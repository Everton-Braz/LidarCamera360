"""In-process dual-track decoding and timestamp-aware sharp frame selection."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np


def compute_laplacian_sharpness(image):
    """Compute variance of the Laplacian as a blur/sharpness metric."""
    if image.ndim == 3:
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        if image.shape[0] != 512 or image.shape[1] != 512:
            gray = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        else:
            gray = image
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())


@lru_cache(maxsize=8)
def _timestamps(path, mtime_ns, size):
    return json.loads(Path(path).read_text(encoding='utf-8'))['timestamps']


def frame_time(dataset_dir, name, fps=2.0):
    """Read measured presentation time; support older, one-based frame datasets."""
    manifest = Path(dataset_dir) / 'images' / 'frames.json'
    if manifest.is_file():
        stat = manifest.stat()
        data = _timestamps(str(manifest), stat.st_mtime_ns, stat.st_size)
        key = str(name).replace('\\', '/')
        if key in data:
            return float(data[key])
    digits = ''.join(filter(str.isdigit, Path(name).stem))
    return (int(digits) - 1) / fps


def _open_decoder(path, track, backend, threads):
    import av
    kwargs = {}
    if backend != 'cpu':
        from av.codec.hwaccel import HWAccel
        kwargs['hwaccel'] = HWAccel(backend, allow_software_fallback=False)
    container = av.open(str(path), **kwargs)
    try:
        if len(container.streams.video) != 2:
            raise ValueError('Input must contain exactly two video tracks')
        stream = container.streams.video[track]
        stream.codec_context.thread_count = threads
        stream.thread_type = 'AUTO'
        return container, stream
    except BaseException:
        container.close()
        raise


def _select_decoder(path, requested, threads):
    if requested != 'auto':
        return requested
    import av
    with av.open(str(path)) as probe:
        if len(probe.streams.video) != 2:
            return 'cpu'
        if probe.streams.video[0].codec_context.name not in ('hevc', 'h264', 'av1', 'vp9', 'mpeg2video'):
            return 'cpu'
    from av.codec.hwaccel import hwdevices_available
    available = hwdevices_available()
    # Probe the actual codec/driver, not just FFmpeg's compiled-in device list.
    for backend in ('cuda', 'd3d11va', 'videotoolbox', 'vaapi'):
        if backend not in available:
            continue
        try:
            container, stream = _open_decoder(path, 0, backend, threads)
            with container:
                next(container.decode(stream))
            return backend
        except Exception:
            # Failed hardware initialization may leave cyclic PyAV references.
            import gc
            gc.collect()
            continue
    return 'cpu'


def _frame_sharpness(frame):
    # Scale native YUV directly to 512x512 grayscale in C/SIMD via FFmpeg swscale.
    # Eliminates full-resolution RGB conversion and Python float matrix operations for candidates.
    small = frame.reformat(width=512, height=512, format='gray', interpolation='FAST_BILINEAR')
    return float(cv2.Laplacian(small.to_ndarray(), cv2.CV_32F, ksize=1).var())


def _write_pair(stage, number, frames, jpeg_quality):
    for lens, frame in enumerate(frames):
        image = frame.to_ndarray(format='bgr24')
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise IOError('JPEG encoding failed')
        encoded.tofile(stage / f'cam{lens}/frame_{number:06d}.jpg')


def extract_video_frames(insv_path, output_dir, fps=2.0, sharp_window=5,
                         progress_cb=None, *, decoder='auto', threads=None,
                         jpeg_quality=95):
    """Extract synchronized sharp pairs using GPU decoding with CPU fallback.

    Two bounded decoder workers overlap lenses; dedicated encoder jobs overlap
    JPEG writing with decoding. Original presentation timestamps are retained.
    """
    insv_path, output_dir = Path(insv_path), Path(output_dir)
    if not math.isfinite(fps) or fps <= 0 or not isinstance(sharp_window, int) or sharp_window < 1:
        raise ValueError('fps must be positive and finite; sharp_window must be a positive integer')
    threads = threads if threads is not None else max(1, min(4, (os.cpu_count() or 2)//2))
    if threads < 1 or not 1 <= jpeg_quality <= 100:
        raise ValueError('Invalid decoder thread count or JPEG quality')
    if decoder not in ('auto', 'cpu', 'cuda', 'd3d11va', 'videotoolbox', 'vaapi'):
        raise ValueError('Unsupported decoder backend')
    stat = insv_path.stat()
    signature = dict(source=str(insv_path.resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, fps=fps, sharp_window=sharp_window,
                     jpeg_quality=jpeg_quality, version=2)
    images = output_dir / 'images'
    manifest = images / 'frames.json'
    if manifest.is_file():
        try:
            old = json.loads(manifest.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            old = {}
        if old.get('signature') == signature and old.get('timestamps') and all(
                (images / name).is_file() for name in old['timestamps']):
            print('[*] Reusing completed video extraction.')
            return True
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = _select_decoder(insv_path, decoder, threads)
    attempts = [backend] + (['cpu'] if decoder == 'auto' and backend != 'cpu' else [])
    for backend in attempts:
        print(f'[*] Video decoder: {backend}; threads per lens: {threads}', flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix='frames-', dir=output_dir) as tmp, ExitStack() as stack:
                stage = Path(tmp)
                for cam in ('cam0', 'cam1'):
                    (stage / cam).mkdir()
                sources = []
                for lens in range(2):
                    container, stream = _open_decoder(insv_path, lens, backend, threads)
                    stack.enter_context(container)
                    sources.append((container, stream))
                decode_pool = stack.enter_context(ThreadPoolExecutor(max_workers=2, thread_name_prefix='video'))
                encode_workers = max(2, min(os.cpu_count() or 4, 4))
                encode_pool = stack.enter_context(ThreadPoolExecutor(max_workers=encode_workers, thread_name_prefix='jpeg'))
                iterators = [container.decode(stream) for container, stream in sources]
                stream = sources[0][1]
                rate = float(stream.average_rate or 30)
                origin = float((stream.start_time or 0) * stream.time_base)
                timestamps, pending = {}, deque()
                interval, best, count, candidates, index = None, None, 0, 0, 0

                def finish_job():
                    future, number = pending.popleft()
                    future.result()
                    if progress_cb:
                        for lens in range(2):
                            progress_cb(lens, number)

                def save(candidate):
                    nonlocal count
                    if candidate is None:
                        return
                    if len(pending) >= 4:
                        finish_job()
                    count += 1
                    _, pair, times = candidate
                    pending.append((encode_pool.submit(_write_pair, stage, count, pair, jpeg_quality), count))
                    for lens in range(2):
                        timestamps[f'cam{lens}/frame_{count:06d}.jpg'] = times[lens]

                while True:
                    futures = [decode_pool.submit(next, iterator, None) for iterator in iterators]
                    pair = tuple(f.result() for f in futures)
                    if all(f is None for f in pair):
                        break
                    if any(f is None for f in pair):
                        raise ValueError('Video tracks have different frame counts')
                    times = [float(f.time) - origin if f.time is not None else index / rate for f in pair]
                    index += 1
                    if abs(times[0] - times[1]) > 0.5 / rate:
                        raise ValueError('Video tracks have mismatched presentation timestamps')
                    bucket = int(math.floor(max(0.0, times[0]) * fps + 1e-7))
                    if bucket != interval:
                        save(best)
                        interval, best, candidates = bucket, None, 0
                    if candidates < sharp_window:
                        if sharp_window == 1:
                            best = 0.0, pair, times
                            candidates = 1
                        else:
                            score = sum(_frame_sharpness(frame) for frame in pair)
                            if best is None or score > best[0]:
                                best = score, pair, times
                            candidates += 1
                save(best)
                while pending:
                    finish_job()
                if not count:
                    raise ValueError('No video frames decoded')
                images.mkdir(exist_ok=True)
                manifest.unlink(missing_ok=True)
                for cam in ('cam0', 'cam1'):
                    target = images / cam
                    target.mkdir(exist_ok=True)
                    for old in target.glob('frame_*.jpg'):
                        old.unlink()
                    for file in (stage / cam).iterdir():
                        shutil.move(str(file), target / file.name)
                manifest.write_text(json.dumps(dict(signature=signature, timestamps=timestamps,
                                                    decoder=backend), indent=2), encoding='utf-8')
                print(f'[+] Extracted {count} synchronized sharp frame pairs.')
                return True
        except Exception as exc:
            print(f'[!] {backend} video extraction failed: {exc}', flush=True)
    return False


# Compatibility alias for existing integrations.
extract_insv_frames_pyav = extract_video_frames
