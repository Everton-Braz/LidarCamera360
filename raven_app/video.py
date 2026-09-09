"""In-process dual-track decoding and timestamp-aware sharp frame selection."""
import json
from functools import lru_cache
import math
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np


def compute_laplacian_sharpness(image):
    small = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = small @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    gray -= gray.mean()
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())


@lru_cache(maxsize=8)
def _timestamps(path, mtime_ns, size):
    return json.loads(Path(path).read_text(encoding='utf-8'))['timestamps']


def frame_time(dataset_dir, name, fps=1.0):
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


def extract_insv_frames_pyav(insv_path, output_dir, fps=1.0, sharp_window=5, progress_cb=None):
    """Choose up to sharp_window candidates per output interval, preserving PTS.

    Both lenses select the same source instant using their summed sharpness.
    A completion manifest prevents partial or differently sampled runs being reused.
    """
    import av
    insv_path, output_dir = Path(insv_path), Path(output_dir)
    if not math.isfinite(fps) or fps <= 0 or sharp_window < 1:
        raise ValueError('fps must be finite and positive; sharp_window must be >= 1')
    stat = insv_path.stat()
    signature = dict(source=str(insv_path.resolve()), size=stat.st_size,
                     mtime_ns=stat.st_mtime_ns, fps=fps, sharp_window=sharp_window, version=1)
    images = output_dir / 'images'
    manifest = images / 'frames.json'
    if manifest.is_file():
        try:
            old = json.loads(manifest.read_text(encoding='utf-8'))
        except (ValueError, OSError):
            old = {}
        if old.get('signature') == signature and old.get('timestamps') and all(
                (images / name).is_file() for name in old['timestamps']):
            print('[*] Reusing completed PyAV extraction.')
            return True
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='insv-', dir=output_dir) as tmp:
            stage = Path(tmp)
            for cam in ('cam0', 'cam1'):
                (stage / cam).mkdir()
            timestamps = {}
            # Independent demuxers keep decoder state isolated and memory bounded.
            with av.open(str(insv_path)) as front, av.open(str(insv_path)) as rear:
                if len(front.streams.video) < 2:
                    raise ValueError('INSV must contain two video tracks')
                s0, s1 = front.streams.video[0], rear.streams.video[1]
                rate = float(s0.average_rate or 30)
                origin = float((s0.start_time or 0) * s0.time_base)
                interval, best, count, candidates = None, None, 0, 0

                def save(candidate):
                    nonlocal count
                    if candidate is None:
                        return
                    count += 1
                    _, pair, times = candidate
                    for lens in range(2):
                        name = f'cam{lens}/frame_{count:06d}.jpg'
                        ok, encoded = cv2.imencode('.jpg', pair[lens], [cv2.IMWRITE_JPEG_QUALITY, 95])
                        if not ok:
                            raise IOError(f'JPEG encoding failed: {name}')
                        encoded.tofile(stage / name)
                        timestamps[name] = times[lens]
                        if progress_cb:
                            progress_cb(lens, count)

                from itertools import zip_longest
                for index, pair in enumerate(zip_longest(front.decode(s0), rear.decode(s1))):
                    if any(f is None for f in pair):
                        raise ValueError('Video tracks have different frame counts')
                    times = [float(f.time) - origin if f.time is not None else index / rate for f in pair]
                    if abs(times[0] - times[1]) > 0.5 / rate:
                        raise ValueError('Video tracks have mismatched presentation timestamps')
                    bucket = int(math.floor(max(0.0, times[0]) * fps + 1e-7))
                    if bucket != interval:
                        save(best)
                        interval, best, candidates = bucket, None, 0
                    if candidates < sharp_window:
                        arrays = [f.to_ndarray(format='bgr24') for f in pair]
                        score = sum(compute_laplacian_sharpness(a) for a in arrays)
                        if best is None or score > best[0]:
                            best = score, arrays, times
                        candidates += 1
                save(best)
            if not count:
                raise ValueError('No video frames decoded')
            images.mkdir(exist_ok=True)
            # Invalidate completion before publishing; failed publication cannot be reused.
            manifest.unlink(missing_ok=True)
            for cam in ('cam0', 'cam1'):
                target = images / cam
                target.mkdir(exist_ok=True)
                for old in target.glob('frame_*.jpg'):
                    old.unlink()
                for file in (stage / cam).iterdir():
                    shutil.move(str(file), target / file.name)
            manifest.write_text(json.dumps(dict(signature=signature, timestamps=timestamps), indent=2), encoding='utf-8')
            print(f'[+] PyAV extracted {count} synchronized sharp frame pairs.')
            return True
    except Exception as exc:
        print(f'[!] INSV extraction failed: {exc}')
        return False
