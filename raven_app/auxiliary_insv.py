"""Extraction helpers for auxiliary Insta360 INSV lens streams.

The primary video decoder already understands the paired streams and the
camera's exposure-time trailer. This module reuses that decoder in an
independent cache, then publishes one or both lens streams under auxiliary
camera names without changing the primary cam0/cam1 extraction.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable


_CAMERA_NAME = re.compile(r"^cam(\d+)$")
_EXTRACTION_VERSION = 1


def inspect_auxiliary_insv(path: str | Path) -> dict[str, Any]:
    """Return stream geometry and timing metadata for an auxiliary INSV file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Auxiliary INSV file does not exist: {source}")
    if source.suffix.lower() != ".insv":
        raise ValueError(f"Expected an .insv file, received: {source}")

    try:
        import av
    except ImportError as exc:  # pragma: no cover - application packages PyAV
        raise RuntimeError("PyAV is required to inspect auxiliary INSV streams") from exc

    with av.open(str(source)) as container:
        streams = list(container.streams.video)
        if not streams:
            raise ValueError(f"No video streams were found in INSV file: {source}")
        stream_info = []
        for index, stream in enumerate(streams):
            try:
                rate = float(stream.average_rate or 0.0)
            except (TypeError, ValueError, ZeroDivisionError):
                rate = 0.0
            if not math.isfinite(rate) or rate <= 0:
                rate = None
            duration = None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration:
                duration = float(container.duration) / 1_000_000.0
            if duration is not None and (not math.isfinite(duration) or duration <= 0):
                duration = None
            stream_info.append({
                "index": index,
                "width": int(stream.codec_context.width or 0),
                "height": int(stream.codec_context.height or 0),
                "fps": rate,
                "duration_s": duration,
                "frame_count": int(stream.frames or 0),
            })

    first = stream_info[0]
    return {
        "type": "insv",
        "path": str(source.resolve()),
        "lens_count": len(stream_info),
        "width": first["width"],
        "height": first["height"],
        "fps": first["fps"],
        "duration_s": first["duration_s"],
        "streams": stream_info,
    }


def _camera_index(name: str) -> int:
    match = _CAMERA_NAME.fullmatch(str(name))
    if not match:
        raise ValueError(f"Auxiliary INSV camera name must look like camN: {name!r}")
    return int(match.group(1))


def _initial_fisheye_intrinsics(width: int, height: int) -> dict[str, float]:
    """Build a dimension-aware fisheye starting model, not phone pinhole values."""
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid decoded INSV lens dimensions: {width}x{height}")
    # Keep the initial focal length conservative for a circular fisheye image;
    # SfM estimates the final intrinsics during the required session calibration.
    focal = 0.32 * min(width, height)
    return {
        "width": float(width), "height": float(height),
        "fx": focal, "fy": focal,
        "cx": width / 2.0, "cy": height / 2.0,
        "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "p1": 0.0, "p2": 0.0, "sx1": 0.0, "sy1": 0.0,
    }


def expand_auxiliary_insv_config(
    config: Any,
    used_camera_names: Iterable[str] = (),
) -> list[Any]:
    """Expand an INSV config into one selected lens or both consecutive lenses.

    ``used_camera_names`` contains names already reserved by configured cameras.
    A config with no stream selection keeps its configured name for stream 0
    (or single stream) and gives stream 1 the next free ``camN`` when dual streams exist.
    An explicit stream selection keeps the configured name exactly. Previous
    per-camera calibration is invalidated because the lens/source has changed.
    """
    source = Path(str(getattr(config, "source_path", "")))
    if source.suffix.lower() != ".insv":
        raise ValueError(f"Expected an auxiliary .insv source, received: {source}")

    selected = getattr(config, "source_stream_index", None)
    original_name = str(getattr(config, "camera_name", "cam2"))
    base_index = _camera_index(original_name)
    used = {str(name) for name in used_camera_names}
    # The configured name belongs to this camera and must survive expansion;
    # callers may pass the full existing-name set, including this entry.
    used.discard(original_name)

    if selected is not None:
        if isinstance(selected, bool) or selected not in (0, 1):
            raise ValueError("INSV source_stream_index must be None, 0, or 1")
        # Persisted expanded entries can pass through configuration loading
        # several times; return them untouched so calibrated values survive.
        if str(getattr(config, "source_type", "")).lower() == "insv":
            return [config]
        metadata = inspect_auxiliary_insv(source)
        lens_count = int(metadata.get("lens_count", 2) or 2)
        if int(selected) >= lens_count:
            raise ValueError(
                f"INSV stream {selected} is unavailable; "
                f"the source contains {lens_count} video stream(s)"
            )
        item = copy.deepcopy(config)
        item.source_type = "insv"
        item.camera_model = "THIN_PRISM_FISHEYE"
        item.source_stream_index = int(selected)
        stream = metadata["streams"][int(selected)]
        item.intrinsics = _initial_fisheye_intrinsics(stream["width"], stream["height"])
        return [item]

    metadata = inspect_auxiliary_insv(source)
    lens_count = int(metadata.get("lens_count", 2) or 2)
    if lens_count == 1:
        stream_indices = (0,)
        camera_names = (original_name,)
    else:
        stream_indices = (0, 1)
        if max(stream_indices) >= lens_count:
            raise ValueError(
                f"INSV stream {max(stream_indices)} is unavailable; "
                f"the source contains {lens_count} video stream(s)"
            )

        second_index = base_index + 1
        while f"cam{second_index}" in used:
            second_index += 1
        camera_names = (original_name, f"cam{second_index}")

    expanded = []
    for offset, stream_index in enumerate(stream_indices):
        item = copy.deepcopy(config)
        item.camera_name = camera_names[offset]
        item.source_type = "insv"
        item.camera_model = "THIN_PRISM_FISHEYE"
        item.source_stream_index = stream_index
        stream = metadata["streams"][stream_index]
        item.intrinsics = _initial_fisheye_intrinsics(stream["width"], stream["height"])
        report = copy.deepcopy(getattr(item, "calibration_report", {}) or {})
        report["intrinsics"] = {"status": "uncalibrated", "source": "INSV lens stream"}
        report["extrinsics"] = {"status": "uncalibrated", "source": "INSV lens stream"}
        report.setdefault("time_offset", {"status": "unknown", "source": "configuration"})
        item.calibration_report = report
        item.description = f"Insta360 INSV lens stream {stream_index}"
        expanded.append(item)
    return expanded


def _cache_signature(source: Path, fps: float, sharpness_window: int,
                     jpeg_quality: int) -> dict[str, Any]:
    stat = source.stat()
    return {
        "source": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "fps": float(fps),
        "sharpness_window": int(sharpness_window),
        "jpeg_quality": int(jpeg_quality),
        "version": _EXTRACTION_VERSION,
    }


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"schema": 1, "timestamps": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema": 1, "timestamps": {}}
    if not isinstance(value, dict) or not isinstance(value.get("timestamps"), dict):
        return {"schema": 1, "timestamps": {}}
    return value


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _encode_and_write_jpeg(path: Path, image: Any, quality: int) -> None:
    import cv2
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError(f"Failed to encode JPEG: {path}")
    buf.tofile(str(path))


def extract_single_stream_insv_frames_pyav(
    source: Path,
    cache_dir: Path,
    stream_index: int = 0,
    fps: float = 2.0,
    sharp_window: int = 3,
    jpeg_quality: int = 95,
) -> bool:
    """Extract a single video track from an INSV file using PyAV."""
    import av
    from raven_app.video import _timelapse_times, _frame_sharpness
    from raven_app.third_camera import (
        _apply_pyav_frame_orientation,
        _auxiliary_seek_eligible,
        _seek_auxiliary_windows,
        _AuxiliarySeekError,
    )

    source = Path(source)
    cache_dir = Path(cache_dir)
    images_root = cache_dir / "images"
    cam_name = f"cam{stream_index}"
    cam_dir = images_root / cam_name
    manifest_path = images_root / "frames.json"

    signature = _cache_signature(source, fps, sharp_window, jpeg_quality)
    signature["stream_index"] = int(stream_index)

    if manifest_path.is_file():
        cached_manifest = _read_manifest(manifest_path)
        if cached_manifest.get("signature") == signature:
            timestamp_values = cached_manifest.get("timestamps", {})
            prefix = f"{cam_name}/"
            matching = [name for name in timestamp_values if name.startswith(prefix)]
            if matching and all((images_root / name).is_file() for name in matching):
                print(f"[*] Reusing completed single-stream INSV extraction for stream {stream_index}.")
                return True

    cam_dir.mkdir(parents=True, exist_ok=True)
    for old in cam_dir.glob("frame_*.jpg"):
        old.unlink(missing_ok=True)

    capture_times = _timelapse_times(source) if source.suffix.lower() == ".insv" else None
    threads = max(2, min(8, os.cpu_count() or 4))

    container = av.open(str(source))
    try:
        if stream_index >= len(container.streams.video):
            raise ValueError(f"Video stream {stream_index} not found in {source}")
        stream = container.streams.video[stream_index]
        stream.codec_context.thread_count = threads
        stream.thread_type = "AUTO"

        native_fps = float(stream.average_rate or 30.0)
        if not math.isfinite(native_fps) or native_fps <= 0:
            native_fps = 30.0
        effective_fps = min(fps, native_fps)
        origin = float((stream.start_time or 0) * stream.time_base) if stream.time_base else 0.0

        seek_sampling = capture_times is None
        sparse_seek = False
        if seek_sampling:
            try:
                sparse_seek = _auxiliary_seek_eligible(container, stream, effective_fps, sharp_window)
            except Exception:
                sparse_seek = False

        if sparse_seek:
            print(f"[*] {cam_name}: seeking between sharpness windows (single stream).", flush=True)

        encode_workers = max(2, min(4, os.cpu_count() or 4))
        timestamps: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=encode_workers) as pool:
            pending: deque = deque()
            saved_idx = 1
            interval = None
            best = None
            candidates = 0

            def flush_jobs(max_pending=0):
                while len(pending) > max_pending:
                    future, out_name, t_sec = pending.popleft()
                    future.result()
                    timestamps[f"{cam_name}/{out_name}"] = t_sec

            def save_candidate(candidate):
                nonlocal saved_idx
                if candidate is None:
                    return
                _score, frame, t_sec = candidate
                out_name = f"frame_{saved_idx:06d}.jpg"
                out_path = cam_dir / out_name
                bgr = frame.to_ndarray(format="bgr24")
                bgr = _apply_pyav_frame_orientation(frame, bgr)
                flush_jobs(max_pending=8)
                pending.append((pool.submit(_encode_and_write_jpeg, out_path, bgr, jpeg_quality), out_name, float(t_sec)))
                saved_idx += 1

            try:
                frames_iter = (
                    _seek_auxiliary_windows(container, stream, effective_fps, sharp_window)
                    if sparse_seek else container.decode(stream)
                )
                source_frame_idx = 0
                for frame in frames_iter:
                    if capture_times is not None:
                        if source_frame_idx >= len(capture_times):
                            break
                        t_sec = float(capture_times[source_frame_idx])
                    else:
                        t_sec = float(
                            frame.time - origin if frame.time is not None
                            else (frame.pts * stream.time_base if frame.pts is not None else 0.0)
                        )
                    source_frame_idx += 1

                    bucket = int(math.floor(max(0.0, t_sec) * effective_fps + 1e-7))
                    if bucket != interval:
                        save_candidate(best)
                        interval = bucket
                        best = None
                        candidates = 0

                    if candidates < sharp_window:
                        if sharp_window == 1:
                            best = (0.0, frame, t_sec)
                            candidates = 1
                        else:
                            score = _frame_sharpness(frame)
                            if best is None or score > best[0]:
                                best = (score, frame, t_sec)
                            candidates += 1

                save_candidate(best)
            except _AuxiliarySeekError:
                print(f"[!] {cam_name}: seeking failed, falling back to sequential decoding.", flush=True)
                container.seek(stream.start_time or 0, stream=stream, backward=True)
                source_frame_idx = 0
                interval = None
                best = None
                candidates = 0
                for frame in container.decode(stream):
                    if capture_times is not None:
                        if source_frame_idx >= len(capture_times):
                            break
                        t_sec = float(capture_times[source_frame_idx])
                    else:
                        t_sec = float(
                            frame.time - origin if frame.time is not None
                            else (frame.pts * stream.time_base if frame.pts is not None else 0.0)
                        )
                    source_frame_idx += 1

                    bucket = int(math.floor(max(0.0, t_sec) * effective_fps + 1e-7))
                    if bucket != interval:
                        save_candidate(best)
                        interval = bucket
                        best = None
                        candidates = 0

                    if candidates < sharp_window:
                        if sharp_window == 1:
                            best = (0.0, frame, t_sec)
                            candidates = 1
                        else:
                            score = _frame_sharpness(frame)
                            if best is None or score > best[0]:
                                best = (score, frame, t_sec)
                            candidates += 1
                save_candidate(best)
            finally:
                flush_jobs(max_pending=0)
    finally:
        container.close()

    if not timestamps:
        raise RuntimeError(f"No frames were extracted from {source} stream {stream_index}")

    cached_manifest = _read_manifest(manifest_path)
    old_timestamps = cached_manifest.get("timestamps", {})
    if not isinstance(old_timestamps, dict):
        old_timestamps = {}
    prefix = f"{cam_name}/"
    for k in list(old_timestamps.keys()):
        if k.startswith(prefix):
            old_timestamps.pop(k, None)
    old_timestamps.update(timestamps)

    manifest_data = {
        "signature": signature,
        "timestamps": old_timestamps,
        "time_source": "insv_timelapse" if capture_times is not None else "video_pts",
        "decoder": "pyav",
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return True


def extract_auxiliary_insv_frames(
    config: Any,
    dataset_dir: str | Path,
    target_fps: float | None = None,
    sharpness_window: int = 3,
    jpeg_quality: int = 95,
    cache_root: str | Path | None = None,
) -> tuple[list[Path], dict[str, float]]:
    """Extract/publish the configured INSV lens and merge its measured times.

    The existing primary extractor decodes both synchronized streams in a
    source-specific cache when multiple streams exist. For single-stream INSV sources,
    dedicated single-stream decoding is used.
    Only ``config.source_stream_index`` is published into ``dataset_dir/images/<camera_name>``.
    The primary extraction's ``timestamps`` and ``time_source`` are preserved;
    per-camera timing source metadata is recorded separately.
    """
    source = Path(str(getattr(config, "source_path", "")))
    if source.suffix.lower() != ".insv":
        raise ValueError(f"Expected an auxiliary .insv source, received: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"Auxiliary INSV file does not exist: {source}")

    metadata = inspect_auxiliary_insv(source)
    lens_count = int(metadata.get("lens_count", 0) or 0)
    stream_index = getattr(config, "source_stream_index", None)
    if stream_index is None and lens_count == 1:
        stream_index = 0

    if isinstance(stream_index, bool) or stream_index not in (0, 1):
        raise ValueError("Expand an INSV config before extraction; source_stream_index must be 0 or 1")
    if int(stream_index) >= lens_count:
        raise ValueError(
            f"Auxiliary INSV stream {stream_index} is unavailable; "
            f"found {lens_count} stream(s) in {source.name}"
        )

    camera_name = str(getattr(config, "camera_name", ""))
    _camera_index(camera_name)

    fps = float(getattr(config, "fps", 2.0) if target_fps is None else target_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Auxiliary camera extraction FPS must be positive; received {fps!r}")
    if not isinstance(sharpness_window, int) or sharpness_window < 1:
        raise ValueError("sharpness_window must be a positive integer")
    if not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    signature = _cache_signature(source, fps, sharpness_window, jpeg_quality)
    publish_signature = {**signature, "source_stream_index": int(stream_index)}
    dataset = Path(dataset_dir)
    images_root = dataset / "images"
    manifest_path = images_root / "frames.json"
    current_manifest = _read_manifest(manifest_path)
    current_signatures = current_manifest.get("auxiliary_insv_signatures", {})
    cached_signature = current_signatures.get(camera_name) if isinstance(current_signatures, dict) else None
    prefix = f"{camera_name}/"
    current_timestamps = {
        str(name).replace("\\", "/"): float(stamp)
        for name, stamp in current_manifest["timestamps"].items()
        if str(name).replace("\\", "/").startswith(prefix)
    }
    if (cached_signature == publish_signature and current_timestamps
            and all((images_root / name).is_file() for name in current_timestamps)):
        return [images_root / name for name in sorted(current_timestamps)], current_timestamps

    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cache = Path(cache_root) if cache_root is not None else dataset / ".raven_aux_insv_cache"
    cache_dir = cache / key

    if lens_count == 1:
        if not extract_single_stream_insv_frames_pyav(source, cache_dir, stream_index=int(stream_index),
                                                      fps=fps, sharp_window=sharpness_window,
                                                      jpeg_quality=jpeg_quality):
            raise RuntimeError(f"Could not extract auxiliary single-stream INSV from {source}")
    else:
        from raven_app.video import extract_insv_frames_pyav

        try:
            extracted_ok = extract_insv_frames_pyav(source, cache_dir, fps=fps,
                                                    sharp_window=sharpness_window,
                                                    jpeg_quality=jpeg_quality)
        except Exception:
            extracted_ok = False

        if not extracted_ok:
            if not extract_single_stream_insv_frames_pyav(source, cache_dir, stream_index=int(stream_index),
                                                          fps=fps, sharp_window=sharpness_window,
                                                          jpeg_quality=jpeg_quality):
                raise RuntimeError(f"Could not extract auxiliary INSV streams from {source}")

    cached_images = cache_dir / "images"
    cached_manifest_path = cached_images / "frames.json"
    cached_manifest = _read_manifest(cached_manifest_path)
    timestamp_values = cached_manifest.get("timestamps", {})
    source_prefix = f"cam{stream_index}/"
    selected = []
    for name, stamp in timestamp_values.items():
        normalized = str(name).replace("\\", "/")
        if not normalized.startswith(source_prefix):
            continue
        relative_file = normalized[len(source_prefix):]
        if Path(relative_file).name != relative_file or not relative_file.lower().endswith(".jpg"):
            raise ValueError(f"Invalid INSV frame name in extraction manifest: {name!r}")
        frame = cached_images / f"cam{stream_index}" / relative_file
        if not frame.is_file():
            raise FileNotFoundError(f"Cached INSV frame is missing: {frame}")
        selected.append((relative_file, float(stamp), frame))
    selected.sort(key=lambda row: row[0])
    if not selected:
        raise RuntimeError(f"The INSV extractor produced no frames for lens stream {stream_index}")

    images_root = dataset / "images"
    images_root.mkdir(parents=True, exist_ok=True)
    manifest = _read_manifest(manifest_path)
    final_dir = images_root / camera_name
    published_times: dict[str, float] = {}

    # Stage all output before touching the existing camera directory.
    with tempfile.TemporaryDirectory(prefix=".aux-insv-stage-", dir=images_root) as temp:
        stage_dir = Path(temp) / camera_name
        stage_dir.mkdir()
        for filename, stamp, cached_frame in selected:
            _link_or_copy(cached_frame, stage_dir / filename)
            published_times[f"{camera_name}/{filename}"] = stamp

        final_dir.mkdir(parents=True, exist_ok=True)
        for old in final_dir.glob("frame_*.jpg"):
            old.unlink(missing_ok=True)
        for staged in stage_dir.iterdir():
            staged.replace(final_dir / staged.name)

    timestamps = manifest["timestamps"]
    prefix = f"{camera_name}/"
    for old_name in list(timestamps):
        if str(old_name).replace("\\", "/").startswith(prefix):
            timestamps.pop(old_name, None)
    timestamps.update(published_times)

    signatures = manifest.get("auxiliary_insv_signatures", {})
    if not isinstance(signatures, dict):
        signatures = {}
    signatures[camera_name] = publish_signature
    manifest["auxiliary_insv_signatures"] = signatures
    time_sources = manifest.get("time_sources", {})
    if not isinstance(time_sources, dict):
        time_sources = {}
    time_sources[camera_name] = cached_manifest.get("time_source", "video_pts")
    manifest["time_sources"] = time_sources

    temp_manifest = images_root / f".frames-{key}-{camera_name}.json"
    temp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp_manifest.replace(manifest_path)

    paths = [final_dir / filename for filename, _, _ in selected]
    return paths, published_times
