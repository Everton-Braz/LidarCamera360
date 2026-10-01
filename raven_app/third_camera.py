"""Third camera / auxiliary video and image source integration for RavenCalibrator.

Supports smartphone or secondary camera integration to improve point cloud
colorization and 3D Gaussian Splatting (3DGS) multi-view coverage.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
import shutil
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from raven_app.video import compute_laplacian_sharpness, frame_time


@dataclass
class ThirdCameraConfig:
    """Configuration for auxiliary image/video source (e.g. frontal smartphone)."""

    camera_name: str = "cam2"
    source_path: str = ""
    source_type: str = "video"  # "video" or "folder"
    enabled: bool = True
    fps: float = 2.0
    time_offset_s: float = 0.0
    camera_model: str = "PINHOLE"  # "PINHOLE", "OPENCV", "OPENCV_FISHEYE"
    intrinsics: Dict[str, float] = field(
        default_factory=lambda: {
            "width": 1920.0,
            "height": 1080.0,
            "fx": 1536.0,
            "fy": 1536.0,
            "cx": 960.0,
            "cy": 540.0,
            "k1": 0.0,
            "k2": 0.0,
            "k3": 0.0,
            "k4": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        }
    )
    T_lidar_to_cam2_rigid_4x4: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.15],
            [0.0, 0.0, 1.0, 0.20],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    description: str = "Frontal smartphone camera"
    calibration_report: Dict[str, Any] = field(
        default_factory=lambda: {
            "intrinsics": {"status": "uncalibrated", "source": "configuration"},
            "time_offset": {"status": "unknown", "source": "configuration"},
            "extrinsics": {"status": "uncalibrated", "source": "configuration"},
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        transform_key = f"T_lidar_to_{self.camera_name}_rigid_4x4"
        data[transform_key] = copy.deepcopy(self.T_lidar_to_cam2_rigid_4x4)
        report = copy.deepcopy(self.calibration_report or {})
        report['camera_name'] = self.camera_name
        report['camera_model'] = self.camera_model
        report['extrinsics'] = dict(report.get('extrinsics') or {})
        report['extrinsics'].setdefault('camera_name', self.camera_name)
        report['extrinsics']['transform_key'] = transform_key
        report['intrinsics'] = dict(report.get('intrinsics') or {})
        report['intrinsics'].setdefault('camera_name', self.camera_name)
        report['time_offset'] = dict(report.get('time_offset') or {})
        report['time_offset'].setdefault('camera_name', self.camera_name)
        data['calibration_report'] = report
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThirdCameraConfig:
        camera_name = str(data.get('camera_name', 'cam2'))
        clean = {}
        for k in (
            "camera_name",
            "source_path",
            "source_type",
            "enabled",
            "fps",
            "time_offset_s",
            "camera_model",
            "intrinsics",
            "T_lidar_to_cam2_rigid_4x4",
            "description",
            "calibration_report",
        ):
            if k in data:
                clean[k] = data[k]
        camera_transform_key = f"T_lidar_to_{camera_name}_rigid_4x4"
        if camera_transform_key in data:
            clean['T_lidar_to_cam2_rigid_4x4'] = data[camera_transform_key]
        return cls(**clean)


def default_smartphone_intrinsics(width: int = 1920, height: int = 1080) -> Dict[str, float]:
    """Return standard rectilinear pinhole intrinsics for a typical smartphone camera (~68° HFOV)."""
    fx = float(width) * 0.80
    fy = float(width) * 0.80
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return {
        "width": float(width),
        "height": float(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": 0.0,
        "k2": 0.0,
        "k3": 0.0,
        "k4": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def load_third_camera_config(path_or_dict: Union[str, Path, Dict[str, Any]]) -> ThirdCameraConfig:
    """Load ThirdCameraConfig from a file path or dictionary."""
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        raise TypeError(f"Expected path or dict, got {type(path_or_dict)}")
    return ThirdCameraConfig.from_dict(data)


def load_third_camera_configs(
    path_or_dict: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]
) -> List[ThirdCameraConfig]:
    """Load legacy single-camera or multi-camera bundle configuration."""
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = path_or_dict

    if isinstance(data, list):
        cameras = data
    elif isinstance(data, dict) and isinstance(data.get("cameras"), list):
        cameras = data["cameras"]
    elif isinstance(data, dict):
        cameras = [data]
    else:
        raise TypeError("Camera configuration must be an object, a camera list, or an object with a 'cameras' list")

    if any(not isinstance(camera, dict) for camera in cameras):
        raise TypeError("Every entry in the camera configuration must be an object")
    return [ThirdCameraConfig.from_dict(camera) for camera in cameras]


def save_third_camera_config(
    config: Union[ThirdCameraConfig, Sequence[ThirdCameraConfig]],
    target_path: Union[str, Path],
) -> Path:
    """Persist a legacy single camera or a ``{"cameras": [...]}`` bundle."""
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(config, ThirdCameraConfig):
        payload: Dict[str, Any] = config.to_dict()
    else:
        cameras = list(config)
        if any(not isinstance(camera, ThirdCameraConfig) for camera in cameras):
            raise TypeError("Camera config bundles must contain ThirdCameraConfig values")
        payload = {"cameras": [camera.to_dict() for camera in cameras]}
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def detect_source_metadata(source_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect a video file or image folder and return dimensions and count."""
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {p}")

    if p.is_file():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {p}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = float(count / fps) if fps > 0 else 0.0
        cap.release()
        return {
            "type": "video",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": count,
            "duration_s": dur,
            "path": str(p.resolve()),
        }

    if p.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(
            [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )
        if not images:
            raise ValueError(f"No image files found in folder: {p}")
        sample = cv2.imread(str(images[0]))
        if sample is None:
            raise ValueError(f"Cannot read sample image: {images[0]}")
        h, w = sample.shape[:2]
        return {
            "type": "folder",
            "width": int(w),
            "height": int(h),
            "image_count": len(images),
            "images": [str(img.resolve()) for img in images],
            "path": str(p.resolve()),
        }

    raise ValueError(f"Invalid source path: {p}")


def _frame_sharpness_pyav(frame) -> float:
    small = frame.reformat(width=512, height=512, format="gray", interpolation="FAST_BILINEAR")
    return float(cv2.Laplacian(small.to_ndarray(), cv2.CV_32F, ksize=1).var())


def _frame_sharpness_cv2(image: np.ndarray) -> float:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    if gray.shape[0] != 512 or gray.shape[1] != 512:
        gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_LINEAR)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=1).var())


def _write_jpeg_frame(path: Path, image: np.ndarray, quality: int):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"JPEG encoding failed for {path}")
    encoded.tofile(str(path))


def _apply_pyav_frame_orientation(frame, image: np.ndarray) -> np.ndarray:
    """Apply PyAV's display-matrix rotation before saving decoded pixels."""
    try:
        rotation = float(frame.rotation)
    except (AttributeError, TypeError, ValueError):
        return image
    if not math.isfinite(rotation):
        return image

    degrees = int(round(rotation)) % 360
    if degrees % 90:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


class _AuxiliarySeekError(RuntimeError):
    pass


def _auxiliary_seek_eligible(container, stream, fps, window):
    """Skip unused GOP tails only for low-rate, long, densely-keyed clips."""
    rate = float(stream.average_rate or 0)
    if (not stream.duration or not stream.time_base or fps > 1 or
            rate < window * fps * 4 or float(stream.duration * stream.time_base) < 30):
        return False
    keys = []
    try:
        for index, packet in enumerate(container.demux(stream)):
            if packet.is_keyframe and packet.pts is not None:
                keys.append(float(packet.pts * stream.time_base))
                if len(keys) == 3:
                    return all(0 < b - a <= 1.25 / fps for a, b in zip(keys, keys[1:]))
            if index >= min(1000, int(rate * 4 / fps)):
                break
        return False
    finally:
        container.seek(stream.start_time or 0, stream=stream, backward=True)


def _seek_auxiliary_windows(container, stream, fps, window):
    """Decode the original first window candidates in each target time bucket."""
    origin = float((stream.start_time or 0) * stream.time_base)
    buckets = int(math.ceil(float(stream.duration * stream.time_base) * fps - 1e-9))
    try:
        for bucket in range(buckets):
            offset = int((origin + bucket / fps) / stream.time_base)
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            count = 0
            for frame in container.decode(stream):
                if frame.time is None:
                    raise ValueError('Sparse extraction requires presentation timestamps')
                current = int(math.floor(max(0.0, float(frame.time) - origin) * fps + 1e-7))
                if current < bucket:
                    continue
                if current > bucket:
                    break
                yield frame
                count += 1
                if count == window:
                    break
            if count == 0:
                raise ValueError(f'No candidates found for time bucket {bucket}')
    except Exception as exc:
        raise _AuxiliarySeekError(str(exc)) from exc


def _decode_auxiliary_video_pyav(
    source_path: Path,
    cam_dir: Path,
    camera_name: str,
    target_fps: float,
    sharpness_window: int,
    jpeg_quality: int,
    *, seek_sampling: bool = True,
) -> Tuple[List[Path], Dict[str, float], float]:
    """Select sharp windows, seeking past unused GOP tails when appropriate."""
    import av

    container = av.open(str(source_path))
    extracted: List[Path] = []
    timestamps: Dict[str, float] = {}
    native_fps = 30.0
    try:
        if not container.streams.video:
            return extracted, timestamps, native_fps
        stream = container.streams.video[0]
        stream.codec_context.thread_count = max(2, min(8, os.cpu_count() or 4))
        stream.thread_type = "AUTO"

        native_fps = float(stream.average_rate or 30.0)
        if not math.isfinite(native_fps) or native_fps <= 0:
            native_fps = 30.0
        effective_fps = min(target_fps, native_fps)
        try:
            sparse_seek = seek_sampling and _auxiliary_seek_eligible(
                container, stream, effective_fps, sharpness_window)
        except Exception as exc:
            raise _AuxiliarySeekError(str(exc)) from exc
        if sparse_seek:
            print(f'[*] {camera_name}: seeking between sharpness windows (all candidates retained).')

        origin = (
            float((stream.start_time or 0) * stream.time_base)
            if stream.time_base else 0.0
        )
        encode_workers = max(2, min(4, os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=encode_workers) as pool:
            pending = deque()
            saved_idx = 1
            interval = None
            best = None
            candidates = 0

            def flush_jobs(max_pending=0):
                while len(pending) > max_pending:
                    future, _ = pending.popleft()
                    future.result()

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
                pending.append((pool.submit(_write_jpeg_frame, out_path, bgr, jpeg_quality), out_name))
                extracted.append(out_path)
                timestamps[f"{camera_name}/{out_name}"] = float(t_sec)
                saved_idx += 1

            try:
                frames = (_seek_auxiliary_windows(container, stream, effective_fps, sharpness_window)
                          if sparse_seek else container.decode(stream))
                for frame in frames:
                    t_sec = float(
                        frame.time - origin if frame.time is not None
                        else (frame.pts * stream.time_base if frame.pts is not None else 0.0)
                    )
                    bucket = int(math.floor(max(0.0, t_sec) * effective_fps + 1e-7))
                    if bucket != interval:
                        save_candidate(best)
                        interval = bucket
                        best = None
                        candidates = 0

                    if candidates < sharpness_window:
                        if sharpness_window == 1:
                            best = (0.0, frame, t_sec)
                            candidates = 1
                        else:
                            score = _frame_sharpness_pyav(frame)
                            if best is None or score > best[0]:
                                best = (score, frame, t_sec)
                            candidates += 1
                save_candidate(best)
            finally:
                flush_jobs(max_pending=0)
    finally:
        container.close()

    return extracted, timestamps, native_fps


def extract_third_camera_frames(
    config: ThirdCameraConfig,
    dataset_dir: Union[str, Path],
    target_fps: Optional[float] = None,
    sharpness_window: int = 3,
    jpeg_quality: int = 95,
) -> Tuple[List[Path], Dict[str, float]]:
    """Extract frames from third camera video or directory into dataset/images/cam2.

    Returns:
        (extracted_paths, timestamp_dict) where timestamp_dict maps relative name
        (e.g. 'cam2/frame_000001.jpg') to video PTS in seconds.
    """
    dataset_dir = Path(dataset_dir)
    images_root = dataset_dir / "images"
    cam_dir = images_root / config.camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    fps = float(config.fps if target_fps is None else target_fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Auxiliary camera extraction FPS must be positive; received {fps!r}")
    source_p = Path(config.source_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Third camera source does not exist: {source_p}")
    extraction_started = time.perf_counter()

    extracted: List[Path] = []
    timestamps: Dict[str, float] = {}
    native_fps: float = 30.0
    decoder_name = "image folder"

    # 1. Check cached extraction
    manifest_path = images_root / "frames.json"
    existing_data: Dict[str, Any] = {"schema": 1, "timestamps": {}}
    if manifest_path.is_file():
        try:
            existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "timestamps" not in existing_data or not isinstance(existing_data["timestamps"], dict):
                existing_data["timestamps"] = {}
        except Exception:
            existing_data = {"schema": 1, "timestamps": {}}

    existing_timestamps = existing_data["timestamps"]
    source_stat = source_p.stat()
    aux_sig_key = f"signature_{config.camera_name}"
    curr_sig = {
        "source": str(source_p.resolve()),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "fps": fps,
        "sharpness_window": sharpness_window,
        "jpeg_quality": jpeg_quality,
        "version": 3,
    }

    if existing_data.get(aux_sig_key) == curr_sig and source_p.resolve() != cam_dir.resolve():
        prefix = f"{config.camera_name}/"
        matched = {k: v for k, v in existing_timestamps.items() if k.startswith(prefix)}
        if matched and all((images_root / k).is_file() for k in matched):
            extracted = sorted([images_root / k for k in matched])
            elapsed = time.perf_counter() - extraction_started
            print(
                f"[*] Reusing completed auxiliary frame extraction for '{config.camera_name}' "
                f"({len(extracted)} frames in {elapsed:.2f}s; decoder=cache)."
            )
            return extracted, matched

    if config.source_type == "video" or source_p.is_file():
        pyav_success = False
        try:
            try:
                extracted, timestamps, native_fps = _decode_auxiliary_video_pyav(
                    source_p, cam_dir, config.camera_name, fps,
                    sharpness_window, jpeg_quality,
                )
            except _AuxiliarySeekError as exc:
                print(f'[*] Sparse seek unavailable ({exc}); retaining full PyAV decoding.')
                extracted, timestamps, native_fps = _decode_auxiliary_video_pyav(
                    source_p, cam_dir, config.camera_name, fps,
                    sharpness_window, jpeg_quality, seek_sampling=False,
                )
            pyav_success = bool(extracted)
            if pyav_success:
                decoder_name = "PyAV CPU"
        except Exception as exc:
            print(f"[!] PyAV auxiliary extraction fallback to OpenCV: {exc}", flush=True)
            extracted.clear()
            timestamps.clear()
            pyav_success = False

        if not pyav_success:
            decoder_name = "OpenCV"
            cap = cv2.VideoCapture(str(source_p))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video source: {source_p}")
            if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
                cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

            native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if not math.isfinite(native_fps) or native_fps <= 0:
                native_fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            effective_fps = min(fps, native_fps)
            frame_step = native_fps / effective_fps
            frame_idx = 0
            saved_idx = 1
            sample_idx = 0
            window_size = max(1, int(sharpness_window))

            encode_workers = max(2, min(4, os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=encode_workers) as pool:
                pending = deque()

                def flush_jobs_cv(max_pending=0):
                    while len(pending) > max_pending:
                        fut, _ = pending.popleft()
                        fut.result()

                try:
                    while total_frames <= 0 or int(round(sample_idx * frame_step)) < total_frames:
                        target_idx = int(round(sample_idx * frame_step))
                        if target_idx - frame_idx > 30:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                            frame_idx = target_idx
                        else:
                            while frame_idx < target_idx:
                                if not cap.grab():
                                    break
                                frame_idx += 1
                        if frame_idx < target_idx:
                            break

                        next_target_idx = int(round((sample_idx + 1) * frame_step))
                        interval = max(1, next_target_idx - target_idx)
                        candidate_count = min(window_size, interval)
                        if total_frames > 0:
                            candidate_count = min(candidate_count, total_frames - target_idx)

                        candidates = []
                        for _ in range(candidate_count):
                            ret, frame = cap.read()
                            if not ret:
                                break
                            sharpness = _frame_sharpness_cv2(frame) if candidate_count > 1 else 0.0
                            candidates.append((sharpness, frame, frame_idx))
                            frame_idx += 1
                        if not candidates:
                            break

                        _score, best_frame, best_fidx = max(candidates, key=lambda item: item[0])
                        out_name = f"frame_{saved_idx:06d}.jpg"
                        out_path = cam_dir / out_name
                        flush_jobs_cv(max_pending=8)
                        pending.append((pool.submit(_write_jpeg_frame, out_path, best_frame, jpeg_quality), out_name))
                        extracted.append(out_path)
                        timestamps[f"{config.camera_name}/{out_name}"] = float(best_fidx / native_fps)
                        saved_idx += 1
                        sample_idx += 1
                finally:
                    cap.release()
                    flush_jobs_cv(max_pending=0)

    else:
        # Image folder
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        files = sorted(
            [f for f in source_p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )
        saved_idx = 1
        for idx, f in enumerate(files):
            out_name = f"frame_{saved_idx:06d}.jpg"
            out_path = cam_dir / out_name
            if f.resolve() == out_path.resolve():
                pass
            elif f.suffix.lower() in (".jpg", ".jpeg"):
                try:
                    os.link(f, out_path)
                except Exception:
                    try:
                        out_path.symlink_to(f)
                    except Exception:
                        shutil.copy2(f, out_path)
            else:
                img = cv2.imread(str(f))
                if img is not None:
                    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                else:
                    continue

            extracted.append(out_path)
            rel_name = f"{config.camera_name}/{out_name}"
            # Time spacing according to fps
            timestamps[rel_name] = float(idx / fps)
            saved_idx += 1

    # Update or merge with existing images/frames.json
    if extracted and source_p.resolve() != cam_dir.resolve():
        prefix = f"{config.camera_name}/"
        for name in list(existing_timestamps):
            normalized = str(name).replace("\\", "/")
            if normalized.startswith(prefix) and normalized not in timestamps:
                existing_timestamps.pop(name, None)
        current_names = {path.name for path in extracted}
        for old_frame in cam_dir.glob("frame_*.jpg"):
            if old_frame.name not in current_names:
                old_frame.unlink(missing_ok=True)
    existing_timestamps.update(timestamps)
    existing_data[aux_sig_key] = curr_sig
    manifest_path.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")

    rate_note = (f" at {min(fps, native_fps):.2f} fps" if source_p.is_file() else "")
    elapsed = max(time.perf_counter() - extraction_started, 1e-9)
    throughput = len(extracted) / elapsed
    print(f"[+] Extracted {len(extracted)} auxiliary frames from '{source_p.name}'"
          f"{rate_note} into: {cam_dir} in {elapsed:.2f}s "
          f"({throughput:.2f} output frames/s; decoder={decoder_name})")
    return extracted, timestamps


def match_features_between_images(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "SIFT",
    max_features: int = 3000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extract and match features between two images using SIFT or ORB."""
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    if method.upper() == "SIFT" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING

    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), 0

    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    ratio_thresh = 0.75
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, len(good)


def estimate_relative_camera_pose(
    pts_cam0: np.ndarray,
    pts_cam2: np.ndarray,
    K_cam0: np.ndarray,
    K_cam2: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Estimate relative rotation R and translation unit direction t between cam0 and cam2.

    Returns (R_02, t_02, inliers) such that P_cam2 = R_02 @ P_cam0 + t_02.
    """
    if len(pts_cam0) < 8 or len(pts_cam2) < 8:
        return None, None, 0

    # Essential Matrix via normalized points
    pts0_norm = cv2.undistortPoints(
        np.expand_dims(pts_cam0, axis=1), K_cam0, None
    ).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(
        np.expand_dims(pts_cam2, axis=1), K_cam2, None
    ).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        pts0_norm, pts2_norm, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=0.003
    )
    if E is None or mask is None:
        return None, None, 0

    inliers = int(np.sum(mask))
    if inliers < 8:
        return None, None, 0

    _, R, t, mask_pose = cv2.recoverPose(E, pts0_norm, pts2_norm)
    inliers_pose = int(np.sum(mask_pose))
    return R, t.flatten(), inliers_pose


_COLMAP_CAMERA_PARAM_COUNTS = {
    0: 3,   # SIMPLE_PINHOLE
    1: 4,   # PINHOLE
    2: 4,   # SIMPLE_RADIAL
    3: 5,   # RADIAL
    4: 8,   # OPENCV
    5: 8,   # OPENCV_FISHEYE
    6: 12,  # FULL_OPENCV
    7: 5,   # FOV
    8: 4,   # SIMPLE_RADIAL_FISHEYE
    9: 5,   # RADIAL_FISHEYE
    10: 12, # THIN_PRISM_FISHEYE
}
_COLMAP_CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}


def _read_colmap_cameras(path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """Read the camera records needed to transfer SfM calibration to cam2."""
    cameras: Dict[int, Dict[str, Any]] = {}
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Invalid COLMAP cameras.bin header")
        count = struct.unpack("<Q", header)[0]
        for _ in range(count):
            header = f.read(24)
            if len(header) != 24:
                raise ValueError("Truncated COLMAP camera record")
            camera_id, model_id, width, height = struct.unpack("<iiQQ", header)
            param_count = _COLMAP_CAMERA_PARAM_COUNTS.get(model_id)
            if param_count is None:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            param_data = f.read(param_count * 8)
            if len(param_data) != param_count * 8:
                raise ValueError("Truncated COLMAP camera parameters")
            params = struct.unpack(f"<{param_count}d", param_data)
            cameras[camera_id] = {
                "model_id": model_id,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _third_camera_intrinsics_from_colmap(
    camera: Dict[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Convert supported COLMAP camera models to the app's camera schema."""
    model_id = int(camera["model_id"])
    p = tuple(float(value) for value in camera["params"])
    width, height = float(camera["width"]), float(camera["height"])
    common = {"width": width, "height": height}

    if model_id == 0:  # SIMPLE_PINHOLE: f, cx, cy
        fx = fy = p[0]
        cx, cy = p[1:3]
        model = "PINHOLE"
    elif model_id == 1:  # PINHOLE: fx, fy, cx, cy
        fx, fy, cx, cy = p[:4]
        model = "PINHOLE"
    elif model_id in (2, 3):  # SIMPLE_RADIAL / RADIAL
        fx = fy = p[0]
        cx, cy = p[1:3]
        model = "OPENCV"
        common.update(k1=p[3], k2=p[4] if model_id == 3 else 0.0)
    elif model_id == 4:  # OPENCV
        fx, fy, cx, cy = p[:4]
        model = "OPENCV"
        common.update(k1=p[4], k2=p[5], p1=p[6], p2=p[7])
    elif model_id == 5:  # OPENCV_FISHEYE
        fx, fy, cx, cy = p[:4]
        model = "OPENCV_FISHEYE"
        common.update(k1=p[4], k2=p[5], k3=p[6], k4=p[7])
    elif model_id in (8, 9):  # SIMPLE/RADIAL_FISHEYE
        fx = fy = p[0]
        cx, cy = p[1:3]
        model = "OPENCV_FISHEYE"
        common.update(k1=p[3], k2=p[4] if model_id == 9 else 0.0)
    elif model_id == 10:  # THIN_PRISM_FISHEYE is not an auxiliary camera model.
        raise ValueError("COLMAP THIN_PRISM_FISHEYE cannot calibrate an auxiliary camera")
    else:
        raise ValueError(
            f"COLMAP camera model {model_id} cannot be represented by the auxiliary camera schema"
        )

    intrinsics = {
        **common,
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }
    if model == "PINHOLE":
        intrinsics.update(k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0)
    else:
        intrinsics.setdefault("k1", 0.0)
        intrinsics.setdefault("k2", 0.0)
        intrinsics.setdefault("k3", 0.0)
        intrinsics.setdefault("k4", 0.0)
        intrinsics.setdefault("p1", 0.0)
        intrinsics.setdefault("p2", 0.0)
    if not all(math.isfinite(value) for value in intrinsics.values()):
        raise ValueError("COLMAP camera intrinsics contain non-finite values")
    if width <= 0 or height <= 0 or fx <= 0 or fy <= 0:
        raise ValueError("COLMAP camera intrinsics have invalid dimensions or focal lengths")
    return model, intrinsics


def _pose_consensus(
    rotations: List[np.ndarray], translations: List[np.ndarray]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float]]:
    """Robustly reject inconsistent per-frame rig poses before accepting calibration."""
    if len(rotations) < 6 or len(translations) != len(rotations):
        return None, None, {"candidate_poses": float(len(rotations)), "accepted_poses": 0.0}

    rotation_array = np.asarray(rotations, dtype=np.float64)
    translation_array = np.asarray(translations, dtype=np.float64)
    if (
        not np.all(np.isfinite(rotation_array))
        or not np.all(np.isfinite(translation_array))
        or translation_array.shape != (len(rotations), 3)
    ):
        return None, None, {"candidate_poses": float(len(rotations)), "accepted_poses": 0.0}

    try:
        center_rotation = Rot.from_matrix(rotation_array).mean()
    except ValueError:
        return None, None, {"candidate_poses": float(len(rotations)), "accepted_poses": 0.0}
    center_translation = np.median(translation_array, axis=0)
    translation_errors = np.linalg.norm(translation_array - center_translation, axis=1)
    rotation_errors_deg = np.degrees(
        (center_rotation.inv() * Rot.from_matrix(rotation_array)).magnitude()
    )

    translation_median = float(np.median(translation_errors))
    translation_mad = float(np.median(np.abs(translation_errors - translation_median)))
    rotation_median = float(np.median(rotation_errors_deg))
    rotation_mad = float(np.median(np.abs(rotation_errors_deg - rotation_median)))
    translation_gate = min(0.15, max(0.04, translation_median + 3.0 * 1.4826 * translation_mad))
    rotation_gate = min(8.0, max(2.0, rotation_median + 3.0 * 1.4826 * rotation_mad))
    inliers = (translation_errors <= translation_gate) & (rotation_errors_deg <= rotation_gate)
    inlier_count = int(np.sum(inliers))
    stats = {
        "candidate_poses": float(len(rotations)),
        "accepted_poses": float(inlier_count),
        "translation_p90_m": float(np.percentile(translation_errors[inliers], 90)) if inlier_count else math.inf,
        "rotation_p90_deg": float(np.percentile(rotation_errors_deg[inliers], 90)) if inlier_count else math.inf,
        "translation_gate_m": float(translation_gate),
        "rotation_gate_deg": float(rotation_gate),
        "consensus_fraction": float(inlier_count / len(rotations)),
    }
    if inlier_count < max(6, int(math.ceil(0.70 * len(rotations)))):
        return None, None, stats

    final_rotation = Rot.from_matrix(rotation_array[inliers]).mean().as_matrix()
    final_translation = np.median(translation_array[inliers], axis=0)
    final_translation_error = np.linalg.norm(translation_array[inliers] - final_translation, axis=1)
    final_rotation_error = np.degrees(
        (Rot.from_matrix(final_rotation).inv() * Rot.from_matrix(rotation_array[inliers])).magnitude()
    )
    stats["translation_p90_m"] = float(np.percentile(final_translation_error, 90))
    stats["rotation_p90_deg"] = float(np.percentile(final_rotation_error, 90))
    if stats["translation_p90_m"] > 0.10 or stats["rotation_p90_deg"] > 5.0:
        return None, None, stats

    baseline = float(np.linalg.norm(final_translation))
    if baseline < 0.02 or baseline > 2.0:
        stats["lever_arm_m"] = baseline
        return None, None, stats
    stats["lever_arm_m"] = baseline
    return final_rotation, final_translation, stats


def _rig_poses_at_offset(
    camera_samples: List[Dict[str, Any]],
    t_slam: np.ndarray,
    pos_slam: np.ndarray,
    slerp: Slerp,
    dt_sync: float,
    camera_offset: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Build per-frame LiDAR-to-camera poses for one camera-clock offset."""
    if not camera_samples:
        return [], []
    video_times = np.asarray([sample["video_time"] for sample in camera_samples], dtype=np.float64)
    query_times = t_slam[0] + (video_times - dt_sync - camera_offset)
    valid = (query_times >= t_slam[0]) & (query_times <= t_slam[-1])
    if not np.any(valid):
        return [], []

    query_times = query_times[valid]
    body_rotations = slerp(query_times).as_matrix()
    body_positions = np.column_stack(
        [np.interp(query_times, t_slam, pos_slam[:, axis]) for axis in range(3)]
    )
    camera_centers = np.asarray(
        [sample["center_lidar"] for sample, ok in zip(camera_samples, valid) if ok],
        dtype=np.float64,
    )
    camera_rotations = np.asarray(
        [sample["rotation_world_cam"] for sample, ok in zip(camera_samples, valid) if ok],
        dtype=np.float64,
    )
    body_rotations_t = np.transpose(body_rotations, (0, 2, 1))
    relative_rotations = np.einsum("nij,njk->nik", body_rotations_t, camera_rotations)
    relative_translations = np.einsum(
        "nij,nj->ni", body_rotations_t, camera_centers - body_positions
    )
    return list(relative_rotations), list(relative_translations)


def _estimate_auxiliary_time_offset(
    camera_samples: List[Dict[str, Any]],
    t_slam: np.ndarray,
    pos_slam: np.ndarray,
    slerp: Slerp,
    dt_sync: float,
    extraction_fps: float,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Estimate an unknown phone-video offset only when rigid-pose consensus is unique."""
    if len(camera_samples) < 8:
        return None, {"reason": "Fewer than eight timestamped auxiliary SfM poses"}

    video_times = np.asarray([sample["video_time"] for sample in camera_samples], dtype=np.float64)
    duration = float(t_slam[-1] - t_slam[0])
    low = float(np.min(video_times) - dt_sync - duration)
    high = float(np.max(video_times) - dt_sync)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return None, {"reason": "No overlapping phone-video and SLAM time range"}

    coarse_step = max(0.25, min(1.0, 1.0 / max(float(extraction_fps), 0.1)))

    def overlap_count(offset: float) -> int:
        query_times = t_slam[0] + (video_times - dt_sync - offset)
        return int(np.sum((query_times >= t_slam[0]) & (query_times <= t_slam[-1])))

    coarse_offsets = np.arange(low, high + coarse_step * 0.5, coarse_step)
    max_overlap = max((overlap_count(float(value)) for value in coarse_offsets), default=0)
    minimum_views = max(8, int(math.ceil(0.35 * len(camera_samples))), int(math.ceil(0.60 * max_overlap)))
    if max_overlap < minimum_views:
        return None, {
            "reason": "Insufficient timestamp overlap for a stable offset estimate",
            "maximum_overlapping_poses": max_overlap,
        }

    evaluated: Dict[float, Dict[str, Any]] = {}
    scored_candidates: Dict[float, Dict[str, Any]] = {}

    def evaluate(offset: float) -> None:
        key = round(float(offset), 4)
        if key in scored_candidates or overlap_count(key) < minimum_views:
            return
        rotations, translations = _rig_poses_at_offset(
            camera_samples, t_slam, pos_slam, slerp, dt_sync, key
        )
        if len(rotations) < minimum_views:
            return
        rotation, translation, quality = _pose_consensus(rotations, translations)
        coverage = len(rotations) / max_overlap
        translation_p90 = float(quality.get("translation_p90_m", math.inf))
        rotation_p90 = float(quality.get("rotation_p90_deg", math.inf))
        consensus_fraction = float(quality.get("consensus_fraction", 0.0))
        if not all(math.isfinite(value) for value in (
            translation_p90, rotation_p90, consensus_fraction, coverage
        )):
            return
        score = (
            translation_p90 / 0.05
            + rotation_p90 / 2.0
            + 2.0 * (1.0 - consensus_fraction)
            + 0.5 * (1.0 - coverage)
        )
        scored_candidates[key] = {
            "score": float(score),
            "quality": quality,
            "coverage": coverage,
            "passed": rotation is not None and translation is not None,
        }
        if rotation is not None and translation is not None:
            evaluated[key] = scored_candidates[key]

    for value in coarse_offsets:
        evaluate(float(value))
    if not scored_candidates:
        return None, {
            "reason": "No offset candidate passed the rigid-pose quality gates",
            "maximum_overlapping_poses": max_overlap,
        }

    # Coarse candidates are only seeds. Fast rig motion can make the valid
    # rigid-pose consensus window much narrower than either the video interval
    # or the coarse step, so refine the best separated coarse hypotheses even
    # when none of them has passed the final pose gates yet.
    coarse_ranked = sorted(
        scored_candidates,
        key=lambda value: scored_candidates[value]["score"],
    )
    seeds: List[float] = []
    for value in coarse_ranked:
        if all(abs(value - seed) >= coarse_step for seed in seeds):
            seeds.append(value)
        if len(seeds) >= 5:
            break

    fine_step = 0.01
    for seed in seeds:
        start = max(low, seed - coarse_step)
        end = min(high, seed + coarse_step)
        for value in np.arange(start, end + fine_step * 0.5, fine_step):
            evaluate(float(value))

    if not evaluated:
        best_coarse = scored_candidates[coarse_ranked[0]]
        return None, {
            "reason": "No offset candidate passed the rigid-pose quality gates",
            "maximum_overlapping_poses": max_overlap,
            "best_coarse_candidate_seconds": coarse_ranked[0],
            "best_coarse_score": best_coarse["score"],
            **best_coarse["quality"],
        }

    ranked = sorted(evaluated, key=lambda value: evaluated[value]["score"])
    best = ranked[0]
    # Compare against every separated time hypothesis, including candidates
    # whose fit missed a quality gate. A single narrow, strong minimum can be
    # observable; a constant-motion trajectory still has equally good scores
    # at separated offsets and remains ambiguous.
    competitors = [value for value in scored_candidates if abs(value - best) >= 1.0]
    if not competitors:
        return None, {
            "reason": "Offset minimum is not observable across a separated candidate",
            "candidate_count": len(evaluated),
            "best_candidate_seconds": best,
            "best_score": evaluated[best]["score"],
        }
    runner_up = min(competitors, key=lambda value: scored_candidates[value]["score"])
    best_score = float(evaluated[best]["score"])
    runner_up_score = float(scored_candidates[runner_up]["score"])
    required_margin = max(0.25, 0.35 * max(best_score, 0.1))
    if runner_up_score - best_score < required_margin:
        return None, {
            "reason": "Multiple time offsets produce comparable rigid-pose consensus",
            "candidate_count": len(evaluated),
            "best_candidate_seconds": best,
            "runner_up_seconds": runner_up,
            "best_score": best_score,
            "runner_up_score": runner_up_score,
            "runner_up_passed_quality_gates": scored_candidates[runner_up]["passed"],
        }

    return best, {
        "candidate_count": len(evaluated),
        "score": best_score,
        "runner_up_seconds": runner_up,
        "runner_up_score": runner_up_score,
        "runner_up_passed_quality_gates": scored_candidates[runner_up]["passed"],
        "overlapping_poses": int(overlap_count(best)),
        **evaluated[best]["quality"],
    }


def calibrate_third_camera_intrinsics(
    dataset_dir: Union[str, Path], third_config: ThirdCameraConfig
) -> ThirdCameraConfig:
    """Return a config updated with COLMAP-fitted intrinsics, when available.

    This stage only needs the sparse model. It can run immediately after SfM,
    before camera-to-LiDAR alignment, and never changes rig extrinsics.
    """
    updated_config = copy.deepcopy(third_config)
    report = copy.deepcopy(updated_config.calibration_report or {})
    report.setdefault("time_offset", {"status": "unknown", "source": "configuration"})
    report.setdefault("extrinsics", {"status": "uncalibrated", "source": "configuration"})
    report["intrinsics"] = {"status": "uncalibrated", "source": "COLMAP cameras.bin"}
    sparse_dir = Path(dataset_dir) / "sparse" / "0"
    images_bin = sparse_dir / "images.bin"
    cameras_bin = sparse_dir / "cameras.bin"
    if not images_bin.is_file() or not cameras_bin.is_file():
        report["intrinsics"]["reason"] = "COLMAP sparse camera or image model is missing"
        updated_config.calibration_report = report
        return updated_config

    try:
        from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images

        images = load_colmap_images(images_bin)
        camera_images = [
            image for image in images
            if str(image.get("name", "")).replace("\\", "/").startswith(
                f"{third_config.camera_name}/"
            )
        ]
        if len(camera_images) < 3:
            raise ValueError(f"Fewer than three registered {third_config.camera_name} SfM poses")
        camera_ids = [int(image["cam_id"]) for image in camera_images if "cam_id" in image]
        if not camera_ids:
            raise ValueError("Registered auxiliary poses do not include camera ids")
        values, counts = np.unique(camera_ids, return_counts=True)
        camera_id = int(values[int(np.argmax(counts))])
        camera_views = int(np.max(counts))
        if camera_views < max(3, int(math.ceil(0.8 * len(camera_images)))):
            raise ValueError("Auxiliary SfM images use multiple camera models")

        selected_model = str(third_config.camera_model or "PINHOLE").strip().upper()
        expected_model_ids = {"PINHOLE": 1, "OPENCV": 4, "OPENCV_FISHEYE": 5}
        camera = _read_colmap_cameras(cameras_bin)[camera_id]
        model_id = int(camera["model_id"])
        expected_model_id = expected_model_ids.get(selected_model)
        if expected_model_id is None or model_id != expected_model_id:
            fitted_model = _COLMAP_CAMERA_MODEL_NAMES.get(model_id, f"MODEL_{model_id}")
            report["intrinsics"] = {
                "status": "model_mismatch",
                "source": "COLMAP cameras.bin",
                "camera_id": camera_id,
                "registered_views": camera_views,
                "model_id": model_id,
                "selected_model": selected_model,
                "fitted_model": fitted_model,
                "expected_model_id": expected_model_id,
                "reason": (
                    f"COLMAP fitted {fitted_model} (model id {model_id}), but "
                    f"{selected_model} was selected; configured intrinsics were preserved."
                ),
            }
            updated_config.calibration_report = report
            print(
                f"[!] {third_config.camera_name} SfM intrinsics use {fitted_model}, but "
                f"{selected_model} was selected; keeping the configured model and intrinsics."
            )
            return updated_config

        model, intrinsics = _third_camera_intrinsics_from_colmap(camera)
        updated_config.camera_model = model
        updated_config.intrinsics = intrinsics
        report["intrinsics"] = {
            "status": "sfm_calibrated",
            "source": "COLMAP cameras.bin",
            "camera_id": camera_id,
            "registered_views": camera_views,
            "model_id": int(camera["model_id"]),
            "model": model,
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
            "cx": intrinsics["cx"],
            "cy": intrinsics["cy"],
        }
        print(
            f"[+] {third_config.camera_name} intrinsics from SfM: "
            f"{intrinsics['width']:.0f}x{intrinsics['height']:.0f}, "
            f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}"
        )
    except (ImportError, KeyError, OSError, ValueError, struct.error) as exc:
        report["intrinsics"].update(reason=str(exc))

    updated_config.calibration_report = report
    return updated_config


def align_third_camera_to_rig(
    dataset_dir: Union[str, Path],
    third_config: ThirdCameraConfig,
    rig_profile_path: Optional[Union[str, Path]] = None,
    fps: float = 2.0,
    alignment_data: Optional[Dict[str, Any]] = None,
    trajectory_path: Optional[Union[str, Path]] = None,
) -> ThirdCameraConfig:
    """Transfer SfM intrinsics and quality-gate metric cam2 extrinsics.

    Relative two-view matching has unknown translation scale and often pairs
    unsynchronized frames. It cannot establish a metric rig transform, so only
    registered SfM poses tied to a current COLMAP-to-LiDAR alignment are used.
    """
    del rig_profile_path  # kept for compatibility with older callers
    dataset_dir = Path(dataset_dir)
    updated_config = calibrate_third_camera_intrinsics(dataset_dir, third_config)
    images_bin = dataset_dir / "sparse" / "0" / "images.bin"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    report = copy.deepcopy(updated_config.calibration_report or {})
    report['camera_name'] = updated_config.camera_name
    for section, default in (
        ("intrinsics", {"status": "uncalibrated", "source": "configuration"}),
        ("time_offset", {"status": "unknown", "source": "configuration"}),
        ("extrinsics", {"status": "uncalibrated", "source": "configuration"}),
    ):
        if not isinstance(report.get(section), dict):
            report[section] = default.copy()
    configured_offset = float(updated_config.time_offset_s)
    offset_status = report["time_offset"].get("status")
    if offset_status not in ("calibrated", "verified"):
        offset_status = "configured" if abs(configured_offset) > 1e-9 else "unknown"
    report["time_offset"].update(
        seconds=configured_offset,
        applied=True,
        status=offset_status,
    )

    images: List[Dict[str, Any]] = []
    cam2_imgs: List[Dict[str, Any]] = []
    if images_bin.is_file():
        try:
            from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images

            images = load_colmap_images(images_bin)
            cam2_imgs = [
                im for im in images
                if str(im.get("name", "")).replace("\\", "/").startswith(
                    f"{third_config.camera_name}/"
                )
            ]
        except Exception as exc:
            report["intrinsics"].update(status="uncalibrated", reason=f"Could not read SfM poses: {exc}")

    # A metric extrinsic fit needs the current rigid alignment between SfM and
    # LiDAR. Never infer a metric baseline from essential-matrix direction.
    if not align_json.is_file():
        report["extrinsics"] = {
            "status": "uncalibrated",
            "source": "SfM poses",
            "reason": "colmap_to_lidar_alignment.json is missing",
        }
        updated_config.calibration_report = report
        print(
            f"[*] {third_config.camera_name} extrinsics remain uncalibrated: "
            "no COLMAP-to-LiDAR alignment is available."
        )
        return updated_config
    if len(cam2_imgs) < 6:
        report["extrinsics"] = {
            "status": "uncalibrated",
            "source": "SfM poses",
            "reason": "At least six registered auxiliary poses are required",
            "registered_views": len(cam2_imgs),
        }
        updated_config.calibration_report = report
        return updated_config

    try:
        from scripts.pipeline_auto_calibrator_and_colorizer import (
            CALIBRATION_VERSION,
            current_alignment,
            get_slam_trajectory_path,
            load_trajectory,
        )

        alignment = (alignment_data if alignment_data is not None
                     else current_alignment(dataset_dir, fps))
        if not isinstance(alignment, dict):
            raise ValueError("Current COLMAP-to-LiDAR alignment is unavailable")
        if alignment.get("calibration_version") != CALIBRATION_VERSION:
            raise ValueError("COLMAP-to-LiDAR alignment is stale")

        manifest_path = dataset_dir / "images" / "frames.json"
        time_source = "video_pts"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            time_source = manifest.get("time_source", time_source)
        if time_source == "insv_timelapse":
            from raven_app.timelapse_calibration import VERSION as TIMELAPSE_VERSION

            if (
                alignment.get("quality_status") != "accepted"
                or alignment.get("timelapse_calibration_version") != TIMELAPSE_VERSION
            ):
                raise ValueError("Timelapse COLMAP-to-LiDAR alignment is not currently accepted")

        scale = float(alignment["scale"])
        rotation_sim = np.asarray(alignment["R"], dtype=np.float64)
        translation_sim = np.asarray(alignment["t"], dtype=np.float64)
        dt_sync = float(alignment["dt_sync_seconds"])
        if (
            not math.isfinite(scale)
            or scale <= 0
            or rotation_sim.shape != (3, 3)
            or translation_sim.shape != (3,)
            or not np.all(np.isfinite(rotation_sim))
            or not np.all(np.isfinite(translation_sim))
            or not math.isfinite(dt_sync)
            or np.linalg.det(rotation_sim) <= 0
        ):
            raise ValueError("COLMAP-to-LiDAR alignment contains invalid parameters")

        trajectory_path = trajectory_path or get_slam_trajectory_path(dataset_dir)
        if not trajectory_path or not Path(trajectory_path).is_file():
            raise ValueError("LiDAR SLAM trajectory is missing")
        t_slam, pos_slam, rot_slam = load_trajectory(trajectory_path)
        t_slam = np.asarray(t_slam, dtype=np.float64)
        pos_slam = np.asarray(pos_slam, dtype=np.float64)
        if (
            t_slam.ndim != 1
            or len(t_slam) < 2
            or pos_slam.shape != (len(t_slam), 3)
            or len(rot_slam) != len(t_slam)
            or not np.all(np.isfinite(t_slam))
            or not np.all(np.isfinite(pos_slam))
            or np.any(np.diff(t_slam) <= 0)
        ):
            raise ValueError("LiDAR SLAM trajectory is malformed")
        slerp = Slerp(t_slam, rot_slam)

        camera_samples = []
        missing_timestamps = 0
        for image in cam2_imgs:
            try:
                video_time = frame_time(dataset_dir, image["name"], fps)
            except (KeyError, OSError, ValueError):
                missing_timestamps += 1
                continue
            center_lidar = scale * (rotation_sim @ np.asarray(image["C"], dtype=np.float64)) + translation_sim
            rotation_cw_lidar = np.asarray(image["R_cw"], dtype=np.float64) @ rotation_sim.T
            camera_samples.append({
                "video_time": float(video_time),
                "center_lidar": center_lidar,
                "rotation_world_cam": rotation_cw_lidar.T,
            })

        time_report = report["time_offset"]
        offset_status = time_report.get("status", "unknown")
        auto_offset_quality = None
        if abs(configured_offset) <= 1e-9 and offset_status not in ("calibrated", "verified"):
            estimated_offset, auto_offset_quality = _estimate_auxiliary_time_offset(
                camera_samples, t_slam, pos_slam, slerp, dt_sync, updated_config.fps
            )
            if estimated_offset is None:
                time_report.update(
                    status="unknown",
                    applied=False,
                    reason=auto_offset_quality.get("reason", "Auxiliary offset is ambiguous"),
                    **{key: value for key, value in auto_offset_quality.items() if key != "reason"},
                )
                report["extrinsics"] = {
                    "status": "uncalibrated",
                    "source": "SfM poses and current COLMAP-to-LiDAR alignment",
                    "reason": "Auxiliary camera time offset is unknown or ambiguous",
                    "missing_timestamps": missing_timestamps,
                    **auto_offset_quality,
                }
                updated_config.calibration_report = report
                print(
                    f"[*] {third_config.camera_name} extrinsics remain uncalibrated: "
                    "phone-video time offset is unknown or ambiguous."
                )
                return updated_config
            configured_offset = float(estimated_offset)
            updated_config.time_offset_s = configured_offset
            time_report.update(
                status="calibrated",
                source="rig-pose consensus search",
                estimated_seconds=configured_offset,
                **auto_offset_quality,
            )

        offset_total = dt_sync + configured_offset
        rotations_lidar_cam, translations_lidar_cam = _rig_poses_at_offset(
            camera_samples, t_slam, pos_slam, slerp, dt_sync, configured_offset
        )

        rotation_lc, translation_lc, quality = _pose_consensus(
            rotations_lidar_cam, translations_lidar_cam
        )
        report["time_offset"].update(
            dt_sync_seconds=dt_sync,
            total_applied_offset_seconds=offset_total,
            applied=True,
            source=time_report.get("source", "COLMAP alignment plus auxiliary config"),
        )
        if rotation_lc is None or translation_lc is None:
            report["extrinsics"] = {
                "status": "uncalibrated",
                "source": "SfM poses and current COLMAP-to-LiDAR alignment",
                "reason": "Per-frame rigid-pose consensus failed",
                "missing_timestamps": missing_timestamps,
                **quality,
            }
            updated_config.calibration_report = report
            print(
                f"[*] {third_config.camera_name} metric extrinsics remain uncalibrated: "
                f"rig-pose consensus failed ({int(quality.get('accepted_poses', 0))}/"
                f"{int(quality.get('candidate_poses', 0))} accepted poses)."
            )
            return updated_config

        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation_lc
        transform[:3, 3] = translation_lc
        updated_config.T_lidar_to_cam2_rigid_4x4 = transform.tolist()
        report["extrinsics"] = {
            "status": "calibrated",
            "source": "SfM poses and current COLMAP-to-LiDAR alignment",
            "alignment_dt_sync_seconds": dt_sync,
            "auxiliary_time_offset_seconds": configured_offset,
            **quality,
        }
        updated_config.calibration_report = report
        print(
            f"[+] {third_config.camera_name} metric extrinsics calibrated from "
            f"{int(quality['accepted_poses'])}/{int(quality['candidate_poses'])} poses; "
            f"lever arm {quality['lever_arm_m'] * 100:.2f} cm."
        )
        return updated_config
    except Exception as exc:
        report["extrinsics"] = {
            "status": "uncalibrated",
            "source": "SfM poses and COLMAP-to-LiDAR alignment",
            "reason": str(exc),
        }
        updated_config.calibration_report = report
        print(f"[*] {third_config.camera_name} metric extrinsics remain uncalibrated: {exc}")
        return updated_config


def synthesize_third_camera_poses(
    dataset_dir: Union[str, Path],
    third_config: ThirdCameraConfig,
    fps: float = 2.0,
    dt_sync: float = 0.0,
) -> List[Dict[str, Any]]:
    """Synthesize 6DoF camera poses for cam2 frames along the SLAM trajectory."""
    dataset_dir = Path(dataset_dir)
    from scripts.pipeline_auto_calibrator_and_colorizer import (
        get_slam_trajectory_path,
        load_trajectory,
    )

    trj_path = get_slam_trajectory_path(dataset_dir)
    if not trj_path or not Path(trj_path).is_file():
        return []

    t_slam, pos_slam, rot_slam = load_trajectory(trj_path)
    slerp = Slerp(t_slam, rot_slam)

    t_start, t_end = t_slam[0], t_slam[-1]
    cam2_dir = dataset_dir / "images" / third_config.camera_name
    cam2_files = sorted(cam2_dir.glob("*.jpg"))

    T_LC2 = np.array(third_config.T_lidar_to_cam2_rigid_4x4)
    R_LC2, t_LC2 = T_LC2[:3, :3], T_LC2[:3, 3]

    poses: List[Dict[str, Any]] = []
    dt_total = dt_sync + third_config.time_offset_s

    for img_file in cam2_files:
        rel_name = f"{third_config.camera_name}/{img_file.name}"
        t_vid = frame_time(dataset_dir, rel_name, fps)
        t_q = t_start + (t_vid - dt_total)

        if not (t_start <= t_q <= t_end):
            continue

        idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
        w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L = slerp(t_q).as_matrix()

        p_cam = p_L + R_L @ t_LC2
        R_world_cam = R_L @ R_LC2
        R_cw = R_world_cam.T
        t_cw = -R_cw @ p_cam
        q_cw = Rot.from_matrix(R_cw).as_quat()  # x, y, z, w

        poses.append({
            "name": rel_name,
            "R_cw": R_cw,
            "t_cw": t_cw,
            "q_cw": q_cw,
            "C": p_cam,
            "camera_id": 3,
        })

    return poses


def inject_third_camera_into_colmap(
    sparse_out: Path,
    third_config: ThirdCameraConfig,
    poses: List[Dict[str, Any]],
    camera_id: int = 3,
) -> None:
    """Inject Camera 3 and its synthesized or aligned poses into sparse/0/cameras.txt and images.txt."""
    sparse_out = Path(sparse_out)
    sparse_out.mkdir(parents=True, exist_ok=True)
    cameras_txt = sparse_out / "cameras.txt"
    images_txt = sparse_out / "images.txt"

    # 1. Update cameras.txt
    int2 = third_config.intrinsics
    w = int(int2["width"])
    h = int(int2["height"])
    fx, fy = int2["fx"], int2["fy"]
    cx, cy = int2["cx"], int2["cy"]

    cam_model = third_config.camera_model.upper()
    if cam_model == "PINHOLE":
        cam_line = f"{camera_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"
    elif cam_model == "OPENCV":
        k1, k2, p1, p2 = int2.get("k1", 0.0), int2.get("k2", 0.0), int2.get("p1", 0.0), int2.get("p2", 0.0)
        cam_line = f"{camera_id} OPENCV {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n"
    elif cam_model == "OPENCV_FISHEYE":
        k1, k2 = int2.get("k1", 0.0), int2.get("k2", 0.0)
        k3, k4 = int2.get("k3", 0.0), int2.get("k4", 0.0)
        cam_line = f"{camera_id} OPENCV_FISHEYE {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {k3} {k4}\n"
    else:
        cam_line = f"{camera_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"

    existing_cams = cameras_txt.read_text(encoding="utf-8") if cameras_txt.is_file() else "# Camera list\n"
    if f"\n{camera_id} " not in f"\n{existing_cams}":
        with open(cameras_txt, "a", encoding="utf-8") as f:
            f.write(cam_line)
        print(f"  [+] Injected camera {camera_id} ({cam_model}) into cameras.txt")

    # 2. Update images.txt with third camera poses
    if poses and images_txt.is_file():
        existing_images = images_txt.read_text(encoding="utf-8")
        lines_to_add = []
        max_id = 0
        for line in existing_images.splitlines():
            line_s = line.strip()
            if line_s and not line_s.startswith("#"):
                parts = line_s.split()
                if len(parts) >= 9 and parts[0].isdigit():
                    max_id = max(max_id, int(parts[0]))

        cur_id = max_id + 1
        for p in poses:
            if p["name"] not in existing_images:
                q = p["q_cw"]  # x, y, z, w
                t = p["t_cw"]
                lines_to_add.append(
                    f"{cur_id} {q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {camera_id} {p['name']}\n\n"
                )
                cur_id += 1

        if lines_to_add:
            with open(images_txt, "a", encoding="utf-8") as f:
                f.writelines(lines_to_add)
            print(f"  [+] Injected {len(lines_to_add)} poses for {third_config.camera_name} into images.txt")
