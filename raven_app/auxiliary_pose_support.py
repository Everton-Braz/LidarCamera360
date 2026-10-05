"""Validated per-image auxiliary camera poses from accepted sample SfM models."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


SUPPORT_VERSION = 2
SUPPORT_RELATIVE_PATH = Path("calibration") / "auxiliary_pose_support.json"


def sfm_pose_source_signature(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    """Identify the aligned COLMAP model whose per-image poses are being reused."""
    root = Path(dataset_dir).resolve()
    files = {
        "alignment": root / "colmap_to_lidar_alignment.json",
        "cameras": root / "sparse" / "0" / "cameras.bin",
        "images": root / "sparse" / "0" / "images.bin",
    }
    signature = {}
    for key, path in files.items():
        try:
            stat = path.stat()
        except OSError:
            return None
        if not path.is_file():
            return None
        signature[key] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return signature


def _active_trajectory_signature(dataset_dir: Path, trajectory_path=None) -> Dict[str, Any]:
    from scripts.pipeline_auto_calibrator_and_colorizer import (
        _active_slam_signature,
        get_slam_trajectory_path,
    )

    trajectory = Path(trajectory_path) if trajectory_path is not None else get_slam_trajectory_path(dataset_dir)
    signature = _active_slam_signature(
        dataset_dir, trajectory_path=trajectory, include_cloud=False)
    stat = trajectory.stat()
    signature['trajectory_size'] = stat.st_size
    return signature


def write_auxiliary_pose_support(
    dataset_dir: Path,
    cameras: Dict[str, Dict[str, Any]],
    *,
    trajectory_path=None,
    sfm_source_signature: Optional[Dict[str, Any]] = None,
) -> Path:
    """Atomically publish only camera poses backed by accepted, aligned SfM."""
    root = Path(dataset_dir).resolve()
    target = root / SUPPORT_RELATIVE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_cameras = {}
    for camera_name, source in cameras.items():
        if not isinstance(source, dict) or source.get('status') != 'accepted':
            continue
        minimum = max(6, int(source.get('minimum_poses', 6)))
        poses = []
        seen = set()
        for item in source.get('poses', []):
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).replace('\\', '/')
            relative = Path(name)
            if (not name.startswith(f'{camera_name}/') or relative.is_absolute()
                    or '..' in relative.parts or name in seen):
                continue
            image_path = root / 'images' / relative
            if not image_path.is_file():
                continue
            try:
                rotation = np.asarray(item['R_cw'], dtype=np.float64)
                center = np.asarray(item['C'], dtype=np.float64)
                video_time = float(item['video_time'])
            except (KeyError, TypeError, ValueError):
                continue
            if (rotation.shape != (3, 3) or center.shape != (3,)
                    or not np.all(np.isfinite(rotation))
                    or not np.all(np.isfinite(center)) or not np.isfinite(video_time)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                    or np.linalg.det(rotation) <= 0):
                continue
            stat = image_path.stat()
            pose = dict(item)
            pose.update(name=name, R_cw=rotation.tolist(), C=center.tolist(),
                        video_time=video_time,
                        image_signature={'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
            poses.append(pose)
            seen.add(name)
        if len(poses) < minimum:
            raise ValueError(
                f'{camera_name} auxiliary SfM pose support has only {len(poses)} valid images; '
                f'{minimum} are required')
        record = dict(source)
        record.update(minimum_poses=minimum, poses=poses)
        safe_cameras[str(camera_name)] = record

    if not safe_cameras:
        raise ValueError('No validated auxiliary SfM poses were available to publish')
    manifest = root / 'images' / 'frames.json'
    manifest_signature = None
    if manifest.is_file():
        stat = manifest.stat()
        manifest_signature = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    payload = {
        "version": SUPPORT_VERSION,
        "source": "registered_sfm_poses_aligned_to_lidar",
        "trajectory_signature": _active_trajectory_signature(root, trajectory_path),
        "frame_manifest_signature": manifest_signature,
        "cameras": safe_cameras,
    }
    if sfm_source_signature is not None:
        current_sfm_signature = sfm_pose_source_signature(root)
        if current_sfm_signature is None or current_sfm_signature != sfm_source_signature:
            raise ValueError("The aligned COLMAP model changed before auxiliary poses were published")
        payload["sfm_pose_source_signature"] = current_sfm_signature
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def load_camera_pose_support(
    dataset_dir: Path,
    camera_name: str,
    *,
    camera_model: Optional[str] = None,
    intrinsics: Optional[Dict[str, Any]] = None,
    trajectory_path=None,
    require_sfm_source_signature: bool = False,
) -> Optional[Dict[str, Any]]:
    """Load fresh, safe, metric per-image poses for one auxiliary camera."""
    root = Path(dataset_dir).resolve()
    path = root / SUPPORT_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != SUPPORT_VERSION:
            return None
        saved_signature = payload.get("trajectory_signature")
        if not isinstance(saved_signature, dict):
            return None
        if saved_signature != _active_trajectory_signature(root, trajectory_path):
            return None
        saved_sfm_signature = payload.get("sfm_pose_source_signature")
        if require_sfm_source_signature and not isinstance(saved_sfm_signature, dict):
            return None
        if (saved_sfm_signature is not None
                and saved_sfm_signature != sfm_pose_source_signature(root)):
            return None
        manifest = root / 'images' / 'frames.json'
        saved_manifest_signature = payload.get('frame_manifest_signature')
        if saved_manifest_signature is not None:
            if not manifest.is_file():
                return None
            stat = manifest.stat()
            if (saved_manifest_signature.get('size') != stat.st_size
                    or saved_manifest_signature.get('mtime_ns') != stat.st_mtime_ns):
                return None
        record = payload.get("cameras", {}).get(str(camera_name))
        if (not isinstance(record, dict) or record.get("status") != "accepted"
                or record.get('alignment_quality_status') != 'accepted'):
            return None
        model = str(record.get("camera_model", "")).strip().upper()
        if camera_model and model != str(camera_model).strip().upper():
            return None
        saved_intrinsics = record.get("intrinsics")
        if not isinstance(saved_intrinsics, dict):
            return None
        numeric_intrinsics = {str(key): float(value) for key, value in saved_intrinsics.items()}
        if not all(np.isfinite(value) for value in numeric_intrinsics.values()):
            return None
        if any(numeric_intrinsics.get(key, 0.0) <= 0 for key in ("width", "height", "fx", "fy")):
            return None
        if intrinsics is not None:
            comparable = ('width', 'height', 'fx', 'fy', 'cx', 'cy',
                          'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')
            try:
                for key in comparable:
                    saved_value = numeric_intrinsics.get(key, 0.0)
                    current_value = float(intrinsics.get(key, 0.0))
                    if not np.isclose(saved_value, current_value, rtol=1e-8, atol=1e-8):
                        return None
            except (TypeError, ValueError):
                return None

        poses = []
        seen = set()
        for item in record.get("poses", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).replace("\\", "/")
            if not name.startswith(f"{camera_name}/") or name in seen:
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            image_path = root / "images" / relative
            if not image_path.is_file():
                continue
            image_signature = item.get('image_signature')
            if not isinstance(image_signature, dict):
                continue
            stat = image_path.stat()
            if (image_signature.get('size') != stat.st_size
                    or image_signature.get('mtime_ns') != stat.st_mtime_ns):
                continue
            try:
                rotation = np.asarray(item["R_cw"], dtype=np.float64)
                center = np.asarray(item["C"], dtype=np.float64)
                video_time = float(item["video_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if (rotation.shape != (3, 3) or center.shape != (3,)
                    or not np.all(np.isfinite(rotation))
                    or not np.all(np.isfinite(center))
                    or not np.isfinite(video_time)
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                    or np.linalg.det(rotation) <= 0):
                continue
            pose = dict(item)
            pose.update(name=name, R_cw=rotation, C=center, video_time=video_time,
                        image_path=image_path)
            poses.append(pose)
            seen.add(name)
        minimum = max(6, int(record.get("minimum_poses", 6)))
        if len(poses) < minimum:
            return None
        return {**record, "camera_model": model, "intrinsics": numeric_intrinsics,
                "poses": poses, "path": path}
    except (OSError, ValueError, TypeError, KeyError):
        return None
