"""Astronomical solar compass from calibrated 360-degree camera frames.

A gravity-aligned SLAM scan has an unknown horizontal yaw, while every photo's
3D orientation in the SLAM frame is known from COLMAP or the calibrated rig
trajectory. Given the capture's GPS coordinates and UTC timestamps, the sun's
exact astronomical elevation and true azimuth are known to ~0.01 deg.

Detecting unoccluded sun disks in the calibrated images, checking that their
measured elevation matches the astronomical solar elevation, and clustering the
resulting 3D world-frame rays across many camera poses recovers true north
without relying on GPS track length or satellite imagery alone. Indoor lights,
clouds, and roof-clipped glares fail the circularity, sky-halo, elevation, or
3D directional-constancy gates and are rejected cleanly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from raven_app.pose_ba import read_images_txt, unproject_thin_prism

# Astronomical elevation limits: below 5 deg horizon clutter dominates; above 84 deg
# the 1/cos(alt) azimuth sensitivity exceeds 10x and heading becomes degenerate.
MIN_SUN_ALT_DEG = 5.0
MAX_SUN_ALT_DEG = 84.0
# Per-blob unoccluded sky-halo detector gates (evaluated on ~480 px downsampled frames).
SAT_THRESHOLD = 252
MIN_BLOB_AREA_PX = 8
MAX_BLOB_AREA_FRAC = 0.15
MIN_CORE_RADIUS_PX = 2.0
MAX_CIRCULARITY = 2.2
MIN_HALO_RING_P10 = 180.0
MAX_LENS_ANGLE_DEG = 86.0
MAX_OBS_ELEV_ERR_DEG = 3.5
# 3D spherical consensus gates.
CLUSTER_CONE_DEG = 3.5
RIVAL_SEPARATION_DEG = 15.0
MIN_SOLAR_INLIERS = 4
MAX_RIVAL_RATIO = 0.45
MAX_MEDIAN_ELEV_ERR_DEG = 3.0
MAX_3D_RMS_DEG = 2.5
MAX_SOLAR_SIGMA_DEG = 15.0
# Finite-distance parallax rejection (indoor skylights / ceiling fixtures converge at finite range).
# A 3 m ceiling light inside the +/-3.5 deg cluster cone has RMS baseline ~0.11 m.
PARALLAX_MIN_BASELINE_M = 0.08
PARALLAX_MAX_FINITE_DIST_M = 50.0
PARALLAX_MIN_R2 = 0.55
PARALLAX_MIN_RMS_DROP_DEG = 0.4
# Standalone high-confidence solar compass gates (e.g. offline mode without imagery).
STRONG_MIN_INLIERS = 10
STRONG_MAX_RIVAL_RATIO = 0.25
STRONG_MAX_ELEV_ERR_DEG = 1.5
STRONG_MAX_SIGMA_DEG = 5.0


def solar_position(lat_deg: float, lon_deg: float, epoch_utc: float) -> Tuple[float, float]:
    """Return (altitude_deg, azimuth_true_deg) of the sun via NOAA/Meeus ephemeris.

    ``altitude_deg`` includes standard atmospheric refraction above the horizon.
    ``azimuth_true_deg`` is clockwise from true north in [0, 360).
    """
    jd = float(epoch_utc) / 86400.0 + 2440587.5
    t = (jd - 2451545.0) / 36525.0
    mean_lon = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    mean_anom = (357.52911 + t * (35999.05029 - 0.0001537 * t)) % 360.0
    m_rad = math.radians(mean_anom)
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    eq_center = (
        (1.914602 - t * (0.004817 + 0.000014 * t)) * math.sin(m_rad)
        + (0.019993 - 0.000101 * t) * math.sin(2.0 * m_rad)
        + 0.000289 * math.sin(3.0 * m_rad)
    )
    true_lon = mean_lon + eq_center
    omega = 125.04 - 1934.136 * t
    app_lon = true_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    obliq0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    obliq = obliq0 + 0.00256 * math.cos(math.radians(omega))
    sin_dec = math.sin(math.radians(obliq)) * math.sin(math.radians(app_lon))
    dec = math.asin(max(-1.0, min(1.0, sin_dec)))

    y = math.tan(math.radians(obliq / 2.0)) ** 2
    l0_rad = math.radians(mean_lon)
    eq_time_min = 4.0 * math.degrees(
        y * math.sin(2.0 * l0_rad)
        - 2.0 * ecc * math.sin(m_rad)
        + 4.0 * ecc * y * math.sin(m_rad) * math.cos(2.0 * l0_rad)
        - 0.5 * y * y * math.sin(4.0 * l0_rad)
        - 1.25 * ecc * ecc * math.sin(2.0 * m_rad)
    )
    utc_min = (float(epoch_utc) % 86400.0) / 60.0
    true_solar_min = (utc_min + eq_time_min + 4.0 * float(lon_deg)) % 1440.0
    hour_angle = math.radians(true_solar_min / 4.0 - 180.0)
    lat = math.radians(float(lat_deg))

    sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(hour_angle)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    geom_alt_rad = math.asin(sin_alt)
    alt_deg = math.degrees(geom_alt_rad)
    if alt_deg > -0.575:
        alt_deg += 1.02 / math.tan(math.radians(alt_deg + 10.3 / (alt_deg + 5.11))) / 60.0

    denom = max(math.cos(lat) * math.cos(geom_alt_rad), 1e-12)
    cos_az = max(-1.0, min(1.0, (math.sin(dec) - math.sin(lat) * sin_alt) / denom))
    az_deg = math.degrees(math.acos(cos_az))
    if math.sin(hour_angle) > 0.0:
        az_deg = 360.0 - az_deg
    return float(alt_deg), float(az_deg % 360.0)


def unproject_pixels(uv: np.ndarray, model: str, params: np.ndarray,
                     width: int = 0, height: int = 0) -> np.ndarray:
    """Return unit 3D camera-frame bearing vectors for 2D pixel coordinates ``uv``."""
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    name = str(model).upper()
    if name == 'THIN_PRISM_FISHEYE' and len(params) >= 12:
        return unproject_thin_prism(uv, params[:12])
    if name == 'OPENCV_FISHEYE' and len(params) >= 4:
        fx, fy, cx, cy = params[:4]
        k1, k2, k3, k4 = (list(params[4:8]) + [0.0] * 4)[:4]
        xd = (uv[:, 0] - cx) / fx
        yd = (uv[:, 1] - cy) / fy
        theta_d = np.hypot(xd, yd)
        theta = theta_d.copy()
        for _ in range(12):
            t2 = theta * theta
            f = theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - theta_d
            df = 1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)))
            theta = theta - f / np.where(np.abs(df) < 1e-9, 1e-9, df)
        scale = np.where(theta_d > 1e-12, np.sin(theta) / np.maximum(theta_d, 1e-12), 1.0)
        return np.stack([xd * scale, yd * scale, np.cos(theta)], axis=1)
    if name in ('PINHOLE', 'SIMPLE_PINHOLE', 'OPENCV', 'SIMPLE_RADIAL', 'RADIAL') and len(params) >= 3:
        if name in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL'):
            fx = fy = float(params[0])
            cx, cy = float(params[1]), float(params[2])
        else:
            fx, fy, cx, cy = [float(x) for x in params[:4]]
        x = (uv[:, 0] - cx) / fx
        y = (uv[:, 1] - cy) / fy
        vec = np.stack([x, y, np.ones_like(x)], axis=1)
        return vec / np.linalg.norm(vec, axis=1, keepdims=True)
    if name == 'EQUIRECTANGULAR' and width > 0 and height > 0:
        lon = (uv[:, 0] / float(width) - 0.5) * 2.0 * math.pi
        lat = (0.5 - uv[:, 1] / float(height)) * math.pi
        return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat), np.cos(lat) * np.cos(lon)], axis=1)
    return np.full((len(uv), 3), np.nan, dtype=np.float64)


def _read_cameras_with_dims(path: Path) -> Dict[int, Tuple[str, int, int, np.ndarray]]:
    cameras: Dict[int, Tuple[str, int, int, np.ndarray]] = {}
    for line in path.read_text(encoding='utf-8', errors='ignore').splitlines():
        parts = line.split()
        if len(parts) < 4 or line.startswith('#'):
            continue
        model = parts[1].upper()
        if model != 'EQUIRECTANGULAR' and len(parts) < 5:
            continue
        try:
            cam_id = int(parts[0])
            width = int(parts[2])
            height = int(parts[3])
            params = np.array(parts[4:], dtype=np.float64)
        except ValueError:
            continue
        cameras[cam_id] = (model, width, height, params)
    return cameras


def _colmap_model_paths(candidate: Path) -> Optional[Tuple[Path, Path, Path]]:
    """Return (cameras.txt, images.txt, images_dir) inside a COLMAP dataset folder, or None."""
    if not candidate.is_dir():
        return None
    images_dir = candidate / 'images'
    if not images_dir.is_dir():
        return None
    for sparse in (candidate / 'sparse' / '0', candidate / 'sparse', candidate):
        cams = sparse / 'cameras.txt'
        imgs = sparse / 'images.txt'
        if cams.is_file() and imgs.is_file():
            return cams, imgs, images_dir
    return None


def _is_workflow_run_config(data: Any) -> bool:
    """True for a workflow run configuration, False for a SLAM profile marker inside ``slam_out``."""
    return isinstance(data, dict) and any(
        k in data for k in ('runs', 'schema_version', 'fps', 'extraction', 'settings', 'paths')
    )


def _read_workflow_config(folder: Path) -> Dict[str, Any]:
    cfg = folder / 'workflow_config.json'
    if not cfg.is_file():
        return {}
    try:
        data = json.loads(cfg.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return data if _is_workflow_run_config(data) else {}


def _is_run_root(folder: Path) -> bool:
    """True when ``folder`` is a self-contained workflow/project run directory."""
    return bool(
        _read_workflow_config(folder)
        or (folder / 'project.json').is_file()
        or (folder / 'images' / 'cam0').is_dir()
        or (folder / 'rig_calibration.json').is_file()
        or (folder / 'rig_calibration_photometric.json').is_file()
    )


def find_colmap_dir(paths: Sequence[Any]) -> Optional[Path]:
    """Locate a COLMAP export directory belonging to the scan's run folder."""
    seen = set()
    for raw in filter(None, paths):
        start = Path(raw).resolve()
        if _colmap_model_paths(start) is not None:
            return start
        base = start if start.is_dir() else start.parent
        for folder in (base, *base.parents)[:5]:
            if folder not in seen:
                seen.add(folder)
                if _colmap_model_paths(folder) is not None:
                    return folder
                colmap_3dgs = folder / 'colmap_3dgs'
                if _colmap_model_paths(colmap_3dgs) is not None:
                    return colmap_3dgs
            # Do not walk past this run's own root into a parent dataset directory that may hold an older SLAM run.
            if _is_run_root(folder):
                break
    return None


def _frame_number(name: str) -> int:
    digits = ''.join(ch for ch in Path(name).name if ch.isdigit())
    return int(digits) if digits else 0


def _read_frame_timestamps(candidates: Sequence[Path]) -> Dict[str, float]:
    """Read per-frame video presentation timestamps from ``images/frames.json`` (or legacy ``frame_timestamps.json``)."""
    for ts_file in candidates:
        if not ts_file.is_file():
            continue
        try:
            raw = json.loads(ts_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        mapping = raw.get('timestamps') if isinstance(raw.get('timestamps'), dict) else raw
        parsed = {
            str(k).replace('\\', '/'): float(v)
            for k, v in mapping.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
        }
        if parsed:
            return parsed
    return {}


def _workflow_fps(cfg: Dict[str, Any]) -> Optional[float]:
    """Extract frame extraction FPS from either real multi-run ``workflow_config.json`` or legacy flat configs."""
    runs = cfg.get('runs')
    if isinstance(runs, list) and runs:
        last = runs[-1]
        if isinstance(last, dict):
            sigs = (last.get('image_extraction') or {}).get('camera_signatures')
            if isinstance(sigs, dict):
                for cam_sig in sigs.values():
                    val = cam_sig.get('fps') if isinstance(cam_sig, dict) else None
                    if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                        return float(val)
            for section in (last.get('settings'), last.get('extraction'), last):
                val = section.get('fps') if isinstance(section, dict) else None
                if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                    return float(val)
    for section in (cfg, cfg.get('extraction')):
        val = section.get('fps') if isinstance(section, dict) else None
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
            return float(val)
    return None


def _workflow_calibration_path(cfg: Dict[str, Any]) -> Optional[Path]:
    runs = cfg.get('runs')
    if isinstance(runs, list) and runs and isinstance(runs[-1], dict):
        cal_used = runs[-1].get('calibration_used')
        if isinstance(cal_used, dict) and cal_used.get('path'):
            cand = Path(str(cal_used['path']))
            if cand.is_file():
                return cand
    return None


def _load_colmap_frames(colmap_dir: Path) -> List[Dict[str, Any]]:
    resolved = _colmap_model_paths(colmap_dir)
    if resolved is None:
        return []
    cams_path, imgs_path, images_dir = resolved
    cameras = _read_cameras_with_dims(cams_path)
    ts_map = _read_frame_timestamps((
        images_dir / 'frames.json',
        colmap_dir / 'frames.json',
        colmap_dir.parent / 'images' / 'frames.json',
        images_dir / 'frame_timestamps.json',
        colmap_dir.parent / 'frame_timestamps.json',
    ))
    frames: List[Dict[str, Any]] = []
    for _, R_cw, t_cw, cam_id, name in read_images_txt(imgs_path):
        if cam_id not in cameras:
            continue
        model, width, height, params = cameras[cam_id]
        # Prefer wide/360 cameras where the sky is visible; ignore narrow pinhole cams when wide cams exist.
        img_path = images_dir / name
        if not img_path.is_file():
            continue
        norm_name = str(name).replace('\\', '/')
        entry: Dict[str, Any] = {
            'path': img_path,
            'name': norm_name,
            'frame_num': _frame_number(norm_name),
            'R_cw': R_cw,
            'cam_pos': -R_cw.T @ t_cw,
            'model': model,
            'width': width,
            'height': height,
            'params': params,
        }
        if norm_name in ts_map:
            entry['video_time_s'] = ts_map[norm_name]
        frames.append(entry)
    wide = [f for f in frames if f['model'] in ('THIN_PRISM_FISHEYE', 'OPENCV_FISHEYE', 'EQUIRECTANGULAR')]
    return wide if wide else frames


def _intrinsics_to_params(intr: Dict[str, Any], lens_model: str) -> Optional[Tuple[str, int, int, np.ndarray]]:
    if not isinstance(intr, dict):
        return None
    model = str(intr.get('camera_model') or lens_model or 'THIN_PRISM_FISHEYE').upper()
    width = int(intr.get('width', 0))
    height = int(intr.get('height', 0))
    if width <= 0 or height <= 0:
        return None
    if model == 'THIN_PRISM_FISHEYE':
        keys = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')
        if not all(k in intr for k in ('fx', 'fy', 'cx', 'cy')):
            return None
        return model, width, height, np.array([float(intr.get(k, 0.0)) for k in keys], dtype=np.float64)
    if model == 'OPENCV_FISHEYE':
        keys = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4')
        if not all(k in intr for k in ('fx', 'fy', 'cx', 'cy')):
            return None
        return model, width, height, np.array([float(intr.get(k, 0.0)) for k in keys], dtype=np.float64)
    if model == 'EQUIRECTANGULAR':
        return model, width, height, np.zeros(0, dtype=np.float64)
    return None


def _load_direct_rig_frames(paths: Sequence[Any], trajectory_path: Optional[Path],
                            dt_sync: Optional[float]) -> List[Dict[str, Any]]:
    """Fallback when colmap_3dgs does not exist yet: build camera poses from rig_calibration.json + trajectory."""
    if trajectory_path is None or not Path(trajectory_path).is_file() or dt_sync is None:
        return []
    try:
        rows = np.loadtxt(str(trajectory_path), ndmin=2)
    except (OSError, ValueError):
        return []
    if rows.shape[1] < 8 or len(rows) < 2:
        return []
    # Deduplicate strictly increasing timestamps for Slerp
    _, keep = np.unique(rows[:, 0], return_index=True)
    rows = rows[np.sort(keep)]
    if len(rows) < 2:
        return []

    rig_data = None
    images_dir = None
    wf_cfg: Dict[str, Any] = {}
    seen = set()
    for raw in filter(None, (trajectory_path, *paths)):
        start = Path(raw).resolve()
        base = start if start.is_dir() else start.parent
        for folder in (base, *base.parents)[:5]:
            if folder in seen:
                continue
            seen.add(folder)
            if not wf_cfg:
                wf_cfg = _read_workflow_config(folder)
            if rig_data is None:
                cal_candidates: List[Path] = []
                wf_cal = _workflow_calibration_path(wf_cfg)
                if wf_cal is not None:
                    cal_candidates.append(wf_cal)
                cal_candidates.extend(
                    folder / name
                    for name in ('rig_calibration_photometric.json', 'rig_calibration.json', 'calibracao_rigida_auto.json')
                )
                for cand in cal_candidates:
                    if cand.is_file():
                        try:
                            data = json.loads(cand.read_text(encoding='utf-8'))
                            if (isinstance(data, dict) and 'T_lidar_to_cam0_rigid_4x4' in data
                                    and data.get('valid') is not False):
                                rig_data = data
                                break
                        except (OSError, ValueError):
                            pass
            if images_dir is None and (folder / 'images' / 'cam0').is_dir():
                images_dir = folder / 'images'
            if (rig_data is not None and images_dir is not None) or _is_run_root(folder):
                break
    if rig_data is None or images_dir is None:
        return []

    ts_map = _read_frame_timestamps((
        images_dir / 'frames.json',
        images_dir.parent / 'frames.json',
        images_dir / 'frame_timestamps.json',
        images_dir.parent / 'frame_timestamps.json',
    ))

    t_slam = rows[:, 0]
    t0 = float(t_slam[0])
    span_s = float(t_slam[-1] - t0)
    if span_s <= 0.0:
        return []
    try:
        slerp = Slerp(t_slam, Rotation.from_quat(rows[:, 4:8]))
    except ValueError:
        return []

    lens_model = str(rig_data.get('lens_model') or 'THIN_PRISM_FISHEYE')
    cams_spec = []
    if 'T_lidar_to_cam0_rigid_4x4' in rig_data:
        parsed = _intrinsics_to_params(rig_data.get('cam0_front_intrinsics') or rig_data.get('cam0') or {}, lens_model)
        if parsed is not None:
            cams_spec.append(('cam0', np.asarray(rig_data['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64), parsed))
    if 'T_lidar_to_cam1_rigid_4x4' in rig_data:
        parsed = _intrinsics_to_params(rig_data.get('cam1_rear_intrinsics') or rig_data.get('cam1') or {}, lens_model)
        if parsed is not None:
            cams_spec.append(('cam1', np.asarray(rig_data['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64), parsed))
    if not cams_spec:
        return []

    # Collect image files and infer fps if frame_timestamps.json is absent
    raw_items: List[Tuple[str, Path, int, np.ndarray, Tuple[str, int, int, np.ndarray]]] = []
    for cam_name, T_lc, parsed in cams_spec:
        cam_dir = images_dir / cam_name
        if not cam_dir.is_dir():
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in sorted(cam_dir.glob(ext)):
                rel = f'{cam_name}/{img_path.name}'
                raw_items.append((rel, img_path, _frame_number(img_path.name), T_lc, parsed))
    if not raw_items:
        return []

    fps: Optional[float] = None
    if not ts_map:
        fps = _workflow_fps(wf_cfg or _read_workflow_config(images_dir.parent))
        if fps is None:
            return []  # Cannot map frame index to trajectory time safely without fps or timestamps.

    frames: List[Dict[str, Any]] = []
    for rel, img_path, fnum, T_lc, (model, width, height, params) in raw_items:
        video_t = ts_map.get(rel)
        if video_t is None:
            if fps is None:
                continue
            video_t = (fnum - 1) / fps if fnum >= 1 else 0.0
        t_query = t0 + (video_t - float(dt_sync))
        if not (t_slam[0] <= t_query <= t_slam[-1]):
            continue
        R_lidar = slerp(t_query).as_matrix()
        R_cw = (R_lidar @ T_lc[:3, :3]).T
        p_lidar = np.array([np.interp(t_query, t_slam, rows[:, axis]) for axis in (1, 2, 3)])
        p_cam = p_lidar + R_lidar @ T_lc[:3, 3]
        frames.append({
            'path': img_path,
            'name': rel,
            'frame_num': fnum,
            'video_time_s': float(video_t),
            'R_cw': R_cw,
            'cam_pos': p_cam,
            'model': model,
            'width': width,
            'height': height,
            'params': params,
        })
    return frames


@lru_cache(maxsize=4)
def _sky_hemisphere_grid(size: int = 480) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Precomputed upward-looking equidistant hemisphere unit vectors in world frame."""
    center = 0.5 * float(size - 1)
    f_rad = (0.5 * float(size)) / (0.5 * math.pi)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dx = xx - center
    dy = yy - center
    theta = np.hypot(dx, dy) / f_rad
    valid = theta <= 0.5 * math.pi
    phi = np.arctan2(dy, dx)
    bw_x = (np.sin(theta) * np.cos(phi)).astype(np.float32)
    bw_y = (np.sin(theta) * np.sin(phi)).astype(np.float32)
    bw_z = np.cos(theta).astype(np.float32)
    return bw_x, bw_y, bw_z, valid, center, f_rad


def _remap_equirectangular_sky(
    small: np.ndarray,
    full_width: int,
    full_height: int,
    R_cw: np.ndarray,
    size: int = 480,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Project an equirectangular frame onto an undistorted, seam-free upward sky hemisphere.

    Equirectangular images stretch horizontally by ``1 / cos(lat)`` near zenith and split blobs
    across the ``lon = +/-180 deg`` seam. Remapping the upper world hemisphere to an equidistant
    zenith-centered fisheye restores true spherical circularity without loosening indoor rejection.
    """
    import cv2

    sh, sw = small.shape[:2]
    sx = float(full_width) / float(sw)
    sy = float(full_height) / float(sh)
    bw_x, bw_y, bw_z, valid_hemi, center, f_rad = _sky_hemisphere_grid(size)
    # Explicit 3x3 linear combinations avoid concurrent BLAS calls inside ThreadPoolExecutor.
    r00, r01, r02 = float(R_cw[0, 0]), float(R_cw[0, 1]), float(R_cw[0, 2])
    r10, r11, r12 = float(R_cw[1, 0]), float(R_cw[1, 1]), float(R_cw[1, 2])
    r20, r21, r22 = float(R_cw[2, 0]), float(R_cw[2, 1]), float(R_cw[2, 2])
    bc_x = r00 * bw_x + r01 * bw_y + r02 * bw_z
    bc_y = r10 * bw_x + r11 * bw_y + r12 * bw_z
    bc_z = r20 * bw_x + r21 * bw_y + r22 * bw_z
    lat_c = np.arcsin(np.clip(-bc_y, -1.0, 1.0))
    lon_c = np.arctan2(bc_x, bc_z)
    u_full = (lon_c / (2.0 * math.pi) + 0.5) * float(full_width)
    v_full = (0.5 - lat_c / math.pi) * float(full_height)
    map_x = (((u_full + 0.5) / sx - 0.5) % float(sw)).astype(np.float32)
    map_y = np.clip((v_full + 0.5) / sy - 0.5, 0.0, float(sh - 1)).astype(np.float32)
    padded = np.hstack([small, small[:, :1]])
    sky = cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sky[~valid_hemi] = 0
    return sky, valid_hemi, center, f_rad


def _detect_frame_sun_candidates(
    frame: Dict[str, Any],
    alt_f: float,
    az_f: float,
    alt_mid: float,
    az_mid: float,
    convergence_deg: float,
) -> List[Dict[str, Any]]:
    """Detect unoccluded sky-halo sun disks in one calibrated frame."""
    import cv2

    path = str(frame['path'])
    model_upper = str(frame['model']).upper()
    is_equirect = model_upper == 'EQUIRECTANGULAR'
    ref_dim_full = int(frame['height']) if is_equirect else max(int(frame['width']), int(frame['height']))
    is_jpeg = Path(path).suffix.lower() in ('.jpg', '.jpeg')
    # Fast reduced DCT JPEG decode targeting ~480 px reference dimension across 2K/4K/5.7K/8K captures.
    if is_jpeg and ref_dim_full >= 3072:
        small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_8)
    elif is_jpeg and ref_dim_full >= 1536:
        small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    elif is_jpeg and ref_dim_full >= 768:
        small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    else:
        small = cv2.imread(path, cv2.IMREAD_COLOR)
    if small is None or small.size == 0:
        return []
    ref_dim = small.shape[0] if is_equirect else max(small.shape[:2])
    if ref_dim > 640:
        scale_down = 480.0 / float(ref_dim)
        small = cv2.resize(small, (max(1, int(round(small.shape[1] * scale_down))),
                                   max(1, int(round(small.shape[0] * scale_down)))),
                           interpolation=cv2.INTER_AREA)

    R_cw = frame['R_cw']
    valid_mask: Optional[np.ndarray] = None
    sky_center = 0.0
    sky_f_rad = 1.0
    if is_equirect:
        small, valid_mask, sky_center, sky_f_rad = _remap_equirectangular_sky(
            small, int(frame['width']), int(frame['height']), R_cw,
        )

    sh, sw = small.shape[:2]
    mask = ((small[:, :, 0] >= SAT_THRESHOLD)
            & (small[:, :, 1] >= SAT_THRESHOLD)
            & (small[:, :, 2] >= SAT_THRESHOLD))
    if valid_mask is not None:
        mask &= valid_mask
    if int(mask.sum()) < MIN_BLOB_AREA_PX:
        return []

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    max_area = int(MAX_BLOB_AREA_FRAC * sh * sw)
    ksize = max(5, int(round(9.0 * min(sh, sw) / 480.0)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    min_cos_lens = math.cos(math.radians(MAX_LENS_ANGLE_DEG))
    is_fisheye = model_upper in ('THIN_PRISM_FISHEYE', 'OPENCV_FISHEYE')

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    sx = float(frame['width']) / float(sw)
    sy = float(frame['height']) / float(sh)
    sun_en_math = 90.0 - (az_f - convergence_deg)
    daz_deg = az_f - az_mid
    dalt_deg = alt_mid - alt_f

    results: List[Dict[str, Any]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < MIN_BLOB_AREA_PX or area > max_area:
            continue
        sub = (labels == label).astype(np.uint8)
        dist = cv2.distanceTransform(sub, cv2.DIST_L2, 5)
        dmax = float(dist.max())
        if dmax < MIN_CORE_RADIUS_PX:
            continue
        circularity = float(area) / (math.pi * dmax * dmax)
        if circularity > MAX_CIRCULARITY:
            continue
        ring = (cv2.dilate(sub, kernel) > 0) & (sub == 0)
        if valid_mask is not None:
            ring &= valid_mask
        if not ring.any():
            continue
        ring_p10 = float(np.percentile(gray[ring], 10))
        if ring_p10 < MIN_HALO_RING_P10:
            continue

        ys, xs = np.nonzero(dist >= 0.88 * dmax)
        if is_equirect:
            ux = float(xs.mean()) - sky_center
            uy = float(ys.mean()) - sky_center
            th = math.hypot(ux, uy) / sky_f_rad
            ph = math.atan2(uy, ux)
            b_world = np.array([
                math.sin(th) * math.cos(ph),
                math.sin(th) * math.sin(ph),
                math.cos(th),
            ], dtype=np.float64)
        else:
            uv = np.array([[(float(xs.mean()) + 0.5) * sx - 0.5, (float(ys.mean()) + 0.5) * sy - 0.5]], dtype=np.float64)
            b_cam = unproject_pixels(uv, frame['model'], frame['params'], frame['width'], frame['height'])[0]
            if not np.all(np.isfinite(b_cam)):
                continue
            if is_fisheye and b_cam[2] < min_cos_lens:
                continue
            if not is_fisheye and b_cam[2] <= 0.0:
                continue
            b_world = R_cw.T @ b_cam

        elev_deg = math.degrees(math.asin(max(-1.0, min(1.0, float(b_world[2])))))
        elev_err = elev_deg - alt_f
        if abs(elev_err) > MAX_OBS_ELEV_ERR_DEG:
            continue

        local_math = math.degrees(math.atan2(float(b_world[1]), float(b_world[0])))
        yaw_deg = float((sun_en_math - local_math) % 360.0)
        # Compensate both solar azimuth and altitude motion into the mid-scan reference frame so
        # long scans cluster tightly on the sphere without inducing apparent parallax drift.
        elev_mid_rad = math.radians(max(-90.0, min(90.0, elev_deg + dalt_deg)))
        local_math_mid_rad = math.radians(local_math + daz_deg)
        cos_em = math.cos(elev_mid_rad)
        b_mid = np.array([
            cos_em * math.cos(local_math_mid_rad),
            cos_em * math.sin(local_math_mid_rad),
            math.sin(elev_mid_rad),
        ], dtype=np.float64)
        results.append({
            'yaw_deg': yaw_deg,
            'elev_err_deg': float(elev_err),
            'vec_mid': b_mid,
            'cam_pos': np.asarray(frame['cam_pos'], dtype=np.float64),
            'ring_p10': ring_p10,
            'circularity': circularity,
            'frame': frame['name'],
        })
    # There is only one sun in the sky: keep at most the strongest unoccluded halo per frame.
    if len(results) > 1:
        results = [max(results, key=lambda r: (r['ring_p10'], -abs(r['elev_err_deg'])))]
    return results


def _finite_distance_parallax(vecs: np.ndarray, cam_pos: np.ndarray, mean_vec: np.ndarray) -> Tuple[Optional[float], float, float, float]:
    """Fit a converging finite-distance light source to inlier rays; return (dist_m_or_None, r2, rms_drop_deg, baseline_m)."""
    if len(vecs) < 3 or len(cam_pos) != len(vecs):
        return None, 0.0, 0.0, 0.0
    dp = cam_pos - cam_pos.mean(axis=0)
    d_perp = dp - (dp @ mean_vec)[:, None] * mean_vec[None, :]
    r_perp = vecs - (vecs @ mean_vec)[:, None] * mean_vec[None, :]
    denom = float((d_perp * d_perp).sum())
    r_ss = float((r_perp * r_perp).sum())
    baseline_m = math.sqrt(denom / float(len(vecs)))
    if denom < 1e-6 or r_ss < 1e-10:
        return None, 0.0, 0.0, baseline_m
    kappa = max(0.0, -float((d_perp * r_perp).sum()) / denom)
    if kappa <= 1e-6:
        return None, 0.0, 0.0, baseline_m
    res_ss = float(((r_perp + kappa * d_perp) ** 2).sum())
    r2 = float(max(0.0, 1.0 - res_ss / r_ss))
    rms_before = math.degrees(math.sqrt(r_ss / float(len(vecs))))
    rms_after = math.degrees(math.sqrt(max(res_ss, 0.0) / float(len(vecs))))
    return float(1.0 / kappa), r2, float(rms_before - rms_after), float(baseline_m)


def cluster_solar_observations(
    observations: Sequence[Dict[str, Any]],
    sun_alt_deg: float,
    sun_az_true_deg: float,
) -> Dict[str, Any]:
    """Consensus cluster 3D solar ray observations on the unit sphere."""
    base = {
        'qualifies': False,
        'strong': False,
        'candidates': len(observations),
        'inliers': 0,
        'rival_inliers': 0,
        'rival_ratio': 1.0,
        'sun_altitude_deg': float(sun_alt_deg),
        'sun_azimuth_true_deg': float(sun_az_true_deg),
    }
    if not observations:
        base['reason'] = 'no_unoccluded_sun_observations'
        return base

    vecs = np.array([o['vec_mid'] for o in observations], dtype=np.float64)
    yaws = np.array([o['yaw_deg'] for o in observations], dtype=np.float64)
    elev_errs = np.array([o['elev_err_deg'] for o in observations], dtype=np.float64)

    cos_cone = math.cos(math.radians(CLUSTER_CONE_DEG))
    cos_rival = math.cos(math.radians(RIVAL_SEPARATION_DEG))
    dots = np.clip(vecs @ vecs.T, -1.0, 1.0)
    counts = (dots >= cos_cone).sum(axis=1)
    best_idx = int(np.argmax(counts))
    inliers_mask = dots[best_idx] >= cos_cone
    inliers = int(inliers_mask.sum())

    mean_vec = vecs[inliers_mask].mean(axis=0)
    mean_vec /= max(float(np.linalg.norm(mean_vec)), 1e-12)
    # Re-center membership around the cluster mean vector.
    inliers_mask = (vecs @ mean_vec) >= cos_cone
    inliers = int(inliers_mask.sum())
    if inliers == 0:
        base['reason'] = 'empty_solar_cluster'
        return base
    mean_vec = vecs[inliers_mask].mean(axis=0)
    mean_vec /= max(float(np.linalg.norm(mean_vec)), 1e-12)

    # Strongest competing cluster at least RIVAL_SEPARATION_DEG away on the 3D sphere.
    outside = (vecs @ mean_vec) < cos_rival
    rival_inliers = 0
    if outside.any():
        out_vecs = vecs[outside]
        out_dots = np.clip(out_vecs @ out_vecs.T, -1.0, 1.0)
        rival_inliers = int((out_dots >= cos_cone).sum(axis=1).max())
    rival_ratio = float(rival_inliers) / float(inliers)

    in_yaws = yaws[inliers_mask]
    rad = np.radians(in_yaws)
    c_mean = math.degrees(math.atan2(float(np.sin(rad).mean()), float(np.cos(rad).mean()))) % 360.0
    residuals = (in_yaws - c_mean + 180.0) % 360.0 - 180.0
    yaw_med = float((c_mean + float(np.median(residuals))) % 360.0)
    yaw_mad = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))

    ang_err_3d = np.degrees(np.arccos(np.clip(vecs[inliers_mask] @ mean_vec, -1.0, 1.0)))
    rms_3d = float(np.sqrt(np.mean(ang_err_3d ** 2)))
    elev_err_med = float(np.median(elev_errs[inliers_mask]))

    parallax_dist_m: Optional[float] = None
    parallax_r2 = 0.0
    finite_source = False
    if all('cam_pos' in o for o in observations):
        cpos = np.array([o['cam_pos'] for o in observations], dtype=np.float64)[inliers_mask]
        parallax_dist_m, parallax_r2, rms_drop, baseline_m = _finite_distance_parallax(vecs[inliers_mask], cpos, mean_vec)
        finite_source = bool(
            parallax_dist_m is not None
            and baseline_m >= PARALLAX_MIN_BASELINE_M
            and parallax_dist_m <= PARALLAX_MAX_FINITE_DIST_M
            and parallax_r2 >= PARALLAX_MIN_R2
            and rms_drop >= PARALLAX_MIN_RMS_DROP_DEG
        )

    # Geometric azimuth sensitivity scales as 1 / cos(solar_altitude).
    sigma_3d = max(rms_3d, abs(elev_err_med) / 1.5, 0.5)
    sigma_deg = float(min(sigma_3d / max(math.cos(math.radians(sun_alt_deg)), 0.1), 180.0))

    qualifies = bool(
        inliers >= MIN_SOLAR_INLIERS
        and not finite_source
        and rival_ratio <= MAX_RIVAL_RATIO
        and abs(elev_err_med) <= MAX_MEDIAN_ELEV_ERR_DEG
        and rms_3d <= MAX_3D_RMS_DEG
        and sigma_deg <= MAX_SOLAR_SIGMA_DEG
    )
    strong = bool(
        qualifies
        and inliers >= STRONG_MIN_INLIERS
        and rival_ratio <= STRONG_MAX_RIVAL_RATIO
        and abs(elev_err_med) <= STRONG_MAX_ELEV_ERR_DEG
        and sigma_deg <= STRONG_MAX_SIGMA_DEG
    )

    reason = 'ok'
    if not qualifies:
        if inliers < MIN_SOLAR_INLIERS:
            reason = 'too_few_unoccluded_sun_frames'
        elif finite_source:
            reason = 'finite_distance_light_source'
        elif rival_ratio > MAX_RIVAL_RATIO:
            reason = 'competing_bright_directions'
        elif abs(elev_err_med) > MAX_MEDIAN_ELEV_ERR_DEG:
            reason = 'solar_elevation_mismatch'
        elif rms_3d > MAX_3D_RMS_DEG:
            reason = 'inconsistent_3d_sun_direction'
        else:
            reason = 'high_solar_azimuth_uncertainty'

    return {
        'yaw_grid_deg': yaw_med,
        'sigma_deg': sigma_deg,
        'yaw_mad_deg': yaw_mad,
        'rms_3d_deg': rms_3d,
        'elev_err_med_deg': elev_err_med,
        'inliers': inliers,
        'candidates': len(observations),
        'rival_inliers': rival_inliers,
        'rival_ratio': rival_ratio,
        'parallax_dist_m': parallax_dist_m,
        'parallax_r2': parallax_r2,
        'sun_altitude_deg': float(sun_alt_deg),
        'sun_azimuth_true_deg': float(sun_az_true_deg),
        'qualifies': qualifies,
        'strong': strong,
        'reason': reason,
    }


def solve_solar_north(
    gps_info: Dict[str, Any],
    convergence_deg: float,
    *,
    colmap_dir: Optional[Path] = None,
    search_paths: Sequence[Any] = (),
    trajectory_path: Optional[Path] = None,
    dt_sync: Optional[float] = None,
    capture_origin_s: Optional[float] = 0.0,
    max_frames: int = 240,
    log: Callable[[str], None] = print,
) -> Optional[Dict[str, Any]]:
    """Estimate grid yaw from the sun in calibrated 360-degree frames, or None if no frames exist."""
    records = gps_info.get('records') or []
    mean_wgs = gps_info.get('mean_wgs84') or {}
    if (not records or any('timestamp_epoch' not in r for r in records)
            or 'latitude_deg' not in mean_wgs or 'longitude_deg' not in mean_wgs):
        return None

    epochs = np.array([float(r['timestamp_epoch']) for r in records], dtype=np.float64)
    lat = float(mean_wgs['latitude_deg'])
    lon = float(mean_wgs['longitude_deg'])
    if not (np.all(np.isfinite(epochs)) and math.isfinite(lat) and math.isfinite(lon)):
        return None
    t_start = float(epochs.min())
    t_end = float(epochs.max())
    t_gps_mid = 0.5 * (t_start + t_end)

    probe_alts = [solar_position(lat, lon, ep)[0] for ep in (t_start, t_gps_mid, t_end)]
    alt_gps_mid, az_gps_mid = solar_position(lat, lon, t_gps_mid)
    if all(a < MIN_SUN_ALT_DEG for a in probe_alts) or all(a > MAX_SUN_ALT_DEG for a in probe_alts):
        return {
            'qualifies': False,
            'strong': False,
            'sun_altitude_deg': alt_gps_mid,
            'sun_azimuth_true_deg': az_gps_mid,
            'reason': 'sun_below_horizon_or_too_low' if alt_gps_mid < MIN_SUN_ALT_DEG else 'sun_too_near_zenith',
        }

    resolved_colmap = find_colmap_dir((colmap_dir, *search_paths))
    frames: List[Dict[str, Any]] = []
    source_kind = ''
    if resolved_colmap is not None:
        frames = _load_colmap_frames(resolved_colmap)
        source_kind = 'colmap'
    if not frames:
        frames = _load_direct_rig_frames((colmap_dir, *search_paths), trajectory_path, dt_sync)
        source_kind = 'rig_trajectory'
    if not frames:
        return None

    # Sample evenly across all frames while preserving temporal coverage.
    frames.sort(key=lambda f: (f.get('video_time_s', float(f['frame_num'])), f['frame_num'], f['name']))
    f_min = frames[0]['frame_num']
    f_max = frames[-1]['frame_num']
    step = max(1, len(frames) // max(1, int(max_frames)))
    sampled = frames[::step]

    def frame_epoch(frame: Dict[str, Any]) -> float:
        if 'video_time_s' in frame:
            return t_start + float(capture_origin_s or 0.0) + float(frame['video_time_s'])
        if f_max > f_min and t_end > t_start:
            frac = float(frame['frame_num'] - f_min) / float(f_max - f_min)
            return t_start + frac * (t_end - t_start)
        return t_gps_mid

    sampled_epochs = [frame_epoch(f) for f in sampled]
    t_mid = float(np.median(sampled_epochs)) if sampled_epochs else t_gps_mid
    alt_mid, az_mid = solar_position(lat, lon, t_mid)
    if alt_mid < MIN_SUN_ALT_DEG or alt_mid > MAX_SUN_ALT_DEG:
        return {
            'qualifies': False,
            'strong': False,
            'sun_altitude_deg': alt_mid,
            'sun_azimuth_true_deg': az_mid,
            'reason': 'sun_below_horizon_or_too_low' if alt_mid < MIN_SUN_ALT_DEG else 'sun_too_near_zenith',
        }

    def process_one(item: Tuple[Dict[str, Any], float]) -> List[Dict[str, Any]]:
        frame, ep = item
        alt_f, az_f = solar_position(lat, lon, ep)
        if alt_f < MIN_SUN_ALT_DEG or alt_f > MAX_SUN_ALT_DEG:
            return []
        return _detect_frame_sun_candidates(frame, alt_f, az_f, alt_mid, az_mid, convergence_deg)

    if any(str(f['model']).upper() == 'EQUIRECTANGULAR' for f in sampled):
        _sky_hemisphere_grid()

    observations: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for batch in pool.map(process_one, zip(sampled, sampled_epochs)):
            observations.extend(batch)

    result = cluster_solar_observations(observations, alt_mid, az_mid)
    result['frames_sampled'] = len(sampled)
    result['source'] = source_kind
    if resolved_colmap is not None and source_kind == 'colmap':
        result['colmap_dir'] = str(resolved_colmap)

    if result['qualifies']:
        log(f"[north] Solar compass ({source_kind}, {result['inliers']}/{result['frames_sampled']} frames, "
            f"sun alt {alt_mid:.1f} deg): yaw {result['yaw_grid_deg']:.2f} +/- {result['sigma_deg']:.1f} deg "
            f"(elev err {result['elev_err_med_deg']:+.2f} deg, 3D RMS {result['rms_3d_deg']:.2f} deg).")
    else:
        log(f"[north] Solar compass ({source_kind}, sun alt {alt_mid:.1f} deg): rejected "
            f"({result['reason']}, {result['inliers']} inliers).")
    return result
