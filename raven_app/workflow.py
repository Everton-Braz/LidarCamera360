"""Unified end-to-end workflow engine for RavenCalibrator.

Orchestrates:
1. Video frame & gyro telemetry extraction from Insta360 .insv
2. FAST-LIVO2 LiDAR-inertial odometry and mapping (.bag -> .pcd + trajectory)
3. Sub-millisecond temporal cross-correlation (IMU Gyro Δt)
4. Calibrated LiDAR-camera point cloud colorization
5. Deliverables generation (.PLY, .PCD, and COLMAP dataset ready for 3DGS training).
"""

import json
import hashlib
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
from raven_app.subprocess_utils import hidden_window_options
from scipy.spatial.transform import Rotation as Rot

from raven_app import __version__
from raven_app.cli import engine, resources
from raven_app.bag_io import export_bags
from raven_app.slam_source import active_slam_dir
from raven_app.video import compute_laplacian_sharpness, extract_insv_frames_pyav, frame_time

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass



def process_gps_metadata(insv_path, output_dir, formats=('geojson', 'gpx', 'csv')):
    """Export GPS metadata independently of reconstruction, without placing geometry."""
    from raven_app.insv_gps import export_gps
    selected = set(formats)
    if not selected or not selected <= {'csv', 'gpx', 'geojson'}:
        raise ValueError('Select one or more GPS formats: csv, gpx, geojson')
    try:
        report = export_gps(insv_path, output_dir, formats=selected)
        report['status'] = 'complete' if report['valid_records'] else 'no_valid_gps'
    except ValueError as exc:
        # Unsupported/missing telemetry should not cancel SLAM and colorization.
        # File/permission failures still propagate instead of reporting success.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report = {'status': 'unavailable', 'source_insv': str(Path(insv_path).resolve()),
                  'reason': str(exc), 'unique_fixes': 0, 'valid_records': 0,
                  'georeferencing_status': 'gps_metadata_unavailable',
                  'selected_formats': sorted(selected), 'output_files': {}}
    report['geometry_transformed'] = False
    (Path(output_dir) / 'gps_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return report


def extract_insv_frames(insv_path: Path, output_dir: Path, fps: float = 2.0) -> bool:
    return extract_insv_frames_pyav(insv_path, output_dir, fps=fps)


def _workflow_path_snapshot(path):
    if path is None:
        return None
    source = Path(path)
    result = {'path': str(source.resolve())}
    try:
        stat = source.stat()
        result.update(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    except OSError:
        pass
    return result


def _persist_workflow_run(output_dir, record):
    """Append/update the structured settings snapshot for the current workflow run."""
    path = Path(output_dir) / 'workflow_config.json'
    try:
        data = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    runs = data.get('runs', []) if data.get('schema_version') == 1 else []
    runs = [item for item in runs if item.get('run_id') != record['run_id']]
    record['updated_at_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    runs.append(record)

    def _sanitize(val):
        if isinstance(val, float):
            return val if math.isfinite(val) else None
        if isinstance(val, dict):
            return {k: _sanitize(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_sanitize(v) for v in val]
        return val

    data = {'schema_version': 1, 'runs': _sanitize(runs[-20:])}
    temporary = path.with_name(f'.{path.name}.{record["run_id"]}.tmp')
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    os.replace(temporary, path)


def _workflow_image_extraction_snapshot(output_dir, third_cameras):
    manifest_path = Path(output_dir) / 'images' / 'frames.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    signatures = {'cam0': manifest.get('signature'), 'cam1': manifest.get('signature')}
    insv_signatures = manifest.get('auxiliary_insv_signatures', {})
    if not isinstance(insv_signatures, dict):
        insv_signatures = {}
    for camera in third_cameras:
        is_insv = (str(getattr(camera, 'source_type', '')).lower() == 'insv'
                   or Path(str(getattr(camera, 'source_path', ''))).suffix.lower() == '.insv')
        signatures[camera.camera_name] = (
            insv_signatures.get(camera.camera_name, manifest.get(f'signature_{camera.camera_name}'))
            if is_insv else manifest.get(f'signature_{camera.camera_name}'))
    frame_counts = {}
    for name in manifest.get('timestamps', {}):
        camera_name = str(name).replace('\\', '/').split('/', 1)[0]
        frame_counts[camera_name] = frame_counts.get(camera_name, 0) + 1
    return {
        'manifest': 'images/frames.json',
        'camera_signatures': signatures,
        'frame_counts': frame_counts,
    }


def _workflow_settings_snapshot(bag_path, insv_path, output_dir, *, fps, method,
                                third_cameras, **settings):
    camera_settings = [
        {'camera_name': name, 'source_type': 'insv',
         'source': _workflow_path_snapshot(insv_path), 'fps': float(fps)}
        for name in ('cam0', 'cam1')
    ]
    for camera in third_cameras:
        config = camera.to_dict()
        config['source_snapshot'] = _workflow_path_snapshot(camera.source_path)
        config['fps'] = float(camera.fps)
        camera_settings.append(config)
    return {
        'paths': {
            'bag': _workflow_path_snapshot(bag_path),
            'insv': _workflow_path_snapshot(insv_path),
            'output': str(Path(output_dir).resolve()),
        },
        'settings': {
            **settings,
            'method': method,
            'fps_by_camera': {camera['camera_name']: camera['fps']
                              for camera in camera_settings},
        },
        'cameras': camera_settings,
    }


def extract_insv_gyro(insv_path: Path):
    """Extract gyroscope angular velocities and timestamps from INSV binary trailer."""
    path = Path(insv_path)
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        if size < 78:
            raise ValueError("INSV file is too short for a trailer")
        stream.seek(size - 72)
        header = stream.read(72)
        if header[-32:] != b"8db42d694ccc418790edff439fe026bf":
            raise ValueError("Invalid INSV magic footer")

        length = struct.unpack_from("<I", header, 32)[0]
        if not 78 <= length <= size:
            raise ValueError("Invalid INSV trailer length")
        start, end = size - length, size - 72

        stream.seek(end - 6)
        fmt, kind, count = struct.unpack("<BBI", stream.read(6))
        entries = {}
        if kind == 0 and fmt == 0:
            if count % 10 or count > length - 78:
                raise ValueError("Malformed INSV trailer directory")
            stream.seek(end - 6 - count)
            directory = stream.read(count)
            for k, f, c, offset in struct.iter_unpack("<BBII", directory):
                if k and c:
                    entries[k] = (f, c, offset)
        else:
            curr_end = end
            while curr_end > start:
                if curr_end - start < 6:
                    break
                stream.seek(curr_end - 6)
                f, k, c = struct.unpack("<BBI", stream.read(6))
                begin = curr_end - 6 - c
                if begin < start:
                    break
                if k and c and k not in entries:
                    entries[k] = (f, c, begin - start)
                curr_end = begin

        if 3 not in entries or 4 not in entries:
            raise ValueError("INSV lacks gyro or exposure telemetry track")

        _, g_size, g_off = entries[3]
        stream.seek(start + g_off)
        gyro_bytes = stream.read(g_size)

        _, e_size, e_off = entries[4]
        stream.seek(start + e_off)
        exp_bytes = stream.read(e_size)

    cam_imu_raw = np.frombuffer(gyro_bytes, dtype=[
        ('t', '<u8'),
        ('ax', '<u2'), ('ay', '<u2'), ('az', '<u2'),
        ('gx', '<u2'), ('gy', '<u2'), ('gz', '<u2')
    ])
    cam_exp_raw = np.frombuffer(exp_bytes, dtype=[('t', '<u8'), ('exp', '<f8')])

    t_exp0_us = cam_exp_raw['t'][0]
    t_cam_s = (cam_imu_raw['t'].astype(np.float64) - t_exp0_us) / 1e6

    cam_gx = cam_imu_raw['gx'].astype(np.float64) - 32768.0
    cam_gy = cam_imu_raw['gy'].astype(np.float64) - 32768.0
    cam_gz = cam_imu_raw['gz'].astype(np.float64) - 32768.0
    cam_gyro_norm = np.sqrt(cam_gx**2 + cam_gy**2 + cam_gz**2)

    return t_cam_s, cam_gyro_norm


def auto_sync_imu_gyro(bag_path: Path, insv_path: Path,
                       imu_topic: str = None) -> Optional[float]:
    """Estimate the clip offset inside the complete LiDAR recording.

    Weak or ambiguous matches return ``None``. A guessed fallback offset can
    make a valid SfM reconstruction look like a bad rig calibration.
    """
    print("[*] Performing programmatic IMU gyro cross-correlation for time sync...")
    try:
        from rosbags.highlevel import AnyReader
        from raven_app.bag_io import detect_bag_topics
        t_cam_s, cam_gyro_norm = extract_insv_gyro(insv_path)

        if not imu_topic:
            detected = detect_bag_topics([bag_path])
            imu_topic = detected['imu_topic']

        lidar_t = []
        lidar_wx, lidar_wy, lidar_wz = [], [], []
        with AnyReader([bag_path]) as reader:
            conn_topics = {c.topic for c in reader.connections}
            if imu_topic not in conn_topics:
                detected = detect_bag_topics([bag_path])
                imu_topic = detected['imu_topic']

            imu_connections = [c for c in reader.connections if c.topic == imu_topic]
            if not imu_connections:
                raise ValueError(f"No IMU connection found for topic {imu_topic}")
            for conn, _, raw in reader.messages(connections=imu_connections):
                msg = reader.deserialize(raw, conn.msgtype)
                t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                lidar_t.append(t_sec)
                lidar_wx.append(msg.angular_velocity.x)
                lidar_wy.append(msg.angular_velocity.y)
                lidar_wz.append(msg.angular_velocity.z)

        if not lidar_t:
            print("[!] No LiDAR IMU packets found; temporal offset remains unknown.")
            return None

        from raven_app.time_sync import estimate_gyro_time_sync
        lidar_t = np.asarray(lidar_t, dtype=np.float64)
        lidar_gyro_norm = np.sqrt(np.asarray(lidar_wx)**2 + np.asarray(lidar_wy)**2
                                  + np.asarray(lidar_wz)**2)
        estimate = estimate_gyro_time_sync(
            t_cam_s, cam_gyro_norm, lidar_t, lidar_gyro_norm)
        if not estimate.accepted:
            print("[!] IMU time sync rejected: "
                  f"{estimate.reason}; best correlation={estimate.score!s}, "
                  f"overlap={estimate.overlap_seconds:.1f}s, "
                  f"peak margin={estimate.ambiguity_margin!s}. "
                  "Searching camera poses against the full LiDAR trajectory.")
            return None

        print("[+] Programmatic IMU Gyro Cross-Correlation resolved: "
              f"Δt = {estimate.dt_seconds:.4f}s "
              f"(correlation={estimate.score:.3f}, "
              f"overlap={estimate.overlap_seconds:.1f}s)")
        return float(estimate.dt_seconds)
    except Exception as e:
        print(f"[!] IMU time sync unavailable ({e}); temporal offset remains unknown.")
        return None


def load_lidar_seed_points(dataset_dir: Path, max_points: Optional[int] = None,
                           seed_percent: float = 100.0,
                           preferred_method: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load colorized or raw LiDAR points to use as dense geometric 3DGS seed.

    Defaults to full sensor density unless a percentage or point cap is selected.
    """
    from raven_app.seed_export import sample_seed_points, validate_seed_percent
    seed_percent = validate_seed_percent(seed_percent)
    ply_candidates = sorted((dataset_dir / "deliverables").glob("lidar_colored_*.ply"))
    if not ply_candidates:
        ply_candidates = sorted(dataset_dir.glob("*.ply"))
    method = str(preferred_method or '').strip().lower()
    if method in ('direct', 'sfm'):
        preferred_tokens = ('direct', 'rigid') if method == 'direct' else ('sfm', 'consensus')
        preferred = [path for path in ply_candidates
                     if any(token in path.stem.lower() for token in preferred_tokens)]
        fallback = [path for path in ply_candidates if path not in preferred]
        ply_candidates = fallback + preferred

    xyz = None
    rgb = None

    for ply_path in reversed(ply_candidates):
        if ply_path.name.startswith("points3D"):
            continue
        try:
            with open(ply_path, "rb") as f:
                n_points = 0
                has_rgb = False
                while True:
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    if line.startswith("element vertex"):
                        n_points = int(line.split()[-1])
                    if "property uchar red" in line or "property uint8 red" in line or "property uchar r" in line:
                        has_rgb = True
                    if line == "end_header":
                        break
                if n_points > 0:
                    if has_rgb:
                        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
                        data = np.fromfile(f, dtype=dt, count=n_points)
                        xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
                        rgb = np.column_stack([data["r"], data["g"], data["b"]]).astype(np.uint8)
                    else:
                        dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
                        data = np.fromfile(f, dtype=dt, count=n_points)
                        xyz = np.column_stack([data["x"], data["y"], data["z"]]).astype(np.float64)
                        rgb = np.full((n_points, 3), 180, dtype=np.uint8)
                    if len(xyz) > 0:
                        print(f"  [+] Loaded {len(xyz):,} metric LiDAR points from {ply_path.name} as 3DGS seed")
                        break
        except Exception as e:
            print(f"  [!] Note reading {ply_path.name}: {e}")

    if xyz is None:
        pcd_candidates = sorted((dataset_dir / "deliverables").glob("*.pcd"))
        pcd_candidates += [
            active_slam_dir(dataset_dir) / "pcd" / "all_raw_points.pcd",
            dataset_dir / "all_raw_points.pcd",
        ]
        for pcd_path in pcd_candidates:
            if pcd_path.is_file():
                try:
                    from scripts.pipeline_auto_calibrator_and_colorizer import load_pcd
                    xyz = load_pcd(pcd_path)
                    rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
                    print(f"  [+] Loaded {len(xyz):,} raw LiDAR points from {pcd_path.name} as 3DGS seed")
                    break
                except Exception as e:
                    pass

    if xyz is None or len(xyz) == 0:
        return None, None

    source_count = len(xyz)
    xyz, rgb = sample_seed_points(xyz, rgb, seed_percent, max_points)
    print(f"  [+] 3DGS seed export: {len(xyz):,} / {source_count:,} points "
          f"(requested {seed_percent:g}%; evenly spaced source indices)")

    return xyz, rgb


def write_colmap_points3d(dst_dir: Path, xyz: np.ndarray, rgb: np.ndarray, errors: np.ndarray = None):
    """Write points3D.ply, points3D.bin, and points3D.txt in COLMAP format using fast vectorized buffers."""
    n = len(xyz)
    dst_dir.mkdir(parents=True, exist_ok=True)
    if errors is None:
        errors = np.full(n, 0.1, dtype=np.float64)

    # 1. Write binary PLY (points3D.ply)
    ply_path = dst_dir / "points3D.ply"
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"] = xyz[:, 0]
    arr["y"] = xyz[:, 1]
    arr["z"] = xyz[:, 2]
    arr["r"] = rgb[:, 0]
    arr["g"] = rgb[:, 1]
    arr["b"] = rgb[:, 2]
    with open(ply_path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())

    # 2. Write binary COLMAP (points3D.bin) via fast vectorized structured array
    bin_path = dst_dir / "points3D.bin"
    bin_dt = np.dtype([
        ("id", "<u8"),
        ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("error", "<f8"),
        ("track_len", "<u8")
    ], align=False)
    bin_arr = np.empty(n, dtype=bin_dt)
    bin_arr["id"] = np.arange(1, n + 1, dtype=np.uint64)
    bin_arr["x"] = xyz[:, 0]
    bin_arr["y"] = xyz[:, 1]
    bin_arr["z"] = xyz[:, 2]
    bin_arr["r"] = rgb[:, 0]
    bin_arr["g"] = rgb[:, 1]
    bin_arr["b"] = rgb[:, 2]
    bin_arr["error"] = errors
    bin_arr["track_len"] = 0
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(bin_arr.tobytes())

    # 3. Write text COLMAP (points3D.txt) in buffered chunks
    txt_path = dst_dir / "points3D.txt"
    chunk_size = 250000
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_xyz = xyz[start:end]
            chunk_rgb = rgb[start:end]
            chunk_err = errors[start:end]
            lines = [
                f"{start + i + 1} {chunk_xyz[i, 0]:.6f} {chunk_xyz[i, 1]:.6f} {chunk_xyz[i, 2]:.6f} "
                f"{chunk_rgb[i, 0]} {chunk_rgb[i, 1]} {chunk_rgb[i, 2]} {chunk_err[i]:.4f}\n"
                for i in range(len(chunk_xyz))
            ]
            f.writelines(lines)

    # Link points3D.ply to root of colmap_3dgs if dst_dir is sparse/0 for root loaders without duplicating disk space
    if dst_dir.parent.name == "sparse" and dst_dir.name == "0":
        try:
            root_ply = dst_dir.parent.parent / "points3D.ply"
            if root_ply.exists():
                root_ply.unlink()
            try:
                os.link(ply_path, root_ply)
            except Exception:
                try:
                    root_ply.symlink_to(ply_path)
                except Exception:
                    shutil.copy2(ply_path, root_ply)
        except Exception:
            pass


def fuse_colmap_points3d(sparse_dir: Path, lidar_xyz: np.ndarray, lidar_rgb: np.ndarray) -> Tuple[int, int, int]:
    """Fuse reconstruction SfM points (preserving 2D camera tracks) and dense LiDAR points into a single model."""
    pts_bin = sparse_dir / "points3D.bin"
    pts_ply = sparse_dir / "points3D.ply"
    pts_txt = sparse_dir / "points3D.txt"

    n_lidar = len(lidar_xyz)
    if not pts_bin.is_file():
        write_colmap_points3d(sparse_dir, lidar_xyz, lidar_rgb)
        return 0, n_lidar, n_lidar

    # 1. Preserve pure SfM reconstruction points as points3D_sfm_sparse.*
    sfm_sparse_bin = sparse_dir / "points3D_sfm_sparse.bin"
    sfm_sparse_ply = sparse_dir / "points3D_sfm_sparse.ply"
    sfm_sparse_txt = sparse_dir / "points3D_sfm_sparse.txt"
    if not sfm_sparse_bin.is_file():
        try:
            shutil.copy2(pts_bin, sfm_sparse_bin)
            if pts_ply.is_file():
                shutil.copy2(pts_ply, sfm_sparse_ply)
            if pts_txt.is_file():
                shutil.copy2(pts_txt, sfm_sparse_txt)
        except Exception:
            pass

    # 2. Read SfM points binary data & tracks
    with open(sfm_sparse_bin, "rb") as f:
        n_sfm = struct.unpack("<Q", f.read(8))[0]
        sfm_bin_bytes = f.read()

    # Read SfM coordinates & colors from PLY (sub-millisecond)
    sfm_xyz = np.empty((0, 3), dtype=np.float64)
    sfm_rgb = np.empty((0, 3), dtype=np.uint8)
    if sfm_sparse_ply.is_file():
        try:
            with open(sfm_sparse_ply, "rb") as f:
                while True:
                    line = f.readline().decode("latin1", errors="ignore").strip()
                    if line == "end_header":
                        break
                dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
                sfm_data = np.fromfile(f, dtype=dt, count=n_sfm)
                sfm_xyz = np.column_stack([sfm_data["x"], sfm_data["y"], sfm_data["z"]]).astype(np.float64)
                sfm_rgb = np.column_stack([sfm_data["r"], sfm_data["g"], sfm_data["b"]]).astype(np.uint8)
        except Exception as e:
            print(f"  [!] Note reading sfm ply: {e}")

    n_total = n_sfm + n_lidar
    print(f"  [*] Fusing {n_sfm:,} SfM reconstruction points with {n_lidar:,} metric LiDAR points -> Total: {n_total:,} points")

    # 3. Write fused points3D.bin
    with open(pts_bin, "wb") as f:
        f.write(struct.pack("<Q", n_total))
        f.write(sfm_bin_bytes)

        # Append LiDAR points with track_len=0
        bin_dt = np.dtype([
            ("id", "<u8"),
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
            ("error", "<f8"),
            ("track_len", "<u8")
        ], align=False)
        lidar_arr = np.empty(n_lidar, dtype=bin_dt)
        lidar_arr["id"] = np.arange(n_sfm + 1, n_total + 1, dtype=np.uint64)
        lidar_arr["x"] = lidar_xyz[:, 0]
        lidar_arr["y"] = lidar_xyz[:, 1]
        lidar_arr["z"] = lidar_xyz[:, 2]
        lidar_arr["r"] = lidar_rgb[:, 0]
        lidar_arr["g"] = lidar_rgb[:, 1]
        lidar_arr["b"] = lidar_rgb[:, 2]
        lidar_arr["error"] = 0.1
        lidar_arr["track_len"] = 0
        f.write(lidar_arr.tobytes())

    # 4. Write fused points3D.ply
    fused_xyz = np.vstack([sfm_xyz, lidar_xyz]) if len(sfm_xyz) > 0 else lidar_xyz
    fused_rgb = np.vstack([sfm_rgb, lidar_rgb]) if len(sfm_rgb) > 0 else lidar_rgb
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(fused_xyz)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(len(fused_xyz), dtype=dt)
    arr["x"] = fused_xyz[:, 0]
    arr["y"] = fused_xyz[:, 1]
    arr["z"] = fused_xyz[:, 2]
    arr["r"] = fused_rgb[:, 0]
    arr["g"] = fused_rgb[:, 1]
    arr["b"] = fused_rgb[:, 2]
    with open(pts_ply, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())

    # Link root points3D.ply
    if sparse_dir.parent.name == "sparse" and sparse_dir.name == "0":
        try:
            root_ply = sparse_dir.parent.parent / "points3D.ply"
            if root_ply.exists():
                root_ply.unlink()
            try:
                os.link(pts_ply, root_ply)
            except Exception:
                try:
                    root_ply.symlink_to(pts_ply)
                except Exception:
                    shutil.copy2(pts_ply, root_ply)
        except Exception:
            pass

    # 5. Write fused points3D.txt
    chunk_size = 250000
    with open(pts_txt, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        if sfm_sparse_txt.is_file():
            with open(sfm_sparse_txt, "r", encoding="utf-8", errors="ignore") as f_sfm:
                for line in f_sfm:
                    if not line.startswith("#") and line.strip():
                        f.write(line)
        err_val = 0.1
        for start in range(0, n_lidar, chunk_size):
            end = min(start + chunk_size, n_lidar)
            c_xyz = lidar_xyz[start:end]
            c_rgb = lidar_rgb[start:end]
            lines = [
                f"{n_sfm + start + i + 1} {c_xyz[i, 0]:.6f} {c_xyz[i, 1]:.6f} {c_xyz[i, 2]:.6f} "
                f"{c_rgb[i, 0]} {c_rgb[i, 1]} {c_rgb[i, 2]} {err_val:.4f}\n"
                for i in range(len(c_xyz))
            ]
            f.writelines(lines)

    return n_sfm, n_lidar, n_total


def _copy_person_masks(dataset_dir: Path, masks_dir: Path, out_dir: Path, image_files) -> int:
    """Validate and export keep masks alongside the COLMAP image tree."""
    source_root = Path(masks_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Person mask directory does not exist: {source_root}")
    source_manifest = source_root / "manifest.json"
    manifest_data = None
    if source_manifest.is_file():
        manifest_data = json.loads(source_manifest.read_text(encoding="utf-8"))
        polarity = manifest_data.get("polarity")
        if polarity is not None and polarity not in ("white_keep_black_person", "white_keep_black_foreground"):
            raise ValueError(f"Unsupported person mask polarity in {source_manifest}: {polarity}")

    images_root = (Path(dataset_dir) / "images").resolve()
    mask_out = Path(out_dir) / "masks"
    expected = set()
    validated = []
    for image_path in image_files:
        relative = Path(image_path).resolve().relative_to(images_root)
        relative_mask = Path(relative.as_posix() + ".png")
        mask_path = source_root / relative_mask
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing person mask for {relative.as_posix()}: {mask_path}")
        image = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(mask_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or mask.shape != image.shape[:2]:
            raise ValueError(f"Person mask dimensions are invalid for {relative.as_posix()}")
        if not np.isin(mask, (0, 255)).all():
            raise ValueError(f"Person mask must use binary white-keep/black-exclude values: {mask_path}")
        expected.add(relative_mask.as_posix())
        validated.append((mask_path, relative_mask))

    if not validated:
        raise ValueError("No camera images were available for person mask export")
    mask_out.mkdir(parents=True, exist_ok=True)
    if source_root != mask_out.resolve():
        for old_mask in mask_out.rglob("*.jpg.png"):
            if old_mask.relative_to(mask_out).as_posix() not in expected:
                old_mask.unlink()
        for source_mask, relative_mask in validated:
            target = mask_out / relative_mask
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mask, target)

    if source_manifest.is_file():
        if source_root != mask_out.resolve():
            shutil.copy2(source_manifest, mask_out / "manifest.json")
    else:
        exported_manifest = {
            "schema": 1,
            "polarity": "white_keep_black_person",
            "naming": "image_relative_name_plus_png",
            "image_count": len(validated),
            "sources": {
                relative.with_suffix("").as_posix(): [
                    (Path(dataset_dir) / "images" / relative.with_suffix("")).stat().st_size,
                    (Path(dataset_dir) / "images" / relative.with_suffix("")).stat().st_mtime_ns,
                ]
                for _, relative in validated
            },
        }
        (mask_out / "manifest.json").write_text(
            json.dumps(exported_manifest, indent=2), encoding="utf-8"
        )
    print(f"  [+] Exported {len(validated)} validated person masks to: {mask_out}")
    return len(validated)


def _select_bounded_sfm_frames(timed_frames, frame_limit, source_fps=1.0, center_seconds=None):
    """Keep a temporally continuous clip for video-mode sequential matching."""
    if len(timed_frames) <= frame_limit:
        return timed_frames
    source_fps = float(source_fps)
    sample_fps = min(1.0, source_fps) if np.isfinite(source_fps) and source_fps > 0 else 1.0
    first_time = float(timed_frames[0][0])
    last_time = float(timed_frames[-1][0])
    duration = max(0.0, last_time - first_time)
    selected_count = min(frame_limit, max(1, int(np.floor(duration * sample_fps + 1e-9)) + 1))
    if selected_count >= len(timed_frames):
        return timed_frames

    window_duration = selected_count / sample_fps
    start_time = first_time + max(0.0, duration - window_duration) / 2.0
    if center_seconds is not None:
        start_time = max(first_time, min(float(center_seconds) - window_duration / 2, last_time - window_duration))
    end_time = start_time + window_duration
    window = [item for item in timed_frames
              if start_time - 1e-6 <= float(item[0]) <= end_time + 1e-6]
    if len(window) <= selected_count:
        return window
    window_times = np.asarray([item[0] for item in window])
    targets = window_times[0] + np.arange(selected_count) / sample_fps
    right = np.clip(np.searchsorted(window_times, targets), 0, len(window) - 1)
    left = np.maximum(0, right - 1)
    indices = np.unique(np.where(abs(window_times[left] - targets) <= abs(window_times[right] - targets), left, right))
    return [window[index] for index in indices]


def _preserve_sample_sfm_attempt(dataset_dir, sample_root, sample_images, metadata=None):
    """Keep every bounded-SfM attempt, including partial models rejected before alignment."""
    attempt_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')
    attempt_root = Path(dataset_dir) / 'calibration' / 'sample_sfm_attempt' / attempt_id
    sparse_source = Path(sample_root) / 'sparse'
    copied_sparse = []
    if sparse_source.is_dir():
        for source in sparse_source.rglob('*'):
            if not source.is_file():
                continue
            relative = source.relative_to(sparse_source)
            target = attempt_root / 'sparse' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_sparse.append(relative.as_posix())

    manifest_source = Path(sample_images) / 'frames.json'
    if manifest_source.is_file():
        manifest_target = attempt_root / 'images' / 'frames.json'
        manifest_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest_source, manifest_target)
    attempt_root.mkdir(parents=True, exist_ok=True)
    preserved = {
        'attempt_id': attempt_id,
        'sparse_files': copied_sparse,
        'sample_manifest': 'images/frames.json' if manifest_source.is_file() else None,
    }
    if isinstance(metadata, dict):
        preserved.update(metadata)
    (attempt_root / 'attempt.json').write_text(
        json.dumps(preserved, indent=2), encoding='utf-8')
    print(f'[*] Preserved bounded SfM attempt at: {attempt_root}')
    return attempt_root


def _validate_sample_sfm_component_coverage(sparse_dir, selected_files, attempt_root):
    """Require useful per-camera coverage in COLMAP's connected sparse/0 component."""
    from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images

    images_path = Path(sparse_dir) / 'images.bin'
    if not images_path.is_file():
        raise RuntimeError('Sample SfM did not produce sparse/0/images.bin')
    registered = {}
    for image in load_colmap_images(images_path):
        normalized = image['name'].replace('\\', '/')
        camera_name = normalized.split('/', 1)[0]
        registered[camera_name] = registered.get(camera_name, 0) + 1

    coverage = {}
    failures = []
    for camera_name, selected in selected_files.items():
        selected_count = len(selected)
        registered_count = registered.get(camera_name, 0)
        primary = camera_name in ('cam0', 'cam1')
        required_ratio = 0.60 if primary else 0.35
        required_count = max(20, int(np.ceil(selected_count * required_ratio)))
        coverage[camera_name] = {
            'registered_frames': registered_count,
            'selected_frames': selected_count,
            'required_frames': required_count,
            'required_fraction': required_ratio,
            'registered_fraction': (registered_count / selected_count) if selected_count else 0.0,
            'role': 'primary' if primary else 'auxiliary',
        }
        if selected_count < 20 or registered_count < required_count:
            failures.append(
                f'{camera_name} registered {registered_count}/{selected_count}; '
                f'requires at least {required_count} ({required_ratio:.0%}) in the connected component')

    metadata_path = Path(attempt_root) / 'attempt.json'
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        metadata = {}
    metadata['component_coverage'] = coverage
    metadata['coverage_status'] = 'rejected' if failures else 'candidate'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError('Bounded sample sparse/0 lacks per-camera coverage: ' + '; '.join(failures))
    print('[+] Sample sparse/0 camera coverage: ' + ', '.join(
        f'{name} {item["registered_frames"]}/{item["selected_frames"]}'
        for name, item in coverage.items()))
    return coverage


def _calibrate_aux_cameras_from_sample_sfm(dataset_dir, cameras, fps, dt_sync):
    """Calibrate the per-session rig from bounded SfM, then recover with all frames."""
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    from raven_app.third_camera import align_third_camera_to_rig, ThirdCameraConfig

    dataset_dir = Path(dataset_dir)
    images_root = dataset_dir / 'images'
    manifest_path = images_root / 'frames.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    all_times = manifest.get('timestamps', {})
    camera_names = ['cam0', 'cam1'] + [camera.camera_name for camera in cameras]
    per_camera_limit = max(20, min(40, 320 // max(1, len(camera_names))))
    camera_fps = {'cam0': float(fps), 'cam1': float(fps)}
    camera_fps.update({camera.camera_name: float(camera.fps) for camera in cameras})
    primary_times = [float(t) for name, t in all_times.items() if name.startswith('cam0/')]
    primary_times.extend(float(t) for name, t in all_times.items() if name.startswith('cam1/'))
    primary_start_seconds = min(primary_times) if primary_times else None
    primary_end_seconds = max(primary_times) if primary_times else None
    def filename_start(path):
        match = re.search(r'VID_(\d{8})_(\d{6})', Path(path).name, re.IGNORECASE)
        if not match:
            return None
        try:
            return datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S')
        except ValueError:
            return None
    extraction = manifest.get('extraction', {})
    primary_source = manifest.get('signature', {}).get('source', '') or extraction.get('source_path', '')
    if not primary_source:
        # Persisted manifests can store the source inside the extraction signature.
        signature = manifest.get('extraction_signature', {})
        primary_source = signature.get('source_path', '') if isinstance(signature, dict) else ''
    primary_start = filename_start(primary_source)
    timed_by_camera = {}
    for camera_name in camera_names:
        camera_dir = images_root / camera_name
        files = sorted(camera_dir.glob('*.jpg')) if camera_dir.is_dir() else []
        timed = []
        for path in files:
            relative = f'{camera_name}/{path.name}'
            timestamp = all_times.get(relative)
            if timestamp is None:
                try:
                    timestamp = frame_time(dataset_dir, relative, fps)
                except (KeyError, ValueError):
                    continue
            timed.append((float(timestamp), path, relative))
        timed.sort(key=lambda item: item[0])
        timed_by_camera[camera_name] = timed

    missing = [name for name in camera_names if len(timed_by_camera.get(name, ())) < 20]
    if missing:
        details = ', '.join(
            f'{name} has {len(timed_by_camera.get(name, ()))} timestamped frames'
            for name in missing)
        raise RuntimeError(
            f'Required sample SfM cannot include every camera: {details}; at least 20 are required per camera')

    trajectory_path = pipeline.get_slam_trajectory_path(dataset_dir)
    if not trajectory_path or not Path(trajectory_path).is_file():
        raise FileNotFoundError('FAST-LIVO2 trajectory is missing for sample camera calibration')

    auxiliary_by_name = {camera.camera_name: camera for camera in cameras}
    retry_time_hints = {}
    failures = []
    attempts = []
    last_failed_reports = {}
    best_attempt = None
    pose_support_attempts = []
    fractions = (0.50, 0.25, 0.75, None)

    def primary_center(fraction):
        if primary_start_seconds is not None and primary_end_seconds is not None:
            return primary_start_seconds + fraction * (primary_end_seconds - primary_start_seconds)
        primary = timed_by_camera.get('cam0') or timed_by_camera.get('cam1') or []
        return float(primary[0][0]) + fraction * (float(primary[-1][0]) - float(primary[0][0]))

    for attempt_index, fraction in enumerate(fractions, start=1):
        full_frame_recovery = fraction is None
        sample_method = ('full_frame_sfm_pose_recovery' if full_frame_recovery
                         else 'bounded_temporal_window')
        selected_files = {}
        selected_timestamps = {}
        windows = {}
        for camera_name in camera_names:
            timed = timed_by_camera[camera_name]
            if full_frame_recovery:
                center = None
                selected = list(timed)
            else:
                center = primary_center(fraction)
            if not full_frame_recovery and camera_name not in ('cam0', 'cam1'):
                camera = auxiliary_by_name[camera_name]
                if camera_name in retry_time_hints:
                    center += retry_time_hints[camera_name]
                else:
                    time_report = (camera.calibration_report or {}).get('time_offset', {}) or {}
                    if time_report.get('status') in ('configured', 'calibrated', 'verified'):
                        center += float(camera.time_offset_s)
                    else:
                        auxiliary_start = filename_start(camera.source_path)
                        if primary_start and auxiliary_start:
                            center -= (auxiliary_start - primary_start).total_seconds()
                        else:
                            center = float(timed[0][0]) + fraction * (float(timed[-1][0]) - float(timed[0][0]))
            if not full_frame_recovery:
                selected = _select_bounded_sfm_frames(
                    timed, per_camera_limit, source_fps=camera_fps[camera_name],
                    center_seconds=center)
            selected_files[camera_name] = selected
            if selected:
                selected_times = [float(item[0]) for item in selected]
                gaps = np.diff(selected_times)
                windows[camera_name] = {
                    'available_frames': len(timed), 'selected_frames': len(selected),
                    'source_fps': camera_fps[camera_name],
                    'center_seconds': float(center) if center is not None else None,
                    'start_seconds': selected_times[0], 'end_seconds': selected_times[-1],
                    'median_gap_seconds': float(np.median(gaps)) if len(gaps) else 0.0,
                    'mode': 'all_available_timestamped_frames' if full_frame_recovery else 'bounded_window',
                }
            for timestamp, _source, relative in selected:
                selected_timestamps[relative] = timestamp

        metadata = {
            'attempt_index': attempt_index,
            'window_fraction': fraction,
            'method': sample_method,
            'sample_windows': windows,
            'time_offset_hints_used_seconds': dict(retry_time_hints),
            'status': 'running',
        }
        attempt_error = None
        attempt_root = None
        calibrated = []
        candidate_root = None
        tempdir = None
        try:
            from contextlib import nullcontext
            tempdir = tempfile.TemporaryDirectory(prefix='raven-aux-sfm-', dir=str(dataset_dir))
            with nullcontext(tempdir.name) as tmp:
                sample_root = Path(tmp)
                candidate_root = sample_root
                sample_images = sample_root / 'images'
                for camera_name, selected in selected_files.items():
                    target_dir = sample_images / camera_name
                    target_dir.mkdir(parents=True)
                    for _timestamp, source, _relative in selected:
                        target = target_dir / source.name
                        try:
                            os.link(source, target)
                        except OSError:
                            shutil.copy2(source, target)
                sample_manifest = dict(manifest)
                sample_manifest['timestamps'] = selected_timestamps
                sample_manifest['sampled_sfm'] = not full_frame_recovery
                sample_manifest['full_frame_sfm_pose_recovery'] = full_frame_recovery
                sample_manifest['sample_windows'] = windows
                (sample_images / 'frames.json').write_text(
                    json.dumps(sample_manifest, indent=2), encoding='utf-8')
                short = [name for name, selected in selected_files.items() if len(selected) < 20]
                if short:
                    counts = ', '.join(f'{name} {len(selected_files[name])}/20' for name in short)
                    raise RuntimeError(f'This window cannot include every camera: {counts}')

                # Every attempt starts from the caller's original config.
                attempt_cameras = [ThirdCameraConfig.from_dict(camera.to_dict()) for camera in cameras]
                for camera in attempt_cameras:
                    if camera.camera_name in retry_time_hints:
                        report = dict(camera.calibration_report or {})
                        time_report = dict(report.get('time_offset', {}) or {})
                        time_report['sample_window_hint_seconds'] = retry_time_hints[camera.camera_name]
                        report['time_offset'] = time_report
                        camera.calibration_report = report
                if full_frame_recovery:
                    print('[*] Bounded calibration windows failed; starting full-frame SfM pose recovery. '
                          'This performs extra feature extraction and matching over every available '
                          f'timestamped frame ({sum(map(len, selected_files.values()))} frames).')
                else:
                    print('[*] Running bounded sample SfM for per-session camera calibration '
                          f'({sum(map(len, selected_files.values()))} frames, window {fraction:.0%}).')
                candidate = pipeline.run_spirula_sfm_auto(
                    sample_root, quality='medium', auxiliary_cameras=attempt_cameras,
                    allow_partial_calibration=True)
                if not candidate:
                    raise RuntimeError(
                        'Spirula did not produce a full-frame recovery candidate'
                        if full_frame_recovery else
                        'Spirula did not produce a bounded calibration candidate')
                try:
                    coverage = _validate_sample_sfm_component_coverage(
                        sample_root / 'sparse' / '0', selected_files, sample_root)
                except RuntimeError as exc:
                    if full_frame_recovery:
                        message = str(exc).replace(
                            'Bounded sample sparse/0', 'Full-frame recovery sparse/0')
                        raise RuntimeError(message) from exc
                    raise
                metadata.update(coverage_status='accepted', component_coverage=coverage)
                alignment = pipeline.align_colmap_to_lidar(
                    sample_root, fps=fps, dt_hint=dt_sync,
                    trajectory_path=trajectory_path, use_icp=False)
                if alignment.get('quality_status') not in (None, 'accepted'):
                    raise RuntimeError(
                        f"Sample SfM alignment rejected: {alignment.get('quality_status')} "
                        f"{alignment.get('quality_reason', '')}".strip())
                try:
                    scale = float(alignment['scale'])
                    rotation = np.asarray(alignment['R'], dtype=np.float64)
                    translation = np.asarray(alignment['t'], dtype=np.float64)
                    sync = float(alignment['dt_sync_seconds'])
                    rmse = float(alignment['trajectory_rmse_cm'])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f'Sample SfM alignment is incomplete: {exc}') from exc
                if (not np.isfinite(scale) or scale <= 0 or not np.isfinite(sync)
                        or not np.isfinite(rmse) or rmse < 0
                        or rotation.shape != (3, 3) or translation.shape != (3,)
                        or not np.all(np.isfinite(rotation))
                        or not np.all(np.isfinite(translation))
                        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                        or np.linalg.det(rotation) <= 0):
                    raise RuntimeError('Sample SfM alignment failed finite Sim(3) validation')
                support_alignment_accepted = alignment.get('quality_status') == 'accepted'
                if support_alignment_accepted and alignment.get('frame_time_source') == 'insv_timelapse':
                    from raven_app.timelapse_calibration import VERSION as TIMELAPSE_VERSION
                    support_alignment_accepted = (
                        alignment.get('timelapse_calibration_version') == TIMELAPSE_VERSION)
                if support_alignment_accepted:
                    sample_time_source = sample_manifest.get('time_source')
                    support_alignment_accepted = (
                        not sample_time_source
                        or alignment.get('frame_time_source') == sample_time_source)
                metadata.update(
                    alignment_status='accepted',
                    alignment_rmse_cm=alignment.get('trajectory_rmse_cm'),
                    alignment_method=alignment.get('alignment_method'))
                rig = pipeline.recalibrate_from_sfm(
                    sample_root, fps=fps, alignment=alignment, write_outputs=False,
                    trajectory_path=trajectory_path)

                camera_failures = []
                for camera in attempt_cameras:
                    try:
                        result = align_third_camera_to_rig(
                            sample_root, camera, fps=fps, alignment_data=alignment,
                            trajectory_path=trajectory_path)
                    except Exception as exc:
                        result = camera
                        report = dict(result.calibration_report or {})
                        report['calibration_attempt'] = {'status': 'failed', 'reason': str(exc)}
                        report['extrinsics'] = {
                            'status': 'uncalibrated', 'source': 'sample SfM attempt',
                            'camera_name': result.camera_name, 'reason': str(exc)}
                    else:
                        report = dict(result.calibration_report or {})
                    report['camera_name'] = result.camera_name
                    ready = _aux_camera_calibration_ready(result)
                    extrinsics = report.get('extrinsics', {}) or {}
                    sample_report = {
                        'status': 'complete' if ready else 'rejected',
                        'method': sample_method,
                        'attempt_index': attempt_index,
                        'window_fraction': fraction,
                        'frames_per_camera': {name: len(items) for name, items in selected_files.items()},
                        'alignment_rmse_cm': alignment.get('trajectory_rmse_cm'),
                        'alignment_method': alignment.get('alignment_method', 'timelapse_pose_consensus'),
                    }
                    if not ready:
                        sample_report['reason'] = (
                            extrinsics.get('reason') or
                            (report.get('time_offset', {}) or {}).get('reason') or
                            'intrinsics, time offset, or metric extrinsics failed the calibration gates')
                        camera_failures.append(f'{result.camera_name}: {sample_report["reason"]}')
                    report['sample_sfm'] = sample_report
                    result.calibration_report = report
                    calibrated.append(result)
                    hint = (report.get('time_offset', {}) or {}).get('best_candidate_seconds')
                    if hint is not None:
                        try:
                            hint = float(hint)
                            if np.isfinite(hint):
                                retry_time_hints[result.camera_name] = hint
                        except (TypeError, ValueError):
                            pass

                if best_attempt is None or rmse < best_attempt['alignment_rmse_cm']:
                    best_attempt = {
                        'alignment_rmse_cm': rmse,
                        'alignment': dict(alignment),
                        'rig': dict(rig),
                        'cameras': [ThirdCameraConfig.from_dict(camera.to_dict())
                                    for camera in calibrated],
                        'window_fraction': fraction,
                        'method': sample_method,
                        'frames_per_camera': {name: len(items)
                                              for name, items in selected_files.items()},
                        'attempt_index': attempt_index,
                    }

                # Keep exact registered poses only when the sample-to-SLAM
                # transform passed an explicit, current alignment acceptance.
                # These per-image poses do not claim a fixed camera baseline or
                # phone clock offset.
                if support_alignment_accepted and auxiliary_by_name:
                    selected_names = {
                        name: {relative for _time, _path, relative in selected_files[name]}
                        for name in auxiliary_by_name
                    }
                    allowed_models = {
                        'PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'THIN_PRISM_FISHEYE'}
                    attempt_support = {}
                    for result in calibrated:
                        report = result.calibration_report or {}
                        candidate = report.get('sample_sfm_pose_support') or {}
                        camera_name = result.camera_name
                        model = str(candidate.get('camera_model', '')).strip().upper()
                        intrinsics = candidate.get('intrinsics')
                        if (candidate.get('status') != 'candidate'
                                or model not in allowed_models
                                or model != str(result.camera_model).strip().upper()
                                or not isinstance(intrinsics, dict)):
                            continue
                        try:
                            checked_intrinsics = {
                                str(key): float(value) for key, value in intrinsics.items()}
                            if (not all(np.isfinite(value) for value in checked_intrinsics.values())
                                    or any(checked_intrinsics.get(key, 0.0) <= 0
                                           for key in ('width', 'height', 'fx', 'fy'))):
                                continue
                        except (TypeError, ValueError):
                            continue
                        normalized = []
                        for pose in candidate.get('poses', []):
                            if not isinstance(pose, dict):
                                continue
                            name = str(pose.get('name', '')).replace('\\', '/')
                            if (name not in selected_names.get(camera_name, set())
                                    or not (dataset_dir / 'images' / name).is_file()):
                                continue
                            try:
                                center = np.asarray(pose['C'], dtype=np.float64)
                                rotation_cw = np.asarray(pose['R_cw'], dtype=np.float64)
                                video_time = float(pose['video_time'])
                            except (KeyError, TypeError, ValueError):
                                continue
                            if (center.shape != (3,) or rotation_cw.shape != (3, 3)
                                    or not np.all(np.isfinite(center))
                                    or not np.all(np.isfinite(rotation_cw))
                                    or not np.isfinite(video_time)
                                    or not np.allclose(rotation_cw.T @ rotation_cw,
                                                       np.eye(3), atol=1e-3)
                                    or np.linalg.det(rotation_cw) <= 0):
                                continue
                            normalized.append({
                                'name': name, 'video_time': video_time,
                                'C': center.tolist(), 'R_cw': rotation_cw.tolist(),
                                'alignment_rmse_cm': rmse,
                                'sample_window_fraction': fraction,
                            })
                        if len(normalized) < 6:
                            continue
                        attempt_support[camera_name] = {
                            'poses': normalized,
                            'camera_model': model,
                            'intrinsics': checked_intrinsics,
                            'config': ThirdCameraConfig.from_dict(result.to_dict()),
                        }
                    support_missing = sorted(set(auxiliary_by_name) - set(attempt_support))
                    metadata['registered_pose_support'] = {
                        'status': 'candidate' if not support_missing else 'incomplete',
                        'registered_views_by_camera': {
                            name: len(item['poses']) for name, item in attempt_support.items()},
                        'missing_cameras': support_missing,
                        'alignment_rmse_cm': rmse,
                    }
                    if not support_missing:
                        accepted_coverage = sum(
                            int((coverage.get(name) or {}).get('registered_frames', 0))
                            for name in camera_names)
                        pose_support_attempts.append({
                            'attempt_index': attempt_index,
                            'method': sample_method,
                            'window_fraction': fraction,
                            'coverage_total': accepted_coverage,
                            'coverage': dict(coverage),
                            'alignment_rmse_cm': rmse,
                            'alignment': dict(alignment),
                            'camera_support': attempt_support,
                        })
                last_failed_reports = {
                    camera.camera_name: dict(camera.calibration_report or {})
                    for camera in calibrated
                }
                if camera_failures:
                    metadata['auxiliary_calibration'] = {
                        camera.camera_name: camera.calibration_report.get('sample_sfm', {})
                        for camera in calibrated}
                    raise RuntimeError('; '.join(camera_failures))

                # Publish the final primary rig and sample evidence only after
                # every enabled auxiliary camera passed its calibration gates.
                evidence = dataset_dir / 'calibration' / 'sample_sfm'
                (evidence / 'sparse' / '0').mkdir(parents=True, exist_ok=True)
                (evidence / 'images').mkdir(parents=True, exist_ok=True)
                for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
                    shutil.copy2(sample_root / 'sparse' / '0' / name,
                                 evidence / 'sparse' / '0' / name)
                shutil.copy2(sample_images / 'frames.json', evidence / 'images' / 'frames.json')
                (evidence / 'colmap_to_lidar_alignment.json').write_text(
                    json.dumps(alignment, indent=2), encoding='utf-8')
                rig['calibration_source'] = 'sample_sfm'
                rig['sample_sfm_evidence'] = 'calibration/sample_sfm'
                rig['sample_sfm_method'] = sample_method
                rig['sample_sfm_window_fraction'] = fraction
                rig['sample_sfm_frames_per_camera'] = {
                    name: len(items) for name, items in selected_files.items()}
                metadata.update(
                    status='passed', coverage_status='accepted',
                    alignment_status='accepted', component_coverage=coverage,
                    alignment_rmse_cm=alignment.get('trajectory_rmse_cm'),
                    auxiliary_calibration={
                        camera.camera_name: camera.calibration_report.get('sample_sfm', {})
                        for camera in calibrated})
                for name in ('rig_calibration.json', 'calibracao_rigida_auto.json'):
                    (dataset_dir / name).write_text(json.dumps(rig, indent=2), encoding='utf-8')
        except Exception as exc:
            attempt_error = exc
            metadata['status'] = 'failed'
            metadata['failure'] = str(exc)
            if calibrated:
                metadata.setdefault('auxiliary_calibration', {
                    camera.camera_name: camera.calibration_report.get('sample_sfm', {})
                    for camera in calibrated})
        finally:
            try:
                if candidate_root is not None:
                    attempt_root = _preserve_sample_sfm_attempt(
                        dataset_dir, candidate_root, candidate_root / 'images', metadata)
                    attempts.append(attempt_root)
            finally:
                if tempdir is not None:
                    tempdir.cleanup()

        if attempt_error is not None:
            label = ('full-frame recovery' if full_frame_recovery
                     else f'bounded window {fraction:.0%}')
            failures.append(f'{label}: {attempt_error}')
            print(f'[!] Sample SfM {label} failed: {attempt_error}')
            continue

        rig['sample_sfm_attempt'] = str(attempt_root.relative_to(dataset_dir)) if attempt_root else None
        for name in ('rig_calibration.json', 'calibracao_rigida_auto.json'):
            (dataset_dir / name).write_text(json.dumps(rig, indent=2), encoding='utf-8')
        if full_frame_recovery:
            print(f"[+] Primary and all auxiliary cameras calibrated from full-frame SfM: "
                  f"dt={rig['dt_sync_seconds']:.4f}s.")
        else:
            print(f"[+] Primary and all auxiliary cameras calibrated from sample SfM: "
                  f"dt={rig['dt_sync_seconds']:.4f}s; window {fraction:.0%}.")
        return calibrated

    if best_attempt is not None:
        best_cameras = {camera.camera_name: camera for camera in best_attempt['cameras']}
        # Keep poses from one COLMAP reconstruction only. Combining images from
        # independent bounded models can silently mix camera intrinsics or
        # incompatible local reconstructions even when each alignment passed.
        best_support = min(
            pose_support_attempts,
            key=lambda item: (-int(item['coverage_total']),
                              float(item['alignment_rmse_cm']),
                              int(item['attempt_index'])),
            default=None)
        resolved_cameras = []
        support_records = {}
        unresolved = []
        for original in cameras:
            candidate = best_cameras.get(original.camera_name)
            if candidate is not None and _aux_camera_calibration_ready(candidate):
                resolved_cameras.append(candidate)
                continue

            collected = ((best_support or {}).get('camera_support', {})
                         .get(original.camera_name))
            support_config = collected.get('config') if collected else None
            poses = list(collected.get('poses', [])) if collected else []
            if support_config is None or len(poses) < 6:
                unresolved.append(original.camera_name)
                continue
            alignment = (best_support or {}).get('alignment') or {}
            existing_report = dict(support_config.calibration_report or {})
            rigid_reason = (
                (existing_report.get('extrinsics', {}) or {}).get('reason')
                or (existing_report.get('time_offset', {}) or {}).get('reason')
                or 'A rigid time offset or camera-to-LiDAR transform did not pass calibration gates')
            existing_report.pop('sample_sfm_pose_support', None)
            existing_report['camera_name'] = original.camera_name
            existing_report['sample_sfm'] = {
                **(existing_report.get('sample_sfm', {}) or {}),
                'status': 'per_image_sfm_pose_support',
                'method': (best_support or {}).get(
                    'method', 'full_frame_sfm_pose_recovery'),
                'pose_source': 'registered_sample_sfm_poses_aligned_to_lidar',
                'reason': rigid_reason,
                'registered_views': len(poses),
                'accepted_registered_coverage': (best_support or {}).get('coverage', {}),
                'alignment_rmse_cm': (best_support or {}).get('alignment_rmse_cm'),
                'alignment_quality_status': alignment.get('quality_status'),
                'attempt_index': (best_support or {}).get('attempt_index'),
                'window_fraction': (best_support or {}).get('window_fraction'),
                'pose_support_path': 'calibration/auxiliary_pose_support.json',
                'attempts': [str(path.relative_to(dataset_dir)) for path in attempts],
            }
            support_config.calibration_report = existing_report
            support_records[original.camera_name] = {
                'status': 'accepted',
                'source': 'registered_sample_sfm_poses_aligned_to_lidar',
                'method': (best_support or {}).get(
                    'method', 'full_frame_sfm_pose_recovery'),
                'camera_name': original.camera_name,
                'camera_model': collected['camera_model'],
                'intrinsics': collected['intrinsics'],
                'alignment_quality_status': alignment.get('quality_status'),
                'alignment_frame_time_source': alignment.get('frame_time_source'),
                'alignment_calibration_version': alignment.get('calibration_version'),
                'alignment_rmse_cm': (best_support or {}).get('alignment_rmse_cm'),
                'sample_window_fraction': (best_support or {}).get('window_fraction'),
                'accepted_registered_coverage': (best_support or {}).get('coverage', {}),
                'attempt_index': (best_support or {}).get('attempt_index'),
                'minimum_poses': 6,
                'poses': sorted(poses, key=lambda pose: (pose['video_time'], pose['name'])),
            }
            resolved_cameras.append(support_config)

        if not unresolved:
            if support_records:
                from raven_app.auxiliary_pose_support import write_auxiliary_pose_support
                support_path = write_auxiliary_pose_support(
                    dataset_dir, support_records, trajectory_path=trajectory_path)
                print('[+] Auxiliary camera per-image SfM poses accepted for exact registered views: '
                      + ', '.join(
                          f'{name} ({len(record["poses"])} images)'
                          for name, record in support_records.items()))
                print(f'    Pose support: {support_path}')
            rig = dict(best_attempt['rig'])
            rig['calibration_source'] = 'sample_sfm'
            rig['sample_sfm_alignment_rmse_cm'] = best_attempt['alignment_rmse_cm']
            rig['sample_sfm_window_fraction'] = best_attempt['window_fraction']
            rig['sample_sfm_method'] = best_attempt.get('method', 'bounded_temporal_window')
            rig['sample_sfm_frames_per_camera'] = best_attempt.get('frames_per_camera', {})
            if best_support is not None:
                rig['auxiliary_pose_support_method'] = best_support['method']
                rig['auxiliary_pose_support_attempt_index'] = best_support['attempt_index']
                rig['auxiliary_pose_support_registered_coverage'] = best_support['coverage']
            rig['auxiliary_pose_support_path'] = (
                'calibration/auxiliary_pose_support.json' if support_records else None)
            rig['sample_sfm_attempts'] = [str(path.relative_to(dataset_dir)) for path in attempts]
            for name in ('rig_calibration.json', 'calibracao_rigida_auto.json'):
                (dataset_dir / name).write_text(json.dumps(rig, indent=2), encoding='utf-8')
            return resolved_cameras

    for camera in cameras:
        report = dict(last_failed_reports.get(camera.camera_name, camera.calibration_report or {}))
        report['camera_name'] = camera.camera_name
        report['sample_sfm'] = {
            'status': 'failed',
            'method': 'full_frame_sfm_pose_recovery',
            'methods_attempted': ['bounded_temporal_window', 'full_frame_sfm_pose_recovery'],
            'reason': 'Bounded calibration and full-frame SfM recovery failed: '
                      + '; '.join(failures),
            'attempts': [str(path.relative_to(dataset_dir)) for path in attempts],
            'last_attempt': dict(last_failed_reports.get(camera.camera_name, {})),
        }
        camera.calibration_report = report
    details = []
    for camera in cameras:
        report = camera.calibration_report or {}
        reason = ((report.get('extrinsics', {}) or {}).get('reason')
                  or (report.get('time_offset', {}) or {}).get('reason'))
        details.append(f'{camera.camera_name}: {reason or "calibration quality gates failed"}')
    attempts_text = ', '.join(str(path) for path in attempts)
    raise RuntimeError(
        'Required per-session sample SfM calibration failed for enabled cameras: '
        + '; '.join(details) + '; ' + '; '.join(failures) + f'. Attempts: {attempts_text}')


def _aux_camera_calibration_ready(camera):
    """Only use auxiliary views with calibrated intrinsics, time offset, and metric extrinsics."""
    report = getattr(camera, 'calibration_report', {}) or {}
    time_report = report.get('time_offset', {}) or {}
    time_status = time_report.get('status')
    return (
        report.get('intrinsics', {}).get('status') in ('sfm_calibrated', 'calibrated', 'verified')
        and time_status in ('calibrated', 'verified', 'configured')
        and time_report.get('applied', True) is not False
        and report.get('extrinsics', {}).get('status') == 'calibrated'
    )


def _aux_camera_pose_support_ready(
    dataset_dir, camera, *, require_sfm_source_signature=False
):
    """Return true only for fresh per-image poses from an accepted SfM alignment."""
    from raven_app.auxiliary_pose_support import load_camera_pose_support
    return load_camera_pose_support(
        Path(dataset_dir), getattr(camera, 'camera_name', ''),
        camera_model=getattr(camera, 'camera_model', None),
        intrinsics=getattr(camera, 'intrinsics', None),
        require_sfm_source_signature=require_sfm_source_signature) is not None


def _full_sfm_aux_pose_support_record(dataset_dir, camera):
    """Build support only from registered images in the current accepted full-SfM model."""
    report = getattr(camera, 'calibration_report', {}) or {}
    candidate = report.get('sample_sfm_pose_support') or {}
    if candidate.get('status') != 'candidate':
        return None, 'current SfM calibration did not provide registered per-image poses'
    quality_status = candidate.get('alignment_quality_status')
    if quality_status not in (None, 'accepted'):
        return None, f'COLMAP-to-LiDAR alignment was rejected: {quality_status}'

    from raven_app.auxiliary_pose_support import sfm_pose_source_signature
    from scripts.pipeline_auto_calibrator_and_colorizer import CALIBRATION_VERSION, load_colmap_images

    root = Path(dataset_dir)
    alignment_path = root / 'colmap_to_lidar_alignment.json'
    sparse_dir = root / 'sparse' / '0'
    try:
        alignment = json.loads(alignment_path.read_text(encoding='utf-8'))
        alignment_quality = alignment.get('quality_status')
        if (alignment.get('calibration_version') != CALIBRATION_VERSION
                or alignment_quality not in (None, 'accepted')
                or candidate.get('alignment_calibration_version') != CALIBRATION_VERSION):
            return None, 'the current COLMAP-to-LiDAR alignment is stale or not accepted'
        if (quality_status or 'accepted') != (alignment_quality or 'accepted'):
            return None, 'the registered poses do not match the current alignment quality state'
        manifest = json.loads((root / 'images' / 'frames.json').read_text(encoding='utf-8'))
        time_source = manifest.get('time_source', 'video_pts')
        alignment_time_source = alignment.get('frame_time_source', time_source)
        if (candidate.get('alignment_frame_time_source') != alignment_time_source
                or time_source != alignment_time_source):
            return None, 'the aligned poses use a different frame-time source'
        if time_source == 'insv_timelapse':
            from raven_app.timelapse_calibration import VERSION as TIMELAPSE_VERSION
            if (alignment.get('timelapse_calibration_version') != TIMELAPSE_VERSION
                    or candidate.get('alignment_timelapse_calibration_version') != TIMELAPSE_VERSION):
                return None, 'the timelapse alignment calibration is stale'
        saved_model_signature = candidate.get('sfm_pose_source_signature')
        if (not isinstance(saved_model_signature, dict)
                or saved_model_signature != sfm_pose_source_signature(root)):
            return None, 'the COLMAP model or alignment changed after its poses were generated'
        if not (sparse_dir / 'images.bin').is_file() or not (sparse_dir / 'cameras.bin').is_file():
            return None, 'the current COLMAP camera/image model is incomplete'
        registered = {
            str(image.get('name', '')).replace('\\', '/')
            for image in load_colmap_images(sparse_dir / 'images.bin')
        }
        timestamps = manifest.get('timestamps', {})
        model = str(candidate.get('camera_model', '')).strip().upper()
        if model != str(getattr(camera, 'camera_model', '')).strip().upper():
            return None, 'SfM poses do not use the selected auxiliary optical model'
        intrinsics = candidate.get('intrinsics')
        if not isinstance(intrinsics, dict):
            return None, 'SfM did not provide auxiliary camera intrinsics'
        checked_intrinsics = {str(key): float(value) for key, value in intrinsics.items()}
        if (not all(np.isfinite(value) for value in checked_intrinsics.values())
                or any(checked_intrinsics.get(key, 0.0) <= 0
                       for key in ('width', 'height', 'fx', 'fy'))):
            return None, 'SfM auxiliary camera intrinsics are invalid'
        if (str((report.get('intrinsics') or {}).get('status', ''))
                not in ('sfm_calibrated', 'calibrated', 'verified')):
            return None, 'SfM auxiliary camera intrinsics did not pass calibration'
        poses = []
        for pose in candidate.get('poses', []):
            if not isinstance(pose, dict):
                continue
            name = str(pose.get('name', '')).replace('\\', '/')
            if not name.startswith(f'{camera.camera_name}/') or name not in registered:
                continue
            try:
                video_time = float(pose['video_time'])
                manifest_time = float(timestamps[name])
            except (KeyError, TypeError, ValueError):
                continue
            if not np.isfinite(video_time) or abs(video_time - manifest_time) > 1e-6:
                continue
            poses.append({**pose, 'video_time': video_time})
        if len(poses) < 6:
            return None, f'only {len(poses)} current registered auxiliary poses are valid; six are required'
        return {
            'status': 'accepted',
            'source': 'registered_full_sfm_poses_aligned_to_lidar',
            'camera_name': camera.camera_name,
            'camera_model': model,
            'intrinsics': checked_intrinsics,
            'alignment_quality_status': 'accepted',
            'alignment_calibration_version': CALIBRATION_VERSION,
            'alignment_timelapse_calibration_version': alignment.get(
                'timelapse_calibration_version'),
            'alignment_frame_time_source': alignment_time_source,
            'alignment_rmse_cm': candidate.get('alignment_rmse_cm'),
            'minimum_poses': 6,
            'poses': poses,
            'sfm_pose_source_signature': saved_model_signature,
        }, None
    except (OSError, ValueError, TypeError, KeyError, ImportError, struct.error) as exc:
        return None, f'could not validate current registered SfM poses: {exc}'


_COLMAP_TEXT_MODEL_IDS = {
    'SIMPLE_PINHOLE': 0, 'PINHOLE': 1, 'SIMPLE_RADIAL': 2, 'RADIAL': 3,
    'OPENCV': 4, 'OPENCV_FISHEYE': 5, 'FULL_OPENCV': 6, 'FOV': 7,
    'SIMPLE_RADIAL_FISHEYE': 8, 'RADIAL_FISHEYE': 9, 'THIN_PRISM_FISHEYE': 10,
}


def _sync_colmap_text_binary(sparse_dir: Path) -> None:
    """Regenerate COLMAP binary camera/image files from their text counterparts."""
    from raven_app.third_camera import (
        _COLMAP_CAMERA_MODEL_NAMES,
        _COLMAP_CAMERA_PARAM_COUNTS,
    )

    sparse_dir = Path(sparse_dir)
    camera_txt = sparse_dir / 'cameras.txt'
    image_txt = sparse_dir / 'images.txt'
    if camera_txt.is_file():
        cameras = []
        for line in camera_txt.read_text(encoding='utf-8').splitlines():
            row = line.strip()
            if not row or row.startswith('#'):
                continue
            parts = row.split()
            if len(parts) < 5:
                raise ValueError(f'Invalid COLMAP camera row: {row}')
            camera_id, model_name, width, height = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            model_id = _COLMAP_TEXT_MODEL_IDS.get(model_name)
            if model_id is None:
                raise ValueError(f'Unsupported COLMAP camera model: {model_name}')
            params = [float(value) for value in parts[4:]]
            if len(params) != _COLMAP_CAMERA_PARAM_COUNTS[model_id]:
                raise ValueError(
                    f'{model_name} camera {camera_id} has {len(params)} parameters; '
                    f'expected {_COLMAP_CAMERA_PARAM_COUNTS[model_id]}'
                )
            cameras.append((camera_id, model_id, width, height, params))
        with (sparse_dir / 'cameras.bin').open('wb') as stream:
            stream.write(struct.pack('<Q', len(cameras)))
            for camera_id, model_id, width, height, params in cameras:
                stream.write(struct.pack('<iiQQ', camera_id, model_id, width, height))
                stream.write(struct.pack(f'<{len(params)}d', *params))

    if image_txt.is_file():
        rows = [line.rstrip('\r\n') for line in image_txt.read_text(encoding='utf-8').splitlines()]
        images = []
        index = 0
        while index < len(rows):
            row = rows[index].strip()
            if not row or row.startswith('#'):
                index += 1
                continue
            header = row.split(maxsplit=9)
            if len(header) != 10:
                raise ValueError(f'Invalid COLMAP image row: {rows[index]}')
            if index + 1 >= len(rows):
                raise ValueError(f'COLMAP images.txt is missing a POINTS2D row for {header[9]}')
            image_id = int(header[0])
            qvec = tuple(float(value) for value in header[1:5])
            tvec = tuple(float(value) for value in header[5:8])
            camera_id = int(header[8])
            name = header[9]
            tokens = rows[index + 1].split()
            if len(tokens) % 3:
                raise ValueError(f'Invalid COLMAP POINTS2D row for {name}')
            points2d = []
            for point_offset in range(0, len(tokens), 3):
                x, y = float(tokens[point_offset]), float(tokens[point_offset + 1])
                point_id = int(tokens[point_offset + 2])
                if point_id < -1:
                    raise ValueError(f'Invalid POINT3D_ID {point_id} for {name}')
                points2d.append((x, y, (1 << 64) - 1 if point_id == -1 else point_id))
            images.append((image_id, qvec, tvec, camera_id, name, points2d))
            index += 2
        camera_ids = {camera[0] for camera in cameras} if camera_txt.is_file() else set()
        if any(image[3] not in camera_ids for image in images):
            raise ValueError('COLMAP image references a camera id absent from cameras.txt')
        with (sparse_dir / 'images.bin').open('wb') as stream:
            stream.write(struct.pack('<Q', len(images)))
            for image_id, qvec, tvec, camera_id, name, points2d in images:
                stream.write(struct.pack('<I', image_id))
                stream.write(struct.pack('<4d', *qvec))
                stream.write(struct.pack('<3d', *tvec))
                stream.write(struct.pack('<I', camera_id))
                stream.write(name.encode('ascii') + b'\x00')
                stream.write(struct.pack('<Q', len(points2d)))
                for x, y, point_id in points2d:
                    stream.write(struct.pack('<2dQ', x, y, point_id))


def _validate_rigid_transform(value, label):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f'{label} must be a finite 4x4 rigid transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or np.linalg.det(rotation) <= 0
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3)):
        raise ValueError(f'{label} does not contain a proper rigid rotation')
    return matrix


def _colmap_model_parameters(intrinsics, lens_model, camera_id):
    model = str(intrinsics.get('camera_model') or intrinsics.get('lens_model') or lens_model or 'OPENCV_FISHEYE').upper()
    width, height = int(intrinsics.get('width', 3840)), int(intrinsics.get('height', 3840))
    fx, fy, cx, cy = (float(intrinsics[key]) for key in ('fx', 'fy', 'cx', 'cy'))
    values = {
        'SIMPLE_PINHOLE': [fx, cx, cy],
        'PINHOLE': [fx, fy, cx, cy],
        'SIMPLE_RADIAL': [fx, cx, cy, float(intrinsics.get('k1', 0.0))],
        'RADIAL': [fx, cx, cy, float(intrinsics.get('k1', 0.0)), float(intrinsics.get('k2', 0.0))],
        'OPENCV': [fx, fy, cx, cy, *[float(intrinsics.get(key, 0.0)) for key in ('k1', 'k2', 'p1', 'p2')]],
        'OPENCV_FISHEYE': [fx, fy, cx, cy, *[float(intrinsics.get(key, 0.0)) for key in ('k1', 'k2', 'k3', 'k4')]],
        'FULL_OPENCV': [fx, fy, cx, cy, *[float(intrinsics.get(key, 0.0)) for key in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')]],
        'FOV': [fx, fy, cx, cy, float(intrinsics.get('omega', 0.0))],
        'SIMPLE_RADIAL_FISHEYE': [fx, cx, cy, float(intrinsics.get('k1', 0.0))],
        'RADIAL_FISHEYE': [fx, cx, cy, float(intrinsics.get('k1', 0.0)), float(intrinsics.get('k2', 0.0))],
        'THIN_PRISM_FISHEYE': [fx, fy, cx, cy, *[float(intrinsics.get(key, 0.0)) for key in
            ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')]],
    }
    if model not in values:
        raise ValueError(f'Unsupported calibrated COLMAP camera model for camera {camera_id}: {model}')
    if not np.all(np.isfinite(values[model])):
        raise ValueError(f'Camera {camera_id} intrinsics contain non-finite values')
    return model, width, height, values[model]


def _valid_pose_window(dataset_dir, name, fps, dt_sync, slam_duration, offset=0.0):
    try:
        video_time = frame_time(dataset_dir, name, fps)
    except (KeyError, OSError, ValueError):
        return False
    query = video_time - float(dt_sync) - float(offset)
    return bool(np.isfinite(query) and 0.0 <= query <= slam_duration)


def _link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def _auxiliary_camera_model_matches(camera, sparse_dir):
    from raven_app.third_camera import _COLMAP_CAMERA_MODEL_NAMES, _read_colmap_cameras
    selected = str(getattr(camera, 'camera_model', 'PINHOLE')).upper()
    expected = {
        'PINHOLE': 'PINHOLE', 'OPENCV': 'OPENCV',
        'OPENCV_FISHEYE': 'OPENCV_FISHEYE',
        'THIN_PRISM_FISHEYE': 'THIN_PRISM_FISHEYE',
    }.get(selected)
    if expected is None:
        return False
    images_bin = Path(sparse_dir) / 'images.bin'
    cameras_bin = Path(sparse_dir) / 'cameras.bin'
    if not images_bin.is_file() or not cameras_bin.is_file():
        return False
    from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images
    prefix = f"{getattr(camera, 'camera_name', 'cam2')}/"
    records = [item for item in load_colmap_images(images_bin) if item['name'].replace('\\', '/').startswith(prefix)]
    if not records:
        return True
    cameras = _read_colmap_cameras(cameras_bin)
    names = _COLMAP_CAMERA_MODEL_NAMES
    return all(names.get(cameras[item['cam_id']]['model_id']) == expected for item in records)


def _clear_colmap_image_observations(sparse_dir):
    """Keep camera poses while clearing feature links after replacing SfM points by LiDAR seed points."""
    path = Path(sparse_dir) / 'images.txt'
    if not path.is_file():
        return
    rows = path.read_text(encoding='utf-8').splitlines()
    output = []
    index = 0
    while index < len(rows):
        line = rows[index]
        output.append(line)
        if line.strip() and not line.lstrip().startswith('#'):
            if index + 1 >= len(rows):
                raise ValueError('COLMAP images.txt is missing a POINTS2D row')
            points = rows[index + 1].split()
            if len(points) % 3:
                raise ValueError(f'Invalid COLMAP POINTS2D row after: {line}')
            output.append(' '.join(f'{points[i]} {points[i+1]} -1' for i in range(0, len(points), 3)))
            index += 2
        else:
            index += 1
    path.write_text('\n'.join(output) + '\n', encoding='utf-8')


def export_colmap_3dgs(dataset_dir: Path, calib_path: Path, fps: float = 2.0, dt_sync: float = 0.0,
                       max_points: Optional[int] = None, masks_dir: Optional[Path] = None,
                       third_camera: Optional[Any] = None, seed_percent: float = 100.0,
                       pose_source: str = 'auto', export_path: Optional[Path] = None) -> Path:
    """Build a clean COLMAP export, preserving any prior export as a recoverable backup."""
    dataset_dir = Path(dataset_dir)
    target_dir = Path(export_path) if export_path is not None else dataset_dir / 'colmap_3dgs'
    if export_path is not None and target_dir.exists():
        raise FileExistsError(f'COLMAP export directory already exists: {target_dir}')
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f'.{target_dir.name}.stage-', dir=str(target_dir.parent)))
    try:
        _build_colmap_3dgs_dataset(
            dataset_dir, calib_path, fps, dt_sync, max_points, masks_dir, third_camera,
            seed_percent, pose_source, stage_dir,
        )
        backup_dir = None
        if target_dir.exists():
            stamp = time.strftime('%Y%m%d_%H%M%S')
            backup_dir = target_dir.with_name(f'{target_dir.name}.previous-{stamp}')
            suffix = 1
            while backup_dir.exists():
                backup_dir = target_dir.with_name(f'{target_dir.name}.previous-{stamp}-{suffix}')
                suffix += 1
            os.replace(target_dir, backup_dir)
        try:
            os.replace(stage_dir, target_dir)
        except Exception:
            if backup_dir is not None and backup_dir.exists() and not target_dir.exists():
                os.replace(backup_dir, target_dir)
            raise
        _postprocess_3dgs_dataset(target_dir, dataset_dir)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        raise
    print(f"[+] COLMAP 3DGS dataset generation complete: {target_dir}")
    return target_dir


def _build_colmap_3dgs_dataset(dataset_dir, calib_path, fps, dt_sync, max_points,
                              masks_dir, third_camera, seed_percent, pose_source, out_dir):
    """Generate a complete COLMAP dataset configured for 3D Gaussian Splatting (3DGS) training.

    Transforms camera poses into the LiDAR metric coordinate frame and seeds the model with
    the true metric LiDAR point cloud (points3D.ply, points3D.bin, points3D.txt).
    Defaults to 100% of the metric LiDAR cloud; seed_percent budgets the exported seed.
    If SfM tie-points exist from reconstruction, they are preserved separately in points3D_sfm_sparse.*.
    """
    from raven_app.seed_export import validate_seed_percent
    seed_percent = validate_seed_percent(seed_percent)
    dataset_dir = Path(dataset_dir)
    out_dir = Path(out_dir)
    pose_source = str(pose_source).strip().lower()
    if pose_source not in ('auto', 'direct', 'sfm'):
        raise ValueError("pose_source must be 'auto', 'direct', or 'sfm'")
    images_out = out_dir / "images"
    sparse_out = out_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)
    print(f"\n[*] Generating COLMAP Dataset for 3DGS Training under: {out_dir}")

    third_cameras = (third_camera if isinstance(third_camera, (list, tuple))
                     else [third_camera] if third_camera else [])
    camera_cfg_by_name = {
        str(getattr(camera, 'camera_name', 'cam2')): camera for camera in third_cameras
        if getattr(camera, 'enabled', True)
    }

    sfm_sparse = dataset_dir / "sparse" / "0"
    if not (sfm_sparse / "points3D.bin").is_file():
        alt = dataset_dir / "spirula_out" / "workspace" / "sparse" / "0"
        if (alt / "points3D.bin").is_file():
            sfm_sparse = alt
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"
    has_sfm = (sfm_sparse / 'points3D.bin').is_file() and align_json.is_file()
    if pose_source == 'auto':
        pose_source = 'sfm' if has_sfm else 'direct'
    if pose_source == 'sfm' and not has_sfm:
        raise FileNotFoundError('SfM pose source was requested, but sparse points or COLMAP-to-LiDAR alignment is missing')

    from scripts.pipeline_auto_calibrator_and_colorizer import get_slam_trajectory_path
    trj_path = get_slam_trajectory_path(dataset_dir)
    if not trj_path.is_file():
        raise FileNotFoundError(f'SLAM trajectory not found: {trj_path}')
    trajectory = np.loadtxt(trj_path, ndmin=2)
    if (trajectory.ndim != 2 or trajectory.shape[1] < 8 or len(trajectory) < 2
            or not np.all(np.isfinite(trajectory[:, :8]))
            or np.any(np.diff(trajectory[:, 0]) <= 0)):
        raise ValueError(f'Malformed SLAM trajectory: {trj_path}')
    t_slam = trajectory[:, 0]
    pos_slam = trajectory[:, 1:4]
    rot_slam = Rot.from_quat(trajectory[:, 4:8])
    from scipy.spatial.transform import Slerp
    slerp = Slerp(t_slam, rot_slam)
    slam_duration = float(t_slam[-1] - t_slam[0])

    direct_calib = None
    alignment = None
    sync_for_export = float(dt_sync)
    source_image_records = []
    allowed_names = set()
    selected_images = []
    auxiliary_decisions = {}
    auxiliary_pose_records = {}
    if pose_source == 'sfm':
        alignment = json.loads(align_json.read_text(encoding='utf-8'))
        if alignment.get('quality_status') not in (None, 'accepted'):
            raise ValueError(f"SfM alignment is not accepted: {alignment.get('quality_status')}")
        sync_for_export = float(alignment.get('dt_sync_seconds', dt_sync))
        scale = float(alignment.get('scale', 0.0))
        rotation = np.asarray(alignment.get('R'), dtype=np.float64)
        translation = np.asarray(alignment.get('t'), dtype=np.float64)
        if (not np.isfinite(sync_for_export) or not np.isfinite(scale) or scale <= 0
                or rotation.shape != (3, 3) or translation.shape != (3,)
                or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
                or np.linalg.det(rotation) <= 0):
            raise ValueError('SfM alignment contains invalid scale, rotation, translation, or time offset')
        if alignment.get('frame_time_source') == 'insv_timelapse':
            from raven_app.timelapse_calibration import VERSION as TIMELAPSE_VERSION
            if (alignment.get('quality_status') != 'accepted'
                    or alignment.get('timelapse_calibration_version') != TIMELAPSE_VERSION):
                raise ValueError('Timelapse SfM alignment is stale or not accepted')
        from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images
        image_bin = sfm_sparse / 'images.bin'
        if not image_bin.is_file() or not (sfm_sparse / 'cameras.bin').is_file():
            raise FileNotFoundError(f'SfM camera/image model is incomplete: {sfm_sparse}')
        source_image_records = load_colmap_images(image_bin)
        accepted_value = alignment.get('accepted_image_names')
        accepted_names = ({str(name).replace('\\', '/') for name in accepted_value}
                          if isinstance(accepted_value, list) else None)
        rejected_names = {str(name).replace('\\', '/')
                          for name in alignment.get('rejected_image_names', [])}
        for record in source_image_records:
            name = str(record['name']).replace('\\', '/')
            camera_name = name.split('/', 1)[0]
            if camera_name in ('cam0', 'cam1'):
                if ((accepted_names is not None and name not in accepted_names)
                        or (accepted_names is None and name in rejected_names)):
                    continue
                offset = 0.0
            else:
                camera = camera_cfg_by_name.get(camera_name)
                if camera is None:
                    auxiliary_decisions[camera_name] = 'excluded: camera is not enabled'
                    continue
                if not _auxiliary_camera_model_matches(camera, sfm_sparse):
                    auxiliary_decisions[camera_name] = 'excluded: sparse model differs from selected optical model'
                    continue
                if _aux_camera_calibration_ready(camera):
                    auxiliary_decisions[camera_name] = 'accepted: calibrated sparse poses'
                    offset = float(getattr(camera, 'time_offset_s', 0.0))
                    if not _valid_pose_window(
                            dataset_dir, name, fps, sync_for_export, slam_duration, offset):
                        continue
                elif alignment.get('quality_status') == 'accepted':
                    # A registered SfM pose is already in the aligned world frame;
                    # it does not need a fabricated per-camera clock offset.
                    auxiliary_decisions[camera_name] = 'accepted: registered aligned SfM pose'
                    offset = None
                else:
                    auxiliary_decisions[camera_name] = 'excluded: no accepted alignment or rigid calibration'
                    continue
            if camera_name in ('cam0', 'cam1') and not _valid_pose_window(
                    dataset_dir, name, fps, sync_for_export, slam_duration, 0.0):
                continue
            image_path = dataset_dir / 'images' / Path(name)
            if not image_path.is_file():
                continue
            allowed_names.add(name)
            selected_images.append((name, image_path))
        if not any(name.startswith(('cam0/', 'cam1/')) for name in allowed_names):
            raise ValueError('No accepted cam0/cam1 SfM poses remain inside the SLAM time window')
        print(f"  [*] SfM pose filter: {len(allowed_names)} accepted/in-window images; "
              f"{len(source_image_records) - len(allowed_names)} rejected, out-of-window, uncalibrated, or missing")
    else:
        direct_calib = json.loads(Path(calib_path).read_text(encoding='utf-8'))
        for camera_id, camera_key in ((1, 'cam0_front_intrinsics'), (2, 'cam1_rear_intrinsics')):
            _colmap_model_parameters(direct_calib[camera_key], direct_calib.get('lens_model'), camera_id)
            _validate_rigid_transform(
                direct_calib[f'T_lidar_to_cam{camera_id - 1}_rigid_4x4'],
                f'LiDAR-to-cam{camera_id - 1} calibration',
            )
        for camera_name in ('cam0', 'cam1'):
            source_dir = dataset_dir / 'images' / camera_name
            for image_path in sorted(source_dir.glob('*.jpg')) if source_dir.is_dir() else []:
                name = f'{camera_name}/{image_path.name}'
                if _valid_pose_window(dataset_dir, name, fps, sync_for_export, slam_duration):
                    allowed_names.add(name)
                    selected_images.append((name, image_path))
        for camera_name, camera in camera_cfg_by_name.items():
            if not _aux_camera_calibration_ready(camera):
                from raven_app.auxiliary_pose_support import load_camera_pose_support
                support = load_camera_pose_support(
                    dataset_dir, camera_name,
                    camera_model=getattr(camera, 'camera_model', None),
                    intrinsics=getattr(camera, 'intrinsics', None),
                    trajectory_path=trj_path)
                if support is None:
                    auxiliary_decisions[camera_name] = 'excluded: no rigid calibration or fresh registered SfM poses'
                    print(f'  [!] Skipping {camera_name}: no rigid calibration or fresh registered SfM poses')
                    continue
                try:
                    _colmap_model_parameters(
                        support['intrinsics'], support['camera_model'], 3)
                except (KeyError, TypeError, ValueError) as exc:
                    auxiliary_decisions[camera_name] = f'excluded: invalid SfM optical model ({exc})'
                    print(f'  [!] Skipping {camera_name}: invalid SfM optical model ({exc})')
                    continue
                auxiliary_pose_records[camera_name] = support
                auxiliary_decisions[camera_name] = (
                    f'accepted: {len(support["poses"])} registered sample SfM poses')
                for pose in support['poses']:
                    name = pose['name']
                    image_path = Path(pose['image_path'])
                    if name not in allowed_names and image_path.is_file():
                        allowed_names.add(name)
                        selected_images.append((name, image_path))
                print(f'  [+] Including {len(support["poses"])} exact registered SfM poses for {camera_name}')
                continue
            try:
                _validate_rigid_transform(
                    getattr(camera, 'T_lidar_to_cam2_rigid_4x4', None),
                    f'LiDAR-to-{camera_name} calibration',
                )
                _colmap_model_parameters(
                    getattr(camera, 'intrinsics', {}) or {},
                    getattr(camera, 'camera_model', 'PINHOLE'), 3,
                )
                if str(getattr(camera, 'camera_model', 'PINHOLE')).upper() not in (
                        'PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'THIN_PRISM_FISHEYE'):
                    raise ValueError(f'Unsupported selected COLMAP model for {camera_name}')
            except ValueError as exc:
                auxiliary_decisions[camera_name] = f'excluded: {exc}'
                print(f'  [!] Skipping {camera_name}: {exc}')
                continue
            auxiliary_decisions[camera_name] = 'accepted: SLAM synthesis'
            source_dir = dataset_dir / 'images' / camera_name
            for image_path in sorted(source_dir.glob('*.jpg')) if source_dir.is_dir() else []:
                name = f'{camera_name}/{image_path.name}'
                if _valid_pose_window(dataset_dir, name, fps, sync_for_export, slam_duration,
                                      float(getattr(camera, 'time_offset_s', 0.0))):
                    allowed_names.add(name)
                    selected_images.append((name, image_path))
        if not any(name.startswith(('cam0/', 'cam1/')) for name in allowed_names):
            raise ValueError('No cam0/cam1 images overlap the SLAM trajectory time window')

    camera_files = []
    for name, source in selected_images:
        target = images_out / Path(name)
        _link_or_copy(source, target)
        camera_files.append(source)
    if masks_dir is not None:
        _copy_person_masks(dataset_dir, masks_dir, out_dir, camera_files)

    poses_ready = False
    if pose_source == 'sfm':
        print("  [*] Transforming Spirula SfM cameras and images into true LiDAR METRIC frame via Sim(3)...")
        from scripts.pipeline_auto_calibrator_and_colorizer import transform_colmap_to_metric
        transform_colmap_to_metric(
            sfm_sparse, sparse_out, alignment['scale'],
            np.asarray(alignment['R'], dtype=np.float64),
            np.asarray(alignment['t'], dtype=np.float64),
            include_images=allowed_names,
        )
        poses_ready = True

    if not poses_ready:
        print("  [*] Using requested direct pose source: SLAM trajectory + calibrated rig")
        calib = direct_calib
        c0 = calib["cam0_front_intrinsics"]
        c1 = calib["cam1_rear_intrinsics"]
        T_LC0 = _validate_rigid_transform(calib["T_lidar_to_cam0_rigid_4x4"], 'LiDAR-to-cam0 calibration')
        T_LC1 = _validate_rigid_transform(calib["T_lidar_to_cam1_rigid_4x4"], 'LiDAR-to-cam1 calibration')
        R_LC0, t_LC0 = T_LC0[:3, :3], T_LC0[:3, 3]
        R_LC1, t_LC1 = T_LC1[:3, :3], T_LC1[:3, 3]

        cameras_txt = sparse_out / "cameras.txt"
        with open(cameras_txt, "w", encoding="utf-8") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            for camera_id, intrinsics in ((1, c0), (2, c1)):
                model, width, height, params = _colmap_model_parameters(
                    intrinsics, calib.get('lens_model'), camera_id)
                f.write(f"{camera_id} {model} {width} {height} "
                        f"{' '.join(format(value, '.17g') for value in params)}\n")
        print("  [+] Written cameras.txt")
        images_txt = sparse_out / "images.txt"
        with open(images_txt, "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            image_id = 1
            for camera_id, camera_name, rotation_lc, translation_lc in (
                (1, 'cam0', R_LC0, t_LC0), (2, 'cam1', R_LC1, t_LC1),
            ):
                for name, image_path in selected_images:
                    if not name.startswith(f'{camera_name}/'):
                        continue
                    video_time = frame_time(dataset_dir, name, fps)
                    t_query = t_slam[0] + (video_time - sync_for_export)
                    if not (t_slam[0] <= t_query <= t_slam[-1]):
                        continue
                    index = int(np.clip(np.searchsorted(t_slam, t_query), 1, len(t_slam) - 1))
                    weight = (t_query - t_slam[index - 1]) / (t_slam[index] - t_slam[index - 1])
                    p_lidar = (1.0 - weight) * pos_slam[index - 1] + weight * pos_slam[index]
                    R_lidar = slerp(t_query).as_matrix()
                    p_cam = p_lidar + R_lidar @ translation_lc
                    R_world_cam = R_lidar @ rotation_lc
                    R_cw = R_world_cam.T
                    t_cw = -R_cw @ p_cam
                    q_cw = Rot.from_matrix(R_cw).as_quat()
                    f.write(
                        f"{image_id} {q_cw[3]:.17g} {q_cw[0]:.17g} {q_cw[1]:.17g} {q_cw[2]:.17g} "
                        f"{t_cw[0]:.17g} {t_cw[1]:.17g} {t_cw[2]:.17g} {camera_id} {name}\n\n"
                    )
                    image_id += 1
        print(f"  [+] Written direct images.txt with {image_id - 1} camera poses")

    # Seed 3DGS point cloud exclusively with true metric LiDAR points (full raw points)
    lidar_xyz, lidar_rgb = load_lidar_seed_points(
        dataset_dir, max_points=max_points, seed_percent=seed_percent,
        preferred_method=pose_source)
    if lidar_xyz is not None and len(lidar_xyz) > 0:
        # If previous SfM points existed from reconstruction, preserve them as points3D_sfm_sparse
        if (sparse_out / "points3D.bin").is_file() and not (sparse_out / "points3D_sfm_sparse.bin").is_file():
            try:
                shutil.copy2(sparse_out / "points3D.bin", sparse_out / "points3D_sfm_sparse.bin")
                if (sparse_out / "points3D.ply").is_file():
                    shutil.copy2(sparse_out / "points3D.ply", sparse_out / "points3D_sfm_sparse.ply")
                if (sparse_out / "points3D.txt").is_file():
                    shutil.copy2(sparse_out / "points3D.txt", sparse_out / "points3D_sfm_sparse.txt")
            except Exception:
                pass

        method_tag = "Reconstruction-Based" if poses_ready else "Trajectory-Based"
        print(f"  [*] [{method_tag} 3DGS] Writing {seed_percent:g}% metric LiDAR seed "
              f"({len(lidar_xyz):,} points) to points3D.ply, points3D.bin, points3D.txt...")
        write_colmap_points3d(sparse_out, lidar_xyz, lidar_rgb)
        _clear_colmap_image_observations(sparse_out)
        print(f"  [+] LiDAR 3DGS seed successfully written to: {sparse_out / 'points3D.ply'}")
    else:
        print("  [*] Using existing sparse points3D as 3DGS seed (no LiDAR cloud available).")

    if pose_source == 'direct' and third_cameras:
        from raven_app.third_camera import synthesize_third_camera_poses, inject_third_camera_into_colmap
        for camera_index, camera in enumerate(third_cameras, start=3):
            if not _aux_camera_calibration_ready(camera):
                support = auxiliary_pose_records.get(camera.camera_name)
                if support is None:
                    print(f"  [!] Skipping {camera.camera_name}: no rigid or registered SfM poses")
                    continue
                poses = []
                for item in support['poses']:
                    rotation_cw = np.asarray(item['R_cw'], dtype=np.float64)
                    center = np.asarray(item['C'], dtype=np.float64)
                    translation_cw = -rotation_cw @ center
                    quaternion = Rot.from_matrix(rotation_cw).as_quat()
                    poses.append({
                        'name': item['name'], 'R_cw': rotation_cw,
                        't_cw': translation_cw, 'q_cw': quaternion,
                        'C': center, 'camera_id': camera_index,
                    })
                if poses:
                    inject_third_camera_into_colmap(
                        sparse_out, camera, poses, camera_id=camera_index)
                    print(f"  [+] Injected {len(poses)} registered sample SfM poses "
                          f"({camera.camera_name}) into 3DGS COLMAP dataset")
                continue
            try:
                poses = synthesize_third_camera_poses(
                    dataset_dir, camera, fps=fps, dt_sync=sync_for_export)
                if poses:
                    inject_third_camera_into_colmap(
                        sparse_out, camera, poses, camera_id=camera_index)
                    print(f"  [+] Injected {len(poses)} poses ({camera.camera_name}) into 3DGS COLMAP dataset")
            except Exception as e:
                print(f"  [!] Warning: Failed to inject {camera.camera_name} poses: {e}")

    _sync_colmap_text_binary(sparse_out)
    from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images
    exported_images = load_colmap_images(sparse_out / 'images.bin')
    counts_by_camera = {}
    for image in exported_images:
        camera_name = image['name'].replace('\\', '/').split('/', 1)[0]
        counts_by_camera[camera_name] = counts_by_camera.get(camera_name, 0) + 1
    report = {
        'pose_source': pose_source,
        'dt_sync_seconds': sync_for_export,
        'slam_duration_seconds': slam_duration,
        'exported_images_by_camera': counts_by_camera,
        'excluded_sfm_images': (len(source_image_records) - len(allowed_names)
                                if pose_source == 'sfm' else 0),
        'auxiliary_cameras': auxiliary_decisions,
        'lidar_seed_percent': seed_percent,
    }
    (out_dir / 'colmap_export_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def setup_sparse_compatibility(out_dir: Path):
    """Ensure both colmap_3dgs and colmap_3dgs/sparse work seamlessly in Spirula Studio without duplicating files."""
    sparse_dir = out_dir / "sparse"
    sparse_0 = sparse_dir / "0"
    if sparse_0.is_dir():
        for fname in ("cameras.bin", "cameras.txt", "images.bin", "images.txt", "points3D.bin", "points3D.txt", "points3D.ply"):
            src = sparse_0 / fname
            dst = sparse_dir / fname
            if src.is_file() and not dst.exists():
                try:
                    os.link(src, dst)
                except Exception:
                    try:
                        dst.symlink_to(src)
                    except Exception:
                        pass


def _postprocess_3dgs_dataset(out_dir: Path, dataset_dir: Path):
    """Adds point cloud seed, compatibility links, and instructions to the 3DGS dataset."""

    readme_txt = out_dir / "README_3DGS_DATASET.txt"
    try:
        readme_txt.write_text(
            "================================================================================\n"
            "3D GAUSSIAN SPLATTING (3DGS) DATASET GUIDE\n"
            "================================================================================\n\n"
            "This dataset has been transformed into true metric LiDAR scale and is configured\n"
            "for training 3D Gaussian Splatting models.\n\n"
            "HOW TO LOAD IN SPIRULA STUDIO:\n"
            "1. Open Spirula Studio.\n"
            "2. Click 'Open Dataset' (or 'Change...') and select THIS folder:\n"
            f"   {out_dir}\n"
            "   (Important: Select this folder, NOT the 'sparse' subfolder).\n"
            "3. Choose preset 'General purpose (3dgs)' or '360-camera'.\n"
            "4. Click Train.\n\n"
            "COMPATIBILITY:\n"
            "- Spirula Studio: Native COLMAP dataset.\n"
            "- Nerfstudio / PostShot / LichtFeld Studio: Supported via standard COLMAP models.\n"
            "\nOPTIONAL PERSON MASKS:\n"
            "- When present, masks are under masks/ with the same camera-relative image name plus .png (for example, cam0/frame.jpg.png).\n"
            "- White (255) means keep; black (0) means exclude the person.\n"
            "- Configure your trainer to read these masks explicitly; trainers do not all load them automatically.\n"
            "================================================================================\n",
            encoding="utf-8"
        )
    except Exception:
        pass

    setup_sparse_compatibility(out_dir)


def execute_unified_workflow(
    bag_path: Path,
    insv_path: Path,
    output_dir: Path,
    *,
    lio: bool = True,
    threads: int = 4,
    lidar_topic: str = "/vanjee_722z",
    imu_topic: str = "/vanjee_imu_packets",
    fps: float = 2.0,
    method: str = "sfm",
    calib_json: Path = None,
    dt_override: float = None,
    recalibrate: bool = True,
    run_spirula: bool = False,
    use_vulkan: bool = True,
    export_ply: bool = True,
    export_pcd: bool = True,
    export_colmap: bool = True,
    seed_percent: float = 100.0,
    process_gps: bool = False,
    gps_formats=('geojson', 'gpx', 'csv'),
    geo_formats=('laz', 'geojson'),
    mask_persons: bool = False,
    masks_dir: Path = None,
    mask_model: Path = None,
    mask_backend: str = "vulkan",
    mask_threshold: float = 0.5,
    mask_margin: float = 0.03,
    mask_config: Path = None,
    operator_radius: float = 0.0,
    photometric: str = "off",
    photometric_params: Path = None,
    third_camera: Optional[Any] = None,
    colmap_export_dir: Optional[Path] = None,
) -> int:
    """Run end-to-end unified workflow: Extract -> SLAM -> Sync -> SfM/Recalibrate -> Colorize -> Deliverables."""
    t_start = time.time()
    from raven_app.seed_export import validate_seed_percent
    seed_percent = validate_seed_percent(seed_percent)
    if operator_radius > 0 and not (mask_persons or masks_dir is not None):
        raise ValueError('Operator removal requires person masks; a trajectory radius alone can erase fixed objects')
    if method == "trajectory":
        method = "direct"
    elif method == "reconstruction":
        method = "sfm"
    if mask_persons and masks_dir is not None:
        raise ValueError("Choose either generated person masks or --masks-dir reuse, not both")
    if mask_persons:
        from raven_app.person_masks import read_mask_config
        read_mask_config(mask_config)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir = output_dir / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)

    third_cfgs = []
    if third_camera:
        from raven_app.third_camera import (
            ThirdCameraConfig,
            load_third_camera_configs,
        )
        if isinstance(third_camera, ThirdCameraConfig):
            third_cfgs = [third_camera]
        elif isinstance(third_camera, (list, tuple)):
            third_cfgs = [item if isinstance(item, ThirdCameraConfig)
                          else ThirdCameraConfig.from_dict(item) for item in third_camera]
        elif isinstance(third_camera, (str, Path)):
            p = Path(third_camera)
            if p.is_file() and p.suffix.lower() == ".json":
                third_cfgs = load_third_camera_configs(p)
            elif p.exists():
                third_cfgs = [ThirdCameraConfig(source_path=str(p), enabled=True)]
        elif isinstance(third_camera, dict):
            third_cfgs = load_third_camera_configs(third_camera)
    third_cfgs = [cfg for cfg in third_cfgs if cfg.enabled and cfg.source_path]
    if third_cfgs:
        from raven_app.third_camera import expand_auxiliary_camera_configs
        third_cfgs = expand_auxiliary_camera_configs(third_cfgs)

    run_record = {
        'run_id': str(time.time_ns()),
        'started_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'status': 'started',
        'app_version': __version__,
        **_workflow_settings_snapshot(
            bag_path, insv_path, output_dir, fps=fps, method=method,
            third_cameras=third_cfgs,
            lio=bool(lio), threads=int(threads), lidar_topic=lidar_topic,
            imu_topic=imu_topic, dt_override_seconds=dt_override,
            recalibrate=bool(recalibrate), run_spirula=bool(run_spirula),
            use_vulkan=bool(use_vulkan),
            deliverables={'ply': bool(export_ply), 'pcd': bool(export_pcd),
                          'colmap_3dgs': bool(export_colmap)},
            seed_percent=float(seed_percent), process_gps=bool(process_gps),
            gps_formats=list(gps_formats), geo_formats=list(geo_formats),
            masks={'generate_person_masks': bool(mask_persons),
                   'directory': _workflow_path_snapshot(masks_dir),
                   'model': _workflow_path_snapshot(mask_model),
                   'backend': mask_backend, 'threshold': float(mask_threshold),
                   'margin': float(mask_margin),
                   'config': _workflow_path_snapshot(mask_config)},
            operator_radius_m=float(operator_radius),
            photometric={'mode': photometric,
                         'parameters': _workflow_path_snapshot(photometric_params)},
            calibration_file=_workflow_path_snapshot(calib_json),
        ),
    }
    _persist_workflow_run(output_dir, run_record)
    print(f"[*] Workflow settings saved to {output_dir / 'workflow_config.json'}")

    print("=" * 80)
    print(" RAVENCALIBRATOR UNIFIED WORKFLOW STUDIO")
    print(f" Bag File:    {bag_path}")
    print(f" Video File:  {insv_path}")
    print(f" Output Dir:  {output_dir}")
    print(f" SLAM Mode:   {'LIO (LiDAR+IMU)' if lio else 'VIO (Visual+LiDAR+IMU)'}")
    print(f" Method:      {method.upper()}")
    print(f" Recalibrate: {'YES (Dynamic SfM-to-LiDAR)' if recalibrate else 'NO (Static)'}")
    print(f" Deliverables: PLY={export_ply}, PCD={export_pcd}, COLMAP/3DGS={export_colmap}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # STAGE 1: Extract frames from INSV
    # --------------------------------------------------------------------------
    print("\n[STAGE 1/7] Extracting video frames from INSV...")
    frames_manifest = output_dir / "images" / "frames.json"
    previous_frames = frames_manifest.stat().st_mtime_ns if frames_manifest.is_file() else None
    if not extract_insv_frames(insv_path, output_dir, fps=fps):
        raise RuntimeError("Failed to extract video frames from INSV")
    frames_changed = previous_frames is not None and frames_manifest.stat().st_mtime_ns != previous_frames
    is_timelapse = json.loads(frames_manifest.read_text(encoding='utf-8')).get('time_source') == 'insv_timelapse'

    for third_cfg in third_cfgs:
        print(f"\n[*] Extracting auxiliary frames from {third_cfg.camera_name}...")
        from raven_app.third_camera import extract_third_camera_frames
        try:
            extracted_aux, _ = extract_third_camera_frames(third_cfg, output_dir)
            print(f"[+] Successfully extracted {len(extracted_aux)} frames for {third_cfg.camera_name}")
        except Exception as e:
            raise RuntimeError(f"Could not extract auxiliary camera {third_cfg.camera_name}: {e}") from e

    run_record['status'] = 'frames_ready'
    run_record['image_extraction'] = _workflow_image_extraction_snapshot(output_dir, third_cfgs)
    _persist_workflow_run(output_dir, run_record)

    # Auxiliary orientation/source changes invalidate SfM just like INSV changes.
    frames_changed = previous_frames is not None and frames_manifest.stat().st_mtime_ns != previous_frames

    # --------------------------------------------------------------------------
    # STAGE 2: Time synchronization (IMU Cross-Correlation)
    # --------------------------------------------------------------------------
    print("\n[STAGE 2/7] Temporal Synchronization...")
    if dt_override is not None:
        dt_sync = dt_override
        print(f"[*] Using manual time offset Δt = {dt_sync:.4f}s")
    else:
        dt_sync = auto_sync_imu_gyro(bag_path, insv_path, imu_topic=imu_topic)
    run_record['resolved_dt_sync_seconds'] = float(dt_sync)
    run_record['synchronization_source'] = 'manual_override' if dt_override is not None else 'imu_gyro_correlation'
    _persist_workflow_run(output_dir, run_record)

    # --------------------------------------------------------------------------
    # STAGE 3: Run Native FAST-LIVO2 SLAM
    # --------------------------------------------------------------------------
    print("\n[STAGE 3/7] Running FAST-LIVO2 Native SLAM...")
    slam_out = output_dir / "slam_out"
    is_eagle = ('livox' in (lidar_topic or '').lower()) or ('eagle' in str(bag_path).lower()) or (lidar_topic == '/livox/lidar')
    timelapse_raven_profile = bool(is_timelapse and lio and not is_eagle)
    base_config = resources() / ('FAST-LIVO2/config/eagle.yaml' if is_eagle else 'FAST-LIVO2/config/raven.yaml')
    if not base_config.is_file():
        base_config = resources() / 'FAST-LIVO2/config/raven.yaml'
    native_engine = engine()
    slam_inputs = {'bag': str(Path(bag_path).resolve()), 'size': Path(bag_path).stat().st_size,
                   'mtime_ns': Path(bag_path).stat().st_mtime_ns, 'lio': bool(lio),
                   'lidar_topic': lidar_topic, 'imu_topic': imu_topic,
                   'timelapse_profile': timelapse_raven_profile, 'provenance_version': 1,
                   'config_sha256': hashlib.sha256(base_config.read_bytes()).hexdigest(),
                   'engine_sha256': hashlib.sha256(native_engine.read_bytes()).hexdigest()
                   if lio and native_engine.is_file() else 'not-used'}
    if lio:
        # A settings-keyed raw cache preserves old datasets and invalidates SLAM
        # when input, sensor topics, engine or estimator configuration changes.
        key = hashlib.sha256(json.dumps(slam_inputs, sort_keys=True).encode()).hexdigest()[:20]
        slam_out = output_dir / f'slam_raw_{key}'
    raw_pcd = slam_out / "pcd" / "all_raw_points.pcd"
    raw_trj = slam_out / "result" / "Raven_3DMakerPro_Scan.txt"
    if not raw_trj.is_file() and (slam_out / "result").is_dir():
        txts = sorted((slam_out / "result").glob("*.txt"))
        if txts:
            raw_trj = txts[0]

    profile_marker = slam_out / 'workflow_config.json'
    expected_profile = {
        'name': 'raven_timelapse_stable_v1',
        'filter_size_surf_m': 0.06,
        'voxel_size_m': 0.20,
    }
    profile_matches = False
    if timelapse_raven_profile and profile_marker.is_file():
        try:
            profile_matches = json.loads(profile_marker.read_text(encoding='utf-8')) == expected_profile
        except (OSError, ValueError):
            profile_matches = False

    if raw_pcd.is_file() and raw_trj.is_file() and (
            not timelapse_raven_profile or profile_matches):
        print(f"[*] Existing SLAM trajectory ({raw_trj.name}) and PCD found. Re-using output.")
    else:
        eng = engine()
        if not eng.is_file():
            raise FileNotFoundError(f"Native engine missing at {eng}; run tools/build_windows.ps1")

        if is_eagle and (resources() / "FAST-LIVO2/config/eagle.yaml").is_file():
            config_path = resources() / "FAST-LIVO2/config/eagle.yaml"
        else:
            config_path = resources() / "FAST-LIVO2/config/raven.yaml"
        temporary_config = None
        if timelapse_raven_profile or lio:
            config_text = config_path.read_text(encoding='utf-8')
            if timelapse_raven_profile:
                config_text, n_surf = re.subn(r'filter_size_surf:\s*[\d\.]+', 'filter_size_surf: 0.06', config_text, count=1)
                config_text, n_vox = re.subn(r'voxel_size:\s*[\d\.]+', 'voxel_size: 0.20', config_text, count=1)
                if n_surf != 1 or n_vox != 1:
                    raise RuntimeError('Cannot build the validated timelapse Raven profile: missing filter_size_surf or voxel_size')
            if lio:
                config_text, count = re.subn(r'(?m)^pcd_save:\s*$',
                                             'pcd_save:\n  scan_provenance_en: true', config_text, count=1)
                if count != 1:
                    raise RuntimeError('SLAM config lacks the required pcd_save section')
            temporary_config = output_dir / f'.raven_timelapse_{time.time_ns()}.yaml'
            temporary_config.write_text(config_text, encoding='utf-8')
            config_path = temporary_config
            profile_marker.unlink(missing_ok=True)
            if raw_pcd.is_file() and raw_trj.is_file():
                print('[*] Existing SLAM output has no matching timelapse profile; recomputing it.')
            if timelapse_raven_profile:
                print('[*] Using tested timelapse LIO settings: surface filter 0.06 m, map voxel 0.20 m.')
        camera_path = resources() / "FAST-LIVO2/config/camera_raven.yaml"
        cmd = [
            str(eng), "--input", "-", "--config", str(config_path),
            "--camera", str(camera_path), "--output", str(slam_out),
            "--threads", str(threads)
        ]
        if lio:
            cmd.append("--lio")

        slam_out.mkdir(parents=True, exist_ok=True)
        try:
            with subprocess.Popen(cmd, stdin=subprocess.PIPE, **hidden_window_options()) as child:
                export_bags(
                    [bag_path], child.stdin,
                    lidar_topic=lidar_topic,
                    imu_topic=imu_topic,
                    lio=lio
                )
                child.stdin.close()
                code = child.wait()
                if code != 0:
                    raise RuntimeError(f"FAST-LIVO2 SLAM execution failed with exit code {code}")
            if timelapse_raven_profile and raw_pcd.is_file() and raw_trj.is_file():
                profile_marker.write_text(
                    json.dumps(expected_profile, indent=2) + '\n', encoding='utf-8')
            (slam_out/'input_config.json').write_text(json.dumps(slam_inputs, indent=2), encoding='utf-8')
        finally:
            if temporary_config is not None:
                temporary_config.unlink(missing_ok=True)

        if not raw_trj.is_file() and (slam_out / "result").is_dir():
            txts = sorted((slam_out / "result").glob("*.txt"))
            if txts:
                raw_trj = txts[0]

    if lio:
        from raven_app.slam_refinement import refine, VERSION
        if not (slam_out / 'scan_ranges.csv').is_file():
            raise RuntimeError('SLAM output lacks exact scan provenance; recompute the raw SLAM in a fresh output directory')
        refinement_dir = output_dir / (f'slam_refined_{VERSION}_'
                                      f'{raw_pcd.stat().st_mtime_ns}_{raw_trj.stat().st_mtime_ns}')
        refinement = refine(slam_out, refinement_dir, log=lambda message: print(f'[*] {message}', flush=True))
        run_record['slam_refinement'] = refinement
        _persist_workflow_run(output_dir, run_record)
        if refinement.get('status') == 'accepted':
            slam_out = refinement_dir
            raw_pcd = Path(refinement['output_pcd'])
            raw_trj = Path(refinement['output_trajectory'])
            print('[+] Global SLAM correction accepted by independent geometry validation.')
        elif refinement.get('closure_improved') and Path(refinement.get('output_pcd', '')).is_file():
            slam_out = refinement_dir
            raw_pcd = Path(refinement['output_pcd'])
            raw_trj = Path(refinement['output_trajectory'])
            print('[+] Global SLAM correction accepted based on pose graph closure improvement.')
        elif refinement.get('status') == 'no_supported_revisits':
            print('[*] No validated revisit correction; retaining raw SLAM geometry.')
        else:
            print('[!] Warning: Global SLAM refinement candidate rejected by geometry check; retaining raw SLAM geometry.')
    else:
        run_record['slam_refinement'] = {'status': 'unsupported_visual_mode',
                                          'reason': 'Exact world-frame scan provenance requires LiDAR + IMU mode'}
        print('[*] Global revisit correction requires LiDAR + IMU mode.')
    active_manifest = output_dir / 'active_slam.json'
    active_data = {'relative_directory': str(slam_out.relative_to(output_dir)),
                   'refinement_status': run_record['slam_refinement']['status']}
    active_temp = output_dir / '.active_slam.tmp'
    active_temp.write_text(json.dumps(active_data, indent=2), encoding='utf-8')
    active_temp.replace(active_manifest)

    # --------------------------------------------------------------------------
    # STAGE 4: SfM Alignment & Dynamic Spatial Extrinsics Recalibration
    # --------------------------------------------------------------------------
    sparse_bin = output_dir / "sparse" / "0" / "points3D.bin"
    stage4_sfm = (method in ("sfm", "all") or run_spirula or
                  (frames_changed and sparse_bin.is_file()))
    direct_sample_calibration_required = (
        method == 'direct' and (bool(third_cfgs) or (recalibrate and not sparse_bin.is_file())))
    sample_sfm_required = direct_sample_calibration_required
    stage4_title = ("Multi-View SfM Alignment & Spatial Recalibration"
                    if stage4_sfm else
                    ("Direct trajectory + sample SfM rig calibration (full SfM skipped)"
                     if sample_sfm_required else
                     "Direct Trajectory Preparation (full SfM skipped)"))
    print(f"\n[STAGE 4/7] {stage4_title}...")
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    if third_cfgs:
        from raven_app.third_camera import (
            align_third_camera_to_rig,
            calibrate_third_camera_intrinsics,
            save_third_camera_config,
        )
    calib = calib_json or (output_dir / "rig_calibration.json")
    if not calib.is_file():
        calib = output_dir / "calibracao_rigida_auto.json"
    if not calib.is_file():
        if is_eagle and (resources() / "configs/rig_profile_eagle.json").is_file():
            calib = resources() / "configs/rig_profile_eagle.json"
        elif (resources() / "configs/rig_profile.json").is_file():
            calib = resources() / "configs/rig_profile.json"
        else:
            calib = resources() / "calibracao_rigida_raven_insta360.json"

    if (method in ("sfm", "all") or run_spirula or (frames_changed and sparse_bin.is_file())) and (not sparse_bin.is_file() or frames_changed):
        print("[*] Running Spirula SfM (Vulkan GPU Headless)...")
        try:
            rebuilt = pipeline.run_spirula_sfm_auto(
                output_dir, auxiliary_cameras=third_cfgs, force_rebuild=frames_changed)
            if frames_changed and not rebuilt:
                raise RuntimeError("SfM rebuild failed after video frames changed")
        except Exception as e:
            if frames_changed:
                raise
            print(f"[!] Spirula SfM run notice: {e}")

    sparse_bin = output_dir / "sparse" / "0" / "points3D.bin"
    if sparse_bin.is_file() and third_cfgs:
        print("[*] Calibrating auxiliary camera intrinsics from the shared SfM model...")
        intrinsic_cameras = []
        for camera in third_cfgs:
            try:
                camera = calibrate_third_camera_intrinsics(output_dir, camera)
                print(f"[+] {camera.camera_name}: intrinsics fitted from SfM")
            except Exception as e:
                print(f"[!] {camera.camera_name} intrinsics remain uncalibrated: {e}")
            intrinsic_cameras.append(camera)
        third_cfgs = intrinsic_cameras
        save_third_camera_config(
            third_cfgs, output_dir / "third_camera_calibrated.json")

    if sparse_bin.is_file() and not (method == 'direct' and third_cfgs):
        print("[*] Performing Metric Sim(3) Alignment (Spirula COLMAP -> LiDAR Ground Truth)...")
        try:
            align_data = pipeline.align_colmap_to_lidar(output_dir, fps=fps, dt_hint=dt_sync)
            dt_sync = align_data.get("dt_sync_seconds", dt_sync)
            rmse_cm = align_data.get('trajectory_rmse_cm')
            if rmse_cm is not None and rmse_cm > 2.0:
                print(f"[QUALITY WARNING] SfM-to-FAST-LIVO2 trajectory RMSE is {rmse_cm:.2f} cm; "
                      "this is above the requested 1.5–2 cm wall-thickness precision.")
                print("[QUALITY WARNING] SfM alignment calibrates camera poses; geometric correction is independently validated in the SLAM stage.")
        except Exception as e:
            if is_timelapse:
                raise RuntimeError(f'Timelapse alignment failed: {e}') from e
            print(f"[!] Metric alignment notice: {e}")

        if recalibrate:
            print("[*] Auto-Recalibrating spatial mounting extrinsics (T_LC0, T_LC1) from SfM...")
            try:
                pipeline.recalibrate_from_sfm(output_dir, fps=fps)
                calib = output_dir / "rig_calibration.json"
                if not calib.is_file():
                    calib = output_dir / "calibracao_rigida_auto.json"
            except Exception as e:
                if is_timelapse:
                    raise RuntimeError(f'Timelapse rig calibration failed: {e}') from e
                print(f"[!] Extrinsics recalibration notice: {e}")

    if method == 'direct' and (recalibrate or third_cfgs) and (
            not sparse_bin.is_file() or bool(third_cfgs)):
        print("[*] Per-session rig calibration requires a bounded SfM sample; "
              "the full SfM reconstruction remains skipped.")
        try:
            third_cfgs = _calibrate_aux_cameras_from_sample_sfm(
                output_dir, third_cfgs, fps, dt_sync)
            calib = output_dir / 'rig_calibration.json'
            dt_sync = json.loads(calib.read_text(encoding='utf-8'))['dt_sync_seconds']
        except Exception as e:
            run_record.update(
                status='failed',
                failure={'stage': 'sample_sfm_rig_calibration', 'reason': str(e)},
            )
            _persist_workflow_run(output_dir, run_record)
            for camera in third_cfgs:
                report = dict(camera.calibration_report or {})
                report['camera_name'] = camera.camera_name
                report['sample_sfm'] = {
                    **(report.get('sample_sfm', {}) or {}),
                    'status': 'failed', 'reason': str(e),
                }
                camera.calibration_report = report
            if third_cfgs:
                from raven_app.third_camera import save_third_camera_config
                save_third_camera_config(third_cfgs, output_dir / 'third_camera_calibrated.json')
            raise RuntimeError(
                f'Required per-session sample SfM camera calibration failed: {e}') from e

    calibrated_cameras = []
    updated_third_cfgs = []
    full_sfm_pose_support_records = {}
    for third_cfg in third_cfgs:
        report = third_cfg.calibration_report or {}
        sample_status = report.get('sample_sfm', {}).get('status')
        if sample_status in {'complete', 'rejected', 'failed', 'skipped',
                             'per_image_sfm_pose_support'}:
            pose_support_ready = _aux_camera_pose_support_ready(
                output_dir, third_cfg, require_sfm_source_signature=stage4_sfm)
            rigid_ready = _aux_camera_calibration_ready(third_cfg)
            if rigid_ready or (pose_support_ready and not stage4_sfm):
                calibrated_cameras.append(third_cfg)
                updated_third_cfgs.append(third_cfg)
                continue
            if not stage4_sfm:
                reason = (report.get('sample_sfm', {}).get('reason') or
                          report.get('extrinsics', {}).get('reason') or
                          'sample-SfM calibration did not pass quality gates')
                print(f"[!] Skipping {third_cfg.camera_name} in point-cloud colorization: {reason}")
                updated_third_cfgs.append(third_cfg)
                continue
            action = ('refreshing' if pose_support_ready else 'replacing stale')
            print(f"[*] {third_cfg.camera_name} per-image SfM poses are {action} from the "
                  "current aligned model.")
        print(f"\n[*] Calibrating {third_cfg.camera_name} intrinsics and rig extrinsics...")
        try:
            third_cfg = align_third_camera_to_rig(
                output_dir, third_cfg, rig_profile_path=calib, fps=fps)
        except Exception as e:
            print(f"[!] {third_cfg.camera_name} calibration notice: {e}")
            report = dict(third_cfg.calibration_report or {})
            report['camera_name'] = third_cfg.camera_name
            report['calibration_attempt'] = {'status': 'failed', 'reason': str(e)}
            if report.get('extrinsics', {}).get('status') != 'calibrated':
                report['extrinsics'] = {
                    'status': 'uncalibrated',
                    'source': 'calibration attempt',
                    'camera_name': third_cfg.camera_name,
                    'reason': str(e),
                }
            third_cfg.calibration_report = report
        updated_third_cfgs.append(third_cfg)
        if _aux_camera_calibration_ready(third_cfg):
            calibrated_cameras.append(third_cfg)
        elif stage4_sfm:
            support_record, support_reason = _full_sfm_aux_pose_support_record(
                output_dir, third_cfg)
            if support_record is not None:
                full_sfm_pose_support_records[third_cfg.camera_name] = support_record
                print(f"[+] {third_cfg.camera_name}: {len(support_record['poses'])} exact registered "
                      "SfM poses are valid for this accepted LiDAR alignment; "
                      "rigid offset/baseline remain uncalibrated.")
            else:
                print(f"[!] {third_cfg.camera_name} per-image SfM poses not accepted: {support_reason}")
    third_cfgs = updated_third_cfgs
    if full_sfm_pose_support_records:
        try:
            from raven_app.auxiliary_pose_support import write_auxiliary_pose_support

            source_signatures = {
                json.dumps(record['sfm_pose_source_signature'], sort_keys=True)
                for record in full_sfm_pose_support_records.values()
            }
            if len(source_signatures) != 1:
                raise ValueError('Auxiliary cameras were aligned against different COLMAP models')
            sfm_source_signature = next(iter(full_sfm_pose_support_records.values()))[
                'sfm_pose_source_signature']
            support_path = write_auxiliary_pose_support(
                output_dir, full_sfm_pose_support_records,
                sfm_source_signature=sfm_source_signature)
            for camera in third_cfgs:
                record = full_sfm_pose_support_records.get(camera.camera_name)
                if record is None:
                    continue
                report = dict(camera.calibration_report or {})
                reason = ((report.get('extrinsics') or {}).get('reason')
                          or 'a rigid per-camera clock offset and baseline were not established')
                report['sample_sfm'] = {
                    **(report.get('sample_sfm') or {}),
                    'status': 'per_image_sfm_pose_support',
                    'method': 'registered_full_sfm_poses_aligned_to_lidar',
                    'reason': reason,
                    'registered_views': len(record['poses']),
                    'alignment_quality_status': 'accepted',
                    'alignment_rmse_cm': record.get('alignment_rmse_cm'),
                    'pose_support_path': str(support_path.relative_to(output_dir)),
                }
                camera.calibration_report = report
            print(f"[+] Accepted exact registered auxiliary poses from the current full SfM model: "
                  f"{support_path}")
        except (OSError, ValueError, TypeError) as exc:
            print(f"[!] Could not publish full-SfM auxiliary pose support: {exc}")
    if third_cfgs:
        save_third_camera_config(third_cfgs, output_dir / "third_camera_calibrated.json")
        support_required_for_run = stage4_sfm
        calibrated_cameras = [
            camera for camera in third_cfgs
            if (_aux_camera_calibration_ready(camera)
                or _aux_camera_pose_support_ready(
                    output_dir, camera,
                    require_sfm_source_signature=support_required_for_run))
        ]
        failed_cameras = [camera for camera in third_cfgs
                          if camera not in calibrated_cameras]
        if failed_cameras:
            reasons = []
            for camera in failed_cameras:
                report = camera.calibration_report or {}
                reason = ((report.get('sample_sfm', {}) or {}).get('reason')
                          or (report.get('extrinsics', {}) or {}).get('reason')
                          or (report.get('time_offset', {}) or {}).get('reason')
                          or 'intrinsics, time offset, or metric extrinsics were not calibrated')
                reasons.append(f'{camera.camera_name}: {reason}')
            message = ('Enabled auxiliary cameras must all pass calibration or have fresh '
                       'registered SfM poses before colorization: ' + '; '.join(reasons))
            run_record.update(status='failed', failure={
                'stage': 'auxiliary_camera_calibration', 'reason': message})
            _persist_workflow_run(output_dir, run_record)
            raise RuntimeError(message)

    # --------------------------------------------------------------------------
    # STAGE 5: Person-mask generation
    # --------------------------------------------------------------------------
    mask_stage = ('Preparing person masks...' if mask_persons or masks_dir is not None
                  else 'Skipping person-mask generation...')
    print(f"\n[STAGE 5/7] {mask_stage}")
    active_masks_dir = masks_dir
    if mask_persons:
        print("[*] Generating RF-DETR person masks before colorization...")
        from raven_app.person_masks import generate_person_masks
        active_masks_dir = generate_person_masks(
            output_dir, model_path=mask_model,
            threshold=mask_threshold, margin=mask_margin,
            mask_config=mask_config, backend=mask_backend
        )
    elif active_masks_dir is not None:
        active_masks_dir = Path(active_masks_dir).resolve()
        if not active_masks_dir.is_dir():
            raise FileNotFoundError(f"Person mask directory does not exist: {active_masks_dir}")
        print(f"[*] Reusing person masks from {active_masks_dir}")
    if active_masks_dir is not None:
        print(f"[*] Person masks: {active_masks_dir}")
    if operator_radius > 0:
        print(f"[*] Operator removal radius: {operator_radius:.3f} m (person-mask agreement with planar-surface protection)")

    # --------------------------------------------------------------------------
    # STAGE 6: Point Cloud Colorization
    # --------------------------------------------------------------------------
    print("\n[STAGE 6/7] Running Point Cloud Colorization...")
    cloud_state_before = {p: p.stat().st_mtime_ns for p in deliverables_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in {'.las', '.laz', '.ply', '.pcd'}}
    sfm_done = False
    if method in ("sfm", "all") and sparse_bin.is_file():
        try:
            pipeline.colorize_via_spirula_sfm(
                output_dir, fps=fps, use_vulkan=use_vulkan,
                masks_dir=active_masks_dir, operator_radius=operator_radius,
                photometric=photometric, photometric_params=photometric_params
            )
            sfm_done = True
        except Exception as e:
            print(f"[!] SfM colorization notice: {e}")

    if method in ("direct", "all") or (method == "sfm" and not sfm_done):
        pipeline.colorize_via_direct_rigid(
            output_dir, calib, fps=fps, dt_override=dt_sync, use_vulkan=use_vulkan,
            masks_dir=active_masks_dir, operator_radius=operator_radius,
            photometric=photometric, photometric_params=photometric_params,
            third_camera=calibrated_cameras
        )

    # --------------------------------------------------------------------------
    # STAGE 7: Georeferencing and packaging deliverables
    # --------------------------------------------------------------------------
    print("\n[STAGE 7/7] Georeferencing and packaging deliverables...")
    if process_gps:
        print("\n[GPS] Extracting native GPS metadata and selected formats...")
        gps_report = process_gps_metadata(insv_path, deliverables_dir / 'gps', gps_formats)
        print(f"[GPS] {gps_report['unique_fixes']} unique fixes; {gps_report['georeferencing_status']}")
        print(f"[GPS] Metadata saved in {deliverables_dir / 'gps'}")

    # Automatic placement is deliberately after colorization so only this run's
    # produced clouds are considered, and before local-format cleanup.
    if process_gps:
        try:
            from raven_app.automatic_georeference import automatic_georeference
            produced = [p for p in deliverables_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in {'.las', '.laz', '.ply', '.pcd'}
                        and (p not in cloud_state_before or p.stat().st_mtime_ns != cloud_state_before[p])]
            gps_info = None
            try:
                from raven_app.georeference import calculate_insv_gps
                gps_info = calculate_insv_gps(insv_path)
            except ValueError:
                gps_info = {'records': []}
            automatic_georeference(insv_path, deliverables_dir, candidate_clouds=produced,
                                   formats=geo_formats, trajectory_path=raw_trj, gps_info=gps_info)
        except ImportError as e:
            print(f"[!] Automatic georeferencing skipped: dependency missing ({e}). Install with 'pip install -r requirements.txt'.")
        except Exception as e:
            print(f"[!] Automatic georeferencing notice: {e}")

    # --------------------------------------------------------------------------
    # Packaging Deliverables
    # --------------------------------------------------------------------------
    print("\n[*] Packaging Deliverables...")
    if not export_ply:
        for f in deliverables_dir.glob("*.ply"):
            try: f.unlink()
            except Exception: pass
    if not export_pcd:
        for f in deliverables_dir.glob("*.pcd"):
            try: f.unlink()
            except Exception: pass

    if export_colmap:
        export_colmap_3dgs(output_dir, calib, fps=fps, dt_sync=dt_sync,
                           masks_dir=active_masks_dir, third_camera=third_cfgs,
                           seed_percent=seed_percent,
                           pose_source='direct' if method == 'direct' else 'sfm',
                           export_path=colmap_export_dir)

    print("\n[+] Deliverables currently in folder:")
    for deliv_item in sorted(deliverables_dir.iterdir()):
        if deliv_item.is_file():
            print(f"    - {deliv_item.name} ({deliv_item.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    run_record.update(
        status='complete',
        completed_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        elapsed_seconds=round(elapsed, 3),
        resolved_dt_sync_seconds=float(dt_sync),
        calibration_used=_workflow_path_snapshot(calib),
        auxiliary_calibration=[camera.to_dict() for camera in third_cfgs],
    )
    _persist_workflow_run(output_dir, run_record)
    print("\n" + "=" * 80)
    print(f" [SUCCESS] Unified Workflow Complete in {elapsed:.1f}s!")
    print(f" Deliverables saved in: {deliverables_dir}")
    print("=" * 80)
    return 0
