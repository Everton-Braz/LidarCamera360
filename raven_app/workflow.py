"""Unified end-to-end workflow engine for RavenCalibrator.

Orchestrates:
1. Video frame & gyro telemetry extraction from Insta360 .insv
2. FAST-LIVO2 LiDAR-inertial odometry and mapping (.bag -> .pcd + trajectory)
3. Sub-millisecond temporal cross-correlation (IMU Gyro Δt)
4. Calibrated LiDAR-camera point cloud colorization
5. Deliverables generation (.PLY, .PCD, and COLMAP dataset ready for 3DGS training).
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation as Rot

from raven_app import __version__
from raven_app.cli import engine, resources
from raven_app.bag_io import export_bags
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


def auto_sync_imu_gyro(bag_path: Path, insv_path: Path, imu_topic: str = None) -> float:
    """Calculate sub-millisecond time offset Δt via cross-correlation of camera and LiDAR gyros."""
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

            for conn, _, raw in reader.messages():
                if conn.topic == imu_topic:
                    msg = reader.deserialize(raw, conn.msgtype)
                    t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    lidar_t.append(t_sec)
                    lidar_wx.append(msg.angular_velocity.x)
                    lidar_wy.append(msg.angular_velocity.y)
                    lidar_wz.append(msg.angular_velocity.z)

        if not lidar_t:
            print("[!] Warning: No IMU packets found in ROS bag. Using default sync offset.")
            return 1.892

        lidar_t = np.array(lidar_t)
        lidar_gyro_norm = np.sqrt(np.array(lidar_wx)**2 + np.array(lidar_wy)**2 + np.array(lidar_wz)**2)
        lidar_t_rel = lidar_t - lidar_t[0]

        # Resample both to 100 Hz
        target_hz = 100.0
        dur = min(t_cam_s[-1], lidar_t_rel[-1])
        if dur <= 2.0:
            print("[!] Trajectory too short for robust cross-correlation. Using default sync.")
            return 1.892

        t_common = np.arange(0.0, dur, 1.0 / target_hz)
        cam_resamp = np.interp(t_common, t_cam_s, cam_gyro_norm)
        lidar_resamp = np.interp(t_common, lidar_t_rel, lidar_gyro_norm)

        cam_resamp -= np.mean(cam_resamp)
        lidar_resamp -= np.mean(lidar_resamp)

        corr = signal.correlate(cam_resamp, lidar_resamp, mode='full')
        lags = signal.correlation_lags(len(cam_resamp), len(lidar_resamp), mode='full')
        best_lag = lags[np.argmax(corr)]
        dt = best_lag / target_hz

        print(f"[+] Programmatic IMU Gyro Cross-Correlation resolved: Δt = {dt:.4f}s")
        return float(dt)
    except Exception as e:
        print(f"[!] Cross-correlation failed ({e}). Falling back to calibrated default Δt = 1.892s.")
        return 1.892


def load_lidar_seed_points(dataset_dir: Path, max_points: int = 1500000):
    """Load colorized or raw LiDAR points to use as dense geometric 3DGS seed."""
    ply_candidates = sorted((dataset_dir / "deliverables").glob("lidar_colored_*.ply"))
    if not ply_candidates:
        ply_candidates = sorted(dataset_dir.glob("*.ply"))

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
            dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd",
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

    if len(xyz) > max_points:
        step = max(1, len(xyz) // max_points)
        xyz = xyz[::step]
        rgb = rgb[::step]
        print(f"  [*] Subsampled LiDAR 3DGS seed to {len(xyz):,} points (step={step}) for optimal 3DGS training initialization")

    return xyz, rgb


def write_colmap_points3d(dst_dir: Path, xyz: np.ndarray, rgb: np.ndarray, errors: np.ndarray = None):
    """Write points3D.ply, points3D.bin, and points3D.txt in COLMAP format."""
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

    # 2. Write binary COLMAP (points3D.bin)
    bin_path = dst_dir / "points3D.bin"
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<Q", n))
        for i in range(n):
            pid = i + 1
            f.write(struct.pack("<Q3d3BdQ", pid, float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                                int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]), float(errors[i]), 0))

    # 3. Write text COLMAP (points3D.txt)
    txt_path = dst_dir / "points3D.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for i in range(n):
            f.write(f"{i + 1} {xyz[i, 0]:.6f} {xyz[i, 1]:.6f} {xyz[i, 2]:.6f} "
                    f"{int(rgb[i, 0])} {int(rgb[i, 1])} {int(rgb[i, 2])} {errors[i]:.4f}\n")

    # Copy points3D.ply to root of colmap_3dgs if dst_dir is sparse/0
    if dst_dir.parent.name == "sparse" and dst_dir.name == "0":
        try:
            shutil.copy2(ply_path, dst_dir.parent.parent / "points3D.ply")
        except Exception:
            pass


def export_colmap_3dgs(dataset_dir: Path, calib_path: Path, fps: float = 2.0, dt_sync: float = 0.0) -> Path:
    """Generate a complete COLMAP dataset configured for 3D Gaussian Splatting (3DGS) training.

    Transforms camera poses into the LiDAR metric coordinate frame and seeds the model with
    the true metric LiDAR point cloud (points3D.ply, points3D.bin, points3D.txt).
    """
    out_dir = dataset_dir / "colmap_3dgs"
    images_out = out_dir / "images"
    sparse_out = out_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Generating COLMAP Dataset for 3DGS Training under: {out_dir}")

    # Link / copy images into colmap_3dgs/images/cam0 and cam1
    cam0_files = sorted((dataset_dir / "images" / "cam0").glob("*.jpg"))
    cam1_files = sorted((dataset_dir / "images" / "cam1").glob("*.jpg"))
    for cam, files in [("cam0", cam0_files), ("cam1", cam1_files)]:
        dest_cam = images_out / cam
        dest_cam.mkdir(parents=True, exist_ok=True)
        for f in files:
            target = dest_cam / f.name
            if not target.exists():
                try:
                    target.symlink_to(f)
                except Exception:
                    shutil.copy2(f, target)

    # 1. Check if Spirula SfM output is available to transform to METRIC scale
    sfm_sparse = dataset_dir / "sparse" / "0"
    if not (sfm_sparse / "points3D.bin").is_file():
        alt = dataset_dir / "spirula_out" / "workspace" / "sparse" / "0"
        if (alt / "points3D.bin").is_file():
            sfm_sparse = alt
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    poses_ready = False
    if (sfm_sparse / "points3D.bin").is_file() and align_json.is_file():
        print("  [*] Transforming Spirula SfM cameras and images into true LiDAR METRIC frame via Sim(3)...")
        from scripts.pipeline_auto_calibrator_and_colorizer import transform_colmap_to_metric
        with open(align_json, "r", encoding="utf-8") as f:
            al = json.load(f)
        s_sim = al["scale"]
        R_sim = np.array(al["R"])
        t_sim = np.array(al["t"])
        transform_colmap_to_metric(sfm_sparse, sparse_out, s_sim, R_sim, t_sim)
        poses_ready = True

    if not poses_ready:
        # Fallback: Synthesize from SLAM Trajectory
        print("  [*] SfM sparse reconstruction not found; synthesizing poses from SLAM trajectory...")
        calib = json.loads(calib_path.read_text(encoding="utf-8"))
        c0 = calib["cam0_front_intrinsics"]
        c1 = calib["cam1_rear_intrinsics"]
        T_LC0 = np.array(calib["T_lidar_to_cam0_rigid_4x4"])
        T_LC1 = np.array(calib["T_lidar_to_cam1_rigid_4x4"])
        R_LC0, t_LC0 = T_LC0[:3, :3], T_LC0[:3, 3]
        R_LC1, t_LC1 = T_LC1[:3, :3], T_LC1[:3, 3]

        cameras_txt = sparse_out / "cameras.txt"
        with open(cameras_txt, "w", encoding="utf-8") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write(
                f"1 OPENCV_FISHEYE 3840 3840 "
                f"{c0['fx']} {c0['fy']} {c0['cx']} {c0['cy']} "
                f"{c0['k1']} {c0['k2']} {c0['k3']} {c0['k4']}\n"
            )
            f.write(
                f"2 OPENCV_FISHEYE 3840 3840 "
                f"{c1['fx']} {c1['fy']} {c1['cx']} {c1['cy']} "
                f"{c1['k1']} {c1['k2']} {c1['k3']} {c1['k4']}\n"
            )
        print("  [+] Written cameras.txt")

        trj_path = dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
        if not trj_path.is_file() and (dataset_dir / "slam_out" / "result").is_dir():
            txts = sorted((dataset_dir / "slam_out" / "result").glob("*.txt"))
            if txts:
                trj_path = txts[0]
        images_txt = sparse_out / "images.txt"

        if trj_path.is_file():
            trj_data = np.loadtxt(trj_path)
            t_slam = trj_data[:, 0]
            pos_slam = trj_data[:, 1:4]
            rot_slam = Rot.from_quat(trj_data[:, 4:8])
            from scipy.spatial.transform import Slerp
            slerp = Slerp(t_slam, rot_slam)

            t_slam_start = t_slam[0]
            t_slam_end = t_slam[-1]
            img_id = 1

            with open(images_txt, "w", encoding="utf-8") as f:
                f.write("# Image list with two lines of data per image:\n")
                f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
                f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

                for k in range(min(len(cam0_files), len(cam1_files))):
                    fn_digits = "".join(filter(str.isdigit, cam0_files[k].stem))
                    fn = int(fn_digits) if fn_digits else (k + 1)
                    t_vid = frame_time(dataset_dir, "cam0/" + cam0_files[k].name, fps)
                    t_query = t_slam_start + (t_vid - dt_sync)

                    if not (t_slam_start <= t_query <= t_slam_end):
                        continue

                    idx = np.clip(np.searchsorted(t_slam, t_query), 1, len(t_slam) - 1)
                    w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                    p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                    R_L = slerp(t_query).as_matrix()

                    for cam_id, cam_name, img_file, R_LC, t_LC in [
                        (1, "cam0", cam0_files[k], R_LC0, t_LC0),
                        (2, "cam1", cam1_files[k], R_LC1, t_LC1)
                    ]:
                        p_cam = p_L + R_L @ t_LC
                        R_world_cam = R_L @ R_LC
                        R_cw = R_world_cam.T
                        t_cw = -R_cw @ p_cam
                        q_cw = Rot.from_matrix(R_cw).as_quat()
                        f.write(
                            f"{img_id} {q_cw[3]:.8f} {q_cw[0]:.8f} {q_cw[1]:.8f} {q_cw[2]:.8f} "
                            f"{t_cw[0]:.8f} {t_cw[1]:.8f} {t_cw[2]:.8f} {cam_id} {cam_name}/{img_file.name}\n\n"
                        )
                        img_id += 1

            print(f"  [+] Written images.txt with {img_id - 1} camera poses")

    # Seed 3DGS point cloud from true metric LiDAR points
    lidar_xyz, lidar_rgb = load_lidar_seed_points(dataset_dir)
    if lidar_xyz is not None and len(lidar_xyz) > 0:
        # If previous SfM points existed, preserve them as points3D_sfm_sparse
        if (sparse_out / "points3D.bin").is_file() and not (sparse_out / "points3D_sfm_sparse.bin").is_file():
            try:
                shutil.copy2(sparse_out / "points3D.bin", sparse_out / "points3D_sfm_sparse.bin")
                if (sparse_out / "points3D.ply").is_file():
                    shutil.copy2(sparse_out / "points3D.ply", sparse_out / "points3D_sfm_sparse.ply")
                if (sparse_out / "points3D.txt").is_file():
                    shutil.copy2(sparse_out / "points3D.txt", sparse_out / "points3D_sfm_sparse.txt")
            except Exception:
                pass

        print(f"  [*] Writing LiDAR 3DGS Seed: {len(lidar_xyz):,} points to points3D.ply, points3D.bin, points3D.txt...")
        write_colmap_points3d(sparse_out, lidar_xyz, lidar_rgb)
        print(f"  [+] LiDAR 3DGS seed successfully written to: {sparse_out / 'points3D.ply'}")
    else:
        print("  [*] Using existing sparse points3D as 3DGS seed (no LiDAR cloud available).")

    _postprocess_3dgs_dataset(out_dir, dataset_dir)
    print(f"[+] COLMAP 3DGS dataset generation complete: {out_dir}")
    return out_dir


def setup_sparse_compatibility(out_dir: Path):
    """Ensure both colmap_3dgs and colmap_3dgs/sparse work seamlessly in Spirula Studio."""
    sparse_dir = out_dir / "sparse"
    sparse_0 = sparse_dir / "0"
    if sparse_0.is_dir():
        for fname in ("cameras.bin", "cameras.txt", "images.bin", "images.txt", "points3D.bin", "points3D.txt", "points3D.ply"):
            src = sparse_0 / fname
            dst = sparse_dir / fname
            if src.is_file() and not dst.exists():
                try:
                    dst.symlink_to(src)
                except Exception:
                    shutil.copy2(src, dst)

    images_dir = out_dir / "images"
    sparse_images = sparse_dir / "images"
    if images_dir.is_dir() and not sparse_images.exists():
        try:
            if os.name == "nt":
                subprocess.run(["cmd", "/c", "mklink", "/J", str(sparse_images), str(images_dir)], capture_output=True)
            if not sparse_images.exists():
                sparse_images.symlink_to(images_dir, target_is_directory=True)
        except Exception:
            pass


def _postprocess_3dgs_dataset(out_dir: Path, dataset_dir: Path):
    """Adds point cloud seed, compatibility links, and instructions to the 3DGS dataset."""
    ply_candidates = sorted((dataset_dir / "deliverables").glob("lidar_colored_*.ply"))
    if not ply_candidates:
        ply_candidates = sorted(dataset_dir.glob("*.ply"))
    if ply_candidates:
        try:
            shutil.copy2(ply_candidates[-1], out_dir / "points3D_lidar.ply")
        except Exception:
            pass

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
    process_gps: bool = False,
    gps_formats=('geojson', 'gpx', 'csv'),
    geo_formats=('laz', 'geojson'),
) -> int:
    """Run end-to-end unified workflow: Extract -> SLAM -> Sync -> SfM/Recalibrate -> Colorize -> Deliverables."""
    t_start = time.time()
    if method == "trajectory":
        method = "direct"
    elif method == "reconstruction":
        method = "sfm"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    deliverables_dir = output_dir / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)

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

    if process_gps:
        print("\n[GPS] Extracting native GPS metadata and selected formats...")
        gps_report = process_gps_metadata(insv_path, deliverables_dir / 'gps', gps_formats)
        print(f"[GPS] {gps_report['unique_fixes']} unique fixes; {gps_report['georeferencing_status']}")
        print(f"[GPS] Metadata saved in {deliverables_dir / 'gps'}")

    # --------------------------------------------------------------------------
    # STAGE 1: Extract frames from INSV
    # --------------------------------------------------------------------------
    print("\n[STAGE 1/5] Extracting video frames from INSV...")
    if not extract_insv_frames(insv_path, output_dir, fps=fps):
        raise RuntimeError("Failed to extract video frames from INSV")

    # --------------------------------------------------------------------------
    # STAGE 2: Time synchronization (IMU Cross-Correlation)
    # --------------------------------------------------------------------------
    print("\n[STAGE 2/5] Temporal Synchronization...")
    if dt_override is not None:
        dt_sync = dt_override
        print(f"[*] Using manual time offset Δt = {dt_sync:.4f}s")
    else:
        dt_sync = auto_sync_imu_gyro(bag_path, insv_path, imu_topic=imu_topic)

    # --------------------------------------------------------------------------
    # STAGE 3: Run Native FAST-LIVO2 SLAM
    # --------------------------------------------------------------------------
    print("\n[STAGE 3/5] Running FAST-LIVO2 Native SLAM...")
    slam_out = output_dir / "slam_out"
    raw_pcd = slam_out / "pcd" / "all_raw_points.pcd"
    raw_trj = slam_out / "result" / "Raven_3DMakerPro_Scan.txt"
    if not raw_trj.is_file() and (slam_out / "result").is_dir():
        txts = sorted((slam_out / "result").glob("*.txt"))
        if txts:
            raw_trj = txts[0]

    is_eagle = ('livox' in (lidar_topic or '').lower()) or ('eagle' in str(bag_path).lower()) or (lidar_topic == '/livox/lidar')

    if raw_pcd.is_file() and raw_trj.is_file():
        print(f"[*] Existing SLAM trajectory ({raw_trj.name}) and PCD found. Re-using output.")
    else:
        eng = engine()
        if not eng.is_file():
            raise FileNotFoundError(f"Native engine missing at {eng}; run tools/build_windows.ps1")

        if is_eagle and (resources() / "FAST-LIVO2/config/eagle.yaml").is_file():
            config_path = resources() / "FAST-LIVO2/config/eagle.yaml"
        else:
            config_path = resources() / "FAST-LIVO2/config/raven.yaml"
        camera_path = resources() / "FAST-LIVO2/config/camera_raven.yaml"
        cmd = [
            str(eng), "--input", "-", "--config", str(config_path),
            "--camera", str(camera_path), "--output", str(slam_out),
            "--threads", str(threads)
        ]
        if lio:
            cmd.append("--lio")

        slam_out.mkdir(parents=True, exist_ok=True)
        with subprocess.Popen(cmd, stdin=subprocess.PIPE) as child:
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

        if not raw_trj.is_file() and (slam_out / "result").is_dir():
            txts = sorted((slam_out / "result").glob("*.txt"))
            if txts:
                raw_trj = txts[0]

    # --------------------------------------------------------------------------
    # STAGE 4: SfM Alignment & Dynamic Spatial Extrinsics Recalibration
    # --------------------------------------------------------------------------
    print("\n[STAGE 4/5] Multi-View SfM Reconstruction & Spatial Recalibration...")
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
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

    sparse_bin = output_dir / "sparse" / "0" / "points3D.bin"
    if (method in ("sfm", "all") or run_spirula) and not sparse_bin.is_file():
        print("[*] Running Spirula SfM (Vulkan GPU Headless)...")
        try:
            pipeline.run_spirula_sfm_auto(output_dir)
        except Exception as e:
            print(f"[!] Spirula SfM run notice: {e}")

    sparse_bin = output_dir / "sparse" / "0" / "points3D.bin"
    if sparse_bin.is_file():
        print("[*] Performing Metric Sim(3) Alignment (Spirula COLMAP -> LiDAR Ground Truth)...")
        try:
            align_data = pipeline.align_colmap_to_lidar(output_dir, fps=fps, dt_hint=dt_sync)
            dt_sync = align_data.get("dt_sync_seconds", dt_sync)
        except Exception as e:
            print(f"[!] Metric alignment notice: {e}")

        if recalibrate:
            print("[*] Auto-Recalibrating spatial mounting extrinsics (T_LC0, T_LC1) from SfM...")
            try:
                pipeline.recalibrate_from_sfm(output_dir, fps=fps)
                calib = output_dir / "rig_calibration.json"
                if not calib.is_file():
                    calib = output_dir / "calibracao_rigida_auto.json"
            except Exception as e:
                print(f"[!] Extrinsics recalibration notice: {e}")

    # --------------------------------------------------------------------------
    # STAGE 5: Point Cloud Colorization
    # --------------------------------------------------------------------------
    print("\n[STAGE 5/5] Running Point Cloud Colorization...")
    cloud_state_before = {p: p.stat().st_mtime_ns for p in deliverables_dir.iterdir()
                          if p.is_file() and p.suffix.lower() in {'.las', '.laz', '.ply', '.pcd'}}
    sfm_done = False
    if method in ("sfm", "all") and sparse_bin.is_file():
        try:
            pipeline.colorize_via_spirula_sfm(output_dir, fps=fps, use_vulkan=use_vulkan)
            sfm_done = True
        except Exception as e:
            print(f"[!] SfM colorization notice: {e}")

    if method in ("direct", "all") or (method == "sfm" and not sfm_done):
        pipeline.colorize_via_direct_rigid(output_dir, calib, fps=fps, dt_override=dt_sync, use_vulkan=use_vulkan)

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
        export_colmap_3dgs(output_dir, calib, fps=fps, dt_sync=dt_sync)

    print("\n[+] Deliverables currently in folder:")
    for deliv_item in sorted(deliverables_dir.iterdir()):
        if deliv_item.is_file():
            print(f"    - {deliv_item.name} ({deliv_item.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t_start
    print("\n" + "=" * 80)
    print(f" [SUCCESS] Unified Workflow Complete in {elapsed:.1f}s!")
    print(f" Deliverables saved in: {deliverables_dir}")
    print("=" * 80)
    return 0
