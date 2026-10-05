#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified pipeline: Auto-Calibração, Sincronização Temporal e Coloração LiDAR-Câmera 360
========================================================================================
Complete integration:
1. Executa o Spirula Studio SfM (spirula.exe) de forma headless (se solicitado ou se sparse não existir).
2. Realiza o alinhamento de alta precisão Sim(3) + ICP dos tie-points do COLMAP contra a nuvem LiDAR.
3. Automatically synchronize the exact time (Δt) entre o vídeo INSV e o SLAM (0 ms de erro) via SfM ou IMU Gyro.
4. Automatically recalibrate and compensate variações mecânicas de aperto do suporte (Yaw/Roll/Pitch).
5. Colorize the point cloud with occlusion via Z-Buffer (960x960) e ponderação de nitidez óptica.
6. Suporta os dois métodos:
   - Method A: SfM Spirula Alinhado (Padrão Ouro, 100% autônomo e aprumado).
   - Method B: Direto Rígido (Rápido, universal para qualquer dataset com o mesmo rig fixo).
"""

import os
import sys
import time
import struct
import json
import math
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from scipy.ndimage import minimum_filter
from raven_app.video import frame_time
from raven_app.slam_source import active_slam_dir
from raven_app.person_masks import load_keep_mask, keep_samples, masked_operator_keep
from raven_app.vulkan_engine import get_vulkan_bin, colorize_views

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
SPIRULA_EXE = WORKSPACE_DIR / "spirula" / "spirula.exe"
DEFAULT_CALIB_JSON = WORKSPACE_DIR / "calibracao_rigida_raven_insta360.json"
CALIBRATION_VERSION = 3


def _active_slam_signature(dataset_dir, *, trajectory_path=None, cloud_path=None,
                           include_cloud=None):
    """Fingerprint the exact trajectory/cloud used for an alignment.

    Sample-SfM alignments may use the parent dataset's trajectory without
    copying a multi-gigabyte cloud. In that case the trajectory remains the
    full source signature and the cloud is deliberately omitted.
    """
    trajectory = Path(trajectory_path) if trajectory_path is not None else get_slam_trajectory_path(dataset_dir)
    signature = {'trajectory': str(trajectory.resolve()),
                 'trajectory_mtime_ns': trajectory.stat().st_mtime_ns}
    if include_cloud is None:
        include_cloud = cloud_path is not None
        if cloud_path is None:
            try:
                candidate = active_slam_dir(dataset_dir) / 'pcd' / 'all_raw_points.pcd'
                include_cloud = candidate.is_file()
            except (FileNotFoundError, ValueError):
                include_cloud = False
    if include_cloud:
        cloud = Path(cloud_path) if cloud_path is not None else (
            active_slam_dir(dataset_dir) / 'pcd' / 'all_raw_points.pcd')
        signature.update(cloud=str(cloud.resolve()), cloud_mtime_ns=cloud.stat().st_mtime_ns)
    return signature


# ==============================================================================
# LEITURA DE DADOS
# ==============================================================================

def load_pcd(path):
    print(f"[*] Loading PCD cloud: {path.name}...")
    with open(path, "rb") as f:
        header = {}
        for _ in range(100):
            raw_line = f.readline()
            if not raw_line:
                raise ValueError('Truncated PCD header')
            line = raw_line.decode('ascii').strip().split()
            if not line or line[0].startswith('#'):
                continue
            header[line[0]] = line[1:]
            if line[0] == 'DATA':
                break
        else:
            raise ValueError('PCD header exceeds 100 lines')
        if header.get('DATA') != ['binary']:
            raise ValueError('Expected uncompressed binary PCD')
        names = header['FIELDS']
        sizes = list(map(int, header['SIZE']))
        counts = list(map(int, header.get('COUNT', ['1'] * len(names))))
        types = {'F':'f', 'I':'i', 'U':'u'}
        if not (len(names) == len(sizes) == len(counts) == len(header['TYPE'])):
            raise ValueError('Inconsistent PCD field layout')
        formats = [('<' + types[t] + str(s), (c,)) if c != 1 else '<' + types[t] + str(s)
                   for t, s, c in zip(header['TYPE'], sizes, counts)]
        dtype = np.dtype(list(zip(names, formats)))
        n = int(header['POINTS'][0])
        if n <= 0 or not all(name in names for name in ('x','y','z')):
            raise ValueError('PCD must contain points with XYZ fields')
        if os.fstat(f.fileno()).st_size - f.tell() != n * dtype.itemsize:
            raise ValueError('PCD payload size does not match its header')
        data = np.fromfile(f, dtype=dtype, count=n)
        xyz = np.column_stack([data[name] for name in ('x','y','z')]).astype(np.float64)
    print(f"    Loaded points: {len(xyz):,}")
    return xyz


def get_slam_trajectory_path(dataset_dir):
    """Find the SLAM trajectory file dynamically regardless of scanner model."""
    result_dir = active_slam_dir(dataset_dir) / "result"
    for name in ("Eagle_Scan.txt", "Eagle_X6_Scan.txt", "Raven_3DMakerPro_Scan.txt", "trajectory.txt"):
        p = result_dir / name
        if p.is_file():
            return p
    if result_dir.is_dir():
        txts = sorted(result_dir.glob("*.txt"))
        if txts:
            return txts[0]
    return result_dir / "Eagle_Scan.txt"


def load_trajectory(path):
    path = Path(path)
    if not path.is_file():
        # Try finding any trajectory file in the same directory
        if path.parent.is_dir():
            txts = sorted(path.parent.glob("*.txt"))
            if txts:
                path = txts[0]
    print(f"[*] Loading SLAM trajectory: {path.name}...")
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    rotations = Rot.from_quat(data[:, 4:8])
    print(f"    Total poses: {len(data):,}, Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    return timestamps, positions, rotations


def load_colmap_cameras(cameras_bin):
    from raven_app.third_camera import _read_colmap_cameras
    return _read_colmap_cameras(cameras_bin)


def _colorization_camera_params(camera):
    """Pack mixed COLMAP models into the native/CPU projection protocol."""
    if camera['model_id'] == 10:
        return camera['params']
    from raven_app.third_camera import _third_camera_intrinsics_from_colmap
    model, intr = _third_camera_intrinsics_from_colmap(camera)
    marker = -999.0 if model == 'PINHOLE' else -998.0 if model == 'OPENCV' else 0.0
    return tuple(float(intr.get(key, 0.0)) for key in
                 ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4')) + (0.0, marker)


def _auxiliary_colorization_camera_params(camera):
    """Pack an auxiliary camera's selected model into the shared CPU/Vulkan protocol."""
    model = str(camera.camera_model or 'PINHOLE').strip().upper()
    intr = camera.intrinsics
    supported = {'PINHOLE', 'OPENCV', 'OPENCV_FISHEYE', 'THIN_PRISM_FISHEYE'}
    if model not in supported:
        raise ValueError(f'Unsupported auxiliary camera model: {camera.camera_model}')

    base = [float(intr[key]) for key in ('fx', 'fy', 'cx', 'cy')]
    if not np.isfinite(base).all() or min(base[:2]) <= 0:
        raise ValueError(f'Invalid auxiliary camera intrinsics for {camera.camera_name}')
    get = lambda key: float(intr.get(key, 0.0))

    if model == 'PINHOLE':
        distortion = [0.0] * 7 + [-999.0]
    elif model == 'OPENCV':
        distortion = [get('k1'), get('k2'), get('p1'), get('p2'), 0.0, 0.0, 0.0, -998.0]
    elif model == 'OPENCV_FISHEYE':
        distortion = [get('k1'), get('k2'), 0.0, 0.0,
                      get('k3'), get('k4'), 0.0, 0.0]
    else:
        distortion = [get(key) for key in
                      ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')]

    params = tuple(base + distortion)
    if len(params) != 12 or not np.isfinite(params).all():
        raise ValueError(f'Invalid auxiliary camera distortion for {camera.camera_name}')
    return params


def load_colmap_images(images_bin):
    images = []
    with open(images_bin, "rb") as f:
        num_reg = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c.decode("ascii")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, 1)

            R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
            C = -R_cw.T @ np.array(tvec)
            images.append({
                "img_id": img_id,
                "name": name,
                "cam_id": cam_id,
                "R_cw": R_cw,
                "C": C,
                "tvec": np.array(tvec)
            })
    return images


def load_colmap_points(points_bin):
    pts = []
    rgb = []
    with open(points_bin, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            c = struct.unpack("<3B", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, 1)
            pts.append(xyz)
            rgb.append(c)
    return np.array(pts, dtype=np.float64), np.array(rgb, dtype=np.uint8)


# ==============================================================================
# ALGORITMOS MATEMÁTICOS (SIM3, THIN-PRISM, ICP)
# ==============================================================================

def solve_umeyama_sim3(X, Y):
    """Resolve Y = s * R * X + t (onde Y é métrico LiDAR e X é COLMAP)"""
    n = len(X)
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    var_X = np.mean(np.sum((X - mu_X) ** 2, axis=1))

    Sigma = (Y - mu_Y).T @ (X - mu_X) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_X
    t = mu_Y - s * R @ mu_X

    Y_pred = (s * R @ X.T).T + t
    err = np.linalg.norm(Y - Y_pred, axis=1)
    rmse = np.sqrt(np.mean(err ** 2))
    return s, R, t, rmse


def project_thin_prism(P_v, params):
    """Projeção de pontos 3D da câmera no modelo fisheye Thin Prism de 12 parâmetros"""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    x = P_v[:, 0] / np.maximum(P_v[:, 2], 1e-12)
    y = P_v[:, 1] / np.maximum(P_v[:, 2], 1e-12)
    if sy1 < -997.0:  # Auxiliary pinhole / OpenCV model marker for native RVC views.
        rd2 = x*x + y*y
        radial = 1 + rd2 * (k1 + rd2 * (k2 + rd2 * (k3 + rd2 * k4)))
        xd = x*radial + 2*p1*x*y + p2*(rd2 + 2*x*x)
        yd = y*radial + p1*(rd2 + 2*y*y) + 2*p2*x*y
        u, v = fx*xd + cx, fy*yd + cy
        r = np.hypot(u-cx, v-cy)
    else:
        r = np.hypot(P_v[:, 0], P_v[:, 1])
        theta = np.arctan2(r, P_v[:, 2])
        scale = np.divide(theta, r, out=np.zeros_like(theta), where=r > 1e-12)
        xd, yd = P_v[:, 0] * scale, P_v[:, 1] * scale
        rd2 = theta**2
        radial = 1 + rd2 * (k1 + rd2 * (k2 + rd2 * (k3 + rd2 * k4)))
        du = 2*p1*xd*yd + p2*(rd2 + 2*xd**2) + sx1*rd2
        dv = p1*(rd2 + 2*yd**2) + 2*p2*xd*yd + sy1*rd2
        u, v = fx*(xd*radial + du) + cx, fy*(yd*radial + dv) + cy
    return u, v, r


def visible_depths(distances, flat_indices, width, height):
    """Conservative radial-depth test, including adjacent sparse raster cells."""
    depth = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(depth, flat_indices, np.floor(distances * 1000).astype(np.float32) / 1000)
    nearest = minimum_filter(depth.reshape(height, width), size=3,
                             mode='constant', cval=np.inf).ravel()[flat_indices]
    return distances <= nearest + np.maximum(0.03, 0.01 * nearest)


def sample_bilinear(image, u, v):
    x, y = np.floor(u).astype(int), np.floor(v).astype(int)
    dx, dy = (u-x)[:, None], (v-y)[:, None]
    value = ((1-dx)*(1-dy)*image[y,x] + dx*(1-dy)*image[y,x+1]
             + (1-dx)*dy*image[y+1,x] + dx*dy*image[y+1,x+1])
    return np.clip(value, 0, 255).astype(np.uint8)


def write_ply(path, points, colors_rgb):
    n = len(points)
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

    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")]
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["r"] = colors_rgb[:, 0]
    arr["g"] = colors_rgb[:, 1]
    arr["b"] = colors_rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Saved PLY: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


def write_pcd(path, points, colors_rgb):
    n = len(points)
    rgb_packed = (
        (colors_rgb[:, 0].astype(np.uint32) << 16)
        | (colors_rgb[:, 1].astype(np.uint32) << 8)
        | (colors_rgb[:, 2].astype(np.uint32))
    ).view(np.float32)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")

    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["rgb"] = rgb_packed

    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Saved PCD: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


# ==============================================================================
# SINCRONIZAÇÃO AUTOMÁTICA VIA IMU GYRO (INSV + ROS BAG)
# ==============================================================================

def sync_via_gyro_cross_correlation(insv_path, bag_path, trj_path):
    """Sincronização temporal automática sub-milissegundo entre INSV e ROS Bag via IMU Gyro"""
    try:
        from rosbags.highlevel import AnyReader
        HEADER_SIZE = 72
        with open(insv_path, "rb") as fin:
            fin.seek(-HEADER_SIZE, 2)
            header = fin.read(HEADER_SIZE)
            extra_size = struct.unpack('<I', header[32:36])[0]
            file_size = fin.seek(0, 2)
            extra_start = file_size - extra_size

            fin.seek(-(HEADER_SIZE + 6 + 250), 2)
            offsets_data = fin.read(250)
            offsets = {}
            for i in range(0, len(offsets_data), 10):
                oid, ofmt, osize, ooff = struct.unpack('<BBII', offsets_data[i:i+10])
                if oid > 0:
                    offsets[oid] = (ofmt, osize, ooff)

            if 3 not in offsets or 4 not in offsets:
                return None
            _, g_size, g_off = offsets[3]
            fin.seek(extra_start + g_off)
            gyro_bytes = fin.read(g_size)

            _, e_size, e_off = offsets[4]
            fin.seek(extra_start + e_off)
            exp_bytes = fin.read(e_size)

        cam_imu = np.frombuffer(gyro_bytes, dtype=[('t', '<u8'), ('ax', '<u2'), ('ay', '<u2'), ('az', '<u2'), ('gx', '<u2'), ('gy', '<u2'), ('gz', '<u2')])
        cam_exp = np.frombuffer(exp_bytes, dtype=[('t', '<u8'), ('exp', '<f8')])
        t_exp0 = cam_exp['t'][0]
        t_cam_s = (cam_imu['t'].astype(np.float64) - t_exp0) / 1e6

        gx = cam_imu['gx'].astype(np.float64) - 32768.0
        gy = cam_imu['gy'].astype(np.float64) - 32768.0
        gz = cam_imu['gz'].astype(np.float64) - 32768.0
        cam_gyro_norm = np.sqrt(gx**2 + gy**2 + gz**2)

        lidar_t, lidar_wx, lidar_wy, lidar_wz = [], [], [], []
        with AnyReader([bag_path]) as reader:
            for conn, ts, raw in reader.messages():
                if 'imu' in conn.topic.lower():
                    msg = reader.deserialize(raw, conn.msgtype)
                    t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    lidar_t.append(t_sec)
                    lidar_wx.append(msg.angular_velocity.x)
                    lidar_wy.append(msg.angular_velocity.y)
                    lidar_wz.append(msg.angular_velocity.z)

        lidar_t = np.array(lidar_t)
        lidar_norm = np.sqrt(np.array(lidar_wx)**2 + np.array(lidar_wy)**2 + np.array(lidar_wz)**2)

        trj = np.loadtxt(trj_path)
        t_slam_start = trj[0, 0]
        t_lidar_rel = lidar_t - t_slam_start

        fs = 200.0
        t_eval = np.arange(1.0, min(80.0, t_cam_s[-1] - 1.0), 1.0 / fs)
        cam_interp = np.interp(t_eval, t_cam_s, cam_gyro_norm)
        cam_interp -= np.median(cam_interp)

        shifts = np.arange(-30.0, 30.0, 0.005)
        corrs = []
        for shift in shifts:
            t_l_q = t_eval - shift
            valid = (t_l_q >= t_lidar_rel[0]) & (t_l_q <= t_lidar_rel[-1])
            if np.sum(valid) > 1000:
                lid_interp = np.interp(t_l_q[valid], t_lidar_rel, lidar_norm)
                lid_interp -= np.median(lid_interp)
                c = np.corrcoef(cam_interp[valid], lid_interp)[0, 1]
                corrs.append(c)
            else:
                corrs.append(0.0)

        corrs = np.array(corrs)
        best_idx = np.argmax(corrs)
        best_dt = shifts[best_idx]
        best_r = corrs[best_idx]
        if best_r > 0.5:
            print(f"[+] Sincronização IMU Gyro Cross-Correlation bem-sucedida: Δt = {best_dt:.4f}s (Pearson r = {best_r:.4f})")
            return float(best_dt)
        return None
    except Exception as e:
        print(f"[*] Gyro synchronization unavailable ({e}), proceeding...")
        return None


# ==============================================================================
# PIPELINE MÉTODO 1: SPIRULA SFM ALINHADO (PADRÃO OURO)
# ==============================================================================

def get_best_vulkan_device() -> int:
    """Retorna o índice do melhor dispositivo Vulkan (priorizando GPU discreta dedicada como NVIDIA/AMD)."""
    try:
        from raven_app.subprocess_utils import hidden_window_options
        res = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True,
                             timeout=5, **hidden_window_options())
        current_dev = -1
        dev_scores = {}
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("GPU") and line_str.endswith(":") and len(line_str) <= 6:
                digits = "".join(filter(str.isdigit, line_str))
                if digits:
                    current_dev = int(digits)
                    dev_scores[current_dev] = 0
            elif current_dev >= 0:
                if "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" in line_str:
                    dev_scores[current_dev] += 100
                if any(k in line_str.lower() for k in ("nvidia", "geforce", "rtx", "radeon")):
                    dev_scores[current_dev] += 50
        if dev_scores:
            best = max(dev_scores, key=dev_scores.get)
            if dev_scores[best] > 0:
                return best
    except Exception:
        pass
    return -1


def _spirula_auxiliary_flags(dataset_dir, auxiliary_cameras=None):
    from raven_app.third_camera import load_third_camera_configs
    if auxiliary_cameras is None:
        config = Path(dataset_dir) / 'third_camera.json'
        auxiliary_cameras = load_third_camera_configs(config) if config.is_file() else []
    flags = []
    for camera in auxiliary_cameras:
        if not camera.enabled or not any((Path(dataset_dir) / 'images' / camera.camera_name).glob('*.jpg')):
            continue
        model = camera.camera_model.lower().replace('_', '-')
        if model not in ('pinhole', 'opencv', 'opencv-fisheye', 'thin-prism-fisheye'):
            raise ValueError(f'Unsupported auxiliary camera model: {camera.camera_model}')
        focal = float(camera.intrinsics['fx'])
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError(f'Invalid focal length for {camera.camera_name}')
        flags.extend(['--camera-model', f'{camera.camera_name}={model}',
                      '--focal', f'{camera.camera_name}={focal:g}'])
        if model != 'pinhole':
            keys = {
                'opencv-fisheye': ('k1', 'k2', 'k3', 'k4'),
                'opencv': ('k1', 'k2', 'p1', 'p2'),
                'thin-prism-fisheye': (
                    'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1'),
            }[model]
            values = ','.join(f'{float(camera.intrinsics.get(key, 0)):g}' for key in keys)
            flags.extend(['--distortion', f'{camera.camera_name}={values}'])
        print(f'[*] SfM camera {camera.camera_name}: {model}, focal prior {focal:g} px')
    return flags


def run_spirula_sfm_auto(dataset_dir, quality="medium", auxiliary_cameras=None,
                         force_rebuild=False, allow_partial_calibration=False):
    """Executa o Spirula Studio via CLI se a pasta sparse/0 não existir"""
    dataset_dir = Path(dataset_dir)
    auxiliary_flags = _spirula_auxiliary_flags(dataset_dir, auxiliary_cameras)
    img_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"

    if not force_rebuild and all((sparse_dir / name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        _validate_sfm_camera_models(sparse_dir, auxiliary_flags)
        print(f"[+] Existing SfM reconstruction found at: {sparse_dir}")
        return True

    try:
        from raven_app.config import get_spirula_bin
        spirula_bin = get_spirula_bin()
    except Exception:
        spirula_bin = SPIRULA_EXE

    if not spirula_bin.exists():
        print(f"[!] spirula.exe not found at: {spirula_bin}")
        return False

    vulkan_dev = get_best_vulkan_device()
    print("=" * 80)
    print(f" INICIANDO SPIRULA STUDIO SFM (VULKAN GPU HEADLESS, DEVICE: {vulkan_dev if vulkan_dev >= 0 else 'DEFAULT'})...")
    print(f" Executable: {spirula_bin}")
    print(f" Images:     {img_dir}")
    print(f" Quality:    {quality}")
    print("=" * 80)

    cmd = [
        str(spirula_bin), "sfm", "auto",
        str(img_dir),
        "-o", str(dataset_dir),
        "--data-type", "video",
        "--quality", quality,
        "--camera-mode", "folder",
        "--focal", "1080.19",
        "--camera-model", "thin-prism-fisheye"
    ]
    cmd.extend(auxiliary_flags)
    if force_rebuild:
        cmd.append('--no-resume')
    if vulkan_dev >= 0:
        cmd.extend(["--device", str(vulkan_dev)])

    t0 = time.time()
    from raven_app.subprocess_utils import run_hidden_stream
    code = run_hidden_stream(cmd)
    model_files_present = all(
        (sparse_dir / name).is_file()
        for name in ("cameras.bin", "images.bin", "points3D.bin"))
    if code != 0:
        if code == 3 and allow_partial_calibration and model_files_present:
            cameras = load_colmap_cameras(sparse_dir / "cameras.bin")
            if not cameras:
                print("[!] Spirula partial sample has no valid COLMAP cameras")
                return False
            _validate_sfm_camera_models(sparse_dir, auxiliary_flags)
            print("[!] Spirula returned code 3; retaining sparse/0 only as a sample-calibration candidate. "
                  "Per-camera coverage and SLAM rig-calibration gates are still required.")
            return True
        print(f"[!] Spirula SFM exited with code {code}")
        return False
    if not model_files_present:
        print("[!] Spirula SFM did not produce a complete sparse/0 model")
        return False

    cameras = load_colmap_cameras(sparse_dir / "cameras.bin")
    if not cameras:
        return False
    _validate_sfm_camera_models(sparse_dir, auxiliary_flags)
    print(f"[+] Spirula SFM completed in {time.time() - t0:.1f}s!")
    return True


def _validate_sfm_camera_models(sparse_dir, auxiliary_flags):
    cameras = load_colmap_cameras(sparse_dir / 'cameras.bin')
    expected = {'cam0': 10, 'cam1': 10}
    model_ids = {
        'pinhole': 1,
        'opencv': 4,
        'opencv-fisheye': 5,
        'thin-prism-fisheye': 10,
    }
    for index, flag in enumerate(auxiliary_flags):
        if flag == '--camera-model':
            prefix, model = auxiliary_flags[index + 1].split('=', 1)
            expected[prefix] = model_ids[model]
    for image in load_colmap_images(sparse_dir / 'images.bin'):
        prefix = image['name'].replace('\\', '/').split('/')[0]
        if prefix in expected and cameras[image['cam_id']]['model_id'] != expected[prefix]:
            raise ValueError(f'SfM camera model mismatch for {prefix}; rebuild SfM with the selected camera models')


def align_colmap_to_lidar(dataset_dir, fps=2.0, dt_hint=None, *,
                          trajectory_path=None, use_icp=True):
    """Executa o alinhamento de alta precisão Sim(3) + ICP entre COLMAP e LiDAR"""
    manifest = dataset_dir / 'images/frames.json'
    if manifest.is_file() and json.loads(manifest.read_text(encoding='utf-8')).get('time_source') == 'insv_timelapse':
        from raven_app.timelapse_calibration import align_dataset
        result = align_dataset(dataset_dir, dt_hint, trajectory_path=trajectory_path)
        source_trajectory = Path(trajectory_path) if trajectory_path is not None else get_slam_trajectory_path(dataset_dir)
        result['slam_source_signature'] = _active_slam_signature(
            dataset_dir, trajectory_path=source_trajectory, include_cloud=False)
        (dataset_dir / 'colmap_to_lidar_alignment.json').write_text(
            json.dumps(result, indent=2), encoding='utf-8')
        return result
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_trj = Path(trajectory_path) if trajectory_path is not None else get_slam_trajectory_path(dataset_dir)
    slam_pcd = (active_slam_dir(dataset_dir) / "pcd" / "all_raw_points.pcd") if use_icp else None
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    print("=" * 80)
    print(" METRIC ALIGNMENT: COLMAP (SPIRULA) -> LIDAR SLAM")
    print("=" * 80)

    pts_colmap, rgb_colmap = load_colmap_points(sparse_dir / "points3D.bin")
    pts_lidar = load_pcd(slam_pcd) if use_icp else None
    t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
    col_imgs = load_colmap_images(sparse_dir / "images.bin")

    # Filtrar cam0 para busca temporal
    cam0 = [im for im in col_imgs if im["name"].startswith("cam0/")]
    def get_frame_num(name):
        digits = "".join(filter(str.isdigit, Path(name).stem))
        return int(digits) if digits else 0
    cam0.sort(key=lambda x: get_frame_num(x["name"]))

    t0_slam = t_slam[0]
    t1_slam = t_slam[-1]
    min_overlap = max(10, int(len(cam0) * 0.40))

    # Nominal physical camera arm offset for initial camera center approximation
    # The trajectory is gravity-aligned (world +Z up). Express the upright
    # camera lever in the initial body frame, rather than reusing another
    # capture's gravity vector (which previously placed the camera BELOW it).
    c_L_phys = rot_slam[0].inv().apply([0.0, 0.0, 0.185])
    trajectory_slerp = Slerp(t_slam, rot_slam)

    # Busca ótima do dt (coarse-to-fine adaptativa)
    print(f"[*] Searching for optimal time synchronization Δt (FPS = {fps:.1f}, hint = {dt_hint})...")
    best_rmse = 1e9
    best_dt = 0
    best_sim3 = None

    if dt_hint is not None:
        dts = np.linspace(dt_hint - 2.0, dt_hint + 2.0, 401)
    else:
        dts = np.linspace(-100.0, 50.0, 1501)

    for dt in dts:
        X_trj = []
        Y_trj = []
        for im in cam0:
            frame_num = get_frame_num(im["name"])
            t_vid = frame_time(dataset_dir, im["name"], fps)
            t_q = t0_slam + (t_vid - dt)
            if t0_slam <= t_q <= t1_slam:
                idx = np.searchsorted(t_slam, t_q)
                idx = np.clip(idx, 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                X_trj.append(im["C"])
                Y_trj.append(p_L + trajectory_slerp(t_q).apply(c_L_phys))

        if len(X_trj) >= min_overlap:
            s, R, t_trans, rmse = solve_umeyama_sim3(np.array(X_trj), np.array(Y_trj))
            if rmse < best_rmse:
                best_rmse = rmse
                best_dt = dt
                best_sim3 = (s, R, t_trans)

    if best_sim3 is None:
        # Fallback if window was too tight
        print("[!] Warning: Restricted search window found insufficient alignment. Widening search...")
        dts = np.linspace(-150.0, 50.0, 2001)
        for dt in dts:
            X_trj, Y_trj = [], []
            for im in cam0:
                frame_num = get_frame_num(im["name"])
                t_vid = frame_time(dataset_dir, im["name"], fps)
                t_q = t0_slam + (t_vid - dt)
                if t0_slam <= t_q <= t1_slam:
                    idx = np.searchsorted(t_slam, t_q)
                    idx = np.clip(idx, 1, len(t_slam) - 1)
                    w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                    p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                    X_trj.append(im["C"])
                    Y_trj.append(p_L + trajectory_slerp(t_q).apply(c_L_phys))

            if len(X_trj) >= min_overlap:
                s, R, t_trans, rmse = solve_umeyama_sim3(np.array(X_trj), np.array(Y_trj))
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_dt = dt
                    best_sim3 = (s, R, t_trans)

    if best_sim3 is None:
        raise RuntimeError("Could not align COLMAP trajectory against LiDAR SLAM.")

    s_init, R_init, t_init = best_sim3
    print(f"    Optimal synchronization: Δt = {best_dt:.4f}s | Trajectory RMSE: {best_rmse*100:.2f} cm (Scale s = {s_init:.4f})")

    s_comp, R_comp, t_comp = s_init, R_init.copy(), t_init.copy()
    rmse_icp = best_rmse
    if use_icp:
        # Refinamento Trimmed ICP nos tie-points 3D
        step_lidar = max(1, len(pts_lidar) // 200000)
        sub_lidar = pts_lidar[::step_lidar]
        tree = cKDTree(sub_lidar)
        step_col = max(1, len(pts_colmap) // 50000)
        raw_sub_col = pts_colmap[::step_col].copy()
        current_pts = s_init * (R_init @ raw_sub_col.T).T + t_init
        print(f"[*] Running Trimmed ICP refinement ({len(raw_sub_col):,} tie-points)...")
        for _ in range(10):
            dists, idxs = tree.query(current_pts, k=1)
            thresh = min(0.35, float(np.percentile(dists, 70)))
            mask = dists < thresh
            if np.sum(mask) < 20:
                break
            try:
                s_cand, R_cand, t_cand, r_cand = solve_umeyama_sim3(
                    raw_sub_col[mask], sub_lidar[idxs[mask]])
                if np.isfinite(r_cand):
                    s_comp, R_comp, t_comp, rmse_icp = s_cand, R_cand, t_cand, r_cand
                    current_pts = s_comp * (R_comp @ raw_sub_col.T).T + t_comp
            except Exception:
                break
        print(f"[+] ICP converged: RMSE = {rmse_icp*100:.2f} cm, Scale s = {s_comp:.6f}")
    else:
        print(f"[+] Sample-SfM trajectory alignment accepted without cloud ICP: "
              f"RMSE = {best_rmse*100:.2f} cm, Scale s = {s_comp:.6f}")

    align_data = {
        "calibration_version": CALIBRATION_VERSION,
        "slam_source_signature": _active_slam_signature(
            dataset_dir, trajectory_path=slam_trj, cloud_path=slam_pcd,
            include_cloud=use_icp),
        "frame_time_source": json.loads((dataset_dir / "images" / "frames.json").read_text(encoding="utf-8")).get("time_source", "video_pts") if (dataset_dir / "images" / "frames.json").is_file() else "legacy_frame_rate",
        "initial_camera_lever_body_m": c_L_phys.tolist(),
        "trajectory_rmse_cm": float(best_rmse * 100),
        "scale": float(s_comp),
        "R": R_comp.tolist(),
        "t": t_comp.tolist(),
        "rmse_cm": float(rmse_icp * 100),
        "dt_sync_seconds": float(best_dt),
        "alignment_method": "trajectory_only_sample_sfm" if not use_icp else "trajectory_and_cloud_icp",
    }
    with open(align_json, "w", encoding="utf-8") as f:
        json.dump(align_data, f, indent=2)

    return align_data


def current_alignment(dataset_dir, fps=2.0):
    """Upgrade alignments made with the old inverted lever initialization."""
    path = dataset_dir / "colmap_to_lidar_alignment.json"
    saved = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}
    saved_signature = saved.get('slam_source_signature') or {}
    trajectory = None
    cloud = None
    try:
        candidate = get_slam_trajectory_path(dataset_dir)
        if candidate.is_file():
            trajectory = candidate
    except (FileNotFoundError, ValueError):
        pass
    if trajectory is None and not (dataset_dir / 'active_slam.json').is_file():
        candidate = Path(saved_signature['trajectory']) if saved_signature.get('trajectory') else None
        if candidate is not None and candidate.is_file():
            trajectory = candidate

    uses_cloud = 'cloud' in saved_signature
    if not saved_signature:
        try:
            cloud = active_slam_dir(dataset_dir) / 'pcd' / 'all_raw_points.pcd'
            uses_cloud = cloud.is_file()
        except (FileNotFoundError, ValueError):
            uses_cloud = False
    if uses_cloud:
        try:
            candidate = active_slam_dir(dataset_dir) / 'pcd' / 'all_raw_points.pcd'
            if candidate.is_file():
                cloud = candidate
        except (FileNotFoundError, ValueError):
            pass
        if cloud is None and not (dataset_dir / 'active_slam.json').is_file():
            candidate = Path(saved_signature['cloud'])
            if candidate.is_file():
                cloud = candidate

    active_signature = None
    if trajectory is not None and (not uses_cloud or cloud is not None):
        active_signature = _active_slam_signature(
            dataset_dir, trajectory_path=trajectory, cloud_path=cloud,
            include_cloud=uses_cloud)
    if saved_signature != active_signature:
        if trajectory is None:
            trajectory = get_slam_trajectory_path(dataset_dir)
        return align_colmap_to_lidar(
            dataset_dir, fps, saved.get('dt_sync_seconds'),
            trajectory_path=trajectory, use_icp=uses_cloud)
    manifest = dataset_dir / "images" / "frames.json"
    source = json.loads(manifest.read_text(encoding='utf-8')).get('time_source', 'video_pts') if manifest.is_file() else 'legacy_frame_rate'
    if source == 'insv_timelapse':
        from raven_app.timelapse_calibration import VERSION
        if saved.get('timelapse_calibration_version') != VERSION or saved.get('quality_status') != 'accepted':
            return align_colmap_to_lidar(
                dataset_dir, fps, saved.get('dt_sync_seconds'),
                trajectory_path=trajectory, use_icp=uses_cloud)
    if saved.get('calibration_version') != CALIBRATION_VERSION:
        return align_colmap_to_lidar(
            dataset_dir, fps, saved.get('dt_sync_seconds'),
            trajectory_path=trajectory, use_icp=uses_cloud)
    return saved


def transform_colmap_to_metric(src_dir: Path, dst_dir: Path, s_sim: float, R_sim: np.ndarray,
                               t_sim: np.ndarray, include_images=None):
    """Transform COLMAP into LiDAR metric space, filtering poses without orphaned tracks."""
    from raven_app.third_camera import _COLMAP_CAMERA_MODEL_NAMES, _COLMAP_CAMERA_PARAM_COUNTS

    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    scale = float(s_sim)
    rotation = np.asarray(R_sim, dtype=np.float64)
    translation = np.asarray(t_sim, dtype=np.float64)
    if (not np.isfinite(scale) or scale <= 0 or rotation.shape != (3, 3)
            or translation.shape != (3,) or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(translation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or np.linalg.det(rotation) <= 0):
        raise ValueError('Invalid Sim(3) alignment for COLMAP transform')
    allowed = None if include_images is None else {str(name).replace('\\', '/') for name in include_images}

    cameras = []
    with (src_dir / 'cameras.bin').open('rb') as stream:
        camera_count = struct.unpack('<Q', stream.read(8))[0]
        for _ in range(camera_count):
            header = stream.read(24)
            if len(header) != 24:
                raise ValueError('Truncated COLMAP cameras.bin')
            camera_id, model_id, width, height = struct.unpack('<iiQQ', header)
            param_count = _COLMAP_CAMERA_PARAM_COUNTS.get(model_id)
            if param_count is None:
                raise ValueError(f'Unsupported COLMAP camera model id: {model_id}')
            raw = stream.read(param_count * 8)
            if len(raw) != param_count * 8:
                raise ValueError('Truncated COLMAP camera parameters')
            params = struct.unpack(f'<{param_count}d', raw)
            cameras.append((camera_id, model_id, width, height, params))
    with (dst_dir / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            stream.write(struct.pack('<iiQQ', camera_id, model_id, width, height))
            stream.write(struct.pack(f'<{len(params)}d', *params))
    with (dst_dir / 'cameras.txt').open('w', encoding='utf-8') as stream:
        stream.write('# Camera list with one line of data per camera:\n')
        stream.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for camera_id, model_id, width, height, params in cameras:
            model = _COLMAP_CAMERA_MODEL_NAMES[model_id]
            stream.write(f"{camera_id} {model} {width} {height} "
                         f"{' '.join(format(value, '.17g') for value in params)}\n")

    images = []
    image_bin = src_dir / 'images.bin'
    if image_bin.is_file():
        with image_bin.open('rb') as stream:
            image_count = struct.unpack('<Q', stream.read(8))[0]
            for _ in range(image_count):
                image_id = struct.unpack('<I', stream.read(4))[0]
                qw, qx, qy, qz = struct.unpack('<4d', stream.read(32))
                tx, ty, tz = struct.unpack('<3d', stream.read(24))
                camera_id = struct.unpack('<I', stream.read(4))[0]
                name_bytes = bytearray()
                while True:
                    character = stream.read(1)
                    if not character:
                        raise ValueError('Truncated COLMAP image name')
                    if character == b'\x00':
                        break
                    name_bytes.extend(character)
                name = name_bytes.decode('ascii').replace('\\', '/')
                point_count = struct.unpack('<Q', stream.read(8))[0]
                points2d = []
                for _ in range(point_count):
                    x, y = struct.unpack('<2d', stream.read(16))
                    point_id = struct.unpack('<Q', stream.read(8))[0]
                    points2d.append([x, y, point_id])
                if allowed is not None and name not in allowed:
                    continue
                R_cw = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
                center = -R_cw.T @ np.array([tx, ty, tz], dtype=np.float64)
                center_metric = scale * (rotation @ center) + translation
                R_metric = R_cw @ rotation.T
                t_metric = -R_metric @ center_metric
                q_metric = Rot.from_matrix(R_metric).as_quat()
                images.append({
                    'id': image_id, 'q': (q_metric[3], q_metric[0], q_metric[1], q_metric[2]),
                    't': tuple(t_metric), 'camera_id': camera_id, 'name': name,
                    'points2d': points2d,
                })

    used_camera_ids = {image['camera_id'] for image in images}
    camera_ids = {camera[0] for camera in cameras}
    if not used_camera_ids <= camera_ids:
        raise ValueError('COLMAP image references a missing camera record')
    cameras = [camera for camera in cameras if camera[0] in used_camera_ids]
    with (dst_dir / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(cameras)))
        for camera_id, model_id, width, height, params in cameras:
            stream.write(struct.pack('<iiQQ', camera_id, model_id, width, height))
            stream.write(struct.pack(f'<{len(params)}d', *params))
    with (dst_dir / 'cameras.txt').open('w', encoding='utf-8') as stream:
        stream.write('# Camera list with one line of data per camera:\n')
        stream.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for camera_id, model_id, width, height, params in cameras:
            model = _COLMAP_CAMERA_MODEL_NAMES[model_id]
            stream.write(f"{camera_id} {model} {width} {height} "
                         f"{' '.join(format(value, '.17g') for value in params)}\n")

    image_by_id = {image['id']: image for image in images}
    retained_image_ids = set(image_by_id)
    points = []
    point_bin = src_dir / 'points3D.bin'
    if point_bin.is_file():
        with point_bin.open('rb') as stream:
            point_count = struct.unpack('<Q', stream.read(8))[0]
            for _ in range(point_count):
                point_id = struct.unpack('<Q', stream.read(8))[0]
                xyz = struct.unpack('<3d', stream.read(24))
                rgb = struct.unpack('<3B', stream.read(3))
                error = struct.unpack('<d', stream.read(8))[0]
                track_count = struct.unpack('<Q', stream.read(8))[0]
                tracks = [struct.unpack('<2I', stream.read(8)) for _ in range(track_count)]
                valid_tracks = []
                for image_id, point2d_index in tracks:
                    if image_id not in retained_image_ids:
                        continue
                    image = image_by_id[image_id]
                    if (point2d_index >= len(image['points2d'])
                            or image['points2d'][point2d_index][2] != point_id):
                        continue
                    valid_tracks.append((image_id, point2d_index))
                if allowed is not None and not valid_tracks:
                    continue
                xyz_metric = scale * (rotation @ np.asarray(xyz, dtype=np.float64)) + translation
                points.append({
                    'id': point_id, 'xyz': xyz_metric, 'rgb': rgb,
                    'error': scale * error, 'tracks': valid_tracks,
                })

    retained_point_ids = {point['id'] for point in points}
    invalid_id = (1 << 64) - 1
    for image in images:
        for point2d in image['points2d']:
            if point2d[2] != invalid_id and point2d[2] not in retained_point_ids:
                point2d[2] = invalid_id

    with (dst_dir / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(images)))
        for image in images:
            stream.write(struct.pack('<I', image['id']))
            stream.write(struct.pack('<4d', *image['q']))
            stream.write(struct.pack('<3d', *image['t']))
            stream.write(struct.pack('<I', image['camera_id']))
            stream.write(image['name'].encode('ascii') + b'\x00')
            stream.write(struct.pack('<Q', len(image['points2d'])))
            for x, y, point_id in image['points2d']:
                stream.write(struct.pack('<2dQ', x, y, point_id))
    with (dst_dir / 'images.txt').open('w', encoding='utf-8') as stream:
        stream.write('# Image list with two lines of data per image:\n')
        stream.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        stream.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        for image in images:
            stream.write(f"{image['id']} {' '.join(format(v, '.17g') for v in image['q'])} "
                         f"{' '.join(format(v, '.17g') for v in image['t'])} "
                         f"{image['camera_id']} {image['name']}\n")
            stream.write(' '.join(
                f"{format(x, '.17g')} {format(y, '.17g')} {-1 if point_id == invalid_id else point_id}"
                for x, y, point_id in image['points2d']) + '\n')

    with (dst_dir / 'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(points)))
        for point in points:
            stream.write(struct.pack('<Q3d3BdQ', point['id'], *point['xyz'], *point['rgb'],
                                     point['error'], len(point['tracks'])))
            for track in point['tracks']:
                stream.write(struct.pack('<2I', *track))
    with (dst_dir / 'points3D.txt').open('w', encoding='utf-8') as stream:
        stream.write('# 3D point list with one line of data per point:\n')
        stream.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        for point in points:
            tracks = ' '.join(f'{image_id} {point2d_index}' for image_id, point2d_index in point['tracks'])
            stream.write(f"{point['id']} {' '.join(format(v, '.17g') for v in point['xyz'])} "
                         f"{' '.join(str(v) for v in point['rgb'])} {format(point['error'], '.17g')} {tracks}\n")

    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n'
    ).encode('ascii')
    dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]
    cloud = np.empty(len(points), dtype=dtype)
    for index, point in enumerate(points):
        cloud[index] = (*point['xyz'], *point['rgb'])
    ply_path = dst_dir / 'points3D.ply'
    with ply_path.open('wb') as stream:
        stream.write(header)
        stream.write(cloud.tobytes())
    if dst_dir.parent.parent.exists():
        try:
            shutil.copy2(ply_path, dst_dir.parent.parent / 'points3D.ply')
        except OSError:
            pass

    print(f'[+] Metric COLMAP dataset for 3DGS generated successfully at: {dst_dir}')
    print(f'    Cameras:  {len(cameras)}')
    print(f'    Images:   {len(images)}')
    print(f'    Points3D: {len(points):,} metric tie-points')
    return dst_dir


def sync_via_gyro_cross_correlation(insv_path, bag_path, trj_path):
    """Calcula automaticamente com precisão de milissegundos o offset temporal Δt via correlação de giro IMU"""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        print("[!] rosbags not installed for direct IMU reading.")
        return None

    HEADER_SIZE = 72
    try:
        with open(insv_path, "rb") as fin:
            fin.seek(-HEADER_SIZE, 2)
            header = fin.read(HEADER_SIZE)
            extra_size = struct.unpack('<I', header[32:36])[0]
            file_size = fin.seek(0, 2)
            extra_start = file_size - extra_size

            fin.seek(-(HEADER_SIZE + 6 + 250), 2)
            offsets_data = fin.read(250)
            offsets = {}
            for i in range(0, len(offsets_data), 10):
                oid, ofmt, osize, ooff = struct.unpack('<BBII', offsets_data[i:i+10])
                if oid > 0:
                    offsets[oid] = (ofmt, osize, ooff)

            if 3 not in offsets or 4 not in offsets:
                print("[!] Gyro/Exposure records not found in INSV trailer.")
                return None

            _, g_size, g_off = offsets[3]
            fin.seek(extra_start + g_off)
            gyro_bytes = fin.read(g_size)

            _, e_size, e_off = offsets[4]
            fin.seek(extra_start + e_off)
            exp_bytes = fin.read(e_size)

        cam_imu_raw = np.frombuffer(gyro_bytes, dtype=[
            ('t', '<u8'), ('ax', '<u2'), ('ay', '<u2'), ('az', '<u2'),
            ('gx', '<u2'), ('gy', '<u2'), ('gz', '<u2')
        ])
        cam_exp_raw = np.frombuffer(exp_bytes, dtype=[('t', '<u8'), ('exp', '<f8')])

        t_exp0_us = cam_exp_raw['t'][0]
        t_cam_s = (cam_imu_raw['t'].astype(np.float64) - t_exp0_us) / 1e6
        cam_gx = cam_imu_raw['gx'].astype(np.float64) - 32768.0
        cam_gy = cam_imu_raw['gy'].astype(np.float64) - 32768.0
        cam_gz = cam_imu_raw['gz'].astype(np.float64) - 32768.0
        cam_gyro_norm = np.sqrt(cam_gx**2 + cam_gy**2 + cam_gz**2)

        lidar_t = []
        lidar_wx = []
        lidar_wy = []
        lidar_wz = []

        with AnyReader([Path(bag_path)]) as reader:
            for conn, ts, raw in reader.messages():
                if 'imu' in conn.topic.lower():
                    msg = reader.deserialize(raw, conn.msgtype)
                    t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    lidar_t.append(t_sec)
                    lidar_wx.append(msg.angular_velocity.x)
                    lidar_wy.append(msg.angular_velocity.y)
                    lidar_wz.append(msg.angular_velocity.z)

        if len(lidar_t) == 0:
            print("[!] No IMU messages found in the .bag file.")
            return None

        lidar_t = np.array(lidar_t)
        lidar_gyro_norm = np.sqrt(np.array(lidar_wx)**2 + np.array(lidar_wy)**2 + np.array(lidar_wz)**2)

        trj = np.loadtxt(trj_path)
        t_slam_start = trj[0, 0]
        t_lidar_rel_slam = lidar_t - t_slam_start

        fs = 200.0
        t_eval = np.arange(1.0, min(90.0, t_cam_s[-1] - 1.0), 1.0 / fs)
        cam_norm_interp = np.interp(t_eval, t_cam_s, cam_gyro_norm)
        cam_norm_interp -= np.median(cam_norm_interp)

        shifts = np.arange(-5.0, 15.0, 0.01)
        corrs = []
        for shift in shifts:
            t_q = t_eval - shift
            valid = (t_q >= t_lidar_rel_slam[0]) & (t_q <= t_lidar_rel_slam[-1])
            if np.sum(valid) > 500:
                lid_interp = np.interp(t_q[valid], t_lidar_rel_slam, lidar_gyro_norm)
                lid_interp -= np.median(lid_interp)
                c = np.corrcoef(cam_norm_interp[valid], lid_interp)[0, 1]
                corrs.append(c)
            else:
                corrs.append(0.0)

        best_shift = shifts[np.argmax(corrs)]
        fine_shifts = np.arange(best_shift - 0.1, best_shift + 0.1, 0.0005)
        fine_corrs = []
        for shift in fine_shifts:
            t_q = t_eval - shift
            valid = (t_q >= t_lidar_rel_slam[0]) & (t_q <= t_lidar_rel_slam[-1])
            lid_interp = np.interp(t_q[valid], t_lidar_rel_slam, lidar_gyro_norm)
            lid_interp -= np.median(lid_interp)
            c = np.corrcoef(cam_norm_interp[valid], lid_interp)[0, 1]
            fine_corrs.append(c)

        best_exact = fine_shifts[np.argmax(fine_corrs)]
        max_corr = np.max(fine_corrs)
        print(f"  [+] Optimal IMU gyro synchronization: Δt = {best_exact:.4f}s (Pearson r = {max_corr:.4f})")
        return float(best_exact)
    except Exception as e:
        print(f"[!] Error computing IMU synchronization: {e}")
        return None


def recalibrate_from_sfm(dataset_dir, fps=2.0, alignment=None, write_outputs=True,
                         trajectory_path=None):
    """Recalibrate with high precision os parâmetros de montagem T_LC0 e T_LC1 usando as poses do SfM/Spirula"""
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_trj = trajectory_path or get_slam_trajectory_path(dataset_dir)
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    al = alignment if alignment is not None else current_alignment(dataset_dir, fps)
    s_sim = al["scale"]
    R_sim = np.array(al["R"])
    t_sim = np.array(al["t"])
    dt_sync = al.get("dt_sync_seconds", 0.0)

    t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
    slerp = Slerp(t_slam, rot_slam)
    images = load_colmap_images(sparse_dir / "images.bin")

    def get_fn(name):
        return int("".join(filter(str.isdigit, Path(name).stem)))

    def solve_lens(cam_prefix):
        cam_imgs = [im for im in images if im["name"].startswith(cam_prefix)]
        if 'accepted_image_names' in al:
            accepted = set(al['accepted_image_names'])
            cam_imgs = [im for im in cam_imgs if im['name'] in accepted]
        cam_imgs.sort(key=lambda x: get_fn(x["name"]))
        R_list, t_list = [], []
        for im in cam_imgs:
            fn = get_fn(im["name"])
            t_vid = frame_time(dataset_dir, im["name"], fps)
            t_q = t_slam[0] + (t_vid - dt_sync)
            if t_slam[0] <= t_q <= t_slam[-1]:
                R_L = slerp(t_q).as_matrix()
                idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]

                C_lidar = s_sim * (R_sim @ im["C"]) + t_sim
                R_cw_lidar = im["R_cw"] @ R_sim.T
                R_wc_sfm = R_cw_lidar.T

                R_LC = R_L.T @ R_wc_sfm
                t_LC = R_L.T @ (C_lidar - p_L)
                R_list.append(R_LC)
                t_list.append(t_LC)
        mean_rot = Rot.from_matrix(R_list).mean()
        median_t = np.median(t_list, axis=0)
        return mean_rot, median_t

    R0, t0 = solve_lens("cam0/")
    R1, t1 = solve_lens("cam1/")
    if al.get('frame_time_source') == 'insv_timelapse':
        if any(not .08 <= np.linalg.norm(t) <= .30 for t in (t0, t1)) or np.linalg.norm(t0-t1) > .12:
            raise ValueError('Timelapse rig calibration rejected: implausible camera lever arm')
        if abs(np.degrees((R0.inv()*R1).magnitude())-180.) > 5.:
            raise ValueError('Timelapse rig calibration rejected: inconsistent front/rear rotations')

    T_LC0 = np.eye(4)
    T_LC0[:3, :3] = R0.as_matrix()
    T_LC0[:3, 3] = t0

    T_LC1 = np.eye(4)
    T_LC1[:3, :3] = R1.as_matrix()
    T_LC1[:3, 3] = t1

    rel_ang = np.linalg.norm((R0.inv() * R1).as_rotvec(degrees=True))
    e0 = R0.as_euler("xyz", degrees=True)
    e1 = R1.as_euler("xyz", degrees=True)

    print("\n" + "=" * 80)
    print(" PHYSICAL RIG RE-CALIBRATION FROM SFM/SPIRULA COMPLETE:")
    print(f" Cam0: Euler XYZ={e0} | Lever arm={t0*100} cm (norm {np.linalg.norm(t0)*100:.2f} cm)")
    print(f" Cam1: Euler XYZ={e1} | Lever arm={t1*100} cm (norm {np.linalg.norm(t1)*100:.2f} cm)")
    print(f" Relative angle between lenses: {rel_ang:.2f}° (Theoretical: 180°)")
    print("=" * 80)

    calib_dict = {
        "calibration_version": CALIBRATION_VERSION,
        "timelapse_calibration_version": al.get('timelapse_calibration_version'),
        "scanner": "3DMakerPro_Raven_LiDAR",
        "camera": "Insta360_X4_DualFisheye",
        "lens_model": "THIN_PRISM_FISHEYE",
        "T_lidar_to_cam0_rigid_4x4": T_LC0.tolist(),
        "T_lidar_to_cam1_rigid_4x4": T_LC1.tolist(),
        "lever_arm_cam0_meters": {
            "dx": float(t0[0]), "dy": float(t0[1]), "dz": float(t0[2]), "norm_distance_cm": float(np.linalg.norm(t0) * 100)
        },
        "lever_arm_cam1_meters": {
            "dx": float(t1[0]), "dy": float(t1[1]), "dz": float(t1[2]), "norm_distance_cm": float(np.linalg.norm(t1) * 100)
        },
        "rotation_euler_xyz_deg_cam0": {
            "roll": float(e0[0]), "pitch": float(e0[1]), "yaw": float(e0[2])
        },
        "rotation_euler_xyz_deg_cam1": {
            "roll": float(e1[0]), "pitch": float(e1[1]), "yaw": float(e1[2])
        },
        "relative_lens_angle_deg": float(rel_ang),
        "dt_sync_seconds": float(dt_sync)
    }

    # Extrinsics and intrinsics must describe the SAME reconstruction.
    cameras = load_colmap_cameras(sparse_dir / "cameras.bin")
    keys = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')
    for prefix, field in (("cam0/", "cam0_front_intrinsics"),
                          ("cam1/", "cam1_rear_intrinsics")):
        ids = {im['cam_id'] for im in images if im['name'].startswith(prefix)}
        if len(ids) != 1:
            raise ValueError(f"Expected one calibrated camera for {prefix}, got {ids}")
        camera = cameras[ids.pop()]
        if camera['model_id'] != 10:
            raise ValueError(f'Expected THIN_PRISM_FISHEYE for {prefix}')
        calib_dict[field] = dict(zip(keys, camera['params']))
        calib_dict[field].update(width=camera['width'], height=camera['height'])
    calib_dict['transform_convention'] = 'camera_to_trajectory_body'

    if write_outputs:
        deliv_dir = dataset_dir / "deliverables"
        deliv_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(dataset_dir / "rig_calibration.json", "w", encoding="utf-8") as f:
            json.dump(calib_dict, f, indent=2)
        with open(dataset_dir / "calibracao_rigida_auto.json", "w", encoding="utf-8") as f:
            json.dump(calib_dict, f, indent=2)
        with open(deliv_dir / f"rig_calibration_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(calib_dict, f, indent=2)
        print(f"  [+] Dynamic calibration saved to: {dataset_dir / 'rig_calibration.json'}")
    return calib_dict


def colorize_via_spirula_sfm(dataset_dir, fps=2.0, use_vulkan=True, masks_dir=None, operator_radius=0.0,
                           photometric='off', photometric_params=None):
    """Executa a coloração de alta precisão projetando as poses alinhadas do SfM"""
    if operator_radius > 0 and masks_dir is None:
        raise ValueError('Operator removal requires person masks (--mask-persons or --masks-dir); a trajectory radius alone can erase doors and walls')
    from raven_app.photometric import prepare, apply_rgb
    photo_views = prepare(dataset_dir, photometric, photometric_params, masks_dir)
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_pcd = active_slam_dir(dataset_dir) / "pcd" / "all_raw_points.pcd"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    al = current_alignment(dataset_dir, fps)
    s_sim = al["scale"]
    R_sim = np.array(al["R"])
    t_sim = np.array(al["t"])

    cameras = load_colmap_cameras(sparse_dir / "cameras.bin")
    images = load_colmap_images(sparse_dir / "images.bin")
    pose_overrides = {}
    if al.get('frame_time_source') == 'insv_timelapse':
        from raven_app.timelapse_calibration import rejected_pose_overrides
        pose_overrides = rejected_pose_overrides(dataset_dir, al, images)
    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    operator_views = []
    gpu_views = [] if use_vulkan and get_vulkan_bin().is_file() else None
    K_VIEWS = 3
    top_scores = np.zeros((n_pts if gpu_views is None else 0, K_VIEWS), dtype=np.float32)
    top_colors = np.zeros((n_pts if gpu_views is None else 0, K_VIEWS, 3), dtype=np.uint8)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    print("\n" + "=" * 80)
    print(" PROJECTING SFM FRAMES ONTO LIDAR POINT CLOUD (TOP-3 CONSENSUS)...")
    print("=" * 80)

    t0 = time.time()
    images.sort(key=lambda x: x["name"])

    for count, im in enumerate(images, 1):
        img_path = dataset_dir / "images" / im["name"]
        if not img_path.exists():
            continue

        camera = cameras[im["cam_id"]]
        params = _colorization_camera_params(camera)

        # Posição e orientação da câmera no referencial LiDAR
        C_lidar = s_sim * (R_sim @ im["C"]) + t_sim
        R_cw_lidar = im["R_cw"] @ R_sim.T
        if im['name'] in pose_overrides:
            if pose_overrides[im['name']] is None:
                continue
            R_cw_lidar, C_lidar = pose_overrides[im['name']]

        if operator_radius and masks_dir is not None:
            operator_views.append((img_path, R_cw_lidar, C_lidar, params))
        if gpu_views is not None:
            gpu_views.append((img_path, R_cw_lidar, C_lidar, params))
            continue

        P_cam = (R_cw_lidar @ (pts_lidar - C_lidar).T).T
        z_mask = P_cam[:, 2] > 0.15
        idx_valid = np.where(z_mask)[0]
        if len(idx_valid) == 0:
            continue

        P_v = P_cam[idx_valid]
        d = np.linalg.norm(P_v, axis=1)

        cx, cy = params[2], params[3]
        u, v, r = project_thin_prism(P_v, params)

        r_px = np.hypot(u - cx, v - cy)
        radius = (1620.0 * min(camera['width'], camera['height']) / 3840.0
                  if camera['model_id'] in (5, 10) else
                  np.hypot(camera['width'], camera['height']) / 2)
        mask_circle = ((r_px < radius) & (u >= 0.0) & (u < camera['width'] - 1)
                       & (v >= 0.0) & (v < camera['height'] - 1))
        if not np.any(mask_circle):
            continue

        u_c = u[mask_circle]
        v_c = v[mask_circle]
        d_c = d[mask_circle]
        r_c = r_px[mask_circle]
        idx_c = idx_valid[mask_circle]

        # Z-Buffer raster de oclusão
        ug = np.clip(np.floor(u_c * zbuf_w / camera['width']).astype(np.int32), 0, zbuf_w - 1)
        vg = np.clip(np.floor(v_c * zbuf_h / camera['height']).astype(np.int32), 0, zbuf_h - 1)
        flat_idx = vg * zbuf_w + ug

        vis_mask = visible_depths(d_c, flat_idx, zbuf_w, zbuf_h)
        if masks_dir is not None:
            keep_mask = load_keep_mask(img_path, dataset_dir / "images", masks_dir)
            source = cv2.imread(str(img_path))
            if source is None or keep_mask.shape != source.shape[:2]:
                raise ValueError(f"Person mask dimensions do not match {img_path}")
            vis_mask &= keep_samples(keep_mask, u_c, v_c)
        if not np.any(vis_mask):
            continue

        u_vis = u_c[vis_mask]
        v_vis = v_c[vis_mask]
        d_vis = d_c[vis_mask]
        r_vis = r_c[vis_mask]
        idx_vis = idx_c[vis_mask]

        scores = (1.0 - 0.5 * (r_vis / radius)**2) / (np.maximum(d_vis, 0.5)**1.5)

        min_top = np.min(top_scores[idx_vis], axis=1)
        can_insert = scores > min_top
        if not np.any(can_insert):
            continue

        idx_ins = idx_vis[can_insert]
        scores_ins = scores[can_insert]
        u_ins = u_vis[can_insert]
        v_ins = v_vis[can_insert]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb_ins = sample_bilinear(img_rgb, u_ins, v_ins)
        if photo_views is not None:
            rgb_ins = apply_rgb(rgb_ins, u_ins, v_ins, params,
                                photo_views.get(im['name'].replace('\\', '/')))

        worst_slot = np.argmin(top_scores[idx_ins], axis=1)
        top_scores[idx_ins, worst_slot] = scores_ins
        top_colors[idx_ins, worst_slot] = rgb_ins

        if count % 50 == 0 or count == len(images):
            cov = np.count_nonzero(np.any(top_scores > 0, axis=1)) / n_pts * 100
            print(f"    [{count:3d}/{len(images)}] Projected frames | LiDAR coverage: {cov:.1f}%")

    print("[*] Solving statistical consensus SfM...")
    if gpu_views is not None:
        colors = colorize_views(pts_lidar, gpu_views, dataset_dir, masks_dir=masks_dir, images_dir=dataset_dir / "images",
                                **({'photometric': photo_views} if photo_views is not None else {}))
        if colors is None:
            return colorize_via_spirula_sfm(dataset_dir, fps=fps, use_vulkan=False, masks_dir=masks_dir,
                                           operator_radius=operator_radius, photometric=photometric,
                                           photometric_params=photometric_params)
    else:
        colors = np.full((n_pts, 3), 180, dtype=np.uint8)
        n_obs = np.count_nonzero(top_scores > 0, axis=1)

        mask_1 = (n_obs == 1)
        if np.any(mask_1):
            idx_1 = np.where(mask_1)[0]
            best_slot = np.argmax(top_scores[idx_1], axis=1)
            colors[idx_1] = top_colors[idx_1, best_slot]

        mask_2 = (n_obs == 2)
        if np.any(mask_2):
            idx_2 = np.where(mask_2)[0]
            s2 = top_scores[idx_2]
            c2 = top_colors[idx_2].astype(np.float32)
            weights = s2 / np.sum(s2, axis=1, keepdims=True)
            avg2 = np.sum(c2 * weights[:, :, None], axis=1)
            # Two contradictory views cannot establish a consensus. Keep an
            # observed color rather than manufacturing a translucent blend.
            slots = np.argsort(-s2, axis=1)[:, :2]
            observed = np.take_along_axis(c2, slots[:, :, None], axis=1)
            disagree = np.linalg.norm(observed[:, 0] - observed[:, 1], axis=1) >= 45.0
            avg2[disagree] = observed[disagree, 0]
            colors[idx_2] = np.clip(np.round(avg2), 0, 255).astype(np.uint8)

        mask_3 = (n_obs == 3)
        if np.any(mask_3):
            idx_3 = np.where(mask_3)[0]
            c3 = top_colors[idx_3].astype(np.float32)
            s3 = top_scores[idx_3]
            med3 = np.median(c3, axis=1, keepdims=True)
            diff_from_med = np.linalg.norm(c3 - med3, axis=2)
            inliers = diff_from_med < 45.0
            weights = s3 * inliers.astype(np.float32)
            sum_w = np.sum(weights, axis=1, keepdims=True)
            fallback = (sum_w[:, 0] == 0)
            weights[fallback] = 0.0
            best = np.argmax(s3[fallback], axis=1)
            weights[np.where(fallback)[0], best] = 1.0
            sum_w[fallback] = 1.0
            norm_w = weights / sum_w
            final_c3 = np.sum(c3 * norm_w[:, :, None], axis=1)
            colors[idx_3] = np.clip(np.round(final_c3), 0, 255).astype(np.uint8)

    deliv_dir = dataset_dir / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_ply = deliv_dir / f"lidar_colored_sfm_consensus_{ts}.ply"
    out_pcd = deliv_dir / f"lidar_colored_sfm_consensus_{ts}.pcd"
    if operator_radius and masks_dir is not None:
        keep = masked_operator_keep(pts_lidar, operator_views, dataset_dir / "images", masks_dir, operator_radius, project_thin_prism)
        pts_lidar, colors = pts_lidar[keep], colors[keep]
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    print(f"[+] Deliverables saved to: {deliv_dir}")
    print(f"    - {out_ply.name}")
    print(f"    - {out_pcd.name}")
    print(f"[+] Completed in {time.time() - t0:.1f}s!")
    return out_ply, out_pcd


# ==============================================================================
# PIPELINE MÉTODO 2: DIRETO RÍGIDO (SEM SFM)
# ==============================================================================

def _prepare_direct_photometric(dataset_dir, mode, params, masks_dir):
    """Use photometric fitting only when this dataset has the required SfM tracks."""
    if mode == 'off':
        return None
    dataset_dir = Path(dataset_dir)
    sparse = dataset_dir / 'sparse' / '0'
    has_sfm_tracks = all((sparse / name).is_file()
                         for name in ('cameras.bin', 'images.bin', 'points3D.bin'))
    if mode not in ('loglinear', 'ppisp-bilateral'):
        raise ValueError('Unsupported photometric mode')
    if not has_sfm_tracks:
        print(f'[!] {mode} needs COLMAP SfM tracks, which direct/trajectory mode does not create. '
              'Skipping photometric correction and continuing; select Reconstruction to fit it.')
        return None
    from raven_app.photometric import prepare
    return prepare(dataset_dir, mode, params, masks_dir)

def colorize_via_direct_rigid(dataset_dir, calib_json_path=None, fps=2.0, dt_override=None, use_vulkan=True, masks_dir=None, operator_radius=0.0,
                            photometric='off', photometric_params=None, third_camera=None):
    """Executa a coloração direta rápida usando matriz rígida e tempo calibrado"""
    from raven_app.photometric import apply_rgb
    photo_views = _prepare_direct_photometric(
        dataset_dir, photometric, photometric_params, masks_dir)
    if operator_radius > 0 and masks_dir is None:
        raise ValueError('Operator removal requires person masks (--mask-persons or --masks-dir); a trajectory radius alone can erase doors and walls')
    if calib_json_path is None:
        rig_json = dataset_dir / "rig_calibration.json"
        auto_json = dataset_dir / "calibracao_rigida_auto.json"
        azure_json = dataset_dir / "calibracao_rigida_azure_dataset.json"
        recalib_json = dataset_dir / "calibracao_rigida_recalibrada.json"
        if rig_json.exists():
            calib_json_path = rig_json
        elif auto_json.exists():
            calib_json_path = auto_json
        elif recalib_json.exists():
            calib_json_path = recalib_json
        elif azure_json.exists():
            calib_json_path = azure_json
        else:
            candidates = list(dataset_dir.glob("*calibrat*.json")) + list(dataset_dir.glob("calibracao_rigida*.json"))
            calib_json_path = candidates[0] if candidates else DEFAULT_CALIB_JSON

    print("=" * 80)
    print(" DIRECT RIGID COLORIZATION (WITHOUT SFM)")
    print(f" Calibration: {calib_json_path.name}")
    print("=" * 80)

    with open(calib_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    active_alignment = {}
    if (dataset_dir / 'sparse/0/images.bin').is_file():
        active_alignment = current_alignment(dataset_dir, fps)
        if (calib_json_path.name in ('rig_calibration.json', 'calibracao_rigida_auto.json')
                and (cfg.get('calibration_version') != CALIBRATION_VERSION
                     or cfg.get('timelapse_calibration_version') != active_alignment.get('timelapse_calibration_version'))):
            cfg = recalibrate_from_sfm(dataset_dir, fps)

    p0 = cfg["cam0_front_intrinsics"]
    params0 = (p0["fx"], p0["fy"], p0["cx"], p0["cy"], p0["k1"], p0["k2"], p0["p1"], p0["p2"], p0["k3"], p0["k4"], p0["sx1"], p0["sy1"])
    p1 = cfg["cam1_rear_intrinsics"]
    params1 = (p1["fx"], p1["fy"], p1["cx"], p1["cy"], p1["k1"], p1["k2"], p1["p1"], p1["p2"], p1["k3"], p1["k4"], p1["sx1"], p1["sy1"])

    T_LC0 = np.array(cfg["T_lidar_to_cam0_rigid_4x4"])
    R_LC0 = T_LC0[:3, :3]
    t_LC0 = T_LC0[:3, 3]

    T_LC1 = np.array(cfg["T_lidar_to_cam1_rigid_4x4"])
    R_LC1 = T_LC1[:3, :3]
    t_LC1 = T_LC1[:3, 3]

    slam_pcd = active_slam_dir(dataset_dir) / "pcd" / "all_raw_points.pcd"
    slam_trj = get_slam_trajectory_path(dataset_dir)

    # Determinar sincronização temporal
    dt_sync = dt_override
    if dt_sync is None and "dt_sync_seconds" in cfg:
        dt_sync = cfg["dt_sync_seconds"]

    align_json = dataset_dir / "colmap_to_lidar_alignment.json"
    if dt_sync is None and align_json.exists():
        try:
            with open(align_json, "r", encoding="utf-8") as f:
                al = json.load(f)
                dt_sync = al.get("dt_sync_seconds", None)
                if dt_sync is not None:
                    print(f"[*] Synchronization inherited from COLMAP Sim(3): Δt = {dt_sync:.4f}s")
        except Exception:
            pass

    if dt_sync is None:
        insv_files = list(dataset_dir.glob("*.insv"))
        bag_files = list(dataset_dir.glob("*.bag"))
        if insv_files and bag_files:
            print("[*] Synchronizing via Gyro Cross-Correlation...")
            dt_sync = sync_via_gyro_cross_correlation(insv_files[0], bag_files[0], slam_trj)

    if dt_sync is None:
        raise ValueError("No capture-specific time synchronization. Supply --dt or calibrate this dataset first.")
    else:
        print(f"[*] Active time synchronization: Δt = {dt_sync:.4f}s")

    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
    t_slam_start = t_slam[0]
    t_slam_end = t_slam[-1]
    slerp = Slerp(t_slam, rot_slam)

    front = {p.name: p for p in (dataset_dir / "images" / "cam0").glob("*.jpg")}
    rear = {p.name: p for p in (dataset_dir / "images" / "cam1").glob("*.jpg")}
    pairs = sorted(front.keys() & rear.keys(),
                   key=lambda name: frame_time(dataset_dir, "cam0/" + name, fps))
    if not pairs:
        raise ValueError("No synchronized front/rear image pairs found")
    if front.keys() != rear.keys():
        print(f"[!] Skipping {len(front.keys() ^ rear.keys())} unpaired lens images")
    cam0_files = [front[name] for name in pairs]
    cam1_files = [rear[name] for name in pairs]

    operator_views = []
    gpu_views = [] if use_vulkan and get_vulkan_bin().is_file() else None
    K_VIEWS = 3
    top_scores = np.zeros((n_pts if gpu_views is None else 0, K_VIEWS), dtype=np.float32)
    top_colors = np.zeros((n_pts if gpu_views is None else 0, K_VIEWS, 3), dtype=np.uint8)

    zbuf_w, zbuf_h = 960, 960

    t0 = time.time()
    num_frames = min(len(cam0_files), len(cam1_files))

    print("\n" + "=" * 80)
    print(" DIRECT RIGID COLORIZATION (TOP-3 MULTI-VIEW CONSENSUS)")
    print("=" * 80)

    # Se alinhamento SfM existir, calcular correção suave de drift da trajetória (Amostragem de Keyframes)
    use_drift_correction = False
    sparse_img_bin = dataset_dir / "sparse" / "0" / "images.bin"
    is_timelapse = active_alignment.get('frame_time_source') == 'insv_timelapse'
    if align_json.exists() and sparse_img_bin.exists() and not is_timelapse:
        try:
            with open(align_json, "r", encoding="utf-8") as f:
                al = json.load(f)
            s_sim = al["scale"]
            R_sim = np.array(al["R"])
            t_sim = np.array(al["t"])
            sfm_imgs = load_colmap_images(sparse_img_bin)
            cam0_sfm = [im for im in sfm_imgs if im["name"].startswith("cam0")]
            cam0_sfm.sort(key=lambda x: int("".join(filter(str.isdigit, x["name"]))))

            n_samples = len(cam0_sfm)
            sample_indices = np.linspace(0, len(cam0_sfm) - 1, n_samples).astype(int)
            sample_imgs = [cam0_sfm[i] for i in sample_indices]

            sample_times = []
            delta_p_list = []
            delta_R_list = []

            for im in sample_imgs:
                fn = int("".join(filter(str.isdigit, im["name"])))
                t_vid = frame_time(dataset_dir, im["name"], fps)
                t_q = t_slam_start + (t_vid - dt_sync)
                if not (t_slam_start <= t_q <= t_slam_end):
                    continue

                C_sfm = s_sim * (R_sim @ im["C"]) + t_sim
                R_wc_sfm = (im["R_cw"] @ R_sim.T).T

                R_L_sfm = R_wc_sfm @ R_LC0.T
                p_L_sfm = C_sfm - R_L_sfm @ t_LC0

                idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L_slam = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                R_L_slam = slerp(t_q).as_matrix()

                delta_p_list.append(p_L_sfm - p_L_slam)
                delta_R_list.append(R_L_sfm @ R_L_slam.T)
                sample_times.append(t_q)

            if len(sample_times) >= 2:
                sample_times = np.array(sample_times)
                delta_p_arr = np.array(delta_p_list)
                slerp_delta_R = Slerp(sample_times, Rot.from_matrix(delta_R_list))
                use_drift_correction = True
                print(f"[*] Trajectory drift correction enabled with {len(sample_times)} SfM keyframes!")
        except Exception as e:
            print(f"[!] Warning: Could not load SfM drift correction: {e}")

    events = []
    for k in range(num_frames):
        t_vid = frame_time(dataset_dir, "cam0/" + cam0_files[k].name, fps)
        events.extend((
            (t_vid, 0.0, "cam0", cam0_files[k], params0, R_LC0, t_LC0, "THIN_PRISM_FISHEYE", None),
            (t_vid, 0.0, "cam1", cam1_files[k], params1, R_LC1, t_LC1, "THIN_PRISM_FISHEYE", None),
        ))

    auxiliary_cameras = (third_camera if isinstance(third_camera, (list, tuple))
                         else [third_camera] if third_camera else [])
    for camera in auxiliary_cameras:
        if not getattr(camera, 'enabled', True):
            continue
        report = getattr(camera, 'calibration_report', {}) or {}
        intrinsic_status = report.get('intrinsics', {}).get('status')
        extrinsic_status = report.get('extrinsics', {}).get('status')
        if (extrinsic_status == 'calibrated'
                and intrinsic_status in ('sfm_calibrated', 'calibrated', 'verified')):
            model = camera.camera_model.upper()
            params = _auxiliary_colorization_camera_params(camera)
            transform = np.asarray(camera.T_lidar_to_cam2_rigid_4x4, dtype=np.float64)
            camera_dir = dataset_dir / 'images' / camera.camera_name
            added = 0
            for image_path in sorted(camera_dir.glob('*.jpg')) if camera_dir.is_dir() else ():
                try:
                    capture_time = frame_time(
                        dataset_dir, f'{camera.camera_name}/{image_path.name}', camera.fps or fps)
                except (KeyError, OSError, ValueError) as exc:
                    print(f"[!] {camera.camera_name} frame has no capture time ({image_path.name}): {exc}")
                    continue
                events.append((capture_time, float(camera.time_offset_s), camera.camera_name,
                               image_path, params, transform[:3, :3], transform[:3, 3], model, None))
                added += 1
            print(f"[*] Direct colorization calibrated views: {camera.camera_name} "
                  f"({added} frames, {model})")
            continue

        from raven_app.auxiliary_pose_support import load_camera_pose_support
        support = load_camera_pose_support(
            dataset_dir, camera.camera_name, camera_model=camera.camera_model,
            intrinsics=camera.intrinsics,
            trajectory_path=slam_trj)
        if support is None:
            reason = (report.get('extrinsics', {}).get('reason')
                      or report.get('intrinsics', {}).get('reason')
                      or 'No fresh, accepted per-image SfM pose support is available')
            print(f"[!] Direct colorization skipping {camera.camera_name}: {reason}")
            continue
        support_camera = type('AuxiliaryPoseSupportCamera', (), {})()
        support_camera.camera_name = camera.camera_name
        support_camera.camera_model = support['camera_model']
        support_camera.intrinsics = support['intrinsics']
        model = support_camera.camera_model.upper()
        params = _auxiliary_colorization_camera_params(support_camera)
        added = 0
        for pose in support['poses']:
            events.append((float(pose['video_time']), 0.0, camera.camera_name,
                           pose['image_path'], params, None, None, model,
                           (pose['R_cw'], pose['C'])))
            added += 1
        print(f"[*] Direct colorization registered sample-SfM views: {camera.camera_name} "
              f"({added} exact poses, {model}; rigid clock/baseline remain uncalibrated)")

    events.sort(key=lambda event: (event[0], event[2], event[3].name))
    for event_index, (video_time, camera_offset, cam_name, img_path, params,
                      R_LC, t_LC, _model, pose_override) in enumerate(events):
        if pose_override is not None:
            R_cw, p_cam = pose_override
        else:
            t_query = t_slam_start + (video_time - dt_sync - camera_offset)
            if not (t_slam_start <= t_query <= t_slam_end):
                continue
            idx = np.clip(np.searchsorted(t_slam, t_query), 1, len(t_slam) - 1)
            w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_query).as_matrix()
            if use_drift_correction and sample_times[0] <= t_query <= sample_times[-1]:
                dp = np.array([np.interp(t_query, sample_times, delta_p_arr[:, ax]) for ax in range(3)])
                p_L += dp
                R_L = slerp_delta_R(t_query).as_matrix() @ R_L
            p_cam = p_L + R_L @ t_LC
            R_cw = (R_L @ R_LC).T
        if operator_radius and masks_dir is not None:
            operator_views.append((img_path, R_cw, p_cam, params))
        if gpu_views is not None:
            gpu_views.append((img_path, R_cw, p_cam, params))
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        height, width = img_bgr.shape[:2]
        camera_radius = (0.5 * np.hypot(width, height) if params[11] < -997.0
                         else 1620.0 * min(width, height) / 3840.0)
        P_cam = (R_cw @ (pts_lidar - p_cam).T).T
        idx_valid = np.flatnonzero(P_cam[:, 2] > 0.15)
        if not len(idx_valid):
            continue
        P_v = P_cam[idx_valid]
        distances = np.linalg.norm(P_v, axis=1)
        u, v, _ = project_thin_prism(P_v, params)
        r_px = np.hypot(u - params[2], v - params[3])
        valid = ((r_px < camera_radius) & (u >= 0) & (u < width - 1)
                 & (v >= 0) & (v < height - 1))
        if not np.any(valid):
            continue
        u, v, distances = u[valid], v[valid], distances[valid]
        r_px, point_indices = r_px[valid], idx_valid[valid]
        scale_x, scale_y = width / zbuf_w, height / zbuf_h
        ug = np.clip(np.floor(u / scale_x).astype(np.int32), 0, zbuf_w - 1)
        vg = np.clip(np.floor(v / scale_y).astype(np.int32), 0, zbuf_h - 1)
        visible = visible_depths(distances, vg * zbuf_w + ug, zbuf_w, zbuf_h)
        if masks_dir is not None:
            keep_mask = load_keep_mask(img_path, dataset_dir / 'images', masks_dir)
            if keep_mask.shape != img_bgr.shape[:2]:
                raise ValueError(f'Person mask dimensions do not match {img_path}')
            visible &= keep_samples(keep_mask, u, v)
        if not np.any(visible):
            continue
        u_vis, v_vis = u[visible], v[visible]
        scores = (1.0 - 0.5 * (r_px[visible] / camera_radius) ** 2) / np.maximum(distances[visible], 0.5) ** 1.5
        idx_vis = point_indices[visible]
        can_insert = scores > np.min(top_scores[idx_vis], axis=1)
        if not np.any(can_insert):
            continue
        idx_insert = idx_vis[can_insert]
        colors_insert = sample_bilinear(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), u_vis[can_insert], v_vis[can_insert])
        if photo_views is not None:
            photo_key = img_path.relative_to(dataset_dir / 'images').as_posix()
            colors_insert = apply_rgb(
                colors_insert, u_vis[can_insert], v_vis[can_insert], params,
                photo_views.get(photo_key))
        score_insert = scores[can_insert]
        worst_slot = np.argmin(top_scores[idx_insert], axis=1)
        top_scores[idx_insert, worst_slot] = score_insert
        top_colors[idx_insert, worst_slot] = colors_insert

        if gpu_views is None and (event_index + 1) % 50 == 0:
            coverage = np.count_nonzero(np.any(top_scores > 0, axis=1)) / n_pts * 100
            print(f"    [{event_index + 1:3d}/{len(events)}] Accumulated camera views | Coverage: {coverage:.1f}%")

    print("[*] Solving statistical consensus and removing projection outliers...")
    if gpu_views is not None:
        colors = colorize_views(pts_lidar, gpu_views, dataset_dir, masks_dir=masks_dir, images_dir=dataset_dir / "images",
                                **({'photometric': photo_views} if photo_views is not None else {}))
        if colors is None:
            return colorize_via_direct_rigid(dataset_dir, calib_json_path, fps=fps, dt_override=dt_sync, use_vulkan=False,
                                             masks_dir=masks_dir, operator_radius=operator_radius,
                                             photometric=photometric, photometric_params=photometric_params,
                                             third_camera=third_camera)
    else:
        colors = np.full((n_pts, 3), 180, dtype=np.uint8)
        n_obs = np.count_nonzero(top_scores > 0, axis=1)

        mask_1 = (n_obs == 1)
        if np.any(mask_1):
            idx_1 = np.where(mask_1)[0]
            best_slot = np.argmax(top_scores[idx_1], axis=1)
            colors[idx_1] = top_colors[idx_1, best_slot]

        mask_2 = (n_obs == 2)
        if np.any(mask_2):
            idx_2 = np.where(mask_2)[0]
            s2 = top_scores[idx_2]
            c2 = top_colors[idx_2].astype(np.float32)
            weights = s2 / np.sum(s2, axis=1, keepdims=True)
            avg2 = np.sum(c2 * weights[:, :, None], axis=1)
            # Two contradictory views cannot establish a consensus. Keep an
            # observed color rather than manufacturing a translucent blend.
            slots = np.argsort(-s2, axis=1)[:, :2]
            observed = np.take_along_axis(c2, slots[:, :, None], axis=1)
            disagree = np.linalg.norm(observed[:, 0] - observed[:, 1], axis=1) >= 45.0
            avg2[disagree] = observed[disagree, 0]
            colors[idx_2] = np.clip(np.round(avg2), 0, 255).astype(np.uint8)

        mask_3 = (n_obs == 3)
        if np.any(mask_3):
            idx_3 = np.where(mask_3)[0]
            c3 = top_colors[idx_3].astype(np.float32)
            s3 = top_scores[idx_3]
            med3 = np.median(c3, axis=1, keepdims=True)
            diff_from_med = np.linalg.norm(c3 - med3, axis=2)
            inliers = diff_from_med < 45.0
            weights = s3 * inliers.astype(np.float32)
            sum_w = np.sum(weights, axis=1, keepdims=True)
            fallback = (sum_w[:, 0] == 0)
            weights[fallback] = 0.0
            best = np.argmax(s3[fallback], axis=1)
            weights[np.where(fallback)[0], best] = 1.0
            sum_w[fallback] = 1.0
            norm_w = weights / sum_w
            final_c3 = np.sum(c3 * norm_w[:, :, None], axis=1)
            colors[idx_3] = np.clip(np.round(final_c3), 0, 255).astype(np.uint8)

    deliv_dir = dataset_dir / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_ply = deliv_dir / f"lidar_colored_direct_rigid_{ts}.ply"
    out_pcd = deliv_dir / f"lidar_colored_direct_rigid_{ts}.pcd"
    if operator_radius and masks_dir is not None:
        keep = masked_operator_keep(pts_lidar, operator_views, dataset_dir / "images", masks_dir, operator_radius, project_thin_prism)
        pts_lidar, colors = pts_lidar[keep], colors[keep]
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    print(f"[+] Deliverables saved to: {deliv_dir}")
    print(f"    - {out_ply.name}")
    print(f"    - {out_pcd.name}")
    print(f"[+] Direct rigid method completed in {time.time() - t0:.1f}s!")
    return out_ply, out_pcd


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified LiDAR-camera calibration and colorization pipeline for 360-degree data.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset directory.")
    parser.add_argument("--method", type=str, choices=["all", "sfm", "direct", "trajectory", "reconstruction"], default="all",
                        help="Method to run: 'sfm', 'direct', or 'all'.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame sampling rate (FPS, default 2.0).")
    parser.add_argument("--run-spirula", action="store_true", help="Force spirula.exe execution before colorization.")
    parser.add_argument("--recalibrate-from-sfm", action="store_true", help="Recalculate rig parameters from SfM.")
    parser.add_argument("--dt", type=float, default=None, help="Manual time offset Δt in seconds.")
    parser.add_argument("--calib", type=str, default=None, help="Optional calibration JSON path.")

    args = parser.parse_args()
    if args.method == "trajectory":
        args.method = "direct"
    elif args.method == "reconstruction":
        args.method = "sfm"
    dataset_dir = Path(args.dataset)

    print("=" * 80)
    print(" MASTER LIDAR-360 CAMERA COLORIZATION PIPELINE")
    print(f" Dataset: {dataset_dir}")
    print(f" Method:  {args.method.upper()}")
    print(f" FPS:     {args.fps:.1f}")
    print("=" * 80)

    if args.recalibrate_from_sfm:
        recalibrate_from_sfm(dataset_dir, fps=args.fps)

    if args.method in ["sfm", "all"]:
        if args.run_spirula:
            run_spirula_sfm_auto(dataset_dir)
        colorize_via_spirula_sfm(dataset_dir, fps=args.fps)

    if args.method in ["direct", "all"]:
        calib_path = Path(args.calib) if args.calib else None
        colorize_via_direct_rigid(dataset_dir, calib_path, fps=args.fps, dt_override=args.dt)

    print("\n" + "=" * 80)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Colorized files saved in: {dataset_dir}")
    print(f" Deliverables consolidated in: {dataset_dir / 'deliverables'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
