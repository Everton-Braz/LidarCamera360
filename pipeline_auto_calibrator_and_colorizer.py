#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Unificado: Auto-Calibração, Sincronização Temporal e Coloração LiDAR-Câmera 360
========================================================================================
Integração completa:
1. Executa o Spirula Studio SfM (spirula.exe) de forma headless (se solicitado ou se sparse não existir).
2. Realiza o alinhamento de alta precisão Sim(3) + ICP dos tie-points do COLMAP contra a nuvem LiDAR.
3. Auto-sincroniza o tempo exato (Δt) entre o vídeo INSV e o SLAM (0 ms de erro) via SfM ou IMU Gyro.
4. Auto-recalibra e compensa variações mecânicas de aperto do suporte (Yaw/Roll/Pitch).
5. Colore a nuvem com oclusão via Z-Buffer (960x960) e ponderação de nitidez óptica.
6. Suporta os dois métodos:
   - Método A: SfM Spirula Alinhado (Padrão Ouro, 100% autônomo e aprumado).
   - Método B: Direto Rígido (Rápido, universal para qualquer dataset com o mesmo rig fixo).
"""

import os
import sys
import time
import struct
import json
import argparse
import shutil
import subprocess
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE_DIR = Path(__file__).resolve().parent
SPIRULA_EXE = WORKSPACE_DIR / "spirula" / "spirula.exe"
DEFAULT_CALIB_JSON = WORKSPACE_DIR / "calibracao_rigida_raven_insta360.json"


# ==============================================================================
# LEITURA DE DADOS
# ==============================================================================

def load_pcd(path):
    print(f"[*] Carregando nuvem PCD: {path.name}...")
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("POINTS"):
                n = int(line.split()[1])
            elif line.startswith("DATA"):
                break
        raw = f.read(n * 16)
        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        xyz = data[:, :3].astype(np.float64)
    print(f"    Pontos carregados: {len(xyz):,}")
    return xyz


def load_trajectory(path):
    print(f"[*] Carregando trajetória SLAM: {path.name}...")
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    rotations = Rot.from_quat(data[:, 4:8])
    print(f"    Total de poses: {len(data):,}, Duração: {timestamps[-1] - timestamps[0]:.2f}s")
    return timestamps, positions, rotations


def load_colmap_cameras(cameras_bin):
    cameras = {}
    with open(cameras_bin, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cid, model_id, width, height = struct.unpack("<iiQQ", f.read(24))
            params = struct.unpack("<12d", f.read(12 * 8))
            cameras[cid] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras


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
    x = P_v[:, 0] / P_v[:, 2]
    y = P_v[:, 1] / P_v[:, 2]
    r = np.sqrt(x**2 + y**2)
    th = np.arctan(r)
    th2 = th**2
    th4 = th2**2
    th6 = th4 * th2
    th8 = th4**2
    thd = th * (1.0 + k1*th2 + k2*th4 + k3*th6 + k4*th8)

    scale = np.where(r > 1e-8, thd / r, 1.0)
    xd = x * scale
    yd = y * scale
    rd2 = xd**2 + yd**2

    du = 2.0 * p1 * xd * yd + p2 * (rd2 + 2.0 * xd**2) + sx1 * rd2
    dv = p1 * (rd2 + 2.0 * yd**2) + 2.0 * p2 * xd * yd + sy1 * rd2

    u = fx * (xd + du) + cx
    v = fy * (yd + dv) + cy
    return u, v, r


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
    print(f"  [+] PLY salvo: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


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
    print(f"  [+] PCD salvo: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


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
        print(f"[*] Sincronização via giroscópio indisponível ({e}), prosseguindo...")
        return None


# ==============================================================================
# PIPELINE MÉTODO 1: SPIRULA SFM ALINHADO (PADRÃO OURO)
# ==============================================================================

def run_spirula_sfm_auto(dataset_dir, quality="medium"):
    """Executa o Spirula Studio via CLI se a pasta sparse/0 não existir"""
    img_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"

    if sparse_dir.exists() and (sparse_dir / "cameras.bin").exists() and (sparse_dir / "images.bin").exists():
        print(f"[+] Reconstrução SfM existente encontrada em: {sparse_dir}")
        return True

    if not SPIRULA_EXE.exists():
        print(f"[!] spirula.exe não encontrado em: {SPIRULA_EXE}")
        return False

    print("=" * 80)
    print(" INICIANDO SPIRULA STUDIO SFM (VULKAN GPU HEADLESS)...")
    print(f" Imagens:   {img_dir}")
    print(f" Qualidade: {quality}")
    print("=" * 80)

    cmd = [
        str(SPIRULA_EXE), "sfm", "auto",
        str(img_dir),
        "-o", str(dataset_dir),
        "--data-type", "video",
        "--quality", quality,
        "--camera-model", "thin-prism-fisheye"
    ]

    t0 = time.time()
    ret = subprocess.run(cmd)
    if ret.returncode != 0 and not (sparse_dir / "images.bin").exists():
        print(f"[!] Spirula SFM finalizou com código {ret.returncode}")
        return False

    print(f"[+] Spirula SFM concluído em {time.time() - t0:.1f}s!")
    return True


def align_colmap_to_lidar(dataset_dir, fps=1.0):
    """Executa o alinhamento de alta precisão Sim(3) + ICP entre COLMAP e LiDAR"""
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_pcd = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    slam_trj = dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    print("=" * 80)
    print(" ALINHAMENTO MÉTRICO COLMAP (SPIRULA) -> LIDAR SLAM")
    print("=" * 80)

    pts_colmap, rgb_colmap = load_colmap_points(sparse_dir / "points3D.bin")
    pts_lidar = load_pcd(slam_pcd)
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

    # Busca ótima do dt (coarse-to-fine adaptativa)
    print(f"[*] Buscando sincronização temporal ótima Δt (FPS = {fps:.1f})...")
    best_rmse = 1e9
    best_dt = 0
    best_sim3 = None

    # Coarse search de -200s a +50s
    dts = np.linspace(-200.0, 50.0, 1001)
    for dt in dts:
        X_trj = []
        Y_trj = []
        for im in cam0:
            frame_num = get_frame_num(im["name"])
            t_vid = (frame_num - 1) / fps
            t_q = t0_slam + (t_vid - dt)
            if t0_slam <= t_q <= t1_slam:
                idx = np.searchsorted(t_slam, t_q)
                idx = np.clip(idx, 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                X_trj.append(im["C"])
                Y_trj.append(p_L)

        if len(X_trj) >= 20:
            s, R, t_trans, rmse = solve_umeyama_sim3(np.array(X_trj), np.array(Y_trj))
            if rmse < best_rmse:
                best_rmse = rmse
                best_dt = dt
                best_sim3 = (s, R, t_trans)

    # Refinar dt a 2 ms
    fine_dts = np.linspace(best_dt - 1.0, best_dt + 1.0, 501)
    for dt in fine_dts:
        X_trj = []
        Y_trj = []
        for im in cam0:
            frame_num = get_frame_num(im["name"])
            t_vid = (frame_num - 1) / fps
            t_q = t0_slam + (t_vid - dt)
            if t0_slam <= t_q <= t1_slam:
                idx = np.searchsorted(t_slam, t_q)
                idx = np.clip(idx, 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                X_trj.append(im["C"])
                Y_trj.append(p_L)

        if len(X_trj) >= 20:
            s, R, t_trans, rmse = solve_umeyama_sim3(np.array(X_trj), np.array(Y_trj))
            if rmse < best_rmse:
                best_rmse = rmse
                best_dt = dt
                best_sim3 = (s, R, t_trans)

    s_init, R_init, t_init = best_sim3
    print(f"    Sincronização Ótima: Δt = {best_dt:.4f}s | RMSE Trajetória: {best_rmse*100:.2f} cm")

    # Refinamento Trimmed ICP nos tie-points 3D
    step_lidar = max(1, len(pts_lidar) // 200000)
    sub_lidar = pts_lidar[::step_lidar]
    tree = cKDTree(sub_lidar)

    step_col = max(1, len(pts_colmap) // 50000)
    raw_sub_col = pts_colmap[::step_col].copy()
    current_pts = s_init * (R_init @ raw_sub_col.T).T + t_init

    print(f"[*] Executando Refinamento Trimmed ICP ({len(raw_sub_col):,} tie-points)...")
    s_comp, R_comp, t_comp = s_init, R_init.copy(), t_init.copy()

    for it in range(1, 11):
        dists, idxs = tree.query(current_pts, k=1)
        thresh = min(0.25, np.percentile(dists, 70))
        mask = dists < thresh
        match_c = raw_sub_col[mask]
        match_l = sub_lidar[idxs[mask]]

        s_comp, R_comp, t_comp, rmse_icp = solve_umeyama_sim3(match_c, match_l)
        current_pts = s_comp * (R_comp @ raw_sub_col.T).T + t_comp

    print(f"[+] ICP Convergiu: RMSE = {rmse_icp*100:.2f} cm, Escala s = {s_comp:.6f}")

    align_data = {
        "scale": float(s_comp),
        "R": R_comp.tolist(),
        "t": t_comp.tolist(),
        "rmse_cm": float(rmse_icp * 100),
        "dt_sync_seconds": float(best_dt)
    }
    with open(align_json, "w", encoding="utf-8") as f:
        json.dump(align_data, f, indent=2)

    return align_data


def sync_via_gyro_cross_correlation(insv_path, bag_path, trj_path):
    """Calcula automaticamente com precisão de milissegundos o offset temporal Δt via correlação de giro IMU"""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        print("[!] rosbags não instalado para leitura direta de IMU.")
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
                print("[!] Registros de Gyro/Exposure não encontrados no trailer INSV.")
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
            print("[!] Nenhuma mensagem de IMU encontrada no arquivo .bag.")
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
        print(f"  [+] Sincronização IMU Gyro Ótima: Δt = {best_exact:.4f}s (Pearson r = {max_corr:.4f})")
        return float(best_exact)
    except Exception as e:
        print(f"[!] Erro no cálculo de sincronização IMU: {e}")
        return None


def recalibrate_from_sfm(dataset_dir, fps=1.0):
    """Recalibra com alta precisão os parâmetros de montagem T_LC0 e T_LC1 usando as poses do SfM/Spirula"""
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_trj = dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    if not align_json.exists():
        align_colmap_to_lidar(dataset_dir, fps=fps)

    with open(align_json, "r", encoding="utf-8") as f:
        al = json.load(f)
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
        cam_imgs.sort(key=lambda x: get_fn(x["name"]))
        R_list, t_list = [], []
        for im in cam_imgs:
            fn = get_fn(im["name"])
            t_vid = (fn - 1) / fps
            t_q = t_slam[0] + (t_vid - dt_sync)
            if t_slam[0] <= t_q <= t_slam[-1]:
                R_L = slerp(t_q).as_matrix()
                idx = np.searchsorted(t_slam, t_q)
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
    print(" RE-CALIBRAÇÃO FÍSICA A PARTIR DO SFM/SPIRULA CONCLUÍDA:")
    print(f" Cam0: Euler XYZ={e0} | Braço={t0*100} cm (norma {np.linalg.norm(t0)*100:.2f} cm)")
    print(f" Cam1: Euler XYZ={e1} | Braço={t1*100} cm (norma {np.linalg.norm(t1)*100:.2f} cm)")
    print(f" Ângulo relativo entre lentes: {rel_ang:.2f}° (Teórico: 180°)")
    print("=" * 80)

    calib_dict = {
        "scanner": "3DMakerPro_Raven_LiDAR",
        "camera": "Insta360_X4_DualFisheye",
        "lens_model": "THIN_PRISM_FISHEYE",
        "cam0_front_intrinsics": {
            "fx": 1081.46958, "fy": 1081.54143, "cx": 1920.0, "cy": 1920.0,
            "k1": 0.083654, "k2": -0.031306, "p1": -0.000387, "p2": 0.001724,
            "k3": 0.012000, "k4": -0.002967, "sx1": -0.003401, "sy1": 0.000687
        },
        "cam1_rear_intrinsics": {
            "fx": 1079.49331, "fy": 1079.54918, "cx": 1920.0, "cy": 1920.0,
            "k1": 0.080167, "k2": -0.027667, "p1": -0.000449, "p2": 0.000715,
            "k3": 0.010155, "k4": -0.002635, "sx1": 0.000645, "sy1": 0.002474
        },
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

    with open(dataset_dir / "calibracao_rigida_auto.json", "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)
    with open(dataset_dir / "calibracao_rigida_recalibrada.json", "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)
    with open(DEFAULT_CALIB_JSON, "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)

    print(f"  [+] Calibração salva em: {dataset_dir / 'calibracao_rigida_auto.json'}")
    print(f"  [+] Calibração mestre atualizada: {DEFAULT_CALIB_JSON}")
    return calib_dict


def colorize_via_spirula_sfm(dataset_dir, fps=1.0):
    """Executa a coloração de alta precisão projetando as poses alinhadas do SfM"""
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_pcd = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    if not align_json.exists():
        align_colmap_to_lidar(dataset_dir, fps=fps)

    with open(align_json, "r", encoding="utf-8") as f:
        al = json.load(f)
    s_sim = al["scale"]
    R_sim = np.array(al["R"])
    t_sim = np.array(al["t"])

    cameras = load_colmap_cameras(sparse_dir / "cameras.bin")
    images = load_colmap_images(sparse_dir / "images.bin")
    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    K_VIEWS = 3
    top_scores = np.zeros((n_pts, K_VIEWS), dtype=np.float32)
    top_colors = np.zeros((n_pts, K_VIEWS, 3), dtype=np.uint8)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    print("\n" + "=" * 80)
    print(" PROJETANDO QUADROS SFM SOBRE A NUVEM LIDAR (TOP-3 CONSENSUS)...")
    print("=" * 80)

    t0 = time.time()
    images.sort(key=lambda x: x["name"])

    for count, im in enumerate(images, 1):
        img_path = dataset_dir / "images" / im["name"]
        if not img_path.exists():
            continue

        params = cameras[im["cam_id"]]["params"]

        # Posição e orientação da câmera no referencial LiDAR
        C_lidar = s_sim * (R_sim @ im["C"]) + t_sim
        R_cw_lidar = im["R_cw"] @ R_sim.T

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
        mask_circle = (r_px < 1620.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
        if not np.any(mask_circle):
            continue

        u_c = u[mask_circle]
        v_c = v[mask_circle]
        d_c = d[mask_circle]
        r_c = r_px[mask_circle]
        idx_c = idx_valid[mask_circle]

        # Z-Buffer raster de oclusão
        ug = np.clip(np.floor(u_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_w - 1)
        vg = np.clip(np.floor(v_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_h - 1)
        flat_idx = vg * zbuf_w + ug

        dmap = np.full(zbuf_w * zbuf_h, 1e9, dtype=np.float32)
        np.minimum.at(dmap, flat_idx, d_c.astype(np.float32))

        vis_mask = d_c <= (dmap[flat_idx] * 1.08 + 0.15)
        if not np.any(vis_mask):
            continue

        u_vis = np.round(u_c[vis_mask]).astype(int)
        v_vis = np.round(v_c[vis_mask]).astype(int)
        d_vis = d_c[vis_mask]
        r_vis = r_c[vis_mask]
        idx_vis = idx_c[vis_mask]

        scores = (1.0 - 0.5 * (r_vis / 1620.0)**2) / (np.maximum(d_vis, 0.5)**1.5)

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
        rgb_ins = img_rgb[v_ins, u_ins]

        worst_slot = np.argmin(top_scores[idx_ins], axis=1)
        top_scores[idx_ins, worst_slot] = scores_ins
        top_colors[idx_ins, worst_slot] = rgb_ins

        if count % 50 == 0 or count == len(images):
            cov = np.count_nonzero(np.any(top_scores > 0, axis=1)) / n_pts * 100
            print(f"    [{count:3d}/{len(images)}] Quadros projetados | Cobertura LiDAR: {cov:.1f}%")

    print("[*] Resolvendo consenso estatístico SfM...")
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
        weights[fallback] = 1.0
        sum_w[fallback] = 3.0
        norm_w = weights / sum_w
        final_c3 = np.sum(c3 * norm_w[:, :, None], axis=1)
        colors[idx_3] = np.clip(np.round(final_c3), 0, 255).astype(np.uint8)

    out_ply = dataset_dir / "02_NUVEM_LIDAR_COLORIDA_METODO_SFM_SPIRULA_CORRIGIDO.ply"
    out_pcd = dataset_dir / "02_NUVEM_LIDAR_COLORIDA_METODO_SFM_SPIRULA_CORRIGIDO.pcd"
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    deliv_dir = dataset_dir / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(out_ply, deliv_dir / out_ply.name)
        shutil.copy2(out_pcd, deliv_dir / out_pcd.name)
    except Exception:
        pass

    print(f"[+] Concluído em {time.time() - t0:.1f}s!")


# ==============================================================================
# PIPELINE MÉTODO 2: DIRETO RÍGIDO (SEM SFM)
# ==============================================================================

def colorize_via_direct_rigid(dataset_dir, calib_json_path=None, fps=1.0, dt_override=None):
    """Executa a coloração direta rápida usando matriz rígida e tempo calibrado"""
    if calib_json_path is None:
        auto_json = dataset_dir / "calibracao_rigida_auto.json"
        azure_json = dataset_dir / "calibracao_rigida_azure_dataset.json"
        recalib_json = dataset_dir / "calibracao_rigida_recalibrada.json"
        if auto_json.exists():
            calib_json_path = auto_json
        elif recalib_json.exists():
            calib_json_path = recalib_json
        elif azure_json.exists():
            calib_json_path = azure_json
        else:
            candidates = list(dataset_dir.glob("calibracao_rigida*.json"))
            calib_json_path = candidates[0] if candidates else DEFAULT_CALIB_JSON

    print("=" * 80)
    print(" COLORACAO DIRETA RIGIDA (SEM SFM)")
    print(f" Calibração: {calib_json_path.name}")
    print("=" * 80)

    with open(calib_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

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

    slam_pcd = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    slam_trj = dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"

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
                    print(f"[*] Sincronização herdada do COLMAP Sim(3): Δt = {dt_sync:.4f}s")
        except Exception:
            pass

    if dt_sync is None:
        insv_files = list(dataset_dir.glob("*.insv"))
        bag_files = list(dataset_dir.glob("*.bag"))
        if insv_files and bag_files:
            print("[*] Sincronização via Gyro Cross-Correlation...")
            dt_sync = sync_via_gyro_cross_correlation(insv_files[0], bag_files[0], slam_trj)

    if dt_sync is None:
        dt_sync = -4.220
        print(f"[!] Aviso: Δt padrão adotado = {dt_sync:.4f}s")
    else:
        print(f"[*] Sincronização Temporal Ativa: Δt = {dt_sync:.4f}s")

    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
    t_slam_start = t_slam[0]
    t_slam_end = t_slam[-1]
    slerp = Slerp(t_slam, rot_slam)

    cam0_files = sorted(list((dataset_dir / "images" / "cam0").glob("*.jpg")))
    cam1_files = sorted(list((dataset_dir / "images" / "cam1").glob("*.jpg")))

    K_VIEWS = 3
    top_scores = np.zeros((n_pts, K_VIEWS), dtype=np.float32)
    top_colors = np.zeros((n_pts, K_VIEWS, 3), dtype=np.uint8)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    t0 = time.time()
    num_frames = min(len(cam0_files), len(cam1_files))

    print("\n" + "=" * 80)
    print(" COLORACAO DIRETA RIGIDA (TOP-3 MULTI-VIEW CONSENSUS)")
    print("=" * 80)

    # Se alinhamento SfM existir, calcular correção suave de drift da trajetória (Amostragem de Keyframes)
    use_drift_correction = False
    sparse_img_bin = dataset_dir / "sparse" / "0" / "images.bin"
    if align_json.exists() and sparse_img_bin.exists():
        try:
            with open(align_json, "r", encoding="utf-8") as f:
                al = json.load(f)
            s_sim = al["scale"]
            R_sim = np.array(al["R"])
            t_sim = np.array(al["t"])
            sfm_imgs = load_colmap_images(sparse_img_bin)
            cam0_sfm = [im for im in sfm_imgs if im["name"].startswith("cam0")]
            cam0_sfm.sort(key=lambda x: int("".join(filter(str.isdigit, x["name"]))))

            n_samples = min(30, len(cam0_sfm))
            sample_indices = np.linspace(0, len(cam0_sfm) - 1, n_samples).astype(int)
            sample_imgs = [cam0_sfm[i] for i in sample_indices]

            sample_times = []
            delta_p_list = []
            delta_R_list = []

            for im in sample_imgs:
                fn = int("".join(filter(str.isdigit, im["name"])))
                t_vid = (fn - 1) / fps
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
                print(f"[*] Correção de Drift da Trajetória ativada com {len(sample_times)} quadros-chave SfM!")
        except Exception as e:
            print(f"[!] Aviso: não foi possível carregar drift correction do SfM: {e}")

    for k in range(num_frames):
        fn_digits = "".join(filter(str.isdigit, cam0_files[k].stem))
        fn = int(fn_digits) if fn_digits else (k + 1)
        t_vid = (fn - 1) / fps
        t_query = t_slam_start + (t_vid - dt_sync)

        if not (t_slam_start <= t_query <= t_slam_end):
            continue

        idx = np.searchsorted(t_slam, t_query)
        idx = np.clip(idx, 1, len(t_slam) - 1)
        w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L_raw = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L_raw = slerp(t_query).as_matrix()

        if use_drift_correction and (sample_times[0] <= t_query <= sample_times[-1]):
            dp = np.array([np.interp(t_query, sample_times, delta_p_arr[:, ax]) for ax in range(3)])
            dR = slerp_delta_R(t_query).as_matrix()
            p_L = p_L_raw + dp
            R_L = dR @ R_L_raw
        else:
            p_L = p_L_raw
            R_L = R_L_raw

        lens_configs = [
            ("cam0", cam0_files[k], params0, R_LC0, t_LC0),
            ("cam1", cam1_files[k], params1, R_LC1, t_LC1)
        ]

        for cam_name, img_path, params, R_LC, t_LC in lens_configs:
            p_cam = p_L + R_L @ t_LC
            R_world_cam = R_L @ R_LC
            R_cw = R_world_cam.T

            P_cam = (R_cw @ (pts_lidar - p_cam).T).T
            z_mask = P_cam[:, 2] > 0.15
            idx_valid = np.where(z_mask)[0]
            if len(idx_valid) == 0:
                continue

            P_v = P_cam[idx_valid]
            d = np.linalg.norm(P_v, axis=1)

            cx, cy = params[2], params[3]
            u, v, r = project_thin_prism(P_v, params)

            r_px = np.hypot(u - cx, v - cy)
            mask_circle = (r_px < 1620.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
            if not np.any(mask_circle):
                continue

            u_c = u[mask_circle]
            v_c = v[mask_circle]
            d_c = d[mask_circle]
            r_c = r_px[mask_circle]
            idx_c = idx_valid[mask_circle]

            ug = np.clip(np.floor(u_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_w - 1)
            vg = np.clip(np.floor(v_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_h - 1)
            flat_idx = vg * zbuf_w + ug

            dmap = np.full(zbuf_w * zbuf_h, 1e9, dtype=np.float32)
            np.minimum.at(dmap, flat_idx, d_c.astype(np.float32))

            vis_mask = d_c <= (dmap[flat_idx] * 1.08 + 0.15)
            if not np.any(vis_mask):
                continue

            u_vis = np.round(u_c[vis_mask]).astype(int)
            v_vis = np.round(v_c[vis_mask]).astype(int)
            d_vis = d_c[vis_mask]
            r_vis = r_c[vis_mask]
            idx_vis = idx_c[vis_mask]

            scores = (1.0 - 0.5 * (r_vis / 1620.0)**2) / (np.maximum(d_vis, 0.5)**1.5)

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
            rgb_ins = img_rgb[v_ins, u_ins]

            worst_slot = np.argmin(top_scores[idx_ins], axis=1)
            top_scores[idx_ins, worst_slot] = scores_ins
            top_colors[idx_ins, worst_slot] = rgb_ins

        if (k + 1) % 25 == 0 or (k + 1) == num_frames:
            cov = np.count_nonzero(np.any(top_scores > 0, axis=1)) / n_pts * 100
            print(f"    [{k+1:3d}/{num_frames}] Pares de quadros acumulados | Cobertura: {cov:.1f}%")

    print("[*] Resolvendo consenso estatístico e eliminando outliers de projeção...")
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
        weights[fallback] = 1.0
        sum_w[fallback] = 3.0
        norm_w = weights / sum_w
        final_c3 = np.sum(c3 * norm_w[:, :, None], axis=1)
        colors[idx_3] = np.clip(np.round(final_c3), 0, 255).astype(np.uint8)

    out_ply = dataset_dir / "03_NUVEM_LIDAR_COLORIDA_METODO_DIRETO_CALIBRADO.ply"
    out_pcd = dataset_dir / "03_NUVEM_LIDAR_COLORIDA_METODO_DIRETO_CALIBRADO.pcd"
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    deliv_dir = dataset_dir / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(out_ply, deliv_dir / out_ply.name)
        shutil.copy2(out_pcd, deliv_dir / out_pcd.name)
    except Exception:
        pass

    print(f"[+] Método Direto Rígido concluído em {time.time() - t0:.1f}s!")


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Unificado de Calibração e Coloração LiDAR-Câmera 360")
    parser.add_argument("--dataset", type=str, default=r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST",
                        help="Caminho para a pasta do dataset")
    parser.add_argument("--method", type=str, choices=["all", "sfm", "direct"], default="all",
                        help="Método a executar: 'sfm', 'direct' ou 'all' para ambos")
    parser.add_argument("--fps", type=float, default=1.0, help="Taxa de amostragem dos quadros (FPS, padrão 1.0)")
    parser.add_argument("--run-spirula", action="store_true", help="Forçar execução do spirula.exe antes de colorir")
    parser.add_argument("--recalibrate-from-sfm", action="store_true", help="Recalcular parâmetros de montagem usando SfM")
    parser.add_argument("--dt", type=float, default=None, help="Sincronização temporal manual Δt em segundos")
    parser.add_argument("--calib", type=str, default=None, help="Caminho opcional do arquivo de calibração JSON")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset)

    print("=" * 80)
    print(" PIPELINE MASTER DE COLORACAO LIDAR-CAMERA 360")
    print(f" Dataset: {dataset_dir}")
    print(f" Método:  {args.method.upper()}")
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
    print(" PROCESSO FINALIZADO COM SUCESSO!")
    print(f" Os arquivos coloridos foram salvos em: {dataset_dir}")
    print(f" Entregáveis consolidados em: {dataset_dir / 'deliverables'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
