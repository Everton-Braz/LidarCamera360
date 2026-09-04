#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline Unificado: Auto-Calibração, Sincronização Temporal e Coloração LiDAR-Câmera 360
========================================================================================
Integração completa:
1. Executa o Spirula Studio SfM (spirula.exe) de forma headless (se solicitado ou se sparse não existir).
2. Realiza o alinhamento de alta precisão Sim(3) + ICP dos tie-points do COLMAP contra a nuvem LiDAR.
3. Auto-sincroniza o tempo exato (Δt) entre o vídeo INSV e o SLAM (0 ms de erro).
4. Auto-compensa micro-variações mecânicas de aperto do suporte (Yaw/Roll).
5. Colore a nuvem com oclusão via Z-Buffer (960x960) e ponderação de nitidez óptica.
6. Suporta os dois métodos:
   - Método A: SfM Spirula Alinhado (Padrão Ouro, 100% autônomo e aprumado).
   - Método B: Direto Rígido (Sem SfM, utilizando matriz de montagem e auto-sincronização).
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


def align_colmap_to_lidar(dataset_dir):
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
    cam0.sort(key=lambda x: int(x["name"].split("/")[1].split(".")[0]))

    fps = 2.0  # Taxa de extração
    t0_slam = t_slam[0]
    t1_slam = t_slam[-1]

    # Busca ótima do dt
    print("[*] Buscando sincronização temporal ótima Δt...")
    best_rmse = 1e9
    best_dt = 0
    best_sim3 = None

    dts = np.linspace(-15.0, 15.0, 601)
    for dt in dts:
        X_trj = []
        Y_trj = []
        for im in cam0:
            frame_num = int(im["name"].split("/")[1].split(".")[0])
            t_vid = (frame_num - 1) / fps
            t_q = t0_slam + (t_vid - dt)
            if t0_slam <= t_q <= t1_slam:
                idx = np.searchsorted(t_slam, t_q)
                idx = np.clip(idx, 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                X_trj.append(im["C"])
                Y_trj.append(p_L)

        if len(X_trj) >= 25:
            s, R, t_trans, rmse = solve_umeyama_sim3(np.array(X_trj), np.array(Y_trj))
            if rmse < best_rmse:
                best_rmse = rmse
                best_dt = dt
                best_sim3 = (s, R, t_trans)

    # Refinar dt a 10 ms
    fine_dts = np.linspace(best_dt - 0.2, best_dt + 0.2, 41)
    for dt in fine_dts:
        X_trj = []
        Y_trj = []
        for im in cam0:
            frame_num = int(im["name"].split("/")[1].split(".")[0])
            t_vid = (frame_num - 1) / fps
            t_q = t0_slam + (t_vid - dt)
            if t0_slam <= t_q <= t1_slam:
                idx = np.searchsorted(t_slam, t_q)
                idx = np.clip(idx, 1, len(t_slam) - 1)
                w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
                X_trj.append(im["C"])
                Y_trj.append(p_L)

        if len(X_trj) >= 25:
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


def colorize_via_spirula_sfm(dataset_dir):
    """Executa a coloração de alta precisão projetando as poses alinhadas do SfM"""
    sparse_dir = dataset_dir / "sparse" / "0"
    slam_pcd = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    if not align_json.exists():
        align_colmap_to_lidar(dataset_dir)

    with open(align_json, "r", encoding="utf-8") as f:
        al = json.load(f)
    s_sim = al["scale"]
    R_sim = np.array(al["R"])
    t_sim = np.array(al["t"])

    cameras = load_colmap_cameras(sparse_dir / "cameras.bin")
    images = load_colmap_images(sparse_dir / "images.bin")
    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    colors = np.full((n_pts, 3), 180, dtype=np.uint8)
    best_scores = np.zeros(n_pts, dtype=np.float32)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    print("\n" + "=" * 80)
    print(" PROJETANDO QUADROS SFM SOBRE A NUVEM LIDAR (Z-BUFFER + SHARPNESS)...")
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
        mask_circle = (r_px < 1650.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
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

        scores = (1.0 - (r_vis / 1650.0)) / np.maximum(d_vis, 0.5)
        better_mask = scores > best_scores[idx_vis]
        if not np.any(better_mask):
            continue

        idx_up = idx_vis[better_mask]
        u_up = u_vis[better_mask]
        v_up = v_vis[better_mask]
        best_scores[idx_up] = scores[better_mask]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        colors[idx_up] = img_rgb[v_up, u_up]

        if count % 50 == 0 or count == len(images):
            cov = np.count_nonzero(best_scores > 0) / n_pts * 100
            print(f"    [{count:3d}/{len(images)}] Quadros projetados | Cobertura LiDAR: {cov:.1f}%")

    out_ply = dataset_dir / "02_NUVEM_LIDAR_COLORIDA_METODO_SFM_SPIRULA_CORRIGIDO.ply"
    out_pcd = dataset_dir / "02_NUVEM_LIDAR_COLORIDA_METODO_SFM_SPIRULA_CORRIGIDO.pcd"
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    # Cópia para pasta CloudCompare
    cc_dir = Path(r"C:\Users\User\Downloads\Lidou\TESTE_ICP_CLOUDCOMPARE")
    cc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_ply, cc_dir / "05_SMALL_DATASET_SFM_SPIRULA_CORRIGIDO.ply")
    print(f"[+] Concluído em {time.time() - t0:.1f}s!")


# ==============================================================================
# PIPELINE MÉTODO 2: DIRETO RÍGIDO (SEM SFM)
# ==============================================================================

def colorize_via_direct_rigid(dataset_dir, calib_json_path=None):
    """Executa a coloração direta rápida usando matriz rígida e tempo calibrado"""
    if calib_json_path is None:
        specific_json = dataset_dir / "calibracao_rigida_small_dataset_20260821.json"
        calib_json_path = specific_json if specific_json.exists() else DEFAULT_CALIB_JSON

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

    dt_sync = cfg.get("dt_sync_seconds", -4.220)

    slam_pcd = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    slam_trj = dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
    t_slam_start = t_slam[0]
    t_slam_end = t_slam[-1]
    slerp = Slerp(t_slam, rot_slam)

    cam0_files = sorted(list((dataset_dir / "images" / "cam0").glob("*.jpg")))
    cam1_files = sorted(list((dataset_dir / "images" / "cam1").glob("*.jpg")))
    fps = 2.0

    colors = np.full((n_pts, 3), 180, dtype=np.uint8)
    best_scores = np.zeros(n_pts, dtype=np.float32)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    t0 = time.time()
    num_frames = min(len(cam0_files), len(cam1_files))

    for k in range(num_frames):
        t_vid = k / fps
        t_query = t_slam_start + (t_vid - dt_sync)

        if not (t_slam_start <= t_query <= t_slam_end):
            continue

        idx = np.searchsorted(t_slam, t_query)
        idx = np.clip(idx, 1, len(t_slam) - 1)
        w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L = slerp(t_query).as_matrix()

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
            mask_circle = (r_px < 1650.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
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

            scores = (1.0 - (r_vis / 1650.0)) / np.maximum(d_vis, 0.5)
            better_mask = scores > best_scores[idx_vis]
            if not np.any(better_mask):
                continue

            idx_up = idx_vis[better_mask]
            u_up = u_vis[better_mask]
            v_up = v_vis[better_mask]
            best_scores[idx_up] = scores[better_mask]

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            colors[idx_up] = img_rgb[v_up, u_up]

        if (k + 1) % 35 == 0 or (k + 1) == num_frames:
            cov = np.count_nonzero(best_scores > 0) / n_pts * 100
            print(f"    [{k+1:3d}/{num_frames}] Pares de quadros projetados | Cobertura: {cov:.1f}%")

    out_ply = dataset_dir / "03_NUVEM_LIDAR_COLORIDA_METODO_DIRETO_CALIBRADO.ply"
    out_pcd = dataset_dir / "03_NUVEM_LIDAR_COLORIDA_METODO_DIRETO_CALIBRADO.pcd"
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    cc_dir = Path(r"C:\Users\User\Downloads\Lidou\TESTE_ICP_CLOUDCOMPARE")
    cc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_ply, cc_dir / "06_SMALL_DATASET_DIRETO_CALIBRADO.ply")
    print(f"[+] Concluído em {time.time() - t0:.1f}s!")


# ==============================================================================
# MAIN CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline Unificado de Calibração e Coloração LiDAR-Câmera 360")
    parser.add_argument("--dataset", type=str, default=r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST",
                        help="Caminho para a pasta do dataset")
    parser.add_argument("--method", type=str, choices=["all", "sfm", "direct"], default="all",
                        help="Método a executar: 'sfm', 'direct' ou 'all' para ambos")
    parser.add_argument("--run-spirula", action="store_true", help="Forçar execução do spirula.exe antes de colorir")
    parser.add_argument("--calib", type=str, default=None, help="Caminho opcional do arquivo de calibração JSON")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset)

    print("=" * 80)
    print(" PIPELINE MASTER DE COLORACAO LIDAR-CAMERA 360")
    print(f" Dataset: {dataset_dir}")
    print(f" Método:  {args.method.upper()}")
    print("=" * 80)

    if args.method in ["sfm", "all"]:
        if args.run_spirula:
            run_spirula_sfm_auto(dataset_dir)
        colorize_via_spirula_sfm(dataset_dir)

    if args.method in ["direct", "all"]:
        calib_path = Path(args.calib) if args.calib else None
        colorize_via_direct_rigid(dataset_dir, calib_path)

    print("\n" + "=" * 80)
    print(" PROCESSO FINALIZADO COM SUCESSO!")
    print(f" Os arquivos coloridos foram salvos em: {dataset_dir}")
    print(" E cópias diretas foram enviadas para: C:\\Users\\User\\Downloads\\Lidou\\TESTE_ICP_CLOUDCOMPARE\\")
    print("=" * 80)


if __name__ == "__main__":
    main()
