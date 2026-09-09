#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coloração de Nuvem de Pontos com COLMAP / Spirula Studio (SfM Alinhado por Sim(3) + ICP)
========================================================================================
Utiliza os 350 quadros registrados pelo Spirula Studio (cam0 e cam1) e os parâmetros
Thin Prism calibrados, projetando com oclusão via Z-Buffer diretamente sobre a nuvem LiDAR.
Elimina 100% de erros de inclinação, drift do SLAM e braço de alavanca.
"""

import os
import sys
import time
import struct
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = None
SPARSE_DIR = None
SLAM_PCD = None
ALIGN_JSON = None
IMG_DIR = None
OUT_PLY = None
OUT_PCD = None
CLOUDCOMPARE_COPY = None

def load_pcd(path):
    print(f"[*] Loading LiDAR point cloud: {path.name}...")
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
    print(f"    Total points loaded: {len(xyz):,}")
    return xyz

def load_cameras(path):
    cameras = {}
    with open(path, "rb") as f:
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

def load_images(path):
    images = []
    with open(path, "rb") as f:
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

            R_cw_colmap = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
            C_colmap = -R_cw_colmap.T @ np.array(tvec)
            images.append({
                "img_id": img_id,
                "name": name,
                "cam_id": cam_id,
                "R_cw_colmap": R_cw_colmap,
                "C_colmap": C_colmap
            })
    return images

def project_thin_prism(P_v, params):
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Colorize a LiDAR point cloud from aligned COLMAP views.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory.")
    args = parser.parse_args()
    global DATASET_DIR, SPARSE_DIR, SLAM_PCD, ALIGN_JSON, IMG_DIR, OUT_PLY, OUT_PCD
    DATASET_DIR = args.dataset.resolve()
    SPARSE_DIR = DATASET_DIR / "sparse" / "0"
    SLAM_PCD = DATASET_DIR / "slam_out" / "pcd" / "all_raw_points.pcd"
    ALIGN_JSON = DATASET_DIR / "colmap_to_lidar_alignment.json"
    IMG_DIR = DATASET_DIR / "images"
    deliv_dir = DATASET_DIR / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    OUT_PLY = deliv_dir / "reconstruction_colorized.ply"
    OUT_PCD = deliv_dir / "reconstruction_colorized.pcd"
    CLOUDCOMPARE_COPY = deliv_dir / "reconstruction_colorized_cloudcompare.ply"
    print("=" * 80)
    print(" SFM COLORIZATION (SPIRULA STUDIO) ALIGNED VIA SIM(3) + ICP")
    print(f" Dataset: {DATASET_DIR}")
    print("=" * 80)

    # 1. Load Sim(3) + ICP alignment
    with open(ALIGN_JSON, "r", encoding="utf-8") as f:
        al = json.load(f)
    s_sim = al["scale"]
    R_sim = np.array(al["R"])
    t_sim = np.array(al["t"])
    print(f"[*] Loaded composite transformation: Scale={s_sim:.6f}, RMSE={al.get('rmse_cm', 0.0):.2f} cm")

    # 2. Load COLMAP Cameras and Images
    cameras = load_cameras(SPARSE_DIR / "cameras.bin")
    images = load_images(SPARSE_DIR / "images.bin")
    print(f"[*] Cameras: {len(cameras)}, Registered Images: {len(images)}")

    # 3. Carregar Nuvem LiDAR
    pts_lidar = load_pcd(SLAM_PCD)
    n_pts = len(pts_lidar)

    # Buffers de coloração
    colors = np.full((n_pts, 3), 180, dtype=np.uint8)
    best_scores = np.zeros(n_pts, dtype=np.float32)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    t_start = time.time()
    processed_count = 0

    # Ordenar imagens para processar (cam0 e cam1 alternadas ou sequenciais)
    images.sort(key=lambda x: x["name"])

    for im in images:
        img_rel_path = DATASET_DIR / "images" / im["name"]
        if not img_rel_path.exists():
            continue

        cam = cameras[im["cam_id"]]
        params = cam["params"]

        # Posição da câmera no mundo LiDAR:
        # C_lidar = s_sim * R_sim @ C_colmap + t_sim
        C_lidar = s_sim * (R_sim @ im["C_colmap"]) + t_sim

        # Rotação da câmera no referencial LiDAR:
        # R_cw_lidar = R_cw_colmap @ R_sim^T
        R_cw_lidar = im["R_cw_colmap"] @ R_sim.T

        # Mapear nuvem LiDAR métrica para o referencial de câmera:
        # P_cam = R_cw_lidar @ (pts_lidar - C_lidar)
        P_cam = (R_cw_lidar @ (pts_lidar - C_lidar).T).T

        z_mask = P_cam[:, 2] > 0.15
        idx_valid = np.where(z_mask)[0]
        if len(idx_valid) == 0:
            continue

        P_v = P_cam[idx_valid]
        d = np.linalg.norm(P_v, axis=1)

        # Projeção Thin Prism
        cx, cy = params[2], params[3]
        u, v, r = project_thin_prism(P_v, params)

        # Máscara de abertura circular (raio útil < 1650 px)
        r_px = np.hypot(u - cx, v - cy)
        mask_circle = (r_px < 1650.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
        if not np.any(mask_circle):
            continue

        u_c = u[mask_circle]
        v_c = v[mask_circle]
        d_c = d[mask_circle]
        r_c = r_px[mask_circle]
        idx_c = idx_valid[mask_circle]

        # Z-Buffer raster (960x960) para oclusão
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

        # Métrica de Nitidez Óptica
        scores = (1.0 - (r_vis / 1650.0)) / np.maximum(d_vis, 0.5)
        better_mask = scores > best_scores[idx_vis]
        if not np.any(better_mask):
            continue

        idx_update = idx_vis[better_mask]
        u_up = u_vis[better_mask]
        v_up = v_vis[better_mask]
        best_scores[idx_update] = scores[better_mask]

        # Carregar imagem em RGB
        img_bgr = cv2.imread(str(img_rel_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        colors[idx_update] = img_rgb[v_up, u_up]

        processed_count += 1
        if processed_count % 35 == 0 or processed_count == len(images):
            colored_pct = np.count_nonzero(best_scores > 0) / n_pts * 100
            print(f"    [{processed_count:3d}/{len(images)}] Projected frames | LiDAR coverage: {colored_pct:.1f}%")

    print("\n" + "=" * 80)
    print(f" COLORIZATION COMPLETED IN {time.time() - t_start:.1f}s")
    total_colored = np.count_nonzero(best_scores > 0)
    print(f" Colorized points: {total_colored:,} / {n_pts:,} ({total_colored/n_pts*100:.2f}%)")
    print("=" * 80)

    # Save deliverables
    write_ply(OUT_PLY, pts_lidar, colors)
    write_pcd(OUT_PCD, pts_lidar, colors)

    # Copy to deliverables inspection folder
    CLOUDCOMPARE_COPY.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(OUT_PLY, CLOUDCOMPARE_COPY)
    print(f"  [+] Deliverable copy: {CLOUDCOMPARE_COPY.name}")

if __name__ == "__main__":
    main()
