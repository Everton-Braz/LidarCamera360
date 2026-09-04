#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coloração Direta Rígida (Sem SfM) usando Matriz Rígida Calibrada e Sincronização por Giroscópio
================================================================================================
Aplica a calibração rígida mestre (calibracao_rigida_raven_insta360.json) diretamente sobre a
trajetória FAST-LIVO2 do LiDAR e os quadros de alta resolução da câmera 360, sem depender de COLMAP.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.stdout.reconfigure(encoding='utf-8')

DATASET_DIR = Path(r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST")
CALIB_JSON = Path(r"C:\Users\User\Documents\APLICATIVOS\Lidar-Camera-calibrator\calibracao_rigida_raven_insta360.json")

SLAM_PCD = DATASET_DIR / "slam_out" / "pcd" / "all_raw_points.pcd"
SLAM_TRJ = DATASET_DIR / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
IMG_DIR = DATASET_DIR / "images"

OUT_PLY = DATASET_DIR / "01_NUVEM_LIDAR_COLORIDA_METODO_DIRETO_SEM_SFM.ply"
OUT_PCD = DATASET_DIR / "01_NUVEM_LIDAR_COLORIDA_METODO_DIRETO_SEM_SFM.pcd"

# Sincronização temporal ótima calculada via correlação cruzada do giroscópio (r = 0.9993)
DT_SYNC = -4.6290  # segundos


def load_pcd(path):
    print(f"[*] Carregando nuvem LiDAR: {path.name}...")
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
    print(f"    Total de pontos carregados: {len(xyz):,}")
    return xyz


def load_trajectory(path):
    print(f"[*] Carregando trajetória SLAM: {path.name}...")
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    rotations = Rot.from_quat(data[:, 4:8])
    print(f"    Total de poses: {len(data):,}, Duração: {timestamps[-1] - timestamps[0]:.2f}s")
    return timestamps, positions, rotations


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
    print(f"  [+] Arquivo PLY salvo: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


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
    print(f"  [+] Arquivo PCD salvo: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


def main():
    print("=" * 80)
    print(" COLORACAO DIRETA RIGIDA (SEM SFM) - 3DMAKERPRO RAVEN + INSTA360 X4")
    print(" Dataset: SMALL-DATASET-TEST")
    print("=" * 80)
    t_start = time.time()

    # 1. Carregar Configuração Rígida Mestre
    print(f"[*] Carregando arquivo de calibração rígida: {CALIB_JSON.name}...")
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Parâmetros das lentes
    p0 = cfg["cam0_front_intrinsics"]
    params0 = (p0["fx"], p0["fy"], p0["cx"], p0["cy"], p0["k1"], p0["k2"], p0["p1"], p0["p2"], p0["k3"], p0["k4"], p0["sx1"], p0["sy1"])
    
    p1 = cfg["cam1_rear_intrinsics"]
    params1 = (p1["fx"], p1["fy"], p1["cx"], p1["cy"], p1["k1"], p1["k2"], p1["p1"], p1["p2"], p1["k3"], p1["k4"], p1["sx1"], p1["sy1"])

    # Matrizes extrínsecas T_LC (LiDAR -> Camera)
    T_LC0 = np.array(cfg["T_lidar_to_cam0_rigid_4x4"])
    R_LC0 = T_LC0[:3, :3]
    t_LC0 = T_LC0[:3, 3]

    T_LC1 = np.array(cfg["T_lidar_to_cam1_rigid_4x4"])
    R_LC1 = T_LC1[:3, :3]
    t_LC1 = T_LC1[:3, 3]

    print(f"    Braço de Alavanca Físico: {cfg['lever_arm_meters']['norm_distance_cm']:.2f} cm")
    print(f"    Offset Cam0 (t_LC): {t_LC0}")
    print(f"    Offset Cam1 (t_LC): {t_LC1}")

    # 2. Carregar Dados do LiDAR
    pts_lidar = load_pcd(SLAM_PCD)
    n_pts = len(pts_lidar)

    t_slam, pos_slam, rot_slam = load_trajectory(SLAM_TRJ)
    t_slam_start = t_slam[0]
    t_slam_end = t_slam[-1]

    # Objeto Slerp para interpolação contínua da rotação
    slerp = Slerp(t_slam, rot_slam)

    # 3. Listar Quadros Extraídos (cam0 e cam1)
    cam0_files = sorted(list((IMG_DIR / "cam0").glob("*.jpg")))
    cam1_files = sorted(list((IMG_DIR / "cam1").glob("*.jpg")))
    print(f"[*] Quadros encontrados: cam0={len(cam0_files)}, cam1={len(cam1_files)}")

    fps_extracted = 2.0  # Extraído a 2 FPS

    # Buffers de cor e pontuação
    colors = np.full((n_pts, 3), 180, dtype=np.uint8)
    best_scores = np.zeros(n_pts, dtype=np.float32)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    print("\n" + "=" * 80)
    print(" INICIANDO COLORACAO DIRETA RIGIDA MULTI-QUADRO...")
    print(f" Sincronizacao Temporal: dt = {DT_SYNC:.4f} s (via Giroscopio)")
    print("=" * 80)

    t_loop_start = time.time()
    frames_processed = 0

    # Iterar sobre todos os instantes de vídeo extraídos
    num_frames = min(len(cam0_files), len(cam1_files))

    for k in range(num_frames):
        t_vid = k / fps_extracted  # tempo em segundos dentro do vídeo
        # Momento correspondente no relógio SLAM:
        t_query = t_slam_start + (t_vid - DT_SYNC)

        # Apenas processar se estiver dentro do intervalo do SLAM
        if not (t_slam_start <= t_query <= t_slam_end):
            continue

        # Interpolação da pose do LiDAR
        idx = np.searchsorted(t_slam, t_query)
        idx = np.clip(idx, 1, len(t_slam) - 1)
        w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L = slerp(t_query).as_matrix()

        # Processar ambas as lentes: (lente, arquivos, params, R_LC, t_LC)
        lens_configs = [
            ("cam0", cam0_files[k], params0, R_LC0, t_LC0),
            ("cam1", cam1_files[k], params1, R_LC1, t_LC1)
        ]

        for cam_name, img_path, params, R_LC, t_LC in lens_configs:
            # Posição e orientação da câmera no mundo:
            # p_cam = p_L + R_L @ t_LC
            # R_world_cam = R_L @ R_LC
            p_cam = p_L + R_L @ t_LC
            R_world_cam = R_L @ R_LC
            R_cw = R_world_cam.T

            # Mapear nuvem LiDAR métrica para o referencial de câmera:
            # P_cam = R_cw @ (P_world - p_cam)
            P_cam = (R_cw @ (pts_lidar - p_cam).T).T
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
            u_update = u_vis[better_mask]
            v_update = v_vis[better_mask]
            scores_update = scores[better_mask]

            # Ler imagem e amostrar pixels
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            colors[idx_update] = img_rgb[v_update, u_update]
            best_scores[idx_update] = scores_update

        frames_processed += 1
        if (k + 1) % 15 == 0 or k == num_frames - 1:
            n_colored = np.sum(best_scores > 0)
            pct = (n_colored / n_pts) * 100.0
            elapsed = time.time() - t_loop_start
            fps = frames_processed / elapsed
            print(f"  Quadro [{k+1:3d}/{num_frames:3d}] (t={t_vid:4.1f}s) | "
                  f"Coloridos: {n_colored:,}/{n_pts:,} ({pct:.1f}%) | {fps:.1f} fps")

    # 5. Estatísticas Finais
    n_colored_final = np.sum(best_scores > 0)
    pct_final = (n_colored_final / n_pts) * 100.0
    print("\n" + "=" * 80)
    print(" RESULTADO DA COLORACAO DIRETA:")
    print(f"  Total de Pontos no LiDAR: {n_pts:,}")
    print(f"  Pontos 100% Coloridos:    {n_colored_final:,} ({pct_final:.2f}%)")
    print(f"  Quadros Processados:      {frames_processed} quadros sincronizados")
    print(f"  Tempo Total:              {time.time() - t_loop_start:.1f} segundos")
    print("=" * 80)

    # 6. Salvar Arquivos
    print("\n[*] Exportando arquivos finais coloridos...")
    write_ply(OUT_PLY, pts_lidar, colors)
    write_pcd(OUT_PCD, pts_lidar, colors)

    # Copiar também para a pasta de inspeção do CloudCompare
    cc_folder = Path(r"C:\Users\User\Downloads\Lidou\TESTE_ICP_CLOUDCOMPARE")
    cc_folder.mkdir(parents=True, exist_ok=True)
    out_cc = cc_folder / "04_SMALL_DATASET_COLORIDO_DIRETO_SEM_SFM.ply"
    import shutil
    shutil.copy2(OUT_PLY, out_cc)
    print(f"  [+] Cópia pronta para CloudCompare: {out_cc.name}")

    # Salvar Resumo JSON
    summary = {
        "dataset": "SMALL-DATASET-TEST",
        "metodo": "Metodo_Direto_Rigido_Sem_SfM",
        "dt_sync_gyro_sec": float(DT_SYNC),
        "total_pontos_lidar": int(n_pts),
        "pontos_coloridos": int(n_colored_final),
        "taxa_cobertura_percent": float(pct_final),
        "tempo_execucao_sec": float(time.time() - t_start),
        "arquivos": {
            "ply": str(OUT_PLY),
            "pcd": str(OUT_PCD),
            "cloudcompare_copy": str(out_cc)
        }
    }
    with open(DATASET_DIR / "relatorio_coloracao_direta.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" COLORACAO DIRETA CONCLUIDA COM SUCESSO!")
    print(f" Pasta de Saída: {DATASET_DIR}")
    print(f" Pasta CloudCompare: {cc_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()
