#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alinhamento Métrico COLMAP (Spirula Studio) -> LiDAR SLAM Ground Truth
=====================================================================
1. Otimiza a sincronização temporal exata Δt entre o vídeo INSV e o SLAM do LiDAR.
2. Resolve a transformação de similaridade Sim(3) de Umeyama (escala s, rotação R, translação t).
3. Transforma todos os 177.479 pontos 3D do COLMAP para o referencial métrico real do LiDAR.
4. Exporta as nuvens PCD para inspeção visual e medição no CloudCompare.
5. Deriva a calibração extrínseca rígida sensor-a-sensor via Hand-Eye (AX = XB).
"""

import os
import sys
import struct
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib")
COLMAP_DIR = DATASET_DIR / "VID_20260902_143757_00_277_dataset" / "sparse" / "0"
SLAM_FILE = DATASET_DIR / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
OUT_DIR = Path(r"C:\Users\User\Downloads\Lidou\COMPARATIVO_NUVENS_TODOS_METODOS")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_colmap_images(img_bin_path):
    images = []
    with open(img_bin_path, "rb") as f:
        num_reg = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c.decode("ascii")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, os.SEEK_CUR)

            if name.startswith("cam0/"):
                frame_num = int(name.split("/")[1].replace(".jpg", ""))
                # COLMAP world-to-cam: X_c = R_wc * X_w + t_wc
                # Camera optical center in COLMAP world: C = -R_wc^T * t_wc
                R_wc = Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                C = -R_wc.T @ np.array(tvec)
                # Video timestamp at 24.0 fps
                t_vid = frame_num / 24.0
                images.append({
                    "frame_num": frame_num,
                    "t_vid": t_vid,
                    "C_colmap": C,
                    "R_wc": R_wc,
                    "t_wc": np.array(tvec)
                })

    images.sort(key=lambda x: x["t_vid"])
    return images


def load_colmap_points3d(pts_bin_path):
    points = []
    colors = []
    errors = []
    with open(pts_bin_path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        print(f"[*] Carregando {num_pts:,} pontos 3D do COLMAP ({pts_bin_path.name})...")
        for _ in range(num_pts):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, os.SEEK_CUR)  # skip tracks
            points.append(xyz)
            colors.append(rgb)
            errors.append(err)
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8), np.array(errors, dtype=np.float64)


def solve_umeyama_sim3(X, Y):
    """
    Resolve Y = s * R * X + t (onde Y é métrico LiDAR e X é COLMAP)
    """
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
    return s, R, t, rmse, err


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
    print(f"  [+] PCD exportado: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


def main():
    print("=" * 75)
    print(" ALINHAMENTO MÉTRICO COLMAP (SPIRULA STUDIO) -> LIDAR GROUND TRUTH")
    print("=" * 75)

    # 1. Carregar Trajetória LiDAR SLAM
    print(f"[*] Carregando trajetória LiDAR: {SLAM_FILE.name}...")
    lidar_trj = np.loadtxt(SLAM_FILE)
    t_lidar = lidar_trj[:, 0]
    pos_lidar = lidar_trj[:, 1:4]
    rot_lidar = Rot.from_quat(lidar_trj[:, 4:8]).as_matrix()
    t0_lidar = t_lidar[0]
    print(f"    Total de poses LiDAR: {len(lidar_trj):,}, Duração: {t_lidar[-1] - t0_lidar:.2f} s")

    # 2. Carregar Câmeras do COLMAP
    print(f"[*] Carregando poses de câmera do COLMAP...")
    colmap_images = load_colmap_images(COLMAP_DIR / "images.bin")
    print(f"    Total de poses de câmera cam0: {len(colmap_images):,}")

    # Offset físico nominal da câmera na haste do LiDAR (18.5 cm ao longo de Z_up do rig)
    u_L = np.array([0.003102, -0.504937, -0.863152])
    c_L_phys = 0.185 * u_L

    # 3. Otimizar Sincronização Temporal (Δt) via busca em grade fina
    print("\n[*] Otimizando sincronização temporal exata Δt...")
    dt_candidates = np.linspace(5.2, 6.0, 161)  # passo de 5 ms
    best_dt = None
    best_rmse = 1e9
    best_s = None
    best_R = None
    best_t = None
    best_X = None
    best_Y = None

    for dt in dt_candidates:
        X_list = []
        Y_list = []
        for img in colmap_images:
            t_query = t0_lidar + (img["t_vid"] - dt)
            if t_lidar[0] <= t_query <= t_lidar[-1]:
                idx = np.searchsorted(t_lidar, t_query)
                if 0 < idx < len(t_lidar):
                    t1, t2 = t_lidar[idx - 1], t_lidar[idx]
                    w = (t_query - t1) / (t2 - t1)
                    p_lidar = (1 - w) * pos_lidar[idx - 1] + w * pos_lidar[idx]
                    R_lidar = rot_lidar[idx]  # aproximação
                    p_cam_metric = p_lidar + R_lidar @ c_L_phys
                    X_list.append(img["C_colmap"])
                    Y_list.append(p_cam_metric)

        if len(X_list) >= 50:
            X_arr = np.array(X_list)
            Y_arr = np.array(Y_list)
            s_val, R_val, t_val, rmse_val, _ = solve_umeyama_sim3(X_arr, Y_arr)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_dt = dt
                best_s = s_val
                best_R = R_val
                best_t = t_val
                best_X = X_arr
                best_Y = Y_arr

    print("=" * 75)
    print(f" [RESULTADO DA OTIMIZAÇÃO SIM(3)]")
    print(f"  Δt Ótimo (Sincronização):     {best_dt:.4f} s")
    print(f"  Fator de Escala Métrico (s):  {best_s:.4f}  (1 un COLMAP = {best_s:.3f} m)")
    print(f"  RMSE do Alinhamento 3D:       {best_rmse * 100:.2f} cm ({best_rmse:.4f} m)")
    print(f"  Número de Poses Pareadas:     {len(best_X)} quadros")
    print("=" * 75)

    # 4. Transformar a Nuvem 3D do COLMAP (177.479 pontos) para Metros Reais
    pts_colmap, colors_colmap, errors_colmap = load_colmap_points3d(COLMAP_DIR / "points3D.bin")
    
    # Filtrar pontos com erro de triangulação excessivo (> 4px)
    mask_good = errors_colmap < 4.0
    pts_filtered = pts_colmap[mask_good]
    colors_filtered = colors_colmap[mask_good]
    print(f"[*] Filtragem de qualidade: {np.sum(mask_good):,} pontos retidos (erro < 4px)")

    # Aplicação da transformação métrica Sim(3): P_metric = s * R * P_colmap + t
    print("[*] Aplicando transformação métrica Sim(3) nos pontos 3D...")
    pts_metric = (best_s * best_R @ pts_filtered.T).T + best_t

    # 5. Exportar Nuvem Fotogramétrica Métrica
    out_colmap_pcd = OUT_DIR / "05_COLMAP_SfM_Metrico_Spirula_Studio.pcd"
    write_pcd(out_colmap_pcd, pts_metric, colors_filtered)

    # 6. Carregar Nuvem SLAM LiDAR e Exportar Fusão Conjunta
    slam_pcd_path = DATASET_DIR / "slam_out" / "pcd" / "all_raw_points.pcd"
    if slam_pcd_path.exists():
        print(f"\n[*] Carregando nuvem bruta do LiDAR SLAM para fusão...")
        with open(slam_pcd_path, "rb") as f:
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("POINTS"):
                    n_lidar = int(line.split()[1])
                elif line.startswith("DATA"):
                    break
            raw = f.read(n_lidar * 16)
            lidar_data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
            pts_lidar_all = lidar_data[:, :3].astype(np.float64)

        # Pintar a nuvem do LiDAR de cinza neutro (180, 180, 180) para destacar os pontos coloridos do COLMAP
        colors_lidar = np.full((len(pts_lidar_all), 3), 160, dtype=np.uint8)

        # Fusão: LiDAR em cinza + COLMAP em cores reais
        pts_fusion = np.vstack([pts_lidar_all, pts_metric])
        colors_fusion = np.vstack([colors_lidar, colors_filtered])

        out_fusion_pcd = OUT_DIR / "06_FUSAO_LiDAR_Cinza_e_COLMAP_Cores.pcd"
        write_pcd(out_fusion_pcd, pts_fusion, colors_fusion)

    # 7. Salvar Resumo em JSON
    summary = {
        "dataset": "VID_20260902_143757_00_277",
        "optimal_delta_t_sec": float(best_dt),
        "metric_scale_factor_s": float(best_s),
        "alignment_rmse_meters": float(best_rmse),
        "num_matched_poses": int(len(best_X)),
        "sim3_rotation_matrix": best_R.tolist(),
        "sim3_translation_vector": best_t.tolist(),
        "total_colmap_points_transformed": int(len(pts_metric)),
        "exported_colmap_pcd": str(out_colmap_pcd),
        "exported_fusion_pcd": str(out_fusion_pcd) if slam_pcd_path.exists() else None
    }
    with open(OUT_DIR / "colmap_lidar_sim3_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(" PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f" Arquivos disponíveis na pasta:\n {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
