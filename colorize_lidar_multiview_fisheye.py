#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coloração Multi-Visão da Nuvem LiDAR com Imagens Dual-Fisheye 360° (Thin Prism + ICP)
==================================================================================
1. Carrega a nuvem LiDAR métrica completa (2.336.721 pontos).
2. Transforma a nuvem para o referencial de câmeras do COLMAP via Sim(3) + ICP.
3. Projeta os pontos em todas as fotos olho de peixe calibradas (cam0 e cam1).
4. Realiza teste de oclusão por Z-Buffer raster (960x960) para evitar projeções através de paredes.
5. Seleciona a cor da foto com máxima nitidez ótica (menor r ao centro e menor distância d).
6. Exporta a nuvem colorida final em formatos PCD e PLY de alta performance.
"""

import os
import sys
import time
import json
import struct
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding='utf-8')

# Caminhos principais
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib")
COLMAP_DIR = DATASET_DIR / "VID_20260902_143757_00_277_dataset" / "sparse" / "0"
IMG_ROOT = DATASET_DIR / "VID_20260902_143757_00_277_dataset" / "images"
SLAM_PCD = DATASET_DIR / "slam_out" / "pcd" / "all_raw_points.pcd"

OUT_DIR = Path(r"C:\Users\User\Downloads\Lidou\COMPARATIVO_NUVENS_TODOS_METODOS")
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_colmap_cameras(cameras_bin_path):
    cams = {}
    with open(cameras_bin_path, "rb") as f:
        num_cams = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cams):
            cid, mid, w, h = struct.unpack("<IIQQ", f.read(24))
            params = struct.unpack("<12d", f.read(96))
            cams[cid] = {
                "model_id": mid,
                "width": w,
                "height": h,
                "params": params
            }
    return cams


def load_colmap_images(images_bin_path):
    images = []
    with open(images_bin_path, "rb") as f:
        num_reg = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg):
            iid = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            cid = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c.decode("ascii")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, os.SEEK_CUR)

            R_cw = Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
            t_cw = np.array(tvec)
            
            images.append({
                "id": iid,
                "name": name,
                "cid": cid,
                "R_cw": R_cw,
                "t_cw": t_cw
            })
    return images


def project_thin_prism(P_v, params):
    """
    Projeção analítica do modelo THIN_PRISM_FISHEYE (COLMAP Modelo 10)
    P_v: array Nx3 onde Z > 0
    params: (fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1)
    """
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


def main():
    print("=" * 80)
    print(" COLORACAO MULTI-VISAO LIDAR 360 COM CALIBRACAO THIN PRISM + SIM(3) + ICP")
    print("=" * 80)
    t_start = time.time()

    # 1. Carregar Calibração Composta Sim(3) + ICP
    sim3_summary_file = OUT_DIR / "colmap_lidar_sim3_alignment_summary.json"
    icp_res_file = OUT_DIR / "calibracao_automatica_icp_resultado.json"

    if not sim3_summary_file.exists() or not icp_res_file.exists():
        print("[-] Arquivos de calibração não encontrados.")
        return

    with open(sim3_summary_file, "r") as f:
        s3 = json.load(f)
    with open(icp_res_file, "r") as f:
        icp = json.load(f)

    s = s3["metric_scale_factor_s"]
    R_s3 = np.array(s3["sim3_rotation_matrix"])
    t_s3 = np.array(s3["sim3_translation_vector"])

    eul_icp = [
        icp["icp_fine_trims_deg"]["pitch_trim"],
        icp["icp_fine_trims_deg"]["yaw_trim"],
        icp["icp_fine_trims_deg"]["roll_trim"]
    ]
    R_icp = Rot.from_euler("xyz", eul_icp, degrees=True).as_matrix()
    t_icp = np.array(icp["icp_residual_translation_meters"])

    R_tot = R_icp @ R_s3
    t_tot = R_icp @ t_s3 + t_icp

    print(f"[*] Transformação Composta Carregada:")
    print(f"    Escala s: {s:.6f}")
    print(f"    Translação Total: DX={t_tot[0]:+.3f}m, DY={t_tot[1]:+.3f}m, DZ={t_tot[2]:+.3f}m")
    print(f"    Rotação Total Euler XYZ: {Rot.from_matrix(R_tot).as_euler('xyz', degrees=True)}")

    # 2. Carregar a Nuvem de Pontos Bruta do LiDAR (Ground Truth Métrico)
    pts_lidar = load_pcd(SLAM_PCD)
    n_pts = len(pts_lidar)

    # 3. Pré-computar todos os pontos no mundo COLMAP:
    # P_col = (1 / s) * R_tot^T * (P_lidar - t_tot)
    print("\n[*] Mapeando nuvem LiDAR para o referencial de câmeras do COLMAP...")
    pts_col = (1.0 / s) * (R_tot.T @ (pts_lidar - t_tot).T).T
    print("    Nuvem pré-transformada com sucesso!")

    # 4. Carregar Parâmetros Ópticos e Imagens Registradas
    cams = load_colmap_cameras(COLMAP_DIR / "cameras.bin")
    all_images = load_colmap_images(COLMAP_DIR / "images.bin")

    # Separar imagens existentes no disco
    images_valid = []
    for img in all_images:
        path = IMG_ROOT / img["name"]
        if path.exists():
            images_valid.append(img)

    images_valid.sort(key=lambda x: x["name"])
    print(f"[*] Total de fotos calibradas no disco: {len(images_valid):,} (cam0 e cam1)")

    # Amostragem inteligente de quadros (step=2 pega ~190 quadros, cobrindo cada centímetro da trajetória)
    # Isso garante cobertura completa 360° com altíssima velocidade
    step = 2
    keyframes = images_valid[::step]
    print(f"[*] Quadros selecionados para coloração (step={step}): {len(keyframes):,} fotos de alta resolução")

    # Arrays de cor e pontuação de qualidade
    # Cor padrão: cinza neutro (180, 180, 180) para pontos que não aparecem em nenhuma foto
    colors = np.full((n_pts, 3), 180, dtype=np.uint8)
    best_scores = np.zeros(n_pts, dtype=np.float32)

    # 5. Loop de Projeção Multi-Visão
    print("\n" + "=" * 80)
    print(" INICIANDO PROJEÇÃO ÓPTICA E AMOSTRAGEM FOTOGRÁFICA...")
    print("=" * 80)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w  # 4.0

    t_loop_start = time.time()
    for idx_kf, kf in enumerate(keyframes):
        img_name = kf["name"]
        img_path = IMG_ROOT / img_name
        cid = kf["cid"]
        R_cw = kf["R_cw"]
        t_cw = kf["t_cw"]

        # 1. Mapear pontos para o referencial óptico da câmera kf
        P_cam = (R_cw @ pts_col.T).T + t_cw
        z_mask = P_cam[:, 2] > 0.15  # Apenas pontos na frente da lente
        idx_valid = np.where(z_mask)[0]
        if len(idx_valid) == 0:
            continue

        P_v = P_cam[idx_valid]
        d = np.linalg.norm(P_v, axis=1)

        # 2. Projeção Thin Prism
        cam_meta = cams[cid]
        params = cam_meta["params"]
        cx, cy = params[2], params[3]
        u, v, r = project_thin_prism(P_v, params)

        # 3. Máscara de abertura circular (raio útil da lente olho de peixe < 1650 px)
        r_px = np.hypot(u - cx, v - cy)
        mask_circle = (r_px < 1650.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
        if not np.any(mask_circle):
            continue

        u_c = u[mask_circle]
        v_c = v[mask_circle]
        d_c = d[mask_circle]
        r_c = r_px[mask_circle]
        idx_c = idx_valid[mask_circle]

        # 4. Teste de Oclusão por Z-Buffer Raster (960x960)
        ug = np.clip(np.floor(u_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_w - 1)
        vg = np.clip(np.floor(v_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_h - 1)
        flat_idx = vg * zbuf_w + ug

        dmap = np.full(zbuf_w * zbuf_h, 1e9, dtype=np.float32)
        np.minimum.at(dmap, flat_idx, d_c.astype(np.float32))

        # Visível se d <= d_min * 1.08 + 0.15m (margem de tolerância para superfícies rugosas)
        vis_mask = d_c <= (dmap[flat_idx] * 1.08 + 0.15)
        if not np.any(vis_mask):
            continue

        u_vis = np.round(u_c[vis_mask]).astype(int)
        v_vis = np.round(v_c[vis_mask]).astype(int)
        d_vis = d_c[vis_mask]
        r_vis = r_c[vis_mask]
        idx_vis = idx_c[vis_mask]

        # 5. Métrica de Nitidez Óptica (pontos mais próximos e mais no centro da lente ganham prioridade)
        # Score = (1 - r/1650) / max(dist, 0.5)
        scores = (1.0 - (r_vis / 1650.0)) / np.maximum(d_vis, 0.5)

        # 6. Atualizar apenas os pontos onde a foto atual tem pontuação maior que a anterior
        better_mask = scores > best_scores[idx_vis]
        if not np.any(better_mask):
            continue

        idx_update = idx_vis[better_mask]
        u_update = u_vis[better_mask]
        v_update = v_vis[better_mask]
        scores_update = scores[better_mask]

        # 7. Amostragem de Pixel da Imagem Original
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        colors[idx_update] = img_rgb[v_update, u_update]
        best_scores[idx_update] = scores_update

        # Progresso
        if (idx_kf + 1) % 15 == 0 or idx_kf == len(keyframes) - 1:
            n_colored = np.sum(best_scores > 0)
            pct = (n_colored / n_pts) * 100.0
            elapsed = time.time() - t_loop_start
            fps = (idx_kf + 1) / elapsed
            print(f"  [{idx_kf+1:3d}/{len(keyframes):3d}] Foto: {img_name:15s} | "
                  f"Coloridos: {n_colored:,}/{n_pts:,} ({pct:.1f}%) | {fps:.1f} fotos/s")

    # 6. Estatísticas Finais de Coloração
    n_colored_final = np.sum(best_scores > 0)
    pct_final = (n_colored_final / n_pts) * 100.0
    print("\n" + "=" * 80)
    print(" RESULTADO DA COLORACAO:")
    print(f"  Total de Pontos no LiDAR: {n_pts:,}")
    print(f"  Pontos 100% Coloridos:    {n_colored_final:,} ({pct_final:.2f}%)")
    print(f"  Tempo Total de Projeção:  {time.time() - t_loop_start:.1f} segundos")
    print("=" * 80)

    # 7. Exportar Nuvens Finais
    print("\n[*] Exportando arquivos finais de alta resolução...")
    out_pcd = OUT_DIR / "09_NUVEM_LIDAR_COLORIDA_MULTIVIEW_FISHEYE.pcd"
    out_ply = OUT_DIR / "09_NUVEM_LIDAR_COLORIDA_MULTIVIEW_FISHEYE.ply"

    write_pcd(out_pcd, pts_lidar, colors)
    write_ply(out_ply, pts_lidar, colors)

    # 8. Salvar Metadados da Coloração
    summary = {
        "metodo": "Coloracao_MultiVisao_DualFisheye_ThinPrism_ICP",
        "data_execucao": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_pontos_lidar": int(n_pts),
        "pontos_coloridos": int(n_colored_final),
        "taxa_cobertura_percent": float(pct_final),
        "fotos_utilizadas": int(len(keyframes)),
        "modelo_lente": "THIN_PRISM_FISHEYE",
        "resolucao_fotos": "3840x3840",
        "raio_util_abertura_px": 1650.0,
        "tempo_processamento_s": float(time.time() - t_start),
        "arquivos_gerados": {
            "pcd": str(out_pcd),
            "ply": str(out_ply)
        }
    }
    with open(OUT_DIR / "relatorio_coloracao_final.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" PROCESSO 100% CONCLUIDO COM SUCESSO!")
    print(f" Arquivos disponíveis na pasta:\n {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
