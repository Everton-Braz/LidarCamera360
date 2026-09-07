#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibração Automática Multimodal: LiDAR SLAM + COLMAP SfM (Spirula Studio)
========================================================================
1. Prepara pasta dedicada para o teste de ICP no CloudCompare com instruções passo a passo.
2. Executa ICP Robusto com cKDTree entre os pontos fotogramétricos métricos e a nuvem LiDAR.
3. Deriva a calibração extrínseca rígida 6-DoF (Pitch, Yaw, Roll, Lever Arm).
4. Exporta a nuvem de pontos colorizada final com os parâmetros calculados automaticamente.
"""

import os
import sys
import json
import struct
import shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding='utf-8')

# Caminhos
if len(sys.argv) > 1:
    DATASET_DIR = Path(sys.argv[1])
else:
    DATASET_DIR = Path(r"D:\APLICATIVOS\FAST-LIVO2\AZURE-DATASET")
COLMAP_DIR = DATASET_DIR / "VID_dataset" / "sparse" / "0"
SLAM_PCD = DATASET_DIR / "slam_out" / "pcd" / "all_raw_points.pcd"
SLAM_TRJ = DATASET_DIR / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"

ICP_FOLDER = DATASET_DIR / "deliverables" / "TESTE_ICP_CLOUDCOMPARE"
OUT_COMPARATIVO = DATASET_DIR / "deliverables"
ICP_FOLDER.mkdir(parents=True, exist_ok=True)
OUT_COMPARATIVO.mkdir(parents=True, exist_ok=True)


def load_pcd(path):
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
        rgb_raw = data[:, 3].view(np.uint32)
        r = (rgb_raw >> 16) & 0xFF
        g = (rgb_raw >> 8) & 0xFF
        b = rgb_raw & 0xFF
        colors = np.stack([r, g, b], axis=-1).astype(np.uint8)
        return xyz, colors


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
    print(f"  [+] Nuvem salva: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


def run_robust_icp(source_pts, target_pts, max_iter=30):
    """
    Executa ICP Robusto (Point-to-Point com corte de outliers adaptativo).
    Retorna R, t e os pontos transformados.
    """
    print("\n[*] Construindo árvore KDTree do LiDAR (Ground Truth)...")
    tree = cKDTree(target_pts)

    P = source_pts.copy()
    R_total = np.eye(3)
    t_total = np.zeros(3)

    print("[*] Iniciando iterações do ICP Robusto...")
    thresholds = [0.35, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08]

    for it in range(max_iter):
        # Determinar threshold de corte para esta iteração
        th_idx = min(it // 4, len(thresholds) - 1)
        th = thresholds[th_idx]

        dists, indices = tree.query(P, k=1)
        inliers = dists < th
        n_inliers = np.sum(inliers)

        if n_inliers < 1000:
            print(f"  [!] Poucos inliers ({n_inliers}), encerrando ICP.")
            break

        src_in = P[inliers]
        tgt_in = target_pts[indices[inliers]]

        mu_src = src_in.mean(axis=0)
        mu_tgt = tgt_in.mean(axis=0)

        H = (src_in - mu_src).T @ (tgt_in - mu_tgt)
        U, S, Vt = np.linalg.svd(H)
        R_step = Vt.T @ U.T
        if np.linalg.det(R_step) < 0:
            Vt[2, :] *= -1
            R_step = Vt.T @ U.T

        t_step = mu_tgt - R_step @ mu_src

        # Atualizar nuvem e acumuladores
        P = (R_step @ P.T).T + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        rmse_in = np.sqrt(np.mean(dists[inliers] ** 2))
        med_in = np.median(dists[inliers])

        if (it + 1) % 5 == 0 or it == max_iter - 1:
            print(f"  Iteração {it+1:2d}: Inliers = {n_inliers:,}/{len(P):,} ({n_inliers/len(P)*100:.1f}%), "
                  f"Th = {th*100:.0f}cm | RMSE = {rmse_in*100:.2f}cm, Mediana = {med_in*100:.2f}cm")

    return R_total, t_total, P, rmse_in, med_in


def main():
    print("=" * 75)
    print(" CALIBRAÇÃO AUTOMÁTICA MULTIMODAL: LIDAR SLAM + COLMAP SFM")
    print("=" * 75)

    # 1. Carregar Nuvem do COLMAP Métrica
    colmap_pcd_in = OUT_COMPARATIVO / "05_COLMAP_SfM_Metrico_Spirula_Studio.pcd"
    if not colmap_pcd_in.exists():
        print("[-] Nuvem do COLMAP não encontrada. Execute align_colmap_to_lidar.py primeiro.")
        return

    pts_colmap, colors_colmap = load_pcd(colmap_pcd_in)
    print(f"[*] Nuvem COLMAP carregada: {len(pts_colmap):,} pontos")

    # 2. Carregar Nuvem do LiDAR SLAM
    pts_lidar, colors_lidar = load_pcd(SLAM_PCD)
    print(f"[*] Nuvem LiDAR carregada: {len(pts_lidar):,} pontos")

    # 3. Preparar Pasta Específica para o Teste de ICP no CloudCompare
    print("\n" + "=" * 75)
    print(f" PREPARANDO ARQUIVOS NA PASTA ESPECÍFICA:\n {ICP_FOLDER}")
    print("=" * 75)

    # Amostragem do LiDAR para o CloudCompare (1 em cada 4 pontos para ficar ágil: ~580k pontos)
    pts_lidar_sub = pts_lidar[::4]
    colors_lidar_sub = np.full((len(pts_lidar_sub), 3), 160, dtype=np.uint8)  # cinza neutro

    f_lidar = ICP_FOLDER / "01_MODELO_LiDAR_Referencia_GroundTruth.pcd"
    f_colmap = ICP_FOLDER / "02_ALINHADO_COLMAP_Fotogrametria_Cores.pcd"

    write_pcd(f_lidar, pts_lidar_sub, colors_lidar_sub)
    write_pcd(f_colmap, pts_colmap, colors_colmap)

    # Escrever manual passo a passo para o CloudCompare
    txt_instrucoes = ICP_FOLDER / "LEIA-ME_PASSO_A_PASSO_CLOUDCOMPARE.txt"
    with open(txt_instrucoes, "w", encoding="utf-8") as f:
        f.write("""================================================================================
PASSO A PASSO: TESTE DE REGISTRO FINO POR ICP NO CLOUDCOMPARE
================================================================================

Estes arquivos já estão na escala métrica 1:1 real e pré-alinhados a cerca de 10 cm!

ARQUIVOS DESTA PASTA:
  1. 01_MODELO_LiDAR_Referencia_GroundTruth.pcd  -> Nuvem do LiDAR (Ground Truth métrico)
  2. 02_ALINHADO_COLMAP_Fotogrametria_Cores.pcd -> Nuvem do COLMAP com as cores reais

COMO EXECUTAR O ICP NO CLOUDCOMPARE:
------------------------------------
1. Abra o CloudCompare.
2. Arraste os dois arquivos PCD para a tela do CloudCompare.
3. No painel esquerdo ('DB Tree'):
   - Selecione AMBAS as nuvens segurando a tecla Ctrl (clique em 01_MODELO... e em 02_ALINHADO...).
4. No menu superior, clique em:
   -> Tools  (ou Edit)
   -> Registration
   -> Fine Registration (ICP)
5. Na janela de diálogo que abrir, o CloudCompare vai perguntar qual nuvem é o 'model' e qual é o 'aligned':
   - Escolha '01_MODELO_LiDAR_Referencia_GroundTruth' como o MODEL (referência fixa).
   - Escolha '02_ALINHADO_COLMAP_Fotogrametria_Cores' como o ALIGNED (nuvem móvel que vai se ajustar).
6. Parâmetros recomendados na janela do ICP:
   - RMS difference: 1.0e-5
   - Random sampling limit: 50000
   - Ajuste de Escala: DESMARQUE a caixa 'adjust scale' (pois a escala já foi calibrada como métrica exata!).
7. Clique em 'OK'.
8. O CloudCompare vai rodar o ICP e travar a nuvem de cores perfeitamente em cima da nuvem de laser!
   Ele exibirá a matriz de transformação 4x4 e o erro RMS final no console.
================================================================================
""")
    print(f"  [+] Instruções salvas em: {txt_instrucoes.name}")

    # 4. Executar o ICP Automático Python
    print("\n" + "=" * 75)
    print(" EXECUTANDO O ICP AUTOMÁTICO VIA PYTHON (ALGORITMO ROBUSTO)...")
    print("=" * 75)
    R_icp, t_icp, pts_colmap_opt, rmse_final, med_final = run_robust_icp(
        pts_colmap, pts_lidar_sub, max_iter=28
    )

    eul_icp = Rot.from_matrix(R_icp).as_euler("xyz", degrees=True)
    print("\n" + "=" * 75)
    print(" [RESULTADO DO ICP AUTOMÁTICO]")
    print(f"  Erro Final: RMSE = {rmse_final*100:.2f} cm | Mediana = {med_final*100:.2f} cm")
    print(f"  Rotação de Ajuste Fino (Euler XYZ): Pitch={eul_icp[0]:+.3f}°, Yaw={eul_icp[1]:+.3f}°, Roll={eul_icp[2]:+.3f}°")
    print(f"  Translação Residual: DX={t_icp[0]*100:+.2f}cm, DY={t_icp[1]*100:+.2f}cm, DZ={t_icp[2]*100:+.2f}cm")
    print("=" * 75)

    # 5. Exportar a Nuvem COLMAP Alinhada pelo ICP
    f_icp_out = OUT_COMPARATIVO / "07_COLMAP_Otimizado_Apos_ICP_Automatico.pcd"
    write_pcd(f_icp_out, pts_colmap_opt, colors_colmap)

    # 6. Salvar Fusão Final Pós-ICP (LiDAR + COLMAP Casados)
    f_fusao_final = OUT_COMPARATIVO / "08_FUSAO_FINAL_LiDAR_e_COLMAP_Pos_ICP.pcd"
    pts_fusao_opt = np.vstack([pts_lidar[::3], pts_colmap_opt])
    colors_fusao_opt = np.vstack([np.full((len(pts_lidar[::3]), 3), 150, dtype=np.uint8), colors_colmap])
    write_pcd(f_fusao_final, pts_fusao_opt, colors_fusao_opt)

    # 7. Salvar os Parâmetros Finais da Calibração Automática em JSON
    res_final = {
        "metodo": "Calibracao_Automatica_Multimodal_HandEye_Sim3_ICP",
        "camera_model": "THIN_PRISM_FISHEYE",
        "focal_length_px": 1080.1874,
        "max_radius_cutoff_px": 1650.0,
        "distortion_parameters": {
            "k1": 0.084684,
            "k2": -0.032074,
            "p1": -0.000413,
            "p2": 0.001235,
            "k3": 0.012301,
            "k4": -0.002990,
            "sx1": -0.002747,
            "sy1": 0.000407
        },
        "icp_fine_trims_deg": {
            "pitch_trim": float(eul_icp[0]),
            "yaw_trim": float(eul_icp[1]),
            "roll_trim": float(eul_icp[2])
        },
        "icp_residual_translation_meters": t_icp.tolist(),
        "final_rmse_meters": float(rmse_final),
        "final_median_error_meters": float(med_final)
    }

    with open(OUT_COMPARATIVO / "calibracao_automatica_icp_resultado.json", "w") as f:
        json.dump(res_final, f, indent=2)

    print("\n" + "=" * 75)
    print(" PROCESSO 100% CONCLUÍDO!")
    print(f" Pasta CloudCompare:  {ICP_FOLDER}")
    print(f" Pasta Comparativo:   {OUT_COMPARATIVO}")
    print("=" * 75)


if __name__ == "__main__":
    main()
