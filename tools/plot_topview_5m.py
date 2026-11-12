#!/usr/bin/env python3
"""
Render top-down view sliced at ~5 meters height (cutting off ceiling/roof) to reveal floor plan,
walkways, and store interiors in full color.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def render_topview_5m(ply_path: Path, trj_path: Path, out_path: Path, title: str, z_cut: float = 4.5, voxel_size: float = 0.05):
    print(f"[*] Loading {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    
    # Slice ceiling above z_cut (e.g. ~5 meters above floor level)
    # Floor median is around ~-0.5 to 0.0m, so z_cut relative to trajectory / floor
    print(f"[*] Slicing points below Z = {z_cut:.2f} m...")
    mask = pts[:, 2] <= z_cut
    pts_sliced = pts[mask]
    cols_sliced = cols[mask]
    
    trj_pts = []
    if trj_path.exists():
        with open(trj_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        trj_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
    trj = np.array(trj_pts) if len(trj_pts) > 0 else np.zeros((0, 3))
    
    fig, ax = plt.subplots(figsize=(18, 14), dpi=220, facecolor='#111115')
    ax.set_facecolor('#141418')
    
    # Subsample if points > 1.2M for crisp rendering
    if len(pts_sliced) > 1_500_000:
        idx = np.random.choice(len(pts_sliced), 1_500_000, replace=False)
        p_plot = pts_sliced[idx]
        c_plot = cols_sliced[idx]
    else:
        p_plot = pts_sliced
        c_plot = cols_sliced
        
    ax.scatter(p_plot[:, 0], p_plot[:, 1], c=c_plot, s=1.0, alpha=0.9, rasterized=True)
    
    if len(trj) > 0:
        ax.plot(trj[:, 0], trj[:, 1], color='#00ffcc', linewidth=2.0, alpha=0.85, label='Trajetória SLAM', zorder=4)
        ax.scatter(trj[0, 0], trj[0, 1], color='#33ff33', s=60, zorder=5, label='Início')
        ax.scatter(trj[-1, 0], trj[-1, 1], color='#ff3333', s=60, zorder=5, label='Fim')
        
    ax.set_title(f"{title} - Top View (Corte a ~5m de altura sem teto)", color='white', fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('X (metros)', color='white', fontsize=12)
    ax.set_ylabel('Y (metros)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.grid(True, color='#282832', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', facecolor='#1e1e24', edgecolor='#444450', labelcolor='white', fontsize=11)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=220, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved topview to: {out_path}")

if __name__ == '__main__':
    # Generate for HarmonyMall
    render_topview_5m(
        Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT/deliverables/lidar_colored_direct_rigid_20261021_211521.ply'),
        Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT/slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'),
        Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/harmonymall_topview_5m.png'),
        "HarmonyMall",
        z_cut=4.0,
        voxel_size=0.08
    )
    
    # Generate for PIER
    render_topview_5m(
        Path('D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output/deliverables/lidar_colored_direct_rigid_20261021_213644.ply'),
        Path('D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output/slam_raw_f8a83ade028dd052024e/result/Raven_3DMakerPro_Scan.txt'),
        Path('D:/ARQUIVOS_TESTE_2/PIER/pier_topview_5m.png'),
        "PIER",
        z_cut=4.0,
        voxel_size=0.03
    )
