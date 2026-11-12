#!/usr/bin/env python3
"""
Render ultra-high-resolution close-up top views (5m cut) showing detailed colorization,
floor tiles, storefronts, and deck planks up close.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def render_closeup(
    ply_path: Path,
    trj_path: Path,
    out_path: Path,
    title: str,
    center_xy: tuple[float, float],
    span_meters: float = 12.0,
    z_min: float = -3.0,
    z_max: float = 4.0,
    point_size: float = 4.0
):
    print(f"[*] Loading point cloud: {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    
    cx, cy = center_xy
    half_w = span_meters / 2.0
    
    # Crop tight window
    mask = (
        (pts[:, 0] >= cx - half_w) & (pts[:, 0] <= cx + half_w) &
        (pts[:, 1] >= cy - half_w) & (pts[:, 1] <= cy + half_w) &
        (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    )
    pts_sub = pts[mask]
    cols_sub = cols[mask]
    print(f"[+] Window [{cx:.1f}, {cy:.1f}] +/- {half_w:.1f}m: {len(pts_sub):,} points retained.")
    
    trj = np.loadtxt(str(trj_path))
    traj_xyz = trj[:, 1:4]
    mask_trj = (
        (traj_xyz[:, 0] >= cx - half_w) & (traj_xyz[:, 0] <= cx + half_w) &
        (traj_xyz[:, 1] >= cy - half_w) & (traj_xyz[:, 1] <= cy + half_w)
    )
    sub_trj = traj_xyz[mask_trj]
    
    fig, ax = plt.subplots(figsize=(12, 12), dpi=260, facecolor='#111115')
    ax.set_facecolor('#16161c')
    
    ax.scatter(pts_sub[:, 0], pts_sub[:, 1], c=cols_sub, s=point_size, alpha=0.95, rasterized=True)
    
    if len(sub_trj) > 0:
        ax.plot(sub_trj[:, 0], sub_trj[:, 1], color='#00ffcc', linewidth=3.0, alpha=0.9, label='Trajetória do Scanner', zorder=5)
        ax.scatter(sub_trj[0, 0], sub_trj[0, 1], color='#33ff33', s=90, zorder=6, edgecolors='white', linewidth=1.5, label='Início da Seção')
        ax.scatter(sub_trj[-1, 0], sub_trj[-1, 1], color='#ff3333', s=90, zorder=6, edgecolors='white', linewidth=1.5, label='Fim da Seção')
        
    ax.set_title(f"{title} - Close-up Top View ({span_meters:.0f}m x {span_meters:.0f}m)", color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('X (metros)', color='white', fontsize=11)
    ax.set_ylabel('Y (metros)', color='white', fontsize=11)
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.set_xlim(cx - half_w, cx + half_w)
    ax.set_ylim(cy - half_w, cy + half_w)
    ax.grid(True, color='#2c2c36', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', facecolor='#202028', edgecolor='#555566', labelcolor='white', fontsize=10)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=260, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Saved close-up zoom to: {out_path}")

if __name__ == '__main__':
    # HarmonyMall Corridor & Store Close-up (10m x 10m window)
    render_closeup(
        Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT/deliverables/lidar_colored_direct_rigid_20261021_211521.ply'),
        Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT/slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'),
        Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/harmonymall_topview_closeup_10m.png'),
        title="HarmonyMall (Corredor & Vitrines)",
        center_xy=(2.5, -20.0),
        span_meters=12.0,
        z_min=-2.0,
        z_max=4.0,
        point_size=5.0
    )
    
    # PIER Deck Walkway & Railings Close-up (8m x 8m window)
    render_closeup(
        Path('D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output/deliverables/lidar_colored_direct_rigid_20261021_213644.ply'),
        Path('D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output/slam_raw_f8a83ade028dd052024e/result/Raven_3DMakerPro_Scan.txt'),
        Path('D:/ARQUIVOS_TESTE_2/PIER/pier_topview_closeup_8m.png'),
        title="PIER (Deque de Madeira & Guarda-Corpos)",
        center_xy=(6.0, -8.0),
        span_meters=10.0,
        z_min=-2.0,
        z_max=3.5,
        point_size=6.0
    )
