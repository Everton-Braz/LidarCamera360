#!/usr/bin/env python3
"""
Render Close-up Centered at the Start of Trajectory (0, 0)
Using the Colored Point Cloud and High-Density SLAM Trajectory
"""
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    dataset_dir = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
    out_dir = dataset_dir / 'orthogonal_views'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ply_path = dataset_dir / 'deliverables/lidar_colored_direct_rigid_20261021_211521.ply'
    trj_path = dataset_dir / 'slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'
    
    print('[*] Loading trajectory...')
    trj = np.loadtxt(trj_path)
    pos_slam = trj[:, 1:4]
    start_pos = pos_slam[0]
    cx, cy = float(start_pos[0]), float(start_pos[1])
    print(f'    Start position: ({cx:.4f}, {cy:.4f}, {start_pos[2]:.4f})')
    
    print(f'[*] Loading colored point cloud: {ply_path.name}...')
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    # Crop around start position: 10m half-width (20m total window)
    span_m = 10.0
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=np.array([cx - span_m, cy - span_m, -4.0]),
        max_bound=np.array([cx + span_m, cy + span_m, 6.0])
    )
    pcd_crop = pcd.crop(bbox)
    pcd_down = pcd_crop.voxel_down_sample(voxel_size=0.03)
    
    pts = np.asarray(pcd_down.points, dtype=np.float32)
    cols = np.asarray(pcd_down.colors, dtype=np.float32)
    print(f'    Points in {span_m*2}m window: {len(pts):,}')
    
    valid_color = np.sum(cols, axis=1) > 0.08
    pts_val = pts[valid_color]
    cols_val = cols[valid_color]
    
    # Filter trajectory points near start
    mask_trj = (
        (pos_slam[:, 0] >= cx - span_m) & (pos_slam[:, 0] <= cx + span_m) &
        (pos_slam[:, 1] >= cy - span_m) & (pos_slam[:, 1] <= cy + span_m)
    )
    sub_trj = pos_slam[mask_trj]
    
    bg_color = '#0e0e13'
    panel_bg = '#14141c'
    grid_color = '#252533'
    
    fig, ax = plt.subplots(figsize=(14, 14), dpi=260, facecolor=bg_color)
    ax.set_facecolor(panel_bg)
    
    ax.scatter(pts_val[:, 0], pts_val[:, 1], c=cols_val, s=2.8, alpha=0.95, rasterized=True)
    ax.plot(sub_trj[:, 0], sub_trj[:, 1], color='#00f0ff', linewidth=2.8, label='Trajetória LiDAR SLAM', zorder=10)
    ax.scatter(start_pos[0], start_pos[1], color='#00ff66', s=180, marker='o', edgecolors='white', linewidth=2.0, label='Início Trajetória (Centro)', zorder=12)
    
    ax.set_title('HarmonyMall: Close-up no Início da Trajetória (Janela 20x20m)', color='white', fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('X - Lateral (metros)', color='white', fontsize=12)
    ax.set_ylabel('Y - Longitudinal (metros)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.set_xlim(cx - span_m, cx + span_m)
    ax.set_ylim(cy - span_m, cy + span_m)
    ax.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)
    
    out_path = out_dir / 'harmonymall_start_closeup_20m.png'
    plt.savefig(str(out_path), dpi=260, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[SUCCESS] Close-up plot saved: {out_path}')

if __name__ == '__main__':
    main()
