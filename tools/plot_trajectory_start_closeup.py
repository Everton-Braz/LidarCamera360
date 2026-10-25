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
    cx, cy = start_pos[0], start_pos[1]
    print(f'    Start position: ({cx:.4f}, {cy:.4f}, {start_pos[2]:.4f})')
    
    print('[*] Loading colored point cloud...')
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    cols_all = np.asarray(pcd.colors, dtype=np.float32)
    
    # Crop high-density window around start position (span 12 meters, radius 6m)
    hw = 6.0
    mask = (
        (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) &
        (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) &
        (pts_all[:, 2] >= -2.5) & (pts_all[:, 2] <= 4.0)
    )
    pts = pts_all[mask]
    cols = cols_all[mask]
    print(f'    Points in {2*hw}m window around start: {len(pts):,}')
    
    # Filter uncolored points
    valid = np.sum(cols, axis=1) > 0.05
    pts = pts[valid]
    cols = cols[valid]
    print(f'    Colored points: {len(pts):,}')
    
    # Trajectory points within the window + margin
    traj_mask = (
        (pos_slam[:, 0] >= cx - hw - 2.0) & (pos_slam[:, 0] <= cx + hw + 2.0) &
        (pos_slam[:, 1] >= cy - hw - 2.0) & (pos_slam[:, 1] <= cy + hw + 2.0)
    )
    traj_segment = pos_slam[traj_mask]
    
    # Color and visual theme
    bg_color = '#0e0e13'
    panel_bg = '#14141c'
    grid_color = '#252533'
    
    # 1. Close-up 12m view
    fig, ax = plt.subplots(figsize=(14, 14), dpi=280, facecolor=bg_color)
    ax.set_facecolor(panel_bg)
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=2.8, alpha=0.95, rasterized=True)
    
    # Plot trajectory with directional arrows / progressive line
    ax.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=3.0, label='Trajetória SLAM', zorder=10)
    
    # Mark Start precisely at the center
    ax.scatter(cx, cy, color='#00ff66', s=250, marker='o', edgecolors='white', linewidth=2.5, label=f'Início Trajetória ({cx:.2f}, {cy:.2f})', zorder=15)
    
    # Add a crosshair at start
    ax.axvline(x=cx, color='#00ff66', linestyle=':', alpha=0.45, linewidth=1.2)
    ax.axhline(y=cy, color='#00ff66', linestyle=':', alpha=0.45, linewidth=1.2)
    
    ax.set_title(f'Close-Up: Início da Trajetória (Janela {2*hw:.0f}m x {2*hw:.0f}m)', color='white', fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Eixo X - Lateral (metros)', color='white', fontsize=12)
    ax.set_ylabel('Eixo Y - Longitudinal (metros)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy - hw, cy + hw)
    ax.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)
    
    out_img = out_dir / 'harmonymall_start_trajectory_closeup_12m.png'
    plt.savefig(str(out_img), dpi=280, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[+] Saved 12m Close-Up: {out_img}')
    
    # 2. Also generate a tighter 6m close-up for ultra fine floor/structure details
    hw_tight = 3.5
    mask_tight = (
        (pts[:, 0] >= cx - hw_tight) & (pts[:, 0] <= cx + hw_tight) &
        (pts[:, 1] >= cy - hw_tight) & (pts[:, 1] <= cy + hw_tight)
    )
    pts_t = pts[mask_tight]
    cols_t = cols[mask_tight]
    
    fig_t, ax_t = plt.subplots(figsize=(14, 14), dpi=300, facecolor=bg_color)
    ax_t.set_facecolor(panel_bg)
    ax_t.scatter(pts_t[:, 0], pts_t[:, 1], c=cols_t, s=4.2, alpha=0.98, rasterized=True)
    ax_t.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=3.5, label='Trajetória SLAM', zorder=10)
    ax_t.scatter(cx, cy, color='#00ff66', s=300, marker='o', edgecolors='white', linewidth=2.8, label=f'Início Trajetória ({cx:.2f}, {cy:.2f})', zorder=15)
    ax_t.axvline(x=cx, color='#00ff66', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_t.axhline(y=cy, color='#00ff66', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax_t.set_title(f'Ultra Close-Up: Início da Trajetória (Janela {2*hw_tight:.0f}m x {2*hw_tight:.0f}m)', color='white', fontsize=15, fontweight='bold', pad=14)
    ax_t.set_xlabel('Eixo X - Lateral (metros)', color='white', fontsize=12)
    ax_t.set_ylabel('Eixo Y - Longitudinal (metros)', color='white', fontsize=12)
    ax_t.tick_params(colors='white')
    ax_t.axis('equal')
    ax_t.set_xlim(cx - hw_tight, cx + hw_tight)
    ax_t.set_ylim(cy - hw_tight, cy + hw_tight)
    ax_t.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_t.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)
    
    out_tight = out_dir / 'harmonymall_start_trajectory_ultracloseup_7m.png'
    plt.savefig(str(out_tight), dpi=300, facecolor=fig_t.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[+] Saved 7m Ultra Close-Up: {out_tight}')
    print('[SUCCESS] All close-ups generated successfully!')

if __name__ == '__main__':
    main()
