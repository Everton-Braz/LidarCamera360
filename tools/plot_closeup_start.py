#!/usr/bin/env python3
"""
Render High-Resolution Close-up Centered at the Start of the Trajectory
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

    print('[*] Loading point cloud...')
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    cols_all = np.asarray(pcd.colors, dtype=np.float32)

    # Crop window around start (radius 6m -> 12x12m span)
    hw = 6.0
    mask = (
        (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) &
        (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) &
        (pts_all[:, 2] >= -2.5) & (pts_all[:, 2] <= 3.8)
    )
    pts = pts_all[mask]
    cols = cols_all[mask]
    print(f'    Points in {2*hw}m window around start: {len(pts):,}')

    # Filter uncolored/black background points
    valid = np.sum(cols, axis=1) > 0.06
    pts = pts[valid]
    cols = cols[valid]

    # Filter trajectory inside / near window
    traj_mask = (
        (pos_slam[:, 0] >= cx - hw - 2.0) & (pos_slam[:, 0] <= cx + hw + 2.0) &
        (pos_slam[:, 1] >= cy - hw - 2.0) & (pos_slam[:, 1] <= cy + hw + 2.0)
    )
    pos_traj_local = pos_slam[traj_mask]

    fig, ax = plt.subplots(figsize=(14, 14), dpi=280, facecolor='#0e0e13')
    ax.set_facecolor('#14141c')

    # Scatter colored points
    ax.scatter(pts[:, 0], pts[:, 1], c=cols, s=2.8, alpha=0.95, rasterized=True)

    # Plot trajectory
    if len(pos_traj_local) > 1:
        ax.plot(pos_traj_local[:, 0], pos_traj_local[:, 1], color='#00f0ff', linewidth=3.2, label='Trajetória SLAM (LiDAR)', zorder=10)

    # Start marker
    ax.scatter(cx, cy, color='#00ff66', s=180, marker='o', edgecolors='white', linewidth=2.0, label='Início da Trajetória (t=0s)', zorder=15)

    # Direction arrow for first 3 meters of motion
    if len(pos_slam) > 30:
        p_next = pos_slam[25]
        dx = p_next[0] - cx
        dy = p_next[1] - cy
        dist = np.hypot(dx, dy)
        if dist > 0.1:
            ax.annotate('', xy=(cx + (dx/dist)*2.0, cy + (dy/dist)*2.0), xytext=(cx, cy),
                        arrowprops=dict(facecolor='#00ff66', edgecolor='white', width=2.5, headwidth=10, shrink=0.05), zorder=16)

    ax.set_title('HarmonyMall: Close-up no Início da Trajetória (Centro em 0, 0)', color='white', fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Eixo X (metros)', color='white', fontsize=12)
    ax.set_ylabel('Eixo Y (metros)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy - hw, cy + hw)
    ax.grid(True, color='#252533', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)

    out_path = out_dir / 'harmonymall_closeup_start_trajectory.png'
    plt.savefig(str(out_path), dpi=280, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[SUCCESS] Close-up plot saved to: {out_path}')

if __name__ == '__main__':
    main()
