#!/usr/bin/env python3
"""
Generates high-resolution orthogonal projections of the PIER colored point cloud
with the SLAM trajectory overlaid:
1. Top View (XY) — Aerial plan view
2. Longitudinal Profile View (Along Pier Axis vs Z)
3. Perpendicular Cross-Section View (Across Pier Axis vs Z)
"""

import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

DELIVERABLE_PLY = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output/deliverables/lidar_colored_direct_rigid_20261021_194433.ply')
TRJ_TXT = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output/slam_raw_f8a83ade028dd052024e/result/Raven_3DMakerPro_Scan.txt')
OUTPUT_PNG = Path('D:/ARQUIVOS_TESTE_2/PIER/pier_orthogonal_views.png')

def main():
    print(f'[*] Reading colored point cloud: {DELIVERABLE_PLY}...')
    pcd = o3d.io.read_point_cloud(str(DELIVERABLE_PLY))
    pts = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    print(f'[+] Loaded {len(pts):,} points.')

    print(f'[*] Reading SLAM trajectory: {TRJ_TXT}...')
    trj = np.loadtxt(TRJ_TXT)
    traj_xyz = trj[:, 1:4]
    print(f'[+] Loaded {len(traj_xyz):,} trajectory poses.')

    # Subsample points for rendering if very dense
    max_render = 600000
    if len(pts) > max_render:
        idx_sub = np.random.choice(len(pts), max_render, replace=False)
        pts_sub = pts[idx_sub]
        col_sub = colors[idx_sub]
    else:
        pts_sub = pts
        col_sub = colors

    # Pier axis alignment: find principal direction of the pier from trajectory
    v_pier = traj_xyz[-1, :2] - traj_xyz[0, :2]
    # In PIER dataset, trajectory goes from (0,0) to (15.6, -24.6)
    v_dir = np.array([15.65875, -24.62084])
    v_dir = v_dir / np.linalg.norm(v_dir)
    angle_rot = -np.arctan2(v_dir[1], v_dir[0])
    c_rot, s_rot = np.cos(angle_rot), np.sin(angle_rot)
    R_align = np.array([[c_rot, -s_rot], [s_rot, c_rot]])

    # Project points into Aligned Frame: [s_long, d_perp, z]
    pts_rot_xy = (R_align @ pts_sub[:, :2].T).T
    s_long = pts_rot_xy[:, 0]
    d_perp = pts_rot_xy[:, 1]
    z_elev = pts_sub[:, 2]

    traj_rot_xy = (R_align @ traj_xyz[:, :2].T).T
    traj_s = traj_rot_xy[:, 0]
    traj_d = traj_rot_xy[:, 1]
    traj_z = traj_xyz[:, 2]

    fig = plt.figure(figsize=(18, 16), dpi=160, facecolor='#121214')
    plt.rcParams['text.color'] = '#EEEEEE'
    plt.rcParams['axes.labelcolor'] = '#EEEEEE'
    plt.rcParams['xtick.color'] = '#EEEEEE'
    plt.rcParams['ytick.color'] = '#EEEEEE'

    # 1. Top View (XY - World/Plan View)
    ax1 = fig.add_subplot(3, 1, 1, facecolor='#18181b')
    ax1.scatter(pts_sub[:, 0], pts_sub[:, 1], c=col_sub, s=0.8, alpha=0.85, rasterized=True)
    ax1.plot(traj_xyz[:, 0], traj_xyz[:, 1], color='#FF3366', linewidth=2.5, label='FAST-LIVO2 Trajectory')
    ax1.scatter([traj_xyz[0, 0]], [traj_xyz[0, 1]], color='#00FF66', s=80, zorder=5, label='Start')
    ax1.scatter([traj_xyz[-1, 0]], [traj_xyz[-1, 1]], color='#FFCC00', s=80, zorder=5, label='End')
    ax1.set_aspect('equal')
    ax1.set_title('1. Top View (Planimetric XY) — Full Dataset with LiDAR Odometry Trajectory', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('X (meters)', fontsize=10)
    ax1.set_ylabel('Y (meters)', fontsize=10)
    ax1.grid(True, linestyle='--', color='#333338', alpha=0.5)
    ax1.legend(loc='upper right', facecolor='#202024', edgecolor='#44444a')

    # 2. Longitudinal View (Along Pier Walkway Axis vs Height Z)
    ax2 = fig.add_subplot(3, 1, 2, facecolor='#18181b')
    ax2.scatter(s_long, z_elev, c=col_sub, s=0.8, alpha=0.85, rasterized=True)
    ax2.plot(traj_s, traj_z, color='#FF3366', linewidth=2.5, label='Scanner Height Trajectory')
    ax2.scatter([traj_s[0]], [traj_z[0]], color='#00FF66', s=80, zorder=5, label='Start')
    ax2.scatter([traj_s[-1]], [traj_z[-1]], color='#FFCC00', s=80, zorder=5, label='End')
    ax2.set_aspect('equal')
    ax2.set_xlim(-5.0, 35.0)
    ax2.set_ylim(-3.5, 3.5)
    ax2.set_title('2. Longitudinal View (Along Pier Axis vs Height Z) — Side Elevation', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Distance Along Pier Axis (meters)', fontsize=10)
    ax2.set_ylabel('Height Z (meters)', fontsize=10)
    ax2.grid(True, linestyle='--', color='#333338', alpha=0.5)
    ax2.legend(loc='upper right', facecolor='#202024', edgecolor='#44444a')

    # 3. Perpendicular View (Cross Section Across Pier Walkway vs Height Z)
    # Filter a representative section along the pier (e.g. s_long between 8m and 18m)
    mask_cross = (s_long >= 8.0) & (s_long <= 20.0)
    ax3 = fig.add_subplot(3, 1, 3, facecolor='#18181b')
    ax3.scatter(d_perp[mask_cross], z_elev[mask_cross], c=col_sub[mask_cross], s=1.0, alpha=0.85, rasterized=True)
    # Filter trajectory in this section
    traj_mask = (traj_s >= 8.0) & (traj_s <= 20.0)
    if np.any(traj_mask):
        ax3.scatter(traj_d[traj_mask], traj_z[traj_mask], color='#FF3366', s=10, alpha=0.7, label='Scanner Path in Section')
    ax3.set_aspect('equal')
    ax3.set_xlim(-8.0, 8.0)
    ax3.set_ylim(-3.5, 3.5)
    ax3.set_title('3. Perpendicular View (Cross Section Across Pier Walkway vs Height Z) — 12m Section', fontsize=13, fontweight='bold', pad=10)
    ax3.set_xlabel('Distance Across Pier Walkway (meters)', fontsize=10)
    ax3.set_ylabel('Height Z (meters)', fontsize=10)
    ax3.grid(True, linestyle='--', color='#333338', alpha=0.5)
    ax3.legend(loc='upper right', facecolor='#202024', edgecolor='#44444a')

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=160, facecolor=fig.get_facecolor())
    plt.close()
    print(f'\n[SUCCESS] Orthogonal views generated at: {OUTPUT_PNG}')

if __name__ == '__main__':
    main()
