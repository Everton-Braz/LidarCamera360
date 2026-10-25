#!/usr/bin/env python3
"""
High-Performance Orthogonal Views (Top-View, Longitudinal, Latitudinal) with SLAM Trajectory
Visualizing the Fully Colored LiDAR Cloud (Direct Rigid Calibration Candidate)
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
    
    print(f'[*] Loading colored point cloud: {ply_path.name}...')
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    print(f'    Total points: {len(pcd.points):,}. Voxel downsampling (voxel_size=0.08m)...')
    pcd_down = pcd.voxel_down_sample(voxel_size=0.08)
    
    pts = np.asarray(pcd_down.points, dtype=np.float32)
    cols = np.asarray(pcd_down.colors, dtype=np.float32)
    print(f'    Downsampled to {len(pts):,} points.')
    
    # Filter uncolored or invalid black background points
    valid_color = np.sum(cols, axis=1) > 0.08
    pts_val = pts[valid_color]
    cols_val = cols[valid_color]
    print(f'    Valid colored points for rendering: {len(pts_val):,}')
    
    print(f'[*] Loading SLAM trajectory: {trj_path.name}...')
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    print(f'    Trajectory points: {len(pos_slam):,}, Length: ~{np.sum(np.linalg.norm(np.diff(pos_slam, axis=0), axis=1)):.1f}m')
    
    # Visual style settings
    bg_color = '#0e0e13'
    panel_bg = '#14141c'
    grid_color = '#252533'
    traj_color = '#00f0ff'
    
    # =========================================================================
    # 1. MASTER 3-PANEL FIGURE (TOP-VIEW + LONGITUDINAL + LATITUDINAL)
    # =========================================================================
    print('[*] Generating Master 3-Panel Orthogonal Figure...')
    fig = plt.figure(figsize=(26, 17), dpi=220, facecolor=bg_color)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.22, wspace=0.18)
    
    # --- PANEL 1: TOP-VIEW (VISTA SUPERIOR XY) ---
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor(panel_bg)
    ax_top.scatter(pts_val[:, 0], pts_val[:, 1], c=cols_val, s=1.4, alpha=0.92, rasterized=True)
    ax_top.plot(pos_slam[:, 0], pos_slam[:, 1], color=traj_color, linewidth=2.4, label='Trajetória LiDAR SLAM (688m)', zorder=10)
    ax_top.scatter(pos_slam[0, 0], pos_slam[0, 1], color='#00ff66', s=90, marker='o', edgecolors='white', linewidth=1.2, label='Início', zorder=12)
    ax_top.scatter(pos_slam[-1, 0], pos_slam[-1, 1], color='#ff0055', s=90, marker='X', edgecolors='white', linewidth=1.2, label='Fim', zorder=12)
    ax_top.set_title('Vista Superior (Top-View / Plano XY) - Nuvem Colorida com Trajetória SLAM', color='white', fontsize=14, fontweight='bold', pad=10)
    ax_top.set_xlabel('Eixo X - Transversal / Latitudinal (metros)', color='white', fontsize=11)
    ax_top.set_ylabel('Eixo Y - Longitudinal / Corredor (metros)', color='white', fontsize=11)
    ax_top.tick_params(colors='white')
    ax_top.axis('equal')
    ax_top.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_top.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=11)
    
    # --- PANEL 2: LONGITUDINAL PROFILE (VISTA LONGITUDINAL YZ) ---
    ax_long = fig.add_subplot(gs[1, 0])
    ax_long.set_facecolor(panel_bg)
    # Longitudinal axis (Y) vs Elevation (Z)
    ax_long.scatter(pts_val[:, 1], pts_val[:, 2], c=cols_val, s=1.2, alpha=0.88, rasterized=True)
    ax_long.plot(pos_slam[:, 1], pos_slam[:, 2], color=traj_color, linewidth=2.2, label='Trajetória SLAM', zorder=10)
    ax_long.scatter(pos_slam[0, 1], pos_slam[0, 2], color='#00ff66', s=80, marker='o', edgecolors='white', zorder=12)
    ax_long.scatter(pos_slam[-1, 1], pos_slam[-1, 2], color='#ff0055', s=80, marker='X', edgecolors='white', zorder=12)
    ax_long.set_title('Vista Perfil Longitudinal (Plano YZ) - Extensão do Corredor vs Altura', color='white', fontsize=13, fontweight='bold', pad=10)
    ax_long.set_xlabel('Eixo Y - Longitudinal (metros)', color='white', fontsize=11)
    ax_long.set_ylabel('Eixo Z - Altura / Elevação (metros)', color='white', fontsize=11)
    ax_long.tick_params(colors='white')
    ax_long.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_long.set_ylim(-3.5, 7.5)
    ax_long.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white')
    
    # --- PANEL 3: LATITUDINAL PROFILE (VISTA LATITUDINAL XZ) ---
    ax_lat = fig.add_subplot(gs[1, 1])
    ax_lat.set_facecolor(panel_bg)
    # Latitudinal axis (X) vs Elevation (Z)
    ax_lat.scatter(pts_val[:, 0], pts_val[:, 2], c=cols_val, s=1.2, alpha=0.88, rasterized=True)
    ax_lat.plot(pos_slam[:, 0], pos_slam[:, 2], color=traj_color, linewidth=2.2, label='Trajetória SLAM', zorder=10)
    ax_lat.scatter(pos_slam[0, 0], pos_slam[0, 2], color='#00ff66', s=80, marker='o', edgecolors='white', zorder=12)
    ax_lat.scatter(pos_slam[-1, 0], pos_slam[-1, 2], color='#ff0055', s=80, marker='X', edgecolors='white', zorder=12)
    ax_lat.set_title('Vista Perfil Latitudinal / Transversal (Plano XZ) - Largura vs Altura', color='white', fontsize=13, fontweight='bold', pad=10)
    ax_lat.set_xlabel('Eixo X - Lateral / Transversal (metros)', color='white', fontsize=11)
    ax_lat.set_ylabel('Eixo Z - Altura / Elevação (metros)', color='white', fontsize=11)
    ax_lat.tick_params(colors='white')
    ax_lat.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_lat.set_ylim(-3.5, 7.5)
    ax_lat.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white')
    
    master_path = out_dir / 'harmonymall_orthogonal_colored_master.png'
    plt.savefig(str(master_path), dpi=220, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[+] Saved Master Orthogonal Figure: {master_path}')
    
    # =========================================================================
    # 2. STANDALONE HIGH-RES TOP-VIEW (VISTA SUPERIOR XY)
    # =========================================================================
    print('[*] Generating Standalone High-Res Top-View...')
    fig_top, ax_t = plt.subplots(figsize=(15, 22), dpi=260, facecolor=bg_color)
    ax_t.set_facecolor(panel_bg)
    ax_t.scatter(pts_val[:, 0], pts_val[:, 1], c=cols_val, s=1.8, alpha=0.95, rasterized=True)
    ax_t.plot(pos_slam[:, 0], pos_slam[:, 1], color=traj_color, linewidth=2.5, label='Trajetória SLAM (Raven LiDAR)', zorder=10)
    ax_t.scatter(pos_slam[0, 0], pos_slam[0, 1], color='#00ff66', s=120, marker='o', edgecolors='white', linewidth=1.5, label='Início Trajetória', zorder=12)
    ax_t.scatter(pos_slam[-1, 0], pos_slam[-1, 1], color='#ff0055', s=120, marker='X', edgecolors='white', linewidth=1.5, label='Fim Trajetória', zorder=12)
    ax_t.set_title('HarmonyMall: Vista Superior Completa (Top-View XY) + Trajetória SLAM', color='white', fontsize=15, fontweight='bold', pad=14)
    ax_t.set_xlabel('X - Lateral (metros)', color='white', fontsize=12)
    ax_t.set_ylabel('Y - Longitudinal (metros)', color='white', fontsize=12)
    ax_t.tick_params(colors='white')
    ax_t.axis('equal')
    ax_t.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_t.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)
    top_path = out_dir / 'harmonymall_topview_complete.png'
    plt.savefig(str(top_path), dpi=260, facecolor=fig_top.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[+] Saved Standalone Top-View: {top_path}')
    
    # =========================================================================
    # 3. STANDALONE HIGH-RES LONGITUDINAL PROFILE (YZ)
    # =========================================================================
    print('[*] Generating Standalone High-Res Longitudinal Profile...')
    fig_long, ax_l = plt.subplots(figsize=(22, 10), dpi=260, facecolor=bg_color)
    ax_l.set_facecolor(panel_bg)
    ax_l.scatter(pts_val[:, 1], pts_val[:, 2], c=cols_val, s=1.6, alpha=0.90, rasterized=True)
    ax_l.plot(pos_slam[:, 1], pos_slam[:, 2], color=traj_color, linewidth=2.4, label='Trajetória SLAM', zorder=10)
    ax_l.scatter(pos_slam[0, 1], pos_slam[0, 2], color='#00ff66', s=110, marker='o', edgecolors='white', linewidth=1.5, label='Início', zorder=12)
    ax_l.scatter(pos_slam[-1, 1], pos_slam[-1, 2], color='#ff0055', s=110, marker='X', edgecolors='white', linewidth=1.5, label='Fim', zorder=12)
    ax_l.set_title('HarmonyMall: Vista Longitudinal (Perfil YZ) - Extensão vs Altura', color='white', fontsize=15, fontweight='bold', pad=14)
    ax_l.set_xlabel('Y - Longitudinal (metros)', color='white', fontsize=12)
    ax_l.set_ylabel('Z - Altura (metros)', color='white', fontsize=12)
    ax_l.tick_params(colors='white')
    ax_l.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_l.set_ylim(-3.5, 7.5)
    ax_l.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)
    long_path = out_dir / 'harmonymall_longitudinal_profile.png'
    plt.savefig(str(long_path), dpi=260, facecolor=fig_long.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[+] Saved Standalone Longitudinal Profile: {long_path}')
    
    # =========================================================================
    # 4. STANDALONE HIGH-RES LATITUDINAL PROFILE (XZ)
    # =========================================================================
    print('[*] Generating Standalone High-Res Latitudinal Profile...')
    fig_lat, ax_w = plt.subplots(figsize=(16, 10), dpi=260, facecolor=bg_color)
    ax_w.set_facecolor(panel_bg)
    ax_w.scatter(pts_val[:, 0], pts_val[:, 2], c=cols_val, s=1.6, alpha=0.90, rasterized=True)
    ax_w.plot(pos_slam[:, 0], pos_slam[:, 2], color=traj_color, linewidth=2.4, label='Trajetória SLAM', zorder=10)
    ax_w.scatter(pos_slam[0, 0], pos_slam[0, 2], color='#00ff66', s=110, marker='o', edgecolors='white', linewidth=1.5, label='Início', zorder=12)
    ax_w.scatter(pos_slam[-1, 0], pos_slam[-1, 2], color='#ff0055', s=110, marker='X', edgecolors='white', linewidth=1.5, label='Fim', zorder=12)
    ax_w.set_title('HarmonyMall: Vista Latitudinal / Transversal (Perfil XZ) - Largura vs Altura', color='white', fontsize=15, fontweight='bold', pad=14)
    ax_w.set_xlabel('X - Lateral (metros)', color='white', fontsize=12)
    ax_w.set_ylabel('Z - Altura (metros)', color='white', fontsize=12)
    ax_w.tick_params(colors='white')
    ax_w.grid(True, color=grid_color, linestyle='--', alpha=0.6)
    ax_w.set_ylim(-3.5, 7.5)
    ax_w.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)
    lat_path = out_dir / 'harmonymall_latitudinal_profile.png'
    plt.savefig(str(lat_path), dpi=260, facecolor=fig_lat.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[+] Saved Standalone Latitudinal Profile: {lat_path}')
    print('[SUCCESS] All orthogonal views successfully generated!')

if __name__ == '__main__':
    main()
