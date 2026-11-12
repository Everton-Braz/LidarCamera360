#!/usr/bin/env python3
"""
Render Full Dataset Orthogonal Multi-Views (Top-View, Longitudinal, Latitudinal) with Trajectory
Colored with the Decoupled Calibration Rig.
"""
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import open3d as o3d

DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
OUT_DIR = DATASET_DIR / 'orthogonal_views'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def project_thin_prism(P_v, params):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    r = np.hypot(P_v[:, 0], P_v[:, 1])
    theta = np.arctan2(r, P_v[:, 2])
    scale = np.divide(theta, r, out=np.zeros_like(theta), where=r > 1e-12)
    xd, yd = P_v[:, 0] * scale, P_v[:, 1] * scale
    rd2 = theta**2
    radial = 1 + rd2 * (k1 + rd2 * (k2 + rd2 * (k3 + rd2 * k4)))
    du = 2*p1*xd*yd + p2*(rd2 + 2*xd**2) + sx1*rd2
    dv = p1*(rd2 + 2*yd**2) + 2*p2*xd*yd + sy1*rd2
    u, v = fx*(xd*radial + du) + cx, fy*(yd*radial + dv) + cy
    return u, v, r

def main():
    print('[*] Loading PCD and SLAM trajectory for HarmonyMall...')
    pcd_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/pcd/all_raw_points.pcd'
    trj_path = list(DATASET_DIR.glob('slam_raw_*/result/*Scan*.txt'))[0]

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    # Downsample slightly for ultra-fast full-dataset rendering if needed (voxel 4cm)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.04)
    pts = np.asarray(pcd_down.points, dtype=np.float32)
    print(f'[+] Downsampled points for orthogonal view: {len(pts):,}')

    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]

    # Decoupled calib
    with open(DATASET_DIR / 'decoupled_rig_calibration.json') as f:
        calib = json.load(f)
    T0 = np.array(calib['T_lidar_to_cam0_rigid_4x4'])
    T1 = np.array(calib['T_lidar_to_cam1_rigid_4x4'])
    dt_sync = calib['dt_sync_refined']

    # Factory intrinsics
    with open('calibracao_rigida_raven_insta360.json') as f:
        nom = json.load(f)
    p0 = nom['cam0_front_intrinsics']
    params0 = (p0['fx'], p0['fy'], p0['cx'], p0['cy'], p0['k1'], p0['k2'], p0['p1'], p0['p2'], p0['k3'], p0['k4'], p0['sx1'], p0['sy1'])
    p1 = nom['cam1_rear_intrinsics']
    params1 = (p1['fx'], p1['fy'], p1['cx'], p1['cy'], p1['k1'], p1['k2'], p1['p1'], p1['p2'], p1['k3'], p1['k4'], p1['sx1'], p1['sy1'])

    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    with open(DATASET_DIR / 'images/frames.json') as f:
        fj = json.load(f)
    video_times = {p.name: float(fj['timestamps'].get(f'cam0/{p.name}', 0.0)) for p in cam0_files}

    # Colorize full cloud with stride 2 frames for speed
    print('[*] Projecting colors across trajectory...')
    R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
    R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]

    best_dist = np.full(len(pts), 999.0, dtype=np.float32)
    colors = np.zeros((len(pts), 3), dtype=np.uint8)

    stride = 2
    for idx_f, p in enumerate(cam0_files[::stride]):
        t_vid = video_times[p.name]
        t_q = t_slam[0] + (t_vid - dt_sync)
        if not (t_slam_start <= t_q <= t_slam_end):
            continue
        idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
        w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L = slerp(t_q).as_matrix()

        # Bounding box filter around camera pose (radius 18m)
        dist_sq = np.sum((pts[:, :2] - p_L[:2])**2, axis=1)
        mask_local = dist_sq <= (18.0**2)
        if not np.any(mask_local):
            continue

        sub_indices = np.where(mask_local)[0]
        pts_local = pts[sub_indices]
        pts_body = (R_L.T @ (pts_local - p_L).T).T

        img0 = cv2.imread(str(DATASET_DIR / f'images/cam0/{p.name}'))
        img1 = cv2.imread(str(DATASET_DIR / f'images/cam1/{p.name}'))

        # Cam0
        pts_c0 = (R_LC0 @ pts_body.T).T + t_LC0
        m0 = pts_c0[:, 2] > 0.3
        if np.any(m0):
            u0, v0, _ = project_thin_prism(pts_c0[m0], params0)
            val0 = (u0 >= 0) & (u0 < 2880) & (v0 >= 0) & (v0 < 2880)
            if np.any(val0):
                sub_idx = sub_indices[m0][val0]
                dists = pts_c0[m0, 2][val0]
                closer = dists < best_dist[sub_idx]
                if np.any(closer):
                    tk = sub_idx[closer]
                    colors[tk] = img0[np.clip(v0[val0][closer].round().astype(int), 0, 2879), np.clip(u0[val0][closer].round().astype(int), 0, 2879)][:, ::-1]
                    best_dist[tk] = dists[closer]

        # Cam1
        pts_c1 = (R_LC1 @ pts_body.T).T + t_LC1
        m1 = pts_c1[:, 2] > 0.3
        if np.any(m1):
            u1, v1, _ = project_thin_prism(pts_c1[m1], params1)
            val1 = (u1 >= 0) & (u1 < 2880) & (v1 >= 0) & (v1 < 2880)
            if np.any(val1):
                sub_idx = sub_indices[m1][val1]
                dists = pts_c1[m1, 2][val1]
                closer = dists < best_dist[sub_idx]
                if np.any(closer):
                    tk = sub_idx[closer]
                    colors[tk] = img1[np.clip(v1[val1][closer].round().astype(int), 0, 2879), np.clip(u1[val1][closer].round().astype(int), 0, 2879)][:, ::-1]
                    best_dist[tk] = dists[closer]

    valid = np.any(colors > 0, axis=1)
    pts_val = pts[valid]
    cols_val = colors[valid] / 255.0
    print(f'[+] Colorized {len(pts_val):,} points.')

    # Create Master 3-Panel Orthogonal View (Top-View, Longitudinal Side, Latitudinal Front)
    print('[*] Rendering Master Orthogonal Plot...')
    fig = plt.figure(figsize=(26, 18), dpi=220, facecolor='#0f0f13')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.22, wspace=0.18)

    # 1. Top-View (XY Plane) - Main view
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_facecolor('#15151c')
    ax_top.scatter(pts_val[:, 0], pts_val[:, 1], c=cols_val, s=1.2, alpha=0.9, rasterized=True)
    ax_top.plot(pos_slam[:, 0], pos_slam[:, 1], c='#00ffcc', linewidth=2.0, label='Trajetória LiDAR SLAM', zorder=10)
    ax_top.scatter(pos_slam[0, 0], pos_slam[0, 1], c='#00ff00', s=80, marker='o', label='Início', zorder=11)
    ax_top.scatter(pos_slam[-1, 0], pos_slam[-1, 1], c='#ff0055', s=80, marker='X', label='Fim', zorder=11)
    ax_top.set_title('Vista Superior (Top-View / XY) - Colorida com Calibração Desacoplada e Trajetória', color='white', fontsize=14, fontweight='bold', pad=10)
    ax_top.set_xlabel('X - Lateral (metros)', color='white', fontsize=11)
    ax_top.set_ylabel('Y - Longitudinal (metros)', color='white', fontsize=11)
    ax_top.tick_params(colors='white')
    ax_top.axis('equal')
    ax_top.grid(True, color='#282836', linestyle='--', alpha=0.6)
    ax_top.legend(loc='upper right', facecolor='#1f1f2a', edgecolor='white', labelcolor='white')

    # 2. Longitudinal Profile (YZ Plane - looking from the side)
    ax_long = fig.add_subplot(gs[1, 0])
    ax_long.set_facecolor('#15151c')
    # Profile along principal corridor (Y) vs Height (Z)
    ax_long.scatter(pts_val[:, 1], pts_val[:, 2], c=cols_val, s=1.2, alpha=0.85, rasterized=True)
    ax_long.plot(pos_slam[:, 1], pos_slam[:, 2], c='#00ffcc', linewidth=2.0, zorder=10)
    ax_long.set_title('Vista Perfil Longitudinal (YZ) - Extensão vs Altura', color='white', fontsize=13, fontweight='bold', pad=10)
    ax_long.set_xlabel('Y - Longitudinal (metros)', color='white', fontsize=11)
    ax_long.set_ylabel('Z - Altura (metros)', color='white', fontsize=11)
    ax_long.tick_params(colors='white')
    ax_long.grid(True, color='#282836', linestyle='--', alpha=0.6)
    ax_long.set_ylim(-3.0, 8.0)

    # 3. Latitudinal Profile (XZ Plane - looking from the front/end)
    ax_lat = fig.add_subplot(gs[1, 1])
    ax_lat.set_facecolor('#15151c')
    # Profile across corridor width (X) vs Height (Z)
    ax_lat.scatter(pts_val[:, 0], pts_val[:, 2], c=cols_val, s=1.2, alpha=0.85, rasterized=True)
    ax_lat.plot(pos_slam[:, 0], pos_slam[:, 2], c='#00ffcc', linewidth=2.0, zorder=10)
    ax_lat.set_title('Vista Perfil Latitudinal (XZ) - Largura vs Altura', color='white', fontsize=13, fontweight='bold', pad=10)
    ax_lat.set_xlabel('X - Lateral (metros)', color='white', fontsize=11)
    ax_lat.set_ylabel('Z - Altura (metros)', color='white', fontsize=11)
    ax_lat.tick_params(colors='white')
    ax_lat.grid(True, color='#282836', linestyle='--', alpha=0.6)
    ax_lat.set_ylim(-3.0, 8.0)

    out_master = OUT_DIR / 'harmonymall_orthogonal_colored_master.png'
    plt.savefig(str(out_master), dpi=220, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[SUCCESS] Master orthogonal plot saved: {out_master}')

    # Save standalone Top-View High-Res
    fig_top, ax_t = plt.subplots(figsize=(14, 20), dpi=260, facecolor='#0f0f13')
    ax_t.set_facecolor('#15151c')
    ax_t.scatter(pts_val[:, 0], pts_val[:, 1], c=cols_val, s=1.6, alpha=0.92, rasterized=True)
    ax_t.plot(pos_slam[:, 0], pos_slam[:, 1], c='#00ffcc', linewidth=2.2, label='Trajetória LiDAR SLAM', zorder=10)
    ax_t.scatter(pos_slam[0, 0], pos_slam[0, 1], c='#00ff00', s=100, marker='o', label='Início', zorder=11)
    ax_t.scatter(pos_slam[-1, 0], pos_slam[-1, 1], c='#ff0055', s=100, marker='X', label='Fim', zorder=11)
    ax_t.set_title('HarmonyMall: Vista Superior Completa (Top-View XY) + Trajetória', color='white', fontsize=14, fontweight='bold', pad=12)
    ax_t.set_xlabel('X (m)', color='white')
    ax_t.set_ylabel('Y (m)', color='white')
    ax_t.tick_params(colors='white')
    ax_t.axis('equal')
    ax_t.grid(True, color='#282836', linestyle='--', alpha=0.6)
    ax_t.legend(loc='upper right', facecolor='#1f1f2a', edgecolor='white', labelcolor='white')
    out_top = OUT_DIR / 'harmonymall_topview_complete.png'
    plt.savefig(str(out_top), dpi=260, facecolor=fig_top.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f'[SUCCESS] High-Res Top-View saved: {out_top}')

if __name__ == '__main__':
    main()
