#!/usr/bin/env python3
"""
Fast Candidate Generator for HarmonyMall
Tests 5 calibration/temporal hypotheses on a 12m x 12m corridor and storefront window
to identify the exact parameter set that eliminates ghosting and misalignments.
"""
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import open3d as o3d

WORKSPACE = Path('C:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator')
DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
OUT_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/candidates_zoom')
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
    trj_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'

    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)

    # Focus on the corridor and shop window section: center (2.5, -20.0), span 12m
    cx, cy = 2.5, -20.0
    hw = 5.5
    mask = (
        (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) &
        (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) &
        (pts_all[:, 2] >= -2.0) & (pts_all[:, 2] <= 3.8)
    )
    pts = pts_all[mask]
    print(f'[+] Filtered {len(pts):,} points in 11m x 11m window.')

    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]

    with open(WORKSPACE / 'calibracao_rigida_raven_insta360.json', 'r') as f:
        calib_factory = json.load(f)
    with open(DATASET_DIR / 'rig_calibration.json', 'r') as f:
        calib_recal = json.load(f)

    p0 = calib_factory['cam0_front_intrinsics']
    params0 = (p0['fx'], p0['fy'], p0['cx'], p0['cy'], p0['k1'], p0['k2'], p0['p1'], p0['p2'], p0['k3'], p0['k4'], p0['sx1'], p0['sy1'])
    p1 = calib_factory['cam1_rear_intrinsics']
    params1 = (p1['fx'], p1['fy'], p1['cx'], p1['cy'], p1['k1'], p1['k2'], p1['p1'], p1['p2'], p1['k3'], p1['k4'], p1['sx1'], p1['sy1'])

    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    cam1_files = sorted(list((DATASET_DIR / 'images/cam1').glob('*.jpg')))

    with open(DATASET_DIR / 'images/frames.json', 'r') as f:
        fj = json.load(f)
    ts_dict = fj.get('timestamps', {})
    video_times = {p.name: float(ts_dict.get(f'cam0/{p.name}', 0.0)) for p in cam0_files}

    # Build 5 Hypotheses recommended by Gemini 4 Argon
    # Factory extrinsics
    T_fact0 = np.array(calib_factory['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T_fact1 = np.array(calib_factory['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    # Recalibrated extrinsics
    T_rec0 = np.array(calib_recal['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T_rec1 = np.array(calib_recal['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)

    # Yaw perturbation matrices
    R_yaw_neg = R.from_euler('z', -1.2, degrees=True).as_matrix()
    R_yaw_pos = R.from_euler('z', +1.2, degrees=True).as_matrix()

    T_rec0_yaw_neg = T_rec0.copy()
    T_rec0_yaw_neg[:3, :3] = T_rec0[:3, :3] @ R_yaw_neg.T
    T_rec1_yaw_neg = T_rec1.copy()
    T_rec1_yaw_neg[:3, :3] = T_rec1[:3, :3] @ R_yaw_neg.T

    T_rec0_yaw_pos = T_rec0.copy()
    T_rec0_yaw_pos[:3, :3] = T_rec0[:3, :3] @ R_yaw_pos.T
    T_rec1_yaw_pos = T_rec1.copy()
    T_rec1_yaw_pos[:3, :3] = T_rec1[:3, :3] @ R_yaw_pos.T

    candidates = [
        (
            'Cand 1: Factory Nom (dt = 3.337s)',
            3.3368,
            T_fact0, T_fact1
        ),
        (
            'Cand 2: Factory Nom + dt SfM (3.377s)',
            3.3768,
            T_fact0, T_fact1
        ),
        (
            'Cand 3: Recalib SfM Rig (dt = 3.377s)',
            3.3768,
            T_rec0, T_rec1
        ),
        (
            'Cand 4: Recalib + Yaw -1.2 deg',
            3.3768,
            T_rec0_yaw_neg, T_rec1_yaw_neg
        ),
        (
            'Cand 5: Recalib + Yaw +1.2 deg',
            3.3768,
            T_rec0_yaw_pos, T_rec1_yaw_pos
        ),
    ]

    # Find frames covering this window (e.g. scanner near center)
    # Scanner path within 12m
    dists_trj = np.hypot(pos_slam[:, 0] - cx, pos_slam[:, 1] - cy)
    close_poses = np.where(dists_trj < 15.0)[0]
    t_close_min = t_slam[close_poses[0]] - 5.0
    t_close_max = t_slam[close_poses[-1]] + 5.0

    candidate_results = []
    img_cache = {}

    for cand_idx, (cand_name, dt_cand, T0, T1) in enumerate(candidates):
        print(f'[*] Evaluating {cand_name}...')
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]

        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        cand_cols = np.zeros((len(pts), 3), dtype=np.uint8)

        for k, p_cam0 in enumerate(cam0_files):
            t_vid = video_times[p_cam0.name]
            t_query = t_slam[0] + (t_vid - dt_cand)
            if t_query < t_close_min or t_query > t_close_max:
                continue
            if not (t_slam_start <= t_query <= t_slam_end):
                continue

            # Query SLAM pose
            idx = np.clip(np.searchsorted(t_slam, t_query), 1, len(t_slam) - 1)
            w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_query).as_matrix()

            # Only consider points within 12m of camera position
            d_pts = np.linalg.norm(pts - p_L, axis=1)
            mask_local = d_pts < 12.0
            if not np.any(mask_local):
                continue

            idx_local = np.where(mask_local)[0]
            sub_pts = pts[idx_local]

            # Transform world points to LiDAR body
            pts_body = (R_L.T @ (sub_pts - p_L).T).T

            # Check Cam0
            pts_c0 = (R_LC0 @ pts_body.T).T + t_LC0
            front_mask0 = pts_c0[:, 2] > 0.3
            if np.any(front_mask0):
                u0, v0, r0 = project_thin_prism(pts_c0[front_mask0], params0)
                valid0 = (u0 >= 0) & (u0 < 2880) & (v0 >= 0) & (v0 < 2880)
                if np.any(valid0):
                    sub_sub_idx = idx_local[front_mask0][valid0]
                    dists = pts_c0[front_mask0, 2][valid0]
                    closer = dists < best_dist[sub_sub_idx]
                    if np.any(closer):
                        if p_cam0.name not in img_cache:
                            img_cache[p_cam0.name] = cv2.imread(str(p_cam0))
                        img0 = img_cache[p_cam0.name]
                        take_idx = sub_sub_idx[closer]
                        u_take = np.clip(np.round(u0[valid0][closer]).astype(int), 0, 2879)
                        v_take = np.clip(np.round(v0[valid0][closer]).astype(int), 0, 2879)
                        cand_cols[take_idx] = img0[v_take, u_take][:, ::-1]
                        best_dist[take_idx] = dists[closer]

            # Check Cam1
            pts_c1 = (R_LC1 @ pts_body.T).T + t_LC1
            front_mask1 = pts_c1[:, 2] > 0.3
            if np.any(front_mask1):
                u1, v1, r1 = project_thin_prism(pts_c1[front_mask1], params1)
                valid1 = (u1 >= 0) & (u1 < 2880) & (v1 >= 0) & (v1 < 2880)
                if np.any(valid1):
                    sub_sub_idx = idx_local[front_mask1][valid1]
                    dists = pts_c1[front_mask1, 2][valid1]
                    closer = dists < best_dist[sub_sub_idx]
                    if np.any(closer):
                        p_cam1 = DATASET_DIR / f'images/cam1/{p_cam0.name}'
                        if p_cam1.name not in img_cache:
                            img_cache[p_cam1.name] = cv2.imread(str(p_cam1))
                        img1 = img_cache[p_cam1.name]
                        take_idx = sub_sub_idx[closer]
                        u_take = np.clip(np.round(u1[valid1][closer]).astype(int), 0, 2879)
                        v_take = np.clip(np.round(v1[valid1][closer]).astype(int), 0, 2879)
                        cand_cols[take_idx] = img1[v_take, u_take][:, ::-1]
                        best_dist[take_idx] = dists[closer]

        cov = np.count_nonzero(np.any(cand_cols > 0, axis=1)) / len(pts) * 100.0
        print(f'[+] {cand_name}: coverage = {cov:.1f}%')
        candidate_results.append((cand_name, cand_cols))

    # Plot side-by-side comparison
    fig, axes = plt.subplots(1, 5, figsize=(30, 6.5), dpi=220, facecolor='#111115')
    for ax, (c_name, c_cols) in zip(axes, candidate_results):
        ax.set_facecolor('#16161c')
        valid = np.any(c_cols > 0, axis=1)
        ax.scatter(pts[valid, 0], pts[valid, 1], c=c_cols[valid]/255.0, s=2.5, rasterized=True)
        ax.set_title(c_name, color='white', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.tick_params(colors='white')
        ax.axis('equal')
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)
        ax.grid(True, color='#2c2c36', linestyle='--', alpha=0.5)

    out_comp = DATASET_DIR / 'candidates_zoom/mall_candidates_comparison.png'
    plt.tight_layout()
    plt.savefig(str(out_comp), dpi=220, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f'[SUCCESS] Candidates comparison saved to: {out_comp}')

    # Also save standalone high-res images for Candidate 1, 2 and 3
    for idx_c in [0, 1, 2]:
        c_name, c_cols = candidate_results[idx_c]
        fig_single, ax_s = plt.subplots(figsize=(10, 10), dpi=260, facecolor='#111115')
        ax_s.set_facecolor('#16161c')
        valid = np.any(c_cols > 0, axis=1)
        ax_s.scatter(pts[valid, 0], pts[valid, 1], c=c_cols[valid]/255.0, s=3.5, rasterized=True)
        ax_s.set_title(f'{c_name} - Close-up High-Res', color='white', fontsize=13, fontweight='bold', pad=10)
        ax_s.set_xlabel('X (m)', color='white')
        ax_s.set_ylabel('Y (m)', color='white')
        ax_s.tick_params(colors='white')
        ax_s.axis('equal')
        ax_s.set_xlim(cx - hw, cx + hw)
        ax_s.set_ylim(cy - hw, cy + hw)
        ax_s.grid(True, color='#2c2c36', linestyle='--', alpha=0.5)
        single_out = DATASET_DIR / f'candidates_zoom/mall_candidate_{idx_c+1}_detail.png'
        plt.tight_layout()
        plt.savefig(str(single_out), dpi=260, facecolor=fig_single.get_facecolor(), edgecolor='none')
        plt.close()
        print(f'[+] Standalone detail saved: {single_out}')

if __name__ == '__main__':
    main()
