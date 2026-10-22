#!/usr/bin/env python3
"""
Fast Candidate Generator for HarmonyMall (Optimized)
Filters ONLY closest frames near the corridor window to run in seconds.
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

    # Focus on the corridor and shop window section: center (2.5, -20.0), span 11m
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
    with open(DATASET_DIR / 'images/frames.json', 'r') as f:
        fj = json.load(f)
    ts_dict = fj.get('timestamps', {})
    video_times = {p.name: float(ts_dict.get(f'cam0/{p.name}', 0.0)) for p in cam0_files}

    # Identify ONLY the frames where the scanner was within 8 meters of this corridor center
    dt_ref = 3.3768
    relevant_frames = []
    for p in cam0_files:
        t_vid = video_times[p.name]
        t_q = t_slam[0] + (t_vid - dt_ref)
        if t_slam_start <= t_q <= t_slam_end:
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            d_to_center = np.hypot(p_L[0] - cx, p_L[1] - cy)
            if d_to_center <= 7.0:
                relevant_frames.append((p.name, t_vid, p_L))

    print(f'[+] Found {len(relevant_frames)} keyframes within 7m of corridor center. Pre-loading images...')
    img_cache_cam0 = {}
    img_cache_cam1 = {}
    for fn, _, _ in relevant_frames:
        p0 = DATASET_DIR / f'images/cam0/{fn}'
        p1 = DATASET_DIR / f'images/cam1/{fn}'
        img_cache_cam0[fn] = cv2.imread(str(p0))
        img_cache_cam1[fn] = cv2.imread(str(p1))

    # Build 5 Hypotheses recommended by Gemini 4 Argon
    T_fact0 = np.array(calib_factory['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T_fact1 = np.array(calib_factory['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    T_rec0 = np.array(calib_recal['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T_rec1 = np.array(calib_recal['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)

    R_yaw_neg = R.from_euler('z', -1.5, degrees=True).as_matrix()
    R_yaw_pos = R.from_euler('z', +1.5, degrees=True).as_matrix()

    T_rec0_yaw_neg = T_rec0.copy()
    T_rec0_yaw_neg[:3, :3] = T_rec0[:3, :3] @ R_yaw_neg.T
    T_rec1_yaw_neg = T_rec1.copy()
    T_rec1_yaw_neg[:3, :3] = T_rec1[:3, :3] @ R_yaw_neg.T

    T_rec0_yaw_pos = T_rec0.copy()
    T_rec0_yaw_pos[:3, :3] = T_rec0[:3, :3] @ R_yaw_pos.T
    T_rec1_yaw_pos = T_rec1.copy()
    T_rec1_yaw_pos[:3, :3] = T_rec1[:3, :3] @ R_yaw_pos.T

    candidates = [
        ('Cand 1: Factory Nom (dt=3.337s)', 3.3368, T_fact0, T_fact1),
        ('Cand 2: Factory Nom (dt=3.377s)', 3.3768, T_fact0, T_fact1),
        ('Cand 3: Recalib SfM Rig (dt=3.377s)', 3.3768, T_rec0, T_rec1),
        ('Cand 4: Recalib + Yaw -1.5 deg', 3.3768, T_rec0_yaw_neg, T_rec1_yaw_neg),
        ('Cand 5: Recalib + Yaw +1.5 deg', 3.3768, T_rec0_yaw_pos, T_rec1_yaw_pos),
    ]

    candidate_results = []
    for cand_name, dt_cand, T0, T1 in candidates:
        print(f'[*] Evaluating {cand_name}...')
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]

        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        cand_cols = np.zeros((len(pts), 3), dtype=np.uint8)

        for fn, t_vid, _ in relevant_frames:
            t_query = t_slam[0] + (t_vid - dt_cand)
            if not (t_slam_start <= t_query <= t_slam_end):
                continue

            idx = np.clip(np.searchsorted(t_slam, t_query), 1, len(t_slam) - 1)
            w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_query).as_matrix()

            pts_body = (R_L.T @ (pts - p_L).T).T

            # Cam0
            pts_c0 = (R_LC0 @ pts_body.T).T + t_LC0
            front_mask0 = pts_c0[:, 2] > 0.3
            if np.any(front_mask0):
                u0, v0, _ = project_thin_prism(pts_c0[front_mask0], params0)
                valid0 = (u0 >= 0) & (u0 < 2880) & (v0 >= 0) & (v0 < 2880)
                if np.any(valid0):
                    sub_idx = np.where(front_mask0)[0][valid0]
                    dists = pts_c0[front_mask0, 2][valid0]
                    closer = dists < best_dist[sub_idx]
                    if np.any(closer):
                        img0 = img_cache_cam0[fn]
                        take_idx = sub_idx[closer]
                        u_take = np.clip(np.round(u0[valid0][closer]).astype(int), 0, 2879)
                        v_take = np.clip(np.round(v0[valid0][closer]).astype(int), 0, 2879)
                        cand_cols[take_idx] = img0[v_take, u_take][:, ::-1]
                        best_dist[take_idx] = dists[closer]

            # Cam1
            pts_c1 = (R_LC1 @ pts_body.T).T + t_LC1
            front_mask1 = pts_c1[:, 2] > 0.3
            if np.any(front_mask1):
                u1, v1, _ = project_thin_prism(pts_c1[front_mask1], params1)
                valid1 = (u1 >= 0) & (u1 < 2880) & (v1 >= 0) & (v1 < 2880)
                if np.any(valid1):
                    sub_idx = np.where(front_mask1)[0][valid1]
                    dists = pts_c1[front_mask1, 2][valid1]
                    closer = dists < best_dist[sub_idx]
                    if np.any(closer):
                        img1 = img_cache_cam1[fn]
                        take_idx = sub_idx[closer]
                        u_take = np.clip(np.round(u1[valid1][closer]).astype(int), 0, 2879)
                        v_take = np.clip(np.round(v1[valid1][closer]).astype(int), 0, 2879)
                        cand_cols[take_idx] = img1[v_take, u_take][:, ::-1]
                        best_dist[take_idx] = dists[closer]

        cov = np.count_nonzero(np.any(cand_cols > 0, axis=1)) / len(pts) * 100.0
        print(f'[+] {cand_name}: coverage = {cov:.1f}%')
        candidate_results.append((cand_name, cand_cols))

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 5, figsize=(32, 7), dpi=220, facecolor='#111115')
    for ax, (c_name, c_cols) in zip(axes, candidate_results):
        ax.set_facecolor('#16161c')
        valid = np.any(c_cols > 0, axis=1)
        ax.scatter(pts[valid, 0], pts[valid, 1], c=c_cols[valid]/255.0, s=2.8, rasterized=True)
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

    for idx_c in range(5):
        c_name, c_cols = candidate_results[idx_c]
        fig_s, ax_s = plt.subplots(figsize=(11, 11), dpi=260, facecolor='#111115')
        ax_s.set_facecolor('#16161c')
        valid = np.any(c_cols > 0, axis=1)
        ax_s.scatter(pts[valid, 0], pts[valid, 1], c=c_cols[valid]/255.0, s=3.5, rasterized=True)
        ax_s.set_title(f'{c_name} - Close-up High-Res (HarmonyMall)', color='white', fontsize=13, fontweight='bold', pad=10)
        ax_s.set_xlabel('X (m)', color='white')
        ax_s.set_ylabel('Y (m)', color='white')
        ax_s.tick_params(colors='white')
        ax_s.axis('equal')
        ax_s.set_xlim(cx - hw, cx + hw)
        ax_s.set_ylim(cy - hw, cy + hw)
        ax_s.grid(True, color='#2c2c36', linestyle='--', alpha=0.5)
        single_out = DATASET_DIR / f'candidates_zoom/mall_candidate_{idx_c+1}_detail.png'
        plt.tight_layout()
        plt.savefig(str(single_out), dpi=260, facecolor=fig_s.get_facecolor(), edgecolor='none')
        plt.close()
        print(f'[+] Standalone detail saved: {single_out}')

if __name__ == '__main__':
    main()
