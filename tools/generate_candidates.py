#!/usr/bin/env python3
"""
Fast Candidate Generator for PIER Dataset Colorization
Tests multiple calibration, temporal synchronization, and lens orientation hypotheses
without rebuilding the PyInstaller app.
"""

import json
import math
import os
import struct
import sys
import time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt

WORKSPACE = Path('C:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator')
DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output')
OUTPUT_DIR = Path('D:/ARQUIVOS_TESTE_2/PIER/candidates')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_slam_trajectory(trj_path):
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]  # [qx, qy, qz, qw]
    rots = R.from_quat(quats)
    slerp = Slerp(t_slam, rots)
    return t_slam, pos_slam, slerp

def load_pcd_points(pcd_path, max_points=1000000):
    print(f'[*] Loading PCD points from {pcd_path} via Open3D...')
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    print(f'[+] Loaded {len(pts):,} points for candidate testing.')
    return pts

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

def sample_image_colors(img, u, v, valid):
    colors = np.zeros((len(u), 3), dtype=np.uint8)
    if not np.any(valid):
        return colors
    u_coords = np.clip(np.round(u[valid]).astype(int), 0, img.shape[1] - 1)
    v_coords = np.clip(np.round(v[valid]).astype(int), 0, img.shape[0] - 1)
    sampled = img[v_coords, u_coords][:, ::-1]
    colors[valid] = sampled
    return colors

def render_candidate_topdown(pts, colors, out_png_path, title):
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    sub = 100000
    if len(pts) > sub:
        idx = np.random.choice(len(pts), sub, replace=False)
        p_sub = pts[idx]
        c_sub = colors[idx] / 255.0
    else:
        p_sub = pts
        c_sub = colors / 255.0
    
    ax.scatter(p_sub[:, 0], p_sub[:, 1], c=c_sub, s=0.5, alpha=0.8)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png_path)
    plt.close()
    print(f'[+] Rendered: {out_png_path}')

def write_ply(filename, points, colors):
    valid = np.any(colors > 0, axis=1)
    p = points[valid]
    c = colors[valid]
    with open(filename, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write(f'element vertex {len(p)}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(p, c):
            f.write(f'{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n')

def main():
    print('[*] Fast Candidate Generation starting...')
    pcd_path = DATASET_DIR / 'slam_raw_f8a83ade028dd052024e/pcd/all_raw_points.pcd'
    if not pcd_path.exists():
        pcd_path = DATASET_DIR / 'all_raw_points.pcd'
    trj_path = DATASET_DIR / 'slam_raw_f8a83ade028dd052024e/result/Raven_3DMakerPro_Scan.txt'
    
    pts = load_pcd_points(pcd_path, max_points=600000)
    t_slam, pos_slam, slerp = load_slam_trajectory(trj_path)
    t_slam_start = t_slam[0]
    t_slam_end = t_slam[-1]
    
    calib_v020_path = WORKSPACE / 'calibracao_rigida_raven_insta360.json'
    if not calib_v020_path.exists():
        calib_v020_path = WORKSPACE / 'configs/calibracao_rigida_raven_insta360.json'
    with open(calib_v020_path, 'r', encoding='utf-8') as f:
        calib_v020 = json.load(f)
        
    T_LC0_v020 = np.array(calib_v020['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T_LC1_v020 = np.array(calib_v020['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    p0 = calib_v020['cam0_front_intrinsics']
    params0_v020 = (p0['fx'], p0['fy'], p0['cx'], p0['cy'], p0['k1'], p0['k2'], p0['p1'], p0['p2'], p0['k3'], p0['k4'], p0['sx1'], p0['sy1'])
    p1 = calib_v020['cam1_rear_intrinsics']
    params1_v020 = (p1['fx'], p1['fy'], p1['cx'], p1['cy'], p1['k1'], p1['k2'], p1['p1'], p1['p2'], p1['k3'], p1['k4'], p1['sx1'], p1['sy1'])
    
    # Candidate 5: test with dt = -7.1063 (direct raw gyro lag without SLAM delay adjustment)
    candidates = [
        {
            'id': 'cand_1_v020_dt_neg675',
            'name': 'Candidate 1: v0.2.0 Exact (dt = -6.75s, v0.2.0 Extrinsics)',
            'dt': -6.75,
            'T_LC0': T_LC0_v020,
            'T_LC1': T_LC1_v020,
            'params0': params0_v020,
            'params1': params1_v020,
            'swap_lenses': False,
        },
        {
            'id': 'cand_2_v020_dt_pos675',
            'name': 'Candidate 2: Inverted dt (dt = +6.75s, v0.2.0 Extrinsics)',
            'dt': 6.75,
            'T_LC0': T_LC0_v020,
            'T_LC1': T_LC1_v020,
            'params0': params0_v020,
            'params1': params1_v020,
            'swap_lenses': False,
        },
        {
            'id': 'cand_3_v020_dt_neg1277',
            'name': 'Candidate 3: IMU Shift to SLAM Origin (dt = -12.77s)',
            'dt': -12.77,
            'T_LC0': T_LC0_v020,
            'T_LC1': T_LC1_v020,
            'params0': params0_v020,
            'params1': params1_v020,
            'swap_lenses': False,
        },
        {
            'id': 'cand_4_lens_yaw_flip',
            'name': 'Candidate 4: Lenses Swapped / 180 deg Yaw (dt = -6.75s)',
            'dt': -6.75,
            'T_LC0': T_LC1_v020,
            'T_LC1': T_LC0_v020,
            'params0': params1_v020,
            'params1': params0_v020,
            'swap_lenses': True,
        },
        {
            'id': 'cand_5_raw_gyro_neg710',
            'name': 'Candidate 5: Raw Gyro Lag without SLAM delay (dt = -7.1063s)',
            'dt': -7.1063,
            'T_LC0': T_LC0_v020,
            'T_LC1': T_LC1_v020,
            'params0': params0_v020,
            'params1': params1_v020,
            'swap_lenses': False,
        },
    ]
    
    cam0_dir = DATASET_DIR / 'images' / 'cam0'
    cam1_dir = DATASET_DIR / 'images' / 'cam1'
    cam0_files = sorted(list(cam0_dir.glob('*.jpg')))
    cam1_files = sorted(list(cam1_dir.glob('*.jpg')))
    
    frames_json_path = DATASET_DIR / 'images' / 'frames.json'
    video_times = {}
    if frames_json_path.exists():
        with open(frames_json_path, 'r') as f:
            fj = json.load(f)
            ts_dict = fj.get('timestamps', {})
            for img_p in cam0_files:
                k = f'cam0/{img_p.name}'
                if k in ts_dict:
                    video_times[img_p.name] = float(ts_dict[k])
                else:
                    digits = ''.join(filter(str.isdigit, img_p.stem))
                    video_times[img_p.name] = (int(digits) - 1) / 1.0
    
    print(f'[*] Found {len(cam0_files)} cam0 images, {len(cam1_files)} cam1 images, {len(video_times)} timestamps.')
    
    stride = 4
    sample_cam0 = cam0_files[::stride]
    sample_cam1 = cam1_files[::stride]
    
    for cand in candidates:
        c_id = cand['id']
        c_name = cand['name']
        dt = cand['dt']
        T0 = cand['T_LC0']
        T1 = cand['T_LC1']
        p0_cfg = cand['params0']
        p1_cfg = cand['params1']
        
        print(f'\n==================================================')
        print(f'[*] Evaluating {c_name}...')
        
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]
        
        cand_colors = np.zeros((len(pts), 3), dtype=np.uint8)
        colored_counts = np.zeros(len(pts), dtype=np.int32)
        accum_colors = np.zeros((len(pts), 3), dtype=np.float32)
        
        for k in range(len(sample_cam0)):
            f0_path = sample_cam0[k]
            f1_path = sample_cam1[k] if k < len(sample_cam1) else None
            t_vid = video_times.get(f0_path.name, (k * stride) * 1.0)
            
            t_query = t_slam_start + (t_vid - dt)
            if not (t_slam_start <= t_query <= t_slam_end):
                continue
                
            idx = np.searchsorted(t_slam, t_query)
            idx = np.clip(idx, 1, len(t_slam) - 1)
            w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_query).as_matrix()
            
            lenses = [('cam0', f0_path, p0_cfg, R_LC0, t_LC0)]
            if f1_path is not None and not cand['swap_lenses']:
                lenses.append(('cam1', f1_path, p1_cfg, R_LC1, t_LC1))
            elif f1_path is not None and cand['swap_lenses']:
                lenses.append(('cam1', f1_path, p0_cfg, R_LC1, t_LC1))
                
            for l_name, img_file, l_params, R_LC, t_LC in lenses:
                p_cam = p_L + R_L @ t_LC
                R_cw = (R_L @ R_LC).T
                P_cam = (R_cw @ (pts - p_cam).T).T
                
                idx_valid = np.flatnonzero(P_cam[:, 2] > 0.15)
                if len(idx_valid) == 0:
                    continue
                    
                u, v, _ = project_thin_prism(P_cam[idx_valid], l_params)
                r_px = np.hypot(u - l_params[2], v - l_params[3])
                camera_radius = 1620.0
                
                valid = ((r_px < camera_radius) & (u >= 0) & (u < 3839) & (v >= 0) & (v < 3839))
                if not np.any(valid):
                    continue
                    
                u_v = u[valid]
                v_v = v[valid]
                global_valid_idx = idx_valid[valid]
                
                img = cv2.imread(str(img_file))
                if img is not None:
                    u_coords = np.clip(np.round(u_v).astype(int), 0, img.shape[1] - 1)
                    v_coords = np.clip(np.round(v_v).astype(int), 0, img.shape[0] - 1)
                    c_samp = img[v_coords, u_coords][:, ::-1]  # BGR to RGB
                    accum_colors[global_valid_idx] += c_samp
                    colored_counts[global_valid_idx] += 1
                        
        has_color = colored_counts > 0
        cand_colors[has_color] = np.clip(accum_colors[has_color] / colored_counts[has_color, None], 0, 255).astype(np.uint8)
        coverage = np.sum(has_color) / len(pts) * 100.0
        print(f'[+] Coverage: {coverage:.1f}% ({np.sum(has_color):,} points colored)')
        
        ply_out = OUTPUT_DIR / f'{c_id}.ply'
        png_out = OUTPUT_DIR / f'{c_id}_topdown.png'
        write_ply(ply_out, pts, cand_colors)
        render_candidate_topdown(pts, cand_colors, png_out, f'{c_name} (Coverage: {coverage:.1f}%)')
        
    fig, axes = plt.subplots(len(candidates), 1, figsize=(16, 4 * len(candidates)), dpi=120)
    for ax, cand in zip(axes, candidates):
        c_id = cand['id']
        png_cand = OUTPUT_DIR / f'{c_id}_topdown.png'
        img_cand = plt.imread(str(png_cand))
        ax.imshow(img_cand)
        ax.axis('off')
    plt.tight_layout()
    comp_png = OUTPUT_DIR / 'candidates_comparison.png'
    plt.savefig(comp_png)
    plt.close()
    print(f'\n[SUCCESS] All candidates generated! Summary plot at: {comp_png}')

if __name__ == '__main__':
    main()
