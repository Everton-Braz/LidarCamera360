#!/usr/bin/env python3
"""
Zoom-in candidate evaluation on a 15-meter section of the Pier Deck.
Compares fine variations between dt = -6.75s and dt = -7.11s with distance gating.
"""

import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import open3d as o3d

WORKSPACE = Path('C:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator')
DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output')
OUTPUT_DIR = Path('D:/ARQUIVOS_TESTE_2/PIER/candidates_zoom')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
    print('[*] Loading PCD and SLAM trajectory for 15m Deck Zoom...')
    pcd_path = DATASET_DIR / 'slam_raw_f8a83ade028dd052024e/pcd/all_raw_points.pcd'
    trj_path = DATASET_DIR / 'slam_raw_f8a83ade028dd052024e/result/Raven_3DMakerPro_Scan.txt'
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    
    # Select a dense 15-meter section of the pier deck (e.g. X between 25m and 40m)
    # Pier extends roughly along X
    x_min, x_max = 25.0, 40.0
    mask_section = (pts_all[:, 0] >= x_min) & (pts_all[:, 0] <= x_max)
    pts = pts_all[mask_section]
    print(f'[+] Filtered 15m deck section: {len(pts):,} points (from {len(pts_all):,})')
    
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]
    
    calib_path = WORKSPACE / 'calibracao_rigida_raven_insta360.json'
    with open(calib_path, 'r', encoding='utf-8') as f:
        calib = json.load(f)
        
    T_LC0 = np.array(calib['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T_LC1 = np.array(calib['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    R_LC0, t_LC0 = T_LC0[:3, :3], T_LC0[:3, 3]
    R_LC1, t_LC1 = T_LC1[:3, :3], T_LC1[:3, 3]
    
    p0 = calib['cam0_front_intrinsics']
    params0 = (p0['fx'], p0['fy'], p0['cx'], p0['cy'], p0['k1'], p0['k2'], p0['p1'], p0['p2'], p0['k3'], p0['k4'], p0['sx1'], p0['sy1'])
    p1 = calib['cam1_rear_intrinsics']
    params1 = (p1['fx'], p1['fy'], p1['cx'], p1['cy'], p1['k1'], p1['k2'], p1['p1'], p1['p2'], p1['k3'], p1['k4'], p1['sx1'], p1['sy1'])
    
    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    cam1_files = sorted(list((DATASET_DIR / 'images/cam1').glob('*.jpg')))
    
    with open(DATASET_DIR / 'images/frames.json', 'r') as f:
        fj = json.load(f)
    ts_dict = fj.get('timestamps', {})
    video_times = {}
    for p in cam0_files:
        k = f'cam0/{p.name}'
        video_times[p.name] = float(ts_dict.get(k, 0.0))
        
    # Fine candidate values around -6.75s and -7.11s
    dt_candidates = [
        (-6.7555, 'dt = -6.755s (v0.2.0 Exact)'),
        (-6.9300, 'dt = -6.930s (Midpoint)'),
        (-7.1063, 'dt = -7.106s (Raw Gyro Lag)'),
        (-7.2500, 'dt = -7.250s (Fine Lead)'),
    ]
    
    images_cache = {}
    for cand_idx, (dt, cand_name) in enumerate(dt_candidates):
        print(f'[*] Evaluating Zoom Candidate: {cand_name}...')
        
        # Closest frame distance testing (Z-buffering / Nearest camera projection)
        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        cand_colors = np.zeros((len(pts), 3), dtype=np.uint8)
        
        # Process every frame within range of the 15m section
        for k in range(len(cam0_files)):
            f0 = cam0_files[k]
            f1 = cam1_files[k] if k < len(cam1_files) else None
            t_vid = video_times[f0.name]
            t_query = t_slam_start + (t_vid - dt)
            
            if not (t_slam_start <= t_query <= t_slam_end):
                continue
                
            idx = np.clip(np.searchsorted(t_slam, t_query), 1, len(t_slam) - 1)
            w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            
            # Check if camera position is within 8m of the 15m section
            if p_L[0] < x_min - 8.0 or p_L[0] > x_max + 8.0:
                continue
                
            R_L = slerp(t_query).as_matrix()
            
            lenses = [('cam0', f0, params0, R_LC0, t_LC0)]
            if f1 is not None:
                lenses.append(('cam1', f1, params1, R_LC1, t_LC1))
                
            for l_name, img_path, l_params, R_LC, t_LC in lenses:
                p_cam = p_L + R_L @ t_LC
                R_cw = (R_L @ R_LC).T
                P_cam = (R_cw @ (pts - p_cam).T).T
                
                idx_front = np.flatnonzero(P_cam[:, 2] > 0.15)
                if len(idx_front) == 0:
                    continue
                    
                u, v, _ = project_thin_prism(P_cam[idx_front], l_params)
                r_px = np.hypot(u - l_params[2], v - l_params[3])
                
                # Only sharp optical zone (< 1550px) and distance < 6.0m
                dists = np.linalg.norm(P_cam[idx_front], axis=1)
                valid = (r_px < 1550.0) & (u >= 0) & (u < 3839) & (v >= 0) & (v < 3839) & (dists < 6.0)
                if not np.any(valid):
                    continue
                    
                u_v = u[valid]
                v_v = v[valid]
                d_v = dists[valid]
                target_idx = idx_front[valid]
                
                # Nearest frame winner-takes-all for maximum edge sharpness
                closer = d_v < best_dist[target_idx]
                if not np.any(closer):
                    continue
                    
                win_idx = target_idx[closer]
                win_u = u_v[closer]
                win_v = v_v[closer]
                win_d = d_v[closer]
                
                if str(img_path) not in images_cache:
                    img = cv2.imread(str(img_path))
                    images_cache[str(img_path)] = img
                else:
                    img = images_cache[str(img_path)]
                    
                if img is not None:
                    u_c = np.clip(np.round(win_u).astype(int), 0, img.shape[1] - 1)
                    v_c = np.clip(np.round(win_v).astype(int), 0, img.shape[0] - 1)
                    cand_colors[win_idx] = img[v_c, u_c][:, ::-1] # BGR to RGB
                    best_dist[win_idx] = win_d
                    
        colored_ratio = np.sum(best_dist < 999.0) / len(pts) * 100.0
        print(f'[+] Deck section colored: {colored_ratio:.1f}%')
        
        # Save 2D top-down detail plot
        fig, (ax_top, ax_persp) = plt.subplots(2, 1, figsize=(14, 9), dpi=150)
        
        # Top-down view (XY)
        sc1 = ax_top.scatter(pts[:, 0], pts[:, 1], c=cand_colors/255.0, s=1.2, alpha=0.9)
        ax_top.set_xlim(x_min, x_max)
        ax_top.set_ylim(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5)
        ax_top.set_aspect('equal')
        ax_top.set_title(f'{cand_name} — Top-Down Deck Planks View (X: 25m to 40m)', fontsize=12, fontweight='bold')
        ax_top.set_xlabel('X (m)')
        ax_top.set_ylabel('Y (m)')
        ax_top.grid(True, linestyle='--', alpha=0.3)
        
        # Side/Profile view (XZ) showing railing and floor boards
        sc2 = ax_persp.scatter(pts[:, 0], pts[:, 2], c=cand_colors/255.0, s=1.2, alpha=0.9)
        ax_persp.set_xlim(x_min, x_max)
        ax_persp.set_ylim(pts[:, 2].min() - 0.2, pts[:, 2].max() + 0.2)
        ax_persp.set_aspect('equal')
        ax_persp.set_title('Profile / Railing & Floor View (XZ)', fontsize=11, fontweight='bold')
        ax_persp.set_xlabel('X (m)')
        ax_persp.set_ylabel('Z (m)')
        ax_persp.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        out_png = OUTPUT_DIR / f'zoom_cand_{cand_idx + 1}_{dt:.3f}.png'
        plt.savefig(out_png)
        plt.close()
        print(f'[+] Rendered: {out_png}')
        
    # Combined grid plot for direct comparison
    fig, axes = plt.subplots(len(dt_candidates), 1, figsize=(16, 5 * len(dt_candidates)), dpi=140)
    for idx, (dt, cand_name) in enumerate(dt_candidates):
        img = plt.imread(str(OUTPUT_DIR / f'zoom_cand_{idx + 1}_{dt:.3f}.png'))
        axes[idx].imshow(img)
        axes[idx].axis('off')
    plt.tight_layout()
    comp_zoom = OUTPUT_DIR / 'zoom_deck_planks_comparison.png'
    plt.savefig(comp_zoom)
    plt.close()
    print(f'\n[SUCCESS] Zoom comparison generated at: {comp_zoom}')

if __name__ == '__main__':
    main()
