#!/usr/bin/env python3
"""
High-resolution 15-meter Zoom Evaluation of the Pier Deck.
Aligns the pier horizontally along the screen to clearly show wooden planks and railings.
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
    print('[*] Loading PCD and SLAM trajectory for Pier Deck Zoom...')
    pcd_path = DATASET_DIR / 'slam_raw_f8a83ade028dd052024e/pcd/all_raw_points.pcd'
    trj_path = DATASET_DIR / 'slam_raw_f8a83ade028dd052024e/result/Raven_3DMakerPro_Scan.txt'
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]
    
    # Pier axis vector
    v_pier = np.array([15.65875, -24.62084])
    v_dir = v_pier / np.linalg.norm(v_pier)
    angle_rot = -np.arctan2(v_dir[1], v_dir[0])
    c_rot, s_rot = np.cos(angle_rot), np.sin(angle_rot)
    R_align2d = np.array([[c_rot, -s_rot], [s_rot, c_rot]])
    
    # Center of 15m zoom window along the pier walkway (from 6m to 21m along pier axis)
    s_center = 13.5
    center_2d = s_center * v_dir
    
    # Distance from center
    d_from_center = np.linalg.norm(pts_all[:, :2] - center_2d, axis=1)
    mask_section = d_from_center <= 7.5
    pts = pts_all[mask_section]
    print(f'[+] Extracted 15m deck section: {len(pts):,} points')
    
    # Aligned coordinates for visualization
    pts_aligned_xy = (R_align2d @ pts[:, :2].T).T
    
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
        
    dt_candidates = [
        (-6.7555, 'Candidate 1: dt = -6.755s (v0.2.0 Exact)'),
        (-6.9300, 'Candidate Mid: dt = -6.930s (Interpolated)'),
        (-7.1063, 'Candidate 5: dt = -7.106s (Raw Gyro Lag)'),
        (-7.2500, 'Candidate Lead: dt = -7.250s (Early Lag)'),
    ]
    
    images_cache = {}
    comp_images = []
    
    for cand_idx, (dt, cand_name) in enumerate(dt_candidates):
        print(f'[*] Evaluating Zoom: {cand_name}...')
        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        cand_colors = np.zeros((len(pts), 3), dtype=np.uint8)
        
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
            
            # Check if camera is within 12m of the 15m section center
            if np.linalg.norm(p_L[:2] - center_2d) > 12.0:
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
                dists = np.linalg.norm(P_cam[idx_front], axis=1)
                
                # Sharp optical zone (< 1580px) and close range (< 5.0m) for maximum plank resolution
                valid = (r_px < 1580.0) & (u >= 0) & (u < 3839) & (v >= 0) & (v < 3839) & (dists < 5.0)
                if not np.any(valid):
                    continue
                    
                u_v = u[valid]
                v_v = v[valid]
                d_v = dists[valid]
                target_idx = idx_front[valid]
                
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
                    cand_colors[win_idx] = img[v_c, u_c][:, ::-1]
                    best_dist[win_idx] = win_d
                    
        colored_count = np.sum(best_dist < 999.0)
        print(f'[+] Deck points colored: {colored_count:,} ({colored_count/len(pts)*100.1:.1f}%)')
        
        # Render high-resolution dual-view detail plot
        fig, (ax_top, ax_prof) = plt.subplots(2, 1, figsize=(15, 7.5), dpi=160)
        
        # Filter points that received color for crisp display
        has_c = best_dist < 999.0
        pts_c_xy = pts_aligned_xy[has_c]
        pts_c_z = pts[has_c, 2]
        col_c = cand_colors[has_c] / 255.0
        
        # Subsample for scatter rendering if dense
        if len(pts_c_xy) > 100000:
            sub = np.random.choice(len(pts_c_xy), 100000, replace=False)
            pts_c_xy = pts_c_xy[sub]
            pts_c_z = pts_c_z[sub]
            col_c = col_c[sub]
            
        # 1. Aligned Top-Down View of Deck Planks
        ax_top.scatter(pts_c_xy[:, 0], pts_c_xy[:, 1], c=col_c, s=1.5, alpha=0.9)
        ax_top.set_aspect('equal')
        ax_top.set_title(f'{cand_name} — Deck Planks Top-Down (15-Meter Zoom)', fontsize=12, fontweight='bold')
        ax_top.set_xlabel('Along Pier Axis (m)')
        ax_top.set_ylabel('Across Pier (m)')
        ax_top.grid(True, linestyle='--', alpha=0.3)
        
        # 2. Side Elevation showing Railing / Balusters and Floor Deck
        ax_prof.scatter(pts_c_xy[:, 0], pts_c_z, c=col_c, s=1.5, alpha=0.9)
        ax_prof.set_aspect('equal')
        ax_prof.set_title('Pier Railing & Floor Profile (Side Elevation)', fontsize=11, fontweight='bold')
        ax_prof.set_xlabel('Along Pier Axis (m)')
        ax_prof.set_ylabel('Height Z (m)')
        ax_prof.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        out_png = OUTPUT_DIR / f'zoom_cand_{cand_idx + 1}.png'
        plt.savefig(out_png)
        plt.close()
        comp_images.append(out_png)
        print(f'[+] Rendered: {out_png}')
        
    # Build single stacked comparison PNG
    fig, axes = plt.subplots(len(comp_images), 1, figsize=(15, 6 * len(comp_images)), dpi=130)
    for idx, png_p in enumerate(comp_images):
        im = plt.imread(str(png_p))
        axes[idx].imshow(im)
        axes[idx].axis('off')
    plt.tight_layout()
    final_comp = OUTPUT_DIR / 'zoom_deck_planks_comparison.png'
    plt.savefig(final_comp)
    plt.close()
    print(f'\n[SUCCESS] Final 15m Zoom Comparison saved at: {final_comp}')

if __name__ == '__main__':
    main()
