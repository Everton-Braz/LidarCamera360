#!/usr/bin/env python3
"""
Test Everton's Verified Profile 'Raven Lidar + Insta360 X4 (Montagem 1)' (from 20261022home)
on HarmonyMall Parking Spot Region (Yellow Lines at Y=0 to -30m)
"""
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import open3d as o3d

DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
OUT_DIR = DATASET_DIR / 'candidates_zoom'
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
    pcd_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/pcd/all_raw_points.pcd'
    trj_path = list(DATASET_DIR.glob('slam_raw_*/result/*Scan*.txt'))[0]
    
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)
    
    # Parking spots high-density area around Y = -12m (Y from -4 to -22m, X from -2 to 10m)
    cx, cy = 4.0, -14.0
    hw_x, hw_y = 6.0, 9.0
    mask = (
        (pts_all[:, 0] >= cx - hw_x) & (pts_all[:, 0] <= cx + hw_x) &
        (pts_all[:, 1] >= cy - hw_y) & (pts_all[:, 1] <= cy + hw_y) &
        (pts_all[:, 2] >= -2.2) & (pts_all[:, 2] <= 1.5)
    )
    pts = pts_all[mask]
    print(f'[+] Points in parking spots evaluation window: {len(pts):,}')
    
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]
    
    # Load 20261022home profile (Everton's Verified Profile)
    with open('D:/ARQUIVOS_TESTE_2/20261022home/rig_calibration.json') as f:
        home_calib = json.load(f)
    T0_home = np.array(home_calib['T_lidar_to_cam0_rigid_4x4'])
    T1_home = np.array(home_calib['T_lidar_to_cam1_rigid_4x4'])
    
    # Load nominal v0.2.0 (with corrupt Cam1 Ty sign)
    with open('calibracao_rigida_raven_insta360.json') as f:
        nom_calib = json.load(f)
    T0_nom = np.array(nom_calib['T_lidar_to_cam0_rigid_4x4'])
    T1_nom = np.array(nom_calib['T_lidar_to_cam1_rigid_4x4'])
    
    # Camera intrinsics (3840x3840)
    p0 = home_calib['cam0_front_intrinsics']
    params0 = (p0['fx'], p0['fy'], p0['cx'], p0['cy'], p0['k1'], p0['k2'], p0['p1'], p0['p2'], p0['k3'], p0['k4'], p0['sx1'], p0['sy1'])
    p1 = home_calib['cam1_rear_intrinsics']
    params1 = (p1['fx'], p1['fy'], p1['cx'], p1['cy'], p1['k1'], p1['k2'], p1['p1'], p1['p2'], p1['k3'], p1['k4'], p1['sx1'], p1['sy1'])
    
    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    with open(DATASET_DIR / 'images/frames.json') as f:
        fj = json.load(f)
    video_times = {p.name: float(fj['timestamps'].get(f'cam0/{p.name}', 0.0)) for p in cam0_files}
    
    dt_eval = 3.336827
    
    # Find frames relevant to parking area (radius 14m from cx, cy)
    relevant_frames = []
    for p in cam0_files:
        t_vid = video_times[p.name]
        t_q = t_slam[0] + (t_vid - dt_eval)
        if t_slam_start <= t_q <= t_slam_end:
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            if np.hypot(p_L[0] - cx, p_L[1] - cy) <= 14.0:
                relevant_frames.append((p.name, t_vid))
                
    print(f'[+] Relevant video frames: {len(relevant_frames)}')
    # Subsample frames to 25 evenly spaced frames to speed up and stay ultra light
    step = max(1, len(relevant_frames) // 25)
    eval_frames = relevant_frames[::step]
    print(f'[+] Using {len(eval_frames)} keyframes for evaluation')

    configs = [
        ('Calibração Antiga (Nominal v0.2.0 - Ty Invertido)', T0_nom, T1_nom),
        ('Novo Perfil: Raven + X4 Montagem 1 (Validado 20261022home)', T0_home, T1_home)
    ]

    results = []
    for title, T0, T1 in configs:
        print(f'[*] Rendering {title}...')
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]
        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        colors = np.zeros((len(pts), 3), dtype=np.uint8)

        for fn, t_vid in eval_frames:
            t_q = t_slam[0] + (t_vid - dt_eval)
            if not (t_slam_start <= t_q <= t_slam_end):
                continue
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_q).as_matrix()
            pts_body = (R_L.T @ (pts - p_L).T).T

            img0 = None
            img1 = None

            # Cam0 (3840x3840)
            pts_c0 = (R_LC0 @ pts_body.T).T + t_LC0
            m0 = pts_c0[:, 2] > 0.3
            if np.any(m0):
                u0, v0, _ = project_thin_prism(pts_c0[m0], params0)
                val0 = (u0 >= 0) & (u0 < 3840) & (v0 >= 0) & (v0 < 3840)
                if np.any(val0):
                    sub = np.where(m0)[0][val0]
                    dists = pts_c0[m0, 2][val0]
                    closer = dists < best_dist[sub]
                    if np.any(closer):
                        if img0 is None:
                            img0 = cv2.imread(str(DATASET_DIR / f'images/cam0/{fn}'))
                        tk = sub[closer]
                        colors[tk] = img0[np.clip(v0[val0][closer].round().astype(int), 0, 3839), np.clip(u0[val0][closer].round().astype(int), 0, 3839)][:, ::-1]
                        best_dist[tk] = dists[closer]

            # Cam1 (3840x3840)
            pts_c1 = (R_LC1 @ pts_body.T).T + t_LC1
            m1 = pts_c1[:, 2] > 0.3
            if np.any(m1):
                u1, v1, _ = project_thin_prism(pts_c1[m1], params1)
                val1 = (u1 >= 0) & (u1 < 3840) & (v1 >= 0) & (v1 < 3840)
                if np.any(val1):
                    sub = np.where(m1)[0][val1]
                    dists = pts_c1[m1, 2][val1]
                    closer = dists < best_dist[sub]
                    if np.any(closer):
                        if img1 is None:
                            img1 = cv2.imread(str(DATASET_DIR / f'images/cam1/{fn}'))
                        tk = sub[closer]
                        colors[tk] = img1[np.clip(v1[val1][closer].round().astype(int), 0, 3839), np.clip(u1[val1][closer].round().astype(int), 0, 3839)][:, ::-1]
                        best_dist[tk] = dists[closer]
                        
        results.append((title, colors))
        
    # Side by Side comparison figure of Parking Lines
    fig, axes = plt.subplots(1, 2, figsize=(24, 14), dpi=260, facecolor='#0e0e13')
    for ax, (title, cols) in zip(axes, results):
        ax.set_facecolor('#14141c')
        val = np.any(cols > 0, axis=1)
        ax.scatter(pts[val, 0], pts[val, 1], c=cols[val]/255.0, s=3.5, rasterized=True)
        ax.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=2.0, alpha=0.7, label='Trajetória SLAM')
        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('X - Lateral (m)', color='white')
        ax.set_ylabel('Y - Longitudinal (m)', color='white')
        ax.tick_params(colors='white')
        ax.axis('equal')
        ax.set_xlim(cx - hw_x, cx + hw_x)
        ax.set_ylim(cy - hw_y, cy + hw_y)
        ax.grid(True, color='#252533', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white')
        
    out_comp = OUT_DIR / 'parking_spots_old_vs_montagem1_comparison.png'
    plt.tight_layout()
    plt.savefig(str(out_comp), dpi=260, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f'[SUCCESS] Comparison of parking spots saved: {out_comp}')

if __name__ == '__main__':
    main()
