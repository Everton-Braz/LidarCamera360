#!/usr/bin/env python3
"""
Dense High-Fidelity Parking Stall Candidate Generator
Implements Argon's Plan: Continuous video buffer (no frame skips),
incidence-weighted multiview projection, and ultra close-up rendering (7m x 7m window).
"""
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp

DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
OUT_DIR = DATASET_DIR / 'candidates_dense_zoom'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def project_thin_prism(pts_c, p):
    fx, fy = p['fx'], p['fy']
    cx, cy = p['cx'], p['cy']
    k1, k2, k3, k4 = p['k1'], p['k2'], p['k3'], p['k4']
    p1, p2 = p['p1'], p['p2']
    sx1, sy1 = p['sx1'], p['sy1']

    x = pts_c[:, 0] / pts_c[:, 2]
    y = pts_c[:, 1] / pts_c[:, 2]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)

    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    scale = np.where(r > 1e-8, theta_d / r, 1.0)
    xd = x * scale
    yd = y * scale

    u = fx * (xd + 2 * p1 * xd * yd + p2 * (r**2 + 2 * xd**2) + sx1 * r**2) + cx
    v = fy * (yd + p1 * (r**2 + 2 * yd**2) + 2 * p2 * xd * yd + sy1 * r**2) + cy
    return u, v, pts_c[:, 2]

def main():
    print('[*] Loading SLAM trajectory...')
    trj_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_start, t_end = t_slam[0], t_slam[-1]

    cx, cy = pos_slam[0, 0], pos_slam[0, 1]
    print(f'    Start position: ({cx:.4f}, {cy:.4f})')

    print('[*] Loading point cloud...')
    ply_path = DATASET_DIR / 'deliverables/lidar_colored_direct_rigid_20261021_211521.ply'
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)

    # Crop 7m x 7m window around trajectory start (floor points)
    hw = 3.5
    mask = (
        (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) &
        (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) &
        (pts_all[:, 2] >= -2.2) & (pts_all[:, 2] <= 1.0)
    )
    pts = pts_all[mask]
    print(f'[+] Extracted {len(pts):,} points in 7m x 7m window.')

    # Base rig calibration
    with open(DATASET_DIR / 'deliverables/rig_calibration_20261022_074718.json') as f:
        base_calib = json.load(f)

    T0_base = np.array(base_calib['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T1_base = np.array(base_calib['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    p0 = base_calib['cam0_front_intrinsics']
    p1 = base_calib['cam1_rear_intrinsics']

    # Video frame timestamps
    with open(DATASET_DIR / 'images/frames.json') as f:
        fj = json.load(f)
    ts_dict = fj.get('timestamps', {})
    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    frame_times = [(p.name, float(ts_dict.get(f'cam0/{p.name}', 0.0))) for p in cam0_files]

    # Select ALL consecutive frames within 8 meters of origin
    dt_nominal = 3.336827
    relevant_frames = []
    for fn, t_vid in frame_times:
        t_q = t_slam[0] + (t_vid - dt_nominal)
        if t_start <= t_q <= t_end:
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            if np.hypot(p_L[0] - cx, p_L[1] - cy) <= 8.0:
                relevant_frames.append((fn, t_vid))

    print(f'[+] Full continuous frame window: {len(relevant_frames)} consecutive frames (no skip)')

    def make_rot(T_in, roll_d=0.0, pitch_d=0.0, yaw_d=0.0):
        T = T_in.copy()
        R_delta = R.from_euler('xyz', [roll_d, pitch_d, yaw_d], degrees=True).as_matrix()
        T[:3, :3] = T[:3, :3] @ R_delta.T
        return T

    # High-signal candidate grid based on Argon's plan:
    # Decimal temporal sweeps (dt) and Pitch/Yaw micro-rotations
    candidates = [
        (
            'candidate_dense_01_baseline_dt_3.337s',
            'Baseline Montagem 1 (dt = 3.337s)',
            3.336827, T0_base, T1_base
        ),
        (
            'candidate_dense_02_dt_3.357s',
            'Ajuste Temporal dt = 3.357s (+20ms)',
            3.356827, T0_base, T1_base
        ),
        (
            'candidate_dense_03_dt_3.377s',
            'Ajuste Temporal dt = 3.377s (+40ms)',
            3.376827, T0_base, T1_base
        ),
        (
            'candidate_dense_04_pitch_minus_0.30deg',
            'Pitch -0.30 deg (dt = 3.357s)',
            3.356827,
            make_rot(T0_base, pitch_d=-0.30), make_rot(T1_base, pitch_d=-0.30)
        ),
        (
            'candidate_dense_05_pitch_plus_0.30deg',
            'Pitch +0.30 deg (dt = 3.357s)',
            3.356827,
            make_rot(T0_base, pitch_d=+0.30), make_rot(T1_base, pitch_d=+0.30)
        ),
        (
            'candidate_dense_06_yaw_minus_0.35deg',
            'Yaw -0.35 deg (dt = 3.357s)',
            3.356827,
            make_rot(T0_base, yaw_d=-0.35), make_rot(T1_base, yaw_d=-0.35)
        ),
    ]

    for file_tag, title, dt_eval, T0, T1 in candidates:
        print(f'[*] Processing dense projection: {title}...')
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]

        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        colors = np.zeros((len(pts), 3), dtype=np.uint8)

        for fn, t_vid in relevant_frames:
            t_q = t_slam[0] + (t_vid - dt_eval)
            if not (t_start <= t_q <= t_end):
                continue
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_q).as_matrix()
            pts_body = (R_L.T @ (pts - p_L).T).T

            img0 = None
            img1 = None

            # Front Cam (Cam0)
            pts_c0 = (R_LC0 @ pts_body.T).T + t_LC0
            m0 = pts_c0[:, 2] > 0.3
            if np.any(m0):
                u0, v0, _ = project_thin_prism(pts_c0[m0], p0)
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

            # Rear Cam (Cam1)
            pts_c1 = (R_LC1 @ pts_body.T).T + t_LC1
            m1 = pts_c1[:, 2] > 0.3
            if np.any(m1):
                u1, v1, _ = project_thin_prism(pts_c1[m1], p1)
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

        val = np.any(colors > 0, axis=1)
        hsv = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        mask_y = (hsv[:, 0, 0] > 18) & (hsv[:, 0, 0] < 36) & (hsv[:, 0, 1] > 60) & (hsv[:, 0, 2] > 60)
        n_yellow = np.sum(mask_y)
        print(f'    Colored points: {np.sum(val):,} / {len(pts):,} | Yellow points: {n_yellow:,}')

        fig, ax = plt.subplots(figsize=(14, 14), dpi=260, facecolor='#0e0e13')
        ax.set_facecolor('#14141c')
        ax.scatter(pts[val, 0], pts[val, 1], c=colors[val] / 255.0, s=4.5, alpha=0.98, rasterized=True)
        ax.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=3.5, label='Trajetória SLAM', zorder=10)
        ax.scatter(cx, cy, color='#00ff66', s=300, marker='o', edgecolors='white', linewidth=2.8, label=f'Início Trajetória ({cx:.2f}, {cy:.2f})', zorder=15)
        ax.axvline(x=cx, color='#00ff66', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.axhline(y=cy, color='#00ff66', linestyle=':', alpha=0.5, linewidth=1.5)

        ax.set_title(f'{title}\nMarcas de Vagas no Piso (Janela Densa 7m x 7m)', color='white', fontsize=14, fontweight='bold', pad=14)
        ax.set_xlabel('Eixo X - Lateral (metros)', color='white', fontsize=12)
        ax.set_ylabel('Eixo Y - Longitudinal (metros)', color='white', fontsize=12)
        ax.tick_params(colors='white')
        ax.axis('equal')
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)
        ax.grid(True, color='#252533', linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=12)

        out_path = OUT_DIR / f'{file_tag}.png'
        plt.savefig(str(out_path), dpi=260, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close()
        print(f'[+] Saved: {out_path.name}')

    print('[SUCCESS] Dense candidate generation complete.')

if __name__ == '__main__':
    main()
