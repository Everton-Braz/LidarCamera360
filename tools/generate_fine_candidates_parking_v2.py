#!/usr/bin/env python3
"""
Argon-Compliant Candidate Generator & Closer Post Renderer for HarmonyMall (v0.3.0)
- Preserves 100% video frame density (no subsampling)
- Baseline dt = -6.75549s (the real convergent temporal sync)
- Sweeps fine dt (-30ms to +30ms) and fine orientation (Pitch/Yaw/Roll)
- Generates both 7m x 7m overview and 2.5m x 2.5m ultra-close-up post on parking stall lines
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
OUT_DIR = DATASET_DIR / 'candidates_zoom'
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

def make_rotated_T(T_in, roll_d=0.0, pitch_d=0.0, yaw_d=0.0):
    T = T_in.copy()
    R_delta = R.from_euler('xyz', [roll_d, pitch_d, yaw_d], degrees=True).as_matrix()
    T[:3, :3] = T[:3, :3] @ R_delta.T
    return T

def main():
    print('[*] Loading SLAM trajectory...')
    trj_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'
    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_start, t_end = t_slam[0], t_slam[-1]

    start_pos = pos_slam[0]
    cx, cy = start_pos[0], start_pos[1]
    print(f'    Start position: ({cx:.4f}, {cy:.4f})')

    print('[*] Loading LiDAR point cloud around start...')
    ply_path = DATASET_DIR / 'deliverables/lidar_colored_direct_rigid_20261021_211521.ply'
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)

    hw_tight = 3.5
    mask = (
        (pts_all[:, 0] >= cx - hw_tight) & (pts_all[:, 0] <= cx + hw_tight) &
        (pts_all[:, 1] >= cy - hw_tight) & (pts_all[:, 1] <= cy + hw_tight) &
        (pts_all[:, 2] >= -2.2) & (pts_all[:, 2] <= 1.0)
    )
    pts = pts_all[mask]
    print(f'[+] Extracted {len(pts):,} points in 7m x 7m ROI.')

    # Profile Montagem 1
    profile_path = Path('C:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/configs/profiles/raven_x4_montagem_1.json')
    with open(profile_path) as f:
        profile = json.load(f)

    T0_base = np.array(profile['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T1_base = np.array(profile['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    p0 = profile['cam0_front_intrinsics']
    p1 = profile['cam1_rear_intrinsics']

    # Timestamps
    with open(DATASET_DIR / 'images/frames.json') as f:
        fj = json.load(f)
    ts_dict = fj.get('timestamps', {})
    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    frame_times = [(p.name, float(ts_dict.get(f'cam0/{p.name}', 0.0))) for p in cam0_files]

    dt_base = -6.755490
    rel_frames = [
        ft for ft in frame_times
        if np.hypot(pos_slam[np.clip(np.searchsorted(t_slam, t_slam[0] + ft[1] - dt_base), 0, len(t_slam)-1), 0] - cx,
                    pos_slam[np.clip(np.searchsorted(t_slam, t_slam[0] + ft[1] - dt_base), 0, len(t_slam)-1), 1] - cy) <= 12.0
    ]
    print(f'[+] All relevant frames in passage (100% density): {len(rel_frames)}')

    # Preload frames in memory for high-speed colorization sweeps
    print('[*] Preloading passage frames into memory...')
    img0_cache = {fn: cv2.imread(str(DATASET_DIR / f'images/cam0/{fn}')) for fn, _ in rel_frames}
    img1_cache = {fn: cv2.imread(str(DATASET_DIR / f'images/cam1/{fn}')) for fn, _ in rel_frames}
    print('[+] Preloaded frames successfully.')

    candidates = [
        (
            'cand_01_baseline_montagem1',
            'Candidato 1: Baseline Montagem 1 (dt = -6.755s)',
            dt_base,
            T0_base, T1_base
        ),
        (
            'cand_02_dt_minus_20ms',
            'Candidato 2: dt = -6.775s (-20ms)',
            dt_base - 0.020,
            T0_base, T1_base
        ),
        (
            'cand_03_dt_plus_20ms',
            'Candidato 3: dt = -6.735s (+20ms)',
            dt_base + 0.020,
            T0_base, T1_base
        ),
        (
            'cand_04_pitch_minus_0.25deg',
            'Candidato 4: Pitch -0.25 deg (dt = -6.755s)',
            dt_base,
            make_rotated_T(T0_base, pitch_d=-0.25), make_rotated_T(T1_base, pitch_d=-0.25)
        ),
        (
            'cand_05_pitch_plus_0.25deg',
            'Candidato 5: Pitch +0.25 deg (dt = -6.755s)',
            dt_base,
            make_rotated_T(T0_base, pitch_d=+0.25), make_rotated_T(T1_base, pitch_d=+0.25)
        ),
        (
            'cand_06_yaw_minus_0.30deg',
            'Candidato 6: Yaw -0.30 deg (dt = -6.755s)',
            dt_base,
            make_rotated_T(T0_base, yaw_d=-0.30), make_rotated_T(T1_base, yaw_d=-0.30)
        ),
        (
            'cand_07_yaw_plus_0.30deg',
            'Candidato 7: Yaw +0.30 deg (dt = -6.755s)',
            dt_base,
            make_rotated_T(T0_base, yaw_d=+0.30), make_rotated_T(T1_base, yaw_d=+0.30)
        ),
        (
            'cand_08_roll_minus_0.25deg',
            'Candidato 8: Roll -0.25 deg (dt = -6.755s)',
            dt_base,
            make_rotated_T(T0_base, roll_d=-0.25), make_rotated_T(T1_base, roll_d=-0.25)
        ),
    ]

    results = []
    for tag, title, dt_eval, T0, T1 in candidates:
        print(f'[*] Evaluating {title}...')
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]
        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        colors = np.zeros((len(pts), 3), dtype=np.uint8)

        for fn, t_vid in rel_frames:
            t_q = t_slam[0] + (t_vid - dt_eval)
            if not (t_start <= t_q <= t_end):
                continue
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            R_L = slerp(t_q).as_matrix()
            pts_body = (R_L.T @ (pts - p_L).T).T

            # Cam0
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
                        tk = sub[closer]
                        img0 = img0_cache[fn]
                        colors[tk] = img0[np.clip(v0[val0][closer].round().astype(int), 0, 3839), np.clip(u0[val0][closer].round().astype(int), 0, 3839)][:, ::-1]
                        best_dist[tk] = dists[closer]

            # Cam1
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
                        tk = sub[closer]
                        img1 = img1_cache[fn]
                        colors[tk] = img1[np.clip(v1[val1][closer].round().astype(int), 0, 3839), np.clip(u1[val1][closer].round().astype(int), 0, 3839)][:, ::-1]
                        best_dist[tk] = dists[closer]

        hsv = cv2.cvtColor(colors.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
        mask_y = (hsv[:,:,0] > 18) & (hsv[:,:,0] < 36) & (hsv[:,:,1] > 60) & (hsv[:,:,2] > 60)
        n_yellow_pts = int(np.sum(mask_y))
        print(f'  [+] Yellow points: {n_yellow_pts:,} / {len(pts):,}')

        # 1. Overview Plot (7m x 7m)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=220, facecolor='#0e0e13')
        ax.set_facecolor('#14141c')
        val = np.any(colors > 0, axis=1)
        ax.scatter(pts[val, 0], pts[val, 1], c=colors[val]/255.0, s=4.5, alpha=0.98, rasterized=True)
        ax.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=3.5, label='Trajetória SLAM', zorder=10)
        ax.scatter(cx, cy, color='#00ff66', s=250, marker='o', edgecolors='white', linewidth=2.5, label=f'Início ({cx:.2f}, {cy:.2f})', zorder=15)

        # Box demarcating the Closer Post ROI on parking stall lines (around x=-0.5, y=+1.2)
        zoom_cx, zoom_cy, zoom_hw = -0.5, 1.2, 1.25
        rect = plt.Rectangle((zoom_cx - zoom_hw, zoom_cy - zoom_hw), 2*zoom_hw, 2*zoom_hw, fill=False, edgecolor='#ff0055', linewidth=2.5, linestyle='--', label='Janela Closer Post (2.5m x 2.5m)', zorder=20)
        ax.add_patch(rect)

        ax.set_title(f'{title}\nVisão Geral (7m x 7m) | Amostragem 100% Frames', color='white', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Eixo X - Lateral (m)', color='white')
        ax.set_ylabel('Eixo Y - Longitudinal (m)', color='white')
        ax.tick_params(colors='white')
        ax.axis('equal')
        ax.set_xlim(cx - hw_tight, cx + hw_tight)
        ax.set_ylim(cy - hw_tight, cy + hw_tight)
        ax.grid(True, color='#252533', linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=11)
        out_ov = OUT_DIR / f'{tag}_overview_7m.png'
        plt.savefig(str(out_ov), dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # 2. Closer Post Plot (2.5m x 2.5m Ultra-Close-up on Yellow Stall Lines)
        fig, ax = plt.subplots(figsize=(10, 10), dpi=260, facecolor='#0e0e13')
        ax.set_facecolor('#14141c')
        m_zoom = (
            (pts[:, 0] >= zoom_cx - zoom_hw) & (pts[:, 0] <= zoom_cx + zoom_hw) &
            (pts[:, 1] >= zoom_cy - zoom_hw) & (pts[:, 1] <= zoom_cy + zoom_hw) &
            val
        )
        ax.scatter(pts[m_zoom, 0], pts[m_zoom, 1], c=colors[m_zoom]/255.0, s=7.0, alpha=0.98, rasterized=True)
        ax.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=3.0, alpha=0.8, label='Trajetória SLAM')
        ax.set_title(f'{title}\nCloser Post: Detalhe das Faixas de Estacionamento (2.5m x 2.5m)', color='white', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Eixo X - Lateral (m)', color='white')
        ax.set_ylabel('Eixo Y - Longitudinal (m)', color='white')
        ax.tick_params(colors='white')
        ax.axis('equal')
        ax.set_xlim(zoom_cx - zoom_hw, zoom_cx + zoom_hw)
        ax.set_ylim(zoom_cy - zoom_hw, zoom_cy + zoom_hw)
        ax.grid(True, color='#252533', linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', facecolor='#1b1b26', edgecolor='white', labelcolor='white', fontsize=11)
        out_cl = OUT_DIR / f'{tag}_closer_post_2.5m.png'
        plt.savefig(str(out_cl), dpi=260, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()

        # Measure edge gradient on closer post image
        img_cl = cv2.imread(str(out_cl))
        hsv_cl = cv2.cvtColor(img_cl, cv2.COLOR_BGR2HSV)
        my_cl = (hsv_cl[:,:,0] > 18) & (hsv_cl[:,:,0] < 36) & (hsv_cl[:,:,1] > 60) & (hsv_cl[:,:,2] > 60)
        n_y_cl = int(np.sum(my_cl))
        gray = cv2.cvtColor(img_cl, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        score = float(mag[my_cl].mean()) if n_y_cl > 0 else 0.0
        print(f'  [+] Closer post: yellow_pixels={n_y_cl:,}, edge_gradient={score:.2f}')
        results.append({
            'tag': tag,
            'title': title,
            'yellow_pts': n_yellow_pts,
            'closer_yellow_pixels': n_y_cl,
            'edge_gradient': score,
            'overview_img': str(out_ov),
            'closer_img': str(out_cl)
        })

    with open(OUT_DIR / 'candidates_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('[SUCCESS] All candidates evaluated and closer posts generated!')

if __name__ == '__main__':
    main()
