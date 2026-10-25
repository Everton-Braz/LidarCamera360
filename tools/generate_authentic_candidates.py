#!/usr/bin/env python3
"""
Authentic Candidate Generator anchored directly on the 20261021_211521 deliverable geometry.
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
OUT_DIR = DATASET_DIR / 'candidates_reference_grid'
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

def make_rot(T_in, roll_d=0.0, pitch_d=0.0, yaw_d=0.0):
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
    cx, cy = pos_slam[0, 0], pos_slam[0, 1]

    print('[*] Loading point cloud near trajectory start...')
    ply_path = DATASET_DIR / 'deliverables/lidar_colored_direct_rigid_20261021_211521.ply'
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)

    hw = 3.5
    mask = (
        (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) &
        (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) &
        (pts_all[:, 2] >= -2.2) & (pts_all[:, 2] <= 1.0)
    )
    pts = pts_all[mask]
    print(f'[+] Window points: {len(pts):,}')

    # Exact parameters from automatic_calibration_20261021_211521.json
    with open(DATASET_DIR / 'deliverables/automatic_calibration_20261021_211521.json') as f:
        ref_calib = json.load(f)

    T0_ref = np.array(ref_calib['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
    T1_ref = np.array(ref_calib['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    p0 = ref_calib['cam0_front_intrinsics']
    p1 = ref_calib['cam1_rear_intrinsics']

    with open(DATASET_DIR / 'images/frames.json') as f:
        fj = json.load(f)
    ts_dict = fj.get('timestamps', {})
    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    frame_times = [(p.name, float(ts_dict.get(f'cam0/{p.name}', 0.0))) for p in cam0_files]

    rel_frames = [
        ft for ft in frame_times
        if np.hypot(
            pos_slam[np.clip(np.searchsorted(t_slam, t_slam[0] + ft[1] - 3.337), 0, len(t_slam)-1), 0] - cx,
            pos_slam[np.clip(np.searchsorted(t_slam, t_slam[0] + ft[1] - 3.337), 0, len(t_slam)-1), 1] - cy
        ) <= 8.0
    ]
    print(f'[+] Continuous frames in window: {len(rel_frames)}')

    candidates = [
        ('cand_A_base_20261021', 'Candidato A: Base 20261021 Puro (dt=3.337s)', 3.336827, T0_ref, T1_ref),
        ('cand_B_dt_minus30ms', 'Candidato B: dt = 3.307s (-30ms)', 3.306827, T0_ref, T1_ref),
        ('cand_C_dt_plus30ms', 'Candidato C: dt = 3.367s (+30ms)', 3.366827, T0_ref, T1_ref),
        ('cand_D_pitch_minus025', 'Candidato D: Pitch -0.25 deg (dt=3.337s)', 3.336827, make_rot(T0_ref, pitch_d=-0.25), make_rot(T1_ref, pitch_d=-0.25)),
        ('cand_E_pitch_plus025', 'Candidato E: Pitch +0.25 deg (dt=3.337s)', 3.336827, make_rot(T0_ref, pitch_d=+0.25), make_rot(T1_ref, pitch_d=+0.25)),
        ('cand_F_yaw_minus030', 'Candidato F: Yaw -0.30 deg (dt=3.337s)', 3.336827, make_rot(T0_ref, yaw_d=-0.30), make_rot(T1_ref, yaw_d=-0.30)),
    ]

    rendered = {}
    for tag, title, dt_eval, T0, T1 in candidates:
        print(f'[*] Rendering {title}...')
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
            img0, img1 = None, None

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
                        if img0 is None:
                            img0 = cv2.imread(str(DATASET_DIR / f'images/cam0/{fn}'))
                        tk = sub[closer]
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
                        if img1 is None:
                            img1 = cv2.imread(str(DATASET_DIR / f'images/cam1/{fn}'))
                        tk = sub[closer]
                        colors[tk] = img1[np.clip(v1[val1][closer].round().astype(int), 0, 3839), np.clip(u1[val1][closer].round().astype(int), 0, 3839)][:, ::-1]
                        best_dist[tk] = dists[closer]

        val = np.any(colors > 0, axis=1)
        fig, ax = plt.subplots(figsize=(12, 12), dpi=220, facecolor='#0e0e13')
        ax.set_facecolor('#14141c')
        ax.scatter(pts[val, 0], pts[val, 1], c=colors[val]/255.0, s=4.0, alpha=0.98, rasterized=True)
        ax.plot(pos_slam[:, 0], pos_slam[:, 1], color='#00f0ff', linewidth=3.0, label='Trajetória SLAM', zorder=10)
        ax.scatter(cx, cy, color='#00ff66', s=250, marker='o', edgecolors='white', linewidth=2.5, zorder=15)
        ax.set_title(f'{title}\n(Janela 7m x 7m - Base 20261021)', color='white', fontsize=13, fontweight='bold', pad=12)
        ax.axis('equal'); ax.set_xlim(cx - hw, cx + hw); ax.set_ylim(cy - hw, cy + hw)
        ax.tick_params(colors='white'); ax.grid(True, color='#252533', linestyle='--', alpha=0.6)
        out_p = OUT_DIR / f'{tag}.png'
        plt.savefig(str(out_p), dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        print(f'[+] Rendered {tag}')
        rendered[tag] = cv2.imread(str(out_p))

    # Create side-by-side zoom collage
    base_img = rendered['cand_A_base_20261021']
    h, w, _ = base_img.shape
    # Crop the parking spot lines area in lower-right
    hsv = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    mask_y = (hsv[:,:,0] > 18) & (hsv[:,:,0] < 36) & (hsv[:,:,1] > 60) & (hsv[:,:,2] > 60)
    mask_y[:1200, :] = 0
    ys, xs = np.where(mask_y)
    y1, y2 = max(0, ys.min() - 30), min(h, ys.max() + 30)
    x1, x2 = max(0, xs.min() - 30), min(w, xs.max() + 30)

    crop_a = rendered['cand_A_base_20261021'][y1:y2, x1:x2]
    crop_b = rendered['cand_B_dt_minus30ms'][y1:y2, x1:x2]
    crop_d = rendered['cand_D_pitch_minus025'][y1:y2, x1:x2]
    comp = np.hstack([crop_a, crop_b, crop_d])
    cv2.putText(comp, 'A: Base 20261021', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(comp, 'B: dt -30ms', (crop_a.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(comp, 'D: Pitch -0.25 deg', (2*crop_a.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
    cv2.imwrite(str(OUT_DIR / 'parking_spots_zoom_authentic_grid.png'), comp)
    print('[SUCCESS] Authentic grid and zoom collage created successfully.')

if __name__ == '__main__':
    main()
