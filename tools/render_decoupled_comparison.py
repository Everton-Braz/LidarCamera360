#!/usr/bin/env python3
"""
Render Close-up Comparison of the Decoupled Rig vs Best Previous Candidate
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
    trj_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    pts_all = np.asarray(pcd.points, dtype=np.float32)

    # Exactly matching the user's marked area: center (2.5, -20.0), span 10m
    cx, cy = 2.5, -20.0
    hw = 5.0
    mask = (
        (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) &
        (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) &
        (pts_all[:, 2] >= -2.0) & (pts_all[:, 2] <= 3.8)
    )
    pts = pts_all[mask]
    print(f'[+] Points in 10m window: {len(pts):,}')

    trj = np.loadtxt(trj_path)
    t_slam = trj[:, 0]
    pos_slam = trj[:, 1:4]
    quats = trj[:, 4:8]
    slerp = Slerp(t_slam, R.from_quat(quats))
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]

    # Load decoupled calib
    with open(DATASET_DIR / 'decoupled_rig_calibration.json') as f:
        decoupled_calib = json.load(f)
    T_dec_cam0 = np.array(decoupled_calib['T_lidar_to_cam0_rigid_4x4'])
    T_dec_cam1 = np.array(decoupled_calib['T_lidar_to_cam1_rigid_4x4'])
    dt_dec = decoupled_calib['dt_sync_refined']

    # Load nominal calib (from candidate 211521)
    with open('calibracao_rigida_raven_insta360.json') as f:
        nom_calib = json.load(f)
    T_nom_cam0 = np.array(nom_calib['T_lidar_to_cam0_rigid_4x4'])
    T_nom_cam1 = np.array(nom_calib['T_lidar_to_cam1_rigid_4x4'])
    dt_nom = 3.336827

    p0 = nom_calib['cam0_front_intrinsics']
    params0 = (p0['fx'], p0['fy'], p0['cx'], p0['cy'], p0['k1'], p0['k2'], p0['p1'], p0['p2'], p0['k3'], p0['k4'], p0['sx1'], p0['sy1'])
    p1 = nom_calib['cam1_rear_intrinsics']
    params1 = (p1['fx'], p1['fy'], p1['cx'], p1['cy'], p1['k1'], p1['k2'], p1['p1'], p1['p2'], p1['k3'], p1['k4'], p1['sx1'], p1['sy1'])

    cam0_files = sorted(list((DATASET_DIR / 'images/cam0').glob('*.jpg')))
    with open(DATASET_DIR / 'images/frames.json') as f:
        fj = json.load(f)
    video_times = {p.name: float(fj['timestamps'].get(f'cam0/{p.name}', 0.0)) for p in cam0_files}

    # Select frames in radius 7m
    relevant_frames = []
    for p in cam0_files:
        t_vid = video_times[p.name]
        t_q = t_slam[0] + (t_vid - dt_nom)
        if t_slam_start <= t_q <= t_slam_end:
            idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
            w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
            p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
            if np.hypot(p_L[0] - cx, p_L[1] - cy) <= 7.0:
                relevant_frames.append((p.name, t_vid))

    img_cache0 = {fn: cv2.imread(str(DATASET_DIR / f'images/cam0/{fn}')) for fn, _ in relevant_frames}
    img_cache1 = {fn: cv2.imread(str(DATASET_DIR / f'images/cam1/{fn}')) for fn, _ in relevant_frames}

    comparisons = [
        ('Candidato Anterior (Direct Rigid Nominal, dt=3.337s)', dt_nom, T_nom_cam0, T_nom_cam1),
        ('Novo Candidato (Decoupled Boresight + dt refinado, dt=3.344s)', dt_dec, T_dec_cam0, T_dec_cam1)
    ]

    rendered = []
    for title, dt_eval, T0, T1 in comparisons:
        print(f'[*] Rendering {title}...')
        R_LC0, t_LC0 = T0[:3, :3], T0[:3, 3]
        R_LC1, t_LC1 = T1[:3, :3], T1[:3, 3]
        best_dist = np.full(len(pts), 999.0, dtype=np.float32)
        colors = np.zeros((len(pts), 3), dtype=np.uint8)

        for fn, t_vid in relevant_frames:
            t_q = t_slam[0] + (t_vid - dt_eval)
            if not (t_slam_start <= t_q <= t_slam_end):
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
                u0, v0, _ = project_thin_prism(pts_c0[m0], params0)
                val0 = (u0 >= 0) & (u0 < 2880) & (v0 >= 0) & (v0 < 2880)
                if np.any(val0):
                    sub = np.where(m0)[0][val0]
                    dists = pts_c0[m0, 2][val0]
                    closer = dists < best_dist[sub]
                    if np.any(closer):
                        tk = sub[closer]
                        colors[tk] = img_cache0[fn][np.clip(v0[val0][closer].round().astype(int), 0, 2879), np.clip(u0[val0][closer].round().astype(int), 0, 2879)][:, ::-1]
                        best_dist[tk] = dists[closer]

            # Cam1
            pts_c1 = (R_LC1 @ pts_body.T).T + t_LC1
            m1 = pts_c1[:, 2] > 0.3
            if np.any(m1):
                u1, v1, _ = project_thin_prism(pts_c1[m1], params1)
                val1 = (u1 >= 0) & (u1 < 2880) & (v1 >= 0) & (v1 < 2880)
                if np.any(val1):
                    sub = np.where(m1)[0][val1]
                    dists = pts_c1[m1, 2][val1]
                    closer = dists < best_dist[sub]
                    if np.any(closer):
                        tk = sub[closer]
                        colors[tk] = img_cache1[fn][np.clip(v1[val1][closer].round().astype(int), 0, 2879), np.clip(u1[val1][closer].round().astype(int), 0, 2879)][:, ::-1]
                        best_dist[tk] = dists[closer]

        rendered.append((title, colors))

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), dpi=250, facecolor='#111115')
    for ax, (title, cols) in zip(axes, rendered):
        ax.set_facecolor('#16161c')
        val = np.any(cols > 0, axis=1)
        ax.scatter(pts[val, 0], pts[val, 1], c=cols[val]/255.0, s=3.2, rasterized=True)
        ax.set_title(title, color='white', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.tick_params(colors='white')
        ax.axis('equal')
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)
        ax.grid(True, color='#2c2c36', linestyle='--', alpha=0.5)

    out_img = OUT_DIR / 'decoupled_vs_previous_comparison.png'
    plt.tight_layout()
    plt.savefig(str(out_img), dpi=250, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f'[SUCCESS] Rendered comparison saved to: {out_img}')

    # Also save single high-res close-up of the new candidate
    fig_s, ax_s = plt.subplots(figsize=(12, 12), dpi=280, facecolor='#111115')
    ax_s.set_facecolor('#16161c')
    val = np.any(rendered[1][1] > 0, axis=1)
    ax_s.scatter(pts[val, 0], pts[val, 1], c=rendered[1][1][val]/255.0, s=3.8, rasterized=True)
    ax_s.set_title('Novo Candidato: Calibração Desacoplada (Argon Tier 1+2)', color='white', fontsize=14, fontweight='bold', pad=12)
    ax_s.set_xlabel('X (m)', color='white')
    ax_s.set_ylabel('Y (m)', color='white')
    ax_s.tick_params(colors='white')
    ax_s.axis('equal')
    ax_s.set_xlim(cx - hw, cx + hw)
    ax_s.set_ylim(cy - hw, cy + hw)
    ax_s.grid(True, color='#2c2c36', linestyle='--', alpha=0.5)
    single_out = OUT_DIR / 'new_decoupled_candidate_detail.png'
    plt.tight_layout()
    plt.savefig(str(single_out), dpi=280, facecolor=fig_s.get_facecolor(), edgecolor='none')
    plt.close()
    print(f'[SUCCESS] Single high-res detail saved to: {single_out}')

if __name__ == '__main__':
    main()
