#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct Rigid Colorization (without SfM) using Calibrated Rig Matrix and Gyro Sync
=====================================================================================
Applies the calibrated sensor rig profile directly over the LiDAR SLAM trajectory
and high-resolution 360 dual-fisheye imagery with Z-buffer occlusion testing and
sharpness weighting.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_pcd(path):
    print(f"[*] Loading LiDAR point cloud: {path.name}...")
    with open(path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("POINTS"):
                n = int(line.split()[1])
            elif line.startswith("DATA"):
                break
        raw = f.read(n * 16)
        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        xyz = data[:, :3].astype(np.float64)
    print(f"    Total points loaded: {len(xyz):,}")
    return xyz


def load_trajectory(path):
    print(f"[*] Loading SLAM trajectory: {path.name}...")
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    rotations = Rot.from_quat(data[:, 4:8])
    print(f"    Total poses: {len(data):,}, Duration: {timestamps[-1] - timestamps[0]:.2f}s")
    return timestamps, positions, rotations


def project_thin_prism(P_v, params):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    x = P_v[:, 0]
    y = P_v[:, 1]
    z = P_v[:, 2]
    r = np.sqrt(x**2 + y**2)
    r = np.maximum(r, 1e-7)
    theta = np.arctan2(r, z)
    theta_d = theta * (1.0 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)
    cos_phi = x / r
    sin_phi = y / r
    u_rad = theta_d * cos_phi
    v_rad = theta_d * sin_phi
    r2 = u_rad**2 + v_rad**2
    du_tan = 2.0 * p1 * u_rad * v_rad + p2 * (r2 + 2.0 * u_rad**2)
    dv_tan = p1 * (r2 + 2.0 * v_rad**2) + 2.0 * p2 * u_rad * v_rad
    du_tp = sx1 * r2
    dv_tp = sy1 * r2
    u_dist = u_rad + du_tan + du_tp
    v_dist = v_rad + dv_tan + dv_tp
    u_px = fx * u_dist + cx
    v_px = fy * v_dist + cy
    return u_px, v_px, r


def write_ply(path, points, colors):
    n = len(points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".encode("ascii")
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["r"] = colors[:, 0]
    arr["g"] = colors[:, 1]
    arr["b"] = colors[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Saved PLY: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


def write_pcd(path, points, colors):
    n = len(points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z rgb
SIZE 4 4 4 4
TYPE F F F U
COUNT 1 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
""".encode("ascii")
    r = colors[:, 0].astype(np.uint32)
    g = colors[:, 1].astype(np.uint32)
    b = colors[:, 2].astype(np.uint32)
    rgb_packed = (r << 16) | (g << 8) | b
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["rgb"] = rgb_packed
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Saved PCD: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")


def find_calibration_file(dataset_dir, explicit=None):
    if explicit and Path(explicit).is_file():
        return Path(explicit).resolve()
    for name in ("rig_profile.json", "calibration.json", "calibracao_rigida_auto.json", "calibracao_rigida_raven_insta360.json"):
        p = dataset_dir / name
        if p.is_file():
            return p
    default_config = PROJECT_ROOT / "configs" / "rig_profile.json"
    if default_config.is_file():
        return default_config
    fallback = PROJECT_ROOT / "calibracao_rigida_raven_insta360.json"
    if fallback.is_file():
        return fallback
    return default_config


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Colorize a LiDAR point cloud with the calibrated rigid camera rig.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--calib", type=Path, default=None, help="Optional calibration JSON path.")
    parser.add_argument("--dt", type=float, default=None, help="Time offset Δt in seconds (video vs SLAM).")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame rate of extracted images.")
    parser.add_argument("--output", type=Path, default=None, help="Output PLY/PCD prefix or directory.")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    calib_json = find_calibration_file(dataset_dir, args.calib)
    slam_pcd = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    slam_trj = (
        dataset_dir / "slam_out" / "result" / "trajectory.txt"
        if (dataset_dir / "slam_out" / "result" / "trajectory.txt").is_file()
        else dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
    )
    img_dir = dataset_dir / "images"

    out_dir = dataset_dir / "deliverables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ply = out_dir / "trajectory_colorized.ply"
    out_pcd = out_dir / "trajectory_colorized.pcd"

    print("=" * 80)
    print(" DIRECT RIGID COLORIZATION (WITHOUT SFM)")
    print(f" Dataset:     {dataset_dir}")
    print(f" Calibration: {calib_json.name}")
    print("=" * 80)
    t_start = time.time()

    # 1. Load Calibration
    print(f"[*] Loading rigid calibration file: {calib_json.name}...")
    with open(calib_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    p0 = cfg["cam0_front_intrinsics"]
    params0 = (p0["fx"], p0["fy"], p0["cx"], p0["cy"], p0["k1"], p0["k2"], p0["p1"], p0["p2"], p0["k3"], p0["k4"], p0["sx1"], p0["sy1"])

    p1 = cfg["cam1_rear_intrinsics"]
    params1 = (p1["fx"], p1["fy"], p1["cx"], p1["cy"], p1["k1"], p1["k2"], p1["p1"], p1["p2"], p1["k3"], p1["k4"], p1["sx1"], p1["sy1"])

    T_LC0 = np.array(cfg["T_lidar_to_cam0_rigid_4x4"])
    R_LC0 = T_LC0[:3, :3]
    t_LC0 = T_LC0[:3, 3]

    T_LC1 = np.array(cfg["T_lidar_to_cam1_rigid_4x4"])
    R_LC1 = T_LC1[:3, :3]
    t_LC1 = T_LC1[:3, 3]

    # 2. Determine Time Sync (Δt)
    dt_sync = args.dt
    if dt_sync is None:
        align_json = dataset_dir / "deliverables" / "colmap_lidar_sim3_alignment_summary.json"
        if align_json.is_file():
            try:
                with open(align_json, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)
                    dt_sync = summary_data.get("optimal_delta_t_sec", None)
                    if dt_sync is not None:
                        print(f"[*] Inherited Δt from Sim(3) summary: {dt_sync:.4f}s")
            except Exception:
                pass
    if dt_sync is None:
        dt_sync = -4.6290
        print(f"[*] Using nominal default Δt: {dt_sync:.4f}s")

    # 3. Load LiDAR Point Cloud and Trajectory
    pts_lidar = load_pcd(slam_pcd)
    n_pts = len(pts_lidar)

    t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
    t_slam_start, t_slam_end = t_slam[0], t_slam[-1]
    slerp = Slerp(t_slam, rot_slam)

    # 4. List Frames
    cam0_files = sorted(list((img_dir / "cam0").glob("*.jpg")))
    cam1_files = sorted(list((img_dir / "cam1").glob("*.jpg")))
    print(f"[*] Image frames found: cam0={len(cam0_files)}, cam1={len(cam1_files)}")

    fps_extracted = args.fps
    colors = np.full((n_pts, 3), 180, dtype=np.uint8)
    best_scores = np.zeros(n_pts, dtype=np.float32)

    zbuf_w, zbuf_h = 960, 960
    scale_factor_zbuf = 3840.0 / zbuf_w

    print("\n" + "=" * 80)
    print(" STARTING MULTI-FRAME DIRECT RIGID COLORIZATION...")
    print(f" Time Synchronization: dt = {dt_sync:.4f} s")
    print("=" * 80)

    t_loop_start = time.time()
    frames_processed = 0
    num_frames = min(len(cam0_files), len(cam1_files))

    for k in range(num_frames):
        t_vid = k / fps_extracted
        t_query = t_slam_start + (t_vid - dt_sync)

        if not (t_slam_start <= t_query <= t_slam_end):
            continue

        idx = np.searchsorted(t_slam, t_query)
        idx = np.clip(idx, 1, len(t_slam) - 1)
        w = (t_query - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L = slerp(t_query).as_matrix()

        lens_configs = [
            ("cam0", cam0_files[k], params0, R_LC0, t_LC0),
            ("cam1", cam1_files[k], params1, R_LC1, t_LC1)
        ]

        for cam_name, img_path, params, R_LC, t_LC in lens_configs:
            p_cam = p_L + R_L @ t_LC
            R_world_cam = R_L @ R_LC
            R_cw = R_world_cam.T

            P_cam = (R_cw @ (pts_lidar - p_cam).T).T
            z_mask = P_cam[:, 2] > 0.15
            idx_valid = np.where(z_mask)[0]
            if len(idx_valid) == 0:
                continue

            P_v = P_cam[idx_valid]
            d = np.linalg.norm(P_v, axis=1)

            cx, cy = params[2], params[3]
            u, v, r = project_thin_prism(P_v, params)

            r_px = np.hypot(u - cx, v - cy)
            mask_circle = (r_px < 1650.0) & (u >= 0.0) & (u < 3839.0) & (v >= 0.0) & (v < 3839.0)
            if not np.any(mask_circle):
                continue

            u_c = u[mask_circle]
            v_c = v[mask_circle]
            d_c = d[mask_circle]
            r_c = r_px[mask_circle]
            idx_c = idx_valid[mask_circle]

            ug = np.clip(np.floor(u_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_w - 1)
            vg = np.clip(np.floor(v_c / scale_factor_zbuf).astype(np.int32), 0, zbuf_h - 1)
            flat_idx = vg * zbuf_w + ug

            dmap = np.full(zbuf_w * zbuf_h, 1e9, dtype=np.float32)
            np.minimum.at(dmap, flat_idx, d_c.astype(np.float32))

            vis_mask = d_c <= (dmap[flat_idx] * 1.08 + 0.15)
            if not np.any(vis_mask):
                continue

            u_vis = np.round(u_c[vis_mask]).astype(int)
            v_vis = np.round(v_c[vis_mask]).astype(int)
            d_vis = d_c[vis_mask]
            r_vis = r_c[vis_mask]
            idx_vis = idx_c[vis_mask]

            scores = (1.0 - (r_vis / 1650.0)) / np.maximum(d_vis, 0.5)

            better_mask = scores > best_scores[idx_vis]
            if not np.any(better_mask):
                continue

            idx_update = idx_vis[better_mask]
            u_update = u_vis[better_mask]
            v_update = v_vis[better_mask]
            scores_update = scores[better_mask]

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            colors[idx_update] = img_rgb[v_update, u_update]
            best_scores[idx_update] = scores_update

        frames_processed += 1
        if (k + 1) % 15 == 0 or k == num_frames - 1:
            n_colored = np.sum(best_scores > 0)
            pct = (n_colored / n_pts) * 100.0
            elapsed = time.time() - t_loop_start
            fps_act = frames_processed / max(elapsed, 0.001)
            print(f"  Frame [{k+1:3d}/{num_frames:3d}] (t={t_vid:4.1f}s) | "
                  f"Colorized: {n_colored:,}/{n_pts:,} ({pct:.1f}%) | {fps_act:.1f} fps")

    n_colored_final = np.sum(best_scores > 0)
    pct_final = (n_colored_final / n_pts) * 100.0
    print("\n" + "=" * 80)
    print(" DIRECT RIGID COLORIZATION RESULT:")
    print(f"  Total LiDAR points:        {n_pts:,}")
    print(f"  Colorized points:          {n_colored_final:,} ({pct_final:.2f}%)")
    print(f"  Processed frames:          {frames_processed} synchronized frame pairs")
    print(f"  Execution time:            {time.time() - t_loop_start:.1f} seconds")
    print("=" * 80)

    print("\n[*] Exporting deliverables...")
    write_ply(out_ply, pts_lidar, colors)
    write_pcd(out_pcd, pts_lidar, colors)

    summary = {
        "dataset": str(dataset_dir),
        "method": "direct_rigid",
        "dt_sync_sec": float(dt_sync),
        "total_lidar_points": int(n_pts),
        "colorized_points": int(n_colored_final),
        "coverage_rate_percent": float(pct_final),
        "execution_time_sec": float(time.time() - t_start),
        "files": {
            "ply": str(out_ply),
            "pcd": str(out_pcd)
        }
    }
    with open(out_dir / "direct_colorization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(" DIRECT COLORIZATION FINISHED SUCCESSFULLY!")
    print(f" Output folder: {out_dir}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
