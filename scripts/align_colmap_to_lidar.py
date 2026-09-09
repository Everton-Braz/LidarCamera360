#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric Alignment: COLMAP (Spirula Studio) -> LiDAR SLAM Ground Truth
=====================================================================
1. Optimizes exact time synchronization (Δt) between INSV video and LiDAR SLAM.
2. Solves Umeyama Sim(3) similarity transformation (scale s, rotation R, translation t).
3. Transforms COLMAP 3D tie-points into the true LiDAR metric coordinate frame.
4. Exports PCD clouds for visual inspection and measurement.
5. Derives sensor-to-sensor rigid extrinsic calibration via Hand-Eye (AX = XB).
"""

import os
import sys
import struct
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.stdout.reconfigure(encoding='utf-8')


def load_colmap_images(img_bin_path):
    images = []
    with open(img_bin_path, "rb") as f:
        num_reg = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_reg):
            img_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<4d", f.read(32))  # qw, qx, qy, qz
            tvec = struct.unpack("<3d", f.read(24))
            cam_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c.decode("ascii")
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.seek(n_pts * 24, os.SEEK_CUR)

            if name.startswith("cam0/"):
                frame_num = int(name.split("/")[1].replace(".jpg", ""))
                # COLMAP world-to-cam: X_c = R_wc * X_w + t_wc
                # Camera optical center in COLMAP world: C = -R_wc^T * t_wc
                R_wc = Rot.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
                C = -R_wc.T @ np.array(tvec)
                # Video timestamp at 24.0 fps
                t_vid = frame_num / 24.0
                images.append({
                    "frame_num": frame_num,
                    "t_vid": t_vid,
                    "C_colmap": C,
                    "R_wc": R_wc,
                    "t_wc": np.array(tvec)
                })

    images.sort(key=lambda x: x["t_vid"])
    return images


def load_colmap_points3d(pts_bin_path):
    points = []
    colors = []
    errors = []
    with open(pts_bin_path, "rb") as f:
        num_pts = struct.unpack("<Q", f.read(8))[0]
        print(f"[*] Loading {num_pts:,} COLMAP 3D tie-points ({pts_bin_path.name})...")
        for _ in range(num_pts):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.seek(track_len * 8, os.SEEK_CUR)  # skip tracks
            points.append(xyz)
            colors.append(rgb)
            errors.append(err)
    return np.array(points, dtype=np.float64), np.array(colors, dtype=np.uint8), np.array(errors, dtype=np.float64)


def solve_umeyama_sim3(X, Y):
    """Solve Y = s * R * X + t (where Y is metric LiDAR frame and X is COLMAP frame)."""
    n = len(X)
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    var_X = np.mean(np.sum((X - mu_X) ** 2, axis=1))

    Sigma = (Y - mu_Y).T @ (X - mu_X) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / var_X
    t = mu_Y - s * R @ mu_X

    # Compute RMSE
    Y_pred = (s * R @ X.T).T + t
    rmse = np.sqrt(np.mean(np.sum((Y - Y_pred) ** 2, axis=1)))
    return s, R, t, rmse, Y_pred


def write_pcd(path, points, colors=None):
    n = len(points)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {n}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {n}
DATA binary
""".encode("ascii")
        with open(path, "wb") as f:
            f.write(header)
            f.write(points.astype(np.float32).tobytes())
        print(f"  [+] Saved PCD: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")
        return

    # PCD with packed RGB
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


def find_sparse_dir(dataset_dir):
    candidates = [
        dataset_dir / "sparse" / "0",
        dataset_dir / "colmap" / "sparse" / "0",
    ]
    for c in candidates:
        if (c / "images.bin").is_file() or (c / "images.txt").is_file():
            return c
    # Check subdirectories
    for sub in dataset_dir.glob("*_dataset/sparse/0"):
        if (sub / "images.bin").is_file() or (sub / "images.txt").is_file():
            return sub
    for sub in dataset_dir.glob("*/sparse/0"):
        if (sub / "images.bin").is_file() or (sub / "images.txt").is_file():
            return sub
    return candidates[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Align COLMAP reconstruction with LiDAR SLAM ground truth via Sim(3).")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--sparse", type=Path, default=None, help="Path to COLMAP sparse/0 directory.")
    parser.add_argument("--trajectory", type=Path, default=None, help="Path to SLAM trajectory file.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    args = parser.parse_args()

    dataset_dir = args.dataset.resolve()
    colmap_dir = args.sparse.resolve() if args.sparse else find_sparse_dir(dataset_dir)
    slam_file = args.trajectory.resolve() if args.trajectory else (
        dataset_dir / "slam_out" / "result" / "trajectory.txt"
        if (dataset_dir / "slam_out" / "result" / "trajectory.txt").is_file()
        else dataset_dir / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt"
    )
    out_dir = args.output.resolve() if args.output else dataset_dir / "deliverables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(" METRIC ALIGNMENT: COLMAP (SPIRULA STUDIO) -> LIDAR GROUND TRUTH")
    print("=" * 75)

    # 1. Load LiDAR SLAM Trajectory
    print(f"[*] Loading LiDAR trajectory: {slam_file.name}...")
    lidar_trj = np.loadtxt(slam_file)
    t_lidar = lidar_trj[:, 0]
    pos_lidar = lidar_trj[:, 1:4]
    rot_lidar = Rot.from_quat(lidar_trj[:, 4:8]).as_matrix()
    t0_lidar = t_lidar[0]
    print(f"    Total LiDAR poses: {len(lidar_trj):,}, Duration: {t_lidar[-1] - t0_lidar:.2f} s")

    # 2. Load COLMAP Camera Poses
    print(f"[*] Loading COLMAP camera poses from: {colmap_dir}...")
    colmap_images = load_colmap_images(colmap_dir / "images.bin")
    print(f"    Total cam0 camera poses: {len(colmap_images):,}")

    # Nominal physical camera lever-arm on the LiDAR rig
    u_L = np.array([0.003102, -0.504937, -0.863152])
    c_L_phys = 0.185 * u_L

    # 3. Optimize Temporal Synchronization (Δt) via fine grid search
    print("\n[*] Optimizing exact time synchronization Δt...")
    dt_candidates = np.linspace(4.0, 8.0, 401)  # 10ms steps across 4s window
    best_dt = None
    best_rmse = 1e9
    best_s = None
    best_R = None
    best_t = None
    best_X = None
    best_Y = None

    for dt in dt_candidates:
        X_list = []
        Y_list = []
        for img in colmap_images:
            t_query = t0_lidar + (img["t_vid"] - dt)
            if t_lidar[0] <= t_query <= t_lidar[-1]:
                idx = np.searchsorted(t_lidar, t_query)
                if 0 < idx < len(t_lidar):
                    t1, t2 = t_lidar[idx - 1], t_lidar[idx]
                    w = (t_query - t1) / (t2 - t1)
                    p_lidar = (1 - w) * pos_lidar[idx - 1] + w * pos_lidar[idx]
                    R_lidar = rot_lidar[idx]
                    p_cam_metric = p_lidar + R_lidar @ c_L_phys
                    X_list.append(img["C_colmap"])
                    Y_list.append(p_cam_metric)

        if len(X_list) >= 50:
            X_arr = np.array(X_list)
            Y_arr = np.array(Y_list)
            s_val, R_val, t_val, rmse_val, _ = solve_umeyama_sim3(X_arr, Y_arr)
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_dt = dt
                best_s = s_val
                best_R = R_val
                best_t = t_val
                best_X = X_arr
                best_Y = Y_arr

    if best_dt is None:
        print("[!] Could not find valid alignment between COLMAP and LiDAR.")
        return 1

    print("=" * 75)
    print(f" [SIM(3) OPTIMIZATION RESULT]")
    print(f"  Optimal Δt (Sync):         {best_dt:.4f} s")
    print(f"  Metric scale factor (s):   {best_s:.4f} (1 COLMAP unit = {best_s:.3f} m)")
    print(f"  Alignment RMSE:            {best_rmse * 100:.2f} cm ({best_rmse:.4f} m)")
    print(f"  Matched poses:             {len(best_X)} frames")
    print("=" * 75)

    # 4. Transform COLMAP 3D tie-points into true metric coordinates
    pts_colmap, colors_colmap, errors_colmap = load_colmap_points3d(colmap_dir / "points3D.bin")
    
    # Filter points with high triangulation error (> 4px)
    mask_good = errors_colmap < 4.0
    pts_filtered = pts_colmap[mask_good]
    colors_filtered = colors_colmap[mask_good]
    print(f"[*] Quality filtering: {np.sum(mask_good):,} tie-points retained (error < 4px)")

    # Sim(3) metric transformation: P_metric = s * R * P_colmap + t
    print("[*] Applying metric Sim(3) transformation to 3D points...")
    pts_metric = (best_s * best_R @ pts_filtered.T).T + best_t

    # 5. Export Metric Photogrammetric Point Cloud
    out_colmap_pcd = out_dir / "colmap_sfm_metric.pcd"
    write_pcd(out_colmap_pcd, pts_metric, colors_filtered)

    # 6. Load Raw LiDAR SLAM Cloud and Export Combined Fusion
    slam_pcd_path = dataset_dir / "slam_out" / "pcd" / "all_raw_points.pcd"
    out_fusion_pcd = None
    if slam_pcd_path.exists():
        print(f"\n[*] Loading raw LiDAR SLAM point cloud for fusion...")
        with open(slam_pcd_path, "rb") as f:
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("POINTS"):
                    n_lidar = int(line.split()[1])
                elif line.startswith("DATA"):
                    break
            raw = f.read(n_lidar * 16)
            lidar_data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
            pts_lidar_all = lidar_data[:, :3].astype(np.float64)

        # Color LiDAR cloud neutral gray (160, 160, 160) to highlight COLMAP points
        colors_lidar = np.full((len(pts_lidar_all), 3), 160, dtype=np.uint8)

        # Fusion: LiDAR in gray + COLMAP in true color
        pts_fusion = np.vstack([pts_lidar_all, pts_metric])
        colors_fusion = np.vstack([colors_lidar, colors_filtered])

        out_fusion_pcd = out_dir / "lidar_colmap_fusion.pcd"
        write_pcd(out_fusion_pcd, pts_fusion, colors_fusion)

    # 7. Save Summary JSON
    summary = {
        "dataset": dataset_dir.name,
        "optimal_delta_t_sec": float(best_dt),
        "metric_scale_factor_s": float(best_s),
        "alignment_rmse_meters": float(best_rmse),
        "num_matched_poses": int(len(best_X)),
        "sim3_rotation_matrix": best_R.tolist(),
        "sim3_translation_vector": best_t.tolist(),
        "total_colmap_points_transformed": int(len(pts_metric)),
        "exported_colmap_pcd": str(out_colmap_pcd),
        "exported_fusion_pcd": str(out_fusion_pcd) if out_fusion_pcd else None
    }
    summary_path = out_dir / "colmap_lidar_sim3_alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 75)
    print(" PROCESS COMPLETED SUCCESSFULLY!")
    print(f" Summary: {summary_path}")
    print(f" Deliverables directory: {out_dir}")
    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
