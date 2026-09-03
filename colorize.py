#!/usr/bin/env python3
"""
LiDAR Point Cloud Colorizer using Calibrated Extrinsics
======================================================
Projects 3D LiDAR point clouds onto Insta360 / 360° panoramic video frames
using the extrinsic calibration matrix (T_base_camera_lidar).

Modes:
------
1. SLAM Trajectory Colorization (Multi-Frame 360°):
   Colorizes a globally registered SLAM point cloud (e.g. from RayStudio, FAST-LIVO2)
   using the full 6-DoF trajectory (trj.txt) and 360° video.
   
   python colorize.py --slam-ply dataset-test/pointcloud_raven_raystudio.ply \\
                      --trj dataset-test/shading/trj.txt \\
                      --video dataset-test/VID_20260828_094637_00_270.mp4 \\
                      --calib dataset-test/extrinsics_determined.json \\
                      --out dataset-test/factory_360_colorized.ply

2. Single-Scan Bag Colorization:
   Colorizes raw static LiDAR scans directly from a ROS bag.

   python colorize.py --bag dataset-test/LIDAR_20260828094048Fabrica2.bag \\
                      --video dataset-test/VID_20260828_094637_00_270.mp4 \\
                      --calib dataset-test/extrinsics_determined.json \\
                      --out dataset-test/colorized_cloud.ply
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import core


def write_ply(path, points, colors_rgb, intensities=None):
    """Write little-endian binary PLY point cloud with XYZ, RGB, and optional intensity."""
    n = len(points)
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if intensities is not None:
        header.append("property float intensity")
    header.append("end_header\n")
    header_bytes = "\n".join(header).encode("ascii")

    dt = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if intensities is not None:
        dt.append(("intensity", "<f4"))

    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["red"] = colors_rgb[:, 0].astype(np.uint8)
    arr["green"] = colors_rgb[:, 1].astype(np.uint8)
    arr["blue"] = colors_rgb[:, 2].astype(np.uint8)
    if intensities is not None:
        arr["intensity"] = intensities.astype(np.float32)

    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(arr.tobytes())


def write_pcd(path, points, colors_rgb, intensities=None):
    """Write binary PCD point cloud with XYZ, RGB, and optional intensity."""
    n = len(points)
    rgb_packed = (
        (colors_rgb[:, 0].astype(np.uint32) << 16)
        | (colors_rgb[:, 1].astype(np.uint32) << 8)
        | (colors_rgb[:, 2].astype(np.uint32))
    ).view(np.float32)

    fields = "x y z rgb"
    sizes = "4 4 4 4"
    types = "F F F F"
    counts = "1 1 1 1"

    if intensities is not None:
        fields += " intensity"
        sizes += " 4"
        types += " F"
        counts += " 1"

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {sizes}\n"
        f"TYPE {types}\n"
        f"COUNT {counts}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")

    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
    if intensities is not None:
        dt.append(("intensity", "<f4"))

    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["rgb"] = rgb_packed
    if intensities is not None:
        arr["intensity"] = intensities.astype(np.float32)

    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def read_ply_xyz(path):
    """Read XYZ vertices from PLY file."""
    with open(path, "rb") as f:
        num_verts = 0
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                num_verts = int(line.split()[-1])
            if line == "end_header":
                break

        if num_verts == 0:
            raise ValueError(f"Could not find vertex count in {path}")

        # Check vertex structure
        dt = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1")
        ])
        data = np.fromfile(f, dtype=dt, count=num_verts)
        pts = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float64)
        return pts


def colorize_slam_cloud(
    slam_ply_path,
    trj_path,
    video_path,
    calib_json_path,
    out_path,
    keyframes=35,
    min_dist=0.3,
    max_dist=25.0,
    log=print,
):
    """
    Colorize a registered global SLAM map using 6-DoF trajectory and 360° video.
    """
    log(f"[1/4] Loading extrinsic calibration: {calib_json_path}")
    with open(calib_json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    if "T_base_camera_lidar" in calib:
        T_c_l = np.array(calib["T_base_camera_lidar"], dtype=np.float64)
    elif "T_lidar_camera" in calib:
        T_c_l = np.linalg.inv(np.array(calib["T_lidar_camera"], dtype=np.float64))
    else:
        raise ValueError("JSON file must contain 'T_base_camera_lidar' or 'T_lidar_camera'.")

    R_c_l = T_c_l[:3, :3]
    t_c_l = T_c_l[:3, 3]

    log(f"[2/4] Loading SLAM pointcloud and trajectory...")
    pts = read_ply_xyz(slam_ply_path)
    log(f"      Loaded {len(pts):,} SLAM points from: {slam_ply_path}")

    trj = np.loadtxt(trj_path)
    t_stamps = trj[:, 0]
    t_xyz = trj[:, 1:4]
    t_quat = trj[:, 4:8]
    t_rot = Rot.from_quat(t_quat).as_matrix()

    t_start, t_end = t_stamps[0], t_stamps[-1]
    t_dur = t_end - t_start
    log(f"      Loaded {len(trj):,} poses ({t_dur:.1f}s trajectory) from: {trj_path}")

    log(f"[3/4] Processing 360 video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    v_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    v_dur = v_frames / v_fps
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log(f"      Video: {W}x{H} @ {v_fps:.0f}fps ({v_dur:.1f}s)")

    # Sample keyframes evenly along trajectory
    key_times = np.linspace(0.02, 0.98, keyframes)
    best_dist = np.full(len(pts), 1e9, dtype=np.float32)
    color_accum = np.zeros((len(pts), 3), dtype=np.uint8)

    for ki, rel_t in enumerate(key_times):
        sim_t = t_start + rel_t * t_dur
        pose_idx = np.argmin(np.abs(t_stamps - sim_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(rel_t * (v_frames - 1)))
        ret, frame = cap.read()
        if not ret:
            continue

        # Transform world points into LiDAR frame at time t:
        # P_l = (P_w - t_w_l) @ R_w_l
        P_l = (pts - t_w_l) @ R_w_l

        # Transform LiDAR points into Camera frame:
        # P_c = P_l @ R_c_l.T + t_c_l
        P_c = P_l @ R_c_l.T + t_c_l

        dist = np.linalg.norm(P_c, axis=1)

        # Equirectangular projection
        u, v = core.sph_uv(P_c, W, H)
        u_idx = np.mod(np.round(u).astype(np.int64), W)
        v_idx = np.clip(np.round(v).astype(np.int64), 0, H - 1)

        vis = (dist >= min_dist) & (dist <= max_dist) & (dist < best_dist)
        if np.any(vis):
            bgr = frame[v_idx[vis], u_idx[vis]]
            color_accum[vis] = bgr[:, ::-1]
            best_dist[vis] = dist[vis]

        pct = (np.sum(best_dist < 1e8) / len(pts)) * 100
        log(f"      Frame {ki+1:2d}/{keyframes} ({rel_t*v_dur:4.1f}s): {pct:5.1f}% points colorized")

    cap.release()

    log(f"[4/4] Writing output: {out_path}")
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".pcd":
        write_pcd(out_path, pts, color_accum)
    else:
        if ext != ".ply":
            out_path += ".ply"
        write_ply(out_path, pts, color_accum)

    file_size_mb = os.path.getsize(out_path) / 1e6
    log(f"      Saved successfully ({file_size_mb:.2f} MB)")
    return out_path


def colorize_bag_cloud(
    bag_path,
    video_path,
    calib_json_path,
    out_path,
    frame_pos=0.5,
    timestamp_s=None,
    min_range=0.3,
    max_range=100.0,
    log=print,
):
    """
    Project LiDAR points from bag onto video frame and save colorized point cloud.
    """
    log(f"[1/4] Loading extrinsic calibration: {calib_json_path}")
    with open(calib_json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)

    if "T_base_camera_lidar" in calib:
        T = np.array(calib["T_base_camera_lidar"], dtype=np.float64)
    elif "T_lidar_camera" in calib:
        T = np.linalg.inv(np.array(calib["T_lidar_camera"], dtype=np.float64))
    else:
        raise ValueError("JSON file must contain 'T_base_camera_lidar' or 'T_lidar_camera'.")

    R = T[:3, :3]
    t = T[:3, 3]

    log(f"[2/4] Loading LiDAR bag: {bag_path}")
    bag_data = core.read_bag(bag_path)
    pts = bag_data["points"][:, :3].astype(np.float64)
    refl = (
        bag_data["points"][:, 3].astype(np.float32)
        if bag_data["points"].shape[1] > 3
        else None
    )

    log(f"      Loaded {len(pts):,} points (LiDAR duration: {bag_data.get('duration_s', 0):.1f}s)")

    log(f"[3/4] Grabbing frame from 360 video: {video_path}")
    if timestamp_s is not None:
        dur = bag_data.get("duration_s", 1.0)
        rel_pos = float(np.clip(timestamp_s / max(dur, 1e-3), 0.01, 0.99))
    else:
        rel_pos = float(np.clip(frame_pos, 0.01, 0.99))

    img = core.grab_frame(video_path, rel_pos)
    H, W, _ = img.shape
    log(f"      Frame size: {W}x{H} (at {rel_pos*100:.1f}% of video)")

    X_cam = pts @ R.T + t
    u, v = core.sph_uv(X_cam, W, H)

    u_idx = np.mod(np.round(u).astype(np.int64), W)
    v_idx = np.clip(np.round(v).astype(np.int64), 0, H - 1)

    colors_bgr = img[v_idx, u_idx]
    colors_rgb = colors_bgr[:, ::-1]

    dist = np.linalg.norm(pts, axis=1)
    valid_mask = (dist >= min_range) & (dist <= max_range) & np.isfinite(dist)

    pts_valid = pts[valid_mask]
    rgb_valid = colors_rgb[valid_mask]
    refl_valid = refl[valid_mask] if refl is not None else None

    log(f"      Colorized {len(pts_valid):,} valid points within [{min_range}m .. {max_range}m]")

    log(f"[4/4] Writing output: {out_path}")
    ext = os.path.splitext(out_path)[1].lower()

    if ext == ".ply":
        write_ply(out_path, pts_valid, rgb_valid, refl_valid)
    elif ext == ".pcd":
        write_pcd(out_path, pts_valid, rgb_valid, refl_valid)
    else:
        out_path += ".ply"
        write_ply(out_path, pts_valid, rgb_valid, refl_valid)

    file_size_mb = os.path.getsize(out_path) / 1e6
    log(f"      Saved successfully ({file_size_mb:.2f} MB)")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Colorize LiDAR point clouds using 360 camera video and calibrated extrinsics."
    )
    # SLAM trajectory mode
    parser.add_argument("--slam-ply", help="Path to global SLAM point cloud PLY (e.g. from RayStudio)")
    parser.add_argument("--trj", help="Path to 6-DoF trajectory file (trj.txt)")
    parser.add_argument("--keyframes", type=int, default=35, help="Number of 360 video keyframes to project (default: 35)")

    # Raw Bag mode
    parser.add_argument("--bag", help="Path to ROS .bag file")
    parser.add_argument("--frame-pos", type=float, default=0.5, help="Relative position in video for static bag (0.0 to 1.0, default: 0.5)")
    parser.add_argument("--time", type=float, default=None, help="Timestamp in seconds to sample frame")

    # Common args
    parser.add_argument("--video", required=True, help="Path to 360 equirectangular .mp4 video")
    parser.add_argument("--calib", required=True, help="Path to calibration extrinsics JSON file")
    parser.add_argument("--out", default="colorized_cloud.ply", help="Path for output point cloud (.ply, .pcd)")
    parser.add_argument("--min-range", type=float, default=0.3, help="Minimum distance filter in meters (default: 0.3)")
    parser.add_argument("--max-range", type=float, default=30.0, help="Maximum distance filter in meters (default: 30.0)")

    args = parser.parse_args()

    try:
        if args.slam_ply and args.trj:
            colorize_slam_cloud(
                slam_ply_path=args.slam_ply,
                trj_path=args.trj,
                video_path=args.video,
                calib_json_path=args.calib,
                out_path=args.out,
                keyframes=args.keyframes,
                min_dist=args.min_range,
                max_dist=args.max_range,
                log=print,
            )
        elif args.bag:
            colorize_bag_cloud(
                bag_path=args.bag,
                video_path=args.video,
                calib_json_path=args.calib,
                out_path=args.out,
                frame_pos=args.frame_pos,
                timestamp_s=args.time,
                min_range=args.min_range,
                max_range=args.max_range,
                log=print,
            )
        else:
            print("Error: Specify either (--slam-ply AND --trj) for SLAM colorization, or --bag for raw scan colorization.", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
