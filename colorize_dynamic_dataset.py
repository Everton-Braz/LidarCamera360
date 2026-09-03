#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Frame Dynamic Dual-Fisheye LiDAR Point Cloud Colorizer
============================================================
Projects registered 3D SLAM point clouds from FAST-LIVO2 onto dual-fisheye
Insta360 X4 video streams with exact Left (Stream 0) and Right (Stream 1)
vertical gravity-aligned frames.
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"


def load_pcd_xyz(pcd_path):
    print(f"[*] Reading PCD: {pcd_path}")
    with open(pcd_path, "rb") as f:
        num_points = 0
        fields = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("FIELDS"):
                fields = line.split()[1:]
            elif line.startswith("POINTS"):
                num_points = int(line.split()[1])
            elif line.startswith("DATA"):
                break

        print(f"    Fields: {fields}, Points: {num_points:,}")
        raw = f.read(num_points * 16)
        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        xyz = data[:, :3].astype(np.float64)
        return xyz


def write_ply(path, points, colors_rgb):
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
        "end_header\n",
    ]
    header_bytes = "\n".join(header).encode("ascii")

    dt = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["red"] = colors_rgb[:, 0].astype(np.uint8)
    arr["green"] = colors_rgb[:, 1].astype(np.uint8)
    arr["blue"] = colors_rgb[:, 2].astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(arr.tobytes())
    print(f"  [+] Saved PLY: {path} ({os.path.getsize(path)/1e6:.2f} MB)")


def write_pcd(path, points, colors_rgb):
    n = len(points)
    rgb_packed = (
        (colors_rgb[:, 0].astype(np.uint32) << 16)
        | (colors_rgb[:, 1].astype(np.uint32) << 8)
        | (colors_rgb[:, 2].astype(np.uint32))
    ).view(np.float32)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")

    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
    arr = np.empty(n, dtype=dt)
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["rgb"] = rgb_packed

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Saved PCD: {path} ({os.path.getsize(path)/1e6:.2f} MB)")


class FisheyeProjector:
    def __init__(self, width=3840, height=3840, fov_deg=196.0):
        self.W = width
        self.H = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        fov_rad = np.radians(fov_deg)
        self.f = (width / 2.0) / (fov_rad / 2.0)
        self.max_radius = (min(width, height) / 2.0) * 0.985

    def project(self, P_cam):
        X = P_cam[:, 0]
        Y = P_cam[:, 1]
        Z = P_cam[:, 2]

        dist = np.linalg.norm(P_cam, axis=1)
        r_xy = np.hypot(X, Y)

        # Fisheye equidistant projection
        theta = np.arctan2(r_xy, Z)
        r_img = self.f * theta

        valid = (Z > 0.05) & (dist > 0.25) & (dist < 30.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi

        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid, dist, r_img


def extract_stream_frame(video_path, stream_idx, time_sec):
    cmd = [
        FFMPEG_PATH,
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        video_path,
        "-map",
        f"0:v:{stream_idx}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_img = pipe.stdout.read(3840 * 3840 * 3)
    pipe.kill()
    if len(raw_img) == 3840 * 3840 * 3:
        img = np.frombuffer(raw_img, dtype=np.uint8).reshape((3840, 3840, 3))
        return img
    return None


def run_dynamic_colorization(
    slam_pcd_path,
    trj_path,
    video_path,
    out_ply_path,
    num_keyframes=50,
):
    print("=" * 70)
    print(" DYNAMIC LIDAR POINT CLOUD COLORIZATION WITH CALIBRATED EXTRINSICS")
    print("=" * 70)

    # 1. Setup Upright Gravity-Aligned Orthogonal Frames for Left (Stream 0) & Right (Stream 1)
    print(f"[1/4] Configuring Upright Left (Stream 0) and Right (Stream 1) extrinsic matrices...")

    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)  # Down vector
    u_L = -d_L  # Up vector in LiDAR

    # Left Lens (Stream 0): Z_cam points Left (-X_L), Y_cam points UP (u_L)
    Z_left = np.array([-1.0, 0.0, 0.0])
    Y_left = u_L - np.dot(u_L, Z_left) * Z_left
    Y_left = Y_left / np.linalg.norm(Y_left)
    X_left = np.cross(Y_left, Z_left)
    R_left = np.stack([X_left, Y_left, Z_left], axis=0)
    t_left = np.array([0.0, -0.185, 0.0])

    # Right Lens (Stream 1): Z_cam points Right (+X_L), Y_cam points UP (u_L)
    Z_right = np.array([1.0, 0.0, 0.0])
    Y_right = u_L - np.dot(u_L, Z_right) * Z_right
    Y_right = Y_right / np.linalg.norm(Y_right)
    X_right = np.cross(Y_right, Z_right)
    R_right = np.stack([X_right, Y_right, Z_right], axis=0)
    t_right = np.array([0.0, -0.185, 0.0])

    print("  Stream 0 (Left Lens)  Optical Axis (Z_cam in LiDAR):", R_left[2].round(4))
    print("  Stream 0 (Left Lens)  Vertical Axis (Y_cam in LiDAR):", R_left[1].round(4))
    print("  Stream 1 (Right Lens) Optical Axis (Z_cam in LiDAR):", R_right[2].round(4))
    print("  Stream 1 (Right Lens) Vertical Axis (Y_cam in LiDAR):", R_right[1].round(4))

    # 2. Load SLAM Point Cloud & Trajectory
    print(f"\n[2/4] Loading FAST-LIVO2 SLAM map & trajectory...")
    pts_world = load_pcd_xyz(slam_pcd_path)
    N = len(pts_world)

    trj = np.loadtxt(trj_path)
    t_stamps = trj[:, 0]
    t_xyz = trj[:, 1:4]
    t_quat = trj[:, 4:8]
    t_rot = Rot.from_quat(t_quat).as_matrix()

    t_start, t_end = t_stamps[0], t_stamps[-1]
    t_dur = t_end - t_start
    print(f"  SLAM Map: {N:,} points")
    print(f"  Trajectory: {len(trj)} poses ({t_dur:.2f}s duration: [{t_start:.3f} .. {t_end:.3f}])")

    # 3. Setup Projector & Keyframe sampling
    print(f"\n[3/4] Sampling {num_keyframes} keyframes along trajectory...")
    projector = FisheyeProjector(width=3840, height=3840, fov_deg=196.0)

    color_accum = np.zeros((N, 3), dtype=np.uint8)
    best_cost = np.full(N, 1e9, dtype=np.float32)

    sample_rel_times = np.linspace(0.02, 0.98, num_keyframes)

    t0 = time.time()
    for ki, rel_t in enumerate(sample_rel_times):
        query_t = t_start + rel_t * t_dur
        pose_idx = np.argmin(np.abs(t_stamps - query_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        v_time_sec = rel_t * min(88.0, t_dur)

        # Stream 0 is Left Lens, Stream 1 is Right Lens
        img_left = extract_stream_frame(video_path, stream_idx=0, time_sec=v_time_sec)
        img_right = extract_stream_frame(video_path, stream_idx=1, time_sec=v_time_sec)

        if img_left is None and img_right is None:
            continue

        # Points in current LiDAR body frame: P_l = (P_w - t_w_l) @ R_w_l
        P_l = (pts_world - t_w_l) @ R_w_l

        # 1. Left Lens (Stream 0)
        if img_left is not None:
            P_cam_l = P_l @ R_left.T + t_left
            u_l, v_l, mask_l, dist_l, r_l = projector.project(P_cam_l)
            cost_l = dist_l + (r_l / 1920.0) * 1.5
            better_l = mask_l & (cost_l < best_cost)
            if np.any(better_l):
                u_idx = np.round(u_l[better_l]).astype(np.int64)
                v_idx = np.round(v_l[better_l]).astype(np.int64)
                bgr = img_left[v_idx, u_idx]
                color_accum[better_l] = bgr[:, ::-1]
                best_cost[better_l] = cost_l[better_l]

        # 2. Right Lens (Stream 1)
        if img_right is not None:
            P_cam_r = P_l @ R_right.T + t_right
            u_r, v_r, mask_r, dist_r, r_r = projector.project(P_cam_r)
            cost_r = dist_r + (r_r / 1920.0) * 1.5
            better_r = mask_r & (cost_r < best_cost)
            if np.any(better_r):
                u_idx = np.round(u_r[better_r]).astype(np.int64)
                v_idx = np.round(v_r[better_r]).astype(np.int64)
                bgr = img_right[v_idx, u_idx]
                color_accum[better_r] = bgr[:, ::-1]
                best_cost[better_r] = cost_r[better_r]

        pct = (np.sum(best_cost < 1e8) / N) * 100
        print(f"  Frame {ki+1:2d}/{num_keyframes} (t={v_time_sec:4.1f}s): {pct:5.1f}% map colorized")

    dt = time.time() - t0
    total_pct = (np.sum(best_cost < 1e8) / N) * 100
    print(f"\n[+] Dynamic colorization complete in {dt:.1f}s ({total_pct:.2f}% coverage)!")

    # 4. Save Final Point Clouds
    print(f"\n[4/4] Writing colorized output files...")
    out_pcd_path = os.path.splitext(out_ply_path)[0] + ".pcd"

    write_ply(out_ply_path, pts_world, color_accum)
    write_pcd(out_pcd_path, pts_world, color_accum)

    print("=" * 70)
    print(" COLORIZATION SUCCESSFUL!")
    print(f" PLY: {out_ply_path}")
    print(f" PCD: {out_pcd_path}")
    print("=" * 70)
    return out_ply_path, out_pcd_path


if __name__ == "__main__":
    slam_pcd = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\Log\small_test\pcd\all_raw_points.pcd"
    trj_file = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\Log\small_test\result\Raven_3DMakerPro_Scan.txt"
    insv_file = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST\VID_20260821_105436_00_269.insv"

    out_ply = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST\colorized_slam_small_dataset.ply"

    run_dynamic_colorization(
        slam_pcd_path=slam_pcd,
        trj_path=trj_file,
        video_path=insv_file,
        out_ply_path=out_ply,
        num_keyframes=50,
    )
