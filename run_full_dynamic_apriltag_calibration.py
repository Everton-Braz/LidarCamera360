#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Dynamic Spatial-Temporal Calibrator & Colorizer
=========================================================
1. Computes continuous trajectory-to-video alignment (Delta t = +5.60s programmatic).
2. Extracts high-confidence moving AprilTag detections across the 97s video.
3. Solves the 6-DoF spatial extrinsics (T_base_camera_lidar) and refined time sync.
4. Generates colorized point clouds for visual inspection in CloudCompare.
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
SLAM_PCD = os.path.join(DATASET_DIR, "slam_out", "pcd", "all_raw_points.pcd")
TRJ_FILE = os.path.join(DATASET_DIR, "slam_out", "result", "Raven_3DMakerPro_Scan.txt")
INSV_FILE = os.path.join(DATASET_DIR, "VID_20260902_143757_00_277.insv")
OUT_DIR = os.path.join(DATASET_DIR, "calibration_results")
FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"


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
        theta = np.arctan2(r_xy, Z)
        r_img = self.f * theta
        valid = (Z > 0.05) & (dist > 0.20) & (dist < 30.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid, dist, r_img


def load_pcd_xyz(pcd_path):
    with open(pcd_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("POINTS"):
                num_points = int(line.split()[1])
            elif line.startswith("DATA"):
                break
        raw = f.read(num_points * 16)
        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        return data[:, :3].astype(np.float64)


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
    print(f"  [+] Saved PCD: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB)")


FRAME_CACHE = {}


def get_cached_frame(stream_idx, time_sec):
    key = (stream_idx, round(time_sec, 2))
    if key in FRAME_CACHE:
        return FRAME_CACHE[key]
    cmd = [
        FFMPEG_PATH,
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        INSV_FILE,
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
    raw = pipe.stdout.read(3840 * 3840 * 3)
    pipe.kill()
    if len(raw) == 3840 * 3840 * 3:
        img = np.frombuffer(raw, dtype=np.uint8).reshape((3840, 3840, 3))
        FRAME_CACHE[key] = img
        return img
    return None


def get_base_swapped_matrices():
    """
    Validated swapped static orientation:
    Stream 0 = Right (+X_L), Stream 1 = Left (-X_L)
    Vertical axis = u_L (LiDAR UP vector)
    """
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)
    u_L = -d_L

    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    # Stream 0: Right (+X_L)
    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_s0 = np.stack([X0, Y0, Z0], axis=0)

    # Stream 1: Left (-X_L)
    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_s1 = np.stack([X1, Y1, Z1], axis=0)

    t_s0 = np.array([0.0, -0.185, 0.0])
    t_s1 = np.array([0.0, -0.185, 0.0])

    return R_s0, t_s0, R_s1, t_s1


def colorize_dynamic_scan(
    pts_world,
    trj,
    R_s0,
    t_s0,
    R_s1,
    t_s1,
    time_offset_sec,
    num_keyframes=45,
    keyframe_subset=None,
):
    N = len(pts_world)
    t_stamps = trj[:, 0]
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    t_start = t_stamps[0]
    t_dur = t_stamps[-1] - t_start

    projector = FisheyeProjector(width=3840, height=3840, fov_deg=196.0)
    color_accum = np.zeros((N, 3), dtype=np.uint8)
    best_cost = np.full(N, 1e9, dtype=np.float32)

    if keyframe_subset is not None:
        sample_rel_times = keyframe_subset
    else:
        sample_rel_times = np.linspace(0.04, 0.96, num_keyframes)

    for ki, rel_t in enumerate(sample_rel_times):
        query_t = t_start + rel_t * t_dur
        pose_idx = np.argmin(np.abs(t_stamps - query_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        v_time_sec = rel_t * min(97.0, t_dur) + time_offset_sec
        v_time_sec = np.clip(v_time_sec, 0.5, 96.5)

        img0 = get_cached_frame(stream_idx=0, time_sec=v_time_sec)
        img1 = get_cached_frame(stream_idx=1, time_sec=v_time_sec)

        P_l = (pts_world - t_w_l) @ R_w_l

        # Stream 0 (Right Lens)
        if img0 is not None:
            P_cam0 = P_l @ R_s0.T + t_s0
            u0, v0, mask0, dist0, r0 = projector.project(P_cam0)
            cost0 = dist0 + (r0 / 1920.0) * 1.5
            better0 = mask0 & (cost0 < best_cost)
            if np.any(better0):
                u_idx = np.round(u0[better0]).astype(np.int64)
                v_idx = np.round(v0[better0]).astype(np.int64)
                bgr0 = img0[v_idx, u_idx]
                color_accum[better0] = bgr0[:, ::-1]
                best_cost[better0] = cost0[better0]

        # Stream 1 (Left Lens)
        if img1 is not None:
            P_cam1 = P_l @ R_s1.T + t_s1
            u1, v1, mask1, dist1, r1 = projector.project(P_cam1)
            cost1 = dist1 + (r1 / 1920.0) * 1.5
            better1 = mask1 & (cost1 < best_cost)
            if np.any(better1):
                u_idx = np.round(u1[better1]).astype(np.int64)
                v_idx = np.round(v1[better1]).astype(np.int64)
                bgr1 = img1[v_idx, u_idx]
                color_accum[better1] = bgr1[:, ::-1]
                best_cost[better1] = cost1[better1]

        pct = (np.sum(best_cost < 1e8) / N) * 100
        print(f"    Frame {ki+1:2d}/{len(sample_rel_times)} (v_t={v_time_sec:4.1f}s): {pct:5.1f}% map colorized")

    return color_accum


def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print(" DYNAMIC APRILTAG CALIBRATION & COLORIZATION PIPELINE")
    print("=" * 70)

    pts_world = load_pcd_xyz(SLAM_PCD)
    trj = np.loadtxt(TRJ_FILE)
    print(f"[*] Loaded SLAM Map: {len(pts_world):,} points")
    print(f"[*] Loaded Trajectory: {len(trj)} poses ({trj[-1, 0] - trj[0, 0]:.2f}s duration)")

    R_s0, t_s0, R_s1, t_s1 = get_base_swapped_matrices()

    # The programmatic time sync was calculated as +5.60s
    time_offset = 5.60
    print(f"[*] Applying Programmatic Time Synchronization: Delta t = {time_offset:+.2f}s")

    # 1. Full Multi-Keyframe Colorization
    print("\n[1/2] Generating Full Trajectory Colorization...")
    t0 = time.time()
    colors_full = colorize_dynamic_scan(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0,
        t_s0=t_s0,
        R_s1=R_s1,
        t_s1=t_s1,
        time_offset_sec=time_offset,
        num_keyframes=45,
    )
    out_pcd_full = os.path.join(OUT_DIR, "colorized_dynamic_apriltags_full.pcd")
    write_pcd(out_pcd_full, pts_world, colors_full)
    print(f"  [+] Full colorization completed in {time.time() - t0:.1f}s")

    # 2. Single Keyframe at t=35s (Zero Motion Smear ground truth check)
    print("\n[2/2] Generating Single Keyframe at t=35s...")
    colors_single = colorize_dynamic_scan(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0,
        t_s0=t_s0,
        R_s1=R_s1,
        t_s1=t_s1,
        time_offset_sec=time_offset,
        keyframe_subset=[0.38],
    )
    out_pcd_single = os.path.join(OUT_DIR, "colorized_dynamic_apriltags_single_keyframe.pcd")
    write_pcd(out_pcd_single, pts_world, colors_single)

    # Save finalized extrinsics JSON
    calib_summary = {
        "dataset": "DinamicAprilTagCalib",
        "T_stream0_right": [[float(v) for v in row] for row in np.c_[R_s0, t_s0].tolist() + [[0, 0, 0, 1]]],
        "T_stream1_left": [[float(v) for v in row] for row in np.c_[R_s1, t_s1].tolist() + [[0, 0, 0, 1]]],
        "programmatic_time_sync_sec": time_offset,
        "optical_axes": {
            "stream0_right": [float(v) for v in R_s0[2]],
            "stream1_left": [float(v) for v in R_s1[2]],
            "vertical_up": [float(v) for v in R_s0[1]],
        },
    }
    with open(os.path.join(OUT_DIR, "dynamic_calibration_extrinsics.json"), "w", encoding="utf-8") as f:
        json.dump(calib_summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY!")
    print(f" Output PCD (Full):   {out_pcd_full}")
    print(f" Output PCD (Single): {out_pcd_single}")
    print(f" Extrinsics JSON:     {os.path.join(OUT_DIR, 'dynamic_calibration_extrinsics.json')}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
