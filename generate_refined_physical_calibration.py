#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical Lever-Arm & Bundle Calibrated Colorizer
================================================
Incorporates:
1. True physical mount lever arm along gravity UP vector:
   t_cam = 0.185 * u_L = [0.0, -0.0934, -0.1597] m (fixes the 16cm vertical shift)
2. Bundle adjustment angular trims (pitch -0.42 deg, yaw +0.07 deg, roll +0.51 deg)
3. Bundle-optimized time sync: Delta t = +5.44s
4. Generates single-keyframe and full-trajectory PCDs for CloudCompare inspection.
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
SLAM_PCD = os.path.join(DATASET_DIR, "slam_out", "pcd", "all_raw_points.pcd")
TRJ_FILE = os.path.join(DATASET_DIR, "slam_out", "result", "Raven_3DMakerPro_Scan.txt")
INSV_FILE = os.path.join(DATASET_DIR, "VID_20260902_143757_00_277.insv")
OUT_DIR = os.path.join(DATASET_DIR, "calibration_refined_pcds")
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
        X, Y, Z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
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


def colorize_experiment(
    pts_world,
    trj,
    R_s0,
    t_s0,
    R_s1,
    t_s1,
    time_offset_sec,
    keyframe_subset=None,
    num_keyframes=45,
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

        # Stream 0 (Right)
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

        # Stream 1 (Left)
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

    return color_accum


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 70)
    print(" GENERATING REFINED CALIBRATION CANDIDATES WITH TRUE LEVER ARM")
    print("=" * 70)

    pts_world = load_pcd_xyz(SLAM_PCD)
    trj = np.loadtxt(TRJ_FILE)

    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)
    u_L = -d_L  # Gravity UP in LiDAR frame

    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    # Stream 0: Right (+X_L)
    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_base0 = np.stack([X0, Y0, Z0], axis=0)

    # Stream 1: Left (-X_L)
    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_base1 = np.stack([X1, Y1, Z1], axis=0)

    # TRUE PHYSICAL LEVER ARM (Camera mounted 18.5cm along handle UP vector)
    t_cam_true = 0.185 * u_L
    print(f"[*] True Physical Lever Arm: {t_cam_true.round(4)} m")

    experiments = [
        # Exp 1: Single Keyframe with True Lever Arm & Nominal Orientation (Delta t = 5.44s)
        {
            "name": "cand_opt_01_true_lever_arm_single_keyframe",
            "R0": R_base0,
            "R1": R_base1,
            "t": t_cam_true,
            "dt": 5.44,
            "subset": [0.38],
            "desc": "Single Keyframe (t=35s) with True Physical Lever Arm (18.5cm along UP)",
        },
        # Exp 2: Single Keyframe with True Lever Arm + Bundle Trims (Pitch -0.42, Yaw +0.07, Roll +0.51)
        {
            "name": "cand_opt_02_bundle_trimmed_single_keyframe",
            "R0": Rot.from_euler("xyz", [-0.42, 0.07, 0.51], degrees=True).as_matrix() @ R_base0,
            "R1": Rot.from_euler("xyz", [-0.42, 0.07, 0.51], degrees=True).as_matrix() @ R_base1,
            "t": t_cam_true,
            "dt": 5.44,
            "subset": [0.38],
            "desc": "Single Keyframe with True Lever Arm + Bundle Trims",
        },
        # Exp 3: Single Keyframe with Pitch +2 deg trim (checking wall frame shift)
        {
            "name": "cand_opt_03_pitch_plus2deg_single_keyframe",
            "R0": Rot.from_euler("xyz", [+2.0, 0.0, 0.0], degrees=True).as_matrix() @ R_base0,
            "R1": Rot.from_euler("xyz", [+2.0, 0.0, 0.0], degrees=True).as_matrix() @ R_base1,
            "t": t_cam_true,
            "dt": 5.44,
            "subset": [0.38],
            "desc": "Single Keyframe with Pitch +2.0 deg trim",
        },
        # Exp 4: Single Keyframe with Pitch -2 deg trim
        {
            "name": "cand_opt_04_pitch_minus2deg_single_keyframe",
            "R0": Rot.from_euler("xyz", [-2.0, 0.0, 0.0], degrees=True).as_matrix() @ R_base0,
            "R1": Rot.from_euler("xyz", [-2.0, 0.0, 0.0], degrees=True).as_matrix() @ R_base1,
            "t": t_cam_true,
            "dt": 5.44,
            "subset": [0.38],
            "desc": "Single Keyframe with Pitch -2.0 deg trim",
        },
        # Exp 5: Full Trajectory with True Physical Lever Arm & Bundle Trims (Delta t = 5.44s)
        {
            "name": "cand_opt_05_full_trajectory_perfected",
            "R0": Rot.from_euler("xyz", [-0.42, 0.07, 0.51], degrees=True).as_matrix() @ R_base0,
            "R1": Rot.from_euler("xyz", [-0.42, 0.07, 0.51], degrees=True).as_matrix() @ R_base1,
            "t": t_cam_true,
            "dt": 5.44,
            "subset": None,
            "desc": "Full Trajectory Colorization with True Lever Arm & Bundle Trims",
        },
    ]

    for exp in experiments:
        print(f"\n[*] Processing {exp['name']} ({exp['desc']})...")
        colors = colorize_experiment(
            pts_world=pts_world,
            trj=trj,
            R_s0=exp["R0"],
            t_s0=exp["t"],
            R_s1=exp["R1"],
            t_s1=exp["t"],
            time_offset_sec=exp["dt"],
            keyframe_subset=exp.get("subset", None),
            num_keyframes=45,
        )
        out_pcd = os.path.join(OUT_DIR, f"{exp['name']}.pcd")
        write_pcd(out_pcd, pts_world, colors)

    print("\n" + "=" * 70)
    print(" ALL REFINED CANDIDATES GENERATED!")
    print(f" Output Directory: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run()
