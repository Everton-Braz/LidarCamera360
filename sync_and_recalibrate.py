#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programmatic Video-to-LiDAR Time Synchronization & Refined Calibration Candidates
=================================================================================
1. Programmatically computes the exact time offset Delta t via normalized
   cross-correlation between video optical flow and SLAM trajectory speed.
2. Generates a targeted set of refined candidates around the optimal programmatic
   time offset and fine-tuning angles for CloudCompare evaluation.
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from scipy.signal import correlate
from scipy.spatial.transform import Rotation as Rot

FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
SLAM_PCD = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\Log\small_test\pcd\all_raw_points.pcd"
TRJ_FILE = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\Log\small_test\result\Raven_3DMakerPro_Scan.txt"
INSV_FILE = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST\VID_20260821_105436_00_269.insv"
OUT_DIR = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST\candidates_refined"


def compute_programmatic_time_sync(video_path, trj_path, fps=5.0):
    """
    Computes optimal time offset Delta t by cross-correlating visual optical flow
    with LiDAR SLAM velocity profile. Returns offset in seconds.
    """
    print("\n[+] Computing Programmatic Video-to-LiDAR Time Synchronization...")
    trj = np.loadtxt(trj_path)
    t_trj = trj[:, 0] - trj[0, 0]
    xyz = trj[:, 1:4]
    dt = np.diff(t_trj)
    dxyz = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    v_trj = dxyz / np.maximum(dt, 1e-4)
    t_mid = 0.5 * (t_trj[:-1] + t_trj[1:])

    # Extract downsampled frames at 5 FPS from stream 0
    W, H = 320, 320
    cmd = [
        FFMPEG_PATH,
        "-i",
        video_path,
        "-map",
        "0:v:0",
        "-vf",
        f"fps={fps},scale={W}:{H}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_sz = W * H

    flow_mags = []
    prev_frame = None

    while True:
        raw = pipe.stdout.read(frame_sz)
        if len(raw) < frame_sz:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((H, W))
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = float(np.mean(np.hypot(flow[..., 0], flow[..., 1])))
            flow_mags.append(mag)
        else:
            flow_mags.append(0.0)
        prev_frame = frame

    pipe.kill()
    v_video = np.array(flow_mags)
    t_video = np.arange(len(v_video)) / fps

    # Resample trajectory velocity onto video grid
    grid_t = t_video[(t_video >= t_mid[0]) & (t_video <= t_mid[-1])]
    v_trj_interp = np.interp(grid_t, t_mid, v_trj)
    v_vid_sub = v_video[: len(grid_t)]

    # Normalized cross-correlation
    v_trj_norm = (v_trj_interp - np.mean(v_trj_interp)) / np.std(v_trj_interp)
    v_vid_norm = (v_vid_sub - np.mean(v_vid_sub)) / np.std(v_vid_sub)

    corr = correlate(v_vid_norm, v_trj_norm, mode="full")
    lags = np.arange(-len(v_trj_norm) + 1, len(v_vid_norm)) / fps

    # Search in window [-10s, +10s]
    valid_mask = (lags >= -10.0) & (lags <= 10.0)
    corr_valid = corr[valid_mask]
    lags_valid = lags[valid_mask]

    best_idx = np.argmax(corr_valid)
    # The physical lead of video relative to LiDAR start
    raw_lag = lags_valid[best_idx]
    optimal_offset = abs(raw_lag)

    print(f"    Peak Cross-Correlation: {corr_valid[best_idx]:.2f}")
    print(f"    Programmatically Detected Time Offset: {optimal_offset:+.2f} seconds")
    return optimal_offset


def load_pcd_xyz(pcd_path):
    with open(pcd_path, "rb") as f:
        num_points = 0
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
    print(f"  [+] Saved Refined PCD: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB)")


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
        valid = (Z > 0.05) & (dist > 0.25) & (dist < 30.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid, dist, r_img


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


def get_base_matrices():
    """
    Returns baseline orthogonal frames for Stream 0 (Right, +X) and Stream 1 (Left, -X).
    """
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)
    u_L = -d_L  # Up vector in LiDAR

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

    return R_s0, R_s1


def colorize_run(
    pts_world,
    trj,
    R_s0,
    R_s1,
    time_offset_sec,
    num_keyframes=35,
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

    t0_cam = np.array([0.0, -0.185, 0.0])
    t1_cam = np.array([0.0, -0.185, 0.0])

    if keyframe_subset is not None:
        sample_rel_times = keyframe_subset
    else:
        sample_rel_times = np.linspace(0.05, 0.95, num_keyframes)

    for rel_t in sample_rel_times:
        query_t = t_start + rel_t * t_dur
        pose_idx = np.argmin(np.abs(t_stamps - query_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        v_time_sec = rel_t * min(88.0, t_dur) + time_offset_sec
        v_time_sec = np.clip(v_time_sec, 0.5, 87.5)

        img0 = get_cached_frame(stream_idx=0, time_sec=v_time_sec)
        img1 = get_cached_frame(stream_idx=1, time_sec=v_time_sec)

        P_l = (pts_world - t_w_l) @ R_w_l

        # Stream 0 (Right)
        if img0 is not None:
            P_cam0 = P_l @ R_s0.T + t0_cam
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
            P_cam1 = P_l @ R_s1.T + t1_cam
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


def run_refined_candidates():
    print("=" * 70)
    print(" EXECUTING PROGRAMMATIC SYNC & REFINED CALIBRATION CANDIDATES")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    pts_world = load_pcd_xyz(SLAM_PCD)
    trj = np.loadtxt(TRJ_FILE)

    # 1. Programmatically compute time sync offset
    auto_offset = compute_programmatic_time_sync(INSV_FILE, TRJ_FILE, fps=5.0)

    # 2. Base Swapped Matrices (S0=Right, S1=Left)
    R_base_s0, R_base_s1 = get_base_matrices()

    def make_rotated_pair(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
        # Rotation delta in camera frame (X right, Y down, Z forward)
        R_delta = Rot.from_euler("xyz", [pitch_deg, yaw_deg, roll_deg], degrees=True).as_matrix()
        return R_delta @ R_base_s0, R_delta @ R_base_s1

    # Define refined candidate experiments
    candidates = [
        # Candidate 1: Automatic Programmatic Sync (Exact Delta t)
        {
            "name": f"cand_ref_01_auto_sync_{auto_offset:.2f}s",
            "offset": auto_offset,
            "R0": R_base_s0,
            "R1": R_base_s1,
            "desc": f"Programmatic Auto-Sync ({auto_offset:.2f}s), Nominal Orientation",
        },
        # Candidate 2: Auto-Sync + Pitch -3 deg
        {
            "name": f"cand_ref_02_auto_pitch_minus3deg",
            "offset": auto_offset,
            "R0": make_rotated_pair(pitch_deg=-3.0)[0],
            "R1": make_rotated_pair(pitch_deg=-3.0)[1],
            "desc": f"Auto-Sync ({auto_offset:.2f}s) with Pitch -3.0 deg",
        },
        # Candidate 3: Auto-Sync + Pitch +3 deg
        {
            "name": f"cand_ref_03_auto_pitch_plus3deg",
            "offset": auto_offset,
            "R0": make_rotated_pair(pitch_deg=+3.0)[0],
            "R1": make_rotated_pair(pitch_deg=+3.0)[1],
            "desc": f"Auto-Sync ({auto_offset:.2f}s) with Pitch +3.0 deg",
        },
        # Candidate 4: Auto-Sync + Yaw -4 deg
        {
            "name": f"cand_ref_04_auto_yaw_minus4deg",
            "offset": auto_offset,
            "R0": make_rotated_pair(yaw_deg=-4.0)[0],
            "R1": make_rotated_pair(yaw_deg=-4.0)[1],
            "desc": f"Auto-Sync ({auto_offset:.2f}s) with Yaw -4.0 deg",
        },
        # Candidate 5: Auto-Sync + Yaw +4 deg
        {
            "name": f"cand_ref_05_auto_yaw_plus4deg",
            "offset": auto_offset,
            "R0": make_rotated_pair(yaw_deg=+4.0)[0],
            "R1": make_rotated_pair(yaw_deg=+4.0)[1],
            "desc": f"Auto-Sync ({auto_offset:.2f}s) with Yaw +4.0 deg",
        },
        # Candidate 6: Time Offset Sweep: +4.0s (user's visual favorite)
        {
            "name": "cand_ref_06_sync_plus4.0s",
            "offset": 4.0,
            "R0": R_base_s0,
            "R1": R_base_s1,
            "desc": "Sync +4.0s (User Selected Favorite Time Base)",
        },
        # Candidate 7: Time Offset Sweep: +4.3s
        {
            "name": "cand_ref_07_sync_plus4.3s",
            "offset": 4.3,
            "R0": R_base_s0,
            "R1": R_base_s1,
            "desc": "Sync +4.3s Fine Interpolation",
        },
        # Candidate 8: Single Keyframe at t=35s with Programmatic Sync (Zero Motion Smear)
        {
            "name": "cand_ref_08_single_keyframe_clean",
            "offset": auto_offset,
            "R0": R_base_s0,
            "R1": R_base_s1,
            "subset": [0.40],
            "desc": f"Single Keyframe at t=37s (Auto Sync {auto_offset:.2f}s, Zero Motion Blur)",
        },
    ]

    index_data = []

    for i, c in enumerate(candidates):
        print(f"\n[{i+1}/{len(candidates)}] Processing {c['name']} ({c['desc']})...")
        t0 = time.time()
        colors = colorize_run(
            pts_world=pts_world,
            trj=trj,
            R_s0=c["R0"],
            R_s1=c["R1"],
            time_offset_sec=c["offset"],
            num_keyframes=35,
            keyframe_subset=c.get("subset", None),
        )
        pcd_out = os.path.join(OUT_DIR, f"{c['name']}.pcd")
        write_pcd(pcd_out, pts_world, colors)
        dt = time.time() - t0
        index_data.append({
            "name": c["name"],
            "file": pcd_out,
            "offset_s": c["offset"],
            "desc": c["desc"],
            "elapsed_s": round(dt, 1),
        })

    index_file = os.path.join(OUT_DIR, "refined_index.json")
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=2)

    print("\n" + "=" * 70)
    print(" ALL REFINED CANDIDATES GENERATED!")
    print(f" Output Directory: {OUT_DIR}")
    print(f" Summary Index:    {index_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_refined_candidates()
