#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Candidate Colorization Generator for Systematic Evaluation
================================================================
Generates candidate colorized PCDs across:
1. Orientation / Lens Configurations (Sideways Left-Right, Swapped, Forward-Backward)
2. Time Synchronization Offsets (-6s to +6s)
3. Single-Keyframe Ground Truth Alignment Checks
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
SLAM_PCD = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\Log\small_test\pcd\all_raw_points.pcd"
TRJ_FILE = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION\Log\small_test\result\Raven_3DMakerPro_Scan.txt"
INSV_FILE = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST\VID_20260821_105436_00_269.insv"
OUT_DIR = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST\candidates"


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
    print(f"  [+] Saved Candidate: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB)")


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


# Frame Cache to avoid re-extracting video frames for each candidate
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


def get_orientation_configs():
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)  # Down vector
    u_L = -d_L  # Up vector in LiDAR

    # Forward direction in LiDAR
    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)
    fwd_L = np.cross(u_L, right_L)

    configs = {}

    # Config 1: Standard Sideways (Stream 0 = Left [-X], Stream 1 = Right [+X])
    Z0 = -right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R0 = np.stack([X0, Y0, Z0], axis=0)

    Z1 = right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R1 = np.stack([X1, Y1, Z1], axis=0)

    configs["sideways_standard"] = {
        "desc": "Stream 0 = Left (-X), Stream 1 = Right (+X)",
        "R_stream0": R0,
        "t_stream0": np.array([0.0, -0.185, 0.0]),
        "R_stream1": R1,
        "t_stream1": np.array([0.0, -0.185, 0.0]),
    }

    # Config 2: Swapped Sideways (Stream 0 = Right [+X], Stream 1 = Left [-X])
    configs["sideways_swapped_streams"] = {
        "desc": "Stream 0 = Right (+X), Stream 1 = Left (-X)",
        "R_stream0": R1,
        "t_stream0": np.array([0.0, -0.185, 0.0]),
        "R_stream1": R0,
        "t_stream1": np.array([0.0, -0.185, 0.0]),
    }

    # Config 3: Forward-Backward Mount (Stream 0 = Forward, Stream 1 = Backward)
    Z_fwd = fwd_L
    Y_fwd = u_L - np.dot(u_L, Z_fwd) * Z_fwd
    Y_fwd /= np.linalg.norm(Y_fwd)
    X_fwd = np.cross(Y_fwd, Z_fwd)
    R_fwd = np.stack([X_fwd, Y_fwd, Z_fwd], axis=0)

    Z_bwd = -fwd_L
    Y_bwd = u_L - np.dot(u_L, Z_bwd) * Z_bwd
    Y_bwd /= np.linalg.norm(Y_bwd)
    X_bwd = np.cross(Y_bwd, Z_bwd)
    R_bwd = np.stack([X_bwd, Y_bwd, Z_bwd], axis=0)

    configs["mount_forward_backward"] = {
        "desc": "Stream 0 = Forward (+Fwd), Stream 1 = Backward (-Fwd)",
        "R_stream0": R_fwd,
        "t_stream0": np.array([0.0, -0.185, 0.0]),
        "R_stream1": R_bwd,
        "t_stream1": np.array([0.0, -0.185, 0.0]),
    }

    # Config 4: Forward-Backward Swapped (Stream 0 = Backward, Stream 1 = Forward)
    configs["mount_backward_forward"] = {
        "desc": "Stream 0 = Backward (-Fwd), Stream 1 = Forward (+Fwd)",
        "R_stream0": R_bwd,
        "t_stream0": np.array([0.0, -0.185, 0.0]),
        "R_stream1": R_fwd,
        "t_stream1": np.array([0.0, -0.185, 0.0]),
    }

    return configs


def colorize_candidate(
    pts_world,
    trj,
    cfg,
    time_offset_sec=0.0,
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

    R0 = cfg["R_stream0"]
    t0 = cfg["t_stream0"]
    R1 = cfg["R_stream1"]
    t1 = cfg["t_stream1"]

    if keyframe_subset is not None:
        sample_rel_times = keyframe_subset
    else:
        sample_rel_times = np.linspace(0.05, 0.95, num_keyframes)

    for rel_t in sample_rel_times:
        query_t = t_start + rel_t * t_dur
        pose_idx = np.argmin(np.abs(t_stamps - query_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        # Video time with time offset shift
        v_time_sec = rel_t * min(88.0, t_dur) + time_offset_sec
        v_time_sec = np.clip(v_time_sec, 0.5, 87.5)

        img0 = get_cached_frame(stream_idx=0, time_sec=v_time_sec)
        img1 = get_cached_frame(stream_idx=1, time_sec=v_time_sec)

        P_l = (pts_world - t_w_l) @ R_w_l

        if img0 is not None:
            P_cam0 = P_l @ R0.T + t0
            u0, v0, mask0, dist0, r0 = projector.project(P_cam0)
            cost0 = dist0 + (r0 / 1920.0) * 1.5
            better0 = mask0 & (cost0 < best_cost)
            if np.any(better0):
                u_idx = np.round(u0[better0]).astype(np.int64)
                v_idx = np.round(v0[better0]).astype(np.int64)
                bgr0 = img0[v_idx, u_idx]
                color_accum[better0] = bgr0[:, ::-1]
                best_cost[better0] = cost0[better0]

        if img1 is not None:
            P_cam1 = P_l @ R1.T + t1
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


def run_all_candidates():
    print("=" * 70)
    print(" GENERATING SYSTEMATIC COLORIZATION CANDIDATE POINT CLOUDS")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)
    pts_world = load_pcd_xyz(SLAM_PCD)
    trj = np.loadtxt(TRJ_FILE)
    configs = get_orientation_configs()

    candidates_to_run = [
        # 1. Orientation variations with zero time offset
        {
            "name": "cand_01_sideways_standard_sync0s",
            "cfg": configs["sideways_standard"],
            "offset": 0.0,
            "desc": "Standard Sideways (S0=Left, S1=Right), Sync 0s",
        },
        {
            "name": "cand_02_sideways_swapped_sync0s",
            "cfg": configs["sideways_swapped_streams"],
            "offset": 0.0,
            "desc": "Swapped Sideways (S0=Right, S1=Left), Sync 0s",
        },
        {
            "name": "cand_03_forward_backward_sync0s",
            "cfg": configs["mount_forward_backward"],
            "offset": 0.0,
            "desc": "Forward-Backward Mount (S0=Fwd, S1=Bwd), Sync 0s",
        },
        {
            "name": "cand_04_backward_forward_sync0s",
            "cfg": configs["mount_backward_forward"],
            "offset": 0.0,
            "desc": "Backward-Forward Mount (S0=Bwd, S1=Fwd), Sync 0s",
        },
        # 2. Time Synchronization Sweeps for Standard Sideways
        {
            "name": "cand_05_sideways_sync_minus4s",
            "cfg": configs["sideways_standard"],
            "offset": -4.0,
            "desc": "Standard Sideways, Sync -4.0s offset",
        },
        {
            "name": "cand_06_sideways_sync_minus2s",
            "cfg": configs["sideways_standard"],
            "offset": -2.0,
            "desc": "Standard Sideways, Sync -2.0s offset",
        },
        {
            "name": "cand_07_sideways_sync_plus2s",
            "cfg": configs["sideways_standard"],
            "offset": +2.0,
            "desc": "Standard Sideways, Sync +2.0s offset",
        },
        {
            "name": "cand_08_sideways_sync_plus4s",
            "cfg": configs["sideways_standard"],
            "offset": +4.0,
            "desc": "Standard Sideways, Sync +4.0s offset",
        },
        # 3. Time Synchronization Sweeps for Swapped Sideways
        {
            "name": "cand_09_swapped_sync_minus4s",
            "cfg": configs["sideways_swapped_streams"],
            "offset": -4.0,
            "desc": "Swapped Sideways, Sync -4.0s offset",
        },
        {
            "name": "cand_10_swapped_sync_plus4s",
            "cfg": configs["sideways_swapped_streams"],
            "offset": +4.0,
            "desc": "Swapped Sideways, Sync +4.0s offset",
        },
        # 4. Single-Keyframe Ground Truth Check (at t=30s)
        {
            "name": "cand_11_single_keyframe_sideways_t30s",
            "cfg": configs["sideways_standard"],
            "offset": 0.0,
            "subset": [0.35],
            "desc": "Single Keyframe (t=30s) Standard Sideways",
        },
        {
            "name": "cand_12_single_keyframe_swapped_t30s",
            "cfg": configs["sideways_swapped_streams"],
            "offset": 0.0,
            "subset": [0.35],
            "desc": "Single Keyframe (t=30s) Swapped Sideways",
        },
    ]

    summary = []

    for i, c in enumerate(candidates_to_run):
        print(f"\n[{i+1}/{len(candidates_to_run)}] Running {c['name']} ({c['desc']})...")
        t0 = time.time()
        colors = colorize_candidate(
            pts_world=pts_world,
            trj=trj,
            cfg=c["cfg"],
            time_offset_sec=c.get("offset", 0.0),
            num_keyframes=30,
            keyframe_subset=c.get("subset", None),
        )
        pcd_out = os.path.join(OUT_DIR, f"{c['name']}.pcd")
        write_pcd(pcd_out, pts_world, colors)
        dt = time.time() - t0
        summary.append({
            "candidate": c["name"],
            "file": pcd_out,
            "description": c["desc"],
            "elapsed_s": round(dt, 1),
        })

    summary_file = os.path.join(OUT_DIR, "candidates_index.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" ALL CANDIDATES GENERATED SUCCESSFULLY!")
    print(f" Directory: {OUT_DIR}")
    print(f" Index:     {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_candidates()
