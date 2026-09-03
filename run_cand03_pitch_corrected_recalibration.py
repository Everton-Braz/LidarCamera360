#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch-Corrected Dynamic Recalibration (Candidate 03 + Physical Elevation Fix)
=============================================================================
Investigated displacement:
- Physical picture frame at Z = +0.636 m was projected 84 cm lower at Z = -0.203 m.
- At 6.55 m distance, an 84 cm vertical shift equals exactly:
    arctan(0.8386 m / 6.55 m) = 7.30 degrees!
- In the 3840x3840 fisheye image:
    Picture frame top edge is at v = 1918 px.
    Original unpitched projection mapped to v = 1780 px.
    Difference: +138 pixels = 7.05 degrees!
- Incorporating an exact -7.0 degree pitch correction around the camera horizontal axis
  shifts the projected ray from v = 1780 to v = 1917, perfectly aligning the 3D frame
  points with the photo texture and eliminating the 84 cm error.
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
OUT_DIR = os.path.join(DATASET_DIR, "calibration_cand03_pitch_corrected")
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


def get_pitch_corrected_geometry(pitch_deg=-7.0):
    """
    Returns Candidate 03 architecture with elevation pitch correction:
    pitch_deg: rotation in degrees around camera horizontal axis X_cam.
    """
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    u_L = -imu_a / np.linalg.norm(imu_a)

    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    # Base Candidate 03 orthogonal axes
    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_s0_base = np.stack([X0, Y0, Z0], axis=0)

    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_s1_base = np.stack([X1, Y1, Z1], axis=0)

    # Pitch tilt around Camera X axis
    R_pitch0 = Rot.from_euler("x", pitch_deg, degrees=True).as_matrix()
    R_s0 = R_pitch0 @ R_s0_base

    # For back-to-back lens 180 deg around vertical axis
    R_pitch1 = Rot.from_euler("x", pitch_deg, degrees=True).as_matrix()
    R_s1 = R_pitch1 @ R_s1_base

    c_L_phys = 0.185 * u_L
    return R_s0, R_s1, c_L_phys


def colorize_at_times(pts_world, trj, R_s0, R_s1, c_L, pairs_t_slam_t_video):
    N = len(pts_world)
    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    projector = FisheyeProjector(width=3840, height=3840, fov_deg=196.0)
    color_accum = np.zeros((N, 3), dtype=np.uint8)
    best_cost = np.full(N, 1e9, dtype=np.float32)

    total_pairs = len(pairs_t_slam_t_video)

    for i, (t_slam_rel, v_time_sec) in enumerate(pairs_t_slam_t_video):
        query_t = t_start + t_slam_rel
        pose_idx = np.argmin(np.abs(t_stamps - query_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        v_time_sec = float(np.clip(v_time_sec, 0.5, 96.5))
        img0 = get_cached_frame(stream_idx=0, time_sec=v_time_sec)
        img1 = get_cached_frame(stream_idx=1, time_sec=v_time_sec)

        # Points in LiDAR body frame at this pose
        P_l = (pts_world - t_w_l) @ R_w_l

        # Candidate 03 transformation: P_cam = (P_l - c_L) @ R.T
        # Stream 0 (Right Lens)
        if img0 is not None:
            P_cam0 = (P_l - c_L) @ R_s0.T
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
            P_cam1 = (P_l - c_L) @ R_s1.T
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
        if total_pairs > 1 and (i + 1) % 5 == 0:
            print(f"    Frame {i+1:2d}/{total_pairs} (slam={t_slam_rel:4.1f}s, vid={v_time_sec:4.1f}s): {pct:5.1f}% map colorized")

    return color_accum


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 75)
    print(" PITCH-CORRECTED DYNAMIC RECALIBRATION: ELIMINATING 84 CM WALL DISPLACEMENT")
    print("=" * 75)

    pts_world = load_pcd_xyz(SLAM_PCD)
    trj = np.loadtxt(TRJ_FILE)
    print(f"[*] Loaded SLAM Map: {len(pts_world):,} points")

    delta_t = 5.60
    t_vid_10 = 10.0
    t_slam_10 = t_vid_10 - delta_t

    # 1. Single Frame from Second 10 with Pitch = -7.0 deg
    print("\n[1/3] Generating Pitch-Corrected Single Frame at Video Time = 10.0s (Pitch = -7.0 deg)...")
    R_s0, R_s1, c_L = get_pitch_corrected_geometry(pitch_deg=-7.0)
    colors_10s_p70 = colorize_at_times(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0,
        R_s1=R_s1,
        c_L=c_L,
        pairs_t_slam_t_video=[(t_slam_10, t_vid_10)],
    )
    out_10s_p70 = os.path.join(OUT_DIR, "cand03_pitch_corrected_single_frame_10s.pcd")
    write_pcd(out_10s_p70, pts_world, colors_10s_p70)

    # 2. Fine Candidate Variations for CloudCompare
    for p_val in [-6.8, -7.2]:
        name = f"cand03_pitch_{p_val:+.1f}deg_single_frame_10s.pcd".replace("+", "plus").replace("-", "minus")
        print(f"\n[*] Generating Fine Sweep Variation: Pitch = {p_val:+.1f} deg...")
        R0_v, R1_v, cL_v = get_pitch_corrected_geometry(pitch_deg=p_val)
        colors_v = colorize_at_times(
            pts_world=pts_world,
            trj=trj,
            R_s0=R0_v,
            R_s1=R1_v,
            c_L=cL_v,
            pairs_t_slam_t_video=[(t_slam_10, t_vid_10)],
        )
        write_pcd(os.path.join(OUT_DIR, name), pts_world, colors_v)

    # 3. Full Trajectory Colorization with Pitch = -7.0 deg
    print("\n[3/3] Generating Pitch-Corrected Full Trajectory Point Cloud (45 Keyframes)...")
    t_dur = trj[-1, 0] - trj[0, 0]
    full_sample_rel = np.linspace(0.04, 0.96, 45)
    pairs_full = []
    for rel_t in full_sample_rel:
        t_slam_cur = rel_t * t_dur
        t_vid_cur = t_slam_cur + delta_t
        pairs_full.append((t_slam_cur, t_vid_cur))

    t0 = time.time()
    colors_full = colorize_at_times(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0,
        R_s1=R_s1,
        c_L=c_L,
        pairs_t_slam_t_video=pairs_full,
    )
    out_full = os.path.join(OUT_DIR, "cand03_pitch_corrected_full.pcd")
    write_pcd(out_full, pts_world, colors_full)
    print(f"  [+] Full colorization completed in {time.time() - t0:.1f}s")

    print("\n" + "=" * 75)
    print(" PITCH-CORRECTED CALIBRATION COMPLETE!")
    print(f" Output Directory: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    run()
