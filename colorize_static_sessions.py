#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colorize Static AprilTag LiDAR Scans
====================================
Projects raw static LiDAR point clouds (.bag) onto dual fisheye lenses
(lens1_front.jpg, lens2_back.jpg) for the 3 test sessions in C:\\Users\\User\\Downloads\\Lidou.
Zero motion, zero time-sync ambiguity — perfectly checks spatial extrinsics.
"""

import os
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, r"c:\Users\User\Documents\APLICATIVOS\Lidar-Camera-calibrator")
import core

BASE_DIR = r"C:\Users\User\Downloads\Lidou"
SESSIONS = ["20260902092818", "20260902092520", "20260902092658"]


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
    print(f"  [+] Saved Static PCD: {path} ({os.path.getsize(path)/1e6:.2f} MB)")


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


def get_extrinsics(mode="swapped"):
    """
    Returns (R_lens1, t_lens1, R_lens2, t_lens2).
    mode='swapped': Lens 1 = Right (+X), Lens 2 = Left (-X)
    mode='standard': Lens 1 = Left (-X), Lens 2 = Right (+X)
    """
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)
    u_L = -d_L  # Up vector in LiDAR

    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    # Right Lens (+X_L)
    Z_r = right_L
    Y_r = u_L - np.dot(u_L, Z_r) * Z_r
    Y_r /= np.linalg.norm(Y_r)
    X_r = np.cross(Y_r, Z_r)
    R_right = np.stack([X_r, Y_r, Z_r], axis=0)

    # Left Lens (-X_L)
    Z_l = -right_L
    Y_l = u_L - np.dot(u_L, Z_l) * Z_l
    Y_l /= np.linalg.norm(Y_l)
    X_l = np.cross(Y_l, Z_l)
    R_left = np.stack([X_l, Y_l, Z_l], axis=0)

    t_cam = np.array([0.0, -0.185, 0.0])

    if mode == "swapped":
        # Lens 1 (Stream 0) = Right, Lens 2 (Stream 1) = Left
        return R_right, t_cam, R_left, t_cam
    else:
        # Lens 1 (Stream 0) = Left, Lens 2 (Stream 1) = Right
        return R_left, t_cam, R_right, t_cam


def colorize_static_session(session_id, mode="swapped"):
    session_dir = os.path.join(BASE_DIR, session_id)
    bag_p = os.path.join(session_dir, f"LIDAR_{session_id}.bag")
    img1_p = os.path.join(session_dir, "extracted_fisheye", "lens1_front.jpg")
    img2_p = os.path.join(session_dir, "extracted_fisheye", "lens2_back.jpg")

    if not os.path.exists(bag_p) or not os.path.exists(img1_p) or not os.path.exists(img2_p):
        print(f"[-] Session {session_id} missing required files, skipping.")
        return

    print(f"\n[*] Processing Session {session_id} [Mode: {mode}]...")
    d_bag = core.read_bag(bag_p)
    pts = d_bag["points"][:, :3].astype(np.float64)
    N = len(pts)
    print(f"    LiDAR Points: {N:,}")

    img1 = cv2.imread(img1_p)
    img2 = cv2.imread(img2_p)

    R1, t1, R2, t2 = get_extrinsics(mode=mode)
    projector = FisheyeProjector(width=3840, height=3840, fov_deg=196.0)

    color_accum = np.zeros((N, 3), dtype=np.uint8)
    best_cost = np.full(N, 1e9, dtype=np.float32)

    # 1. Project Lens 1
    P_cam1 = pts @ R1.T + t1
    u1, v1, mask1, dist1, r1 = projector.project(P_cam1)
    cost1 = dist1 + (r1 / 1920.0) * 1.5
    better1 = mask1 & (cost1 < best_cost)
    if np.any(better1):
        u_idx = np.round(u1[better1]).astype(np.int64)
        v_idx = np.round(v1[better1]).astype(np.int64)
        bgr1 = img1[v_idx, u_idx]
        color_accum[better1] = bgr1[:, ::-1]
        best_cost[better1] = cost1[better1]

    # 2. Project Lens 2
    P_cam2 = pts @ R2.T + t2
    u2, v2, mask2, dist2, r2 = projector.project(P_cam2)
    cost2 = dist2 + (r2 / 1920.0) * 1.5
    better2 = mask2 & (cost2 < best_cost)
    if np.any(better2):
        u_idx = np.round(u2[better2]).astype(np.int64)
        v_idx = np.round(v2[better2]).astype(np.int64)
        bgr2 = img2[v_idx, u_idx]
        color_accum[better2] = bgr2[:, ::-1]
        best_cost[better2] = cost2[better2]

    out_pcd = os.path.join(session_dir, f"colorized_static_{mode}.pcd")
    write_pcd(out_pcd, pts, color_accum)


def main():
    print("=" * 70)
    print(" COLORIZING STATIC APRILTAG LIDAR SCANS (LIDOU DATASET)")
    print("=" * 70)

    for s in SESSIONS:
        # Generate swapped (confirmed closest by user)
        colorize_static_session(s, mode="swapped")
        # Also generate standard for session 20260902092818 for side-by-side comparison
        if s == "20260902092818":
            colorize_static_session(s, mode="standard")

    print("\n" + "=" * 70)
    print(" STATIC SCAN COLORIZATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
