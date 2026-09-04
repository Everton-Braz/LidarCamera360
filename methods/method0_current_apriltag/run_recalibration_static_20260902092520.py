#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-calibrate Using Static Session 20260902092520
================================================
1. Extracts 3D retroreflective AprilTag targets (Tag 0, Tag 1, Tag 5).
2. Uses exact sub-pixel detected 2D corners in lens1_front.jpg and lens2_back.jpg.
3. Solves optimal 6-DoF transformation with true physical lever arm:
   c_L = 0.185 * u_L = [0.0006, -0.0934, -0.1597] m.
4. Generates colorized PCD files for CloudCompare visual validation.
"""

import os
import sys
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize

sys.path.insert(0, r"c:\Users\User\Documents\APLICATIVOS\Lidar-Camera-calibrator")
import core

SESSION_DIR = r"C:\Users\User\Downloads\Lidou\20260902092520"
BAG_PATH = os.path.join(SESSION_DIR, "LIDAR_20260902092520.bag")
IMG1_PATH = os.path.join(SESSION_DIR, "extracted_fisheye", "lens1_front.jpg")
IMG2_PATH = os.path.join(SESSION_DIR, "extracted_fisheye", "lens2_back.jpg")


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
    print(f"  [+] Saved Static PCD: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB)")


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


def colorize_with_extrinsics(pts, img1, img2, R1, c_L1, R2, c_L2):
    N = len(pts)
    projector = FisheyeProjector(width=3840, height=3840, fov_deg=196.0)
    color_accum = np.zeros((N, 3), dtype=np.uint8)
    best_cost = np.full(N, 1e9, dtype=np.float32)

    # Lens 1 (Stream 0)
    P_cam1 = (pts - c_L1) @ R1.T
    u1, v1, mask1, dist1, r1 = projector.project(P_cam1)
    cost1 = dist1 + (r1 / 1920.0) * 1.5
    better1 = mask1 & (cost1 < best_cost)
    if np.any(better1):
        u_idx = np.round(u1[better1]).astype(np.int64)
        v_idx = np.round(v1[better1]).astype(np.int64)
        bgr1 = img1[v_idx, u_idx]
        color_accum[better1] = bgr1[:, ::-1]
        best_cost[better1] = cost1[better1]

    # Lens 2 (Stream 1)
    P_cam2 = (pts - c_L2) @ R2.T
    u2, v2, mask2, dist2, r2 = projector.project(P_cam2)
    cost2 = dist2 + (r2 / 1920.0) * 1.5
    better2 = mask2 & (cost2 < best_cost)
    if np.any(better2):
        u_idx = np.round(u2[better2]).astype(np.int64)
        v_idx = np.round(v2[better2]).astype(np.int64)
        bgr2 = img2[v_idx, u_idx]
        color_accum[better2] = bgr2[:, ::-1]
        best_cost[better2] = cost2[better2]

    return color_accum


def main():
    print("=" * 70)
    print(" RE-CALIBRATING STATIC APRILTAG SESSION 20260902092520")
    print("=" * 70)

    # 1. Load LiDAR points
    print("[*] Loading LiDAR point cloud from bag...")
    d_bag = core.read_bag(BAG_PATH)
    pts = d_bag["points"][:, :3].astype(np.float64)
    print(f"    Loaded {len(pts):,} points")

    img1 = cv2.imread(IMG1_PATH)
    img2 = cv2.imread(IMG2_PATH)

    # 2. Gravity and physical mount vector
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    u_L = -imu_a / np.linalg.norm(imu_a)  # UP vector
    c_L_phys = 0.185 * u_L  # Camera center in LiDAR coordinates

    # 3D Tag locations identified from retroreflective targets
    P0_L = np.array([-2.893, 0.129, 0.131])
    P1_L = np.array([0.722, -1.160, 0.129])
    P5_L = np.array([0.638, -1.247, 0.132])

    targets = [
        (1, P0_L, 3758.2, 1659.2),
        (1, P1_L, 1345.0, 2052.0),
        (1, P5_L, 1457.0, 1937.0),
        (2, P0_L, 110.8, 1632.5),
    ]

    # Fisheye params
    fov_rad = np.radians(196.0)
    f = (3840 / 2.0) / (fov_rad / 2.0)
    cx, cy = 1920.0, 1920.0

    def project_single(P_cam):
        X, Y, Z = P_cam[0], P_cam[1], P_cam[2]
        r_xy = np.hypot(X, Y)
        theta = np.arctan2(r_xy, Z)
        r_img = f * theta
        u = cx + r_img * (X / max(r_xy, 1e-7))
        v = cy + r_img * (Y / max(r_xy, 1e-7))
        return u, v

    # Model A: SVD Wahba's Solution
    v0_L = (P0_L - c_L_phys) / np.linalg.norm(P0_L - c_L_phys)
    v1_L = (P1_L - c_L_phys) / np.linalg.norm(P1_L - c_L_phys)
    v5_L = (P5_L - c_L_phys) / np.linalg.norm(P5_L - c_L_phys)

    r0_C = np.array([0.9866, -0.1401, -0.0822])
    r1_C = np.array([-0.4889, 0.1124, 0.8651])
    r5_C = np.array([-0.4007, 0.0147, 0.9161])

    H = np.outer(r0_C, v0_L) + np.outer(r1_C, v1_L) + np.outer(r5_C, v5_L)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(U @ Vt)
    R_svd1 = U @ np.diag([1, 1, d]) @ Vt
    R_svd2 = np.diag([-1, 1, -1]) @ R_svd1

    # Model B: Non-Linear Refinement with Physical Lever Arm
    def cost_func(rv):
        R1 = Rot.from_rotvec(rv).as_matrix()
        R2 = np.diag([-1, 1, -1]) @ R1
        errs = []
        for lens, P_L, u_act, v_act in targets:
            R = R1 if lens == 1 else R2
            P_C = R @ (P_L - c_L_phys)
            u_p, v_p = project_single(P_C)
            errs.append(u_p - u_act)
            errs.append(v_p - v_act)
        return np.mean(np.square(errs))

    rotvec_init = Rot.from_matrix(R_svd1).as_rotvec()
    res = minimize(cost_func, rotvec_init, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-5})
    R_opt1 = Rot.from_rotvec(res.x).as_matrix()
    R_opt2 = np.diag([-1, 1, -1]) @ R_opt1

    # Model C: Swapped Sideways Baseline with True Lever Arm
    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_sideways1 = np.stack([X0, Y0, Z0], axis=0)

    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_sideways2 = np.stack([X1, Y1, Z1], axis=0)

    candidates = [
        ("cand01_svd_optimal", R_svd1, R_svd2, "SVD Wahba Optimal Solution"),
        ("cand02_nonlinear_refined", R_opt1, R_opt2, "Non-Linear Levenberg-Nelder Refinement"),
        ("cand03_sideways_true_lever_arm", R_sideways1, R_sideways2, "Sideways Swapped with True Lever Arm"),
    ]

    for name, R1, R2, desc in candidates:
        print(f"\n[*] Generating {name} ({desc})...")
        colors = colorize_with_extrinsics(pts, img1, img2, R1, c_L_phys, R2, c_L_phys)
        out_pcd = os.path.join(SESSION_DIR, f"{name}.pcd")
        write_pcd(out_pcd, pts, colors)

    # Save JSON summary
    summary = {
        "session": "20260902092520",
        "c_L_physical_m": c_L_phys.tolist(),
        "R_lens1_nonlinear": R_opt1.tolist(),
        "R_lens2_nonlinear": R_opt2.tolist(),
        "R_lens1_svd": R_svd1.tolist(),
        "R_lens2_svd": R_svd2.tolist(),
    }
    with open(os.path.join(SESSION_DIR, "recalibration_static_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" RECALIBRATION COMPLETE!")
    print(f" Files saved in: {SESSION_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
