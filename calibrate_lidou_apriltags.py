#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precise AprilTag Extrinsic Calibrator for Raven LiDAR + Insta360 X4
===================================================================
Calibrates 6-DoF transformation matrix T_base_camera_lidar using the 3 static
AprilTag multi-room sessions in C:\\Users\\User\\Downloads\\Lidou.
"""

import os
import sys
import json
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize, least_squares
from scipy.spatial import cKDTree

import core

BASE_DIR = r"C:\Users\User\Downloads\Lidou"
SESSIONS = ["20260902092520", "20260902092658", "20260902092818"]

fov_rad = np.radians(196.0)
f_nominal = (3840 / 2.0) / (fov_rad / 2.0)
cx, cy = 1920.0, 1920.0

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def calibrate_all():
    print("=" * 70)
    print("   OPTIMIZING RAVEN LIDAR -> INSTA360 X4 EXTRINSIC CALIBRATION")
    print("=" * 70)

    # Physical baseline from IMU gravity vector and 90 deg azimuth:
    # In Raven Vanjee 722z:
    # Vanjee LiDAR has Z spin axis tilted forward ~30 deg.
    # IMU gravity points down at [0, 4.93, 8.43].
    # Scanner forward is -X_L (or -Y_L).
    # Camera front lens is looking forward (+Z_C) and up is -Y_C.

    imu_a = np.array([-0.03, 4.9335, 8.4338])
    up_L = -imu_a / np.linalg.norm(imu_a)

    # Forward direction in LiDAR frame:
    # Looking from handle forward
    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, up_L) * up_L
    right_L = right_L / np.linalg.norm(right_L)
    fwd_L = np.cross(up_L, right_L)

    R_base = np.stack([right_L, -up_L, fwd_L], axis=0)
    R_z90 = Rot.from_rotvec(np.radians(90.0) * np.array([0, 1, 0])).as_matrix()
    R_init = R_z90 @ R_base
    t_init = np.array([0.0, -0.18, 0.0])

    print("\n[1/3] Initial Physical Extrinsics:")
    print("  R_init Euler ZYX (deg):", Rot.from_matrix(R_init).as_euler("ZYX", degrees=True).round(2))
    print("  t_init (m):             ", t_init)

    # Validated 3D LiDAR cluster positions and corresponding camera rays:
    pairs = [
        # Session 20260902092818
        {
            "name": "20260902092818_Tag0",
            "P_lidar": np.array([-2.130, 0.007, 0.098]),
            "ray_cam": np.array([0.161, -0.147, 0.976]),
        },
        # Session 20260902092520
        {
            "name": "20260902092520_Tag0",
            "P_lidar": np.array([-2.907, 0.130, 0.132]),
            "ray_cam": np.array([0.987, -0.140, -0.083]),
        },
        {
            "name": "20260902092520_Tag1",
            "P_lidar": np.array([0.723, -1.160, 0.129]),
            "ray_cam": np.array([-0.489, 0.112, 0.865]),
        },
        # Session 20260902092658
        {
            "name": "20260902092658_Tag2",
            "P_lidar": np.array([3.094, 0.038, -0.006]),
            "ray_cam": np.array([0.995, 0.029, -0.099]),
        },
    ]

    print(f"\n[2/3] Solving Non-Linear Extrinsics across {len(pairs)} target constraints...")

    def cost_scalar(params):
        rv = params[:3]
        t = params[3:6]
        R = Rot.from_rotvec(rv).as_matrix()
        total_err = 0.0
        for p in pairs:
            P_l = p["P_lidar"]
            ray_c = p["ray_cam"]
            P_c = R @ P_l + t
            pred_ray = P_c / np.linalg.norm(P_c)
            cos_ang = np.clip(np.dot(pred_ray, ray_c), -1.0, 1.0)
            ang_err = np.arccos(cos_ang)
            total_err += ang_err**2
        return total_err

    x0 = np.r_[Rot.from_matrix(R_init).as_rotvec(), t_init]
    opt_res = minimize(cost_scalar, x0, method="Powell", options={"maxiter": 1000, "ftol": 1e-8})

    R_opt = Rot.from_rotvec(opt_res.x[:3]).as_matrix()
    t_opt = opt_res.x[3:6]
    C_opt = -R_opt.T @ t_opt

    T_final = np.eye(4)
    T_final[:3, :3] = R_opt
    T_final[:3, 3] = t_opt

    euler_zyx = Rot.from_matrix(R_opt).as_euler("ZYX", degrees=True)
    quat_xyzw = Rot.from_matrix(R_opt).as_quat()

    mean_err_deg = np.degrees(np.sqrt(opt_res.fun / len(pairs)))
    print("\n[3/3] Calibration Results:")
    print(f"  Residual RMS Angular Error: {mean_err_deg:.3f} deg")
    print(f"  Euler ZYX (deg):             Yaw={euler_zyx[0]:.3f}°, Pitch={euler_zyx[1]:.3f}°, Roll={euler_zyx[2]:.3f}°")
    print(f"  Camera Translation (t):      [{t_opt[0]:.4f}, {t_opt[1]:.4f}, {t_opt[2]:.4f}] m")
    print(f"  Camera Lever Arm (C):        [{C_opt[0]:.4f}, {C_opt[1]:.4f}, {C_opt[2]:.4f}] m")
    print(f"  T_base_camera_lidar Matrix:\n{T_final.round(5)}")

    out_dict = {
        "T_base_camera_lidar": [[float(v) for v in row] for row in T_final],
        "_determined": {
            "method": "AprilTag + Retroreflective Multi-Session Non-Linear Optimization",
            "residual_rms_deg": float(mean_err_deg),
            "euler_ZYX_deg": [float(v) for v in euler_zyx],
            "translation_m": [float(v) for v in t_opt],
            "lever_arm_cam_in_lidar_m": [float(v) for v in C_opt],
        },
        "_calibration": {
            "camera_axes": "X right, Y down, Z forward (OpenCV)",
            "lidar_axes": "sensor body frame (Vanjee 722z)",
            "transform": "X_cam = R @ X_lidar + t",
            "quaternion_xyzw": [float(v) for v in quat_xyzw],
            "T_lidar_camera": [[float(v) for v in row] for row in np.linalg.inv(T_final)],
        },
    }

    out_p1 = r"C:\Users\User\Documents\APLICATIVOS\Lidar-Camera-calibrator\extrinsics_determined.json"
    out_p2 = r"C:\Users\User\Downloads\Lidou\extrinsics_determined.json"

    with open(out_p1, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    with open(out_p2, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\n[+] Saved updated calibration to: {out_p1}")
    return T_final


if __name__ == "__main__":
    calibrate_all()
