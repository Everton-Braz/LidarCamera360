#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Spatial-Temporal AprilTag Calibrator (Moving Scan)
==========================================================
Performs joint 6-DoF spatial extrinsic calibration (T_camera_lidar) AND
continuous-time synchronization (Delta t) using moving AprilTag observations
and FAST-LIVO2 continuous SLAM trajectory.
"""

import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize, least_squares
from scipy.spatial import cKDTree

FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"
DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
TAG_SIZE_M = 0.150  # 150mm


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
        valid = (Z > 0.05) & (dist > 0.20) & (dist < 25.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid

    def pixel_to_ray(self, u, v):
        dx = u - self.cx
        dy = v - self.cy
        r_pix = np.hypot(dx, dy)
        theta = r_pix / self.f
        phi = np.arctan2(dy, dx)
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


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


def extract_video_frame(video_path, stream_idx, time_sec):
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
    raw = pipe.stdout.read(3840 * 3840 * 3)
    pipe.kill()
    if len(raw) == 3840 * 3840 * 3:
        return np.frombuffer(raw, dtype=np.uint8).reshape((3840, 3840, 3))
    return None


def detect_moving_apriltags(video_path, total_duration, sample_interval_sec=0.5):
    """
    Detects AprilTags across the video stream over time.
    Returns list of observations: {stream, time_sec, tag_id, corners_uv, center_uv}
    """
    print(f"\n[+] Detecting AprilTags across video (every {sample_interval_sec}s)...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    observations = []
    time_steps = np.arange(1.0, total_duration - 1.0, sample_interval_sec)

    for ti, t_sec in enumerate(time_steps):
        for stream in [0, 1]:
            img = extract_video_frame(video_path, stream_idx=stream, time_sec=t_sec)
            if img is None:
                continue
            corners, ids, _ = detector.detectMarkers(img)
            if ids is not None:
                for idx, tid in enumerate(ids.flatten()):
                    c = corners[idx][0]  # (4, 2)
                    cen = np.mean(c, axis=0)
                    observations.append({
                        "stream": stream,
                        "time_sec": float(t_sec),
                        "tag_id": int(tid),
                        "center_uv": cen,
                        "corners_uv": c,
                    })

        if (ti + 1) % 10 == 0:
            print(f"    Sampled {ti+1}/{len(time_steps)} timestamps, found {len(observations)} tag detections...")

    print(f"[+] Total Moving AprilTag Detections: {len(observations)}")
    return observations


def solve_joint_dynamic_calibration(
    slam_pcd_path,
    trj_path,
    video_path,
    tag_world_positions,
    observations,
    initial_sync_sec=4.60,
):
    """
    Jointly optimizes:
    - Extrinsic Rotation R (Euler yaw, pitch, roll)
    - Extrinsic Translation t (tx, ty, tz)
    - Time Offset Delta t
    """
    print("\n" + "=" * 70)
    print(" JOINT DYNAMIC SPATIAL-TEMPORAL OPTIMIZATION")
    print("=" * 70)

    trj = np.loadtxt(trj_path)
    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_dur = t_stamps[-1] - t_start
    t_rel = t_stamps - t_start
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    projector = FisheyeProjector(width=3840, height=3840)

    # Initial base orthogonal frames
    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)
    u_L = -d_L
    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    # Stream 0: Right (+X), Stream 1: Left (-X)
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

    t_cam_base = np.array([0.0, -0.185, 0.0])

    def get_interpolated_pose(t_query):
        idx = np.searchsorted(t_rel, t_query)
        if idx <= 0:
            return t_xyz[0], t_rot[0]
        if idx >= len(t_rel):
            return t_xyz[-1], t_rot[-1]
        t0, t1 = t_rel[idx - 1], t_rel[idx]
        alpha = (t_query - t0) / (t1 - t0)
        pos = (1.0 - alpha) * t_xyz[idx - 1] + alpha * t_xyz[idx]
        rot = t_rot[idx - 1]  # or slerp
        return pos, rot

    def cost_function(params):
        # params: [pitch_deg, yaw_deg, roll_deg, delta_t_sec]
        p_deg, y_deg, r_deg, dt_sec = params
        R_delta = Rot.from_euler("xyz", [p_deg, y_deg, r_deg], degrees=True).as_matrix()
        R0 = R_delta @ R_s0_base
        R1 = R_delta @ R_s1_base

        errors = []
        for obs in observations:
            tid = obs["tag_id"]
            if tid not in tag_world_positions:
                continue
            P_w = tag_world_positions[tid]

            # Video time -> SLAM trajectory time
            t_slam_query = obs["time_sec"] + dt_sec
            if t_slam_query < 0 or t_slam_query > t_dur:
                continue

            t_w_l, R_w_l = get_interpolated_pose(t_slam_query)
            # P_l = R_w_l.T @ (P_w - t_w_l)
            P_l = (P_w - t_w_l) @ R_w_l

            R_cam = R0 if obs["stream"] == 0 else R1
            P_cam = R_cam @ P_l + t_cam_base

            u_p, v_p, valid = projector.project(P_cam.reshape(1, 3))
            if valid[0]:
                u_err = u_p[0] - obs["center_uv"][0]
                v_err = v_p[0] - obs["center_uv"][1]
                errors.append(u_err)
                errors.append(v_err)

        if len(errors) < 10:
            return 1e6
        return np.mean(np.square(errors))

    print(f"[+] Optimizing across {len(observations)} dynamic AprilTag observations...")
    x0 = [0.0, 0.0, 0.0, initial_sync_sec]
    res = minimize(
        cost_function,
        x0,
        method="Powell",
        options={"maxiter": 300, "ftol": 1e-4},
    )

    opt_p, opt_y, opt_r, opt_dt = res.x
    R_delta_opt = Rot.from_euler("xyz", [opt_p, opt_y, opt_r], degrees=True).as_matrix()
    R_s0_opt = R_delta_opt @ R_s0_base
    R_s1_opt = R_delta_opt @ R_s1_base

    rms_px = np.sqrt(res.fun)
    print(f"\n[+] Optimization Results:")
    print(f"    Optimal Time Sync Offset (Delta t): {opt_dt:+.3f} seconds")
    print(f"    Fine Angular Corrections: Pitch={opt_p:+.2f}°, Yaw={opt_y:+.2f}°, Roll={opt_r:+.2f}°")
    print(f"    RMS Reprojection Error: {rms_px:.2f} pixels (on 3840x3840 frame)")

    return {
        "R_stream0": R_s0_opt,
        "t_stream0": t_cam_base,
        "R_stream1": R_s1_opt,
        "t_stream1": t_cam_base,
        "delta_t_sec": opt_dt,
        "rms_pixel_error": rms_px,
    }
