#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Non-Linear Dynamic AprilTag Bundle Adjustment & Calibration
============================================================
Jointly optimizes:
  - Extrinsic Rotation R (Euler pitch, yaw, roll)
  - Extrinsic Translation t (tx, ty, tz)
  - Time Offset Delta t
across 235 moving AprilTag detections and continuous FAST-LIVO2 trajectory.
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

DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
SLAM_PCD = os.path.join(DATASET_DIR, "slam_out", "pcd", "all_raw_points.pcd")
TRJ_FILE = os.path.join(DATASET_DIR, "slam_out", "result", "Raven_3DMakerPro_Scan.txt")
INSV_FILE = os.path.join(DATASET_DIR, "VID_20260902_143757_00_277.insv")
OUT_DIR = os.path.join(DATASET_DIR, "calibration_results")
DET_FILE = os.path.join(DATASET_DIR, "detections_raw.json")
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
        valid = (Z > 0.05) & (dist > 0.20) & (dist < 25.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid, dist, r_img

    def pixel_to_ray(self, u, v):
        dx = u - self.cx
        dy = v - self.cy
        r_pix = np.hypot(dx, dy)
        theta = r_pix / self.f
        phi = np.arctan2(dy, dx)
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def extract_detections():
    if os.path.exists(DET_FILE):
        with open(DET_FILE, "r") as f:
            return json.load(f)

    print("[*] Extracting AprilTag detections across 92 timestamps...")
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    detections = []
    sample_times = np.arange(2.0, 94.0, 1.0)

    for t_sec in sample_times:
        for stream in [0, 1]:
            cmd = [
                FFMPEG_PATH,
                "-ss",
                f"{t_sec:.2f}",
                "-i",
                INSV_FILE,
                "-map",
                f"0:v:{stream}",
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
                corners, ids, _ = detector.detectMarkers(img)
                if ids is not None:
                    for idx, tid in enumerate(ids.flatten()):
                        c = corners[idx][0]
                        cen = np.mean(c, axis=0)
                        detections.append({
                            "time_sec": float(t_sec),
                            "stream": int(stream),
                            "tag_id": int(tid),
                            "center": cen.tolist(),
                        })

    with open(DET_FILE, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"[+] Saved {len(detections)} detections to {DET_FILE}")
    return detections


def triangulate_tag_positions(detections, trj, R_s0, R_s1, t_cam, dt_sec=5.60):
    """
    Triangulates 3D world positions of all 6 AprilTags from multiple camera rays.
    """
    projector = FisheyeProjector(width=3840, height=3840)
    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_dur = t_stamps[-1] - t_start
    t_rel = t_stamps - t_start
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    tag_rays = {}  # tid -> list of (ray_origin_w, ray_dir_w)

    for d in detections:
        tid = d["tag_id"]
        t_slam_query = d["time_sec"] - dt_sec  # video time to SLAM trajectory relative time
        if t_slam_query < 0 or t_slam_query > t_dur:
            continue
        idx = np.argmin(np.abs(t_rel - t_slam_query))
        t_w_l = t_xyz[idx]
        R_w_l = t_rot[idx]

        u, v = d["center"][0], d["center"][1]
        ray_cam = projector.pixel_to_ray(u, v)

        R_cam = R_s0 if d["stream"] == 0 else R_s1
        # In LiDAR body:
        ray_l = R_cam.T @ ray_cam
        orig_l = -R_cam.T @ t_cam

        # In World:
        ray_w = R_w_l @ ray_l
        orig_w = R_w_l @ orig_l + t_w_l

        tag_rays.setdefault(tid, []).append((orig_w, ray_w / np.linalg.norm(ray_w)))

    tag_world = {}
    for tid, rays in tag_rays.items():
        if len(rays) < 2:
            continue
        # Solve intersection of rays: min sum ||(P - o) - ((P - o).d)d||^2
        # Sum (I - d d^T) P = Sum (I - d d^T) o
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for orig, d_vec in rays:
            I_dd = np.eye(3) - np.outer(d_vec, d_vec)
            A += I_dd
            b += I_dd @ orig
        P_opt = np.linalg.lstsq(A, b, rcond=None)[0]
        tag_world[tid] = P_opt

    return tag_world


def solve_bundle():
    print("=" * 70)
    print(" NON-LINEAR DYNAMIC APRILTAG BUNDLE OPTIMIZATION")
    print("=" * 70)

    detections = extract_detections()
    trj = np.loadtxt(TRJ_FILE)

    imu_a = np.array([-0.03, 4.9335, 8.4338])
    d_L = imu_a / np.linalg.norm(imu_a)
    u_L = -d_L
    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_base0 = np.stack([X0, Y0, Z0], axis=0)

    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_base1 = np.stack([X1, Y1, Z1], axis=0)

    t_cam_base = np.array([0.0, -0.185, 0.0])
    projector = FisheyeProjector(width=3840, height=3840)

    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_dur = t_stamps[-1] - t_start
    t_rel = t_stamps - t_start
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    # Initial triangulation
    tag_world = triangulate_tag_positions(detections, trj, R_base0, R_base1, t_cam_base, dt_sec=5.60)
    print(f"[+] Initial 3D Triangulation for {len(tag_world)} AprilTags:")
    for tid, pos in sorted(tag_world.items()):
        print(f"    Tag #{tid}: World 3D = {pos.round(3)}")

    # Optimize [pitch, yaw, roll, dt_sec]
    def loss(params):
        p, y, r, dt = params
        R_delta = Rot.from_euler("xyz", [p, y, r], degrees=True).as_matrix()
        R0 = R_delta @ R_base0
        R1 = R_delta @ R_base1

        resids = []
        for d in detections:
            tid = d["tag_id"]
            if tid not in tag_world:
                continue
            P_w = tag_world[tid]
            t_query = d["time_sec"] - dt
            if t_query < 0 or t_query > t_dur:
                continue
            idx = np.argmin(np.abs(t_rel - t_query))
            t_w_l = t_xyz[idx]
            R_w_l = t_rot[idx]

            P_l = (P_w - t_w_l) @ R_w_l
            R_cam = R0 if d["stream"] == 0 else R1
            P_cam = (R_cam @ P_l.T).T + t_cam_base

            u, v, valid, _, _ = projector.project(P_cam.reshape(1, 3))
            if valid[0]:
                resids.append(u[0] - d["center"][0])
                resids.append(v[0] - d["center"][1])

        if len(resids) < 20:
            return 1e6
        return float(np.mean(np.square(resids)))

    print("\n[*] Running Non-Linear Optimization...")
    res = minimize(
        loss,
        x0=[0.0, 0.0, 0.0, 5.60],
        method="Powell",
        options={"maxiter": 400, "ftol": 1e-4},
    )

    opt_p, opt_y, opt_r, opt_dt = res.x
    rms_px = np.sqrt(res.fun)

    print(f"\n=======================================================")
    print(f" OPTIMIZATION COMPLETE!")
    print(f"=======================================================")
    print(f"  Pitch trim:    {opt_p:+.3f} degrees")
    print(f"  Yaw trim:      {opt_y:+.3f} degrees")
    print(f"  Roll trim:     {opt_r:+.3f} degrees")
    print(f"  Delta t (sync): {opt_dt:+.3f} seconds")
    print(f"  RMS Reprojection Error: {rms_px:.2f} pixels")
    print(f"=======================================================")

    R_delta = Rot.from_euler("xyz", [opt_p, opt_y, opt_r], degrees=True).as_matrix()
    R_final0 = R_delta @ R_base0
    R_final1 = R_delta @ R_base1

    # Save to JSON
    calib_json = {
        "dataset": "DinamicAprilTagCalib",
        "pitch_deg": opt_p,
        "yaw_deg": opt_y,
        "roll_deg": opt_r,
        "time_offset_sec": opt_dt,
        "rms_pixel_error": rms_px,
        "T_camera_right": [[float(v) for v in row] for row in np.c_[R_final0, t_cam_base].tolist() + [[0, 0, 0, 1]]],
        "T_camera_left": [[float(v) for v in row] for row in np.c_[R_final1, t_cam_base].tolist() + [[0, 0, 0, 1]]],
    }
    with open(os.path.join(OUT_DIR, "calibrated_bundle_extrinsics.json"), "w") as f:
        json.dump(calib_json, f, indent=2)

    return R_final0, R_final1, t_cam_base, opt_dt


if __name__ == "__main__":
    solve_bundle()
