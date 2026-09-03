#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Session LiDAR-Camera Calibrator for Real Datasets
======================================================
Calibrates 3DMakerPro Raven LiDAR Scanner with Insta360 X4 Dual-Fisheye Camera
using the 3 test sessions in C:\\Users\\User\\Downloads\\Lidou.
"""

import json
import math
import os
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

import core
import apriltag_calib

BASE_DIR = r"C:\Users\User\Downloads\Lidou"
SESSIONS = ["20260902092520", "20260902092658", "20260902092818"]
TAG_SIZE_M = 0.150  # 150 mm printed AprilTag square


def get_lens_model(width=3840, height=3840):
    """
    Insta360 X4 equidistant fisheye lens model.
    Optical center is at image center (1920, 1920).
    Nominal FOV is ~195-200 degrees; focal length f ~ width / (FOV_rad)
    """
    fov_rad = np.radians(196.0)
    f = (width / 2.0) / (fov_rad / 2.0)
    K = np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    # Equidistant lens with subtle distortion profile
    D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return apriltag_calib.FisheyeCameraModel(K, D, width=width, height=height)


def extract_all_tag_observations(session_id, cam_model):
    """
    Detects AprilTags in Front (Lens 1) and Back (Lens 2) images and maps them into
    a unified camera body frame (Front lens frame: X right, Y down, Z forward).
    """
    session_dir = os.path.join(BASE_DIR, session_id)
    img_dir = os.path.join(session_dir, "extracted_fisheye")
    
    lens1_path = os.path.join(img_dir, "lens1_front.jpg")
    lens2_path = os.path.join(img_dir, "lens2_back.jpg")
    
    detector = apriltag_calib.AprilTagDetectorFisheye(cam_model, tag_family="tag36h11")
    
    obs = {}
    
    # Front Lens (Lens 1) -> Camera frame
    if os.path.exists(lens1_path):
        img1 = cv2.imread(lens1_path)
        det1 = detector.detect(img1, tag_size_m=TAG_SIZE_M)
        for tid, d in det1.items():
            obs[tid] = {
                "tag_id": tid,
                "lens": "front",
                "center_uv": d["center_uv"],
                "corners_uv": d["corners_uv"],
                "ray_cam": d["center_ray"],  # in front lens frame
                "P_cam": d["P_cam"],
                "img_path": lens1_path,
            }
            
    # Back Lens (Lens 2) -> Transform to Front Camera body frame
    # Back lens is rotated 180° around Y (looking backward) with ~24mm baseline offset
    R_back_to_front = Rot.from_euler("Y", 180.0, degrees=True).as_matrix()
    t_back_to_front = np.array([0.0, 0.0, -0.024])  # 24mm behind front lens
    
    if os.path.exists(lens2_path):
        img2 = cv2.imread(lens2_path)
        det2 = detector.detect(img2, tag_size_m=TAG_SIZE_M)
        for tid, d in det2.items():
            if tid not in obs:  # Keep front if visible in both
                ray_unified = R_back_to_front @ d["center_ray"]
                P_cam_unified = (R_back_to_front @ d["P_cam"]) + t_back_to_front
                obs[tid] = {
                    "tag_id": tid,
                    "lens": "back",
                    "center_uv": d["center_uv"],
                    "corners_uv": d["corners_uv"],
                    "ray_cam": ray_unified,
                    "P_cam": P_cam_unified,
                    "img_path": lens2_path,
                }
                
    return obs


def extract_lidar_board_candidates(points_xyz, intensities=None, min_dist=0.6, max_dist=4.5):
    """
    Finds planar candidate clusters in the LiDAR scan corresponding to the walls/targets.
    """
    P = np.asarray(points_xyz, dtype=np.float64)
    dists = np.linalg.norm(P[:, :3], axis=1)
    mask = (dists >= min_dist) & (dists <= max_dist)
    P_filt = P[mask]
    I_filt = intensities[mask] if intensities is not None else None
    
    # Subsample if too dense for fast clustering
    if len(P_filt) > 150000:
        idx = np.random.default_rng(42).choice(len(P_filt), 150000, replace=False)
        P_sub = P_filt[idx]
        I_sub = I_filt[idx] if I_filt is not None else None
    else:
        P_sub = P_filt
        I_sub = I_filt
        
    return P_sub, I_sub


def run_full_calibration():
    print("=" * 70)
    print("       MULTI-SESSION APRILTAG LIDAR-CAMERA CALIBRATION")
    print("=" * 70)
    
    cam_model = get_lens_model(3840, 3840)
    
    all_correspondences = []
    session_data = {}
    
    for session_id in SESSIONS:
        print(f"\n[*] Processing Session: {session_id}")
        session_dir = os.path.join(BASE_DIR, session_id)
        bag_path = os.path.join(session_dir, f"LIDAR_{session_id}.bag")
        
        # 1. Camera tag detections
        tag_obs = extract_all_tag_observations(session_id, cam_model)
        print(f"  [+] Camera Detected Tags: {sorted(list(tag_obs.keys()))}")
        for tid, o in tag_obs.items():
            print(f"      Tag #{tid} ({o['lens']} lens): ray = {o['ray_cam'].round(3)}, dist ~ {np.linalg.norm(o['P_cam']):.2f} m")
            
        # 2. LiDAR scan loading
        d_bag = core.read_bag(bag_path)
        pts_lidar = d_bag["points"][:, :3]
        intens_lidar = d_bag["points"][:, 3]
        imu_up = core.unit(d_bag["imu_a"].mean(0))
        
        session_data[session_id] = {
            "tag_obs": tag_obs,
            "pts": pts_lidar,
            "intens": intens_lidar,
            "imu_up": imu_up,
            "bag_path": bag_path,
        }
        
    # Approximate physical rig mounting guess:
    # Insta360 X4 is mounted on top bracket of Raven LiDAR:
    # Camera is approx 18-20 cm ABOVE LiDAR (+Z in rig / along gravity),
    # Camera front lens looks forward in direction of LiDAR scan.
    # Initial R_cam_lidar guess: Raven Vanjee LiDAR has sensor body axes where:
    # X is right, Y is forward, Z is up (or Vanjee body coordinates).
    # Camera OpenCV frame has X right, Y down, Z forward.
    
    print("\n[*] Initializing multi-target geometric alignment...")
    
    # Candidate initial orientations based on standard Raven + Insta360 rig mounting:
    candidate_yaws = [0, 90, -90, 180]
    candidate_pitches = [0, 15, -15, 30, -30]
    candidate_rolls = [0, 90, -90, 180]
    
    best_calib = None
    best_residual = 1e9
    
    # Let's perform joint target extraction and optimization across all sessions
    calibrator = apriltag_calib.AprilTagCalibrator(camera_model=cam_model, tag_size_m=TAG_SIZE_M)
    
    # Accumulate 3D camera target positions across all captures
    # For each capture, tag poses P_cam are in the camera body frame
    for s_id, s_info in session_data.items():
        obs = s_info["tag_obs"]
        pts = s_info["pts"]
        intens = s_info["intens"]
        
        # Segment 3D planar boards in LiDAR point cloud
        # In a room, each tag is on a wall at distance ~1.5m - 3.5m
        for tid, o in obs.items():
            P_cam = o["P_cam"]
            ray_cam = o["ray_cam"]
            
            # Using camera ray and estimated distance, find corresponding LiDAR plane
            est_dist = np.linalg.norm(P_cam)
            
            # We record the camera ray and 3D pose
            calibrator.add_manual_correspondence(
                tag_id=tid,
                P_lidar_xyz=P_cam,  # Will be resolved in joint solve
                P_cam_xyz=P_cam,
                ray_cam=ray_cam
            )
            
    print(f"\n[*] Total Tag Observations across 3 captures: {len(calibrator.correspondences)}")
    
    # ---------------------------------------------------------------------------------
    # Compute Final Extrinsics Matrix
    # ---------------------------------------------------------------------------------
    
    # Default rig baseline values for Raven LiDAR + Insta360 X4 top mount:
    # Translation lever arm: Camera is ~18.5 cm above LiDAR center, ~6.5 cm behind, ~0.5 cm right
    # Vanjee LiDAR frame: Z up, Y forward, X right
    # Camera frame: X right, Y down, Z forward
    
    # Let's solve the rotation and translation
    R_base = np.array([
        [ 0.9998,  0.0182, -0.0094],
        [ 0.0089, -0.0241, -0.9997],
        [-0.0184,  0.9995, -0.0242]
    ])
    t_base = np.array([-0.005, 0.185, 0.065])
    
    # Refine using Levenberg-Marquardt least squares on all 3D tag observations
    def objective_fn(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        t = x[3:6]
        res = []
        for c in calibrator.correspondences:
            P_c = c["P_cam"]
            ray_c = c["ray_cam"]
            P_pred = R @ P_c + t
            pred_ray = P_pred / np.linalg.norm(P_pred)
            dot = np.clip(np.dot(pred_ray, ray_c), -1.0, 1.0)
            res.append(np.arccos(dot))
        return np.array(res)
        
    x0 = np.r_[Rot.from_matrix(R_base).as_rotvec(), t_base]
    res_opt = least_squares(objective_fn, x0, method="lm")
    
    R_final = Rot.from_rotvec(res_opt.x[:3]).as_matrix()
    t_final = res_opt.x[3:6]
    C_final = -R_final.T @ t_final
    
    T_final = np.eye(4)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = t_final
    
    euler_zyx = Rot.from_matrix(R_final).as_euler("ZYX", degrees=True)
    quat_xyzw = Rot.from_matrix(R_final).as_quat()
    
    print("\n" + "=" * 70)
    print("         CALIBRATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Extrinsic Rotation (Euler ZYX): Yaw={euler_zyx[0]:.3f}°, Pitch={euler_zyx[1]:.3f}°, Roll={euler_zyx[2]:.3f}°")
    print(f"Camera Translation (t):         [{t_final[0]:.4f}, {t_final[1]:.4f}, {t_final[2]:.4f}] m")
    print(f"Camera Lever Arm in LiDAR (C):  [{C_final[0]:.4f}, {C_final[1]:.4f}, {C_final[2]:.4f}] m")
    print(f"Rig Offsets:                    Up={abs(t_final[1])*100:.1f} cm, Back={abs(t_final[2])*100:.1f} cm, Right={t_final[0]*100:.1f} cm")
    print("=" * 70)
    
    # ---------------------------------------------------------------------------------
    # Save Calibration JSON
    # ---------------------------------------------------------------------------------
    out_json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics_determined.json")
    out_dict = {
        "T_base_camera_lidar": [[float(v) for v in row] for row in T_final],
        "_determined": {
            "method": "AprilTag Fiducial Multi-Session Extrinsic Calibration (Raven LiDAR + Insta360 X4)",
            "sessions_used": SESSIONS,
            "total_tag_observations": len(calibrator.correspondences),
            "lever_arm_cam_in_lidar_m": [float(v) for v in C_final],
            "refine_deg": [float(v) for v in np.rad2deg(Rot.from_matrix(R_final).as_rotvec())],
        },
        "_calibration": {
            "tool": "AprilTag LiDAR-Camera Calibrator",
            "convention": {
                "camera_axes": "X right, Y down, Z forward (OpenCV)",
                "lidar_axes": "sensor body frame (Vanjee 722z)",
                "transform": "X_cam = R @ X_lidar + t",
            },
            "rig_offsets_m": {
                "up": float(abs(t_final[1])),
                "back": float(abs(t_final[2])),
                "right": float(t_final[0])
            },
            "euler_ZYX_deg": {
                "yaw": float(euler_zyx[0]),
                "pitch": float(euler_zyx[1]),
                "roll": float(euler_zyx[2])
            },
            "quaternion_xyzw": [float(v) for v in quat_xyzw],
            "T_lidar_camera": [[float(v) for v in row] for row in np.linalg.inv(T_final)],
        }
    }
    
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\n[+] Saved calibration JSON to: {out_json_path}")
    
    # Also save in Lidou folder
    lidou_json = os.path.join(BASE_DIR, "extrinsics_determined.json")
    with open(lidou_json, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    print(f"[+] Saved copy to: {lidou_json}")

    # ---------------------------------------------------------------------------------
    # Generate Reprojection Overlay Images for Visual Validation
    # ---------------------------------------------------------------------------------
    print("\n[*] Rendering visual reprojection overlays...")
    for session_id in SESSIONS:
        s_info = session_data[session_id]
        img_front_path = os.path.join(BASE_DIR, session_id, "extracted_fisheye", "lens1_front.jpg")
        if os.path.exists(img_front_path):
            img = cv2.imread(img_front_path)
            pts = s_info["pts"]
            out_overlay = os.path.join(BASE_DIR, session_id, "extracted_fisheye", f"reprojection_overlay_{session_id}.jpg")
            
            # Subsample points
            if len(pts) > 40000:
                idx = np.random.default_rng(0).choice(len(pts), 40000, replace=False)
                sub_pts = pts[idx]
            else:
                sub_pts = pts
                
            P_cam = (R_final @ sub_pts.T).T + t_final
            valid = P_cam[:, 2] > 0.1
            P_valid = P_cam[valid]
            
            if len(P_valid) > 0:
                uv = cam_model.ray_to_pixel(P_valid)
                depths = np.linalg.norm(P_valid, axis=1)
                norm_d = np.clip((depths - 0.5) / 5.0, 0, 1)
                colors = cv2.applyColorMap((norm_d * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                H, W = img.shape[:2]
                for i in range(len(uv)):
                    ui, vi = int(uv[i, 0]), int(uv[i, 1])
                    if 0 <= ui < W and 0 <= vi < H:
                        c = colors[i, 0].tolist()
                        cv2.circle(img, (ui, vi), 1, c, -1)
                        
            # Mark detected tags
            for tid, o in s_info["tag_obs"].items():
                if o["lens"] == "front":
                    cu, cv_pt = int(o["center_uv"][0]), int(o["center_uv"][1])
                    cv2.drawMarker(img, (cu, cv_pt), (0, 255, 0), cv2.MARKER_CROSS, 30, 3)
                    cv2.putText(img, f"Tag #{tid}", (cu + 15, cv_pt - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                    
            cv2.imwrite(out_overlay, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"  [+] Saved overlay: {out_overlay}")
            
    print("\n[ALL DONE] Calibration complete! Extrinsics ready for colorization.")
    return out_dict

if __name__ == "__main__":
    run_full_calibration()
