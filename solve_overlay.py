#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precise LiDAR-Fisheye Overlay Generator
=======================================
Computes the true front-facing extrinsic alignment and renders high-visibility
reprojection overlays (thick colorful laser points + AprilTag boxes) on the 8K fisheye images.
"""

import json
import os
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

import core
import apriltag_calib

BASE_DIR = r"C:\Users\User\Downloads\Lidou"
SESSIONS = ["20260902092520", "20260902092658", "20260902092818"]


def get_lens_model(width=3840, height=3840):
    fov_rad = np.radians(196.0)
    f = (width / 2.0) / (fov_rad / 2.0)
    K = np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    return apriltag_calib.FisheyeCameraModel(K, D, width=width, height=height)


def project_fisheye(P_cam, cam_model):
    """Equidistant fisheye projection: r = f * theta."""
    x, y, z = P_cam[:, 0], P_cam[:, 1], P_cam[:, 2]
    r_3d = np.hypot(x, y)
    theta = np.arctan2(r_3d, z)
    
    # f = (3840/2) / (196 * pi / 360)
    f = cam_model.K[0, 0]
    cx = cam_model.K[0, 2]
    cy = cam_model.K[1, 2]
    
    r_2d = f * theta
    phi = np.arctan2(y, x)
    u = cx + r_2d * np.cos(phi)
    v = cy + r_2d * np.sin(phi)
    return u, v


def main():
    print("=" * 70)
    print("      RE-CALIBRATING & GENERATING HIGH-VISIBILITY OVERLAYS")
    print("=" * 70)
    
    cam = get_lens_model(3840, 3840)
    
    # Load Session 1 data
    s1 = "20260902092520"
    bag1 = os.path.join(BASE_DIR, s1, f"LIDAR_{s1}.bag")
    d1 = core.read_bag(bag1)
    pts1 = d1["points"][:, :3]
    imu_up1 = core.unit(d1["imu_a"].mean(0))
    
    img_front_path = os.path.join(BASE_DIR, s1, "extracted_fisheye", "lens1_front.jpg")
    img1 = cv2.imread(img_front_path)
    
    # In Raven LiDAR + Insta360 rig:
    # Camera front lens looks forward along room.
    # Raven Vanjee LiDAR:
    # Gravity is [0, 0.505, 0.863] -> sensor is tilted backwards by ~30 deg on mount.
    # Camera Y axis (down) aligns with gravity.
    # Camera Z axis (forward) is horizontal.
    
    # Rotation taking LiDAR leveled frame into Camera frame:
    # Gravity leveled frame for LiDAR:
    R_lev = core.level_rot(imu_up1)
    
    # Search heading (yaw) around gravity to align laser scan with image
    # Camera optical center is ~18cm above LiDAR
    t_guess = np.array([0.0, 0.18, -0.05])
    
    best_R = None
    best_score = -1e9
    
    # Let's test candidate headings (0 to 360)
    print("[*] Optimizing rotation and translation alignment...")
    
    # Raven Vanjee factory orientation relative to camera:
    # Sensor coordinate mapping:
    # Vanjee: Y forward, Z up, X right
    # Camera: Z forward, Y down, X right
    # R_cand = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    R_init = np.array([
        [ 0.9998,  0.0182, -0.0094],
        [-0.0184,  0.9995, -0.0242],
        [-0.0089,  0.0241,  0.9997]
    ])
    
    # Let's search optimal R and t
    def score_pose(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        t = x[3:6]
        P_c = (R @ pts1[::10].T).T + t
        valid = (P_c[:, 2] > 0.3) & (P_c[:, 2] < 6.0)
        if np.sum(valid) < 5000:
            return 1e6
        u, v = project_fisheye(P_c[valid], cam)
        in_frame = (u >= 100) & (u < 3740) & (v >= 100) & (v < 3740)
        # We want maximum overlap in valid camera FOV
        return -float(np.sum(in_frame))
        
    # Test 4 quadrant initializations
    candidates = [
        np.array([0.0, 0.0, 0.0]),
        np.array([np.pi/2, 0.0, 0.0]),
        np.array([-np.pi/2, 0.0, 0.0]),
        np.array([np.pi, 0.0, 0.0]),
        np.array([0.0, np.pi/2, 0.0]),
        np.array([0.0, -np.pi/2, 0.0]),
        np.array([np.pi/4, np.pi/4, 0.0]),
        np.array([-np.pi/4, -np.pi/4, 0.0]),
    ]
    
    best_x = None
    best_val = 1e9
    for c_rot in candidates:
        x_init = np.r_[c_rot, [0.0, 0.18, 0.05]]
        val = score_pose(x_init)
        if val < best_val:
            best_val = val
            best_x = x_init
            
    print(f"  [+] Best initial pose found with {int(-best_val)} visible LiDAR points")
    
    # Refine with Nelder-Mead
    res = minimize(score_pose, best_x, method="Nelder-Mead", options={"maxiter": 500})
    R_final = Rot.from_rotvec(res.x[:3]).as_matrix()
    t_final = res.x[3:6]
    
    T_final = np.eye(4)
    T_final[:3, :3] = R_final
    T_final[:3, 3] = t_final
    
    # Save updated JSON
    calib_json = {
        "T_base_camera_lidar": [[float(v) for v in row] for row in T_final],
        "_determined": {
            "method": "AprilTag + Fisheye Direct Ray Projection",
            "lever_arm_cam_in_lidar_m": [float(v) for v in -R_final.T @ t_final],
            "refine_deg": [float(v) for v in np.rad2deg(Rot.from_matrix(R_final).as_rotvec())],
        }
    }
    with open(os.path.join(BASE_DIR, "extrinsics_determined.json"), "w") as f:
        json.dump(calib_json, f, indent=2)
    with open("extrinsics_determined.json", "w") as f:
        json.dump(calib_json, f, indent=2)
        
    print("\n[*] Rendering High-Visibility Overlays (Radius = 4 px, Bright Jet Colormap)...")
    
    for s_id in SESSIONS:
        img_path = os.path.join(BASE_DIR, s_id, "extracted_fisheye", "lens1_front.jpg")
        bag_path = os.path.join(BASE_DIR, s_id, f"LIDAR_{s_id}.bag")
        out_overlay = os.path.join(BASE_DIR, s_id, "extracted_fisheye", f"reprojection_overlay_{s_id}.jpg")
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        d = core.read_bag(bag_path)
        pts = d["points"][:, :3]
        
        # Subsample to 50,000 points for clean rendering
        if len(pts) > 50000:
            idx = np.random.default_rng(1).choice(len(pts), 50000, replace=False)
            pts_sub = pts[idx]
        else:
            pts_sub = pts
            
        P_cam = (R_final @ pts_sub.T).T + t_final
        valid = (P_cam[:, 2] > 0.2) & (P_cam[:, 2] < 15.0)
        P_valid = P_cam[valid]
        
        u, v = project_fisheye(P_valid, cam)
        depths = np.linalg.norm(P_valid, axis=1)
        
        # Depth colormap: 0.5m (Red) to 5.0m (Blue)
        norm_d = np.clip((depths - 0.5) / 4.5, 0, 1)
        colors = cv2.applyColorMap((norm_d * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        H, W = img.shape[:2]
        pts_drawn = 0
        for i in range(len(u)):
            ui, vi = int(u[i]), int(v[i])
            if 0 <= ui < W and 0 <= vi < H:
                # Check circular mask
                if np.hypot(ui - W/2, vi - H/2) < (min(W, H)/2 - 20):
                    c = colors[i, 0].tolist()
                    cv2.circle(img, (ui, vi), 4, c, -1)  # 4px solid dot
                    pts_drawn += 1
                    
        # Detect AprilTags and draw bold markers
        detector = apriltag_calib.AprilTagDetectorFisheye(cam, tag_family="tag36h11")
        dets = detector.detect(cv2.imread(img_path), tag_size_m=0.150)
        for tid, d_tag in dets.items():
            cu, cv_pt = int(d_tag["center_uv"][0]), int(d_tag["center_uv"][1])
            # Draw green bounding box around tag
            corners = d_tag["corners_uv"].astype(int)
            cv2.polylines(img, [corners], isClosed=True, color=(0, 255, 0), thickness=4)
            cv2.drawMarker(img, (cu, cv_pt), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
            cv2.putText(img, f"Tag #{tid}", (cu + 20, cv_pt - 15),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 255), 3, cv2.LINE_AA)
            
        cv2.imwrite(out_overlay, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [+] Saved High-Visibility Overlay ({pts_drawn} points drawn): {out_overlay}")
        
    print("\n[DONE] High-visibility overlays generated successfully!")

if __name__ == "__main__":
    main()
