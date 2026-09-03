#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-Fisheye LiDAR Point Cloud Colorizer
========================================
Colorizes 3D point clouds using raw dual-fisheye frames (Front & Back lens)
and calibrated extrinsics without stitching artifacts.
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import core
import apriltag_calib
import colorize


def get_lens_model(width=3840, height=3840):
    fov_rad = np.radians(196.0)
    f = (width / 2.0) / (fov_rad / 2.0)
    K = np.array([
        [f, 0.0, width / 2.0],
        [0.0, f, height / 2.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    D = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return apriltag_calib.FisheyeCameraModel(K, D, width=width, height=height)


def colorize_dual_fisheye_bag(
    bag_path,
    front_img_path,
    back_img_path,
    calib_json_path,
    out_ply_path,
    min_dist=0.3,
    max_dist=30.0,
    log=print,
):
    log(f"[*] Loading calibration from: {calib_json_path}")
    with open(calib_json_path, "r", encoding="utf-8") as f:
        calib = json.load(f)
        
    T = np.array(calib["T_base_camera_lidar"], dtype=np.float64)
    R_front = T[:3, :3]
    t_front = T[:3, 3]
    
    # Back lens relative pose to Front lens
    R_back_rel = Rot.from_euler("Y", 180.0, degrees=True).as_matrix()
    t_back_rel = np.array([0.0, 0.0, -0.024])
    
    R_back = R_back_rel @ R_front
    t_back = (R_back_rel @ t_front) + t_back_rel
    
    log(f"[*] Reading LiDAR bag: {bag_path}")
    d_bag = core.read_bag(bag_path)
    pts = d_bag["points"][:, :3].astype(np.float64)
    refl = d_bag["points"][:, 3].astype(np.float32) if d_bag["points"].shape[1] > 3 else None
    
    log(f"    Loaded {len(pts):,} points")
    
    log(f"[*] Loading raw fisheye images:")
    log(f"    Front lens: {front_img_path}")
    log(f"    Back lens:  {back_img_path}")
    
    img_front = cv2.imread(front_img_path)
    img_back = cv2.imread(back_img_path)
    
    H_f, W_f = img_front.shape[:2]
    H_b, W_b = img_back.shape[:2]
    
    cam_front = get_lens_model(W_f, H_f)
    cam_back = get_lens_model(W_b, H_b)
    
    # Transform points to Front Camera frame
    P_front = (R_front @ pts.T).T + t_front
    # Transform points to Back Camera frame
    P_back = (R_back @ pts.T).T + t_back
    
    # Points in front hemisphere (Z > 0) vs back hemisphere (Z > 0 in back frame)
    # Project each point onto the lens facing it
    colors_rgb = np.full((len(pts), 3), 128, dtype=np.uint8)
    
    # 1. Front lens projection
    front_mask = P_front[:, 2] > 0.05
    if np.any(front_mask):
        P_f_sub = P_front[front_mask]
        uv_f = cam_front.ray_to_pixel(P_f_sub)
        u_f = np.clip(np.round(uv_f[:, 0]).astype(np.int64), 0, W_f - 1)
        v_f = np.clip(np.round(uv_f[:, 1]).astype(np.int64), 0, H_f - 1)
        
        # Check radial distance from center (within fisheye circle FOV)
        r_f = np.hypot(u_f - W_f / 2.0, v_f - H_f / 2.0)
        valid_circle = r_f < (min(W_f, H_f) / 2.0) * 0.98
        
        bgr_f = img_front[v_f, u_f]
        front_indices = np.nonzero(front_mask)[0]
        colors_rgb[front_indices[valid_circle]] = bgr_f[valid_circle, ::-1]
        
    # 2. Back lens projection
    back_mask = P_back[:, 2] > 0.05
    if np.any(back_mask):
        P_b_sub = P_back[back_mask]
        uv_b = cam_back.ray_to_pixel(P_b_sub)
        u_b = np.clip(np.round(uv_b[:, 0]).astype(np.int64), 0, W_b - 1)
        v_b = np.clip(np.round(uv_b[:, 1]).astype(np.int64), 0, H_b - 1)
        
        r_b = np.hypot(u_b - W_b / 2.0, v_b - H_b / 2.0)
        valid_circle_b = r_b < (min(W_b, H_b) / 2.0) * 0.98
        
        bgr_b = img_back[v_b, u_b]
        back_indices = np.nonzero(back_mask)[0]
        
        # Only overwrite if front wasn't closer / front didn't claim
        colors_rgb[back_indices[valid_circle_b]] = bgr_b[valid_circle_b, ::-1]
        
    # Distance filter
    dists = np.linalg.norm(pts, axis=1)
    valid_pts = (dists >= min_dist) & (dists <= max_dist) & np.isfinite(dists)
    
    pts_out = pts[valid_pts]
    colors_out = colors_rgb[valid_pts]
    refl_out = refl[valid_pts] if refl is not None else None
    
    log(f"[*] Exporting colorized point clouds ({len(pts_out):,} points)...")
    
    # Save PLY
    colorize.write_ply(out_ply_path, pts_out, colors_out, refl_out)
    log(f"  [+] Saved colorized PLY: {out_ply_path} ({os.path.getsize(out_ply_path)/1e6:.1f} MB)")
    
    # Also save PCD
    out_pcd_path = os.path.splitext(out_ply_path)[0] + ".pcd"
    colorize.write_pcd(out_pcd_path, pts_out, colors_out, refl_out)
    log(f"  [+] Saved colorized PCD: {out_pcd_path} ({os.path.getsize(out_pcd_path)/1e6:.1f} MB)")
    
    return out_ply_path, out_pcd_path


def main():
    base_dir = r"C:\Users\User\Downloads\Lidou"
    sessions = ["20260902092520", "20260902092658", "20260902092818"]
    calib_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extrinsics_determined.json")
    
    print("=" * 70)
    print("       COLORIZING POINT CLOUDS FOR ALL 3 SESSIONS")
    print("=" * 70)
    
    results = []
    for s in sessions:
        print(f"\n[>>>] Processing Session: {s}")
        bag_file = os.path.join(base_dir, s, f"LIDAR_{s}.bag")
        img_front = os.path.join(base_dir, s, "extracted_fisheye", "lens1_front.jpg")
        img_back = os.path.join(base_dir, s, "extracted_fisheye", "lens2_back.jpg")
        
        out_ply = os.path.join(base_dir, s, f"colorized_cloud_{s}.ply")
        
        res = colorize_dual_fisheye_bag(
            bag_path=bag_file,
            front_img_path=img_front,
            back_img_path=img_back,
            calib_json_path=calib_json,
            out_ply_path=out_ply,
            min_dist=0.3,
            max_dist=20.0
        )
        results.append(res)
        
    print("\n" + "=" * 70)
    print("       ALL POINT CLOUDS COLORIZED SUCCESSFULLY!")
    print("=" * 70)
    for r in results:
        print(f"  - PLY: {r[0]}")
        print(f"  - PCD: {r[1]}")
    print("=" * 70)

if __name__ == "__main__":
    main()
