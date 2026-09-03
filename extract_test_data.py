#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract raw fisheye frames from .insv and inspect LiDAR bags for all 3 test captures.
"""

import os
import subprocess
import cv2
import numpy as np

import core
import apriltag_calib

BASE_DIR = r"C:\Users\User\Downloads\Lidou"
SESSIONS = ["20260902092520", "20260902092658", "20260902092818"]

def extract_session_frames(session_id):
    session_dir = os.path.join(BASE_DIR, session_id)
    out_dir = os.path.join(session_dir, "extracted_fisheye")
    os.makedirs(out_dir, exist_ok=True)
    
    # Find .insv file
    insv_files = [f for f in os.listdir(session_dir) if f.lower().endswith(".insv")]
    if not insv_files:
        print(f"[!] No .insv file in {session_dir}")
        return None
    insv_path = os.path.join(session_dir, insv_files[0])
    
    lens1_path = os.path.join(out_dir, "lens1_front.jpg")
    lens2_path = os.path.join(out_dir, "lens2_back.jpg")
    
    print(f"[*] Extracting frames from {insv_files[0]}...")
    cmd1 = ["ffmpeg", "-ss", "10.0", "-i", insv_path, "-map", "0:v:0", "-vframes", "1", "-q:v", "2", "-y", lens1_path]
    cmd2 = ["ffmpeg", "-ss", "10.0", "-i", insv_path, "-map", "0:v:1", "-vframes", "1", "-q:v", "2", "-y", lens2_path]
    
    subprocess.run(cmd1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    
    print(f"  [+] Extracted Lens 1 (Front): {lens1_path}")
    print(f"  [+] Extracted Lens 2 (Back):  {lens2_path}")
    return lens1_path, lens2_path

def inspect_lidar_bag(session_id):
    session_dir = os.path.join(BASE_DIR, session_id)
    bag_files = [f for f in os.listdir(session_dir) if f.startswith("LIDAR_") and f.endswith(".bag")]
    if not bag_files:
        print(f"[!] No LIDAR bag in {session_dir}")
        return
    bag_path = os.path.join(session_dir, bag_files[0])
    print(f"[*] Reading LiDAR bag: {bag_files[0]}...")
    d = core.read_bag(bag_path)
    pts = d["points"]
    print(f"  [+] LiDAR points: {len(pts)} points, Duration: {d['duration']:.2f} s")
    print(f"  [+] IMU samples: {len(d['imu_a'])}, Gravity up: {np.mean(d['imu_a'], 0).round(3)}")

def check_apriltags(img_path):
    if not os.path.exists(img_path):
        return
    img = cv2.imread(img_path)
    detector = apriltag_calib.AprilTagDetectorFisheye(apriltag_calib.FisheyeCameraModel(width=img.shape[1], height=img.shape[0]))
    detections = detector.detect(img, tag_size_m=0.150)
    print(f"  [+] Detections in {os.path.basename(img_path)}: {list(detections.keys())}")
    for tid, det in detections.items():
        print(f"      Tag #{tid}: center_uv = {det['center_uv'].round(1)}, ray = {det['center_ray'].round(3)}")

if __name__ == "__main__":
    for s in SESSIONS:
        print("\n" + "=" * 60)
        print(f"  PROCESSING TEST SESSION: {s}")
        print("=" * 60)
        frames = extract_session_frames(s)
        if frames:
            check_apriltags(frames[0])
            check_apriltags(frames[1])
        inspect_lidar_bag(s)
