#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run FAST-LIVO2 SLAM on the Dynamic AprilTag Dataset
===================================================
1. Merges LIDAR_DinamicAprilTagCalib.bag + IMAGE_DinamicAprilTagCalib.bag
2. Executes FAST-LIVO2 direct LiDAR-inertial-visual SLAM in WSL2
3. Saves the registered 3D SLAM map (all_raw_points.pcd) and 6-DoF trajectory (Raven_3DMakerPro_Scan.txt)
"""

import os
import sys
import time
import subprocess
import shutil

WSL_DISTRO = "Ubuntu-20.04"
DATASET_WIN = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
DATASET_WSL = "/mnt/c/Users/User/Downloads/Lidou/DinamicAprilTagCalib"
REPO_WIN = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION"
REPO_WSL = "/mnt/c/Users/User/Documents/APLICATIVOS/RAVEN-SCAN-INSTA360-COLORIZATION"


def run_wsl(cmd_str, desc="WSL command"):
    print(f"\n[*] {desc}")
    full_cmd = ["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "bash", "-c", cmd_str]
    t0 = time.time()
    subprocess.run(full_cmd, check=True)
    print(f"[+] {desc} finished in {time.time() - t0:.1f}s")


def merge_bags():
    print("=" * 70)
    print(" STEP 1: MERGING LIDAR + FRONT CAMERA BAGS")
    print("=" * 70)

    merge_py = f"""
import rosbag
import os

lidar_bag = '{DATASET_WSL}/LIDAR_DinamicAprilTagCalib.bag'
img_bag = '{DATASET_WSL}/IMAGE_DinamicAprilTagCalib.bag'
out_bag = '{DATASET_WSL}/dynamic_apriltag_merged.bag'

print(f'Merging: {{lidar_bag}} + {{img_bag}} -> {{out_bag}}')

with rosbag.Bag(out_bag, 'w', compression='lz4') as out:
    count = 0
    with rosbag.Bag(lidar_bag, 'r') as b1:
        for topic, msg, t in b1.read_messages():
            out.write(topic, msg, t)
            count += 1
            if count % 5000 == 0:
                print(f'  LiDAR messages written: {{count}}...')
                
    with rosbag.Bag(img_bag, 'r') as b2:
        for topic, msg, t in b2.read_messages():
            out.write('/camera_front/image/compressed', msg, t)
            count += 1

print(f'Done! Total merged messages: {{count}}')
"""
    tmp_py = os.path.join(DATASET_WIN, "merge_wsl.py")
    with open(tmp_py, "w", newline="\n", encoding="utf-8") as f:
        f.write(merge_py)

    run_wsl(
        f"source /opt/ros/noetic/setup.bash && python3 {DATASET_WSL}/merge_wsl.py",
        desc="ROS Bag Merge",
    )


def run_slam():
    print("=" * 70)
    print(" STEP 2: RUNNING FAST-LIVO2 SLAM IN WSL2")
    print("=" * 70)

    slam_sh = f"""#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

BAG_PATH="{DATASET_WSL}/dynamic_apriltag_merged.bag"
LOG_DIR="{REPO_WSL}/Log"
DEST_DIR="{DATASET_WSL}/slam_out"

echo "=== FAST-LIVO2: Processing Dynamic AprilTag Scan ==="
echo "Bag: $BAG_PATH"

rm -rf "$LOG_DIR/pcd" "$LOG_DIR/result" "$LOG_DIR/image"
mkdir -p "$LOG_DIR/pcd" "$LOG_DIR/result" "$LOG_DIR/image"
mkdir -p "$DEST_DIR"

echo "Starting FAST-LIVO2 mapping node..."
roslaunch fast_livo mapping_raven.launch rviz:=false &
LAUNCH_PID=$!

sleep 5

echo "Playing ROS bag at 1.0x..."
rosbag play "$BAG_PATH" --clock -r 1.0

echo "Bag finished. Waiting 4s..."
sleep 4

echo "Sending SIGINT to save PCD map..."
kill -INT $LAUNCH_PID || true
wait $LAUNCH_PID 2>/dev/null || true

echo "Copying output maps and trajectory..."
cp -rv "$LOG_DIR/pcd" "$DEST_DIR/"
cp -rv "$LOG_DIR/result" "$DEST_DIR/"
echo "FAST-LIVO2 SLAM Finished Successfully!"
"""
    sh_file = os.path.join(DATASET_WIN, "run_slam.sh")
    with open(sh_file, "w", newline="\n", encoding="utf-8") as f:
        f.write(slam_sh)

    run_wsl(f"bash {DATASET_WSL}/run_slam.sh", desc="FAST-LIVO2 SLAM Execution")


if __name__ == "__main__":
    out_dir = os.path.join(DATASET_WIN, "slam_out")
    os.makedirs(out_dir, exist_ok=True)
    merge_bags()
    run_slam()
