#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import json
import numpy as np

WSL_DISTRO = "Ubuntu-20.04"
DATASET_DIR_WIN = r"C:\Users\User\Documents\ARQUIVOS_TESTE\SMALL-DATASET-TEST"
DATASET_DIR_WSL = "/mnt/c/Users/User/Documents/ARQUIVOS_TESTE/SMALL-DATASET-TEST"
REPO_DIR_WIN = r"C:\Users\User\Documents\APLICATIVOS\RAVEN-SCAN-INSTA360-COLORIZATION"
REPO_DIR_WSL = "/mnt/c/Users/User/Documents/APLICATIVOS/RAVEN-SCAN-INSTA360-COLORIZATION"
CALIB_JSON_WIN = r"C:\Users\User\Documents\APLICATIVOS\Lidar-Camera-calibrator\extrinsics_determined.json"


def run_wsl_bash(cmd_str, desc="WSL command"):
    print(f"\n=======================================================")
    print(f" [*] {desc}")
    print(f" Command: {cmd_str}")
    print(f"=======================================================")
    full_cmd = ["wsl.exe", "-d", WSL_DISTRO, "-u", "root", "bash", "-c", cmd_str]
    t0 = time.time()
    res = subprocess.run(full_cmd, check=True)
    dt = time.time() - t0
    print(f"[+] Completed in {dt:.1f}s")
    return res


def merge_bags():
    print("\n--- STEP 1: Merging Bags for FAST-LIVO2 ---")
    merge_script = f"""
import rosbag
import os

lidar_bag = '{DATASET_DIR_WSL}/LIDAR_20260821105002Teste.bag'
img_bag = '{DATASET_DIR_WSL}/IMAGE_20260821105002Teste.bag'
out_bag = '{DATASET_DIR_WSL}/small_merged.bag'

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
    tmp_py = os.path.join(REPO_DIR_WIN, "scripts", "merge_small_dataset_wsl.py")
    with open(tmp_py, "w", newline="\n", encoding="utf-8") as f:
        f.write(merge_script)

    run_wsl_bash(
        f"source /opt/ros/noetic/setup.bash && python3 {REPO_DIR_WSL}/scripts/merge_small_dataset_wsl.py",
        desc="ROS Bag Merge",
    )


def run_slam():
    print("\n--- STEP 2: Running FAST-LIVO2 SLAM ---")
    slam_sh = f"""#!/usr/bin/env bash
set -e
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

BAG_PATH="{DATASET_DIR_WSL}/small_merged.bag"
LOG_DIR="{REPO_DIR_WSL}/Log"
DEST_DIR="{REPO_DIR_WSL}/Log/small_test"

echo "=== FAST-LIVO2 SLAM: Processing SMALL-DATASET-TEST ==="
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

echo "Bag playback finished. Waiting for queues..."
sleep 4

echo "Sending SIGINT to save PCD..."
kill -INT $LAUNCH_PID || true
wait $LAUNCH_PID 2>/dev/null || true

echo "Archiving results..."
rm -rf "$DEST_DIR"
mkdir -p "$DEST_DIR/pcd" "$DEST_DIR/result" "$DEST_DIR/image"

if [ -f "$LOG_DIR/pcd/all_raw_points.pcd" ]; then
    cp "$LOG_DIR/pcd/all_raw_points.pcd" "$DEST_DIR/pcd/"
fi
if [ -f "$LOG_DIR/pcd/all_downsampled_points.pcd" ]; then
    cp "$LOG_DIR/pcd/all_downsampled_points.pcd" "$DEST_DIR/pcd/"
fi
if [ -f "$LOG_DIR/result/Raven_3DMakerPro_Scan.txt" ]; then
    cp "$LOG_DIR/result/Raven_3DMakerPro_Scan.txt" "$DEST_DIR/result/"
fi

echo "SLAM Complete!"
"""
    tmp_sh = os.path.join(REPO_DIR_WIN, "scripts", "run_small_dataset_slam_wsl.sh")
    with open(tmp_sh, "w", newline="\n", encoding="utf-8") as f:
        f.write(slam_sh)

    run_wsl_bash(f"bash {REPO_DIR_WSL}/scripts/run_small_dataset_slam_wsl.sh", desc="FAST-LIVO2 SLAM")


if __name__ == "__main__":
    print("Helper script ready.")
