#!/usr/bin/env bash
# Script to run FAST-LIVO2 against 3DMakerPro Raven dataset in WSL2

RATE=${1:-"2.0"}
BAG_PATH=${2:-"/mnt/d/APLICATIVOS/FAST-LIVO2/DATA_TEST/raven_merged.bag"}

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

echo "========================================================="
echo " FAST-LIVO2 Dataset Runner (WSL2 Ubuntu 20.04)"
echo " Dataset Bag: $BAG_PATH"
echo " Playback Rate: ${RATE}x"
echo "========================================================="

# Ensure directories exist
mkdir -p /mnt/d/APLICATIVOS/FAST-LIVO2/Log/pcd
mkdir -p /mnt/d/APLICATIVOS/FAST-LIVO2/Log/result
mkdir -p /mnt/d/APLICATIVOS/FAST-LIVO2/Log/image
mkdir -p /mnt/d/APLICATIVOS/FAST-LIVO2/Log/Colmap/images
mkdir -p /mnt/d/APLICATIVOS/FAST-LIVO2/Log/Colmap/sparse/0

rosparam set /use_sim_time true

# Launch FAST-LIVO2
roslaunch fast_livo mapping_raven.launch rviz:=false > /mnt/d/APLICATIVOS/FAST-LIVO2/Log/fast_livo_run.log 2>&1 &
LAUNCH_PID=$!
echo "[INFO] FAST-LIVO2 node started (PID: $LAUNCH_PID). Streaming output to Log/fast_livo_run.log"

sleep 5

echo "[INFO] Starting rosbag playback at ${RATE}x rate..."
rosbag play "$BAG_PATH" -r "$RATE" --clock /camera_front/image_raw:=/camera_front/image/compressed

echo "[INFO] Playback finished. Waiting for node to finalize..."
sleep 5

# Send SIGINT directly to fastlivo_mapping so it triggers savePCD and flushes to disk
NODE_PID=$(pgrep -f fastlivo_mapping || true)
if [ -n "$NODE_PID" ]; then
  echo "[INFO] Triggering PCD save on fastlivo_mapping (PID: $NODE_PID)..."
  kill -INT $NODE_PID || true
  while kill -0 $NODE_PID 2>/dev/null; do
    echo "[INFO] Saving PCD point cloud map to disk, please wait..."
    sleep 3
  done
  echo "[INFO] fastlivo_mapping finished saving PCD files!"
fi

# Now terminate roslaunch cleanly
kill -INT $LAUNCH_PID 2>/dev/null || true
wait $LAUNCH_PID 2>/dev/null || true

echo "========================================================="
echo " Processing Completed!"
echo " Outputs:"
echo " - Trajectory:  /mnt/d/APLICATIVOS/FAST-LIVO2/Log/result/Raven_3DMakerPro_Scan.txt"
echo " - Filtered Map: /mnt/d/APLICATIVOS/FAST-LIVO2/Log/pcd/all_downsampled_points.pcd"
echo " - Dense Map:   /mnt/d/APLICATIVOS/FAST-LIVO2/Log/pcd/all_raw_points.pcd"
echo "========================================================="
