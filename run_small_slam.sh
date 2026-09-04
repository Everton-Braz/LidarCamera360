#!/usr/bin/env bash
set -e

source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash

BAG_PATH="/mnt/c/Users/User/Documents/ARQUIVOS_TESTE/SMALL-DATASET-TEST/small_merged.bag"
OUT_DIR="/mnt/c/Users/User/Documents/ARQUIVOS_TESTE/SMALL-DATASET-TEST/slam_out"
LOG_DIR="/mnt/c/Users/User/Documents/APLICATIVOS/RAVEN-SCAN-INSTA360-COLORIZATION/Log"

mkdir -p "$LOG_DIR/pcd" "$LOG_DIR/result" "$LOG_DIR/image"
mkdir -p "$OUT_DIR/pcd" "$OUT_DIR/result"

echo "[1/5] Starting ROS core..."
roscore &
ROSCORE_PID=$!
sleep 4

echo "[2/5] Setting sim time and starting FAST-LIVO2..."
rosparam set /use_sim_time true
roslaunch fast_livo mapping_raven.launch rviz:=false > /tmp/fastlivo.log 2>&1 &
LAUNCH_PID=$!

sleep 6

echo "[3/5] Playing bag at 2.0x rate..."
rosbag play "$BAG_PATH" --clock -r 2.0

echo "[4/5] Playback complete. Finalizing SLAM and saving point cloud..."
sleep 3

NODE_PID=$(pgrep -f fastlivo_mapping || echo "")
if [ -n "$NODE_PID" ]; then
    echo "Sending SIGINT to fastlivo_mapping (PID: $NODE_PID)..."
    kill -INT $NODE_PID || true
    while kill -0 $NODE_PID 2>/dev/null; do
        echo "Waiting for PCD write..."
        sleep 2
    done
fi

kill -INT $LAUNCH_PID 2>/dev/null || true
kill -INT $ROSCORE_PID 2>/dev/null || true
wait $LAUNCH_PID 2>/dev/null || true
wait $ROSCORE_PID 2>/dev/null || true

echo "[5/5] Copying results to $OUT_DIR..."
cp -r "$LOG_DIR/pcd"/* "$OUT_DIR/pcd/" 2>/dev/null || true
cp -r "$LOG_DIR/result"/* "$OUT_DIR/result/" 2>/dev/null || true

echo "SLAM FINISHED SUCCESSFULLY!"
ls -lh "$OUT_DIR/pcd" "$OUT_DIR/result"
