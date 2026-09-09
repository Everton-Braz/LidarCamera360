import os
import struct
import argparse
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy import signal
from scipy.spatial.transform import Rotation as Rot

parser = argparse.ArgumentParser(description="Estimate camera-to-LiDAR time offset from gyro cross-correlation.")
parser.add_argument("--insv", required=True, help="Path to the Insta360 INSV file.")
parser.add_argument("--bag", required=True, help="Path to the LiDAR ROS bag.")
parser.add_argument("--trajectory", required=True, help="Path to the LiDAR SLAM trajectory.")
args = parser.parse_args()
INSV_FILE = args.insv
BAG_FILE = args.bag
TRJ_FILE = args.trajectory

print("=" * 80)
print(" 1. EXTRACTING INSTA360 IMU DATA")
print("=" * 80)

HEADER_SIZE = 72
with open(INSV_FILE, "rb") as fin:
    fin.seek(-HEADER_SIZE, 2)
    header = fin.read(HEADER_SIZE)
    extra_size = struct.unpack('<I', header[32:36])[0]
    file_size = fin.seek(0, 2)
    extra_start = file_size - extra_size

    fin.seek(-(HEADER_SIZE + 6 + 250), 2)
    offsets_data = fin.read(250)
    offsets = {}
    for i in range(0, len(offsets_data), 10):
        oid, ofmt, osize, ooff = struct.unpack('<BBII', offsets_data[i:i+10])
        if oid > 0:
            offsets[oid] = (ofmt, osize, ooff)

    # Gyro record (id=3)
    _, g_size, g_off = offsets[3]
    fin.seek(extra_start + g_off)
    gyro_bytes = fin.read(g_size)

    # Exposure record (id=4)
    _, e_size, e_off = offsets[4]
    fin.seek(extra_start + e_off)
    exp_bytes = fin.read(e_size)

# Parse Insta360 Gyro
cam_imu_raw = np.frombuffer(gyro_bytes, dtype=[
    ('t', '<u8'),
    ('ax', '<u2'), ('ay', '<u2'), ('az', '<u2'),
    ('gx', '<u2'), ('gy', '<u2'), ('gz', '<u2')
])

# Parse Insta360 Exposure
cam_exp_raw = np.frombuffer(exp_bytes, dtype=[
    ('t', '<u8'),
    ('exp', '<f8')
])

t_exp0_us = cam_exp_raw['t'][0]
print(f"Insta360 Gyro samples: {len(cam_imu_raw):,}")
print(f"Insta360 Exposure samples: {len(cam_exp_raw):,} (video frames)")
print(f"First frame timestamp: {t_exp0_us} us")
print(f"First gyro timestamp:  {cam_imu_raw['t'][0]} us")

# Time relative to video start (frame 0) in seconds:
t_cam_s = (cam_imu_raw['t'].astype(np.float64) - t_exp0_us) / 1e6

# Gyro raw in counts (-32768)
cam_gx = cam_imu_raw['gx'].astype(np.float64) - 32768.0
cam_gy = cam_imu_raw['gy'].astype(np.float64) - 32768.0
cam_gz = cam_imu_raw['gz'].astype(np.float64) - 32768.0

# Accel raw in counts (-32768)
cam_ax = cam_imu_raw['ax'].astype(np.float64) - 32768.0
cam_ay = cam_imu_raw['ay'].astype(np.float64) - 32768.0
cam_az = cam_imu_raw['az'].astype(np.float64) - 32768.0

cam_gyro_norm = np.sqrt(cam_gx**2 + cam_gy**2 + cam_gz**2)
print(f"Camera gyro time range: {t_cam_s[0]:.3f}s to {t_cam_s[-1]:.3f}s (dt = {t_cam_s[-1]-t_cam_s[0]:.3f}s)")

print("\n" + "=" * 80)
print(" 2. EXTRACTING LIDAR IMU DATA FROM ROS BAG")
print("=" * 80)

lidar_t = []
lidar_wx = []
lidar_wy = []
lidar_wz = []
lidar_ax = []
lidar_ay = []
lidar_az = []

with AnyReader([Path(BAG_FILE)]) as reader:
    for conn, ts, raw in reader.messages():
        if conn.topic == '/vanjee_imu_packets':
            msg = reader.deserialize(raw, conn.msgtype)
            t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            lidar_t.append(t_sec)
            lidar_wx.append(msg.angular_velocity.x)
            lidar_wy.append(msg.angular_velocity.y)
            lidar_wz.append(msg.angular_velocity.z)
            lidar_ax.append(msg.linear_acceleration.x)
            lidar_ay.append(msg.linear_acceleration.y)
            lidar_az.append(msg.linear_acceleration.z)

lidar_t = np.array(lidar_t)
lidar_wx = np.array(lidar_wx)
lidar_wy = np.array(lidar_wy)
lidar_wz = np.array(lidar_wz)
lidar_ax = np.array(lidar_ax)
lidar_ay = np.array(lidar_ay)
lidar_az = np.array(lidar_az)

lidar_gyro_norm = np.sqrt(lidar_wx**2 + lidar_wy**2 + lidar_wz**2)
print(f"LiDAR IMU samples: {len(lidar_t):,}")
print(f"LiDAR time range: {lidar_t[0]:.3f} to {lidar_t[-1]:.3f} (span: {lidar_t[-1]-lidar_t[0]:.3f}s)")

# Load SLAM trajectory start time
trj = np.loadtxt(TRJ_FILE)
t_slam_start = trj[0, 0]
print(f"SLAM Trajectory start time: {t_slam_start:.3f}")
print(f"LiDAR bag start time:       {lidar_t[0]:.3f} (diff = {t_slam_start - lidar_t[0]:.3f}s)")

# Time of LiDAR IMU relative to SLAM trajectory start:
t_lidar_rel_slam = lidar_t - t_slam_start

print("\n" + "=" * 80)
print(" 3. CROSS-CORRELATION: TIME SYNCHRONIZATION")
print("=" * 80)

# Resample both gyro norms to a common 200 Hz grid
fs = 200.0  # 200 Hz
t_grid_max = min(t_cam_s[-1], t_lidar_rel_slam[-1] + 30.0)

# Resample camera norm on a grid from 0 to 95s
t_eval = np.arange(1.0, 90.0, 1.0 / fs)

# Camera gyro norm interpolated on t_eval
cam_norm_interp = np.interp(t_eval, t_cam_s, cam_gyro_norm)
cam_norm_interp -= np.median(cam_norm_interp)

# We want to find dt_shift such that: t_lidar_rel_slam = t_cam_s - dt_shift
# Or query_lidar_t = t_eval - dt_shift
# Test shifts from 0.0s to 12.0s in steps of 0.005s (5 ms)
shifts = np.arange(2.0, 10.0, 0.005)
corrs = []

for shift in shifts:
    t_lidar_query = t_eval - shift
    # Mask valid in lidar range
    valid = (t_lidar_query >= t_lidar_rel_slam[0]) & (t_lidar_query <= t_lidar_rel_slam[-1])
    if np.sum(valid) > 1000:
        lid_norm_interp = np.interp(t_lidar_query[valid], t_lidar_rel_slam, lidar_gyro_norm)
        lid_norm_interp -= np.median(lid_norm_interp)
        c = np.corrcoef(cam_norm_interp[valid], lid_norm_interp)[0, 1]
        corrs.append(c)
    else:
        corrs.append(0.0)

corrs = np.array(corrs)
best_idx = np.argmax(corrs)
best_dt = shifts[best_idx]
best_corr = corrs[best_idx]

print(f"Optimal dt from Gyro Cross-Correlation: dt = {best_dt:.4f} s (Correlation = {best_corr:.4f})")
print(f"Previous programmatic/visual dt was:    dt = 5.4990 s")
print(f"Difference:                             {abs(best_dt - 5.499)*1000:.1f} ms")

# Fine search around best_dt at 0.5 ms resolution
fine_shifts = np.arange(best_dt - 0.1, best_dt + 0.1, 0.0005)
fine_corrs = []
for shift in fine_shifts:
    t_lidar_query = t_eval - shift
    valid = (t_lidar_query >= t_lidar_rel_slam[0]) & (t_lidar_query <= t_lidar_rel_slam[-1])
    lid_norm_interp = np.interp(t_lidar_query[valid], t_lidar_rel_slam, lidar_gyro_norm)
    lid_norm_interp -= np.median(lid_norm_interp)
    c = np.corrcoef(cam_norm_interp[valid], lid_norm_interp)[0, 1]
    fine_corrs.append(c)

fine_corrs = np.array(fine_corrs)
fine_best_idx = np.argmax(fine_corrs)
exact_dt = fine_shifts[fine_best_idx]
exact_corr = fine_corrs[fine_best_idx]

print(f"\n[>>>] HIGH PRECISION GYRO SYNC: dt = {exact_dt:.4f} s (Pearson r = {exact_corr:.4f})")
