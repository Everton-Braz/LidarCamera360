import os
import struct
import numpy as np
from pathlib import Path
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as Rot

INSV_FILE = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib\VID_20260902_143757_00_277.insv"
BAG_FILE = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib\LIDAR_DinamicAprilTagCalib.bag"
TRJ_FILE = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib\slam_out\result\Raven_3DMakerPro_Scan.txt"

# 1. Load Insta360 IMU
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

    _, g_size, g_off = offsets[3]
    fin.seek(extra_start + g_off)
    gyro_bytes = fin.read(g_size)

    _, e_size, e_off = offsets[4]
    fin.seek(extra_start + e_off)
    exp_bytes = fin.read(e_size)

cam_imu_raw = np.frombuffer(gyro_bytes, dtype=[
    ('t', '<u8'),
    ('ax', '<u2'), ('ay', '<u2'), ('az', '<u2'),
    ('gx', '<u2'), ('gy', '<u2'), ('gz', '<u2')
])
cam_exp_raw = np.frombuffer(exp_bytes, dtype=[('t', '<u8'), ('exp', '<f8')])
t_exp0_us = cam_exp_raw['t'][0]

t_cam_s = (cam_imu_raw['t'].astype(np.float64) - t_exp0_us) / 1e6

# Gyro raw counts (-32768)
cam_gx = cam_imu_raw['gx'].astype(np.float64) - 32768.0
cam_gy = cam_imu_raw['gy'].astype(np.float64) - 32768.0
cam_gz = cam_imu_raw['gz'].astype(np.float64) - 32768.0

cam_ax = cam_imu_raw['ax'].astype(np.float64) - 32768.0
cam_ay = cam_imu_raw['ay'].astype(np.float64) - 32768.0
cam_az = cam_imu_raw['az'].astype(np.float64) - 32768.0

# 2. Load LiDAR IMU
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

trj = np.loadtxt(TRJ_FILE)
t_slam_start = trj[0, 0]
t_lidar_rel_slam = lidar_t - t_slam_start

# EXACT GYRO SYNC TIME SHIFT:
dt_sync = 5.7075
print(f"Applying exact gyro sync dt = {dt_sync:.4f} s")

# Resample on synchronized common timestamps where rotation is active
# Evaluate at LiDAR IMU timestamps
t_cam_query = t_lidar_rel_slam + dt_sync
valid = (t_cam_query >= t_cam_s[0]) & (t_cam_query <= t_cam_s[-1])

t_eval = t_lidar_rel_slam[valid]
t_cam_eval = t_cam_query[valid]

# Interpolate Camera Gyro to LiDAR timestamps
w_cam_x = np.interp(t_cam_eval, t_cam_s, cam_gx)
w_cam_y = np.interp(t_cam_eval, t_cam_s, cam_gy)
w_cam_z = np.interp(t_cam_eval, t_cam_s, cam_gz)
W_cam = np.stack([w_cam_x, w_cam_y, w_cam_z], axis=1)

# LiDAR Gyro at these timestamps
w_lid_x = lidar_wx[valid]
w_lid_y = lidar_wy[valid]
w_lid_z = lidar_wz[valid]
W_lid = np.stack([w_lid_x, w_lid_y, w_lid_z], axis=1)

# Filter for active rotation periods to avoid stationary noise
lid_speed = np.linalg.norm(W_lid, axis=1)
cam_speed = np.linalg.norm(W_cam, axis=1)

active_mask = (lid_speed > 0.05) # > 0.05 rad/s (~2.8 deg/s)
print(f"Total synchronized samples: {len(W_lid):,}")
print(f"Active rotation samples:     {np.sum(active_mask):,}")

W_lid_act = W_lid[active_mask]
W_cam_act = W_cam[active_mask]

# Normalize each vector to unit direction so magnitude scaling doesn't affect rotation
W_lid_u = W_lid_act / np.linalg.norm(W_lid_act, axis=1, keepdims=True)
W_cam_u = W_cam_act / np.linalg.norm(W_cam_act, axis=1, keepdims=True)

# Kabsch Algorithm (SVD): Find R such that W_cam_u ≈ R @ W_lid_u
# H = W_lid_u^T @ W_cam_u
H = W_lid_u.T @ W_cam_u
U, S, Vt = np.linalg.svd(H)
R_gyro_est = Vt.T @ U.T

# Check determinant to ensure proper rotation (not reflection)
det = np.linalg.det(R_gyro_est)
print(f"Kabsch SVD determinant: {det:.4f}")
if det < 0:
    Vt[-1, :] *= -1
    R_gyro_est = Vt.T @ U.T
    det = np.linalg.det(R_gyro_est)
    print(f"Corrected determinant: {det:.4f}")

# Alignment residual
W_cam_pred = (R_gyro_est @ W_lid_u.T).T
dot_prod = np.sum(W_cam_pred * W_cam_u, axis=1)
mean_angle_err_deg = np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0))).mean()
median_angle_err_deg = np.median(np.degrees(np.arccos(np.clip(dot_prod, -1.0, 1.0))))

print("\n" + "=" * 80)
print(" 4. ESTIMATED ROTATION MATRIX: LIDAR IMU -> CAMERA IMU")
print("=" * 80)
print("R_gyro_est:")
for row in R_gyro_est:
    print(f"  [{row[0]:+8.5f}, {row[1]:+8.5f}, {row[2]:+8.5f}]")

euler_xyz = Rot.from_matrix(R_gyro_est).as_euler('xyz', degrees=True)
euler_zxy = Rot.from_matrix(R_gyro_est).as_euler('zxy', degrees=True)
print(f"Euler XYZ (deg): {euler_xyz}")
print(f"Euler ZXY (deg): {euler_zxy}")
print(f"Mean angular alignment error:   {mean_angle_err_deg:.2f}°")
print(f"Median angular alignment error: {median_angle_err_deg:.2f}°")

# Also let's check accelerometer alignment under resting conditions
rest_mask = (lid_speed < 0.02)
A_lid_rest = np.median(np.stack([lidar_ax[valid][rest_mask], lidar_ay[valid][rest_mask], lidar_az[valid][rest_mask]], axis=1), axis=0)
A_cam_rest = np.median(np.stack([
    np.interp(t_cam_eval[rest_mask], t_cam_s, cam_ax),
    np.interp(t_cam_eval[rest_mask], t_cam_s, cam_ay),
    np.interp(t_cam_eval[rest_mask], t_cam_s, cam_az)
], axis=1), axis=0)

print("\n" + "=" * 80)
print(" 5. ACCELEROMETER GRAVITY ALIGNMENT AT REST")
print("=" * 80)
u_g_lid = A_lid_rest / np.linalg.norm(A_lid_rest)
u_g_cam = A_cam_rest / np.linalg.norm(A_cam_rest)
print(f"LiDAR resting gravity unit vector:  [{u_g_lid[0]:+8.4f}, {u_g_lid[1]:+8.4f}, {u_g_lid[2]:+8.4f}]")
print(f"Camera resting gravity unit vector: [{u_g_cam[0]:+8.4f}, {u_g_cam[1]:+8.4f}, {u_g_cam[2]:+8.4f}]")

# Check if R_gyro_est maps u_g_lid to u_g_cam
u_g_cam_pred = R_gyro_est @ u_g_lid
g_err_deg = np.degrees(np.arccos(np.clip(np.dot(u_g_cam_pred, u_g_cam), -1.0, 1.0)))
print(f"Predicted camera gravity unit:      [{u_g_cam_pred[0]:+8.4f}, {u_g_cam_pred[1]:+8.4f}, {u_g_cam_pred[2]:+8.4f}]")
print(f"Gravity direction error with R_gyro: {g_err_deg:.2f}°")
