import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 1. Paths
dataset_dir = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output')
trj_path = dataset_dir / 'slam_refined_revisit-plane-bundle-v2_1792601135242622700_1792601135237248800/result/Raven_3DMakerPro_Scan.txt'
pcd_path = dataset_dir / 'slam_refined_revisit-plane-bundle-v2_1792601135242622700_1792601135237248800/result/scan.pcd'
out_png = dataset_dir / 'deliverables/pier_trajectory_and_colorization_diagnostic.png'

print('[*] Loading LiDAR trajectory...')
trj_data = np.loadtxt(trj_path)
t_slam = trj_data[:, 0]
pos_slam = trj_data[:, 1:4]

print('[*] Loading point cloud (downsampled)...')
# Read pcd
points = []
with open(pcd_path, 'r', errors='ignore') as f:
    header = True
    for line in f:
        if header:
            if line.startswith('DATA'):
                header = False
            continue
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
points = np.array(points)
print(f'[+] Total PCD points: {len(points)}')

# Subsample points for plotting (e.g. 50,000 points)
step = max(1, len(points) // 50000)
sub_pts = points[::step]

# 2. Compute camera trajectory with wrong dt (-7.106s) vs correct dt (+5.47s)
# Camera factory calibration lever arm
factory_lever = np.array([-0.0336, -0.1293, 0.0988])

from scipy.spatial.transform import Rotation as R_scipy, Slerp
from scipy.interpolate import interp1d

quats = trj_data[:, 4:8]  # x, y, z, w
slerp = Slerp(t_slam, R_scipy.from_quat(quats))
pos_interp = interp1d(t_slam, pos_slam, axis=0, bounds_error=False, fill_value='extrapolate')

# Video times: 0 to 203s at 2 fps
t_vids = np.arange(0, 203.5, 0.5)
t0_slam = t_slam[0]

# WRONG trajectory (dt = -7.106)
t_q_wrong = t0_slam + (t_vids - (-7.106))
t_q_wrong = np.clip(t_q_wrong, t_slam[0], t_slam[-1])
pos_cam_wrong = pos_interp(t_q_wrong)

# CORRECT trajectory (dt = +5.470)
t_q_correct = t0_slam + (t_vids - 5.470)
t_q_correct = np.clip(t_q_correct, t_slam[0], t_slam[-1])
pos_cam_correct = pos_interp(t_q_correct)

print('[*] Generating diagnostic figures...')
fig = plt.figure(figsize=(20, 10), dpi=150)

# Subplot 1: Top-down XY map
ax1 = fig.add_subplot(1, 2, 1)
# Plot point cloud colored by height (Z)
sc = ax1.scatter(sub_pts[:, 0], sub_pts[:, 1], c=sub_pts[:, 2], cmap='viridis', s=1, alpha=0.4, label='LiDAR Map (scan.pcd)')
# Plot LiDAR trajectory
ax1.plot(pos_slam[:, 0], pos_slam[:, 1], 'k-', linewidth=3, label='LiDAR Trajectory (True Path)')
# Plot start and end
ax1.scatter(pos_slam[0, 0], pos_slam[0, 1], c='green', s=120, marker='o', edgecolors='black', label='LiDAR Start (t=0s)', zorder=5)
ax1.scatter(pos_slam[-1, 0], pos_slam[-1, 1], c='black', s=120, marker='X', edgecolors='white', label='LiDAR End (t=214s)', zorder=5)

# Plot Camera with WRONG dt
ax1.plot(pos_cam_wrong[:, 0], pos_cam_wrong[:, 1], 'r--', linewidth=2, alpha=0.8, label='Cam with WRONG sync (dt = -7.1s, 12.6s error!)')
ax1.scatter(pos_cam_wrong[0, 0], pos_cam_wrong[0, 1], c='red', s=100, marker='^', edgecolors='black', label='Wrong Cam Start Pose', zorder=6)

# Plot Camera with CORRECT dt
ax1.plot(pos_cam_correct[:, 0], pos_cam_correct[:, 1], 'b-', linewidth=2.5, alpha=0.9, label='Cam with TRUE physical sync (dt = +5.47s, r = 0.82)')
ax1.scatter(pos_cam_correct[0, 0], pos_cam_correct[0, 1], c='blue', s=100, marker='^', edgecolors='black', label='True Cam Start Pose', zorder=6)

ax1.set_title('PIER Dataset: Top-Down (X-Y) Trajectory & Point Cloud Alignment', fontsize=14, fontweight='bold')
ax1.set_xlabel('X [meters]', fontsize=12)
ax1.set_ylabel('Y [meters]', fontsize=12)
ax1.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax1.axis('equal')
ax1.grid(True, linestyle=':', alpha=0.6)

# Subplot 2: 3D Perspective
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.scatter(sub_pts[::2, 0], sub_pts[::2, 1], sub_pts[::2, 2], c=sub_pts[::2, 2], cmap='viridis', s=0.8, alpha=0.3)
ax2.plot(pos_slam[:, 0], pos_slam[:, 1], pos_slam[:, 2], 'k-', linewidth=3, label='LiDAR Path')
ax2.plot(pos_cam_wrong[:, 0], pos_cam_wrong[:, 1], pos_cam_wrong[:, 2], 'r--', linewidth=2, label='Wrong Sync Path (12.6s gap)')
ax2.plot(pos_cam_correct[:, 0], pos_cam_correct[:, 1], pos_cam_correct[:, 2], 'b-', linewidth=2.5, label='Correct Sync Path')
ax2.set_title('3D Perspective View: Physical Motion Trajectories', fontsize=14, fontweight='bold')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_zlabel('Z [m]')
ax2.legend(loc='upper right', fontsize=9)
ax2.view_init(elev=35, azim=-60)

plt.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_png, dpi=150)
print(f'[+] Diagnostic plot saved successfully: {out_png}')
