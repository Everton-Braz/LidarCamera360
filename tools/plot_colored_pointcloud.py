import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ply_path = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output/deliverables/lidar_colored_direct.ply')
out_png = Path('D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output/deliverables/pier_colored_pointcloud_render.png')

print('[*] Loading colored PLY...')
with open(ply_path, 'rb') as f:
    while True:
        line = f.readline().decode('ascii', errors='ignore').strip()
        if line.startswith('element vertex'):
            n_pts = int(line.split()[-1])
        if line == 'end_header':
            break
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    data = np.fromfile(f, dtype=dt, count=n_pts)

print(f'[+] Read {len(data)} points')
# Subsample for rendering
# Filter points that are colored (not pure black)
colored_mask = (data['r'] > 10) | (data['g'] > 10) | (data['b'] > 10)
colored_data = data[colored_mask]
print(f'[+] Colored points: {len(colored_data)} ({len(colored_data)/len(data)*100:.1f}%)')

step = max(1, len(colored_data) // 80000)
sub = colored_data[::step]
rgb = np.column_stack([sub['r'], sub['g'], sub['b']]) / 255.0

fig = plt.figure(figsize=(24, 8), dpi=150)

# View 1: Top-Down (X-Y)
ax1 = fig.add_subplot(1, 3, 1)
ax1.scatter(sub['x'], sub['y'], c=rgb, s=1.5, alpha=0.8)
ax1.set_title('Top-Down View (X-Y Plane)', fontsize=14, fontweight='bold')
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_facecolor('#1a1a1a')
ax1.axis('equal')
ax1.grid(True, color='#333333', linestyle=':')

# View 2: Perspective 3D
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(sub['x'], sub['y'], sub['z'], c=rgb, s=1.2, alpha=0.8)
ax2.set_title('3D Perspective View', fontsize=14, fontweight='bold')
ax2.set_xlabel('X [m]')
ax2.set_ylabel('Y [m]')
ax2.set_zlabel('Z [m]')
ax2.set_facecolor('#1a1a1a')
ax2.view_init(elev=25, azim=-45)

# View 3: Side/Front View (X-Z)
ax3 = fig.add_subplot(1, 3, 3)
ax3.scatter(sub['x'], sub['z'], c=rgb, s=1.5, alpha=0.8)
ax3.set_title('Side Profile View (X-Z Plane)', fontsize=14, fontweight='bold')
ax3.set_xlabel('X [m]')
ax3.set_ylabel('Z [m]')
ax3.set_facecolor('#1a1a1a')
ax3.set_facecolor('#1a1a1a')
ax3.axis('equal')
ax3.grid(True, color='#333333', linestyle=':')

fig.suptitle('PIER Dataset: Colorized Point Cloud with Physical Sync (dt = +5.470s)', fontsize=16, fontweight='bold', color='white')
fig.patch.set_facecolor('#121212')
plt.tight_layout()
plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
print(f'[+] Render saved: {out_png}')
