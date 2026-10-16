import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_trajectory(traj_path):
    data = []
    with open(traj_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) >= 4:
                try:
                    t = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    data.append([t, x, y, z])
                except ValueError:
                    continue
    return np.array(data) if data else None

def load_points_pcd_or_ply(cloud_path, max_pts=300000):
    cloud_path = Path(cloud_path)
    if cloud_path.suffix.lower() == '.ply':
        with open(cloud_path, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('latin1', errors='ignore').strip()
                header.append(line)
                if line == 'end_header':
                    break
            header_str = ' '.join(header)
            is_binary = 'format binary_little_endian' in header_str
            num_verts = 0
            has_rgb = False
            for h in header:
                if h.startswith('element vertex'):
                    num_verts = int(h.split()[-1])
                if 'property uchar red' in h:
                    has_rgb = True
            if is_binary and num_verts > 0:
                dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]) if has_rgb else np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
                raw_bytes = f.read(num_verts * dtype.itemsize)
                data = np.frombuffer(raw_bytes, dtype=dtype)
                pts = np.vstack([data['x'], data['y'], data['z']]).T
                valid = np.isfinite(pts).all(axis=1)
                pts = pts[valid]
                if len(pts) > max_pts:
                    idx = np.random.choice(len(pts), max_pts, replace=False)
                    pts = pts[idx]
                return pts
    elif cloud_path.suffix.lower() == '.pcd':
        with open(cloud_path, 'rb') as f:
            header = []
            while True:
                line = f.readline().decode('latin1', errors='ignore').strip()
                header.append(line)
                if line.startswith('DATA'):
                    data_mode = line.split()[-1]
                    break
            num_pts = 0
            fields = []
            for h in header:
                if h.startswith('POINTS'):
                    num_pts = int(h.split()[-1])
                if h.startswith('FIELDS'):
                    fields = h.split()[1:]
            if data_mode == 'binary' and num_pts > 0:
                if len(fields) == 8:
                    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'), ('i', '<f4'), ('c', '<f4')])
                elif len(fields) == 4:
                    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('i', '<f4')])
                else:
                    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')] + [(f'f{i}', '<f4') for i in range(len(fields)-3)])
                raw_bytes = f.read(num_pts * dtype.itemsize)
                data = np.frombuffer(raw_bytes, dtype=dtype)
                pts = np.vstack([data['x'], data['y'], data['z']]).T
                valid = np.isfinite(pts).all(axis=1)
                pts = pts[valid]
                if len(pts) > max_pts:
                    idx = np.random.choice(len(pts), max_pts, replace=False)
                    pts = pts[idx]
                return pts
    return None

def generate_views(dataset_dir, output_png, title):
    dataset_dir = Path(dataset_dir)
    traj_candidates = list(dataset_dir.glob('slam_refined*/**/lidar_poses.txt')) + \
                      list(dataset_dir.glob('slam_raw*/**/lidar_poses.txt')) + \
                      list(dataset_dir.glob('slam_refined*/**/Raven_3DMakerPro_Scan.txt')) + \
                      list(dataset_dir.glob('slam_raw*/**/Raven_3DMakerPro_Scan.txt')) + \
                      list(dataset_dir.glob('slam_out/**/lidar_poses.txt')) + \
                      list(dataset_dir.glob('slam_out/**/Raven_3DMakerPro_Scan.txt'))
    traj = None
    for tc in traj_candidates:
        if tc.is_file():
            traj = load_trajectory(tc)
            if traj is not None and len(traj) > 0:
                print(f'Using trajectory: {tc.name} ({len(traj)} poses)')
                break

    cloud_candidates = list(dataset_dir.glob('colmap_3dgs/sparse/0/points3D.ply')) + \
                       list(dataset_dir.glob('slam_refined*/**/all_raw_points.pcd')) + \
                       list(dataset_dir.glob('slam_raw*/**/all_raw_points.pcd')) + \
                       list(dataset_dir.glob('slam_out/**/all_downsampled_points.pcd')) + \
                       list(dataset_dir.glob('slam_out/**/all_raw_points.pcd'))
    cloud = None
    for cc in cloud_candidates:
        if cc.is_file():
            print(f'Loading cloud: {cc}...')
            cloud = load_points_pcd_or_ply(cc, max_pts=300000)
            if cloud is not None and len(cloud) > 0:
                print(f'Loaded {len(cloud)} points from {cc.name}')
                break

    if cloud is None and traj is None:
        print(f'No data found for {dataset_dir}')
        return

    fig = plt.figure(figsize=(19, 6), dpi=160)
    fig.suptitle(f'{title} — Nuvem ({len(cloud):,} pts) + Trajetória SLAM', fontsize=15, fontweight='bold', y=0.98)

    # 1. Top-Down View (X vs Y)
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title('Vista Superior (Top-Down X-Y)', fontsize=12, fontweight='bold')
    if cloud is not None:
        sc1 = ax1.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], cmap='turbo', s=0.5, alpha=0.5, rasterized=True)
        cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Z (m) / Elevação', fontsize=9)
    if traj is not None:
        ax1.plot(traj[:, 1], traj[:, 2], color='red', linewidth=2.0, label='Trajetória SLAM', zorder=10)
        ax1.scatter(traj[0, 1], traj[0, 2], color='lime', s=55, marker='o', edgecolors='black', label='Início', zorder=11)
        ax1.scatter(traj[-1, 1], traj[-1, 2], color='magenta', s=55, marker='X', edgecolors='black', label='Fim', zorder=11)
        ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 2. Longitudinal Cut (X vs Z)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('Corte Longitudinal (X-Z)', fontsize=12, fontweight='bold')
    if cloud is not None:
        sc2 = ax2.scatter(cloud[:, 0], cloud[:, 2], c=cloud[:, 1], cmap='plasma', s=0.5, alpha=0.5, rasterized=True)
        cbar2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Y (m) / Profundidade', fontsize=9)
    if traj is not None:
        ax2.plot(traj[:, 1], traj[:, 3], color='red', linewidth=2.0, label='Trajetória SLAM', zorder=10)
        ax2.scatter(traj[0, 1], traj[0, 3], color='lime', s=55, marker='o', edgecolors='black', zorder=11)
        ax2.scatter(traj[-1, 1], traj[-1, 3], color='magenta', s=55, marker='X', edgecolors='black', zorder=11)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z / Altura (m)')
    ax2.axis('equal')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # 3. Transverse Cut (Y vs Z)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Corte Transversal (Y-Z)', fontsize=12, fontweight='bold')
    if cloud is not None:
        sc3 = ax3.scatter(cloud[:, 1], cloud[:, 2], c=cloud[:, 0], cmap='viridis', s=0.5, alpha=0.5, rasterized=True)
        cbar3 = plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('X (m) / Lateral', fontsize=9)
    if traj is not None:
        ax3.plot(traj[:, 2], traj[:, 3], color='red', linewidth=2.0, label='Trajetória SLAM', zorder=10)
        ax3.scatter(traj[0, 2], traj[0, 3], color='lime', s=55, marker='o', edgecolors='black', zorder=11)
        ax3.scatter(traj[-1, 2], traj[-1, 3], color='magenta', s=55, marker='X', edgecolors='black', zorder=11)
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z / Altura (m)')
    ax3.axis('equal')
    ax3.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=160, bbox_inches='tight')
    plt.close()
    print(f'[OK] Vista ortogonal e cortes salvos em: {output_png}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python generate_sections_plot.py <dataset_dir> <output_png> <title>')
        sys.exit(1)
    generate_views(sys.argv[1], sys.argv[2], sys.argv[3])
