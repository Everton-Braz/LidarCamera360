#!/usr/bin/env python3
"""
Universal Multi-View Orthogonal & Trajectory Plotter for 3DGS & Calibration Datasets.
Generates 3-view orthogonal analysis: Top View (XY), Longitudinal Profile, and Latitudinal/Transverse Cross-Section.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def plot_dataset(
    ply_path: Path,
    trajectory_path: Path,
    output_image: Path,
    title_prefix: str = "Dataset",
    voxel_size: float = 0.08,
    max_points: int = 500_000
):
    print(f"[*] Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)

    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    print(f"[*] Loading trajectory: {trajectory_path}")
    trj_pts = []
    if trajectory_path.exists():
        with open(trajectory_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        trj_pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        continue
    trj = np.array(trj_pts) if len(trj_pts) > 0 else np.zeros((0, 3))

    fig, axes = plt.subplots(3, 1, figsize=(15, 20), facecolor='#111115')

    # 1. Top View (X-Y plane)
    ax0 = axes[0]
    ax0.set_facecolor('#1a1a20')
    ax0.scatter(pts[:, 0], pts[:, 1], c=cols, s=0.9, alpha=0.85, rasterized=True)
    if len(trj) > 0:
        ax0.plot(trj[:, 0], trj[:, 1], color='#00ffcc', linewidth=2.2, label='SLAM Trajectory', zorder=4)
        ax0.scatter(trj[0, 0], trj[0, 1], color='#33ff33', s=50, zorder=5, label=f'Start ({trj[0, 0]:.1f}, {trj[0, 1]:.1f})')
        ax0.scatter(trj[-1, 0], trj[-1, 1], color='#ff3333', s=50, zorder=5, label=f'End ({trj[-1, 0]:.1f}, {trj[-1, 1]:.1f})')
    ax0.set_title(f'{title_prefix} - Top-Down View (X-Y Plane)', color='white', fontsize=14, fontweight='bold', pad=10)
    ax0.set_xlabel('X (meters)', color='white', fontsize=11)
    ax0.set_ylabel('Y (meters)', color='white', fontsize=11)
    ax0.tick_params(colors='white')
    ax0.axis('equal')
    ax0.grid(True, color='#333340', linestyle='--', alpha=0.5)
    ax0.legend(loc='upper right', facecolor='#22222a', edgecolor='none', labelcolor='white')

    # Detect principal axis
    dx = np.ptp(trj[:, 0]) if len(trj) > 0 else np.ptp(pts[:, 0])
    dy = np.ptp(trj[:, 1]) if len(trj) > 0 else np.ptp(pts[:, 1])

    # 2. Longitudinal Profile vs Z
    ax1 = axes[1]
    ax1.set_facecolor('#1a1a20')
    if dx >= dy:
        long_idx = 0
        long_label = 'Longitudinal X (meters)'
        trans_idx = 1
        trans_label = 'Latitudinal / Transverse Y (meters)'
    else:
        long_idx = 1
        long_label = 'Longitudinal Y (meters)'
        trans_idx = 0
        trans_label = 'Latitudinal / Transverse X (meters)'

    ax1.scatter(pts[:, long_idx], pts[:, 2], c=cols, s=0.9, alpha=0.85, rasterized=True)
    if len(trj) > 0:
        ax1.plot(trj[:, long_idx], trj[:, 2], color='#00ffcc', linewidth=2.2, label='Trajectory Elevation', zorder=4)
        ax1.scatter(trj[0, long_idx], trj[0, 2], color='#33ff33', s=50, zorder=5)
        ax1.scatter(trj[-1, long_idx], trj[-1, 2], color='#ff3333', s=50, zorder=5)
    ax1.set_title(f'{title_prefix} - Longitudinal Profile ({long_label.split()[0]}-Z Elevation)', color='white', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel(long_label, color='white', fontsize=11)
    ax1.set_ylabel('Elevation Z (meters)', color='white', fontsize=11)
    ax1.tick_params(colors='white')
    ax1.axis('equal')
    ax1.grid(True, color='#333340', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', facecolor='#22222a', edgecolor='none', labelcolor='white')

    # 3. Latitudinal / Transverse Cross-Section vs Z
    ax2 = axes[2]
    ax2.set_facecolor('#1a1a20')
    ax2.scatter(pts[:, trans_idx], pts[:, 2], c=cols, s=0.9, alpha=0.85, rasterized=True)
    if len(trj) > 0:
        ax2.plot(trj[:, trans_idx], trj[:, 2], color='#00ffcc', linewidth=2.2, label='Trajectory Cross-Section', zorder=4)
        ax2.scatter(trj[0, trans_idx], trj[0, 2], color='#33ff33', s=50, zorder=5)
        ax2.scatter(trj[-1, trans_idx], trj[-1, 2], color='#ff3333', s=50, zorder=5)
    ax2.set_title(f'{title_prefix} - Latitudinal / Transverse Cross-Section ({trans_label.split()[0]}-Z Profile)', color='white', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel(trans_label, color='white', fontsize=11)
    ax2.set_ylabel('Elevation Z (meters)', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.axis('equal')
    ax2.grid(True, color='#333340', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', facecolor='#22222a', edgecolor='none', labelcolor='white')

    plt.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_image), dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Orthogonal 3-view plot saved to: {output_image}")

def main():
    parser = argparse.ArgumentParser(description="Plot orthogonal views for colored point cloud + trajectory")
    parser.add_argument("--ply", type=str, required=True, help="Path to colorized pointcloud PLY")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory.txt")
    parser.add_argument("--output", type=str, required=True, help="Path to output PNG")
    parser.add_argument("--title", type=str, default="Dataset", help="Title prefix")
    parser.add_argument("--voxel-size", type=float, default=0.08, help="Downsample voxel size in meters")
    args = parser.parse_args()

    plot_dataset(
        Path(args.ply),
        Path(args.trajectory),
        Path(args.output),
        title_prefix=args.title,
        voxel_size=args.voxel_size
    )

if __name__ == '__main__':
    main()
