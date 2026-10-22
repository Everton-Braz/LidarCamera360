"""Generate high-resolution top-down colored point cloud view."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import open3d as o3d

def main():
    ply_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output/deliverables/lidar_colored.ply")
    out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/ARQUIVOS_TESTE_2/PIER/pier_topdown_view.png")

    print(f"[*] Reading {ply_path}...")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    print(f"[+] Points: {len(pts):,}, Colors: {len(colors):,}")

    # Filter uncolored/black points if any
    valid = np.any(colors > 0.02, axis=1)
    pts = pts[valid]
    colors = colors[valid]

    n = len(pts)
    if n > 1500000:
        step = n // 1500000
        pts = pts[::step]
        colors = colors[::step]
        print(f"[*] Subsampled to {len(pts):,} points for rendering")

    fig, ax = plt.subplots(figsize=(14, 14), facecolor='#111111')
    ax.set_facecolor('#111111')

    # Top-Down (X vs Y)
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=0.4, alpha=0.85, edgecolors='none', rasterized=True)
    ax.set_title("PIER - Top-Down View (X vs Y)\n1 FPS | PPISP-Bilateral: SIM | Mask: SIM", color='white', fontsize=16, pad=15)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=300, bbox_inches='tight', facecolor='#111111')
    print(f"[+] Saved top-down view to {out_png}")

if __name__ == '__main__':
    main()
