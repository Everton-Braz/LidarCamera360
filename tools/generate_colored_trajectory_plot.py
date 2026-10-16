import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_trajectory(output_dir):
    patterns = [
        "**/*trajectory*.txt", "**/*poses*.txt", "**/*.tum",
        "**/*odom*.txt", "**/*slam*.txt", "**/*traj*.csv"
    ]
    for pat in patterns:
        files = glob.glob(os.path.join(output_dir, pat), recursive=True)
        for f in sorted(files, key=os.path.getsize, reverse=True):
            try:
                data = np.loadtxt(f, comments='#')
                if data.ndim == 2 and data.shape[0] > 2 and data.shape[1] >= 4:
                    xyz = data[:, 1:4]
                    return xyz, os.path.basename(f)
                elif data.ndim == 2 and data.shape[0] > 2 and data.shape[1] == 3:
                    return data[:, :3], os.path.basename(f)
            except Exception:
                continue
    return None, None

def load_colored_cloud(output_dir, max_points=250000):
    cloud_files = []
    for ext in ("**/*.laz", "**/*.las", "**/*.ply"):
        cloud_files.extend(glob.glob(os.path.join(output_dir, ext), recursive=True))
    
    cloud_files.sort(key=lambda x: ('color' in os.path.basename(x).lower(), os.path.getsize(x)), reverse=True)

    for cf in cloud_files:
        ext = os.path.splitext(cf)[1].lower()
        try:
            if ext in ('.las', '.laz'):
                import laspy
                las = laspy.read(cf)
                pts = np.vstack((las.x, las.y, las.z)).T
                if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
                    r = np.asarray(las.red, dtype=np.float32)
                    g = np.asarray(las.green, dtype=np.float32)
                    b = np.asarray(las.blue, dtype=np.float32)
                    max_val = max(r.max(), g.max(), b.max(), 1.0)
                    scale = 65535.0 if max_val > 255.0 else 255.0
                    rgb = np.vstack((r, g, b)).T / scale
                    rgb = np.clip(rgb, 0.0, 1.0)
                    if np.std(rgb) > 0.01:
                        return subsample(pts, rgb, max_points), os.path.basename(cf), len(pts), True
                return subsample(pts, None, max_points), os.path.basename(cf), len(pts), False

            elif ext == '.ply':
                try:
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(cf)
                    pts = np.asarray(pcd.points)
                    if len(pts) == 0:
                        continue
                    if pcd.has_colors():
                        rgb = np.asarray(pcd.colors)
                        if np.std(rgb) > 0.01:
                            return subsample(pts, rgb, max_points), os.path.basename(cf), len(pts), True
                    return subsample(pts, None, max_points), os.path.basename(cf), len(pts), False
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Falha ao ler {cf}: {e}")
            continue
    return (None, None), None, 0, False

def subsample(pts, rgb, max_points):
    n = len(pts)
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        pts = pts[idx]
        if rgb is not None:
            rgb = rgb[idx]
    mask = np.ones(len(pts), dtype=bool)
    for axis in range(3):
        p1, p99 = np.percentile(pts[:, axis], [0.5, 99.5])
        span = max(p99 - p1, 1.0)
        mask &= (pts[:, axis] >= p1 - 0.5 * span) & (pts[:, axis] <= p99 + 0.5 * span)
    pts = pts[mask]
    if rgb is not None:
        rgb = rgb[mask]
    return pts, rgb

def main():
    if len(sys.argv) < 3:
        print("Uso: python generate_colored_trajectory_plot.py <output_dir> <title>")
        sys.exit(1)

    output_dir = sys.argv[1]
    title = sys.argv[2]
    out_png = os.path.join(output_dir, "plot_colored_trajectory.png")

    traj, traj_name = load_trajectory(output_dir)
    (pts, rgb), cloud_name, total_pts, has_rgb = load_colored_cloud(output_dir)

    if traj is None and pts is None:
        print(f"[ERROR] Nenhuma trajetória ou nuvem encontrada em {output_dir}")
        sys.exit(2)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 9), dpi=160)
    fig.suptitle(f"LidarCamera360 — {title}", fontsize=16, fontweight='bold', color='#00ffcc', y=0.97)

    ax1 = fig.add_subplot(1, 2, 1)
    if pts is not None:
        if has_rgb and rgb is not None:
            ax1.scatter(pts[:, 0], pts[:, 1], c=rgb, s=0.6, alpha=0.85, rasterized=True)
        else:
            sc = ax1.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='turbo', s=0.6, alpha=0.8, rasterized=True)
            plt.colorbar(sc, ax=ax1, label='Elevação Z (m)', fraction=0.046, pad=0.04)

    dist_total = 0.0
    if traj is not None and len(traj) > 1:
        diffs = np.diff(traj, axis=0)
        dist_total = float(np.sum(np.linalg.norm(diffs, axis=1)))
        ax1.plot(traj[:, 0], traj[:, 1], color='#ff00ff', linewidth=2.2, label=f'Trajetória ({dist_total:.1f}m)')
        ax1.scatter(traj[0, 0], traj[0, 1], color='#00ff00', s=90, zorder=10, edgecolors='black', label='Início')
        ax1.scatter(traj[-1, 0], traj[-1, 1], color='#ff3300', s=90, zorder=10, edgecolors='black', label='Fim')
        ax1.legend(loc='upper right', framealpha=0.8)

    ax1.set_title(f"Vista Superior 2D (XY) | Nuvem: {cloud_name or 'N/A'} ({'RGB' if has_rgb else 'Z-Color'})", fontsize=12)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.axis('equal')
    ax1.grid(True, linestyle='--', alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if pts is not None:
        step = max(1, len(pts) // 60000)
        pts3d = pts[::step]
        if has_rgb and rgb is not None:
            ax2.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c=rgb[::step], s=0.5, alpha=0.6, rasterized=True)
        else:
            ax2.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c=pts3d[:, 2], cmap='turbo', s=0.5, alpha=0.6, rasterized=True)

    if traj is not None and len(traj) > 1:
        ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='#00ffff', linewidth=2.5, label='Trajetória 3D')
        ax2.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='#00ff00', s=70, label='Início')
        ax2.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='#ff3300', s=70, label='Fim')
        ax2.legend(loc='upper right')

    stats_txt = (
        f"Pontos Totais: {total_pts:,} | Poses SLAM: {len(traj) if traj is not None else 0} | "
        f"Distância: {dist_total:.2f} m | Traj File: {traj_name or 'N/A'}"
    )
    ax2.set_title("Vista 3D Isométrica (Nuvem Colorida + Trajetória)", fontsize=12)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")
    fig.text(0.5, 0.02, stats_txt, ha='center', fontsize=11, color='#cccccc',
             bbox=dict(facecolor='#222222', edgecolor='#555555', boxstyle='round,pad=0.4'))

    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[OK] Plot salvo em: {out_png}")

if __name__ == '__main__':
    main()
