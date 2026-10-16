import sys
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_colored_ply(ply_path, max_pts=350000):
    with open(ply_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('latin1', errors='ignore').strip()
            header.append(line)
            if line == 'end_header':
                break
        num_verts = 0
        for h in header:
            if h.startswith('element vertex'):
                num_verts = int(h.split()[-1])
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        data = np.fromfile(f, dtype=dtype, count=num_verts)
        pts = np.vstack([data['x'], data['y'], data['z']]).T
        rgb = np.vstack([data['r'], data['g'], data['b']]).T / 255.0
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        rgb = rgb[valid]
        if len(pts) > max_pts:
            idx = np.random.choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
            rgb = rgb[idx]
        return pts, rgb

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

def generate_colored_views(dataset_dir, output_png, title):
    dataset_dir = Path(dataset_dir)
    ply_candidates = list(dataset_dir.glob('deliverables/*colored*.ply')) + list(dataset_dir.glob('deliverables/*.ply'))
    if not ply_candidates:
        print(f'Nenhum PLY colorido em {dataset_dir}')
        return
    ply_path = ply_candidates[0]
    print(f'Carregando nuvem colorida por câmera: {ply_path.name}...')
    pts, rgb = load_colored_ply(ply_path, max_pts=350000)
    print(f'  {len(pts):,} pontos coloridos carregados.')

    traj_candidates = list(dataset_dir.glob('slam_refined*/**/lidar_poses.txt')) + \
                      list(dataset_dir.glob('slam_raw*/**/lidar_poses.txt')) + \
                      list(dataset_dir.glob('slam_out/**/lidar_poses.txt')) + \
                      list(dataset_dir.glob('slam_refined*/**/Raven_3DMakerPro_Scan.txt')) + \
                      list(dataset_dir.glob('slam_raw*/**/Raven_3DMakerPro_Scan.txt')) + \
                      list(dataset_dir.glob('slam_out/**/Raven_3DMakerPro_Scan.txt'))
    traj = None
    for tc in traj_candidates:
        if tc.is_file():
            traj = load_trajectory(tc)
            if traj is not None and len(traj) > 0:
                print(f'  Trajetória: {tc.name} ({len(traj)} poses)')
                break

    fig = plt.figure(figsize=(20, 6.5), dpi=160, facecolor='#111111')
    fig.suptitle(f'{title}\nNuvem de Pontos Colorida com Imagens das Câmeras ({ply_path.name}: {len(pts):,} pts)',
                 fontsize=14, fontweight='bold', color='white', y=0.98)

    # 1. Top-Down (X vs Y)
    ax1 = fig.add_subplot(1, 3, 1, facecolor='#1e1e1e')
    ax1.set_title('Vista Superior (Top-Down X-Y)', fontsize=12, fontweight='bold', color='white')
    ax1.scatter(pts[:, 0], pts[:, 1], c=rgb, s=0.6, alpha=0.85, rasterized=True)
    if traj is not None:
        ax1.plot(traj[:, 1], traj[:, 2], color='#ff3333', linewidth=2.2, label='Trajetória SLAM', zorder=10)
        ax1.scatter(traj[0, 1], traj[0, 2], color='#00ff66', s=60, marker='o', edgecolors='black', label='Início', zorder=11)
        ax1.scatter(traj[-1, 1], traj[-1, 2], color='#ff00ff', s=60, marker='X', edgecolors='black', label='Fim', zorder=11)
        leg = ax1.legend(loc='upper right', fontsize=8, facecolor='#222222', edgecolor='gray')
        for text in leg.get_texts():
            text.set_color('white')
    ax1.set_xlabel('X (m)', color='white')
    ax1.set_ylabel('Y (m)', color='white')
    ax1.axis('equal')
    ax1.tick_params(colors='white')
    ax1.grid(True, linestyle=':', alpha=0.3, color='gray')

    # 2. Corte Longitudinal (X vs Z)
    ax2 = fig.add_subplot(1, 3, 2, facecolor='#1e1e1e')
    ax2.set_title('Corte Longitudinal (X-Z)', fontsize=12, fontweight='bold', color='white')
    ax2.scatter(pts[:, 0], pts[:, 2], c=rgb, s=0.6, alpha=0.85, rasterized=True)
    if traj is not None:
        ax2.plot(traj[:, 1], traj[:, 3], color='#ff3333', linewidth=2.2, label='Trajetória SLAM', zorder=10)
        ax2.scatter(traj[0, 1], traj[0, 3], color='#00ff66', s=60, marker='o', edgecolors='black', zorder=11)
        ax2.scatter(traj[-1, 1], traj[-1, 3], color='#ff00ff', s=60, marker='X', edgecolors='black', zorder=11)
    ax2.set_xlabel('X (m)', color='white')
    ax2.set_ylabel('Z / Altura (m)', color='white')
    ax2.axis('equal')
    ax2.tick_params(colors='white')
    ax2.grid(True, linestyle=':', alpha=0.3, color='gray')

    # 3. Corte Transversal (Y vs Z)
    ax3 = fig.add_subplot(1, 3, 3, facecolor='#1e1e1e')
    ax3.set_title('Corte Transversal (Y-Z)', fontsize=12, fontweight='bold', color='white')
    ax3.scatter(pts[:, 1], pts[:, 2], c=rgb, s=0.6, alpha=0.85, rasterized=True)
    if traj is not None:
        ax3.plot(traj[:, 2], traj[:, 3], color='#ff3333', linewidth=2.2, label='Trajetória SLAM', zorder=10)
        ax3.scatter(traj[0, 2], traj[0, 3], color='#00ff66', s=60, marker='o', edgecolors='black', zorder=11)
        ax3.scatter(traj[-1, 2], traj[-1, 3], color='#ff00ff', s=60, marker='X', edgecolors='black', zorder=11)
    ax3.set_xlabel('Y (m)', color='white')
    ax3.set_ylabel('Z / Altura (m)', color='white')
    ax3.axis('equal')
    ax3.tick_params(colors='white')
    ax3.grid(True, linestyle=':', alpha=0.3, color='gray')

    plt.tight_layout()
    plt.savefig(output_png, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] Plot salvo com sucesso em: {output_png}')

if __name__ == '__main__':
    generate_colored_views(sys.argv[1], sys.argv[2], sys.argv[3])
