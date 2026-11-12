#!/usr/bin/env python3
"""
Diagnose why Spirula COLMAP Sim(3) alignment is rejected against LiDAR SLAM
in both HarmonyMall and PIER datasets.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path('C:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator')
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_images, load_trajectory, solve_umeyama_sim3, frame_time

def inspect_dataset(name, p):
    base = Path(p)
    sparse_dir = base / 'sparse/0'
    if not sparse_dir.exists():
        print(f'[-] {name}: sparse/0 not found')
        return

    imgs = load_colmap_images(sparse_dir / 'images.bin')
    cam0 = [im for im in imgs if im['name'].startswith('cam0/')]
    def get_frame_num(n):
        d = ''.join(filter(str.isdigit, Path(n).stem))
        return int(d) if d else 0
    cam0.sort(key=lambda x: get_frame_num(x['name']))

    # find trajectory
    txts = list(base.glob('slam*/result/Raven_3DMakerPro_Scan.txt'))
    if not txts:
        print(f'[-] {name}: trajectory not found')
        return
    trj_path = txts[0]
    t_slam, pos_slam, rot_slam = load_trajectory(trj_path)

    print(f'=== {name} ===')
    print(f'COLMAP cam0 count: {len(cam0)} | SLAM duration: {t_slam[-1] - t_slam[0]:.2f}s ({len(t_slam)} poses)')
    
    centers_colmap = np.array([im['C'] for im in cam0])
    print(f'COLMAP cam0 centers: min={centers_colmap.min(axis=0)}, max={centers_colmap.max(axis=0)}')
    print(f'SLAM pos: min={pos_slam.min(axis=0)}, max={pos_slam.max(axis=0)}')

    # Let us compute the trajectory distance traveled in COLMAP vs SLAM
    d_col = np.sum(np.linalg.norm(np.diff(centers_colmap, axis=0), axis=1))
    d_slam = np.sum(np.linalg.norm(np.diff(pos_slam, axis=0), axis=1))
    print(f'Distance traveled: COLMAP={d_col:.2f} (arb units), SLAM={d_slam:.2f} meters')
    if d_col > 0:
        print(f'Empirical scale ratio: {d_slam / d_col:.4f}')

if __name__ == '__main__':
    inspect_dataset('HarmonyMall', 'D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
    inspect_dataset('PIER', 'D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output')
