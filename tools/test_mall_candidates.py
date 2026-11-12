import json, math, os, sys, time
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt
import open3d as o3d

WORKSPACE = Path('C:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator')
DATASET_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
OUT_DIR = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/candidates_zoom')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print('[*] Loading PCD cloud for HarmonyMall...')
pcd_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/pcd/all_raw_points.pcd'
trj_path = DATASET_DIR / 'slam_raw_9c24d84a679f1e63715d/result/Raven_3DMakerPro_Scan.txt'

pcd = o3d.io.read_point_cloud(str(pcd_path))
pts_all = np.asarray(pcd.points, dtype=np.float32)

# Corridor window around X=[0, 8], Y=[-25, -15], Z=[-1.5, 3.5]
cx, cy = 2.5, -20.0
hw = 5.0
mask = (pts_all[:, 0] >= cx - hw) & (pts_all[:, 0] <= cx + hw) & \n       (pts_all[:, 1] >= cy - hw) & (pts_all[:, 1] <= cy + hw) & \n       (pts_all[:, 2] >= -1.5) & (pts_all[:, 2] <= 3.5)
pts = pts_all[mask]
print(f'[+] Filtered {len(pts):,} points in 10m x 10m window.')

trj = np.loadtxt(trj_path)
t_slam = trj[:, 0]
pos_slam = trj[:, 1:4]
quats = trj[:, 4:8]
slerp = Slerp(t_slam, R.from_quat(quats))

# Load factory calib vs recalibrated calib
with open(WORKSPACE / 'calibracao_rigida_raven_insta360.json', 'r') as f:
    calib_factory = json.load(f)
with open(DATASET_DIR / 'rig_calibration.json', 'r') as f:
    calib_recal = json.load(f)

print('[+] Calibrations loaded successfully.')
