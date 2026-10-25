#!/usr/bin/env python3
"""
Test Decoupled Calibrator on HarmonyMall Dataset
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scripts.decoupled_calibrator import calibrate_decoupled_extrinsics
from scripts.pipeline_auto_calibrator_and_colorizer import load_trajectory, load_colmap_images, frame_time

def main():
    dataset_dir = Path('D:/ARQUIVOS_TESTE_2/HarmonyMall/OUTPUT')
    sparse_bin = dataset_dir / 'sfm_workspace_colmap_dense_0.50fps/sparse/0/images.bin'
    if not sparse_bin.exists():
        sparse_bin = dataset_dir / 'sparse/0/images.bin'
    
    trj_path = list(dataset_dir.glob('slam_raw_*/result/*Scan*.txt'))[0]
    base_calib_path = Path('calibracao_rigida_raven_insta360.json')
    
    with open(base_calib_path) as f:
        base_calib = json.load(f)
    nominal_T0 = np.array(base_calib['T_lidar_to_cam0_rigid_4x4'])
    nominal_T1 = np.array(base_calib['T_lidar_to_cam1_rigid_4x4'])
    
    images = load_colmap_images(sparse_bin)
    t_slam, pos_slam, rot_slam = load_trajectory(trj_path)
    slerp = Slerp(t_slam, rot_slam)
    t_min, t_max = t_slam[0], t_slam[-1]
    
    dt_sync_init = 3.336827  # from IMU gyro sync
    fps = 0.50
    
    paired_cam0 = []
    paired_cam1 = []
    for im in images:
        name = im['name']
        t_vid = frame_time(dataset_dir, name, fps)
        t_lidar = t_min + (t_vid - dt_sync_init)
        if t_lidar < t_min or t_lidar > t_max:
            continue
        R_cw = im['R_cw']
        C = im['C']
        T_w_c = np.eye(4)
        T_w_c[:3, :3] = R_cw.T
        T_w_c[:3, 3] = C
        if 'cam0' in name or 'front' in name.lower():
            paired_cam0.append((T_w_c, np.eye(4), t_vid))
        elif 'cam1' in name or 'rear' in name.lower():
            paired_cam1.append((T_w_c, np.eye(4), t_vid))
            
    print(f'Paired frames: Cam0={len(paired_cam0)}, Cam1={len(paired_cam1)}')
    
    print('\n[*] Running Decoupled Calibration for Cam0...')
    R0, t0, dt0, diag0 = calibrate_decoupled_extrinsics(
        paired_data=paired_cam0,
        slerp_slam=slerp,
        slam_times=t_slam,
        slam_positions=pos_slam,
        nominal_R_lc=nominal_T0[:3, :3],
        nominal_t_lc=nominal_T0[:3, 3],
        dt_init=dt_sync_init
    )
    
    print('\n[*] Running Decoupled Calibration for Cam1...')
    R1, t1, dt1, diag1 = calibrate_decoupled_extrinsics(
        paired_data=paired_cam1,
        slerp_slam=slerp,
        slam_times=t_slam,
        slam_positions=pos_slam,
        nominal_R_lc=nominal_T1[:3, :3],
        nominal_t_lc=nominal_T1[:3, 3],
        dt_init=dt_sync_init
    )
    
    print('\n================ CALIBRATION SUMMARY ================')
    print(f'Cam0 t: {t0} (norm: {np.linalg.norm(t0)*100:.2f} cm)')
    print(f'Cam1 t: {t1} (norm: {np.linalg.norm(t1)*100:.2f} cm)')
    print(f'Refined dt: Cam0={dt0:.6f}s, Cam1={dt1:.6f}s (Diff from IMU: {(dt0-dt_sync_init)*1000:+.2f} ms)')
    
    # Save test rig calibration
    out_rig = dataset_dir / 'decoupled_rig_calibration.json'
    T0_mat = np.eye(4)
    T0_mat[:3, :3] = R0
    T0_mat[:3, 3] = t0
    T1_mat = np.eye(4)
    T1_mat[:3, :3] = R1
    T1_mat[:3, 3] = t1
    
    calib_out = dict(base_calib)
    calib_out['calibration_version'] = 3
    calib_out['method'] = 'decoupled_sfm_slam_argon'
    calib_out['T_lidar_to_cam0_rigid_4x4'] = T0_mat.tolist()
    calib_out['T_lidar_to_cam1_rigid_4x4'] = T1_mat.tolist()
    calib_out['dt_sync_refined'] = float(0.5 * (dt0 + dt1))
    calib_out['diagnostics'] = {'cam0': diag0, 'cam1': diag1}
    
    with open(out_rig, 'w') as f:
        json.dump(calib_out, f, indent=2)
    print(f'\n[+] Saved decoupled rig calibration to: {out_rig}')

if __name__ == '__main__':
    main()
