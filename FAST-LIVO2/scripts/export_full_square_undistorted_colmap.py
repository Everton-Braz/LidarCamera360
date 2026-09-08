#!/usr/bin/env python3
import os
import cv2
import sqlite3
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation as R

def build_undistortion_map(in_w, in_h, out_w, out_h, fov_deg, cam_params):
    fx, fy, cx, cy = cam_params[0], cam_params[1], cam_params[2], cam_params[3]
    k1, k2, p1, p2 = cam_params[4], cam_params[5], cam_params[6], cam_params[7]
    k3, k4, sx1, sy1 = cam_params[8], cam_params[9], cam_params[10], cam_params[11]
    
    f_pinhole = (out_w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
    cx_out, cy_out = out_w / 2.0, out_h / 2.0
    
    u_out, v_out = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x_pin = (u_out - cx_out) / f_pinhole
    y_pin = (v_out - cy_out) / f_pinhole
    
    r = np.sqrt(x_pin**2 + y_pin**2)
    theta = np.arctan(r)
    r_safe = np.where(r < 1e-6, 1.0, r)
    theta2 = theta**2
    theta4 = theta2**2
    theta6 = theta4 * theta2
    theta8 = theta4**2
    theta_d = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8)
    scale = np.where(r < 1e-6, 1.0, theta_d / r_safe)
    
    x_d = scale * x_pin
    y_d = scale * y_pin
    r_d2 = x_d**2 + y_d**2
    x_dd = x_d + 2.0*p1*x_d*y_d + p2*(r_d2 + 2.0*x_d**2) + sx1*r_d2
    y_dd = y_d + 2.0*p2*x_d*y_d + p1*(r_d2 + 2.0*y_d**2) + sy1*r_d2
    
    map_x = (fx * x_dd + cx).astype(np.float32)
    map_y = (fy * y_dd + cy).astype(np.float32)
    return map_x, map_y, f_pinhole, cx_out, cy_out

def main():
    print("===================================================================")
    print(" Exporting Full-Frame Square Undistorted COLMAP Dataset (3DGS)")
    print("===================================================================")
    
    src_images_dir = r'd:\APLICATIVOS\FAST-LIVO2\SMALL-DATASET-TEST\videos\VID_20260821_105436_00_269_dataset\images'
    src_colmap_dir = r'd:\APLICATIVOS\FAST-LIVO2\Log\small_test\Colmap_Metric_Fisheye'
    out_dir = r'd:\APLICATIVOS\FAST-LIVO2\Log\small_test\Colmap_Undistorted_3DGS'
    
    out_imgs_dir = os.path.join(out_dir, 'images')
    out_sparse_dir = os.path.join(out_dir, 'sparse')
    out_sparse_0 = os.path.join(out_sparse_dir, '0')
    
    os.makedirs(os.path.join(out_imgs_dir, 'cam0'), exist_ok=True)
    os.makedirs(os.path.join(out_imgs_dir, 'cam1'), exist_ok=True)
    os.makedirs(out_sparse_dir, exist_ok=True)
    os.makedirs(out_sparse_0, exist_ok=True)
    
    # 1. Parse Input Cameras
    cameras = {}
    with open(os.path.join(src_colmap_dir, 'cameras.txt')) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            cam_id = int(parts[0])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = params
            
    # Target Pinhole Image Size: 1536 x 1536 (FOV 110 degrees for wide, uncropped coverage)
    out_w, out_h = 1536, 1536
    fov_deg = 110.0
    
    # Build precomputed remap tables for cam0 (ID 1) and cam1 (ID 2)
    map_x0, map_y0, f0, cx0, cy0 = build_undistortion_map(3840, 3840, out_w, out_h, fov_deg, cameras[1])
    map_x1, map_y1, f1, cx1, cy1 = build_undistortion_map(3840, 3840, out_w, out_h, fov_deg, cameras.get(2, cameras[1]))
    
    # 2. Undistort all images to full square 1536x1536
    print(f"\n[1/4] Undistorting all frames to square {out_w}x{out_h} rectilinear images (FOV={fov_deg} deg)...")
    total_undist = 0
    for cam_name, map_x, map_y in [('cam0', map_x0, map_y0), ('cam1', map_x1, map_y1)]:
        c_dir = os.path.join(src_images_dir, cam_name)
        if not os.path.exists(c_dir):
            continue
        for fname in sorted(os.listdir(c_dir)):
            if not fname.endswith('.jpg'):
                continue
            src_p = os.path.join(c_dir, fname)
            dst_p = os.path.join(out_imgs_dir, cam_name, fname)
            
            img = cv2.imread(src_p)
            if img is None:
                continue
            undist = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
            cv2.imwrite(dst_p, undist, [cv2.IMWRITE_JPEG_QUALITY, 94])
            total_undist += 1
            if total_undist % 50 == 0:
                print(f"  Undistorted {total_undist} images...")
                
    print(f"  Finished undistorting {total_undist} full-frame square images!")
    
    # 3. Write Target cameras.txt
    print("\n[2/4] Writing standard PINHOLE cameras.txt...")
    with open(os.path.join(out_sparse_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {out_w} {out_h} {f0:.6f} {f0:.6f} {cx0:.1f} {cy0:.1f}\n")
        f.write(f"2 PINHOLE {out_w} {out_h} {f1:.6f} {f1:.6f} {cx1:.1f} {cy1:.1f}\n")
        
    # 4. Load points3D.txt (metric scale)
    print("\n[3/4] Processing 3D tie points and 2D observation tracks...")
    points3d = {}
    with open(os.path.join(src_colmap_dir, 'points3D.txt'), 'r') as f_in, open(os.path.join(out_sparse_dir, 'points3D.txt'), 'w') as f_out:
        for line in f_in:
            if line.startswith('#') or not line.strip():
                f_out.write(line)
                continue
            parts = line.split()
            pid = int(parts[0])
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            points3d[pid] = xyz
            f_out.write(line) # Write metric 3D point
            
    # 5. Transform 2D point tracks in images.txt to undistorted pinhole coordinates
    with open(os.path.join(src_colmap_dir, 'images.txt'), 'r') as f_in, open(os.path.join(out_sparse_dir, 'images.txt'), 'w') as f_out:
        lines = f_in.readlines()
        
    for i in range(0, len(lines), 2):
        line1 = lines[i]
        if line1.startswith('#') or not line1.strip():
            with open(os.path.join(out_sparse_dir, 'images.txt'), 'a') as f_out:
                f_out.write(line1)
            continue
            
        parts = line1.split()
        if len(parts) < 10:
            continue
            
        img_id = parts[0]
        qw, qx, qy, qz = [float(x) for x in parts[1:5]]
        tx, ty, tz = [float(x) for x in parts[5:8]]
        cam_id = int(parts[8])
        name = parts[9]
        
        R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
        t_w2c = np.array([tx, ty, tz])
        
        f_curr = f0 if cam_id == 1 else f1
        cx_curr = cx0 if cam_id == 1 else cx1
        cy_curr = cy0 if cam_id == 1 else cy1
        
        # Parse 2D-3D observation track on line 2
        line2 = lines[i+1] if (i+1) < len(lines) else ""
        obs_parts = line2.split()
        
        new_obs = []
        for j in range(0, len(obs_parts), 3):
            pid = int(obs_parts[j+2])
            if pid != -1 and pid in points3d:
                # Project 3D point into pinhole camera frame
                P_cam = R_w2c @ points3d[pid] + t_w2c
                if P_cam[2] > 0.1:
                    u_pin = f_curr * (P_cam[0] / P_cam[2]) + cx_curr
                    v_pin = f_curr * (P_cam[1] / P_cam[2]) + cy_curr
                    if 0 <= u_pin < out_w and 0 <= v_pin < out_h:
                        new_obs.append(f"{u_pin:.2f} {v_pin:.2f} {pid}")
                    else:
                        new_obs.append(f"-1 -1 -1")
                else:
                    new_obs.append(f"-1 -1 -1")
            else:
                new_obs.append(f"{obs_parts[j]} {obs_parts[j+1]} -1")
                
        with open(os.path.join(out_sparse_dir, 'images.txt'), 'a') as f_out:
            f_out.write(line1)
            f_out.write(" ".join(new_obs) + "\n")
            
    # 6. Convert to Binary format for sparse/0
    print("\n[4/4] Converting sparse model to binary for sparse/0...")
    colmap_exe = r'D:\APLICATIVOS\COLMAP 4.1.0\bin\colmap.exe'
    cmd_conv = [
        colmap_exe, 'model_converter',
        '--input_path', out_sparse_dir,
        '--output_path', out_sparse_0,
        '--output_type', 'BIN'
    ]
    subprocess.run(cmd_conv, check=True)
    
    # Copy txt files to sparse/0 as well
    for f in ['cameras.txt', 'images.txt', 'points3D.txt']:
        import shutil
        shutil.copy(os.path.join(out_sparse_dir, f), os.path.join(out_sparse_0, f))
        
    print("\n=== Full-Frame Square Undistorted 3DGS Dataset Ready at: ===")
    print(f"  {out_dir}")

if __name__ == '__main__':
    main()
