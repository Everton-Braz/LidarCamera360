#!/usr/bin/env python3
"""
apply_camera_rotation_minus90.py

Rotates only the camera registrations (images.txt, binary images.bin, and image EXIF GPS tags)
around the X-AXIS (Blue Arrow / East) for model_3_upright_yaw180_reverse, while keeping points3D.txt unchanged.
"""

import os
import sys
import shutil
import json
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R
import subprocess

try:
    import piexif
except ImportError:
    piexif = None

sys.path.append(r'd:\APLICATIVOS\FAST-LIVO2')
import scripts.fix_realityscan_colmap_rotation_and_import as fix
import scripts.generate_perfect_realityscan_models as gen

def rotate_cameras_x_axis(rx_degrees=90.0, target_dir=r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\RealityScan_Dataset"):
    sparse_dir = os.path.join(target_dir, "sparse")
    sparse_0_dir = os.path.join(sparse_dir, "0")
    images_dir = os.path.join(target_dir, "images")

    os.makedirs(sparse_dir, exist_ok=True)
    os.makedirs(sparse_0_dir, exist_ok=True)

    print(f"[*] Rotating camera registrations by {rx_degrees:+.1f}° around X-AXIS (East) in: {target_dir}")

    # 1. Load cameras & images from base source
    src_dir = r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\Colmap_Undistorted_3DGS"
    cameras = fix.load_cameras(os.path.join(src_dir, "sparse", "cameras.txt"))
    images = fix.load_images(os.path.join(src_dir, "sparse", "images.txt"))

    # Model 3 base transformation on world: R_pts = R_z(180) @ R_base
    R_base = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0],
        [0.0, -1.0,  0.0]
    ], dtype=np.float64)
    R_pts = R.from_euler("z", 180, degrees=True).as_matrix() @ R_base

    # Camera rotation: Rx(rx_degrees) applied on top of R_pts
    R_rx = R.from_euler("x", rx_degrees, degrees=True).as_matrix()
    R_cam_total = R_rx @ R_pts

    transformed_images = {}
    cam_centers_list = []

    for img_id, img in images.items():
        C_new = R_cam_total @ img["C_world"]
        r_wc_new = R_cam_total @ img["r_wc"]
        r_cw_new = r_wc_new.T
        t_cw_new = -r_cw_new @ C_new

        q_scipy = R.from_matrix(r_cw_new).as_quat() # [qx, qy, qz, qw]
        q_wxyz = np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])

        transformed_images[img_id] = {
            "id": img_id,
            "q_cw": q_wxyz,
            "t_cw": t_cw_new,
            "C_world": C_new,
            "r_wc": r_wc_new,
            "r_cw": r_cw_new,
            "cam_id": img["cam_id"],
            "name": img["name"],
            "points2D_line": img["points2D_line"]
        }
        cam_centers_list.append(C_new)

    cam_centers_arr = np.array(cam_centers_list)
    print(f"[+] Transformed {len(transformed_images)} camera poses with Rx({rx_degrees:+.1f}°).")
    print(f"    New Camera X range: [{np.min(cam_centers_arr[:,0]):.2f}m, {np.max(cam_centers_arr[:,0]):.2f}m] span={np.ptp(cam_centers_arr[:,0]):.2f}m")
    print(f"    New Camera Y range: [{np.min(cam_centers_arr[:,1]):.2f}m, {np.max(cam_centers_arr[:,1]):.2f}m] span={np.ptp(cam_centers_arr[:,1]):.2f}m")
    print(f"    New Camera Z range: [{np.min(cam_centers_arr[:,2]):.2f}m, {np.max(cam_centers_arr[:,2]):.2f}m] span={np.ptp(cam_centers_arr[:,2]):.2f}m")

    # 2. Write updated images.txt into sparse/ and sparse/0/
    for dst_dir in [sparse_dir, sparse_0_dir]:
        img_txt_path = os.path.join(dst_dir, "images.txt")
        with open(img_txt_path, "w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for img_id in sorted(transformed_images.keys()):
                img = transformed_images[img_id]
                qw, qx, qy, qz = img["q_cw"]
                tx, ty, tz = img["t_cw"]
                f.write(f"{img['id']} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.6f} {ty:.6f} {tz:.6f} {img['cam_id']} {img['name']}\n")
                if img["points2D_line"]:
                    f.write(img["points2D_line"] + "\n")
                else:
                    f.write("\n")
        print(f"[+] Updated {img_txt_path}")

    # 3. Recompile binary model
    colmap_exe = r"D:\APLICATIVOS\COLMAP 4.1.0\bin\colmap.exe"
    cmd = [
        colmap_exe, "model_converter",
        "--input_path", sparse_0_dir,
        "--output_path", sparse_0_dir,
        "--output_type", "BIN"
    ]
    subprocess.run(cmd, check=True)
    print(f"[+] Recompiled binary COLMAP model in: {sparse_0_dir}")

    # 4. Synchronize EXIF GPS
    with open(r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\Georeferenced\local_to_geographic.json", "r", encoding="utf-8") as f:
        geo_dict = json.load(f)

    transformer = pyproj.Transformer.from_crs("EPSG:31984", "EPSG:4326", always_xy=True)
    R_geo_orig = np.array(geo_dict["rotation_matrix"])
    t_geo = np.array(geo_dict["translation_utm_m"])
    R_world_to_geo = R_geo_orig @ R_cam_total.T

    gen.embed_exif_gps_to_images(
        transformed_images,
        images_dir,
        transformer,
        geo_dict["timestamp_sync"]["video_start_epoch"],
        R_world_to_geo,
        t_geo
    )

    print(f"\n[SUCCESS] Camera registrations rotated {rx_degrees:+.1f}° around X-Axis (East) successfully for RealityScan_Dataset!")

if __name__ == "__main__":
    angle = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    rotate_cameras_x_axis(angle)
