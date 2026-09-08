#!/usr/bin/env python3
"""
export_raw_fisheye_3dgs_colmap.py

Generates a complete COLMAP dataset configured for 3D Gaussian Splatting (3DGS)
and LichtFeld Studio that PRESERVES the raw circular fisheye images (no pinhole undistortion).

Directory Structure:
Log/<dataset_name>/Colmap_Fisheye_3DGS/
├── images/
│   ├── cam0/ (raw fisheye front images)
│   └── cam1/ (raw fisheye rear images)
├── sparse/0/
│   ├── cameras.bin / cameras.txt
│   ├── images.bin / images.txt
│   └── points3D.bin / points3D.txt
└── colorized_sparse_cloud.ply
"""

import os
import sys
import shutil
import subprocess
import numpy as np

def export_raw_fisheye_3dgs(
    src_colmap_metric=r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\Colmap_Metric_Fisheye",
    src_images_dir=r"d:\APLICATIVOS\FAST-LIVO2\SMALL-DATASET-TEST\videos\VID_20260821_105436_00_269_dataset\images",
    out_dir=r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\Colmap_Fisheye_3DGS"
):
    print("===================================================================")
    print(" Exporting Raw Dual-Fisheye COLMAP Dataset for 3DGS / LichtFeld")
    print(f" Source Metric COLMAP: {src_colmap_metric}")
    print(f" Source Images:        {src_images_dir}")
    print(f" Output Directory:     {out_dir}")
    print("===================================================================")

    images_out = os.path.join(out_dir, "images")
    sparse_out = os.path.join(out_dir, "sparse")
    sparse_0_out = os.path.join(sparse_out, "0")

    os.makedirs(images_out, exist_ok=True)
    os.makedirs(sparse_out, exist_ok=True)
    os.makedirs(sparse_0_out, exist_ok=True)

    # 1. Copy raw fisheye images into images/cam0 and images/cam1
    print("\n[1/4] Copying raw circular fisheye images...")
    for cam in ["cam0", "cam1"]:
        src_cam = os.path.join(src_images_dir, cam)
        dst_cam = os.path.join(images_out, cam)
        if os.path.exists(src_cam):
            if os.path.exists(dst_cam):
                shutil.rmtree(dst_cam)
            shutil.copytree(src_cam, dst_cam)
            img_count = len([f for f in os.listdir(dst_cam) if f.endswith(('.jpg', '.png'))])
            print(f"  [+] Copied {img_count} raw fisheye frames to images/{cam}")

    # 2. Copy and ensure text models in sparse/ and sparse/0/
    print("\n[2/4] Setting up Metric COLMAP text files...")
    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        src_f = os.path.join(src_colmap_metric, fname)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(sparse_out, fname))
            shutil.copy(src_f, os.path.join(sparse_0_out, fname))
            print(f"  [+] Copied {fname}")

    # 3. Export Colorized Sparse PLY for 3DGS initial point cloud
    print("\n[3/4] Exporting Colorized Sparse PLY Point Cloud...")
    pts_file = os.path.join(sparse_out, "points3D.txt")
    ply_path = os.path.join(out_dir, "colorized_sparse_cloud.ply")

    pts_xyz = []
    pts_rgb = []
    if os.path.exists(pts_file):
        with open(pts_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.split()
                if len(p) >= 8:
                    pts_xyz.append([float(p[1]), float(p[2]), float(p[3])])
                    pts_rgb.append([int(p[4]), int(p[5]), int(p[6])])

    if pts_xyz:
        xyz = np.array(pts_xyz, dtype=np.float32)
        rgb = np.array(pts_rgb, dtype=np.uint8)
        num_pts = len(xyz)
        with open(ply_path, "w", encoding="ascii") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {num_pts}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for i in range(num_pts):
                f.write(f"{xyz[i,0]:.4f} {xyz[i,1]:.4f} {xyz[i,2]:.4f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}\n")
        print(f"  [+] Saved: {ply_path} ({num_pts} tie points, {os.path.getsize(ply_path)/(1024*1024):.2f} MB)")

    # 4. Convert model to binary for sparse/0/
    print("\n[4/4] Converting sparse model to Binary (.bin) for 3DGS loaders...")
    colmap_exe = r"D:\APLICATIVOS\COLMAP 4.1.0\bin\colmap.exe"
    if os.path.exists(colmap_exe):
        cmd = [
            colmap_exe, "model_converter",
            "--input_path", sparse_0_out,
            "--output_path", sparse_0_out,
            "--output_type", "BIN"
        ]
        subprocess.run(cmd, check=True)
        print(f"  [+] Binary model compiled in: {sparse_0_out}")

    print("\n===================================================================")
    print(f" [SUCCESS] Raw Fisheye 3DGS Dataset Ready at:")
    print(f"   {out_dir}")
    print("===================================================================")

if __name__ == "__main__":
    export_raw_fisheye_3dgs()
