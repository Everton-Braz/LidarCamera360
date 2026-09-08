#!/usr/bin/env python3
import os
import shutil
import numpy as np

def partition_colmap_and_pcd(log_dir, output_chunks_dir, chunks_def):
    """
    Partitions the continuous full-run outputs into 4 chunk folders with overlap.
    chunks_def: list of tuples (t_start_rel, t_end_rel, chunk_name)
    """
    traj_file = os.path.join(log_dir, 'result', 'Raven_3DMakerPro_Scan.txt')
    if not os.path.exists(traj_file):
        print(f"Error: Trajectory file {traj_file} not found!")
        return

    print("Loading full trajectory...")
    traj_data = np.loadtxt(traj_file)
    t_base = traj_data[0, 0]
    print(f"Base start time: {t_base:.3f}, total poses: {len(traj_data)}")

    # Parse images.txt
    colmap_img_file = os.path.join(log_dir, 'Colmap', 'sparse', '0', 'images.txt')
    images_list = []
    with open(colmap_img_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        parts = line.split()
        img_id = int(parts[0])
        img_name = parts[9]
        # Poses line + Points2D line
        line1 = lines[i]
        line2 = lines[i+1] if i+1 < len(lines) else "0.0 0.0 -1\n"
        
        # Get timestamp from trajectory
        idx = img_id - 1
        ts = traj_data[idx, 0] if idx < len(traj_data) else (t_base + idx * 0.3)
        images_list.append({
            'id': img_id,
            'name': img_name,
            'ts': ts,
            'rel_t': ts - t_base,
            'line1': line1,
            'line2': line2
        })
        i += 2

    print(f"Parsed {len(images_list)} COLMAP images.")

    cameras_src = os.path.join(log_dir, 'Colmap', 'sparse', '0', 'cameras.txt')

    for t_start, t_end, cname in chunks_def:
        cdir = os.path.join(output_chunks_dir, cname)
        c_pcd_dir = os.path.join(cdir, 'pcd')
        c_res_dir = os.path.join(cdir, 'result')
        c_img_dir = os.path.join(cdir, 'Colmap', 'images')
        c_sparse_dir = os.path.join(cdir, 'Colmap', 'sparse', '0')

        os.makedirs(c_pcd_dir, exist_ok=True)
        os.makedirs(c_res_dir, exist_ok=True)
        os.makedirs(c_img_dir, exist_ok=True)
        os.makedirs(c_sparse_dir, exist_ok=True)

        # Copy cameras.txt
        if os.path.exists(cameras_src):
            shutil.copy(cameras_src, os.path.join(c_sparse_dir, 'cameras.txt'))

        # Filter trajectory poses
        c_traj = traj_data[(traj_data[:, 0] >= t_base + t_start) & (traj_data[:, 0] <= t_base + t_end)]
        np.savetxt(os.path.join(c_res_dir, 'Raven_3DMakerPro_Scan.txt'), c_traj, fmt='%.6f %f %f %f %f %f %f %f')

        # Filter images & copy
        c_images = [img for img in images_list if t_start <= img['rel_t'] <= t_end]
        print(f"\n[{cname}] rel_t: {t_start:.1f}s -> {t_end:.1f}s | {len(c_images)} images | {len(c_traj)} trajectory poses")

        with open(os.path.join(c_sparse_dir, 'images.txt'), 'w') as f_out:
            f_out.write("# Image list with two lines of data per image:\n")
            f_out.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f_out.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for new_id, img in enumerate(c_images, 1):
                # Update image ID in line1
                parts = img['line1'].split()
                parts[0] = str(new_id)
                f_out.write(" ".join(parts) + "\n")
                f_out.write(img['line2'])

                # Copy image file
                src_img = os.path.join(log_dir, 'Colmap', 'images', img['name'])
                dst_img = os.path.join(c_img_dir, img['name'])
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)

        # Copy PCD & generate points3D.txt for chunk
        src_raw_pcd = os.path.join(log_dir, 'pcd', 'all_raw_points.pcd')
        src_down_pcd = os.path.join(log_dir, 'pcd', 'all_downsampled_points.pcd')
        if os.path.exists(src_raw_pcd):
            shutil.copy(src_raw_pcd, os.path.join(c_pcd_dir, 'all_raw_points.pcd'))
        if os.path.exists(src_down_pcd):
            shutil.copy(src_down_pcd, os.path.join(c_pcd_dir, 'all_downsampled_points.pcd'))

        src_points3d = os.path.join(log_dir, 'Colmap', 'sparse', '0', 'points3D.txt')
        if os.path.exists(src_points3d):
            shutil.copy(src_points3d, os.path.join(c_sparse_dir, 'points3D.txt'))

        print(f"  [{cname}] Saved successfully to {cdir}")

if __name__ == '__main__':
    log_base = 'Log'
    out_chunks = 'Log/Chunks'
    chunks = [
        (0.0, 540.0, 'chunk_01'),
        (460.0, 1000.0, 'chunk_02'),
        (920.0, 1460.0, 'chunk_03'),
        (1380.0, 1876.55, 'chunk_04')
    ]
    partition_colmap_and_pcd(log_base, out_chunks, chunks)
