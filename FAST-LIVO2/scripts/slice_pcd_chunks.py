#!/usr/bin/env python3
import os
import struct
import numpy as np
from rosbags.rosbag1 import Reader

def slice_pcd_binary(src_pcd, dst_pcd, start_idx, end_idx):
    """
    Slices a binary PCD file between start_idx and end_idx (point indices)
    and writes a valid PCD file.
    """
    with open(src_pcd, 'rb') as f_in:
        header_lines = []
        while True:
            line = f_in.readline().decode('ascii', errors='ignore')
            if line.startswith('DATA'):
                break
            header_lines.append(line)
        
        num_points = end_idx - start_idx
        # Build new header
        new_header = []
        for l in header_lines:
            if l.startswith('WIDTH'):
                new_header.append(f"WIDTH {num_points}\n")
            elif l.startswith('POINTS'):
                new_header.append(f"POINTS {num_points}\n")
            else:
                new_header.append(l)
        new_header.append("DATA binary\n")
        
        # Seek to start point (16 bytes per point for XYZRGB)
        f_in.seek(start_idx * 16, os.SEEK_CUR)
        chunk_bytes = f_in.read(num_points * 16)
        
    os.makedirs(os.path.dirname(dst_pcd), exist_ok=True)
    with open(dst_pcd, 'wb') as f_out:
        f_out.write("".join(new_header).encode('ascii'))
        f_out.write(chunk_bytes)
        
    sz_mb = os.path.getsize(dst_pcd) / (1024 * 1024)
    print(f"  Saved {dst_pcd}: {sz_mb:.2f} MB ({num_points} points)")

def export_colmap_points3d_for_chunk(pcd_path, output_points3d_path, max_points=1000000):
    """
    Generates points3D.txt from a chunk PCD file.
    """
    with open(pcd_path, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore')
            if line.startswith('POINTS'):
                total_pts = int(line.split()[1])
            if line.startswith('DATA'):
                break
        
        step = max(1, total_pts // max_points)
        num_export = total_pts // step
        
        os.makedirs(os.path.dirname(output_points3d_path), exist_ok=True)
        with open(output_points3d_path, 'w') as f_out:
            f_out.write("# 3D point list with one line of data per point\n")
            f_out.write("#  POINT_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            
            raw = f.read()
            out_id = 1
            for i in range(0, total_pts, step):
                offset = i * 16
                if offset + 16 > len(raw):
                    break
                x, y, z, rgb_u = struct.unpack_from('fffl', raw, offset)
                r = (rgb_u >> 16) & 0xFF
                g = (rgb_u >> 8) & 0xFF
                b = rgb_u & 0xFF
                f_out.write(f"{out_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.5\n")
                out_id += 1
                
    sz_mb = os.path.getsize(output_points3d_path) / (1024 * 1024)
    print(f"  Saved {output_points3d_path}: {sz_mb:.2f} MB ({out_id-1} points)")

def main():
    bag_path = 'DATA_TEST/raven_merged.bag'
    src_pcd = 'Log/pcd/all_raw_points.pcd'
    chunks_base = 'Log/Chunks'
    
    if not os.path.exists(src_pcd):
        print(f"Error: Source PCD {src_pcd} does not exist!")
        return

    print("Analyzing timestamps in bag to compute exact chunk boundaries...")
    timestamps = []
    with Reader(bag_path) as bag:
        start_t = None
        for conn, ts, raw in bag.messages():
            if conn.topic == '/vanjee_722z':
                if start_t is None:
                    start_t = ts
                timestamps.append((ts - start_t) / 1e9)

    timestamps = np.array(timestamps)
    total_scans = len(timestamps)
    
    # Read total points in source PCD
    with open(src_pcd, 'rb') as f:
        while True:
            line = f.readline().decode('ascii', errors='ignore')
            if line.startswith('POINTS'):
                total_pts_pcd = int(line.split()[1])
                break
                
    pts_per_scan = total_pts_pcd / total_scans
    print(f"Total LiDAR scans: {total_scans}, Total points: {total_pts_pcd} (~{pts_per_scan:.1f} pts/scan)")

    chunks = [
        (0.0, 540.0, 'chunk_01'),
        (460.0, 1000.0, 'chunk_02'),
        (920.0, 1460.0, 'chunk_03'),
        (1380.0, 1876.55, 'chunk_04')
    ]

    for t_start, t_end, cname in chunks:
        idx = np.where((timestamps >= t_start) & (timestamps <= t_end))[0]
        p_start = int(idx[0] * pts_per_scan)
        p_end = int(min((idx[-1] + 1) * pts_per_scan, total_pts_pcd))
        
        print(f"\n[{cname}] Slicing points {p_start} -> {p_end} ({p_end - p_start} points, time {t_start}s -> {t_end}s)...")
        
        # Raw PCD
        dst_raw_pcd = os.path.join(chunks_base, cname, 'pcd', 'all_raw_points.pcd')
        slice_pcd_binary(src_pcd, dst_raw_pcd, p_start, p_end)
        
        # Downsampled PCD
        dst_down_pcd = os.path.join(chunks_base, cname, 'pcd', 'all_downsampled_points.pcd')
        slice_pcd_binary(src_pcd, dst_down_pcd, p_start, p_end)
        
        # Sparse points3D.txt
        dst_points3d = os.path.join(chunks_base, cname, 'Colmap', 'sparse', '0', 'points3D.txt')
        export_colmap_points3d_for_chunk(dst_down_pcd, dst_points3d, max_points=1000000)

    print("\nAll 4 chunk PCDs and points3D.txt successfully sliced!")

if __name__ == '__main__':
    main()
