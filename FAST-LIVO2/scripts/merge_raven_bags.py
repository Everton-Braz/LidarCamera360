#!/usr/bin/env python3
"""
Merge separate 3DMakerPro Raven LiDAR/IMU bag and Camera bag into a single,
chronologically ordered ROS 1 bag file for FAST-LIVO2 processing.
"""

import os
import sys
import argparse
import heapq
from pathlib import Path
from tqdm import tqdm
from rosbags.rosbag1 import Reader, Writer

def merge_bags(lidar_bag_path: str, img_bag_path: str, output_bag_path: str, duration_sec: float = None, start_sec: float = 0.0):
    lidar_bag_path = Path(lidar_bag_path)
    img_bag_path = Path(img_bag_path)
    output_bag_path = Path(output_bag_path)

    if not lidar_bag_path.exists():
        raise FileNotFoundError(f"LiDAR bag not found: {lidar_bag_path}")
    if not img_bag_path.exists():
        raise FileNotFoundError(f"Image bag not found: {img_bag_path}")

    print(f"[Merge] LiDAR/IMU bag: {lidar_bag_path}")
    print(f"[Merge] Image bag:     {img_bag_path}")
    print(f"[Merge] Output bag:    {output_bag_path}")

    with Reader(lidar_bag_path) as r_lidar, Reader(img_bag_path) as r_img:
        total_msgs = r_lidar.message_count + r_img.message_count
        print(f"[Merge] Total messages to process: {total_msgs} (LiDAR/IMU: {r_lidar.message_count}, Camera: {r_img.message_count})")

        # Determine start and end timestamps
        t_min = min(r_lidar.start_time, r_img.start_time)
        t_start_ns = t_min + int(start_sec * 1e9)
        t_end_ns = t_start_ns + int(duration_sec * 1e9) if duration_sec else max(r_lidar.end_time, r_img.end_time)

        output_bag_path.parent.mkdir(parents=True, exist_ok=True)
        if output_bag_path.exists():
            output_bag_path.unlink()
        with Writer(output_bag_path) as writer:
            # Map connections
            conn_map = {}
            for conn in r_lidar.connections:
                msgdef_str = conn.msgdef.data if hasattr(conn.msgdef, 'data') else str(conn.msgdef)
                conn_map[(0, conn.id)] = writer.add_connection(conn.topic, conn.msgtype, md5sum=conn.digest, msgdef=msgdef_str)
            for conn in r_img.connections:
                msgdef_str = conn.msgdef.data if hasattr(conn.msgdef, 'data') else str(conn.msgdef)
                conn_map[(1, conn.id)] = writer.add_connection(conn.topic, conn.msgtype, md5sum=conn.digest, msgdef=msgdef_str)

            # Generators for both readers
            def gen_msgs(reader, reader_id):
                for conn, timestamp, rawdata in reader.messages():
                    if timestamp < t_start_ns:
                        continue
                    if timestamp > t_end_ns:
                        break
                    yield timestamp, reader_id, conn.id, rawdata

            gen0 = gen_msgs(r_lidar, 0)
            gen1 = gen_msgs(r_img, 1)

            # Merge sorted streams
            merged = heapq.merge(gen0, gen1, key=lambda x: x[0])

            written_count = 0
            with tqdm(total=total_msgs, desc="Merging bags", unit="msg") as pbar:
                for timestamp, reader_id, conn_id, rawdata in merged:
                    out_conn = conn_map[(reader_id, conn_id)]
                    writer.write(out_conn, timestamp, rawdata)
                    written_count += 1
                    pbar.update(1)

    print(f"\n[Merge] Successfully wrote {written_count} messages to {output_bag_path}!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge Raven LiDAR and Camera bag files.")
    parser.add_argument('--lidar_bag', default='DATA_TEST/LIDAR_20260804142234Azure1.bag', help='Path to LiDAR/IMU bag')
    parser.add_argument('--img_bag', default='DATA_TEST/IMAGE_20260804142234Azure1.bag', help='Path to Camera bag')
    parser.add_argument('--output_bag', default='DATA_TEST/raven_merged.bag', help='Path to output merged bag')
    parser.add_argument('--duration', type=float, default=None, help='Duration in seconds to extract (optional)')
    parser.add_argument('--start', type=float, default=0.0, help='Start time offset in seconds (default: 0.0)')

    args = parser.parse_args()
    merge_bags(args.lidar_bag, args.img_bag, args.output_bag, args.duration, args.start)
