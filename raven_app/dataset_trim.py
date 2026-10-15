"""Dataset trimming and chunking with time synchronization for LidarCamera360.

Handles synchronized temporal slicing of both ROS bags (.bag) and 360 videos (.insv / .mp4)
using automatic or manual IMU gyro time-synchronization (dt_sync).

Time synchronization convention:
    t_lidar = ROS bag timestamp
    t_video = camera video timestamp
    dt_sync = t_lidar_rel - t_video_rel (or t_lidar - t_video)
    Specifically, with normalized starting times:
        t_video = t_lidar + dt_sync
        (i.e. if dt_sync > 0, camera video began before the bag recorded).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from rosbags.rosbag1 import Reader, Writer


@dataclass
class DatasetInterval:
    start_sec: float
    end_sec: float
    name: str = ""


def get_bag_time_bounds(bag_path: Path | str) -> tuple[int, int, float]:
    """Return (start_time_ns, end_time_ns, duration_seconds) for a ROS bag."""
    bag_path = Path(bag_path)
    with Reader(bag_path) as r:
        start_ns = r.start_time
        end_ns = r.end_time
        duration = (end_ns - start_ns) / 1e9
    return start_ns, end_ns, duration


def get_video_duration(video_path: Path | str) -> float:
    """Return duration in seconds of a video file (.insv / .mp4)."""
    try:
        import av
        with av.open(str(video_path)) as container:
            for stream in container.streams:
                if stream.type == 'video' and stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
            if container.duration is not None:
                return float(container.duration / av.time.AV_TIME_BASE)
    except Exception:
        pass
    return 0.0


def slice_bag(
    src_bag: Path | str,
    dst_bag: Path | str,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict:
    """Slice a ROS 1 bag temporally from start_sec to end_sec (relative to bag start).

    Args:
        src_bag: Input .bag file
        dst_bag: Output .bag file
        start_sec: Seconds from beginning of bag to start (default 0.0)
        end_sec: Seconds from beginning of bag to end (None = till end)
        progress_callback: Optional progress callback receiving float in [0, 1]

    Returns:
        dict with slice summary (messages, duration, start_time, end_time)
    """
    src_bag = Path(src_bag)
    dst_bag = Path(dst_bag)
    dst_bag.parent.mkdir(parents=True, exist_ok=True)

    with Reader(src_bag) as reader:
        bag_t0 = reader.start_time
        bag_t1 = reader.end_time
        total_dur = (bag_t1 - bag_t0) / 1e9

        t_min_ns = bag_t0 + int(max(0.0, start_sec) * 1e9)
        if end_sec is not None and end_sec < total_dur:
            t_max_ns = bag_t0 + int(end_sec * 1e9)
        else:
            t_max_ns = bag_t1

        if t_min_ns >= t_max_ns:
            raise ValueError(f"Invalid bag slice range: start_sec={start_sec}s >= end_sec={end_sec}s")

        total_msgs = reader.message_count
        written_count = 0
        first_written_ts = None
        last_written_ts = None

        with Writer(dst_bag) as writer:
            conn_map = {
                c.id: writer.add_connection(
                    c.topic,
                    c.msgtype,
                    md5sum=c.digest,
                    msgdef=c.msgdef.data if hasattr(c.msgdef, "data") else str(c.msgdef),
                )
                for c in reader.connections
            }

            for idx, (conn, ts, raw) in enumerate(reader.messages()):
                if ts < t_min_ns:
                    continue
                if ts > t_max_ns:
                    continue

                if first_written_ts is None:
                    first_written_ts = ts
                last_written_ts = ts

                writer.write(conn_map[conn.id], ts, raw)
                written_count += 1

                if progress_callback and idx % 2000 == 0:
                    progress_callback(min(1.0, idx / max(1, total_msgs)))

        if progress_callback:
            progress_callback(1.0)

    slice_dur = ((last_written_ts - first_written_ts) / 1e9) if (first_written_ts and last_written_ts) else 0.0
    return {
        "messages": written_count,
        "duration_seconds": slice_dur,
        "start_time_ns": first_written_ts,
        "end_time_ns": last_written_ts,
        "output_path": str(dst_bag),
    }


def slice_video(
    src_video: Path | str,
    dst_video: Path | str,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> dict:
    """Slice video temporally preserving all streams (dual fisheye cam0 + cam1 + audio).

    Uses stream copy (-c copy) when possible with precise seeking (-ss before -i).
    """
    src_video = Path(src_video)
    dst_video = Path(dst_video)
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    start_sec = max(0.0, float(start_sec))

    cmd = [ffmpeg_bin, "-y"]
    if start_sec > 0.001:
        cmd.extend(["-ss", f"{start_sec:.4f}"])
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.4f}"])

    cmd.extend(["-i", str(src_video)])
    # Map dual camera streams and audio
    cmd.extend(["-map", "0:v:0", "-map", "0:v:1", "-map", "0:a:0?"])
    cmd.extend(["-c", "copy", "-f", "mp4", str(dst_video)])

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        # Fallback without explicit -map if video has only single stream
        fallback_cmd = [ffmpeg_bin, "-y"]
        if start_sec > 0.001:
            fallback_cmd.extend(["-ss", f"{start_sec:.4f}"])
        if duration_sec is not None and duration_sec > 0:
            fallback_cmd.extend(["-t", f"{duration_sec:.4f}"])
        fallback_cmd.extend(["-i", str(src_video), "-c", "copy", str(dst_video)])
        res2 = subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res2.returncode != 0:
            raise RuntimeError(f"FFmpeg slicing failed: {res.stderr or res2.stderr}")

    actual_dur = get_video_duration(dst_video)
    return {
        "duration_seconds": actual_dur,
        "output_path": str(dst_video),
        "start_sec": start_sec,
    }


def trim_dataset_synchronized(
    bag_path: Path | str,
    video_path: Path | str,
    output_dir: Path | str,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
    dt_sync: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict:
    """Trim synchronized ROS bag and video from beginning and end.

    Args:
        bag_path: Path to input ROS bag (.bag)
        video_path: Path to input 360 video (.insv / .mp4)
        output_dir: Target directory for trimmed files
        trim_start_sec: Seconds to cut from the beginning
        trim_end_sec: Seconds to cut from the end
        dt_sync: Time sync offset (t_video = t_lidar + dt_sync). If None, auto-calculated.
        ffmpeg_bin: ffmpeg executable path
        progress_callback: Optional callback(stage, progress_float)

    Returns:
        dict with details of trimmed bag and video, effective times and dt_sync.
    """
    bag_path = Path(bag_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag file not found: {bag_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Determine time synchronization offset
    if dt_sync is None:
        if progress_callback:
            progress_callback("Computing time synchronization...", 0.05)
        from .workflow import auto_sync_imu_gyro, compute_time_sync_offset
        try:
            dt_sync = auto_sync_imu_gyro(bag_path, video_path)
        except Exception as e:
            print(f"[!] auto_sync_imu_gyro failed: {e}. Falling back to metadata offset.")
            dt_sync = compute_time_sync_offset(bag_path, video_path)

    dt_sync = float(dt_sync or 0.0)

    # 2. Inspect source durations
    _, _, bag_dur = get_bag_time_bounds(bag_path)
    video_dur = get_video_duration(video_path)

    trim_start_sec = max(0.0, float(trim_start_sec))
    trim_end_sec = max(0.0, float(trim_end_sec))

    # Bag target interval
    bag_slice_start = trim_start_sec
    bag_slice_end = max(0.0, bag_dur - trim_end_sec)
    if bag_slice_start >= bag_slice_end:
        raise ValueError(
            f"Trim settings exceed bag duration: start={trim_start_sec}s, end={trim_end_sec}s, bag={bag_dur:.2f}s"
        )
    mission_dur = bag_slice_end - bag_slice_start

    # Synchronized video interval:
    # t_video = t_lidar + dt_sync
    video_slice_start = max(0.0, bag_slice_start + dt_sync)
    video_slice_dur = mission_dur

    out_bag = output_dir / f"{bag_path.stem}_trimmed{bag_path.suffix}"
    out_video = output_dir / f"{video_path.stem}_trimmed{video_path.suffix}"

    # 3. Slice bag
    if progress_callback:
        progress_callback("Trimming ROS bag...", 0.15)
    bag_res = slice_bag(
        bag_path,
        out_bag,
        start_sec=bag_slice_start,
        end_sec=bag_slice_end,
        progress_callback=lambda p: progress_callback("Trimming ROS bag...", 0.15 + 0.35 * p) if progress_callback else None,
    )

    # 4. Slice video
    if progress_callback:
        progress_callback("Trimming 360 video...", 0.55)
    video_res = slice_video(
        video_path,
        out_video,
        start_sec=video_slice_start,
        duration_sec=video_slice_dur,
        ffmpeg_bin=ffmpeg_bin,
    )

    if progress_callback:
        progress_callback("Trim complete", 1.0)

    manifest = {
        "bag_path": str(out_bag),
        "video_path": str(out_video),
        "dt_sync": dt_sync,
        "trim_start_sec": trim_start_sec,
        "trim_end_sec": trim_end_sec,
        "bag_duration": bag_res["duration_seconds"],
        "video_duration": video_res["duration_seconds"],
        "bag_messages": bag_res["messages"],
    }

    manifest_path = output_dir / "trim_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def compute_overlapping_chunks(
    total_duration: float,
    chunk_duration: float,
    overlap_duration: float,
) -> list[tuple[float, float, str]]:
    """Compute start and end seconds for overlapping temporal chunks.

    Args:
        total_duration: Total mission duration in seconds
        chunk_duration: Nominal duration of each chunk in seconds
        overlap_duration: Temporal overlap shared between consecutive chunks

    Returns:
        List of (t_start, t_end, chunk_name)
    """
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if overlap_duration < 0:
        raise ValueError("overlap_duration cannot be negative")
    if overlap_duration >= chunk_duration:
        raise ValueError("overlap_duration must be strictly less than chunk_duration")

    if total_duration <= chunk_duration:
        return [(0.0, total_duration, "chunk_01")]

    step = chunk_duration - overlap_duration
    chunks = []
    t_start = 0.0
    idx = 1

    while t_start < total_duration:
        t_end = min(total_duration, t_start + chunk_duration)
        chunk_name = f"chunk_{idx:02d}"
        chunks.append((t_start, t_end, chunk_name))
        if t_end >= total_duration:
            break
        t_start += step
        idx += 1

    return chunks


def split_dataset_into_chunks(
    bag_path: Path | str,
    video_path: Path | str,
    output_dir: Path | str,
    chunk_duration_sec: float = 300.0,
    overlap_sec: float = 30.0,
    trim_start_sec: float = 0.0,
    trim_end_sec: float = 0.0,
    dt_sync: float | None = None,
    ffmpeg_bin: str = "ffmpeg",
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[dict]:
    """Divide a dataset into overlapping chunks with synchronized bag and video.

    Each chunk shares `overlap_sec` seconds with its neighbor, providing continuous
    overlap for smooth spatial/temporal interpolation and seamless map merging.

    Args:
        bag_path: Input ROS bag
        video_path: Input 360 video
        output_dir: Base directory where chunk subfolders will be created
        chunk_duration_sec: Duration of each chunk in seconds (default 300.0s)
        overlap_sec: Overlap duration shared between consecutive chunks (default 30.0s)
        trim_start_sec: Optional initial trim from mission start
        trim_end_sec: Optional final trim from mission end
        dt_sync: Time sync offset (auto-computed if None)
        ffmpeg_bin: ffmpeg binary
        progress_callback: Optional progress reporter

    Returns:
        List of chunk manifests containing paths and time boundaries
    """
    bag_path = Path(bag_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine sync
    if dt_sync is None:
        if progress_callback:
            progress_callback("Resolving time sync...", 0.05)
        from .workflow import auto_sync_imu_gyro, compute_time_sync_offset
        try:
            dt_sync = auto_sync_imu_gyro(bag_path, video_path)
        except Exception as e:
            print(f"[!] auto_sync_imu_gyro failed: {e}")
            dt_sync = compute_time_sync_offset(bag_path, video_path)
    dt_sync = float(dt_sync or 0.0)

    _, _, total_bag_dur = get_bag_time_bounds(bag_path)
    valid_start = max(0.0, float(trim_start_sec))
    valid_end = max(0.0, total_bag_dur - float(trim_end_sec))
    if valid_start >= valid_end:
        raise ValueError(f"Trim bounds exceed dataset duration: {valid_start}s -> {valid_end}s")

    mission_dur = valid_end - valid_start
    raw_chunks = compute_overlapping_chunks(mission_dur, chunk_duration_sec, overlap_sec)

    chunk_results = []
    total_n = len(raw_chunks)

    for i, (rel_start, rel_end, cname) in enumerate(raw_chunks):
        c_start_bag = valid_start + rel_start
        c_end_bag = valid_start + rel_end
        c_dur = c_end_bag - c_start_bag

        c_start_video = max(0.0, c_start_bag + dt_sync)

        chunk_dir = output_dir / cname
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_bag = chunk_dir / f"{bag_path.stem}_{cname}.bag"
        chunk_video = chunk_dir / f"{video_path.stem}_{cname}{video_path.suffix}"

        if progress_callback:
            msg = f"[{i+1}/{total_n}] Slicing {cname} ({c_dur:.1f}s, overlap {overlap_sec:.1f}s)..."
            progress_callback(msg, (i) / total_n)

        bag_res = slice_bag(bag_path, chunk_bag, start_sec=c_start_bag, end_sec=c_end_bag)
        video_res = slice_video(
            video_path,
            chunk_video,
            start_sec=c_start_video,
            duration_sec=c_dur,
            ffmpeg_bin=ffmpeg_bin,
        )

        item = {
            "name": cname,
            "index": i + 1,
            "total_chunks": total_n,
            "chunk_dir": str(chunk_dir),
            "bag_path": str(chunk_bag),
            "video_path": str(chunk_video),
            "bag_start_sec": c_start_bag,
            "bag_end_sec": c_end_bag,
            "bag_duration": bag_res["duration_seconds"],
            "video_start_sec": c_start_video,
            "video_duration": video_res["duration_seconds"],
            "overlap_sec": overlap_sec if i > 0 else 0.0,
            "dt_sync": dt_sync,
        }
        chunk_results.append(item)

    summary = {
        "dataset_bag": str(bag_path),
        "dataset_video": str(video_path),
        "dt_sync": dt_sync,
        "chunk_duration_sec": chunk_duration_sec,
        "overlap_sec": overlap_sec,
        "total_chunks": total_n,
        "chunks": chunk_results,
    }
    (output_dir / "chunks_manifest.json").write_text(json.dumps(summary, indent=2))

    if progress_callback:
        progress_callback(f"Finished dividing into {total_n} overlapping chunks", 1.0)

    return chunk_results
