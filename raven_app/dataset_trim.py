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

from raven_app.insv_gps import write_gps_sidecar


@dataclass
class DatasetInterval:
    start_sec: float
    end_sec: float
    name: str = ""



def cut_dataset_sync(bag_path):
    """Offset left between a bag and video cut by Trim/Chunks, read from the manifest next to them.

    Cut videos lose the INSV telemetry trailer, so the gyro cannot sync them again; the cut
    recorded the offset instead. Returns None when the bag is not a Trim/Chunks output.
    """
    bag_path = Path(bag_path)
    folder = bag_path.parent
    try:
        trim = folder / "trim_manifest.json"
        if trim.is_file():
            data = json.loads(trim.read_text(encoding="utf-8"))
            if Path(data.get("bag_path", "")).name == bag_path.name and data.get("dt_sync_trimmed") is not None:
                return float(data["dt_sync_trimmed"])
        chunks = folder.parent / "chunks_manifest.json"
        if chunks.is_file():
            for ch in json.loads(chunks.read_text(encoding="utf-8")).get("chunks", []):
                if Path(ch.get("bag_path", "")).name == bag_path.name and ch.get("dt_sync_trimmed") is not None:
                    return float(ch["dt_sync_trimmed"])
    except (OSError, ValueError):
        return None
    return None


def _resolve_time_sync(bag_path, video_path):
    """IMU-gyro offset of the clip inside the LiDAR recording; a guessed offset is never substituted."""
    from .workflow import auto_sync_imu_gyro
    known = cut_dataset_sync(bag_path)
    if known is not None:
        print(f"[*] Already-cut dataset: Δt = {known:.4f}s from its manifest")
        return known
    try:
        dt_sync = auto_sync_imu_gyro(bag_path, video_path)
    except Exception as e:
        print(f"[!] auto_sync_imu_gyro failed: {e}")
        dt_sync = None
    if dt_sync is None:
        raise ValueError("Could not determine the LiDAR/video time offset automatically. "
                         "Untick 'Auto IMU Gyro Cross-Correlation' and enter the Manual Δt (s), then try again.")
    return dt_sync

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



SIDECAR_SUFFIX = ".timelapse.json"


def _ffmpeg_error(stderr: str, lines: int = 6) -> str:
    """The lines of ffmpeg's stderr that say what went wrong (the banner and stream listing come first)."""
    tail = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    return " | ".join(tail[-lines:]) or "no output"


def _partner_video(video_path: Path, secondary_video_path) -> Path | None:
    """Rear-lens file of a dual-file camera (ONE RS / ONE R): the one given, else the _10_ sibling of a _00_ file."""
    if secondary_video_path:
        p = Path(secondary_video_path)
        if not p.is_file():
            raise FileNotFoundError(f"Second lens video not found: {p}")
        return p
    from raven_app.video import find_partner_insv
    cam0, cam1 = find_partner_insv(video_path)
    return Path(cam1) if cam0 and cam1 and Path(cam0) == Path(video_path) else None


AUX_PAD_KNOWN_SEC = 10.0    # margin around an auxiliary cut when its clock offset was calibrated
AUX_PAD_UNKNOWN_SEC = 60.0  # wider margin when the offset is only the configured guess
AUX_CONFIG_NAME = "third_camera.json"


def _auxiliary_configs(auxiliary_cameras) -> list:
    from raven_app.third_camera import ThirdCameraConfig
    configs = []
    for camera in auxiliary_cameras or []:
        cfg = camera if isinstance(camera, ThirdCameraConfig) else ThirdCameraConfig.from_dict(camera)
        if cfg.enabled and str(cfg.source_path).strip():
            configs.append(cfg)
    return configs


def cut_auxiliary_cameras(cameras, out_dir, tag, bag_start, bag_dur, dt_sync, video_start,
                          ffmpeg_bin="ffmpeg") -> list:
    """Cut each auxiliary video to the kept interval and re-express its clock offset.

    The pipeline reads an auxiliary frame at video time ``t_lidar + dt_sync + time_offset_s``. After
    the bag is cut at ``bag_start`` and the main video at ``video_start``, a source cut at ``a`` keeps
    that mapping with ``time_offset_s' = time_offset_s + video_start - a`` (``a = 0`` when uncut).
    Video files are cut; INSV (the trailer would be lost), image folders and bag topics stay whole and
    only get the new offset; bag topics follow the bag clock and are left as they are.
    """
    import copy
    out_dir = Path(out_dir)
    result = []
    for cfg in cameras:
        new = copy.deepcopy(cfg)
        src = Path(cfg.source_path)
        offset = float(cfg.time_offset_s or 0.0)
        if cfg.source_type == "bag" or src.suffix.lower() == ".bag":
            result.append(new)
            continue
        cut_at = 0.0
        if src.is_file() and src.suffix.lower() != ".insv":
            status = str(((cfg.calibration_report or {}).get("time_offset") or {}).get("status", "unknown"))
            pad = AUX_PAD_UNKNOWN_SEC if status == "unknown" else AUX_PAD_KNOWN_SEC
            aux_dur = get_video_duration(src)
            start = bag_start + dt_sync + offset
            a, b = max(0.0, start - pad), min(aux_dur, start + bag_dur + pad)
            if b - a < 1.0:
                print(f"[aux] {cfg.camera_name}: {src.name} does not cover this interval with offset "
                      f"{offset:+.2f}s; left uncut.")
            else:
                dst = out_dir / f"{src.stem}_{tag}{src.suffix}"
                slice_video(src, dst, start_sec=a, duration_sec=b - a, ffmpeg_bin=ffmpeg_bin)
                new.source_path = str(dst)
                cut_at = a
                if status == "unknown":
                    print(f"[aux] {cfg.camera_name}: clock offset not calibrated; cut around the configured "
                          f"{offset:+.2f}s with a {pad:.0f}s margin.")
        new.time_offset_s = offset + video_start - cut_at
        result.append(new)
    if result:
        from raven_app.third_camera import save_third_camera_config
        save_third_camera_config(result, out_dir / AUX_CONFIG_NAME)
    return result


def _slice_lens(src, dst, cap_times, start, dur, ffmpeg_bin):
    """Cut one lens file; a timelapse is cut by capture time, a normal video by playback time."""
    if cap_times is not None:
        return slice_timelapse_video(src, dst, cap_times, start, start + dur, ffmpeg_bin)
    return slice_video(src, dst, start_sec=start, duration_sec=dur, ffmpeg_bin=ffmpeg_bin)


def capture_times(video_path: Path | str):
    """Per-frame capture seconds of a timelapse video (INSV trailer or our sidecar), else None.

    A timelapse plays 15 min of capture in about a minute, so playback seconds are NOT capture seconds.
    """
    from .video import _timelapse_times
    return _timelapse_times(Path(video_path))


def _video_packets(video_path: Path | str):
    import av
    pts, key = [], []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            pts.append(float(packet.pts * stream.time_base))
            key.append(bool(packet.is_keyframe))
    order = np.argsort(pts)
    return np.asarray(pts)[order], np.asarray(key)[order]


def slice_timelapse_video(src_video, dst_video, times, t_start, t_end, ffmpeg_bin="ffmpeg") -> dict:
    """Cut a timelapse video by CAPTURE time. The cut starts on a keyframe so no frame is left without a timestamp,
    and the capture times of the kept frames are written beside the output (rebased to its first frame)."""
    pts, key = _video_packets(src_video)
    if len(pts) > len(times):
        raise RuntimeError(f"Video has {len(pts)} frames but its telemetry has only {len(times)} timestamps")
    times = times[:len(pts)]  # the camera may log one more exposure than it encoded
    wanted = np.nonzero((times >= t_start) & (times <= t_end))[0]
    if not len(wanted):
        raise ValueError(f"No video frame was captured between {t_start:.1f}s and {t_end:.1f}s "
                         f"(the video covers 0 - {times[-1]:.1f}s)")
    first = int(wanted[0])
    last = int(wanted[-1])
    first = int(np.nonzero(key[:first + 1])[0][-1])
    step = float(np.median(np.diff(pts))) if len(pts) > 1 else 0.0
    dur = (pts[last] - pts[first]) + step
    out = Path(dst_video)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_bin, "-y", "-ss", f"{pts[first]:.6f}", "-t", f"{dur:.6f}", "-i", str(src_video),
           "-map", "0:v:0", "-map", "0:v:1", "-map", "0:a:0?", "-c", "copy", "-f", "mp4", str(out)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        cmd = [ffmpeg_bin, "-hide_banner", "-y", "-ss", f"{pts[first]:.6f}", "-t", f"{dur:.6f}", "-i", str(src_video),
               "-map", "0:v", "-map", "0:a:0?", "-c", "copy", "-f", "mp4", str(out)]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg slicing failed for {Path(src_video).name}: {_ffmpeg_error(res.stderr)}")
    n_out = len(_video_packets(out)[0])
    if n_out < last - first + 1:
        last = first + n_out - 1
    kept = times[first:last + 1]
    Path(str(out) + SIDECAR_SUFFIX).write_text(
        json.dumps({"times": (kept - kept[0]).tolist(), "origin_capture_sec": float(kept[0])}))
    return {"duration_seconds": float(kept[-1] - kept[0]), "output_path": str(out), "start_sec": float(kept[0]),
            "frames": int(len(kept)), "timelapse": True}


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
    """Slice video temporally preserving every video stream and the audio.

    An X4/X5 INSV holds both lenses as two streams; a ONE RS INSV holds one lens per file. Mapping
    all video streams covers both. The output is always written as MP4 because ffmpeg cannot pick a
    muxer for the .insv extension. Uses stream copy with seeking before -i.
    """
    src_video = Path(src_video)
    dst_video = Path(dst_video)
    dst_video.parent.mkdir(parents=True, exist_ok=True)

    start_sec = max(0.0, float(start_sec))

    cmd = [ffmpeg_bin, "-hide_banner", "-y"]
    if start_sec > 0.001:
        cmd.extend(["-ss", f"{start_sec:.4f}"])
    if duration_sec is not None and duration_sec > 0:
        cmd.extend(["-t", f"{duration_sec:.4f}"])
    cmd.extend(["-i", str(src_video), "-map", "0:v", "-map", "0:a:0?", "-c", "copy", "-f", "mp4", str(dst_video)])

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg slicing failed for {src_video.name}: {_ffmpeg_error(res.stderr)}")

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
    end_at_sec: float | None = None,
    secondary_video_path: Path | str | None = None,
    auxiliary_cameras=None,
) -> dict:
    """Trim synchronized ROS bag and video from beginning and end.

    The rear-lens file of a dual-file camera (given, or the ``_10_`` sibling of a ``_00_`` file) is cut
    with the same interval. Auxiliary cameras are cut by their own clock offset and written to
    ``third_camera.json`` in ``output_dir``.

    Args:
        bag_path: Path to input ROS bag (.bag)
        video_path: Path to input 360 video (.insv / .mp4)
        output_dir: Target directory for trimmed files
        trim_start_sec: Seconds to cut from the beginning
        trim_end_sec: Seconds to cut from the end
        end_at_sec: Absolute end time (seconds from the bag start) to keep; overrides trim_end_sec when > 0
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
    partner = _partner_video(video_path, secondary_video_path)
    aux_cameras = _auxiliary_configs(auxiliary_cameras)

    # 1. Determine time synchronization offset
    if dt_sync is None:
        if progress_callback:
            progress_callback("Computing time synchronization...", 0.05)
        dt_sync = _resolve_time_sync(bag_path, video_path)

    dt_sync = float(dt_sync or 0.0)

    # 2. Inspect source durations
    _, _, bag_dur = get_bag_time_bounds(bag_path)
    cap_times = capture_times(video_path)
    video_dur = float(cap_times[-1]) if cap_times is not None else get_video_duration(video_path)
    if cap_times is not None:
        print(f"[trim] Timelapse video: {len(cap_times)} frames spanning {video_dur:.1f}s of capture time.")

    trim_start_sec = max(0.0, float(trim_start_sec))
    trim_end_sec = max(0.0, float(trim_end_sec))

    # Bag target interval
    bag_slice_start = trim_start_sec
    bag_slice_end = max(0.0, bag_dur - trim_end_sec)
    if end_at_sec is not None and float(end_at_sec) > 0:
        bag_slice_end = min(bag_dur, float(end_at_sec))
        trim_end_sec = bag_dur - bag_slice_end
    if bag_slice_start >= bag_slice_end:
        raise ValueError(
            f"Trim settings exceed bag duration: start={trim_start_sec}s, end={trim_end_sec}s, bag={bag_dur:.2f}s"
        )
    mission_dur = bag_slice_end - bag_slice_start

    # Synchronized video interval:
    # t_video = t_lidar + dt_sync
    video_slice_start = max(0.0, bag_slice_start + dt_sync)
    # The video can be shorter than the bag: colour only exists where both overlap.
    video_slice_dur = min(mission_dur, max(0.0, video_dur - video_slice_start))
    if video_slice_dur <= 0:
        raise ValueError(f"The video ({video_dur:.1f}s) ends before the selected LiDAR interval starts "
                         f"(video starts at {video_slice_start:.1f}s).")
    if video_slice_dur < mission_dur - 0.5:
        print(f"[trim] WARNING: video covers only {video_slice_dur:.1f}s of the {mission_dur:.1f}s LiDAR interval; "
              "the rest will have no colour.")
    # Stream copy makes the requested start the new t=0 of the video, so the offset between the
    # trimmed files is what is left after both cuts, not the original dt_sync.

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
    requested_video_start = video_slice_start
    video_res = _slice_lens(video_path, out_video, cap_times, video_slice_start, video_slice_dur, ffmpeg_bin)
    if cap_times is not None:
        video_slice_start = video_res["start_sec"]
    dt_sync_trimmed = dt_sync - video_slice_start + bag_slice_start
    write_gps_sidecar(video_path, out_video, video_slice_start)

    out_partner = None
    if partner is not None:
        if progress_callback:
            progress_callback("Trimming second lens video...", 0.75)
        out_partner = output_dir / f"{partner.stem}_trimmed{partner.suffix}"
        _slice_lens(partner, out_partner, capture_times(partner) if cap_times is not None else None,
                    requested_video_start, video_slice_dur, ffmpeg_bin)

    aux_out = []
    if aux_cameras:
        if progress_callback:
            progress_callback("Trimming auxiliary cameras...", 0.85)
        aux_out = cut_auxiliary_cameras(aux_cameras, output_dir, "trimmed", bag_slice_start, mission_dur,
                                        dt_sync, video_slice_start, ffmpeg_bin)

    if progress_callback:
        progress_callback("Trim complete", 1.0)

    manifest = {
        "bag_path": str(out_bag),
        "video_path": str(out_video),
        "secondary_video_path": str(out_partner) if out_partner else None,
        "auxiliary_config": str(output_dir / AUX_CONFIG_NAME) if aux_out else None,
        "bag_start_sec": bag_slice_start,
        "bag_kept_sec": mission_dur,
        "video_start_sec": video_slice_start,
        "dt_sync": dt_sync,
        "dt_sync_trimmed": dt_sync_trimmed,
        "video_covered_seconds": video_slice_dur,
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
    end_at_sec: float | None = None,
    num_chunks: int = 0,
    secondary_video_path: Path | str | None = None,
    auxiliary_cameras=None,
) -> list[dict]:
    """Divide a dataset into overlapping chunks with synchronized bag and video.

    The rear-lens file of a dual-file camera and the auxiliary cameras are cut per chunk as in
    :func:`trim_dataset_synchronized`.

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
        num_chunks: If > 0, split into exactly this many chunks (chunk_duration_sec is derived from it)
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
    partner = _partner_video(video_path, secondary_video_path)
    aux_cameras = _auxiliary_configs(auxiliary_cameras)

    # Determine sync
    if dt_sync is None:
        if progress_callback:
            progress_callback("Resolving time sync...", 0.05)
        dt_sync = _resolve_time_sync(bag_path, video_path)
    dt_sync = float(dt_sync or 0.0)

    _, _, total_bag_dur = get_bag_time_bounds(bag_path)
    valid_start = max(0.0, float(trim_start_sec))
    valid_end = max(0.0, total_bag_dur - float(trim_end_sec))
    if end_at_sec is not None and float(end_at_sec) > 0:
        valid_end = min(total_bag_dur, float(end_at_sec))
    if valid_start >= valid_end:
        raise ValueError(f"Trim bounds exceed dataset duration: {valid_start}s -> {valid_end}s")

    mission_dur = valid_end - valid_start
    cap_times = capture_times(video_path)
    partner_times = capture_times(partner) if partner is not None and cap_times is not None else None
    video_dur = float(cap_times[-1]) if cap_times is not None else get_video_duration(video_path)
    if num_chunks:
        n = max(1, int(num_chunks))
        chunk_duration_sec = (mission_dur + (n - 1) * overlap_sec) / n
    raw_chunks = compute_overlapping_chunks(mission_dur, chunk_duration_sec, overlap_sec)

    chunk_results = []
    total_n = len(raw_chunks)

    for i, (rel_start, rel_end, cname) in enumerate(raw_chunks):
        c_start_bag = valid_start + rel_start
        c_end_bag = valid_start + rel_end
        c_dur = c_end_bag - c_start_bag

        c_start_video = max(0.0, c_start_bag + dt_sync)
        c_video_dur = min(c_dur, max(0.0, video_dur - c_start_video))
        if c_video_dur <= 0:
            print(f"[chunks] {cname}: the video ends before this chunk, skipped (no colour available).")
            continue
        if c_video_dur < c_dur - 0.5:
            print(f"[chunks] WARNING {cname}: video covers only {c_video_dur:.1f}s of {c_dur:.1f}s.")

        chunk_dir = output_dir / cname
        chunk_dir.mkdir(parents=True, exist_ok=True)

        chunk_bag = chunk_dir / f"{bag_path.stem}_{cname}.bag"
        chunk_video = chunk_dir / f"{video_path.stem}_{cname}{video_path.suffix}"

        if progress_callback:
            msg = f"[{i+1}/{total_n}] Slicing {cname} ({c_dur:.1f}s, overlap {overlap_sec:.1f}s)..."
            progress_callback(msg, (i) / total_n)

        bag_res = slice_bag(bag_path, chunk_bag, start_sec=c_start_bag, end_sec=c_end_bag)
        requested_start = c_start_video
        video_res = _slice_lens(video_path, chunk_video, cap_times, c_start_video, c_video_dur, ffmpeg_bin)
        if cap_times is not None:
            c_start_video = video_res["start_sec"]
        write_gps_sidecar(video_path, chunk_video, c_start_video)

        chunk_partner = None
        if partner is not None:
            chunk_partner = chunk_dir / f"{partner.stem}_{cname}{partner.suffix}"
            _slice_lens(partner, chunk_partner, partner_times, requested_start, c_video_dur, ffmpeg_bin)
        aux_out = (cut_auxiliary_cameras(aux_cameras, chunk_dir, cname, c_start_bag, c_dur, dt_sync,
                                         c_start_video, ffmpeg_bin) if aux_cameras else [])

        item = {
            "name": cname,
            "index": i + 1,
            "total_chunks": total_n,
            "chunk_dir": str(chunk_dir),
            "bag_path": str(chunk_bag),
            "video_path": str(chunk_video),
            "secondary_video_path": str(chunk_partner) if chunk_partner else None,
            "auxiliary_config": str(chunk_dir / AUX_CONFIG_NAME) if aux_out else None,
            "bag_start_sec": c_start_bag,
            "bag_end_sec": c_end_bag,
            "bag_duration": bag_res["duration_seconds"],
            "video_start_sec": c_start_video,
            "video_duration": video_res["duration_seconds"],
            "overlap_sec": overlap_sec if i > 0 else 0.0,
            "dt_sync": dt_sync,
            "dt_sync_trimmed": dt_sync - c_start_video + c_start_bag,
        }
        chunk_results.append(item)

    summary = {
        "dataset_bag": str(bag_path),
        "dataset_video": str(video_path),
        "dataset_secondary_video": str(partner) if partner else None,
        "dt_sync": dt_sync,
        "chunk_duration_sec": chunk_duration_sec,
        "overlap_sec": overlap_sec,
        "total_chunks": len(chunk_results),
        "chunks": chunk_results,
    }
    (output_dir / "chunks_manifest.json").write_text(json.dumps(summary, indent=2))

    if progress_callback:
        progress_callback(f"Finished dividing into {total_n} overlapping chunks", 1.0)

    return chunk_results
