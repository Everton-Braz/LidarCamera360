"""Unit and integration tests for dataset trimming and chunking."""
import pytest
from pathlib import Path
import numpy as np

from raven_app.dataset_trim import (
    compute_overlapping_chunks,
    slice_bag,
    slice_video,
    trim_dataset_synchronized,
    split_dataset_into_chunks,
)


def test_compute_overlapping_chunks_basic():
    # 100 seconds divided into 30s chunks with 10s overlap
    # step = 20s -> chunks: [0, 30], [20, 50], [40, 70], [60, 90], [80, 100]
    chunks = compute_overlapping_chunks(100.0, 30.0, 10.0)
    assert len(chunks) == 5
    assert chunks[0] == (0.0, 30.0, "chunk_01")
    assert chunks[1] == (20.0, 50.0, "chunk_02")
    assert chunks[2] == (40.0, 70.0, "chunk_03")
    assert chunks[3] == (60.0, 90.0, "chunk_04")
    assert chunks[4] == (80.0, 100.0, "chunk_05")


def test_compute_overlapping_chunks_single():
    # If duration is shorter than chunk duration, exactly 1 chunk is produced
    chunks = compute_overlapping_chunks(50.0, 100.0, 20.0)
    assert len(chunks) == 1
    assert chunks[0] == (0.0, 50.0, "chunk_01")


def test_compute_overlapping_chunks_validation():
    with pytest.raises(ValueError):
        compute_overlapping_chunks(100.0, -10.0, 5.0)
    with pytest.raises(ValueError):
        compute_overlapping_chunks(100.0, 30.0, 35.0)
    with pytest.raises(ValueError):
        compute_overlapping_chunks(100.0, 30.0, -1.0)


def test_trim_and_chunk_with_real_dataset(tmp_path):
    bag_path = Path("C:/Users/Everton-PC/Downloads/Lidou/20261005fab5/LIDAR_20261005fab5.bag")
    video_path = Path("C:/Users/Everton-PC/Downloads/Lidou/20261005fab5/VID_20261005_143357_00_296.insv")
    if not bag_path.exists() or not video_path.exists():
        pytest.skip("Real dataset files not present on machine")

    # 1. Test synchronized trim (small 6s slice)
    out_trim = tmp_path / "trim_test"
    res = trim_dataset_synchronized(
        bag_path,
        video_path,
        out_trim,
        trim_start_sec=10.0,
        trim_end_sec=322.0,  # ~6.5s duration
        dt_sync=6.938723,
    )
    assert Path(res["bag_path"]).exists()
    assert Path(res["video_path"]).exists()
    assert 5.0 <= res["bag_duration"] <= 8.0
    assert 5.0 <= res["video_duration"] <= 8.0
    assert res["bag_messages"] > 0

    # 2. Test chunking with overlap
    out_chunks = tmp_path / "chunks_test"
    chunks = split_dataset_into_chunks(
        bag_path,
        video_path,
        out_chunks,
        chunk_duration_sec=6.0,
        overlap_sec=2.0,
        trim_start_sec=10.0,
        trim_end_sec=320.0,
        dt_sync=6.938723,
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert Path(c["bag_path"]).exists()
        assert Path(c["video_path"]).exists()
        assert c["bag_duration"] > 0
        assert c["video_duration"] > 0


def test_end_at_keeps_requested_seconds_and_residual_dt(tmp_path, monkeypatch):
    from raven_app import dataset_trim as dt
    calls = {}
    monkeypatch.setattr(dt, "get_bag_time_bounds", lambda p: (0.0, 914.0, 914.0))
    monkeypatch.setattr(dt, "get_video_duration", lambda p: 60.0)

    def fake_bag(src, dst, start_sec, end_sec, progress_callback=None):
        calls["bag"] = (start_sec, end_sec)
        return {"duration_seconds": end_sec - start_sec, "messages": 1}

    def fake_video(src, dst, start_sec, duration_sec, ffmpeg_bin="ffmpeg"):
        calls["video"] = (start_sec, duration_sec)
        return {"duration_seconds": duration_sec}

    monkeypatch.setattr(dt, "slice_bag", fake_bag)
    monkeypatch.setattr(dt, "slice_video", fake_video)
    (tmp_path / "a.bag").write_bytes(b"x")
    (tmp_path / "a.mp4").write_bytes(b"x")
    res = dt.trim_dataset_synchronized(tmp_path / "a.bag", tmp_path / "a.mp4", tmp_path / "o",
                                       end_at_sec=240.0, dt_sync=2.81)
    assert calls["bag"] == (0.0, 240.0)
    assert calls["video"][0] == pytest.approx(2.81)
    assert calls["video"][1] == pytest.approx(57.19)   # video is shorter than the LiDAR
    assert res["dt_sync_trimmed"] == pytest.approx(0.0)
    assert res["trim_end_sec"] == pytest.approx(674.0)
