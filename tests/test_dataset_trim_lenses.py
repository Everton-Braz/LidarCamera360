"""Trim and chunks cut the second lens file and the auxiliary videos with the main one."""
import json

import pytest


@pytest.fixture
def fake_cuts(monkeypatch):
    from raven_app import dataset_trim as dt
    calls = []
    monkeypatch.setattr(dt, "get_bag_time_bounds", lambda p: (0.0, 600.0, 600.0))
    monkeypatch.setattr(dt, "get_video_duration", lambda p: 1300.0 if "phone" in str(p) else 552.0)
    monkeypatch.setattr(dt, "slice_bag", lambda src, dst, start_sec, end_sec, progress_callback=None:
                        {"duration_seconds": end_sec - start_sec, "messages": 1})

    def fake_video(src, dst, start_sec, duration_sec, ffmpeg_bin="ffmpeg"):
        calls.append((src.name, dst.name, start_sec, duration_sec))
        return {"duration_seconds": duration_sec}

    monkeypatch.setattr(dt, "slice_video", fake_video)
    return dt, calls


def _dataset(tmp_path):
    for name in ("run.bag", "VID_1_00_012.insv", "VID_1_10_012.insv", "phone.mp4"):
        (tmp_path / name).write_bytes(b"x")
    return tmp_path / "run.bag", tmp_path / "VID_1_00_012.insv"


def _phone(offset, status="calibrated"):
    from raven_app.third_camera import ThirdCameraConfig
    cfg = ThirdCameraConfig(camera_name="cam2", time_offset_s=offset)
    cfg.calibration_report["time_offset"] = {"status": status}
    return cfg


def test_trim_cuts_rear_lens_found_by_name(tmp_path, fake_cuts):
    dt, calls = fake_cuts
    bag, video = _dataset(tmp_path)
    res = dt.trim_dataset_synchronized(bag, video, tmp_path / "o", trim_start_sec=100.0,
                                       end_at_sec=200.0, dt_sync=5.0)
    assert [c[:2] for c in calls] == [("VID_1_00_012.insv", "VID_1_00_012_trimmed.insv"),
                                      ("VID_1_10_012.insv", "VID_1_10_012_trimmed.insv")]
    assert calls[0][2:] == calls[1][2:] == (105.0, 100.0)
    assert res["secondary_video_path"].endswith("VID_1_10_012_trimmed.insv")


def test_chunks_cut_rear_lens_and_shift_phone_offset(tmp_path, fake_cuts):
    dt, calls = fake_cuts
    bag, video = _dataset(tmp_path)
    phone = _phone(-20.0)
    phone.source_path = str(tmp_path / "phone.mp4")
    out = tmp_path / "chunks"
    chunks = dt.split_dataset_into_chunks(bag, video, out, num_chunks=2, overlap_sec=30.0,
                                          dt_sync=5.0, auxiliary_cameras=[phone])
    assert len(chunks) == 2
    second = chunks[1]
    assert second["secondary_video_path"].endswith("VID_1_10_012_chunk_02.insv")
    phone_cut = [c for c in calls if c[0] == "phone.mp4" and "chunk_02" in c[1]][0]
    # bag 285..600 s -> phone time = t + 5 - 20, with a 10 s margin each side
    assert phone_cut[2] == pytest.approx(285.0 + 5.0 - 20.0 - 10.0)
    assert phone_cut[3] == pytest.approx(315.0 + 20.0)

    saved = json.loads((out / "chunk_02" / "third_camera.json").read_text())["cameras"][0]
    assert saved["source_path"].endswith("phone_chunk_02.mp4")
    # A frame at LiDAR time t (chunk clock) must land on the same phone frame as before the cut.
    t = 42.0
    before = (t + 285.0) + 5.0 + (-20.0)
    after = t + second["dt_sync_trimmed"] + saved["time_offset_s"] + phone_cut[2]
    assert after == pytest.approx(before)


def test_uncut_sources_only_get_the_offset_shift(tmp_path, fake_cuts):
    dt, _ = fake_cuts
    folder = tmp_path / "frames"
    folder.mkdir()
    cfg = _phone(1.5)
    cfg.source_path, cfg.source_type = str(folder), "folder"
    out = dt.cut_auxiliary_cameras([cfg], tmp_path / "o", "trimmed", 100.0, 50.0, 5.0, 105.0)
    assert out[0].source_path == str(folder)
    assert out[0].time_offset_s == pytest.approx(106.5)
    assert cfg.time_offset_s == 1.5  # the caller's config is left alone


def test_slice_video_maps_every_video_stream_as_mp4(tmp_path, monkeypatch):
    from raven_app import dataset_trim as dt
    seen = {}

    class Done:
        returncode, stderr = 0, ""

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return Done()

    monkeypatch.setattr(dt.subprocess, "run", fake_run)
    monkeypatch.setattr(dt, "get_video_duration", lambda p: 5.0)
    dt.slice_video(tmp_path / "a_00_1.insv", tmp_path / "b.insv", 10.0, 5.0)
    cmd = " ".join(seen["cmd"])
    assert "-map 0:v -map 0:a:0?" in cmd and "-f mp4" in cmd and "0:v:1" not in cmd
