import json
from types import SimpleNamespace

from raven_app import auxiliary_insv


def _config(path, camera_name="cam2", stream_index=None):
    return SimpleNamespace(
        source_path=str(path),
        camera_name=camera_name,
        source_stream_index=stream_index,
        source_type="video",
        camera_model="PINHOLE",
        description="configured auxiliary camera",
        fps=1.0,
    )


def _metadata(_path):
    return {
        "type": "insv", "lens_count": 2, "width": 1000, "height": 800,
        "streams": [
            {"index": 0, "width": 1000, "height": 800},
            {"index": 1, "width": 960, "height": 720},
        ],
    }


def test_expand_insv_keeps_first_name_and_allocates_next_free_for_second(tmp_path, monkeypatch):
    source = tmp_path / "aux.insv"
    source.touch()
    monkeypatch.setattr(auxiliary_insv, "inspect_auxiliary_insv", _metadata)

    expanded = auxiliary_insv.expand_auxiliary_insv_config(
        _config(source), used_camera_names=("cam3",)
    )

    assert [(item.camera_name, item.source_stream_index) for item in expanded] == [
        ("cam2", 0), ("cam4", 1)
    ]
    assert all(item.source_type == "insv" for item in expanded)
    assert all(item.camera_model == "THIN_PRISM_FISHEYE" for item in expanded)
    assert (expanded[0].intrinsics["width"], expanded[0].intrinsics["height"]) == (1000, 800)
    assert expanded[0].intrinsics["fx"] == 256
    assert (expanded[1].intrinsics["width"], expanded[1].intrinsics["height"]) == (960, 720)
    assert expanded[0].calibration_report["extrinsics"]["status"] == "uncalibrated"
    assert _config(source).source_type == "video"


def test_explicit_stream_keeps_existing_camera_name(tmp_path, monkeypatch):
    source = tmp_path / "aux.insv"
    source.touch()
    monkeypatch.setattr(auxiliary_insv, "inspect_auxiliary_insv", _metadata)

    expanded = auxiliary_insv.expand_auxiliary_insv_config(
        _config(source, camera_name="cam7", stream_index=1),
        used_camera_names=("cam8",),
    )

    assert len(expanded) == 1
    assert expanded[0].camera_name == "cam7"
    assert expanded[0].source_stream_index == 1


def test_reloading_expanded_calibrated_config_is_idempotent(tmp_path, monkeypatch):
    source = tmp_path / "aux.insv"
    source.touch()
    monkeypatch.setattr(
        auxiliary_insv,
        "inspect_auxiliary_insv",
        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected metadata read")),
    )
    config = _config(source, camera_name="cam3", stream_index=1)
    config.source_type = "insv"
    config.camera_model = "THIN_PRISM_FISHEYE"
    config.time_offset_s = 1.375
    config.intrinsics = {"width": 1234.0, "fx": 321.0}
    config.calibration_report = {"intrinsics": {"status": "sfm_calibrated"}}

    expanded = auxiliary_insv.expand_auxiliary_insv_config(config, used_camera_names=("cam3",))

    assert expanded == [config]
    assert expanded[0].camera_name == "cam3"
    assert expanded[0].time_offset_s == 1.375
    assert expanded[0].intrinsics == {"width": 1234.0, "fx": 321.0}
    assert expanded[0].calibration_report["intrinsics"]["status"] == "sfm_calibrated"


def test_explicit_stream_extracts_only_that_lens_and_keeps_timestamps(tmp_path, monkeypatch):
    source = tmp_path / "aux.insv"
    source.write_bytes(b"synthetic source")
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    (images / "frames.json").write_text(json.dumps({
        "time_source": "video_pts",
        "timestamps": {"cam0/frame_000001.jpg": 0.25},
    }), encoding="utf-8")
    monkeypatch.setattr(auxiliary_insv, "inspect_auxiliary_insv", lambda _path: {
        "type": "insv", "lens_count": 2, "width": 16, "height": 16,
    })

    def fake_primary_extract(_source, output, fps, sharp_window, jpeg_quality):
        cache_images = output / "images"
        for stream in (0, 1):
            folder = cache_images / f"cam{stream}"
            folder.mkdir(parents=True)
            (folder / "frame_000001.jpg").write_bytes(bytes([stream + 1]))
        (cache_images / "frames.json").write_text(json.dumps({
            "signature": {"fps": fps, "sharp_window": sharp_window,
                         "jpeg_quality": jpeg_quality},
            "time_source": "insv_timelapse",
            "timestamps": {
                "cam0/frame_000001.jpg": 2.125,
                "cam1/frame_000001.jpg": 2.125,
            },
        }), encoding="utf-8")
        return True

    monkeypatch.setattr("raven_app.video.extract_insv_frames_pyav", fake_primary_extract)
    config = _config(source, camera_name="cam7", stream_index=1)
    paths, timestamps = auxiliary_insv.extract_auxiliary_insv_frames(config, dataset)

    assert [path.relative_to(images).as_posix() for path in paths] == [
        "cam7/frame_000001.jpg"
    ]
    assert paths[0].read_bytes() == b"\x02"
    assert timestamps == {"cam7/frame_000001.jpg": 2.125}
    manifest = json.loads((images / "frames.json").read_text(encoding="utf-8"))
    assert manifest["timestamps"]["cam0/frame_000001.jpg"] == 0.25
    assert manifest["timestamps"]["cam7/frame_000001.jpg"] == 2.125
    assert manifest["time_source"] == "video_pts"
    assert manifest["time_sources"]["cam7"] == "insv_timelapse"
    assert manifest["auxiliary_insv_signatures"]["cam7"]["source_stream_index"] == 1
    assert auxiliary_insv.extract_auxiliary_insv_frames(config, dataset) == (paths, timestamps)


def test_expand_insv_single_lens_stream_keeps_single_camera(tmp_path, monkeypatch):
    source = tmp_path / "single.insv"
    source.touch()
    monkeypatch.setattr(auxiliary_insv, "inspect_auxiliary_insv", lambda _p: {
        "type": "insv", "lens_count": 1, "width": 3072, "height": 3072,
        "streams": [{"index": 0, "width": 3072, "height": 3072}],
    })

    expanded = auxiliary_insv.expand_auxiliary_insv_config(_config(source))
    assert len(expanded) == 1
    assert expanded[0].camera_name == "cam2"
    assert expanded[0].source_stream_index == 0
    assert expanded[0].source_type == "insv"
    assert expanded[0].camera_model == "THIN_PRISM_FISHEYE"


def test_single_stream_insv_extracts_stream_0(tmp_path, monkeypatch):
    source = tmp_path / "single.insv"
    source.write_bytes(b"synthetic single stream source")
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    (images / "frames.json").write_text(json.dumps({
        "time_source": "video_pts",
        "timestamps": {"cam0/frame_000001.jpg": 0.1},
    }), encoding="utf-8")

    monkeypatch.setattr(auxiliary_insv, "inspect_auxiliary_insv", lambda _path: {
        "type": "insv", "lens_count": 1, "width": 3072, "height": 3072,
        "streams": [{"index": 0, "width": 3072, "height": 3072}],
    })

    def fake_single_extract(source, output, stream_index=0, fps=1.0, sharp_window=3, jpeg_quality=95):
        cache_images = output / "images"
        folder = cache_images / f"cam{stream_index}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "frame_000001.jpg").write_bytes(b"single_frame")
        (cache_images / "frames.json").write_text(json.dumps({
            "signature": {"fps": fps, "sharp_window": sharp_window, "jpeg_quality": jpeg_quality},
            "time_source": "video_pts",
            "timestamps": {
                f"cam{stream_index}/frame_000001.jpg": 0.5,
            },
        }), encoding="utf-8")
        return True

    monkeypatch.setattr(auxiliary_insv, "extract_single_stream_insv_frames_pyav", fake_single_extract)

    config = _config(source, camera_name="cam2", stream_index=0)
    paths, timestamps = auxiliary_insv.extract_auxiliary_insv_frames(config, dataset)

    assert [path.relative_to(images).as_posix() for path in paths] == [
        "cam2/frame_000001.jpg"
    ]
    assert paths[0].read_bytes() == b"single_frame"
    assert timestamps == {"cam2/frame_000001.jpg": 0.5}


def test_single_stream_insv_rejects_unavailable_stream_1(tmp_path, monkeypatch):
    import pytest
    source = tmp_path / "single.insv"
    source.touch()
    monkeypatch.setattr(auxiliary_insv, "inspect_auxiliary_insv", lambda _path: {
        "type": "insv", "lens_count": 1, "width": 3072, "height": 3072,
        "streams": [{"index": 0, "width": 3072, "height": 3072}],
    })

    config = _config(source, camera_name="cam2", stream_index=1)
    with pytest.raises(ValueError, match="unavailable"):
        auxiliary_insv.extract_auxiliary_insv_frames(config, tmp_path / "dataset")

