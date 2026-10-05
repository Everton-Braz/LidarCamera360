"""Regression coverage for trajectory colorization after bounded sample SfM."""

import json
from pathlib import Path

import numpy as np
import pytest

from raven_app import workflow
from raven_app.third_camera import ThirdCameraConfig


def test_bounded_sfm_sample_preserves_temporal_overlap(tmp_path):
    frames = [
        (index / 3.0, tmp_path / f"frame_{index:05d}.jpg", f"cam2/frame_{index:05d}.jpg")
        for index in range(3000)
    ]

    selected = workflow._select_bounded_sfm_frames(frames, 80, source_fps=3.0)

    gaps = [right[0] - left[0] for left, right in zip(selected, selected[1:])]
    assert len(selected) == 80
    assert selected[-1][0] - selected[0][0] <= 80.0
    assert max(gaps) <= 1.01
    assert min(gaps) >= 0.99


def _install_bounded_sample_stubs(tmp_path, monkeypatch, *, accept_at):
    output = tmp_path / "sample-dataset"
    names = ("cam0", "cam1", "cam2")
    timestamps = {}
    for camera_name in names:
        camera_dir = output / "images" / camera_name
        camera_dir.mkdir(parents=True)
        for index in range(100):
            filename = f"frame_{index:05d}.jpg"
            (camera_dir / filename).write_bytes(b"frame")
            timestamps[f"{camera_name}/{filename}"] = index * 0.5
    manifest = {
        "timestamps": timestamps,
        "signature": {"source": str(tmp_path / "main.insv")},
    }
    (output / "images" / "frames.json").write_text(json.dumps(manifest), encoding="utf-8")
    trajectory = output / "trajectory.txt"
    trajectory.write_text("trajectory", encoding="utf-8")
    camera_source = tmp_path / "phone.mp4"
    camera_source.touch()
    camera = ThirdCameraConfig(
        camera_name="cam2", source_path=str(camera_source), fps=2.0,
    )
    calls = []

    def run_sfm(sample_root, **_kwargs):
        sparse = Path(sample_root) / "sparse" / "0"
        sparse.mkdir(parents=True)
        for name in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse / name).write_bytes(b"fixture")
        return True

    import scripts.pipeline_auto_calibrator_and_colorizer as pipeline
    monkeypatch.setattr(pipeline, "get_slam_trajectory_path", lambda _dataset: trajectory)
    monkeypatch.setattr(pipeline, "run_spirula_sfm_auto", run_sfm)
    monkeypatch.setattr(pipeline, "align_colmap_to_lidar", lambda *_args, **_kwargs: {
        "scale": 1.0, "R": np.eye(3).tolist(), "t": [0.0, 0.0, 0.0],
        "dt_sync_seconds": 0.0, "trajectory_rmse_cm": 1.0,
        "alignment_method": "test", "quality_status": "accepted",
    })
    monkeypatch.setattr(pipeline, "recalibrate_from_sfm", lambda *_args, **_kwargs: {
        "dt_sync_seconds": 0.0,
    })
    monkeypatch.setattr(
        workflow, "_validate_sample_sfm_component_coverage",
        lambda _sparse, selected, _attempt: {
            name: {"selected_frames": len(frames), "registered_frames": len(frames)}
            for name, frames in selected.items()
        },
    )

    def align(sample_root, config, **_kwargs):
        attempt_index = len(calls) + 1
        frames = json.loads(
            (Path(sample_root) / "images" / "frames.json").read_text(encoding="utf-8"))
        calls.append({
            "attempt": attempt_index,
            "center": frames["sample_windows"]["cam0"]["center_seconds"],
            "hint": (config.calibration_report.get("time_offset", {}) or {}).get(
                "sample_window_hint_seconds"),
        })
        result = ThirdCameraConfig.from_dict(config.to_dict())
        ready = attempt_index >= accept_at
        result.calibration_report = {
            "intrinsics": {"status": "sfm_calibrated"},
            "time_offset": {
                "status": "calibrated" if ready else "unknown",
                "applied": ready,
                "best_candidate_seconds": -12.5,
            },
            "extrinsics": {
                "status": "calibrated" if ready else "uncalibrated",
                "reason": None if ready else "rig consensus was rejected",
            },
        }
        return result

    monkeypatch.setattr("raven_app.third_camera.align_third_camera_to_rig", align)
    return output, camera, calls


def test_bounded_sample_sfm_retries_with_fine_clock_hint_and_accepts_all_cameras(
    tmp_path, monkeypatch
):
    output, camera, calls = _install_bounded_sample_stubs(
        tmp_path, monkeypatch, accept_at=2)

    calibrated = workflow._calibrate_aux_cameras_from_sample_sfm(
        output, [camera], fps=2.0, dt_sync=0.0)

    assert len(calibrated) == 1
    assert workflow._aux_camera_calibration_ready(calibrated[0])
    assert [call["center"] for call in calls] == [24.75, 12.375]
    assert calls[0]["hint"] is None
    assert calls[1]["hint"] == -12.5
    assert len(list((output / "calibration" / "sample_sfm_attempt").iterdir())) == 2
    assert (output / "calibration" / "sample_sfm" / "colmap_to_lidar_alignment.json").is_file()
    assert json.loads((output / "rig_calibration.json").read_text(encoding="utf-8"))[
        "calibration_source"] == "sample_sfm"


def test_bounded_sample_sfm_preserves_three_rejected_windows_without_final_rig(
    tmp_path, monkeypatch
):
    output, camera, calls = _install_bounded_sample_stubs(
        tmp_path, monkeypatch, accept_at=99)

    with pytest.raises(RuntimeError, match="Required per-session sample SfM calibration failed"):
        workflow._calibrate_aux_cameras_from_sample_sfm(
            output, [camera], fps=2.0, dt_sync=0.0)

    assert len(calls) == 3
    attempts = list((output / "calibration" / "sample_sfm_attempt").iterdir())
    assert len(attempts) == 3
    assert not (output / "rig_calibration.json").exists()
    assert not (output / "calibration" / "sample_sfm").exists()
    assert camera.calibration_report["sample_sfm"]["status"] == "failed"
    assert len(camera.calibration_report["sample_sfm"]["attempts"]) == 3


def _dataset(tmp_path, monkeypatch):
    output = tmp_path / "dataset"
    (output / "slam_out" / "pcd").mkdir(parents=True)
    (output / "slam_out" / "result").mkdir(parents=True)
    (output / "slam_out" / "pcd" / "all_raw_points.pcd").write_bytes(b"pcd")
    (output / "slam_out" / "result" / "Raven_3DMakerPro_Scan.txt").write_text(
        "trajectory", encoding="utf-8"
    )
    bag = tmp_path / "scan.bag"
    video = tmp_path / "camera.insv"
    bag.touch()
    video.touch()

    def extract(_video, destination, fps):
        images = Path(destination) / "images"
        images.mkdir(parents=True, exist_ok=True)
        (images / "frames.json").write_text(
            json.dumps({"time_source": "insv", "fps": fps, "timestamps": {}}),
            encoding="utf-8",
        )
        return True

    monkeypatch.setattr(workflow, "extract_insv_frames", extract)
    monkeypatch.setattr(workflow, "auto_sync_imu_gyro", lambda *args, **kwargs: -0.125)
    return output, bag, video


def _install_sample_calibration(monkeypatch, reports, calls):
    def calibrate(output_dir, cameras, fps, dt_sync):
        calls.append((tuple(camera.camera_name for camera in cameras), fps, dt_sync))
        rig = {
            "calibration_source": "sample_sfm",
            "dt_sync_seconds": 2.75,
            "T_LC0": "sample calibrated primary transform",
            "T_LC1": "sample calibrated secondary transform",
        }
        (Path(output_dir) / "rig_calibration.json").write_text(
            json.dumps(rig), encoding="utf-8"
        )
        for camera in cameras:
            camera.calibration_report = reports[camera.camera_name]
        return cameras

    monkeypatch.setattr(workflow, "_calibrate_aux_cameras_from_sample_sfm", calibrate)


def _run_direct(output, bag, video, *, third_camera=None, export_colmap=False):
    return workflow.execute_unified_workflow(
        bag,
        video,
        output,
        method="direct",
        lio=False,
        recalibrate=True,
        export_ply=False,
        export_pcd=False,
        export_colmap=export_colmap,
        third_camera=third_camera,
    )


def test_direct_without_aux_camera_uses_sample_rig_and_clock(tmp_path, monkeypatch):
    output, bag, video = _dataset(tmp_path, monkeypatch)
    calibration_calls = []
    _install_sample_calibration(monkeypatch, {}, calibration_calls)
    colorization = {}

    def colorize(dataset, calib_path, **kwargs):
        colorization.update(dataset=dataset, calib_path=calib_path, **kwargs)

    monkeypatch.setattr(
        "scripts.pipeline_auto_calibrator_and_colorizer.colorize_via_direct_rigid",
        colorize,
    )

    assert _run_direct(output, bag, video) == 0

    assert calibration_calls == [((), 2.0, -0.125)]
    assert Path(colorization["calib_path"]) == output.resolve() / "rig_calibration.json"
    assert colorization["dt_override"] == 2.75
    assert colorization["third_camera"] == []
    saved_rig = json.loads((output / "rig_calibration.json").read_text(encoding="utf-8"))
    assert saved_rig["calibration_source"] == "sample_sfm"
    assert saved_rig["T_LC0"] == "sample calibrated primary transform"
    saved_run = json.loads((output / "workflow_config.json").read_text(encoding="utf-8"))["runs"][-1]
    assert saved_run["status"] == "complete"
    assert saved_run["settings"]["fps_by_camera"] == {"cam0": 2.0, "cam1": 2.0}
    assert [camera["camera_name"] for camera in saved_run["cameras"]] == ["cam0", "cam1"]
    assert saved_run["image_extraction"]["manifest"] == "images/frames.json"


def test_workflow_expands_auxiliary_insv_into_both_lens_views(tmp_path, monkeypatch):
    output, bag, video = _dataset(tmp_path, monkeypatch)
    auxiliary_insv = tmp_path / "auxiliary.insv"
    auxiliary_insv.touch()
    input_camera = ThirdCameraConfig(
        camera_name="cam2", source_path=str(auxiliary_insv), source_type="insv",
        source_stream_index=None, fps=1.0, camera_model="THIN_PRISM_FISHEYE",
    )
    expanded = [
        ThirdCameraConfig(
            camera_name=f"cam{index}", source_path=str(auxiliary_insv),
            source_type="insv", source_stream_index=stream, fps=1.0,
            camera_model="THIN_PRISM_FISHEYE",
        )
        for index, stream in ((2, 0), (3, 1))
    ]
    expansion_calls = []

    def expand(cameras):
        expansion_calls.append(tuple(camera.camera_name for camera in cameras))
        return expanded

    monkeypatch.setattr("raven_app.third_camera.expand_auxiliary_camera_configs", expand)
    reports = {
        camera.camera_name: {
            "intrinsics": {"status": "sfm_calibrated"},
            "time_offset": {"status": "calibrated", "applied": True},
            "extrinsics": {"status": "calibrated"},
            "sample_sfm": {"status": "complete"},
        }
        for camera in expanded
    }
    calibration_calls = []
    _install_sample_calibration(monkeypatch, reports, calibration_calls)
    extracted = []

    def extract(camera, _output):
        extracted.append((camera.camera_name, camera.source_stream_index))
        return [], None

    monkeypatch.setattr("raven_app.third_camera.extract_third_camera_frames", extract)
    colorization = {}
    monkeypatch.setattr(
        "scripts.pipeline_auto_calibrator_and_colorizer.colorize_via_direct_rigid",
        lambda dataset, calib_path, **kwargs: colorization.update(
            dataset=dataset, calib_path=calib_path, **kwargs),
    )

    assert _run_direct(output, bag, video, third_camera=input_camera) == 0

    assert expansion_calls == [("cam2",)]
    assert extracted == [("cam2", 0), ("cam3", 1)]
    assert [camera.camera_name for camera in colorization["third_camera"]] == ["cam2", "cam3"]
    saved_run = json.loads((output / "workflow_config.json").read_text(encoding="utf-8"))["runs"][-1]
    assert [camera["camera_name"] for camera in saved_run["cameras"]] == [
        "cam0", "cam1", "cam2", "cam3",
    ]
    assert saved_run["settings"]["fps_by_camera"]["cam2"] == 1.0
    assert saved_run["settings"]["fps_by_camera"]["cam3"] == 1.0


def test_insv_expansion_signatures_are_snapshotted_per_lens(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    signatures = {"cam2": {"fps": 1.0, "source_stream_index": 0},
                  "cam3": {"fps": 1.0, "source_stream_index": 1}}
    (images / "frames.json").write_text(json.dumps({
        "timestamps": {},
        "auxiliary_insv_signatures": signatures,
        "signature_cam2": {"stale": True},
    }), encoding="utf-8")
    cameras = [
        ThirdCameraConfig(camera_name=name, source_path="auxiliary.insv", source_type="insv")
        for name in ("cam2", "cam3")
    ]

    snapshot = workflow._workflow_image_extraction_snapshot(tmp_path, cameras)

    assert snapshot["camera_signatures"]["cam2"] == signatures["cam2"]
    assert snapshot["camera_signatures"]["cam3"] == signatures["cam3"]


def test_rejected_aux_camera_is_saved_and_blocks_colorization(
    tmp_path, monkeypatch
):
    output, bag, video = _dataset(tmp_path, monkeypatch)
    camera_path = tmp_path / "phone.mp4"
    camera_path.touch()
    camera_configs = [
        {
            "camera_name": "cam2",
            "source_path": str(camera_path),
            "fps": 3.0,
            "calibration_report": {},
        },
        {
            "camera_name": "cam3",
            "source_path": str(camera_path),
            "fps": 1.5,
            "calibration_report": {},
        },
    ]
    reports = {
        "cam2": {
            "intrinsics": {"status": "sfm_calibrated"},
            "extrinsics": {
                "status": "uncalibrated",
                "reason": "phone-video time offset is ambiguous",
            },
            "sample_sfm": {
                "status": "rejected",
                "reason": "no rigid-pose candidate passed quality gates",
            },
        },
        "cam3": {
            "intrinsics": {"status": "sfm_calibrated"},
            "time_offset": {"status": "configured", "applied": True},
            "extrinsics": {"status": "calibrated"},
            "sample_sfm": {"status": "complete"},
        },
    }
    calibration_calls = []
    _install_sample_calibration(monkeypatch, reports, calibration_calls)
    monkeypatch.setattr(
        "raven_app.third_camera.extract_third_camera_frames",
        lambda camera, _output: ([], None),
    )
    colorization = {}

    def colorize(dataset, calib_path, **kwargs):
        colorization.update(dataset=dataset, calib_path=calib_path, **kwargs)

    monkeypatch.setattr(
        "scripts.pipeline_auto_calibrator_and_colorizer.colorize_via_direct_rigid",
        colorize,
    )
    exported = {}

    def export(dataset, calib_path, **kwargs):
        exported.update(dataset=dataset, calib_path=calib_path, **kwargs)
        return Path(dataset) / "colmap_3dgs"

    monkeypatch.setattr(workflow, "export_colmap_3dgs", export)

    with pytest.raises(RuntimeError, match="Enabled auxiliary cameras must all pass calibration"):
        _run_direct(output, bag, video, third_camera=camera_configs, export_colmap=True)

    assert calibration_calls == [(('cam2', 'cam3'), 2.0, -0.125)]
    assert colorization == {}
    assert exported == {}

    saved = json.loads(
        (output / "third_camera_calibrated.json").read_text(encoding="utf-8")
    )
    assert [camera["camera_name"] for camera in saved["cameras"]] == ["cam2", "cam3"]
    cam2 = saved["cameras"][0]
    assert cam2["calibration_report"]["sample_sfm"]["status"] == "rejected"
    assert "ambiguous" in cam2["calibration_report"]["extrinsics"]["reason"]
    saved_run = json.loads((output / "workflow_config.json").read_text(encoding="utf-8"))["runs"][-1]
    assert saved_run["status"] == "failed"
    assert saved_run["failure"]["stage"] == "auxiliary_camera_calibration"
    assert saved_run["settings"]["fps_by_camera"] == {
        "cam0": 2.0, "cam1": 2.0, "cam2": 3.0, "cam3": 1.5,
    }
    assert [camera["fps"] for camera in saved_run["cameras"]] == [2.0, 2.0, 3.0, 1.5]


def test_required_sample_rig_calibration_failure_is_recorded_and_stops_workflow(
    tmp_path, monkeypatch
):
    output, bag, video = _dataset(tmp_path, monkeypatch)

    def reject_sample(*_args, **_kwargs):
        raise RuntimeError("partial sparse model")

    monkeypatch.setattr(workflow, "_calibrate_aux_cameras_from_sample_sfm", reject_sample)
    with pytest.raises(RuntimeError, match="Required per-session sample SfM camera calibration failed"):
        _run_direct(output, bag, video)

    saved_run = json.loads((output / "workflow_config.json").read_text(encoding="utf-8"))["runs"][-1]
    assert saved_run["status"] == "failed"
    assert saved_run["failure"] == {
        "stage": "sample_sfm_rig_calibration",
        "reason": "partial sparse model",
    }
