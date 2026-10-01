"""Regression coverage for trajectory colorization after bounded sample SfM."""

import json
from pathlib import Path

from raven_app import workflow


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


def test_rejected_aux_camera_is_saved_but_excluded_from_colorization(
    tmp_path, monkeypatch
):
    output, bag, video = _dataset(tmp_path, monkeypatch)
    camera_path = tmp_path / "phone.mp4"
    camera_path.touch()
    camera_configs = [
        {
            "camera_name": "cam2",
            "source_path": str(camera_path),
            "calibration_report": {},
        },
        {
            "camera_name": "cam3",
            "source_path": str(camera_path),
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

    assert _run_direct(
        output, bag, video, third_camera=camera_configs, export_colmap=True
    ) == 0

    assert calibration_calls == [(('cam2', 'cam3'), 2.0, -0.125)]
    assert [camera.camera_name for camera in colorization["third_camera"]] == ["cam3"]
    assert colorization["dt_override"] == 2.75
    assert [camera.camera_name for camera in exported["third_camera"]] == ["cam2", "cam3"]

    saved = json.loads(
        (output / "third_camera_calibrated.json").read_text(encoding="utf-8")
    )
    assert [camera["camera_name"] for camera in saved["cameras"]] == ["cam2", "cam3"]
    cam2 = saved["cameras"][0]
    assert cam2["calibration_report"]["sample_sfm"]["status"] == "rejected"
    assert "ambiguous" in cam2["calibration_report"]["extrinsics"]["reason"]
