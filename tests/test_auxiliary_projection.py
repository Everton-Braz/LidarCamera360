import json
import struct

import numpy as np
import pytest

from raven_app.third_camera import ThirdCameraConfig
from raven_app import vulkan_engine
from scripts import pipeline_auto_calibrator_and_colorizer as pipeline


def _camera(model='THIN_PRISM_FISHEYE'):
    camera = ThirdCameraConfig(camera_name='cam2', camera_model=model)
    camera.intrinsics.update({
        'width': 640, 'height': 480, 'fx': 410.0, 'fy': 405.0,
        'cx': 320.0, 'cy': 240.0, 'k1': .01, 'k2': -.002,
        'p1': .003, 'p2': -.004, 'k3': .0005, 'k4': -.0001,
        'sx1': .002, 'sy1': -.001,
    })
    return camera


def test_auxiliary_thin_prism_model_and_all_distortion_reach_sfm(tmp_path):
    folder = tmp_path / 'images' / 'cam2'
    folder.mkdir(parents=True)
    (folder / 'frame_000001.jpg').touch()
    camera = _camera()

    flags = pipeline._spirula_auxiliary_flags(tmp_path, [camera])

    assert 'cam2=thin-prism-fisheye' in flags
    assert 'cam2=410' in flags
    assert 'cam2=0.01,-0.002,0.003,-0.004,0.0005,-0.0001,0.002,-0.001' in flags


def test_sfm_validation_accepts_auxiliary_thin_prism_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'load_colmap_cameras', lambda _: {4: {'model_id': 10}})
    monkeypatch.setattr(pipeline, 'load_colmap_images', lambda _: [
        {'cam_id': 4, 'name': 'cam2/frame_000001.jpg'}])

    pipeline._validate_sfm_camera_models(
        tmp_path, ['--camera-model', 'cam2=thin-prism-fisheye'])


@pytest.mark.parametrize(('model', 'tail'), [
    ('PINHOLE', (0., 0., 0., 0., 0., 0., 0., -999.)),
    ('OPENCV', (.01, -.002, .003, -.004, 0., 0., 0., -998.)),
    ('OPENCV_FISHEYE', (.01, -.002, 0., 0., .0005, -.0001, 0., 0.)),
    ('THIN_PRISM_FISHEYE', (.01, -.002, .003, -.004, .0005, -.0001, .002, -.001)),
])
def test_auxiliary_projection_parameters_follow_selected_model(model, tail):
    params = pipeline._auxiliary_colorization_camera_params(_camera(model))
    assert params[:4] == (410., 405., 320., 240.)
    assert params[4:] == tail


def test_thin_prism_auxiliary_projection_uses_prism_coefficients():
    camera = _camera()
    params = pipeline._auxiliary_colorization_camera_params(camera)
    point = np.array([[.4, -.25, 1.7]], dtype=np.float64)
    u, v, _ = pipeline.project_thin_prism(point, params)

    theta = np.arctan2(np.hypot(point[0, 0], point[0, 1]), point[0, 2])
    x, y = point[0, 0] * theta / np.hypot(*point[0, :2]), point[0, 1] * theta / np.hypot(*point[0, :2])
    theta2 = theta * theta
    radial = 1 + theta2 * (camera.intrinsics['k1'] + theta2 * (
        camera.intrinsics['k2'] + theta2 * (camera.intrinsics['k3'] + theta2 * camera.intrinsics['k4'])))
    xd = x * radial + 2 * camera.intrinsics['p1'] * x * y + camera.intrinsics['p2'] * (theta2 + 2 * x * x) + camera.intrinsics['sx1'] * theta2
    yd = y * radial + camera.intrinsics['p1'] * (theta2 + 2 * y * y) + 2 * camera.intrinsics['p2'] * x * y + camera.intrinsics['sy1'] * theta2
    np.testing.assert_allclose(u, [camera.intrinsics['fx'] * xd + camera.intrinsics['cx']])
    np.testing.assert_allclose(v, [camera.intrinsics['fy'] * yd + camera.intrinsics['cy']])


def test_vulkan_protocol_serializes_auxiliary_thin_prism_coefficients(tmp_path, monkeypatch):
    params = pipeline._auxiliary_colorization_camera_params(_camera())
    binary = tmp_path / 'vulkan_colorizer.exe'
    binary.touch()
    monkeypatch.setattr(vulkan_engine, 'get_vulkan_bin', lambda: binary)
    seen = {}

    def run_fake(command):
        job = command[command.index('--job') + 1]
        output = command[command.index('--out') + 1]
        with open(job, 'rb') as stream:
            magic, point_count, view_count = struct.unpack('<4sII', stream.read(12))
            stream.seek(point_count * 16, 1)
            name_length = struct.unpack('<I', stream.read(4))[0]
            stream.seek(name_length, 1)
            packed = np.frombuffer(stream.read(24 * 4), dtype='<f4')
        seen['header'] = (magic, point_count, view_count)
        seen['params'] = packed[-12:]
        with open(output, 'wb') as stream:
            stream.write(bytes((1, 2, 3, 255)) * point_count)
        return 0

    monkeypatch.setattr(vulkan_engine, 'run_hidden_stream', run_fake)
    points = np.array([[0., 0., 0.], [1., .5, 2.]])
    views = [(tmp_path / 'images/cam2/frame.jpg', np.eye(3), np.zeros(3), params)]

    colors = vulkan_engine.colorize_views(points, views, tmp_path)

    assert seen['header'] == (b'RVC1', 2, 1)
    np.testing.assert_allclose(seen['params'], params, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(colors, [[1, 2, 3], [1, 2, 3]])


def test_sample_alignment_cache_uses_external_trajectory_signature(tmp_path, monkeypatch):
    sample = tmp_path / 'sample_sfm'
    sample.mkdir()
    trajectory = tmp_path / 'active_slam' / 'trajectory.txt'
    trajectory.parent.mkdir()
    trajectory.write_text('trajectory source')
    signature = pipeline._active_slam_signature(
        sample, trajectory_path=trajectory, include_cloud=False)
    saved = {
        'calibration_version': pipeline.CALIBRATION_VERSION,
        'slam_source_signature': signature,
        'dt_sync_seconds': -2.0,
    }
    alignment_path = sample / 'colmap_to_lidar_alignment.json'
    alignment_path.write_text(json.dumps(saved))

    monkeypatch.setattr(pipeline, 'align_colmap_to_lidar',
                        lambda *args, **kwargs: pytest.fail('valid sample alignment was recomputed'))
    assert pipeline.current_alignment(sample) == saved

    trajectory.write_text('updated trajectory source')
    seen = {}
    monkeypatch.setattr(pipeline, 'align_colmap_to_lidar',
                        lambda *args, **kwargs: seen.update(kwargs) or {'recomputed': True})
    assert pipeline.current_alignment(sample) == {'recomputed': True}
    assert seen['trajectory_path'] == trajectory
    assert seen['use_icp'] is False


def test_timelapse_alignment_persists_source_signature(tmp_path, monkeypatch):
    dataset = tmp_path / 'dataset'
    manifest = dataset / 'images' / 'frames.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'time_source': 'insv_timelapse'}))
    trajectory = tmp_path / 'slam' / 'trajectory.txt'
    trajectory.parent.mkdir()
    trajectory.write_text('trajectory source')

    from raven_app import timelapse_calibration
    monkeypatch.setattr(timelapse_calibration, 'align_dataset', lambda *_args, **_kwargs: {
        'calibration_version': pipeline.CALIBRATION_VERSION,
        'frame_time_source': 'insv_timelapse',
        'timelapse_calibration_version': timelapse_calibration.VERSION,
        'quality_status': 'accepted',
    })

    result = pipeline.align_colmap_to_lidar(
        dataset, trajectory_path=trajectory, use_icp=False)

    assert result['slam_source_signature'] == pipeline._active_slam_signature(
        dataset, trajectory_path=trajectory, include_cloud=False)
    persisted = json.loads((dataset / 'colmap_to_lidar_alignment.json').read_text())
    assert persisted['slam_source_signature'] == result['slam_source_signature']
