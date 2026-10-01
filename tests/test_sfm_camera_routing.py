import struct

import numpy as np
import pytest

from raven_app.third_camera import ThirdCameraConfig
from scripts import pipeline_auto_calibrator_and_colorizer as pipeline


def test_per_folder_camera_models_and_focal_priors(tmp_path):
    for name in ('cam2', 'cam3'):
        folder = tmp_path / 'images' / name
        folder.mkdir(parents=True)
        (folder / 'frame_000001.jpg').touch()
    cameras = [ThirdCameraConfig(camera_name='cam2', camera_model='PINHOLE'),
               ThirdCameraConfig(camera_name='cam3', camera_model='OPENCV_FISHEYE')]
    cameras[0].intrinsics['fx'] = 3072
    cameras[1].intrinsics.update(fx=800, k3=.03, k4=.04)
    flags = pipeline._spirula_auxiliary_flags(tmp_path, cameras)
    assert 'cam2=pinhole' in flags
    assert 'cam2=3072' in flags
    assert 'cam3=opencv-fisheye' in flags
    assert 'cam3=0,0,0.03,0.04' in flags
    assert 'thin-prism-fisheye' not in flags


def test_mixed_colmap_binary_and_pinhole_projection(tmp_path):
    path = tmp_path / 'cameras.bin'
    with path.open('wb') as stream:
        stream.write(struct.pack('<Q', 2))
        stream.write(struct.pack('<iiQQ12d', 1, 10, 2880, 2880,
                                 1080, 1080, 1440, 1440, *([0] * 8)))
        stream.write(struct.pack('<iiQQ4d', 2, 1, 3840, 2160, 3072, 3072, 1920, 1080))
    cameras = pipeline.load_colmap_cameras(path)
    assert len(cameras[1]['params']) == 12
    assert len(cameras[2]['params']) == 4
    params = pipeline._colorization_camera_params(cameras[2])
    u, v, _ = pipeline.project_thin_prism(np.array([[1., .25, 2.]]), params)
    np.testing.assert_allclose(u, [3456])
    np.testing.assert_allclose(v, [1464])
    assert params[-1] == -999


def test_cached_wrong_auxiliary_model_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'load_colmap_cameras', lambda _: {3: {'model_id': 10}})
    monkeypatch.setattr(pipeline, 'load_colmap_images', lambda _: [
        {'cam_id': 3, 'name': 'cam2/frame_000001.jpg'}])
    with pytest.raises(ValueError, match='model mismatch for cam2'):
        pipeline._validate_sfm_camera_models(tmp_path, ['--camera-model', 'cam2=pinhole'])


def test_changed_frames_force_native_recompute_with_mixed_models(tmp_path, monkeypatch):
    folder = tmp_path / 'images' / 'cam2'
    folder.mkdir(parents=True)
    (folder / 'frame_000001.jpg').touch()
    sparse = tmp_path / 'sparse' / '0'
    sparse.mkdir(parents=True)
    for name in ('cameras.bin', 'images.bin', 'points3D.bin'):
        (sparse / name).touch()
    binary = tmp_path / 'spirula.exe'
    binary.touch()
    monkeypatch.setattr('raven_app.config.get_spirula_bin', lambda: binary)
    monkeypatch.setattr(pipeline, 'get_best_vulkan_device', lambda: -1)
    monkeypatch.setattr(pipeline, 'load_colmap_cameras', lambda _: {3: {'model_id': 1}})
    monkeypatch.setattr(pipeline, 'load_colmap_images', lambda _: [
        {'cam_id': 3, 'name': 'cam2/frame_000001.jpg'}])
    commands = []
    monkeypatch.setattr('raven_app.subprocess_utils.run_hidden_stream',
                        lambda command: commands.append(command) or 0)
    camera = ThirdCameraConfig(camera_name='cam2', camera_model='PINHOLE')
    assert pipeline.run_spirula_sfm_auto(tmp_path, auxiliary_cameras=[camera], force_rebuild=True)
    assert len(commands) == 1
    assert '--no-resume' in commands[0]
    assert 'thin-prism-fisheye' in commands[0]
    assert 'cam2=pinhole' in commands[0]


def test_photometric_sampling_handles_mixed_camera_models(tmp_path, monkeypatch):
    import cv2
    from raven_app.photometric import dataset
    sparse = tmp_path / 'sparse' / '0'
    sparse.mkdir(parents=True)
    cameras = {
        1: {'model_id': 10, 'width': 256, 'height': 256,
            'params': (100., 100., 128., 128.) + (0.,) * 8},
        2: {'model_id': 1, 'width': 256, 'height': 256,
            'params': (100., 100., 128., 128.)},
        3: {'model_id': 4, 'width': 256, 'height': 256,
            'params': (100., 100., 128., 128.) + (0.,) * 4},
    }
    points = np.column_stack((np.linspace(-.2, .2, 32), np.full(32, .1), np.full(32, 2.)))
    records = [(index, point, np.array([[1, index], [2, index], [3, index]]))
               for index, point in enumerate(points)]
    monkeypatch.setattr(dataset, '_sample_tracks', lambda *_: records)
    monkeypatch.setattr(pipeline, 'load_colmap_cameras', lambda _: cameras)
    with (sparse / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', 3))
        for camera_id, camera in cameras.items():
            name = f'cam{camera_id}/frame_000001.jpg'
            path = tmp_path / 'images' / name
            path.parent.mkdir(parents=True)
            cv2.imwrite(str(path), np.full((256, 256, 3), 100, dtype=np.uint8))
            stream.write(struct.pack('<i7di', camera_id, 1., 0., 0., 0., 0., 0., 0., camera_id))
            stream.write(name.encode() + b'\0' + struct.pack('<Q', len(points)))
            u, v, _ = pipeline.project_thin_prism(points, pipeline._colorization_camera_params(camera))
            for index in range(len(points)):
                stream.write(struct.pack('<ddq', u[index], v[index], index))
    observations, names, lenses = dataset.collect_observations(tmp_path, sample_points=32, max_observations=96)
    assert len(observations['rgb']) == 96
    assert len(names) == len(lenses) == 3
