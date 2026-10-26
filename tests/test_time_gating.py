import json
import struct

import cv2
import numpy as np
import pytest

from raven_app import vulkan_engine
from raven_app.time_gating import point_scan_times, window_slice
from scripts import pipeline_auto_calibrator_and_colorizer as pipeline


def _write_scan_ranges(slam_dir, rows):
    slam_dir.mkdir(parents=True, exist_ok=True)
    lines = ['timestamp,point_start,point_count'] + [f'{t},{s},{c}' for t, s, c in rows]
    (slam_dir / 'scan_ranges.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _write_pcd(path, xyz):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype='<f4')
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
              f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n')
    with open(path, 'wb') as stream:
        stream.write(header.encode('ascii'))
        xyz.tofile(stream)


def test_point_scan_times_follow_scan_ranges_and_finite_mask(tmp_path):
    _write_scan_ranges(tmp_path, [(100.0, 0, 2), (100.1, 2, 3)])
    finite = np.array([True, False, True, True, False])

    times = point_scan_times(tmp_path, finite)

    np.testing.assert_allclose(times, [100.0, 100.1, 100.1])


@pytest.mark.parametrize('rows', [
    [(100.0, 0, 2)],                      # fewer points listed than the cloud has
    [(100.0, 0, 2), (100.1, 3, 3)],       # gap between blocks
    [(100.1, 0, 2), (100.0, 2, 3)],       # timestamps going backwards
])
def test_point_scan_times_disable_gating_when_table_does_not_describe_cloud(tmp_path, rows):
    _write_scan_ranges(tmp_path, rows)
    assert point_scan_times(tmp_path, np.ones(5, bool)) is None


def test_point_scan_times_disable_gating_without_table(tmp_path):
    assert point_scan_times(tmp_path, np.ones(3, bool)) is None


def test_window_slice_is_inclusive_on_both_ends():
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert window_slice(times, 2.5, 1.5) == slice(1, 5)
    assert window_slice(times, 10.0, 1.0) == slice(6, 6)


def test_vulkan_protocol_serializes_time_gate(tmp_path, monkeypatch):
    binary = tmp_path / 'vulkan_colorizer.exe'
    binary.touch()
    monkeypatch.setattr(vulkan_engine, 'get_vulkan_bin', lambda: binary)
    seen = {}

    def run_fake(command):
        job = command[command.index('--job') + 1]
        output = command[command.index('--out') + 1]
        with open(job, 'rb') as stream:
            magic, point_count, view_count = struct.unpack('<4sII', stream.read(12))
            points = np.frombuffer(stream.read(point_count * 16), dtype='<f4').reshape(-1, 4)
            gates = []
            for _ in range(view_count):
                name_length = struct.unpack('<I', stream.read(4))[0]
                stream.seek(name_length + 24 * 4, 1)
                gates.append(np.frombuffer(stream.read(8), dtype='<f4'))
        seen.update(magic=magic, w=points[:, 3].copy(), gates=gates)
        with open(output, 'wb') as stream:
            stream.write(bytes((9, 8, 7, 255)) * point_count)
        return 0

    monkeypatch.setattr(vulkan_engine, 'run_hidden_stream', run_fake)
    params = (10., 10., 5., 5., 0., 0., 0., 0., 0., 0., 0., -999.)
    views = [(tmp_path / 'images/cam0/a.jpg', np.eye(3), np.zeros(3), params),
             (tmp_path / 'images/cam0/b.jpg', np.eye(3), np.zeros(3), params)]
    points = np.array([[0., 0., 3.], [0., 0., 4.], [0., 0., 5.]])

    colors = vulkan_engine.colorize_views(
        points, views, tmp_path, point_times=np.array([1000.0, 1000.5, 1015.0]),
        view_times=[1000.2, None], time_window=6.0)

    assert seen['magic'] == b'RVT1'
    np.testing.assert_allclose(seen['w'], [0.0, 0.5, 15.0])
    np.testing.assert_allclose(seen['gates'][0], [0.2, 6.0], atol=1e-5)
    np.testing.assert_allclose(seen['gates'][1], [0.0, 0.0])  # views without a capture time stay ungated
    np.testing.assert_array_equal(colors, [[9, 8, 7]] * 3)


def test_vulkan_protocol_without_time_gate_is_unchanged(tmp_path, monkeypatch):
    binary = tmp_path / 'vulkan_colorizer.exe'
    binary.touch()
    monkeypatch.setattr(vulkan_engine, 'get_vulkan_bin', lambda: binary)
    seen = {}

    def run_fake(command):
        with open(command[command.index('--job') + 1], 'rb') as stream:
            seen['magic'] = stream.read(4)
            stream.seek(8, 1)
            seen['w'] = np.frombuffer(stream.read(16), dtype='<f4')[3]
        with open(command[command.index('--out') + 1], 'wb') as stream:
            stream.write(bytes(4))
        return 0

    monkeypatch.setattr(vulkan_engine, 'run_hidden_stream', run_fake)
    params = (10., 10., 5., 5., 0., 0., 0., 0., 0., 0., 0., -999.)
    vulkan_engine.colorize_views(np.array([[0., 0., 3.]]), [(tmp_path / 'a.jpg', np.eye(3), np.zeros(3), params)],
                                 tmp_path, point_times=np.array([5.0]), view_times=[5.0], time_window=0.0)

    assert seen == {'magic': b'RVC1', 'w': 0.0}


def _dynamic_dataset(tmp_path):
    """One spot scanned on two passes 15 s apart: red on pass 1, blue on pass 2."""
    root = tmp_path / 'dataset'
    slam = root / 'slam_out'
    t0 = 1000.0
    times = np.arange(t0, t0 + 20.0, 0.1)
    trajectory = np.column_stack([times, np.zeros((len(times), 3)),
                                  np.tile([0., 0., 0., 1.], (len(times), 1))])
    (slam / 'result').mkdir(parents=True)
    np.savetxt(slam / 'result' / 'Raven_3DMakerPro_Scan.txt', trajectory, fmt='%.6f')
    grid = np.stack(np.meshgrid(np.linspace(-.6, .6, 10), np.linspace(-.6, .6, 10)), -1).reshape(-1, 2)
    patch = np.column_stack([grid, np.full(len(grid), 3.0)])
    _write_pcd(slam / 'pcd' / 'all_raw_points.pcd', np.vstack([patch, patch]))
    _write_scan_ranges(slam, [(t0 + 0.5, 0, len(patch)), (t0 + 15.5, len(patch), len(patch))])

    for cam in ('cam0', 'cam1'):
        (root / 'images' / cam).mkdir(parents=True)
    colours = {'frame_000001.jpg': (0, 0, 255), 'frame_000002.jpg': (255, 0, 0)}  # BGR: red, blue
    for name, bgr in colours.items():
        cv2.imwrite(str(root / 'images' / 'cam0' / name), np.full((64, 64, 3), bgr, np.uint8))
        cv2.imwrite(str(root / 'images' / 'cam1' / name), np.zeros((64, 64, 3), np.uint8))
    stamps = {f'{cam}/frame_000001.jpg': 0.5 for cam in ('cam0', 'cam1')}
    stamps.update({f'{cam}/frame_000002.jpg': 15.5 for cam in ('cam0', 'cam1')})
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': stamps}), encoding='utf-8')

    pinhole = dict(fx=32., fy=32., cx=32., cy=32., k1=0., k2=0., p1=0., p2=0., k3=0., k4=0., sx1=0., sy1=-999.)
    backwards = np.diag([-1., 1., -1., 1.])  # cam1 looks away from the patch
    calib = root / 'calib.json'
    calib.write_text(json.dumps({
        'T_lidar_to_cam0_rigid_4x4': np.eye(4).tolist(), 'T_lidar_to_cam1_rigid_4x4': backwards.tolist(),
        'cam0_front_intrinsics': pinhole, 'cam1_rear_intrinsics': pinhole,
    }), encoding='utf-8')
    return root, calib, len(patch)


def _read_ply_rgb(path):
    with open(path, 'rb') as stream:
        while stream.readline().strip() != b'end_header':
            pass
        data = np.frombuffer(stream.read(), dtype=[('xyz', '<f4', 3), ('rgb', 'u1', 3)])
    return data['rgb']


@pytest.mark.parametrize('window', [6.0, 0.0])
def test_direct_colorization_uses_only_views_from_the_points_own_pass(tmp_path, window):
    root, calib, n = _dynamic_dataset(tmp_path)

    ply, _ = pipeline.colorize_via_direct_rigid(root, calib, fps=1.0, dt_override=0.0,
                                               use_vulkan=False, time_window=window)

    rgb = _read_ply_rgb(ply).astype(int)
    first_pass, second_pass = rgb[:n], rgb[n:]
    if window:
        assert (first_pass[:, 0] > 200).all() and (first_pass[:, 2] < 50).all()   # red
        assert (second_pass[:, 2] > 200).all() and (second_pass[:, 0] < 50).all()  # blue
    else:
        # Ungated, both passes collapse to one colour: the ghosting this guards against.
        np.testing.assert_array_equal(first_pass, second_pass)
