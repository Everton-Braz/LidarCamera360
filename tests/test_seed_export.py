import json
import struct

import numpy as np
import pytest

from raven_app.seed_export import (
    cloud_header_point_count, dataset_seed_point_count,
    sample_seed_points, seed_point_count,
)
from raven_app.workflow import export_colmap_3dgs, load_lidar_seed_points, write_colmap_points3d


def test_default_preserves_full_cloud_and_percent_has_exact_budget():
    xyz = np.arange(303).reshape(101, 3)
    rgb = (xyz % 255).astype(np.uint8)
    full_xyz, full_rgb = sample_seed_points(xyz, rgb)
    assert full_xyz is xyz and full_rgb is rgb
    points, colors = sample_seed_points(xyz, rgb, 25)
    assert len(points) == seed_point_count(101, 25) == 25
    np.testing.assert_array_equal(colors, points % 255)
    assert np.unique(points[:, 0]).size == 25
    assert seed_point_count(1, 1) == 1
    assert seed_point_count(0, 100) == 0


@pytest.mark.parametrize('percent', [0, -1, 101, float('nan'), float('inf')])
def test_invalid_percent_rejected(percent):
    with pytest.raises(ValueError, match='Seed percentage'):
        seed_point_count(100, percent)


def test_seed_export_counts_match_all_formats_and_source_is_unchanged(tmp_path):
    deliverables = tmp_path / 'deliverables'
    xyz = np.arange(300, dtype=np.float32).reshape(100, 3)
    rgb = (xyz % 255).astype(np.uint8)
    write_colmap_points3d(deliverables, xyz, rgb)
    source = deliverables / 'lidar_colored_test.ply'
    (deliverables / 'points3D.ply').rename(source)
    original = source.read_bytes()
    assert dataset_seed_point_count(tmp_path) == 100
    loaded, _ = load_lidar_seed_points(tmp_path)
    np.testing.assert_array_equal(loaded, xyz)
    intrinsics = dict(fx=100, fy=100, cx=128, cy=128)
    calib = tmp_path / 'rig.json'
    calib.write_text(json.dumps({
        'cam0_front_intrinsics': intrinsics, 'cam1_rear_intrinsics': intrinsics,
        'T_lidar_to_cam0_rigid_4x4': np.eye(4).tolist(),
        'T_lidar_to_cam1_rigid_4x4': np.eye(4).tolist(),
    }))
    output = export_colmap_3dgs(tmp_path, calib, seed_percent=25)
    sparse = output / 'sparse' / '0'
    assert cloud_header_point_count(sparse / 'points3D.ply') == 25
    with (sparse / 'points3D.bin').open('rb') as stream:
        assert struct.unpack('<Q', stream.read(8))[0] == 25
    text = (sparse / 'points3D.txt').read_text()
    assert len([line for line in text.splitlines() if line and not line.startswith('#')]) == 25
    assert source.read_bytes() == original


def test_cli_percentage_default_and_explicit_value():
    from raven_app.cli import parse
    args = ['workflow', '--bag', 'scan.bag', '--insv', 'cam.insv', '--output', 'out']
    assert parse(args).seed_percent == 100
    assert parse(args + ['--seed-percent', '25.5']).seed_percent == 25.5
    with pytest.raises(SystemExit):
        parse(args + ['--seed-percent', '101'])
