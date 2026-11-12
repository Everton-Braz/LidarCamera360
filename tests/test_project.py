import json
from pathlib import Path

import numpy as np
import pytest

from raven_app import project


def _ply(path: Path, n=50):
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 5, (n, 3)).astype('<f4')
    rgb = rng.integers(0, 255, (n, 3)).astype('u1')
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode()
    rec = np.zeros(n, dtype=[('p', '<f4', 3), ('c', 'u1', 3)])
    rec['p'], rec['c'] = pts, rgb
    path.write_bytes(head + rec.tobytes())


def _dataset(root: Path):
    (root / 'deliverables').mkdir(parents=True)
    _ply(root / 'deliverables' / 'cloud_colorized.ply')
    slam = root / 'slam_raw_abc' / 'result'
    slam.mkdir(parents=True)
    (slam / 'traj.txt').write_text('100.0 0 0 0 0 0 0 1\n101.0 1 0 0 0 0 0 1\n')
    (root / 'active_slam.json').write_text(json.dumps({'relative_directory': 'slam_raw_abc'}))
    sparse = root / 'colmap_3dgs' / 'sparse' / '0'
    sparse.mkdir(parents=True)
    (sparse / 'cameras.txt').write_text('1 THIN_PRISM_FISHEYE 100 100 50 50 50 50 0 0 0 0 0 0 0 0\n')
    (sparse / 'images.txt').write_text('1 1 0 0 0 1 2 3 1 cam0/frame_000001.jpg\n\n')
    (root / 'images').mkdir()
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': {'cam0/frame_000001.jpg': 0.5}}))


def test_write_project_builds_manifest_and_poses(tmp_path):
    _dataset(tmp_path)
    manifest = project.write_project(tmp_path, dt_sync=0.25, run={'run_id': 'a'})
    assert manifest['schema'] == 'lc360-project'
    assert manifest['assets']['cloud'] == 'deliverables/cloud_colorized.ply'
    assert manifest['assets']['trajectory'] == 'slam_raw_abc/result/traj.txt'
    poses = json.loads((tmp_path / manifest['assets']['camera_poses']).read_text())
    image = poses['images'][0]
    assert image['cam'] == 'cam0' and image['frame'] == 1
    assert image['t'] == pytest.approx(100.0 + 0.5 - 0.25)
    # identity rotation, t=(1,2,3): camera centre is -t
    assert image['position'] == [-1.0, -2.0, -3.0]


def test_second_run_updates_the_same_project(tmp_path):
    _dataset(tmp_path)
    project.write_project(tmp_path, run={'run_id': 'a'}, insv='x.insv')
    manifest = project.write_project(tmp_path, run={'run_id': 'b'})
    assert [r['run_id'] for r in manifest['runs']] == ['a', 'b']
    assert manifest['capture']['insv'] == 'x.insv'


def test_export_formats_from_project(tmp_path):
    _dataset(tmp_path)
    project.write_project(tmp_path)
    made = project.export_from_project(tmp_path, ['pcd', 'laz', 'trajectory'])
    assert Path(made['pcd'][0]).is_file() and Path(made['laz'][0]).is_file()
    assert Path(made['trajectory'][1]).read_text().startswith('t,x,y,z')
    with pytest.raises(ValueError, match='INSV'):
        project.export_from_project(tmp_path, ['gps'])
    with pytest.raises(ValueError, match='unknown'):
        project.export_from_project(tmp_path, ['dwg'])
