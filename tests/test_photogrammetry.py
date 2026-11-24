import math
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from raven_app import openmvs_dense as od
from raven_app import photogrammetry_export as pe
from raven_app.dense_fusion import write_ply

W = 400
LENS = (W / 3.5, W / 3.5, W / 2, W / 2, 0.02, -0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture
def colmap(tmp_path):
    """Three fisheye photos, 0.5 m apart, in front of a coloured wall."""
    root = tmp_path / 'colmap_3dgs'
    sparse = root / 'sparse' / '0'
    sparse.mkdir(parents=True)
    (root / 'images' / 'cam0').mkdir(parents=True)
    (root / 'masks' / 'cam0').mkdir(parents=True)
    (sparse / 'cameras.txt').write_text(f'1 THIN_PRISM_FISHEYE {W} {W} ' + ' '.join(map(str, LENS)) + '\n')
    lines = []
    for i in range(3):
        name = f'cam0/frame_{i:06d}.jpg'
        lines += [f'{i + 1} 1 0 0 0 {-0.5 * i} 0 0 1 {name}', '']
        cv2.imwrite(str(root / 'images' / name), np.full((W, W, 3), 60 * i + 60, np.uint8))
        mask = np.full((W, W), 255, np.uint8)
        mask[:, :W // 4] = 0
        cv2.imwrite(str(root / 'masks' / (name + '.png')), mask)
    (sparse / 'images.txt').write_text('\n'.join(lines) + '\n')
    g = np.arange(-1, 1, 0.05)
    x, y = np.meshgrid(g, g)
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 2.0)])
    write_ply(sparse / 'points3D.ply', xyz, np.full((len(xyz), 3), 120, np.uint8))
    return root


def test_the_stereographic_rays_are_realityscans_division_lens():
    size = 301
    rays, theta = od.stereographic_rays(size)
    f = od.stereographic_focal(size)
    c = (np.arange(size) - (size - 1) / 2) / f
    u, v = (a.ravel() for a in np.meshgrid(c, c))
    k = 1 + pe.STEREOGRAPHIC_K1 * (u * u + v * v)
    front = k > 1e-3                                  # undistorted = distorted / (1 + k1 r^2)
    np.testing.assert_allclose(rays[front, 0] / rays[front, 2], u[front] / k[front], atol=1e-6, rtol=1e-6)
    assert math.isclose(theta[size // 2, -1], od.MAX_THETA, abs_tol=0.02)


def test_equidistant_rays_project_back_on_their_pixel():
    size = 201
    rays, theta = od.equidistant_rays(size)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-9)
    xy = od.equidistant_project(rays, size)
    xy = xy[0] if isinstance(xy, tuple) else xy
    gx, gy = np.meshgrid(np.arange(size), np.arange(size))
    inside = theta.ravel() < od.MAX_THETA
    np.testing.assert_allclose(xy[inside], np.column_stack([gx.ravel(), gy.ravel()])[inside], atol=1e-5)


def test_metashape_export_keeps_whole_fisheyes_masks_and_poses(colmap):
    import json
    out = pe.export_metashape(colmap, size=256, log=lambda *_: None)
    data = json.loads((out / 'cameras.json').read_text())
    assert [c['image'] for c in data['cameras']] == [f'cam0_frame_{i:06d}.jpg' for i in range(3)]
    np.testing.assert_allclose([c['camera_to_world'][0][3] for c in data['cameras']], [0, 0.5, 1.0])
    assert math.isclose(data['focal'], 256 / od.FISHEYE_FOV)
    mask = cv2.imread(str(out / 'masks' / 'cam0_frame_000000.png'), 0)
    assert mask[128, 128] == 255 and mask[128, 20] == 0 and mask[0, 0] == 0
    compile((out / 'import_metashape.py').read_text(), 'import_metashape.py', 'exec')


def test_realityscan_export_locks_pose_and_lens(colmap):
    out = pe.export_realityscan(colmap, size=256, log=lambda *_: None)
    xmp = (out / 'images' / 'cam0_frame_000002.xmp').read_text()
    assert 'DistortionModel="division"' in xmp and '>-0.25 0 0' in xmp and 'PosePrior="locked"' in xmp
    assert '<xcr:Position>1 ' in xmp
    assert (out / 'images' / 'cam0_frame_000002.jpg.mask.png').is_file()
    assert (out / 'lidar.las').is_file()


def test_exporting_again_keeps_the_users_project(colmap):
    out = pe.export_metashape(colmap, size=128, log=lambda *_: None)
    (out / 'mine.psx').write_text('x')
    pe.export_metashape(colmap, size=128, log=lambda *_: None)
    assert (out / 'mine.psx').is_file()


def test_realityscan_is_opened_with_the_division_lens(tmp_path):
    command, = pe.open_commands('realityscan', Path('RealityScan.exe'), tmp_path)
    assert 'sfmDistortionModel=Division' in command
    assert len(pe.open_commands('metashape', Path('metashape.exe'), tmp_path)) == 2


def test_stock_openmvs_is_refused(tmp_path):
    stock = tmp_path / 'DensifyPointCloud.exe'
    stock.write_bytes(b'not the fisheye build')
    with pytest.raises(Exception):
        od.check_binary(stock)


def test_mvs_scene_header(tmp_path, colmap):
    cameras, images = od.read_colmap_text(colmap / 'sparse' / '0')
    xyz = np.array([[0.0, 0, 2], [0.2, 0, 2]])
    path = tmp_path / 'scene.mvs'
    n = od.write_mvs(path, 256, images, xyz, np.zeros((2, 3), np.uint8), np.array([[0, 1], [1, 2]]))
    head = path.read_bytes()[:12]
    assert n == 2 and head[:4] == b'MVSI' and struct.unpack('<I', head[4:8])[0] == 8


def test_failed_dense_run_changes_nothing(colmap, tmp_path):
    before = sorted(p.relative_to(colmap) for p in colmap.rglob('*'))
    with pytest.raises(Exception):
        od.densify_openmvs(colmap, tmp_path / 'missing.exe', log=lambda *_: None)
    assert sorted(p.relative_to(colmap) for p in colmap.rglob('*')) == before
