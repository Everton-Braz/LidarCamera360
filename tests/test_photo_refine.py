import json
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app import openmvs_dense as od
from raven_app import photo_refine as pr

SIZE, READOUT = 800, 0.03                     # px; seconds from the last row to the first
LENS = np.array([260.0, 260.0, 400.0, 400.0, 0.02, -0.005, 0, 0, 0, 0, 0, 0])
FACING = [np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float), np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)]
LEVER = [np.array([0, 0.05, 0.1]), np.array([0, -0.05, 0.1])]


def rig(t):
    """Rig -> world rotation and rig position at time t."""
    t = np.atleast_1d(t)
    z = np.zeros_like(t)
    return (Rot.from_rotvec(np.stack([0.1 * np.sin(1.3 * t), z, 1.8 * np.sin(0.9 * t)], 1)),
            np.stack([0.3 * t - 4, 0.3 * np.sin(t), 1.5 + 0.05 * np.sin(3 * t)], 1))


def see(points, when, lens):
    """Pixels of world points in a photo read row by row around `when`, and which ones it sees."""
    tau = np.zeros(len(points))
    for _ in range(6):
        turn, pos = rig(when + tau)
        Xc = turn.inv().apply(points - pos - turn.apply(np.tile(LEVER[lens], (len(points), 1)))) @ FACING[lens].T
        px = pr.project_thin_prism(Xc, np.tile(LENS, (len(points), 1)))
        tau = np.clip(-READOUT * (px[:, 1] - SIZE / 2) / SIZE, -READOUT, READOUT)
    angle = np.arctan2(np.hypot(Xc[:, 0], Xc[:, 1]), Xc[:, 2])
    return px, (angle < 1.4) & (px > 0).all(axis=1) & (px < SIZE).all(axis=1)


def room(rng):
    """Points on two walls and the floor."""
    points = rng.uniform([-9, -5, 0], [9, 5, 4], (900, 3))
    wall = rng.integers(0, 3, len(points))
    points[wall == 0, 1], points[wall == 1, 1], points[wall == 2, 2] = 5, -5, 0
    return points


def make_run(root, upside_down=True):
    """Tracks, photo times and trajectory of a run with a rolling shutter. Returns the photos (name, camera, R, C)."""
    rng = np.random.default_rng(1)
    points = room(rng)
    images, blob, times = [], b'', {}
    for k, when in enumerate(np.arange(1.0, 25.0, 1.0)):
        turn, pos = rig(when)
        for lens in range(2):
            name = f'cam{lens}/frame_{k:06d}.jpg'
            px, ok = see(points, when, lens)
            px = px[ok] + rng.normal(0, 0.1, (ok.sum(), 2))
            if lens == 1 and upside_down:     # this lens is mounted upside down; the tracks are not turned
                px = SIZE - px
            rows = np.zeros(ok.sum(), np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')]))
            rows['x'], rows['y'], rows['id'] = px[:, 0], px[:, 1], np.flatnonzero(ok)
            blob += struct.pack('<I7dI', len(images) + 1, 1, 0, 0, 0, 0, 0, 0, lens + 1) + name.encode() + b'\0'
            blob += struct.pack('<Q', len(rows)) + rows.tobytes()
            R = FACING[lens] @ turn.as_matrix()[0].T
            images.append((name, lens + 1, R, pos[0] + turn.apply(LEVER[lens])[0]))
            times[name] = when - 0.7
    (root / 'sparse' / '0').mkdir(parents=True)
    (root / 'sparse' / '0' / 'images.bin').write_bytes(struct.pack('<Q', len(images)) + blob)
    (root / 'images').mkdir()
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': times}))
    t = np.arange(0, 26, 0.1)
    turn, pos = rig(t)
    (root / 'slam_raw_1' / 'result').mkdir(parents=True)
    np.savetxt(root / 'slam_raw_1' / 'result' / 'scan.txt', np.column_stack([t + 1.79e9, pos, turn.as_quat()]), fmt='%.9f')
    return images


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp('run')
    images = make_run(root)
    colmap = root / 'colmap_3dgs'
    colmap.mkdir()
    (colmap / 'colmap_export_report.json').write_text(json.dumps({'dt_sync_seconds': -0.7}))
    (colmap / 'upright_rotation.json').write_text(json.dumps({'camera_ids': [2]}))
    cameras = {c: ('THIN_PRISM_FISHEYE', SIZE, SIZE, list(LENS)) for c in (1, 2)}
    return colmap, cameras, images


def quiet(*_):
    pass


def test_the_projection_and_its_derivatives():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3)) * 0.5 + [0, 0, 1.5]
    K = np.tile(LENS, (40, 1))
    K[:, 4:] += rng.normal(0, 0.01, (40, 8))
    px, Jx, Jk = pr.project_thin_prism(X, K, jacobian=True)
    other, _ = od.project_rays('THIN_PRISM_FISHEYE', K[3], X[3:4] / np.linalg.norm(X[3]))
    np.testing.assert_allclose(px[3:4], other, atol=1e-8)
    for J, value, at in ((Jx, X, lambda d: pr.project_thin_prism(X + d, K)),
                         (Jk, K, lambda d: pr.project_thin_prism(X, K + d))):
        for j in range(value.shape[1]):
            d = np.zeros(value.shape[1])
            d[j] = 1e-6
            np.testing.assert_allclose((at(d) - at(-d)) / 2e-6, J[:, :, j], rtol=1e-4, atol=1e-2)


def test_the_rolling_shutter_is_found_and_fits_tracks_it_never_saw(run):
    colmap, cameras, images = run
    _, new_images, shutter, report = pr.refine_photos(colmap, cameras, images, log=quiet)
    assert report['status'] == 'accepted' and report['rolling_shutter']
    assert report['held_out_px'] < 0.2 < 0.4 < report['held_out_px_before']
    for lens in report['lenses']:
        assert abs(lens['readout_ms'] + READOUT * 1000) < 2 and abs(lens['readout_across_ms']) < 2
    assert all(m is not None for m in shutter)
    assert max(np.linalg.norm(a[3] - b[3]) for a, b in zip(images, new_images)) < 0.05
    assert [i[:2] for i in images] == [i[:2] for i in new_images]
    assert cameras[1][3] == list(LENS)        # the export's own values stay


def test_the_photo_maps_undo_the_rolling_shutter(run):
    colmap, cameras, images = run
    size, n = 256, 12                         # a photo taken while the rig turns
    name, camera, R, C = images[n]
    when = json.loads((colmap.parent / 'images' / 'frames.json').read_text())['timestamps'][name] + 0.7
    w, v, known = pr.photo_motion(colmap.parent, colmap, [i[0] for i in images], np.array([i[3] for i in images]))
    assert known.all()
    motion = (np.array([0.0, -READOUT]), R @ w[n], R @ v[n])
    rays, theta = od.equidistant_rays(size)
    base = od.build_remap(*cameras[camera], size)
    mx, my, inside = pr.shutter_maps(cameras[camera], size, od.equidistant_rays, base, motion,
                                     np.full((64, 64), 1 / 3.0, np.float32))
    px, seen = see(C + 3.0 * rays @ R, when, camera - 1)
    use = inside.ravel() & (theta.ravel() < 1.3) & seen
    assert use.sum() > 10_000
    found = np.column_stack([mx.ravel(), my.ravel()])[use] + 0.5
    still = np.column_stack([base[0].ravel(), base[1].ravel()])[use] + 0.5
    error = np.linalg.norm(found - px[use], axis=1)
    assert np.percentile(error, 99) < 0.15 and np.linalg.norm(still - px[use], axis=1).max() > 1.0


def test_a_photo_that_stands_still_keeps_its_map(run):
    _, cameras, _ = run
    base = od.build_remap(*cameras[1], 128)
    still = (np.array([0.0, -READOUT]), np.zeros(3), np.zeros(3))
    mx, my, inside = pr.shutter_maps(cameras[1], 128, od.equidistant_rays, base, still, np.full((64, 64), 0.5, np.float32))
    np.testing.assert_allclose(mx, base[0], atol=1e-4)
    np.testing.assert_allclose(my, base[1], atol=1e-4)
    assert (inside <= base[2]).all()


def test_without_tracks_the_export_is_used(run, tmp_path):
    colmap, cameras, images = run
    result = pr.refine_photos(tmp_path / 'colmap_3dgs', cameras, images, log=quiet)
    assert result[0] is cameras and result[1] is images and result[2] is None
    assert result[3]['status'] == 'skipped'


def test_without_photo_times_no_shutter_is_claimed(run, tmp_path):
    colmap, cameras, images = run
    (tmp_path / 'sparse' / '0').mkdir(parents=True)
    (tmp_path / 'sparse' / '0' / 'images.bin').write_bytes((colmap.parent / 'sparse' / '0' / 'images.bin').read_bytes())
    (tmp_path / 'colmap_3dgs').mkdir()
    (tmp_path / 'colmap_3dgs' / 'upright_rotation.json').write_text(json.dumps({'camera_ids': [2]}))
    _, _, shutter, report = pr.refine_photos(tmp_path / 'colmap_3dgs', cameras, images, log=quiet)
    assert shutter is None and report['rolling_shutter'] is False


def test_an_adjustment_that_does_not_help_is_not_used(run, tmp_path, monkeypatch):
    colmap, cameras, images = run
    monkeypatch.setattr(pr._Adjustment, 'step', lambda self, X, a, planes: X)
    new_cameras, new_images, shutter, report = pr.refine_photos(colmap, cameras, images, log=quiet)
    assert report['status'] == 'rejected'
    assert new_cameras is cameras and new_images is images and shutter is None


def test_a_failure_leaves_the_export_as_it_is(run, monkeypatch):
    colmap, cameras, images = run

    def broken(*_):
        raise RuntimeError('no')
    monkeypatch.setattr(pr, '_tracks', broken)
    new_cameras, new_images, shutter, report = pr.refine_photos(colmap, cameras, images, log=quiet)
    assert report['status'] == 'failed' and new_cameras is cameras and new_images is images and shutter is None


def test_the_surface_distance_fills_the_gaps_between_points():
    g = np.arange(-4, 4, 0.2)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    inverse = pr.inverse_range(np.column_stack([x, y, np.full(x.size, 2.0)]), np.eye(3), np.zeros(3), 128)
    assert (inverse > 0).all() and abs(1 / inverse[64, 64] - 2.0) < 0.1
