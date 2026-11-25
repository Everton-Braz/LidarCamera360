import json
import os

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app import photo_alignment as pa
from raven_app import photo_refine as pr
from raven_app import workflow
from test_photo_refine import FACING, LENS, LEVER, READOUT, SIZE, make_run, quiet, rig, room, see
from test_time_gating import _dynamic_dataset, _read_ply_rgb, _write_pcd

from scripts import pipeline_auto_calibrator_and_colorizer as pipeline

KEYS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    """A run with a rolling shutter: photos, person masks, scan, tracks, trajectory and calibration."""
    root = tmp_path_factory.mktemp('dataset')
    images = make_run(root, upside_down=False)
    (root / 'slam_raw_1').rename(root / 'slam_out')
    intrinsics = {'camera_model': 'THIN_PRISM_FISHEYE', 'width': SIZE, 'height': SIZE, **dict(zip(KEYS, LENS))}
    calib = {'cam0_front_intrinsics': intrinsics, 'cam1_rear_intrinsics': intrinsics, 'dt_sync_seconds': -0.7}
    for lens in range(2):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = FACING[lens].T, LEVER[lens]
        calib[f'T_lidar_to_cam{lens}_rigid_4x4'] = T.tolist()
        (root / 'images' / f'cam{lens}').mkdir(parents=True)
        (root / 'masks' / f'cam{lens}').mkdir(parents=True)
    (root / 'calib.json').write_text(json.dumps(calib))
    photo = cv2.GaussianBlur(np.random.default_rng(0).integers(0, 256, (SIZE, SIZE, 3), np.uint8), (0, 0), 9)
    mask = np.full((SIZE, SIZE), 255, np.uint8)
    mask[300:500, 100:200] = 0
    for name, *_ in images:
        cv2.imwrite(str(root / 'images' / name), photo)
        cv2.imwrite(str(root / 'masks' / f'{name}.png'), mask)
    _write_pcd(pipeline.active_slam_dir(root) / 'pcd' / 'all_raw_points.pcd', room(np.random.default_rng(1)))
    return root, images


@pytest.fixture(scope='module')
def aligned(dataset):
    root, _ = dataset
    found = pa.align_photos(root, root / 'calib.json', 1.0, None, root / 'masks', log=quiet)
    assert found is not None
    return found


def test_the_photos_get_one_pose_each_and_fit_better(dataset, aligned):
    root, images = dataset
    report = aligned.report
    assert report['held_out_px'] < 0.2 < 0.4 < report['held_out_px_before']
    assert set(aligned.poses) == {n for n, *_ in images}
    assert report['photos'] == report['rolling_shutter_removed'] == len(images)
    for name, camera_id, R, C in images:
        found_R, found_C = aligned.poses[name]
        assert np.linalg.norm(found_C - C) < 0.05
        assert np.degrees(Rot.from_matrix(found_R @ R.T).magnitude()) < 0.3
    assert aligned.camera_of('cam1/frame_000000.jpg')[:3] == ('THIN_PRISM_FISHEYE', SIZE, SIZE)


def test_photos_and_masks_are_moved_and_the_runs_own_stay(dataset, aligned):
    root, images = dataset
    name = images[12][0]
    for tree, file in (('images', name), ('masks', f'{name}.png')):
        own, moved = root / tree / file, root / pa.FOLDER / tree / file
        assert not os.path.samefile(own, moved)
        assert (cv2.imread(str(own), cv2.IMREAD_UNCHANGED) != cv2.imread(str(moved), cv2.IMREAD_UNCHANGED)).any()
    assert set(np.unique(cv2.imread(str(aligned.masks_dir / f'{name}.png'), cv2.IMREAD_UNCHANGED))) == {0, 255}
    assert os.path.samefile(root / 'images/frames.json', aligned.images_dir / 'frames.json')


def test_the_alignment_is_reused_until_a_photo_or_mask_changes(dataset, aligned, monkeypatch):
    root, images = dataset
    monkeypatch.setattr(pr, 'refine_photos', lambda *a, **k: pytest.fail('not reused'))
    again = pa.align_photos(root, root / 'calib.json', 1.0, None, root / 'masks', log=quiet)
    assert again.poses.keys() == aligned.poses.keys()
    log = []
    mask = root / 'masks' / f'{images[0][0]}.png'
    before = mask.stat()
    os.utime(mask, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
    monkeypatch.setattr(pr, 'refine_photos', lambda *a, **k: (_ for _ in ()).throw(ValueError('asked again')))
    assert pa.align_photos(root, root / 'calib.json', 1.0, None, root / 'masks', log=log.append) is None
    assert 'asked again' in log[-1]
    os.utime(mask, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_the_lens_maps_undo_the_rolling_shutter(dataset):
    _, images = dataset
    name, camera_id, R, C = images[24]
    when = 13.0
    camera = ('THIN_PRISM_FISHEYE', SIZE, SIZE, list(LENS))
    step = 1e-3
    turn, pos = rig(when + np.array([-step, step]))
    omega = (turn[1] * turn[0].inv()).as_rotvec() / (2 * step)
    lever = turn.apply(np.tile(LEVER[camera_id - 1], (2, 1)))
    speed = ((pos + lever)[1] - (pos + lever)[0]) / (2 * step)
    mx, my = pa.lens_shutter_maps(camera, (np.array([0.0, -READOUT]), R @ omega, R @ speed),
                                  np.full((64, 64), 1 / 3.0, np.float32))
    rng = np.random.default_rng(3)
    angle, around = rng.uniform(0, 1.2, 4000), rng.uniform(0, 2 * np.pi, 4000)
    rays = np.column_stack([np.sin(angle) * np.cos(around), np.sin(angle) * np.sin(around), np.cos(angle)])
    still = pr.project_thin_prism(rays, np.tile(LENS, (len(rays), 1)))
    read, seen = see(C + 3.0 * rays @ R, when, camera_id - 1)
    at = np.floor(still).astype(int)
    use = seen & (at > 1).all(axis=1) & (at < SIZE - 2).all(axis=1)
    assert use.sum() > 3000
    still, read, at = still[use], read[use], at[use]
    found = np.column_stack([mx[at[:, 1], at[:, 0]], my[at[:, 1], at[:, 0]]]) + 0.5 + (still - at - 0.5)
    assert np.percentile(np.linalg.norm(found - read, axis=1), 99) < 0.3
    assert np.linalg.norm(still - read, axis=1).max() > 2.0


def test_no_alignment_without_a_time_offset_tracks_or_fisheyes(dataset, tmp_path):
    root, _ = dataset
    calib = json.loads((root / 'calib.json').read_text())
    for change, reason in (({'dt_sync_seconds': None}, 'no time offset'),
                           ({'lens_model': 'EQUIRECTANGULAR', 'cam0_front_intrinsics': None, 'cam1_rear_intrinsics': None},
                            'needs fisheye photos')):
        path = tmp_path / 'calib.json'
        path.write_text(json.dumps({**calib, **change}))
        log = []
        assert pa.align_photos(root, path, 1.0, None, root / 'masks', log=log.append) is None
        assert reason in log[-1]
    bare = tmp_path / 'bare'
    (bare / 'images' / 'cam0').mkdir(parents=True)
    (bare / 'images' / 'cam0' / 'frame_000000.jpg').write_bytes((root / 'images/cam0/frame_000000.jpg').read_bytes())
    (bare / 'images' / 'frames.json').write_text((root / 'images/frames.json').read_text())
    (bare / 'slam_out' / 'result').mkdir(parents=True)
    (bare / 'slam_out/result/scan.txt').write_text((root / 'slam_out/result/scan.txt').read_text())
    log = []
    assert pa.align_photos(bare, root / 'calib.json', 1.0, None, log=log.append) is None
    assert 'no feature tracks' in log[-1] and not (bare / pa.FOLDER).exists()


def test_the_export_uses_the_aligned_photos_poses_and_lenses(dataset, aligned, tmp_path):
    root, images = dataset
    out = workflow.export_colmap_3dgs(root, root / 'calib.json', dt_sync=-0.7, masks_dir=root / 'masks',
                                      pose_source='ba', export_path=tmp_path / 'export', aligned=aligned)
    report = workflow.read_export_report(out)
    assert report['pose_source'] == 'aligned' and report['photo_alignment']['held_out_px'] == aligned.report['held_out_px']
    poses = {}
    for line in (out / 'sparse/0/images.txt').read_text().splitlines():
        cells = line.split()
        if len(cells) == 10 and cells[0] != '#':
            R = Rot.from_quat([*map(float, cells[2:5]), float(cells[1])]).as_matrix()
            poses[cells[9]] = (R, -R.T @ np.array(cells[5:8], float), cells[8])
    assert set(poses) == set(aligned.poses)
    turned = json.loads((out / 'upright_rotation.json').read_text())['camera_ids'] \
        if (out / 'upright_rotation.json').is_file() else []
    lenses = {line.split()[0]: np.array(line.split()[4:], float)
              for line in (out / 'sparse/0/cameras.txt').read_text().splitlines() if line and line[0] != '#'}
    for name, (R, C, camera) in poses.items():
        own_R, own_C = aligned.poses[name]
        np.testing.assert_allclose(C, own_C, atol=1e-6)
        photo = cv2.imread(str(aligned.images_dir / name))
        mask = cv2.imread(str(aligned.masks_dir / f'{name}.png'), cv2.IMREAD_UNCHANGED)
        if int(camera) in turned:                  # the export turns upside-down lenses upright, pose and photo alike
            own_R = np.diag([-1.0, -1.0, 1.0]) @ own_R
            mask = mask[::-1, ::-1]
        else:
            assert (cv2.imread(str(out / 'images' / name)) == photo).all()
            np.testing.assert_allclose(lenses[camera], aligned.camera_of(name)[3], rtol=1e-6)
        np.testing.assert_allclose(R, own_R, atol=1e-6)
        assert (cv2.imread(str(out / 'masks' / f'{name}.png'), cv2.IMREAD_UNCHANGED) == mask).all()
    skipped = workflow.refine_colmap_3dgs_poses(out, root / 'sparse/0')
    assert skipped['status'] == 'skipped'


def test_an_export_the_alignment_does_not_cover_uses_the_runs_own(dataset, aligned, tmp_path):
    root, images = dataset
    poses = dict(aligned.poses)
    poses.pop(images[0][0])
    partial = pa.AlignedPhotos(aligned.images_dir, aligned.masks_dir, aligned.cameras, poses, aligned.report)
    for photos, source in ((partial, 'ba'), (aligned, 'direct')):
        out = workflow.export_colmap_3dgs(root, root / 'calib.json', dt_sync=-0.7, pose_source=source,
                                          export_path=tmp_path / source, aligned=photos)
        report = workflow.read_export_report(out)
        assert report['pose_source'] == source and 'photo_alignment' not in report


def test_the_colours_come_from_the_aligned_photos_at_their_poses(tmp_path):
    root, calib, n = _dynamic_dataset(tmp_path)
    tree = tmp_path / 'aligned'
    for cam in ('cam0', 'cam1'):
        (tree / cam).mkdir(parents=True)
        for frame in ('frame_000001.jpg', 'frame_000002.jpg'):
            # green photos; of the second pose they must not be used: it looks the other way
            cv2.imwrite(str(tree / cam / frame), np.full((64, 64, 3), (0, 255, 0) if cam == 'cam0' else 0, np.uint8))
    away = np.diag([-1.0, 1.0, -1.0])
    lens = ('THIN_PRISM_FISHEYE', 64, 64, [32.0, 32.0, 32.0, 32.0, 0, 0, 0, 0, 0, 0, 0, -999.0])
    photos = pa.AlignedPhotos(tree, None, {'cam0': lens, 'cam1': lens}, {
        'cam0/frame_000001.jpg': (np.eye(3), np.zeros(3)), 'cam0/frame_000002.jpg': (away, np.zeros(3)),
        'cam1/frame_000001.jpg': (away, np.zeros(3)), 'cam1/frame_000002.jpg': (away, np.zeros(3))})
    ply, _ = pipeline.colorize_via_direct_rigid(root, calib, fps=1.0, dt_override=0.0, use_vulkan=False,
                                                time_window=6.0, aligned=photos)
    rgb = _read_ply_rgb(ply).astype(int)
    assert (rgb[:n, 1] > 200).all() and (rgb[:n, [0, 2]] < 50).all()
    assert not (rgb[n:, 1] > 200).any()


def test_an_aligned_export_is_known_by_its_report(tmp_path):
    (tmp_path / 'colmap_export_report.json').write_text(json.dumps({'pose_source': 'aligned'}))
    assert workflow.read_export_report(tmp_path)['pose_source'] == 'aligned'
    assert workflow.read_export_report(tmp_path / 'missing') == {}
    assert workflow.refine_colmap_3dgs_poses(tmp_path, tmp_path)['status'] == 'skipped'


def test_the_command_line_switches():
    from raven_app.cli import parse
    args = ['workflow', '--bag', 'scan.bag', '--insv', 'cam.insv', '--output', 'out']
    assert parse(args).photo_alignment and not parse(args + ['--no-photo-alignment']).photo_alignment
