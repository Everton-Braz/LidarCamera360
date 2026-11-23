import json

import cv2
import numpy as np
import pytest

from raven_app.dynamic_filter import (CarveParams, _wide_groups, removal_fringe_keep, carve_dynamic, load_static_keep,
                                      person_vote_keep, run_dynamic_filter)
from raven_app.mcp_server import _cli_arguments


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


def _wall(x, y_range, z_range, step):
    ys = np.arange(*y_range, step)
    zs = np.arange(*z_range, step)
    yy, zz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def _scene(seed=0, person_scans=3):
    """A wall seen by 30 scans (1 s apart) and a person in front of it during the first scans."""
    rng = np.random.default_rng(seed)
    wall = _wall(8.0, (-1.5, 1.5), (0.0, 2.0), 0.04)
    person = np.column_stack([rng.uniform(3.9, 4.1, 400), rng.uniform(-0.2, 0.2, 400),
                              rng.uniform(0.5, 1.5, 400)])
    points, scans, is_person = [], [], []
    for scan in range(30):
        block = [wall + rng.normal(0, 0.003, wall.shape)]
        flags = [np.zeros(len(wall), bool)]
        if scan < person_scans:
            block.append(person + rng.normal(0, 0.003, person.shape))
            flags.append(np.ones(len(person), bool))
        block = np.concatenate(block)
        points.append(block)
        scans.append(np.full(len(block), scan))
        is_person.append(np.concatenate(flags))
    return (np.concatenate(points), np.concatenate(scans), np.arange(30, dtype=float),
            np.tile([0.0, 0.0, 1.0], (30, 1)), np.concatenate(is_person))


def test_carving_removes_a_passer_by_and_keeps_the_wall():
    points, scans, times, origins, is_person = _scene()

    dynamic, stats = carve_dynamic(points, scans, times, origins, log=lambda *_: None)

    # The far side of the body is never seen through within the ray margin.
    assert dynamic[is_person].mean() > 0.8
    assert not dynamic[~is_person].any()
    assert stats['dynamic_points'] == int(dynamic.sum())


def test_carving_keeps_objects_present_in_every_scan():
    # Foliage or a fence is hit and seen through by the same scans: never carved.
    points, scans, times, origins, is_object = _scene(person_scans=30)

    dynamic, _ = carve_dynamic(points, scans, times, origins, log=lambda *_: None)

    # Only voxels on cell borders, hit intermittently because of range noise, may flip.
    assert dynamic[is_object].mean() < 0.03
    assert not dynamic[~is_object].any()


def test_run_dynamic_filter_caches_a_raw_point_mask(tmp_path):
    points, scans, times, origins, is_person = _scene()
    raw = points.copy()
    raw[5] = np.nan  # load_pcd drops non-finite rows; the cached mask is per raw row
    slam = tmp_path / 'slam_out'
    _write_pcd(slam / 'pcd' / 'all_raw_points.pcd', raw)
    counts = np.bincount(scans)
    starts = np.cumsum(counts) - counts
    _write_scan_ranges(slam, [(100.0 + t, s, c) for t, s, c in zip(times, starts, counts)])
    (slam / 'result').mkdir()
    np.savetxt(slam / 'result' / 'trajectory.txt',
               np.column_stack([100.0 + times, origins, np.tile([0, 0, 0, 1.0], (30, 1))]))

    run_dynamic_filter(tmp_path, log=lambda *_: None)
    report = json.loads((tmp_path / 'dynamic_filter' / 'report.json').read_text(encoding='utf-8'))
    finite = np.isfinite(raw).all(1)
    keep = load_static_keep(tmp_path, finite, log=lambda *_: None)

    assert len(keep) == finite.sum()
    assert (~keep).sum() == report['dynamic_points'] > 0.8 * is_person.sum()
    assert keep[~is_person[finite]].all()
    assert load_static_keep(tmp_path, np.ones(len(raw) + 1, bool), log=lambda *_: None) is None
    messages = []
    run_dynamic_filter(tmp_path, log=messages.append)
    assert messages and messages[0].startswith('[*] Reusing')


def _pinhole(cam, params):
    fx, fy, cx, cy = params[:4]
    return fx * cam[:, 0] / cam[:, 2] + cx, fy * cam[:, 1] / cam[:, 2] + cy, None


def test_person_votes_remove_a_standing_person_but_not_the_wall_behind(tmp_path):
    rng = np.random.default_rng(1)
    wall = _wall(10.0, (-3.0, 3.0), (0.0, 2.5), 0.05)
    # Torso-sized cylinder, sampled as densely as an accumulated LiDAR cloud.
    angle, height = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(0.3, 1.7, 3000)
    person = np.column_stack([3.0 + 0.15 * np.cos(angle), 0.15 * np.sin(angle), height])
    # Seen in every frame through the operator's mask (bottom rows), never a person.
    direction = rng.normal(size=(800, 3))
    behind_operator = np.array([5.0, 0.0, -1.5]) + 0.2 * direction / np.linalg.norm(direction, axis=1, keepdims=True)
    points = np.concatenate([wall, person, behind_operator])
    is_person = np.r_[np.zeros(len(wall), bool), np.ones(len(person), bool), np.zeros(len(behind_operator), bool)]
    is_operator_background = np.r_[np.zeros(len(wall) + len(person), bool), np.ones(len(behind_operator), bool)]
    params = (400.0, 400.0, 320.0, 240.0, 0, 0, 0, 0, 0, 0, 0.0, -999.0)
    images, masks = tmp_path / 'images', tmp_path / 'masks'
    (masks / 'cam0').mkdir(parents=True)
    views = []
    looking_x = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    looking_back = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], float)
    poses = [(looking_x, np.array([0.0, y, 1.0])) for y in np.arange(-2.5, 2.6, 0.5)]
    poses += [(looking_back, np.array([6.0, y, 1.0])) for y in (-0.5, 0.5)]
    for i, (rotation, center) in enumerate(poses):
        image = images / 'cam0' / f'f{i}.jpg'
        cam = (rotation @ (person - center).T).T
        u, v, _ = _pinhole(cam, params)
        mask = np.full((480, 640), 255, np.uint8)
        for x, y in zip(u, v):
            cv2.circle(mask, (int(x), int(y)), 6, 0, -1)
        mask[380:] = 0  # the operator, masked in every frame
        cv2.imwrite(str(masks / 'cam0' / 'f{}.jpg.png'.format(i)), mask)
        views.append((image, rotation, center, params))

    keep = person_vote_keep(points, views, images, masks, _pinhole, log=lambda *_: None)

    assert (~keep[is_person]).mean() > 0.8
    assert keep[~is_person].all()
    assert keep[is_operator_background].all()
    # Views vote on worker threads; the outcome must not depend on how many.
    serial = person_vote_keep(points, views, images, masks, _pinhole, workers=1, log=lambda *_: None)
    assert (serial == keep).all()


def test_wide_groups_separate_a_facade_from_a_person():
    rng = np.random.default_rng(2)
    # A noisy facade strip (4 m wide, 0.3 m thick) and a person-sized blob 3 m away.
    facade = np.column_stack([rng.uniform(-2, 2, 4000), rng.uniform(5.0, 5.3, 4000), rng.uniform(0, 2, 4000)])
    person = np.column_stack([rng.normal(0, 0.12, 1500), rng.normal(2.0, 0.12, 1500), rng.uniform(0, 1.7, 1500)])

    wide = _wide_groups(np.concatenate([facade, person]), link=0.15, max_extent=2.0)

    assert wide[:len(facade)].all()
    assert not wide[len(facade):].any()


def test_remove_dynamic_is_exposed_to_mcp():
    assert '--remove-dynamic' in _cli_arguments('workflow', {'bag': 'b', 'insv': 'i', 'output': 'o',
                                                            'remove_dynamic': True})
    assert '--remove-dynamic' in _cli_arguments('colorize', {'dataset': 'd', 'remove_dynamic': True})


def test_carve_params_are_validated():
    with pytest.raises(ValueError):
        CarveParams(voxel=0).validate()


def test_removal_fringe_drops_stranded_points_only():
    rng = np.random.default_rng(1)
    wall = np.column_stack([rng.uniform(0, 4, 4000), np.zeros(4000), rng.uniform(0, 3, 4000)])
    person = rng.normal([2.0, 1.5, 1.0], [0.15, 0.15, 0.4], (800, 3))
    stray = np.array([[2.05, 1.5, 1.1]])
    # The wall next to the removed person keeps its points; the leftover inside it goes.
    near_wall = rng.normal([2.0, 0.0, 1.0], 0.05, (5, 3)) * [1, 0, 1]
    kept = np.concatenate([wall, near_wall, stray])
    keep = removal_fringe_keep(kept, person, log=lambda *_: None)
    assert not keep[-1]
    assert keep[:-1].all()
