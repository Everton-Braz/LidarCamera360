import struct

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from raven_app.pose_ba import read_images_txt, refine_poses, unproject_thin_prism
from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism

PARAMS = np.array([1081.5, 1082.2, 1920.0, 1920.0, 0.082, -0.029, -0.0005, 0.0012,
                   0.0109, -0.0028, -0.0022, 0.0007])


def test_unprojection_inverts_the_thin_prism_projection():
    rng = np.random.default_rng(0)
    rays = rng.normal(size=(5000, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays[rays[:, 2] > 0.0]
    u, v, _ = project_thin_prism(rays, PARAMS)
    bearings = unproject_thin_prism(np.stack([u, v], 1), PARAMS)
    assert np.degrees(np.arccos(np.clip((bearings * rays).sum(1), -1, 1))).max() < 1e-4


def _write_tracks(path, observations):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(observations)))
        for image_id, (name, xy, ids) in enumerate(observations, start=1):
            f.write(struct.pack('<I', image_id) + struct.pack('<4d', 1, 0, 0, 0)
                    + struct.pack('<3d', 0, 0, 0) + struct.pack('<I', 1))
            f.write(name.encode() + b'\x00' + struct.pack('<Q', len(ids)))
            for (x, y), pid in zip(xy, ids):
                f.write(struct.pack('<ddq', x, y, pid))


def _scene(tmp_path, wild=None, noise_deg=0.35, wild_deg=0.5):
    """40 rig frames with ``noise_deg`` attitude noise; frame ``wild`` gets ``wild_deg`` instead."""
    rng = np.random.default_rng(1)
    points = np.column_stack([rng.uniform(-15, 15, 4000), rng.uniform(-15, 15, 4000), rng.uniform(-1, 6, 4000)])
    rig = {'cam0': Rot.from_euler('xyz', [-90, -30, -90], degrees=True).as_matrix(),
           'cam1': Rot.from_euler('xyz', [-90, 30, 90], degrees=True).as_matrix()}
    lever = {'cam0': np.array([0.02, 0.12, 0.12]), 'cam1': np.array([-0.01, 0.12, 0.11])}
    sparse = tmp_path / 'colmap' / 'sparse' / '0'
    sparse.mkdir(parents=True)
    (sparse / 'cameras.txt').write_text(''.join(
        f"{i} THIN_PRISM_FISHEYE 3840 3840 {' '.join(map(str, PARAMS))}\n" for i in (1, 2)))
    lines, tracks, truth = [], [], {}
    image_id = 1
    for k in range(40):
        body_R = Rot.from_euler('z', 9 * k, degrees=True).as_matrix()
        body_t = np.array([0.4 * k - 8, 0.1 * k, 1.5])
        noise = Rot.from_rotvec(np.radians(noise_deg) * rng.normal(size=3) / np.sqrt(3)).as_matrix()
        if k == wild:
            noise = Rot.from_rotvec(np.radians(wild_deg) * np.array([0.6, 0.0, 0.8])).as_matrix()
        for camera_id, lens in ((1, 'cam0'), (2, 'cam1')):
            name = f'{lens}/frame_{k:06d}.jpg'
            R_wc = body_R @ rig[lens]
            C = body_t + body_R @ lever[lens]
            truth[name] = (R_wc.T, C)
            v = (points - C) @ R_wc
            front = v[:, 2] / np.linalg.norm(v, axis=1) > 0.05
            u, w, _ = project_thin_prism(v[front], PARAMS)
            xy = np.stack([u, w], 1) + rng.normal(scale=0.5, size=(front.sum(), 2))
            tracks.append((name, xy, np.flatnonzero(front)))
            R_cw_noisy = (noise @ R_wc).T
            q = Rot.from_matrix(R_cw_noisy).as_quat()
            t = -R_cw_noisy @ C
            lines.append(f'{image_id} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {camera_id} {name}\n')
            image_id += 1
    (sparse / 'images.txt').write_text('\n'.join(lines) + '\n')
    tracks_dir = tmp_path / 'sfm'
    tracks_dir.mkdir()
    _write_tracks(tracks_dir / 'images.bin', tracks)
    return sparse, tracks_dir, truth


def test_bundle_adjustment_removes_attitude_noise_and_keeps_the_lidar_frame(tmp_path):
    sparse, tracks_dir, truth = _scene(tmp_path)
    report = refine_poses(sparse, tracks_dir, backup_path=tmp_path / 'images_direct.txt', log=lambda *_: None)

    assert report['status'] == 'accepted'
    assert report['after']['median_px'] < 1.0 < report['direct_same_observations']['median_px']
    errors = []
    for _, R_cw, t_cw, _, name in read_images_txt(sparse / 'images.txt'):
        R_true, C_true = truth[name]
        errors.append((np.degrees(Rot.from_matrix(R_cw @ R_true.T).magnitude()),
                       np.linalg.norm(-R_cw.T @ t_cw - C_true)))
    errors = np.array(errors)
    assert np.median(errors[:, 0]) < 0.1   # from ~0.35 deg of attitude noise
    assert np.median(errors[:, 1]) < 0.03  # still in the LiDAR frame
    assert (tmp_path / 'images_direct.txt').is_file()


def test_one_wild_frame_follows_its_neighbours_and_the_rest_is_adjusted(tmp_path):
    # scaled down: a frame several degrees off loses its tracks to the pruning and is never moved
    sparse, tracks_dir, truth = _scene(tmp_path, wild=20, noise_deg=0.08, wild_deg=0.5)
    direct = {im[4]: im for im in read_images_txt(sparse / 'images.txt')}

    report = refine_poses(sparse, tracks_dir, max_rotation_deg=0.3, log=lambda *_: None)

    assert report['status'] == 'accepted', report.get('reason')
    assert report['neighbour_corrected_frames'] == ['frame_000020.jpg']
    for _, R_cw, t_cw, _, name in read_images_txt(sparse / 'images.txt'):
        R_true, _ = truth[name]
        error = np.degrees(Rot.from_matrix(R_cw @ R_true.T).magnitude())
        if 'frame_000020' in name:
            # its own 0.5 deg correction is dropped; it takes its neighbours' ~0.08 deg one instead
            assert np.degrees(Rot.from_matrix(R_cw @ direct[name][1].T).magnitude()) < 0.25
        else:
            assert error < 0.15
