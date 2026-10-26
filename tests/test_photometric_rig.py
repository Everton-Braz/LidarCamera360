import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app.photometric_rig import main_body_from_calibration, refine_rig

ROOM = np.array([[-5.0, 25.0], [-6.0, 6.0], [-1.5, 3.0]])   # x, y, z bounds (m)
R_CAM0 = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])   # looks along body +x
R_CAM1 = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])   # looks along body -x
T_CAM0, T_CAM1 = np.array([0.01, 0., 0.1]), np.array([-0.01, 0., 0.1])
F, SIZE = 120.0, 400


def _texture(points):
    # Metre-scale pattern: the solver works on 1/4-resolution images blurred by up to
    # 4 px, so with these 400 px frames finer texture aliases away.
    x, y, z = points.T
    return 0.5 + 0.5 * np.sin(2 * np.pi * x / 2.1) * np.sin(2 * np.pi * y / 2.7) * np.sin(2 * np.pi * z / 2.4)


def _body(t):
    """Walk along x at 1 m/s while yawing ±34 deg (up to ~54 deg/s), like a handheld rig."""
    yaw = 0.6 * np.sin(2 * np.pi * t / 4.0)
    return np.array([t, 0.0, 0.0]), Rot.from_euler('z', yaw).as_matrix()


def _render(t, R_cam, t_cam):
    position, R_body = _body(t)
    centre = position + R_body @ t_cam
    vv, uu = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    xd, yd = (uu - SIZE / 2) / F, (vv - SIZE / 2) / F
    theta = np.hypot(xd, yd)
    s = np.divide(np.sin(theta), theta, out=np.ones_like(theta), where=theta > 1e-9)
    rays = np.stack([xd * s, yd * s, np.cos(theta)], -1).reshape(-1, 3) @ (R_body @ R_cam).T
    best = np.full(len(rays), np.inf)
    for axis in range(3):
        for bound in ROOM[axis]:
            with np.errstate(divide='ignore', invalid='ignore'):
                hit = (bound - centre[axis]) / rays[:, axis]
            best = np.where((hit > 1e-6) & (hit < best), hit, best)
    image = _texture(centre + rays * best[:, None]) * 255
    return image.reshape(SIZE, SIZE).astype(np.uint8)


def _write_pcd(path, xyz):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype='<f4')
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n'
              f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n')
    with open(path, 'wb') as stream:
        stream.write(header.encode('ascii'))
        xyz.tofile(stream)


def _room_cloud(step=0.1):
    (x0, x1), (y0, y1), (z0, z1) = ROOM
    xs, ys, zs = np.arange(x0, x1, step), np.arange(y0, y1, step), np.arange(z0, z1, step)
    faces = []
    for y in (y0, y1):
        gx, gz = np.meshgrid(xs, zs)
        faces.append(np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()]))
    for z in (z0, z1):
        gx, gy = np.meshgrid(xs, ys)
        faces.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)]))
    return np.vstack(faces)


@pytest.fixture(scope='module')
def synthetic_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp('photometric_rig') / 'dataset'
    slam = root / 'slam_out'
    times = np.arange(0.0, 20.0, 0.05)
    poses = [_body(t) for t in times]
    quats = Rot.from_matrix(np.stack([R for _, R in poses])).as_quat()
    trajectory = np.column_stack([1000.0 + times, np.stack([p for p, _ in poses]), quats])
    (slam / 'result').mkdir(parents=True)
    np.savetxt(slam / 'result' / 'Raven_3DMakerPro_Scan.txt', trajectory, fmt='%.6f')
    _write_pcd(slam / 'pcd' / 'all_raw_points.pcd', _room_cloud())
    stamps = {}
    for cam, R_cam, t_cam in (('cam0', R_CAM0, T_CAM0), ('cam1', R_CAM1, T_CAM1)):
        (root / 'images' / cam).mkdir(parents=True)
        for index, t in enumerate(np.arange(1.0, 19.0, 0.5), start=1):
            name = f'{cam}/frame_{index:06d}.png'
            cv2.imwrite(str(root / 'images' / name), _render(t, R_cam, t_cam))  # true dt = 0
            stamps[name] = float(t)
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': stamps}), encoding='utf-8')
    intrinsics = dict(fx=F, fy=F, cx=SIZE / 2, cy=SIZE / 2, k1=0., k2=0., p1=0., p2=0., k3=0., k4=0., sx1=0., sy1=0.)
    profile = root / 'profile.json'
    rig = {}
    for key, R_cam, t_cam in (('T_lidar_to_cam0_rigid_4x4', R_CAM0, T_CAM0), ('T_lidar_to_cam1_rigid_4x4', R_CAM1, T_CAM1)):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_cam, t_cam
        rig[key] = T.tolist()
    profile.write_text(json.dumps({**rig, 'lens_model': 'THIN_PRISM_FISHEYE', 'cam0_front_intrinsics': intrinsics,
                                   'cam1_rear_intrinsics': intrinsics}), encoding='utf-8')
    return root, profile


def test_dt_refinement_recovers_the_true_offset_and_is_validated_on_held_out_frames(synthetic_capture):
    root, profile = synthetic_capture

    result = refine_rig(root, profile, 0.15, mode='dt', frames=12, dt_window=0.3,
                        reference_profile=profile, output=root / 'refined.json', log=lambda *_: None)

    assert abs(result['dt_sync_seconds']) <= 0.04
    holdout = result['photometric_report']['holdout']
    assert holdout['refined']['cost'] < holdout['initial']['cost']
    assert result['quality_status'] == 'accepted'
    assert result['dt_sync_reference'] == 'slam_trajectory_first_pose'


def test_correct_calibration_is_not_replaced_without_held_out_improvement(synthetic_capture):
    root, profile = synthetic_capture

    result = refine_rig(root, profile, 0.0, mode='dt', frames=12, dt_window=0.3,
                        reference_profile=profile, output=root / 'unchanged.json', log=lambda *_: None)

    assert result['quality_status'] == 'rejected'
    assert 'held-out cost improved' in result['rejection_reason']


def test_refinement_result_is_reused_for_the_same_inputs(synthetic_capture):
    root, profile = synthetic_capture
    output = root / 'cached.json'
    first = refine_rig(root, profile, 0.15, mode='dt', frames=12, dt_window=0.3,
                       reference_profile=profile, output=output, log=lambda *_: None)
    messages = []

    second = refine_rig(root, profile, 0.15, mode='dt', frames=12, dt_window=0.3,
                        reference_profile=profile, output=output, log=messages.append)

    assert second == first
    assert any('Reusing photometric rig refinement' in m for m in messages)


def test_existing_calibration_from_other_inputs_is_kept_as_previous(synthetic_capture):
    root, profile = synthetic_capture
    output = root / 'handmade.json'
    output.write_text('{"made": "elsewhere"}', encoding='utf-8')

    refine_rig(root, profile, 0.15, mode='dt', frames=12, dt_window=0.3,
               reference_profile=profile, output=output, log=lambda *_: None)

    assert json.loads((root / 'handmade.previous.json').read_text(encoding='utf-8')) == {'made': 'elsewhere'}
    assert 'photometric_report' in json.loads(output.read_text(encoding='utf-8'))


def test_main_body_accepts_any_supported_model_and_refuses_incomplete_calibrations():
    pinhole = dict(fx=1., fy=1., cx=0., cy=0., k1=0., k2=0., p1=0., p2=0., k3=0., k4=0., sx1=0., sy1=0.)
    calib = {'T_lidar_to_cam0_rigid_4x4': np.eye(4).tolist(), 'cam0_front_intrinsics': {**pinhole, 'camera_model': 'PINHOLE'}}
    body = main_body_from_calibration(calib, 1.0)
    assert [c.name for c in body.cameras] == ['cam0'] and body.cameras[0].model.model == 'PINHOLE'
    equirect = {'lens_model': 'EQUIRECTANGULAR', 'T_lidar_to_cam0_rigid_4x4': np.eye(4).tolist(),
                'T_lidar_to_cam1_rigid_4x4': np.eye(4).tolist(), 'cam1_rear_intrinsics': pinhole,
                'cam0_front_intrinsics': {'camera_model': 'EQUIRECTANGULAR', 'width': 2000, 'height': 1000}}
    assert [c.name for c in main_body_from_calibration(equirect, 0.0).cameras] == ['cam0']
    assert main_body_from_calibration({'lens_model': 'EQUIRECTANGULAR'}, 0.0) is None
    assert main_body_from_calibration({**calib, 'cam0_front_intrinsics': {**pinhole, 'camera_model': 'KANNALA'}}, 0.0) is None
