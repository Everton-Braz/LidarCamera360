"""Generic rigs: equirectangular main camera and an auxiliary pinhole body with its own clock."""
import json

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app.third_camera import ThirdCameraConfig
from raven_app.photometric_rig import refine_rig
from test_photometric_rig import R_CAM0, R_CAM1, ROOM, T_CAM0, T_CAM1, _body, _room_cloud, _texture, _write_pcd

R_PHONE = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])  # phone looks along body +x, like cam0
T_PHONE = np.array([0.05, 0.12, 0.25])
PHONE_OFFSET = 0.8   # phone clock runs 0.8 s ahead of the main camera clock


def _rays(model, width, height, f=None):
    vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
    if model == 'EQUIRECTANGULAR':
        lon = (uu / width - 0.5) * 2 * np.pi
        lat = (vv / height - 0.5) * np.pi
        rays = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    elif model == 'PINHOLE':
        rays = np.stack([(uu - width / 2) / f, (vv - height / 2) / f, np.ones_like(uu)], -1)
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    else:
        xd, yd = (uu - width / 2) / f, (vv - height / 2) / f
        theta = np.hypot(xd, yd)
        s = np.divide(np.sin(theta), theta, out=np.ones_like(theta), where=theta > 1e-9)
        rays = np.stack([xd * s, yd * s, np.cos(theta)], -1)
    return rays.reshape(-1, 3)


def _render(t, R_cam, t_cam, rays, shape):
    position, R_body = _body(t)
    centre = position + R_body @ t_cam
    world = rays @ (R_body @ R_cam).T
    best = np.full(len(world), np.inf)
    for axis in range(3):
        for bound in ROOM[axis]:
            with np.errstate(divide='ignore', invalid='ignore'):
                hit = (bound - centre[axis]) / world[:, axis]
            best = np.where((hit > 1e-6) & (hit < best), hit, best)
    return (_texture(centre + world * best[:, None]) * 255).reshape(shape).astype(np.uint8)


def _write_slam(root):
    slam = root / 'slam_out'
    times = np.arange(0.0, 20.0, 0.05)
    poses = [_body(t) for t in times]
    quats = Rot.from_matrix(np.stack([R for _, R in poses])).as_quat()
    (slam / 'result').mkdir(parents=True)
    np.savetxt(slam / 'result' / 'Raven_3DMakerPro_Scan.txt',
               np.column_stack([1000.0 + times, np.stack([p for p, _ in poses]), quats]), fmt='%.6f')
    _write_pcd(slam / 'pcd' / 'all_raw_points.pcd', _room_cloud())


def _rig_json(path, cameras):
    data = {'lens_model': cameras[0][1]['camera_model']}
    for name, intrinsics, R, t in cameras:
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        key = 'cam0' if name == 'cam0' else 'cam1'
        data[f'T_lidar_to_{key}_rigid_4x4'] = T.tolist()
        data['cam0_front_intrinsics' if key == 'cam0' else 'cam1_rear_intrinsics'] = intrinsics
    path.write_text(json.dumps(data), encoding='utf-8')
    return path


def test_equirectangular_main_camera_recovers_time_offset(tmp_path):
    root = tmp_path / 'equirect'
    _write_slam(root)
    W, H = 1600, 800
    rays = _rays('EQUIRECTANGULAR', W, H)
    (root / 'images' / 'cam0').mkdir(parents=True)
    stamps = {}
    for index, t in enumerate(np.arange(1.0, 19.0, 0.5), start=1):
        name = f'cam0/frame_{index:06d}.png'
        cv2.imwrite(str(root / 'images' / name), _render(t, R_CAM0, T_CAM0, rays, (H, W)))
        stamps[name] = float(t)
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': stamps}), encoding='utf-8')
    calib = _rig_json(root / 'equirect.json', [('cam0', {'camera_model': 'EQUIRECTANGULAR', 'width': W, 'height': H},
                                                R_CAM0, T_CAM0)])

    result = refine_rig(root, calib, 0.15, mode='dt', frames=12, dt_window=0.3, output=root / 'out.json',
                        log=lambda *_: None)

    assert result['quality_status'] == 'accepted'
    assert abs(result['dt_sync_seconds']) <= 0.04


def test_equirectangular_export_orientation_is_found_from_a_fisheye_profile(tmp_path):
    root = tmp_path / 'export'
    _write_slam(root)
    W, H = 1600, 800
    rays = _rays('EQUIRECTANGULAR', W, H)
    # The stitched export is centred 90 degrees away from the profile's cam0 lens.
    R_true = R_CAM0 @ Rot.from_euler('y', 90, degrees=True).as_matrix()
    (root / 'images' / 'cam0').mkdir(parents=True)
    stamps = {}
    for index, t in enumerate(np.arange(1.0, 19.0, 0.5), start=1):
        name = f'cam0/frame_{index:06d}.jpg'
        cv2.imwrite(str(root / 'images' / name), _render(t, R_true, T_CAM0, rays, (H, W)))
        stamps[name] = float(t)
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': stamps, 'video_mode': 'equirectangular'}),
                                                 encoding='utf-8')
    fisheye = dict(camera_model='THIN_PRISM_FISHEYE', fx=120., fy=120., cx=200., cy=200., k1=0., k2=0., p1=0., p2=0.,
                   k3=0., k4=0., sx1=0., sy1=0., width=400, height=400)
    profile = _rig_json(root / 'x4_profile.json', [('cam0', fisheye, R_CAM0, T_CAM0), ('cam1', fisheye, R_CAM1, T_CAM1)])

    result = refine_rig(root, profile, 0.0, mode='photometric', fix_dt=True, frames=12, output=root / 'out.json',
                        log=lambda *_: None)

    assert result['lens_model'] == 'EQUIRECTANGULAR' and 'T_lidar_to_cam1_rigid_4x4' not in result
    assert result['cam0_front_intrinsics'] == {'camera_model': 'EQUIRECTANGULAR', 'width': W, 'height': H}
    R = np.asarray(result['T_lidar_to_cam0_rigid_4x4'])[:3, :3]
    assert np.degrees(Rot.from_matrix(R_true.T @ R).magnitude()) < 1.0
    assert result['quality_status'] == 'accepted'


@pytest.fixture(scope='module')
def two_body_capture(tmp_path_factory):
    root = tmp_path_factory.mktemp('two_body') / 'dataset'
    _write_slam(root)
    size, f_fish = 400, 120.0
    fish_rays = _rays('THIN_PRISM_FISHEYE', size, size, f_fish)
    pw, ph, f_phone = 960, 540, 600.0
    phone_rays = _rays('PINHOLE', pw, ph, f_phone)
    stamps = {}
    for cam, R_cam, t_cam in (('cam0', R_CAM0, T_CAM0), ('cam1', R_CAM1, T_CAM1)):
        (root / 'images' / cam).mkdir(parents=True)
        for index, t in enumerate(np.arange(1.0, 19.0, 0.5), start=1):
            name = f'{cam}/frame_{index:06d}.png'
            cv2.imwrite(str(root / 'images' / name), _render(t, R_cam, t_cam, fish_rays, (size, size)))
            stamps[name] = float(t)
    (root / 'images' / 'cam2').mkdir(parents=True)
    for index, t in enumerate(np.arange(2.0, 18.0, 0.5), start=1):
        name = f'cam2/frame_{index:06d}.png'
        cv2.imwrite(str(root / 'images' / name), _render(t, R_PHONE, T_PHONE, phone_rays, (ph, pw)))
        stamps[name] = float(t + PHONE_OFFSET)   # phone clock
    (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': stamps}), encoding='utf-8')
    fisheye = dict(camera_model='THIN_PRISM_FISHEYE', fx=f_fish, fy=f_fish, cx=size / 2, cy=size / 2, k1=0., k2=0.,
                   p1=0., p2=0., k3=0., k4=0., sx1=0., sy1=0., width=size, height=size)
    calib = _rig_json(root / 'x4.json', [('cam0', fisheye, R_CAM0, T_CAM0), ('cam1', fisheye, R_CAM1, T_CAM1)])
    guess = np.eye(4)
    guess[:3, :3] = Rot.from_euler('z', 3, degrees=True).as_matrix() @ R_PHONE
    guess[:3, 3] = T_PHONE + [0.05, 0.0, 0.0]
    phone = ThirdCameraConfig(camera_name='cam2', camera_model='PINHOLE', fps=2.0, time_offset_s=0.0)
    phone.intrinsics.update(width=pw, height=ph, fx=f_phone, fy=f_phone, cx=pw / 2, cy=ph / 2)
    phone.T_lidar_to_cam2_rigid_4x4 = guess.tolist()
    phone.calibration_report = {'intrinsics': {'status': 'calibrated'}, 'extrinsics': {'status': 'uncalibrated'}}
    return root, calib, phone


def test_auxiliary_body_recovers_its_own_clock_and_mount(two_body_capture):
    root, calib, phone = two_body_capture

    result = refine_rig(root, calib, 0.0, mode='photometric', fix_dt=True, frames=10, aux_cameras=[phone],
                        aux_dt_window=1.5, output=root / 'refined.json', log=lambda *_: None)

    aux = ThirdCameraConfig.from_dict(result['auxiliary_cameras'][0])
    assert aux.calibration_report['extrinsics']['status'] == 'calibrated'
    assert abs(aux.time_offset_s - PHONE_OFFSET) <= 0.05
    T = np.asarray(aux.T_lidar_to_cam2_rigid_4x4)
    rotation_error = np.degrees(Rot.from_matrix(R_PHONE.T @ T[:3, :3]).magnitude())
    assert rotation_error < 1.5   # started 3 degrees off
    body = next(b for b in result['photometric_report']['bodies'] if b['body'] == 'cam2')
    assert body['holdout']['refined']['cost'] < body['holdout']['initial']['cost']


def test_auxiliary_body_without_calibrated_intrinsics_is_skipped(two_body_capture, tmp_path):
    root, calib, phone = two_body_capture
    unknown = ThirdCameraConfig.from_dict(phone.to_dict())
    unknown.calibration_report = {'intrinsics': {'status': 'uncalibrated'}}

    result = refine_rig(root, calib, 0.0, mode='dt', fix_dt=True, frames=10, aux_cameras=[unknown],
                        output=tmp_path / 'skip.json', log=lambda *_: None)

    assert result['photometric_report']['skipped'][0]['camera'] == 'cam2'
    assert ThirdCameraConfig.from_dict(result['auxiliary_cameras'][0]).calibration_report.get(
        'extrinsics', {}).get('status') != 'calibrated'
