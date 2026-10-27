"""Rig profile v2: migration of every v1 file, round trips, validation, wizard profiles."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app import profile_manager
from raven_app.rig_bootstrap import CAMERA_CATALOGUE, build_profile, camera_intrinsics, perturb_profile
from raven_app.rig_profile import (RigProfile, apply_profile_to_aux_cameras, load_profile,
                                   profile_from_calibration, save_profile, validate_profile)
from raven_app.third_camera import ThirdCameraConfig

ROOT = Path(__file__).resolve().parents[1]
V1_FILES = [ROOT / 'calibracao_rigida_raven_insta360.json', ROOT / 'configs' / 'rig_profile.json']
BUNDLED = [ROOT / 'configs' / 'profiles' / 'raven_x4_montagem_1.json', ROOT / 'configs' / 'rig_profile_eagle.json']


def _angle(A, B):
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(np.asarray(A)[:3, :3].T @ np.asarray(B)[:3, :3]).as_rotvec())))


@pytest.mark.parametrize('path', V1_FILES + BUNDLED, ids=lambda p: p.name)
def test_every_shipped_profile_migrates_validates_and_round_trips(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    profile = RigProfile.from_dict(data, path=path)
    assert validate_profile(profile) == []
    assert [c.name for c in profile.main.cameras] == ['cam0', 'cam1']
    out = profile.to_dict()
    # The v1 keys the colorizers read are unchanged.
    for key in ('T_lidar_to_cam0_rigid_4x4', 'T_lidar_to_cam1_rigid_4x4'):
        assert np.allclose(out[key], data[key])
    for key in ('cam0_front_intrinsics', 'cam1_rear_intrinsics'):
        strip = lambda d: {k: v for k, v in d.items() if k != 'camera_model'}
        assert strip(out[key]) == strip(data[key])
    again = RigProfile.from_dict(json.loads(json.dumps(out)))
    assert again.to_dict() == out


def test_bundled_profiles_are_v2():
    for path in BUNDLED:
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['schema'] == 2 and data['bodies'][0]['cameras'][0]['name'] == 'cam0'


def test_calibration_with_auxiliary_camera_becomes_a_two_body_profile(tmp_path):
    base = json.loads(BUNDLED[0].read_text(encoding='utf-8'))
    aux = ThirdCameraConfig(camera_name='cam3', camera_model='OPENCV', time_offset_s=-2.15,
                            intrinsics={'width': 3840, 'height': 2160, 'fx': 2646, 'fy': 2650, 'cx': 1920,
                                        'cy': 1080, 'k1': 0.01, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0})
    aux.calibration_report['extrinsics'] = {'status': 'calibrated'}
    rejected = ThirdCameraConfig(camera_name='cam4')
    data = dict(base, auxiliary_cameras=[aux.to_dict(), rejected.to_dict()], quality_status='accepted',
                photometric_report={'holdout': {}})
    profile = profile_from_calibration(data, 'Raven + X4 + phone', base=load_profile(BUNDLED[0]))
    assert [b.body_id for b in profile.bodies] == ['main', 'cam3']
    assert profile.auxiliary[0].time_offset_s == pytest.approx(-2.15)
    assert 'photometric_report' not in profile.to_dict()
    path = save_profile(profile, tmp_path / 'p.json')
    configs = load_profile(path).auxiliary_camera_configs()
    assert configs[0].camera_model == 'OPENCV' and configs[0].intrinsics['fx'] == 2646
    assert configs[0].calibration_report['extrinsics']['status'] == 'calibrated'

    # A capture that only names the source gets the profile's calibration.
    capture = ThirdCameraConfig(camera_name='cam3', source_path='phone.mp4')
    filled = apply_profile_to_aux_cameras([capture, ThirdCameraConfig(camera_name='cam9')], load_profile(path),
                                          log=lambda *_: None)
    assert filled[0].source_path == 'phone.mp4' and filled[0].intrinsics['fx'] == 2646
    assert filled[0].time_offset_s == pytest.approx(-2.15)
    assert filled[1].intrinsics['fx'] == ThirdCameraConfig().intrinsics['fx']


def test_validation_reports_bad_profiles():
    profile = load_profile(BUNDLED[0])
    data = profile.to_dict()
    data['bodies'][0]['cameras'][1]['name'] = 'cam0'
    data['bodies'][0]['cameras'][0]['model'] = 'NOPE'
    errors = validate_profile(data)
    assert any('unique' in e for e in errors) and any('NOPE' in e for e in errors)
    assert validate_profile({'name': 'empty'})


def test_wizard_presets_reproduce_the_validated_rigs():
    raven = build_profile('r', camera='insta360_x4', scanner_preset='raven', offset_cm=(0.8, 12.4, 11.6))
    eagle = build_profile('e', camera='insta360_x4', scanner_preset='eagle', offset_cm=(-9.0, -0.2, 6.6))
    for built, path in ((raven, BUNDLED[0]), (eagle, BUNDLED[1])):
        reference = load_profile(path)
        assert validate_profile(built) == []
        for a, b in zip(built.main.cameras, reference.main.cameras):
            assert _angle(a.T, b.T) < 2.0
            assert np.linalg.norm(np.asarray(a.T)[:3, 3] - np.asarray(b.T)[:3, 3]) < 0.01


def test_wizard_handles_single_cameras_and_auxiliaries():
    profile = build_profile('pole', camera='equirectangular_360', orientation='lens_left',
                            aux_cameras=[{'camera': 'phone_ultrawide', 'offset_cm': (10, 0, -5)}])
    assert validate_profile(profile) == []
    assert [c.model for c in profile.cameras] == ['EQUIRECTANGULAR', 'OPENCV']
    T = np.asarray(profile.main.cameras[0].T)
    assert np.allclose(T[:3, 2], [0, 1, 0], atol=1e-9)    # optical axis points left
    wide = camera_intrinsics(CAMERA_CATALOGUE['phone_ultrawide'], 1920, 1080)
    assert np.degrees(2 * np.arctan(960 / wide['fx'])) == pytest.approx(105.0)
    with pytest.raises(ValueError):
        build_profile('x', camera='unknown')


def test_perturb_moves_the_whole_main_body():
    profile = load_profile(BUNDLED[0])
    moved = perturb_profile(profile, rot_deg=(9, -8, 8), trans_cm=(6, -6, 5))
    for a, b in zip(profile.main.cameras, moved.main.cameras):
        assert 14.0 < _angle(a.T, b.T) < 16.0
    T0, T1 = (np.asarray(c.T) for c in moved.main.cameras)
    R0, R1 = (np.asarray(c.T) for c in profile.main.cameras)
    assert np.linalg.norm(T0[:3, 3] - T1[:3, 3]) == pytest.approx(np.linalg.norm(R0[:3, 3] - R1[:3, 3]))


def test_user_profiles_override_and_become_default(tmp_path, monkeypatch):
    bundled, user = tmp_path / 'bundled', tmp_path / 'user'
    bundled.mkdir()
    monkeypatch.setattr(profile_manager, 'get_profiles_dir', lambda: bundled)
    monkeypatch.setattr(profile_manager, 'get_user_profiles_dir', lambda: user)
    (bundled / 'raven_x4_montagem_1.json').write_text(BUNDLED[0].read_text(encoding='utf-8'), encoding='utf-8')
    assert profile_manager.resolve_profile() == bundled / 'raven_x4_montagem_1.json'

    saved = profile_manager.save_profile_from_calibration(json.loads(BUNDLED[0].read_text(encoding='utf-8')),
                                                          'My Rig', base_profile='raven_x4_montagem_1')
    assert saved == user / 'my_rig.json'
    listed = {p['profile_id']: p for p in profile_manager.list_available_profiles()}
    assert listed['my_rig']['source'] == 'user' and not listed['my_rig']['is_default']
    profile_manager.set_default_profile('my_rig')
    assert profile_manager.resolve_profile() == saved
    assert profile_manager.list_available_profiles()[0]['profile_id'] == 'my_rig'
    profile_manager.set_default_profile('raven_x4_montagem_1')
    assert profile_manager.resolve_profile() == bundled / 'raven_x4_montagem_1.json'
