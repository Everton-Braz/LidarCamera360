import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app import profile_manager
from raven_app.person_masks import check_masks_match_frames
from raven_app.rig_gates import rig_gate_failures
from raven_app.workflow import resolve_calibration_file, validate_calibration_file

PROFILE = Path(__file__).resolve().parents[1] / 'configs' / 'profiles' / 'raven_x4_montagem_1.json'


def _profile():
    return json.loads(PROFILE.read_text(encoding='utf-8'))


def _nudged(profile, rot_deg=(0, 0, 0), trans_cm=(0, 0, 0), cam1_shift_cm=(0, 0, 0)):
    """Move the whole rig rigidly in the LiDAR body frame, and optionally cam1 alone."""
    out = dict(profile)
    delta = np.eye(4)
    delta[:3, :3] = Rot.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    delta[:3, 3] = np.asarray(trans_cm) / 100.0
    T0 = delta @ np.asarray(profile['T_lidar_to_cam0_rigid_4x4'])
    T1 = delta @ np.asarray(profile['T_lidar_to_cam1_rigid_4x4'])
    T1[:3, 3] += np.asarray(cam1_shift_cm) / 100.0
    out['T_lidar_to_cam0_rigid_4x4'] = T0.tolist()
    out['T_lidar_to_cam1_rigid_4x4'] = T1.tolist()
    return out


def test_profile_and_small_photometric_style_refinement_are_accepted():
    profile = _profile()
    assert rig_gate_failures(profile, profile) == []
    # PIER / HarmonyMall photometric refinements moved the rig by <= 3 deg and <= 3.4 cm.
    assert rig_gate_failures(_nudged(profile, rot_deg=(0, 3, 0), trans_cm=(-3.4, 0.5, -0.7)), profile) == []


def test_lens_baseline_far_from_the_camera_body_is_rejected():
    # PIER sample-SfM rig: 6.6 cm baseline for a 2.8 cm Insta360 X4.
    reasons = rig_gate_failures(_nudged(_profile(), cam1_shift_cm=(0, 0, 4.0)), _profile())
    assert any('lens baseline' in r for r in reasons)


def test_rig_far_from_the_profile_is_rejected():
    # HarmonyMall SfM rig: lever arms ~29 cm away from the profile.
    reasons = rig_gate_failures(_nudged(_profile(), trans_cm=(0, -29, 0)), _profile())
    assert any('position differs' in r for r in reasons)
    reasons = rig_gate_failures(_nudged(_profile(), rot_deg=(0, 0, 15)), _profile())
    assert any('rotation differs' in r for r in reasons)


def test_without_reference_only_the_absolute_lever_arm_applies():
    far = _nudged(_profile(), trans_cm=(0, -29, 0))
    assert rig_gate_failures(far) == []
    huge = _nudged(_profile(), trans_cm=(0, 0, 60))
    assert any('lever arm' in r for r in rig_gate_failures(huge))


def test_profile_comparison_is_skipped_for_a_different_lens_model():
    equirect = _nudged(_profile(), rot_deg=(0, 90, 0))
    equirect['lens_model'] = 'EQUIRECTANGULAR'
    assert rig_gate_failures(equirect, _profile()) == []


def test_resolver_ignores_implausible_dataset_calibration_and_uses_profile(tmp_path):
    (tmp_path / 'rig_calibration.json').write_text(
        json.dumps(_nudged(_profile(), trans_cm=(0, -29, 0))), encoding='utf-8')

    resolved = resolve_calibration_file(tmp_path, profile=str(PROFILE))

    assert resolved == PROFILE
    assert not validate_calibration_file(tmp_path / 'rig_calibration.json', reference_profile=PROFILE)
    # The legacy lever-arm-only check (no reference) still accepts it.
    assert validate_calibration_file(tmp_path / 'rig_calibration.json')


def test_resolver_keeps_plausible_dataset_calibration(tmp_path):
    rig = tmp_path / 'rig_calibration.json'
    rig.write_text(json.dumps(_nudged(_profile(), rot_deg=(0, 2, 0))), encoding='utf-8')
    assert resolve_calibration_file(tmp_path, profile=str(PROFILE)) == rig


def test_resolve_profile_by_path_id_and_default(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_manager, 'get_profiles_dir', lambda: tmp_path)
    data = dict(_profile(), profile_id='custom_rig', is_default=False)
    (tmp_path / 'my_rig.json').write_text(json.dumps(data), encoding='utf-8')
    default = dict(_profile(), is_default=True)
    (tmp_path / 'other.json').write_text(json.dumps(default), encoding='utf-8')

    assert profile_manager.resolve_profile(str(tmp_path / 'my_rig.json')) == tmp_path / 'my_rig.json'
    assert profile_manager.resolve_profile('my_rig') == tmp_path / 'my_rig.json'
    assert profile_manager.resolve_profile('custom_rig') == tmp_path / 'my_rig.json'
    assert profile_manager.resolve_profile() == tmp_path / 'other.json'
    with pytest.raises(ValueError, match='Unknown rig profile'):
        profile_manager.resolve_profile('missing')


def test_saved_profiles_never_claim_to_be_the_default(tmp_path, monkeypatch):
    monkeypatch.setattr(profile_manager, 'get_profiles_dir', lambda: tmp_path)
    saved = profile_manager.save_profile_from_calibration(dict(_profile(), is_default=True), 'Copy')
    assert json.loads(saved.read_text(encoding='utf-8'))['is_default'] is False


def _manifest(path, fps, stamps):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'frames.json').write_text(json.dumps({'signature': {'fps': fps}, 'timestamps': stamps}),
                                      encoding='utf-8')


def test_masks_from_another_extraction_fps_are_refused(tmp_path):
    # PIER: studio masks extracted at 1 fps reused for 2 fps CLI images.
    _manifest(tmp_path / 'studio' / 'images', 1.0, {'cam0/frame_000002.jpg': 1.0})
    _manifest(tmp_path / 'cli' / 'images', 2.0, {'cam0/frame_000002.jpg': 0.5})
    with pytest.raises(ValueError, match='1.0 fps extraction'):
        check_masks_match_frames(tmp_path / 'studio' / 'masks', tmp_path / 'cli' / 'images')


def test_masks_with_shifted_frame_times_are_refused(tmp_path):
    _manifest(tmp_path / 'a' / 'images', 2.0, {'cam0/frame_000002.jpg': 0.5})
    _manifest(tmp_path / 'b' / 'images', 2.0, {'cam0/frame_000002.jpg': 0.625})
    with pytest.raises(ValueError, match='different frame extraction'):
        check_masks_match_frames(tmp_path / 'a' / 'masks', tmp_path / 'b' / 'images')


def test_masks_of_the_same_extraction_are_accepted(tmp_path):
    _manifest(tmp_path / 'out' / 'images', 2.0, {'cam0/frame_000002.jpg': 0.5})
    check_masks_match_frames(tmp_path / 'out' / 'masks', tmp_path / 'out' / 'images')
    # A mask set without its own manifest cannot be checked and is left to the per-file check.
    check_masks_match_frames(tmp_path / 'elsewhere' / 'masks', tmp_path / 'out' / 'images')
