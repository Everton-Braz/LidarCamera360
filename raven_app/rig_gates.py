"""Physical plausibility gates for per-capture rig calibrations.

A rigid mount does not change between captures, so a per-capture solve (full SfM,
sample SfM, photometric) may only nudge the hardware profile.  PIER's sample-SfM rig
(6.6 cm lens baseline for a 2.8 cm camera) and HarmonyMall's SfM rig (lever arms
~29 cm away from the profile) both passed the old 40 cm lever-arm check and produced
misaligned colours; these gates reject them while accepting the photometric
refinements (<= 3.4 cm, <= 3 deg from the profile).
"""
import json
from pathlib import Path

import numpy as np

LIMITS = {
    'max_lever_arm_cm': 40.0,
    'max_baseline_delta_cm': 1.5,
    'max_lens_angle_delta_deg': 2.0,
    'max_rotation_delta_deg': 10.0,
    'max_translation_delta_cm': 8.0,
}


def _matrix(value):
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return matrix if matrix.shape == (4, 4) and np.all(np.isfinite(matrix)) else None


def rig_transforms(data):
    """Return (T_lidar_to_cam0, T_lidar_to_cam1) of a calibration dict; missing ones are None."""
    if 'Cam0' in data and isinstance(data['Cam0'], dict):
        return (_matrix(data['Cam0'].get('T_lidar_cam')),
                _matrix((data.get('Cam1') or {}).get('T_lidar_cam')))
    cam0 = next((data[key] for key in ('T_lidar_to_cam0_rigid_4x4', 'T_lidar_to_camera_rigid_4x4',
                                       'T_lidar_cam', 'T_CL') if key in data), None)
    return _matrix(cam0), _matrix(data.get('T_lidar_to_cam1_rigid_4x4'))


def _angle_deg(rotation):
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def rig_gate_failures(candidate, reference=None, limits=None):
    """List the reasons ``candidate`` is physically implausible (empty = accepted).

    Without a reference only the absolute lever-arm limit applies.  Profile
    comparisons are skipped when the lens models differ (e.g. an equirectangular
    capture checked against a dual-fisheye profile).
    """
    limits = {**LIMITS, **(limits or {})}
    if reference is not None and isinstance(reference, dict):
        prior = reference.get('mount_prior') or {}
        user_limits = limits or {}
        if 'max_rotation_delta_deg' not in user_limits and prior.get('rot_sigma_deg'):
            limits['max_rotation_delta_deg'] = max(limits['max_rotation_delta_deg'], float(prior['rot_sigma_deg']) * 2.5)
        if 'max_translation_delta_cm' not in user_limits and prior.get('trans_sigma_cm'):
            limits['max_translation_delta_cm'] = max(limits['max_translation_delta_cm'], float(prior['trans_sigma_cm']) * 2.0)
    T0, T1 = rig_transforms(candidate)
    if T0 is None:
        return ['calibration has no lidar-to-camera transform']
    reasons = []
    for name, T in (('cam0', T0), ('cam1', T1)):
        if T is not None:
            lever = float(np.linalg.norm(T[:3, 3]) * 100.0)
            if lever > limits['max_lever_arm_cm']:
                reasons.append(f'{name} lever arm {lever:.1f} cm exceeds {limits["max_lever_arm_cm"]:.0f} cm')
    if reference is None:
        return reasons
    model, reference_model = candidate.get('lens_model'), reference.get('lens_model')
    if model and reference_model and model != reference_model:
        return reasons
    R0, R1 = rig_transforms(reference)
    if R0 is None:
        return reasons
    for name, T, R in (('cam0', T0, R0), ('cam1', T1, R1)):
        if T is None or R is None:
            continue
        rotation = _angle_deg(R[:3, :3].T @ T[:3, :3])
        translation = float(np.linalg.norm(T[:3, 3] - R[:3, 3]) * 100.0)
        if rotation > limits['max_rotation_delta_deg']:
            reasons.append(f'{name} rotation differs from the profile by {rotation:.1f}° '
                           f'(limit {limits["max_rotation_delta_deg"]:.0f}°)')
        if translation > limits['max_translation_delta_cm']:
            reasons.append(f'{name} position differs from the profile by {translation:.1f} cm '
                           f'(lever arm {np.linalg.norm(T[:3, 3]) * 100:.1f} cm vs '
                           f'{np.linalg.norm(R[:3, 3]) * 100:.1f} cm in the profile; '
                           f'limit {limits["max_translation_delta_cm"]:.0f} cm)')
    if T1 is not None and R1 is not None:
        baseline = float(np.linalg.norm(T0[:3, 3] - T1[:3, 3]) * 100.0)
        reference_baseline = float(np.linalg.norm(R0[:3, 3] - R1[:3, 3]) * 100.0)
        if abs(baseline - reference_baseline) > limits['max_baseline_delta_cm']:
            reasons.append(f'lens baseline {baseline:.1f} cm vs {reference_baseline:.1f} cm in the profile '
                           f'(limit ±{limits["max_baseline_delta_cm"]:.1f} cm)')
        lens_angle = _angle_deg(T0[:3, :3].T @ T1[:3, :3])
        reference_angle = _angle_deg(R0[:3, :3].T @ R1[:3, :3])
        if abs(lens_angle - reference_angle) > limits['max_lens_angle_delta_deg']:
            reasons.append(f'angle between lenses {lens_angle:.2f}° vs {reference_angle:.2f}° in the profile '
                           f'(limit ±{limits["max_lens_angle_delta_deg"]:.0f}°)')
    return reasons


def load_reference(profile_path):
    """Read a profile JSON, or None when it is missing or unreadable."""
    if not profile_path:
        return None
    try:
        return json.loads(Path(profile_path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
