"""Starting point for a rig that has no validated profile yet.

A new user knows their scanner, their camera model and roughly how the camera sits
on the scanner (a ruler measurement and "lens facing forward").  This builds a
profile from that: catalogue intrinsics for the camera, the mount from the measured
offset plus an orientation preset.  That is good to about ±15° and ±10 cm; the
coarse photometric search in ``photometric_rig.refine_rig(coarse=True)`` takes it
from there.

Frames (all right-handed):
* LiDAR body: x forward, y left, z up (REP-103), tilted by the scanner preset.
* Camera body: the same axes for a camera standing upright with its first lens
  (or its only lens) looking forward.
* Camera: OpenCV (x right, y down, z along the optical axis).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from raven_app.rig_profile import SCANNER_PRESETS, Body, CameraSpec, RigProfile, slugify

# Camera body -> camera rotation of a lens looking forward / backward (columns: camera axes).
_FORWARD = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
_BACKWARD = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)

# How each scanner holds its LiDAR body relative to an upright camera (roll, pitch, yaw in
# degrees, 'xyz' extrinsic), measured on the validated Raven and Eagle profiles.
SCANNER_TILT_DEG = {'raven': (-31.0, 0.0, 0.0), 'eagle': (0.0, -30.0, 0.0), 'generic': (0.0, 0.0, 0.0)}

_X4_FISHEYE = {'fx': 1080.0, 'fy': 1080.0, 'cx': 1920.0, 'cy': 1920.0, 'k1': 0.083, 'k2': -0.0295,
               'p1': 0.0, 'p2': 0.0, 'k3': 0.0109, 'k4': -0.00278, 'sx1': 0.0, 'sy1': 0.0}


def _dual_fisheye(width, scale, baseline_cm):
    intrinsics = {k: (v * scale if k in ('fx', 'fy', 'cx', 'cy') else v) for k, v in _X4_FISHEYE.items()}
    return {'kind': 'dual_fisheye', 'model': 'THIN_PRISM_FISHEYE', 'width': width, 'height': width,
            'intrinsics': intrinsics, 'baseline_cm': baseline_cm, 'sync': 'gyro'}


# Starting intrinsics per camera.  Dual-fisheye entries share the X4 lens shape scaled to
# the sensor; perspective entries use a field of view guess the sample SfM / photometric
# refinement corrects.
CAMERA_CATALOGUE: Dict[str, Dict] = {
    'insta360_x4': {'label': 'Insta360 X4 (dual fisheye .insv)', **_dual_fisheye(3840, 1.0, 2.8)},
    'insta360_x3': {'label': 'Insta360 X3 (dual fisheye .insv)', **_dual_fisheye(2880, 0.75, 2.6)},
    'insta360_x5': {'label': 'Insta360 X5 (dual fisheye .insv)', **_dual_fisheye(3840, 1.0, 2.9)},
    'insta360_one_rs_1inch': {'label': 'Insta360 ONE RS 1-inch 360 (dual fisheye .insv)',
                              **_dual_fisheye(3072, 0.8, 3.6)},
    'equirectangular_360': {'label': 'Any 360 camera, stitched equirectangular .mp4', 'kind': 'equirectangular',
                            'model': 'EQUIRECTANGULAR', 'width': 7680, 'height': 3840, 'intrinsics': {},
                            'sync': 'gyro'},
    'gopro_max_equirect': {'label': 'GoPro MAX (equirectangular export)', 'kind': 'equirectangular',
                           'model': 'EQUIRECTANGULAR', 'width': 5376, 'height': 2688, 'intrinsics': {},
                           'sync': 'photometric'},
    'phone_main': {'label': 'Phone main camera 1x (≈ 75° HFOV)', 'kind': 'perspective', 'model': 'OPENCV',
                   'width': 3840, 'height': 2160, 'hfov_deg': 75.0, 'sync': 'photometric'},
    'phone_ultrawide': {'label': 'Phone ultra-wide 0.5x/0.6x (≈ 105° HFOV)', 'kind': 'perspective',
                        'model': 'OPENCV', 'width': 3840, 'height': 2160, 'hfov_deg': 105.0,
                        'sync': 'photometric'},
    'gopro_wide': {'label': 'GoPro (Wide lens, fisheye)', 'kind': 'fisheye', 'model': 'OPENCV_FISHEYE',
                   'width': 3840, 'height': 2160, 'hfov_deg': 120.0, 'sync': 'photometric'},
    'generic_pinhole': {'label': 'Generic pinhole camera', 'kind': 'perspective', 'model': 'PINHOLE',
                        'width': 1920, 'height': 1080, 'hfov_deg': 70.0, 'sync': 'photometric'},
}

# Camera body orientation on the scanner: yaw of the first lens around the scanner's up axis.
MOUNT_ORIENTATIONS = {
    'lens_forward': {'label': 'Upright, first lens facing forward', 'rpy_deg': (0.0, 0.0, 0.0)},
    'lens_left': {'label': 'Upright, first lens facing left', 'rpy_deg': (0.0, 0.0, 90.0)},
    'lens_right': {'label': 'Upright, first lens facing right', 'rpy_deg': (0.0, 0.0, -90.0)},
    'lens_backward': {'label': 'Upright, first lens facing backward', 'rpy_deg': (0.0, 0.0, 180.0)},
    'upside_down_forward': {'label': 'Upside down, first lens facing forward', 'rpy_deg': (180.0, 0.0, 0.0)},
}

CAPTURE_ADVICE = (
    'Record 2-3 minutes; walk past textured structures 3-15 m away; include a few turns; '
    'keep other people away from the rig; start the camera before the scanner.')


def catalogue_choices() -> List[Dict[str, str]]:
    return [{'id': key, 'label': value['label'], 'model': value['model']} for key, value in CAMERA_CATALOGUE.items()]


def camera_intrinsics(entry: Dict, width: Optional[int] = None, height: Optional[int] = None) -> Dict[str, float]:
    """Intrinsics of a catalogue entry for the actual image size."""
    width = int(width or entry['width'])
    height = int(height or entry['height'])
    if entry['kind'] == 'equirectangular':
        return {}
    if entry['kind'] == 'dual_fisheye':
        scale = width / entry['width']
        return {k: (v * scale if k in ('fx', 'fy', 'cx', 'cy') else v) for k, v in entry['intrinsics'].items()}
    half = np.deg2rad(entry['hfov_deg']) / 2.0
    if entry['kind'] == 'fisheye':
        f = (width / 2.0) / half          # equidistant: r = f * theta
    else:
        f = (width / 2.0) / np.tan(half)
    return {'fx': f, 'fy': f, 'cx': width / 2.0, 'cy': height / 2.0, 'k1': 0.0, 'k2': 0.0,
            'p1': 0.0, 'p2': 0.0, 'k3': 0.0, 'k4': 0.0}


def mount_rotation(scanner_preset='generic', orientation='lens_forward', extra_rpy_deg=(0.0, 0.0, 0.0)):
    """LiDAR body <- camera body rotation from the scanner tilt, the orientation preset and a tweak."""
    tilt = Rot.from_euler('xyz', SCANNER_TILT_DEG.get(scanner_preset, (0.0, 0.0, 0.0)), degrees=True)
    preset = Rot.from_euler('xyz', MOUNT_ORIENTATIONS[orientation]['rpy_deg'], degrees=True)
    extra = Rot.from_euler('xyz', extra_rpy_deg, degrees=True)
    return (tilt * preset * extra).as_matrix()


def build_profile(name: str, *, camera: str, scanner_preset: str = 'generic', scanner_model: str = '',
                  offset_cm=(0.0, 0.0, 0.0), orientation: str = 'lens_forward',
                  extra_rpy_deg=(0.0, 0.0, 0.0), width: Optional[int] = None, height: Optional[int] = None,
                  aux_cameras: Optional[List[Dict]] = None) -> RigProfile:
    """Initial profile for a new rig.

    ``offset_cm`` is the camera centre (between the two lenses of a dual fisheye) in the
    LiDAR body frame, measured with a ruler.  ``aux_cameras`` adds auxiliary bodies:
    dicts with ``name``, ``camera`` (catalogue id), ``offset_cm``, ``orientation`` and
    optional ``extra_rpy_deg`` / ``width`` / ``height``.
    """
    if camera not in CAMERA_CATALOGUE:
        raise ValueError(f'Unknown camera {camera}; choose one of {", ".join(CAMERA_CATALOGUE)}')
    if scanner_preset not in SCANNER_PRESETS:
        raise ValueError(f'Unknown scanner preset {scanner_preset}')
    if orientation not in MOUNT_ORIENTATIONS:
        raise ValueError(f'Unknown mount orientation {orientation}')
    main = _body('main', ['cam0', 'cam1'], CAMERA_CATALOGUE[camera], scanner_preset, offset_cm,
                 orientation, extra_rpy_deg, width, height)
    bodies = [main]
    for i, aux in enumerate(aux_cameras or []):
        entry = CAMERA_CATALOGUE[aux['camera']]
        if entry['kind'] == 'dual_fisheye':
            raise ValueError('Auxiliary cameras must be single-lens (export one lens or a stitched video)')
        name = aux.get('name') or f'cam{i + 2}'
        body = _body(name, [name], entry, scanner_preset, aux.get('offset_cm', (0, 0, 0)),
                     aux.get('orientation', 'lens_forward'), aux.get('extra_rpy_deg', (0, 0, 0)),
                     aux.get('width'), aux.get('height'))
        body.dt_tolerance_s = 3.0
        bodies.append(body)
    preset = SCANNER_PRESETS[scanner_preset]
    return RigProfile(
        profile_id=slugify(name), name=name, bodies=bodies,
        scanner={'model': scanner_model or preset['model'], 'preset': scanner_preset,
                 'lidar_topic': preset['lidar_topic'], 'imu_topic': preset['imu_topic']},
        description=f'New rig from the wizard: {CAMERA_CATALOGUE[camera]["label"]}, '
                    f'{MOUNT_ORIENTATIONS[orientation]["label"].lower()}, offset {tuple(offset_cm)} cm. '
                    'Not validated yet: calibrate it on a capture (coarse search) before use.',
        mount_prior={'rot_sigma_deg': 15.0, 'trans_sigma_cm': 10.0},
        extra={'camera': CAMERA_CATALOGUE[camera]['label'], 'bootstrap': {
            'camera': camera, 'orientation': orientation, 'offset_cm': list(offset_cm),
            'extra_rpy_deg': list(extra_rpy_deg), 'needs_coarse_calibration': True}})


def _body(body_id, names, entry, scanner_preset, offset_cm, orientation, extra_rpy_deg, width, height):
    width = int(width or entry['width'])
    height = int(height or entry['height'])
    R_mount = mount_rotation(scanner_preset, orientation, extra_rpy_deg)
    centre = np.asarray(offset_cm, dtype=float) / 100.0
    intrinsics = camera_intrinsics(entry, width, height)
    lenses = [(names[0], _FORWARD, 0.0)]
    if entry['kind'] == 'dual_fisheye':
        half = entry['baseline_cm'] / 200.0
        lenses = [(names[0], _FORWARD, half), (names[1], _BACKWARD, -half)]
    cameras = []
    for name, R_lens, x in lenses:
        T = np.eye(4)
        T[:3, :3] = R_mount @ R_lens
        T[:3, 3] = centre + R_mount @ np.array([x, 0.0, 0.0])
        cameras.append(CameraSpec(name, entry['model'], width, height, dict(intrinsics), T.tolist()))
    return Body(body_id, cameras, sync=entry.get('sync', 'photometric'),
                dt_tolerance_s=0.6 if entry.get('sync') == 'gyro' else 10.0,
                source='insv' if entry['kind'] == 'dual_fisheye' else 'video')


def perturb_profile(profile: RigProfile, rot_deg=(0.0, 0.0, 0.0), trans_cm=(0.0, 0.0, 0.0)) -> RigProfile:
    """Move the main body rigidly in the LiDAR frame (tests and robustness checks)."""
    out = copy.deepcopy(profile)
    D = np.eye(4)
    D[:3, :3] = Rot.from_euler('xyz', rot_deg, degrees=True).as_matrix()
    D[:3, 3] = np.asarray(trans_cm, dtype=float) / 100.0
    for camera in out.main.cameras:
        camera.T = (D @ np.asarray(camera.T)).tolist()
    return out
