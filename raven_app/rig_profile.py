"""Rig profile v2: one file describes any LiDAR + cameras + mount.

A profile is a scanner plus *bodies*; a body is cameras sharing one housing and one
clock (an Insta360 = cam0 + cam1, a stitched 360 export, a phone or GoPro).  The
first body is the main camera (cam0 [+ cam1]); further bodies are auxiliary cameras
with their own clock offset relative to the main one.

Transforms use the app convention ``camera_to_trajectory_body``: ``T[:3, :3]``
holds the camera axes (x right, y down, z forward) in the LiDAR body frame and
``T[:3, 3]`` the camera position in it.

Version 1 files (the flat ``cam0_front_intrinsics`` / ``T_lidar_to_cam0_rigid_4x4``
layout of every calibration JSON) are migrated on load.  Saved v2 files also carry
the v1 keys, so every colorizer that reads a calibration file keeps working.
"""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from raven_app.camera_models import INTRINSIC_KEYS, SUPPORTED_MODELS, CameraModel

SCHEMA_VERSION = 2
SYNC_METHODS = ('gyro', 'photometric', 'manual')
SCANNER_PRESETS = {
    'raven': {'model': '3DMakerPro Raven', 'lidar_topic': '/vanjee_722z', 'imu_topic': '/vanjee_imu_packets'},
    'eagle': {'model': '3DMakerPro Eagle', 'lidar_topic': '/livox/lidar', 'imu_topic': '/livox/imu'},
    'generic': {'model': 'Generic LiDAR', 'lidar_topic': '', 'imu_topic': ''},
}
_V1_CAMERA_KEYS = {'cam0': 'cam0_front_intrinsics', 'cam1': 'cam1_rear_intrinsics'}
_V2_KEYS = ('schema', 'profile_id', 'name', 'description', 'scanner', 'bodies', 'mount_prior',
            'validated_on', 'is_default')


def user_profiles_dir() -> Path:
    """Profiles the user creates live outside the app folder, so updates keep them."""
    base = os.environ.get('APPDATA') or str(Path.home() / '.config')
    return Path(base) / 'LidarCamera360' / 'profiles'


def slugify(name: str) -> str:
    slug = re.sub(r'[^a-z0-9_-]+', '_', str(name).strip().lower()).strip('_')
    return slug or 'rig'


def _matrix(value, what):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f'{what}: needs a finite 4x4 matrix')
    return matrix


@dataclass
class CameraSpec:
    name: str
    model: str
    width: int
    height: int
    intrinsics: Dict[str, float] = field(default_factory=dict)
    T: List[List[float]] = field(default_factory=lambda: np.eye(4).tolist())
    valid_radius_px: Optional[float] = None

    def camera_model(self) -> CameraModel:
        model = CameraModel.from_intrinsics(self.model, self.intrinsics, self.width, self.height)
        model.valid_radius_px = self.valid_radius_px
        return model

    def to_dict(self) -> Dict[str, Any]:
        data = {'name': self.name, 'model': self.model, 'width': int(self.width), 'height': int(self.height),
                'intrinsics': {k: float(v) for k, v in self.intrinsics.items()},
                'T_lidar_to_cam': [[float(v) for v in row] for row in self.T]}
        if self.valid_radius_px:
            data['valid_radius_px'] = float(self.valid_radius_px)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CameraSpec':
        intrinsics = {k: float(v) for k, v in (data.get('intrinsics') or {}).items()
                      if k in INTRINSIC_KEYS}
        return cls(name=str(data['name']), model=str(data.get('model', 'PINHOLE')).upper(),
                   width=int(data.get('width') or 0), height=int(data.get('height') or 0),
                   intrinsics=intrinsics, T=_matrix(data.get('T_lidar_to_cam'), data['name']).tolist(),
                   valid_radius_px=data.get('valid_radius_px'))

    def v1_intrinsics(self) -> Dict[str, Any]:
        return {**{k: float(v) for k, v in self.intrinsics.items()}, 'width': int(self.width),
                'height': int(self.height), 'camera_model': self.model}


@dataclass
class Body:
    body_id: str
    cameras: List[CameraSpec]
    sync: str = 'gyro'
    time_offset_s: float = 0.0      # auxiliary bodies: clock offset from the main body
    dt_tolerance_s: float = 0.6
    source: str = ''                # free text: insv, video, image_folder, bag_topic

    def to_dict(self) -> Dict[str, Any]:
        return {'body_id': self.body_id, 'source': self.source,
                'clock': {'sync': self.sync, 'time_offset_s': float(self.time_offset_s),
                          'dt_tolerance_s': float(self.dt_tolerance_s)},
                'cameras': [camera.to_dict() for camera in self.cameras]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Body':
        clock = data.get('clock') or {}
        return cls(body_id=str(data.get('body_id') or 'body'),
                   cameras=[CameraSpec.from_dict(c) for c in data.get('cameras') or []],
                   sync=str(clock.get('sync', 'gyro')), time_offset_s=float(clock.get('time_offset_s', 0.0)),
                   dt_tolerance_s=float(clock.get('dt_tolerance_s', 0.6)), source=str(data.get('source', '')))


@dataclass
class RigProfile:
    profile_id: str
    name: str
    bodies: List[Body]
    scanner: Dict[str, Any] = field(default_factory=lambda: {'model': '', 'preset': 'generic'})
    description: str = ''
    mount_prior: Dict[str, float] = field(default_factory=lambda: {'rot_sigma_deg': 3.0, 'trans_sigma_cm': 5.0})
    validated_on: List[Dict[str, Any]] = field(default_factory=list)
    is_default: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)   # unknown keys, kept on save
    path: Optional[Path] = None

    @property
    def main(self) -> Body:
        return self.bodies[0]

    @property
    def auxiliary(self) -> List[Body]:
        return self.bodies[1:]

    @property
    def cameras(self) -> List[CameraSpec]:
        return [camera for body in self.bodies for camera in body.cameras]

    def camera(self, name) -> Optional[CameraSpec]:
        return next((c for c in self.cameras if c.name == name), None)

    # -- v1 view ---------------------------------------------------------------
    def legacy_calibration(self) -> Dict[str, Any]:
        """The flat v1 calibration dict every colorizer understands."""
        main = self.main.cameras
        cam0 = main[0]
        data = {
            'scanner': self.scanner.get('model', ''), 'camera': self.extra.get('camera', self.name),
            'lens_model': cam0.model, 'transform_convention': 'camera_to_trajectory_body',
            'cam0_front_intrinsics': cam0.v1_intrinsics(), 'T_lidar_to_cam0_rigid_4x4': cam0.T,
            'lever_arm_cam0_meters': _lever(cam0.T),
        }
        if len(main) > 1:
            cam1 = main[1]
            T0, T1 = np.asarray(cam0.T), np.asarray(cam1.T)
            data.update({
                'cam1_rear_intrinsics': cam1.v1_intrinsics(), 'T_lidar_to_cam1_rigid_4x4': cam1.T,
                'lever_arm_cam1_meters': _lever(cam1.T),
                'relative_lens_angle_deg': float(np.degrees(np.arccos(np.clip(
                    (np.trace(T0[:3, :3].T @ T1[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))),
                'lens_baseline_cm': float(np.linalg.norm(T0[:3, 3] - T1[:3, 3]) * 100.0),
            })
        if self.scanner.get('lidar_topic') or self.scanner.get('imu_topic'):
            data['topics'] = {'lidar': self.scanner.get('lidar_topic', ''), 'imu': self.scanner.get('imu_topic', '')}
        if self.auxiliary:
            data['auxiliary_cameras'] = [config.to_dict() for config in self.auxiliary_camera_configs()]
        return data

    def auxiliary_camera_configs(self):
        """Auxiliary bodies as ``ThirdCameraConfig`` (one per camera, no source path)."""
        from raven_app.third_camera import ThirdCameraConfig
        configs = []
        for body in self.auxiliary:
            for camera in body.cameras:
                status = {'status': 'calibrated', 'source': 'rig_profile', 'profile_id': self.profile_id}
                configs.append(ThirdCameraConfig(
                    camera_name=camera.name, source_type=body.source or 'video', enabled=True,
                    time_offset_s=float(body.time_offset_s), camera_model=camera.model,
                    intrinsics={**camera.intrinsics, 'width': camera.width, 'height': camera.height},
                    T_lidar_to_cam2_rigid_4x4=[list(row) for row in camera.T],
                    description=f'{body.body_id} from rig profile {self.profile_id}',
                    calibration_report={'intrinsics': dict(status),
                                        'time_offset': {**status, 'applied': True,
                                                        'seconds': float(body.time_offset_s),
                                                        'status': 'configured'},
                                        'extrinsics': dict(status)}))
        return configs

    # -- serialisation ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        data = {**copy.deepcopy(self.extra), **self.legacy_calibration()}
        data.update({
            'schema': SCHEMA_VERSION, 'profile_id': self.profile_id, 'name': self.name,
            'description': self.description, 'scanner_spec': dict(self.scanner),
            'bodies': [body.to_dict() for body in self.bodies], 'mount_prior': dict(self.mount_prior),
            'validated_on': list(self.validated_on), 'is_default': bool(self.is_default),
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, path=None) -> 'RigProfile':
        if not isinstance(data, dict):
            raise ValueError('A rig profile must be a JSON object')
        if int(data.get('schema') or 1) >= 2 and isinstance(data.get('bodies'), list):
            bodies = [Body.from_dict(b) for b in data['bodies']]
            scanner = dict(data.get('scanner_spec') or {})
            if not scanner and isinstance(data.get('scanner'), dict):
                scanner = dict(data['scanner'])
        else:
            bodies, scanner = _migrate_v1(data)
        if not bodies or not bodies[0].cameras:
            raise ValueError('A rig profile needs at least one camera')
        known = set(_V2_KEYS) | {'scanner_spec', 'cam0_front_intrinsics', 'cam1_rear_intrinsics',
                                 'T_lidar_to_cam0_rigid_4x4', 'T_lidar_to_cam1_rigid_4x4', 'auxiliary_cameras',
                                 'lever_arm_cam0_meters', 'lever_arm_cam1_meters', 'relative_lens_angle_deg',
                                 'lens_baseline_cm', 'lens_model', 'topics', 'transform_convention'}
        stem = Path(path).stem if path else ''
        return cls(
            profile_id=str(data.get('profile_id') or stem or slugify(data.get('name', 'rig'))),
            name=str(data.get('name') or stem or 'Rig'), bodies=bodies, scanner=scanner,
            description=str(data.get('description') or ''),
            mount_prior=dict(data.get('mount_prior') or {'rot_sigma_deg': 3.0, 'trans_sigma_cm': 5.0}),
            validated_on=list(data.get('validated_on') or []), is_default=bool(data.get('is_default', False)),
            extra={k: v for k, v in data.items() if k not in known}, path=Path(path) if path else None)


def _lever(T):
    t = np.asarray(T, dtype=np.float64)[:3, 3]
    return {'dx': float(t[0]), 'dy': float(t[1]), 'dz': float(t[2]), 'norm_distance_cm': float(np.linalg.norm(t) * 100)}


def _v1_camera(name, intrinsics, transform, lens_model):
    intrinsics = dict(intrinsics or {})
    model = str(intrinsics.get('camera_model') or lens_model or 'THIN_PRISM_FISHEYE').upper()
    return CameraSpec(name=name, model=model, width=int(intrinsics.get('width') or 0),
                      height=int(intrinsics.get('height') or 0),
                      intrinsics={k: float(v) for k, v in intrinsics.items() if k in INTRINSIC_KEYS},
                      T=_matrix(transform, name).tolist())


def _migrate_v1(data):
    lens_model = data.get('lens_model')
    main = []
    for name, key in _V1_CAMERA_KEYS.items():
        transform = data.get(f'T_lidar_to_{name}_rigid_4x4')
        if transform is None or (key not in data and name != 'cam0'):
            continue
        intrinsics = data.get(key) or {}
        if not intrinsics and str(lens_model).upper() == 'EQUIRECTANGULAR':
            intrinsics = {'camera_model': 'EQUIRECTANGULAR'}
        main.append(_v1_camera(name, intrinsics, transform, lens_model))
    if not main:
        raise ValueError('Profile has no T_lidar_to_cam0_rigid_4x4')
    bodies = [Body('main', main, sync='gyro', source='insv' if len(main) > 1 else '',
                   dt_tolerance_s=0.6)]
    for aux in data.get('auxiliary_cameras') or []:
        from raven_app.third_camera import ThirdCameraConfig
        config = ThirdCameraConfig.from_dict(aux)
        camera = CameraSpec(config.camera_name, str(config.camera_model).upper(),
                            int(config.intrinsics.get('width') or 0), int(config.intrinsics.get('height') or 0),
                            {k: float(v) for k, v in config.intrinsics.items() if k in INTRINSIC_KEYS},
                            _matrix(config.T_lidar_to_cam2_rigid_4x4, config.camera_name).tolist())
        bodies.append(Body(config.camera_name, [camera], sync='photometric',
                           time_offset_s=float(config.time_offset_s or 0.0), dt_tolerance_s=3.0,
                           source=config.source_type))
    topics = data.get('topics') if isinstance(data.get('topics'), dict) else {}
    scanner_name = str(data.get('scanner') or '')
    preset = 'eagle' if 'eagle' in scanner_name.lower() else 'raven' if 'raven' in scanner_name.lower() else 'generic'
    scanner = {'model': scanner_name, 'preset': preset,
               'lidar_topic': topics.get('lidar', SCANNER_PRESETS[preset]['lidar_topic']),
               'imu_topic': topics.get('imu', SCANNER_PRESETS[preset]['imu_topic'])}
    return bodies, scanner


def validate_profile(profile_or_data) -> List[str]:
    """Problems that make a profile unusable (empty list = valid)."""
    try:
        profile = (profile_or_data if isinstance(profile_or_data, RigProfile)
                   else RigProfile.from_dict(profile_or_data))
    except (ValueError, TypeError, KeyError) as exc:
        return [str(exc) or type(exc).__name__]
    errors = []
    names = [c.name for c in profile.cameras]
    if len(set(names)) != len(names):
        errors.append('camera names must be unique')
    if profile.main.cameras[0].name != 'cam0':
        errors.append('the main body must start with cam0')
    if any(c.name in ('cam0', 'cam1') for b in profile.auxiliary for c in b.cameras):
        errors.append('auxiliary cameras cannot use cam0/cam1')
    if len(profile.main.cameras) > 2:
        errors.append('the main body has at most two cameras (cam0, cam1); add others as auxiliary bodies')
    for body in profile.bodies:
        if body.sync not in SYNC_METHODS:
            errors.append(f'{body.body_id}: clock sync must be one of {", ".join(SYNC_METHODS)}')
        for camera in body.cameras:
            if camera.model not in SUPPORTED_MODELS:
                errors.append(f'{camera.name}: unsupported model {camera.model}')
                continue
            if camera.width <= 0 or camera.height <= 0:
                errors.append(f'{camera.name}: image width/height missing')
            try:
                camera.camera_model()
            except (ValueError, TypeError) as exc:
                errors.append(f'{camera.name}: {exc}')
            T = np.asarray(camera.T)
            if abs(np.linalg.det(T[:3, :3]) - 1.0) > 1e-3 or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3):
                errors.append(f'{camera.name}: rotation is not orthonormal')
            if np.linalg.norm(T[:3, 3]) > 3.0:
                errors.append(f'{camera.name}: lever arm {np.linalg.norm(T[:3, 3]):.2f} m is implausible')
    return errors


def load_profile(path) -> RigProfile:
    path = Path(path)
    return RigProfile.from_dict(json.loads(path.read_text(encoding='utf-8')), path=path)


def save_profile(profile: RigProfile, path=None) -> Path:
    """Write ``profile`` (v2 with v1 keys); default location is the user profiles folder."""
    errors = validate_profile(profile)
    if errors:
        raise ValueError('Invalid rig profile: ' + '; '.join(errors))
    if path is None:
        path = user_profiles_dir() / f'{slugify(profile.profile_id)}.json'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding='utf-8')
    profile.path = path
    return path


def profile_from_calibration(data: Dict[str, Any], name: str, *, base: Optional[RigProfile] = None,
                             validated_on: Optional[Dict[str, Any]] = None) -> RigProfile:
    """A new profile from a calibration result (e.g. ``rig_calibration_photometric.json``).

    The result's accepted auxiliary cameras become auxiliary bodies; scanner and
    priors come from ``base`` (the profile the calibration started from) when given.
    """
    data = dict(data)
    aux = [a for a in data.get('auxiliary_cameras') or []
           if ((a.get('calibration_report') or {}).get('extrinsics') or {}).get('status') == 'calibrated']
    data['auxiliary_cameras'] = aux
    profile = RigProfile.from_dict({k: v for k, v in data.items() if k not in ('schema', 'bodies')})
    profile.name, profile.profile_id, profile.is_default = name, slugify(name), False
    if base is not None:
        profile.scanner = dict(base.scanner)
        profile.mount_prior = dict(base.mount_prior)
        for body, base_body in zip(profile.bodies, base.bodies):
            body.sync, body.dt_tolerance_s = base_body.sync, base_body.dt_tolerance_s
            for camera in body.cameras:
                previous = base.camera(camera.name)
                if previous is not None and previous.model == camera.model:
                    camera.valid_radius_px = previous.valid_radius_px
    for key in ('photometric_report', 'quality_status', 'rejection_reason', 'valid', 'calibration_init',
                'dt_sync_seconds', 'dt_sync_reference', 'calibration_source'):
        profile.extra.pop(key, None)
    if validated_on:
        profile.validated_on.append(validated_on)
    return profile


def apply_profile_to_aux_cameras(configs, profile: Optional[RigProfile], log=print):
    """Fill auxiliary camera configs that lack a calibration from the profile's auxiliary bodies.

    A capture supplies only the source (video / folder) of an auxiliary camera; when the
    profile knows a camera of the same name, its model, intrinsics, mount and clock
    offset are used instead of a fresh sample-SfM calibration.
    """
    if profile is None or not profile.auxiliary:
        return list(configs)
    known = {c.camera_name: c for c in profile.auxiliary_camera_configs()}
    result = []
    for config in configs:
        source = known.get(config.camera_name)
        report = config.calibration_report or {}
        calibrated = all((report.get(key) or {}).get('status') in ('sfm_calibrated', 'calibrated', 'verified')
                         for key in ('intrinsics', 'extrinsics'))
        if source is not None and not calibrated:
            config = copy.deepcopy(config)
            config.camera_model = source.camera_model
            config.intrinsics = dict(source.intrinsics)
            config.T_lidar_to_cam2_rigid_4x4 = [list(row) for row in source.T_lidar_to_cam2_rigid_4x4]
            config.time_offset_s = source.time_offset_s
            config.calibration_report = copy.deepcopy(source.calibration_report)
            log(f'[*] {config.camera_name}: model, intrinsics, mount and clock offset from rig profile '
                f'{profile.profile_id}')
        result.append(config)
    return result
