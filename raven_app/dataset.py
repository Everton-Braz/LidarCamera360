"""Canonical dataset paths and read-only compatibility with older captures."""
from pathlib import Path
import json
from raven_app.config import get_app_root

RECONSTRUCTION_COLORS = 'reconstruction_colorized'
TRAJECTORY_COLORS = 'trajectory_colorized'


def trajectory_path(dataset):
    directory = Path(dataset) / 'slam_out' / 'result'
    preferred = directory / 'trajectory.txt'
    if preferred.is_file():
        return preferred
    legacy = directory / 'Raven_3DMakerPro_Scan.txt'
    if legacy.is_file():
        return legacy
    return preferred


def calibration_path(dataset=None, explicit=None):
    if explicit is not None:
        return Path(explicit)
    if dataset is not None:
        for name in ('calibration.json', 'rig_profile.json', 'calibracao_rigida_auto.json',
                     'calibracao_rigida_recalibrada.json'):
            candidate = Path(dataset) / name
            if candidate.is_file():
                return candidate
    return get_app_root() / 'configs' / 'rig_profile.json'


def load_profile(path=None):
    base = json.loads(calibration_path().read_text(encoding='utf-8'))
    if path is not None and Path(path).resolve() != calibration_path().resolve():
        custom = json.loads(Path(path).read_text(encoding='utf-8'))
        for key, value in custom.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
    return base


def normalize_method(method):
    return {'sfm': 'reconstruction', 'direct': 'trajectory'}.get(method, method)
