"""
Rig Profile Manager for LiDAR-Camera Calibration.
Lists, resolves and saves hardware rig profiles: bundled ones in ``configs/profiles``
and the user's own in ``%APPDATA%/LidarCamera360/profiles`` (kept across app updates).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

from raven_app.rig_profile import (RigProfile, profile_from_calibration, save_profile, slugify,
                                   user_profiles_dir)


def get_app_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_profiles_dir() -> Path:
    """Bundled profiles shipped with the app."""
    profiles_dir = get_app_root() / "configs" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir


def get_user_profiles_dir() -> Path:
    return user_profiles_dir()


def _profile_dirs() -> List[Path]:
    dirs = [get_profiles_dir()]
    user = get_user_profiles_dir()
    if user.is_dir() and user.resolve() != dirs[0].resolve():
        dirs.append(user)
    return dirs


def _profile_files() -> List[Path]:
    """Profile files, bundled first; a user profile with the same file name replaces a bundled one."""
    files = {}
    for directory in _profile_dirs():
        for path in sorted(directory.glob('*.json')):
            files[path.name] = path
    return list(files.values())


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def list_available_profiles() -> List[Dict[str, str]]:
    """List bundled and user profiles; the default profile comes first."""
    profiles = []
    default = resolve_profile()
    user_dir = get_user_profiles_dir()
    eagle_profile = get_app_root() / EAGLE_PROFILE
    for path in _profile_files() + ([eagle_profile] if eagle_profile.is_file() else []):
        data = _read(path)
        try:
            profile = RigProfile.from_dict(data, path=path)
            cameras = ', '.join(f'{c.name} {c.model}' for c in profile.cameras)
        except (ValueError, TypeError, KeyError):
            cameras = ''
        profiles.append({
            'name': data.get('name', path.stem.replace('_', ' ').title()),
            'profile_id': data.get('profile_id', path.stem),
            'path': str(path), 'is_default': default is not None and path == default,
            'source': 'user' if path.parent == user_dir else 'bundled',
            'schema': int(data.get('schema') or 1), 'cameras': cameras,
        })
    profiles.sort(key=lambda p: not p['is_default'])
    return profiles

DEFAULT_PROFILE_FILE = "raven_x4_montagem_1.json"
EAGLE_PROFILE = Path("configs") / "rig_profile_eagle.json"


def resolve_profile(profile: Optional[str] = None, is_eagle: bool = False) -> Optional[Path]:
    """Resolve the hardware rig profile used as prior and plausibility reference.

    ``profile`` may be a JSON path or a profile id (file stem or ``profile_id``).
    Without one, Eagle captures use the Eagle profile and everything else the profile
    marked ``is_default`` (falling back to the Raven + X4 mount 1 profile).
    """
    files = _profile_files()
    if profile:
        candidate = Path(profile)
        if candidate.suffix.lower() == '.json' and candidate.is_file():
            return candidate
        for p in files:
            if p.stem == profile:
                return p
        for p in files:
            if _read(p).get('profile_id') == profile:
                return p
        eagle = get_app_root() / EAGLE_PROFILE
        if profile == eagle.stem and eagle.is_file():
            return eagle
        raise ValueError(f'Unknown rig profile: {profile}')
    if is_eagle and (get_app_root() / EAGLE_PROFILE).is_file():
        return get_app_root() / EAGLE_PROFILE
    defaults = [p for p in files if _read(p).get('is_default') is True]
    user_defaults = [p for p in defaults if p.parent != get_profiles_dir()]
    if user_defaults:
        return user_defaults[-1]
    preferred = get_profiles_dir() / DEFAULT_PROFILE_FILE
    if preferred in defaults or (not defaults and preferred.is_file()):
        return preferred
    return defaults[0] if defaults else None


def set_default_profile(profile: str) -> Path:
    """Make ``profile`` the default; only user profiles are rewritten (bundled ones stay read-only)."""
    target = resolve_profile(profile)
    user_dir = get_user_profiles_dir()
    for path in _profile_files():
        if path.parent != user_dir:
            continue
        data = _read(path)
        wanted = path == target
        if data and bool(data.get('is_default')) != wanted:
            data['is_default'] = wanted
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    # A bundled target needs no write: it is the default once no user profile claims it.
    return target


def save_profile_from_calibration(data: dict, profile_name: str, base_profile: Optional[str] = None,
                                  validated_on: Optional[dict] = None) -> Path:
    """
    Save a confirmed rig calibration as a named v2 profile that persists for future runs.
    """
    base = None
    if base_profile:
        try:
            from raven_app.rig_profile import load_profile
            base = load_profile(resolve_profile(base_profile))
        except (ValueError, OSError):
            base = None
    profile = profile_from_calibration(data, profile_name, base=base, validated_on=validated_on)
    return save_profile(profile, get_user_profiles_dir() / f'{slugify(profile_name)}.json')
