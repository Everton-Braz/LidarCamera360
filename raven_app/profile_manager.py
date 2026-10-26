"""
Rig Profile Manager for LiDAR-Camera Calibration.
Supports saving, loading, listing, and exporting user hardware rig profiles.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def get_app_root() -> Path:
    return Path(__file__).resolve().parent.parent

def get_profiles_dir() -> Path:
    profiles_dir = get_app_root() / "configs" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir

def list_available_profiles() -> List[Dict[str, str]]:
    """List all saved profiles in configs/profiles plus built-in presets."""
    profiles = []
    pdir = get_profiles_dir()
    
    # Default standard profile
    default_file = pdir / "raven_x4_montagem_1.json"
    if default_file.is_file():
        try:
            data = json.loads(default_file.read_text(encoding='utf-8'))
            name = data.get('name', 'Raven Lidar + Insta360 X4 (Montagem 1)')
            profiles.append({'name': name, 'path': str(default_file), 'is_default': True})
        except Exception:
            profiles.append({'name': 'Raven Lidar + Insta360 X4 (Montagem 1)', 'path': str(default_file), 'is_default': True})
            
    # Scan other user profiles in configs/profiles
    for p in sorted(pdir.glob('*.json')):
        if p.name == 'raven_x4_montagem_1.json':
            continue
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            name = data.get('name', p.stem.replace('_', ' ').title())
            profiles.append({'name': name, 'path': str(p), 'is_default': False})
        except Exception:
            profiles.append({'name': p.stem, 'path': str(p), 'is_default': False})
            
    # Legacy built-ins if present
    eagle_profile = get_app_root() / "configs" / "rig_profile_eagle.json"
    if eagle_profile.is_file():
        profiles.append({'name': 'Eagle (Livox + Insta360)', 'path': str(eagle_profile), 'is_default': False})
        
    return profiles

DEFAULT_PROFILE_FILE = "raven_x4_montagem_1.json"
EAGLE_PROFILE = Path("configs") / "rig_profile_eagle.json"


def resolve_profile(profile: Optional[str] = None, is_eagle: bool = False) -> Optional[Path]:
    """Resolve the hardware rig profile used as prior and plausibility reference.

    ``profile`` may be a JSON path or a profile id (file stem or ``profile_id``).
    Without one, Eagle captures use the Eagle profile and everything else the profile
    marked ``is_default`` (falling back to the Raven + X4 mount 1 profile).
    """
    pdir = get_profiles_dir()
    if profile:
        candidate = Path(profile)
        if candidate.suffix.lower() == '.json' and candidate.is_file():
            return candidate
        by_stem = pdir / f'{profile}.json'
        if by_stem.is_file():
            return by_stem
        for p in sorted(pdir.glob('*.json')):
            try:
                if json.loads(p.read_text(encoding='utf-8')).get('profile_id') == profile:
                    return p
            except (OSError, ValueError):
                continue
        raise ValueError(f'Unknown rig profile: {profile}')
    if is_eagle and (get_app_root() / EAGLE_PROFILE).is_file():
        return get_app_root() / EAGLE_PROFILE
    defaults = []
    for p in sorted(pdir.glob('*.json')):
        try:
            if json.loads(p.read_text(encoding='utf-8')).get('is_default') is True:
                defaults.append(p)
        except (OSError, ValueError):
            continue
    preferred = pdir / DEFAULT_PROFILE_FILE
    if preferred in defaults or (not defaults and preferred.is_file()):
        return preferred
    return defaults[0] if defaults else None


def save_profile_from_calibration(data: dict, profile_name: str) -> Path:
    """
    Save a confirmed rig calibration as a named profile that persists for future runs.
    """
    pdir = get_profiles_dir()
    safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in profile_name).strip()
    slug = safe_name.replace(' ', '_').lower()
    out_path = pdir / f'{slug}.json'
    
    data_copy = dict(data)
    data_copy['name'] = profile_name
    data_copy['profile_id'] = slug
    data_copy['is_default'] = False
    
    out_path.write_text(json.dumps(data_copy, indent=2), encoding='utf-8')
    return out_path
