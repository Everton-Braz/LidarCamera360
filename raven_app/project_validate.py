"""Validate a `.lc360` v1 package: a folder with `project.json` that names the files inside it.

One validator for writers and readers. LidarCamera360 calls it before it writes a manifest; any other reader can
vendor a copy of this file and call it before it opens a package. The set of codes is part of the v1
contract (`spec/SPEC.md`); a new code needs a new conformance example under `spec/conformance/`.

Level 1 (schema):     E_JSON, E_SCHEMA, E_VERSION
Level 2 (structural): E_PATH_ESCAPE, E_MISSING_ASSET, E_POSES, and the warning W_NO_ID
                      Level 2 runs only when level 1 found no error.
Warning W_DUP_ID (two folders with one id) needs two packages, so a reader raises it, not this module.

This file depends only on the standard library and `jsonschema`, and finds its schemas next to itself, so the same
file works in both repositories.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import jsonschema

PROJECT_FILE = 'project.json'
SUPPORTED_MAJOR = 1
MAX_POSE_FINDINGS = 20

E_JSON, E_SCHEMA, E_VERSION = 'E_JSON', 'E_SCHEMA', 'E_VERSION'
E_PATH_ESCAPE, E_MISSING_ASSET, E_POSES = 'E_PATH_ESCAPE', 'E_MISSING_ASSET', 'E_POSES'
W_NO_ID, W_DUP_ID = 'W_NO_ID', 'W_DUP_ID'

ERROR_CODES = (E_JSON, E_SCHEMA, E_VERSION, E_PATH_ESCAPE, E_MISSING_ASSET, E_POSES)
WARNING_CODES = (W_NO_ID, W_DUP_ID)
LEVEL_OF = {E_JSON: 1, E_SCHEMA: 1, E_VERSION: 1, E_PATH_ESCAPE: 2, E_MISSING_ASSET: 2, E_POSES: 2,
            W_NO_ID: 2, W_DUP_ID: 2}

_MESSAGES = {
    E_JSON: 'project.json is missing or is not valid JSON',
    E_SCHEMA: 'project.json does not fit the lc360-project schema',
    E_VERSION: 'unsupported package version',
    E_PATH_ESCAPE: 'path is not a relative path inside the package',
    E_MISSING_ASSET: 'listed file does not exist',
    E_POSES: 'camera pose file does not fit lc360-camera-poses-1.0',
    W_NO_ID: 'manifest has no id (an exporter that writes ids will add one)',
    W_DUP_ID: 'another package folder carries the same id',
}

_SINGLE_PATH_KEYS = (('crs', 'transform'), ('capture', 'rig_profile'))


class UnsafePath(ValueError):
    """A manifest path is not a relative path inside the package."""


@dataclass(frozen=True)
class Finding:
    level: int
    code: str
    path: str
    message: str

    def to_dict(self) -> dict:
        return {'level': self.level, 'code': self.code, 'path': self.path, 'message': self.message}

    @property
    def is_error(self) -> bool:
        return self.code.startswith('E_')


@dataclass
class Report:
    level: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.is_error]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.is_error]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set[str]:
        return {f.code for f in self.findings}

    def to_dict(self) -> dict:
        return {'level': self.level, 'findings': [f.to_dict() for f in self.findings]}


def _finding(code: str, path: str, detail: str = '') -> Finding:
    message = _MESSAGES[code] + (f': {detail}' if detail else '')
    return Finding(LEVEL_OF[code], code, path, message)


@lru_cache(maxsize=None)
def schema(name: str) -> dict:
    """Load `lc360-project` or `lc360-camera-poses-1.0` from next to this file (`schema/` or the same folder)."""
    here = Path(__file__).resolve().parent
    for folder in (here / 'schema', here):
        candidate = folder / f'{name}.schema.json'
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding='utf-8'))
    raise FileNotFoundError(f'{name}.schema.json not found next to {Path(__file__).name}')


def safe_path(root, relative) -> Path:
    """Resolve `relative` under `root`, or raise `UnsafePath`.

    Rejects by string test first (leading `/` or `\\`, drive letter, UNC, backslash, `..` segment), then resolves
    symlinks and checks the result is still inside `root` with `is_relative_to` (never a string prefix test, so
    `/pkg-evil` does not match `/pkg`).
    """
    if not isinstance(relative, str) or not relative:
        raise UnsafePath(f'{relative!r} is not a path')
    if relative[0] in '/\\' or re.match(r'^[A-Za-z]:', relative):
        raise UnsafePath(f'{relative!r} is absolute')
    if '\\' in relative:
        raise UnsafePath(f'{relative!r} contains a backslash')
    if '..' in relative.split('/'):
        raise UnsafePath(f'{relative!r} contains ..')
    try:
        base = Path(root).resolve()
        resolved = (base / relative).resolve()
    except (OSError, ValueError) as exc:
        raise UnsafePath(f'{relative!r} cannot be resolved: {exc}') from exc
    if not resolved.is_relative_to(base):
        raise UnsafePath(f'{relative!r} resolves outside the package')
    return resolved


def _dotted(parts) -> str:
    return '.'.join(str(p) for p in parts)


def _level1(manifest) -> list[Finding]:
    version = manifest.get('version') if isinstance(manifest, dict) else None
    if isinstance(version, str):
        m = re.match(r'^(\d+)\.\d+$', version)
        if m and int(m.group(1)) != SUPPORTED_MAJOR:
            return [_finding(E_VERSION, 'version', f'{version!r}; this reader accepts {SUPPORTED_MAJOR}.x')]
    validator = jsonschema.Draft202012Validator(schema('lc360-project'))
    errors = sorted(validator.iter_errors(manifest), key=lambda e: _dotted(e.absolute_path))
    return [_finding(E_SCHEMA, _dotted(e.absolute_path), e.message) for e in errors]


def manifest_paths(manifest: dict) -> list[tuple[str, str]]:
    """Every (manifest location, relative path) the validator checks, in a stable order."""
    found = [(f'assets.{key}', rel) for key, rel in sorted(manifest.get('assets', {}).items())]
    for section, key in _SINGLE_PATH_KEYS:
        rel = (manifest.get(section) or {}).get(key)
        if rel is not None:
            found.append((f'{section}.{key}', rel))
    if manifest.get('quality') is not None:
        found.append(('quality', manifest['quality']))
    return found


def _check_poses(root: Path, manifest: dict, poses_file: Path) -> list[Finding]:
    try:
        poses = json.loads(poses_file.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return [_finding(E_POSES, 'assets.camera_poses', f'cannot read: {exc}')]
    validator = jsonschema.Draft202012Validator(schema('lc360-camera-poses-1.0'))
    errors = sorted(validator.iter_errors(poses), key=lambda e: _dotted(e.absolute_path))
    findings = [_finding(E_POSES, f'assets.camera_poses:{_dotted(e.absolute_path)}', e.message)
                for e in errors[:MAX_POSE_FINDINGS]]
    if len(errors) > MAX_POSE_FINDINGS:
        findings.append(_finding(E_POSES, 'assets.camera_poses', f'{len(errors) - MAX_POSE_FINDINGS} more not shown'))
    images_rel = manifest.get('assets', {}).get('images')
    if images_rel is not None and not errors:
        try:
            images_root = safe_path(root, images_rel)
        except UnsafePath:
            return findings
        bad = [im['image'] for im in poses['images'] if not _inside(images_root, im['image'])]
        findings += [_finding(E_POSES, 'assets.camera_poses:images', f'{name!r} is not under assets.images')
                     for name in bad[:MAX_POSE_FINDINGS]]
    return findings


def _inside(images_root: Path, name: str) -> bool:
    try:
        safe_path(images_root, name)
        return True
    except UnsafePath:
        return False


def _level2(root: Path, manifest: dict) -> list[Finding]:
    findings: list[Finding] = []
    poses_file = None
    for where, rel in manifest_paths(manifest):
        try:
            target = safe_path(root, rel)
        except UnsafePath as exc:
            findings.append(_finding(E_PATH_ESCAPE, where, str(exc)))
            continue
        if not target.exists():
            findings.append(_finding(E_MISSING_ASSET, where, rel))
        elif where == 'assets.camera_poses' and target.is_file():
            poses_file = target
    if poses_file is not None:
        findings += _check_poses(root, manifest, poses_file)
    if 'id' not in manifest:
        findings.append(_finding(W_NO_ID, 'id'))
    return findings


def validate_project(folder, manifest: dict | None = None) -> Report:
    """Validate the package in `folder`. Pass `manifest` to check a dict that has not been written yet."""
    root = Path(folder)
    if manifest is None:
        try:
            manifest = json.loads((root / PROJECT_FILE).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            return Report(0, [_finding(E_JSON, PROJECT_FILE, str(exc))])
    level1 = _level1(manifest)
    if level1:
        return Report(0, level1)
    level2 = _level2(root, manifest)
    return Report(1 if any(f.is_error for f in level2) else 2, level2)
