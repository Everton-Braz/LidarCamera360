"""The v1 package validator: every conformance example, every code, and the path rule."""
import json
import os
import shutil
from pathlib import Path

import pytest

from raven_app.project_validate import (ERROR_CODES, UnsafePath, safe_path, validate_project)

CONFORMANCE = Path(__file__).resolve().parent.parent / 'spec' / 'conformance'
CASES = sorted(p for kind in ('valid', 'invalid') for p in (CONFORMANCE / kind).iterdir() if p.is_dir())

ID = '3f2b8c1e-5d4a-4b7e-9c60-1a2b3c4d5e6f'


def _manifest(**over):
    m = {'schema': 'lc360-project', 'version': '1.0', 'created_by': {'app': 't', 'version': '1'}, 'units': 'm',
         'frame': {'name': 'w', 'up': '+Z'}, 'id': ID, 'assets': {'cloud': 'cloud/c.laz'}}
    m.update(over)
    return m


def _package(tmp_path, manifest=None, files=('cloud/c.laz',)):
    root = tmp_path / 'pkg'
    root.mkdir()
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(b'x')
    if manifest is not None:
        (root / 'project.json').write_text(json.dumps(manifest), encoding='utf-8')
    return root


def _prepare(case: Path, tmp_path: Path, expect: dict) -> Path:
    """Copy an example out of the repo; create the symlink an example asks for at test time."""
    root = tmp_path / case.name
    shutil.copytree(case, root)
    link = expect.get('symlink_outside')
    if link:
        outside = tmp_path / 'outside'
        outside.mkdir()
        (outside / 'colored.laz').write_bytes(b'x')
        try:
            os.symlink(outside, root / link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip('this OS refuses to create a symlink here (on Windows enable Developer Mode)')
    return root


@pytest.mark.parametrize('case', CASES, ids=lambda p: f'{p.parent.name}/{p.name}')
def test_conformance_example(case, tmp_path):
    expect = json.loads((case / 'expect.json').read_text(encoding='utf-8'))
    report = validate_project(_prepare(case, tmp_path, expect))
    assert {f.code for f in report.errors} == set(expect['errors'])
    assert {f.code for f in report.warnings} == set(expect['warnings'])
    assert (case.parent.name == 'valid') == report.ok


def test_every_error_code_has_an_example():
    covered = set()
    for case in CASES:
        covered |= set(json.loads((case / 'expect.json').read_text(encoding='utf-8'))['errors'])
    assert covered == set(ERROR_CODES)


def test_e_json(tmp_path):
    root = _package(tmp_path)
    assert validate_project(root).codes() == {'E_JSON'}
    (root / 'project.json').write_text('{oops', encoding='utf-8')
    report = validate_project(root)
    assert report.codes() == {'E_JSON'} and report.level == 0


def test_e_schema_reports_every_error(tmp_path):
    root = _package(tmp_path, _manifest(units='ft', frame={'name': 'w', 'up': '+X'}, id='nope'))
    report = validate_project(root)
    assert report.codes() == {'E_SCHEMA'}
    assert {f.path for f in report.findings} >= {'units', 'frame.up', 'id'}


def test_missing_version_is_schema_not_version(tmp_path):
    m = _manifest()
    del m['version']
    assert validate_project(_package(tmp_path, m)).codes() == {'E_SCHEMA'}


def test_e_version(tmp_path):
    report = validate_project(_package(tmp_path, _manifest(version='2.1')))
    assert report.codes() == {'E_VERSION'} and report.level == 0


def test_minor_versions_are_accepted(tmp_path):
    assert validate_project(_package(tmp_path, _manifest(version='1.7'))).ok


def test_e_path_escape_on_every_path_field(tmp_path):
    m = _manifest(assets={'cloud': 'cloud/c.laz', 'splat': '../x'}, crs={'transform': '/abs'},
                  capture={'rig_profile': 'a\\b', 'insv': 'D:/not/checked.insv'}, quality='C:/q.json')
    report = validate_project(_package(tmp_path, m))
    assert {f.path for f in report.findings if f.code == 'E_PATH_ESCAPE'} == {
        'assets.splat', 'crs.transform', 'capture.rig_profile', 'quality'}


def test_e_missing_asset_on_cloud_and_on_another_asset(tmp_path):
    m = _manifest(assets={'cloud': 'cloud/gone.laz', 'trajectory': 'trajectory/gone.txt'})
    report = validate_project(_package(tmp_path, m, files=()))
    assert {f.path for f in report.findings if f.code == 'E_MISSING_ASSET'} == {'assets.cloud', 'assets.trajectory'}
    assert report.level == 1


def test_e_poses_unreadable_and_wrong_lens(tmp_path):
    m = _manifest(assets={'cloud': 'cloud/c.laz', 'camera_poses': 'cameras/poses.json'})
    root = _package(tmp_path, m, files=('cloud/c.laz', 'cameras/poses.json'))
    assert validate_project(root).codes() == {'E_POSES'}
    (root / 'cameras' / 'poses.json').write_text('{"lens_model": "PINHOLE", "intrinsics": {}, "images": []}')
    assert validate_project(root).codes() == {'E_POSES'}


def test_missing_pose_file_is_a_missing_asset_not_poses(tmp_path):
    m = _manifest(assets={'cloud': 'cloud/c.laz', 'camera_poses': 'cameras/poses.json'})
    assert validate_project(_package(tmp_path, m)).codes() == {'E_MISSING_ASSET'}


def test_poses_image_outside_is_skipped_without_assets_images(tmp_path):
    poses = {'lens_model': 'THIN_PRISM_FISHEYE',
             'intrinsics': {'c': {'width': 1, 'height': 1, 'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0}},
             'images': [{'image': '../../x.jpg', 'cam': 'c', 'frame': 1, 't': 0, 'position': [0, 0, 0],
                         'q_world_cam': [0, 0, 0, 1], 'forward': [0, 1, 0], 'yaw_deg': 0, 'pitch_deg': 0}]}
    m = _manifest(assets={'cloud': 'cloud/c.laz', 'camera_poses': 'cameras/poses.json'})
    root = _package(tmp_path, m, files=('cloud/c.laz',))
    (root / 'cameras').mkdir()
    (root / 'cameras' / 'poses.json').write_text(json.dumps(poses))
    assert validate_project(root).ok


def test_level_two_does_not_run_after_a_level_one_error(tmp_path):
    root = _package(tmp_path, _manifest(units='ft', assets={'cloud': '../escape'}))
    assert validate_project(root).codes() == {'E_SCHEMA'}


def test_warning_w_no_id_and_level_two(tmp_path):
    m = _manifest()
    del m['id']
    report = validate_project(_package(tmp_path, m))
    assert report.ok and report.codes() == {'W_NO_ID'} and report.level == 2
    assert [f.level for f in report.findings] == [2]


def test_manifest_argument_is_checked_without_a_file(tmp_path):
    root = _package(tmp_path)
    assert validate_project(root, _manifest()).ok


def test_unknown_keys_are_ignored(tmp_path):
    assert validate_project(_package(tmp_path, _manifest(brand_new={'a': [1]}))).ok


@pytest.mark.parametrize('bad', ['/etc/x', '\\x', 'C:/x', 'C:x', '//server/share/x', 'a\\b', '../x', 'a/../../x',
                                 'a/..', '', None, 5])
def test_safe_path_rejects(tmp_path, bad):
    with pytest.raises(UnsafePath):
        safe_path(tmp_path, bad)


def test_safe_path_accepts_nested_and_nonexistent(tmp_path):
    assert safe_path(tmp_path, 'a/b/c.txt') == (tmp_path / 'a' / 'b' / 'c.txt').resolve()


def test_safe_path_is_not_a_string_prefix_test(tmp_path):
    (tmp_path / 'pkg').mkdir()
    (tmp_path / 'pkg-evil').mkdir()
    (tmp_path / 'pkg-evil' / 'f').write_bytes(b'x')
    with pytest.raises(UnsafePath):
        safe_path(tmp_path / 'pkg', '../pkg-evil/f')
    link = tmp_path / 'pkg' / 'link'
    try:
        os.symlink(tmp_path / 'pkg-evil', link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('this OS refuses to create a symlink here')
    with pytest.raises(UnsafePath):
        safe_path(tmp_path / 'pkg', 'link/f')
