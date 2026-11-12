import json

from raven_app.timelapse_calibration import VERSION
from raven_app.workflow import timelapse_alignment_dt


def _dataset(tmp_path, time_source='insv_timelapse', **alignment):
    (tmp_path / 'images').mkdir(parents=True)
    (tmp_path / 'images' / 'frames.json').write_text(
        json.dumps({'time_source': time_source, 'timestamps': {}}), encoding='utf-8')
    data = {'dt_sync_seconds': 2.4818, 'frame_time_source': 'insv_timelapse',
            'quality_status': 'accepted', 'timelapse_calibration_version': VERSION}
    data.update(alignment)
    (tmp_path / 'colmap_to_lidar_alignment.json').write_text(json.dumps(data), encoding='utf-8')
    return tmp_path


def test_an_accepted_timelapse_alignment_gives_its_clock_offset(tmp_path):
    assert timelapse_alignment_dt(_dataset(tmp_path)) == 2.4818


def test_the_alignment_can_be_passed_in_directly(tmp_path):
    root = _dataset(tmp_path)
    alignment = json.loads((root / 'colmap_to_lidar_alignment.json').read_text(encoding='utf-8'))
    alignment['dt_sync_seconds'] = 1.5
    assert timelapse_alignment_dt(root, alignment) == 1.5


def test_video_frames_keep_the_requested_offset(tmp_path):
    root = _dataset(tmp_path, time_source='insv', frame_time_source='insv')
    assert timelapse_alignment_dt(root) is None


def test_a_stale_manifest_does_not_take_the_timelapse_offset(tmp_path):
    assert timelapse_alignment_dt(_dataset(tmp_path, time_source='insv')) is None


def test_a_rejected_or_outdated_alignment_is_ignored(tmp_path):
    assert timelapse_alignment_dt(_dataset(tmp_path / 'a', quality_status='rejected')) is None
    assert timelapse_alignment_dt(_dataset(tmp_path / 'b', timelapse_calibration_version=VERSION - 1)) is None
    assert timelapse_alignment_dt(_dataset(tmp_path / 'c', valid=False)) is None


def test_missing_files_give_none(tmp_path):
    assert timelapse_alignment_dt(tmp_path) is None
