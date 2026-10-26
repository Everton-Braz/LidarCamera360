import cv2
import numpy as np

from raven_app.workflow import gyro_telemetry_source


def _video(path, frames):
    # OpenCV only writes known extensions; an INSV is an MP4-family container.
    staging = path.with_name(path.stem + '_staging.mp4')
    writer = cv2.VideoWriter(str(staging), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 32))
    for _ in range(frames):
        writer.write(np.zeros((32, 64, 3), np.uint8))
    writer.release()
    staging.replace(path)
    return path


def test_exported_mp4_uses_the_sibling_insv_with_the_same_duration(tmp_path):
    mp4 = _video(tmp_path / 'VID_20260821_105436_00_269.mp4', 40)
    insv = _video(tmp_path / 'VID_20260821_105436_00_269.insv', 40)
    assert gyro_telemetry_source(mp4) == insv
    assert gyro_telemetry_source(insv) == insv


def test_trimmed_export_keeps_the_mp4(tmp_path):
    mp4 = _video(tmp_path / 'VID_x.mp4', 20)       # 2 s
    _video(tmp_path / 'VID_x.insv', 40)            # 4 s: different time base
    assert gyro_telemetry_source(mp4) == mp4


def test_mp4_without_sibling_is_unchanged(tmp_path):
    mp4 = _video(tmp_path / 'clip.mp4', 20)
    assert gyro_telemetry_source(mp4) == mp4
