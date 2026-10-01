import json

import av
import numpy as np

from raven_app import third_camera


def _video(path, gop=12):
    rng = np.random.default_rng(21)
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=12)
        stream.width = stream.height = 128
        stream.pix_fmt = 'yuv420p'
        stream.options = {'g': str(gop), 'bf': '0',
                          'x264-params': f'keyint={gop}:min-keyint={gop}:scenecut=0'}
        for index in range(384):
            pixels = rng.integers(20, 230, (128, 128, 3), dtype=np.uint8)
            if index % 12 != 1:
                pixels[::2] = pixels[1::2]
            frame = av.VideoFrame.from_ndarray(pixels, format='bgr24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_seek_preserves_windows_selected_timestamps_and_pixels(tmp_path):
    source = tmp_path / 'phone.mp4'
    _video(source)
    results = []
    for seek in (False, True):
        destination = tmp_path / str(seek)
        destination.mkdir()
        paths, stamps, _ = third_camera._decode_auxiliary_video_pyav(
            source, destination, 'cam2', 1, 3, 95, seek_sampling=seek)
        results.append((stamps, [path.read_bytes() for path in paths]))
    assert len(results[0][0]) == 32
    assert results[0] == results[1]


def test_long_gops_use_full_decode(tmp_path):
    source = tmp_path / 'long-gop.mp4'
    _video(source, gop=72)
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        assert not third_camera._auxiliary_seek_eligible(container, stream, 1, 3)
        assert float(next(container.decode(stream)).time) == 0


def test_seek_failure_retries_full_pyav(tmp_path, monkeypatch):
    source = tmp_path / 'phone.mp4'
    _video(source)
    monkeypatch.setattr(third_camera, '_auxiliary_seek_eligible', lambda *_: True)

    def unavailable(*_):
        raise third_camera._AuxiliarySeekError('unsupported seek')

    monkeypatch.setattr(third_camera, '_seek_auxiliary_windows', unavailable)
    config = third_camera.ThirdCameraConfig(source_path=str(source), fps=1)
    output = tmp_path / 'output'
    paths, stamps = third_camera.extract_third_camera_frames(config, output)
    assert len(paths) == len(stamps) == 32
    manifest = json.loads((output / 'images/frames.json').read_text())
    assert len(manifest['timestamps']) == 32
