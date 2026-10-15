import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import av

from raven_app.video import (
    is_equirectangular_video,
    extract_insv_frames,
    has_cached_frame_extraction,
)
from scripts.pipeline_auto_calibrator_and_colorizer import project_equirectangular


class TestEquirectangularSupport(unittest.TestCase):
    def make_equirect_video(self, path, width=128, height=64, num_frames=10):
        with av.open(str(path), 'w', format='mp4') as out:
            stream = out.add_stream('h264', rate=10)
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'
            rng = np.random.default_rng(42)
            for _ in range(num_frames):
                img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(img, format='bgr24')
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)

    def test_is_equirectangular_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'pano.mp4'
            self.make_equirect_video(p, width=128, height=64)
            self.assertTrue(is_equirectangular_video(p))

            not_pano = Path(tmp) / 'square.mp4'
            self.make_equirect_video(not_pano, width=64, height=64)
            self.assertFalse(is_equirectangular_video(not_pano))

    def test_extraction_equirectangular_single_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / 'equirect.mp4'
            out_dir = Path(tmp) / 'output'
            self.make_equirect_video(video_path, width=128, height=64, num_frames=10)

            success = extract_insv_frames(video_path, out_dir, fps=2.0)
            self.assertTrue(success)
            self.assertTrue(has_cached_frame_extraction(video_path, out_dir, fps=2.0))

            manifest_path = out_dir / 'images' / 'frames.json'
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            self.assertEqual(manifest.get('video_mode'), 'equirectangular')
            self.assertEqual(manifest.get('camera_names'), ['cam0'])
            self.assertTrue(len(manifest.get('timestamps', {})) > 0)

            cam0_files = list((out_dir / 'images' / 'cam0').glob('*.jpg'))
            self.assertTrue(len(cam0_files) > 0)

    def test_project_equirectangular_geometry(self):
        # Test key directions:
        # Points: [X, Y, Z]
        # [0, 0, 10]: Forward -> lon=0, lat=0 -> u = 0.5 * W, v = 0.5 * H
        # [10, 0, 0]: Right -> lon=pi/2 -> u = (0.25 + 0.5)*W = 0.75 * W, v = 0.5 * H
        # [-10, 0, 0]: Left -> lon=-pi/2 -> u = (-0.25 + 0.5)*W = 0.25 * W, v = 0.5 * H
        # [0, 10, 0]: Down -> lat=pi/2 -> v = (0.5 + 0.5)*H = 1.0 * H (border)
        # [0, -10, 0]: Up -> lat=-pi/2 -> v = (-0.5 + 0.5)*H = 0.0 * H
        W, H = 3600, 1800
        pts = np.array([
            [0.0, 0.0, 10.0],
            [10.0, 0.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, -10.0, 0.0],
        ], dtype=np.float64)

        u, v, r = project_equirectangular(pts, W, H)
        self.assertTrue(np.all(r > 0))
        self.assertAlmostEqual(u[0], 1800.0, delta=1e-3)
        self.assertAlmostEqual(v[0], 900.0, delta=1e-3)

        self.assertAlmostEqual(u[1], 2700.0, delta=1e-3)
        self.assertAlmostEqual(v[1], 900.0, delta=1e-3)

        self.assertAlmostEqual(u[2], 900.0, delta=1e-3)
        self.assertAlmostEqual(v[2], 900.0, delta=1e-3)

        self.assertAlmostEqual(v[3], 0.0, delta=1e-3)


if __name__ == '__main__':
    unittest.main()
