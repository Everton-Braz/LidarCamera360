import unittest, tempfile, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
import av
from raven_app.video import (extract_insv_frames_pyav, frame_time,
                             has_cached_frame_extraction, _timelapse_times,
                             video_preview_timeline, extract_video_preview_frame)
from raven_app.vulkan_engine import colorize_views, get_vulkan_bin
from raven_app import vulkan_engine
from raven_app.photometric.model import apply_rgb
from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism

class VideoTests(unittest.TestCase):
    def make_video(self, path, tracks=2):
        with av.open(str(path), 'w', format='matroska') as out:
            streams = [out.add_stream('ffv1', rate=10) for _ in range(tracks)]
            for stream in streams:
                stream.width=stream.height=64
                stream.pix_fmt='yuv420p'
            rng=np.random.default_rng(1)
            image=rng.integers(0,256,(64,64,3),dtype=np.uint8)
            for i in range(23):
                for stream in streams:
                    arr=image if i%10==2 else cv2.GaussianBlur(image,(15,15),5)
                    frame=av.VideoFrame.from_ndarray(arr,format='bgr24')
                    for packet in stream.encode(frame):out.mux(packet)
            for stream in streams:
                for packet in stream.encode():out.mux(packet)
    def test_sharp_pairs_cadence_pts_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'test.insv'; self.make_video(source)
            self.assertTrue(extract_insv_frames_pyav(source,root/'out',fps=1))
            self.assertTrue(has_cached_frame_extraction(source, root/'out', fps=1))
            self.assertFalse(has_cached_frame_extraction(source, root/'out', fps=2))
            images=root/'out/images'
            self.assertEqual(len(list(images.glob('cam0/*.jpg'))),3)
            self.assertAlmostEqual(frame_time(root/'out','cam0/frame_000001.jpg'),.2)
            self.assertEqual(frame_time(root/'out','cam0/frame_000001.jpg'),frame_time(root/'out','cam1/frame_000001.jpg'))
            (images/'cam1/frame_000002.jpg').unlink()
            self.assertTrue(extract_insv_frames_pyav(source,root/'out',fps=1))
            self.assertTrue((images/'cam1/frame_000002.jpg').is_file())
            self.assertTrue(extract_insv_frames_pyav(source,root/'out',fps=2))
            self.assertTrue(has_cached_frame_extraction(source, root/'out', fps=2))
            self.assertEqual(len(list(images.glob('cam0/*.jpg'))),5)
    def test_single_track_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.make_video(root/'test.insv',1)
            self.assertFalse(extract_insv_frames_pyav(root/'test.insv',root/'out'))
            self.assertFalse((root/'out/images/frames.json').exists())

    def test_preview_extracts_only_requested_timeline_slots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'preview.insv'
            self.make_video(source)
            self.assertEqual(video_preview_timeline(source, fps=1), (0, 1, 2))
            cache = root / 'preview-cache'
            self.assertTrue(extract_video_preview_frame(source, cache, 0, fps=1,
                                                        sharp_window=1, decoder='cpu'))
            self.assertTrue(extract_video_preview_frame(source, cache, 2, fps=1,
                                                        sharp_window=1, decoder='cpu'))
            for camera in ('cam0', 'cam1'):
                files = sorted((cache / 'images' / camera).glob('*.jpg'))
                self.assertEqual([path.name for path in files],
                                 ['frame_000001.jpg', 'frame_000003.jpg'])
            self.assertFalse((cache / 'images' / 'frames.json').exists())
    def test_legacy_one_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(frame_time(tmp,'cam1/frame_000003.jpg',2),1)

    def test_timelapse_uses_capture_time_for_selection_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'test.insv'; self.make_video(source)
            capture=np.arange(24,dtype=float)*2.0 + .3
            with patch('raven_app.video._timelapse_times',return_value=capture):
                timeline = video_preview_timeline(source, fps=1)
                self.assertEqual(len(timeline), 23)
                self.assertEqual(timeline[-1], 44)
                self.assertTrue(extract_insv_frames_pyav(source,root/'out',fps=1,sharp_window=1))
            manifest=json.loads((root/'out/images/frames.json').read_text())
            self.assertEqual(manifest['time_source'],'insv_timelapse')
            self.assertEqual(len(list((root/'out/images/cam0').glob('*.jpg'))),23)
            self.assertAlmostEqual(frame_time(root/'out','cam0/frame_000023.jpg'),44.3)

    def test_timelapse_count_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'test.insv'; self.make_video(source)
            with patch('raven_app.video._timelapse_times',return_value=np.arange(21,dtype=float)*2):
                self.assertFalse(extract_insv_frames_pyav(source,root/'out',fps=1))
            self.assertFalse((root/'out/images/frames.json').exists())

@unittest.skipUnless(get_vulkan_bin().is_file(),'Native Vulkan build unavailable')
class VulkanTests(unittest.TestCase):
    def test_gpu_colorizes_mixed_image_dimensions_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = np.array([[0.0, 0.0, 1.0]])
            views = []
            for index, (width, height) in enumerate(((64, 64), (96, 72))):
                path = root / f'mixed_{index}.png'
                rgb = np.full((height, width, 3), [40, 140, 210], np.uint8)
                cv2.imwrite(str(path), rgb[:, :, ::-1])
                params = [width * .28, height * .28, width / 2, height / 2] + [0] * 8
                views.append((path, np.eye(3), np.zeros(3), params))

            native_codes = []
            run_native = vulkan_engine.run_hidden_stream
            def capture_native(*args, **kwargs):
                code = run_native(*args, **kwargs)
                native_codes.append(code)
                return code

            with patch.object(vulkan_engine, 'run_hidden_stream', side_effect=capture_native):
                colors = colorize_views(points, views, root)
            self.assertEqual(native_codes, [0], 'mixed sizes must succeed on Vulkan, not CPU fallback')
            np.testing.assert_allclose(colors[0], [40, 140, 210], atol=1)

    def test_photometric_rvc3_matches_cpu_and_identity(self):
        """RVC3 corrects each observation in linear sRGB and keeps zero-model bytes exact."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'
            camera_dir = images / 'cam0'
            camera_dir.mkdir(parents=True)
            rgb0 = np.array([118, 96, 72], dtype=np.uint8)
            path = camera_dir / 'frame_000001.png'
            # 3840 px preserves the model's physical 1620 px normalization.
            cv2.imwrite(str(path), np.broadcast_to(rgb0[::-1], (3840, 3840, 3)).copy())
            params = [1080, 1080, 1920, 1920] + [0] * 8
            radii = (0.0, 760.0, 1320.0)
            points = np.array([[np.tan(r / params[0]), 0.0, 1.0] for r in radii])
            views = [(path, np.eye(3), np.zeros(3), params)]
            coeff = {'log_gain': [0.18, -0.09, 0.04],
                     'vignette': [-0.24, 0.12, 0.01]}
            model = {'cam0/frame_000001.png': coeff}

            baseline = colorize_views(points, views, root)
            identity = colorize_views(points, views, root, images_dir=images,
                                      photometric={})
            corrected = colorize_views(points, views, root, images_dir=images,
                                       photometric=model)
            self.assertIsNotNone(baseline)
            self.assertIsNotNone(identity)
            self.assertIsNotNone(corrected)
            np.testing.assert_array_equal(identity, baseline)
            u, v, _ = project_thin_prism(points, params)
            expected = np.stack([apply_rgb(rgb0, x, y, params, coeff)
                                 for x, y in zip(u, v)])
            np.testing.assert_allclose(corrected, expected, atol=1)

    def test_occlusion_consensus_and_large_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            origin=np.array([600000.,7500000.,100.])
            points=np.array([[0,0,1],[0,0,2],[0,0,-1],[.2,0,1]])+origin
            views=[]
            for i,color in enumerate(([30,80,120],[30,80,120],[240,10,10])):
                path=root/f'{i}.png';cv2.imwrite(str(path),np.full((64,64,3),color[::-1],np.uint8))
                views.append((path,np.eye(3),origin,[18,18,32,32,0,0,0,0,0,0,0,0]))
            rgb=colorize_views(points,views,root)
            self.assertIsNotNone(rgb)
            np.testing.assert_allclose(rgb[[0,3]],[[30,80,120],[30,80,120]],atol=1)
            np.testing.assert_array_equal(rgb[[1,2]],[[180]*3,[180]*3])
    def test_visibility_rejects_shallow_occluder_and_neighbor_hole(self):
        """A 5.2 m sample must not leak through a nearby 5 m foreground cell."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Keep the foreground and background projections in adjacent z-buffer
            # cells (the latter exercises the 3x3 sparse-hole lookup).
            points = np.array([[0.0, 0.0, 5.0], [0.0041, 0.0, 5.2]])
            image = np.zeros((64, 64, 3), np.uint8)
            image[:] = [30, 80, 120][::-1]  # RGB [30, 80, 120] after decode
            path = root / 'view.png'
            cv2.imwrite(str(path), image)
            params = [18, 18, 32, 32] + [0] * 8
            rgb = colorize_views(points, [(path, np.eye(3), np.zeros(3), params)], root)
            self.assertIsNotNone(rgb)
            np.testing.assert_allclose(rgb[0], [30, 80, 120], atol=1)
            np.testing.assert_allclose(rgb[1], [180, 180, 180], atol=1)
    def test_resolve_two_disagreeing_frames_keeps_best_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = np.array([[0.0, 0.0, 1.0]])
            colors = ([255, 0, 0], [0, 0, 255])
            views = []
            for i, color in enumerate(colors):
                path = root / f'frame_{i}.png'
                cv2.imwrite(str(path), np.full((64, 64, 3), color[::-1], np.uint8))
                # The second camera is closer, so its score is higher.
                center = np.array([0.0, 0.0, 0.2 if i else 0.0])
                views.append((path, np.eye(3), center, [18, 18, 32, 32] + [0] * 8))
            rgb = colorize_views(points, views, root)
            self.assertIsNotNone(rgb)
            np.testing.assert_allclose(rgb[0], [0, 0, 255], atol=1)

    def test_resolve_three_without_median_inlier_keeps_best_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            points = np.array([[0.0, 0.0, 1.0]])
            colors = ([255, 0, 0], [0, 255, 0], [0, 0, 255])
            views = []
            centers = (0.0, 0.2, -0.2)
            for i, (color, z_center) in enumerate(zip(colors, centers)):
                path = root / f'orthogonal_{i}.png'
                cv2.imwrite(str(path), np.full((64, 64, 3), color[::-1], np.uint8))
                views.append((path, np.eye(3), np.array([0.0, 0.0, z_center]),
                              [18, 18, 32, 32] + [0] * 8))
            rgb = colorize_views(points, views, root)
            self.assertIsNotNone(rgb)
            np.testing.assert_allclose(rgb[0], [0, 255, 0], atol=1)
    def test_missing_frame_returns_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=colorize_views(np.array([[0.,0.,1.]]),[(Path(tmp)/'missing.jpg',np.eye(3),np.zeros(3),[18,18,32,32]+[0]*8)],tmp)
            self.assertIsNone(result)
    def test_gpu_projection_with_distortion(self):
        # Coordinate-encoded texture detects projection errors across 100k points.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); size=3840
            xx=np.arange(size,dtype=np.float32)[None,:];yy=xx.T
            image=np.zeros((size,size,3),np.uint8)
            image[:,:,0]=np.floor(xx/16);image[:,:,1]=np.floor(yy/16);image[:,:,2]=80
            path=root/'gradient.png';cv2.imwrite(str(path),image[:,:,::-1])
            rng=np.random.default_rng(7);points=np.column_stack((rng.uniform(-1,1,100000),rng.uniform(-1,1,100000),np.ones(100000)))
            params=[1080,1090,1920,1920,.05,-.01,.003,-.002,.001,-.0001,.002,-.001]
            rgb=colorize_views(points,[(path,np.eye(3),np.zeros(3),params)],root)
            self.assertIsNotNone(rgb)
            u,v,_=project_thin_prism(points,params)
            expected=np.column_stack((np.floor(u/16),np.floor(v/16),np.full(len(u),80)))
            self.assertLess(np.abs(rgb.astype(float)-expected).max(),2)

if __name__=='__main__':unittest.main()
