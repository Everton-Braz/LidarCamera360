import unittest, tempfile, json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
import av
from raven_app.video import extract_insv_frames_pyav, frame_time
from raven_app.vulkan_engine import colorize_views, get_vulkan_bin
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
            images=root/'out/images'
            self.assertEqual(len(list(images.glob('cam0/*.jpg'))),3)
            self.assertAlmostEqual(frame_time(root/'out','cam0/frame_000001.jpg'),.2)
            self.assertEqual(frame_time(root/'out','cam0/frame_000001.jpg'),frame_time(root/'out','cam1/frame_000001.jpg'))
            (images/'cam1/frame_000002.jpg').unlink()
            self.assertTrue(extract_insv_frames_pyav(source,root/'out',fps=1))
            self.assertTrue((images/'cam1/frame_000002.jpg').is_file())
            self.assertTrue(extract_insv_frames_pyav(source,root/'out',fps=2))
            self.assertEqual(len(list(images.glob('cam0/*.jpg'))),5)
    def test_single_track_fails_without_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.make_video(root/'test.insv',1)
            self.assertFalse(extract_insv_frames_pyav(root/'test.insv',root/'out'))
            self.assertFalse((root/'out/images/frames.json').exists())
    def test_legacy_one_based(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(frame_time(tmp,'cam1/frame_000003.jpg',2),1)

@unittest.skipUnless(get_vulkan_bin().is_file(),'Native Vulkan build unavailable')
class VulkanTests(unittest.TestCase):
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
