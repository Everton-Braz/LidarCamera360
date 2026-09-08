import io
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np
from raven_app.bag_io import cloud_payload, point_array, image_payload
from raven_app.cli import parse
from scripts.pipeline_auto_calibrator_and_colorizer import load_pcd


class BagAdapterTests(unittest.TestCase):
    def cloud(self, endian='<'):
        fields=[NS(name=n,datatype=t,offset=o,count=1) for n,t,o in
                [('x',7,0),('y',7,4),('z',7,8),('intensity',7,12),('timestamp',8,16)]]
        data=struct.pack(endian+'4fd',1,2,3,4,1000.)+b'padding!'+struct.pack(endian+'4fd',4,5,6,7,1000.1)+b'padding!'
        return NS(fields=fields,is_bigendian=endian=='>',data=data,point_step=24,row_step=32,width=1,height=2)

    def test_strided_and_big_endian_points(self):
        for endian in ('<','>'):
            data,count=cloud_payload(self.cloud(endian))
            points=np.frombuffer(data,dtype='<f4').reshape(-1,5)
            self.assertEqual(count,2)
            np.testing.assert_allclose(points,[[1,2,3,4,0],[4,5,6,7,100]],atol=.001)

    def test_missing_timing_rejected(self):
        with self.assertRaisesRegex(ValueError,'lacks'):
            cloud_payload(self.cloud(),time_field='time')

    def test_rgb_image_is_lossless_bgr(self):
        import cv2
        msg=NS(encoding='rgb8',step=6,height=1,width=2,data=bytes([255,0,0,0,0,255]))
        decoded=cv2.imdecode(np.frombuffer(image_payload(msg),np.uint8),cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(decoded,[[[0,0,255],[255,0,0]]])

    def test_headless_never_defaults_to_gui(self):
        with self.assertRaises(SystemExit) as ctx:parse(['--headless'])
        self.assertEqual(ctx.exception.code,2)

    def test_bad_fps_rejected(self):
        for v in ('0','-1','nan','inf'):
            with self.assertRaises(SystemExit):parse(['colorize','--dataset','.','--fps',v])


class PcdTests(unittest.TestCase):
    def test_native_xyzinormal_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cloud.pcd'
            p.write_bytes(b'FIELDS x y z intensity normal_x normal_y normal_z curvature\nSIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\nPOINTS 2\nDATA binary\n'+struct.pack('<16f',1,2,3,4,0,0,0,99,5,6,7,8,0,0,0,100))
            np.testing.assert_array_equal(load_pcd(p),[[1,2,3],[5,6,7]])

    def test_truncated_header_and_payload_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'cloud.pcd'
            p.write_bytes(b'POINTS 3\n')
            with self.assertRaisesRegex(ValueError,'Truncated'):load_pcd(p)
            p.write_bytes(b'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 3\nDATA binary\n')
            with self.assertRaisesRegex(ValueError,'payload'):load_pcd(p)



class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_fluent_main_window_creation(self):
        from raven_app.gui import create_main_window
        win = create_main_window()
        self.assertIn('RavenCalibrator', win.windowTitle())
        views = [
            win.workflow_view, win.slam_view, win.colorize_view,
            win.inspect_view, win.calib_view, win.doctor_view, win.settings_view
        ]
        for v in views:
            win.switchTo(v)
            self.assertTrue(len(v.objectName()) > 0)
        win.inspect_view.apply_to_slam_requested.emit('/test/path.bag', '/custom_lidar', '/custom_imu', '/custom_cam')
        self.assertEqual(win.workflow_view.bag_input.text(), '/test/path.bag')
        self.assertEqual(win.workflow_view.lidar_topic.text(), '/custom_lidar')
        self.assertEqual(win.workflow_view.imu_topic.text(), '/custom_imu')
        win.close()


class WorkflowTests(unittest.TestCase):
    def test_workflow_cli_parsing(self):
        args = parse([
            'workflow',
            '--bag', 'test.bag',
            '--insv', 'test.insv',
            '--output', 'test_out',
            '--export-ply',
            '--export-pcd',
            '--export-colmap'
        ])
        self.assertEqual(args.command, 'workflow')
        self.assertTrue(args.export_ply)
        self.assertTrue(args.export_pcd)
        self.assertTrue(args.export_colmap)

    def test_workflow_cli_recalibrate_and_spirula(self):
        args = parse([
            'workflow',
            '--bag', 'test.bag',
            '--insv', 'test.insv',
            '--output', 'test_out',
            '--recalibrate',
            '--run-spirula'
        ])
        self.assertTrue(args.recalibrate)
        self.assertTrue(args.run_spirula)

        args_no = parse([
            'workflow',
            '--bag', 'test.bag',
            '--insv', 'test.insv',
            '--output', 'test_out',
            '--no-recalibrate'
        ])
        self.assertFalse(args_no.recalibrate)

    def test_transform_colmap_to_metric(self):
        from scipy.spatial.transform import Rotation as Rot
        from scripts.pipeline_auto_calibrator_and_colorizer import transform_colmap_to_metric
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src_sparse"
            dst = Path(tmp) / "dst_sparse"
            src.mkdir(parents=True)

            # 1. Synthetic cameras.bin (OPENCV_FISHEYE model_id=9)
            with open(src / "cameras.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<iiQQ", 1, 9, 3840, 3840))
                f.write(struct.pack("<8d", 1000.0, 1000.0, 1920.0, 1920.0, 0.1, -0.01, 0.001, -0.0001))

            # 2. Synthetic images.bin (1 camera at [0, 0, 5], pointing down -Z in world coords)
            with open(src / "images.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))  # qw, qx, qy, qz (R = I)
                f.write(struct.pack("<3d", 0.0, 0.0, -5.0))       # tx, ty, tz -> C = [0, 0, 5]
                f.write(struct.pack("<I", 1))
                f.write(b"cam0/frame_000001.jpg\x00")
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<2d", 1920.0, 1920.0))
                f.write(struct.pack("<Q", 1))

            # 3. Synthetic points3D.bin (1 point at [0, 0, 5])
            with open(src / "points3D.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<3d", 0.0, 0.0, 5.0))
                f.write(struct.pack("<3B", 255, 128, 64))
                f.write(struct.pack("<d", 0.5))
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<2I", 1, 0))

            # Apply Sim(3): scale=2.0, R=RotZ(90 deg), t=[10, 20, 30]
            s_sim = 2.0
            R_sim = Rot.from_euler('z', 90, degrees=True).as_matrix()
            t_sim = np.array([10.0, 20.0, 30.0], dtype=np.float64)

            transform_colmap_to_metric(src, dst, s_sim, R_sim, t_sim)

            # Verify files created
            self.assertTrue((dst / "cameras.txt").is_file())
            self.assertTrue((dst / "cameras.bin").is_file())
            self.assertTrue((dst / "images.txt").is_file())
            self.assertTrue((dst / "images.bin").is_file())
            self.assertTrue((dst / "points3D.txt").is_file())
            self.assertTrue((dst / "points3D.bin").is_file())
            self.assertTrue((dst / "points3D.ply").is_file())

            # Verify image transformation: C' = s * R * C + t
            with open(dst / "images.bin", "rb") as f:
                num_imgs = struct.unpack("<Q", f.read(8))[0]
                self.assertEqual(num_imgs, 1)
                _ = struct.unpack("<I", f.read(4))[0]
                qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
                tx, ty, tz = struct.unpack("<3d", f.read(24))
                R_cw_m = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
                C_metric = -R_cw_m.T @ np.array([tx, ty, tz])
                expected_C = s_sim * (R_sim @ np.array([0.0, 0.0, 5.0])) + t_sim
                np.testing.assert_allclose(C_metric, expected_C, atol=1e-5)

            # Verify point transformation: P' = s * R * P + t, err' = s * err
            with open(dst / "points3D.bin", "rb") as f:
                num_pts = struct.unpack("<Q", f.read(8))[0]
                self.assertEqual(num_pts, 1)
                pid = struct.unpack("<Q", f.read(8))[0]
                xyz = struct.unpack("<3d", f.read(24))
                rgb = struct.unpack("<3B", f.read(3))
                err = struct.unpack("<d", f.read(8))[0]
                expected_P = s_sim * (R_sim @ np.array([0.0, 0.0, 5.0])) + t_sim
                np.testing.assert_allclose(xyz, expected_P, atol=1e-5)
                self.assertEqual(rgb, (255, 128, 64))
                self.assertAlmostEqual(err, 1.0, places=5)

    def test_colmap_3dgs_export_from_spirula(self):
        import json
        from raven_app.workflow import export_colmap_3dgs
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dataset = tmp / "dataset"
            sparse_0 = dataset / "sparse" / "0"
            sparse_0.mkdir(parents=True)

            with open(sparse_0 / "cameras.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<iiQQ", 1, 9, 3840, 3840))
                f.write(struct.pack("<8d", 1000.0, 1000.0, 1920.0, 1920.0, 0.1, -0.01, 0.001, -0.0001))

            with open(sparse_0 / "images.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
                f.write(struct.pack("<3d", 0.0, 0.0, -1.0))
                f.write(struct.pack("<I", 1))
                f.write(b"cam0/frame_000001.jpg\x00")
                f.write(struct.pack("<Q", 0))

            with open(sparse_0 / "points3D.bin", "wb") as f:
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<3d", 0.0, 0.0, 1.0))
                f.write(struct.pack("<3B", 200, 100, 50))
                f.write(struct.pack("<d", 0.2))
                f.write(struct.pack("<Q", 0))

            # Alignment json
            align_data = {
                "scale": 2.5,
                "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "t": [5.0, 10.0, 15.0]
            }
            with open(dataset / "colmap_to_lidar_alignment.json", "w", encoding="utf-8") as f:
                json.dump(align_data, f)

            for cam in ("cam0", "cam1"):
                d = dataset / "images" / cam
                d.mkdir(parents=True)
                (d / "frame_000001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")

            root = Path(__file__).resolve().parents[1]
            calib = root / "calibracao_rigida_raven_insta360.json"
            out = export_colmap_3dgs(dataset, calib, fps=1.0, dt_sync=0.0)

            self.assertTrue((out / "sparse" / "0" / "cameras.txt").is_file())
            self.assertTrue((out / "sparse" / "0" / "images.txt").is_file())
            self.assertTrue((out / "sparse" / "0" / "points3D.txt").is_file())
            self.assertTrue((out / "points3D.ply").is_file())


if __name__=='__main__':unittest.main()


