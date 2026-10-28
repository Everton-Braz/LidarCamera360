import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
from raven_app.third_camera import ThirdCameraConfig
from raven_app.cli import parse
from scripts.pipeline_auto_calibrator_and_colorizer import recalibrate_from_sfm


class ImageBagNoInsvTests(unittest.TestCase):
    def test_third_camera_config_keeps_the_frame_rotation(self):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, 3] = (0.12, -0.34, 0.56)
        cfg = ThirdCameraConfig(camera_name="cam2", source_path="IMAGE.bag", source_type="bag",
                                T_lidar_to_cam2_rigid_4x4=mat.tolist(), image_rotation_deg=270)
        cfg2 = ThirdCameraConfig.from_dict(cfg.to_dict())
        self.assertEqual(cfg2.image_rotation_deg, 270)
        self.assertEqual(cfg2.T_lidar_to_cam2_rigid_4x4[2][3], 0.56)

    def test_bag_camera_clock_is_the_trajectory_start_minus_the_frame_time_origin(self):
        from raven_app.bag_io import bag_time_origin
        from raven_app.workflow import _apply_bag_camera_clocks
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds = Path(tmp_dir)
            (ds / "images").mkdir()
            manifest = {"timestamps": {"cam2/frame_000001.jpg": 3.1, "cam2/frame_000002.jpg": 3.7},
                        "first_timestamp_sec": 1000.1}
            (ds / "images" / "frames.json").write_text(json.dumps(manifest))
            # Older manifests: the origin is recovered from the first frame.
            self.assertAlmostEqual(bag_time_origin(ds, "cam2"), 997.0)
            manifest["time_origins"] = {"cam2": 996.5}
            (ds / "images" / "frames.json").write_text(json.dumps(manifest))
            trajectory = ds / "trajectory.txt"
            trajectory.write_text("1000.0 0 0 0 0 0 0 1\n1001.0 0 0 0 0 0 0 1\n")
            camera = ThirdCameraConfig(camera_name="cam2", source_type="bag")
            clocks = _apply_bag_camera_clocks(ds, [camera], trajectory, dt_sync=1.0)
            self.assertAlmostEqual(clocks["cam2"], 3.5)
            # t_query = t_slam0 + t_frame - dt_sync - offset must be the frame's bag time.
            self.assertAlmostEqual(1000.0 + 3.1 - 1.0 - camera.time_offset_s, 996.5 + 3.1)
            self.assertEqual(camera.calibration_report["time_offset"]["status"], "configured")

    def test_factory_extrinsic_is_stored_as_the_camera_pose_in_the_lidar_frame(self):
        from unittest import mock
        from raven_app.workflow import _bag_camera_factory_calibration
        # BSJADE factory transform_matrix: LiDAR points into the camera (Rcl/Pcl).
        T_cl = [[-0.9996107670269285, 0.02221961472673402, -0.01687018569047376, -0.006810373674206506],
                [0.0037024618385771337, -0.49369624636929954, -0.8696265336897201, -0.06977653344994117],
                [-0.027651513885670723, -0.8693505075872788, 0.4934218162359374, -0.0024897936956511484],
                [0, 0, 0, 1]]
        with tempfile.TemporaryDirectory() as tmp_dir:
            calib = Path(tmp_dir) / "calib.json"
            calib.write_text(json.dumps({
                "camera_info": {"front": {"K": [1123.5, 0, 1542.6, 0, 1124.2, 2110.5, 0, 0, 1],
                                          "coeff": [-0.043, 0.0024, -0.0009, 0.0001]}},
                "out_put": {"front": {"transform_matrix": T_cl}}}))
            with mock.patch("raven_app.bag_io.first_image_size", return_value=(4000, 3000)):
                factory = _bag_camera_factory_calibration(Path(tmp_dir) / "LIDAR.bag", "IMAGE.bag", calib, "/cam")
        T_lc = np.array(factory["T_lidar_to_cam"])
        np.testing.assert_allclose(T_lc @ np.array(T_cl), np.eye(4), atol=1e-9)
        # The camera centre, about 7 cm from the LiDAR origin (not the matrix's own translation).
        np.testing.assert_allclose(T_lc[:3, 3], [-0.0066, -0.0365, -0.0596], atol=2e-4)
        self.assertEqual(factory["image_rotation_deg"], 270)

    def test_cli_parsing_no_insv_with_image_bag(self):
        # 1. No --insv passed at all
        args = parse([
            'workflow',
            '--bag', 'test.bag',
            '--image-bag', 'image.bag',
            '--use-image-bag',
            '--output', 'test_out',
        ])
        self.assertIsNone(args.insv)
        self.assertTrue(args.use_image_bag)

        # 2. Empty string --insv "" passed (as GUI previously did)
        args_empty = parse([
            'workflow',
            '--bag', 'test.bag',
            '--insv', '',
            '--image-bag', 'image.bag',
            '--use-image-bag',
            '--output', 'test_out',
        ])
        self.assertIsNone(args_empty.insv)

    def test_recalibrate_from_sfm_skips_when_no_cam0(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            ds = Path(tmp_dir)
            images_dir = ds / "images"
            cam2_dir = images_dir / "cam2"
            cam2_dir.mkdir(parents=True)
            sparse_0 = ds / "sparse" / "0"
            sparse_0.mkdir(parents=True)

            # points3D.bin with 1 dummy point
            with open(sparse_0 / "points3D.bin", "wb") as f:
                import struct
                f.write(struct.pack("<Q", 0))

            # images.bin with cam2 image only
            with open(sparse_0 / "images.bin", "wb") as f:
                import struct
                f.write(struct.pack("<Q", 1))
                f.write(struct.pack("<I", 1))
                f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))
                f.write(struct.pack("<3d", 0.0, 0.0, 0.0))
                f.write(struct.pack("<I", 1))
                f.write(b"cam2/frame_000001.jpg\x00")
                f.write(struct.pack("<Q", 0))

            res = recalibrate_from_sfm(ds, fps=2.0)
            self.assertIsNone(res)


if __name__ == '__main__':
    unittest.main()
