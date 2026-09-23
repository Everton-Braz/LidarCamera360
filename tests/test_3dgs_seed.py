import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from raven_app.workflow import export_colmap_3dgs, load_lidar_seed_points, write_colmap_points3d


class TestLidar3DGSSeed(unittest.TestCase):
    def test_write_colmap_points3d_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            dst = tmp / "sparse" / "0"

            xyz = np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ], dtype=np.float64)

            rgb = np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
            ], dtype=np.uint8)

            write_colmap_points3d(dst, xyz, rgb)

            # Check points3D.ply
            ply_path = dst / "points3D.ply"
            self.assertTrue(ply_path.is_file())
            # Check points3D.bin
            bin_path = dst / "points3D.bin"
            self.assertTrue(bin_path.is_file())
            with open(bin_path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.assertEqual(n, 3)
                pid, x, y, z, r, g, b, err, track_len = struct.unpack("<Q3d3BdQ", f.read(8 + 24 + 3 + 8 + 8))
                self.assertEqual(pid, 1)
                self.assertAlmostEqual(x, 1.0)
                self.assertAlmostEqual(y, 2.0)
                self.assertAlmostEqual(z, 3.0)
                self.assertEqual((r, g, b), (255, 0, 0))
                self.assertEqual(track_len, 0)

            # Check points3D.txt
            txt_path = dst / "points3D.txt"
            self.assertTrue(txt_path.is_file())
            content = txt_path.read_text(encoding="utf-8")
            self.assertIn("1 1.000000 2.000000 3.000000 255 0 0", content)

            # Check root copy
            self.assertTrue((tmp / "points3D.ply").is_file())

    def test_export_colmap_3dgs_seeds_from_lidar(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset"
            deliv = dataset / "deliverables"
            deliv.mkdir(parents=True)

            # Write a synthetic colored PLY into deliverables
            ply_file = deliv / "lidar_colored_direct_rigid_20260917_120000.ply"
            xyz = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
            rgb = np.array([[120, 130, 140], [200, 210, 220]], dtype=np.uint8)
            write_colmap_points3d(deliv, xyz, rgb)
            if (deliv / "points3D.ply").is_file():
                os.replace(deliv / "points3D.ply", ply_file)

            # Add sample camera frames
            for cam in ("cam0", "cam1"):
                d = dataset / "images" / cam
                d.mkdir(parents=True)
                (d / "frame_000001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01")

            calib_data = {
                "cam0_front_intrinsics": {"fx": 1000, "fy": 1000, "cx": 1920, "cy": 1920, "k1": 0, "k2": 0, "k3": 0, "k4": 0},
                "cam1_rear_intrinsics": {"fx": 1000, "fy": 1000, "cx": 1920, "cy": 1920, "k1": 0, "k2": 0, "k3": 0, "k4": 0},
                "T_lidar_to_cam0_rigid_4x4": np.eye(4).tolist(),
                "T_lidar_to_cam1_rigid_4x4": np.eye(4).tolist(),
            }
            calib_file = dataset / "rig_profile.json"
            calib_file.write_text(json.dumps(calib_data), encoding="utf-8")

            out = export_colmap_3dgs(dataset, calib_file, fps=1.0, dt_sync=0.0)

            pts_ply = out / "sparse" / "0" / "points3D.ply"
            pts_bin = out / "sparse" / "0" / "points3D.bin"
            pts_txt = out / "sparse" / "0" / "points3D.txt"

            self.assertTrue(pts_ply.is_file())
            self.assertTrue(pts_bin.is_file())
            self.assertTrue(pts_txt.is_file())

            # Verify that points3D.bin contains the 2 LiDAR points
            with open(pts_bin, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.assertEqual(n, 2)
                pid, x, y, z, r, g, b, err, _ = struct.unpack("<Q3d3BdQ", f.read(8 + 24 + 3 + 8 + 8))
                self.assertEqual(pid, 1)
                self.assertAlmostEqual(x, 10.0, places=4)
                self.assertAlmostEqual(y, 20.0, places=4)
                self.assertAlmostEqual(z, 30.0, places=4)
                self.assertEqual((r, g, b), (120, 130, 140))


if __name__ == '__main__':
    unittest.main()
