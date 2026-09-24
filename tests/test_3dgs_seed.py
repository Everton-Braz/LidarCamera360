import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from raven_app.workflow import export_colmap_3dgs, load_lidar_seed_points, write_colmap_points3d, fuse_colmap_points3d


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

    def test_fuse_colmap_points3d(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse" / "0"
            sparse_dir.mkdir(parents=True)

            # 1. Write initial SfM sparse points (e.g. 2 points with 2D tracks)
            bin_path = sparse_dir / "points3D.bin"
            ply_path = sparse_dir / "points3D.ply"
            txt_path = sparse_dir / "points3D.txt"

            sfm_bytes = bytearray()
            # Point 1: track_len=1
            sfm_bytes += struct.pack("<Q3d3BdQ2I", 1, 1.0, 2.0, 3.0, 10, 20, 30, 0.5, 1, 101, 5)
            # Point 2: track_len=2
            sfm_bytes += struct.pack("<Q3d3BdQ4I", 2, 4.0, 5.0, 6.0, 40, 50, 60, 0.4, 2, 101, 8, 102, 12)

            with open(bin_path, "wb") as f:
                f.write(struct.pack("<Q", 2))
                f.write(sfm_bytes)

            # Also write corresponding SfM PLY
            header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode("ascii")
            dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
            arr = np.empty(2, dtype=dt)
            arr["x"] = [1.0, 4.0]
            arr["y"] = [2.0, 5.0]
            arr["z"] = [3.0, 6.0]
            arr["r"] = [10, 40]
            arr["g"] = [20, 50]
            arr["b"] = [30, 60]
            with open(ply_path, "wb") as f:
                f.write(header)
                f.write(arr.tobytes())

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("# 3D point list\n1 1.0 2.0 3.0 10 20 30 0.5 101 5\n2 4.0 5.0 6.0 40 50 60 0.4 101 8 102 12\n")

            # 2. Fuse with 3 LiDAR points
            lidar_xyz = np.array([
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0]
            ], dtype=np.float64)
            lidar_rgb = np.array([
                [100, 101, 102],
                [103, 104, 105],
                [106, 107, 108]
            ], dtype=np.uint8)

            n_sfm, n_lidar, n_total = fuse_colmap_points3d(sparse_dir, lidar_xyz, lidar_rgb)
            self.assertEqual(n_sfm, 2)
            self.assertEqual(n_lidar, 3)
            self.assertEqual(n_total, 5)

            # 3. Verify fused points3D.bin
            with open(bin_path, "rb") as f:
                total_pts = struct.unpack("<Q", f.read(8))[0]
                self.assertEqual(total_pts, 5)

                # Point 1 (SfM)
                p1_id, p1_x, p1_y, p1_z, p1_r, p1_g, p1_b, p1_err, p1_tlen = struct.unpack("<Q3d3BdQ", f.read(51))
                self.assertEqual(p1_id, 1)
                self.assertEqual(p1_tlen, 1)
                t1 = struct.unpack("<2I", f.read(8))
                self.assertEqual(t1, (101, 5))

                # Point 2 (SfM)
                p2_id, p2_x, p2_y, p2_z, p2_r, p2_g, p2_b, p2_err, p2_tlen = struct.unpack("<Q3d3BdQ", f.read(51))
                self.assertEqual(p2_id, 2)
                self.assertEqual(p2_tlen, 2)
                t2 = struct.unpack("<4I", f.read(16))
                self.assertEqual(t2, (101, 8, 102, 12))

                # Point 3 (LiDAR point 1)
                p3_id, p3_x, p3_y, p3_z, p3_r, p3_g, p3_b, p3_err, p3_tlen = struct.unpack("<Q3d3BdQ", f.read(51))
                self.assertEqual(p3_id, 3)
                self.assertAlmostEqual(p3_x, 10.0)
                self.assertEqual((p3_r, p3_g, p3_b), (100, 101, 102))
                self.assertEqual(p3_tlen, 0)

            # Check that backup points3D_sfm_sparse.* files exist
            self.assertTrue((sparse_dir / "points3D_sfm_sparse.bin").is_file())
            self.assertTrue((sparse_dir / "points3D_sfm_sparse.ply").is_file())
            self.assertTrue((sparse_dir / "points3D_sfm_sparse.txt").is_file())


if __name__ == '__main__':
    unittest.main()
