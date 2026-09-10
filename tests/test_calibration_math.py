import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripts import pipeline_auto_calibrator_and_colorizer as pipeline


class CalibrationMathTests(unittest.TestCase):
    def test_known_sim3(self):
        x = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
        r = Rot.from_euler("z", 27, degrees=True).as_matrix()
        y = 2.5 * (r @ x.T).T + np.array([4., -2., .5])
        scale, got_r, translation, rmse = pipeline.solve_umeyama_sim3(x, y)
        self.assertAlmostEqual(scale, 2.5, places=10)
        np.testing.assert_allclose(got_r, r, atol=1e-10)
        np.testing.assert_allclose(translation, [4., -2., .5], atol=1e-10)
        self.assertLess(rmse, 1e-10)

    def test_visibility_rejects_foreground_neighbor(self):
        distances = np.array([5.0, 5.2, 5.2], dtype=float)
        # The second point is in the adjacent cell; the third is farther but
        # in the same cell. The 3x3 conservative raster must keep only 5.0.
        flat = np.array([10 * 100 + 10, 10 * 100 + 11, 10 * 100 + 10])
        visible = pipeline.visible_depths(distances, flat, 100, 100)
        np.testing.assert_array_equal(visible, [True, False, False])

    def test_recalibrate_uses_first_trajectory_pose_at_first_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sparse" / "0").mkdir(parents=True)
            (root / "deliverables").mkdir()
            (root / "colmap_to_lidar_alignment.json").write_text(
                json.dumps({"calibration_version": pipeline.CALIBRATION_VERSION, "scale": 1.0, "R": np.eye(3).tolist(), "t": [0, 0, 0], "dt_sync_seconds": 0.0})
            )
            identity = Rot.identity()
            images = [
                {"name": "cam0/frame_000001.jpg", "C": np.array([11., 0., 0.]), "R_cw": np.eye(3), "cam_id": 1},
                {"name": "cam1/frame_000001.jpg", "C": np.array([9., 0., 0.]), "R_cw": np.eye(3), "cam_id": 2},
            ]
            cameras = {
                1: {"params": tuple(np.arange(12, dtype=float) + 100), "width": 3840, "height": 3840},
                2: {"params": tuple(np.arange(12, dtype=float) + 200), "width": 3840, "height": 3840},
            }
            with patch.object(pipeline, "load_trajectory", return_value=(np.array([0., 1.]), np.array([[10., 0., 0.], [20., 0., 0.]]), Rot.from_euler("xyz", [[0, 0, 0], [0, 0, 0]], degrees=True))), \
                 patch.object(pipeline, "load_colmap_images", return_value=images), \
                 patch.object(pipeline, "load_colmap_cameras", return_value=cameras), \
                 patch.object(pipeline, "frame_time", return_value=0.0):
                pipeline.recalibrate_from_sfm(root, fps=2.0)
            result = json.loads((root / "rig_calibration.json").read_text())
            np.testing.assert_allclose([result["T_lidar_to_cam0_rigid_4x4"][i][3] for i in range(3)], [1., 0., 0.], atol=1e-12)
            np.testing.assert_allclose([result["T_lidar_to_cam1_rigid_4x4"][i][3] for i in range(3)], [-1., 0., 0.], atol=1e-12)
            self.assertEqual(result["cam0_front_intrinsics"]["fx"], 100.0)
            self.assertEqual(result["cam1_rear_intrinsics"]["fx"], 200.0)
            self.assertEqual(result["cam0_front_intrinsics"]["sx1"], 110.0)

    def test_depth_visibility_accepts_equal_foreground_samples(self):
        distances = np.array([5.0, 5.02], dtype=float)
        flat = np.array([0, 1])
        visible = pipeline.visible_depths(distances, flat, 2, 1)
        np.testing.assert_array_equal(visible, [True, True])

    def test_current_alignment_uses_versioned_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = {"calibration_version": pipeline.CALIBRATION_VERSION, "scale": 1.0}
            (root / "colmap_to_lidar_alignment.json").write_text(json.dumps(saved))
            with patch.object(pipeline, "align_colmap_to_lidar") as align:
                self.assertEqual(pipeline.current_alignment(root), saved)
                align.assert_not_called()

    def test_current_alignment_upgrades_old_version_with_dt_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "colmap_to_lidar_alignment.json").write_text(
                json.dumps({"calibration_version": 1, "dt_sync_seconds": 1.25})
            )
            upgraded = {"calibration_version": pipeline.CALIBRATION_VERSION, "dt_sync_seconds": 1.25}
            with patch.object(pipeline, "align_colmap_to_lidar", return_value=upgraded) as align:
                self.assertEqual(pipeline.current_alignment(root, fps=2.0), upgraded)
                align.assert_called_once_with(root, 2.0, 1.25)

    def test_direct_colorization_requires_capture_specific_dt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calib = root / "calibration.json"
            intr = {k: 0.0 for k in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")}
            intr.update(fx=1.0, fy=1.0)
            calib.write_text(json.dumps({
                "cam0_front_intrinsics": intr, "cam1_rear_intrinsics": intr,
                "T_lidar_to_cam0_rigid_4x4": np.eye(4).tolist(),
                "T_lidar_to_cam1_rigid_4x4": np.eye(4).tolist(),
            }))
            with self.assertRaisesRegex(ValueError, "time synchronization"):
                pipeline.colorize_via_direct_rigid(root, calib_json_path=calib)


if __name__ == "__main__":
    unittest.main()
