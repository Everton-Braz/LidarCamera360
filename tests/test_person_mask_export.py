import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from raven_app.third_camera import ThirdCameraConfig
from raven_app.workflow import _copy_person_masks, export_colmap_3dgs


class PersonMaskExportTests(unittest.TestCase):
    def test_exports_validated_masks_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            images = dataset / "images"
            masks = dataset / "masks"
            image_files = []
            for camera in ("cam0", "cam1"):
                image_dir = images / camera
                image_dir.mkdir(parents=True)
                image_path = image_dir / "frame.jpg"
                cv2.imwrite(str(image_path), np.full((12, 16, 3), 100, np.uint8))
                image_files.append(image_path)
                mask_path = masks / camera / "frame.jpg.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask = np.full((12, 16), 255, np.uint8)
                mask[2:4, 3:5] = 0
                cv2.imwrite(str(mask_path), mask)
            manifest = {"schema": 1, "polarity": "white_keep_black_person", "image_count": 2}
            (masks / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            output = root / "colmap_3dgs"
            stale = output / "masks" / "cam0" / "old.jpg.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old")
            count = _copy_person_masks(dataset, masks, output, image_files)

            self.assertEqual(count, 2)
            self.assertFalse(stale.exists())
            self.assertEqual(
                json.loads((output / "masks" / "manifest.json").read_text(encoding="utf-8")),
                manifest,
            )
            for camera in ("cam0", "cam1"):
                exported = cv2.imdecode(
                    np.fromfile(output / "masks" / camera / "frame.jpg.png", np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                self.assertEqual(exported.shape, (12, 16))
                self.assertEqual(int(exported[2, 3]), 0)
                self.assertEqual(int(exported[0, 0]), 255)

    def test_rejects_wrong_mask_polarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "dataset" / "images" / "cam0"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "frame.jpg"
            cv2.imwrite(str(image_path), np.full((8, 8, 3), 80, np.uint8))
            masks = root / "source_masks"
            masks.mkdir()
            (masks / "manifest.json").write_text(
                json.dumps({"polarity": "white_person_black_keep"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _copy_person_masks(root / "dataset", masks, root / "out", [image_path])

    def test_3dgs_export_preserves_masks_for_all_cameras_and_removes_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            masks = dataset / "masks"
            image_files = []
            for camera in ("cam0", "cam1", "cam2", "cam3"):
                image_dir = dataset / "images" / camera
                image_dir.mkdir(parents=True)
                image = image_dir / "frame_000001.jpg"
                cv2.imwrite(str(image), np.full((12, 16, 3), 90, np.uint8))
                image_files.append(image)
                mask = masks / camera / "frame_000001.jpg.png"
                mask.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask), np.full((12, 16), 255, np.uint8))

            calib = root / "rig.json"
            identity = np.eye(4).tolist()
            intrinsics = {key: 1.0 for key in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4")}
            calib.write_text(json.dumps({
                "cam0_front_intrinsics": intrinsics,
                "cam1_rear_intrinsics": intrinsics,
                "T_lidar_to_cam0_rigid_4x4": identity,
                "T_lidar_to_cam1_rigid_4x4": identity,
            }), encoding="utf-8")
            cameras = [
                ThirdCameraConfig(
                    camera_name=name,
                    camera_model="PINHOLE",
                    intrinsics={"fx": 10.0, "fy": 10.0, "cx": 8.0, "cy": 6.0, "width": 16, "height": 12},
                    T_lidar_to_cam2_rigid_4x4=identity,
                    calibration_report={
                        "intrinsics": {"status": "calibrated"},
                        "time_offset": {"status": "calibrated", "applied": True},
                        "extrinsics": {"status": "calibrated"},
                    },
                )
                for name in ("cam2", "cam3")
            ]
            output = dataset / "colmap_3dgs"
            stale = output / "masks" / "cam4" / "old.jpg.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"stale")
            slam_res = dataset / "slam_out" / "result"
            slam_res.mkdir(parents=True)
            (slam_res / "Eagle_Scan.txt").write_text(
                "0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n"
                "10.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n",
                encoding="utf-8"
            )
            with (
                patch("raven_app.workflow.load_lidar_seed_points", return_value=(None, None)),
                patch("raven_app.workflow._postprocess_3dgs_dataset"),
            ):
                export_colmap_3dgs(dataset, calib, masks_dir=masks, third_camera=cameras)

            self.assertFalse(stale.exists())
            for camera in ("cam0", "cam1", "cam2", "cam3"):
                self.assertTrue((output / "masks" / camera / "frame_000001.jpg.png").is_file())


if __name__ == "__main__":
    unittest.main()
