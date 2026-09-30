"""Unit tests for third camera / auxiliary source integration."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from raven_app.third_camera import (
    ThirdCameraConfig,
    align_third_camera_to_rig,
    default_smartphone_intrinsics,
    detect_source_metadata,
    estimate_relative_camera_pose,
    extract_third_camera_frames,
    inject_third_camera_into_colmap,
    load_third_camera_config,
    match_features_between_images,
    save_third_camera_config,
    synthesize_third_camera_poses,
)


class TestThirdCamera(unittest.TestCase):
    def test_config_serialization(self):
        cfg = ThirdCameraConfig(
            camera_name="cam2",
            source_path="video.mp4",
            source_type="video",
            fps=3.0,
            time_offset_s=0.5,
        )
        d = cfg.to_dict()
        self.assertEqual(d["camera_name"], "cam2")
        self.assertEqual(d["fps"], 3.0)

        cfg2 = ThirdCameraConfig.from_dict(d)
        self.assertEqual(cfg2.time_offset_s, 0.5)

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "third_cam.json"
            save_third_camera_config(cfg, p)
            loaded = load_third_camera_config(p)
            self.assertEqual(loaded.camera_name, "cam2")
            self.assertEqual(loaded.fps, 3.0)

    def test_default_intrinsics(self):
        intrinsics = default_smartphone_intrinsics(1920, 1080)
        self.assertEqual(intrinsics["width"], 1920.0)
        self.assertEqual(intrinsics["height"], 1080.0)
        self.assertAlmostEqual(intrinsics["cx"], 960.0)
        self.assertAlmostEqual(intrinsics["cy"], 540.0)
        self.assertAlmostEqual(intrinsics["fx"], 1920.0 * 0.8)

    def test_detect_source_metadata_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input_imgs"
            p.mkdir()
            for i in range(3):
                img = np.zeros((100, 200, 3), dtype=np.uint8)
                cv2.imwrite(str(p / f"img_{i}.jpg"), img)

            meta = detect_source_metadata(p)
            self.assertEqual(meta["type"], "folder")
            self.assertEqual(meta["image_count"], 3)
            self.assertEqual(meta["width"], 200)
            self.assertEqual(meta["height"], 100)

    def test_extract_third_camera_frames_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_dir = Path(tmp) / "source"
            source_dir.mkdir()
            for i in range(4):
                img = np.full((120, 160, 3), i * 50, dtype=np.uint8)
                cv2.imwrite(str(source_dir / f"frame_{i}.png"), img)

            dataset_dir = Path(tmp) / "dataset"
            dataset_dir.mkdir()

            cfg = ThirdCameraConfig(
                camera_name="cam2",
                source_path=str(source_dir),
                source_type="folder",
                fps=2.0,
            )

            extracted, timestamps = extract_third_camera_frames(cfg, dataset_dir, target_fps=2.0)
            self.assertEqual(len(extracted), 4)
            self.assertEqual(len(timestamps), 4)
            self.assertTrue((dataset_dir / "images" / "cam2" / "frame_000001.jpg").is_file())

            # Check frames.json was written
            manifest = dataset_dir / "images" / "frames.json"
            self.assertTrue(manifest.is_file())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("cam2/frame_000001.jpg", manifest_data["timestamps"])

    def test_match_features_and_relative_pose(self):
        # Create patterned images with corners
        img1 = np.zeros((400, 400, 3), dtype=np.uint8)
        for y in range(50, 350, 40):
            for x in range(50, 350, 40):
                cv2.circle(img1, (x, y), 8, (255, 255, 255), -1)
                cv2.rectangle(img1, (x - 4, y - 4), (x + 4, y + 4), (100, 100, 100), 2)

        # Slight shift for img2
        M = np.float32([[1, 0, 10], [0, 1, 5]])
        img2 = cv2.warpAffine(img1, M, (400, 400))

        pts1, pts2, n_match = match_features_between_images(img1, img2, method="SIFT")
        self.assertGreater(n_match, 10)

        K = np.array([[300.0, 0.0, 200.0], [0.0, 300.0, 200.0], [0.0, 0.0, 1.0]])
        R, t, inl = estimate_relative_camera_pose(pts1, pts2, K, K)
        self.assertIsNotNone(R)
        self.assertIsNotNone(t)
        self.assertGreater(inl, 5)

    def test_inject_third_camera_into_colmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse_out = Path(tmp) / "sparse" / "0"
            sparse_out.mkdir(parents=True)

            cameras_txt = sparse_out / "cameras.txt"
            cameras_txt.write_text(
                "1 OPENCV_FISHEYE 3840 3840 1080 1080 1920 1920 0 0 0 0\n"
                "2 OPENCV_FISHEYE 3840 3840 1080 1080 1920 1920 0 0 0 0\n"
            )
            images_txt = sparse_out / "images.txt"
            images_txt.write_text(
                "1 1 0 0 0 0 0 0 1 cam0/frame_000001.jpg\n\n"
                "2 1 0 0 0 0 0 0 2 cam1/frame_000001.jpg\n\n"
            )

            cfg = ThirdCameraConfig(
                camera_name="cam2",
                camera_model="PINHOLE",
                intrinsics={"width": 1920, "height": 1080, "fx": 1500, "fy": 1500, "cx": 960, "cy": 540},
            )

            fake_poses = [
                {
                    "name": "cam2/frame_000001.jpg",
                    "R_cw": np.eye(3),
                    "t_cw": np.array([0.1, 0.2, 0.3]),
                    "q_cw": np.array([0.0, 0.0, 0.0, 1.0]),  # x, y, z, w
                    "camera_id": 3,
                }
            ]

            inject_third_camera_into_colmap(sparse_out, cfg, fake_poses, camera_id=3)

            cam_content = cameras_txt.read_text()
            self.assertIn("3 PINHOLE 1920 1080", cam_content)

            img_content = images_txt.read_text()
            self.assertIn("cam2/frame_000001.jpg", img_content)
            self.assertIn("3 cam2/frame_000001.jpg", img_content)

    def test_cli_parsing_third_camera(self):
        from raven_app.cli import parse
        wf_args = parse([
            "workflow", "--bag", "scan.bag", "--insv", "video.insv", "--output", "out_dir",
            "--third-camera", "configs/third_camera.json"
        ])
        self.assertEqual(wf_args.third_camera, Path("configs/third_camera.json"))

        color_args = parse([
            "colorize", "--dataset", "my_dataset",
            "--third-camera", "configs/third_camera.json"
        ])
        self.assertEqual(color_args.third_camera, Path("configs/third_camera.json"))

    def test_add_source_dialog_initialization(self):
        try:
            from PyQt6.QtWidgets import QApplication
            from raven_app.views.add_source_dialog import AddSourceDialog

            app = QApplication.instance() or QApplication([])
            cfg = ThirdCameraConfig(
                camera_name="cam2",
                source_path="test_phone.mp4",
                source_type="video",
                fps=3.0,
                time_offset_s=0.25,
            )
            dialog = AddSourceDialog(config=cfg)
            res_cfg = dialog.get_config()
            self.assertEqual(res_cfg.camera_name, "cam2")
            self.assertEqual(res_cfg.source_path, "test_phone.mp4")
            self.assertEqual(res_cfg.fps, 3.0)
            self.assertEqual(res_cfg.time_offset_s, 0.25)
            dialog.close()
        except ImportError:
            pass


if __name__ == "__main__":
    unittest.main()
