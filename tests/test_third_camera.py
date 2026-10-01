"""Unit tests for third camera / auxiliary source integration."""

import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from raven_app.third_camera import (
    ThirdCameraConfig,
    _apply_pyav_frame_orientation,
    _estimate_auxiliary_time_offset,
    _pose_consensus,
    align_third_camera_to_rig,
    calibrate_third_camera_intrinsics,
    default_smartphone_intrinsics,
    detect_source_metadata,
    estimate_relative_camera_pose,
    extract_third_camera_frames,
    inject_third_camera_into_colmap,
    load_third_camera_config,
    load_third_camera_configs,
    match_features_between_images,
    save_third_camera_config,
    synthesize_third_camera_poses,
)
from scipy.spatial.transform import Rotation as Rot, Slerp


class TestThirdCamera(unittest.TestCase):
    app = None

    @staticmethod
    def _write_sfm_fixture(root, poses, camera_id=3, model_id=10):
        sparse = root / "sparse" / "0"
        sparse.mkdir(parents=True, exist_ok=True)
        prism_params = (3291.165, 3294.470, 1920.0, 1080.0,
                        0.376, -0.014, 0.003, -0.001, 0.410, -0.413, 0.001, -0.004)
        if model_id == 5:
            params = prism_params[:8]
        elif model_id == 1:
            params = prism_params[:4]
        else:
            params = prism_params
        with (sparse / "cameras.bin").open("wb") as f:
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<iiQQ", camera_id, model_id, 3840, 2160))
            f.write(struct.pack(f"<{len(params)}d", *params))

        with (sparse / "images.bin").open("wb") as f:
            f.write(struct.pack("<Q", len(poses)))
            for image_id, (name, center, rotation_cw) in enumerate(poses, start=1):
                qx, qy, qz, qw = Rot.from_matrix(rotation_cw).as_quat()
                tvec = -rotation_cw @ np.asarray(center, dtype=np.float64)
                f.write(struct.pack("<I4d3dI", image_id, qw, qx, qy, qz,
                                    *tvec, camera_id))
                f.write(name.encode("ascii") + b"\0")
                f.write(struct.pack("<Q", 0))
        return sparse

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
        self.assertIn("k3", intrinsics)

    def test_load_third_camera_configs_legacy_and_bundle(self):
        first = {"camera_name": "cam2", "source_path": "phone.mp4"}
        second = {"camera_name": "cam3", "source_path": "action.mp4"}
        self.assertEqual(load_third_camera_configs(first)[0].camera_name, "cam2")
        bundle = load_third_camera_configs({"cameras": [first, second]})
        self.assertEqual([cfg.camera_name for cfg in bundle], ["cam2", "cam3"])
        self.assertEqual(len(load_third_camera_configs([first, second])), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cameras.json"
            configs = [ThirdCameraConfig.from_dict(first), ThirdCameraConfig.from_dict(second)]
            save_third_camera_config(configs, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload), {"cameras"})
            self.assertEqual(
                [cfg.camera_name for cfg in load_third_camera_configs(path)],
                ["cam2", "cam3"],
            )

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

    def test_video_extraction_honors_camera_fps_and_prunes_stale_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.avi"
            writer = cv2.VideoWriter(
                str(source), cv2.VideoWriter_fourcc(*"MJPG"), 24.0, (64, 48)
            )
            self.assertTrue(writer.isOpened(), "OpenCV MJPG video writer is unavailable")
            for index in range(240):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                cv2.putText(frame, str(index), (4, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 255, 255), 1)
                writer.write(frame)
            writer.release()

            dataset = root / "dataset"
            camera_dir = dataset / "images" / "cam2"
            camera_dir.mkdir(parents=True)
            cv2.imwrite(str(camera_dir / "frame_000011.jpg"), np.zeros((48, 64, 3), np.uint8))
            manifest = dataset / "images" / "frames.json"
            manifest.write_text(json.dumps({
                "schema": 1,
                "timestamps": {
                    "cam0/frame_000001.jpg": 0.0,
                    "cam2/frame_000011.jpg": 10.0,
                },
            }), encoding="utf-8")

            config = ThirdCameraConfig(
                camera_name="cam2", source_path=str(source), source_type="video", fps=1.0
            )
            extracted, timestamps = extract_third_camera_frames(config, dataset)

            self.assertEqual(len(extracted), 10)
            self.assertEqual(len(timestamps), 10)
            self.assertEqual(len(list(camera_dir.glob("*.jpg"))), 10)
            self.assertFalse((camera_dir / "frame_000011.jpg").exists())
            manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertIn("cam0/frame_000001.jpg", manifest_data["timestamps"])
            self.assertNotIn("cam2/frame_000011.jpg", manifest_data["timestamps"])
            self.assertAlmostEqual(timestamps["cam2/frame_000001.jpg"], 0.0, delta=0.1)

    def test_pyav_frame_orientation_applies_display_rotation(self):
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        for rotation, expected in (
            (-180, cv2.rotate(image, cv2.ROTATE_180)),
            (90, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            (-90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ):
            with self.subTest(rotation=rotation):
                result = _apply_pyav_frame_orientation(
                    SimpleNamespace(rotation=rotation), image
                )
                np.testing.assert_array_equal(result, expected)

    def test_sfm_intrinsics_model_mismatch_preserves_user_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = [
                (f"cam2/frame_{i:06d}.jpg", np.zeros(3), np.eye(3))
                for i in range(1, 9)
            ]
            self._write_sfm_fixture(root, poses)
            configured_intrinsics = {
                "width": 3840.0, "height": 2160.0, "fx": 3072.0,
                "fy": 3072.0, "cx": 1920.0, "cy": 1080.0,
                "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0,
                "p1": 0.0, "p2": 0.0,
            }
            config = ThirdCameraConfig(
                camera_name="cam2",
                camera_model="PINHOLE",
                intrinsics=configured_intrinsics,
            )

            result = calibrate_third_camera_intrinsics(root, config)

            self.assertEqual(result.camera_model, "PINHOLE")
            self.assertEqual(result.intrinsics, configured_intrinsics)
            self.assertEqual(result.calibration_report["intrinsics"]["status"], "model_mismatch")
            self.assertEqual(result.calibration_report["intrinsics"]["fitted_model"], "THIN_PRISM_FISHEYE")

            fisheye_config = ThirdCameraConfig(
                camera_name="cam2",
                camera_model="OPENCV_FISHEYE",
                intrinsics=configured_intrinsics,
            )
            fisheye_result = calibrate_third_camera_intrinsics(root, fisheye_config)
            self.assertEqual(fisheye_result.camera_model, "OPENCV_FISHEYE")
            self.assertEqual(fisheye_result.intrinsics, configured_intrinsics)
            self.assertEqual(
                fisheye_result.calibration_report["intrinsics"]["fitted_model"],
                "THIN_PRISM_FISHEYE",
            )

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

    def test_sfm_intrinsics_are_loaded_but_extrinsics_need_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = [
                (f"cam2/frame_{i:06d}.jpg", np.zeros(3), np.eye(3))
                for i in range(1, 9)
            ]
            self._write_sfm_fixture(root, poses, model_id=5)
            (root / "images" / "cam0").mkdir(parents=True)
            (root / "images" / "cam2").mkdir(parents=True)
            cv2.imwrite(str(root / "images" / "cam0" / "frame_000001.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
            cv2.imwrite(str(root / "images" / "cam2" / "frame_000001.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
            config = ThirdCameraConfig(camera_name="cam2", camera_model="OPENCV_FISHEYE")
            preset_transform = np.asarray(config.T_lidar_to_cam2_rigid_4x4)
            with (
                patch("raven_app.third_camera.match_features_between_images") as feature_match,
                patch("raven_app.third_camera.estimate_relative_camera_pose") as relative_pose,
            ):
                result = align_third_camera_to_rig(root, config)

            self.assertEqual(result.camera_model, "OPENCV_FISHEYE")
            self.assertAlmostEqual(result.intrinsics["fx"], 3291.165)
            self.assertAlmostEqual(result.intrinsics["fy"], 3294.470)
            self.assertAlmostEqual(result.intrinsics["k3"], 0.003)
            self.assertEqual(result.calibration_report["intrinsics"]["status"], "sfm_calibrated")
            self.assertEqual(result.calibration_report["extrinsics"]["status"], "uncalibrated")
            self.assertIn("alignment.json is missing", result.calibration_report["extrinsics"]["reason"])
            np.testing.assert_allclose(result.T_lidar_to_cam2_rigid_4x4, preset_transform)
            feature_match.assert_not_called()
            relative_pose.assert_not_called()

    def test_current_alignment_and_camera_time_offset_calibrate_rigid_extrinsics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rotations_lidar = Rot.from_euler("z", np.linspace(0.0, 0.7, 10)[:, None])
            positions_lidar = np.column_stack((np.arange(10.0), np.square(np.arange(10.0)) * 0.03,
                                               np.zeros(10)))
            rotation_lc = Rot.from_euler("xyz", [0.02, -0.03, 0.05]).as_matrix()
            translation_lc = np.array([0.30, 0.20, 0.40])
            offset_total = 0.25 + 0.50
            poses = []
            timestamps = {}
            for frame_idx, query_time in enumerate(range(1, 9), start=1):
                slam_rotation = rotations_lidar[query_time].as_matrix()
                center = positions_lidar[query_time] + slam_rotation @ translation_lc
                rotation_cw = (slam_rotation @ rotation_lc).T
                name = f"cam2/frame_{frame_idx:06d}.jpg"
                poses.append((name, center, rotation_cw))
                timestamps[name] = float(query_time) + offset_total
            self._write_sfm_fixture(root, poses)
            (root / "colmap_to_lidar_alignment.json").write_text("{}", encoding="utf-8")
            (root / "images").mkdir(parents=True)
            (root / "images" / "frames.json").write_text(
                json.dumps({"timestamps": timestamps}), encoding="utf-8"
            )
            trajectory_path = root / "trajectory.txt"
            trajectory_path.write_text("fixture", encoding="utf-8")

            from scripts.pipeline_auto_calibrator_and_colorizer import CALIBRATION_VERSION

            alignment = {
                "calibration_version": CALIBRATION_VERSION,
                "scale": 1.0,
                "R": np.eye(3).tolist(),
                "t": [0.0, 0.0, 0.0],
                "dt_sync_seconds": 0.25,
            }
            config = ThirdCameraConfig(camera_name="cam2")
            with (
                patch("scripts.pipeline_auto_calibrator_and_colorizer.current_alignment", return_value=alignment),
                patch("scripts.pipeline_auto_calibrator_and_colorizer.get_slam_trajectory_path", return_value=trajectory_path),
                patch("scripts.pipeline_auto_calibrator_and_colorizer.load_trajectory",
                      return_value=(np.arange(10.0), positions_lidar, rotations_lidar)),
            ):
                result = align_third_camera_to_rig(root, config)

            self.assertEqual(result.calibration_report["extrinsics"]["status"], "calibrated")
            self.assertEqual(result.calibration_report["time_offset"]["status"], "calibrated")
            self.assertAlmostEqual(result.time_offset_s, 0.5, delta=0.1)
            self.assertAlmostEqual(
                result.calibration_report["time_offset"]["total_applied_offset_seconds"], 0.75
            )
            transform = np.asarray(result.T_lidar_to_cam2_rigid_4x4)
            np.testing.assert_allclose(transform[:3, :3], rotation_lc, atol=1e-6)
            np.testing.assert_allclose(transform[:3, 3], translation_lc, atol=1e-6)

    def test_rigid_pose_consensus_rejects_inconsistent_frames(self):
        rotations = [np.eye(3) for _ in range(10)]
        translations = [np.array([0.3, 0.2, 0.4]) for _ in range(6)]
        translations += [np.array([1.0 + i, -0.5, 0.1]) for i in range(4)]
        rotation, translation, quality = _pose_consensus(rotations, translations)
        self.assertIsNone(rotation)
        self.assertIsNone(translation)
        self.assertLess(quality["consensus_fraction"], 0.7)

    def test_unknown_offset_is_rejected_when_motion_cannot_disambiguate_it(self):
        t_slam = np.arange(0.0, 31.0)
        positions = np.column_stack((t_slam, np.zeros_like(t_slam), np.zeros_like(t_slam)))
        rotations = Rot.from_euler("xyz", np.zeros((len(t_slam), 3)))
        slerp = Slerp(t_slam, rotations)
        dt_sync, true_offset = 0.25, 0.5
        samples = []
        for video_time in np.arange(6.0, 18.0):
            slam_time = video_time - dt_sync - true_offset
            samples.append({
                "video_time": video_time,
                "center_lidar": np.array([slam_time + 0.3, 0.2, 0.4]),
                "rotation_world_cam": np.eye(3),
            })

        estimated, details = _estimate_auxiliary_time_offset(
            samples, t_slam, positions, slerp, dt_sync, extraction_fps=1.0
        )
        self.assertIsNone(estimated)
        self.assertIn("comparable rigid-pose consensus", details["reason"])

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

    def test_inject_opencv_fisheye_uses_sfm_radial_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            sparse_out = Path(tmp) / "sparse" / "0"
            sparse_out.mkdir(parents=True)
            (sparse_out / "cameras.txt").write_text("# Camera list\n", encoding="utf-8")
            config = ThirdCameraConfig(
                camera_model="OPENCV_FISHEYE",
                intrinsics={
                    "width": 3840, "height": 2160, "fx": 3291, "fy": 3294,
                    "cx": 1920, "cy": 1080, "k1": 0.1, "k2": 0.2,
                    "k3": 0.3, "k4": 0.4,
                },
            )
            inject_third_camera_into_colmap(sparse_out, config, [], camera_id=3)
            camera_lines = (sparse_out / "cameras.txt").read_text(encoding="utf-8")
            self.assertIn("3 OPENCV_FISHEYE 3840 2160 3291 3294 1920 1080 0.1 0.2 0.3 0.4", camera_lines)

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

            # Keep the Python wrapper alive for the full test suite.  Some
            # qfluentwidgets globals are children of QApplication and become
            # invalid if this local reference is collected after the test.
            self.__class__.app = QApplication.instance() or QApplication([])
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
