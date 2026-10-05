import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QScrollArea

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from raven_app.process_runner import ProcessRunner
from raven_app.third_camera import ThirdCameraConfig
from raven_app.views.colorize_view import ColorizeView
from raven_app.views.add_source_dialog import AddSourceDialog
from raven_app.views.unified_workflow_view import UnifiedWorkflowView


class AuxiliaryCameraViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_add_source_dialog_fits_narrow_and_high_dpi_layout(self):
        dialog = AddSourceDialog()
        dialog.resize(520, 620)
        dialog.show()
        self.app.processEvents()

        scroll = dialog.findChild(QScrollArea)
        self.assertIsNotNone(scroll)
        self.assertEqual(
            scroll.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        self.assertLessEqual(scroll.widget().width(), scroll.viewport().width())

        for widget in (
            dialog.source_input,
            dialog.btn_browse_video,
            dialog.btn_browse_insv,
            dialog.btn_browse_folder,
            dialog.cam_name_input,
            dialog.fps_spin,
            dialog.time_offset_spin,
            dialog.model_combo,
            dialog.btn_reset_intrinsics,
            dialog.width_spin,
            dialog.height_spin,
            dialog.fx_spin,
            dialog.fy_spin,
            dialog.cx_spin,
            dialog.cy_spin,
            dialog.tx_spin,
            dialog.ty_spin,
            dialog.tz_spin,
            dialog.roll_spin,
            dialog.pitch_spin,
            dialog.yaw_spin,
        ):
            x = widget.mapTo(scroll.viewport(), widget.rect().topLeft()).x()
            self.assertGreaterEqual(x, 0, widget.objectName() or type(widget).__name__)
            self.assertLessEqual(
                x + widget.width(),
                scroll.viewport().width(),
                widget.objectName() or type(widget).__name__,
            )

        for widget in (
            dialog.btn_load_json,
            dialog.btn_save_json,
            dialog.btn_cancel,
            dialog.btn_confirm,
        ):
            x = widget.mapTo(dialog, widget.rect().topLeft()).x()
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + widget.width(), dialog.width())

        image = QImage(dialog.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        dialog.render(painter)
        painter.end()
        self.assertFalse(image.isNull())
        dialog.close()

    def test_insv_source_defaults_to_thin_prism_and_both_lens_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "auxiliary.insv"
            source.write_bytes(b"metadata is mocked")
            metadata = {
                "type": "insv",
                "width": 2880,
                "height": 2880,
                "lens_count": 2,
            }
            dialog = AddSourceDialog()
            with patch("raven_app.views.add_source_dialog.detect_source_metadata", return_value=metadata):
                dialog._set_source(str(source))

            self.assertEqual(dialog.model_combo.currentText(), "THIN_PRISM_FISHEYE")
            self.assertFalse(dialog.insv_stream_combo.isHidden())
            self.assertEqual(dialog.width_spin.value(), 2880)
            self.assertEqual(dialog.height_spin.value(), 2880)
            self.assertEqual(dialog.cx_spin.value(), 1440)
            self.assertEqual(dialog.cy_spin.value(), 1440)
            self.assertGreater(dialog.fx_spin.value(), 0)
            self.assertGreater(dialog.fy_spin.value(), 0)

            config = dialog.get_config()
            self.assertEqual(config.source_type, "insv")
            self.assertIsNone(config.source_stream_index)
            self.assertEqual(config.camera_model, "THIN_PRISM_FISHEYE")
            self.assertEqual(config.intrinsics["width"], 2880)
            self.assertEqual(config.intrinsics["sx1"], 0.0)
            self.assertEqual(config.intrinsics["sy1"], 0.0)

            dialog.insv_stream_combo.setCurrentIndex(1)
            self.assertEqual(dialog.get_config().source_stream_index, 0)
            dialog.insv_stream_combo.setCurrentIndex(2)
            self.assertEqual(dialog.get_config().source_stream_index, 1)
            dialog.close()

    def test_single_lens_insv_source_configures_single_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "single_lens.insv"
            source.write_bytes(b"mock insv")
            metadata = {
                "type": "insv",
                "width": 3072,
                "height": 3072,
                "lens_count": 1,
            }
            dialog = AddSourceDialog()
            with patch("raven_app.views.add_source_dialog.detect_source_metadata", return_value=metadata):
                dialog._set_source(str(source))

            self.assertEqual(dialog.insv_stream_combo.count(), 1)
            self.assertEqual(dialog.insv_stream_combo.currentIndex(), 0)
            self.assertEqual(dialog.get_config().source_stream_index, 0)
            dialog.close()

    def test_thin_prism_choice_and_coefficients_survive_config_edit(self):
        config = ThirdCameraConfig(
            source_path="phone.mp4",
            source_type="video",
            camera_model="THIN_PRISM_FISHEYE",
            intrinsics={
                "width": 1920, "height": 1080, "fx": 1500, "fy": 1490,
                "cx": 960, "cy": 540, "k1": 0.01, "k2": -0.02,
                "p1": 0.003, "p2": -0.004, "k3": 0.005, "k4": -0.006,
                "sx1": 0.007, "sy1": -0.008,
            },
        )
        dialog = AddSourceDialog(config=config)
        self.assertEqual(dialog.model_combo.currentText(), "THIN_PRISM_FISHEYE")
        self.assertFalse(dialog.distortion_spins["sx1"].isHidden())
        result = dialog.get_config()
        self.assertEqual(result.camera_model, "THIN_PRISM_FISHEYE")
        for name in ("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"):
            self.assertAlmostEqual(result.intrinsics[name], config.intrinsics[name])
        dialog.close()

    def test_both_views_allocate_stable_camera_names_and_save_bundle(self):
        for view_type in (ColorizeView, UnifiedWorkflowView):
            with self.subTest(view=view_type.__name__):
                view = view_type(ProcessRunner())
                first = ThirdCameraConfig(
                    camera_name=view._next_aux_camera_name(),
                    source_path="phone_a.mp4",
                )
                second = ThirdCameraConfig(
                    camera_name=view._next_aux_camera_name(),
                    source_path="phone_b.mp4",
                )
                disabled = ThirdCameraConfig(
                    camera_name=view._next_aux_camera_name(),
                    source_path="phone_c.mp4",
                    enabled=False,
                )
                view.third_camera_configs.extend((first, second, disabled))
                view._refresh_aux_sources_ui()

                self.assertEqual(
                    [cfg.camera_name for cfg in view.third_camera_configs],
                    ["cam2", "cam3", "cam4"],
                )
                self.assertEqual(view.aux_sources_layout.count(), 3)
                self.assertEqual(view._configured_aux_camera_names(), ["cam2", "cam3", "cam4"])

                view._remove_aux_source(0)
                self.assertEqual(view.third_camera_configs[0].camera_name, "cam3")
                self.assertEqual(view._next_aux_camera_name(), "cam5")

                with tempfile.TemporaryDirectory() as tmp:
                    target = Path(tmp) / "third_camera.json"
                    view._save_aux_camera_bundle(target)
                    bundle = json.loads(target.read_text(encoding="utf-8"))
                self.assertEqual([item["camera_name"] for item in bundle["cameras"]], ["cam3"])
                view.close()


class MaskCameraSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_unextracted_auxiliary_sources_appear_in_mask_settings(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog

        dialog = MaskSettingsDialog(
            None,
            camera_names=("cam2", "cam3", "cam2", ""),
        )
        cameras = [dialog.camera_combo.itemText(i) for i in range(dialog.camera_combo.count())]
        self.assertEqual(cameras, ["cam0", "cam1", "cam2", "cam3"])
        dialog.close()

    def test_selected_and_extracted_cameras_are_merged_once(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog

        with tempfile.TemporaryDirectory() as tmp:
            images = Path(tmp) / "images"
            (images / "cam2").mkdir(parents=True)
            (images / "cam10").mkdir()
            dialog = MaskSettingsDialog(
                tmp,
                camera_names=("cam2", "cam3"),
            )
            cameras = [dialog.camera_combo.itemText(i) for i in range(dialog.camera_combo.count())]
            self.assertEqual(cameras, ["cam0", "cam1", "cam2", "cam3", "cam10"])
            dialog.close()

    def test_auxiliary_frame_loads_when_primary_video_preview_is_lazy(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            insv = root / "main.insv"
            insv.write_bytes(b"not a real video")
            image_dir = root / "images" / "cam2"
            image_dir.mkdir(parents=True)
            image = image_dir / "frame_000001.jpg"
            cv2.imwrite(str(image), np.full((24, 32, 3), 120, np.uint8))
            dialog = MaskSettingsDialog(root, insv_path=insv, camera_names=("cam2",))
            dialog.camera_combo.setCurrentText("cam2")
            self.assertEqual(dialog._current_source, image)
            self.assertFalse(dialog.canvas.image.isNull())
            dialog.close()

    def test_per_camera_border_value_switches_and_persists(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog

        dialog = MaskSettingsDialog(
            None,
            config={"fisheye_border_percent": 2,
                    "fisheye_border_percent_by_camera": {"cam2": 9}},
            camera_names=("cam2",),
        )
        dialog.camera_combo.setCurrentText("cam2")
        self.assertEqual(dialog.border_spin.value(), 9)
        dialog.border_spin.setValue(6)
        dialog.camera_combo.setCurrentText("cam0")
        self.assertEqual(dialog.border_spin.value(), 2)
        dialog.camera_combo.setCurrentText("cam2")
        self.assertEqual(dialog.border_spin.value(), 6)
        self.assertEqual(dialog.settings()["fisheye_border_percent_by_camera"]["cam2"], 6)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
