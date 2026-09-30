"""Dialog for adding and configuring auxiliary image/video sources (third camera / smartphone)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    CaptionLabel,
    CardWidget,
    CheckBox,
    ComboBox,
    DoubleSpinBox,
    FluentIcon,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
    PushButton,
    SpinBox,
    SubtitleLabel,
    TitleLabel,
)
from scipy.spatial.transform import Rotation as Rot

from raven_app.i18n import tr
from raven_app.third_camera import (
    ThirdCameraConfig,
    default_smartphone_intrinsics,
    detect_source_metadata,
    load_third_camera_config,
    save_third_camera_config,
)


class AddSourceDialog(QDialog):
    """Configuration dialog for auxiliary camera (smartphone / third camera)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        config: Optional[Union[ThirdCameraConfig, Dict[str, Any], Path, str]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(tr("Add Source Image/Video"))
        self.resize(720, 780)
        self.setMinimumWidth(640)

        # Parse or default config
        if config is None:
            self._config = ThirdCameraConfig()
        elif isinstance(config, ThirdCameraConfig):
            self._config = config
        elif isinstance(config, (str, Path)):
            p = Path(config)
            if p.is_file() and p.suffix.lower() == ".json":
                self._config = load_third_camera_config(p)
            elif p.exists():
                self._config = ThirdCameraConfig(source_path=str(p))
            else:
                self._config = ThirdCameraConfig()
        elif isinstance(config, dict):
            self._config = ThirdCameraConfig.from_dict(config)
        else:
            self._config = ThirdCameraConfig()

        self._build_ui()
        self._load_from_config(self._config)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        # Title and description
        self.title_label = TitleLabel(tr("Add Source Image/Video"))
        self.desc_label = CaptionLabel(
            tr(
                "Configure an auxiliary camera source (smartphone video or image sequence) "
                "to supplement the dual fisheye cameras for high-resolution 3DGS multi-view coverage."
            )
        )
        self.desc_label.setWordWrap(True)
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.desc_label)

        # Scroll area for clean layout
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ----------------------------------------------------------------------
        # Card 1: Source Media Selection
        # ----------------------------------------------------------------------
        source_card = CardWidget(scroll_content)
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(16, 14, 16, 14)
        source_layout.setSpacing(10)

        source_title = SubtitleLabel(tr("1. Media Source (Video or Image Folder)"))
        source_layout.addWidget(source_title)

        path_row = QHBoxLayout()
        self.source_input = LineEdit()
        self.source_input.setPlaceholderText(
            tr("Select video file (.mp4, .mov) or folder containing extracted images...")
        )
        self.btn_browse_video = PushButton(tr("Browse Video"), icon=FluentIcon.VIDEO)
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.btn_browse_folder = PushButton(tr("Browse Folder"), icon=FluentIcon.FOLDER)
        self.btn_browse_folder.clicked.connect(self._browse_folder)

        path_row.addWidget(self.source_input)
        path_row.addWidget(self.btn_browse_video)
        path_row.addWidget(self.btn_browse_folder)
        source_layout.addLayout(path_row)

        self.meta_label = CaptionLabel(tr("No source selected."))
        self.meta_label.setStyleSheet("color: #0078D4; font-weight: bold;")
        source_layout.addWidget(self.meta_label)

        # Checkbox enable
        enable_row = QHBoxLayout()
        self.enabled_check = CheckBox(tr("Enable this auxiliary source in pipeline processing"))
        self.enabled_check.setChecked(True)
        enable_row.addWidget(self.enabled_check)
        source_layout.addLayout(enable_row)

        layout.addWidget(source_card)

        # ----------------------------------------------------------------------
        # Card 2: Acquisition & Synchronization
        # ----------------------------------------------------------------------
        sync_card = CardWidget(scroll_content)
        sync_layout = QVBoxLayout(sync_card)
        sync_layout.setContentsMargins(16, 14, 16, 14)
        sync_layout.setSpacing(10)

        sync_title = SubtitleLabel(tr("2. Timing & Frame Rate"))
        sync_layout.addWidget(sync_title)

        sync_grid = QGridLayout()
        sync_grid.setHorizontalSpacing(16)
        sync_grid.setVerticalSpacing(8)

        sync_grid.addWidget(BodyLabel(tr("Camera Identifier:")), 0, 0)
        self.cam_name_input = LineEdit()
        self.cam_name_input.setText("cam2")
        self.cam_name_input.setFixedWidth(140)
        sync_grid.addWidget(self.cam_name_input, 0, 1)

        sync_grid.addWidget(BodyLabel(tr("Extraction FPS:")), 0, 2)
        self.fps_spin = DoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(2.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setFixedWidth(120)
        sync_grid.addWidget(self.fps_spin, 0, 3)

        sync_grid.addWidget(BodyLabel(tr("Time Offset Δt (s):")), 1, 0)
        self.time_offset_spin = DoubleSpinBox()
        self.time_offset_spin.setRange(-3600.0, 3600.0)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setSingleStep(0.05)
        self.time_offset_spin.setDecimals(4)
        self.time_offset_spin.setFixedWidth(140)
        sync_grid.addWidget(self.time_offset_spin, 1, 1)

        sync_desc = CaptionLabel(
            tr("Offset shifts auxiliary timestamps relative to the primary rig clock.")
        )
        sync_grid.addWidget(sync_desc, 1, 2, 1, 2)

        sync_layout.addLayout(sync_grid)
        layout.addWidget(sync_card)

        # ----------------------------------------------------------------------
        # Card 3: Camera Model & Optical Intrinsics
        # ----------------------------------------------------------------------
        opt_card = CardWidget(scroll_content)
        opt_layout = QVBoxLayout(opt_card)
        opt_layout.setContentsMargins(16, 14, 16, 14)
        opt_layout.setSpacing(10)

        opt_title = SubtitleLabel(tr("3. Optical Intrinsics"))
        opt_layout.addWidget(opt_title)

        model_row = QHBoxLayout()
        model_row.addWidget(BodyLabel(tr("Camera Model:")))
        self.model_combo = ComboBox()
        self.model_combo.addItems(["PINHOLE", "OPENCV", "OPENCV_FISHEYE"])
        model_row.addWidget(self.model_combo)
        model_row.addStretch()

        self.btn_reset_intrinsics = PushButton(
            tr("Auto-Calculate from Resolution"), icon=FluentIcon.SYNC
        )
        self.btn_reset_intrinsics.clicked.connect(self._auto_calculate_intrinsics)
        model_row.addWidget(self.btn_reset_intrinsics)
        opt_layout.addLayout(model_row)

        intrin_grid = QGridLayout()
        intrin_grid.setHorizontalSpacing(16)
        intrin_grid.setVerticalSpacing(8)

        intrin_grid.addWidget(BodyLabel(tr("Width (px):")), 0, 0)
        self.width_spin = SpinBox()
        self.width_spin.setRange(100, 16384)
        self.width_spin.setValue(1920)
        intrin_grid.addWidget(self.width_spin, 0, 1)

        intrin_grid.addWidget(BodyLabel(tr("Height (px):")), 0, 2)
        self.height_spin = SpinBox()
        self.height_spin.setRange(100, 16384)
        self.height_spin.setValue(1080)
        intrin_grid.addWidget(self.height_spin, 0, 3)

        intrin_grid.addWidget(BodyLabel(tr("fx:")), 1, 0)
        self.fx_spin = DoubleSpinBox()
        self.fx_spin.setRange(1.0, 50000.0)
        self.fx_spin.setValue(1536.0)
        self.fx_spin.setDecimals(2)
        intrin_grid.addWidget(self.fx_spin, 1, 1)

        intrin_grid.addWidget(BodyLabel(tr("fy:")), 1, 2)
        self.fy_spin = DoubleSpinBox()
        self.fy_spin.setRange(1.0, 50000.0)
        self.fy_spin.setValue(1536.0)
        self.fy_spin.setDecimals(2)
        intrin_grid.addWidget(self.fy_spin, 1, 3)

        intrin_grid.addWidget(BodyLabel(tr("cx:")), 2, 0)
        self.cx_spin = DoubleSpinBox()
        self.cx_spin.setRange(0.0, 16384.0)
        self.cx_spin.setValue(960.0)
        self.cx_spin.setDecimals(2)
        intrin_grid.addWidget(self.cx_spin, 2, 1)

        intrin_grid.addWidget(BodyLabel(tr("cy:")), 2, 2)
        self.cy_spin = DoubleSpinBox()
        self.cy_spin.setRange(0.0, 16384.0)
        self.cy_spin.setValue(540.0)
        self.cy_spin.setDecimals(2)
        intrin_grid.addWidget(self.cy_spin, 2, 3)

        opt_layout.addLayout(intrin_grid)
        layout.addWidget(opt_card)

        # ----------------------------------------------------------------------
        # Card 4: Nominal Lever Arm / Extrinsics (LiDAR Body Frame)
        # ----------------------------------------------------------------------
        rig_card = CardWidget(scroll_content)
        rig_layout = QVBoxLayout(rig_card)
        rig_layout.setContentsMargins(16, 14, 16, 14)
        rig_layout.setSpacing(10)

        rig_title = SubtitleLabel(tr("4. Nominal Lever Arm & Extrinsics"))
        rig_layout.addWidget(rig_title)

        rig_hint = CaptionLabel(
            tr(
                "Nominal translation in LiDAR body frame: X (right), Y (forward), Z (up). "
                "Initial guess will be refined automatically via multi-camera SfM."
            )
        )
        rig_hint.setWordWrap(True)
        rig_layout.addWidget(rig_hint)

        extrin_grid = QGridLayout()
        extrin_grid.setHorizontalSpacing(16)
        extrin_grid.setVerticalSpacing(8)

        extrin_grid.addWidget(BodyLabel(tr("Translation X (m):")), 0, 0)
        self.tx_spin = DoubleSpinBox()
        self.tx_spin.setRange(-5.0, 5.0)
        self.tx_spin.setValue(0.0)
        self.tx_spin.setSingleStep(0.01)
        self.tx_spin.setDecimals(3)
        extrin_grid.addWidget(self.tx_spin, 0, 1)

        extrin_grid.addWidget(BodyLabel(tr("Translation Y (m):")), 0, 2)
        self.ty_spin = DoubleSpinBox()
        self.ty_spin.setRange(-5.0, 5.0)
        self.ty_spin.setValue(0.15)
        self.ty_spin.setSingleStep(0.01)
        self.ty_spin.setDecimals(3)
        extrin_grid.addWidget(self.ty_spin, 0, 3)

        extrin_grid.addWidget(BodyLabel(tr("Translation Z (m):")), 1, 0)
        self.tz_spin = DoubleSpinBox()
        self.tz_spin.setRange(-5.0, 5.0)
        self.tz_spin.setValue(0.20)
        self.tz_spin.setSingleStep(0.01)
        self.tz_spin.setDecimals(3)
        extrin_grid.addWidget(self.tz_spin, 1, 1)

        extrin_grid.addWidget(BodyLabel(tr("Euler Roll / Pitch / Yaw (°):")), 1, 2)
        rpy_box = QHBoxLayout()
        self.roll_spin = DoubleSpinBox()
        self.roll_spin.setRange(-180.0, 180.0)
        self.roll_spin.setValue(0.0)
        self.roll_spin.setToolTip(tr("Roll (deg)"))
        self.pitch_spin = DoubleSpinBox()
        self.pitch_spin.setRange(-180.0, 180.0)
        self.pitch_spin.setValue(0.0)
        self.pitch_spin.setToolTip(tr("Pitch (deg)"))
        self.yaw_spin = DoubleSpinBox()
        self.yaw_spin.setRange(-180.0, 180.0)
        self.yaw_spin.setValue(0.0)
        self.yaw_spin.setToolTip(tr("Yaw (deg)"))
        rpy_box.addWidget(self.roll_spin)
        rpy_box.addWidget(self.pitch_spin)
        rpy_box.addWidget(self.yaw_spin)
        extrin_grid.addLayout(rpy_box, 1, 3)

        rig_layout.addLayout(extrin_grid)
        layout.addWidget(rig_card)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # ----------------------------------------------------------------------
        # Footer Action Buttons
        # ----------------------------------------------------------------------
        footer = QHBoxLayout()
        self.btn_load_json = PushButton(tr("Load JSON..."), icon=FluentIcon.FOLDER)
        self.btn_load_json.clicked.connect(self._load_json)
        self.btn_save_json = PushButton(tr("Save JSON..."), icon=FluentIcon.SAVE)
        self.btn_save_json.clicked.connect(self._save_json)
        footer.addWidget(self.btn_load_json)
        footer.addWidget(self.btn_save_json)

        footer.addStretch()

        self.btn_cancel = PushButton(tr("Cancel"), icon=FluentIcon.CANCEL)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm = PrimaryPushButton(tr("Confirm"), icon=FluentIcon.ACCEPT)
        self.btn_confirm.clicked.connect(self._on_confirm)
        footer.addWidget(self.btn_cancel)
        footer.addWidget(self.btn_confirm)

        main_layout.addLayout(footer)

    def _browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Auxiliary Video File"),
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*.*)",
        )
        if file_path:
            self._set_source(file_path)

    def _browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            tr("Select Folder with Extracted Images"),
            "",
        )
        if folder_path:
            self._set_source(folder_path)

    def _set_source(self, path: str):
        self.source_input.setText(path)
        try:
            meta = detect_source_metadata(path)
            if meta["type"] == "video":
                info = (
                    f"Video: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps, "
                    f"duration {meta['duration_s']:.1f}s ({meta['frame_count']} frames)"
                )
            else:
                info = (
                    f"Image Folder: {meta['image_count']} images, {meta['width']}x{meta['height']}"
                )
            self.meta_label.setText(info)
            self.width_spin.setValue(meta["width"])
            self.height_spin.setValue(meta["height"])
            self._auto_calculate_intrinsics()
        except Exception as e:
            self.meta_label.setText(f"[!] {e}")

    def _auto_calculate_intrinsics(self):
        w = int(self.width_spin.value())
        h = int(self.height_spin.value())
        intrin = default_smartphone_intrinsics(w, h)
        self.fx_spin.setValue(intrin["fx"])
        self.fy_spin.setValue(intrin["fy"])
        self.cx_spin.setValue(intrin["cx"])
        self.cy_spin.setValue(intrin["cy"])

    def _load_from_config(self, cfg: ThirdCameraConfig):
        self.cam_name_input.setText(cfg.camera_name or "cam2")
        self.source_input.setText(cfg.source_path or "")
        self.enabled_check.setChecked(cfg.enabled)
        self.fps_spin.setValue(cfg.fps or 2.0)
        self.time_offset_spin.setValue(cfg.time_offset_s or 0.0)

        # Model
        idx = self.model_combo.findText(cfg.camera_model.upper())
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        # Intrinsics
        intr = cfg.intrinsics
        self.width_spin.setValue(int(intr.get("width", 1920)))
        self.height_spin.setValue(int(intr.get("height", 1080)))
        self.fx_spin.setValue(float(intr.get("fx", 1536.0)))
        self.fy_spin.setValue(float(intr.get("fy", 1536.0)))
        self.cx_spin.setValue(float(intr.get("cx", 960.0)))
        self.cy_spin.setValue(float(intr.get("cy", 540.0)))

        # Extrinsics
        T = np.array(cfg.T_lidar_to_cam2_rigid_4x4)
        if T.shape == (4, 4):
            self.tx_spin.setValue(float(T[0, 3]))
            self.ty_spin.setValue(float(T[1, 3]))
            self.tz_spin.setValue(float(T[2, 3]))
            rpy = Rot.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            self.roll_spin.setValue(float(rpy[0]))
            self.pitch_spin.setValue(float(rpy[1]))
            self.yaw_spin.setValue(float(rpy[2]))

        if cfg.source_path and Path(cfg.source_path).exists():
            try:
                meta = detect_source_metadata(cfg.source_path)
                if meta["type"] == "video":
                    self.meta_label.setText(
                        f"Video: {meta['width']}x{meta['height']} @ {meta['fps']:.1f} fps, "
                        f"{meta['duration_s']:.1f}s ({meta['frame_count']} frames)"
                    )
                else:
                    self.meta_label.setText(
                        f"Image Folder: {meta['image_count']} images, {meta['width']}x{meta['height']}"
                    )
            except Exception:
                pass

    def get_config(self) -> ThirdCameraConfig:
        """Construct ThirdCameraConfig from current dialog inputs."""
        src_path = self.source_input.text().strip()
        src_type = "video"
        if src_path and Path(src_path).is_dir():
            src_type = "folder"

        # Build 4x4 rigid transformation
        R = Rot.from_euler(
            "xyz",
            [self.roll_spin.value(), self.pitch_spin.value(), self.yaw_spin.value()],
            degrees=True,
        ).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[0, 3] = self.tx_spin.value()
        T[1, 3] = self.ty_spin.value()
        T[2, 3] = self.tz_spin.value()

        intrinsics = {
            "width": float(self.width_spin.value()),
            "height": float(self.height_spin.value()),
            "fx": float(self.fx_spin.value()),
            "fy": float(self.fy_spin.value()),
            "cx": float(self.cx_spin.value()),
            "cy": float(self.cy_spin.value()),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        }

        return ThirdCameraConfig(
            camera_name=self.cam_name_input.text().strip() or "cam2",
            source_path=src_path,
            source_type=src_type,
            enabled=self.enabled_check.isChecked(),
            fps=float(self.fps_spin.value()),
            time_offset_s=float(self.time_offset_spin.value()),
            camera_model=self.model_combo.currentText(),
            intrinsics=intrinsics,
            T_lidar_to_cam2_rigid_4x4=T.tolist(),
            description=f"Auxiliary source ({self.cam_name_input.text().strip()})",
        )

    def _load_json(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, tr("Load Third Camera Config"), "", "JSON files (*.json);;All files (*.*)"
        )
        if file_path:
            try:
                cfg = load_third_camera_config(file_path)
                self._load_from_config(cfg)
                InfoBar.success(
                    title=tr("Config Loaded"),
                    content=tr("Auxiliary camera configuration successfully loaded."),
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=2500,
                )
            except Exception as e:
                InfoBar.error(
                    title=tr("Error"),
                    content=str(e),
                    parent=self,
                    position=InfoBarPosition.TOP,
                )

    def _save_json(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, tr("Save Third Camera Config"), "third_camera.json", "JSON files (*.json)"
        )
        if file_path:
            try:
                cfg = self.get_config()
                save_third_camera_config(cfg, file_path)
                InfoBar.success(
                    title=tr("Config Saved"),
                    content=tr("Auxiliary camera configuration saved successfully."),
                    parent=self,
                    position=InfoBarPosition.TOP,
                    duration=2500,
                )
            except Exception as e:
                InfoBar.error(
                    title=tr("Error"),
                    content=str(e),
                    parent=self,
                    position=InfoBarPosition.TOP,
                )

    def _on_confirm(self):
        cfg = self.get_config()
        if cfg.enabled and not cfg.source_path:
            InfoBar.warning(
                title=tr("Missing Media Source"),
                content=tr("Please select a video file or image folder, or disable auxiliary source."),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3500,
            )
            return
        self._config = cfg
        self.accept()
