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
    QSizePolicy,
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
        self.resize(720, 760)
        self.setMinimumWidth(520)

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
                "Configure an auxiliary camera source (smartphone or INSV video, or image sequence) "
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        scroll_content.setMinimumWidth(0)
        scroll_content.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # ----------------------------------------------------------------------
        # Card 1: Source Media Selection
        # ----------------------------------------------------------------------
        source_card = CardWidget(scroll_content)
        source_layout = QVBoxLayout(source_card)
        source_layout.setContentsMargins(12, 10, 12, 10)
        source_layout.setSpacing(10)

        source_title = SubtitleLabel(tr("1. Media Source (Video, INSV, or Image Folder)"))
        source_title.setWordWrap(True)
        source_layout.addWidget(source_title)

        path_layout = QVBoxLayout()
        path_layout.setSpacing(8)
        self.source_input = LineEdit()
        self.source_input.setPlaceholderText(
            tr("Select video file (.mp4, .mov) or folder containing extracted images...")
        )
        path_layout.addWidget(self.source_input)

        browse_buttons = QVBoxLayout()
        browse_buttons.setSpacing(8)
        self.btn_browse_video = PushButton(tr("Browse Video"), icon=FluentIcon.VIDEO)
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.btn_browse_insv = PushButton(tr("Browse INSV"), icon=FluentIcon.VIDEO)
        self.btn_browse_insv.clicked.connect(self._browse_insv)
        self.btn_browse_folder = PushButton(tr("Browse Folder"), icon=FluentIcon.FOLDER)
        self.btn_browse_folder.clicked.connect(self._browse_folder)
        browse_buttons.addWidget(self.btn_browse_video, 0, Qt.AlignmentFlag.AlignLeft)
        browse_buttons.addWidget(self.btn_browse_insv, 0, Qt.AlignmentFlag.AlignLeft)
        browse_buttons.addWidget(self.btn_browse_folder, 0, Qt.AlignmentFlag.AlignLeft)
        path_layout.addLayout(browse_buttons)
        source_layout.addLayout(path_layout)

        self.meta_label = CaptionLabel(tr("No source selected."))
        self.meta_label.setWordWrap(True)
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
        sync_layout.setContentsMargins(12, 10, 12, 10)
        sync_layout.setSpacing(10)

        sync_title = SubtitleLabel(tr("2. Timing & Frame Rate"))
        sync_title.setWordWrap(True)
        sync_layout.addWidget(sync_title)

        sync_grid = QGridLayout()
        sync_grid.setHorizontalSpacing(8)
        sync_grid.setVerticalSpacing(8)

        camera_label = BodyLabel(tr("Camera Identifier:"))
        camera_label.setWordWrap(True)
        sync_grid.addWidget(camera_label, 0, 0)
        fps_label = BodyLabel(tr("Extraction FPS:"))
        fps_label.setWordWrap(True)
        sync_grid.addWidget(fps_label, 0, 1)
        self.cam_name_input = LineEdit()
        self.cam_name_input.setText("cam2")
        self.cam_name_input.setMaximumWidth(180)
        sync_grid.addWidget(self.cam_name_input, 1, 0)

        self.fps_spin = DoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(2.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setMaximumWidth(150)
        sync_grid.addWidget(self.fps_spin, 1, 1)

        offset_label = BodyLabel(tr("Time Offset Δt (s):"))
        offset_label.setWordWrap(True)
        sync_grid.addWidget(offset_label, 2, 0, 1, 2)
        self.time_offset_spin = DoubleSpinBox()
        self.time_offset_spin.setRange(-3600.0, 3600.0)
        self.time_offset_spin.setValue(0.0)
        self.time_offset_spin.setSingleStep(0.05)
        self.time_offset_spin.setDecimals(4)
        self.time_offset_spin.setMaximumWidth(170)
        sync_grid.addWidget(self.time_offset_spin, 3, 0)

        sync_desc = CaptionLabel(
            tr("Offset shifts auxiliary timestamps relative to the primary rig clock.")
        )
        sync_desc.setWordWrap(True)
        sync_grid.addWidget(sync_desc, 3, 1)

        sync_layout.addLayout(sync_grid)

        self.insv_stream_label = BodyLabel(tr("INSV lens streams:"))
        self.insv_stream_combo = ComboBox()
        self.insv_stream_combo.addItems([
            tr("Both lens streams (recommended)"),
            tr("Lens stream 0"),
            tr("Lens stream 1"),
        ])
        self.insv_stream_combo.setToolTip(
            tr("Use both lenses from this INSV source, or select one lens stream.")
        )
        self.insv_stream_label.setVisible(False)
        self.insv_stream_combo.setVisible(False)
        stream_row = QHBoxLayout()
        stream_row.addWidget(self.insv_stream_label)
        stream_row.addWidget(self.insv_stream_combo, 1)
        sync_layout.addLayout(stream_row)
        layout.addWidget(sync_card)

        # ----------------------------------------------------------------------
        # Card 3: Camera Model & Optical Intrinsics
        # ----------------------------------------------------------------------
        opt_card = CardWidget(scroll_content)
        opt_layout = QVBoxLayout(opt_card)
        opt_layout.setContentsMargins(12, 10, 12, 10)
        opt_layout.setSpacing(10)

        opt_title = SubtitleLabel(tr("3. Optical Intrinsics"))
        opt_title.setWordWrap(True)
        opt_layout.addWidget(opt_title)

        model_row = QHBoxLayout()
        model_row.addWidget(BodyLabel(tr("Camera Model:")))
        self.model_combo = ComboBox()
        self.model_combo.addItems(
            ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"]
        )
        self.model_combo.setMaximumWidth(190)
        self.model_combo.currentTextChanged.connect(self._update_distortion_visibility)
        model_row.addWidget(self.model_combo)
        model_row.addStretch()

        self.btn_reset_intrinsics = PushButton(
            tr("Auto-Calculate"), icon=FluentIcon.SYNC
        )
        self.btn_reset_intrinsics.setToolTip(
            tr("Auto-calculate intrinsics from the image resolution.")
        )
        self.btn_reset_intrinsics.clicked.connect(self._auto_calculate_intrinsics)
        opt_layout.addLayout(model_row)
        opt_layout.addWidget(
            self.btn_reset_intrinsics, 0, Qt.AlignmentFlag.AlignRight
        )

        intrin_grid = QGridLayout()
        intrin_grid.setHorizontalSpacing(8)
        intrin_grid.setVerticalSpacing(8)

        intrin_grid.addWidget(BodyLabel(tr("Width (px):")), 0, 0)
        intrin_grid.addWidget(BodyLabel(tr("Height (px):")), 0, 1)
        self.width_spin = SpinBox()
        self.width_spin.setRange(100, 16384)
        self.width_spin.setValue(1920)
        self.width_spin.setMaximumWidth(160)
        intrin_grid.addWidget(self.width_spin, 1, 0)

        self.height_spin = SpinBox()
        self.height_spin.setRange(100, 16384)
        self.height_spin.setValue(1080)
        self.height_spin.setMaximumWidth(160)
        intrin_grid.addWidget(self.height_spin, 1, 1)

        intrin_grid.addWidget(BodyLabel(tr("fx:")), 2, 0)
        intrin_grid.addWidget(BodyLabel(tr("fy:")), 2, 1)
        self.fx_spin = DoubleSpinBox()
        self.fx_spin.setRange(1.0, 50000.0)
        self.fx_spin.setValue(1536.0)
        self.fx_spin.setDecimals(2)
        self.fx_spin.setMaximumWidth(160)
        intrin_grid.addWidget(self.fx_spin, 3, 0)

        self.fy_spin = DoubleSpinBox()
        self.fy_spin.setRange(1.0, 50000.0)
        self.fy_spin.setValue(1536.0)
        self.fy_spin.setDecimals(2)
        self.fy_spin.setMaximumWidth(160)
        intrin_grid.addWidget(self.fy_spin, 3, 1)

        intrin_grid.addWidget(BodyLabel(tr("cx:")), 4, 0)
        intrin_grid.addWidget(BodyLabel(tr("cy:")), 4, 1)
        self.cx_spin = DoubleSpinBox()
        self.cx_spin.setRange(0.0, 16384.0)
        self.cx_spin.setValue(960.0)
        self.cx_spin.setDecimals(2)
        self.cx_spin.setMaximumWidth(160)
        intrin_grid.addWidget(self.cx_spin, 5, 0)

        self.cy_spin = DoubleSpinBox()
        self.cy_spin.setRange(0.0, 16384.0)
        self.cy_spin.setValue(540.0)
        self.cy_spin.setDecimals(2)
        self.cy_spin.setMaximumWidth(160)
        intrin_grid.addWidget(self.cy_spin, 5, 1)

        opt_layout.addLayout(intrin_grid)

        distortion_title = CaptionLabel(tr("Distortion coefficients (used by the selected model):"))
        distortion_title.setWordWrap(True)
        opt_layout.addWidget(distortion_title)
        distortion_grid = QGridLayout()
        distortion_grid.setHorizontalSpacing(8)
        distortion_grid.setVerticalSpacing(5)
        self.distortion_spins = {}
        self._distortion_labels = {}
        coefficient_names = ("k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
        for index, name in enumerate(coefficient_names):
            row, pair = divmod(index, 2)
            label_col = pair * 2
            label = BodyLabel(f"{name}:")
            spin = DoubleSpinBox()
            spin.setRange(-10.0, 10.0)
            spin.setDecimals(8)
            spin.setSingleStep(0.001)
            spin.setMaximumWidth(120)
            distortion_grid.addWidget(label, row, label_col)
            distortion_grid.addWidget(spin, row, label_col + 1)
            self._distortion_labels[name] = label
            self.distortion_spins[name] = spin
        opt_layout.addLayout(distortion_grid)
        self._update_distortion_visibility(self.model_combo.currentText())
        layout.addWidget(opt_card)

        # ----------------------------------------------------------------------
        # Card 4: Nominal Lever Arm / Extrinsics (LiDAR Body Frame)
        # ----------------------------------------------------------------------
        rig_card = CardWidget(scroll_content)
        rig_layout = QVBoxLayout(rig_card)
        rig_layout.setContentsMargins(12, 10, 12, 10)
        rig_layout.setSpacing(10)

        rig_title = SubtitleLabel(tr("4. Nominal Lever Arm & Extrinsics"))
        rig_title.setWordWrap(True)
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
        extrin_grid.setHorizontalSpacing(8)
        extrin_grid.setVerticalSpacing(8)

        extrin_grid.addWidget(BodyLabel(tr("Translation X (m):")), 0, 0)
        self.tx_spin = DoubleSpinBox()
        self.tx_spin.setRange(-5.0, 5.0)
        self.tx_spin.setValue(0.0)
        self.tx_spin.setSingleStep(0.01)
        self.tx_spin.setDecimals(3)
        self.tx_spin.setMaximumWidth(160)
        extrin_grid.addWidget(self.tx_spin, 0, 1)

        extrin_grid.addWidget(BodyLabel(tr("Translation Y (m):")), 1, 0)
        self.ty_spin = DoubleSpinBox()
        self.ty_spin.setRange(-5.0, 5.0)
        self.ty_spin.setValue(0.15)
        self.ty_spin.setSingleStep(0.01)
        self.ty_spin.setDecimals(3)
        self.ty_spin.setMaximumWidth(160)
        extrin_grid.addWidget(self.ty_spin, 1, 1)

        extrin_grid.addWidget(BodyLabel(tr("Translation Z (m):")), 2, 0)
        self.tz_spin = DoubleSpinBox()
        self.tz_spin.setRange(-5.0, 5.0)
        self.tz_spin.setValue(0.20)
        self.tz_spin.setSingleStep(0.01)
        self.tz_spin.setDecimals(3)
        self.tz_spin.setMaximumWidth(160)
        extrin_grid.addWidget(self.tz_spin, 2, 1)

        extrin_grid.addWidget(
            BodyLabel(tr("Euler Roll / Pitch / Yaw (°):")), 3, 0, 1, 2
        )
        self.roll_spin = DoubleSpinBox()
        self.roll_spin.setRange(-180.0, 180.0)
        self.roll_spin.setValue(0.0)
        self.roll_spin.setMaximumWidth(160)
        self.roll_spin.setToolTip(tr("Roll (deg)"))
        self.pitch_spin = DoubleSpinBox()
        self.pitch_spin.setRange(-180.0, 180.0)
        self.pitch_spin.setValue(0.0)
        self.pitch_spin.setMaximumWidth(160)
        self.pitch_spin.setToolTip(tr("Pitch (deg)"))
        self.yaw_spin = DoubleSpinBox()
        self.yaw_spin.setRange(-180.0, 180.0)
        self.yaw_spin.setValue(0.0)
        self.yaw_spin.setMaximumWidth(160)
        self.yaw_spin.setToolTip(tr("Yaw (deg)"))
        # Two controls per row keep each spin box wide enough to read without
        # forcing a third column beyond the visible card width.
        extrin_grid.addWidget(self.roll_spin, 4, 0)
        extrin_grid.addWidget(self.pitch_spin, 4, 1)
        extrin_grid.addWidget(self.yaw_spin, 5, 0)

        rig_layout.addLayout(extrin_grid)
        layout.addWidget(rig_card)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # ----------------------------------------------------------------------
        # Footer Action Buttons
        # ----------------------------------------------------------------------
        footer = QVBoxLayout()
        footer.setSpacing(8)
        config_buttons = QHBoxLayout()
        config_buttons.setSpacing(8)
        self.btn_load_json = PushButton(tr("Load JSON..."), icon=FluentIcon.FOLDER)
        self.btn_load_json.clicked.connect(self._load_json)
        self.btn_save_json = PushButton(tr("Save JSON..."), icon=FluentIcon.SAVE)
        self.btn_save_json.clicked.connect(self._save_json)
        config_buttons.addWidget(self.btn_load_json)
        config_buttons.addWidget(self.btn_save_json)
        config_buttons.addStretch(1)
        footer.addLayout(config_buttons)

        action_buttons = QHBoxLayout()
        action_buttons.addStretch(1)
        self.btn_cancel = PushButton(tr("Cancel"), icon=FluentIcon.CANCEL)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm = PrimaryPushButton(tr("Confirm"), icon=FluentIcon.ACCEPT)
        self.btn_confirm.clicked.connect(self._on_confirm)
        action_buttons.addWidget(self.btn_cancel)
        action_buttons.addWidget(self.btn_confirm)
        footer.addLayout(action_buttons)

        main_layout.addLayout(footer)

    def _browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Auxiliary Video File"),
            "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.m4v *.insv);;All files (*.*)",
        )
        if file_path:
            self._set_source(file_path)

    def _browse_insv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr("Select Auxiliary Insta360 INSV File"),
            "",
            "Insta360 videos (*.insv);;All files (*.*)",
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

    def _update_insv_stream_combo(self, lens_count: int, selected_stream: Optional[int] = None):
        self.insv_stream_combo.blockSignals(True)
        self.insv_stream_combo.clear()
        if lens_count <= 1:
            self.insv_stream_combo.addItem(tr("Lens stream 0"))
            self.insv_stream_combo.setCurrentIndex(0)
        else:
            self.insv_stream_combo.addItems([
                tr("Both lens streams (recommended)"),
                tr("Lens stream 0"),
                tr("Lens stream 1"),
            ])
            if selected_stream is None:
                self.insv_stream_combo.setCurrentIndex(0)
            elif selected_stream in (0, 1):
                self.insv_stream_combo.setCurrentIndex(selected_stream + 1)
            else:
                self.insv_stream_combo.setCurrentIndex(0)
        self.insv_stream_combo.blockSignals(False)

    def _set_source(self, path: str):
        self.source_input.setText(path)
        try:
            meta = detect_source_metadata(path)
            suffix = Path(path).suffix.lower()
            is_insv = suffix == ".insv" or str(meta.get("type", "")).lower() == "insv"
            self.insv_stream_label.setVisible(is_insv)
            self.insv_stream_combo.setVisible(is_insv)
            if is_insv:
                self.model_combo.setCurrentText("THIN_PRISM_FISHEYE")
                streams = int(meta.get("lens_count", 2) or 2)
                self._update_insv_stream_combo(streams)
                info = (
                    f"INSV: {meta['width']}x{meta['height']}, "
                    f"{streams} lens streams"
                )
            elif meta["type"] == "video":
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
            self._auto_calculate_intrinsics(meta=meta, is_insv=is_insv)
        except Exception as e:
            self.meta_label.setText(f"[!] {e}")

    def _update_distortion_visibility(self, model: str):
        model = str(model or "PINHOLE").strip().upper()
        if model == "THIN_PRISM_FISHEYE":
            visible = {"k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1"}
        elif model == "OPENCV_FISHEYE":
            visible = {"k1", "k2", "k3", "k4"}
        elif model == "OPENCV":
            visible = {"k1", "k2", "p1", "p2"}
        else:
            visible = set()
        for name, spin in self.distortion_spins.items():
            spin.setVisible(name in visible)
            self._distortion_labels[name].setVisible(name in visible)

    def _auto_calculate_intrinsics(self, *, meta=None, is_insv=False):
        w = int(self.width_spin.value())
        h = int(self.height_spin.value())
        intrin = default_smartphone_intrinsics(w, h)
        supplied = {}
        if isinstance(meta, dict):
            candidate = meta.get("intrinsics")
            if isinstance(candidate, dict):
                supplied.update(candidate)
            supplied.update({
                key: meta[key]
                for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "sx1", "sy1")
                if key in meta
            })
        for key in ("fx", "fy", "cx", "cy"):
            if key in supplied:
                intrin[key] = float(supplied[key])
        self.fx_spin.setValue(intrin["fx"])
        self.fy_spin.setValue(intrin["fy"])
        self.cx_spin.setValue(intrin["cx"])
        self.cy_spin.setValue(intrin["cy"])
        for name, spin in self.distortion_spins.items():
            spin.setValue(float(supplied.get(name, 0.0)))

    def _load_from_config(self, cfg: ThirdCameraConfig):
        self.cam_name_input.setText(cfg.camera_name or "cam2")
        self.source_input.setText(cfg.source_path or "")
        source_is_insv = (
            str(cfg.source_type or "").lower() == "insv"
            or Path(cfg.source_path or "").suffix.lower() == ".insv"
        )
        self.insv_stream_label.setVisible(source_is_insv)
        self.insv_stream_combo.setVisible(source_is_insv)
        stream_index = getattr(cfg, "source_stream_index", None)
        try:
            stream_index = None if stream_index is None else int(stream_index)
        except (TypeError, ValueError):
            stream_index = None
        self.insv_stream_combo.setCurrentIndex(
            0 if stream_index not in (0, 1) else stream_index + 1
        )
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
        for name, spin in self.distortion_spins.items():
            spin.setValue(float(intr.get(name, 0.0)))
        self._update_distortion_visibility(self.model_combo.currentText())

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
                if (Path(cfg.source_path).suffix.lower() == ".insv"
                        or str(meta.get("type", "")).lower() == "insv"):
                    streams = int(meta.get("lens_count", 2) or 2)
                    self.meta_label.setText(
                        f"INSV: {meta['width']}x{meta['height']}, "
                        f"{streams} lens streams"
                    )
                    self.insv_stream_label.setVisible(True)
                    self.insv_stream_combo.setVisible(True)
                    self._update_insv_stream_combo(streams, stream_index)
                elif meta["type"] == "video":
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
        elif src_path and Path(src_path).suffix.lower() == ".insv":
            src_type = "insv"

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
        intrinsics.update({
            name: float(spin.value())
            for name, spin in self.distortion_spins.items()
        })

        config = ThirdCameraConfig(
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
        if src_type != "insv":
            stream_index = None
        elif self.insv_stream_combo.count() == 1:
            stream_index = 0
        elif self.insv_stream_combo.currentIndex() == 0:
            stream_index = None
        else:
            stream_index = self.insv_stream_combo.currentIndex() - 1
        config.source_stream_index = stream_index
        return config

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
                content=tr("Please select a video, INSV file, or image folder, or disable auxiliary source."),
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3500,
            )
            return
        self._config = cfg
        self.accept()
