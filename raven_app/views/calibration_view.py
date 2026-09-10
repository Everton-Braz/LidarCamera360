"""Calibration and rig geometry view."""
import json
import math
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QSizePolicy
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PushButton,
    PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition
)
from raven_app.i18n import tr
from raven_app.config import get_app_root


class CalibrationView(QWidget):
    """WinUI Fluent view for inspecting and editing LiDAR-Camera calibration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CalibrationView")
        self._calib_path = get_app_root() / "calibracao_rigida_raven_insta360.json"
        self._init_ui()
        self._load_calibration()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        self.header_title = TitleLabel(tr("LiDAR-Camera Rig Calibration"))
        self.header_subtitle = CaptionLabel(
            tr("Spatial rigid lever-arm extrinsics & Thin Prism dual-fisheye optical models")
        )
        header_layout.addWidget(self.header_title)
        header_layout.addWidget(self.header_subtitle)
        layout.addLayout(header_layout)

        # Summary Metrics Row
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(16)

        # Card 1: Hardware Rig
        rig_card = CardWidget(self)
        rig_layout = QVBoxLayout(rig_card)
        rig_layout.setContentsMargins(16, 14, 16, 14)
        self.title_hardware = SubtitleLabel(tr("Sensor Hardware"))
        rig_layout.addWidget(self.title_hardware)
        self.lbl_scanner = StrongBodyLabel("LiDAR: 3DMakerPro Raven")
        self.lbl_camera = StrongBodyLabel("Camera: Insta360 X4 Dual Fisheye")
        self.lbl_optics = CaptionLabel("Model: THIN_PRISM_FISHEYE (OpenCV)")
        rig_layout.addWidget(self.lbl_scanner)
        rig_layout.addWidget(self.lbl_camera)
        rig_layout.addWidget(self.lbl_optics)
        metrics_row.addWidget(rig_card)

        # Card 2: Lever-Arm Offsets
        lever_card = CardWidget(self)
        lever_layout = QVBoxLayout(lever_card)
        lever_layout.setContentsMargins(16, 14, 16, 14)
        self.title_lever = SubtitleLabel(tr("Rigid Lever-Arm"))
        lever_layout.addWidget(self.title_lever)
        self.lbl_lever_cam0 = StrongBodyLabel()
        self.lbl_lever_cam1 = StrongBodyLabel()
        self.lbl_norm_dist = CaptionLabel()
        lever_layout.addWidget(self.lbl_lever_cam0)
        lever_layout.addWidget(self.lbl_lever_cam1)
        lever_layout.addWidget(self.lbl_norm_dist)
        metrics_row.addWidget(lever_card)

        # Card 3: Rotation & Angles
        rot_card = CardWidget(self)
        rot_layout = QVBoxLayout(rot_card)
        rot_layout.setContentsMargins(16, 14, 16, 14)
        self.title_rot = SubtitleLabel(tr("Orientation & Sync"))
        rot_layout.addWidget(self.title_rot)
        self.lbl_lens_angle = StrongBodyLabel()
        self.lbl_euler_cam0 = CaptionLabel()
        self.lbl_euler_cam1 = CaptionLabel()
        self.lbl_capture_offset = CaptionLabel()
        rot_layout.addWidget(self.lbl_lens_angle)
        rot_layout.addWidget(self.lbl_euler_cam0)
        rot_layout.addWidget(self.lbl_euler_cam1)
        rot_layout.addWidget(self.lbl_capture_offset)
        metrics_row.addWidget(rot_card)

        for card in (rig_card, lever_card, rot_card):
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        for label in (self.lbl_scanner, self.lbl_camera, self.lbl_optics,
                      self.lbl_lever_cam0, self.lbl_lever_cam1, self.lbl_norm_dist,
                      self.lbl_lens_angle, self.lbl_euler_cam0, self.lbl_euler_cam1,
                      self.lbl_capture_offset):
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        layout.addLayout(metrics_row)

        # JSON Configuration Card
        json_card = CardWidget(self)
        json_layout = QVBoxLayout(json_card)
        json_layout.setContentsMargins(16, 14, 16, 14)
        json_layout.setSpacing(10)

        json_title = SubtitleLabel("Full Calibration Configuration (JSON)")
        json_title.setWordWrap(True)
        json_title.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        json_layout.addWidget(json_title)

        json_actions = QHBoxLayout()
        json_actions.setSpacing(8)
        self.btn_reload = PushButton("Reload", icon=FluentIcon.SYNC)
        self.btn_reload.clicked.connect(self._load_calibration)
        self.btn_save = PushButton("Save Changes", icon=FluentIcon.SAVE)
        self.btn_save.clicked.connect(self._save_calibration)
        self.btn_export = PushButton("Export As...", icon=FluentIcon.SHARE)
        self.btn_export.clicked.connect(self._export_calibration)

        json_actions.addWidget(self.btn_reload)
        json_actions.addWidget(self.btn_save)
        json_actions.addWidget(self.btn_export)
        json_actions.addStretch()
        json_layout.addLayout(json_actions)

        self.json_edit = PlainTextEdit()
        self.json_edit.setStyleSheet(
            "PlainTextEdit { background-color: #0c1219; color: #a9c7df; font-family: 'Consolas', 'Courier New'; font-size: 12px; border-radius: 6px; padding: 8px; }"
        )
        json_layout.addWidget(self.json_edit)
        layout.addWidget(json_card, stretch=1)

    def _load_calibration(self):
        if self._calib_path.is_file():
            try:
                content = self._calib_path.read_text(encoding='utf-8')
                data = json.loads(content)
                self.json_edit.setPlainText(json.dumps(data, indent=2))
                self.lbl_scanner.setText(f"LiDAR: {str(data.get('scanner', 'Raven')).replace('_', ' ')}")
                self.lbl_camera.setText(f"Camera: {str(data.get('camera', 'Insta360 X4')).replace('_', ' ')}")
                self.lbl_optics.setText(f"Lens Model: {data.get('lens_model', 'THIN_PRISM_FISHEYE')}")
                self._refresh_summary(data)
            except Exception as e:
                self.json_edit.setPlainText(f"Error loading calibration: {e}")

    @staticmethod
    def _transform_summary(data, camera):
        matrix = data.get(f"T_lidar_to_{camera}_rigid_4x4")
        if not isinstance(matrix, list) or len(matrix) < 3 or any(len(row) < 4 for row in matrix[:3]):
            raise ValueError(f"missing transform for {camera}")
        r = [row[:3] for row in matrix[:3]]
        t = [float(matrix[i][3]) for i in range(3)]
        norm = math.sqrt(sum(value * value for value in t)) * 100.0
        # Matches the calibration pipeline's rotation_euler_xyz_deg convention.
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[2][0]))))
        roll = math.degrees(math.atan2(r[2][1], r[2][2]))
        yaw = math.degrees(math.atan2(r[1][0], r[0][0]))
        return t, norm, (roll, pitch, yaw), r

    def _refresh_summary(self, data):
        try:
            c0, c1 = (self._transform_summary(data, camera) for camera in ("cam0", "cam1"))
            self.lbl_lever_cam0.setText(
                f"Cam0: ΔX={c0[0][0] * 100:+.2f} cm, ΔY={c0[0][1] * 100:+.2f} cm, ΔZ={c0[0][2] * 100:+.2f} cm"
            )
            self.lbl_lever_cam1.setText(
                f"Cam1: ΔX={c1[0][0] * 100:+.2f} cm, ΔY={c1[0][1] * 100:+.2f} cm, ΔZ={c1[0][2] * 100:+.2f} cm"
            )
            self.lbl_norm_dist.setText(f"Lever-arm norms: Cam0 {c0[1]:.2f} cm · Cam1 {c1[1]:.2f} cm")
            self.lbl_euler_cam0.setText("Cam0 Euler XYZ: " + self._format_euler(c0[2]))
            self.lbl_euler_cam1.setText("Cam1 Euler XYZ: " + self._format_euler(c1[2]))
            relative = [[sum(c1[3][i][k] * c0[3][j][k] for k in range(3)) for j in range(3)] for i in range(3)]
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (sum(relative[i][i] for i in range(3)) - 1.0) / 2.0))))
            self.lbl_lens_angle.setText(f"Relative lens angle: {angle:.2f}°")
            reference = data.get("reference_capture")
            offset = reference.get("dt_sync_seconds") if isinstance(reference, dict) else data.get("dt_sync_seconds")
            label = "Reference capture offset" if isinstance(reference, dict) else "Capture time offset"
            self.lbl_capture_offset.setText(f"{label}: {float(offset):+.3f} s" if offset is not None else f"{label}: unavailable")
        except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
            self.lbl_lens_angle.setText(f"Transform summary unavailable: {exc}")

    @staticmethod
    def _format_euler(values):
        return ", ".join(f"{axis} {value:+.2f}°" for axis, value in zip(("R", "P", "Y"), values))

    def _save_calibration(self):
        text = self.json_edit.toPlainText().strip()
        try:
            data = json.loads(text)
            self._calib_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            self._refresh_summary(data)
            InfoBar.success(
                title="Saved",
                content="Calibration parameters successfully saved.",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
        except json.JSONDecodeError as e:
            InfoBar.error(
                title="JSON Syntax Error",
                content=f"Invalid JSON format: {e}",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self
            )
        except OSError as e:
            InfoBar.error(title="Save Error", content=f"Could not write calibration: {e}",
                          orient=Qt.Orientation.Horizontal, position=InfoBarPosition.TOP,
                          duration=5000, parent=self)

    def _export_calibration(self):
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Calibration", "calibration.json", "JSON files (*.json)"
        )
        if dest:
            Path(dest).write_text(self.json_edit.toPlainText(), encoding='utf-8')
            InfoBar.success(
                title="Exported",
                content=f"Calibration exported to {dest}",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )

    def retranslate_ui(self):
        """Update all text elements dynamically when the language changes."""
        self.header_title.setText(tr("LiDAR-Camera Rig Calibration"))
        self.header_subtitle.setText(
            tr("Spatial rigid lever-arm extrinsics & Thin Prism dual-fisheye optical models")
        )
        self.title_hardware.setText(tr("Sensor Hardware"))
        self.title_lever.setText(tr("Rigid Lever-Arm"))
        self.title_rot.setText(tr("Orientation & Sync"))
