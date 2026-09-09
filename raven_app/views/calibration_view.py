"""Calibration and rig geometry view."""
import json
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QFrame
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PushButton,
    PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition
)
from raven_app.i18n import tr


class CalibrationView(QWidget):
    """WinUI Fluent view for inspecting and editing LiDAR-Camera calibration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CalibrationView")
        self._calib_path = Path(__file__).resolve().parents[2] / "calibracao_rigida_raven_insta360.json"
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
        self.lbl_lever_cam0 = StrongBodyLabel("Cam0: ΔX=+0.58cm, ΔY=+13.53cm, ΔZ=+9.20cm")
        self.lbl_lever_cam1 = StrongBodyLabel("Cam1: ΔX=-1.77cm, ΔY=+13.71cm, ΔZ=+9.07cm")
        self.lbl_norm_dist = CaptionLabel("Norm baseline distance: ~16.4 cm")
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
        self.lbl_lens_angle = StrongBodyLabel("Dual Lens Angle: 179.87°")
        self.lbl_euler_cam0 = CaptionLabel("Cam0 Euler: R -87.86°, P -31.57°, Y -94.85°")
        self.lbl_euler_cam1 = CaptionLabel("Cam1 Euler: R -91.70°, P +30.47°, Y +85.34°")
        rot_layout.addWidget(self.lbl_lens_angle)
        rot_layout.addWidget(self.lbl_euler_cam0)
        rot_layout.addWidget(self.lbl_euler_cam1)
        metrics_row.addWidget(rot_card)

        layout.addLayout(metrics_row)

        # JSON Configuration Card
        json_card = CardWidget(self)
        json_layout = QVBoxLayout(json_card)
        json_layout.setContentsMargins(16, 14, 16, 14)
        json_layout.setSpacing(10)

        json_header = QHBoxLayout()
        json_title = SubtitleLabel("Full Calibration Configuration (JSON)")
        self.btn_reload = PushButton("Reload", icon=FluentIcon.SYNC)
        self.btn_reload.clicked.connect(self._load_calibration)
        self.btn_save = PushButton("Save Changes", icon=FluentIcon.SAVE)
        self.btn_save.clicked.connect(self._save_calibration)
        self.btn_export = PushButton("Export As...", icon=FluentIcon.SHARE)
        self.btn_export.clicked.connect(self._export_calibration)

        json_header.addWidget(json_title)
        json_header.addStretch()
        json_header.addWidget(self.btn_reload)
        json_header.addWidget(self.btn_save)
        json_header.addWidget(self.btn_export)
        json_layout.addLayout(json_header)

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
                self.lbl_scanner.setText(f"LiDAR: {data.get('scanner', 'Raven')}")
                self.lbl_camera.setText(f"Camera: {data.get('camera', 'Insta360 X4')}")
                self.lbl_optics.setText(f"Lens Model: {data.get('lens_model', 'THIN_PRISM_FISHEYE')}")
            except Exception as e:
                self.json_edit.setPlainText(f"Error loading calibration: {e}")

    def _save_calibration(self):
        text = self.json_edit.toPlainText().strip()
        try:
            data = json.loads(text)
            self._calib_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
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

