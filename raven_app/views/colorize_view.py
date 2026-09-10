"""LiDAR point cloud colorization view."""
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QFrame
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton, ToolButton,
    LineEdit, DoubleSpinBox, ComboBox, CheckBox, ProgressBar,
    PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition
)
from raven_app.process_runner import ProcessRunner
from raven_app.i18n import tr


class ColorizeView(QWidget):
    """WinUI Fluent view for running dual-method point cloud colorization."""

    def __init__(self, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("ColorizeView")
        self.runner = runner
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        self.header_title = TitleLabel(tr("Point Cloud Colorization Studio"))
        self.header_subtitle = CaptionLabel(
            tr("Colorize 3D LiDAR point clouds using Calibrated Direct Rigid projection or SfM Consensus")
        )
        header_layout.addWidget(self.header_title)
        header_layout.addWidget(self.header_subtitle)
        layout.addLayout(header_layout)

        # Dataset & Paths Card
        paths_card = CardWidget(self)
        paths_layout = QVBoxLayout(paths_card)
        paths_layout.setContentsMargins(20, 18, 20, 18)
        paths_layout.setSpacing(14)

        self.paths_title = SubtitleLabel(tr("Dataset & Calibration Inputs"))
        paths_layout.addWidget(self.paths_title)

        # Dataset folder row
        ds_row = QHBoxLayout()
        self.ds_label = BodyLabel(tr("Dataset Directory:"))
        self.ds_label.setFixedWidth(140)
        self.ds_input = LineEdit()
        self.ds_input.setPlaceholderText(tr("Select folder containing slam_out/ and images/..."))
        self.ds_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.ds_browse.clicked.connect(self._browse_dataset)
        ds_row.addWidget(self.ds_label)
        ds_row.addWidget(self.ds_input)
        ds_row.addWidget(self.ds_browse)
        paths_layout.addLayout(ds_row)

        # Calib JSON row
        calib_row = QHBoxLayout()
        self.calib_label = BodyLabel(tr("Calibration JSON:"))
        self.calib_label.setFixedWidth(140)
        self.calib_input = LineEdit()
        self.calib_input.setPlaceholderText(tr("Leave empty to use default Raven rigid calibration..."))
        self.calib_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.calib_browse.clicked.connect(self._browse_calib)
        calib_row.addWidget(self.calib_label)
        calib_row.addWidget(self.calib_input)
        calib_row.addWidget(self.calib_browse)
        paths_layout.addLayout(calib_row)

        layout.addWidget(paths_card)

        # Method & Parameters Card
        params_card = CardWidget(self)
        params_layout = QVBoxLayout(params_card)
        params_layout.setContentsMargins(20, 18, 20, 18)
        params_layout.setSpacing(14)

        self.params_title = SubtitleLabel(tr("Method & Synchronization"))
        params_layout.addWidget(self.params_title)

        method_row = QHBoxLayout()
        self.method_label = BodyLabel(tr("Colorization Method:"))
        self.method_label.setFixedWidth(140)
        self.method_combo = ComboBox()
        self.method_combo.addItems([
            tr("Trajectory-based colorization"),
            tr("Reconstruction-based colorization"),
            tr("Compare both methods")
        ])
        self.method_combo.setCurrentIndex(0)
        method_row.addWidget(self.method_label)
        method_row.addWidget(self.method_combo)
        params_layout.addLayout(method_row)

        # Timing row
        timing_row = QHBoxLayout()
        self.fps_label = BodyLabel(tr("Extraction FPS:"))
        self.fps_spin = DoubleSpinBox()
        self.fps_spin.setRange(0.1, 120.0)
        self.fps_spin.setValue(2.0)
        self.fps_spin.setSingleStep(0.5)

        self.dt_label = BodyLabel(tr("Manual Δt (s):"))
        self.dt_input = LineEdit()
        self.dt_input.setPlaceholderText("Optional (e.g. -4.2 or 5.7075)")

        timing_row.addWidget(self.fps_label)
        timing_row.addWidget(self.fps_spin)
        timing_row.addSpacing(20)
        timing_row.addWidget(self.dt_label)
        timing_row.addWidget(self.dt_input)
        timing_row.addStretch()
        params_layout.addLayout(timing_row)

        # Checkboxes
        check_row = QHBoxLayout()
        self.chk_spirula = CheckBox(tr("Run Spirula SfM automatically if sparse reconstruction is missing"))
        self.chk_recalib = CheckBox(tr("Recalibrate spatial extrinsics from SfM alignment"))
        check_row.addWidget(self.chk_spirula)
        check_row.addWidget(self.chk_recalib)
        check_row.addStretch()
        params_layout.addLayout(check_row)

        layout.addWidget(params_card)

        # Action & Status Card
        action_card = CardWidget(self)
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(20, 14, 20, 14)

        status_box = QVBoxLayout()
        self.status_title = StrongBodyLabel(tr("Status: Ready"))
        self.status_desc = CaptionLabel(tr("Select a dataset folder with PCD trajectory and camera frames."))
        status_box.addWidget(self.status_title)
        status_box.addWidget(self.status_desc)
        action_layout.addLayout(status_box)

        action_layout.addStretch()

        self.progress_bar = ProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        self.btn_run = PrimaryPushButton(tr("Run Colorization"), icon=FluentIcon.PALETTE)
        self.btn_run.clicked.connect(self._start_colorization)
        self.btn_cancel = PushButton(tr("Cancel"), icon=FluentIcon.CLOSE)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setEnabled(False)

        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_cancel)
        layout.addWidget(action_card)

        # Live Console Card
        log_card = CardWidget(self)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 12, 16, 12)
        log_layout.setSpacing(8)

        log_header = QHBoxLayout()
        self.log_title = SubtitleLabel(tr("Colorization Log"))
        self.btn_clear_log = ToolButton(FluentIcon.DELETE)
        self.btn_clear_log.setToolTip(tr("Clear log"))
        self.btn_clear_log.clicked.connect(self._clear_log)
        log_header.addWidget(self.log_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_header)

        self.log_console = PlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "PlainTextEdit { background-color: #0c1219; color: #a9c7df; font-family: 'Consolas', 'Courier New'; font-size: 12px; border-radius: 6px; padding: 8px; }"
        )
        log_layout.addWidget(self.log_console)
        layout.addWidget(log_card, stretch=1)

    def _connect_signals(self):
        self.runner.started.connect(self._on_job_started)
        self.runner.finished.connect(self._on_job_finished)
        self.runner.log_received.connect(self._on_log_received)

    def _browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Directory")
        if folder:
            self.ds_input.setText(folder)

    def _browse_calib(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration JSON", "", "JSON files (*.json);;All files (*.*)"
        )
        if path:
            self.calib_input.setText(path)

    def _clear_log(self):
        self.log_console.clear()

    def _start_colorization(self):
        ds = self.ds_input.text().strip()
        if not ds or not Path(ds).is_dir():
            InfoBar.error(
                title=tr("Invalid Dataset Directory"),
                content=tr("Please specify a valid dataset directory containing slam_out and images."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        method_key = ['direct', 'sfm', 'all'][self.method_combo.currentIndex()]
        args = [
            'colorize',
            '--dataset', ds,
            '--method', method_key,
            '--fps', str(self.fps_spin.value())
        ]

        calib = self.calib_input.text().strip()
        if calib and Path(calib).is_file():
            args.extend(['--calib', calib])

        dt = self.dt_input.text().strip()
        if dt:
            args.extend(['--dt', dt])

        if self.chk_spirula.isChecked():
            args.append('--run-spirula')

        if self.chk_recalib.isChecked():
            args.append('--recalibrate-from-sfm')

        self.log_console.appendPlainText(f"\n>>> Starting Colorization: {' '.join(args)}\n")
        self.runner.start_job(args)

    def _cancel(self):
        self.btn_cancel.setEnabled(False)
        self.runner.cancel()

    def _on_job_started(self):
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_title.setText("Status: Colorizing Point Cloud")
        self.status_desc.setText("Processing ray-casting and camera frame projections...")

    def _on_job_finished(self, code: int):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        if code == 0:
            self.status_title.setText("Status: Colorization Completed")
            self.status_desc.setText("Deliverable PCDs generated under dataset/deliverables/")
            InfoBar.success(
                title=tr("Colorization Finished"),
                content=tr("Point cloud colorization completed successfully."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self
            )
        elif code == 130:
            self.status_title.setText("Status: Cancelled")
            self.status_desc.setText("Colorization was cancelled by the user.")
        else:
            self.status_title.setText(f"Status: Failed (exit code {code})")
            self.status_desc.setText("Error during colorization. Review the log console.")
            InfoBar.error(
                title=tr("Colorization Error"),
                content=tr("Process exited with code {code}.", code=code),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self
            )

    def _on_log_received(self, line: str):
        self.log_console.insertPlainText(line)
        self.log_console.ensureCursorVisible()

    def retranslate_ui(self):
        """Update all text elements dynamically when the language changes."""
        self.header_title.setText(tr("Point Cloud Colorization Studio"))
        self.header_subtitle.setText(
            tr("Colorize 3D LiDAR point clouds using Calibrated Direct Rigid projection or SfM Consensus")
        )
        self.paths_title.setText(tr("Dataset & Calibration Inputs"))
        self.ds_label.setText(tr("Dataset Directory:"))
        self.ds_input.setPlaceholderText(tr("Select folder containing slam_out/ and images/..."))
        self.ds_browse.setText(tr("Browse"))
        self.calib_label.setText(tr("Calibration JSON:"))
        self.calib_input.setPlaceholderText(tr("Leave empty to use default Raven rigid calibration..."))
        self.calib_browse.setText(tr("Browse"))

        self.params_title.setText(tr("Method & Synchronization"))
        self.method_label.setText(tr("Colorization Method:"))

        cur_idx = self.method_combo.currentIndex()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItems([
            tr("Trajectory-based colorization"),
            tr("Reconstruction-based colorization"),
            tr("Compare both methods")
        ])
        self.method_combo.setCurrentIndex(cur_idx)
        self.method_combo.blockSignals(False)

        self.fps_label.setText(tr("Extraction FPS:"))
        self.dt_label.setText(tr("Manual Δt (s):"))
        self.chk_spirula.setText(tr("Run Spirula SfM automatically if sparse reconstruction is missing"))
        self.chk_recalib.setText(tr("Recalibrate spatial extrinsics from SfM alignment"))
        self.btn_run.setText(tr("Run Colorization"))
        self.btn_cancel.setText(tr("Cancel"))
        self.log_title.setText(tr("Colorization Log"))
        self.btn_clear_log.setToolTip(tr("Clear log"))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Select a dataset folder with PCD trajectory and camera frames."))

