"""LiDAR point cloud colorization view."""
import json
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QFrame, QDialog, QMessageBox
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton, ToolButton,
    LineEdit, DoubleSpinBox, ComboBox, CheckBox, ProgressBar,
    PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition
)
from raven_app.process_runner import ProcessRunner
from raven_app.mask_download import MaskResourceDownload
from raven_app.mask_resources import resolve_model, resolve_runtime_dir
from raven_app.i18n import tr


class ColorizeView(QWidget):
    """WinUI Fluent view for running dual-method point cloud colorization."""

    def __init__(self, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("ColorizeView")
        self.runner = runner
        self._mask_downloader = None
        self.mask_config = None
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

        mask_row = QHBoxLayout()
        self.chk_mask_persons = CheckBox(tr("Generate masks"))
        self.chk_mask_persons.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        self.btn_mask_model = PushButton(tr("Download model RF-DETR"), icon=FluentIcon.DOWNLOAD)
        self.btn_mask_model.setToolTip(tr("Download the RF-DETR model and, when needed, its TensorRT runtime to the user cache. The portable app stays small."))
        self.btn_mask_model.clicked.connect(self._download_rf_detr)
        self.btn_mask_settings = PushButton(tr("Mask settings..."))
        self.btn_mask_settings.clicked.connect(self._open_mask_settings)
        self.operator_radius_label = BodyLabel(tr("Operator removal radius (m):"))
        self.operator_radius_spin = DoubleSpinBox()
        self.operator_radius_spin.setRange(0.0, 5.0)
        self.operator_radius_spin.setDecimals(2)
        self.operator_radius_spin.setSingleStep(0.10)
        self.operator_radius_spin.setValue(0.0)
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.operator_radius_spin.valueChanged.connect(lambda value: self.chk_mask_persons.setChecked(True) if value > 0 else None)
        self.chk_mask_persons.toggled.connect(lambda checked: self.operator_radius_spin.setValue(0.0) if not checked else None)
        mask_row.addWidget(self.chk_mask_persons)
        mask_row.addWidget(self.btn_mask_model)
        mask_row.addWidget(self.btn_mask_settings)
        mask_row.addSpacing(12)
        mask_row.addWidget(self.operator_radius_label)
        mask_row.addWidget(self.operator_radius_spin)
        mask_row.addStretch()
        params_layout.addLayout(mask_row)

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

    def _download_rf_detr(self):
        if self.runner.is_busy:
            InfoBar.warning(title=tr('Processing is active'), content=tr('Wait for the current task to finish before downloading RF-DETR resources.'), parent=self)
            return
        if resolve_model() is not None and resolve_runtime_dir() is not None:
            InfoBar.success(title=tr('RF-DETR is ready'), content=tr('The model and TensorRT runtime are already available.'), parent=self)
            return
        answer = QMessageBox.question(
            self, tr('Download RF-DETR resources'),
            tr('Verify or download the 139 MB model as needed. If the NVIDIA TensorRT runtime is missing, download about 1.3 GB; extraction may need up to 3 GB of temporary disk space. Files are cached for this Windows user and are not added to the portable app.'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        downloader = MaskResourceDownload(self)
        self._mask_downloader = downloader
        downloader.progress.connect(self._on_mask_download_progress)
        downloader.completed.connect(self._on_mask_download_completed)
        self.btn_mask_model.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_mask_model.setText(tr('Preparing RF-DETR...'))
        if not downloader.start():
            self._on_mask_download_completed(2, tr('Could not start the resource downloader.'))

    def _on_mask_download_progress(self, resource, percent):
        title = tr('RF-DETR model') if resource == 'model' else (tr('Installing TensorRT') if resource == 'install' else tr('TensorRT runtime'))
        self.btn_mask_model.setText(f'{title}: {percent}%')

    def _on_mask_download_completed(self, code, output):
        downloader, self._mask_downloader = self._mask_downloader, None
        self.btn_mask_model.setEnabled(True)
        self.btn_mask_model.setText(tr('Download model RF-DETR'))
        self.btn_run.setEnabled(True)
        if downloader is not None:
            downloader.deleteLater()
        if code == 0 and resolve_model() is not None and resolve_runtime_dir() is not None:
            InfoBar.success(title=tr('RF-DETR resources ready'), content=tr('The verified model and TensorRT runtime are saved in the user cache.'), parent=self)
        else:
            detail = output[-700:] if output else tr('Check your internet connection and available disk space, then retry.')
            InfoBar.error(title=tr('RF-DETR download failed'), content=detail, parent=self, duration=8000)

    def _open_mask_settings(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog
        from raven_app.person_masks import read_mask_config
        dataset = Path(self.ds_input.text().strip()) if self.ds_input.text().strip() else None
        config = self.mask_config
        settings_path = dataset / 'mask_settings.json' if dataset else None
        if config is None and settings_path and settings_path.is_file():
            try:
                config = read_mask_config(settings_path)
            except (OSError, ValueError) as exc:
                InfoBar.error(title=tr('Invalid mask settings'), content=str(exc),
                              position=InfoBarPosition.TOP, parent=self)
                return
        dialog = MaskSettingsDialog(dataset, config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.mask_config = dialog.settings()
            self.chk_mask_persons.setChecked(True)
            count = sum(sum(map(len, self.mask_config[key].values())) for key in ('rectangles', 'ellipses', 'polygons'))
            self.btn_mask_settings.setText(tr('Mask settings ({n} shapes)').format(n=count))

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

        if self.chk_mask_persons.isChecked():
            args.append('--mask-persons')
            settings_path = Path(ds).resolve() / 'mask_settings.json'
            if self.mask_config is not None:
                try:
                    temporary = settings_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(self.mask_config, indent=2), encoding='utf-8')
                    temporary.replace(settings_path)
                except OSError as exc:
                    InfoBar.error(title=tr('Cannot save mask settings'), content=str(exc),
                                  position=InfoBarPosition.TOP, parent=self)
                    return
            if settings_path.is_file():
                args.extend(['--mask-config', str(settings_path)])

        if self.operator_radius_spin.value() > 0:
            args.extend(['--operator-radius', str(self.operator_radius_spin.value())])

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
        self.chk_mask_persons.setText(tr("Generate masks"))
        self.chk_mask_persons.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        shapes = sum(sum(map(len, self.mask_config[key].values()))
                     for key in ('rectangles', 'ellipses', 'polygons')) if self.mask_config else 0
        self.btn_mask_settings.setText(tr('Mask settings ({n} shapes)').format(n=shapes) if self.mask_config else tr('Mask settings...'))
        if self._mask_downloader is None:
            self.btn_mask_model.setText(tr("Download model RF-DETR"))
            self.btn_mask_model.setToolTip(tr("Download the RF-DETR model and, when needed, its TensorRT runtime to the user cache. The portable app stays small."))
        self.operator_radius_label.setText(tr("Operator removal radius (m):"))
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.btn_run.setText(tr("Run Colorization"))
        self.btn_cancel.setText(tr("Cancel"))
        self.log_title.setText(tr("Colorization Log"))
        self.btn_clear_log.setToolTip(tr("Clear log"))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Select a dataset folder with PCD trajectory and camera frames."))
