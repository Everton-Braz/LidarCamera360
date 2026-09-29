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
from raven_app.process_runner import ProcessRunner, make_run_log_path
from raven_app.mask_download import MaskResourceDownload
from raven_app.mask_resources import resource_status, resources_ready
from raven_app.i18n import tr


class ColorizeView(QWidget):
    """WinUI Fluent view for running dual-method point cloud colorization."""

    def __init__(self, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("ColorizeView")
        self.runner = runner
        self._mask_downloader = None
        self.mask_config = None
        self._current_log_path = None
        self._init_ui()
        self._connect_signals()
        self._refresh_mask_resource_state()

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
        self.fps_spin.setToolTip(tr("Extraction rate is capped by the source video. For a 2 fps timelapse, use 2; a higher value does not create extra frames."))

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

        photometric_row = QHBoxLayout()
        self.photometric_label = BodyLabel(tr("Photometric compensation:"))
        self.photometric_label.setFixedWidth(180)
        self.photometric_combo = ComboBox()
        self.photometric_combo.addItem(tr("Off"), userData="off")
        self.photometric_combo.addItem(tr("Log-linear"), userData="loglinear")
        self.photometric_combo.setCurrentIndex(0)
        self.photometric_combo.setFixedWidth(130)
        self.photometric_combo.currentIndexChanged.connect(self._update_photometric_controls)
        self.photometric_params_label = BodyLabel(tr("Photometric parameters:"))
        self.photometric_params_label.setFixedWidth(180)
        self.photometric_params_input = LineEdit()
        self.photometric_params_input.setPlaceholderText(tr("Optional fitted JSON file"))
        self.photometric_params_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.photometric_params_browse.clicked.connect(self._browse_photometric_params)
        photometric_row.addWidget(self.photometric_label)
        photometric_row.addWidget(self.photometric_combo)
        photometric_row.addSpacing(12)
        photometric_row.addWidget(self.photometric_params_label)
        photometric_row.addWidget(self.photometric_params_input)
        photometric_row.addWidget(self.photometric_params_browse)
        params_layout.addLayout(photometric_row)
        self._update_photometric_controls()

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
        self.mask_backend_label = BodyLabel(tr("Mask backend:"))
        self.mask_backend_combo = ComboBox()
        self.mask_backend_combo.addItem(tr("Vulkan"), userData="vulkan")
        self.mask_backend_combo.addItem(tr("TensorRT"), userData="tensorrt")
        self.mask_backend_combo.setCurrentIndex(0)
        self.mask_backend_combo.currentIndexChanged.connect(self._refresh_mask_resource_state)
        self.mask_backend_combo.currentIndexChanged.connect(self._update_mask_backend_tooltip)
        self._update_mask_backend_tooltip()
        self.btn_mask_model = PushButton(tr("Download model RF-DETR"), icon=FluentIcon.DOWNLOAD)
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
        mask_row.addWidget(self.mask_backend_label)
        mask_row.addWidget(self.mask_backend_combo)
        mask_row.addStretch()
        params_layout.addLayout(mask_row)

        mask_controls_row = QHBoxLayout()
        mask_controls_row.addWidget(self.btn_mask_model)
        mask_controls_row.addWidget(self.btn_mask_settings)
        mask_controls_row.addSpacing(12)
        mask_controls_row.addWidget(self.operator_radius_label)
        mask_controls_row.addWidget(self.operator_radius_spin)
        mask_controls_row.addStretch()
        params_layout.addLayout(mask_controls_row)

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
        self.btn_clear_log.setToolTip(tr("Clear visible log (saved file remains)"))
        self.btn_clear_log.clicked.connect(self._clear_log)
        self.btn_export_log = ToolButton(FluentIcon.DOWNLOAD)
        self.btn_export_log.setToolTip(tr("Export log..."))
        self.btn_export_log.clicked.connect(self._export_log)
        log_header.addWidget(self.log_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_export_log)
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

    def _browse_photometric_params(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Select photometric parameter JSON"), "", "JSON files (*.json);;All files (*.*)"
        )
        if path:
            self.photometric_params_input.setText(path)

    def _update_photometric_controls(self, *_):
        enabled = self.photometric_combo.currentData() == "loglinear"
        self.photometric_params_label.setEnabled(enabled)
        self.photometric_params_input.setEnabled(enabled)
        self.photometric_params_browse.setEnabled(enabled)

    def _download_rf_detr(self):
        if self.runner.is_busy:
            InfoBar.warning(title=tr('Processing is active'), content=tr('Wait for the current task to finish before downloading RF-DETR resources.'), parent=self)
            return
        backend = self.mask_backend_combo.currentData()
        if resources_ready(backend):
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            InfoBar.success(title=tr('RF-DETR is ready'), content=tr('The model and {backend} mask resources are verified.').format(backend=label), parent=self)
            return
        if resource_status(backend) == 'executor-missing':
            InfoBar.error(title=tr('Mask backend unavailable'), content=tr('This app build is missing the selected RF-DETR executor. Rebuild or update the app.'), parent=self)
            return
        if backend == 'vulkan':
            prompt = tr('Verify or download the 139 MB RF-DETR model and prepare its Vulkan cache (~130 MB). This uses Vulkan and does not download TensorRT. Resources stay outside the portable app.')
        else:
            prompt = tr('Verify or download the 139 MB RF-DETR model. If needed, download about 1.3 GB of NVIDIA TensorRT runtime; extraction may need up to 3 GB of temporary disk space. Resources stay outside the portable app.')
        answer = QMessageBox.question(
            self, tr('Prepare RF-DETR masks'), prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        downloader = MaskResourceDownload(backend, self)
        self._mask_downloader = downloader
        downloader.progress.connect(self._on_mask_download_progress)
        downloader.completed.connect(self._on_mask_download_completed)
        self.btn_mask_model.setEnabled(False)
        self.mask_backend_combo.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.btn_mask_model.setText(tr('Preparing RF-DETR...'))
        self.btn_mask_model.setIcon(FluentIcon.SYNC)
        if not downloader.start():
            self._on_mask_download_completed(2, tr('Could not start the resource downloader.'))

    def _on_mask_download_progress(self, resource, percent):
        title = {
            'model': tr('RF-DETR model'),
            'runtime': tr('TensorRT runtime'),
            'install': tr('Installing TensorRT'),
            'vulkan': tr('Preparing Vulkan model'),
        }.get(resource, tr('Preparing RF-DETR'))
        self.btn_mask_model.setText(f'{title}: {percent}%' if percent else f'{title}...')

    def _on_mask_download_completed(self, code, output):
        downloader, self._mask_downloader = self._mask_downloader, None
        self.btn_mask_model.setEnabled(True)
        self.mask_backend_combo.setEnabled(True)
        self.btn_run.setEnabled(True)
        if downloader is not None:
            downloader.deleteLater()
        self._refresh_mask_resource_state()
        backend = self.mask_backend_combo.currentData()
        if code == 0 and resources_ready(backend):
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            InfoBar.success(title=tr('RF-DETR resources ready'), content=tr('The model and {backend} mask resources are verified and saved in the user cache.').format(backend=label), parent=self)
        else:
            detail = output[-700:] if output else tr('Check your internet connection and available disk space, then retry.')
            InfoBar.error(title=tr('RF-DETR download failed'), content=detail, parent=self, duration=8000)

    def _refresh_mask_resource_state(self, *_args):
        if self._mask_downloader is not None:
            return
        backend = self.mask_backend_combo.currentData()
        status = resource_status(backend)
        self.btn_mask_model.setEnabled(status != 'executor-missing')
        if status == 'ready':
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            self.btn_mask_model.setText(tr('RF-DETR ready ({backend})').format(backend=label))
            self.btn_mask_model.setIcon(FluentIcon.ACCEPT)
            self.btn_mask_model.setToolTip(tr('The model, selected mask executor and required resources are verified.'))
        elif status == 'model-missing':
            self.btn_mask_model.setText(tr('Download model RF-DETR'))
            self.btn_mask_model.setIcon(FluentIcon.DOWNLOAD)
            self.btn_mask_model.setToolTip(tr('Download the verified RF-DETR model and prepare the selected mask backend.'))
        elif status == 'runtime-missing':
            self.btn_mask_model.setText(tr('Download TensorRT runtime'))
            self.btn_mask_model.setIcon(FluentIcon.DOWNLOAD)
            self.btn_mask_model.setToolTip(tr('The RF-DETR model is present. Download the optional NVIDIA TensorRT runtime for this backend.'))
        elif status == 'schedule-missing':
            self.btn_mask_model.setText(tr('Prepare Vulkan masks'))
            self.btn_mask_model.setIcon(FluentIcon.SYNC)
            self.btn_mask_model.setToolTip(tr('The model is downloaded. Prepare the Vulkan schedule; TensorRT is not needed.'))
        else:
            self.btn_mask_model.setText(tr('Mask executor missing'))
            self.btn_mask_model.setIcon(FluentIcon.INFO)
            self.btn_mask_model.setToolTip(tr('Update or rebuild the app with the selected RF-DETR executor.'))

    def _update_mask_backend_tooltip(self, *_args):
        if self.mask_backend_combo.currentData() == 'vulkan':
            text = tr('Use Vulkan for person-mask inference; no TensorRT runtime is required. Point-cloud colorization has a separate setting.')
        else:
            text = tr('Use NVIDIA TensorRT for person-mask inference; its optional runtime is downloaded separately. Point-cloud colorization has a separate setting.')
        self.mask_backend_combo.setToolTip(text)

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
        dialog = MaskSettingsDialog(dataset, config, self,
                                    backend=self.mask_backend_combo.currentData())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.mask_config = dialog.settings()
            self.chk_mask_persons.setChecked(True)
            count = sum(sum(map(len, self.mask_config[key].values())) for key in ('rectangles', 'ellipses', 'polygons'))
            self.btn_mask_settings.setText(tr('Mask settings ({n} shapes)').format(n=count))

    def _clear_log(self):
        self.log_console.clear()

    def _export_log(self):
        source = self._current_log_path
        default_path = str(source) if source else 'colorization.log'
        target, _ = QFileDialog.getSaveFileName(
            self, tr('Export colorization log'), default_path,
            tr('Log files (*.log);;Text files (*.txt);;All files (*.*)')
        )
        if not target:
            return
        try:
            content = source.read_text(encoding='utf-8') if source and source.is_file() else self.log_console.toPlainText()
            Path(target).write_text(content, encoding='utf-8')
        except OSError as exc:
            InfoBar.error(title=tr('Cannot export log'), content=str(exc),
                          position=InfoBarPosition.TOP, parent=self)

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
            '--fps', str(self.fps_spin.value()),
            '--photometric', str(self.photometric_combo.currentData())
        ]

        photometric_params = self.photometric_params_input.text().strip()
        if photometric_params and self.photometric_combo.currentData() == "loglinear":
            args.extend(['--photometric-params', photometric_params])

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
            args.extend(['--mask-backend', self.mask_backend_combo.currentData()])
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

        log_path = make_run_log_path(ds, f'{Path(ds).name}_colorization')
        if self.runner.start_job(
            args, log_path=log_path,
            initial_log=f">>> Starting Colorization: {' '.join(args)}"
        ):
            self._current_log_path = log_path

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

        self.photometric_label.setText(tr("Photometric compensation:"))
        photo_idx = self.photometric_combo.currentIndex()
        self.photometric_combo.blockSignals(True)
        self.photometric_combo.setItemText(0, tr("Off"))
        self.photometric_combo.setItemText(1, tr("Log-linear"))
        self.photometric_combo.setCurrentIndex(photo_idx)
        self.photometric_combo.blockSignals(False)
        self.photometric_params_label.setText(tr("Photometric parameters:"))
        self.photometric_params_input.setPlaceholderText(tr("Optional fitted JSON file"))
        self.photometric_params_browse.setText(tr("Browse"))
        self._update_photometric_controls()

        self.fps_label.setText(tr("Extraction FPS:"))
        self.fps_spin.setToolTip(tr("Extraction rate is capped by the source video. For a 2 fps timelapse, use 2; a higher value does not create extra frames."))
        self.dt_label.setText(tr("Manual Δt (s):"))
        self.chk_spirula.setText(tr("Run Spirula SfM automatically if sparse reconstruction is missing"))
        self.chk_recalib.setText(tr("Recalibrate spatial extrinsics from SfM alignment"))
        self.chk_mask_persons.setText(tr("Generate masks"))
        self.chk_mask_persons.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        self.mask_backend_label.setText(tr("Mask backend:"))
        self.mask_backend_combo.setItemText(0, tr("Vulkan"))
        self.mask_backend_combo.setItemText(1, tr("TensorRT"))
        self._update_mask_backend_tooltip()
        shapes = sum(sum(map(len, self.mask_config[key].values()))
                     for key in ('rectangles', 'ellipses', 'polygons')) if self.mask_config else 0
        self.btn_mask_settings.setText(tr('Mask settings ({n} shapes)').format(n=shapes) if self.mask_config else tr('Mask settings...'))
        if self._mask_downloader is None:
            self._refresh_mask_resource_state()
        self.operator_radius_label.setText(tr("Operator removal radius (m):"))
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.btn_run.setText(tr("Run Colorization"))
        self.btn_cancel.setText(tr("Cancel"))
        self.log_title.setText(tr("Colorization Log"))
        self.btn_clear_log.setToolTip(tr("Clear visible log (saved file remains)"))
        self.btn_export_log.setToolTip(tr("Export log..."))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Select a dataset folder with PCD trajectory and camera frames."))
