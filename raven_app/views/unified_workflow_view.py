"""Unified processing workflow view."""
import json
import time
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QFrame, QDialog,
    QMessageBox
)
from qfluentwidgets import (
    CardWidget, ElevatedCardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton, ToolButton,
    LineEdit, DoubleSpinBox, SpinBox, ComboBox, CheckBox, SwitchButton,
    ProgressBar, PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition,
    SmoothScrollArea
)
from raven_app.process_runner import ProcessRunner, make_run_log_path
from raven_app.workflow_progress import (
    STAGE_NAMES, estimate_stage_seconds, load_stage_history,
    parse_progress_line, save_stage_history, stage_duration_estimates,
)
from raven_app.mask_download import MaskResourceDownload
from raven_app.mask_resources import resource_status, resources_ready
from raven_app.bag_io import detect_bag_topics
from raven_app.i18n import tr


class UnifiedWorkflowView(QWidget):
    """WinUI Fluent view providing single-screen end-to-end processing.

    Inputs: ROS Bag (.bag) + Insta360 Video (.insv) + Output Folder
    Workflow: Video Extraction -> SLAM -> Gyro Sync -> Colorization -> Deliverables
    Outputs: .PLY, .PCD, and COLMAP Dataset ready for 3DGS Training.
    """

    def __init__(self, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("UnifiedWorkflowView")
        self.runner = runner
        self._mask_downloader = None
        self.mask_config = None
        self._current_log_path = None
        self._workflow_started_at = None
        self._stage_started_at = None
        self._stage_root = None
        self._current_stage = 0
        self._current_stage_skipped = False
        self._stage_fraction = 0.0
        self._stage_fraction_from_log = False
        self._stage_history = [[] for _ in STAGE_NAMES]
        self._stage_estimates = [60.0] * len(STAGE_NAMES)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_workflow_progress)
        self._init_ui()
        self._connect_signals()
        self._refresh_mask_resource_state()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area to comfortably accommodate all cards
        scroll = SmoothScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # ----------------------------------------------------------------------
        # Header
        # ----------------------------------------------------------------------
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        self.header_title = TitleLabel(tr("Unified LiDAR-Camera Studio"))
        self.header_subtitle = CaptionLabel(
            tr("Automated end-to-end pipeline: Inputs Selection → FAST-LIVO2 SLAM → "
               "Gyro Auto-Sync → Point Cloud Colorization → 3DGS Deliverables")
        )
        header_layout.addWidget(self.header_title)
        header_layout.addWidget(self.header_subtitle)
        layout.addLayout(header_layout)

        # ----------------------------------------------------------------------
        # 1. Dataset Inputs Card
        # ----------------------------------------------------------------------
        inputs_card = CardWidget(content)
        inputs_layout = QVBoxLayout(inputs_card)
        inputs_layout.setContentsMargins(20, 18, 20, 18)
        inputs_layout.setSpacing(12)

        self.inputs_title = SubtitleLabel(tr("1. Input Datasets & Destination"))
        inputs_layout.addWidget(self.inputs_title)

        # Row 1: Bag File
        bag_row = QHBoxLayout()
        self.bag_label = BodyLabel(tr("LiDAR ROS Bag (.bag):"))
        self.bag_label.setFixedWidth(170)
        self.bag_input = LineEdit()
        self.bag_input.setPlaceholderText(tr("Select merged or raw ROS1 bag file (.bag)..."))
        self.btn_browse_bag = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.btn_browse_bag.clicked.connect(self._browse_bag)
        bag_row.addWidget(self.bag_label)
        bag_row.addWidget(self.bag_input)
        bag_row.addWidget(self.btn_browse_bag)
        inputs_layout.addLayout(bag_row)

        # Row 2: INSV Video
        insv_row = QHBoxLayout()
        self.insv_label = BodyLabel(tr("Insta360 Video (.insv):"))
        self.insv_label.setFixedWidth(170)
        self.insv_input = LineEdit()
        self.insv_input.setPlaceholderText(tr("Select Insta360 X4 / X6 video (.insv or .mp4)..."))
        self.btn_browse_insv = PushButton(tr("Browse"), icon=FluentIcon.VIDEO)
        self.btn_browse_insv.clicked.connect(self._browse_insv)
        insv_row.addWidget(self.insv_label)
        insv_row.addWidget(self.insv_input)
        insv_row.addWidget(self.btn_browse_insv)
        inputs_layout.addLayout(insv_row)

        # Row 3: Output Folder
        out_row = QHBoxLayout()
        self.out_label = BodyLabel(tr("Output Directory:"))
        self.out_label.setFixedWidth(170)
        self.out_input = LineEdit()
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.btn_browse_out = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.btn_browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_label)
        out_row.addWidget(self.out_input)
        out_row.addWidget(self.btn_browse_out)
        inputs_layout.addLayout(out_row)

        layout.addWidget(inputs_card)

        # ----------------------------------------------------------------------
        # 2. Pipeline Configuration Card
        # ----------------------------------------------------------------------
        config_card = CardWidget(content)
        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(20, 18, 20, 18)
        config_layout.setSpacing(14)

        self.config_title = SubtitleLabel(tr("2. Pipeline Processing Options"))
        config_layout.addWidget(self.config_title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)

        # Scanner Preset Selector
        scanner_row = QHBoxLayout()
        self.scanner_label = CaptionLabel(tr("Scanner Hardware Preset:"))
        self.scanner_combo = ComboBox()
        self.scanner_combo.addItems([
            tr("Auto-Detect (Eagle / Raven)"),
            tr("Eagle (Livox Mid-360)"),
            tr("Raven (Vanjee 722z)")
        ])
        self.scanner_combo.currentIndexChanged.connect(self._on_preset_changed)
        scanner_row.addWidget(self.scanner_label)
        scanner_row.addWidget(self.scanner_combo)
        grid.addLayout(scanner_row, 0, 0)

        threads_layout = QHBoxLayout()
        self.threads_caption = CaptionLabel(tr("CPU Threads:"))
        self.threads_spin = SpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        threads_layout.addWidget(self.threads_caption)
        threads_layout.addWidget(self.threads_spin)
        grid.addLayout(threads_layout, 0, 1)

        # SLAM options
        self.lio_switch = SwitchButton(text=tr("Enable LiDAR + IMU Odometry (Fast LIO)"))
        self.lio_switch.setOnText(tr("LiDAR + IMU (LIO)"))
        self.lio_switch.setOffText(tr("LiDAR + camera + IMU (VIO)"))
        self.lio_switch.setChecked(True)
        grid.addWidget(self.lio_switch, 1, 0, 1, 2)

        # Topics
        topics_layout = QHBoxLayout()
        self.lidar_caption = CaptionLabel(tr("LiDAR Topic:"))
        self.lidar_topic = LineEdit()
        self.lidar_topic.setText("/vanjee_722z")
        topics_layout.addWidget(self.lidar_caption)
        topics_layout.addWidget(self.lidar_topic)
        self.imu_caption = CaptionLabel(tr("IMU Topic:"))
        self.imu_topic = LineEdit()
        self.imu_topic.setText("/vanjee_imu_packets")
        topics_layout.addWidget(self.imu_caption)
        topics_layout.addWidget(self.imu_topic)
        grid.addLayout(topics_layout, 2, 0, 1, 2)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        config_layout.addLayout(grid)
        config_layout.addWidget(sep)

        # Sync & Colorization row
        sync_row = QHBoxLayout()
        self.auto_sync_chk = CheckBox(tr("Auto IMU Gyro Cross-Correlation"))
        self.auto_sync_chk.setChecked(True)
        self.auto_sync_chk.stateChanged.connect(self._toggle_auto_sync)
        sync_row.addWidget(self.auto_sync_chk)

        self.dt_caption = CaptionLabel(tr("Manual Δt (s):"))
        self.dt_input = LineEdit()
        self.dt_input.setPlaceholderText("Optional (e.g. 1.892)")
        self.dt_input.setFixedWidth(120)
        self.dt_input.setEnabled(False)
        sync_row.addWidget(self.dt_caption)
        sync_row.addWidget(self.dt_input)
        sync_row.addStretch()

        self.fps_caption = CaptionLabel(tr("Extraction FPS:"))
        self.fps_spin = DoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(2.0)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setToolTip(tr("Extraction rate is capped by the source video. For a 2 fps timelapse, use 2; a higher value does not create extra frames."))
        sync_row.addWidget(self.fps_caption)
        sync_row.addWidget(self.fps_spin)

        config_layout.addLayout(sync_row)

        # Colorization & Recalibration Method
        method_row = QHBoxLayout()
        self.method_label = BodyLabel(tr("Colorization & Recalibration:"))
        method_row.addWidget(self.method_label)
        self.method_combo = ComboBox()
        self.method_combo.addItems([
            tr("Reconstruction-based colorization"),
            tr("Trajectory-based colorization"),
            tr("Compare both methods")
        ])
        self.method_combo.setCurrentIndex(0)
        method_row.addWidget(self.method_combo)
        method_row.addStretch()
        config_layout.addLayout(method_row)

        recalib_row = QHBoxLayout()
        self.recalibrate_chk = CheckBox(tr("Auto-Recalibrate Spatial Extrinsics from SfM Alignment"))
        self.recalibrate_chk.setChecked(True)
        recalib_row.addWidget(self.recalibrate_chk)
        recalib_row.addStretch()
        config_layout.addLayout(recalib_row)

        mask_row = QHBoxLayout()
        self.mask_persons_chk = CheckBox(tr("Generate masks"))
        self.mask_persons_chk.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        self.mask_model_btn = PushButton(tr("Download model RF-DETR"), icon=FluentIcon.DOWNLOAD)
        self.mask_backend_label = CaptionLabel(tr("Mask backend:"))
        self.mask_backend_combo = ComboBox()
        self.mask_backend_combo.addItem(tr("Vulkan"), userData="vulkan")
        self.mask_backend_combo.addItem(tr("TensorRT"), userData="tensorrt")
        self.mask_backend_combo.setCurrentIndex(0)
        self.mask_backend_combo.currentIndexChanged.connect(self._refresh_mask_resource_state)
        self.mask_backend_combo.currentIndexChanged.connect(self._update_mask_backend_tooltip)
        self._update_mask_backend_tooltip()
        self.mask_model_btn.clicked.connect(self._download_rf_detr)
        self.mask_settings_btn = PushButton(tr("Mask settings..."))
        self.mask_settings_btn.clicked.connect(self._open_mask_settings)
        self.operator_radius_caption = CaptionLabel(tr("Operator removal radius (m):"))
        self.operator_radius_spin = DoubleSpinBox()
        self.operator_radius_spin.setRange(0.0, 5.0)
        self.operator_radius_spin.setDecimals(2)
        self.operator_radius_spin.setSingleStep(0.10)
        self.operator_radius_spin.setValue(0.0)
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.operator_radius_spin.valueChanged.connect(lambda value: self.mask_persons_chk.setChecked(True) if value > 0 else None)
        self.mask_persons_chk.toggled.connect(lambda checked: self.operator_radius_spin.setValue(0.0) if not checked else None)
        mask_row.addWidget(self.mask_persons_chk)
        mask_row.addWidget(self.mask_backend_label)
        mask_row.addWidget(self.mask_backend_combo)
        mask_row.addStretch()
        config_layout.addLayout(mask_row)

        mask_controls_row = QHBoxLayout()
        mask_controls_row.addWidget(self.mask_model_btn)
        mask_controls_row.addWidget(self.mask_settings_btn)
        mask_controls_row.addSpacing(12)
        mask_controls_row.addWidget(self.operator_radius_caption)
        mask_controls_row.addWidget(self.operator_radius_spin)
        mask_controls_row.addStretch()
        config_layout.addLayout(mask_controls_row)

        self.vulkan_chk = CheckBox(tr("Use Vulkan for point-cloud colorization"))
        self.vulkan_chk.setChecked(True)
        self.vulkan_chk.setToolTip(tr("This setting accelerates point-cloud colorization. Choose the RF-DETR mask backend separately above."))
        config_layout.addWidget(self.vulkan_chk)

        self.process_gps_chk = CheckBox(tr("Automatically georeference using INSV GPS (when available)"))
        self.process_gps_chk.setChecked(True)
        self.process_gps_chk.stateChanged.connect(self._toggle_gps_formats)
        config_layout.addWidget(self.process_gps_chk)

        layout.addWidget(config_card)

        # ----------------------------------------------------------------------
        # 3. Deliverables Output Selection Card
        # ----------------------------------------------------------------------
        output_card = ElevatedCardWidget(content)
        output_layout = QVBoxLayout(output_card)
        output_layout.setContentsMargins(20, 18, 20, 18)
        output_layout.setSpacing(10)

        self.output_title = SubtitleLabel(tr("3. Select Deliverables to Generate"))
        self.output_subtitle = CaptionLabel(tr("Choose which output formats will be built in this run:"))
        output_layout.addWidget(self.output_title)
        output_layout.addWidget(self.output_subtitle)

        self.chk_ply = CheckBox(tr("Export Colored Point Cloud (.PLY)"))
        self.chk_ply.setChecked(True)
        output_layout.addWidget(self.chk_ply)

        self.chk_pcd = CheckBox(tr("Export Colored Point Cloud (.PCD)"))
        self.chk_pcd.setChecked(True)
        output_layout.addWidget(self.chk_pcd)

        self.chk_colmap = CheckBox(tr("Metric-Scaled 3DGS COLMAP Dataset"))
        self.chk_colmap.setChecked(True)
        output_layout.addWidget(self.chk_colmap)

        self.georef_formats_label = CaptionLabel(tr("Automatic georeferenced cloud formats:"))
        output_layout.addWidget(self.georef_formats_label)
        georef_formats_layout = QHBoxLayout()
        self.chk_geo_laz = CheckBox(tr("LAZ")); self.chk_geo_laz.setChecked(True)
        self.chk_geo_las = CheckBox(tr("LAS"))
        self.chk_geo_ply = CheckBox(tr("PLY"))
        self.chk_geo_pcd = CheckBox(tr("PCD"))
        self.chk_geo_geojson = CheckBox(tr("GeoJSON")); self.chk_geo_geojson.setChecked(True)
        for checkbox in (self.chk_geo_laz, self.chk_geo_las, self.chk_geo_ply, self.chk_geo_pcd, self.chk_geo_geojson):
            georef_formats_layout.addWidget(checkbox)
        georef_formats_layout.addStretch()
        output_layout.addLayout(georef_formats_layout)

        self.gps_formats_label = CaptionLabel(tr("GPS metadata formats (select one or more):"))
        output_layout.addWidget(self.gps_formats_label)
        gps_formats_layout = QHBoxLayout()
        self.chk_gps_geojson = CheckBox(tr("GeoJSON"))
        self.chk_gps_geojson.setChecked(True)
        self.chk_gps_gpx = CheckBox(tr("GPX"))
        self.chk_gps_gpx.setChecked(True)
        self.chk_gps_csv = CheckBox(tr("CSV"))
        self.chk_gps_csv.setChecked(True)
        gps_formats_layout.addWidget(self.chk_gps_geojson)
        gps_formats_layout.addWidget(self.chk_gps_gpx)
        gps_formats_layout.addWidget(self.chk_gps_csv)
        gps_formats_layout.addStretch()
        output_layout.addLayout(gps_formats_layout)
        self._toggle_gps_formats()

        layout.addWidget(output_card)

        # ----------------------------------------------------------------------
        # 4. Action & Status Bar
        # ----------------------------------------------------------------------
        action_card = CardWidget(content)
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(20, 14, 20, 14)
        action_layout.setSpacing(10)

        action_header = QHBoxLayout()

        status_box = QVBoxLayout()
        self.status_title = StrongBodyLabel(tr("Status: Ready"))
        self.status_desc = CaptionLabel(tr("Select Bag and INSV files, then click 'Start Unified Workflow'."))
        status_box.addWidget(self.status_title)
        status_box.addWidget(self.status_desc)
        action_header.addLayout(status_box)
        action_header.addStretch()

        self.btn_run = PrimaryPushButton(tr("Start Unified Workflow"), icon=FluentIcon.PLAY)
        self.btn_run.clicked.connect(self._start_workflow)
        self.btn_cancel = PushButton(tr("Cancel & Save"), icon=FluentIcon.CLOSE)
        self.btn_cancel.clicked.connect(self._cancel_workflow)
        self.btn_cancel.setEnabled(False)

        action_header.addWidget(self.btn_run)
        action_header.addWidget(self.btn_cancel)
        action_layout.addLayout(action_header)

        overall_row = QHBoxLayout()
        self.overall_progress_label = BodyLabel(tr("Overall progress: 0%"))
        self.overall_progress_label.setMinimumWidth(180)
        self.overall_progress_bar = ProgressBar()
        self.overall_progress_bar.setRange(0, 100)
        self.overall_progress_bar.setValue(0)
        self.eta_label = CaptionLabel(tr("Estimated time: waiting to start"))
        self.eta_label.setMinimumWidth(260)
        overall_row.addWidget(self.overall_progress_label)
        overall_row.addWidget(self.overall_progress_bar, 1)
        overall_row.addWidget(self.eta_label)
        action_layout.addLayout(overall_row)

        self.current_step_label = StrongBodyLabel(tr("Workflow steps"))
        action_layout.addWidget(self.current_step_label)
        steps_grid = QGridLayout()
        steps_grid.setHorizontalSpacing(14)
        steps_grid.setVerticalSpacing(7)
        self.step_progress_bars = []
        self.step_progress_labels = []
        for index, name in enumerate(STAGE_NAMES):
            column = (index % 2) * 2
            row = index // 2
            label = CaptionLabel(f"{index + 1}. {tr(name)}")
            bar = ProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setMinimumWidth(120)
            self.step_progress_labels.append(label)
            self.step_progress_bars.append(bar)
            steps_grid.addWidget(label, row, column)
            steps_grid.addWidget(bar, row, column + 1)
        action_layout.addLayout(steps_grid)
        layout.addWidget(action_card)

        # ----------------------------------------------------------------------
        # 5. Live Streaming Console
        # ----------------------------------------------------------------------
        log_card = CardWidget(content)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 12, 16, 12)
        log_layout.setSpacing(8)

        log_header = QHBoxLayout()
        self.log_title = SubtitleLabel(tr("Live Workflow Execution Console"))
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
        self.log_console.setMinimumHeight(240)
        self.log_console.setStyleSheet(
            "PlainTextEdit { background-color: #0c1219; color: #a9c7df; "
            "font-family: 'Consolas', 'Courier New'; font-size: 12px; border-radius: 6px; padding: 8px; }"
        )
        log_layout.addWidget(self.log_console)
        layout.addWidget(log_card)

        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def _connect_signals(self):
        self.runner.started.connect(self._on_job_started)
        self.runner.finished.connect(self._on_job_finished)
        self.runner.log_received.connect(self._on_log_received)
        self.runner.error_occurred.connect(self._on_error_occurred)

    def _on_preset_changed(self, index: int):
        if index == 1:  # Eagle
            self.lidar_topic.setText("/livox/lidar")
            self.imu_topic.setText("/livox/imu")
        elif index == 2:  # Raven
            self.lidar_topic.setText("/vanjee_722z")
            self.imu_topic.setText("/vanjee_imu_packets")
        else:  # Auto-Detect
            bag = self.bag_input.text().strip()
            if bag and Path(bag).is_file():
                self._auto_detect_bag(bag)

    def _auto_detect_bag(self, bag_path: str):
        try:
            detected = detect_bag_topics([bag_path])
            if detected.get('lidar_topic'):
                self.lidar_topic.setText(detected['lidar_topic'])
            if detected.get('imu_topic'):
                self.imu_topic.setText(detected['imu_topic'])
            stype = detected.get('scanner_type', '')
            if stype == 'eagle':
                self.scanner_combo.blockSignals(True)
                self.scanner_combo.setCurrentIndex(1)
                self.scanner_combo.blockSignals(False)
            elif stype == 'raven':
                self.scanner_combo.blockSignals(True)
                self.scanner_combo.setCurrentIndex(2)
                self.scanner_combo.blockSignals(False)
        except Exception:
            pass

    def _toggle_auto_sync(self, state):
        self.dt_input.setEnabled(not self.auto_sync_chk.isChecked())

    def _toggle_gps_formats(self, state=None):
        enabled = self.process_gps_chk.isChecked()
        for checkbox in (self.chk_gps_geojson, self.chk_gps_gpx, self.chk_gps_csv,
                         self.chk_geo_laz, self.chk_geo_las, self.chk_geo_ply,
                         self.chk_geo_pcd, self.chk_geo_geojson):
            checkbox.setEnabled(enabled)

    def _browse_bag(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROS Bag (.bag)", "", "ROS Bag files (*.bag);;All files (*.*)"
        )
        if path:
            self.bag_input.setText(path)
            self._auto_detect_bag(path)
            # Auto-suggest output folder if empty
            if not self.out_input.text().strip():
                p = Path(path)
                self.out_input.setText(str(p.parent / f"{p.stem}_studio_output"))

    def _browse_insv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Insta360 Video (.insv)", "", "Insta360 Video (*.insv *.mp4);;All files (*.*)"
        )
        if path:
            self.insv_input.setText(path)

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
        self.mask_model_btn.setEnabled(False)
        self.mask_backend_combo.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.mask_model_btn.setText(tr('Preparing RF-DETR...'))
        self.mask_model_btn.setIcon(FluentIcon.SYNC)
        if not downloader.start():
            self._on_mask_download_completed(2, tr('Could not start the resource downloader.'))

    def _update_mask_backend_tooltip(self, *_args):
        if self.mask_backend_combo.currentData() == 'vulkan':
            text = tr('Use Vulkan for person-mask inference; no TensorRT runtime is required. Point-cloud colorization has a separate setting.')
        else:
            text = tr('Use NVIDIA TensorRT for person-mask inference; its optional runtime is downloaded separately. Point-cloud colorization has a separate setting.')
        self.mask_backend_combo.setToolTip(text)

    def _on_mask_download_progress(self, resource, percent):
        title = {
            'model': tr('RF-DETR model'),
            'runtime': tr('TensorRT runtime'),
            'install': tr('Installing TensorRT'),
            'vulkan': tr('Preparing Vulkan model'),
        }.get(resource, tr('Preparing RF-DETR'))
        self.mask_model_btn.setText(f'{title}: {percent}%' if percent else f'{title}...')

    def _on_mask_download_completed(self, code, output):
        downloader, self._mask_downloader = self._mask_downloader, None
        self.mask_model_btn.setEnabled(True)
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
        self.mask_model_btn.setEnabled(status != 'executor-missing')
        if status == 'ready':
            label = tr('Vulkan') if backend == 'vulkan' else tr('TensorRT')
            self.mask_model_btn.setText(tr('RF-DETR ready ({backend})').format(backend=label))
            self.mask_model_btn.setIcon(FluentIcon.ACCEPT)
            self.mask_model_btn.setToolTip(tr('The model, selected mask executor and required resources are verified.'))
        elif status == 'model-missing':
            self.mask_model_btn.setText(tr('Download model RF-DETR'))
            self.mask_model_btn.setIcon(FluentIcon.DOWNLOAD)
            self.mask_model_btn.setToolTip(tr('Download the verified RF-DETR model and prepare the selected mask backend.'))
        elif status == 'runtime-missing':
            self.mask_model_btn.setText(tr('Download TensorRT runtime'))
            self.mask_model_btn.setIcon(FluentIcon.DOWNLOAD)
            self.mask_model_btn.setToolTip(tr('The RF-DETR model is present. Download the optional NVIDIA TensorRT runtime for this backend.'))
        elif status == 'schedule-missing':
            self.mask_model_btn.setText(tr('Prepare Vulkan masks'))
            self.mask_model_btn.setIcon(FluentIcon.SYNC)
            self.mask_model_btn.setToolTip(tr('The model is downloaded. Prepare the Vulkan schedule; TensorRT is not needed.'))
        else:
            self.mask_model_btn.setText(tr('Mask executor missing'))
            self.mask_model_btn.setIcon(FluentIcon.INFO)
            self.mask_model_btn.setToolTip(tr('Update or rebuild the app with the selected RF-DETR executor.'))

    def _open_mask_settings(self):
        from raven_app.mask_settings_dialog import MaskSettingsDialog
        from raven_app.person_masks import read_mask_config
        output = Path(self.out_input.text().strip()) if self.out_input.text().strip() else None
        config = self.mask_config
        settings_path = output / 'mask_settings.json' if output else None
        if config is None and settings_path and settings_path.is_file():
            try:
                config = read_mask_config(settings_path)
            except (OSError, ValueError) as exc:
                InfoBar.error(title=tr('Invalid mask settings'), content=str(exc),
                              position=InfoBarPosition.TOP, parent=self)
                return
        dialog = MaskSettingsDialog(output, config, self,
                                    backend=self.mask_backend_combo.currentData(),
                                    insv_path=self.insv_input.text().strip() or None,
                                    fps=self.fps_spin.value())
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.mask_config = dialog.settings()
            self.mask_persons_chk.setChecked(True)
            count = sum(sum(map(len, self.mask_config[key].values())) for key in ('rectangles', 'ellipses', 'polygons'))
            self.mask_settings_btn.setText(tr('Mask settings ({n} shapes)').format(n=count))

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_input.setText(folder)

    def _clear_log(self):
        self.log_console.clear()

    def _export_log(self):
        source = self._current_log_path
        default_path = str(source) if source else 'workflow.log'
        target, _ = QFileDialog.getSaveFileName(
            self, tr('Export workflow log'), default_path,
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

    def _start_workflow(self):
        bag = self.bag_input.text().strip()
        insv = self.insv_input.text().strip()
        out = self.out_input.text().strip()

        if not bag or not Path(bag).is_file():
            InfoBar.error(
                title=tr("Invalid LiDAR Bag"),
                content=tr("Please select a valid ROS bag file (.bag)."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if not insv or not Path(insv).is_file():
            InfoBar.error(
                title=tr("Invalid Insta360 Video"),
                content=tr("Please select an Insta360 video file (.insv or .mp4)."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if not out:
            InfoBar.error(
                title=tr("Missing Output Folder"),
                content=tr("Please specify an output folder for deliverables."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if self.process_gps_chk.isChecked() and not any(
            checkbox.isChecked() for checkbox in (
                self.chk_gps_geojson, self.chk_gps_gpx, self.chk_gps_csv
            )
        ):
            InfoBar.error(
                title=tr("GPS output format required"),
                content=tr("Select at least one GPS metadata format or disable GPS processing."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if self.process_gps_chk.isChecked() and not any(
            checkbox.isChecked() for checkbox in (
                self.chk_geo_laz, self.chk_geo_las, self.chk_geo_ply,
                self.chk_geo_pcd, self.chk_geo_geojson
            )
        ):
            InfoBar.error(
                title=tr("Georeferenced output format required"),
                content=tr("Select at least one georeferenced output format or disable automatic GPS georeferencing."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        method_key = ["sfm", "direct", "all"][self.method_combo.currentIndex()]
        scanner_key = ["auto", "eagle", "raven"][self.scanner_combo.currentIndex()]

        args = [
            "workflow",
            "--bag", bag,
            "--insv", insv,
            "--output", out,
            "--scanner", scanner_key,
            "--lidar-topic", self.lidar_topic.text().strip() or "/vanjee_722z",
            "--imu-topic", self.imu_topic.text().strip() or "/vanjee_imu_packets",
            "--threads", str(self.threads_spin.value()),
            "--fps", str(self.fps_spin.value()),
            "--method", method_key
        ]

        if self.recalibrate_chk.isChecked():
            args.append("--recalibrate")
        else:
            args.append("--no-recalibrate")

        if method_key in ("sfm", "all"):
            args.append("--run-spirula")

        if self.mask_persons_chk.isChecked():
            args.append("--mask-persons")
            args.extend(['--mask-backend', self.mask_backend_combo.currentData()])
            settings_path = Path(out).resolve() / 'mask_settings.json'
            if self.mask_config is not None:
                try:
                    settings_path.parent.mkdir(parents=True, exist_ok=True)
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
            args.extend(["--operator-radius", str(self.operator_radius_spin.value())])

        if self.lio_switch.isChecked():
            args.append("--lio")
        else:
            args.append("--vio")

        if not self.auto_sync_chk.isChecked() and self.dt_input.text().strip():
            args.extend(["--dt", self.dt_input.text().strip()])

        if not self.vulkan_chk.isChecked():
            args.append("--no-vulkan")

        if self.chk_ply.isChecked():
            args.append("--export-ply")
        if self.chk_pcd.isChecked():
            args.append("--export-pcd")
        if self.chk_colmap.isChecked():
            args.append("--export-colmap")

        if self.process_gps_chk.isChecked():
            args.append("--process-gps")
            gps_formats = []
            if self.chk_gps_geojson.isChecked():
                gps_formats.append("geojson")
            if self.chk_gps_gpx.isChecked():
                gps_formats.append("gpx")
            if self.chk_gps_csv.isChecked():
                gps_formats.append("csv")
            args.extend(["--gps-formats", *gps_formats])
            geo_formats = [name for name, checkbox in (
                ("laz", self.chk_geo_laz), ("las", self.chk_geo_las),
                ("ply", self.chk_geo_ply), ("pcd", self.chk_geo_pcd),
                ("geojson", self.chk_geo_geojson)) if checkbox.isChecked()]
            if geo_formats:
                args.extend(["--geo-formats", *geo_formats])

        self._prepare_workflow_progress(out, bag, insv, method_key)
        log_path = make_run_log_path(out, f'{Path(out).name}_workflow')
        if self.runner.start_job(
            args, log_path=log_path,
            initial_log=f">>> Starting Unified Workflow: {' '.join(args)}"
        ):
            self._current_log_path = log_path

    def _cancel_workflow(self):
        self.status_title.setText("Status: Cancelling...")
        self.status_desc.setText("Gracefully saving accumulated SLAM and colorization data...")
        self.btn_cancel.setEnabled(False)
        self.runner.cancel()

    def _on_job_started(self):
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status_title.setText("Status: Workflow Running")
        self.status_desc.setText("Executing end-to-end LiDAR-camera processing. SfM runs in the background.")
        self._workflow_started_at = time.monotonic()
        self._stage_started_at = None
        self._current_stage = 0
        self._current_stage_skipped = False
        self._stage_fraction = 0.0
        self._stage_fraction_from_log = False
        for bar in self.step_progress_bars:
            bar.setValue(0)
        self.overall_progress_bar.setValue(0)
        self.current_step_label.setText(tr("Workflow steps"))
        self._refresh_workflow_progress()
        self._progress_timer.start()

    def _on_job_finished(self, code: int):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._progress_timer.stop()

        if code == 0:
            if self._current_stage:
                self._complete_progress_stage(self._current_stage)
            for bar in self.step_progress_bars:
                bar.setValue(100)
            self.overall_progress_bar.setValue(100)
            self.overall_progress_label.setText(tr("Overall progress: 100%"))
            elapsed = time.monotonic() - self._workflow_started_at if self._workflow_started_at else 0
            self.eta_label.setText(tr("Completed in {time}").format(time=self._format_duration(elapsed)))
            self.current_step_label.setText(tr("All workflow steps completed"))
            self.status_title.setText("Status: Completed Successfully")
            self.status_desc.setText("Deliverables saved in 'deliverables/'. 3DGS dataset in 'colmap_3dgs/'.")
            InfoBar.success(
                title=tr("Unified Workflow Complete"),
                content=tr("Deliverables saved in 'deliverables/'. For 3DGS, select the 'colmap_3dgs' folder."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=6000,
                parent=self
            )
        elif code == 130:
            self.status_title.setText("Status: Cancelled & Flushed")
            self.status_desc.setText("Workflow cancelled by user. Completed steps and partial data are saved.")
        else:
            self.status_title.setText(f"Status: Failed (exit code {code})")
            current = tr(STAGE_NAMES[self._current_stage - 1]) if self._current_stage else tr("preparation")
            self.status_desc.setText(f"Failed during {current}. Check the saved log below.")
            InfoBar.error(
                title=tr("Workflow Failed"),
                content=tr("Execution exited with code {code}.", code=code),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=5000,
                parent=self
            )

    def _on_log_received(self, line: str):
        self.log_console.insertPlainText(line)
        self.log_console.ensureCursorVisible()
        for item in line.splitlines():
            self._update_workflow_progress_from_line(item)

    def _prepare_workflow_progress(self, output_dir, bag_path, insv_path, method):
        self._stage_root = Path(output_dir)
        self._stage_history = load_stage_history(self._stage_root)
        defaults = estimate_stage_seconds(
            self._stage_root, bag_path, insv_path,
            use_masks=self.mask_persons_chk.isChecked(),
            use_vulkan=self.vulkan_chk.isChecked(),
            export_colmap=self.chk_colmap.isChecked(),
            method=method,
            recalibrate=self.recalibrate_chk.isChecked(),
        )
        self._stage_estimates = stage_duration_estimates(defaults, self._stage_history)
        total = sum(self._stage_estimates)
        self.eta_label.setText(tr("Estimated remaining: about {time}").format(
            time=self._format_duration(total)))
        for index, bar in enumerate(self.step_progress_bars):
            bar.setValue(0)
            bar.setToolTip(tr("Approximate progress; the estimate improves from saved runs."))
            self.step_progress_labels[index].setText(f"{index + 1}. {tr(STAGE_NAMES[index])}")

    def _update_workflow_progress_from_line(self, line):
        event = parse_progress_line(line)
        if not event:
            return
        if event['kind'] == 'stage':
            stage = min(max(event['stage'], 1), len(STAGE_NAMES))
            name = event.get('name') or STAGE_NAMES[stage - 1]
            if self._current_stage and stage > self._current_stage:
                for finished in range(self._current_stage, stage):
                    self._complete_progress_stage(finished)
            if stage != self._current_stage:
                self._current_stage = stage
                self._stage_started_at = time.monotonic()
                self._stage_fraction = 0.0
                self._stage_fraction_from_log = False
                self._current_stage_skipped = (
                    stage == 5 and name.casefold().startswith('skipping'))
            self.current_step_label.setText(tr("Step {current}/{total}: {name}").format(
                current=stage, total=len(STAGE_NAMES), name=tr(name)))
            self.status_desc.setText(tr("Current step: {name}").format(name=tr(name)))
        elif event['kind'] == 'stage_fraction' and self._current_stage:
            fraction = event['fraction']
            self._stage_fraction = max(self._stage_fraction, fraction)
            self._stage_fraction_from_log = True
        elif event['kind'] == 'masks_ready' and self._current_stage == 5:
            self._stage_fraction = max(self._stage_fraction, 0.99)
            self._stage_fraction_from_log = True
        elif event['kind'] == 'complete':
            for stage in range(1, len(STAGE_NAMES) + 1):
                self._complete_progress_stage(stage)
            self.current_step_label.setText(tr("All workflow steps completed"))
        self._refresh_workflow_progress()

    def _complete_progress_stage(self, stage):
        index = stage - 1
        if index < 0 or index >= len(self.step_progress_bars):
            return
        if self.step_progress_bars[index].value() >= 100:
            return
        self.step_progress_bars[index].setValue(100)
        if self._current_stage == stage and self._stage_started_at is not None:
            duration = max(0.0, time.monotonic() - self._stage_started_at)
            if not self._current_stage_skipped:
                self._stage_history[index].append(duration)
                self._stage_history[index] = self._stage_history[index][-5:]
                if self._stage_root is not None:
                    try:
                        save_stage_history(self._stage_root, self._stage_history)
                    except OSError:
                        pass
            self._stage_started_at = None

    def _refresh_workflow_progress(self):
        if self._workflow_started_at is None:
            return
        elapsed_total = max(0.0, time.monotonic() - self._workflow_started_at)
        fractions = [bar.value() / 100.0 for bar in self.step_progress_bars]
        current_remaining = 0.0
        if self._current_stage:
            index = self._current_stage - 1
            stage_elapsed = max(0.0, time.monotonic() - (self._stage_started_at or time.monotonic()))
            estimate = max(1.0, self._stage_estimates[index])
            if not self._stage_fraction_from_log and stage_elapsed > estimate:
                estimate = stage_elapsed * 1.25
                self._stage_estimates[index] = estimate
            elapsed_fraction = min(0.95, stage_elapsed / estimate)
            fraction = max(self._stage_fraction, elapsed_fraction)
            self._stage_fraction = fraction
            fractions[index] = fraction
            current_remaining = estimate * (1.0 - fraction)
            if self._stage_fraction_from_log and fraction > 0.05:
                current_remaining = max(current_remaining,
                                        stage_elapsed * (1.0 - fraction) / fraction)

        weights = self._stage_estimates
        total_weight = max(1.0, sum(weights))
        overall = sum(weight * fraction for weight, fraction in zip(weights, fractions)) / total_weight
        percent = max(0, min(99, round(overall * 100)))
        self.overall_progress_bar.setValue(percent)
        self.overall_progress_label.setText(tr("Overall progress: {percent}%").format(percent=percent))
        for index, (bar, fraction) in enumerate(zip(self.step_progress_bars, fractions)):
            bar.setValue(100 if fraction >= 1.0 else max(0, min(99, round(fraction * 100))))

        if self._current_stage:
            future = sum(weights[self._current_stage:])
            remaining = current_remaining + future
            estimate_text = tr("Estimated left: about {time}").format(
                time=self._format_duration(remaining))
        else:
            remaining = max(0.0, total_weight - elapsed_total)
            estimate_text = tr("Estimated remaining: about {time}").format(
                time=self._format_duration(remaining))
        self.eta_label.setText(f"{self._format_duration(elapsed_total)} elapsed · {estimate_text}")

    @staticmethod
    def _format_duration(seconds):
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _on_error_occurred(self, err: str):
        self.log_console.appendPlainText(f"\n[ERROR] {err}\n")

    def retranslate_ui(self):
        """Update all text elements dynamically when the language changes."""
        self.header_title.setText(tr("Unified LiDAR-Camera Studio"))
        self.header_subtitle.setText(
            tr("Automated end-to-end pipeline: Inputs Selection → FAST-LIVO2 SLAM → "
               "Gyro Auto-Sync → Point Cloud Colorization → 3DGS Deliverables")
        )
        self.inputs_title.setText(tr("1. Input Datasets & Destination"))
        self.bag_label.setText(tr("LiDAR ROS Bag (.bag):"))
        self.bag_input.setPlaceholderText(tr("Select merged or raw ROS1 bag file (.bag)..."))
        self.btn_browse_bag.setText(tr("Browse"))
        self.insv_label.setText(tr("Insta360 Video (.insv):"))
        self.insv_input.setPlaceholderText(tr("Select Insta360 X4 / X6 video (.insv or .mp4)..."))
        self.btn_browse_insv.setText(tr("Browse"))
        self.out_label.setText(tr("Output Directory:"))
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.btn_browse_out.setText(tr("Browse"))

        self.config_title.setText(tr("2. Pipeline Processing Options"))
        self.scanner_label.setText(tr("Scanner Hardware Preset:"))
        self.lio_switch.setText(tr("Enable LiDAR + IMU Odometry (Fast LIO)"))
        self.lio_switch.setOnText(tr("LiDAR + IMU (LIO)"))
        self.lio_switch.setOffText(tr("LiDAR + camera + IMU (VIO)"))
        self.threads_caption.setText(tr("CPU Threads:"))
        self.lidar_caption.setText(tr("LiDAR Topic:"))
        self.imu_caption.setText(tr("IMU Topic:"))
        self.auto_sync_chk.setText(tr("Auto IMU Gyro Cross-Correlation"))
        self.dt_caption.setText(tr("Manual Δt (s):"))
        self.fps_caption.setText(tr("Extraction FPS:"))
        self.fps_spin.setToolTip(tr("Extraction rate is capped by the source video. For a 2 fps timelapse, use 2; a higher value does not create extra frames."))
        self.method_label.setText(tr("Colorization & Recalibration:"))

        cur_idx = self.method_combo.currentIndex()
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        self.method_combo.addItems([
            tr("Reconstruction-based colorization"),
            tr("Trajectory-based colorization"),
            tr("Compare both methods")
        ])
        self.method_combo.setCurrentIndex(cur_idx)
        self.method_combo.blockSignals(False)

        self.recalibrate_chk.setText(tr("Auto-Recalibrate Spatial Extrinsics from SfM Alignment"))
        self.mask_persons_chk.setText(tr("Generate masks"))
        self.mask_persons_chk.setToolTip(tr("RF-DETR masks people; use Mask settings to mark your scanner or mount."))
        self.mask_backend_label.setText(tr("Mask backend:"))
        self.mask_backend_combo.setItemText(0, tr("Vulkan"))
        self.mask_backend_combo.setItemText(1, tr("TensorRT"))
        self._update_mask_backend_tooltip()
        shapes = sum(sum(map(len, self.mask_config[key].values()))
                     for key in ('rectangles', 'ellipses', 'polygons')) if self.mask_config else 0
        self.mask_settings_btn.setText(tr('Mask settings ({n} shapes)').format(n=shapes) if self.mask_config else tr('Mask settings...'))
        if self._mask_downloader is None:
            self._refresh_mask_resource_state()
        self.operator_radius_caption.setText(tr("Operator removal radius (m):"))
        self.operator_radius_spin.setToolTip(tr("0 disables geometry removal. A positive radius enables person masks and protects dense planar surfaces such as doors and walls."))
        self.vulkan_chk.setText(tr("Use Vulkan for point-cloud colorization"))
        self.vulkan_chk.setToolTip(tr("This setting accelerates point-cloud colorization. Choose the RF-DETR mask backend separately above."))
        self.process_gps_chk.setText(tr("Automatically georeference using INSV GPS (when available)"))
        self.output_title.setText(tr("3. Select Deliverables to Generate"))
        self.output_subtitle.setText(tr("Choose which output formats will be built in this run:"))
        self.chk_ply.setText(tr("Export Colored Point Cloud (.PLY)"))
        self.chk_pcd.setText(tr("Export Colored Point Cloud (.PCD)"))
        self.chk_colmap.setText(tr("Metric-Scaled 3DGS COLMAP Dataset"))
        self.georef_formats_label.setText(tr("Automatic georeferenced cloud formats:"))
        self.gps_formats_label.setText(tr("GPS metadata formats (select one or more):"))
        self.chk_gps_geojson.setText(tr("GeoJSON"))
        self.chk_gps_gpx.setText(tr("GPX"))
        self.chk_gps_csv.setText(tr("CSV"))
        self.btn_run.setText(tr("Start Unified Workflow"))
        self.btn_cancel.setText(tr("Cancel & Save"))
        for index, name in enumerate(STAGE_NAMES):
            self.step_progress_labels[index].setText(f"{index + 1}. {tr(name)}")
        self.log_title.setText(tr("Live Workflow Execution Console"))
        self.btn_clear_log.setToolTip(tr("Clear visible log (saved file remains)"))
        self.btn_export_log.setToolTip(tr("Export log..."))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Select Bag and INSV files, then click 'Start Unified Workflow'."))
