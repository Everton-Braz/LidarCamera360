"""Unified processing workflow view."""
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog, QFrame
)
from qfluentwidgets import (
    CardWidget, ElevatedCardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton, ToolButton,
    LineEdit, DoubleSpinBox, SpinBox, ComboBox, CheckBox, SwitchButton,
    ProgressBar, PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition,
    SmoothScrollArea
)
from raven_app.process_runner import ProcessRunner
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
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area to comfortably accommodate all cards
        scroll = SmoothScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        content = QWidget()


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
        self._init_ui()
        self._connect_signals()

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
        self.insv_input.setPlaceholderText(tr("Select Insta360 X4 8K video (.insv or .mp4)..."))
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

        # SLAM options
        self.lio_switch = SwitchButton(text=tr("LiDAR + IMU Odometry (Fast LIO)"))
        self.lio_switch.setChecked(True)
        grid.addWidget(self.lio_switch, 0, 0)

        threads_layout = QHBoxLayout()
        self.threads_caption = CaptionLabel(tr("CPU Threads:"))
        self.threads_spin = SpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        threads_layout.addWidget(self.threads_caption)
        threads_layout.addWidget(self.threads_spin)
        grid.addLayout(threads_layout, 0, 1)

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
        grid.addLayout(topics_layout, 1, 0, 1, 2)

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
        self.fps_spin.setValue(1.0)
        self.fps_spin.setSingleStep(0.5)
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
        self.recalibrate_chk = CheckBox(tr("Auto-Recalibrate Spatial Extrinsics (T_LC0, T_LC1) from SfM Alignment"))
        self.recalibrate_chk.setChecked(True)
        recalib_row.addWidget(self.recalibrate_chk)
        recalib_row.addStretch()
        config_layout.addLayout(recalib_row)

        self.vulkan_chk = CheckBox(tr("Enable Vulkan GPU Compute Acceleration"))
        self.vulkan_chk.setChecked(True)
        config_layout.addWidget(self.vulkan_chk)

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

        self.chk_ply = CheckBox(tr("Export Colored Point Cloud (.PLY)  •  Open standard (CloudCompare, MeshLab, Blender)"))
        self.chk_ply.setChecked(True)
        output_layout.addWidget(self.chk_ply)

        self.chk_pcd = CheckBox(tr("Export Colored Point Cloud (.PCD)  •  PCL format with packed RGB fields"))
        self.chk_pcd.setChecked(True)
        output_layout.addWidget(self.chk_pcd)

        self.chk_colmap = CheckBox(tr("Metric-Scaled 3DGS COLMAP Dataset  •  Spirula SfM converted to LiDAR Metric Ground Truth (Nerfstudio/PostShot/LichtFeld)"))
        self.chk_colmap.setChecked(True)
        output_layout.addWidget(self.chk_colmap)

        layout.addWidget(output_card)

        # ----------------------------------------------------------------------
        # 4. Action & Status Bar
        # ----------------------------------------------------------------------
        action_card = CardWidget(content)
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(20, 14, 20, 14)

        status_box = QVBoxLayout()
        self.status_title = StrongBodyLabel(tr("Status: Ready"))
        self.status_desc = CaptionLabel(tr("Select Bag and INSV files, then click 'Start Unified Workflow'."))
        status_box.addWidget(self.status_title)
        status_box.addWidget(self.status_desc)
        action_layout.addLayout(status_box)

        action_layout.addStretch()

        self.progress_bar = ProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        self.btn_run = PrimaryPushButton(tr("Start Unified Workflow"), icon=FluentIcon.PLAY)
        self.btn_run.clicked.connect(self._start_workflow)
        self.btn_cancel = PushButton(tr("Cancel & Save"), icon=FluentIcon.CLOSE)
        self.btn_cancel.clicked.connect(self._cancel_workflow)
        self.btn_cancel.setEnabled(False)

        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_cancel)
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
        self.btn_clear_log.setToolTip(tr("Clear console"))
        self.btn_clear_log.clicked.connect(self._clear_log)
        log_header.addWidget(self.log_title)
        log_header.addStretch()
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

    def _toggle_auto_sync(self, state):
        self.dt_input.setEnabled(not self.auto_sync_chk.isChecked())

    def _browse_bag(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROS Bag (.bag)", "", "ROS Bag files (*.bag);;All files (*.*)"
        )
        if path:
            self.bag_input.setText(path)
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

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_input.setText(folder)

    def _clear_log(self):
        self.log_console.clear()

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

        method_key = ["reconstruction", "trajectory", "all"][self.method_combo.currentIndex()]

        args = [
            "workflow",
            "--bag", bag,
            "--insv", insv,
            "--output", out,
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

        if self.lio_switch.isChecked():
            args.append("--lio")

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

        self.log_console.appendPlainText(f"\n>>> Starting Unified Workflow: {' '.join(args)}\n")
        self.runner.start_job(args)
    def _cancel_workflow(self):
        self.status_title.setText("Status: Cancelling...")
        self.status_desc.setText("Gracefully saving accumulated SLAM and colorization data...")
        self.btn_cancel.setEnabled(False)
        self.runner.cancel()

    def _on_job_started(self):
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.status_title.setText("Status: Workflow Running")
        self.status_desc.setText("Executing end-to-end LiDAR-camera processing...")

    def _on_job_finished(self, code: int):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)

        if code == 0:
            self.status_title.setText("Status: Completed Successfully")
            self.status_desc.setText("All deliverables generated and saved.")
            InfoBar.success(
                title=tr("Unified Workflow Complete"),
                content=tr("Point clouds and 3DGS COLMAP models generated successfully!"),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self
            )
        elif code == 130:
            self.status_title.setText("Status: Cancelled & Flushed")
            self.status_desc.setText("Workflow cancelled by user. Partial data saved.")
        else:
            self.status_title.setText(f"Status: Failed (exit code {code})")
            self.status_desc.setText("Process encountered an error. Check console output below.")
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
        self.insv_input.setPlaceholderText(tr("Select Insta360 X4 8K video (.insv or .mp4)..."))
        self.btn_browse_insv.setText(tr("Browse"))
        self.out_label.setText(tr("Output Directory:"))
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.btn_browse_out.setText(tr("Browse"))

        self.config_title.setText(tr("2. Pipeline Processing Options"))
        self.lio_switch.setText(tr("LiDAR + IMU Odometry (Fast LIO)"))
        self.threads_caption.setText(tr("CPU Threads:"))
        self.lidar_caption.setText(tr("LiDAR Topic:"))
        self.imu_caption.setText(tr("IMU Topic:"))
        self.auto_sync_chk.setText(tr("Auto IMU Gyro Cross-Correlation"))
        self.dt_caption.setText(tr("Manual Δt (s):"))
        self.fps_caption.setText(tr("Extraction FPS:"))
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

        self.recalibrate_chk.setText(tr("Auto-Recalibrate Spatial Extrinsics (T_LC0, T_LC1) from SfM Alignment"))
        self.vulkan_chk.setText(tr("Enable Vulkan GPU Compute Acceleration"))
        self.output_title.setText(tr("3. Select Deliverables to Generate"))
        self.output_subtitle.setText(tr("Choose which output formats will be built in this run:"))
        self.chk_ply.setText(tr("Export Colored Point Cloud (.PLY)  •  Open standard (CloudCompare, MeshLab, Blender)"))
        self.chk_pcd.setText(tr("Export Colored Point Cloud (.PCD)  •  PCL format with packed RGB fields"))
        self.chk_colmap.setText(tr("Metric-Scaled 3DGS COLMAP Dataset  •  Spirula SfM converted to LiDAR Metric Ground Truth (Nerfstudio/PostShot/LichtFeld)"))
        self.btn_run.setText(tr("Start Unified Workflow"))
        self.btn_cancel.setText(tr("Cancel & Save"))
        self.log_title.setText(tr("Live Workflow Execution Console"))
        self.btn_clear_log.setToolTip(tr("Clear console"))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Select Bag and INSV files, then click 'Start Unified Workflow'."))
