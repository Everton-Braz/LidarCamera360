"""Unified Studio Workflow View - Microsoft UI XAML / Fluent Design System."""
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
        title = TitleLabel("Unified LiDAR-Camera Studio")
        subtitle = CaptionLabel(
            "Automated end-to-end pipeline: Inputs Selection → FAST-LIVO2 SLAM → "
            "Gyro Auto-Sync → Point Cloud Colorization → 3DGS Deliverables"
        )
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # ----------------------------------------------------------------------
        # 1. Dataset Inputs Card
        # ----------------------------------------------------------------------
        inputs_card = CardWidget(content)
        inputs_layout = QVBoxLayout(inputs_card)
        inputs_layout.setContentsMargins(20, 18, 20, 18)
        inputs_layout.setSpacing(12)

        inputs_layout.addWidget(SubtitleLabel("1. Input Datasets & Destination"))

        # Row 1: Bag File
        bag_row = QHBoxLayout()
        bag_label = BodyLabel("LiDAR ROS Bag (.bag):")
        bag_label.setFixedWidth(170)
        self.bag_input = LineEdit()
        self.bag_input.setPlaceholderText("Select merged or raw ROS1 bag file (.bag)...")
        self.btn_browse_bag = PushButton("Browse", icon=FluentIcon.FOLDER)
        self.btn_browse_bag.clicked.connect(self._browse_bag)
        bag_row.addWidget(bag_label)
        bag_row.addWidget(self.bag_input)
        bag_row.addWidget(self.btn_browse_bag)
        inputs_layout.addLayout(bag_row)

        # Row 2: INSV Video
        insv_row = QHBoxLayout()
        insv_label = BodyLabel("Insta360 Video (.insv):")
        insv_label.setFixedWidth(170)
        self.insv_input = LineEdit()
        self.insv_input.setPlaceholderText("Select Insta360 X4 8K video (.insv or .mp4)...")
        self.btn_browse_insv = PushButton("Browse", icon=FluentIcon.VIDEO)
        self.btn_browse_insv.clicked.connect(self._browse_insv)
        insv_row.addWidget(insv_label)
        insv_row.addWidget(self.insv_input)
        insv_row.addWidget(self.btn_browse_insv)
        inputs_layout.addLayout(insv_row)

        # Row 3: Output Folder
        out_row = QHBoxLayout()
        out_label = BodyLabel("Output Directory:")
        out_label.setFixedWidth(170)
        self.out_input = LineEdit()
        self.out_input.setPlaceholderText("Select output folder for deliverables, SLAM, and images...")
        self.btn_browse_out = PushButton("Browse", icon=FluentIcon.FOLDER)
        self.btn_browse_out.clicked.connect(self._browse_output)
        out_row.addWidget(out_label)
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

        config_layout.addWidget(SubtitleLabel("2. Pipeline Processing Options"))

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(10)

        # SLAM options
        self.lio_switch = SwitchButton(text="LiDAR + IMU Odometry (Fast LIO)")
        self.lio_switch.setChecked(True)
        grid.addWidget(self.lio_switch, 0, 0)

        threads_layout = QHBoxLayout()
        threads_layout.addWidget(CaptionLabel("CPU Threads:"))
        self.threads_spin = SpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        threads_layout.addWidget(self.threads_spin)
        grid.addLayout(threads_layout, 0, 1)

        # Topics
        topics_layout = QHBoxLayout()
        topics_layout.addWidget(CaptionLabel("LiDAR Topic:"))
        self.lidar_topic = LineEdit()
        self.lidar_topic.setText("/vanjee_722z")
        topics_layout.addWidget(self.lidar_topic)
        topics_layout.addWidget(CaptionLabel("IMU Topic:"))
        self.imu_topic = LineEdit()
        self.imu_topic.setText("/vanjee_imu_packets")
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
        self.auto_sync_chk = CheckBox("Auto IMU Gyro Cross-Correlation (sub-ms Δt)")
        self.auto_sync_chk.setChecked(True)
        self.auto_sync_chk.stateChanged.connect(self._toggle_auto_sync)
        sync_row.addWidget(self.auto_sync_chk)

        sync_row.addWidget(CaptionLabel("Manual Δt (s):"))
        self.dt_input = LineEdit()
        self.dt_input.setPlaceholderText("Optional (e.g. 1.892)")
        self.dt_input.setFixedWidth(120)
        self.dt_input.setEnabled(False)
        sync_row.addWidget(self.dt_input)
        sync_row.addStretch()

        sync_row.addWidget(CaptionLabel("Extraction FPS:"))
        self.fps_spin = DoubleSpinBox()
        self.fps_spin.setRange(0.1, 60.0)
        self.fps_spin.setValue(1.0)
        self.fps_spin.setSingleStep(0.5)
        sync_row.addWidget(self.fps_spin)

        config_layout.addLayout(sync_row)

        # Colorization Method
        method_row = QHBoxLayout()
        method_row.addWidget(BodyLabel("Colorization Method:"))
        self.method_combo = ComboBox()
        self.method_combo.addItems([
            "Direct Calibrated Rigid (Fast, Sub-Millimeter Sync)",
            "Spirula SfM Consensus (Bundle Adjustment + ICP)",
            "All / Dual-Method Comparison"
        ])
        self.method_combo.setCurrentIndex(0)
        method_row.addWidget(self.method_combo)
        method_row.addStretch()
        config_layout.addLayout(method_row)

        layout.addWidget(config_card)

        # ----------------------------------------------------------------------
        # 3. Deliverables Output Selection Card
        # ----------------------------------------------------------------------
        output_card = ElevatedCardWidget(content)
        output_layout = QVBoxLayout(output_card)
        output_layout.setContentsMargins(20, 18, 20, 18)
        output_layout.setSpacing(10)

        output_layout.addWidget(SubtitleLabel("3. Select Deliverables to Generate"))
        output_layout.addWidget(CaptionLabel("Choose which output formats will be built in this run:"))

        self.chk_ply = CheckBox("Export Colored Point Cloud (.PLY)  •  Open standard (CloudCompare, MeshLab, Blender)")
        self.chk_ply.setChecked(True)
        output_layout.addWidget(self.chk_ply)

        self.chk_pcd = CheckBox("Export Colored Point Cloud (.PCD)  •  PCL format with packed RGB fields")
        self.chk_pcd.setChecked(True)
        output_layout.addWidget(self.chk_pcd)

        self.chk_colmap = CheckBox("COLMAP Dataset Ready for 3DGS Training  •  cameras.txt, images.txt, points3D.txt, points3D.ply")
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
        self.status_title = StrongBodyLabel("Status: Ready")
        self.status_desc = CaptionLabel("Select Bag and INSV files, then click 'Start Unified Workflow'.")
        status_box.addWidget(self.status_title)
        status_box.addWidget(self.status_desc)
        action_layout.addLayout(status_box)

        action_layout.addStretch()

        self.progress_bar = ProgressBar()
        self.progress_bar.setFixedWidth(220)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        self.btn_run = PrimaryPushButton("Start Unified Workflow", icon=FluentIcon.PLAY)
        self.btn_run.clicked.connect(self._start_workflow)
        self.btn_cancel = PushButton("Cancel & Save", icon=FluentIcon.CLOSE)
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
        log_title = SubtitleLabel("Live Workflow Execution Console")
        self.btn_clear_log = ToolButton(FluentIcon.DELETE)
        self.btn_clear_log.setToolTip("Clear console")
        self.btn_clear_log.clicked.connect(self._clear_log)
        log_header.addWidget(log_title)
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
                title="Invalid LiDAR Bag",
                content="Please select a valid ROS bag file (.bag).",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if not insv or not Path(insv).is_file():
            InfoBar.error(
                title="Invalid Insta360 Video",
                content="Please select an Insta360 video file (.insv or .mp4).",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if not out:
            InfoBar.error(
                title="Missing Output Folder",
                content="Please specify an output folder for deliverables.",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        method_key = ["direct", "sfm", "all"][self.method_combo.currentIndex()]

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

        if self.lio_switch.isChecked():
            args.append("--lio")

        if not self.auto_sync_chk.isChecked() and self.dt_input.text().strip():
            args.extend(["--dt", self.dt_input.text().strip()])

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
                title="Unified Workflow Complete",
                content="Point clouds and 3DGS COLMAP models generated successfully!",
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
                title="Workflow Failed",
                content=f"Execution exited with code {code}.",
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
