"""FAST-LIVO2 mapping view."""
from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QFrame
)
from qfluentwidgets import (
    CardWidget, ElevatedCardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton, ToolButton,
    LineEdit, SpinBox, SwitchButton, ProgressBar, IndeterminateProgressBar,
    PlainTextEdit, FluentIcon, InfoBar, InfoBarPosition
)
from raven_app.process_runner import ProcessRunner
from raven_app.i18n import tr


class SlamView(QWidget):
    """WinUI Fluent view for running native FAST-LIVO2 SLAM."""

    def __init__(self, runner: ProcessRunner, parent=None):
        super().__init__(parent)
        self.setObjectName("SlamView")
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
        self.header_title = TitleLabel(tr("FAST-LIVO2 SLAM Engine"))
        self.header_subtitle = CaptionLabel(
            tr("High-performance native MSVC C++17 odometry and mapping without ROS or WSL")
        )
        header_layout.addWidget(self.header_title)
        header_layout.addWidget(self.header_subtitle)
        layout.addLayout(header_layout)

        # Configuration Card
        config_card = CardWidget(self)
        card_layout = QVBoxLayout(config_card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(14)

        self.card_title = SubtitleLabel(tr("Input & Trajectory Settings"))
        card_layout.addWidget(self.card_title)

        # Bag file row
        bag_row = QHBoxLayout()
        self.bag_label = BodyLabel(tr("ROS Bag File:"))
        self.bag_label.setFixedWidth(140)
        self.bag_input = LineEdit()
        self.bag_input.setPlaceholderText(tr("Select merged or raw ROS1 bag file (.bag)..."))
        self.bag_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.bag_browse.clicked.connect(self._browse_bag)
        bag_row.addWidget(self.bag_label)
        bag_row.addWidget(self.bag_input)
        bag_row.addWidget(self.bag_browse)
        card_layout.addLayout(bag_row)

        # Output folder row
        out_row = QHBoxLayout()
        self.out_label = BodyLabel(tr("Output Folder:"))
        self.out_label.setFixedWidth(140)
        self.out_input = LineEdit()
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.out_browse = PushButton(tr("Browse"), icon=FluentIcon.FOLDER)
        self.out_browse.clicked.connect(self._browse_output)
        out_row.addWidget(self.out_label)
        out_row.addWidget(self.out_input)
        out_row.addWidget(self.out_browse)
        card_layout.addLayout(out_row)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        card_layout.addWidget(sep)

        # LIO mode and Topic settings
        toggle_row = QHBoxLayout()
        self.lio_switch = SwitchButton(text=tr("LiDAR + IMU mode (camera fusion disabled)"))
        self.lio_switch.setChecked(True)
        self.lio_switch.checkedChanged.connect(self._toggle_lio)
        toggle_row.addWidget(self.lio_switch)
        toggle_row.addStretch()

        self.threads_label = BodyLabel(tr("CPU Threads:"))
        self.threads_spin = SpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        toggle_row.addWidget(self.threads_label)
        toggle_row.addWidget(self.threads_spin)
        card_layout.addLayout(toggle_row)

        # Topic configuration row
        topics_row = QHBoxLayout()
        topics_row.setSpacing(12)

        self.lidar_label = CaptionLabel(tr("LiDAR Topic:"))
        self.lidar_input = LineEdit()
        self.lidar_input.setText("/vanjee_722z")

        self.imu_label = CaptionLabel(tr("IMU Topic:"))
        self.imu_input = LineEdit()
        self.imu_input.setText("/vanjee_imu_packets")

        self.cam_label = CaptionLabel(tr("Camera Topic:"))
        self.cam_input = LineEdit()
        self.cam_input.setText("/camera_front/image/compressed")
        self.cam_input.setEnabled(False)

        topics_row.addWidget(self.lidar_label)
        topics_row.addWidget(self.lidar_input)
        topics_row.addWidget(self.imu_label)
        topics_row.addWidget(self.imu_input)
        topics_row.addWidget(self.cam_label)
        topics_row.addWidget(self.cam_input)
        card_layout.addLayout(topics_row)

        layout.addWidget(config_card)

        # Action & Status Card
        action_card = CardWidget(self)
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(20, 14, 20, 14)

        status_box = QVBoxLayout()
        self.status_title = StrongBodyLabel(tr("Status: Ready"))
        self.status_desc = CaptionLabel(tr("Ready to start FAST-LIVO2 mapping execution."))
        status_box.addWidget(self.status_title)
        status_box.addWidget(self.status_desc)
        action_layout.addLayout(status_box)

        action_layout.addStretch()

        self.progress_bar = ProgressBar()
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setVisible(False)
        action_layout.addWidget(self.progress_bar)

        self.btn_run = PrimaryPushButton(tr("Start Mapping"), icon=FluentIcon.PLAY)
        self.btn_run.clicked.connect(self._start_mapping)
        self.btn_cancel = PushButton(tr("Cancel & Save"), icon=FluentIcon.CLOSE)
        self.btn_cancel.clicked.connect(self._cancel_mapping)
        self.btn_cancel.setEnabled(False)

        action_layout.addWidget(self.btn_run)
        action_layout.addWidget(self.btn_cancel)
        layout.addWidget(action_card)

        # Terminal & Log Card
        log_card = CardWidget(self)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 12, 16, 12)
        log_layout.setSpacing(8)

        log_header = QHBoxLayout()
        self.log_title = SubtitleLabel(tr("Live Execution Console"))
        self.btn_clear_log = ToolButton(FluentIcon.DELETE)
        self.btn_clear_log.setToolTip(tr("Clear console"))
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
        self.runner.error_occurred.connect(self._on_error_occurred)

    def _toggle_lio(self, checked: bool):
        self.cam_input.setEnabled(not checked)

    def _browse_bag(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROS Bag", "", "ROS Bag files (*.bag);;All files (*.*)"
        )
        if path:
            self.bag_input.setText(path)

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder:
            self.out_input.setText(folder)

    def set_bag_and_topics(self, bag_path: str, lidar_topic: str = None, imu_topic: str = None, img_topic: str = None):
        """Helper to pre-fill settings from Bag Inspector."""
        self.bag_input.setText(bag_path)
        if lidar_topic:
            self.lidar_input.setText(lidar_topic)
        if imu_topic:
            self.imu_input.setText(imu_topic)
        if img_topic:
            self.cam_input.setText(img_topic)
            self.lio_switch.setChecked(False)

    def _clear_log(self):
        self.log_console.clear()

    def _start_mapping(self):
        bag = self.bag_input.text().strip()
        out = self.out_input.text().strip()

        if not bag or not Path(bag).is_file():
            InfoBar.error(
                title=tr("Invalid Input"),
                content=tr("Please specify a valid ROS bag file (.bag)."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        if not out:
            InfoBar.error(
                title=tr("Missing Output Folder"),
                content=tr("Please choose an output directory for SLAM deliverables."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        args = [
            'slam',
            '--bag', bag,
            '--output', out,
            '--lidar-topic', self.lidar_input.text().strip() or '/vanjee_722z',
            '--imu-topic', self.imu_input.text().strip() or '/vanjee_imu_packets',
            '--threads', str(self.threads_spin.value())
        ]
        if self.lio_switch.isChecked():
            args.append('--lio')
        else:
            args.extend(['--image-topic', self.cam_input.text().strip() or '/camera_front/image/compressed'])

        self.log_console.appendPlainText(f"\n>>> Starting FAST-LIVO2: {' '.join(args)}\n")
        self.runner.start_job(args)

    def _cancel_mapping(self):
        self.status_title.setText("Status: Cancelling...")
        self.status_desc.setText("Flushing accumulated voxel map data to PCD...")
        self.btn_cancel.setEnabled(False)
        self.runner.cancel()

    def _on_job_started(self):
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_title.setText("Status: Processing FAST-LIVO2")
        self.status_desc.setText("Streaming sensor frames through native estimator pipeline...")

    def _on_job_finished(self, code: int):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        if code == 0:
            self.status_title.setText("Status: Completed Successfully")
            self.status_desc.setText("Point cloud and trajectory written to output directory.")
            InfoBar.success(
                title=tr("Mapping Completed"),
                content=tr("FAST-LIVO2 execution finished with return code 0."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self
            )
        elif code == 130:
            self.status_title.setText("Status: Cancelled & Flushed")
            self.status_desc.setText("Run interrupted by user. Accumulated partial map saved.")
            InfoBar.warning(
                title=tr("Job Cancelled"),
                content=tr("Accumulated map data was saved."),
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=4000,
                parent=self
            )
        else:
            self.status_title.setText(f"Status: Failed (exit code {code})")
            self.status_desc.setText("Execution encountered an error. Check console log below.")
            InfoBar.error(
                title=tr("Execution Failed"),
                content=tr("FAST-LIVO2 stopped with exit code {code}.", code=code),
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
        self.header_title.setText(tr("FAST-LIVO2 SLAM Engine"))
        self.header_subtitle.setText(
            tr("High-performance native MSVC C++17 odometry and mapping without ROS or WSL")
        )
        self.card_title.setText(tr("Input & Trajectory Settings"))
        self.bag_label.setText(tr("ROS Bag File:"))
        self.bag_input.setPlaceholderText(tr("Select merged or raw ROS1 bag file (.bag)..."))
        self.bag_browse.setText(tr("Browse"))
        self.out_label.setText(tr("Output Folder:"))
        self.out_input.setPlaceholderText(tr("Select output folder for deliverables, SLAM, and images..."))
        self.out_browse.setText(tr("Browse"))
        self.lio_switch.setText(tr("LiDAR + IMU mode (camera fusion disabled)"))
        self.threads_label.setText(tr("CPU Threads:"))
        self.lidar_label.setText(tr("LiDAR Topic:"))
        self.imu_label.setText(tr("IMU Topic:"))
        self.cam_label.setText(tr("Camera Topic:"))
        self.btn_run.setText(tr("Start Mapping"))
        self.btn_cancel.setText(tr("Cancel & Save"))
        self.log_title.setText(tr("Live Execution Console"))
        self.btn_clear_log.setToolTip(tr("Clear console"))

        if not self.runner.is_busy:
            self.status_title.setText(tr("Status: Ready"))
            self.status_desc.setText(tr("Ready to start FAST-LIVO2 mapping execution."))

