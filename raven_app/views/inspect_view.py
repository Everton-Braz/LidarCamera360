"""ROS Bag Inspector View - Microsoft UI XAML / Fluent Design System."""
from pathlib import Path
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QHeaderView, QTableWidgetItem
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton, PushButton,
    LineEdit, TableWidget, FluentIcon, InfoBar, InfoBarPosition
)
from raven_app.bag_io import inspect_bags


class InspectWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, bag_path: str):
        super().__init__()
        self.bag_path = bag_path

    def run(self):
        try:
            info = inspect_bags([self.bag_path])
            self.finished.emit(info)
        except Exception as e:
            self.error.emit(str(e))


class InspectView(QWidget):
    """WinUI Fluent view for inspecting ROS1 bags without ROS."""
    apply_to_slam_requested = pyqtSignal(str, str, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("InspectView")
        self._current_bag = ""
        self._worker = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = TitleLabel("ROS Bag Inspector")
        subtitle = CaptionLabel(
            "Extract stream metadata, topic structures, and sensor counts from ROS1 bags offline"
        )
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # File Selection Card
        card = CardWidget(self)
        card_layout = QHBoxLayout(card)
        card_layout.setContentsMargins(20, 16, 20, 16)

        label = BodyLabel("ROS Bag:")
        self.bag_input = LineEdit()
        self.bag_input.setPlaceholderText("Select a .bag file to analyze...")
        self.btn_browse = PushButton("Browse", icon=FluentIcon.FOLDER)
        self.btn_browse.clicked.connect(self._browse)
        self.btn_inspect = PrimaryPushButton("Inspect Bag", icon=FluentIcon.SEARCH)
        self.btn_inspect.clicked.connect(self._inspect)

        card_layout.addWidget(label)
        card_layout.addWidget(self.bag_input)
        card_layout.addWidget(self.btn_browse)
        card_layout.addWidget(self.btn_inspect)
        layout.addWidget(card)

        # Overview Stats Card
        self.stats_card = CardWidget(self)
        stats_layout = QHBoxLayout(self.stats_card)
        stats_layout.setContentsMargins(20, 16, 20, 16)

        self.lbl_duration = StrongBodyLabel("Duration: --")
        self.lbl_msgs = StrongBodyLabel("Total Messages: --")
        self.lbl_lidar = CaptionLabel("Detected LiDAR: --")
        self.lbl_imu = CaptionLabel("Detected IMU: --")
        self.lbl_cam = CaptionLabel("Detected Camera: --")

        stat_col1 = QVBoxLayout()
        stat_col1.addWidget(self.lbl_duration)
        stat_col1.addWidget(self.lbl_msgs)

        stat_col2 = QVBoxLayout()
        stat_col2.addWidget(self.lbl_lidar)
        stat_col2.addWidget(self.lbl_imu)
        stat_col2.addWidget(self.lbl_cam)

        stats_layout.addLayout(stat_col1)
        stats_layout.addSpacing(40)
        stats_layout.addLayout(stat_col2)
        stats_layout.addStretch()

        self.btn_apply = PushButton("Apply Topics to SLAM", icon=FluentIcon.SEND)
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply_to_slam)
        stats_layout.addWidget(self.btn_apply)

        layout.addWidget(self.stats_card)

        # Topics Table Card
        table_card = CardWidget(self)
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(16, 14, 16, 14)
        table_layout.setSpacing(10)

        table_title = SubtitleLabel("Bag Topic Catalog")
        table_layout.addWidget(table_title)

        self.table = TableWidget(self)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Topic Name", "Message Type", "Message Count"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setBorderVisible(True)
        self.table.setBorderRadius(6)
        table_layout.addWidget(self.table)

        layout.addWidget(table_card, stretch=1)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ROS Bag", "", "ROS Bag files (*.bag);;All files (*.*)"
        )
        if path:
            self.bag_input.setText(path)
            self._inspect()

    def _inspect(self):
        bag = self.bag_input.text().strip()
        if not bag or not Path(bag).is_file():
            InfoBar.error(
                title="Invalid Bag File",
                content="Please select an existing .bag file to inspect.",
                orient=Qt.Orientation.Horizontal,
                position=InfoBarPosition.TOP,
                duration=3500,
                parent=self
            )
            return

        self._current_bag = bag
        self.btn_inspect.setEnabled(False)
        self.btn_apply.setEnabled(False)

        self._worker = InspectWorker(bag)
        self._worker.finished.connect(self._on_inspection_finished)
        self._worker.error.connect(self._on_inspection_error)
        self._worker.start()

    def _on_inspection_finished(self, info: dict):
        self.btn_inspect.setEnabled(True)
        dur = info.get('duration_seconds', 0.0)
        mins = int(dur // 60)
        secs = dur % 60
        self.lbl_duration.setText(f"Duration: {mins}m {secs:.1f}s ({dur:.1f}s)")
        self.lbl_msgs.setText(f"Total Messages: {info.get('messages', 0):,}")

        connections = info.get('connections', [])
        self.table.setRowCount(len(connections))

        best_lidar = None
        best_imu = None
        best_cam = None

        for row, conn in enumerate(connections):
            topic = conn.get('topic', '')
            msgtype = conn.get('type', '')
            count = conn.get('messages', 0)

            self.table.setItem(row, 0, QTableWidgetItem(topic))
            self.table.setItem(row, 1, QTableWidgetItem(msgtype))
            self.table.setItem(row, 2, QTableWidgetItem(f"{count:,}"))

            if 'PointCloud2' in msgtype or 'vanjee_722z' in topic:
                best_lidar = topic
            if 'Imu' in msgtype or 'imu' in topic:
                best_imu = topic
            if 'Image' in msgtype or 'camera' in topic:
                best_cam = topic

        self._detected_lidar = best_lidar or "/vanjee_722z"
        self._detected_imu = best_imu or "/vanjee_imu_packets"
        self._detected_cam = best_cam or "/camera_front/image/compressed"

        self.lbl_lidar.setText(f"Detected LiDAR: {self._detected_lidar}")
        self.lbl_imu.setText(f"Detected IMU: {self._detected_imu}")
        self.lbl_cam.setText(f"Detected Camera: {self._detected_cam}")

        self.btn_apply.setEnabled(True)
        InfoBar.success(
            title="Inspection Complete",
            content=f"Found {len(connections)} topics across {info.get('messages', 0):,} records.",
            orient=Qt.Orientation.Horizontal,
            position=InfoBarPosition.TOP,
            duration=3500,
            parent=self
        )

    def _on_inspection_error(self, err: str):
        self.btn_inspect.setEnabled(True)
        InfoBar.error(
            title="Inspection Error",
            content=f"Failed to read bag: {err}",
            orient=Qt.Orientation.Horizontal,
            position=InfoBarPosition.TOP,
            duration=4000,
            parent=self
        )

    def _apply_to_slam(self):
        self.apply_to_slam_requested.emit(
            self._current_bag,
            getattr(self, '_detected_lidar', ''),
            getattr(self, '_detected_imu', ''),
            getattr(self, '_detected_cam', '')
        )
