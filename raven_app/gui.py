"""Desktop interface built with PyQt6 and qfluentwidgets."""
import os
import sys
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox
from qfluentwidgets import (
    FluentWindow, NavigationItemPosition, FluentIcon,
    setTheme, Theme, isDarkTheme
)

from raven_app.process_runner import ProcessRunner
from raven_app.views.unified_workflow_view import UnifiedWorkflowView
from raven_app.views.slam_view import SlamView
from raven_app.views.colorize_view import ColorizeView
from raven_app.views.inspect_view import InspectView
from raven_app.views.calibration_view import CalibrationView
from raven_app.views.doctor_view import DoctorView
from raven_app.views.settings_view import SettingsView
from raven_app.config import load_config, get_app_root
from raven_app.i18n import set_language, tr
try:
    from raven_app.branding import APP_NAME, APP_DESCRIPTION
except ImportError:
    APP_NAME, APP_DESCRIPTION = "RavenCalibrator", "LiDAR and camera calibration"


class RavenMainWindow(FluentWindow):
    """Main application window built with PyQt6 and qfluentwidgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} • {APP_DESCRIPTION}")
        self.resize(1160, 820)
        self.setMinimumSize(960, 660)

        # Initialize shared execution runner
        self.runner = ProcessRunner(self)

        # Initialize Sub-interfaces / Views
        self.workflow_view = UnifiedWorkflowView(self.runner, self)
        self.slam_view = SlamView(self.runner, self)
        self.colorize_view = ColorizeView(self.runner, self)
        self.inspect_view = InspectView(self)
        self.calib_view = CalibrationView(self)
        self.doctor_view = DoctorView(self)
        self.settings_view = SettingsView(self)

        # Wire up inter-view communication
        self.inspect_view.apply_to_slam_requested.connect(self._on_apply_topics_to_slam)

        # Setup Navigation Rail
        self._init_navigation()

    def _init_navigation(self):
        self.addSubInterface(self.workflow_view, FluentIcon.APPLICATION, tr("Unified Studio"))
        self.addSubInterface(self.slam_view, FluentIcon.SPEED_HIGH, tr("FAST-LIVO2 SLAM"))
        self.addSubInterface(self.colorize_view, FluentIcon.PALETTE, tr("Colorize Cloud"))
        self.addSubInterface(self.inspect_view, FluentIcon.FOLDER, tr("Bag Inspector"))
        self.addSubInterface(self.calib_view, FluentIcon.TILES, tr("Rig Calibration"))
        self.addSubInterface(self.doctor_view, FluentIcon.HEART, tr("System Doctor"))

        # Bottom navigation item
        self.addSubInterface(
            self.settings_view,
            FluentIcon.SETTING,
            tr("Settings"),
            NavigationItemPosition.BOTTOM
        )

        # Default start page: Unified Studio
        self.switchTo(self.workflow_view)

    def _on_apply_topics_to_slam(self, bag_path: str, lidar: str, imu: str, cam: str):
        self.slam_view.set_bag_and_topics(bag_path, lidar, imu, cam)
        self.workflow_view.bag_input.setText(bag_path)
        if lidar:
            self.workflow_view.lidar_topic.setText(lidar)
        if imu:
            self.workflow_view.imu_topic.setText(imu)
        self.switchTo(self.workflow_view)

    def closeEvent(self, event):
        if self.runner.is_busy:
            reply = QMessageBox.question(
                self,
                tr("Job in Progress"),
                tr("A FAST-LIVO2 or Colorization job is currently running.\nDo you want to cancel it and wait for data to flush before closing?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.runner.cancel()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def create_main_window():
    """Factory function for tests and embedding."""
    return RavenMainWindow()


def launch(*, on_ready=None):
    """Launch the Fluent UI desktop application."""
    if os.name == 'nt':
        import ctypes
        processes = (ctypes.c_ulong * 2)()
        if ctypes.windll.kernel32.GetConsoleProcessList(processes, 2) == 1:
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

    # Enable High-DPI scaling
    set_language(load_config().get("language", "en"))
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    window = RavenMainWindow()
    window.show()

    if on_ready is not None:
        QTimer.singleShot(50, lambda: on_ready(window))

    return app.exec()
