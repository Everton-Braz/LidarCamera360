"""System Diagnostics & Doctor View - Microsoft UI XAML / Fluent Design System."""
import os
import platform
import subprocess
import sys
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, PrimaryPushButton,
    FluentIcon, InfoBar, InfoBarPosition
)
from raven_app import __version__
from raven_app.cli import engine, resources


class DoctorView(QWidget):
    """WinUI Fluent view for inspecting system dependencies and native binaries."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DoctorView")
        self._init_ui()
        self.refresh_diagnostics()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = TitleLabel("System Diagnostics & Engine Doctor")
        subtitle = CaptionLabel(
            "Verify native C++ estimator, mathematical runtimes, and photogrammetry engines"
        )
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # Overview Status Card
        overview_card = CardWidget(self)
        overview_layout = QHBoxLayout(overview_card)
        overview_layout.setContentsMargins(20, 16, 20, 16)

        self.overall_status = TitleLabel("Checking System...")
        self.overall_desc = CaptionLabel("Analyzing dependencies...")
        status_box = QVBoxLayout()
        status_box.addWidget(self.overall_status)
        status_box.addWidget(self.overall_desc)
        overview_layout.addLayout(status_box)
        overview_layout.addStretch()

        self.btn_refresh = PrimaryPushButton("Re-run Diagnostics", icon=FluentIcon.SYNC)
        self.btn_refresh.clicked.connect(self.refresh_diagnostics)
        overview_layout.addWidget(self.btn_refresh)
        layout.addWidget(overview_card)

        # Grid of detailed cards
        grid = QGridLayout()
        grid.setSpacing(16)

        # Native Engine Card
        self.card_native = CardWidget(self)
        native_layout = QVBoxLayout(self.card_native)
        native_layout.setContentsMargins(16, 14, 16, 14)
        native_layout.addWidget(SubtitleLabel("Native FAST-LIVO2 Engine"))
        self.lbl_native_status = StrongBodyLabel("--")
        self.lbl_native_path = CaptionLabel("--")
        self.lbl_native_ver = CaptionLabel("--")
        native_layout.addWidget(self.lbl_native_status)
        native_layout.addWidget(self.lbl_native_path)
        native_layout.addWidget(self.lbl_native_ver)
        grid.addWidget(self.card_native, 0, 0)

        # Spirula Card
        self.card_spirula = CardWidget(self)
        spirula_layout = QVBoxLayout(self.card_spirula)
        spirula_layout.setContentsMargins(16, 14, 16, 14)
        spirula_layout.addWidget(SubtitleLabel("Spirula SfM Engine"))
        self.lbl_spirula_status = StrongBodyLabel("--")
        self.lbl_spirula_path = CaptionLabel("--")
        spirula_layout.addWidget(self.lbl_spirula_status)
        spirula_layout.addWidget(self.lbl_spirula_path)
        grid.addWidget(self.card_spirula, 0, 1)

        # Python & Math Runtime Card
        self.card_math = CardWidget(self)
        math_layout = QVBoxLayout(self.card_math)
        math_layout.setContentsMargins(16, 14, 16, 14)
        math_layout.addWidget(SubtitleLabel("Mathematical Runtime"))
        self.lbl_py = StrongBodyLabel("--")
        self.lbl_np = CaptionLabel("--")
        self.lbl_scipy = CaptionLabel("--")
        self.lbl_cv = CaptionLabel("--")
        math_layout.addWidget(self.lbl_py)
        math_layout.addWidget(self.lbl_np)
        math_layout.addWidget(self.lbl_scipy)
        math_layout.addWidget(self.lbl_cv)
        grid.addWidget(self.card_math, 1, 0)

        # Hardware & OS Card
        self.card_hw = CardWidget(self)
        hw_layout = QVBoxLayout(self.card_hw)
        hw_layout.setContentsMargins(16, 14, 16, 14)
        hw_layout.addWidget(SubtitleLabel("Host Environment"))
        self.lbl_os = StrongBodyLabel(f"OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
        self.lbl_cpu = CaptionLabel(f"Processor: {platform.processor() or 'x64'}")
        self.lbl_cores = CaptionLabel(f"Available CPU Cores: {os.cpu_count() or 1}")
        self.lbl_app_ver = CaptionLabel(f"RavenCalibrator Version: {__version__}")
        hw_layout.addWidget(self.lbl_os)
        hw_layout.addWidget(self.lbl_cpu)
        hw_layout.addWidget(self.lbl_cores)
        hw_layout.addWidget(self.lbl_app_ver)
        grid.addWidget(self.card_hw, 1, 1)

        layout.addLayout(grid)
        layout.addStretch()

    def refresh_diagnostics(self):
        # Native engine check
        eng = engine()
        has_native = eng.is_file()
        if has_native:
            self.lbl_native_status.setText("Status: [OK] Compiled & Available")
            self.lbl_native_path.setText(f"Path: {eng}")
            try:
                probe = subprocess.run([str(eng), '--version'], capture_output=True, text=True, timeout=5)
                self.lbl_native_ver.setText(f"Version: {probe.stdout.strip() or '0.1.0'}")
            except Exception as e:
                self.lbl_native_ver.setText(f"Version check: {e}")
        else:
            self.lbl_native_status.setText("Status: [MISSING] Binary Not Built")
            self.lbl_native_path.setText("Build using tools/build_windows.ps1")
            self.lbl_native_ver.setText("Requires: Visual Studio 2022/2026 + CMake + vcpkg")

        # Spirula check
        spirula_path = resources() / 'spirula/spirula.exe'
        has_spirula = spirula_path.is_file()
        if has_spirula:
            self.lbl_spirula_status.setText("Status: [OK] Photogrammetry Ready")
            self.lbl_spirula_path.setText(f"Path: {spirula_path}")
        else:
            self.lbl_spirula_status.setText("Status: [OPTIONAL] Spirula Not Found")
            self.lbl_spirula_path.setText("SfM Consensus requires spirula/spirula.exe")

        # Python math libraries
        import numpy as np
        import scipy
        import cv2
        self.lbl_py.setText(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
        self.lbl_np.setText(f"NumPy: {np.__version__}")
        self.lbl_scipy.setText(f"SciPy: {scipy.__version__}")
        self.lbl_cv.setText(f"OpenCV: {cv2.__version__}")

        if has_native:
            self.overall_status.setText("System Operational")
            self.overall_desc.setText("All core runtimes and native estimation engines are operational.")
        else:
            self.overall_status.setText("Native Engine Required")
            self.overall_desc.setText("FAST-LIVO2 native executable is missing. Run tools/build_windows.ps1 to build.")
