"""Settings & Personalization View - Microsoft UI XAML / Fluent Design System."""
from pathlib import Path
import shutil
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFileDialog
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, ComboBox, LineEdit,
    PushButton, PrimaryPushButton, InfoBar, InfoBarPosition,
    FluentIcon, setTheme, Theme, isDarkTheme
)
from raven_app import __version__
from raven_app.config import (
    load_config, save_config, get_spirula_bin,
    validate_tool, auto_detect_tools
)


class SettingsView(QWidget):
    """WinUI Fluent view for managing themes and application preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsView")
        self.cfg = load_config()
        self._init_ui()
        self._refresh_tool_status()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = TitleLabel("Settings & Personalization")
        subtitle = CaptionLabel(
            "Customize interface appearance, external processing binaries, and view application information"
        )
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # 1. External Tools & Dependencies Card
        tools_card = CardWidget(self)
        tools_layout = QVBoxLayout(tools_card)
        tools_layout.setContentsMargins(20, 16, 20, 16)
        tools_layout.setSpacing(12)

        tools_layout.addWidget(SubtitleLabel("External Dependencies & Tools"))
        tools_layout.addWidget(CaptionLabel(
            "Spirula is bundled by default. An executable override is available for development."
        ))

        tools_layout.addWidget(CaptionLabel("Video decoding: bundled PyAV (no external executable required)."))

        # Spirula Row
        sp_header = QHBoxLayout()
        sp_header.addWidget(StrongBodyLabel("Spirula Studio Executable (Vulkan SfM):"))
        self.sp_status_label = CaptionLabel("Checking...")
        sp_header.addWidget(self.sp_status_label)
        sp_header.addStretch()
        tools_layout.addLayout(sp_header)

        sp_row = QHBoxLayout()
        self.spirula_input = LineEdit()
        self.spirula_input.setPlaceholderText("Auto-detected in workspace (e.g., spirula\\spirula.exe)")
        self.spirula_input.setText(self.cfg.get("spirula_path", ""))
        self.spirula_input.textChanged.connect(self._on_paths_edited)
        self.spirula_browse_btn = PushButton("Browse...")
        self.spirula_browse_btn.setIcon(FluentIcon.FOLDER)
        self.spirula_browse_btn.clicked.connect(self._browse_spirula)
        sp_row.addWidget(self.spirula_input, 1)
        sp_row.addWidget(self.spirula_browse_btn)
        tools_layout.addLayout(sp_row)

        # Tools Action Row
        actions_row = QHBoxLayout()
        self.detect_btn = PushButton("Auto-Detect Tools")
        self.detect_btn.setIcon(FluentIcon.SYNC)
        self.detect_btn.clicked.connect(self._auto_detect)

        self.save_btn = PrimaryPushButton("Save Settings")
        self.save_btn.setIcon(FluentIcon.SAVE)
        self.save_btn.clicked.connect(self._save_settings)

        actions_row.addWidget(self.detect_btn)
        actions_row.addStretch()
        actions_row.addWidget(self.save_btn)
        tools_layout.addLayout(actions_row)

        layout.addWidget(tools_card)

        # 2. Appearance Card
        theme_card = CardWidget(self)
        theme_layout = QVBoxLayout(theme_card)
        theme_layout.setContentsMargins(20, 16, 20, 16)
        theme_layout.setSpacing(12)

        theme_layout.addWidget(SubtitleLabel("Appearance"))

        mode_row = QHBoxLayout()
        mode_label = BodyLabel("Application Theme:")
        self.theme_combo = ComboBox()
        self.theme_combo.addItems(["Dark Theme", "Light Theme", "Follow Windows System"])
        current_theme = self.cfg.get("theme", "dark")
        if current_theme == "light":
            self.theme_combo.setCurrentIndex(1)
        elif current_theme == "auto":
            self.theme_combo.setCurrentIndex(2)
        else:
            self.theme_combo.setCurrentIndex(0)
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.theme_combo)
        mode_row.addStretch()
        theme_layout.addLayout(mode_row)

        desc = CaptionLabel(
            "Supports Microsoft Fluent Design System with Mica and Acrylic backdrop materials."
        )
        theme_layout.addWidget(desc)
        layout.addWidget(theme_card)

        # 3. About Card
        about_card = CardWidget(self)
        about_layout = QVBoxLayout(about_card)
        about_layout.setContentsMargins(20, 16, 20, 16)
        about_layout.setSpacing(8)

        about_layout.addWidget(SubtitleLabel("About RavenCalibrator"))
        about_layout.addWidget(StrongBodyLabel(f"RavenCalibrator v{__version__}"))
        about_layout.addWidget(CaptionLabel(
            "Advanced 3D LiDAR-Inertial-Visual Mapping and Colorization Suite\n"
            "Hardware: 3DMakerPro Raven LiDAR + Insta360 X4 Dual Fisheye\n"
            "UI Framework: Microsoft UI XAML Fluent Design System (PyQt6-Fluent-Widgets)\n"
            "Estimator: Native MSVC C++17 FAST-LIVO2 (GPL-2.0)"
        ))
        layout.addWidget(about_card)

        layout.addStretch()

    def _browse_spirula(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select spirula.exe", "", "Executables (spirula*.exe *.exe);;All Files (*)"
        )
        if path:
            self.spirula_input.setText(path)
            self._refresh_tool_status()

    def _on_paths_edited(self):
        self._refresh_tool_status()

    def _refresh_tool_status(self):
        # Validate Spirula
        sp_target = self.spirula_input.text().strip() or str(get_spirula_bin())
        ok_sp, ver_sp = validate_tool("spirula", sp_target)
        if ok_sp:
            self.sp_status_label.setText(f"✓ Ready: {ver_sp[:40]}")
            self.sp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.sp_status_label.setText(f"⚠ Not Available ({ver_sp[:35]})")
            self.sp_status_label.setStyleSheet("color: #FFA000;")

    def _auto_detect(self):
        detected = auto_detect_tools()
        if "spirula_path" in detected:
            self.spirula_input.setText(detected["spirula_path"])
        self._refresh_tool_status()
        InfoBar.info(
            title="Auto-Detect",
            content=f"Detected: Spirula={'Found' if 'spirula_path' in detected else 'Missing'}",
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3000
        )

    def _save_settings(self):
        theme_names = ["dark", "light", "auto"]
        self.cfg["spirula_path"] = self.spirula_input.text().strip()
        self.cfg["theme"] = theme_names[self.theme_combo.currentIndex()]

        if save_config(self.cfg):
            InfoBar.success(
                title="Settings Saved",
                content="Preferences and binary tool locations saved successfully.",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000
            )
        else:
            InfoBar.error(
                title="Save Error",
                content="Could not write settings file to disk.",
                parent=self,
                position=InfoBarPosition.TOP_RIGHT,
                duration=4000
            )

    def _on_theme_changed(self, idx: int):
        theme_names = ["dark", "light", "auto"]
        self.cfg["theme"] = theme_names[idx]
        if idx == 0:
            setTheme(Theme.DARK)
        elif idx == 1:
            setTheme(Theme.LIGHT)
        else:
            setTheme(Theme.AUTO)
        save_config(self.cfg)

