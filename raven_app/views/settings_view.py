"""Settings & Personalization View - Microsoft UI XAML / Fluent Design System."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, ComboBox,
    FluentIcon, setTheme, Theme, isDarkTheme
)
from raven_app import __version__


class SettingsView(QWidget):
    """WinUI Fluent view for managing themes and application preferences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsView")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = TitleLabel("Settings & Personalization")
        subtitle = CaptionLabel(
            "Customize interface appearance, theme modes, and view application information"
        )
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addLayout(header_layout)

        # Appearance Card
        theme_card = CardWidget(self)
        theme_layout = QVBoxLayout(theme_card)
        theme_layout.setContentsMargins(20, 16, 20, 16)
        theme_layout.setSpacing(12)

        theme_layout.addWidget(SubtitleLabel("Appearance"))

        mode_row = QHBoxLayout()
        mode_label = BodyLabel("Application Theme:")
        self.theme_combo = ComboBox()
        self.theme_combo.addItems(["Dark Theme", "Light Theme", "Follow Windows System"])
        self.theme_combo.setCurrentIndex(0 if isDarkTheme() else 1)
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

        # About Card
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

    def _on_theme_changed(self, idx: int):
        if idx == 0:
            setTheme(Theme.DARK)
        elif idx == 1:
            setTheme(Theme.LIGHT)
        else:
            setTheme(Theme.AUTO)
