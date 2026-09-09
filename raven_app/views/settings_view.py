"""Settings and personalization view."""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, ComboBox,
    InfoBar, InfoBarPosition, FluentIcon,
    setTheme, Theme, isDarkTheme
)
from raven_app import __version__
from raven_app.branding import APP_NAME, APP_DESCRIPTION
from raven_app.config import load_config, save_config
from raven_app.i18n import available_languages, get_language, set_language, tr


class SettingsView(QWidget):
    """WinUI Fluent view for managing themes, languages, and application information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsView")
        self.cfg = load_config()
        self._language_codes = [code for code, _ in available_languages()]
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        self.title_label = TitleLabel(tr("Settings & Personalization"))
        self.subtitle_label = CaptionLabel(
            tr("Customize interface appearance, language, and view application information")
        )
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)
        layout.addLayout(header_layout)

        # 1. Language & Region Card
        self.language_card = CardWidget(self)
        lang_card_layout = QVBoxLayout(self.language_card)
        lang_card_layout.setContentsMargins(20, 16, 20, 16)
        lang_card_layout.setSpacing(10)

        self.language_title = SubtitleLabel(tr("Language & Region"))
        lang_card_layout.addWidget(self.language_title)

        lang_row = QHBoxLayout()
        self.language_label = BodyLabel(tr("Language:"))
        self.language_combo = ComboBox()
        for code, name in available_languages():
            self.language_combo.addItem(name, userData=code)

        cur_lang = get_language()
        if cur_lang in self._language_codes:
            self.language_combo.setCurrentIndex(self._language_codes.index(cur_lang))

        self.language_combo.currentIndexChanged.connect(self._on_language_changed)

        lang_row.addWidget(self.language_label)
        lang_row.addWidget(self.language_combo)
        lang_row.addStretch()
        lang_card_layout.addLayout(lang_row)

        self.language_desc = CaptionLabel(
            tr("Changes are applied immediately across all application interfaces.")
        )
        lang_card_layout.addWidget(self.language_desc)
        layout.addWidget(self.language_card)

        # 2. Appearance Card
        self.theme_card = CardWidget(self)
        theme_layout = QVBoxLayout(self.theme_card)
        theme_layout.setContentsMargins(20, 16, 20, 16)
        theme_layout.setSpacing(10)

        self.appearance_title = SubtitleLabel(tr("Appearance"))
        theme_layout.addWidget(self.appearance_title)

        mode_row = QHBoxLayout()
        self.theme_label = BodyLabel(tr("Application Theme:"))
        self.theme_combo = ComboBox()
        self._theme_options = ["dark", "light", "auto"]
        self.theme_combo.addItems([tr("Dark Theme"), tr("Light Theme"), tr("Follow Windows System")])

        current_theme = self.cfg.get("theme", "dark")
        if current_theme == "light":
            self.theme_combo.setCurrentIndex(1)
        elif current_theme == "auto":
            self.theme_combo.setCurrentIndex(2)
        else:
            self.theme_combo.setCurrentIndex(0)

        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        mode_row.addWidget(self.theme_label)
        mode_row.addWidget(self.theme_combo)
        mode_row.addStretch()
        theme_layout.addLayout(mode_row)

        self.theme_desc = CaptionLabel(
            tr("Supports Microsoft Fluent Design System with Mica and Acrylic backdrop materials.")
        )
        theme_layout.addWidget(self.theme_desc)
        layout.addWidget(self.theme_card)

        # 3. Embedded Engines & Capabilities Card
        self.engines_card = CardWidget(self)
        engines_layout = QVBoxLayout(self.engines_card)
        engines_layout.setContentsMargins(20, 16, 20, 16)
        engines_layout.setSpacing(10)

        self.engines_title = SubtitleLabel(tr("Embedded Processing Engines"))
        engines_layout.addWidget(self.engines_title)

        self.engines_desc = CaptionLabel(
            tr("All SLAM, SfM, and GPU colorization engines are fully built into the application bundle.")
        )
        engines_layout.addWidget(self.engines_desc)

        engines_grid = QGridLayout()
        engines_grid.setHorizontalSpacing(24)
        engines_grid.setVerticalSpacing(8)

        self.eng_slam_title = StrongBodyLabel("FAST-LIVO2 SLAM:")
        self.eng_slam_status = CaptionLabel("✓ " + tr("Bundled & Ready (Native MSVC C++17)"))
        self.eng_slam_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        self.eng_sfm_title = StrongBodyLabel("Spirula Studio SfM:")
        self.eng_sfm_status = CaptionLabel("✓ " + tr("Bundled & Ready (Vulkan GPU Accelerated)"))
        self.eng_sfm_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        self.eng_vulkan_title = StrongBodyLabel("Vulkan Cloud Colorizer:")
        self.eng_vulkan_status = CaptionLabel("✓ " + tr("Bundled & Ready (Compute Shader SPIR-V)"))
        self.eng_vulkan_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        self.eng_video_title = StrongBodyLabel("INSV Video & Telemetry:")
        self.eng_video_status = CaptionLabel("✓ " + tr("Bundled & Ready (PyAV SIMD Hardware Accelerated)"))
        self.eng_video_status.setStyleSheet("color: #4CAF50; font-weight: bold;")

        engines_grid.addWidget(self.eng_slam_title, 0, 0)
        engines_grid.addWidget(self.eng_slam_status, 0, 1)
        engines_grid.addWidget(self.eng_sfm_title, 1, 0)
        engines_grid.addWidget(self.eng_sfm_status, 1, 1)
        engines_grid.addWidget(self.eng_vulkan_title, 2, 0)
        engines_grid.addWidget(self.eng_vulkan_status, 2, 1)
        engines_grid.addWidget(self.eng_video_title, 3, 0)
        engines_grid.addWidget(self.eng_video_status, 3, 1)

        engines_layout.addLayout(engines_grid)
        layout.addWidget(self.engines_card)

        # 4. About Card
        self.about_card = CardWidget(self)
        about_layout = QVBoxLayout(self.about_card)
        about_layout.setContentsMargins(20, 16, 20, 16)
        about_layout.setSpacing(8)

        self.about_title = SubtitleLabel(tr("About {app_name}", app_name=APP_NAME))
        self.about_version = StrongBodyLabel(f"{APP_NAME} v{__version__}")
        self.about_desc = CaptionLabel(
            f"{tr(APP_DESCRIPTION)}\n"
            f"{tr('Hardware:')} 3DMakerPro Raven LiDAR + Insta360 X4 ({tr('Modular & Generic Rig Compatible')})\n"
            f"{tr('UI Framework:')} PyQt6 with Fluent Widgets\n"
            f"{tr('License:')} Dual MIT / GPLv3 Open-Source Architecture"
        )
        about_layout.addWidget(self.about_title)
        about_layout.addWidget(self.about_version)
        about_layout.addWidget(self.about_desc)
        layout.addWidget(self.about_card)

        layout.addStretch()

    def _on_language_changed(self, idx: int):
        if idx < 0 or idx >= len(self._language_codes):
            return
        new_lang = self._language_codes[idx]
        if new_lang == get_language():
            return

        # Update language in i18n layer and configuration
        set_language(new_lang)
        self.cfg["language"] = new_lang
        save_config(self.cfg)

        # Show notification in new language
        lang_name = dict(available_languages()).get(new_lang, new_lang)
        InfoBar.success(
            title=tr("Language Updated"),
            content=tr("Interface language switched to {lang}.", lang=lang_name),
            parent=self,
            position=InfoBarPosition.TOP_RIGHT,
            duration=3000
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

    def retranslate_ui(self):
        """Update all text elements when the language changes at runtime."""
        self.title_label.setText(tr("Settings & Personalization"))
        self.subtitle_label.setText(
            tr("Customize interface appearance, language, and view application information")
        )

        self.language_title.setText(tr("Language & Region"))
        self.language_label.setText(tr("Language:"))
        self.language_desc.setText(
            tr("Changes are applied immediately across all application interfaces.")
        )

        self.appearance_title.setText(tr("Appearance"))
        self.theme_label.setText(tr("Application Theme:"))
        self.theme_desc.setText(
            tr("Supports Microsoft Fluent Design System with Mica and Acrylic backdrop materials.")
        )

        # Refresh theme dropdown items while preserving index
        cur_theme_idx = self.theme_combo.currentIndex()
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        self.theme_combo.addItems([tr("Dark Theme"), tr("Light Theme"), tr("Follow Windows System")])
        self.theme_combo.setCurrentIndex(cur_theme_idx)
        self.theme_combo.blockSignals(False)

        # Engines status card
        self.engines_title.setText(tr("Embedded Processing Engines"))
        self.engines_desc.setText(
            tr("All SLAM, SfM, and GPU colorization engines are fully built into the application bundle.")
        )
        self.eng_slam_status.setText("✓ " + tr("Bundled & Ready (Native MSVC C++17)"))
        self.eng_sfm_status.setText("✓ " + tr("Bundled & Ready (Vulkan GPU Accelerated)"))
        self.eng_vulkan_status.setText("✓ " + tr("Bundled & Ready (Compute Shader SPIR-V)"))
        self.eng_video_status.setText("✓ " + tr("Bundled & Ready (PyAV SIMD Hardware Accelerated)"))

        # About card
        self.about_title.setText(tr("About {app_name}", app_name=APP_NAME))
        self.about_desc.setText(
            f"{tr(APP_DESCRIPTION)}\n"
            f"{tr('Hardware:')} 3DMakerPro Raven LiDAR + Insta360 X4 ({tr('Modular & Generic Rig Compatible')})\n"
            f"{tr('UI Framework:')} PyQt6 with Fluent Widgets\n"
            f"{tr('License:')} Dual MIT / GPLv3 Open-Source Architecture"
        )
