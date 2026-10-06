"""Settings and personalization view."""
from pathlib import Path
from typing import Optional
import threading
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
)
from qfluentwidgets import (
    CardWidget, TitleLabel, SubtitleLabel, BodyLabel,
    CaptionLabel, StrongBodyLabel, ComboBox,
    PushButton, PrimaryPushButton, SwitchButton, ProgressBar,
    InfoBar, InfoBarPosition, FluentIcon,
    setTheme, Theme, isDarkTheme
)
from raven_app import __version__
from raven_app.branding import APP_NAME, APP_DESCRIPTION
from raven_app.config import load_config, save_config
from raven_app.i18n import available_languages, get_language, set_language, tr
from raven_app.updater import check_for_updates, download_update, stage_windows_update
from raven_app.telemetry import (
    is_telemetry_enabled, set_telemetry_enabled, get_anonymous_client_id
)
from raven_app.views.feedback_dialog import FeedbackDialog


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

        # 4. Updates & Maintenance Card
        self.updates_card = CardWidget(self)
        updates_layout = QVBoxLayout(self.updates_card)
        updates_layout.setContentsMargins(20, 16, 20, 16)
        updates_layout.setSpacing(10)

        self.updates_title = SubtitleLabel(tr("Software Updates"))
        updates_layout.addWidget(self.updates_title)

        up_row = QHBoxLayout()
        self.updates_ver_label = BodyLabel(f"{tr('Current Version:')} v{__version__}")
        self.check_update_btn = PushButton(tr("Check for Updates"))
        self.check_update_btn.setIcon(FluentIcon.SYNC)
        self.check_update_btn.clicked.connect(self._check_for_updates)
        up_row.addWidget(self.updates_ver_label)
        up_row.addWidget(self.check_update_btn)
        up_row.addStretch()
        updates_layout.addLayout(up_row)

        self.updates_status = CaptionLabel(tr("Click 'Check for Updates' to verify the latest release on GitHub."))
        updates_layout.addWidget(self.updates_status)

        self.download_row = QHBoxLayout()
        self.download_btn = PrimaryPushButton(tr("Download & Install Update"))
        self.download_btn.setIcon(FluentIcon.DOWNLOAD)
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self._download_and_install_update)
        self.download_progress = ProgressBar()
        self.download_progress.setVisible(False)
        self.download_row.addWidget(self.download_btn)
        self.download_row.addWidget(self.download_progress)
        self.download_row.addStretch()
        updates_layout.addLayout(self.download_row)
        layout.addWidget(self.updates_card)

        # 5. Anonymous Telemetry & Diagnostics Card
        self.telemetry_card = CardWidget(self)
        telem_layout = QVBoxLayout(self.telemetry_card)
        telem_layout.setContentsMargins(20, 16, 20, 16)
        telem_layout.setSpacing(10)

        self.telemetry_title = SubtitleLabel(tr("Anonymous Telemetry & Diagnostics"))
        telem_layout.addWidget(self.telemetry_title)

        telem_row = QHBoxLayout()
        self.telemetry_switch = SwitchButton()
        self.telemetry_switch.setChecked(is_telemetry_enabled())
        self.telemetry_switch.checkedChanged.connect(self._on_telemetry_changed)
        self.telemetry_switch_label = BodyLabel(tr("Enable anonymous telemetry and diagnostic reporting"))
        telem_row.addWidget(self.telemetry_switch)
        telem_row.addWidget(self.telemetry_switch_label)
        telem_row.addStretch()
        telem_layout.addLayout(telem_row)

        anon_id = get_anonymous_client_id()
        self.telemetry_desc = CaptionLabel(
            f"{tr('Helps improve LiDAR driver support, sensor synchronization, and SLAM accuracy.')} "
            f"{tr('No video frames, points, paths, or personal info are ever sent.')}\n"
            f"{tr('Anonymous Installation ID:')} {anon_id[:8]}...{anon_id[-4:]}"
        )
        telem_layout.addWidget(self.telemetry_desc)
        layout.addWidget(self.telemetry_card)

        # 6. Feedback & Bug Report Card
        self.feedback_card = CardWidget(self)
        feedback_layout = QVBoxLayout(self.feedback_card)
        feedback_layout.setContentsMargins(20, 16, 20, 16)
        feedback_layout.setSpacing(10)

        self.feedback_title = SubtitleLabel(tr("Feedback & Bug Reporting"))
        feedback_layout.addWidget(self.feedback_title)

        self.feedback_desc = CaptionLabel(
            tr("Encountered an issue, timestamp offset, or have a suggestion? Send your report directly to the development team on GitHub.")
        )
        feedback_layout.addWidget(self.feedback_desc)

        fb_row = QHBoxLayout()
        self.feedback_btn = PushButton(tr("Open Feedback Dialog"))
        self.feedback_btn.setIcon(FluentIcon.FEEDBACK)
        self.feedback_btn.clicked.connect(self._open_feedback_dialog)
        fb_row.addWidget(self.feedback_btn)
        fb_row.addStretch()
        feedback_layout.addLayout(fb_row)
        layout.addWidget(self.feedback_card)

        # 7. About Card
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

    def _on_telemetry_changed(self, enabled: bool):
        set_telemetry_enabled(enabled)
        self.cfg["telemetry_enabled"] = enabled
        save_config(self.cfg)
        msg = tr("Telemetry enabled. Thank you for supporting open-source development!") if enabled else tr("Telemetry disabled.")
        InfoBar.info(title=tr("Telemetry"), content=msg, parent=self, position=InfoBarPosition.TOP_RIGHT, duration=2500)

    def _check_for_updates(self):
        self.check_update_btn.setEnabled(False)
        self.updates_status.setText(tr("Checking for updates on GitHub..."))
        def _worker():
            res = check_for_updates()
            QTimer.singleShot(0, lambda: self._on_check_update_finished(res))
        threading.Thread(target=_worker, daemon=True).start()

    def _on_check_update_finished(self, res: dict):
        self.check_update_btn.setEnabled(True)
        if res.get("status") != "success":
            self.updates_status.setText(f"{tr('Check failed:')} {res.get('message')}")
            InfoBar.warning(title=tr("Update Check"), content=res.get("message", "Error"), parent=self, position=InfoBarPosition.TOP_RIGHT, duration=4000)
            return
        if res.get("has_update"):
            tag = res.get("tag_name")
            self.updates_status.setText(f"{tr('New release available:')} {tag} ({res.get('asset_name', '')})")
            self._latest_update_res = res
            self.download_btn.setVisible(True)
            InfoBar.success(title=tr("Update Available"), content=f"{tr('Version')} {tag} {tr('is ready for download.')}", parent=self, position=InfoBarPosition.TOP_RIGHT, duration=5000)
        else:
            self.updates_status.setText(tr("You are using the latest version (v{ver}).", ver=__version__))
            InfoBar.info(title=tr("Up to Date"), content=tr("LidarCamera360 is up to date."), parent=self, position=InfoBarPosition.TOP_RIGHT, duration=3000)

    def _download_and_install_update(self):
        res = getattr(self, "_latest_update_res", None)
        if not res or not res.get("download_url"):
            return
        self.download_btn.setEnabled(False)
        self.download_progress.setVisible(True)
        self.download_progress.setValue(0)
        self.updates_status.setText(tr("Downloading update package..."))
        def _worker():
            def _cb(cur, total):
                pct = int((cur / max(1, total)) * 100)
                QTimer.singleShot(0, lambda: self.download_progress.setValue(pct))
            zip_path = download_update(res["download_url"], progress_callback=_cb)
            QTimer.singleShot(0, lambda: self._on_download_finished(zip_path))
        threading.Thread(target=_worker, daemon=True).start()

    def _on_download_finished(self, zip_path: Optional[Path]):
        self.download_btn.setEnabled(True)
        self.download_progress.setVisible(False)
        if not zip_path or not zip_path.is_file():
            self.updates_status.setText(tr("Failed to download update."))
            InfoBar.error(title=tr("Update Failed"), content=tr("Download was unsuccessful."), parent=self, position=InfoBarPosition.TOP_RIGHT, duration=4000)
            return
        self.updates_status.setText(tr("Update downloaded. Restarting application to apply..."))
        stage_windows_update(zip_path)

    def _open_feedback_dialog(self):
        dlg = FeedbackDialog(parent=self.window() or self)
        dlg.exec()

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

        # Updates card
        self.updates_title.setText(tr("Software Updates"))
        self.updates_ver_label.setText(f"{tr('Current Version:')} v{__version__}")
        self.check_update_btn.setText(tr("Check for Updates"))
        self.download_btn.setText(tr("Download & Install Update"))

        # Telemetry card
        self.telemetry_title.setText(tr("Anonymous Telemetry & Diagnostics"))
        self.telemetry_switch_label.setText(tr("Enable anonymous telemetry and diagnostic reporting"))

        # Feedback card
        self.feedback_title.setText(tr("Feedback & Bug Reporting"))
        self.feedback_btn.setText(tr("Open Feedback Dialog"))
