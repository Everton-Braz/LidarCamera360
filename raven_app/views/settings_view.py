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
    PushButton, PrimaryPushButton, SwitchButton, ProgressBar, CheckBox,
    InfoBar, InfoBarPosition, FluentIcon, SmoothScrollArea,
    setTheme, Theme, isDarkTheme
)
from raven_app import __version__
from raven_app.branding import APP_NAME, APP_DESCRIPTION
from raven_app.config import load_config, save_config
from raven_app.i18n import available_languages, get_language, set_language, tr, tr_button
from raven_app.updater import check_for_updates, download_update, stage_windows_update
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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        scroll = SmoothScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(36, 24, 36, 24)
        layout.setSpacing(18)

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
        self.language_card = CardWidget(content)
        lang_card_layout = QVBoxLayout(self.language_card)
        lang_card_layout.setContentsMargins(24, 20, 24, 20)
        lang_card_layout.setSpacing(12)

        self.language_title = SubtitleLabel(tr("Language & Region"))
        lang_card_layout.addWidget(self.language_title)

        lang_row = QHBoxLayout()
        self.language_label = BodyLabel(tr("Language:"))
        self.language_combo = ComboBox()
        self.language_combo.setMinimumWidth(220)
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
        self.theme_card = CardWidget(content)
        theme_layout = QVBoxLayout(self.theme_card)
        theme_layout.setContentsMargins(24, 20, 24, 20)
        theme_layout.setSpacing(12)

        self.appearance_title = SubtitleLabel(tr("Appearance"))
        theme_layout.addWidget(self.appearance_title)

        mode_row = QHBoxLayout()
        self.theme_label = BodyLabel(tr("Application Theme:"))
        self.theme_combo = ComboBox()
        self.theme_combo.setMinimumWidth(220)
        self._theme_options = ["dark", "light", "auto"]
        self.theme_combo.addItems([tr("Dark Theme"), tr("Light Theme"), tr("Follow Windows System")])

        current_theme = self.cfg.get("theme", "dark").lower()
        self.theme_combo.blockSignals(True)
        if current_theme == "light":
            self.theme_combo.setCurrentIndex(1)
        elif current_theme == "auto":
            self.theme_combo.setCurrentIndex(2)
        else:
            self.theme_combo.setCurrentIndex(0)
        self.theme_combo.blockSignals(False)

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

        # Deliverables card: extra formats are exported from the project; this only brings them to the main screen
        self.export_card = CardWidget(content)
        export_layout = QVBoxLayout(self.export_card)
        export_layout.setContentsMargins(24, 20, 24, 20)
        export_layout.setSpacing(12)
        self.export_title = SubtitleLabel(tr("Deliverables"))
        export_layout.addWidget(self.export_title)
        self.extra_formats_chk = CheckBox(tr("Offer extra formats on the workflow screen (PCD, georeferenced, GPS)"))
        self.extra_formats_chk.setChecked(bool(self.cfg.get("show_extra_formats", False)))
        self.extra_formats_chk.stateChanged.connect(self._on_extra_formats_changed)
        export_layout.addWidget(self.extra_formats_chk)
        self.export_desc = CaptionLabel(tr(
            "By default a run delivers the coloured PLY and the 3DGS dataset. Everything else is exported "
            "later from the project (Workflow > Export from project)."))
        self.export_desc.setWordWrap(True)
        export_layout.addWidget(self.export_desc)
        self.photogrammetry_chk = CheckBox(tr("Show \"Open in RealityScan / Metashape\" on the workflow screen"))
        self.photogrammetry_chk.setChecked(bool(self.cfg.get("show_photogrammetry_export", False)))
        self.photogrammetry_chk.stateChanged.connect(self._on_photogrammetry_export_changed)
        export_layout.addWidget(self.photogrammetry_chk)
        openmvs_row = QHBoxLayout()
        self.openmvs_label = BodyLabel("")
        self.openmvs_label.setWordWrap(True)
        self.openmvs_btn = PushButton(tr("Choose OpenMVS..."))
        self.openmvs_btn.clicked.connect(self._on_choose_openmvs)
        openmvs_row.addWidget(self.openmvs_label, 1)
        openmvs_row.addWidget(self.openmvs_btn)
        export_layout.addLayout(openmvs_row)
        self._refresh_openmvs_label()
        layout.addWidget(self.export_card)

        # 3. Embedded Engines & Capabilities Card
        self.engines_card = CardWidget(content)
        engines_layout = QVBoxLayout(self.engines_card)
        engines_layout.setContentsMargins(24, 20, 24, 20)
        engines_layout.setSpacing(12)

        self.engines_title = SubtitleLabel(tr("Embedded Processing Engines"))
        engines_layout.addWidget(self.engines_title)

        self.engines_desc = CaptionLabel(
            tr("Processing engines found by this installation. Open Doctor for versions and details.")
        )
        engines_layout.addWidget(self.engines_desc)

        engines_grid = QGridLayout()
        engines_grid.setHorizontalSpacing(24)
        engines_grid.setVerticalSpacing(10)

        self.eng_slam_title = StrongBodyLabel("FAST-LIVO2 SLAM:")
        self.eng_slam_status = CaptionLabel("")

        self.eng_sfm_title = StrongBodyLabel("Spirula Studio SfM:")
        self.eng_sfm_status = CaptionLabel("")

        self.eng_vulkan_title = StrongBodyLabel("Vulkan Cloud Colorizer:")
        self.eng_vulkan_status = CaptionLabel("")

        self.eng_video_title = StrongBodyLabel("INSV Video & Telemetry:")
        self.eng_video_status = CaptionLabel("")

        engines_grid.addWidget(self.eng_slam_title, 0, 0)
        engines_grid.addWidget(self.eng_slam_status, 0, 1)
        engines_grid.addWidget(self.eng_sfm_title, 1, 0)
        engines_grid.addWidget(self.eng_sfm_status, 1, 1)
        engines_grid.addWidget(self.eng_vulkan_title, 2, 0)
        engines_grid.addWidget(self.eng_vulkan_status, 2, 1)
        engines_grid.addWidget(self.eng_video_title, 3, 0)
        engines_grid.addWidget(self.eng_video_status, 3, 1)

        engines_layout.addLayout(engines_grid)
        self._refresh_engine_status()
        layout.addWidget(self.engines_card)

        # 4. Updates & Maintenance Card
        self.updates_card = CardWidget(content)
        updates_layout = QVBoxLayout(self.updates_card)
        updates_layout.setContentsMargins(24, 20, 24, 20)
        updates_layout.setSpacing(12)

        self.updates_title = SubtitleLabel(tr("Software Updates"))
        updates_layout.addWidget(self.updates_title)

        up_row = QHBoxLayout()
        up_row.setSpacing(12)
        self.updates_ver_label = BodyLabel(f"{tr('Current Version:')} v{__version__}")
        self.check_update_btn = PushButton(tr("Check for Updates"))
        self.check_update_btn.setIcon(FluentIcon.SYNC)
        self.check_update_btn.setMinimumHeight(32)
        self.check_update_btn.clicked.connect(self._check_for_updates)
        up_row.addWidget(self.updates_ver_label)
        up_row.addWidget(self.check_update_btn)
        up_row.addStretch()
        updates_layout.addLayout(up_row)

        self.updates_status = CaptionLabel(tr("Click 'Check for Updates' to verify the latest release on GitHub."))
        updates_layout.addWidget(self.updates_status)

        self.download_row = QHBoxLayout()
        self.download_row.setSpacing(12)
        self.download_btn = PrimaryPushButton(tr_button("Download & Install Update"))
        self.download_btn.setIcon(FluentIcon.DOWNLOAD)
        self.download_btn.setMinimumHeight(32)
        self.download_btn.setVisible(False)
        self.download_btn.clicked.connect(self._download_and_install_update)
        self.download_progress = ProgressBar()
        self.download_progress.setVisible(False)
        self.download_row.addWidget(self.download_btn)
        self.download_row.addWidget(self.download_progress)
        self.download_row.addStretch()
        updates_layout.addLayout(self.download_row)
        layout.addWidget(self.updates_card)

        # 5. Feedback & Bug Report Card
        self.feedback_card = CardWidget(content)
        feedback_layout = QVBoxLayout(self.feedback_card)
        feedback_layout.setContentsMargins(24, 20, 24, 20)
        feedback_layout.setSpacing(12)

        self.feedback_title = SubtitleLabel(tr("Feedback & Bug Reporting"))
        feedback_layout.addWidget(self.feedback_title)

        self.feedback_desc = CaptionLabel(
            tr("Encountered an issue, timestamp offset, or have a suggestion? Send your report directly to the development team on GitHub.")
        )
        feedback_layout.addWidget(self.feedback_desc)

        fb_row = QHBoxLayout()
        self.feedback_btn = PushButton(tr("Open Feedback Dialog"))
        self.feedback_btn.setIcon(FluentIcon.FEEDBACK)
        self.feedback_btn.setMinimumHeight(34)
        self.feedback_btn.setMinimumWidth(190)
        self.feedback_btn.clicked.connect(self._open_feedback_dialog)
        fb_row.addWidget(self.feedback_btn)
        fb_row.addStretch()
        feedback_layout.addLayout(fb_row)
        layout.addWidget(self.feedback_card)

        # 7. About Card
        self.about_card = CardWidget(content)
        about_layout = QVBoxLayout(self.about_card)
        about_layout.setContentsMargins(24, 20, 24, 20)
        about_layout.setSpacing(10)

        self.about_title = SubtitleLabel(tr("About {app_name}", app_name=APP_NAME))
        self.about_version = StrongBodyLabel(f"{APP_NAME} v{__version__}")
        self.about_desc = CaptionLabel(
            f"{tr(APP_DESCRIPTION)}\n"
            f"{tr('Hardware:')} {tr('3DMakerPro Raven and Eagle LiDAR, Insta360 and Raven cameras, and custom rigs')}\n"
            f"{tr('UI Framework:')} PyQt6 with Fluent Widgets\n"
            f"{tr('License:')} GPL-3.0-or-later ({tr('third-party notices in THIRD_PARTY_NOTICES.md')})"
        )
        about_layout.addWidget(self.about_title)
        about_layout.addWidget(self.about_version)
        about_layout.addWidget(self.about_desc)
        layout.addWidget(self.about_card)

        layout.addStretch()
        scroll.setWidget(content)
        scroll.enableTransparentBackground()
        content.setAutoFillBackground(False)
        main_layout.addWidget(scroll)

    def _on_extra_formats_changed(self, state=None):
        self.cfg["show_extra_formats"] = self.extra_formats_chk.isChecked()
        save_config(self.cfg)
        InfoBar.info(title=tr("Saved"), content=tr("Takes effect the next time the workflow screen is opened."),
                     position=InfoBarPosition.TOP, parent=self, duration=3000)

    def _on_photogrammetry_export_changed(self, state=None):
        self.cfg["show_photogrammetry_export"] = self.photogrammetry_chk.isChecked()
        save_config(self.cfg)
        InfoBar.info(title=tr("Saved"), content=tr("Takes effect the next time the workflow screen is opened."),
                     position=InfoBarPosition.TOP, parent=self, duration=3000)

    def _refresh_openmvs_label(self):
        from raven_app.config import get_openmvs_bin
        path = get_openmvs_bin()
        self.openmvs_label.setText(tr("OpenMVS (dense photogrammetry, optional): {path}",
                                      path=str(path) if path.is_file() else tr("Not found")))

    def _on_choose_openmvs(self):
        from PyQt6.QtWidgets import QFileDialog
        from raven_app.openmvs_dense import check_binary
        path, _ = QFileDialog.getOpenFileName(self, tr("DensifyPointCloud of the OpenMVS fisheye build"), "",
                                              "DensifyPointCloud (DensifyPointCloud*)")
        if not path:
            return
        try:
            check_binary(path)
        except Exception as exc:
            InfoBar.error(title=tr("OpenMVS was not accepted"), content=str(exc),
                          position=InfoBarPosition.TOP, parent=self, duration=8000)
            return
        self.cfg["openmvs_path"] = path
        save_config(self.cfg)
        self._refresh_openmvs_label()
        InfoBar.info(title=tr("Saved"), content=tr("Takes effect the next time the workflow screen is opened."),
                     position=InfoBarPosition.TOP, parent=self, duration=3000)

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
        if idx < 0 or idx >= len(theme_names):
            return

        theme_name = theme_names[idx]
        self.cfg["theme"] = theme_name
        save_config(self.cfg)

        def _apply_theme():
            try:
                from raven_app.theme_manager import apply_app_theme
                apply_app_theme(theme_name)
            except Exception as exc:
                print(f"[!] Error switching theme: {exc}")

        # Defer theme application to next event loop iteration so ComboBoxMenu
        # can finish dismissing cleanly before stylesheets across the app are recalculated.
        QTimer.singleShot(0, _apply_theme)


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

    def _refresh_engine_status(self):
        """Show whether each processing engine is actually present in this installation."""
        from raven_app.cli import engine
        from raven_app.config import get_spirula_bin
        from raven_app.vulkan_engine import get_vulkan_bin
        try:
            import av  # noqa: F401
            has_av = True
        except ImportError:
            has_av = False
        checks = ((self.eng_slam_status, engine().is_file(), tr("Native FAST-LIVO2 engine")),
                  (self.eng_sfm_status, get_spirula_bin().is_file(), tr("Spirula photogrammetry, optional")),
                  (self.eng_vulkan_status, get_vulkan_bin().is_file(), tr("Vulkan compute colorizer")),
                  (self.eng_video_status, has_av, tr("PyAV video decoder")))
        for label, found, detail in checks:
            if found:
                label.setText("✓ " + tr("Found") + f" ({detail})")
                label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                label.setText("✗ " + tr("Not found") + f" ({detail}) - " + tr("see Doctor"))
                label.setStyleSheet("color: #E0A030; font-weight: bold;")

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

        self.photogrammetry_chk.setText(tr("Show \"Open in RealityScan / Metashape\" on the workflow screen"))
        self.openmvs_btn.setText(tr("Choose OpenMVS..."))
        self._refresh_openmvs_label()

        # Engines status card
        self.engines_title.setText(tr("Embedded Processing Engines"))
        self.engines_desc.setText(
            tr("Processing engines found by this installation. Open Doctor for versions and details.")
        )
        self._refresh_engine_status()

        # About card
        self.about_title.setText(tr("About {app_name}", app_name=APP_NAME))
        self.about_desc.setText(
            f"{tr(APP_DESCRIPTION)}\n"
            f"{tr('Hardware:')} {tr('3DMakerPro Raven and Eagle LiDAR, Insta360 and Raven cameras, and custom rigs')}\n"
            f"{tr('UI Framework:')} PyQt6 with Fluent Widgets\n"
            f"{tr('License:')} GPL-3.0-or-later ({tr('third-party notices in THIRD_PARTY_NOTICES.md')})"
        )

        # Updates card
        self.updates_title.setText(tr("Software Updates"))
        self.updates_ver_label.setText(f"{tr('Current Version:')} v{__version__}")
        self.check_update_btn.setText(tr("Check for Updates"))
        self.download_btn.setText(tr_button("Download & Install Update"))

        # Feedback card
        self.feedback_title.setText(tr("Feedback & Bug Reporting"))
        self.feedback_btn.setText(tr("Open Feedback Dialog"))
