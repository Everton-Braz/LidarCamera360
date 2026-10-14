"""Tests for theme switching and settings view robustness."""
import pytest
from PyQt6.QtWidgets import QApplication
from qfluentwidgets import Theme, isDarkTheme, qconfig

from raven_app.config import load_config, save_config
from raven_app.gui import RavenMainWindow
from raven_app.views.settings_view import SettingsView
from raven_app.viewer_window import PointCloudViewerWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_settings_theme_options_and_switching(qapp, tmp_path, monkeypatch):
    # Mock config location to tmp_path
    cfg_file = tmp_path / "settings.json"
    monkeypatch.setattr("raven_app.config.get_config_path", lambda: cfg_file)
    save_config({"theme": "dark", "language": "en"})

    view = SettingsView()
    assert view.theme_combo.count() == 3
    assert view.theme_combo.currentIndex() == 0  # dark

    # Test out of bounds does not raise
    view._on_theme_changed(-1)
    view._on_theme_changed(99)

    # Test switching to light mode
    view.theme_combo.setCurrentIndex(1)
    qapp.processEvents()
    cfg = load_config()
    assert cfg["theme"] == "light"

    # Test switching to dark mode
    view.theme_combo.setCurrentIndex(0)
    qapp.processEvents()
    cfg = load_config()
    assert cfg["theme"] == "dark"


def test_main_window_theme_and_viewer_lifecycle(qapp, tmp_path, monkeypatch):
    cfg_file = tmp_path / "settings.json"
    monkeypatch.setattr("raven_app.config.get_config_path", lambda: cfg_file)
    save_config({"theme": "dark", "language": "en"})

    win = RavenMainWindow()
    win.show()
    qapp.processEvents()

    # Open viewer
    viewer = PointCloudViewerWindow(win)
    viewer.show()
    qapp.processEvents()

    # Switch theme while viewer is open
    win.settings_view.theme_combo.setCurrentIndex(1)
    qapp.processEvents()

    win.settings_view.theme_combo.setCurrentIndex(0)
    qapp.processEvents()

    # Close viewer and ensure switching theme afterwards does not crash
    viewer.close()
    qapp.processEvents()

    win.settings_view.theme_combo.setCurrentIndex(1)
    qapp.processEvents()

    win.settings_view.theme_combo.setCurrentIndex(0)
    qapp.processEvents()

    win.close()
    qapp.processEvents()


def test_scroll_views_transparent_background(qapp, tmp_path, monkeypatch):
    from qfluentwidgets import SmoothScrollArea

    cfg_file = tmp_path / "settings.json"
    monkeypatch.setattr("raven_app.config.get_config_path", lambda: cfg_file)
    save_config({"theme": "dark", "language": "en"})

    win = RavenMainWindow()
    win.show()
    qapp.processEvents()

    for view in (win.workflow_view, win.colorize_view, win.settings_view):
        scroll = view.findChild(SmoothScrollArea)
        assert scroll is not None
        content = scroll.widget()
        assert content is not None
        assert not content.autoFillBackground(), f"{view.objectName()} content should not auto-fill background"

    win.close()
    qapp.processEvents()

