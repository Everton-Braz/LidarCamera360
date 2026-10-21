"""Centralized theme and global styling management for Fluent UI."""
from __future__ import annotations

from PyQt6.QtWidgets import QApplication
from qfluentwidgets import setTheme, Theme, isDarkTheme, qconfig
from raven_app.config import load_config, save_config


def get_theme_stylesheet() -> str:
    """Global stylesheet ensuring QDialogs, QMainWindows, and nested containers respect dark/light mode."""
    if isDarkTheme():
        return """
            QDialog, QMainWindow, QMainWindow > QWidget {
                background-color: #202020;
                color: #ffffff;
            }
            QScrollArea, QScrollArea > QWidget > QWidget {
                background: transparent;
                background-color: transparent;
            }
        """
    else:
        return """
            QDialog, QMainWindow, QMainWindow > QWidget {
                background-color: #f9f9f9;
                color: #000000;
            }
            QScrollArea, QScrollArea > QWidget > QWidget {
                background: transparent;
                background-color: transparent;
            }
        """


def update_global_theme_stylesheet():
    """Update QApplication stylesheet based on current dark/light status."""
    app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(get_theme_stylesheet())


def apply_app_theme(theme_pref: str | None = None):
    """Apply theme from preference string ('dark', 'light', 'auto') and synchronize global styles."""
    if theme_pref is None:
        theme_pref = load_config().get("theme", "dark").lower()
    else:
        theme_pref = str(theme_pref).lower()

    if theme_pref == "light":
        setTheme(Theme.LIGHT)
    elif theme_pref == "auto":
        setTheme(Theme.AUTO)
    else:
        setTheme(Theme.DARK)

    update_global_theme_stylesheet()


# Hook into QFluentWidgets qconfig.themeChanged
qconfig.themeChanged.connect(lambda *_: update_global_theme_stylesheet())
