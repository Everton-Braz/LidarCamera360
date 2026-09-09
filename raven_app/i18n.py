"""Small, explicit runtime translation layer for the desktop interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sys
from collections.abc import Callable

SUPPORTED_LANGUAGES = {
    "en": "English",
    "pt-BR": "Português (Brasil)",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "zh-CN": "简体中文",
    "ja": "日本語",
}
_language = "en"
_catalog: dict[str, str] = {}
_listeners: list[Callable[[str], None]] = []


def _locale_dir() -> Path:
    # 1. PyInstaller unpacked temporary directory (_MEIPASS)
    if hasattr(sys, "_MEIPASS"):
        p = Path(sys._MEIPASS) / "locales"
        if p.is_dir():
            return p
    # 2. In onedir layout, check next to executable
    exe_dir = Path(sys.executable).resolve().parent
    if (exe_dir / "locales").is_dir():
        return exe_dir / "locales"
    if (exe_dir / "_internal/locales").is_dir():
        return exe_dir / "_internal/locales"
    # 3. Source repository root
    repo_loc = Path(__file__).resolve().parents[1] / "locales"
    if repo_loc.is_dir():
        return repo_loc
    return repo_loc


def available_languages() -> list[tuple[str, str]]:
    return list(SUPPORTED_LANGUAGES.items())


def get_language() -> str:
    return _language


def register_language_listener(listener: Callable[[str], None]) -> None:
    """Register a callback that fires whenever the active language changes."""
    if listener not in _listeners:
        _listeners.append(listener)


def unregister_language_listener(listener: Callable[[str], None]) -> None:
    """Unregister a language change callback."""
    if listener in _listeners:
        _listeners.remove(listener)


def set_language(language: str | None) -> str:
    """Load a locale catalog and return the normalized language code."""
    global _language, _catalog
    requested = language if language in SUPPORTED_LANGUAGES else "en"
    catalog_path = _locale_dir() / f"{requested}.json"
    try:
        if catalog_path.is_file():
            data = json.loads(catalog_path.read_text(encoding="utf-8"))
            _catalog = data if isinstance(data, dict) else {}
        else:
            _catalog = {}
    except (OSError, ValueError):
        _catalog = {}
        requested = "en"
    _language = requested

    for listener in list(_listeners):
        try:
            listener(_language)
        except Exception as e:
            print(f"[!] Language listener error: {e}")

    return requested


def tr(source: str, **values: Any) -> str:
    """Translate an English source key, formatting values when supplied."""
    translated = _catalog.get(source, source)
    try:
        return translated.format(**values) if values else translated
    except (KeyError, IndexError, ValueError):
        return translated


set_language("en")

