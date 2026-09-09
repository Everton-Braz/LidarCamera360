"""Small, explicit runtime translation layer for the desktop interface."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _locale_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "locales"


def available_languages() -> list[tuple[str, str]]:
    return list(SUPPORTED_LANGUAGES.items())


def get_language() -> str:
    return _language


def set_language(language: str | None) -> str:
    """Load a locale catalog and return the normalized language code."""
    global _language, _catalog
    requested = language if language in SUPPORTED_LANGUAGES else "en"
    catalog_path = _locale_dir() / f"{requested}.json"
    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        _catalog = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        _catalog = {}
        requested = "en"
    _language = requested
    return requested


def tr(source: str, **values: Any) -> str:
    """Translate an English source key, formatting values when supplied."""
    translated = _catalog.get(source, source)
    try:
        return translated.format(**values) if values else translated
    except (KeyError, IndexError, ValueError):
        return translated


set_language("en")
