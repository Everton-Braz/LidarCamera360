"""Anonymous, privacy-first telemetry and diagnostic heartbeat."""
import json
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from raven_app import __version__
from raven_app.branding import APP_NAME
from raven_app.config import get_config_dir, load_config, save_config


def get_anonymous_client_id() -> str:
    """Returns or generates a persistent random UUID for this installation."""
    id_file = get_config_dir() / "client_id.txt"
    try:
        if id_file.is_file():
            client_id = id_file.read_text(encoding="utf-8").strip()
            if client_id:
                return client_id
        new_id = str(uuid.uuid4())
        id_file.write_text(new_id, encoding="utf-8")
        return new_id
    except Exception:
        return "anonymous-local"


def is_telemetry_enabled() -> bool:
    """Checks whether anonymous telemetry/heartbeat is permitted by the user."""
    cfg = load_config()
    return bool(cfg.get("telemetry_enabled", True))


def set_telemetry_enabled(enabled: bool) -> bool:
    """Updates telemetry opt-in state in settings."""
    cfg = load_config()
    cfg["telemetry_enabled"] = bool(enabled)
    return save_config(cfg)


def get_system_telemetry_payload() -> Dict[str, Any]:
    """Gathers sanitized system specs for diagnostic reporting."""
    payload: Dict[str, Any] = {
        "app_name": APP_NAME,
        "app_version": __version__,
        "client_id": get_anonymous_client_id(),
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }

    # Attempt lightweight GPU detection without crashing
    try:
        from raven_app.vulkan_engine import vulkan_status
        payload["vulkan_status"] = vulkan_status()
    except Exception:
        payload["vulkan_status"] = {"available": False}

    return payload


def send_heartbeat(endpoint_url: Optional[str] = None, timeout: int = 5) -> Dict[str, Any]:
    """Sends an optional anonymous heartbeat ping if telemetry is enabled and endpoint is provided."""
    if not is_telemetry_enabled():
        return {"status": "opt_out", "message": "Telemetry is disabled by user preference."}

    payload = get_system_telemetry_payload()
    if not endpoint_url:
        cfg = load_config()
        endpoint_url = cfg.get("telemetry_endpoint", "").strip()

    if not endpoint_url:
        return {"status": "noop", "message": "No telemetry endpoint configured; payload collected locally.", "payload": payload}

    import urllib.request
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint_url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": f"{APP_NAME}/{__version__}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"status": "success", "code": resp.status}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
