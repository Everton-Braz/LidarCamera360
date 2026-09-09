"""Application settings and bundled Spirula resolution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def get_config_dir() -> Path:
    """Returns the persistent configuration directory across platforms."""
    app_data = os.getenv("APPDATA")
    if app_data:
        cfg_dir = Path(app_data) / "RavenCalibrator"
    else:
        cfg_dir = Path.home() / ".ravencalibrator"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir


def get_config_path() -> Path:
    """Returns the path to settings.json."""
    return get_config_dir() / "settings.json"


def load_config() -> dict:
    """Loads persistent settings or returns default configuration."""
    defaults = {"spirula_path": "", "theme": "dark"}
    try:
        cfg_path = get_config_path()
        if cfg_path.is_file():
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                defaults.update(data)
    except (OSError, ValueError):
        pass
    return defaults


def save_config(cfg: dict) -> bool:
    """Saves settings dictionary atomically to settings.json."""
    cfg_path = get_config_path()
    try:
        tmp_path = cfg_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        tmp_path.replace(cfg_path)
        return True
    except Exception as e:
        print(f"[!] Failed to save settings: {e}")
        return False


def get_app_root() -> Path:
    """Returns the workspace root or PyInstaller unpacked directory."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))


def get_spirula_bin() -> Path:
    """Resolves path to spirula executable.
    
    Priority:
    1. User configured path in settings.json
    2. Local workspace/bundled spirula (spirula/spirula.exe)
    3. System PATH (which spirula)
    4. Fallback path object
    """
    cfg = load_config()
    custom = cfg.get("spirula_path", "").strip()
    if custom and Path(custom).is_file():
        return Path(custom).resolve()

    root = get_app_root()
    exe_dir = Path(sys.executable).resolve().parent
    cwd = Path.cwd()
    candidates = [
        root / "bin" / "spirula.exe",
        root / "spirula" / "spirula.exe",
        exe_dir / "spirula" / "spirula.exe",
        exe_dir / "spirula.exe",
        cwd / "spirula" / "spirula.exe",
        cwd / "spirula.exe",
        root / "bin" / "spirula.exe",
        root / "spirula.exe"
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()

    which_path = shutil.which("spirula")
    if which_path:
        return Path(which_path).resolve()

    return root / "spirula" / "spirula.exe"


def validate_tool(tool_type: str, path_str: str) -> tuple[bool, str]:
    """Validates if the Spirula executable runs properly.
    
    Returns:
        (is_valid: bool, info_banner: str)
    """
    p = path_str.strip()
    if not p:
        # Test default resolved tool
        resolved = str(get_spirula_bin())
    else:
        resolved = p

    if not Path(resolved).is_file() and not shutil.which(resolved):
        return False, f"File not found: {resolved}"

    try:
        cmd = [resolved, "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            lines = res.stdout.splitlines()
            banner = lines[0].strip() if lines else "Active"
            return True, banner
        else:
            return False, f"Process exited with code {res.returncode}"
    except Exception as e:
        return False, str(e)


def auto_detect_tools() -> dict:
    """Scans system for available Spirula installations."""
    detected = {}
    sp = str(get_spirula_bin())
    ok_sp, ver_sp = validate_tool("spirula", sp)
    if ok_sp:
        detected["spirula_path"] = sp

    return detected
