"""Over-The-Air (OTA) remote update checker and installer for LidarCamera360."""
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from raven_app import __version__
from raven_app.branding import APP_NAME, REPOSITORY_NAME
from raven_app.config import get_app_root, load_config

DEFAULT_REPO = f"Everton-Braz/{REPOSITORY_NAME}"


def parse_version_tuple(v_str: str) -> Tuple[int, ...]:
    """Parses a version string into a comparable tuple of integers."""
    clean = re.sub(r"^[vV]\s*", "", v_str.strip())
    parts = []
    for token in re.split(r"[-._+]", clean):
        digits = re.findall(r"\d+", token)
        if digits:
            parts.append(int(digits[0]))
    return tuple(parts) if parts else (0,)


def is_newer_version(candidate: str, current: str = __version__) -> bool:
    """Returns True if candidate version is strictly greater than current version."""
    cand_tuple = parse_version_tuple(candidate)
    curr_tuple = parse_version_tuple(current)
    # Pad to equal length for comparison
    max_len = max(len(cand_tuple), len(curr_tuple))
    cand_padded = cand_tuple + (0,) * (max_len - len(cand_tuple))
    curr_padded = curr_tuple + (0,) * (max_len - len(curr_tuple))
    return cand_padded > curr_padded


def check_for_updates(
    repo: Optional[str] = None,
    timeout: int = 10,
    current_version: str = __version__
) -> Dict[str, Any]:
    """Queries GitHub Releases for the latest version of the application."""
    cfg = load_config()
    target_repo = repo or cfg.get("update_repo") or DEFAULT_REPO
    manifest_url = cfg.get("update_manifest_url")

    if manifest_url:
        url = manifest_url
    else:
        url = f"https://api.github.com/repos/{target_repo}/releases/latest"

    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": f"{APP_NAME}-Updater/{current_version}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {
            "status": "error",
            "error": f"HTTP {exc.code}: {exc.reason}",
            "current_version": current_version,
            "has_update": False,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "current_version": current_version,
            "has_update": False,
        }

    tag_name = data.get("tag_name", "").strip()
    latest_ver = re.sub(r"^[vV]\s*", "", tag_name)
    has_update = is_newer_version(latest_ver, current_version)

    # Identify download asset
    assets = data.get("assets", [])
    download_url = None
    asset_name = None
    asset_size = 0

    # Prefer windows zip/exe asset
    for asset in assets:
        name = asset.get("name", "").lower()
        if (".zip" in name or ".exe" in name) and ("win" in name or "portable" in name or len(assets) == 1):
            download_url = asset.get("browser_download_url")
            asset_name = asset.get("name")
            asset_size = asset.get("size", 0)
            break

    if not download_url and assets:
        download_url = assets[0].get("browser_download_url")
        asset_name = assets[0].get("name")
        asset_size = assets[0].get("size", 0)

    if not download_url:
        download_url = data.get("zipball_url") or data.get("tarball_url")
        asset_name = f"{tag_name}.zip"

    return {
        "status": "success",
        "current_version": current_version,
        "latest_version": latest_ver,
        "tag_name": tag_name,
        "has_update": has_update,
        "release_name": data.get("name") or tag_name,
        "release_notes": data.get("body", ""),
        "published_at": data.get("published_at", ""),
        "html_url": data.get("html_url", ""),
        "download_url": download_url,
        "asset_name": asset_name,
        "asset_size": asset_size,
    }


def download_update(
    download_url: str,
    target_path: Path,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 65536,
    timeout: int = 300
) -> Path:
    """Downloads an update package to target_path with chunked streaming and progress reporting."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": f"{APP_NAME}-Updater/{__version__}"}
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(target_path, "wb") as out_file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(downloaded, total_size)

    return target_path


def stage_windows_update(
    package_path: Path,
    target_dir: Optional[Path] = None,
    restart_app: bool = True
) -> Path:
    """Generates and launches a standalone Windows updater batch script."""
    app_dir = (target_dir or get_app_root()).resolve()
    pid = os.getpid()
    pkg_path = Path(package_path).resolve()

    temp_dir = Path(tempfile.gettempdir()) / f"{APP_NAME}_updater"
    temp_dir.mkdir(parents=True, exist_ok=True)
    script_path = temp_dir / "apply_update.bat"

    entrypoint = sys.executable
    is_compiled = getattr(sys, 'frozen', False)

    if is_compiled:
        restart_cmd = f'start "" "{entrypoint}"'
    else:
        main_script = app_dir / 'lidarcamera360.py'
        restart_cmd = f'start "" "{sys.executable}" "{main_script}"'

    restart_block = restart_cmd if restart_app else "rem No restart"

    bat_content = f"""@echo off
chcp 65001 > nul
echo [LidarCamera360 Updater] Waiting for application to exit...
:wait_pid
tasklist /fi "PID eq {pid}" 2>nul | find "{pid}" >nul
if %ERRORLEVEL% equ 0 (
    timeout /t 1 /nobreak > nul
    goto wait_pid
)

echo [LidarCamera360 Updater] Extracting update from {pkg_path.name}...
powershell -NoProfile -Command "Expand-Archive -Path '{str(pkg_path)}' -DestinationPath '{str(app_dir)}' -Force"
if %ERRORLEVEL% neq 0 (
    echo [!] Failed to extract update archive.
    pause
    exit /b 1
)

echo [LidarCamera360 Updater] Update applied successfully!
{restart_block}
exit /b 0
"""

    script_path.write_text(bat_content, encoding="utf-8")
    return script_path


def launch_staged_update(script_path: Path) -> subprocess.Popen:
    """Launches the updater script in a fully detached background process."""
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, 'DETACHED_PROCESS', 0x00000008)
    
    return subprocess.Popen(
        ["cmd.exe", "/c", str(script_path)],
        creationflags=flags,
        close_fds=True
    )
