"""Unit and integration tests for telemetry, updater, MCP server, and feedback."""
import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

from raven_app import __version__
from raven_app.telemetry import (
    get_anonymous_client_id,
    is_telemetry_enabled,
    set_telemetry_enabled,
    get_system_telemetry_payload,
)
from raven_app.updater import (
    parse_version_tuple,
    is_newer_version,
    check_for_updates,
    stage_windows_update,
)
from raven_app.mcp_server import LidarCameraMcpServer
from raven_app.feedback import (
    format_markdown_report,
    build_github_issue_url,
    export_feedback_bundle,
)


def test_telemetry_opt_in_toggle():
    set_telemetry_enabled(True)
    assert is_telemetry_enabled() is True
    set_telemetry_enabled(False)
    assert is_telemetry_enabled() is False
    set_telemetry_enabled(True)
    assert is_telemetry_enabled() is True


def test_anonymous_client_id():
    cid = get_anonymous_client_id()
    assert isinstance(cid, str)
    assert len(cid) >= 16
    # Calling again returns stable ID
    assert get_anonymous_client_id() == cid


def test_system_telemetry_payload():
    payload = get_system_telemetry_payload()
    assert "client_id" in payload
    assert "os" in payload
    assert ("python" in payload or "python_version" in payload)
    assert payload.get("app_version") == __version__


def test_version_parsing_and_comparison():
    assert parse_version_tuple("0.2.0") == (0, 2, 0)
    assert parse_version_tuple("v1.0.4-rc1") == (1, 0, 4)
    assert is_newer_version("0.3.0", "0.2.0") is True
    assert is_newer_version("0.2.0", "0.2.0") is False
    assert is_newer_version("0.1.9", "0.2.0") is False
    assert is_newer_version("1.0.0", "0.9.99") is True


def test_stage_windows_update():
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        pkg_path = Path(f.name)
    try:
        script = stage_windows_update(pkg_path, restart_app=False)
        assert script.is_file()
        script_content = script.read_text(encoding="utf-8")
        assert "tar -xf" in script_content or "Expand-Archive" in script_content
    finally:
        if pkg_path.is_file():
            pkg_path.unlink()


def test_feedback_markdown_and_url():
    report = {
        "title": "Camera lag in 4K",
        "description": "Frames drop after 5 mins",
        "category": "bug",
        "steps": "1. Open INSV\n2. Extract at 30fps",
        "contact": "test@example.com",
        "diagnostics": {"gpu": "RTX 5070"}
    }
    md = format_markdown_report(report)
    assert "[BUG] Camera lag in 4K" in md
    assert "Frames drop after 5 mins" in md
    assert "RTX 5070" in md

    url = build_github_issue_url(report, repo="Everton-Braz/LidarCamera360")
    assert url.startswith("https://github.com/Everton-Braz/LidarCamera360/issues/new?")
    assert "title=%5BBug%5D%3A+Camera+lag+in+4K" in url


def test_feedback_bundle_export():
    report = {
        "title": "Bundle test",
        "description": "Testing zip packaging",
        "diagnostics": {"test": True}
    }
    with tempfile.TemporaryDirectory() as td:
        out_zip = Path(td) / "test_bundle.zip"
        created = export_feedback_bundle(report, out_zip)
        assert created.is_file()
        assert created.stat().st_size > 0


def test_mcp_server_protocol():
    server = LidarCameraMcpServer()
    init_resp = server.handle_request({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    })
    assert init_resp is not None
    assert init_resp["result"]["serverInfo"]["name"] == "LidarCamera360-MCP"

    tools_resp = server.handle_request({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    })
    assert tools_resp is not None
    tool_names = [t["name"] for t in tools_resp["result"]["tools"]]
    assert "doctor" in tool_names
    assert "inspect_bag" in tool_names
    assert "extract_gps" in tool_names
    assert "extract_insv_frames" in tool_names
    assert "check_updates" in tool_names
    assert "submit_feedback" in tool_names

    doc_resp = server.handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "doctor", "arguments": {}}
    })
    assert doc_resp is not None
    assert "content" in doc_resp["result"]
