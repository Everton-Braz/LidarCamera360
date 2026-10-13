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

    # Extended tools check
    assert "telemetry_status" in tool_names
    assert "configure_telemetry" in tool_names
    assert "georeference" in tool_names

    # Test telemetry_status tool
    t_stat = server.handle_request({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "telemetry_status", "arguments": {}}
    })
    assert t_stat is not None
    content = json.loads(t_stat["result"]["content"][0]["text"])
    assert "telemetry_enabled" in content
    assert "client_id" in content

    # Test configure_telemetry tool
    t_cfg = server.handle_request({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "configure_telemetry", "arguments": {"enabled": False}}
    })
    assert t_cfg is not None
    content_cfg = json.loads(t_cfg["result"]["content"][0]["text"])
    assert content_cfg["telemetry_enabled"] is False

    # Restore telemetry
    server.handle_request({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {"name": "configure_telemetry", "arguments": {"enabled": True}}
    })

    # Test resources/list and prompts/list
    res_resp = server.handle_request({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "resources/list",
        "params": {}
    })
    assert res_resp is not None and "resources" in res_resp["result"]

    prompt_resp = server.handle_request({
        "jsonrpc": "2.0",
        "id": 8,
        "method": "prompts/list",
        "params": {}
    })
    assert prompt_resp is not None and "prompts" in prompt_resp["result"]


def test_cli_telemetry_commands(capsys):
    from raven_app.cli import parse, run
    args = parse(["telemetry", "--enable"])
    assert args.command == "telemetry"
    assert args.enable is True
    ret = run(args)
    assert ret == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["telemetry_enabled"] is True

    args_stat = parse(["telemetry", "--status"])
    ret_stat = run(args_stat)
    assert ret_stat == 0


def test_cli_update_check_command(capsys):
    from raven_app.cli import parse, run
    args = parse(["update", "check"])
    assert args.command == "update"
    assert args.action == "check"
    ret = run(args)
    assert ret == 0
    captured = capsys.readouterr()
    info = json.loads(captured.out)
    assert info["status"] == "success"
    assert "current_version" in info


def test_cli_feedback_command(tmp_path, capsys):
    from raven_app.cli import parse, run
    out_bundle = tmp_path / "diag_bundle.zip"
    args = parse([
        "feedback",
        "--title", "CLI Test Feedback",
        "--description", "Testing feedback through CLI runner",
        "--export-bundle", str(out_bundle)
    ])
    assert args.command == "feedback"
    ret = run(args)
    assert ret == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out)
    assert res["status"] == "processed"
    assert out_bundle.is_file()

