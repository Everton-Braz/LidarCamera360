"""Unit and integration tests for updater, MCP server, and feedback."""
import json
import os
import sys
import tempfile
from pathlib import Path
import pytest

from raven_app import __version__
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
    assert "georeference" in tool_names

    doc_resp = server.handle_request({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "doctor", "arguments": {}}
    })
    assert doc_resp is not None
    assert "content" in doc_resp["result"]

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
