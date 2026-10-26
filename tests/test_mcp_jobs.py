import json
import sys

import pytest

from raven_app.mcp_jobs import JobManager
from raven_app.mcp_server import LidarCameraMcpServer, _cli_arguments


def _call(server, name, arguments):
    response = server.handle_request({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                      "params": {"name": name, "arguments": arguments}})
    return response["result"]


def test_new_tools_are_listed():
    names = {tool["name"] for tool in LidarCameraMcpServer().tools}
    assert {"list_profiles", "get_profile", "validate_calibration", "calibrate_rig", "run_workflow",
            "colorize", "get_job_status", "cancel_job", "list_jobs"} <= names


def test_cli_arguments_follow_the_cli_flags():
    args = _cli_arguments("workflow", {"bag": "a.bag", "insv": "b.insv", "output": "out", "method": "reconstruction",
                                       "fps": 1, "revisit_refinement": True, "mask_persons": False,
                                       "operator_radius": 1.0, "rig_refine": "photometric"})
    assert args[:6] == ["--bag", "a.bag", "--insv", "b.insv", "--output", "out"]
    assert "--revisit-refinement" in args and "--mask-persons" not in args
    assert args[args.index("--operator-radius") + 1] == "1.0"
    assert "--export-ply" in args   # MCP runs keep the colorized cloud by default
    calibrate = _cli_arguments("calibrate", {"dataset": "d", "mode": "dt", "fix_dt": True})
    assert calibrate[:2] == ["--dataset", "d"] and "--fix-dt" in calibrate
    assert calibrate[calibrate.index("--rig-refine") + 1] == "dt"
    with pytest.raises(ValueError, match="Unsupported argument"):
        _cli_arguments("colorize", {"dataset": "d", "bag": "x"})


def test_profile_and_calibration_tools(tmp_path):
    server = LidarCameraMcpServer()
    profiles = json.loads(_call(server, "list_profiles", {})["content"][0]["text"])
    assert profiles["default"].endswith("raven_x4_montagem_1.json")
    profile = json.loads(_call(server, "get_profile", {})["content"][0]["text"])
    assert "T_lidar_to_cam0_rigid_4x4" in profile["profile"]
    bad = dict(profile["profile"], T_lidar_to_cam1_rigid_4x4=[[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    verdict = json.loads(_call(server, "validate_calibration", {"calibration_path": str(path)})["content"][0]["text"])
    assert verdict["usable"] is False and verdict["reasons"]


def test_calibrate_job_runs_the_cli_and_reports_failure(tmp_path):
    server = LidarCameraMcpServer()
    server._jobs = JobManager(tmp_path / "jobs")
    started = json.loads(_call(server, "calibrate_rig", {"dataset": str(tmp_path / "missing"), "dt": 1.0})["content"][0]["text"])
    assert started["status"] == "running" and started["args"][0] == "calibrate"

    final = server.jobs.wait(started["job_id"], timeout=120)

    assert final["status"] == "failed" and final["exit_code"] != 0
    assert any("missing" in line or "Error" in line or "No such" in line for line in final["log_tail"])
    listed = json.loads(_call(server, "list_jobs", {})["content"][0]["text"])
    assert listed[0]["job_id"] == started["job_id"]


def test_cancel_stops_a_running_job(tmp_path, monkeypatch):
    jobs = JobManager(tmp_path / "jobs")
    monkeypatch.setattr("raven_app.mcp_jobs.cli_command",
                        lambda args: [sys.executable, "-c", "import time; print('[STAGE 1/7] Waiting', flush=True); time.sleep(60)"])
    started = jobs.start("run_workflow", ["workflow"])
    jobs.wait(started["job_id"], timeout=2)

    status = jobs.cancel(started["job_id"])

    assert status["status"] == "cancelled"
    assert status["stage"] == {"index": 1, "total": 7, "name": "Waiting"}


def test_unknown_job_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        JobManager(tmp_path).status("0123456789ab")
    with pytest.raises(ValueError):
        JobManager(tmp_path).status("../../etc")
