"""Native Model Context Protocol (MCP) Server for LidarCamera360."""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure workspace root is in sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from raven_app import __version__
from raven_app.branding import APP_NAME, APP_DESCRIPTION


# MCP argument -> CLI flag per command; booleans become bare flags.
_CLI_OPTIONS = {
    'calibrate': {'dataset': '--dataset', 'calib': '--calib', 'profile': '--profile', 'dt': '--dt',
                  'fix_dt': '--fix-dt', 'masks_dir': '--masks-dir', 'mode': '--rig-refine', 'frames': '--frames'},
    'workflow': {'bag': '--bag', 'insv': '--insv', 'output': '--output', 'method': '--method', 'fps': '--fps',
                 'profile': '--profile', 'rig_refine': '--rig-refine', 'time_window': '--time-window',
                 'revisit_refinement': '--revisit-refinement', 'dt': '--dt', 'calib': '--calib',
                 'mask_persons': '--mask-persons', 'mask_backend': '--mask-backend', 'mask_config': '--mask-config',
                 'masks_dir': '--masks-dir', 'operator_radius': '--operator-radius', 'third_camera': '--third-camera',
                 'export_ply': '--export-ply', 'export_pcd': '--export-pcd'},
    'colorize': {'dataset': '--dataset', 'method': '--method', 'calib': '--calib', 'profile': '--profile',
                 'dt': '--dt', 'masks_dir': '--masks-dir', 'operator_radius': '--operator-radius',
                 'rig_refine': '--rig-refine', 'time_window': '--time-window'},
}


def _cli_arguments(command, arguments):
    options = _CLI_OPTIONS[command]
    unknown = sorted(set(arguments) - set(options))
    if unknown:
        raise ValueError(f"Unsupported argument(s) for {command}: {', '.join(unknown)}")
    if command == 'workflow':
        arguments = {'export_ply': True, **arguments}
    args = []
    for key, flag in options.items():
        if key not in arguments or arguments[key] is None:
            continue
        value = arguments[key]
        if isinstance(value, bool):
            if value:
                args.append(flag)
        else:
            args += [flag, str(value)]
    return args


class LidarCameraMcpServer:
    """JSON-RPC 2.0 stdio MCP Server implementing Model Context Protocol specification."""

    def __init__(self):
        self.server_name = f"{APP_NAME}-MCP"
        self.server_version = __version__
        self.tools = self._register_tools()
        self._jobs = None

    @property
    def jobs(self):
        if self._jobs is None:
            from raven_app.mcp_jobs import JobManager
            self._jobs = JobManager()
        return self._jobs

    def _register_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "doctor",
                "description": "Check health and availability of bundled FAST-LIVO2, Spirula, Vulkan, and GPU engines.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "inspect_bag",
                "description": "Inspect ROS1 .bag files for LiDAR and IMU topics without requiring ROS.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bag_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of absolute file paths to .bag files."
                        }
                    },
                    "required": ["bag_paths"]
                }
            },
            {
                "name": "extract_gps",
                "description": "Extract native GNSS / GPS telemetry from Insta360 .INSV video files.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "insv_path": {"type": "string", "description": "Path to Insta360 .INSV file."},
                        "output_dir": {"type": "string", "description": "Directory to save GeoJSON/GPX/CSV."},
                        "formats": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["geojson", "gpx", "csv"]},
                            "default": ["geojson", "gpx", "csv"]
                        }
                    },
                    "required": ["insv_path", "output_dir"]
                }
            },
            {
                "name": "extract_insv_frames",
                "description": "Extract synchronized dual-lens fisheye frames from Insta360 INSV using PyAV.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "insv_path": {"type": "string", "description": "Path to Insta360 .INSV file."},
                        "output_dir": {"type": "string", "description": "Directory where frames/cam0 and frames/cam1 will be created."},
                        "fps": {"type": "number", "default": 2.0, "description": "Extraction frame rate in FPS."}
                    },
                    "required": ["insv_path", "output_dir"]
                }
            },
            {
                "name": "check_updates",
                "description": "Check for remote OTA releases on GitHub and view changelog.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "description": "Optional custom owner/repo override."}
                    }
                }
            },
            {
                "name": "georeference",
                "description": "Calculate INSV GPS trajectory, georeference point cloud or COLMAP dataset to target CRS, and export GeoJSON.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "insv_path": {"type": "string", "description": "Path to Insta360 .INSV video."},
                        "output_dir": {"type": "string", "description": "Output directory for georeferenced deliverables and GeoJSON."},
                        "cloud_path": {"type": "string", "description": "Optional path to point cloud (PCD/PLY/LAZ/LAS)."},
                        "colmap_dir": {"type": "string", "description": "Optional COLMAP dataset directory."},
                        "trajectory_path": {"type": "string", "description": "Optional trajectory file (.txt)."},
                        "crs": {"type": "string", "description": "Target projected CRS (e.g. EPSG:31984, EPSG:32724)."},
                        "yaw_deg": {"type": "number", "description": "Optional manual True North yaw angle in degrees."},
                        "orthophoto": {"type": "boolean", "default": False, "description": "Generate high-resolution georeferenced orthophoto."}
                    },
                    "required": ["insv_path", "output_dir"]
                }
            },
            {
                "name": "list_profiles",
                "description": "List hardware rig profiles (scanner + camera mount) and which one is the default.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "get_profile",
                "description": "Read a hardware rig profile (id or JSON path; default profile when omitted).",
                "inputSchema": {
                    "type": "object",
                    "properties": {"profile": {"type": "string", "description": "Profile id or JSON path."}}
                }
            },
            {
                "name": "validate_calibration",
                "description": "Check whether a rig calibration JSON is usable: not rejected and physically close to the hardware profile (lens baseline, angle between lenses, per-lens rotation/position, lever arm).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "calibration_path": {"type": "string"},
                        "profile": {"type": "string", "description": "Reference profile id or path (default profile when omitted)."}
                    },
                    "required": ["calibration_path"]
                }
            },
            {
                "name": "calibrate_rig",
                "description": "Start a background job refining the rig mount and time offset(s) of a processed dataset by multi-view colour consistency (accepted only if held-out frames improve). Returns a job_id; poll get_job_status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string", "description": "Processed output directory (images/, SLAM, frames.json)."},
                        "calib": {"type": "string", "description": "Initial calibration (default: hardware profile)."},
                        "profile": {"type": "string"},
                        "dt": {"type": "number", "description": "Initial time offset, SLAM reference."},
                        "fix_dt": {"type": "boolean", "default": False},
                        "masks_dir": {"type": "string"},
                        "mode": {"type": "string", "enum": ["photometric", "dt"], "default": "photometric"},
                        "frames": {"type": "integer", "default": 30}
                    },
                    "required": ["dataset"]
                }
            },
            {
                "name": "run_workflow",
                "description": "Start the end-to-end workflow (bag + 360 video -> SLAM -> sync -> rig refinement -> time-gated colorization -> deliverables) as a background job. Returns a job_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bag": {"type": "string"}, "insv": {"type": "string", "description": "INSV or equirectangular MP4."},
                        "output": {"type": "string"},
                        "method": {"type": "string", "enum": ["direct", "sfm", "all", "trajectory", "reconstruction"], "default": "direct"},
                        "fps": {"type": "number", "default": 2.0},
                        "profile": {"type": "string"},
                        "rig_refine": {"type": "string", "enum": ["photometric", "dt", "none"], "default": "photometric"},
                        "time_window": {"type": "number", "default": 6.0},
                        "revisit_refinement": {"type": "boolean", "default": False, "description": "Loop closure."},
                        "dt": {"type": "number"},
                        "calib": {"type": "string"},
                        "mask_persons": {"type": "boolean", "default": False},
                        "mask_backend": {"type": "string", "enum": ["vulkan", "tensorrt"]},
                        "mask_config": {"type": "string"},
                        "masks_dir": {"type": "string"},
                        "operator_radius": {"type": "number"},
                        "third_camera": {"type": "string"},
                        "export_ply": {"type": "boolean", "default": True},
                        "export_pcd": {"type": "boolean", "default": False}
                    },
                    "required": ["bag", "insv", "output"]
                }
            },
            {
                "name": "colorize",
                "description": "Start a background job colorizing an already processed dataset (rig refinement + time-gated colorization). Returns a job_id.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dataset": {"type": "string"},
                        "method": {"type": "string", "enum": ["direct", "sfm", "all", "trajectory", "reconstruction"], "default": "direct"},
                        "calib": {"type": "string"}, "profile": {"type": "string"}, "dt": {"type": "number"},
                        "masks_dir": {"type": "string"}, "operator_radius": {"type": "number"},
                        "rig_refine": {"type": "string", "enum": ["photometric", "dt", "none"], "default": "photometric"},
                        "time_window": {"type": "number", "default": 6.0}
                    },
                    "required": ["dataset"]
                }
            },
            {
                "name": "get_job_status",
                "description": "Status of a background job: running/succeeded/failed/cancelled, current stage, outputs, rig refinement verdicts and the log tail.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "tail": {"type": "integer", "default": 40}},
                    "required": ["job_id"]
                }
            },
            {
                "name": "cancel_job",
                "description": "Stop a running background job.",
                "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]}
            },
            {
                "name": "list_jobs",
                "description": "Recent background jobs and their status.",
                "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 20}}}
            },
            {
                "name": "submit_feedback",
                "description": "Submit a bug report or feature request with diagnostics.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Short summary of the bug or suggestion."},
                        "description": {"type": "string", "description": "Detailed description."},
                        "category": {"type": "string", "enum": ["bug", "feature", "question"], "default": "bug"},
                        "steps": {"type": "string", "description": "Steps to reproduce if reporting a bug."},
                        "contact": {"type": "string", "description": "Optional email or contact info."}
                    },
                    "required": ["title", "description"]
                }
            }
        ]

    def handle_tool_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Dispatches and executes the requested tool."""
        if name == "doctor":
            from raven_app.cli import engine
            from raven_app.config import get_spirula_bin, validate_tool
            from raven_app.vulkan_engine import vulkan_status
            sp = get_spirula_bin()
            ok_sp, ver_sp = validate_tool("spirula", str(sp))
            eng = engine()
            np_ver, cv_ver = "unknown", "unknown"
            try:
                import numpy
                np_ver = numpy.__version__
            except Exception:
                pass
            try:
                import cv2 # type: ignore
                cv_ver = getattr(cv2, '__version__', 'unknown')
            except Exception:
                pass
            return {
                "version": __version__,
                "app_name": APP_NAME,
                "native_engine": str(eng),
                "native_engine_exists": eng.is_file(),
                "spirula_ready": ok_sp,
                "spirula_version": ver_sp,
                "vulkan_colorizer": vulkan_status(),
                "numpy": np_ver,
                "opencv": cv_ver,
            }

        elif name == "inspect_bag":
            from raven_app.bag_io import inspect_bags
            paths = [Path(p) for p in arguments["bag_paths"]]
            return inspect_bags(paths)

        elif name == "extract_gps":
            from raven_app.insv_gps import export_gps
            insv = Path(arguments["insv_path"])
            out_dir = Path(arguments["output_dir"])
            formats = arguments.get("formats", ["geojson", "gpx", "csv"])
            return export_gps(insv, out_dir / insv.stem, formats=formats)

        elif name == "extract_insv_frames":
            from raven_app.video import extract_insv_frames_pyav
            insv = Path(arguments["insv_path"])
            out_dir = Path(arguments["output_dir"])
            fps = float(arguments.get("fps", 2.0))
            success = extract_insv_frames_pyav(insv, out_dir, fps=fps)
            return {"success": bool(success), "output_dir": str(out_dir)}

        elif name == "check_updates":
            from raven_app.updater import check_for_updates
            repo = arguments.get("repo")
            return check_for_updates(repo=repo)

        elif name == "georeference":
            from raven_app.georeference import run_georeference
            insv = Path(arguments["insv_path"])
            out_dir = Path(arguments["output_dir"])
            cloud = Path(arguments["cloud_path"]) if arguments.get("cloud_path") else None
            colmap_dir = Path(arguments["colmap_dir"]) if arguments.get("colmap_dir") else None
            traj = Path(arguments["trajectory_path"]) if arguments.get("trajectory_path") else None
            crs = arguments.get("crs")
            yaw_deg = arguments.get("yaw_deg")
            ortho = bool(arguments.get("orthophoto", False))
            return run_georeference(
                insv, out_dir,
                cloud=cloud,
                colmap_dir=colmap_dir,
                trajectory=traj,
                crs=crs,
                yaw_deg=yaw_deg,
                generate_orthophoto_file=ortho
            )

        elif name == "list_profiles":
            from raven_app.profile_manager import list_available_profiles, resolve_profile
            default = resolve_profile()
            return {"default": str(default) if default else None, "profiles": list_available_profiles()}

        elif name == "get_profile":
            from raven_app.profile_manager import resolve_profile
            path = resolve_profile(arguments.get("profile"))
            if path is None:
                raise ValueError("No hardware rig profile is available")
            return {"path": str(path), "profile": json.loads(Path(path).read_text(encoding="utf-8"))}

        elif name == "validate_calibration":
            from raven_app.profile_manager import resolve_profile
            from raven_app.workflow import calibration_gate_failures
            reference = resolve_profile(arguments.get("profile"))
            reasons = calibration_gate_failures(Path(arguments["calibration_path"]), reference_profile=reference)
            return {"usable": not reasons, "reasons": reasons, "reference_profile": str(reference) if reference else None}

        elif name in ("calibrate_rig", "run_workflow", "colorize"):
            command = {"calibrate_rig": "calibrate", "run_workflow": "workflow", "colorize": "colorize"}[name]
            return self.jobs.start(name, [command] + _cli_arguments(command, arguments))

        elif name == "get_job_status":
            return self.jobs.status(arguments["job_id"], tail=int(arguments.get("tail", 40)))

        elif name == "cancel_job":
            return self.jobs.cancel(arguments["job_id"])

        elif name == "list_jobs":
            return self.jobs.list(limit=int(arguments.get("limit", 20)))

        elif name == "submit_feedback":
            from raven_app.feedback import gather_diagnostics, build_github_issue_url, send_feedback_webhook
            report = {
                "title": arguments["title"],
                "description": arguments["description"],
                "category": arguments.get("category", "bug"),
                "steps": arguments.get("steps", ""),
                "contact": arguments.get("contact", ""),
                "diagnostics": gather_diagnostics()
            }
            webhook_result = send_feedback_webhook(report)
            issue_url = build_github_issue_url(report)
            return {
                "status": "recorded",
                "issue_url": issue_url,
                "webhook_status": webhook_result
            }

        raise ValueError(f"Unknown tool: {name}")

    def handle_request(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes a JSON-RPC 2.0 request according to the MCP spec."""
        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        # Notifications (no id)
        if req_id is None:
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": self.server_name,
                        "version": self.server_version
                    }
                }
            }

        elif method == "notifications/initialized":
            return None

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": self.tools
                }
            }

        elif method in ("resources/list", "prompts/list"):
            key = "resources" if method == "resources/list" else "prompts"
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    key: []
                }
            }

        elif method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            try:
                result = self.handle_tool_call(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2, ensure_ascii=False)
                            }
                        ]
                    }
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "isError": True,
                        "content": [
                            {
                                "type": "text",
                                "text": f"Error executing {name}: {exc}"
                            }
                        ]
                    }
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found"
            }
        }

    def run_stdio(self):
        """Runs the main stdio loop listening on stdin and emitting to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_request(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError:
                err = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                sys.stdout.write(json.dumps(err) + "\n")
                sys.stdout.flush()


def main():
    server = LidarCameraMcpServer()
    server.run_stdio()
    return 0


if __name__ == "__main__":
    sys.exit(main())
