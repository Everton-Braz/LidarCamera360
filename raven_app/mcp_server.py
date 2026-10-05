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


class LidarCameraMcpServer:
    """JSON-RPC 2.0 stdio MCP Server implementing Model Context Protocol specification."""

    def __init__(self):
        self.server_name = f"{APP_NAME}-MCP"
        self.server_version = __version__
        self.tools = self._register_tools()

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
