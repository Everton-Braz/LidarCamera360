#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serve 3D Aligner: Servidor Web Local para Calibração Interativa LiDAR–Câmera
Porta: 8080 (http://localhost:8080)
"""

import os
import sys
import json
import mimetypes
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Garantir logs em tempo real no terminal/console
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_DIR = PROJECT_ROOT / "web_3d_aligner"
COMPARATIVE_DIR = Path(r"C:\Users\User\Downloads\Lidou\COMPARATIVO_NUVENS_TODOS_METODOS")
COMPARATIVE_DIR.mkdir(parents=True, exist_ok=True)

PORT = 8080

mimetypes.add_type("application/octet-stream", ".bin")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("text/html", ".html")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")


def export_interactive_pcd(pitch, yaw, roll, model="thin_prism", focal_length=None, max_radius=1650.0):
    """Exporta PCD colorizada com os ângulos selecionados no visualizador 3D."""
    pose_file = WEB_DIR / "pose.json"
    pts_file = WEB_DIR / "points.bin"
    tex_file = WEB_DIR / "texture.jpg"

    if not (pose_file.exists() and pts_file.exists() and tex_file.exists()):
        return None, "Arquivos de pose ou textura não encontrados."

    with open(pose_file, "r") as f:
        pose = json.load(f)

    import cv2
    img = cv2.imread(str(tex_file))
    if img is None:
        return None, "Não foi possível carregar texture.jpg"
    H, W = img.shape[:2]

    pts = np.fromfile(str(pts_file), dtype=np.float32).reshape(-1, 3)
    t_wl = np.array(pose["t_w_l"])
    R_wl = np.array(pose["R_w_l"])
    R_base0 = np.array(pose["R_base0"])
    cL = np.array(pose["t_cam"])
    
    if focal_length is not None and float(focal_length) > 0:
        f = float(focal_length)
    elif model == "thin_prism":
        f = 1080.1874
    else:
        f = pose["f"]

    R_trim = Rot.from_euler("xyz", [pitch, yaw, roll], degrees=True).as_matrix()
    R_cam = R_trim @ R_base0

    # 1. World to LiDAR
    diff = pts - t_wl
    P_l = (R_wl.T @ diff.T).T

    # 2. LiDAR to Camera
    P_c = (R_cam @ (P_l - cL).T).T

    # 3. Fisheye Projection
    X = P_c[:, 0]
    Y = P_c[:, 1]
    Z = P_c[:, 2]
    dist = np.linalg.norm(P_c, axis=1)
    r_xy = np.hypot(X, Y)
    theta = np.arctan2(r_xy, Z)

    if model == "thin_prism":
        # Spirula Studio (COLMAP Thin Prism Fisheye)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        thetad = theta * (1.0 + 0.084684 * theta2 - 0.032074 * theta4 + 0.012301 * theta6 - 0.002990 * theta8)
        scale = np.where(r_xy > 1e-7, thetad / np.maximum(r_xy, 1e-7), 1.0)
        du = 2.0 * (-0.000413) * X * Y + 0.001235 * (r_xy * r_xy + 2.0 * X * X) + (-0.002747) * r_xy * r_xy
        dv = (-0.000413) * (r_xy * r_xy + 2.0 * Y * Y) + 2.0 * 0.001235 * X * Y + 0.000407 * r_xy * r_xy
        u = f * (X * scale + du) + (W / 2.0)
        v = f * (Y * scale + dv) + (H / 2.0)
    elif model == "kannala_brandt":
        k1, k2 = -0.04268, 0.00242
        theta2 = theta * theta
        r_img = f * (theta + k1 * theta * theta2 + k2 * theta * theta2 * theta2)
        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)
        u = (W / 2.0) + r_img * cos_phi
        v = (H / 2.0) + r_img * sin_phi
    else:
        r_img = f * theta
        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)
        u = (W / 2.0) + r_img * cos_phi
        v = (H / 2.0) + r_img * sin_phi

    r_center = np.hypot(u - W / 2.0, v - H / 2.0)
    valid = (Z > 0.1) & (dist > 0.3) & (dist < 30.0) & (r_center <= float(max_radius)) & (r_xy > 1e-6)
    valid = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    colors = np.full((len(pts), 3), 60, dtype=np.uint8)  # default cinza
    if np.any(valid):
        u_idx = np.clip(np.round(u[valid]).astype(np.int64), 0, W - 1)
        v_idx = np.clip(np.round(v[valid]).astype(np.int64), 0, H - 1)
        bgr = img[v_idx, u_idx]
        colors[valid] = bgr[:, ::-1]

    out_pcd = COMPARATIVE_DIR / f"Nuvem_Alinhada_P{pitch:+.2f}_Y{yaw:+.2f}_R{roll:+.2f}.pcd"
    
    # Escrever PCD binário
    n = len(pts)
    rgb_packed = (
        (colors[:, 0].astype(np.uint32) << 16)
        | (colors[:, 1].astype(np.uint32) << 8)
        | (colors[:, 2].astype(np.uint32))
    ).view(np.float32)

    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z rgb\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F U\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")

    dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4")]
    arr = np.empty(n, dtype=dt)
    arr["x"] = pts[:, 0].astype(np.float32)
    arr["y"] = pts[:, 1].astype(np.float32)
    arr["z"] = pts[:, 2].astype(np.float32)
    arr["rgb"] = rgb_packed

    with open(out_pcd, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())

    return str(out_pcd), None


class AlignerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.path = "/index.html"
        elif self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "online",
                "points_ready": (WEB_DIR / "points.bin").exists(),
                "pose_ready": (WEB_DIR / "pose.json").exists(),
                "texture_ready": (WEB_DIR / "texture.jpg").exists(),
                "comparative_dir": str(COMPARATIVE_DIR),
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                pitch = float(data.get("pitch", 0.0))
                yaw = float(data.get("yaw", 0.0))
                roll = float(data.get("roll", 0.0))
                model = str(data.get("model", "thin_prism"))
                focal_length = float(data.get("focal_length", 1080.19))
                max_radius = float(data.get("max_radius", 1650.0))

                # Salvar calibration JSON
                calib_out = {
                    "pitch_deg": pitch,
                    "yaw_deg": yaw,
                    "roll_deg": roll,
                    "model": model,
                    "focal_length": focal_length,
                    "max_radius": max_radius,
                    "export_timestamp": str(np.datetime64("now")),
                }
                with open(WEB_DIR / "saved_calibration.json", "w") as f:
                    json.dump(calib_out, f, indent=2)

                # Gerar PCD
                pcd_path, err = export_interactive_pcd(pitch, yaw, roll, model, focal_length, max_radius)
                if err:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": err}).encode("utf-8"))
                    return

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "message": f"Nuvem PCD gerada com sucesso!",
                    "pcd_path": pcd_path
                }).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def run_server():
    server_address = ("127.0.0.1", PORT)
    ThreadingHTTPServer.allow_reuse_address = True
    httpd = ThreadingHTTPServer(server_address, AlignerRequestHandler)
    print("=" * 70)
    print(f" [+] LiDAR–Fisheye 3D Web Aligner Server ONLINE na porta {PORT}!")
    print(f" [URL] http://localhost:{PORT} ou http://127.0.0.1:{PORT}")
    print(f" [Diretório Servido] {WEB_DIR}")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n [!] Servidor encerrado.")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
