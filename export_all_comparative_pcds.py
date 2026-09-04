import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))

from methods.method0_enhanced_current.fisheye_polynomial_model import PolynomialFisheyeCamera
from methods.method4_lidar_intensity_apriltag.intensity_projector import load_bag_points

OUT_DIR = Path(r"C:\Users\User\Downloads\Lidou\COMPARATIVO_NUVENS_TODOS_METODOS")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def write_pcd(path, points, colors_rgb):
    n = len(points)
    rgb_packed = (
        (colors_rgb[:, 0].astype(np.uint32) << 16)
        | (colors_rgb[:, 1].astype(np.uint32) << 8)
        | (colors_rgb[:, 2].astype(np.uint32))
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
    arr["x"] = points[:, 0].astype(np.float32)
    arr["y"] = points[:, 1].astype(np.float32)
    arr["z"] = points[:, 2].astype(np.float32)
    arr["rgb"] = rgb_packed

    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Nuvem PCD salva com sucesso: {path.name} ({os.path.getsize(path)/1e6:.2f} MB)")

def load_pcd_xyz(pcd_path):
    with open(pcd_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("POINTS"):
                num_points = int(line.split()[1])
            elif line.startswith("DATA"):
                break
        raw = f.read(num_points * 16)
        data = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
        return data[:, :3].astype(np.float64)

def colorize_dual_fisheye(pts, img_front, img_back, camera_model, R_front, R_back, c_L):
    N = len(pts)
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_dist = np.full(N, 1e9, dtype=np.float32)

    # Stream 0: Front Lens
    pts_c0 = (R_front @ (pts - c_L).T).T
    u0, v0, valid0 = camera_model.project(pts_c0)
    d0 = np.linalg.norm(pts_c0, axis=1)
    
    better0 = valid0 & (d0 < best_dist)
    if np.any(better0):
        u_idx0 = np.round(u0[better0]).astype(np.int32)
        v_idx0 = np.round(v0[better0]).astype(np.int32)
        bgr0 = img_front[v_idx0, u_idx0]
        colors[better0] = bgr0[:, ::-1] # BGR to RGB
        best_dist[better0] = d0[better0]

    # Stream 1: Back Lens
    if img_back is not None:
        pts_c1 = (R_back @ (pts - c_L).T).T
        u1, v1, valid1 = camera_model.project(pts_c1)
        d1 = np.linalg.norm(pts_c1, axis=1)

        better1 = valid1 & (d1 < best_dist)
        if np.any(better1):
            u_idx1 = np.round(u1[better1]).astype(np.int32)
            v_idx1 = np.round(v1[better1]).astype(np.int32)
            bgr1 = img_back[v_idx1, u_idx1]
            colors[better1] = bgr1[:, ::-1]
            best_dist[better1] = d1[better1]

    return colors

def main():
    print("=" * 70)
    print(" EXPORTADOR DE NUVENS DE PONTOS (.PCD) COMPARATIVAS DE TODOS OS MÉTODOS")
    print(f" Pasta de Destino: {OUT_DIR}")
    print("=" * 70)

    # Base Rig Geometry
    u_L = np.array([0.003102, -0.504937, -0.863152])
    c_L = 0.185 * u_L

    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_base0 = np.stack([X0, Y0, Z0], axis=0)

    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_base1 = np.stack([X1, Y1, Z1], axis=0)

    # Camera models
    cam_linear = PolynomialFisheyeCamera.from_fov(3840, 3840, 196.0, k1=0.0, k2=0.0)
    cam_polynomial = PolynomialFisheyeCamera.from_fov(3840, 3840, 196.0, k1=-0.04268, k2=0.00242)

    # =========================================================================
    # 1. NUVENS DO DATASET ESTÁTICO (20260902092520)
    # =========================================================================
    static_dir = Path(r"C:\Users\User\Downloads\Lidou\20260902092520")
    static_bag = static_dir / "LIDAR_20260902092520.bag"
    img_static_f = cv2.imread(str(static_dir / "extracted_fisheye" / "lens1_front.jpg"))
    img_static_b = cv2.imread(str(static_dir / "extracted_fisheye" / "lens2_back.jpg"))

    print("\n[*] Carregando nuvem de pontos da Sessão Estática 20260902092520...")
    raw_static = load_bag_points(str(static_bag), max_frames=50)
    pts_static = raw_static[:, :3]
    print(f"    Total: {len(pts_static):,} pontos")

    # A. Baseline Cand 03 (Estático)
    print("\n[*] Gerando PCD: 00_ESTATICO_Baseline_Cand03.pcd...")
    colors_00_stat = colorize_dual_fisheye(pts_static, img_static_f, img_static_b,
                                           cam_linear, R_base0, R_base1, c_L)
    write_pcd(OUT_DIR / "00_ESTATICO_Baseline_Cand03.pcd", pts_static, colors_00_stat)

    # B. Método 1: Superfície Planar Kabsch (Estático)
    with open(PROJECT_ROOT / "methods" / "method1_surface_planar_ai" / "results" / "method1_calibration_summary.json") as f:
        m1_res = json.load(f)
    R_m1 = np.array(m1_res["rotation_matrix"])
    print("\n[*] Gerando PCD: 01_ESTATICO_Metodo1_Superficie_Planar_Kabsch.pcd...")
    colors_01_stat = colorize_dual_fisheye(pts_static, img_static_f, img_static_b,
                                           cam_linear, R_m1, R_base1, c_L)
    write_pcd(OUT_DIR / "01_ESTATICO_Metodo1_Superficie_Planar_Kabsch.pcd", pts_static, colors_01_stat)

    # C. Método 2: Target-less NID (Estático)
    with open(PROJECT_ROOT / "methods" / "method2_targetless_nid" / "results" / "method2_nid_summary.json") as f:
        m2_res = json.load(f)
    R_m2 = np.array(m2_res["rotation_matrix"])
    print("\n[*] Gerando PCD: 02_ESTATICO_Metodo2_Targetless_NID.pcd...")
    colors_02_stat = colorize_dual_fisheye(pts_static, img_static_f, img_static_b,
                                           cam_polynomial, R_m2, R_base1, c_L)
    write_pcd(OUT_DIR / "02_ESTATICO_Metodo2_Targetless_NID.pcd", pts_static, colors_02_stat)

    # =========================================================================
    # 2. NUVENS DO DATASET DINÂMICO (DinamicAprilTagCalib)
    # =========================================================================
    dynamic_dir = Path(r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib")
    pcd_single_frame = dynamic_dir / "single_frame_calibrated_pcds" / "single_frame_nominal_pitch0_yaw0.pcd"
    img_dyn_f = cv2.imread(str(dynamic_dir / "preview_frame_10s_stream0.jpg"))
    img_dyn_b = cv2.imread(str(dynamic_dir / "preview_frame_10s_stream1.jpg"))

    print("\n[*] Carregando nuvem da moldura/parede da Sessão Dinâmica (Frame t=10s)...")
    pts_dyn = load_pcd_xyz(str(pcd_single_frame))
    print(f"    Total: {len(pts_dyn):,} pontos")

    if img_dyn_f.shape[:2] != (3840, 3840):
        img_dyn_f = cv2.resize(img_dyn_f, (3840, 3840), interpolation=cv2.INTER_LINEAR)
    if img_dyn_b is not None and img_dyn_b.shape[:2] != (3840, 3840):
        img_dyn_b = cv2.resize(img_dyn_b, (3840, 3840), interpolation=cv2.INTER_LINEAR)

    # D. Baseline Linear (Dinâmico)
    print("\n[*] Gerando PCD: 00_DINAMICO_Baseline_Linear_Pitch0_Yaw0.pcd...")
    colors_dyn_00 = colorize_dual_fisheye(pts_dyn, img_dyn_f, img_dyn_b,
                                          cam_linear, R_base0, R_base1, c_L)
    write_pcd(OUT_DIR / "00_DINAMICO_Baseline_Linear_Pitch0_Yaw0.pcd", pts_dyn, colors_dyn_00)

    # E. Método 0 Aprimorado (Kannala-Brandt Polinomial)
    with open(PROJECT_ROOT / "methods" / "method0_enhanced_current" / "results" / "enhanced_method0_results.json") as f:
        m0_enh = json.load(f)
    R_m0_enh = np.array(m0_enh[1]["R0_matrix"])
    R_trim1_enh = Rot.from_euler("xyz", [-1.3704, -0.2965, -0.4567], degrees=True).as_matrix()
    R_m0_enh1 = R_trim1_enh @ R_base1

    print("\n[*] Gerando PCD: 02_DINAMICO_Metodo0_Aprimorado_Kannala_Brandt.pcd...")
    colors_dyn_m0 = colorize_dual_fisheye(pts_dyn, img_dyn_f, img_dyn_b,
                                          cam_polynomial, R_m0_enh, R_m0_enh1, c_L)
    write_pcd(OUT_DIR / "02_DINAMICO_Metodo0_Aprimorado_Kannala_Brandt.pcd", pts_dyn, colors_dyn_m0)

    # F. Método 3: Global Bundle Adjustment (IPB)
    with open(PROJECT_ROOT / "methods" / "method3_global_bundle_reference" / "results" / "method3_global_bundle_summary.json") as f:
        m3_res = json.load(f)
    R_m3 = np.array(m3_res["rotation_matrix"])
    R_trim1_m3 = Rot.from_euler("xyz", [-2.9723, -0.4254, -0.3596], degrees=True).as_matrix()
    R_m3_1 = R_trim1_m3 @ R_base1

    print("\n[*] Gerando PCD: 03_DINAMICO_Metodo3_Global_Bundle_BA.pcd...")
    colors_dyn_m3 = colorize_dual_fisheye(pts_dyn, img_dyn_f, img_dyn_b,
                                          cam_polynomial, R_m3, R_m3_1, c_L)
    write_pcd(OUT_DIR / "03_DINAMICO_Metodo3_Global_Bundle_BA.pcd", pts_dyn, colors_dyn_m3)

    # G. Ground Truth CloudCompare (Cand 07 - Medição Exata da Moldura)
    R_trim0_c7 = Rot.from_euler("xyz", [-1.236, +2.427, 0.0], degrees=True).as_matrix()
    R_trim1_c7 = Rot.from_euler("xyz", [-1.236, -2.427, 0.0], degrees=True).as_matrix()
    R_c7_0 = R_trim0_c7 @ R_base0
    R_c7_1 = R_trim1_c7 @ R_base1

    print("\n[*] Gerando PCD: 04_DINAMICO_GroundTruth_CloudCompare_Cand07.pcd...")
    colors_dyn_c7 = colorize_dual_fisheye(pts_dyn, img_dyn_f, img_dyn_b,
                                          cam_linear, R_c7_0, R_c7_1, c_L)
    write_pcd(OUT_DIR / "04_DINAMICO_GroundTruth_CloudCompare_Cand07.pcd", pts_dyn, colors_dyn_c7)

    print("\n" + "=" * 70)
    print(" TODAS AS NUVENS FORAM EXPORTADAS COM SUCESSO!")
    print(f" Abra a pasta no CloudCompare para inspecionar lado a lado:\n {OUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
