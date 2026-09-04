import os
import sys
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from fisheye_polynomial_model import PolynomialFisheyeCamera

DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
TRJ_FILE = os.path.join(DATASET_DIR, "slam_out", "result", "Raven_3DMakerPro_Scan.txt")
DET_FILE = os.path.join(DATASET_DIR, "detections_raw.json")

def triangulate_tags(projector, detections, t_rel, t_dur, t_xyz, t_rot, R0, R1, c_L, dt_sec):
    tag_rays = {}
    for d in detections:
        tid = d["tag_id"]
        t_slam_query = d["time_sec"] - dt_sec
        if t_slam_query < 0 or t_slam_query > t_dur:
            continue
        idx = np.argmin(np.abs(t_rel - t_slam_query))
        t_w_l = t_xyz[idx]
        R_w_l = t_rot[idx]

        u, v = d["center"][0], d["center"][1]
        ray_cam = projector.pixel_to_ray(u, v)

        R_cam = R0 if d["stream"] == 0 else R1
        ray_l = R_cam.T @ ray_cam
        orig_l = c_L

        ray_w = R_w_l @ ray_l
        orig_w = R_w_l @ orig_l + t_w_l

        tag_rays.setdefault(tid, []).append((orig_w, ray_w / np.linalg.norm(ray_w)))

    tag_world = {}
    for tid, rays in tag_rays.items():
        if len(rays) < 2:
            continue
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for orig, d_vec in rays:
            I_dd = np.eye(3) - np.outer(d_vec, d_vec)
            A += I_dd
            b += I_dd @ orig
        tag_world[tid] = np.linalg.lstsq(A, b, rcond=None)[0]
    return tag_world

def run_enhanced_calibration():
    output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" METODO ATUAL APRIMORADO (METODO 0+): KANNALA-BRANDT + GYRO SYNC")
    print(" Fundamentação: Seção 3.1 (Distorção Não-Linear de Lente Fisheye)")
    print("=" * 70)

    trj = np.loadtxt(TRJ_FILE)
    with open(DET_FILE, "r") as f:
        detections = json.load(f)

    # Rig base geometry
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

    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_dur = t_stamps[-1] - t_start
    t_rel = t_stamps - t_start
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    DT_GYRO = 5.7075
    print(f"[*] Sincronização temporal de alta precisão via IMU Giroscópio: dt = {DT_GYRO:.4f} s")

    # Models to compare:
    # 1. Linear Equidistant (Current Baseline)
    cam_linear = PolynomialFisheyeCamera.from_fov(width=3840, height=3840, fov_deg=196.0, k1=0.0, k2=0.0)
    # 2. Kannala-Brandt Polynomial (Enhanced with Research parameters)
    cam_enhanced = PolynomialFisheyeCamera.from_fov(width=3840, height=3840, fov_deg=196.0, k1=-0.04268, k2=0.00242)

    experiments = [
        ("Modelo Linear Equidistante (Baseline)", cam_linear),
        ("Modelo Polinomial Kannala-Brandt (Aprimorado)", cam_enhanced)
    ]

    exp_results = []

    for name, camera in experiments:
        print(f"\n[*] Executando Otimização de Bundle: {name}...")
        tag_world = triangulate_tags(camera, detections, t_rel, t_dur, t_xyz, t_rot, R_base0, R_base1, c_L, DT_GYRO)

        def residual_fun(params):
            p, y, r = params
            R_trim0 = Rot.from_euler("xyz", [p, y, r], degrees=True).as_matrix()
            R_trim1 = Rot.from_euler("xyz", [p, -y, -r], degrees=True).as_matrix()
            R0 = R_trim0 @ R_base0
            R1 = R_trim1 @ R_base1

            resids = []
            for d in detections:
                tid = d["tag_id"]
                if tid not in tag_world:
                    continue
                P_w = tag_world[tid]
                t_query = d["time_sec"] - DT_GYRO
                if t_query < 0 or t_query > t_dur:
                    continue
                idx = np.argmin(np.abs(t_rel - t_query))
                t_w_l = t_xyz[idx]
                R_w_l = t_rot[idx]

                u, v = d["center"][0], d["center"][1]
                ray_c = camera.pixel_to_ray(u, v)

                R_cam = R0 if d["stream"] == 0 else R1
                ray_l = R_cam.T @ ray_c
                orig_l = c_L

                ray_w = R_w_l @ ray_l
                ray_w = ray_w / np.linalg.norm(ray_w)
                orig_w = R_w_l @ orig_l + t_w_l

                diff = P_w - orig_w
                dist_vec = diff - np.dot(diff, ray_w) * ray_w
                resids.extend(dist_vec)

            return np.array(resids)

        opt = least_squares(residual_fun, [0.0, 0.0, 0.0], method="lm")
        p_opt, y_opt, r_opt = opt.x
        res_opt = residual_fun(opt.x)
        rms_err = np.sqrt(np.mean(res_opt**2))

        # Check reprojection of physical Picture Frame Corner on wall
        # In frame at t ~ 10s: LiDAR corner is at [6.5288, 0.4949, 0.5895]
        pt_frame_l = np.array([[6.5288, 0.4949, 0.5895]])
        R_trim0 = Rot.from_euler("xyz", [p_opt, y_opt, r_opt], degrees=True).as_matrix()
        R0_opt = R_trim0 @ R_base0
        pt_frame_c = (R0_opt @ (pt_frame_l - c_L).T).T
        u_proj, v_proj, _ = camera.project(pt_frame_c)

        # Ground truth target on image is v = 1918.0 px (moldura na foto)
        v_error_frame = float(v_proj[0] - 1918.0)
        u_error_frame = float(u_proj[0] - 1898.0)

        print(f"  [+] Trims Otimizados: Pitch = {p_opt:+.4f}°, Yaw = {y_opt:+.4f}°, Roll = {r_opt:+.4f}°")
        print(f"  [+] Erro Médio 3D Raio-Alvo: {rms_err * 1000:.1f} mm (RMSE)")
        print(f"  [+] Projeção da Moldura da Parede: u = {u_proj[0]:.1f} px, v = {v_proj[0]:.1f} px")
        print(f"  [+] Desvio Residual da Moldura: Delta_V = {v_error_frame:+.1f} px (Horizontal: Delta_U = {u_error_frame:+.1f} px)")

        exp_results.append({
            "model_name": name,
            "pitch_trim_deg": round(float(p_opt), 4),
            "yaw_trim_deg": round(float(y_opt), 4),
            "roll_trim_deg": round(float(r_opt), 4),
            "ray_distance_rmse_mm": round(float(rms_err * 1000.0), 2),
            "picture_frame_v_error_px": round(v_error_frame, 2),
            "picture_frame_u_error_px": round(u_error_frame, 2),
            "R0_matrix": R0_opt.tolist(),
            "lever_arm_c_L": c_L.tolist()
        })

    # Compare the impact of non-linear polynomial lens correction
    v_improvement = abs(exp_results[0]["picture_frame_v_error_px"]) - abs(exp_results[1]["picture_frame_v_error_px"])
    print("\n" + "=" * 70)
    print(" COMPARAÇÃO DE IMPACTO DO MODELO POLINOMIAL (SEÇÃO 3.1 DA PESQUISA)")
    print("=" * 70)
    print(f"Modelo Linear:     Delta V = {exp_results[0]['picture_frame_v_error_px']:+.1f} px")
    print(f"Modelo Polinomial: Delta V = {exp_results[1]['picture_frame_v_error_px']:+.1f} px")
    print(f"Melhoria direta no erro de distorção de borda: {v_improvement:.1f} pixels corrigidos pelo modelo ótico!")

    summary_file = output_dir / "enhanced_method0_results.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(exp_results, f, indent=4, ensure_ascii=False)
    print(f"[+] Relatório consolidado do Método 0 Aprimorado salvo em: {summary_file}")

if __name__ == "__main__":
    run_enhanced_calibration()
