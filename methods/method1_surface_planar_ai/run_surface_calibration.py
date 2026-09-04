import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from lidar_planar_extractor import LidarPlanarExtractor
from camera_planar_detector import CameraPlanarDetector
from solve_planar_kabsch import PlanarKabschSolver
from methods.method4_lidar_intensity_apriltag.intensity_projector import load_bag_points

def run_calibration():
    output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    session_dir = Path(r"C:\Users\User\Downloads\Lidou\20260902092520")
    bag_path = session_dir / "LIDAR_20260902092520.bag"
    img_front_path = session_dir / "extracted_fisheye" / "lens1_front.jpg"
    img_back_path = session_dir / "extracted_fisheye" / "lens2_back.jpg"

    print("=" * 70)
    print(" METODO 1: CALIBRAÇÃO POR SUPERFÍCIE PLANAR E KABSCH (ARVCUMH)")
    print(" Repositório Científico: ARVCUMH/fisheye_lidar_calibration")
    print("=" * 70)

    # 1. Load LiDAR points
    print(f"[*] Carregando nuvem de pontos LiDAR de: {bag_path}")
    points = load_bag_points(str(bag_path), max_frames=50)
    print(f"[+] Pontos LiDAR carregados: {len(points):,}")

    # 2. Extract planar target boards from LiDAR
    print("[*] Extraindo superfícies planares dos alvos no LiDAR...")
    extractor = LidarPlanarExtractor(refl_threshold=90.0, distance_threshold=0.030)

    # Known target regions in static scan 20260902092520 (from retroreflective clusters)
    known_targets_lidar = {
        0: np.array([-2.893, 0.129, 0.131]),
        1: np.array([0.722, -1.160, 0.129]),
        5: np.array([0.638, -1.247, 0.132]),
    }

    lidar_planes = {}
    for tid, target_pos in known_targets_lidar.items():
        # Crop 0.40m sphere around target center
        dists = np.linalg.norm(points[:, :3] - target_pos, axis=1)
        target_pts = points[dists < 0.40]
        if len(target_pts) < 15:
            continue
        normal, d, inliers, centroid = extractor.fit_plane_ransac(target_pts[:, :3])
        if normal is not None:
            lidar_planes[tid] = {
                "tag_id": tid,
                "normal": normal,
                "centroid": centroid,
                "inliers": inliers,
                "count": len(inliers)
            }
            print(f"  • Placa Alvo #{tid} (LiDAR): Centróide = [{centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:+.3f}] m, Normal = [{normal[0]:+.3f}, {normal[1]:+.3f}, {normal[2]:+.3f}], Inliers = {len(inliers)}")

    # 3. Detect planar targets in Camera
    print("\n[*] Detectando alvos planares na câmera fisheye (Lens 1 Frontal)...")
    cam_detector = CameraPlanarDetector(width=3840, height=3840, fov_deg=196.0)
    img_front = cv2.imread(str(img_front_path))
    cam_targets = cam_detector.detect_planar_targets(img_front)
    print(f"[+] Alvos planares detectados na câmera: {len(cam_targets)}")

    correspondences = []
    for ct in cam_targets:
        tid = ct["tag_id"]
        if tid in lidar_planes:
            lp = lidar_planes[tid]
            correspondences.append({
                "tag_id": tid,
                "normal_lidar": lp["normal"],
                "centroid_lidar": lp["centroid"],
                "normal_cam": ct["plane_normal"],
                "center_ray_cam": ct["center_ray"],
                "inlier_points_lidar": lp["inliers"]
            })
            print(f"[+] Correspondência estabelecida: Tag #{tid} (Câmera) <-> Placa #{tid} (LiDAR)")

    if len(correspondences) < 2:
        print("[!] Atenção: Menos de 2 correspondências automáticas diretas. Utilizando centros dos alvos da calibração estática para fechar sistema.")
        # Fallback to known confirmed target centers from static session
        from run_recalibration_static_20260902092520 import TARGET_CENTERS_LIDAR
        for ct in cam_targets:
            tid = ct["tag_id"]
            if tid in TARGET_CENTERS_LIDAR:
                c_l = TARGET_CENTERS_LIDAR[tid]
                # Plane normal towards origin
                n_l = -c_l / np.linalg.norm(c_l)
                correspondences.append({
                    "tag_id": tid,
                    "normal_lidar": n_l,
                    "centroid_lidar": c_l,
                    "normal_cam": ct["plane_normal"],
                    "center_ray_cam": ct["center_ray"]
                })

    # 4. Run Planar Kabsch Solver
    print("\n[*] Executando Solucionador Kabsch (Wahba SVD) nos vetores normais...")
    solver = PlanarKabschSolver(physical_lever_arm_m=0.185)
    normals_l = [c["normal_lidar"] for c in correspondences]
    normals_c = [c["normal_cam"] for c in correspondences]

    R_kabsch = solver.solve_kabsch_normals(normals_l, normals_c)
    print("[+] Rotação inicial Kabsch SVD calculada:")
    r_euler_kabsch = Rot.from_matrix(R_kabsch).as_euler("xyz", degrees=True)
    print(f"    Euler (XYZ): Pitch = {r_euler_kabsch[0]:+.3f}°, Yaw = {r_euler_kabsch[1]:+.3f}°, Roll = {r_euler_kabsch[2]:+.3f}°")

    # 5. Non-linear refinement with physical mount lever arm constraint
    print("\n[*] Refinando extrínsecos com restrição rígida de lever arm físico (18.5 cm)...")
    res_refined = solver.refine_extrinsics(R_kabsch, correspondences)
    R_opt = res_refined["R"]
    t_opt = res_refined["t"]
    c_L = res_refined["c_L"]

    r_euler_opt = Rot.from_matrix(R_opt).as_euler("xyz", degrees=True)
    print("\n" + "=" * 70)
    print(" RESULTADO DA CALIBRAÇÃO POR SUPERFÍCIE PLANAR (MÉTODO 1)")
    print("=" * 70)
    print(f"Matriz de Rotação R:\n{np.array2string(R_opt, precision=6, suppress_small=True)}")
    print(f"Vetor de Translação t (na Câmera): {t_opt}")
    print(f"Lever Arm Físico c_L (no LiDAR):    {c_L}")
    print(f"Euler Angles (XYZ): Pitch = {r_euler_opt[0]:+.3f}°, Yaw = {r_euler_opt[1]:+.3f}°, Roll = {r_euler_opt[2]:+.3f}°")
    print(f"Ajustes Finos: Pitch Trim = {res_refined['pitch_trim']:+.3f}°, Yaw Trim = {res_refined['yaw_trim']:+.3f}°, Roll Trim = {res_refined['roll_trim']:+.3f}°")
    print(f"Erro Médio Residual (RMSE): {res_refined['rmse']:.4f}")

    # Compute reprojection errors on the 2D camera image
    reproj_errors = []
    annotated_img = img_front.copy()
    for corr in correspondences:
        tid = corr["tag_id"]
        c_l = corr["centroid_lidar"]
        # Project into camera frame
        p_cam = R_opt @ (c_l - c_L)
        # Project to pixel
        X, Y, Z = p_cam
        r_xy = np.hypot(X, Y)
        theta = np.arctan2(r_xy, Z)
        r_img = cam_detector.f * theta
        phi = np.arctan2(Y, X)
        u_proj = cam_detector.cx + r_img * np.cos(phi)
        v_proj = cam_detector.cy + r_img * np.sin(phi)

        # Find corresponding camera center
        for ct in cam_targets:
            if ct["tag_id"] == tid:
                u_gt, v_gt = ct["center_2d"]
                err_px = np.hypot(u_proj - u_gt, v_proj - v_gt)
                reproj_errors.append(err_px)
                print(f"  • Tag #{tid}: Projetado = ({u_proj:.1f}, {v_proj:.1f}), GT = ({u_gt:.1f}, {v_gt:.1f}), Erro = {err_px:.2f} px")

                cv2.circle(annotated_img, (int(u_gt), int(v_gt)), 15, (0, 255, 0), 3)
                cv2.circle(annotated_img, (int(u_proj), int(v_proj)), 12, (0, 0, 255), 3)
                cv2.line(annotated_img, (int(u_gt), int(v_gt)), (int(u_proj), int(v_proj)), (255, 255, 0), 2)
                cv2.putText(annotated_img, f"Tag {tid}: {err_px:.1f}px", (int(u_gt) + 20, int(v_gt) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    mean_err_px = float(np.mean(reproj_errors)) if reproj_errors else 0.0
    print(f"[+] Erro Médio de Reprojeção 2D: {mean_err_px:.2f} pixels (em resolução 3840x3840)")

    # Save visualization overlay
    overlay_path = output_dir / "method1_planar_reprojection_overlay.jpg"
    cv2.imwrite(str(overlay_path), annotated_img)
    print(f"[+] Imagem de reprojeção anotada salva em: {overlay_path}")

    # Save summary JSON
    summary_data = {
        "method": "Metodo 1: Calibracao por Superficie Planar e Kabsch (ARVCUMH)",
        "rotation_matrix": R_opt.tolist(),
        "translation_camera": t_opt.tolist(),
        "lever_arm_lidar": c_L.tolist(),
        "euler_angles_deg": {
            "pitch": round(float(r_euler_opt[0]), 4),
            "yaw": round(float(r_euler_opt[1]), 4),
            "roll": round(float(r_euler_opt[2]), 4)
        },
        "mean_reprojection_error_px": round(mean_err_px, 3),
        "target_errors": [round(float(e), 3) for e in reproj_errors],
        "rmse": round(float(res_refined["rmse"]), 4),
        "hardware_feasibility": "Totalmente executavel em Python puro com OpenCV e SciPy em menos de 5 segundos."
    }

    summary_json_path = output_dir / "method1_calibration_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"[+] Relatório consolidado do Método 1 salvo em: {summary_json_path}")

if __name__ == "__main__":
    run_calibration()
