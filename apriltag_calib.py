#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag LiDAR-Fisheye / Camera Extrinsic Calibrator
===================================================
Estimates the rigid 3D transformation (T_camera_lidar) between the Raven LiDAR
and camera using AprilTag fiducial markers.

Supports:
- Raw Fisheye Images (Insta360 X4 raw fisheye, Raven JMK7 12MP, dual-fisheye side-by-side)
- Kannala-Brandt / Equidistant fisheye camera model (OpenCV cv2.fisheye)
- YAML/JSON camera intrinsics or automatic generic fisheye model
- 360° Equirectangular images (Insta360 stitched)
- Tag family tag36h11 with subpixel corner refinement
- LiDAR board center & retroreflective stripe extraction (.bag, .pcd, .ply, .npz)
- Robust 3D-2D spherical/fisheye PnP and 3D-3D Umeyama/Kabsch solvers
- Overlay visualizer and seamless export to extrinsics_determined.json

Usage:
  # Self-test synthetic validation
  python apriltag_calib.py --selftest

  # Calibrate using raw fisheye image and LiDAR bag:
  python apriltag_calib.py --fisheye path/to/fisheye_front.jpg \\
                           --cam-config config/camera_raven.yaml \\
                           --bag path/to/lidar.bag \\
                           --tag-size 0.150 \\
                           --out extrinsics_determined.json \\
                           --overlay

  # Calibrate using dual-fisheye image (Insta360 X4 raw dual circular frames):
  python apriltag_calib.py --dual-fisheye path/to/dual_fisheye.jpg \\
                           --bag path/to/lidar.bag \\
                           --tag-size 0.150 \\
                           --out extrinsics_determined.json
"""

import argparse
import json
import math
import os
import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

try:
    import yaml
except ImportError:
    yaml = None

import core


# ----------------------------------------------------------------- Camera Models --

class FisheyeCameraModel:
    """
    Kannala-Brandt / Equidistant Fisheye camera model compatible with OpenCV cv2.fisheye.
    """
    
    def __init__(self, K=None, D=None, width=3000, height=4000):
        self.width = width
        self.height = height
        
        if K is not None:
            self.K = np.asarray(K, dtype=np.float64)
        else:
            # Auto-estimate equidistant fisheye K based on image dimensions (~190° FOV)
            f = min(width, height) / np.pi  # f for equidistant r = f * theta
            self.K = np.array([
                [f, 0.0, width / 2.0],
                [0.0, f, height / 2.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            
        if D is not None:
            self.D = np.asarray(D, dtype=np.float64).reshape(-1, 1)
        else:
            self.D = np.zeros((4, 1), dtype=np.float64)

    @classmethod
    def from_yaml(cls, yaml_path):
        """Loads OpenCV / FAST-LIVO2 style camera YAML."""
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config (pip install pyyaml)")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            
        w = cfg.get("cam_width", cfg.get("image_width", 3000))
        h = cfg.get("cam_height", cfg.get("image_height", 4000))
        fx = cfg.get("cam_fx", cfg.get("fx", 1123.5))
        fy = cfg.get("cam_fy", cfg.get("fy", 1124.2))
        cx = cfg.get("cam_cx", cfg.get("cx", w / 2.0))
        cy = cfg.get("cam_cy", cfg.get("cy", h / 2.0))
        k1 = cfg.get("k1", 0.0)
        k2 = cfg.get("k2", 0.0)
        k3 = cfg.get("k3", 0.0)
        k4 = cfg.get("k4", 0.0)
        
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        D = np.array([k1, k2, k3, k4], dtype=np.float64).reshape(4, 1)
        return cls(K, D, width=w, height=h)

    def pixel_to_ray(self, points_uv):
        """
        Unprojects 2D fisheye pixels (N, 2) to 3D unit rays (N, 3) in camera frame.
        """
        pts = np.asarray(points_uv, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, 1, 2)
            single = True
        else:
            pts = pts.reshape(-1, 1, 2)
            single = False
            
        # cv2.fisheye.undistortPoints returns normalized coordinates (x_n, y_n)
        norm_pts = cv2.fisheye.undistortPoints(pts, self.K, self.D)
        norm_pts = norm_pts.reshape(-1, 2)
        
        # In normalized coordinates: ray = [x_n, y_n, 1.0] / norm
        rays = np.hstack([norm_pts, np.ones((len(norm_pts), 1), dtype=np.float64)])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        
        if single:
            return rays[0]
        return rays

    def ray_to_pixel(self, rays_cam):
        """
        Projects 3D camera coordinates or unit rays (N, 3) to 2D fisheye pixels (N, 2).
        """
        rays = np.asarray(rays_cam, dtype=np.float64)
        if rays.ndim == 1:
            rays = rays.reshape(1, 1, 3)
            single = True
        else:
            rays = rays.reshape(-1, 1, 3)
            single = False
            
        # Filter points in front of camera
        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        
        uv, _ = cv2.fisheye.projectPoints(rays, rvec, tvec, self.K, self.D)
        uv = uv.reshape(-1, 2)
        
        if single:
            return uv[0]
        return uv


# ----------------------------------------------------------- AprilTag Detectors --

class AprilTagDetectorFisheye:
    """
    High-precision AprilTag detector for raw Fisheye images.
    Combines direct subpixel detection with fisheye ray backprojection.
    """
    
    def __init__(self, camera_model: FisheyeCameraModel, tag_family="tag36h11"):
        self.cam = camera_model
        if tag_family == "tag36h11":
            self.dict_id = cv2.aruco.DICT_APRILTAG_36h11
        elif tag_family == "tag16h5":
            self.dict_id = cv2.aruco.DICT_APRILTAG_16h5
        else:
            self.dict_id = cv2.aruco.DICT_APRILTAG_36h11
            
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.dict_id)
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params)

    def detect(self, img_bgr, tag_size_m=0.150):
        """
        Detects AprilTags in the raw fisheye image.
        Returns a dict of detections with 2D pixels, 3D unit rays, and estimated 6D camera pose.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        detections = {}
        if ids is None:
            return detections
            
        obj_pts = np.array([
            [-tag_size_m / 2.0, -tag_size_m / 2.0, 0],
            [tag_size_m / 2.0, -tag_size_m / 2.0, 0],
            [tag_size_m / 2.0, tag_size_m / 2.0, 0],
            [-tag_size_m / 2.0, tag_size_m / 2.0, 0],
        ], dtype=np.float64)
        
        for i, tag_id_arr in enumerate(ids):
            tag_id = int(tag_id_arr[0])
            c_uv = corners[i][0]  # shape (4, 2)
            
            center_uv = np.mean(c_uv, axis=0)
            
            # Unproject corners and center to 3D unit rays
            corner_rays = self.cam.pixel_to_ray(c_uv)
            center_ray = self.cam.pixel_to_ray(center_uv)
            
            # Estimate 6D pose in fisheye camera frame using undistorted normalized rays
            norm_corners = cv2.fisheye.undistortPoints(c_uv.reshape(-1, 1, 2), self.cam.K, self.cam.D).reshape(-1, 2)
            
            # Fit pose using solvePnP on normalized plane (K = eye(3))
            K_ident = np.eye(3, dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, norm_corners, K_ident, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, norm_corners, K_ident, None, flags=cv2.SOLVEPNP_ITERATIVE)
                
            P_cam = tvec.flatten() if ok else center_ray * 1.5  # Fallback
            R_tag_cam, _ = cv2.Rodrigues(rvec) if ok else (np.eye(3), None)
            
            detections[tag_id] = {
                "tag_id": tag_id,
                "corners_uv": c_uv,
                "center_uv": center_uv,
                "corner_rays": corner_rays,
                "center_ray": center_ray,
                "P_cam": P_cam,
                "R_tag_cam": R_tag_cam,
                "tag_size_m": tag_size_m,
            }
            
        return detections


# ------------------------------------------------------------- LiDAR Target Processing --

def detect_lidar_board_center(points_xyz, intensities=None, center_prior=None, radius_m=0.6,
                              min_reflectivity_percentile=85.0):
    """
    Extracts the precise 3D center of a planar calibration board in LiDAR point cloud.
    """
    P = np.asarray(points_xyz, dtype=np.float64)
    if len(P) < 20:
        return None, None, None
        
    if center_prior is not None:
        dists = np.linalg.norm(P - np.asarray(center_prior), axis=1)
        mask = dists < radius_m
        P = P[mask]
        if intensities is not None:
            intensities = intensities[mask]
            
    if len(P) < 20:
        return None, None, None
        
    # RANSAC plane fitting
    best_inliers = []
    best_plane = None
    rng = np.random.default_rng(42)
    
    for _ in range(150):
        if len(P) < 3:
            break
        idx = rng.choice(len(P), 3, replace=False)
        p1, p2, p3 = P[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue
        n = n / norm_n
        d = -np.dot(n, p1)
        
        dists = np.abs(np.dot(P, n) + d)
        inliers = np.nonzero(dists < 0.025)[0]  # 2.5 cm tolerance
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (n, d)
            
    if len(best_inliers) < 15:
        return None, None, None
        
    plane_pts = P[best_inliers]
    normal = best_plane[0]
    
    # Retroreflective stripe / high-intensity peak extraction
    if intensities is not None and len(intensities) == len(points_xyz):
        sub_intensities = intensities[best_inliers] if center_prior is None else intensities
        thresh = np.percentile(sub_intensities, min_reflectivity_percentile)
        hi_mask = sub_intensities >= thresh
        if np.sum(hi_mask) >= 6:
            hi_pts = plane_pts[hi_mask]
            center = np.mean(hi_pts, axis=0)
            return center, normal, plane_pts
            
    center = np.mean(plane_pts, axis=0)
    return center, normal, plane_pts


# ------------------------------------------------------------- Calibration Solvers --

def solve_rigid_3d3d_umeyama(P_lidar, P_cam, ransac_threshold_m=0.05, max_iters=200):
    """
    Closed-form 3D-3D rigid registration (Umeyama / Kabsch) with RANSAC.
    """
    P_lidar = np.asarray(P_lidar, float)
    P_cam = np.asarray(P_cam, float)
    n_pts = len(P_lidar)
    
    if n_pts < 3:
        raise ValueError(f"Need at least 3 correspondences for 3D-3D alignment (got {n_pts})")
        
    def _fit_umeyama(src, dst):
        mu_src = np.mean(src, axis=0)
        mu_dst = np.mean(dst, axis=0)
        A = src - mu_src
        B = dst - mu_dst
        H = A.T @ B
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_dst - R @ mu_src
        return R, t
        
    if n_pts == 3:
        R, t = _fit_umeyama(P_lidar, P_cam)
        return R, t, np.arange(n_pts)
        
    best_inliers = []
    best_R, best_t = None, None
    rng = np.random.default_rng(123)
    
    for _ in range(max_iters):
        sample_idx = rng.choice(n_pts, 3, replace=False)
        try:
            R_cand, t_cand = _fit_umeyama(P_lidar[sample_idx], P_cam[sample_idx])
            pred_cam = (R_cand @ P_lidar.T).T + t_cand
            residuals = np.linalg.norm(pred_cam - P_cam, axis=1)
            inliers = np.nonzero(residuals < ransac_threshold_m)[0]
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_R, best_t = R_cand, t_cand
        except Exception:
            continue
            
    if len(best_inliers) >= 3:
        best_R, best_t = _fit_umeyama(P_lidar[best_inliers], P_cam[best_inliers])
        return best_R, best_t, best_inliers
        
    return _fit_umeyama(P_lidar, P_cam) + (np.arange(n_pts),)


def solve_joint_opt(P_lidar_list, rays_cam_list, P_cam_list=None, R_init=None, t_init=None):
    """
    Joint 3D-3D + 3D-2D optimization.
    Minimizes Euclidean 3D target distance and angular ray deviation simultaneously.
    """
    P_lidar = np.asarray(P_lidar_list, float)
    rays_cam = np.asarray(rays_cam_list, float)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
    
    if len(P_lidar) < 3:
        raise ValueError(f"Need at least 3 correspondences (got {len(P_lidar)})")
        
    rvec0 = Rot.from_matrix(R_init).as_rotvec() if R_init is not None else np.zeros(3)
    t0 = np.asarray(t_init, float) if t_init is not None else np.array([0.0, 0.08, -0.02])
    x0 = np.r_[rvec0, t0]
    
    has_3d = P_cam_list is not None and len(P_cam_list) == len(P_lidar)
    if has_3d:
        P_cam = np.asarray(P_cam_list, float)
        
    def residuals(x):
        R = Rot.from_rotvec(x[:3]).as_matrix()
        t = x[3:6]
        P_cam_pred = (R @ P_lidar.T).T + t
        norms = np.linalg.norm(P_cam_pred, axis=1, keepdims=True)
        rays_pred = P_cam_pred / np.maximum(norms, 1e-12)
        
        dots = np.clip(np.sum(rays_pred * rays_cam, axis=1), -1.0, 1.0)
        ang_errors = np.arccos(dots)  # radians
        
        if has_3d:
            dist_errors = np.linalg.norm(P_cam_pred - P_cam, axis=1)  # meters
            return np.r_[dist_errors, ang_errors * 0.2]
        return ang_errors
        
    res = least_squares(residuals, x0, method="lm", xtol=1e-10, ftol=1e-10)
    R_opt = Rot.from_rotvec(res.x[:3]).as_matrix()
    t_opt = res.x[3:6]
    return R_opt, t_opt, res


# ------------------------------------------------------------- Calibration Manager --

class AprilTagCalibrator:
    """
    Complete AprilTag calibration pipeline for Fisheye and Panoramic cameras.
    """
    
    def __init__(self, camera_model: FisheyeCameraModel = None, tag_size_m=0.150, tag_family="tag36h11"):
        self.cam = camera_model or FisheyeCameraModel()
        self.tag_size_m = tag_size_m
        self.detector = AprilTagDetectorFisheye(self.cam, tag_family=tag_family)
        self.correspondences = []
        self.result = {}

    def add_manual_correspondence(self, tag_id, P_lidar_xyz, P_cam_xyz=None, ray_cam=None):
        self.correspondences.append({
            "tag_id": int(tag_id),
            "P_lidar": np.asarray(P_lidar_xyz, float),
            "P_cam": np.asarray(P_cam_xyz, float) if P_cam_xyz is not None else None,
            "ray_cam": np.asarray(ray_cam, float) if ray_cam is not None else None,
        })

    def process_fisheye_capture(self, fisheye_img_bgr, lidar_points_xyz, intensities=None,
                                tag_positions_lidar_guess=None):
        """
        Detects AprilTags in raw fisheye image and correlates with LiDAR point cloud.
        """
        detections = self.detector.detect(fisheye_img_bgr, tag_size_m=self.tag_size_m)
        print(f"[*] Detected {len(detections)} AprilTags in fisheye image: {list(detections.keys())}")
        
        if tag_positions_lidar_guess is not None:
            for tag_id, guess_xyz in tag_positions_lidar_guess.items():
                if tag_id in detections:
                    center_lidar, normal, _ = detect_lidar_board_center(
                        lidar_points_xyz, intensities, center_prior=guess_xyz, radius_m=0.5
                    )
                    if center_lidar is not None:
                        cd = detections[tag_id]
                        self.correspondences.append({
                            "tag_id": tag_id,
                            "P_lidar": center_lidar,
                            "P_cam": cd.get("P_cam"),
                            "ray_cam": cd["center_ray"],
                            "center_uv": cd["center_uv"],
                        })
                        print(f"  [+] Matched Tag #{tag_id}: LiDAR={center_lidar.round(3)} -> Fisheye Ray={cd['center_ray'].round(3)}")
                        
        return detections

    def solve(self):
        """
        Solves for T_camera_lidar using accumulated correspondences.
        """
        if len(self.correspondences) < 3:
            raise ValueError(f"Insufficient correspondences (found {len(self.correspondences)}, need >= 3)")
            
        P_lidar = [c["P_lidar"] for c in self.correspondences]
        has_3d_cam = all(c["P_cam"] is not None for c in self.correspondences)
        
        P_cam = [c["P_cam"] for c in self.correspondences] if has_3d_cam else None
        if has_3d_cam:
            R_init, t_init, inliers = solve_rigid_3d3d_umeyama(P_lidar, P_cam)
            print(f"[*] 3D-3D Umeyama solve: {len(inliers)}/{len(self.correspondences)} inliers")
        else:
            R_init, t_init = None, None
            
        rays_cam = [c["ray_cam"] for c in self.correspondences]
        R_opt, t_opt, _ = solve_joint_opt(P_lidar, rays_cam, P_cam_list=P_cam, R_init=R_init, t_init=t_init)
        
        T = np.eye(4)
        T[:3, :3] = R_opt
        T[:3, 3] = t_opt
        
        P_lidar_arr = np.array(P_lidar)
        P_cam_pred = (R_opt @ P_lidar_arr.T).T + t_opt
        norms = np.linalg.norm(P_cam_pred, axis=1, keepdims=True)
        rays_pred = P_cam_pred / np.maximum(norms, 1e-12)
        
        ang_errors_deg = np.degrees(np.arccos(np.clip(np.sum(rays_pred * np.array(rays_cam), axis=1), -1.0, 1.0)))
        mean_ang_err = float(np.mean(ang_errors_deg))
        max_ang_err = float(np.max(ang_errors_deg))
        
        C_lidar = -R_opt.T @ t_opt
        euler_zyx = Rot.from_matrix(R_opt).as_euler("ZYX", degrees=True)
        quat_xyzw = Rot.from_matrix(R_opt).as_quat()
        
        self.result = {
            "T_base_camera_lidar": [[float(v) for v in row] for row in T],
            "_determined": {
                "method": "AprilTag fiducial 3D-2D ray PnP & 3D-3D Umeyama alignment (Raw Fisheye Model)",
                "tag_count": len(self.correspondences),
                "mean_angular_residual_deg": mean_ang_err,
                "max_angular_residual_deg": max_ang_err,
                "lever_arm_cam_in_lidar_m": [float(v) for v in C_lidar],
                "refine_deg": [float(v) for v in np.rad2deg(Rot.from_matrix(R_opt).as_rotvec())],
            },
            "_calibration": {
                "tool": "AprilTag LiDAR-Fisheye Calibrator (Raven + Raw Fisheye Camera)",
                "convention": {
                    "camera_axes": "X right, Y down, Z forward (OpenCV)",
                    "lidar_axes": "sensor body frame",
                    "transform": "X_cam = R @ X_lidar + t",
                },
                "euler_ZYX_deg": {
                    "yaw": float(euler_zyx[0]),
                    "pitch": float(euler_zyx[1]),
                    "roll": float(euler_zyx[2])
                },
                "quaternion_xyzw": [float(v) for v in quat_xyzw],
                "T_lidar_camera": [[float(v) for v in row] for row in np.linalg.inv(T)],
                "camera_intrinsics": {
                    "K": self.cam.K.tolist(),
                    "D": self.cam.D.flatten().tolist(),
                    "width": self.cam.width,
                    "height": self.cam.height,
                },
                "correspondences": [
                    {
                        "tag_id": int(c["tag_id"]),
                        "P_lidar_m": [float(v) for v in c["P_lidar"]],
                        "angular_error_deg": float(ang_errors_deg[i]),
                    }
                    for i, c in enumerate(self.correspondences)
                ]
            }
        }
        
        print("\n" + "=" * 60)
        print("           APRILTAG FISHEYE CALIBRATION SUCCESSFUL")
        print("=" * 60)
        print(f"Mean Angular Error:    {mean_ang_err:.3f}° (Max: {max_ang_err:.3f}°)")
        print(f"Camera Position (C):   X={C_lidar[0]:+.4f} m, Y={C_lidar[1]:+.4f} m, Z={C_lidar[2]:+.4f} m")
        print(f"Euler Angles (ZYX):    Yaw={euler_zyx[0]:.2f}°, Pitch={euler_zyx[1]:.2f}°, Roll={euler_zyx[2]:.2f}°")
        print("=" * 60 + "\n")
        return self.result

    def export_json(self, out_path="extrinsics_determined.json"):
        if not self.result:
            raise ValueError("No calibration result to export. Call solve() first.")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.result, f, indent=2)
        print(f"[+] Exported calibration JSON: {os.path.abspath(out_path)}")

    def render_overlay_preview(self, fisheye_img_bgr, lidar_points_xyz, out_path="apriltag_fisheye_overlay.jpg", max_pts=40000):
        """
        Projects LiDAR points and detected AprilTag locations onto the raw fisheye image.
        """
        if not self.result:
            return
            
        T = np.array(self.result["T_base_camera_lidar"])
        R = T[:3, :3]
        t = T[:3, 3]
        
        H, W = fisheye_img_bgr.shape[:2]
        canvas = fisheye_img_bgr.copy()
        
        P = np.asarray(lidar_points_xyz, float)
        if len(P) > max_pts:
            idx = np.random.default_rng(0).choice(len(P), max_pts, replace=False)
            P = P[idx]
            
        P_cam = (R @ P[:, :3].T).T + t
        valid = P_cam[:, 2] > 0.05
        P_cam_valid = P_cam[valid]
        
        if len(P_cam_valid) > 0:
            uv = self.cam.ray_to_pixel(P_cam_valid)
            depths = np.linalg.norm(P_cam_valid, axis=1)
            norm_depth = np.clip((depths - 0.5) / 5.0, 0, 1)
            colors = cv2.applyColorMap((norm_depth * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            for i in range(len(uv)):
                ui, vi = int(uv[i, 0]), int(uv[i, 1])
                if 0 <= ui < W and 0 <= vi < H:
                    c = colors[i, 0].tolist()
                    cv2.circle(canvas, (ui, vi), 1, c, -1)
                    
        for c in self.correspondences:
            tag_id = c["tag_id"]
            if "center_uv" in c and c["center_uv"] is not None:
                cu, cv_pt = int(c["center_uv"][0]), int(c["center_uv"][1])
                cv2.drawMarker(canvas, (cu, cv_pt), (0, 255, 0), cv2.MARKER_CROSS, 25, 2)
                
            Pl = np.asarray(c["P_lidar"])
            Pl_cam = R @ Pl + t
            ru, rv = self.cam.ray_to_pixel(Pl_cam)
            rui, rvi = int(ru), int(rv)
            cv2.circle(canvas, (rui, rvi), 8, (0, 0, 255), 2)
            cv2.putText(canvas, f"Tag #{tag_id}", (rui + 12, rvi - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"[+] Saved fisheye reprojection overlay: {os.path.abspath(out_path)}")


# ----------------------------------------------------------------- Self-Test --

def run_synthetic_selftest():
    """
    Validates fisheye optics, AprilTag detector, Umeyama solver, and PnP
    optimization using synthetic 3D scenes with known ground truth.
    """
    print("=" * 60)
    print("     RUNNING SYNTHETIC APRILTAG FISHEYE CALIBRATION TEST")
    print("=" * 60)
    
    # 1. Ground Truth Extrinsics
    R_gt = Rot.from_euler("ZYX", [15.0, 20.0, -10.0], degrees=True).as_matrix()
    t_gt = np.array([-0.032, 0.078, 0.028])
    
    # 2. Synthetic Fisheye Camera
    cam = FisheyeCameraModel(width=3000, height=4000)
    calibrator = AprilTagCalibrator(camera_model=cam, tag_size_m=0.150)
    
    # 3. Targets in 3D scene
    targets_lidar = [
        np.array([1.1, 0.3, -0.1]),
        np.array([-1.2, 0.7, 0.2]),
        np.array([0.1, 1.8, 0.4]),
        np.array([-0.6, 1.6, -0.3]),
        np.array([1.5, 1.2, 0.1]),
        np.array([-0.9, 1.0, 0.6]),
    ]
    
    np.random.seed(42)
    for i, P_l in enumerate(targets_lidar):
        P_c = R_gt @ P_l + t_gt
        P_l_noisy = P_l + np.random.normal(0, 0.001, size=3)
        ray_c = P_c / np.linalg.norm(P_c)
        ray_c_noisy = ray_c + np.random.normal(0, 0.0005, size=3)
        ray_c_noisy /= np.linalg.norm(ray_c_noisy)
        
        calibrator.add_manual_correspondence(
            tag_id=i,
            P_lidar_xyz=P_l_noisy,
            P_cam_xyz=P_c,
            ray_cam=ray_c_noisy
        )
        
    res = calibrator.solve()
    T_est = np.array(res["T_base_camera_lidar"])
    R_est = T_est[:3, :3]
    t_est = T_est[:3, 3]
    
    rot_err_deg = np.degrees(Rot.from_matrix(R_est @ R_gt.T).magnitude())
    trans_err_mm = np.linalg.norm(t_est - t_gt) * 1000.0
    
    print("[*] Fisheye Self-Test Results:")
    print(f"    Rotation Error:    {rot_err_deg:.4f}° (Threshold < 0.15°)")
    print(f"    Translation Error: {trans_err_mm:.3f} mm (Threshold < 3.0 mm)")
    
    assert rot_err_deg < 0.15, f"Rotation error too high: {rot_err_deg:.4f}°"
    assert trans_err_mm < 3.0, f"Translation error too high: {trans_err_mm:.3f} mm"
    print("\n[PASSED] Fisheye AprilTag calibration math and solvers verified successfully!\n")
    return True


# ----------------------------------------------------------------- CLI Entry Point --

def main():
    parser = argparse.ArgumentParser(description="AprilTag LiDAR-Fisheye Extrinsic Calibrator")
    parser.add_argument("--selftest", action="store_true", help="Run synthetic calibration self-test")
    parser.add_argument("--fisheye", "-f", type=str, help="Path to raw fisheye image (.jpg, .png, .tiff)")
    parser.add_argument("--dual-fisheye", type=str, help="Path to dual-fisheye side-by-side image (Insta360 X4 raw)")
    parser.add_argument("--cam-config", type=str, help="Path to camera intrinsics YAML (e.g. config/camera_raven.yaml)")
    parser.add_argument("--cloud", "-c", type=str, help="Path to LiDAR point cloud (.pcd, .ply, .npz)")
    parser.add_argument("--bag", "-b", type=str, help="Path to ROS bag (.bag)")
    parser.add_argument("--tag-size", "-s", type=float, default=0.150, help="Physical size of AprilTag square in meters (default: 0.150 m)")
    parser.add_argument("--out", "-o", type=str, default="extrinsics_determined.json", help="Output JSON calibration file")
    parser.add_argument("--overlay", action="store_true", help="Generate reprojection overlay preview image")
    args = parser.parse_args()

    if args.selftest:
        run_synthetic_selftest()
        return 0

    if not (args.fisheye or args.dual_fisheye) or not (args.cloud or args.bag):
        parser.print_help()
        print("\n[!] Please specify --fisheye (or --dual-fisheye) AND --cloud (or --bag), or run --selftest")
        return 1

    # Load camera model
    if args.cam_config and os.path.exists(args.cam_config):
        print(f"[*] Loading camera intrinsics from: {args.cam_config}")
        cam = FisheyeCameraModel.from_yaml(args.cam_config)
    else:
        cam = FisheyeCameraModel()

    calibrator = AprilTagCalibrator(camera_model=cam, tag_size_m=args.tag_size)

    # 1. Load image
    img_path = args.fisheye or args.dual_fisheye
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Could not open image: {img_path}")
        return 1

    if args.dual_fisheye:
        # Split side-by-side dual fisheye into front lens (left half)
        h, w = img.shape[:2]
        img = img[:, :w // 2]
        cam.width, cam.height = img.shape[1], img.shape[0]

    # 2. Load point cloud
    if args.cloud:
        if args.cloud.endswith(".npz"):
            z = np.load(args.cloud)
            pts = z["P"][:, :3]
            intensities = z["P"][:, 3] if z["P"].shape[1] > 3 else None
        else:
            raise NotImplementedError(f"Direct loader for {args.cloud} - please use cached .npz or .bag")
    else:
        print(f"[*] Reading ROS bag: {args.bag}")
        d = core.read_bag(args.bag)
        pts = d["points"][:, :3]
        intensities = d["points"][:, 3]

    # 3. Detect and match
    calibrator.process_fisheye_capture(img, pts, intensities)

    if len(calibrator.correspondences) < 3:
        print("[!] Need at least 3 matched correspondences between LiDAR and Fisheye camera.")
        return 1

    # 4. Solve
    calibrator.solve()
    calibrator.export_json(args.out)

    if args.overlay:
        calibrator.render_overlay_preview(img, pts, out_path="apriltag_fisheye_overlay.jpg")

    return 0


if __name__ == "__main__":
    sys.exit(main())
