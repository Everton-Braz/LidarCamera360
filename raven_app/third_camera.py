"""Third camera / auxiliary video and image source integration for RavenCalibrator.

Supports smartphone or secondary camera integration to improve point cloud
colorization and 3D Gaussian Splatting (3DGS) multi-view coverage.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from raven_app.video import compute_laplacian_sharpness, frame_time


@dataclass
class ThirdCameraConfig:
    """Configuration for auxiliary image/video source (e.g. frontal smartphone)."""

    camera_name: str = "cam2"
    source_path: str = ""
    source_type: str = "video"  # "video" or "folder"
    enabled: bool = True
    fps: float = 2.0
    time_offset_s: float = 0.0
    camera_model: str = "PINHOLE"  # "PINHOLE", "OPENCV", "OPENCV_FISHEYE"
    intrinsics: Dict[str, float] = field(
        default_factory=lambda: {
            "width": 1920.0,
            "height": 1080.0,
            "fx": 1536.0,
            "fy": 1536.0,
            "cx": 960.0,
            "cy": 540.0,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
        }
    )
    T_lidar_to_cam2_rigid_4x4: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.15],
            [0.0, 0.0, 1.0, 0.20],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    description: str = "Frontal smartphone camera"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ThirdCameraConfig:
        clean = {}
        for k in (
            "camera_name",
            "source_path",
            "source_type",
            "enabled",
            "fps",
            "time_offset_s",
            "camera_model",
            "intrinsics",
            "T_lidar_to_cam2_rigid_4x4",
            "description",
        ):
            if k in data:
                clean[k] = data[k]
        return cls(**clean)


def default_smartphone_intrinsics(width: int = 1920, height: int = 1080) -> Dict[str, float]:
    """Return standard rectilinear pinhole intrinsics for a typical smartphone camera (~68° HFOV)."""
    fx = float(width) * 0.80
    fy = float(width) * 0.80
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    return {
        "width": float(width),
        "height": float(height),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }


def load_third_camera_config(path_or_dict: Union[str, Path, Dict[str, Any]]) -> ThirdCameraConfig:
    """Load ThirdCameraConfig from a file path or dictionary."""
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(path_or_dict, dict):
        data = path_or_dict
    else:
        raise TypeError(f"Expected path or dict, got {type(path_or_dict)}")
    return ThirdCameraConfig.from_dict(data)


def save_third_camera_config(config: ThirdCameraConfig, target_path: Union[str, Path]) -> Path:
    """Persist ThirdCameraConfig to a JSON file."""
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return p


def detect_source_metadata(source_path: Union[str, Path]) -> Dict[str, Any]:
    """Inspect a video file or image folder and return dimensions and count."""
    p = Path(source_path)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {p}")

    if p.is_file():
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {p}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = float(count / fps) if fps > 0 else 0.0
        cap.release()
        return {
            "type": "video",
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": count,
            "duration_s": dur,
            "path": str(p.resolve()),
        }

    if p.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        images = sorted(
            [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )
        if not images:
            raise ValueError(f"No image files found in folder: {p}")
        sample = cv2.imread(str(images[0]))
        if sample is None:
            raise ValueError(f"Cannot read sample image: {images[0]}")
        h, w = sample.shape[:2]
        return {
            "type": "folder",
            "width": int(w),
            "height": int(h),
            "image_count": len(images),
            "images": [str(img.resolve()) for img in images],
            "path": str(p.resolve()),
        }

    raise ValueError(f"Invalid source path: {p}")


def extract_third_camera_frames(
    config: ThirdCameraConfig,
    dataset_dir: Union[str, Path],
    target_fps: Optional[float] = None,
    sharpness_window: int = 3,
    jpeg_quality: int = 95,
) -> Tuple[List[Path], Dict[str, float]]:
    """Extract frames from third camera video or directory into dataset/images/cam2.

    Returns:
        (extracted_paths, timestamp_dict) where timestamp_dict maps relative name
        (e.g. 'cam2/frame_000001.jpg') to video PTS in seconds.
    """
    dataset_dir = Path(dataset_dir)
    images_root = dataset_dir / "images"
    cam_dir = images_root / config.camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    fps = float(target_fps or config.fps or 2.0)
    source_p = Path(config.source_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Third camera source does not exist: {source_p}")

    extracted: List[Path] = []
    timestamps: Dict[str, float] = {}

    if config.source_type == "video" or source_p.is_file():
        cap = cv2.VideoCapture(str(source_p))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source_p}")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step_frames = max(1, int(round(native_fps / fps)))

        frame_idx = 0
        saved_idx = 1
        window_buf = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if sharpness_window > 1:
                sharpness = compute_laplacian_sharpness(frame)
                window_buf.append((sharpness, frame.copy(), frame_idx))
                if len(window_buf) >= sharpness_window:
                    # Pick sharpest in buffer
                    best_sharpness, best_frame, best_fidx = max(window_buf, key=lambda x: x[0])
                    t_sec = float(best_fidx / native_fps)
                    out_name = f"frame_{saved_idx:06d}.jpg"
                    out_path = cam_dir / out_name
                    cv2.imwrite(str(out_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    extracted.append(out_path)
                    rel_name = f"{config.camera_name}/{out_name}"
                    timestamps[rel_name] = t_sec
                    saved_idx += 1
                    window_buf.clear()
                    # Skip to next window
                    for _ in range(step_frames - 1):
                        ret_skip = cap.grab()
                        frame_idx += 1
                        if not ret_skip:
                            break
            else:
                if frame_idx % step_frames == 0:
                    t_sec = float(frame_idx / native_fps)
                    out_name = f"frame_{saved_idx:06d}.jpg"
                    out_path = cam_dir / out_name
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                    extracted.append(out_path)
                    rel_name = f"{config.camera_name}/{out_name}"
                    timestamps[rel_name] = t_sec
                    saved_idx += 1

            frame_idx += 1

        cap.release()

    else:
        # Image folder
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        files = sorted(
            [f for f in source_p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        )
        saved_idx = 1
        for idx, f in enumerate(files):
            out_name = f"frame_{saved_idx:06d}.jpg"
            out_path = cam_dir / out_name
            if f.suffix.lower() in (".jpg", ".jpeg"):
                try:
                    os.link(f, out_path)
                except Exception:
                    try:
                        out_path.symlink_to(f)
                    except Exception:
                        shutil.copy2(f, out_path)
            else:
                img = cv2.imread(str(f))
                if img is not None:
                    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                else:
                    continue

            extracted.append(out_path)
            rel_name = f"{config.camera_name}/{out_name}"
            # Time spacing according to fps
            timestamps[rel_name] = float(idx / fps)
            saved_idx += 1

    # Update or merge with existing images/frames.json
    manifest_path = images_root / "frames.json"
    existing_data: Dict[str, Any] = {"schema": 1, "timestamps": {}}
    if manifest_path.is_file():
        try:
            existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "timestamps" not in existing_data:
                existing_data["timestamps"] = {}
        except Exception:
            existing_data = {"schema": 1, "timestamps": {}}

    existing_data["timestamps"].update(timestamps)
    manifest_path.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")

    print(
        f"[+] Extracted {len(extracted)} auxiliary frames from '{source_p.name}' into: {cam_dir}"
    )
    return extracted, timestamps


def match_features_between_images(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "SIFT",
    max_features: int = 3000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extract and match features between two images using SIFT or ORB."""
    if img1.ndim == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    if img2.ndim == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2

    if method.upper() == "SIFT" and hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features)
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=max_features)
        norm_type = cv2.NORM_HAMMING

    kp1, desc1 = detector.detectAndCompute(gray1, None)
    kp2, desc2 = detector.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), 0

    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(desc1, desc2, k=2)

    good = []
    ratio_thresh = 0.75
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 8:
        return np.empty((0, 2)), np.empty((0, 2)), 0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, len(good)


def estimate_relative_camera_pose(
    pts_cam0: np.ndarray,
    pts_cam2: np.ndarray,
    K_cam0: np.ndarray,
    K_cam2: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Estimate relative rotation R and translation unit direction t between cam0 and cam2.

    Returns (R_02, t_02, inliers) such that P_cam2 = R_02 @ P_cam0 + t_02.
    """
    if len(pts_cam0) < 8 or len(pts_cam2) < 8:
        return None, None, 0

    # Essential Matrix via normalized points
    pts0_norm = cv2.undistortPoints(
        np.expand_dims(pts_cam0, axis=1), K_cam0, None
    ).reshape(-1, 2)
    pts2_norm = cv2.undistortPoints(
        np.expand_dims(pts_cam2, axis=1), K_cam2, None
    ).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        pts0_norm, pts2_norm, focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=0.003
    )
    if E is None or mask is None:
        return None, None, 0

    inliers = int(np.sum(mask))
    if inliers < 8:
        return None, None, 0

    _, R, t, mask_pose = cv2.recoverPose(E, pts0_norm, pts2_norm)
    inliers_pose = int(np.sum(mask_pose))
    return R, t.flatten(), inliers_pose


def align_third_camera_to_rig(
    dataset_dir: Union[str, Path],
    third_config: ThirdCameraConfig,
    rig_profile_path: Optional[Union[str, Path]] = None,
    fps: float = 2.0,
) -> ThirdCameraConfig:
    """Align the third camera extrinsics (T_lidar_to_cam2) using SfM or relative feature matching."""
    dataset_dir = Path(dataset_dir)
    updated_config = copy.deepcopy(third_config)
    sparse_dir = dataset_dir / "sparse" / "0"

    # Strategy 1: Check if multi-camera SfM already reconstructed cam2
    images_bin = sparse_dir / "images.bin"
    align_json = dataset_dir / "colmap_to_lidar_alignment.json"

    if images_bin.is_file() and align_json.is_file():
        from scripts.pipeline_auto_calibrator_and_colorizer import (
            current_alignment,
            get_slam_trajectory_path,
            load_colmap_images,
            load_trajectory,
        )

        images = load_colmap_images(images_bin)
        cam2_imgs = [im for im in images if im["name"].startswith(f"{third_config.camera_name}/")]

        if len(cam2_imgs) >= 3:
            print(
                f"[*] Recovering {third_config.camera_name} extrinsics from {len(cam2_imgs)} SfM poses..."
            )
            al = current_alignment(dataset_dir, fps)
            s_sim = al["scale"]
            R_sim = np.array(al["R"])
            t_sim = np.array(al["t"])
            dt_sync = al.get("dt_sync_seconds", 0.0)

            slam_trj = get_slam_trajectory_path(dataset_dir)
            t_slam, pos_slam, rot_slam = load_trajectory(slam_trj)
            slerp = Slerp(t_slam, rot_slam)

            R_list, t_list = [], []
            for im in cam2_imgs:
                t_vid = frame_time(dataset_dir, im["name"], fps)
                t_q = t_slam[0] + (t_vid - (dt_sync + third_config.time_offset_s))
                if t_slam[0] <= t_q <= t_slam[-1]:
                    R_L = slerp(t_q).as_matrix()
                    idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
                    w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
                    p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]

                    C_lidar = s_sim * (R_sim @ im["C"]) + t_sim
                    R_cw_lidar = im["R_cw"] @ R_sim.T
                    R_wc_sfm = R_cw_lidar.T

                    R_LC2 = R_L.T @ R_wc_sfm
                    t_LC2 = R_L.T @ (C_lidar - p_L)
                    R_list.append(R_LC2)
                    t_list.append(t_LC2)

            if len(R_list) >= 3:
                mean_rot = Rot.from_matrix(R_list).mean()
                median_t = np.median(t_list, axis=0)

                T_LC2 = np.eye(4)
                T_LC2[:3, :3] = mean_rot.as_matrix()
                T_LC2[:3, 3] = median_t
                updated_config.T_lidar_to_cam2_rigid_4x4 = T_LC2.tolist()
                print(
                    f"[+] Third camera {third_config.camera_name} mounting extrinsics resolved via SfM:"
                )
                print(f"    Lever arm: {median_t * 100} cm (norm {np.linalg.norm(median_t)*100:.2f} cm)")
                print(
                    f"    Euler XYZ: {mean_rot.as_euler('xyz', degrees=True)} deg"
                )
                return updated_config

    # Strategy 2: Relative matching with cam0 (Insta360 front camera)
    cam0_files = sorted((dataset_dir / "images" / "cam0").glob("*.jpg"))
    cam2_files = sorted((dataset_dir / "images" / third_config.camera_name).glob("*.jpg"))

    if cam0_files and cam2_files:
        print(f"[*] Estimating relative extrinsics via 2D feature matching (cam0 <-> {third_config.camera_name})...")
        # Load sample pairs
        n_samples = min(5, min(len(cam0_files), len(cam2_files)))
        best_inliers = 0
        best_R02 = None
        best_t02 = None

        K0 = np.array([
            [1080.0, 0.0, 1920.0],
            [0.0, 1080.0, 1920.0],
            [0.0, 0.0, 1.0],
        ])
        int2 = third_config.intrinsics
        K2 = np.array([
            [int2["fx"], 0.0, int2["cx"]],
            [0.0, int2["fy"], int2["cy"]],
            [0.0, 0.0, 1.0],
        ])

        for i in range(n_samples):
            idx = int(i * len(cam0_files) / n_samples)
            im0 = cv2.imread(str(cam0_files[idx]))
            im2 = cv2.imread(str(cam2_files[min(idx, len(cam2_files) - 1)]))
            if im0 is None or im2 is None:
                continue

            pts0, pts2, n_match = match_features_between_images(im0, im2, method="SIFT")
            if n_match >= 15:
                R_cand, t_cand, n_inl = estimate_relative_camera_pose(pts0, pts2, K0, K2)
                if n_inl > best_inliers:
                    best_inliers = n_inl
                    best_R02 = R_cand
                    best_t02 = t_cand

        if best_R02 is not None:
            # Assume nominal forward-mounted phone lever arm distance ~10-15 cm from cam0
            scale_baseline = 0.12  # 12 cm default baseline
            t_rel = best_t02 * scale_baseline

            # T_lidar_to_cam2 = T_cam0_to_cam2 @ T_lidar_to_cam0
            # Read T_LC0 from rig profile if available
            T_LC0 = np.eye(4)
            if rig_profile_path and Path(rig_profile_path).is_file():
                try:
                    rig = json.loads(Path(rig_profile_path).read_text(encoding="utf-8"))
                    T_LC0 = np.array(rig.get("T_lidar_to_cam0_rigid_4x4", np.eye(4)))
                except Exception:
                    pass

            T_02 = np.eye(4)
            T_02[:3, :3] = best_R02
            T_02[:3, 3] = t_rel

            T_LC2 = T_02 @ T_LC0
            updated_config.T_lidar_to_cam2_rigid_4x4 = T_LC2.tolist()
            print(
                f"[+] Successfully estimated {third_config.camera_name} extrinsics via 2D feature matching ({best_inliers} inliers)."
            )
            return updated_config

    print("[*] Retaining nominal/preset third camera extrinsics.")
    return updated_config


def synthesize_third_camera_poses(
    dataset_dir: Union[str, Path],
    third_config: ThirdCameraConfig,
    fps: float = 2.0,
    dt_sync: float = 0.0,
) -> List[Dict[str, Any]]:
    """Synthesize 6DoF camera poses for cam2 frames along the SLAM trajectory."""
    dataset_dir = Path(dataset_dir)
    from scripts.pipeline_auto_calibrator_and_colorizer import (
        get_slam_trajectory_path,
        load_trajectory,
    )

    trj_path = get_slam_trajectory_path(dataset_dir)
    if not trj_path or not Path(trj_path).is_file():
        return []

    t_slam, pos_slam, rot_slam = load_trajectory(trj_path)
    slerp = Slerp(t_slam, rot_slam)

    t_start, t_end = t_slam[0], t_slam[-1]
    cam2_dir = dataset_dir / "images" / third_config.camera_name
    cam2_files = sorted(cam2_dir.glob("*.jpg"))

    T_LC2 = np.array(third_config.T_lidar_to_cam2_rigid_4x4)
    R_LC2, t_LC2 = T_LC2[:3, :3], T_LC2[:3, 3]

    poses: List[Dict[str, Any]] = []
    dt_total = dt_sync + third_config.time_offset_s

    for img_file in cam2_files:
        rel_name = f"{third_config.camera_name}/{img_file.name}"
        t_vid = frame_time(dataset_dir, rel_name, fps)
        t_q = t_start + (t_vid - dt_total)

        if not (t_start <= t_q <= t_end):
            continue

        idx = np.clip(np.searchsorted(t_slam, t_q), 1, len(t_slam) - 1)
        w = (t_q - t_slam[idx - 1]) / (t_slam[idx] - t_slam[idx - 1])
        p_L = (1.0 - w) * pos_slam[idx - 1] + w * pos_slam[idx]
        R_L = slerp(t_q).as_matrix()

        p_cam = p_L + R_L @ t_LC2
        R_world_cam = R_L @ R_LC2
        R_cw = R_world_cam.T
        t_cw = -R_cw @ p_cam
        q_cw = Rot.from_matrix(R_cw).as_quat()  # x, y, z, w

        poses.append({
            "name": rel_name,
            "R_cw": R_cw,
            "t_cw": t_cw,
            "q_cw": q_cw,
            "C": p_cam,
            "camera_id": 3,
        })

    return poses


def inject_third_camera_into_colmap(
    sparse_out: Path,
    third_config: ThirdCameraConfig,
    poses: List[Dict[str, Any]],
    camera_id: int = 3,
) -> None:
    """Inject Camera 3 and its synthesized or aligned poses into sparse/0/cameras.txt and images.txt."""
    sparse_out = Path(sparse_out)
    sparse_out.mkdir(parents=True, exist_ok=True)
    cameras_txt = sparse_out / "cameras.txt"
    images_txt = sparse_out / "images.txt"

    # 1. Update cameras.txt
    int2 = third_config.intrinsics
    w = int(int2["width"])
    h = int(int2["height"])
    fx, fy = int2["fx"], int2["fy"]
    cx, cy = int2["cx"], int2["cy"]

    cam_model = third_config.camera_model.upper()
    if cam_model == "PINHOLE":
        cam_line = f"{camera_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"
    elif cam_model == "OPENCV":
        k1, k2, p1, p2 = int2.get("k1", 0.0), int2.get("k2", 0.0), int2.get("p1", 0.0), int2.get("p2", 0.0)
        cam_line = f"{camera_id} OPENCV {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {p1} {p2}\n"
    elif cam_model == "OPENCV_FISHEYE":
        k1, k2, k3, k4 = int2.get("k1", 0.0), int2.get("k2", 0.0), int2.get("p1", 0.0), int2.get("p2", 0.0)
        cam_line = f"{camera_id} OPENCV_FISHEYE {w} {h} {fx} {fy} {cx} {cy} {k1} {k2} {k3} {k4}\n"
    else:
        cam_line = f"{camera_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n"

    existing_cams = cameras_txt.read_text(encoding="utf-8") if cameras_txt.is_file() else "# Camera list\n"
    if f"\n{camera_id} " not in f"\n{existing_cams}":
        with open(cameras_txt, "a", encoding="utf-8") as f:
            f.write(cam_line)
        print(f"  [+] Injected camera {camera_id} ({cam_model}) into cameras.txt")

    # 2. Update images.txt with third camera poses
    if poses and images_txt.is_file():
        existing_images = images_txt.read_text(encoding="utf-8")
        lines_to_add = []
        max_id = 0
        for line in existing_images.splitlines():
            line_s = line.strip()
            if line_s and not line_s.startswith("#"):
                parts = line_s.split()
                if len(parts) >= 9 and parts[0].isdigit():
                    max_id = max(max_id, int(parts[0]))

        cur_id = max_id + 1
        for p in poses:
            if p["name"] not in existing_images:
                q = p["q_cw"]  # x, y, z, w
                t = p["t_cw"]
                lines_to_add.append(
                    f"{cur_id} {q[3]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} "
                    f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {camera_id} {p['name']}\n\n"
                )
                cur_id += 1

        if lines_to_add:
            with open(images_txt, "a", encoding="utf-8") as f:
                f.writelines(lines_to_add)
            print(f"  [+] Injected {len(lines_to_add)} poses for {third_config.camera_name} into images.txt")
