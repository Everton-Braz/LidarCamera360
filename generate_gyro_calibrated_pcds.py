import os
import sys
import json
import time
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
SLAM_PCD = os.path.join(DATASET_DIR, "slam_out", "pcd", "all_raw_points.pcd")
TRJ_FILE = os.path.join(DATASET_DIR, "slam_out", "result", "Raven_3DMakerPro_Scan.txt")
INSV_FILE = os.path.join(DATASET_DIR, "VID_20260902_143757_00_277.insv")
OUT_DIR = os.path.join(DATASET_DIR, "calibration_gyro_calib")
FFMPEG_PATH = r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe"


class FisheyeProjector:
    def __init__(self, width=3840, height=3840, fov_deg=196.0):
        self.W = width
        self.H = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        fov_rad = np.radians(fov_deg)
        self.f = (width / 2.0) / (fov_rad / 2.0)
        self.max_radius = (min(width, height) / 2.0) * 0.985

    def project(self, P_cam):
        X = P_cam[:, 0]
        Y = P_cam[:, 1]
        Z = P_cam[:, 2]
        dist = np.linalg.norm(P_cam, axis=1)
        r_xy = np.hypot(X, Y)
        theta = np.arctan2(r_xy, Z)
        r_img = self.f * theta
        valid = (Z > 0.05) & (dist > 0.20) & (dist < 30.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid, dist, r_img


def load_pcd_xyz(path):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line.startswith("DATA"):
                break

        fields = {}
        for line in header_lines:
            parts = line.split()
            if parts[0] == "FIELDS":
                fields["names"] = parts[1:]
            elif parts[0] == "SIZE":
                fields["sizes"] = [int(p) for p in parts[1:]]
            elif parts[0] == "TYPE":
                fields["types"] = parts[1:]
            elif parts[0] == "POINTS":
                fields["count"] = int(parts[1])
            elif parts[0] == "DATA":
                data_mode = parts[1]

        n_pts = fields["count"]
        dtype_list = []
        for name, sz, tp in zip(fields["names"], fields["sizes"], fields["types"]):
            code = "f" if tp == "F" else ("u" if tp == "U" else "i")
            dtype_list.append((name, f"<{code}{sz}"))

        data = np.fromfile(f, dtype=dtype_list, count=n_pts)
        pts = np.stack([data["x"], data["y"], data["z"]], axis=-1)
        return pts


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

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())
    print(f"  [+] Saved PCD: {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB)")


FRAME_CACHE = {}


def get_cached_frame(stream_idx, time_sec):
    key = (stream_idx, round(time_sec, 2))
    if key in FRAME_CACHE:
        return FRAME_CACHE[key]
    cmd = [
        FFMPEG_PATH,
        "-ss",
        f"{time_sec:.3f}",
        "-i",
        INSV_FILE,
        "-map",
        f"0:v:{stream_idx}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = p.stdout.read(3840 * 3840 * 3)
    p.kill()
    if len(raw) == 3840 * 3840 * 3:
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((3840, 3840, 3))
        FRAME_CACHE[key] = frame
        return frame
    return None


def get_rig_geometry(pitch_deg, yaw_deg, roll_deg):
    u_L = np.array([0.003102, -0.504937, -0.863152])
    right_L = np.array([1.0, 0.0, 0.0])
    right_L = right_L - np.dot(right_L, u_L) * u_L
    right_L = right_L / np.linalg.norm(right_L)

    Z0 = right_L
    Y0 = u_L - np.dot(u_L, Z0) * Z0
    Y0 /= np.linalg.norm(Y0)
    X0 = np.cross(Y0, Z0)
    R_s0_base = np.stack([X0, Y0, Z0], axis=0)

    Z1 = -right_L
    Y1 = u_L - np.dot(u_L, Z1) * Z1
    Y1 /= np.linalg.norm(Y1)
    X1 = np.cross(Y1, Z1)
    R_s1_base = np.stack([X1, Y1, Z1], axis=0)

    # Stream 0: Right lens
    R_trim0 = Rot.from_euler("xyz", [pitch_deg, yaw_deg, roll_deg], degrees=True).as_matrix()
    R_s0 = R_trim0 @ R_s0_base

    # Stream 1: Left lens
    R_trim1 = Rot.from_euler("xyz", [pitch_deg, -yaw_deg, -roll_deg], degrees=True).as_matrix()
    R_s1 = R_trim1 @ R_s1_base

    c_L_phys = 0.185 * u_L
    return R_s0, R_s1, c_L_phys


def colorize_at_times(pts_world, trj, R_s0, R_s1, c_L, pairs_t_slam_t_video):
    N = len(pts_world)
    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_xyz = trj[:, 1:4]
    t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

    projector = FisheyeProjector(width=3840, height=3840, fov_deg=196.0)
    color_accum = np.zeros((N, 3), dtype=np.uint8)
    best_cost = np.full(N, 1e9, dtype=np.float32)

    total_pairs = len(pairs_t_slam_t_video)

    for i, (t_slam_rel, v_time_sec) in enumerate(pairs_t_slam_t_video):
        query_t = t_start + t_slam_rel
        pose_idx = np.argmin(np.abs(t_stamps - query_t))

        R_w_l = t_rot[pose_idx]
        t_w_l = t_xyz[pose_idx]

        v_time_sec = float(np.clip(v_time_sec, 0.5, 96.5))
        img0 = get_cached_frame(stream_idx=0, time_sec=v_time_sec)
        img1 = get_cached_frame(stream_idx=1, time_sec=v_time_sec)

        # Points in LiDAR body frame at this pose
        P_l = (pts_world - t_w_l) @ R_w_l

        # Stream 0 (Right Lens)
        if img0 is not None:
            P_cam0 = (P_l - c_L) @ R_s0.T
            u0, v0, mask0, dist0, r0 = projector.project(P_cam0)
            cost0 = dist0 + (r0 / 1920.0) * 1.5
            better0 = mask0 & (cost0 < best_cost)
            if np.any(better0):
                u_idx = np.round(u0[better0]).astype(np.int64)
                v_idx = np.round(v0[better0]).astype(np.int64)
                bgr0 = img0[v_idx, u_idx]
                rgb0 = bgr0[:, [2, 1, 0]]
                color_accum[better0] = rgb0
                best_cost[better0] = cost0[better0]

        # Stream 1 (Left Lens)
        if img1 is not None:
            P_cam1 = (P_l - c_L) @ R_s1.T
            u1, v1, mask1, dist1, r1 = projector.project(P_cam1)
            cost1 = dist1 + (r1 / 1920.0) * 1.5
            better1 = mask1 & (cost1 < best_cost)
            if np.any(better1):
                u_idx = np.round(u1[better1]).astype(np.int64)
                v_idx = np.round(v1[better1]).astype(np.int64)
                bgr1 = img1[v_idx, u_idx]
                rgb1 = bgr1[:, [2, 1, 0]]
                color_accum[better1] = rgb1
                best_cost[better1] = cost1[better1]

        if (i + 1) % 10 == 0 or (i + 1) == total_pairs:
            colored_ratio = np.count_nonzero(best_cost < 1e8) / N * 100.0
            print(f"    [{i+1}/{total_pairs}] Processed {v_time_sec:.2f}s: {colored_ratio:.1f}% points colored")

    uncolored = best_cost >= 1e8
    if np.any(uncolored):
        color_accum[uncolored] = [128, 128, 128]

    return color_accum


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 80)
    print(" GENERATING POINT CLOUDS WITH EXACT GYROSCOPE SYNCHRONIZATION")
    print("=" * 80)

    pts_world = load_pcd_xyz(SLAM_PCD)
    trj = np.loadtxt(TRJ_FILE)
    print(f"[*] Loaded SLAM Map: {len(pts_world):,} points")

    # High-precision time sync from gyro correlation:
    dt_gyro = 5.7075
    t_vid_10 = 10.0
    t_slam_10 = t_vid_10 - dt_gyro

    # Configuration 1: Gyro-Sync Bundle Optimized (Pitch = -0.44°, Yaw = +0.61°, Roll = +0.45°)
    print("\n[1/3] Generating Gyro-Sync Bundle Optimized PCD at Video Time = 10.0s...")
    R_s0_bundle, R_s1_bundle, c_L = get_rig_geometry(pitch_deg=-0.443, yaw_deg=0.608, roll_deg=0.446)
    colors_bundle = colorize_at_times(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0_bundle,
        R_s1=R_s1_bundle,
        c_L=c_L,
        pairs_t_slam_t_video=[(t_slam_10, t_vid_10)],
    )
    pcd_bundle = os.path.join(OUT_DIR, "gyro_sync_bundle_optimal_single_frame_10s.pcd")
    write_pcd(pcd_bundle, pts_world, colors_bundle)

    # Configuration 2: Gyro-Sync Cand07 Geometry (Pitch = -1.24°, Yaw = +2.43°, Roll = 0.0°) with exact dt_gyro
    print("\n[2/3] Generating Gyro-Sync Cand07 PCD at Video Time = 10.0s...")
    R_s0_cand07, R_s1_cand07, _ = get_rig_geometry(pitch_deg=-1.236, yaw_deg=2.427, roll_deg=0.0)
    colors_cand07 = colorize_at_times(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0_cand07,
        R_s1=R_s1_cand07,
        c_L=c_L,
        pairs_t_slam_t_video=[(t_slam_10, t_vid_10)],
    )
    pcd_cand07 = os.path.join(OUT_DIR, "gyro_sync_cand07_single_frame_10s.pcd")
    write_pcd(pcd_cand07, pts_world, colors_cand07)

    # Configuration 3: Full Point Cloud with Gyro-Sync Bundle Optimal
    print("\n[3/3] Generating FULL Point Cloud with Gyro-Sync Bundle Optimal...")
    t_stamps = trj[:, 0]
    t_start = t_stamps[0]
    t_end = t_stamps[-1]
    total_dur = t_end - t_start

    pairs_full = []
    # 1.0s interval across the whole scan
    for t_s in np.arange(1.0, total_dur - 1.0, 1.0):
        v_t = t_s + dt_gyro
        if 0.5 <= v_t <= 96.5:
            pairs_full.append((t_s, v_t))

    print(f"[*] Processing {len(pairs_full)} keyframes across {total_dur:.1f}s trajectory...")
    colors_full = colorize_at_times(
        pts_world=pts_world,
        trj=trj,
        R_s0=R_s0_bundle,
        R_s1=R_s1_bundle,
        c_L=c_L,
        pairs_t_slam_t_video=pairs_full,
    )
    pcd_full = os.path.join(OUT_DIR, "gyro_sync_bundle_optimal_full.pcd")
    write_pcd(pcd_full, pts_world, colors_full)

    print("\n" + "=" * 80)
    print(" ALL GYRO-CALIBRATED POINT CLOUDS GENERATED SUCCESSFULLY!")
    print(f" Directory: {OUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
