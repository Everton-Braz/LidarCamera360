import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

DATASET_DIR = r"C:\Users\User\Downloads\Lidou\DinamicAprilTagCalib"
TRJ_FILE = os.path.join(DATASET_DIR, "slam_out", "result", "Raven_3DMakerPro_Scan.txt")
DET_FILE = os.path.join(DATASET_DIR, "detections_raw.json")

trj = np.loadtxt(TRJ_FILE)
with open(DET_FILE, "r") as f:
    detections = json.load(f)

# Fisheye Projector
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
        valid = (Z > 0.05) & (dist > 0.20) & (dist < 25.0) & (r_img <= self.max_radius)
        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)
        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        return u, v, circle_valid

    def pixel_to_ray(self, u, v):
        dx = u - self.cx
        dy = v - self.cy
        r_pix = np.hypot(dx, dy)
        theta = r_pix / self.f
        phi = np.arctan2(dy, dx)
        return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

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

projector = FisheyeProjector(width=3840, height=3840)

t_stamps = trj[:, 0]
t_start = t_stamps[0]
t_dur = t_stamps[-1] - t_start
t_rel = t_stamps - t_start
t_xyz = trj[:, 1:4]
t_rot = Rot.from_quat(trj[:, 4:8]).as_matrix()

# EXACT GYRO DT:
DT_GYRO = 5.7075
print(f"[*] Triangulating AprilTags with exact Gyro Sync dt = {DT_GYRO:.4f}s...")

def triangulate(R0, R1, dt_sec):
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
        orig_l = c_L  # in LiDAR frame, camera position is c_L

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

tag_world = triangulate(R_base0, R_base1, dt_sec=DT_GYRO)
print(f"[+] Triangulated {len(tag_world)} tags in 3D world space:")
for tid, pos in sorted(tag_world.items()):
    print(f"    Tag #{tid}: World pos = [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] m")

# Non-linear optimization of [pitch, yaw, roll]
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
        ray_c = projector.pixel_to_ray(u, v)

        R_cam = R0 if d["stream"] == 0 else R1
        ray_l = R_cam.T @ ray_c
        orig_l = c_L

        ray_w = R_w_l @ ray_l
        ray_w = ray_w / np.linalg.norm(ray_w)
        orig_w = R_w_l @ orig_l + t_w_l

        # Point to ray distance vector: (P_w - orig_w) - dot * ray_w
        diff = P_w - orig_w
        dist_vec = diff - np.dot(diff, ray_w) * ray_w
        resids.extend(dist_vec)

    return np.array(resids)

res_init = residual_fun([0.0, 0.0, 0.0])
print(f"Initial RMS error (zeros): {np.sqrt(np.mean(res_init**2)):.2f} px")

opt = least_squares(residual_fun, [0.0, 0.0, 0.0], method='lm')
p_opt, y_opt, r_opt = opt.x
res_opt = residual_fun(opt.x)
rms_opt = np.sqrt(np.mean(res_opt**2))

print("\n" + "=" * 70)
print(" OPTIMAL BUNDLE EXTRINSICS WITH FIXED GYRO SYNC (dt = 5.7075s)")
print("=" * 70)
print(f"Pitch: {p_opt:+.4f}°")
print(f"Yaw:   {y_opt:+.4f}°")
print(f"Roll:  {r_opt:+.4f}°")
print(f"Optimized RMS error: {rms_opt:.2f} px (across {len(res_opt)//2} ray-tag observations)")
