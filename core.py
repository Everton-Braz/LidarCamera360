# -*- coding: utf-8 -*-
"""Geometry, data loading and the alignment cost used by the calibrator."""

import math
import os

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

# Camera convention: X right, Y down, Z forward.
# Equirect mapping: u = (atan2(X,Z)/2pi + 0.5)*W, v = (0.5 - asin(-Y/r)/pi)*H

ZENITH_DEADZONE_DEG = 0.30
ZENITH_WEIGHT = 0.60
ZENITH_MIN_INLIERS = 25
MAX_CAM_TILT_DEG = 30.0

QUALITY = {
    "Fast": [(512, 256, 3.0, 120000), (1024, 512, 2.5, 200000),
             (2048, 1024, 2.2, 220000)],
    "Normal": [(512, 256, 3.0, 150000), (1024, 512, 2.5, 250000),
               (2048, 1024, 2.2, 300000), (3072, 1536, 2.0, 320000)],
    "High": [(512, 256, 3.0, 150000), (1024, 512, 2.5, 250000),
             (2048, 1024, 2.2, 300000), (3072, 1536, 2.0, 320000),
             (4096, 2048, 1.8, 320000)],
}
SOLO_STAGES = 3


class DataError(Exception):
    pass


# --------------------------------------------------------------- geometry --
def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def level_rot(up_L):
    """Lidar frame -> gravity levelled frame (Y down, Z forward)."""
    u = unit(up_L)
    x = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(x, u)) > 0.98:
        x = np.array([0.0, 1.0, 0.0])
    Yp = -u
    Zp = unit(x - np.dot(x, u) * u)
    return np.stack([np.cross(Yp, Zp), Yp, Zp], 0)


def sph_uv(P, W, H):
    r = np.linalg.norm(P, axis=1)
    lon = np.arctan2(P[:, 0], P[:, 2])
    lat = np.arcsin(np.clip(-P[:, 1] / np.maximum(r, 1e-12), -1, 1))
    return (lon / (2 * np.pi) + 0.5) * W, (0.5 - lat / np.pi) * H


def dirs_from_uv(u, v, W, H):
    lon = (u / W - 0.5) * 2 * np.pi
    lat = (0.5 - v / H) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), -np.sin(lat),
                     np.cos(lat) * np.cos(lon)], -1)


def splat(P, w, W, H):
    """Bilinear accumulation. Nearest-pixel binning makes the cost piecewise
    constant and blind to sub-pixel shifts."""
    u, v = sph_uv(P, W, H)
    u = u - 0.5
    v = np.clip(v - 0.5, 0.0, H - 1.001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu, fv = u - u0, v - v0
    u0 = np.mod(u0, W)
    u1 = np.mod(u0 + 1, W)
    v1 = np.clip(v0 + 1, 0, H - 1)
    v0 = np.clip(v0, 0, H - 1)
    acc = np.zeros(W * H, np.float64)
    for uu, vv, ww in ((u0, v0, (1 - fu) * (1 - fv)), (u1, v0, fu * (1 - fv)),
                       (u0, v1, (1 - fu) * fv), (u1, v1, fu * fv)):
        acc += np.bincount(vv * W + uu, weights=w * ww, minlength=W * H)
    return acc.reshape(H, W).astype(np.float32)


def bilinear(M, u, v):
    H, W = M.shape
    u = u - 0.5
    v = np.clip(v - 0.5, 0, H - 1.001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)
    u0 = np.mod(u0, W)
    u1 = np.mod(u0 + 1, W)
    v1 = np.clip(v0 + 1, 0, H - 1)
    v0 = np.clip(v0, 0, H - 1)
    return (M[v0, u0] * (1 - fu) * (1 - fv) + M[v0, u1] * fu * (1 - fv) +
            M[v1, u0] * (1 - fu) * fv + M[v1, u1] * fu * fv)


def mkR(rv, Rlev):
    return Rot.from_rotvec(rv).as_matrix() @ Rlev


def ang_between(R1, R2):
    return float(np.degrees(Rot.from_matrix(R1 @ R2.T).magnitude()))


def R_for(rv, d, anchor=None):
    """Resolve a shared rotation vector. Every set has its own levelled frame,
    so a common extrinsic must be resolved against one anchor and reused as is;
    resolving it per set gives each set a different extrinsic."""
    return mkR(rv, d.Rlev if anchor is None else anchor.Rlev)


# ---------------------------------------------------------------- loading --
def _pointcloud2(m):
    """sensor_msgs/PointCloud2 -> (N,5) [x, y, z, intensity, tag]."""
    names = {f.name: f for f in m.fields}
    if not {"x", "y", "z"} <= set(names):
        return None
    tmap = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
            5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
    dt, used = [], 0
    for f in sorted(m.fields, key=lambda f: f.offset):
        if f.offset > used:
            dt.append(("_pad%d" % used, np.uint8, f.offset - used))
        base = tmap.get(f.datatype, np.uint8)
        dt.append((f.name, base))
        used = f.offset + np.dtype(base).itemsize
    if m.point_step > used:
        dt.append(("_tail", np.uint8, m.point_step - used))
    arr = np.frombuffer(bytes(m.data), dtype=np.dtype(dt))
    a = np.zeros((len(arr), 5), np.float32)
    a[:, 0], a[:, 1], a[:, 2] = arr["x"], arr["y"], arr["z"]
    for key in ("intensity", "reflectivity", "i"):
        if key in names:
            a[:, 3] = arr[key]
            break
    finite = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1]) & np.isfinite(a[:, 2])
    return a[finite]


def read_bag(path, progress=None):
    """Accumulate all lidar points and IMU samples from a bag."""
    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        raise DataError("rosbags is not installed (pip install rosbags)")
    from pathlib import Path

    chunks, imu_a, imu_g = [], [], []
    lidar_keywords = ("lidar", "point", "vanjee", "722z", "cloud", "velodyne", "hesai", "rslidar", "ouster")
    with AnyReader([Path(path)]) as r:
        lc = [c for c in r.connections
              if (any(k in c.topic.lower() for k in lidar_keywords)
                  or "pointcloud2" in c.msgtype.lower()
                  or "custommsg" in c.msgtype.lower())
              and "imu" not in c.topic.lower()]
        ic = [c for c in r.connections
              if "imu" in c.topic.lower()
              or "imu" in c.msgtype.lower()]
        if not lc:
            raise DataError("no lidar topic in bag (looked for 'vanjee', '722z', 'lidar', 'point', 'PointCloud2')")
        total = sum(c.msgcount for c in lc) or 1
        done = 0
        for conn, ts, raw in r.messages(connections=lc):
            m = r.deserialize(raw, conn.msgtype)
            pts = getattr(m, "points", None)
            if pts is not None:                                # Livox CustomMsg
                a = np.empty((len(pts), 5), np.float32)
                a[:, 0] = [p.x for p in pts]
                a[:, 1] = [p.y for p in pts]
                a[:, 2] = [p.z for p in pts]
                a[:, 3] = [p.reflectivity for p in pts]
                a[:, 4] = [p.tag for p in pts]
                valid = ((a[:, 4].astype(np.uint8) & 0x03) == 0) & np.isfinite(a[:, 0])
                a = a[valid]
            elif hasattr(m, "fields") and hasattr(m, "data"):   # PointCloud2
                a = _pointcloud2(m)
                if a is None:
                    raise DataError("PointCloud2 without x/y/z fields")
            else:
                raise DataError("unsupported lidar message: %s" % conn.msgtype)
            if len(a) > 0:
                chunks.append(a)
            done += 1
            if progress and done % 20 == 0:
                progress(done / total)
        for conn, ts, raw in r.messages(connections=ic):
            m = r.deserialize(raw, conn.msgtype)
            imu_a.append([m.linear_acceleration.x, m.linear_acceleration.y,
                          m.linear_acceleration.z])
            imu_g.append([m.angular_velocity.x, m.angular_velocity.y,
                          m.angular_velocity.z])
        dur = (r.end_time - r.start_time) * 1e-9

    if not chunks:
        raise DataError("bag contains no lidar points")
    P = np.concatenate(chunks, 0)
    finite = np.isfinite(P[:, 0]) & np.isfinite(P[:, 1]) & np.isfinite(P[:, 2])
    P = P[finite]
    d = np.linalg.norm(P[:, :3], axis=1)
    P = P[d > 0.05]
    if len(P) < 20000:
        raise DataError("too few valid lidar points (%d)" % len(P))
    if not imu_a:
        raise DataError("no IMU topic in bag - the gravity reference is required")
    return dict(points=P, imu_a=np.array(imu_a), imu_g=np.array(imu_g), duration=dur)


def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise DataError("cannot open video")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return dict(frames=n, fps=fps, W=W, H=H, duration=(n / fps if fps else 0.0))


def grab_frame(path, rel):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise DataError("cannot open video")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * rel))))
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise DataError("cannot read frame at %.2f" % rel)
    return img


def grab_frames(path, rels):
    """Grab multiple frames in a single video capture pass."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise DataError("cannot open video")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for rel in rels:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * rel))))
        ok, img = cap.read()
        if not ok:
            cap.release()
            raise DataError("cannot read frame at %.2f" % rel)
        frames.append(img)
    cap.release()
    return frames


# ----------------------------------------------------------------- checks --
def check_video_equirect(info):
    r = info["W"] / max(info["H"], 1)
    if abs(r - 2.0) > 0.02:
        return ("video is %dx%d (aspect %.2f), a 2:1 equirectangular panorama "
                "is required" % (info["W"], info["H"], r))
    return None


def check_static(imu_a, imu_g):
    ga = float(np.linalg.norm(np.std(imu_a, 0)))
    gg = float(np.linalg.norm(np.std(imu_g, 0)))
    msg = "IMU: acc sigma %.4f g, gyro sigma %.4f rad/s" % (ga, gg)
    if gg > 0.05 or ga > 0.05:
        return False, msg + "  -> rig was moving"
    return True, msg


def zenith_from_image(img, W=2048, H=1024):
    """Scene vertical in the camera frame, from vertical world lines.
    Independent of the lidar. Returns (z, n_lines, n_inliers) or None."""
    im = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    try:
        lines = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(g)[0]
    except Exception:
        return None
    if lines is None or len(lines) < 12:
        return None
    L = lines.reshape(-1, 4)
    ln = np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1])
    ang = np.degrees(np.arctan2(np.abs(L[:, 2] - L[:, 0]), np.abs(L[:, 3] - L[:, 1])))
    vm = (L[:, 1] + L[:, 3]) / 2
    L = L[(ln > 25) & (ang < 12) & (vm > 0.18 * H) & (vm < 0.82 * H)]
    if len(L) < 10:
        return None

    # a world-vertical line lies on a great circle whose normal is horizontal,
    # so the zenith is the vector orthogonal to all of those normals
    N = np.cross(dirs_from_uv(L[:, 0], L[:, 1], W, H),
                 dirs_from_uv(L[:, 2], L[:, 3], W, H))
    nn = np.linalg.norm(N, axis=1)
    N = N[nn > 1e-9] / nn[nn > 1e-9][:, None]
    if len(N) < 10:
        return None

    rng = np.random.default_rng(0)
    thr = math.sin(math.radians(0.8))
    best = (0, None, None)
    for _ in range(3000):
        i, j = rng.choice(len(N), 2, replace=False)
        z = np.cross(N[i], N[j])
        n = np.linalg.norm(z)
        if n < 1e-6:
            continue
        z /= n
        if z[1] > 0:
            z = -z
        inl = np.abs(N @ z) < thr
        if int(inl.sum()) > best[0]:
            best = (int(inl.sum()), z, inl)
    if best[0] < 8:
        return None
    z = np.linalg.svd(N[best[2]])[2][-1]
    if z[1] > 0:
        z = -z
    return z, len(N), best[0]


def video_frame_spread(path, n=3):
    """Frame-to-frame difference at spread positions: high on a moving rig."""
    fr = grab_frames(path, np.linspace(0.3, 0.7, n))
    sm = [cv2.cvtColor(cv2.resize(f, (512, 256)), cv2.COLOR_BGR2GRAY).astype(np.float32)
          for f in fr]
    return max(float(np.mean(np.abs(sm[i] - sm[0]))) for i in range(1, n))


def zenith_drift(path, n=3):
    """Spread of the scene zenith across the clip. Steady on a static
    unstabilised capture; wanders when stabilisation is active."""
    fr = grab_frames(path, np.linspace(0.3, 0.7, n))
    zs = []
    for f in fr:
        try:
            z = zenith_from_image(f)
        except Exception:
            z = None
        if z:
            zs.append(z[0])
    if len(zs) < 2:
        return None
    return max(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(zs[0], zs[i]))))))
               for i in range(1, len(zs)))


# ------------------------------------------------------------------ edges --
def build_lidar_edges(P3, refl, up_L, W=2048, H=1024):
    """Render the cloud as an equirect panorama and pull edges out of it:
    depth discontinuities, normal breaks and reflectivity gradients.
    Handles both dense solid-state scan patterns and spinning multi-beam rings."""
    X = P3 @ level_rot(up_L).T
    u, v = sph_uv(X, W, H)
    r = np.linalg.norm(X, axis=1)
    idx = (np.clip(v.astype(np.int32), 0, H - 1) * W +
           np.clip(u.astype(np.int32), 0, W - 1))
    order = np.argsort(-r)                      # nearest point wins the pixel
    Rimg = np.full(W * H, np.nan)
    Fimg = np.full(W * H, np.nan)
    SRC = np.full(W * H, -1, np.int64)
    Rimg[idx[order]] = r[order]
    Fimg[idx[order]] = refl[order]
    SRC[idx[order]] = order
    Rimg, Fimg, SRC = Rimg.reshape(H, W), Fimg.reshape(H, W), SRC.reshape(H, W)
    valid = np.isfinite(Rimg)

    # 1. 2D depth discontinuities (for solid-state / dense regions)
    Rs = np.where(valid, cv2.medianBlur(np.where(valid, Rimg, 0).astype(np.float32), 3),
                  np.nan)
    disc2d = np.zeros((H, W), np.float32)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0), (2, 2), (-2, -2), (2, -2), (2, 2)):
        disc2d = np.fmax(disc2d, np.nan_to_num(np.roll(np.roll(Rs, dy, 0), dx, 1) - Rs))

    # 2. Along-ring (azimuth) depth discontinuities (crucial for 16/32-line spinning LiDARs like Raven)
    disc1d = np.zeros((H, W), np.float32)
    for dx in (-4, -3, -2, -1, 1, 2, 3, 4):
        rolled_r = np.roll(Rimg, dx, axis=1)
        valid_pair = valid & np.isfinite(rolled_r)
        diff_r = np.where(valid_pair, np.abs(rolled_r - Rimg), 0.0)
        disc1d = np.maximum(disc1d, diff_r)

    disc = np.maximum(disc2d, disc1d)
    depth_e = np.clip(disc / np.clip(np.nan_to_num(Rimg, nan=1.0), 0.5, None) / 0.18,
                      0, 1) * valid

    # 3. Surface normal breaks
    inv = 1.0 / np.clip(np.nan_to_num(Rs, nan=1e3), 0.1, None)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(inv, (0, 0), 1.6), cv2.CV_32F, ksize=5))
    norm_e = np.clip(lap * np.clip(np.nan_to_num(Rs), 0, 20) / 1.2, 0, 1) * valid
    norm_e = np.where(depth_e > 0.08, 0, norm_e)    # do not count occlusions twice

    # 4. Reflectivity gradients (both 2D and along-ring)
    refl_val = np.clip(np.nan_to_num(Fimg), 0, 255)
    refl_max = float(np.percentile(refl_val[valid], 99.0)) if valid.any() else 255.0
    refl_scale = max(refl_max, 50.0)

    Fs = cv2.GaussianBlur(cv2.medianBlur(refl_val.astype(np.uint8), 5).astype(np.float32),
                          (0, 0), 1.0)
    refl_2d = np.clip(np.hypot(cv2.Sobel(Fs, cv2.CV_32F, 1, 0, ksize=3),
                               cv2.Sobel(Fs, cv2.CV_32F, 0, 1, ksize=3)) / (0.35 * refl_scale),
                      0, 1) * valid

    refl_1d = np.zeros((H, W), np.float32)
    for dx in (-3, -2, -1, 1, 2, 3):
        rolled_f = np.roll(Fimg, dx, axis=1)
        valid_pair = valid & np.isfinite(rolled_f)
        diff_f = np.where(valid_pair, np.abs(rolled_f - Fimg), 0.0)
        refl_1d = np.maximum(refl_1d, diff_f)
    refl_1d = np.clip(refl_1d / (0.30 * refl_scale), 0, 1) * valid
    refl_e = np.maximum(refl_2d, refl_1d)

    E = np.clip((depth_e + 0.7 * norm_e + refl_e) * valid, 0, 2.0)
    ys, xs = np.nonzero(E > 0.10)
    ay, ax = np.nonzero(valid)
    return (P3[SRC[ys, xs]].astype(np.float64),
            E[ys, xs].astype(np.float64),
            P3[SRC[ay, ax]].astype(np.float64))


def image_edge(img_bgr, W, H, sigma):
    im = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                 cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    e = np.clip(e / (np.percentile(e, 99.0) + 1e-6), 0, 1)
    return cv2.GaussianBlur(e, (0, 0), sigma) if sigma > 0 else e


# ------------------------------------------------------------------- cost --
class Cost:
    """Masked NCC between the splatted lidar edges and the image edges.

    Blur is linear, so <blur(L), b> == <L, blur(b)>: the numerator is a sample
    of a pre-blurred map instead of a full render per iteration. The mask and
    the norm are refreshed by set_ref() at the start of each stage.
    """

    def __init__(self, pts, wts, allp, img, W, H, sigma):
        self.P, self.w, self.A = pts, wts, allp
        self.W, self.H, self.sigma = W, H, sigma
        self.I = image_edge(img, W, H, 0.0)
        self.normL = self.bnorm = 1.0
        self.blurb = np.zeros_like(self.I)
        self.mask = np.ones_like(self.I, bool)

    def _coverage(self, R, C):
        M = cv2.GaussianBlur(splat((self.A - C) @ R.T, np.ones(len(self.A)),
                                   self.W, self.H), (0, 0), self.sigma)
        pos = M[M > 0]
        return M > (0.15 * pos.mean() if len(pos) else 0.0)

    def set_ref(self, R, C):
        self.mask = self._coverage(R, C)
        b = np.zeros_like(self.I)
        if self.mask.sum() > 100:
            b[self.mask] = self.I[self.mask] - self.I[self.mask].mean()
        self.bnorm = float(np.sqrt((b * b).sum())) + 1e-12
        self.blurb = cv2.GaussianBlur(b, (0, 0), self.sigma)
        L = cv2.GaussianBlur(splat((self.P - C) @ R.T, self.w, self.W, self.H),
                             (0, 0), self.sigma)
        self.normL = float(np.sqrt((L[self.mask] ** 2).sum())) + 1e-9

    def __call__(self, R, C):
        u, v = sph_uv((self.P - C) @ R.T, self.W, self.H)
        return float(np.dot(self.w, bilinear(self.blurb, u, v))) / (self.normL * self.bnorm)

    def exact(self, R, C):
        L = cv2.GaussianBlur(splat((self.P - C) @ R.T, self.w, self.W, self.H),
                             (0, 0), self.sigma)
        mask = self._coverage(R, C)
        if mask.sum() < 500:
            return -1.0
        a = L[mask].astype(np.float64)
        b = self.I[mask].astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


# ------------------------------------------------------------------ prior --
class Prior:
    """Linear guess of the camera relative to the lidar, in rig terms:
    up along gravity, back against the camera forward axis, right along the
    camera right axis. Used only to reject grossly wrong minima, within +/- tol.
    """

    def __init__(self, up=0.18, back=0.07, right=0.0, tol=0.12):
        self.up, self.back, self.right, self.tol = up, back, right, tol

    @staticmethod
    def _frame(R, up_L):
        zh = R[2] - np.dot(R[2], up_L) * up_L
        n = np.linalg.norm(zh)
        zh = zh / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        return zh, np.cross(up_L, zh)

    def decompose(self, C, R, up_L):
        h = float(np.dot(C, up_L))
        Ch = C - h * up_L
        zh, xh = self._frame(R, up_L)
        return h, -float(np.dot(Ch, zh)), float(np.dot(Ch, xh))

    def start_C(self, up_L, R):
        zh, xh = self._frame(R, up_L)
        return self.up * up_L - self.back * zh + self.right * xh

    def penalty(self, C, R, up_L):
        p = 0.0
        for val, ref in zip(self.decompose(C, R, up_L), (self.up, self.back, self.right)):
            d = abs(val - ref) - self.tol
            if d > 0:
                p += 400.0 * d * d
        return p

    def violation(self, C, R, up_L):
        """Worst overshoot beyond the tolerance box, metres (0 = inside)."""
        h, b, s = self.decompose(C, R, up_L)
        return max(0.0, max(abs(h - self.up), abs(b - self.back),
                            abs(s - self.right)) - self.tol)


# --------------------------------------------------------------- data set --
class DataSet:
    """One static capture: a bag and the matching 360 clip."""

    def __init__(self, idx, bag, video):
        self.idx = idx
        self.bag = bag
        self.video = video
        self.name = os.path.splitext(os.path.basename(bag))[0]
        self.warnings = []
        self.rejected = False
        self.reject_reason = ""
        self.solo = None
        self.solo_score = 0.0
        self.dev_ang = self.dev_lin = 0.0
        self.img = self.img_tilt = self.img_zenith = None
        self.img_zenith_inliers = 0

    def prepare(self, log, progress, cache=True, frame_pos=0.5):
        info = video_info(self.video)
        bad = check_video_equirect(info)
        if bad:
            raise DataError(bad)
        log("   video  %dx%d   %.1f s @ %.0f fps"
            % (info["W"], info["H"], info["duration"], info["fps"]))

        cpath = os.path.splitext(self.bag)[0] + ".calib_cache.npz"
        if cache and os.path.exists(cpath) and os.path.getmtime(cpath) > os.path.getmtime(self.bag):
            z = np.load(cpath)
            P, imu_a, imu_g = z["P"], z["imu_a"], z["imu_g"]
            log("   cloud from cache: %d points" % len(P))
        else:
            log("   reading bag ...")
            d = read_bag(self.bag, progress=lambda f: progress(0.05 + 0.30 * f))
            P, imu_a, imu_g = d["points"], d["imu_a"], d["imu_g"]
            if cache:
                try:
                    np.savez_compressed(cpath, P=P, imu_a=imu_a, imu_g=imu_g,
                                        dur=d["duration"])
                except OSError:
                    pass
            log("   cloud: %d points, %.1f s" % (len(P), d["duration"]))

        ok, msg = check_static(imu_a, imu_g)
        log("   " + msg)
        if not ok:
            self.warnings.append("rig was moving during the scan")

        rng = np.linalg.norm(P[:, :3], axis=1)
        log("   range: median %.1f m, p95 %.1f m, max %.1f m"
            % (np.median(rng), np.percentile(rng, 95), rng.max()))
        if np.percentile(rng, 95) < 3.0:
            self.warnings.append("no distant geometry - angles weakly constrained")
        if np.percentile(rng, 5) > 3.0:
            self.warnings.append("no close geometry - translation weakly constrained")

        self.up = unit(imu_a.mean(0))
        self.Rlev = level_rot(self.up)
        progress(0.45)

        log("   extracting frame at %.0f%% of the clip ..." % (100 * frame_pos))
        self.img = grab_frame(self.video, float(np.clip(frame_pos, 0.02, 0.98)))
        try:
            if video_frame_spread(self.video) > 12.0:
                self.warnings.append("video content changes a lot "
                                     "(moving rig or dynamic scene)")
        except DataError:
            pass
        try:
            drift = zenith_drift(self.video)
            if drift is not None and drift > 0.8:
                self.warnings.append("scene horizon drifts %.1f deg across the clip "
                                     "- stabilisation is likely on" % drift)
        except DataError:
            pass

        z = zenith_from_image(self.img)
        if z:
            self.img_zenith, self.img_zenith_inliers = z[0], z[2]
            self.img_tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(z[0][1])))))
            log("   scene verticals: %d lines, camera tilt %.2f deg" % (z[2], self.img_tilt))
        else:
            self.warnings.append("no reliable vertical lines in the image")
        progress(0.55)

        log("   building lidar edges ...")
        self.pts, self.wts, self.allp = build_lidar_edges(
            P[:, :3].astype(np.float64), P[:, 3], self.up)
        log("   %d edge points" % len(self.pts))
        progress(0.7)

    def cost(self, W, H, sigma, sub):
        pts, wts = self.pts, self.wts
        if sub and len(pts) > sub:
            i = np.random.default_rng(1).choice(len(pts), sub, replace=False)
            pts, wts = pts[i], wts[i]
        allp = self.allp
        if len(allp) > 150000:
            allp = allp[np.random.default_rng(2).choice(len(allp), 150000, replace=False)]
        return Cost(pts, wts, allp, self.img, W, H, sigma)

    def zenith_residual(self, x):
        """Angle between the scene zenith and the IMU gravity mapped into the
        camera frame by this extrinsic."""
        if self.img_zenith is None:
            return None
        pred = unit(mkR(np.asarray(x[:3]), self.Rlev) @ self.up)
        return math.degrees(math.acos(max(-1.0, min(1.0,
                                                    float(np.dot(pred, self.img_zenith))))))


# ------------------------------------------------------------------ solve --
def yaw_scan(ds, prior, W=1024, H=512, sigma=2.5, step=1.0):
    """Full-circle heading search on a single set. The coverage mask rotates
    with the cloud, so the NCC is evaluated exactly at every angle."""
    c = ds.cost(W, H, sigma, 120000)
    curve = []
    for psi in np.arange(-180, 180, step):
        R = mkR(np.array([0.0, math.radians(psi), 0.0]), ds.Rlev)
        curve.append((psi, c.exact(R, prior.start_C(ds.up, R))))
    curve = np.array(curve)
    return float(curve[np.argmax(curve[:, 1]), 0]), curve


def zenith_basis(d):
    """Rotation taking the measured scene zenith onto the image up axis."""
    z = getattr(d, "img_zenith", None)
    if z is None or getattr(d, "img_zenith_inliers", 0) < ZENITH_MIN_INLIERS:
        return None
    a, b = unit(z), np.array([0.0, -1.0, 0.0])
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _orientation_barrier(rv, datasets, anchor=None):
    """Keep the rotation physically plausible. Both terms have a dead zone, so
    they never bias a good solution - they only stop the search wandering."""
    pen = 0.0
    for d in datasets:
        pred = R_for(rv, d, anchor) @ d.up
        n = np.linalg.norm(pred)
        if n < 1e-9:
            continue
        pred = pred / n
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -float(pred[1])))))
        over = tilt - MAX_CAM_TILT_DEG
        if over > 0:
            pen += 0.02 * over * over
        z = getattr(d, "img_zenith", None)
        if z is not None and getattr(d, "img_zenith_inliers", 0) >= ZENITH_MIN_INLIERS:
            a = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(pred, z))))))
            over = a - ZENITH_DEADZONE_DEG
            if over > 0:
                pen += ZENITH_WEIGHT * over * over
    return pen / max(len(datasets), 1)


def optimise_zenith(datasets, theta0, C0, prior, stages, anchor, log=None, tag=""):
    """Four parameters: heading plus lever, with the camera tilt pinned by the
    scene verticals. With all three angles free the optimiser trades tilt
    against the lever arm and slides to the edge of the prior box."""
    B = zenith_basis(anchor)
    if B is None:
        return None, -1.0

    def R_of(theta):
        return B.T @ Rot.from_rotvec([0.0, float(theta), 0.0]).as_matrix() @ anchor.Rlev

    x = np.r_[float(theta0), np.asarray(C0, float)]
    score = -1.0
    for si, (W, H, sg, sub) in enumerate(stages):
        cs = [(d, d.cost(W, H, sg, sub)) for d in datasets]
        for _ in range(2):
            for d, c in cs:
                c.set_ref(R_of(x[0]), x[1:4])

            def f(p):
                R = R_of(p[0])
                s = sum(c(R, p[1:4]) for _, c in cs) / len(cs)
                return -s + prior.penalty(p[1:4], R, anchor.up)

            astep = math.radians(1.5 if si == 0 else 0.5)
            tstep = 0.025 if si == 0 else 0.010
            simplex = np.vstack([x] + [x + np.eye(4)[i] * (astep if i == 0 else tstep)
                                       for i in range(4)])
            x = minimize(f, x, method="Nelder-Mead",
                         options=dict(xatol=2e-6, fatol=1e-10, maxfev=1200,
                                      initial_simplex=simplex)).x
        score = float(np.mean([c.exact(R_of(x[0]), x[1:4]) for _, c in cs]))
        if log:
            h, b, r = prior.decompose(x[1:4], R_of(x[0]), anchor.up)
            log("     %s[%dx%d] score=%.4f theta=%+.2f rig=%.1f/%.1f/%.1f cm"
                % (tag, W, H, score, math.degrees(x[0]), 100 * h, 100 * b, 100 * r))
    rv = Rot.from_matrix(R_of(x[0]) @ anchor.Rlev.T).as_rotvec()
    return np.r_[rv, x[1:4]], score


def optimise(datasets, x0, prior, stages, log=None, progress=None, tag="", anchor=None):
    """Six parameters over the given sets. With anchor set, one rotation matrix
    is shared by every set."""
    x = np.array(x0, float)
    score = -1.0
    ref = anchor if anchor is not None else datasets[0]
    for si, (W, H, sg, sub) in enumerate(stages):
        cs = [(d, d.cost(W, H, sg, sub)) for d in datasets]
        rs = math.radians(2.0 if si == 0 else 0.7)
        ts = 0.03 if si == 0 else 0.012
        for _ in range(2):
            for d, c in cs:
                c.set_ref(R_for(x[:3], d, anchor), x[3:6])

            def f(p):
                s = sum(c(R_for(p[:3], d, anchor), p[3:6]) for d, c in cs) / len(cs)
                return (-s + prior.penalty(p[3:6], R_for(p[:3], ref, anchor), ref.up)
                        + _orientation_barrier(p[:3], datasets, anchor))

            # coordinate descent first: Nelder-Mead alone is unreliable here
            best = f(x)
            for step in (rs, rs / 3.0):
                improved = True
                while improved:
                    improved = False
                    for i in range(6):
                        h = step if i < 3 else (ts if step == rs else ts / 3.0)
                        for delta in (h, -h):
                            y = x.copy()
                            y[i] += delta
                            v = f(y)
                            if v < best - 1e-7:
                                best, x, improved = v, y, True
            simplex = np.vstack([x] + [x + np.eye(6)[i] * (rs if i < 3 else ts)
                                       for i in range(6)])
            x = minimize(f, x, method="Nelder-Mead",
                         options=dict(xatol=2e-6, fatol=1e-10, maxiter=3000,
                                      maxfev=3000, initial_simplex=simplex)).x
        score = float(np.mean([c.exact(R_for(x[:3], d, anchor), x[3:6]) for d, c in cs]))
        if log:
            log("     %s[%dx%d]  score=%.4f  rot=%s  C=%s"
                % (tag, W, H, score, np.rad2deg(x[:3]).round(2), np.round(x[3:6], 4)))
        if progress:
            progress((si + 1) / len(stages))
    return x, score


def to_matrices(x, ds_list, anchor=None):
    """Return R_camera_lidar, the camera centre in lidar coords and the 4x4."""
    if anchor is not None:
        R = mkR(np.asarray(x[:3]), anchor.Rlev)
    else:
        Rs = [mkR(np.asarray(x[:3]), d.Rlev) for d in ds_list]
        R = Rot.from_matrix(np.stack(Rs)).mean().as_matrix() if len(Rs) > 1 else Rs[0]
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    C = np.asarray(x[3:6], float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C
    return R, C, T


def build_json(R, C, T, sets, stats, prior):
    e = Rot.from_matrix(R).as_euler("ZYX", degrees=True)
    per = {}
    for d in sets:
        per[d.name] = {
            "bag": d.bag,
            "video": d.video,
            "C_lidar_m": [float(v) for v in d.solo[3:6]] if d.solo is not None else None,
            "score": float(d.solo_score),
            "dev_from_consensus_deg": float(d.dev_ang),
            "dev_from_consensus_m": float(d.dev_lin),
            "status": "rejected: " + d.reject_reason if d.rejected else "accepted",
            "warnings": list(d.warnings),
        }
    return {
        "T_base_camera_lidar": [[float(v) for v in row] for row in T],
        "_determined": {
            "method": "edge + reflectivity NCC alignment on static scans, "
                      "equirectangular camera model, tilt constrained by scene verticals",
            "handeye_resid_deg": float(stats.get("spread_deg", 0.0)),
            "lever_arm_cam_in_lidar_m": [float(v) for v in C],
            "refine_deg": [float(v) for v in np.rad2deg(Rot.from_matrix(R).as_rotvec())],
            "drift_from_imu_deg": float(stats.get("zenith_resid_deg", 0.0)),
            "_field_notes": {
                "handeye_resid_deg": "not hand-eye: angular spread of the accepted sets "
                                     "around the consensus, an upper bound on the error",
                "refine_deg": "rotation vector of R_camera_lidar, degrees",
                "drift_from_imu_deg": "mean angle between the scene zenith and the IMU "
                                      "gravity mapped through this extrinsic",
                "lever_arm_cam_in_lidar_m": "camera centre in the LIDAR frame, not the rig "
                                            "frame; t = -R @ lever_arm",
            },
        },
        "_calibration": {
            "tool": "Lidar-Camera calibrator",
            "convention": {
                "camera_axes": "X right, Y down, Z forward (OpenCV)",
                "equirect": "u=(atan2(X,Z)/(2*pi)+0.5)*W ; v=(0.5-asin(-Y/r)/pi)*H",
                "lidar_axes": "sensor body frame",
                "transform": "X_cam = R @ X_lidar + t   (R=T[:3,:3], t=T[:3,3])",
            },
            "rig_offsets_m": stats.get("rig", {}),
            "euler_ZYX_deg": {"yaw": float(e[0]), "pitch": float(e[1]), "roll": float(e[2])},
            "quaternion_xyzw": [float(v) for v in Rot.from_matrix(R).as_quat()],
            "T_lidar_camera": [[float(v) for v in row] for row in np.linalg.inv(T)],
            "prior_used_m": {"up": prior.up, "back": prior.back, "right": prior.right,
                             "tolerance": prior.tol},
            "consensus": {
                "sets_accepted": stats.get("n_ok", 0),
                "sets_rejected": stats.get("n_bad", 0),
                "angular_spread_deg": float(stats.get("spread_deg", 0.0)),
                "linear_spread_m": float(stats.get("spread_lin", 0.0)),
                "heading_margin_sigma": float(stats.get("joint_margin", 0.0)),
                "contradiction": bool(stats.get("contradiction", False)),
                "cross_check": stats.get("cross_check"),
                "uncertainty_hint": stats.get("uncertainty", ""),
            },
            "per_set": per,
        },
    }
