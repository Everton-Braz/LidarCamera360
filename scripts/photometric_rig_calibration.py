#!/usr/bin/env python3
"""Photometric LiDAR <-> Insta360 rig calibration (multi-view colour consistency).

Why this exists
---------------
Edge/SfM based rig calibration failed on PIER (Sim(3) RMSE 8.9 m, lever arms of
54-130 cm; the sample-SfM result has a 19-25 cm lever arm and is slightly worse
than the hardware profile).  The dominant error on PIER is not the mount but the
*time offset*: the gyro cross-correlation gives dt = -7.196 s, while the images
say dt = -7.027 s (SLAM-trajectory reference).  170 ms at 60-70 deg/s turns is a
10-12 deg pointing error, i.e. metres of colour smear at 10 m.

Two clock references are mixed in the current code: ``auto_sync_imu_gyro``
measures dt from the first bag IMU packet (PIER: 0.346 s before the first SLAM
pose) and the colourisers apply it from the first SLAM pose.  Correcting only
that (dt = -6.850 s) makes PIER *worse*, so there is a second, opposite
camera-side offset (video PTS vs INSV gyro/exposure clock) of roughly -0.18 s.
Neither can be trusted blindly, so dt is refined here from the images.

Method (Mandate 1 of SPEC-LC360-ARCH-2026-V4, E_photo + E_prior)
-----------------------------------------------------------------
* Sample ~250k LiDAR points near the trajectory.
* Project them into N frames of both lenses with the full THIN_PRISM_FISHEYE
  model (atan2 formulation, valid beyond 180 deg).
* Visibility (z-buffer, person masks, fisheye circle, theta < 96 deg) is
  frozen per stage so the objective is smooth.
* Cost = mean Cauchy(per-point deviation of exposure-normalised grey value from
  the multi-view mean).  Wrong extrinsics / dt => the same 3D point gets
  different colours in different frames => high cost.  This directly measures
  ghosting / double lines rather than edge proximity.
* 1-D dense dt scan first (the most sensitive parameter, wide basin), then
  coarse-to-fine Powell over [rotvec_C0(3), t_C0(3), rotvec_C0C1(3), dt]
  with a Gaussian prior on the hardware profile.
* Output JSON keeps the app schema (``T_lidar_to_cam{0,1}_rigid_4x4``,
  ``dt_sync_seconds`` in the SLAM-trajectory reference) plus a quality report.

Usage
-----
python scripts/photometric_rig_calibration.py --dataset D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output \
    --init configs/profiles/raven_x4_montagem_1.json --dt-init -7.196 --dt-init-reference imu \
    --masks D:/ARQUIVOS_TESTE_2/PIER/LIDAR_20261020115929_studio_output/masks
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

INTR_KEYS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')


# ----------------------------------------------------------------------------- io
def load_pcd(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii').strip().split()
            if not line or line[0].startswith('#'):
                continue
            header[line[0]] = line[1:]
            if line[0] == 'DATA':
                break
        if header['DATA'] != ['binary']:
            raise ValueError('Expected binary PCD')
        names = header['FIELDS']; sizes = list(map(int, header['SIZE'])); types = header['TYPE']
        tmap = {'F': 'f', 'I': 'i', 'U': 'u'}
        dtype = np.dtype([(n, '<' + tmap[t] + str(s)) for n, s, t in zip(names, sizes, types)])
        data = np.fromfile(f, dtype=dtype, count=int(header['POINTS'][0]))
    xyz = np.column_stack([data['x'], data['y'], data['z']]).astype(np.float64)
    return xyz[np.isfinite(xyz).all(1)]


SLAM_DIR_OVERRIDE = None


def find_slam_dir(dataset: Path) -> Path:
    if SLAM_DIR_OVERRIDE is not None:
        return Path(SLAM_DIR_OVERRIDE)
    active = dataset / 'active_slam.json'
    if active.is_file():
        try:
            d = json.loads(active.read_text(encoding='utf-8'))
            for k in ('relative_directory', 'path', 'slam_dir', 'directory', 'name'):
                if k in d:
                    p = Path(d[k]); p = p if p.is_absolute() else dataset / p
                    if p.is_dir():
                        return p
        except Exception:
            pass
    cands = sorted(dataset.glob('slam_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    cands = [c for c in cands if (c / 'pcd' / 'all_raw_points.pcd').is_file()]
    if not cands:
        raise FileNotFoundError(f'No slam_* directory with pcd/all_raw_points.pcd in {dataset}')
    return cands[0]


def load_trajectory(slam_dir: Path):
    res = slam_dir / 'result'
    txts = sorted(res.glob('*.txt'))
    if not txts:
        raise FileNotFoundError(f'No trajectory in {res}')
    d = np.loadtxt(txts[0])
    return d[:, 0], d[:, 1:4], Rot.from_quat(d[:, 4:8])


def load_calibration(path: Path):
    c = json.loads(Path(path).read_text(encoding='utf-8'))
    K0 = tuple(float(c['cam0_front_intrinsics'][k]) for k in INTR_KEYS)
    K1 = tuple(float(c['cam1_rear_intrinsics'][k]) for k in INTR_KEYS)
    return c, np.array(c['T_lidar_to_cam0_rigid_4x4'], float), np.array(c['T_lidar_to_cam1_rigid_4x4'], float), K0, K1


# ----------------------------------------------------------------------- geometry
def project_thin_prism(P, params):
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    r = np.hypot(P[:, 0], P[:, 1])
    theta = np.arctan2(r, P[:, 2])
    scale = np.divide(theta, r, out=np.zeros_like(theta), where=r > 1e-12)
    xd, yd = P[:, 0] * scale, P[:, 1] * scale
    rd2 = theta ** 2
    radial = 1 + rd2 * (k1 + rd2 * (k2 + rd2 * (k3 + rd2 * k4)))
    du = 2 * p1 * xd * yd + p2 * (rd2 + 2 * xd ** 2) + sx1 * rd2
    dv = p1 * (rd2 + 2 * yd ** 2) + 2 * p2 * xd * yd + sy1 * rd2
    return fx * (xd * radial + du) + cx, fy * (yd * radial + dv) + cy, theta


def perturb(T, x6):
    D = np.eye(4); D[:3, :3] = Rot.from_rotvec(x6[:3]).as_matrix(); D[:3, 3] = x6[3:6]
    return T @ D


# ------------------------------------------------------------------------ problem
class PhotometricRigProblem:
    """x = [rotvec_C0(3), t_C0(3), rotvec_C0C1(3), d_dt]  (perturbations, camera frame)."""

    def __init__(self, dataset, T0, T1, K0, K1, dt0, frame_names, masks_dir=None,
                 scale=0.25, voxel=0.04, npts=250000, max_range=18.0, seed=0, log=print):
        self.dataset = Path(dataset); self.T0 = T0; self.T01 = np.linalg.inv(T0) @ T1
        self.K = {'cam0': K0, 'cam1': K1}; self.dt0 = dt0; self.scale = scale; self.log = log
        manifest = json.loads((self.dataset / 'images' / 'frames.json').read_text(encoding='utf-8'))
        self.ft = manifest['timestamps']
        slam = find_slam_dir(self.dataset)
        self.t, self.p, self.R = load_trajectory(slam); self.slerp = Slerp(self.t, self.R)
        names = [n for n in frame_names if n in self.ft]
        tq = np.array([self.t[0] + self.ft[n] - dt0 for n in names])
        inside = (tq > self.t[0] + 0.2) & (tq < self.t[-1] - 0.2)
        self.names = [n for n, ok in zip(names, inside) if ok]
        log(f'[*] {len(self.names)} frames inside the SLAM time window')
        xyz = load_pcd(slam / 'pcd' / 'all_raw_points.pcd')
        key = np.floor(xyz / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True); xyz = xyz[idx]
        pos, _ = self.pose(tq[inside])
        d, _ = cKDTree(pos).query(xyz)
        idx = np.flatnonzero(d < max_range)
        if len(idx) > npts:
            idx = np.random.default_rng(seed).choice(idx, npts, replace=False)
        self.X = xyz[idx]
        log(f'[*] {len(self.X):,} LiDAR sample points')
        self.raw = {}
        for n in self.names:
            img = cv2.imread(str(self.dataset / 'images' / n), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
            H, W = img.shape
            mk = np.ones((H, W), bool)
            if masks_dir is not None:
                mk = self._mask(Path(masks_dir), n, (W, H))
            yy, xx = np.mgrid[0:H, 0:W]
            mk &= np.hypot(xx - W / 2, yy - H / 2) < 0.985 * W / 2
            mk = cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            self.raw[n] = (img, mk)
        self.set_blur(0)

    def _mask(self, masks_dir, name, size):
        """Person masks (white=keep). Matches by timestamp when the mask set has its own
        frames.json (names are not comparable across extraction fps), else by name."""
        cam, fn = name.split('/')
        p = masks_dir / cam / (fn + '.png')
        man = masks_dir.parent / 'images' / 'frames.json'
        if man.is_file():
            ts = json.loads(man.read_text(encoding='utf-8'))['timestamps']
            cands = [(abs(v - self.ft[name]), k) for k, v in ts.items() if k.startswith(cam + '/')]
            p = None
            if cands:
                dtm, k = min(cands)
                if dtm < 0.05:
                    p = masks_dir / (k + '.png')
            if p is None:
                return np.ones(size[::-1], bool)
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.is_file() else None
        if m is None:
            return np.ones(size[::-1], bool)
        return cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) > 127

    def pose(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])
        p = np.column_stack([np.interp(tq, self.t, self.p[:, k]) for k in range(3)])
        return p, self.slerp(tq).as_matrix()

    def set_blur(self, sigma):
        self.img = {}
        for n, (im, mk) in self.raw.items():
            g = cv2.GaussianBlur(im, (0, 0), sigma) if sigma > 0 else im
            v = g[mk]
            self.img[n] = (((g - v.mean()) / (v.std() + 1e-6)).astype(np.float32), mk)

    def extrinsics(self, x):
        T0 = perturb(self.T0, x[:6])
        T01 = perturb(self.T01, np.r_[x[6:9], 0, 0, 0])
        return T0, T0 @ T01, self.dt0 + x[9]

    def _project(self, n, x, idx=None):
        T0, T1, dt = self.extrinsics(x)
        cam = n.split('/')[0]; T = T0 if cam == 'cam0' else T1
        pL, RL = self.pose(np.array([self.t[0] + self.ft[n] - dt]))
        pc = pL[0] + RL[0] @ T[:3, 3]; Rwc = RL[0] @ T[:3, :3]
        X = self.X if idx is None else self.X[idx]
        P = (X - pc) @ Rwc
        u, v, th = project_thin_prism(P, self.K[cam])
        return u * self.scale, v * self.scale, th, np.linalg.norm(P, axis=1)

    def freeze_visibility(self, x, theta_max_deg=96.0, zcell=4):
        self.obs = []
        for n in self.names:
            u, v, th, d = self._project(n, x)
            img, mk = self.img[n]; H, W = img.shape
            ok = (th < np.deg2rad(theta_max_deg)) & (d > 1.0) & (u >= 1) & (v >= 1) & (u < W - 2) & (v < H - 2)
            ii = np.flatnonzero(ok); ui, vi = u[ii].astype(int), v[ii].astype(int)
            k = mk[vi, ui]; ii, ui, vi = ii[k], ui[k], vi[k]
            cell = (vi // zcell) * (W // zcell + 1) + ui // zcell
            zb = np.full(cell.max() + 1 if len(cell) else 1, np.inf); np.minimum.at(zb, cell, d[ii])
            self.obs.append(ii[d[ii] <= zb[cell] * 1.03 + 0.05])
        cnt = np.bincount(np.concatenate(self.obs), minlength=len(self.X))
        self.obs = [o[cnt[o] >= 3] for o in self.obs]
        return int((cnt >= 3).sum()), int(sum(len(o) for o in self.obs))

    def residuals(self, x):
        pid, val = [], []
        for n, ii in zip(self.names, self.obs):
            u, v, _, _ = self._project(n, x, ii)
            img, _ = self.img[n]; H, W = img.shape
            ok = (u >= 0) & (v >= 0) & (u < W - 1) & (v < H - 1)
            u, v, ii = u[ok], v[ok], ii[ok]
            x0 = u.astype(np.int32); y0 = v.astype(np.int32)
            ax = (u - x0).astype(np.float32); ay = (v - y0).astype(np.float32)
            fl = img.ravel(); k = y0 * W + x0
            s = (fl[k] * (1 - ax) + fl[k + 1] * ax) * (1 - ay) + (fl[k + W] * (1 - ax) + fl[k + W + 1] * ax) * ay
            pid.append(ii); val.append(s)
        pid = np.concatenate(pid); val = np.concatenate(val); N = len(self.X)
        cnt = np.bincount(pid, minlength=N)
        mean = np.bincount(pid, val, minlength=N) / np.maximum(cnt, 1)
        return (val - mean[pid]) * np.sqrt(cnt[pid] / np.maximum(cnt[pid] - 1, 1))

    def cost(self, x, c=0.5):
        r = self.residuals(x)
        return float(np.mean(np.log1p((r / c) ** 2)))


def prior_cost(x, sig_rot_deg=3.0, sig_t_m=0.05, sig_c01_deg=1.0, weight=1e-3):
    z = np.r_[x[:3] / np.deg2rad(sig_rot_deg), x[3:6] / sig_t_m, x[6:9] / np.deg2rad(sig_c01_deg)]
    return weight * float(z @ z)


# --------------------------------------------------------------------------- main
def calibrate(args, log=print):
    dataset = Path(args.dataset)
    raw, T0, T1, K0, K1 = load_calibration(args.init)
    t_slam0 = None
    dt0 = float(args.dt_init)
    if args.dt_init_reference == 'imu':
        # Convert gyro-sync dt (relative to first bag IMU packet) to the SLAM-pose reference
        bag_imu0 = args.bag_imu_t0
        slam = find_slam_dir(dataset); t, _, _ = load_trajectory(slam)
        if bag_imu0 is None:
            bag_imu0 = first_bag_imu_stamp(args.bag) if args.bag else None
        if bag_imu0 is not None:
            log(f'[*] dt reference: IMU0 -> SLAM0 shift {t[0] - bag_imu0:+.4f}s')
            dt0 = dt0 + (t[0] - bag_imu0)
    names_all = sorted(n for n in json.loads((dataset / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
                       if (dataset / 'images' / n).is_file())
    by_cam = {c: [n for n in names_all if n.startswith(c + '/')] for c in ('cam0', 'cam1')}
    step = max(1, len(by_cam['cam0']) // args.frames)
    names = [n for c in ('cam0', 'cam1') for n in by_cam[c][::step]]
    pb = PhotometricRigProblem(dataset, T0, T1, K0, K1, dt0, names, masks_dir=args.masks, log=log)
    x = np.zeros(10)
    report = {'dt_init_slam_reference': dt0, 'stages': []}

    # --- stage 0: dense dt scan (widest basin, most sensitive parameter)
    pb.set_blur(4.0); pb.freeze_visibility(x)
    grid = np.arange(-args.dt_window, args.dt_window + 1e-9, 0.02)
    scan = []
    for g in grid:
        xx = x.copy(); xx[9] = g; scan.append(pb.cost(xx))
    scan = np.array(scan); x[9] = grid[int(np.argmin(scan))]
    report['dt_scan'] = {'offsets': grid.round(3).tolist(), 'cost': scan.round(5).tolist()}
    log(f'[*] dt scan: best dt = {dt0 + x[9]:.4f}s (cost {scan.min():.4f} vs {scan[len(grid)//2]:.4f} at init)')

    sc = np.r_[np.full(3, np.deg2rad(1.0)), np.full(3, 0.02), np.full(3, np.deg2rad(0.5)), 0.02]
    for sigma in args.blur:
        pb.set_blur(sigma); npts, nobs = pb.freeze_visibility(x)
        f0 = pb.cost(x); t0 = time.time()
        res = minimize(lambda z: pb.cost(z * sc) + prior_cost(z * sc), x / sc, method='Powell',
                       options=dict(xtol=1e-2, ftol=1e-5, maxfev=args.maxfev))
        x = res.x * sc
        st = dict(sigma=sigma, points=npts, observations=nobs, cost_before=f0, cost_after=float(res.fun),
                  nfev=int(res.nfev), seconds=round(time.time() - t0, 1),
                  rot_deg=np.rad2deg(x[:3]).round(4).tolist(), trans_cm=(x[3:6] * 100).round(3).tolist(),
                  c01_rot_deg=np.rad2deg(x[6:9]).round(4).tolist(), dt=float(dt0 + x[9]))
        report['stages'].append(st); log(f'[*] stage sigma={sigma}: {f0:.4f} -> {res.fun:.4f}  dt={dt0 + x[9]:.4f}')

    # final metric: raw (unblurred) photometric consistency, init vs final
    pb.set_blur(0.0); pb.freeze_visibility(x)
    r_fin = pb.residuals(x); r_ini = pb.residuals(np.r_[np.zeros(9), 0.0])
    report['mv_residual_std_init'] = float(np.std(r_ini)); report['mv_residual_std_final'] = float(np.std(r_fin))
    T0n, T1n, dtn = pb.extrinsics(x)
    out = dict(raw)
    out.update({
        'T_lidar_to_cam0_rigid_4x4': T0n.tolist(), 'T_lidar_to_cam1_rigid_4x4': T1n.tolist(),
        'dt_sync_seconds': float(dtn), 'dt_sync_reference': 'slam_trajectory_first_pose',
        'lever_arm_cam0_meters': dict(zip(('dx', 'dy', 'dz'), T0n[:3, 3].tolist()),
                                      norm_distance_cm=float(np.linalg.norm(T0n[:3, 3]) * 100)),
        'lever_arm_cam1_meters': dict(zip(('dx', 'dy', 'dz'), T1n[:3, 3].tolist()),
                                      norm_distance_cm=float(np.linalg.norm(T1n[:3, 3]) * 100)),
        'relative_lens_angle_deg': float(np.rad2deg(np.linalg.norm(Rot.from_matrix(T0n[:3, :3].T @ T1n[:3, :3]).as_rotvec()))),
        'transform_convention': 'camera_to_trajectory_body',
        'calibration_source': 'photometric_multiview_consistency',
        'calibration_init': str(args.init),
        'quality_status': 'accepted' if report['mv_residual_std_final'] < report['mv_residual_std_init'] else 'rejected',
        'photometric_report': report,
    })
    return out


def first_bag_imu_stamp(bag):
    from rosbags.highlevel import AnyReader
    with AnyReader([Path(bag)]) as r:
        conns = [c for c in r.connections if 'imu' in c.topic.lower()]
        for c, _, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='Output dir with images/, images/frames.json and slam_*/')
    ap.add_argument('--init', required=True, help='Initial calibration JSON (hardware profile)')
    ap.add_argument('--dt-init', type=float, required=True, help='Initial time offset (gyro sync)')
    ap.add_argument('--dt-init-reference', choices=('slam', 'imu'), default='slam',
                    help="'imu' if dt-init is relative to the first bag IMU packet (workflow.auto_sync_imu_gyro)")
    ap.add_argument('--bag', help='LiDAR .bag (to read the first IMU stamp when --dt-init-reference imu)')
    ap.add_argument('--bag-imu-t0', type=float, help='First IMU header stamp (epoch s), avoids reading the bag')
    ap.add_argument('--slam-dir', help='Override SLAM directory (default: active_slam.json / newest slam_*)')
    ap.add_argument('--masks', help='Person mask dir (<cam>/<frame>.jpg.png, white=keep)')
    ap.add_argument('--frames', type=int, default=40, help='Frames per lens used')
    ap.add_argument('--dt-window', type=float, default=0.6)
    ap.add_argument('--blur', type=float, nargs='+', default=[4.0, 2.0, 1.0])
    ap.add_argument('--maxfev', type=int, default=500)
    ap.add_argument('--out', help='Output JSON (default <dataset>/rig_calibration_photometric.json)')
    args = ap.parse_args()
    global SLAM_DIR_OVERRIDE
    SLAM_DIR_OVERRIDE = args.slam_dir
    out = calibrate(args)
    dst = Path(args.out) if args.out else Path(args.dataset) / 'rig_calibration_photometric.json'
    dst.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'[+] wrote {dst}  (dt={out["dt_sync_seconds"]:.4f}s, status={out["quality_status"]})')


if __name__ == '__main__':
    main()
