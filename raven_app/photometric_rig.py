"""Photometric refinement of the LiDAR <-> dual-fisheye rig and time offset.

The hardware profile is a strong prior (a rigid mount does not change between
captures) but the per-capture time offset and small mount deviations decide whether
colours land on the right surface.  This refines [cam0 rotation, cam0 position,
cam0->cam1 rotation, dt] by multi-view colour consistency: wrong extrinsics or dt
make the same LiDAR point read different grey values in different frames.

* ~250k LiDAR points near the trajectory, projected with the THIN_PRISM_FISHEYE
  model (atan2 form, valid past 90 deg) into N frames per lens.
* Visibility (z-buffer, person masks, fisheye circle, theta < 96 deg) is frozen per
  blur stage so the cost is smooth.
* Cost: mean Cauchy deviation of each point's exposure-normalised grey value from
  its multi-view mean, plus a weak prior towards the profile.
* Dense dt scan first (the most sensitive parameter), then coarse-to-fine Powell.
* Acceptance: the result must lower the cost on held-out frames (never used by the
  fit) and pass the rig plausibility gates against the hardware profile.

On PIER this cut the held-out cost by 30 % (dt -7.196 -> -7.027 s, 1.7 deg yaw);
on HarmonyMall by 21 % versus the SfM rig the workflow used to pick.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

from raven_app.rig_gates import load_reference, rig_gate_failures
from raven_app.slam_source import active_slam_dir

INTR_KEYS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')
REFINEMENT_VERSION = 1
MIN_HOLDOUT_IMPROVEMENT = 0.01


def project_thin_prism(P, params):
    """THIN_PRISM_FISHEYE projection using theta = atan2(r, z) (valid beyond 90 deg)."""
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
    D = np.eye(4)
    D[:3, :3] = Rot.from_rotvec(x6[:3]).as_matrix()
    D[:3, 3] = x6[3:6]
    return T @ D


def supported_calibration(calib):
    """Return (T0, T1, K0, K1) for a dual THIN_PRISM_FISHEYE calibration, else None."""
    try:
        K0 = tuple(float(calib['cam0_front_intrinsics'][k]) for k in INTR_KEYS)
        K1 = tuple(float(calib['cam1_rear_intrinsics'][k]) for k in INTR_KEYS)
        T0 = np.asarray(calib['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
        T1 = np.asarray(calib['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if T0.shape != (4, 4) or T1.shape != (4, 4) or min(K0[11], K1[11]) < -997.0:
        return None
    if calib.get('lens_model', 'THIN_PRISM_FISHEYE') != 'THIN_PRISM_FISHEYE':
        return None
    return T0, T1, K0, K1


class PhotometricRigProblem:
    """x = [rotvec_C0(3), t_C0(3), rotvec_C0C1(3), d_dt]  (perturbations, camera frame)."""

    def __init__(self, dataset, T0, T1, K0, K1, dt0, frame_names, *, points, trajectory,
                 masks_dir=None, scale=0.25, voxel=0.04, npts=250000, max_range=18.0, seed=0, log=print):
        self.dataset = Path(dataset)
        self.T0 = T0
        self.T01 = np.linalg.inv(T0) @ T1
        self.K = {'cam0': K0, 'cam1': K1}
        self.dt0 = dt0
        self.scale = scale
        self.ft = json.loads((self.dataset / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
        self.t, self.p, rotations = trajectory
        self.slerp = Slerp(self.t, rotations)
        names = [n for n in frame_names if n in self.ft]
        tq = np.array([self.t[0] + self.ft[n] - dt0 for n in names])
        inside = (tq > self.t[0] + 0.2) & (tq < self.t[-1] - 0.2)
        self.names = [n for n, ok in zip(names, inside) if ok]
        key = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        xyz = points[idx]
        pos, _ = self.pose(tq[inside])
        distance, _ = cKDTree(pos).query(xyz)
        idx = np.flatnonzero(distance < max_range)
        if len(idx) > npts:
            idx = np.random.default_rng(seed).choice(idx, npts, replace=False)
        self.X = xyz[idx]
        log(f'[*] Photometric rig problem: {len(self.names)} frames, {len(self.X):,} LiDAR points')
        self.raw = {}
        for n in self.names:
            img = cv2.imread(str(self.dataset / 'images' / n), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
            H, W = img.shape
            mk = np.ones((H, W), bool)
            if masks_dir is not None:
                m = cv2.imread(str(Path(masks_dir) / (n + '.png')), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mk = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
            yy, xx = np.mgrid[0:H, 0:W]
            mk &= np.hypot(xx - W / 2, yy - H / 2) < 0.985 * W / 2
            mk = cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            self.raw[n] = (img, mk)
        self.set_blur(0)

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
        cam = n.split('/')[0]
        T = T0 if cam == 'cam0' else T1
        pL, RL = self.pose(np.array([self.t[0] + self.ft[n] - dt]))
        pc = pL[0] + RL[0] @ T[:3, 3]
        Rwc = RL[0] @ T[:3, :3]
        X = self.X if idx is None else self.X[idx]
        P = (X - pc) @ Rwc
        u, v, th = project_thin_prism(P, self.K[cam])
        return u * self.scale, v * self.scale, th, np.linalg.norm(P, axis=1)

    def freeze_visibility(self, x, theta_max_deg=96.0, zcell=4):
        self.obs = []
        for n in self.names:
            u, v, th, d = self._project(n, x)
            img, mk = self.img[n]
            H, W = img.shape
            ok = (th < np.deg2rad(theta_max_deg)) & (d > 1.0) & (u >= 1) & (v >= 1) & (u < W - 2) & (v < H - 2)
            ii = np.flatnonzero(ok)
            ui, vi = u[ii].astype(int), v[ii].astype(int)
            k = mk[vi, ui]
            ii, ui, vi = ii[k], ui[k], vi[k]
            cell = (vi // zcell) * (W // zcell + 1) + ui // zcell
            zb = np.full(cell.max() + 1 if len(cell) else 1, np.inf)
            np.minimum.at(zb, cell, d[ii])
            self.obs.append(ii[d[ii] <= zb[cell] * 1.03 + 0.05])
        cnt = np.bincount(np.concatenate(self.obs), minlength=len(self.X)) if self.obs else np.zeros(len(self.X), int)
        self.obs = [o[cnt[o] >= 3] for o in self.obs]
        return int((cnt >= 3).sum()), int(sum(len(o) for o in self.obs))

    def residuals(self, x):
        pid, val = [], []
        for n, ii in zip(self.names, self.obs):
            u, v, _, _ = self._project(n, x, ii)
            img, _ = self.img[n]
            H, W = img.shape
            ok = (u >= 0) & (v >= 0) & (u < W - 1) & (v < H - 1)
            u, v, ii = u[ok], v[ok], ii[ok]
            x0 = u.astype(np.int32)
            y0 = v.astype(np.int32)
            ax = (u - x0).astype(np.float32)
            ay = (v - y0).astype(np.float32)
            fl = img.ravel()
            k = y0 * W + x0
            s = (fl[k] * (1 - ax) + fl[k + 1] * ax) * (1 - ay) + (fl[k + W] * (1 - ax) + fl[k + W + 1] * ax) * ay
            pid.append(ii)
            val.append(s)
        if not pid:
            return np.zeros(0, np.float32)
        pid = np.concatenate(pid)
        val = np.concatenate(val)
        cnt = np.bincount(pid, minlength=len(self.X))
        mean = np.bincount(pid, val, minlength=len(self.X)) / np.maximum(cnt, 1)
        return (val - mean[pid]) * np.sqrt(cnt[pid] / np.maximum(cnt[pid] - 1, 1))

    def cost(self, x, c=0.5):
        r = self.residuals(x)
        return float(np.mean(np.log1p((r / c) ** 2))) if len(r) else float('inf')


def prior_cost(x, sig_rot_deg=3.0, sig_t_m=0.05, sig_c01_deg=1.0, weight=1e-3):
    z = np.r_[x[:3] / np.deg2rad(sig_rot_deg), x[3:6] / sig_t_m, x[6:9] / np.deg2rad(sig_c01_deg)]
    return weight * float(z @ z)


def _frame_names(dataset):
    stamps = json.loads((Path(dataset) / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
    names = sorted(n for n in stamps if (Path(dataset) / 'images' / n).is_file())
    return {cam: [n for n in names if n.startswith(cam + '/')] for cam in ('cam0', 'cam1')}


def _signature(dataset, init_path, dt_init, mode, frames, fix_dt):
    slam = active_slam_dir(dataset)
    trajectory = sorted((slam / 'result').glob('*.txt'))
    digest = hashlib.sha256(Path(init_path).read_bytes()).hexdigest()
    return {
        'version': REFINEMENT_VERSION, 'slam_dir': slam.name,
        'trajectory_mtime_ns': trajectory[0].stat().st_mtime_ns if trajectory else None,
        'frames_mtime_ns': (Path(dataset) / 'images' / 'frames.json').stat().st_mtime_ns,
        'init_sha256': digest, 'dt_init': round(float(dt_init), 6), 'mode': mode,
        'frames_per_lens': frames, 'fix_dt': bool(fix_dt),
    }


def refine_rig(dataset, init_calibration, dt_init, *, masks_dir=None, mode='photometric',
               fix_dt=False, frames=30, dt_window=0.6, blur=(4.0, 2.0), maxfev=400,
               reference_profile=None, output=None, log=print):
    """Refine the rig (``mode='photometric'``) or only dt (``mode='dt'``) for one capture.

    Returns the written calibration dict.  ``quality_status`` is 'accepted' only when
    the held-out cost improves by MIN_HOLDOUT_IMPROVEMENT and the rig gates pass;
    otherwise the result is written with 'rejected' and callers keep the initial
    calibration and dt.  An up-to-date result with the same signature is reused.
    """
    from scripts.pipeline_auto_calibrator_and_colorizer import get_slam_trajectory_path, load_pcd, load_trajectory

    dataset = Path(dataset)
    output = Path(output) if output else dataset / 'rig_calibration.json'
    init = json.loads(Path(init_calibration).read_text(encoding='utf-8'))
    parts = supported_calibration(init)
    if parts is None:
        raise ValueError('Photometric rig refinement supports dual THIN_PRISM_FISHEYE calibrations only')
    if fix_dt and mode == 'dt':
        raise ValueError('Nothing to refine: mode "dt" with a fixed dt')
    signature = _signature(dataset, init_calibration, dt_init, mode, frames, fix_dt)
    if output.is_file():
        try:
            previous = json.loads(output.read_text(encoding='utf-8'))
            if previous.get('photometric_report', {}).get('signature') == signature:
                log(f'[*] Reusing photometric rig refinement ({previous.get("quality_status")}): {output.name}')
                return previous
        except (OSError, ValueError):
            pass

    t_start = time.time()
    T0, T1, K0, K1 = parts
    slam = active_slam_dir(dataset)
    points = load_pcd(slam / 'pcd' / 'all_raw_points.pcd')
    trajectory = load_trajectory(get_slam_trajectory_path(dataset))
    by_cam = _frame_names(dataset)
    step = max(2, len(by_cam['cam0']) // max(frames, 1))
    fit_names = [n for c in ('cam0', 'cam1') for n in by_cam[c][::step]]
    holdout_names = [n for c in ('cam0', 'cam1') for n in by_cam[c][step // 2::step]]
    common = dict(points=points, trajectory=trajectory, masks_dir=masks_dir, log=log)
    fit = PhotometricRigProblem(dataset, T0, T1, K0, K1, dt_init, fit_names, **common)
    if len(fit.names) < 8:
        raise ValueError(f'Too few frames inside the trajectory for rig refinement ({len(fit.names)})')

    x = np.zeros(10)
    report = {'signature': signature, 'mode': mode, 'fix_dt': bool(fix_dt), 'stages': []}
    if not fix_dt:
        fit.set_blur(4.0)
        fit.freeze_visibility(x)
        grid = np.arange(-dt_window, dt_window + 1e-9, 0.02)
        scan = []
        for g in grid:
            trial = x.copy()
            trial[9] = g
            scan.append(fit.cost(trial))
        scan = np.asarray(scan)
        x[9] = grid[int(np.argmin(scan))]
        report['dt_scan'] = {'offsets': grid.round(3).tolist(), 'cost': scan.round(5).tolist()}
        log(f'[*] Photometric dt scan: Δt {dt_init:.4f}s -> {dt_init + x[9]:.4f}s '
            f'(cost {scan.min():.4f} vs {scan[len(grid) // 2]:.4f})')

    if mode == 'photometric':
        free = np.r_[np.ones(9, bool), not fix_dt]
        sc = np.r_[np.full(3, np.deg2rad(1.0)), np.full(3, 0.02), np.full(3, np.deg2rad(0.5)), 0.02]

        def objective(z):
            full = x.copy()
            full[free] = z * sc[free]
            return fit.cost(full) + prior_cost(full)

        for sigma in blur:
            fit.set_blur(sigma)
            fit.freeze_visibility(x)
            before = fit.cost(x)
            t0 = time.time()
            res = minimize(objective, x[free] / sc[free], method='Powell',
                           options=dict(xtol=1e-2, ftol=1e-5, maxfev=maxfev))
            x[free] = res.x * sc[free]
            report['stages'].append(dict(
                sigma=sigma, cost_before=before, cost_after=float(res.fun), nfev=int(res.nfev),
                seconds=round(time.time() - t0, 1), rot_deg=np.rad2deg(x[:3]).round(4).tolist(),
                trans_cm=(x[3:6] * 100).round(3).tolist(), c01_rot_deg=np.rad2deg(x[6:9]).round(4).tolist(),
                dt=float(dt_init + x[9])))
            log(f'[*] Photometric rig stage σ={sigma:g}px: {before:.4f} -> {res.fun:.4f} '
                f'(Δt {dt_init + x[9]:.4f}s, {time.time() - t0:.0f}s)')

    # Held-out frames were never seen by the fit; each candidate freezes its own visibility.
    holdout = {}
    for label, xx in (('initial', np.zeros(10)), ('refined', x)):
        problem = PhotometricRigProblem(dataset, *fit.extrinsics(xx)[:2], K0, K1, fit.extrinsics(xx)[2],
                                        holdout_names, **{**common, 'log': lambda *_: None})
        problem.freeze_visibility(np.zeros(10))
        residuals = problem.residuals(np.zeros(10))
        holdout[label] = {'cost': problem.cost(np.zeros(10)), 'frames': len(problem.names),
                          'median_abs_residual': float(np.median(np.abs(residuals))) if len(residuals) else None}
    improvement = 1.0 - holdout['refined']['cost'] / holdout['initial']['cost']
    report['holdout'] = {**holdout, 'improvement': improvement}

    T0n, T1n, dtn = fit.extrinsics(x)
    candidate = dict(init)
    candidate.update({
        'T_lidar_to_cam0_rigid_4x4': T0n.tolist(), 'T_lidar_to_cam1_rigid_4x4': T1n.tolist(),
        'dt_sync_seconds': float(dtn), 'dt_sync_reference': 'slam_trajectory_first_pose',
        'lever_arm_cam0_meters': dict(zip(('dx', 'dy', 'dz'), T0n[:3, 3].tolist()),
                                      norm_distance_cm=float(np.linalg.norm(T0n[:3, 3]) * 100)),
        'lever_arm_cam1_meters': dict(zip(('dx', 'dy', 'dz'), T1n[:3, 3].tolist()),
                                      norm_distance_cm=float(np.linalg.norm(T1n[:3, 3]) * 100)),
        'relative_lens_angle_deg': float(np.degrees(np.linalg.norm(
            Rot.from_matrix(T0n[:3, :3].T @ T1n[:3, :3]).as_rotvec()))),
        'lens_baseline_cm': float(np.linalg.norm(T0n[:3, 3] - T1n[:3, 3]) * 100),
        'transform_convention': 'camera_to_trajectory_body',
        'calibration_source': 'photometric_multiview_consistency',
        'calibration_init': str(init_calibration),
        'is_default': False,
    })
    reasons = rig_gate_failures(candidate, load_reference(reference_profile))
    if improvement < MIN_HOLDOUT_IMPROVEMENT:
        reasons.append(f'held-out cost improved by {improvement * 100:.1f}% '
                       f'(needs {MIN_HOLDOUT_IMPROVEMENT * 100:.0f}%)')
    candidate.update(quality_status='rejected' if reasons else 'accepted', valid=not reasons,
                     rejection_reason='; '.join(reasons) or None)
    report['seconds'] = round(time.time() - t_start, 1)
    candidate['photometric_report'] = report
    output.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    log(f'[{"+" if not reasons else "!"}] Photometric rig refinement {candidate["quality_status"]}: '
        f'held-out cost {holdout["initial"]["cost"]:.4f} -> {holdout["refined"]["cost"]:.4f} '
        f'({improvement * 100:+.1f}%), Δt {dtn:.4f}s, {report["seconds"]:.0f}s'
        + (f' — {candidate["rejection_reason"]}' if reasons else ''))
    return candidate
