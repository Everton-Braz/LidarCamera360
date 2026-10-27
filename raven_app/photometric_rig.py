"""Photometric refinement of a LiDAR + camera rig and its time offsets.

The hardware profile is a strong prior (a rigid mount does not change between
captures) but the per-capture time offset and small mount deviations decide whether
colours land on the right surface.  This refines the rig by multi-view colour
consistency: wrong extrinsics or clocks make the same LiDAR point read different
grey values in different frames.

A rig is a set of *bodies*; a body is cameras sharing one housing and one clock
(an Insta360 = cam0 + cam1, a single equirectangular camera, a phone or GoPro).
Per body it estimates the mount of its reference camera (rotation + position), the
rotation of each further camera relative to it (tight prior) and the clock offset.
Bodies are refined one after another, the main rig first; each later body is pulled
towards the colours the already-refined bodies see.

* ~250k LiDAR points near the trajectory, projected with each camera's own model
  (``raven_app.camera_models``: thin-prism / OpenCV fisheye, pinhole, OpenCV,
  equirectangular) into N frames per camera.
* Visibility (z-buffer, person masks, usable image area, field of view) is frozen
  per blur stage so the cost is smooth.
* Cost: mean Cauchy deviation of each point's exposure-normalised grey value from
  its multi-view mean, plus a weak prior towards the initial calibration.
* Dense dt scan first (two levels for clocks known only within seconds), then
  coarse-to-fine Powell.
* Acceptance per body: the result must lower the cost on held-out frames (never
  used by the fit) and pass the plausibility gates.

On PIER this cut the held-out cost by 30 % (dt -7.196 -> -7.025 s, 1.7 deg yaw); on
HarmonyMall by 5.4 % against the hardware profile.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp

from raven_app.camera_models import CameraModel
from raven_app.rig_gates import load_reference, rig_gate_failures
from raven_app.slam_source import active_slam_dir

REFINEMENT_VERSION = 3   # bump whenever the solver changes: cached results are keyed on it
MIN_HOLDOUT_IMPROVEMENT = 0.01
MAX_AUX_LEVER_ARM_M = 1.5
# A bootstrapped profile is a ruler measurement: allow the solve to move far from it.
MAX_HANDEYE_RESIDUAL_DEG = 2.0
COARSE_GATE_LIMITS = {'max_rotation_delta_deg': 30.0, 'max_translation_delta_cm': 25.0}


@dataclass
class RigCamera:
    name: str
    model: CameraModel
    T: np.ndarray                      # lidar body <- camera (camera_to_trajectory_body)


@dataclass
class RigBody:
    name: str
    cameras: List[RigCamera]           # cameras[0] is the reference camera
    dt: float                          # SLAM-reference offset: t_query = t_slam0 + t_frame - dt
    dt_window: float = 0.6
    fix_dt: bool = False
    prior_rot_deg: float = 3.0
    prior_trans_m: float = 0.05
    prior_rel_deg: float = 1.0
    bound_rot_deg: float = 6.0          # search box around the initial mount
    bound_trans_m: float = 0.10
    refine_translation: bool = True     # single far-looking cameras barely observe it
    params: np.ndarray = field(default=None)
    initial_T: Optional[Dict[str, np.ndarray]] = None   # mounts before any coarse search
    initial_dt: Optional[float] = None                  # clock before any coarse search

    def __post_init__(self):
        if self.params is None:
            self.params = np.zeros(self.size)

    @property
    def size(self):
        return 6 + 3 * (len(self.cameras) - 1) + 1

    def transforms(self, x=None):
        """Camera transforms and dt for parameter vector ``x`` (perturbations, camera frame)."""
        x = self.params if x is None else x
        ref = self.cameras[0].T
        D = np.eye(4)
        D[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix()
        D[:3, 3] = x[3:6]
        ref_new = ref @ D
        out = {self.cameras[0].name: ref_new}
        for i, cam in enumerate(self.cameras[1:]):
            rel = np.linalg.inv(ref) @ cam.T
            R = np.eye(4)
            R[:3, :3] = Rot.from_rotvec(x[6 + 3 * i: 9 + 3 * i]).as_matrix()
            out[cam.name] = ref_new @ (rel @ R)
        return out, self.dt + x[-1]

    def prior(self, x, weight=1e-3):
        z = [x[:3] / np.deg2rad(self.prior_rot_deg), x[3:6] / self.prior_trans_m,
             x[6:-1] / np.deg2rad(self.prior_rel_deg)]
        z = np.concatenate(z)
        return weight * float(z @ z)

    def scales(self):
        return np.r_[np.full(3, np.deg2rad(1.0)), np.full(3, 0.02),
                     np.full(3 * (len(self.cameras) - 1), np.deg2rad(0.5)), 0.02]

    def bounds(self):
        """Search box (same units as params): mount, relative lens rotations, dt."""
        rot, trans = np.deg2rad(self.bound_rot_deg), self.bound_trans_m
        rel = np.deg2rad(2.0 * self.prior_rel_deg)
        lo = np.r_[np.full(3, -rot), np.full(3, -trans), np.full(3 * (len(self.cameras) - 1), -rel),
                   self.params[-1] - 0.1]
        hi = np.r_[np.full(3, rot), np.full(3, trans), np.full(3 * (len(self.cameras) - 1), rel),
                   self.params[-1] + 0.1]
        return lo, hi


def main_body_from_calibration(calib, dt, *, dataset=None, fix_dt=False, dt_window=0.6):
    """Main rig body (cam0 [+ cam1]) from an app calibration dict, or None if unusable."""
    try:
        T0 = np.asarray(calib['T_lidar_to_cam0_rigid_4x4'], dtype=np.float64)
        intr0 = calib['cam0_front_intrinsics']
    except (KeyError, TypeError, ValueError):
        return None
    if T0.shape != (4, 4):
        return None
    lens = str(intr0.get('camera_model') or calib.get('lens_model') or 'THIN_PRISM_FISHEYE').upper()
    try:
        cameras = [RigCamera('cam0', CameraModel.from_intrinsics(lens, intr0), T0)]
        has_cam1 = (lens != 'EQUIRECTANGULAR' and 'cam1_rear_intrinsics' in calib
                    and 'T_lidar_to_cam1_rigid_4x4' in calib
                    and (dataset is None or (Path(dataset) / 'images' / 'cam1').is_dir()))
        if has_cam1:
            intr1 = calib['cam1_rear_intrinsics']
            lens1 = str(intr1.get('camera_model') or lens).upper()
            cameras.append(RigCamera('cam1', CameraModel.from_intrinsics(lens1, intr1),
                                     np.asarray(calib['T_lidar_to_cam1_rigid_4x4'], dtype=np.float64)))
    except (ValueError, TypeError, KeyError):
        return None
    return RigBody('main', cameras, float(dt), dt_window=dt_window, fix_dt=fix_dt)


def equirectangular_dataset(dataset):
    """True when the dataset's main camera is a stitched 360 video (single cam0, 2:1 frames)."""
    images = Path(dataset) / 'images'
    try:
        if json.loads((images / 'frames.json').read_text(encoding='utf-8')).get('video_mode') == 'equirectangular':
            return True
    except (OSError, ValueError):
        pass
    if (images / 'cam1').is_dir():
        return False
    width, height = _image_size(dataset, 'cam0')
    return bool(width) and abs(width - 2 * height) <= 2


def _image_size(dataset, camera):
    from PIL import Image
    first = next(iter(sorted((Path(dataset) / 'images' / camera).glob('*.jpg'))), None)
    if first is None:
        return 0, 0
    with Image.open(first) as image:
        return image.size


def cube_rotations():
    """The 24 proper rotations mapping coordinate axes onto coordinate axes."""
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                R[row, col] = sign
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


def _coarse_orientation(problem, body, log, sigma=8.0):
    """Pick the axis-aligned orientation of a 360 camera that best explains the images.

    How a stitched equirectangular export is oriented relative to the hardware
    profile's lens is unknown (export settings), so the 24 axis-aligned rotations
    are scored; a 360 camera sees every direction, so one frozen visibility serves all.
    """
    problem.use_bodies([body.name])
    problem.set_blur(sigma)
    problem.freeze_visibility(problem.state())
    camera = body.cameras[0]
    base = camera.T.copy()
    body.initial_T = {camera.name: base.copy()}
    costs = []
    for R in cube_rotations():
        camera.T = base.copy()
        camera.T[:3, :3] = base[:3, :3] @ R
        costs.append(problem.cost(problem.state(), penalize_missing=False))
    order = np.argsort(costs)
    best = cube_rotations()[int(order[0])]
    camera.T = base.copy()
    camera.T[:3, :3] = base[:3, :3] @ best
    log(f'[*] Coarse 360 orientation ({body.name}): best of 24 axis-aligned rotations, cost '
        f'{costs[order[0]]:.4f} (next {costs[order[1]]:.4f}, profile orientation {costs[_identity_index()]:.4f})')
    return {'costs': [round(float(c), 5) for c in costs], 'best_rotation': best.tolist()}


def _identity_index():
    return next(i for i, R in enumerate(cube_rotations()) if np.allclose(R, np.eye(3)))


def _rotate_body(body, base, rotation):
    """Turn every camera of ``body`` by ``rotation`` (LiDAR frame) about the body centre."""
    centre = np.mean([T[:3, 3] for T in base.values()], axis=0)
    G = np.eye(4)
    G[:3, :3] = rotation
    G[:3, 3] = centre - rotation @ centre
    for camera in body.cameras:
        camera.T = G @ base[camera.name]


def _coarse_mount(problem, body, log, *, half_deg, step_deg, sigma):
    """Grid search of the body's mount rotation (roll, pitch, yaw about its centre).

    A measured mount is only good to about ±15°; Powell from there falls into the
    wrong basin, so the rotation is first chosen on a grid with heavily blurred images
    (a wide basin).  Visibility is re-frozen per candidate: a wide-angle rig sees
    almost the same points from any nearby orientation, while a fixed set would
    punish every candidate that turns points out of the frozen views.
    """
    problem.use_bodies([body.name])
    problem.set_blur(sigma)
    base = {c.name: c.T.copy() for c in body.cameras}
    angles = np.arange(-half_deg, half_deg + 1e-9, step_deg)
    results = []
    for rpy in itertools.product(angles, repeat=3):
        _rotate_body(body, base, Rot.from_euler('xyz', rpy, degrees=True).as_matrix())
        state = problem.state()
        problem.freeze_visibility(state)
        results.append((problem.cost(state, penalize_missing=False), rpy))
    results.sort(key=lambda item: item[0])
    best_cost, best = results[0]
    _rotate_body(body, base, Rot.from_euler('xyz', best, degrees=True).as_matrix())
    zero = next(c for c, rpy in results if not any(rpy))
    log(f'[*] Coarse mount search ({body.name}) ±{half_deg:g}° step {step_deg:g}° σ={sigma:g}px: '
        f'roll/pitch/yaw {best[0]:+.1f}/{best[1]:+.1f}/{best[2]:+.1f}° cost {best_cost:.4f} '
        f'(start {zero:.4f}, {len(results)} candidates)')
    return {'half_deg': half_deg, 'step_deg': step_deg, 'sigma': sigma, 'best_rpy_deg': list(best),
            'cost': round(float(best_cost), 5), 'start_cost': round(float(zero), 5)}


def _coarse_dt(problem, body, log, *, window, sigma):
    """Wide dt scan for a clock known only within seconds (no gyro); baked into ``body.dt``."""
    problem.use_bodies([body.name])
    problem.set_blur(sigma)
    problem.freeze_visibility(problem.state())
    saved = body.dt_window
    body.dt_window = window
    try:
        best, scan = _scan_dt(problem, body, log)
    finally:
        body.dt_window = saved
    body.dt += best
    body.params[-1] = 0.0
    return {'window': window, 'sigma': sigma, 'offset': float(best), 'dt': float(body.dt)}


def _coarse_handeye(dataset, body, trajectory, log, *, window, masks_dir=None):
    """Mount rotation and clock of ``body`` from motion (``rig_handeye``), baked into the body."""
    from raven_app.rig_handeye import handeye_initialisation
    reference = body.cameras[0]
    if body.initial_T is None:
        body.initial_T = {c.name: c.T.copy() for c in body.cameras}
    if body.initial_dt is None:
        body.initial_dt = body.dt
    dt, R, report = handeye_initialisation(dataset, reference.name, reference.model, trajectory, body.dt,
                                           window=0.0 if body.fix_dt else window, masks_dir=masks_dir, log=log)
    base = {c.name: c.T.copy() for c in body.cameras}
    G = R @ reference.T[:3, :3].T
    _rotate_body(body, base, G)
    report.update(rotation_change_deg=float(np.degrees(np.linalg.norm(Rot.from_matrix(G).as_rotvec()))),
                  dt_change_s=float(dt - body.dt))
    body.dt = float(dt)
    body.params[-1] = 0.0
    log(f'[*] Hand-eye start ({body.name}): mount turned by {report["rotation_change_deg"]:.1f}°, '
        f'clock moved by {report["dt_change_s"]:+.3f}s')
    return report


def _coarse_main(dataset, body, names, common, log, *, dt_window):
    """Coarse-to-fine start for a mount known to ±15° / ±10 cm and a clock to ±``dt_window``."""
    t0 = time.time()
    if body.initial_T is None:
        body.initial_T = {c.name: c.T.copy() for c in body.cameras}
    body.initial_dt = body.dt
    problem = RigProblem(dataset, [body], names, **{**common, 'scale': 0.125, 'npts': 60000})
    report = {'steps': []}
    if not body.fix_dt and dt_window > body.dt_window:
        report['steps'].append({'dt': _coarse_dt(problem, body, log, window=dt_window, sigma=4.0)})
    report['steps'].append({'mount': _coarse_mount(problem, body, log, half_deg=15.0, step_deg=5.0, sigma=4.0)})
    if not body.fix_dt:
        report['steps'].append({'dt': _coarse_dt(problem, body, log, window=max(1.0, body.dt_window), sigma=2.0)})
    report['steps'].append({'mount': _coarse_mount(problem, body, log, half_deg=4.0, step_deg=2.0, sigma=2.0)})
    report['seconds'] = round(time.time() - t0, 1)
    log(f'[*] Coarse rig search finished in {report["seconds"]:.0f}s, Δt {body.dt:.4f}s')
    return report


def auxiliary_body(camera, main_dt, *, dt_window=3.0):
    """Body for an auxiliary camera (ThirdCameraConfig) with intrinsics already calibrated."""
    status = (camera.calibration_report or {}).get('intrinsics', {}).get('status')
    if status not in ('sfm_calibrated', 'calibrated', 'verified'):
        raise ValueError(f'{camera.camera_name}: intrinsics are {status or "uncalibrated"}')
    intr = dict(camera.intrinsics)
    model = CameraModel.from_intrinsics(camera.camera_model, intr)
    T = np.asarray(camera.T_lidar_to_cam2_rigid_4x4, dtype=np.float64)
    body = RigBody(camera.camera_name, [RigCamera(camera.camera_name, model, T)],
                   float(main_dt) + float(camera.time_offset_s or 0.0), dt_window=dt_window,
                   prior_rot_deg=10.0, prior_trans_m=0.2, bound_rot_deg=12.0, bound_trans_m=0.4,
                   refine_translation=False)
    return body


_EXECUTOR = None


def _executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1)),
                                       thread_name_prefix='photometric')
    return _EXECUTOR


class RigProblem:
    """Multi-view colour consistency of every camera of every body over one point sample."""

    def __init__(self, dataset, bodies, frame_names, *, points, trajectory, masks_dir=None,
                 scale=0.25, voxel=0.04, npts=250000, max_range=18.0, seed=0, log=print):
        self.dataset = Path(dataset)
        self.bodies = {b.name: b for b in bodies}
        self.camera_body = {c.name: b.name for b in bodies for c in b.cameras}
        self.models = {c.name: c.model for b in bodies for c in b.cameras}
        self.scale = scale
        self.active = None   # set of camera names whose frames enter the cost (None = all)
        self.ft = json.loads((self.dataset / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
        self.t, self.p, rotations = trajectory
        self.slerp = Slerp(self.t, rotations)
        names = [n for n in frame_names if n in self.ft and n.split('/')[0] in self.camera_body]
        tq = np.array([self.t[0] + self.ft[n] - self.bodies[self.camera_body[n.split('/')[0]]].dt for n in names])
        inside = (tq > self.t[0] + 0.2) & (tq < self.t[-1] - 0.2)
        self.names = [n for n, ok in zip(names, inside) if ok]
        key = np.floor(points / voxel).astype(np.int64)
        _, idx = np.unique(key, axis=0, return_index=True)
        xyz = points[idx]
        pos, _ = self.pose(tq[inside]) if inside.any() else (self.p[:1], None)
        distance, _ = cKDTree(pos).query(xyz)
        idx = np.flatnonzero(distance < max_range)
        if len(idx) > npts:
            idx = np.random.default_rng(seed).choice(idx, npts, replace=False)
        self.X = xyz[idx].astype(np.float32)
        self._poses = {}
        self.obs_X = {}
        log(f'[*] Photometric rig problem: {len(self.names)} frames '
            f'({", ".join(sorted({n.split("/")[0] for n in self.names}))}), {len(self.X):,} LiDAR points')
        def load(n):
            img = cv2.imread(str(self.dataset / 'images' / n), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)
            H, W = img.shape
            mk = self.models[n.split('/')[0]].image_mask(H, W)
            if masks_dir is not None:
                m = cv2.imread(str(Path(masks_dir) / (n + '.png')), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mk &= cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 127
            mk = cv2.erode(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            return img, mk

        self.raw = dict(zip(self.names, _executor().map(load, self.names)))
        self.set_blur(0)

    def pose(self, tq):
        tq = np.clip(tq, self.t[0], self.t[-1])
        p = np.column_stack([np.interp(tq, self.t, self.p[:, k]) for k in range(3)])
        return p, self.slerp(tq).as_matrix()

    def set_blur(self, sigma):
        def normalised(item):
            im, mk = item
            g = cv2.GaussianBlur(im, (0, 0), sigma) if sigma > 0 else im
            v = g[mk] if mk.any() else g.ravel()
            return ((g - v.mean()) / (v.std() + 1e-6)).astype(np.float32), mk

        self.img = dict(zip(self.raw, _executor().map(normalised, self.raw.values())))

    def state(self, overrides=None):
        """Per-camera transform and per-body dt, with ``overrides`` = {body: params}."""
        transforms, dts = {}, {}
        for name, body in self.bodies.items():
            T, dt = body.transforms((overrides or {}).get(name))
            transforms.update(T)
            dts[name] = dt
        return transforms, dts

    def _frame_pose(self, n, dt):
        key = (n, float(dt))
        pose = self._poses.get(key)
        if pose is None:
            pL, RL = self.pose(np.array([self.t[0] + self.ft[n] - dt]))
            if len(self._poses) > 50000:
                self._poses.clear()
            pose = self._poses[key] = (pL[0], RL[0])
        return pose

    def _project(self, n, state, idx=None, X=None):
        transforms, dts = state
        cam = n.split('/')[0]
        T = transforms[cam]
        pL, RL = self._frame_pose(n, dts[self.camera_body[cam]])
        pc = pL + RL @ T[:3, 3]
        Rwc = RL @ T[:3, :3]
        if X is None:
            X = self.X if idx is None else self.X[idx]
        # float32 is ample for metres around the origin and halves the memory traffic.
        # The 3x3 product is spelled out: frames run in worker threads and a BLAS call
        # from several threads at once can crash some OpenBLAS builds.
        D = X - pc.astype(np.float32)
        R = Rwc.astype(np.float32)
        P = D[:, :1] * R[0] + D[:, 1:2] * R[1] + D[:, 2:3] * R[2]
        u, v, valid, depth = self.models[cam].project(P)
        # Pixel-centre convention of cv2.resize: full-res u maps to (u + 0.5) * s - 0.5.
        return (u + 0.5) * self.scale - 0.5, (v + 0.5) * self.scale - 0.5, valid, depth

    def use_bodies(self, names):
        """Restrict the cost to frames of these bodies (refined or being refined)."""
        self.active = {c for c, b in self.camera_body.items() if b in set(names)}

    def _active_names(self):
        return [n for n in self.names if self.active is None or n.split('/')[0] in self.active]

    def freeze_visibility(self, state, zcell=4):
        def visible(n):
            u, v, valid, d = self._project(n, state)
            img, mk = self.img[n]
            H, W = img.shape
            ok = valid & (d > 1.0) & (u >= 1) & (v >= 1) & (u < W - 2) & (v < H - 2)
            ii = np.flatnonzero(ok)
            ui, vi = u[ii].astype(int), v[ii].astype(int)
            k = mk[vi, ui]
            ii, ui, vi = ii[k], ui[k], vi[k]
            cell = (vi // zcell) * (W // zcell + 1) + ui // zcell
            zb = np.full(cell.max() + 1 if len(cell) else 1, np.inf)
            np.minimum.at(zb, cell, d[ii])
            return ii[d[ii] <= zb[cell] * 1.03 + 0.05]

        self.obs_names = self._active_names()
        self.obs = list(_executor().map(visible, self.obs_names))
        cnt = (np.bincount(np.concatenate(self.obs), minlength=len(self.X)) if self.obs
               else np.zeros(len(self.X), int))
        self.obs = [o[cnt[o] >= 3] for o in self.obs]
        self.obs_X = {n: self.X[o] for n, o in zip(self.obs_names, self.obs)}
        return int((cnt >= 3).sum()), int(sum(len(o) for o in self.obs))

    def _sample(self, n, ii, state):
        """Grey values of observations ``ii`` in frame ``n`` (bilinear) and how many fell outside."""
        u, v, valid, _ = self._project(n, state, ii, X=self.obs_X.get(n))
        img, _ = self.img[n]
        H, W = img.shape
        ok = valid & (u >= 0) & (v >= 0) & (u < W - 1) & (v < H - 1)
        u, v, ii = u[ok], v[ok], ii[ok]
        x0 = u.astype(np.int32)
        y0 = v.astype(np.int32)
        ax = (u - x0).astype(np.float32)
        ay = (v - y0).astype(np.float32)
        fl = img.ravel()
        k = y0 * W + x0
        s = (fl[k] * (1 - ax) + fl[k + 1] * ax) * (1 - ay) + (fl[k + W] * (1 - ax) + fl[k + W + 1] * ax) * ay
        return ii, s, int(len(ok) - len(ii))

    def residuals(self, state, return_missing=False):
        # Frames are independent and numpy releases the GIL in the heavy array work,
        # so they are sampled in parallel; the result is identical to a serial loop.
        jobs = list(zip(self.obs_names, self.obs))
        if len(jobs) > 1:
            parts = list(_executor().map(lambda job: self._sample(job[0], job[1], state), jobs))
        else:
            parts = [self._sample(n, ii, state) for n, ii in jobs]
        missing = sum(p[2] for p in parts)
        if not parts:
            residuals = np.zeros(0, np.float32)
        else:
            pid = np.concatenate([p[0] for p in parts])
            val = np.concatenate([p[1] for p in parts])
            cnt = np.bincount(pid, minlength=len(self.X))
            mean = (np.bincount(pid, val, minlength=len(self.X)) / np.maximum(cnt, 1)).astype(np.float32)
            # Unbiased deviation from the multi-view mean: sqrt(n / (n - 1)) per point.
            unbias = np.sqrt(cnt / np.maximum(cnt - 1, 1)).astype(np.float32)
            residuals = (val - mean[pid]) * unbias[pid]
        return (residuals, missing) if return_missing else residuals

    def cost(self, state, c=0.5, missing_residual=3.0, penalize_missing=True):
        """Mean Cauchy loss over the frozen observations.

        With ``penalize_missing`` an observation pushed out of the image / field of view
        keeps counting with a fixed large residual, so a mount solution cannot lower the
        cost by looking away.  The dt scan turns it off: there the frozen visibility
        belongs to the initial dt and the penalty would bias the scan towards it.
        """
        r, missing = self.residuals(state, return_missing=True)
        if not penalize_missing:
            missing = 0
        total = len(r) + missing
        if not total:
            return float('inf')
        z = r * np.float32(1.0 / c)
        return float((np.sum(np.log1p(z * z), dtype=np.float64) + missing * np.log1p(missing_residual ** 2)) / total)


def _frame_names(dataset, cameras):
    stamps = json.loads((Path(dataset) / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
    names = sorted(n for n in stamps if (Path(dataset) / 'images' / n).is_file())
    return {cam: [n for n in names if n.startswith(cam + '/')] for cam in cameras}


def _signature(dataset, init_path, dt_init, mode, frames, fix_dt, aux_cameras):
    slam = active_slam_dir(dataset)
    trajectory = sorted((slam / 'result').glob('*.txt'))
    aux = [{k: v for k, v in camera.to_dict().items() if k != 'calibration_report'} for camera in aux_cameras]
    return {
        'version': REFINEMENT_VERSION, 'slam_dir': slam.name,
        'trajectory_mtime_ns': trajectory[0].stat().st_mtime_ns if trajectory else None,
        'frames_mtime_ns': (Path(dataset) / 'images' / 'frames.json').stat().st_mtime_ns,
        'init_sha256': hashlib.sha256(Path(init_path).read_bytes()).hexdigest(),
        'dt_init': round(float(dt_init), 6), 'mode': mode, 'frames_per_camera': frames,
        'fix_dt': bool(fix_dt),
        'auxiliary_sha256': hashlib.sha256(json.dumps(aux, sort_keys=True, default=str).encode()).hexdigest(),
    }


def _narrow(body):
    """Bodies whose views lose points at the image border when their clock shifts."""
    return any(not c.model.is_fisheye and c.model.model != 'EQUIRECTANGULAR' for c in body.cameras)


def _dt_cost(problem, body, offset):
    x = body.params.copy()
    x[-1] = offset
    state = problem.state({body.name: x})
    if _narrow(body):
        # A narrow camera's frozen visibility belongs to the initial clock: points that
        # leave the frame would vanish from the mean and favour far-off offsets.
        problem.freeze_visibility(state)
    return problem.cost(state, penalize_missing=False)


def _scan_dt(problem, body, log):
    """Dense dt scan of one body (two levels when the clock is known only within seconds)."""
    def cost_at(offset):
        return _dt_cost(problem, body, offset)

    coarse_step = max(0.02, body.dt_window / 30.0)
    grid = np.arange(-body.dt_window, body.dt_window + 1e-9, coarse_step)
    scan = np.array([cost_at(g) for g in grid])
    best = grid[int(np.argmin(scan))]
    if coarse_step > 0.02:
        fine = np.arange(best - coarse_step, best + coarse_step + 1e-9, 0.02)
        fine_scan = np.array([cost_at(g) for g in fine])
        grid, scan = np.r_[grid, fine], np.r_[scan, fine_scan]
        best = fine[int(np.argmin(fine_scan))]
    order = np.argsort(grid)
    log(f'[*] Photometric dt scan ({body.name}): Δt {body.dt:.4f}s -> {body.dt + best:.4f}s '
        f'(cost {scan.min():.4f} vs {cost_at(0.0):.4f})')
    if _narrow(body):
        problem.freeze_visibility(problem.state())
    return best, {'offsets': grid[order].round(3).tolist(), 'cost': scan[order].round(5).tolist()}


def _rescan_dt(problem, body, half_width=0.1, step=0.01):
    grid = body.params[-1] + np.arange(-half_width, half_width + 1e-9, step)
    best = float(grid[int(np.argmin([_dt_cost(problem, body, g) for g in grid]))])
    if _narrow(body):
        x = body.params.copy()
        x[-1] = best
        problem.freeze_visibility(problem.state({body.name: x}))
    return best


def _refine_body(problem, body, mode, blur, maxfev, log, trusted=()):
    report = {'body': body.name, 'cameras': [c.name for c in body.cameras], 'stages': []}
    problem.use_bodies(list(trusted) + [body.name])
    problem.set_blur(blur[0] if blur else 4.0)
    problem.freeze_visibility(problem.state())
    if not body.fix_dt:
        body.params[-1], report['dt_scan'] = _scan_dt(problem, body, log)
    if mode != 'photometric':
        return report
    free = np.r_[np.ones(body.size - 1, bool), not body.fix_dt]
    if not body.refine_translation:
        # Mount translation of an auxiliary camera trades off against its clock along
        # the walking direction; keep the measured/configured lever arm.
        free[3:6] = False
    sc = body.scales()

    def objective(z):
        x = body.params.copy()
        x[free] = z * sc[free]
        return problem.cost(problem.state({body.name: x})) + body.prior(x)

    for stage, sigma in enumerate(blur):
        problem.set_blur(sigma)
        problem.freeze_visibility(problem.state())
        if stage and not body.fix_dt:
            # Mount rotation and clock trade off while the rig turns: re-centre dt
            # with the rotation the previous stage found.
            body.params[-1] = _rescan_dt(problem, body)
        lo, hi = body.bounds()
        box = list(zip(lo[free] / sc[free], hi[free] / sc[free]))
        before = problem.cost(problem.state())
        t0 = time.time()
        res = minimize(objective, body.params[free] / sc[free], method='Powell', bounds=box,
                       options=dict(xtol=1e-2, ftol=1e-5, maxfev=maxfev))
        body.params[free] = res.x * sc[free]
        report['stages'].append(dict(
            sigma=sigma, cost_before=before, cost_after=float(res.fun), nfev=int(res.nfev),
            seconds=round(time.time() - t0, 1), rot_deg=np.rad2deg(body.params[:3]).round(4).tolist(),
            trans_cm=(body.params[3:6] * 100).round(3).tolist(), dt=float(body.dt + body.params[-1])))
        log(f'[*] Photometric rig stage ({body.name}) σ={sigma:g}px: {before:.4f} -> {res.fun:.4f} '
            f'(Δt {body.dt + body.params[-1]:.4f}s, {time.time() - t0:.0f}s)')
    return report


def _holdout(dataset, bodies, holdout_names, body_name, refined, common, trusted=()):
    """Held-out cost with ``body_name`` at its initial vs refined parameters.

    Both states are scored on the same observations (missing ones penalised), once
    with the visibility of the initial state and once with that of the refined state,
    and the two are averaged: a state must not win by seeing an easier point set.
    """
    trial = []
    for b in bodies:
        clone = copy.copy(b)
        clone.params = b.params.copy()
        trial.append(clone)
    problem = RigProblem(dataset, trial, holdout_names, **{**common, 'log': lambda *_: None})
    problem.use_bodies(list(trusted) + [body_name])
    initial = copy.deepcopy(problem.bodies[body_name])
    for camera in initial.cameras:
        camera.T = (initial.initial_T or {}).get(camera.name, camera.T)
    if initial.initial_dt is not None:
        initial.dt = initial.initial_dt
    transforms, dts = problem.state()
    T_initial, dt_initial = initial.transforms(np.zeros(refined.size))
    states = {'initial': ({**transforms, **T_initial}, {**dts, body_name: dt_initial}),
              'refined': problem.state({body_name: refined})}
    scores = {label: [] for label in states}
    medians = {}
    for frozen_label, frozen in states.items():
        problem.freeze_visibility(frozen)
        for label, state in states.items():
            scores[label].append(problem.cost(state))
        residuals = problem.residuals(frozen)
        medians[frozen_label] = float(np.median(np.abs(residuals))) if len(residuals) else None
    result = {label: {'cost': float(np.mean(values)), 'frames': len(problem.obs_names),
                      'median_abs_residual': medians[label]} for label, values in scores.items()}
    result['improvement'] = 1.0 - result['refined']['cost'] / result['initial']['cost']
    return result


def refine_rig(dataset, init_calibration, dt_init, *, masks_dir=None, mode='photometric',
               fix_dt=False, frames=30, dt_window=0.6, blur=(4.0, 2.0), maxfev=400,
               reference_profile=None, output=None, aux_cameras=None, aux_dt_window=3.0,
               coarse=False, coarse_dt_window=10.0, log=print):
    """Refine the rig (``mode='photometric'``) or only the clocks (``mode='dt'``) for one capture.

    Returns the written main calibration dict.  Its ``quality_status`` is 'accepted'
    only when the main body improves the held-out cost by MIN_HOLDOUT_IMPROVEMENT
    and passes the rig gates; otherwise callers keep the initial calibration and dt.
    Auxiliary cameras (``ThirdCameraConfig`` with calibrated intrinsics) are refined
    as separate bodies afterwards; ``result['auxiliary_cameras']`` holds their updated
    configs, marked calibrated only when their own held-out check passes.
    An up-to-date result with the same signature is reused.

    ``coarse=True`` is for a new rig whose profile comes from measurements
    (``rig_bootstrap``): every body first gets its mount rotation and clock from
    motion (``rig_handeye``, clock searched ±``coarse_dt_window``, kept with
    ``fix_dt``; a photometric grid is the fallback for the main body), the fine
    solve gets a wider box, and the gates allow larger moves away from the
    unvalidated profile but require camera and scanner rotations to agree.
    """
    from scripts.pipeline_auto_calibrator_and_colorizer import get_slam_trajectory_path, load_pcd, load_trajectory

    dataset = Path(dataset)
    output = Path(output) if output else dataset / 'rig_calibration.json'
    init = json.loads(Path(init_calibration).read_text(encoding='utf-8'))
    main = main_body_from_calibration(init, dt_init, dataset=dataset, fix_dt=fix_dt, dt_window=dt_window)
    if main is None:
        raise ValueError('Rig refinement needs T_lidar_to_cam0_rigid_4x4 and cam0_front_intrinsics '
                         'with a supported camera model')
    coarse_orientation = False
    if equirectangular_dataset(dataset) and main.cameras[0].model.model != 'EQUIRECTANGULAR':
        # A stitched 360 video replaces the profile's fisheye pair: one equirectangular
        # camera at the cam0 mount, orientation found by the coarse search below.
        width, height = _image_size(dataset, 'cam0')
        main = RigBody('main', [RigCamera('cam0', CameraModel('EQUIRECTANGULAR', width, height), main.cameras[0].T)],
                       main.dt, dt_window=main.dt_window, fix_dt=main.fix_dt)
        coarse_orientation = True
    if fix_dt and mode == 'dt' and not aux_cameras:
        raise ValueError('Nothing to refine: mode "dt" with a fixed dt')
    aux_cameras = [c for c in (aux_cameras or []) if getattr(c, 'enabled', True)]
    signature = _signature(dataset, init_calibration, dt_init, mode, frames, fix_dt, aux_cameras)
    if coarse:
        if mode != 'photometric':
            raise ValueError('The coarse search needs mode "photometric"')
        signature['coarse'] = {'dt_window': float(coarse_dt_window)}
        main.prior_rot_deg, main.prior_trans_m = 15.0, 0.10
        main.bound_rot_deg, main.bound_trans_m = 6.0, 0.15
    if output.is_file():
        try:
            previous = json.loads(output.read_text(encoding='utf-8'))
            if previous.get('photometric_report', {}).get('signature') == signature:
                log(f'[*] Reusing photometric rig refinement ({previous.get("quality_status")}): {output.name}')
                return previous
        except (OSError, ValueError):
            pass

    t_start = time.time()
    bodies, skipped = [main], []
    for camera in aux_cameras:
        try:
            bodies.append(auxiliary_body(camera, dt_init, dt_window=aux_dt_window))
        except (ValueError, TypeError, KeyError) as exc:
            skipped.append({'camera': camera.camera_name, 'reason': str(exc)})
            log(f'[*] Auxiliary camera {camera.camera_name} not refined: {exc}')
    slam = active_slam_dir(dataset)
    points = load_pcd(slam / 'pcd' / 'all_raw_points.pcd')
    trajectory = load_trajectory(get_slam_trajectory_path(dataset))
    cameras = [c.name for b in bodies for c in b.cameras]
    by_cam = _frame_names(dataset, cameras)
    fit_names, holdout_names = [], []
    for cam, names in by_cam.items():
        step = max(2, len(names) // max(frames, 1))
        fit_names += names[::step]
        holdout_names += names[step // 2::step]
    common = dict(points=points, trajectory=trajectory, masks_dir=masks_dir, log=log)
    report = {'signature': signature, 'mode': mode, 'fix_dt': bool(fix_dt), 'bodies': [], 'skipped': skipped}
    handeye = {}
    if coarse:
        # A measured mount (±15°) and an unknown clock (±seconds) are outside the
        # photometric basin; motion gives both to about a degree / a few ms.
        for body in bodies:
            try:
                handeye[body.name] = _coarse_handeye(dataset, body, trajectory, log, masks_dir=masks_dir,
                                                     window=coarse_dt_window if body is main else
                                                     max(aux_dt_window, coarse_dt_window))
            except (ValueError, cv2.error) as exc:
                log(f'[!] Hand-eye start ({body.name}) failed: {exc}')
                handeye[body.name] = {'error': str(exc)}
        report['handeye'] = handeye
    problem = RigProblem(dataset, bodies, fit_names, **common)
    if sum(n.split('/')[0] in [c.name for c in main.cameras] for n in problem.names) < 8:
        raise ValueError(f'Too few frames inside the trajectory for rig refinement ({len(problem.names)})')

    if coarse_orientation and 'error' not in handeye.get('main', {'error': True}):
        coarse_orientation = False   # hand-eye already found the full orientation
    if coarse_orientation:
        report['coarse_orientation'] = _coarse_orientation(problem, main, log)
    if coarse and 'error' in handeye.get('main', {}):
        report['coarse_search'] = _coarse_main(dataset, main, fit_names, common, log, dt_window=coarse_dt_window)
    verdicts = {}
    trusted = []
    for body in bodies:
        if mode == 'dt' and body.fix_dt:
            trusted.append(body.name)
            continue
        body_report = _refine_body(problem, body, mode, blur, maxfev, log, trusted=trusted)
        body_report['holdout'] = _holdout(dataset, bodies, holdout_names, body.name, body.params.copy(),
                                          common, trusted=trusted)
        reasons = []
        improvement = body_report['holdout']['improvement']
        if improvement < MIN_HOLDOUT_IMPROVEMENT:
            reasons.append(f'held-out cost improved by {improvement * 100:.1f}% '
                           f'(needs {MIN_HOLDOUT_IMPROVEMENT * 100:.0f}%)')
        if coarse:
            motion = handeye.get(body.name, {})
            if 'error' in motion and body is not main:
                reasons.append(f'no hand-eye start: {motion["error"]}')
            elif motion.get('median_residual_deg', 0.0) > MAX_HANDEYE_RESIDUAL_DEG:
                reasons.append(f'camera and scanner rotations disagree by {motion["median_residual_deg"]:.1f}° '
                               f'(median, limit {MAX_HANDEYE_RESIDUAL_DEG:g}°): not rigidly mounted or wrong intrinsics')
        transforms, dt = body.transforms()
        if body is main:
            candidate = {'lens_model': main.cameras[0].model.model,
                         'T_lidar_to_cam0_rigid_4x4': transforms['cam0'].tolist()}
            if 'cam1' in transforms:
                candidate['T_lidar_to_cam1_rigid_4x4'] = transforms['cam1'].tolist()
            reasons += rig_gate_failures(candidate, load_reference(reference_profile),
                                         COARSE_GATE_LIMITS if coarse else None)
        else:
            lever = float(np.linalg.norm(transforms[body.cameras[0].name][:3, 3]))
            if lever > MAX_AUX_LEVER_ARM_M:
                reasons.append(f'lever arm {lever:.2f} m exceeds {MAX_AUX_LEVER_ARM_M} m')
        body_report.update(quality_status='rejected' if reasons else 'accepted',
                           rejection_reason='; '.join(reasons) or None, dt=float(dt))
        verdicts[body.name] = not reasons
        if reasons:
            body.params = np.zeros(body.size)   # later bodies are refined against the initial state
            for camera in body.cameras:
                camera.T = (body.initial_T or {}).get(camera.name, camera.T)
            if body.initial_dt is not None:
                body.dt = body.initial_dt
        log(f'[{"+" if not reasons else "!"}] Photometric rig refinement ({body.name}) '
            f'{"accepted" if not reasons else "rejected"}: held-out cost '
            f'{body_report["holdout"]["initial"]["cost"]:.4f} -> {body_report["holdout"]["refined"]["cost"]:.4f} '
            f'({improvement * 100:+.1f}%), Δt {dt:.4f}s' + (f' — {"; ".join(reasons)}' if reasons else ''))
        report['bodies'].append(body_report)
        if body is main:
            trusted.append(body.name)   # the main rig (refined or initial) anchors the others

    main_report = next((b for b in report['bodies'] if b['body'] == 'main'), None)
    accepted_main = bool(main_report and verdicts.get('main'))
    T_main, dt_main = main.transforms() if accepted_main else main.transforms(np.zeros(main.size))
    candidate = dict(init)
    T0n = T_main['cam0']
    candidate.update({
        'T_lidar_to_cam0_rigid_4x4': T0n.tolist(),
        'dt_sync_seconds': float(dt_main), 'dt_sync_reference': 'slam_trajectory_first_pose',
        'lever_arm_cam0_meters': dict(zip(('dx', 'dy', 'dz'), T0n[:3, 3].tolist()),
                                      norm_distance_cm=float(np.linalg.norm(T0n[:3, 3]) * 100)),
        'transform_convention': 'camera_to_trajectory_body',
        'calibration_source': 'photometric_multiview_consistency',
        'calibration_init': str(init_calibration),
        'is_default': False,
    })
    if 'cam1' in T_main:
        T1n = T_main['cam1']
        candidate.update({
            'T_lidar_to_cam1_rigid_4x4': T1n.tolist(),
            'lever_arm_cam1_meters': dict(zip(('dx', 'dy', 'dz'), T1n[:3, 3].tolist()),
                                          norm_distance_cm=float(np.linalg.norm(T1n[:3, 3]) * 100)),
            'relative_lens_angle_deg': float(np.degrees(np.linalg.norm(
                Rot.from_matrix(T0n[:3, :3].T @ T1n[:3, :3]).as_rotvec()))),
            'lens_baseline_cm': float(np.linalg.norm(T0n[:3, 3] - T1n[:3, 3]) * 100),
        })
    if main.cameras[0].model.model == 'EQUIRECTANGULAR':
        cam = main.cameras[0].model
        candidate['lens_model'] = 'EQUIRECTANGULAR'
        candidate['camera'] = 'Equirectangular_360'
        candidate['cam0_front_intrinsics'] = {'camera_model': 'EQUIRECTANGULAR', 'width': cam.width, 'height': cam.height}
        for key in ('T_lidar_to_cam1_rigid_4x4', 'cam1_rear_intrinsics', 'lever_arm_cam1_meters',
                    'rotation_euler_xyz_deg_cam1', 'relative_lens_angle_deg', 'lens_baseline_cm'):
            candidate.pop(key, None)
    main_reason = (main_report or {}).get('rejection_reason') if main_report else 'main body not refined'
    candidate.update(quality_status='accepted' if accepted_main else 'rejected', valid=accepted_main,
                     rejection_reason=None if accepted_main else main_reason)

    auxiliary = []
    for camera in aux_cameras:
        updated = type(camera).from_dict(camera.to_dict())
        body = next((b for b in bodies if b.name == camera.camera_name), None)
        body_report = next((r for r in report['bodies'] if r['body'] == camera.camera_name), None)
        if body is not None and body_report is not None:
            calib_report = dict(updated.calibration_report or {})
            if verdicts.get(body.name):
                transforms, dt = body.transforms()
                updated.T_lidar_to_cam2_rigid_4x4 = transforms[camera.camera_name].tolist()
                updated.time_offset_s = float(dt - dt_main)
                calib_report['extrinsics'] = {'status': 'calibrated', 'source': 'photometric_multiview_consistency',
                                              'camera_name': camera.camera_name}
                calib_report['time_offset'] = {'status': 'calibrated', 'applied': True,
                                               'seconds': updated.time_offset_s,
                                               'source': 'photometric_multiview_consistency',
                                               'camera_name': camera.camera_name}
            calib_report['photometric_refinement'] = {k: body_report[k] for k in
                                                      ('quality_status', 'rejection_reason', 'holdout', 'dt')}
            updated.calibration_report = calib_report
        auxiliary.append(updated.to_dict())
    candidate['auxiliary_cameras'] = auxiliary
    report['seconds'] = round(time.time() - t_start, 1)
    report['holdout'] = (main_report or {}).get('holdout')
    candidate['photometric_report'] = report
    if output.is_file():
        # Different inputs (or a calibration made elsewhere): keep the last one recoverable.
        output.replace(output.with_name(output.stem + '.previous.json'))
    output.write_text(json.dumps(candidate, indent=2), encoding='utf-8')
    log(f'[*] Photometric rig refinement finished in {report["seconds"]:.0f}s: main '
        f'{candidate["quality_status"]}, Δt {dt_main:.4f}s'
        + ''.join(f', {name} {"accepted" if ok else "rejected"}' for name, ok in verdicts.items() if name != 'main'))
    return candidate
