"""Sub-pixel agreement of the fisheye photos for dense photogrammetry: rolling shutter, lens, lever arm, poses.

A phone-like sensor reads its rows one after the other. While one walks, the top and the bottom of a fisheye
photo are then taken from slightly different poses, several pixels apart when the rig turns, and no single pose
per photo can agree with the neighbours. Dense matching turns that into bumpy walls and missing points.

`refine_photos` adjusts, against the run's SfM feature tracks and the LiDAR surface:

    each frame's pose     a small shift and turn, held to the export's pose
    each lens             its turn and lever arm on the rig and its intrinsics, held to the calibration
    the rolling shutter   how late each row (and column) is read; the motion comes from the SLAM trajectory

Part of the tracks never enters the adjustment. The result is used only when those fit better with it, and it is
never written over the export: `openmvs_dense` resamples its own photos with it (`shutter_maps`).
Cameras other than fisheyes keep their poses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from raven_app import pose_ba
from raven_app.dense_fusion import _voxel_thin

REFERENCE_FOCAL = 945.0                            # px, the focal length the pixel limits below are meant for
ROUNDS = [20.0] * 2 + [12.0] * 3 + [8.0] * 3 + [5.0] * 6    # px, tracks farther off are dropped, round by round
HUBER_PX = 2.0
FRAME_TURN, FRAME_SHIFT = np.radians(0.5), 0.03    # how far (1 sigma, rad and m) a frame leaves its pose
LENS_TURN, LENS_LEVER = np.radians(1.0), 0.02      # and a lens its mount
INTRINSIC_PX, INTRINSIC_K = 8.0, 5e-3              # and its focal length and centre, and its distortion
READOUT_S = 0.05                                   # sigma of the readout time, around none
LIDAR_SIGMA, LIDAR_RADIUS, LIDAR_FLAT, LIDAR_OFF = 0.015, 0.25, 0.012, 0.12
HOLDOUT = 10                                       # one track in 10 only judges the result
MIN_RANGE, MAX_RANGE = 0.3, 60.0
MAX_READOUT_S, MAX_LEVER_M, MAX_SHIFT_M = 0.12, 0.10, 0.15      # beyond these the adjustment is not believed
SPEED_STEP_S = 0.06
LENS_SIZE = 20                                     # parameters of a lens: turn 3, lever 3, intrinsics 12, readout 2
K_SCALE = np.array([1.0] * 4 + [1e-3] * 8)         # pixels, then distortion in thousandths
SHUTTER_STEP = 8                                   # px of the resampled photo per node of the shutter's map


def _skew(v):
    z = np.zeros(len(v))
    return np.stack([z, -v[:, 2], v[:, 1], v[:, 2], z, -v[:, 0], -v[:, 1], v[:, 0], z], 1).reshape(-1, 3, 3)


def _exp(w):
    """Rotation matrices of rotation vectors (N,3)."""
    th2 = (w * w).sum(1)
    th = np.sqrt(th2)
    small = th < 1e-6
    a = np.where(small, 1 - th2 / 6, np.sin(th) / np.where(small, 1, th))
    b = np.where(small, 0.5 - th2 / 24, (1 - np.cos(th)) / np.where(small, 1, th2))
    S = _skew(w)
    return np.eye(3) + a[:, None, None] * S + b[:, None, None] * np.einsum('nij,njk->nik', S, S)


def _apply(M, v):
    return np.einsum('nij,nj->ni', M, v)


def project_thin_prism(Xc, K, jacobian=False):
    """Pixels of camera points (N,3) under THIN_PRISM_FISHEYE parameters (N,12); with `jacobian`, also the
    derivatives by the point (N,2,3) and by the parameters (N,2,12)."""
    x, y, z = Xc.T
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx, sy = K.T
    rho = np.maximum(np.hypot(x, y), 1e-7)
    th = np.arctan2(rho, z)
    s = th / rho
    u, v = x * s, y * s
    r2 = u * u + v * v
    rad = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
    ud = u * (1 + rad) + 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + sx * r2
    vd = v * (1 + rad) + 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + sy * r2
    px = np.stack([fx * ud + cx, fy * vd + cy], 1)
    if not jacobian:
        return px
    d2 = rho * rho + z * z
    a = z / (rho * rho * d2) - th / rho ** 3         # ds/dx = a x, ds/dy = a y, ds/dz = -1 / d2
    du = np.stack([s + a * x * x, a * x * y, -x / d2], 1)
    dv = np.stack([a * x * y, s + a * y * y, -y / d2], 1)
    rd = k1 + r2 * (2 * k2 + r2 * (3 * k3 + r2 * 4 * k4))
    uu = 1 + rad + 2 * u * u * rd + 2 * p1 * v + 6 * p2 * u + 2 * sx * u
    uv = 2 * u * v * rd + 2 * p1 * u + 2 * p2 * v + 2 * sx * v
    vu = 2 * u * v * rd + 2 * p2 * v + 2 * p1 * u + 2 * sy * u
    vv = 1 + rad + 2 * v * v * rd + 2 * p2 * u + 6 * p1 * v + 2 * sy * v
    Jx = np.stack([fx[:, None] * (uu[:, None] * du + uv[:, None] * dv),
                   fy[:, None] * (vu[:, None] * du + vv[:, None] * dv)], 1)
    o, i = np.zeros_like(x), np.ones_like(x)
    r4 = r2 * r2
    Jk = np.stack([np.stack([ud, o, i, o, fx * u * r2, fx * u * r4, fx * 2 * u * v, fx * (r2 + 2 * u * u),
                             fx * u * r2 * r4, fx * u * r4 * r4, fx * r2, o], 1),
                   np.stack([o, vd, o, i, fy * v * r2, fy * v * r4, fy * (r2 + 2 * v * v), fy * 2 * u * v,
                             fy * v * r2 * r4, fy * v * r4 * r4, o, fy * r2], 1)], 1)
    return px, Jx, Jk


def thin_prism_params(model, params):
    """THIN_PRISM_FISHEYE parameters of a fisheye camera, or None for another model."""
    p = [float(a) for a in params]
    if model == 'THIN_PRISM_FISHEYE':
        return np.array(p)
    if model == 'OPENCV_FISHEYE':
        return np.array(p[:6] + [0.0, 0.0] + p[6:8] + [0.0, 0.0])
    return None


def photo_motion(run_dir, colmap_dir, names, centres):
    """Angular (rad/s, world axes) and linear (m/s) velocity of each photo's camera, and which photos have one.

    None when the run keeps no photo times or trajectory, or when the trajectory is in another frame."""
    from raven_app.project import find_trajectory
    try:
        times = json.loads((Path(run_dir) / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
        report = json.loads((Path(colmap_dir) / 'colmap_export_report.json').read_text(encoding='utf-8'))
        dt_sync = float(report['dt_sync_seconds'])
        rows = np.loadtxt(find_trajectory(Path(run_dir)), ndmin=2)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if len(rows) < 10 or rows.shape[1] < 8:
        return None
    t = rows[:, 0] - rows[0, 0]
    keep = np.r_[True, np.diff(t) > 0]
    t, rows = t[keep], rows[keep]
    h = SPEED_STEP_S
    when = np.array([float(times.get(n, np.nan)) - dt_sync for n in names])
    known = np.isfinite(when) & (when >= t[0] - 1) & (when <= t[-1] + 1)
    if known.sum() < 3:
        return None
    when = np.clip(np.where(known, when, t[0]), t[0] + h, t[-1] - h)
    turn = Slerp(t, Rot.from_quat(rows[:, 4:8]))

    def at(q):
        return np.stack([np.interp(q, t, rows[:, k]) for k in (1, 2, 3)], 1)

    arm = centres - at(when)
    if np.median(np.linalg.norm(arm[known], axis=1)) > 1.0:
        return None                        # not the map these poses are in
    w = (turn(when + h) * turn(when - h).inv()).as_rotvec() / (2 * h)
    v = (at(when + h) - at(when - h)) / (2 * h) + np.cross(w, arm)
    w[~known], v[~known] = 0, 0
    return w, v, known


class _Adjustment:
    """Bundle adjustment of the observations (photo, point, pixel), in numpy."""

    def __init__(self, R, C, lens, frame, K, size, w, v, obs_image, obs_point, obs_xy, unit, shutter):
        self.R, self.C, self.lens, self.frame, self.K = R, C, lens, frame, K
        self.w, self.v, self.oi, self.op, self.xy = w, v, obs_image, obs_point, obs_xy
        self.n_frames, self.n_lenses, self.n_points = int(frame.max()) + 1, len(K), int(obs_point.max()) + 1
        self.unit, self.huber = unit, HUBER_PX * unit
        count = np.bincount(frame, minlength=self.n_frames)
        self.pivot = np.stack([np.bincount(frame, C[:, k], self.n_frames) for k in range(3)], 1) / np.maximum(count, 1)[:, None]
        self.s = (obs_xy - size[lens[obs_image]] / 2) / size[lens[obs_image]][:, 1:2]      # image heights from the centre
        self.df, self.dl = np.zeros((self.n_frames, 6)), np.zeros((self.n_lenses, LENS_SIZE))
        self.frame_prior = np.r_[[FRAME_TURN ** -2] * 3, [FRAME_SHIFT ** -2] * 3]
        self.lens_prior = np.r_[[LENS_TURN ** -2] * 3, [LENS_LEVER ** -2] * 3, [(INTRINSIC_PX * unit) ** -2] * 4,
                                (INTRINSIC_K / K_SCALE[4:]) ** -2, [READOUT_S ** -2 if shutter else 1e18] * 2]

    def poses(self):
        """Rotation and centre of every photo at its centre row."""
        D = _exp(self.df[:, :3])[self.frame]
        f, l = self.frame, self.lens
        q = _apply(D, self.C - self.pivot[f])
        M = np.einsum('nij,njk,nlk->nil', _exp(self.dl[:, :3])[l], self.R, D)
        centre = self.pivot[f] + self.df[f, 3:] + q + np.einsum('nji,nj->ni', self.R, self.dl[l, 3:6])
        return M, centre, q

    def intrinsics(self):
        return self.K + self.dl[:, 6:18] * K_SCALE

    def _camera_points(self, X, a):
        M, centre, q = self.poses()
        i = self.oi[a]
        tau = (self.dl[self.lens[i], 18:] * self.s[a]).sum(1)
        E = _exp(-self.w[i] * tau[:, None])
        y = _apply(E, X - centre[i] - self.v[i] * tau[:, None])
        return _apply(M[i], y), y, E, M[i], q[i], centre[i] + self.v[i] * tau[:, None]

    def errors(self, X, a):
        Xc = self._camera_points(X[self.op[a]], a)[0]
        with np.errstate(invalid='ignore'):
            return np.linalg.norm(project_thin_prism(Xc, self.intrinsics()[self.lens[self.oi[a]]]) - self.xy[a], axis=1)

    def triangulate(self, a):
        """Points (NaN where the rays do not cross well) of the observations `a`."""
        _, _, E, M, _, centre = self._camera_points(np.zeros((len(a), 3)), a)
        K, l = self.intrinsics(), self.lens[self.oi[a]]
        b = np.full((len(a), 3), np.nan)
        for k in range(self.n_lenses):
            m = l == k
            if m.any():
                b[m] = pose_ba.unproject_thin_prism(self.xy[a[m]], K[k])
        d = np.einsum('nji,nj->ni', E, np.einsum('nji,nj->ni', M, b))
        good = np.isfinite(d).all(1)
        P = np.eye(3) - d[good][:, :, None] * d[good][:, None, :]
        A = pose_ba._scatter(self.op[a][good], P, self.n_points)
        g = pose_ba._scatter(self.op[a][good], _apply(P, centre[good]), self.n_points)
        ok = np.linalg.det(A) > 1e-6
        X = np.full((self.n_points, 3), np.nan)
        X[ok] = np.linalg.solve(A[ok], g[ok][:, :, None])[:, :, 0]
        return X

    def prune(self, X, a, limit):
        centre = self.poses()[1]
        rng = np.linalg.norm(X[self.op[a]] - centre[self.oi[a]], axis=1)
        e = self.errors(X, a)
        a = a[np.isfinite(e) & (e < limit * self.unit) & (rng > MIN_RANGE) & (rng < MAX_RANGE)]
        return a[np.bincount(self.op[a], minlength=self.n_points)[self.op[a]] >= 2]

    def step(self, X, a, planes):
        """One Gauss-Newton step, with the points eliminated."""
        nf, nl, n = self.n_frames * 6, self.n_lenses * LENS_SIZE, self.n_points
        Hff, gf = np.zeros((self.n_frames, 6, 6)), np.zeros((self.n_frames, 6))
        Hfl = np.zeros((self.n_frames, 6, nl))
        Hll, gl = np.zeros((nl, nl)), np.zeros(nl)
        Hpp, gp = np.zeros((n, 3, 3)), np.zeros((n, 3))
        El = np.zeros((n, nl, 3))
        blocks, rows, cols = [], [], []
        K = self.intrinsics()
        for a in np.array_split(a, max(1, len(a) // 60_000)):
            i = self.oi[a]
            l, f, p = self.lens[i], self.frame[i], self.op[a]
            Xc, y, E, M, q, _ = self._camera_points(X[p], a)
            px, Jx, Jk = project_thin_prism(Xc, K[l], jacobian=True)
            r = px - self.xy[a]
            e = np.linalg.norm(r, axis=1)
            w = np.where(e < self.huber, 1.0, self.huber / np.maximum(e, 1e-12))
            ME = np.einsum('nij,njk->nik', M, E)
            chain = np.einsum('nij,njk->nik', Jx, ME)                  # by a move of the point
            JM = np.einsum('nij,njk->nik', Jx, M)
            turn = np.einsum('nij,njk->nik', JM, _skew(y) + np.einsum('nij,njk->nik', E, _skew(q)))
            Jf = np.concatenate([turn, -chain], 2)
            dtau = np.einsum('nij,nj->ni', JM, -np.cross(self.w[i], y) - _apply(E, self.v[i]))
            Jl = np.concatenate([-np.einsum('nij,njk->nik', Jx, _skew(Xc)),
                                 -np.einsum('nij,nkj->nik', chain, self.R[i]), Jk * K_SCALE,
                                 dtau[:, :, None] * self.s[a][:, None, :]], 2)
            wJf, wJl, wJp = Jf * w[:, None, None], Jl * w[:, None, None], chain * w[:, None, None]
            Hff += pose_ba._scatter(f, np.einsum('nki,nkj->nij', wJf, Jf), self.n_frames)
            gf += pose_ba._scatter(f, np.einsum('nki,nk->ni', wJf, r), self.n_frames)
            Hpp += pose_ba._scatter(p, np.einsum('nki,nkj->nij', wJp, chain), n)
            gp += pose_ba._scatter(p, np.einsum('nki,nk->ni', wJp, r), n)
            for k in range(self.n_lenses):
                m = l == k
                if not m.any():
                    continue
                at = slice(k * LENS_SIZE, (k + 1) * LENS_SIZE)
                Hll[at, at] += np.einsum('nki,nkj->ij', wJl[m], Jl[m])
                gl[at] += np.einsum('nki,nk->i', wJl[m], r[m])
                Hfl[:, :, at] += pose_ba._scatter(f[m], np.einsum('nki,nkj->nij', wJf[m], Jl[m]), self.n_frames)
                El[:, at] += pose_ba._scatter(p[m], np.einsum('nki,nkj->nij', wJl[m], chain[m]), n)
            blocks.append(np.einsum('nki,nkj->nij', wJf, chain))
            rows.append(f)
            cols.append(p)
        if planes is not None:
            normal, point, ok = planes
            ok = ok & np.isfinite(X).all(1)
            off = np.where(ok, np.einsum('ni,ni->n', np.where(ok[:, None], X - point, 0), normal), 0)
            w = ok * np.minimum(1.0, 2 * LIDAR_SIGMA / np.maximum(np.abs(off), 1e-9)) / LIDAR_SIGMA ** 2
            Hpp += w[:, None, None] * normal[:, :, None] * normal[:, None, :]
            gp += (w * off)[:, None] * normal
        Hff += np.diag(self.frame_prior)
        gf += self.frame_prior * self.df
        Hll[np.diag_indices(nl)] += np.tile(self.lens_prior, self.n_lenses)
        gl += (self.lens_prior * self.dl).ravel()
        Hi = np.linalg.inv(Hpp + 1e-3 * np.eye(3))
        blocks, rows, cols = np.concatenate(blocks), np.concatenate(rows), np.concatenate(cols)
        order = np.argsort(rows, kind='stable')
        start = np.r_[0, np.cumsum(np.bincount(rows, minlength=self.n_frames))]
        Ef = sparse.bsr_matrix((blocks[order], cols[order], start), shape=(nf, 3 * n)).tocsr()
        inverse = sparse.bsr_matrix((Hi, np.arange(n), np.arange(n + 1)), shape=(3 * n, 3 * n)).tocsr()
        ElT = El.transpose(0, 2, 1).reshape(3 * n, nl)                 # (points, lens)
        S = np.zeros((nf + nl, nf + nl))
        EfC = Ef @ inverse
        S[:nf, :nf] = -(EfC @ Ef.T).toarray()
        for k in range(self.n_frames):
            S[6 * k:6 * k + 6, 6 * k:6 * k + 6] += Hff[k]
        S[:nf, nf:] = Hfl.reshape(nf, nl) - EfC @ ElT
        S[nf:, :nf] = S[:nf, nf:].T
        CE = inverse @ ElT
        S[nf:, nf:] = Hll - ElT.T @ CE
        g = np.r_[gf.ravel() - EfC @ gp.ravel(), gl - CE.T @ gp.ravel()]
        x = -np.linalg.solve(S + 1e-6 * np.diag(np.diag(S)), g)
        dp = -_apply(Hi, gp + (Ef.T @ x[:nf] + ElT @ x[nf:]).reshape(n, 3))
        dp[np.linalg.norm(dp, axis=1) > 1.0] = np.nan
        self.df += x[:nf].reshape(-1, 6)
        self.dl += x[nf:].reshape(-1, LENS_SIZE)
        return X + dp


def _lidar_planes(tree, lidar, X):
    """Local LiDAR plane (normal, point, usable) of each track point."""
    finite = np.isfinite(X).all(1)
    far, index = np.full(len(X), np.inf), np.zeros((len(X), 12), int)
    if finite.any():
        d, index[finite] = tree.query(X[finite], k=12, workers=-1)
        far[finite] = d[:, -1]
    q = lidar[index]
    centre = q.mean(1)
    u = q - centre[:, None]
    spread, axes = np.linalg.eigh(np.einsum('nki,nkj->nij', u, u) / 12)
    spread, normal = np.sqrt(np.maximum(spread, 0)), axes[:, :, 0]
    off = np.einsum('ni,ni->n', np.where(finite[:, None], X, 0) - centre, normal)
    ok = finite & (far < LIDAR_RADIUS) & (spread[:, 0] < LIDAR_FLAT) & (spread[:, 1] > 0.02) & (np.abs(off) < LIDAR_OFF)
    return normal, centre, ok


def _tracks(run_dir):
    for folder in ('sparse/0', 'sparse'):
        path = Path(run_dir) / folder / 'images.bin'
        if path.is_file():
            return pose_ba.read_colmap_observations(path)
    return {}


def refine_photos(colmap_dir, cameras, images, *, run_dir=None, lidar_xyz=None, log=print):
    """Adjust cameras and poses of a 3DGS export (as `read_colmap_text` gives them) for dense matching.

    Returns (cameras, images, shutter, report). `shutter` is None, or for each photo (readout (2,) in seconds
    over the image height, angular velocity, velocity), both velocities in camera axes. When nothing can be
    adjusted or the held-out tracks do not fit better, the inputs come back with shutter None."""
    try:
        return _refine(Path(colmap_dir), cameras, images, Path(run_dir) if run_dir else Path(colmap_dir).parent,
                       lidar_xyz, log)
    except Exception as exc:               # an optional improvement: the export's own poses always work
        log(f'[!] Photo alignment failed, the poses of the export are used: {exc}')
        return cameras, images, None, {'status': 'failed', 'reason': str(exc)}


def _refine(colmap_dir, cameras, images, run_dir, lidar_xyz, log):
    def skip(reason):
        log(f'[*] Photo alignment skipped: {reason}')
        return cameras, images, None, {'status': 'skipped', 'reason': reason}

    tracks = _tracks(run_dir)
    lens_ids = sorted(c for c in {i[1] for i in images} if thin_prism_params(cameras[c][0], cameras[c][3]) is not None)
    rig = [k for k, i in enumerate(images) if i[1] in lens_ids]
    if not any(images[k][0] in tracks for k in rig):
        return skip('the run keeps no feature tracks of these photos')
    upright = colmap_dir / 'upright_rotation.json'
    turned = json.loads(upright.read_text(encoding='utf-8')).get('camera_ids', []) if upright.is_file() else []
    names = [images[k][0] for k in rig]
    R, C = np.array([images[k][2] for k in rig]), np.array([images[k][3] for k in rig])
    lens = np.array([lens_ids.index(images[k][1]) for k in rig])
    keys, frame = np.unique([pose_ba._frame_key(n) for n in names], return_inverse=True)
    K = np.array([thin_prism_params(cameras[c][0], cameras[c][3]) for c in lens_ids])
    size = np.array([cameras[c][1:3] for c in lens_ids], float)
    oi, op, xy = [], [], []
    for k, name in enumerate(names):
        if name in tracks:
            p, ids = tracks[name]
            oi.append(np.full(len(ids), k))
            op.append(ids)
            xy.append(size[lens[k]] - p if lens_ids[lens[k]] in turned else p)
    oi, op, xy = np.concatenate(oi), np.concatenate(op), np.concatenate(xy)
    kept, op = np.unique(op, return_inverse=True)
    judge = kept[op] % HOLDOUT == 0
    if len(oi) < 2000 or (~judge).sum() < 1000:
        return skip(f'only {len(oi)} track observations')
    motion = photo_motion(run_dir, colmap_dir, names, C)
    if motion is None:
        log('[*] No photo times or trajectory: the lens and the poses are adjusted without the rolling shutter')
    w, v = motion[:2] if motion else (np.zeros_like(C), np.zeros_like(C))
    unit = float(np.median(K[:, 0])) / REFERENCE_FOCAL
    ba = _Adjustment(R, C, lens, frame, K, size, w, v, oi, op, xy, unit, motion is not None)

    def fit(a):
        """Re-triangulate the observations `a` and measure them (NaN behind or beyond the range)."""
        X = ba.triangulate(a)
        e = ba.errors(X, a)
        rng = np.linalg.norm(X[op[a]] - ba.poses()[1][oi[a]], axis=1)
        e[(rng < MIN_RANGE) | (rng > MAX_RANGE)] = np.nan
        return X, e

    test = np.flatnonzero(judge)
    before = fit(test)[1]
    a = np.flatnonzero(~judge)
    X, e = fit(a)
    start = float(np.nanmedian(e))
    a = ba.prune(X, a, 40.0)
    if len(a) < 1000:
        return skip('too few feature tracks agree with the poses of the export')
    tree = lidar = None
    if lidar_xyz is not None and len(lidar_xyz) > 1000:
        lidar = np.asarray(lidar_xyz, float)
        lidar = lidar[_voxel_thin(lidar, 0.02)]
        tree = cKDTree(lidar)
    log(f'[*] Photo alignment: {len(a):,} observations of {len(names)} fisheye photos, median {start:.2f} px')
    planes = None
    for turn, limit in enumerate(ROUNDS):
        if tree is not None and turn % 3 == 0:
            planes = _lidar_planes(tree, lidar, X)
        X = ba.step(X, a, planes)
        a = ba.prune(X, a, limit)
    after = fit(test)[1]
    both = np.isfinite(before) & np.isfinite(after)
    old, new = float(np.median(before[both])), float(np.median(after[both]))
    M, centre, _ = ba.poses()
    shift = np.linalg.norm(ba.df[:, 3:], axis=1)
    report = {'observations': int(len(a)), 'held_out_px_before': round(old, 3), 'held_out_px': round(new, 3),
              'median_px_before': round(start, 3), 'median_px': round(float(np.median(ba.errors(X, a))), 3),
              'rolling_shutter': motion is not None,
              'frame_shift_cm': {'median': round(float(np.median(shift)) * 100, 2), 'max': round(float(shift.max()) * 100, 2)},
              'lenses': [{'camera_id': int(c), 'readout_ms': round(float(ba.dl[k, 19]) * 1000, 2),
                          'readout_across_ms': round(float(ba.dl[k, 18]) * 1000, 2),
                          'lever_cm': [round(float(x) * 100, 2) for x in ba.dl[k, 3:6]],
                          'turn_deg': [round(float(x), 3) for x in np.degrees(ba.dl[k, :3])],
                          'focal_px': [round(float(x), 2) for x in ba.intrinsics()[k, :2]]}
                         for k, c in enumerate(lens_ids)]}
    sane = (np.isfinite(new) and np.abs(ba.dl[:, 18:]).max() < MAX_READOUT_S
            and np.linalg.norm(ba.dl[:, 3:6], axis=1).max() < MAX_LEVER_M and shift.max() < MAX_SHIFT_M)
    if not sane or new > 0.97 * old:
        report.update(status='rejected', reason=f'held-out tracks: {old:.2f} px before, {new:.2f} px after'
                      if sane else 'the adjustment moved farther than a rig can')
        log(f"[*] Photo alignment not used ({report['reason']})")
        return cameras, images, None, report
    report['status'] = 'accepted'
    new_cameras, new_images, shutter = dict(cameras), list(images), [None] * len(images)
    for k, c in enumerate(lens_ids):
        new_cameras[c] = ('THIN_PRISM_FISHEYE', cameras[c][1], cameras[c][2], [float(x) for x in ba.intrinsics()[k]])
    for j, k in enumerate(rig):
        new_images[k] = (names[j], images[k][1], M[j], centre[j])
        if motion is not None and motion[2][j]:
            shutter[k] = (ba.dl[lens[j], 18:].copy(), _apply(M[j:j + 1], w[j:j + 1])[0], _apply(M[j:j + 1], v[j:j + 1])[0])
    log(f'[+] Photo alignment: tracks kept out of it fit to {new:.2f} px (were {old:.2f})'
        + (', rolling shutter ' + ' / '.join(f"{abs(x['readout_ms']):.1f}" for x in report['lenses']) + ' ms' if motion else ''))
    return new_cameras, new_images, shutter if motion is not None else None, report


# --- resampling -----------------------------------------------------------------------------------
def inverse_range(points, R, C, size=256):
    """1 / distance of the surface along each ray of the ideal fisheye (size, size), from a point cloud."""
    from scipy.ndimage import distance_transform_edt
    from raven_app.openmvs_dense import MAX_THETA, equidistant_project
    d = points - C
    px, theta = equidistant_project(d[:, 0:1] * R[:, 0] + d[:, 1:2] * R[:, 1] + d[:, 2:3] * R[:, 2], size)
    rng = np.sqrt((d * d).sum(axis=1))
    q = np.rint(px).astype(np.int64)
    ok = (theta < MAX_THETA + 0.1) & (rng > MIN_RANGE) & (rng < MAX_RANGE) & (q >= 0).all(axis=1) & (q < size).all(axis=1)
    inv = np.zeros(size * size, np.float32)
    np.maximum.at(inv, q[ok, 1] * size + q[ok, 0], (1 / rng[ok]).astype(np.float32))
    inv = inv.reshape(size, size)
    if not (inv > 0).any():
        return inv
    nearest = distance_transform_edt(inv == 0, return_distances=False, return_indices=True)
    return inv[tuple(nearest)]


def shutter_maps(camera, size, rays_of, base, motion, inverse):
    """(map_x, map_y, inside) like `openmvs_dense.build_remap`, of a photo read while it moves.

    `base` is build_remap's own result, `motion` the photo's entry of the shutter, `inverse` its inverse_range.
    The shutter moves pixels smoothly, so it is computed on a coarse grid."""
    import cv2
    from raven_app.openmvs_dense import MAX_THETA, equidistant_project, project_rays
    model, w, h, params = camera
    readout, omega, speed = motion
    rays, theta = rays_of(max(8, size // SHUTTER_STEP))
    low = len(theta)
    n = inverse.shape[0]
    near = equidistant_project(rays, n)[0].astype(np.float32).reshape(low, low, 2)
    inv = cv2.remap(cv2.medianBlur(inverse, 5), near[..., 0], near[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE).ravel()
    first, _ = project_rays(model, params, rays)
    px = first
    for _ in range(3):
        tau = (readout[0] * (px[:, 0] - w / 2) + readout[1] * (px[:, 1] - h / 2)) / h
        d = rays - speed * (tau * inv)[:, None]
        turn = np.cross(omega, d)
        d = d - turn * tau[:, None] + 0.5 * np.cross(omega, turn) * (tau * tau)[:, None]
        px, _ = project_rays(model, params, d / np.sqrt((d * d).sum(axis=1))[:, None])
    move = cv2.resize((px - first).astype(np.float32).reshape(low, low, 2), (size, size), interpolation=cv2.INTER_LINEAR)
    mx, my, inside = base[0] + move[..., 0], base[1] + move[..., 1], base[2]
    return mx, my, inside & (mx >= 0) & (my >= 0) & (mx <= w - 1) & (my <= h - 1)
