"""Mount rotation and clock offset of a new camera from motion alone (hand-eye).

A rigidly mounted camera turns exactly as the scanner turns.  Between two frames the
camera's relative rotation (from feature matches) and the SLAM trajectory's relative
rotation over the same interval have the same angle, and their axes differ only by
the mount rotation:

* the clock offset is the shift that best matches the two angle sequences; angles do
  not depend on how the camera is mounted, so this works from any starting guess;
* the mount rotation is the one that aligns the camera's rotation vectors with the
  scanner's (Kabsch, re-weighted against outliers).

Only intrinsics are needed (catalogue values are enough).  This replaces "±15° and
±3 s" uncertainty with about a degree and a few hundredths of a second, which the
photometric solve then refines.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from raven_app.camera_models import CameraModel

MIN_ANGLE_DEG = 2.0       # pairs turning less than this carry no axis information


def bearings(model: CameraModel, u, v):
    """Unit rays (camera frame) of full-resolution pixels; distortion inverted radially."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if model.model == 'EQUIRECTANGULAR':
        lon = (u / model.width - 0.5) * 2.0 * np.pi
        lat = (v / model.height - 0.5) * np.pi
        return np.column_stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)])
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, _, marker = model.params
    if model.model in ('PINHOLE', 'OPENCV'):
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        dist = np.zeros(4) if model.model == 'PINHOLE' else np.array([k1, k2, p1, p2])
        xy = cv2.undistortPoints(np.column_stack([u, v]).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
        rays = np.column_stack([xy, np.ones(len(xy))])
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)
    # Fisheye: r_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), Newton inversion.
    x, y = (u - cx) / fx, (v - cy) / fy
    rd = np.hypot(x, y)
    theta = rd.copy()
    for _ in range(12):
        t2 = theta * theta
        f = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - rd
        df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        theta = np.clip(theta - f / np.where(np.abs(df) < 1e-6, 1e-6, df), 0.0, np.pi)
    scale = np.divide(np.sin(theta), rd, out=np.zeros_like(rd), where=rd > 1e-12)
    return np.column_stack([x * scale, y * scale, np.cos(theta)])


def relative_rotation(model, features_a, features_b, *, scale, matcher):
    """Rotation of the camera from frame a to frame b (camera-frame rotation vector), or None."""
    (ka, da), (kb, db) = features_a, features_b
    if da is None or db is None or len(ka) < 30 or len(kb) < 30:
        return None
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 30:
        return None
    pa = ka[[m.queryIdx for m in good]] / scale
    pb = kb[[m.trainIdx for m in good]] / scale
    ra, rb = bearings(model, pa[:, 0], pa[:, 1]), bearings(model, pb[:, 0], pb[:, 1])
    # Wide lenses: use rays in the front cone, mapped to a normalised image plane.
    keep = (ra[:, 2] > 0.25) & (rb[:, 2] > 0.25)
    if keep.sum() < 25:
        return None
    na, nb = ra[keep, :2] / ra[keep, 2:], rb[keep, :2] / rb[keep, 2:]
    focal = float(model.params[0]) if model.model != 'EQUIRECTANGULAR' else model.width / (2 * np.pi)
    E, inliers = cv2.findEssentialMat(na, nb, np.eye(3), method=cv2.RANSAC, prob=0.999,
                                      threshold=1.5 / max(focal, 1.0))
    if E is None or E.shape != (3, 3) or inliers is None or int(inliers.sum()) < 20:
        return None
    count, R, _, _ = cv2.recoverPose(E, na, nb, np.eye(3), mask=inliers.copy())
    if count < 15:
        return None
    # recoverPose: X_b = R X_a + t, so the camera itself turned by R^T (in frame a).
    return Rot.from_matrix(R.T).as_rotvec(), int(count)


def camera_rotations(dataset, camera, model, *, max_pairs=200, masks_dir=None, scale=None, log=print):
    """Relative rotations between consecutive frames of one camera: (t_a, t_b, rotvec, inliers)."""
    from raven_app.photometric_rig import _executor
    dataset = Path(dataset)
    stamps = json.loads((dataset / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
    names = sorted((n for n in stamps if n.startswith(camera + '/') and (dataset / 'images' / n).is_file()),
                   key=lambda n: stamps[n])
    if len(names) < 10:
        raise ValueError(f'{camera}: too few frames for hand-eye calibration ({len(names)})')
    step = max(1, (len(names) - 1) // max_pairs)
    pairs = [(names[i], names[i + 1]) for i in range(0, len(names) - 1, step)]
    scale = scale or min(1.0, 1024.0 / max(model.width, model.height, 1))

    def features(name):
        image = cv2.imread(str(dataset / 'images' / name), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mask = model.image_mask(*image.shape).astype(np.uint8) * 255
        if masks_dir is not None:
            person = cv2.imread(str(Path(masks_dir) / (name + '.png')), cv2.IMREAD_GRAYSCALE)
            if person is not None:
                mask &= cv2.resize(person, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
        keypoints, descriptors = cv2.SIFT_create(nfeatures=2000).detectAndCompute(image, mask)
        return np.array([k.pt for k in keypoints], dtype=np.float64).reshape(-1, 2), descriptors

    t0 = time.time()
    frames = sorted({n for pair in pairs for n in pair}, key=lambda n: stamps[n])
    feats = dict(zip(frames, _executor().map(features, frames)))

    def solve(pair):
        a, b = pair
        result = relative_rotation(model, feats[a], feats[b], scale=scale, matcher=cv2.BFMatcher(cv2.NORM_L2))
        return None if result is None else (stamps[a], stamps[b], result[0], result[1])

    out = [r for r in _executor().map(solve, pairs) if r is not None]
    log(f'[*] Hand-eye ({camera}): {len(out)}/{len(pairs)} frame pairs with a relative rotation '
        f'({time.time() - t0:.0f}s)')
    return out


def _trajectory_rotvecs(trajectory, ta, tb, dt):
    t, _, rotations = trajectory
    slerp = Slerp(t, rotations)
    qa = slerp(np.clip(t[0] + ta - dt, t[0], t[-1]))
    qb = slerp(np.clip(t[0] + tb - dt, t[0], t[-1]))
    inside = (t[0] + ta - dt > t[0]) & (t[0] + tb - dt < t[-1])
    return (qa.inv() * qb).as_rotvec(), inside


def estimate_clock(trajectory, rotations, dt_init, window, step=0.02):
    """Clock offset matching camera and scanner rotation angles (independent of the mount)."""
    ta = np.array([r[0] for r in rotations])
    tb = np.array([r[1] for r in rotations])
    angles = np.array([np.linalg.norm(r[2]) for r in rotations])
    grid = np.arange(dt_init - window, dt_init + window + 1e-9, step)
    costs = []
    for dt in grid:
        rv, inside = _trajectory_rotvecs(trajectory, ta, tb, dt)
        if inside.sum() < max(10, 0.5 * len(ta)):
            costs.append(np.inf)
            continue
        diff = np.abs(np.linalg.norm(rv[inside], axis=1) - angles[inside])
        costs.append(float(np.mean(np.minimum(diff, np.deg2rad(5.0)) ** 2)))   # robust to bad pairs
    costs = np.array(costs)
    best = int(np.argmin(costs))
    # Parabolic refinement around the grid minimum.
    dt = grid[best]
    if 0 < best < len(grid) - 1 and np.all(np.isfinite(costs[best - 1:best + 2])):
        c0, c1, c2 = costs[best - 1:best + 2]
        denominator = c0 - 2 * c1 + c2
        if denominator > 0:
            dt += 0.5 * step * (c0 - c2) / denominator
    finite = costs[np.isfinite(costs)]
    return float(dt), {'grid': grid.round(3).tolist(), 'cost': np.where(np.isfinite(costs), costs, -1).tolist(),
                       'contrast': float(np.median(finite) / max(costs[best], 1e-12)) if len(finite) else 0.0}


def estimate_mount_rotation(trajectory, rotations, dt):
    """Camera axes in the scanner frame: Kabsch on rotation vectors, re-weighted against outliers."""
    ta = np.array([r[0] for r in rotations])
    tb = np.array([r[1] for r in rotations])
    cam = np.array([r[2] for r in rotations])
    body, inside = _trajectory_rotvecs(trajectory, ta, tb, dt)
    angle = np.minimum(np.linalg.norm(cam, axis=1), np.linalg.norm(body, axis=1))
    use = inside & (angle > np.deg2rad(MIN_ANGLE_DEG))
    if use.sum() < 6:
        raise ValueError(f'Only {int(use.sum())} frame pairs turn more than {MIN_ANGLE_DEG:g}°: '
                         'turn the rig a few times during the capture')
    a, b = cam[use], body[use]
    weights = np.ones(len(a))
    R = np.eye(3)
    for _ in range(10):
        H = (a * weights[:, None]).T @ b
        U, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
        R = Vt.T @ D @ U.T                       # b ≈ R a
        residual = np.linalg.norm(b - a @ R.T, axis=1)
        scale = max(1.4826 * np.median(residual), np.deg2rad(0.5))
        weights = 1.0 / np.maximum(1.0, residual / (2.0 * scale))   # Huber
    residual_deg = np.degrees(np.linalg.norm(b - a @ R.T, axis=1))
    return R, {'pairs': int(use.sum()), 'median_residual_deg': float(np.median(residual_deg)),
               'inlier_fraction': float(np.mean(residual_deg < 2.0))}


def handeye_initialisation(dataset, camera, model, trajectory, dt_init, *, window=10.0, masks_dir=None,
                           log=print):
    """Clock offset and camera rotation (columns = camera axes in the scanner frame).

    ``window`` is the half-width of the clock search around ``dt_init``; 0 keeps ``dt_init``.
    """
    rotations = camera_rotations(dataset, camera, model, masks_dir=masks_dir, log=log)
    if len(rotations) < 20:
        raise ValueError(f'{camera}: only {len(rotations)} frame pairs could be matched')
    if window > 0:
        dt, clock = estimate_clock(trajectory, rotations, dt_init, window)
        if window > 1.0:
            dt, fine = estimate_clock(trajectory, rotations, dt, 0.5, step=0.005)
            clock['fine'] = {k: fine[k] for k in ('contrast',)}
    else:
        dt, clock = float(dt_init), {'contrast': float('nan')}   # clock fixed by the caller
    R, mount = estimate_mount_rotation(trajectory, rotations, dt)
    log(f'[*] Hand-eye ({camera}): Δt {dt:.4f}s (searched ±{window:g}s, contrast {clock["contrast"]:.1f}), '
        f'mount from {mount["pairs"]} turning pairs, median residual {mount["median_residual_deg"]:.2f}°')
    return dt, R, {'camera': camera, 'dt': dt, 'clock_contrast': clock['contrast'], **mount}
