"""Bundle adjustment of the direct (LiDAR trajectory + rig) camera poses for 3DGS.

The direct poses are metric and drift-free but carry a few tenths of a degree of
per-frame attitude noise (HarmonyMall: 0.35 deg median, ~7 px on a 1080 px-focal
fisheye), which a point cloud hides and a splatting model turns into blur.  A full
SfM fixes the local noise but drifts globally.  This adjusts the direct poses
against the SfM feature tracks while a pose prior holds every frame to its LiDAR
pose, so the result keeps the LiDAR frame and scale and gains the photogrammetric
consistency:

* the two lenses of a rig frame move together (one rotation + translation per frame);
* each lens may rotate slightly against the rig (the lever arms stay as profiled,
  they are not observable from these residuals);
* residuals are angular errors on the bearing sphere scaled by the focal length,
  so the fisheye distortion is inverted once per observation, not per iteration.
"""
import json
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

THIN_PRISM_PARAMS = 12


def read_colmap_observations(images_bin):
    """Return {name: (xy[N,2], point3D_id[N])} for the observed keypoints of a COLMAP model."""
    out = {}
    with open(images_bin, 'rb') as f:
        count = struct.unpack('<Q', f.read(8))[0]
        for _ in range(count):
            f.read(4 + 32 + 24 + 4)
            name = bytearray()
            while (c := f.read(1)) != b'\x00':
                name += c
            n = struct.unpack('<Q', f.read(8))[0]
            rows = np.frombuffer(f.read(24 * n), dtype=np.dtype([('x', '<f8'), ('y', '<f8'), ('id', '<i8')]))
            keep = rows['id'] >= 0
            out[name.decode('utf-8').replace('\\', '/')] = (
                np.stack([rows['x'][keep], rows['y'][keep]], axis=1), rows['id'][keep])
    return out


def read_images_txt(path):
    """Return [(image_id, R_cw, t_cw, camera_id, name)] from a COLMAP images.txt."""
    images = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if len(parts) != 10 or line.startswith('#'):
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        images.append((int(parts[0]), Rot.from_quat([qx, qy, qz, qw]).as_matrix(),
                       np.array([tx, ty, tz]), int(parts[8]), parts[9]))
    return images


def read_cameras_txt(path):
    cameras = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        parts = line.split()
        if parts and not line.startswith('#'):
            cameras[int(parts[0])] = (parts[1], np.array(parts[4:], dtype=np.float64))
    return cameras


def unproject_thin_prism(uv, params, iterations=20):
    """Unit bearing vectors of pixels under COLMAP's THIN_PRISM_FISHEYE model."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    m = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy], axis=1)

    def distort(p):
        x, y = p[:, 0], p[:, 1]
        r2 = x * x + y * y
        radial = 1 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
        return np.stack([x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + sx1 * r2,
                         y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + sy1 * r2], axis=1)

    p = m.copy()
    eps = 1e-7
    for _ in range(iterations):
        f0 = distort(p) - m
        jx = (distort(p + [eps, 0.0]) - distort(p)) / eps
        jy = (distort(p + [0.0, eps]) - distort(p)) / eps
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        p = p - np.stack([(jy[:, 1] * f0[:, 0] - jy[:, 0] * f0[:, 1]) / det,
                          (-jx[:, 1] * f0[:, 0] + jx[:, 0] * f0[:, 1]) / det], axis=1)
        if np.max(np.abs(f0)) < 1e-12:
            break
    theta = np.hypot(p[:, 0], p[:, 1])
    scale = np.where(theta > 1e-12, np.sin(theta) / np.maximum(theta, 1e-12), 1.0)
    bearings = np.stack([p[:, 0] * scale, p[:, 1] * scale, np.cos(theta)], axis=1)
    bearings[np.max(np.abs(distort(p) - m), axis=1) > 1e-6] = np.nan  # past the lens' fold
    return bearings


def _skew(v):
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def _tangent_basis(b):
    helper = np.where(np.abs(b[:, [2]]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    e1 = np.cross(b, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    return np.stack([e1, np.cross(b, e1)], axis=1)


def _scatter(index, values, size):
    """Sum ``values`` rows into ``size`` bins by ``index`` (np.add.at, but fast)."""
    flat = values.reshape(len(values), -1)
    out = np.stack([np.bincount(index, weights=flat[:, k], minlength=size) for k in range(flat.shape[1])], 1)
    return out.reshape((size,) + values.shape[1:])


def _frame_key(name):
    return name.split('/', 1)[1] if '/' in name else name


def refine_poses(sparse_dir, tracks_dir, *, rig_cameras=('cam0', 'cam1'), prior_pos_m=0.03,
                 prior_rot_deg=0.5, rig_prior_deg=1.0, huber_px=2.0, max_error_px=4.0,
                 iterations=25, max_shift_m=0.25, max_rotation_deg=3.0, backup_path=None, log=print):
    """Adjust the rig poses of ``sparse_dir/images.txt`` in place against ``tracks_dir``'s tracks.

    Returns a report dict; ``report['status'] == 'accepted'`` when images.txt was rewritten.
    """
    sparse_dir, tracks_dir = Path(sparse_dir), Path(tracks_dir)
    report = {'status': 'skipped', 'tracks_model': str(tracks_dir)}
    if not (tracks_dir / 'images.bin').is_file():
        report['reason'] = 'no SfM feature tracks'
        return report
    cameras = read_cameras_txt(sparse_dir / 'cameras.txt')
    images = read_images_txt(sparse_dir / 'images.txt')
    lens_of = {}
    for image_id, _, _, camera_id, name in images:
        lens = name.split('/', 1)[0]
        if lens in rig_cameras:
            model, params = cameras[camera_id]
            if model != 'THIN_PRISM_FISHEYE' or len(params) != THIN_PRISM_PARAMS:
                report['reason'] = f'{lens}: {model} is not supported by the pose adjustment'
                return report
            lens_of[name] = lens
    tracks = read_colmap_observations(tracks_dir / 'images.bin')
    used = [im for im in images if im[4] in lens_of and im[4] in tracks]
    if len(used) < 10:
        report['reason'] = f'only {len(used)} exported rig images have SfM tracks'
        return report

    # ---- state: per image pose, per frame correction, per lens camera-side rotation
    lenses = sorted({lens_of[im[4]] for im in images if im[4] in lens_of})
    frames = sorted({_frame_key(im[4]) for im in images if im[4] in lens_of})
    frame_index = {key: i for i, key in enumerate(frames)}
    rig_images = [im for im in images if im[4] in lens_of]
    R_cw_direct = np.array([im[1] for im in rig_images])
    C_direct = np.array([-im[1].T @ im[2] for im in rig_images])
    img_frame = np.array([frame_index[_frame_key(im[4])] for im in rig_images])
    img_lens = np.array([lenses.index(lens_of[im[4]]) for im in rig_images])
    img_row = {im[4]: i for i, im in enumerate(rig_images)}
    focal = {lens: float(np.mean(cameras[next(im[3] for im in rig_images if lens_of[im[4]] == lens)][1][:2]))
             for lens in lenses}

    obs_img, obs_pid, obs_b = [], [], []
    for _, _, _, camera_id, name in used:
        xy, pid = tracks[name]
        obs_img.append(np.full(len(pid), img_row[name]))
        obs_pid.append(pid)
        obs_b.append(unproject_thin_prism(xy, cameras[camera_id][1]))
    obs_b = np.concatenate(obs_b)
    valid = np.isfinite(obs_b).all(1)
    obs_img, obs_b = np.concatenate(obs_img)[valid], obs_b[valid]
    _, obs_pt = np.unique(np.concatenate(obs_pid)[valid], return_inverse=True)
    obs_E = _tangent_basis(obs_b)
    obs_f = np.array([focal[lenses[i]] for i in range(len(lenses))])[img_lens[obs_img]]
    n_points = int(obs_pt.max()) + 1
    n_frames, n_lenses = len(frames), len(lenses)

    D = np.repeat(np.eye(3)[None], n_frames, 0)   # frame rotation correction (world side)
    T = np.zeros((n_frames, 3))                    # frame translation correction
    Q = np.repeat(np.eye(3)[None], n_lenses, 0)    # lens rotation correction (camera side)
    centers0 = np.zeros((n_frames, 3))
    np.add.at(centers0, img_frame, C_direct)
    centers0 /= np.maximum(np.bincount(img_frame, minlength=n_frames), 1)[:, None]

    def poses():
        R_cw = Q[img_lens] @ R_cw_direct @ np.transpose(D[img_frame], (0, 2, 1))
        c = centers0[img_frame]
        C = c + T[img_frame] + np.einsum('nij,nj->ni', D[img_frame], C_direct - c)
        return R_cw, C

    # ---- triangulation (midpoint least squares on world rays)
    def triangulate(active):
        R_cw, C = poses()
        d = np.einsum('nji,nj->ni', R_cw[obs_img[active]], obs_b[active])
        P = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
        A = _scatter(obs_pt[active], P, n_points)
        g = _scatter(obs_pt[active], np.einsum('nij,nj->ni', P, C[obs_img[active]]), n_points)
        count = np.bincount(obs_pt[active], minlength=n_points)
        ok = (count >= 2) & (np.linalg.det(A) > 1e-9)
        X = np.full((n_points, 3), np.nan)
        X[ok] = np.linalg.solve(A[ok], g[ok][:, :, None])[:, :, 0]
        return X

    def residuals(X, active):
        R_cw, C = poses()
        i = obs_img[active]
        v = np.einsum('nij,nj->ni', R_cw[i], X[obs_pt[active]] - C[i])
        norm = np.linalg.norm(v, axis=1)
        u = v / np.maximum(norm, 1e-12)[:, None]
        r = obs_f[active, None] * np.einsum('nkj,nj->nk', obs_E[active], u)
        front = np.einsum('nj,nj->n', u, obs_b[active]) > 0.5
        return r, v, u, norm, front

    def jac_v(active, u, norm):
        """d residual / d v (camera-frame point), shape (n,2,3)."""
        Pu = np.eye(3)[None] - u[:, :, None] * u[:, None, :]
        return obs_f[active, None, None] * np.einsum('nkj,njl->nkl', obs_E[active], Pu) / norm[:, None, None]

    def weights(r):
        e = np.linalg.norm(r, axis=1)
        return np.where(e <= huber_px, 1.0, huber_px / np.maximum(e, 1e-12))

    def point_step(X, active, n_steps=2):
        for _ in range(n_steps):
            r, v, u, norm, _ = residuals(X, active)
            R_cw, _ = poses()
            J = jac_v(active, u, norm) @ R_cw[obs_img[active]]
            w = weights(r)
            H = _scatter(obs_pt[active], w[:, None, None] * np.einsum('nki,nkj->nij', J, J), n_points)
            g = _scatter(obs_pt[active], w[:, None] * np.einsum('nki,nk->ni', J, r), n_points)
            ok = np.isfinite(X).all(1) & (np.linalg.det(H) > 1e-12)
            H[ok] += 1e-6 * np.eye(3)
            X[ok] -= np.linalg.solve(H[ok], g[ok][:, :, None])[:, :, 0]
        return X

    def frame_step(X, active):
        nonlocal D, T
        r, v, u, norm, _ = residuals(X, active)
        R_cw, C = poses()
        i = obs_img[active]
        Jv = jac_v(active, u, norm)
        a = X[obs_pt[active]] - (centers0 + T)[img_frame[i]]
        J = np.concatenate([Jv @ R_cw[i] @ _skew(a), -Jv @ R_cw[i]], axis=2)
        w = weights(r)
        fr = img_frame[i]
        H = _scatter(fr, w[:, None, None] * np.einsum('nki,nkj->nij', J, J), n_frames)
        g = _scatter(fr, w[:, None] * np.einsum('nki,nk->ni', J, r), n_frames)
        wr, wp = 1 / np.radians(prior_rot_deg) ** 2, 1 / prior_pos_m ** 2
        H += np.diag([wr] * 3 + [wp] * 3)[None]
        g[:, :3] += wr * Rot.from_matrix(D).as_rotvec()
        g[:, 3:] += wp * T
        delta = -np.linalg.solve(H, g[:, :, None])[:, :, 0]
        D = Rot.from_rotvec(delta[:, :3]).as_matrix() @ D
        T = T + delta[:, 3:]

    def rig_step(X, active):
        nonlocal Q
        r, v, u, norm, _ = residuals(X, active)
        i = obs_img[active]
        J = jac_v(active, u, norm) @ (-_skew(v))
        w = weights(r)
        H = _scatter(img_lens[i], w[:, None, None] * np.einsum('nki,nkj->nij', J, J), n_lenses)
        g = _scatter(img_lens[i], w[:, None] * np.einsum('nki,nk->ni', J, r), n_lenses)
        wq = 1 / np.radians(rig_prior_deg) ** 2
        H += wq * np.eye(3)[None]
        g += wq * Rot.from_matrix(Q).as_rotvec()
        Q = Rot.from_rotvec(-np.linalg.solve(H, g[:, :, None])[:, :, 0]).as_matrix() @ Q

    def stats(X, active):
        r, *_ = residuals(X, active)
        e = np.linalg.norm(r, axis=1)
        return {'median_px': float(np.median(e)), 'mean_px': float(np.mean(e)),
                'p90_px': float(np.percentile(e, 90)), 'observations': int(active.sum()),
                'points': int(len(np.unique(obs_pt[active])))}

    def prune(X, active, threshold):
        r, _, _, _, front = residuals(X, active)
        keep = active.copy()
        keep[active] = (np.linalg.norm(r, axis=1) <= threshold) & front & np.isfinite(r).all(1)
        count = np.bincount(obs_pt[keep], minlength=n_points)
        return keep & (count[obs_pt] >= 2)

    # Baseline: the best the direct poses can do with their own triangulated points.
    active = np.ones(len(obs_pt), bool)
    X = triangulate(active)
    active &= np.isfinite(X[obs_pt]).all(1)
    for threshold in (40.0, 20.0, 12.0):
        X = point_step(X, active, 3)
        active = prune(X, active, threshold)
    X = point_step(X, active, 3)
    before = stats(X, prune(X, active, 12.0))
    log(f"  [BA] direct poses: {before['median_px']:.2f} px median reprojection "
        f"({before['observations']:,} observations, {before['points']:,} points, {n_frames} frames)")

    schedule = [12.0] * 4 + [8.0] * 4 + [6.0] * 4 + [max_error_px] * max(iterations - 12, 1)
    for k, threshold in enumerate(schedule):
        frame_step(X, active)
        if k >= 3 and n_lenses:
            rig_step(X, active)
        X = point_step(X, active, 2)
        active = prune(X, active, threshold)
        if k == 7:   # re-triangulate once the poses moved, then keep refining
            X = point_step(triangulate(active), active, 3)
    X = point_step(X, active, 3)
    active = prune(X, active, max_error_px)
    after = stats(X, active)
    # The direct poses judged on the same observations, with points triangulated for them.
    saved = D, T, Q
    D, T, Q = np.repeat(np.eye(3)[None], n_frames, 0), np.zeros((n_frames, 3)), np.repeat(np.eye(3)[None], n_lenses, 0)
    direct_same = stats(point_step(triangulate(active), active, 6), active)
    D, T, Q = saved
    log(f"  [BA] adjusted poses: {after['median_px']:.2f} px median reprojection vs "
        f"{direct_same['median_px']:.2f} px for the direct poses on the same "
        f"{after['observations']:,} observations")

    # Frames no track reached follow their neighbours' correction.
    observed = np.bincount(img_frame[obs_img[active]], minlength=n_frames) >= 20
    if observed.any() and not observed.all():
        idx = np.arange(n_frames)
        rv = Rot.from_matrix(D).as_rotvec()
        for arr in (rv, T):
            for j in range(3):
                arr[~observed, j] = np.interp(idx[~observed], idx[observed], arr[observed, j])
        D = Rot.from_rotvec(rv).as_matrix()

    shift_cm = np.linalg.norm(T, axis=1) * 100
    rot_deg = np.degrees(np.linalg.norm(Rot.from_matrix(D).as_rotvec(), axis=1))
    lens_deg = {lens: float(np.degrees(Rot.from_matrix(Q[j]).magnitude())) for j, lens in enumerate(lenses)}
    report.update({
        'before': before, 'after': after, 'direct_same_observations': direct_same,
        'frames': n_frames, 'frames_with_tracks': int(observed.sum()),
        'pose_shift_cm': {'median': float(np.median(shift_cm)), 'p95': float(np.percentile(shift_cm, 95)),
                          'max': float(shift_cm.max())},
        'pose_rotation_deg': {'median': float(np.median(rot_deg)), 'p95': float(np.percentile(rot_deg, 95)),
                              'max': float(rot_deg.max())},
        'lens_rotation_correction_deg': lens_deg,
        'settings': {'prior_pos_m': prior_pos_m, 'prior_rot_deg': prior_rot_deg, 'huber_px': huber_px,
                     'max_error_px': max_error_px, 'iterations': len(schedule)},
    })
    reasons = []
    if not after['median_px'] < direct_same['median_px']:
        reasons.append('reprojection did not improve')
    if shift_cm.max() > max_shift_m * 100:
        reasons.append(f'a frame moved {shift_cm.max():.1f} cm (limit {max_shift_m * 100:.0f} cm)')
    if rot_deg.max() > max_rotation_deg:
        reasons.append(f'a frame rotated {rot_deg.max():.2f} deg (limit {max_rotation_deg:.1f} deg)')
    if reasons:
        report.update(status='rejected', reason='; '.join(reasons))
        log(f"  [BA] adjustment rejected, keeping the direct poses: {report['reason']}")
        return report

    R_cw, C = poses()
    rows = {im[4]: k for k, im in enumerate(rig_images)}
    lines = Path(sparse_dir / 'images.txt').read_text(encoding='utf-8').splitlines()
    if backup_path is not None:
        Path(backup_path).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for n, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 10 and not line.startswith('#') and parts[9] in rows:
            k = rows[parts[9]]
            q = Rot.from_matrix(R_cw[k]).as_quat()
            t = -R_cw[k] @ C[k]
            lines[n] = ' '.join([parts[0], *(format(x, '.17g') for x in (q[3], q[0], q[1], q[2], *t)),
                                 parts[8], parts[9]])
    (sparse_dir / 'images.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report['status'] = 'accepted'
    log(f"  [BA] poses moved {np.median(shift_cm):.1f} cm / {np.median(rot_deg):.2f} deg median "
        f"(max {shift_cm.max():.1f} cm / {rot_deg.max():.2f} deg); lens corrections "
        + ', '.join(f'{k} {v:.2f} deg' for k, v in lens_deg.items()))
    return report


def write_report(path, report):
    Path(path).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
