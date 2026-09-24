"""Robust timelapse pose calibration with rejection of repeated-scene matches."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

VERSION = 1


def similarity(x, y):
    xc, yc = x - x.mean(0), y - y.mean(0)
    variance = np.mean(np.sum(xc * xc, axis=1))
    if variance < 1e-10:
        raise ValueError('Insufficient camera translation for calibration')
    u, d, vt = np.linalg.svd(yc.T @ xc / len(x))
    sign = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ sign @ vt
    scale = np.sum(d * np.diag(sign)) / variance
    translation = y.mean(0) - scale * rotation @ x.mean(0)
    return scale, rotation, translation


def fit_timed_poses(centers, world_to_camera, capture_times, slam_times,
                    positions, rotations, dt_hint=None):
    """Fit similarity and clock offset using consensus positions and rotations."""
    c, tc = np.asarray(centers), np.asarray(capture_times)
    ts = np.asarray(slam_times) - slam_times[0]
    if len(c) < 20 or np.any(np.diff(ts) <= 0):
        raise ValueError('Timelapse calibration requires 20 poses and monotonic SLAM times')
    interpolate = Slerp(ts, rotations)
    arm = rotations[0].inv().apply([0., 0., .185])

    def targets(dt):
        times = np.clip(tc - dt, ts[0], ts[-1])
        body = np.column_stack([np.interp(times, ts, positions[:,axis]) for axis in range(3)])
        return body + interpolate(times).apply(arm)

    def residual(fit, target):
        scale, rotation, translation = fit
        return np.linalg.norm(scale * c @ rotation.T + translation - target, axis=1)

    hint = float(dt_hint) if dt_hint is not None else 0.
    rng = np.random.default_rng(42)
    best_mask, best_dt = np.zeros(len(c), dtype=bool), hint
    for dt in (hint, hint - 4., hint + 4., 0.):
        target = targets(dt)
        valid = (tc-dt >= ts[0]) & (tc-dt <= ts[-1])
        indices = np.flatnonzero(valid)
        if len(indices) < 20:
            continue
        for _ in range(300):
            sample = rng.choice(indices, 4, replace=False)
            try:
                fit = similarity(c[sample], target[sample])
            except ValueError:
                continue
            mask = valid & (residual(fit, target) < 1.2)
            if mask.sum() > best_mask.sum():
                best_mask, best_dt = mask, dt
    if best_mask.sum() < max(20, len(c) * .5):
        raise ValueError('No majority pose consensus: timelapse reconstruction is unreliable')

    mask = best_mask
    for _ in range(3):
        def cost(dt):
            target = targets(dt)
            fit = similarity(c[mask], target[mask])
            return float(np.mean(residual(fit, target)[mask] ** 2))
        grid = np.linspace(best_dt - 8., best_dt + 8., 81)
        coarse = grid[np.argmin([cost(dt) for dt in grid])]
        best_dt = minimize_scalar(cost, bounds=(coarse-.3, coarse+.3), method='bounded').x
        target = targets(best_dt)
        fit = similarity(c[mask], target[mask])
        mask = (residual(fit, target) < .5) & (tc-best_dt >= 0) & (tc-best_dt <= ts[-1])
        if mask.sum() < max(20, len(c) * .5):
            raise ValueError('Insufficient consistent timelapse poses after refinement')

    # Frame timing is most observable during turns. Translation alone confounds
    # clock offset with the camera lever arm while walking at constant speed.
    rotation = fit[1]
    world_camera = Rotation.from_matrix(np.einsum('ij,nkj->nik', rotation, world_to_camera))
    angular_mask = mask & (tc > best_dt + 1.) & (tc < ts[-1] + best_dt - 1.)

    def angular_cost(dt):
        extrinsics = interpolate(tc[angular_mask]-dt).inv() * world_camera[angular_mask]
        errors = (extrinsics.mean().inv() * extrinsics).magnitude()
        return float(np.mean(np.minimum(errors, np.deg2rad(5.)) ** 2))

    angular_samples = [angular_cost(best_dt+shift) for shift in (-.4, 0., .4)]
    if max(angular_samples)-min(angular_samples) > 1e-8:
        best_dt = minimize_scalar(angular_cost, bounds=(best_dt-.5, best_dt+.5), method='bounded').x
    target = targets(best_dt)
    fit = similarity(c[mask], target[mask])
    errors = residual(fit, target)
    mask &= errors < .5
    rmse = float(np.sqrt(np.mean(errors[mask] ** 2)))
    if not np.isfinite(rmse) or rmse > .25:
        raise ValueError(f'Timelapse trajectory calibration rejected: RMSE {rmse:.3f} m')
    return fit, float(best_dt), mask, rmse, arm


def align_dataset(dataset_dir, dt_hint=None):
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    root = Path(dataset_dir)
    images = pipeline.load_colmap_images(root/'sparse/0/images.bin')
    front = sorted([im for im in images if im['name'].startswith('cam0/')], key=lambda im: im['name'])
    ts, positions, rotations = pipeline.load_trajectory(pipeline.get_slam_trajectory_path(root))
    times = np.array([pipeline.frame_time(root, im['name']) for im in front])
    centers = np.array([im['C'] for im in front])
    rcw = np.array([im['R_cw'] for im in front])
    fit, dt, inliers, rmse, arm = fit_timed_poses(centers, rcw, times, ts, positions, rotations, dt_hint)
    scale, rotation, translation = fit
    interpolate = Slerp(ts-ts[0], rotations)
    accepted, rejected = [], []
    angular_errors = []
    for lens in ('cam0/', 'cam1/'):
        group = [im for im in images if im['name'].startswith(lens)]
        capture = np.array([pipeline.frame_time(root, im['name']) for im in group])
        valid = (capture-dt >= 0) & (capture-dt <= ts[-1]-ts[0])
        query = np.clip(capture-dt, 0, ts[-1]-ts[0])
        body_rotations = interpolate(query)
        body = np.column_stack([np.interp(query, ts-ts[0], positions[:,axis]) for axis in range(3)])
        transformed = scale*np.array([im['C'] for im in group])@rotation.T + translation
        errors = np.linalg.norm(transformed-body-body_rotations.apply(arm), axis=1)
        local = body_rotations.inv()*Rotation.from_matrix(np.einsum('ij,nkj->nik', rotation, np.array([im['R_cw'] for im in group])))
        good = valid & (errors < .5)
        if good.sum() < 20:
            raise ValueError(f'Too few consistent timelapse poses for {lens}')
        angle = (local[good].mean().inv()*local).magnitude()*180/np.pi
        good &= angle < 3.
        if good.sum() < max(20, len(group)*.5):
            raise ValueError(f'Timelapse calibration rejected: no majority rotation consensus for {lens}')
        angular_errors.extend(angle[good].tolist())
        accepted.extend(im['name'] for im, ok in zip(group, good) if ok)
        rejected.extend(im['name'] for im, ok in zip(group, good) if not ok)
    result = dict(calibration_version=pipeline.CALIBRATION_VERSION,
                  timelapse_calibration_version=VERSION, frame_time_source='insv_timelapse',
                  initial_camera_lever_body_m=arm.tolist(), trajectory_rmse_cm=rmse*100,
                  scale=float(scale), R=rotation.tolist(), t=translation.tolist(),
                  dt_sync_seconds=dt, accepted_image_names=accepted,
                  rejected_image_names=rejected, quality_status='accepted',
                  rotation_p90_deg=float(np.percentile(angular_errors,90)))
    (root/'colmap_to_lidar_alignment.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[+] Timelapse calibration: dt={dt:.4f}s, trajectory RMSE={rmse*100:.2f} cm; '
          f'{len(accepted)} accepted / {len(rejected)} rejected SfM poses')
    return result


def rejected_pose_overrides(dataset_dir, alignment, images):
    """Replace rejected SfM poses with calibrated camera poses on the LiDAR path."""
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    root = Path(dataset_dir)
    cfg = pipeline.recalibrate_from_sfm(root)
    ts, positions, rotations = pipeline.load_trajectory(pipeline.get_slam_trajectory_path(root))
    interpolate = Slerp(ts-ts[0], rotations)
    rejected = set(alignment['rejected_image_names'])
    result = {}
    for im in images:
        if im['name'] not in rejected:
            continue
        query = pipeline.frame_time(root, im['name'])-alignment['dt_sync_seconds']
        if not 0 <= query <= ts[-1]-ts[0]:
            result[im['name']] = None
            continue
        lens = 0 if im['name'].startswith('cam0/') else 1
        transform = np.array(cfg[f'T_lidar_to_cam{lens}_rigid_4x4'])
        body_rotation = interpolate(query).as_matrix()
        body = np.array([np.interp(query,ts-ts[0],positions[:,axis]) for axis in range(3)])
        result[im['name']] = ((body_rotation@transform[:3,:3]).T, body+body_rotation@transform[:3,3])
    return result
