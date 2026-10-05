"""Robust timelapse pose calibration with rejection of repeated-scene matches."""
import json
import copy
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.spatial.transform import Rotation, Slerp

VERSION = 3
CAMERA_LEVER_LENGTH_M = 0.185


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
                    positions, rotations, dt_hint=None, lever_arm_body_m=None):
    """Fit similarity and clock offset using consensus positions and rotations."""
    c, tc = np.asarray(centers), np.asarray(capture_times)
    ts = np.asarray(slam_times) - slam_times[0]
    if len(c) < 20 or np.any(np.diff(ts) <= 0):
        raise ValueError('Timelapse calibration requires 20 poses and monotonic SLAM times')
    interpolate = Slerp(ts, rotations)
    if lever_arm_body_m is None:
        arm = rotations[0].inv().apply([0., 0., .185])
    else:
        arm = np.asarray(lever_arm_body_m, dtype=float)
        if arm.shape != (3,) or not np.all(np.isfinite(arm)):
            raise ValueError('Camera lever arm must be a finite 3D vector in the SLAM body frame')

    def targets(dt):
        times = np.clip(tc - dt, ts[0], ts[-1])
        body = np.column_stack([np.interp(times, ts, positions[:,axis]) for axis in range(3)])
        return body + interpolate(times).apply(arm)

    def residual(fit, target):
        scale, rotation, translation = fit
        return np.linalg.norm(scale * c @ rotation.T + translation - target, axis=1)

    # Compare consensus to poses that can actually overlap in time. A camera
    # clip may be longer than the SLAM path because recording started/stopped
    # at different moments; out-of-range images cannot be calibration inliers.
    sorted_capture = np.sort(tc)
    slam_span = float(ts[-1] - ts[0])
    right = 0
    maximum_overlap = 0
    for left in range(len(sorted_capture)):
        right = max(right, left)
        while (right < len(sorted_capture)
               and sorted_capture[right] - sorted_capture[left] <= slam_span + 1e-9):
            right += 1
        maximum_overlap = max(maximum_overlap, right - left)
    minimum_consensus = int(np.ceil(max(20., maximum_overlap * .5)))

    rng = np.random.default_rng(42)
    best_mask = np.zeros(len(c), dtype=bool)
    best_dt = float(dt_hint) if dt_hint is not None else 0.
    candidate_dts = [best_dt, best_dt - 4., best_dt + 4., 0.] if dt_hint is not None else []
    coarse_support = []
    if dt_hint is not None:
        # The gyro estimate is a strong prior, not a reason to test only its
        # exact value and two endpoints. A small residual clock error can move
        # a walking camera by more than the 1.2 m consensus threshold. Search
        # the local gyro window before deciding that the sample is unreliable.
        positive_gaps = np.diff(sorted_capture)
        positive_gaps = positive_gaps[positive_gaps > 1e-6]
        step = min(.25, float(np.median(positive_gaps)) / 4.) if len(positive_gaps) else .25
        step = max(.05, step)
        local_grid = np.arange(best_dt - 4., best_dt + 4. + step * .5, step)
        candidate_dts.extend(float(dt) for dt in local_grid)
        candidate_dts = list(dict.fromkeys(round(dt, 6) for dt in candidate_dts))
    else:
        # Search every placement that can cover a majority of camera poses.
        # Using the whole-clip-inside-SLAM interval makes the range empty when
        # the camera recording is slightly longer than the LiDAR recording.
        dt_min = float(sorted_capture[minimum_consensus - 1] - ts[-1])
        dt_max = float(sorted_capture[len(c) - minimum_consensus] - ts[0])
        if dt_min > dt_max:
            raise ValueError(
                'Camera and SLAM recordings have no time offset with majority overlap')
        step = max(0.25, min(1.0, (dt_max - dt_min) / 1200.0))
        grid = np.arange(dt_min, dt_max + step * 0.5, step)
        for dt in grid:
            valid = (tc-dt >= ts[0]) & (tc-dt <= ts[-1])
            if valid.sum() < minimum_consensus:
                continue
            target = targets(dt)
            try:
                fit = similarity(c[valid], target[valid])
            except ValueError:
                continue
            errors = residual(fit, target)[valid]
            coarse_support.append((int(np.sum(errors < 1.2)),
                                   float(np.quantile(errors, 0.60)), float(dt)))
        # Keep several separated peaks: repeated geometry can create local
        # minima, so a single coarse winner is not sufficient for RANSAC.
        for support, _, dt in sorted(coarse_support, key=lambda item: (-item[0], item[1])):
            if all(abs(dt - existing) >= 2.0 for existing in candidate_dts):
                candidate_dts.append(dt)
            if len(candidate_dts) >= 8:
                break

    for dt in candidate_dts:
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
    if best_mask.sum() < minimum_consensus:
        support = int(best_mask.sum())
        total = int(len(c))
        detail = (f'Best camera-to-SLAM time-shift candidate Δt={best_dt:.2f}s '
                  f'matched {support}/{total} camera poses within 1.2 m.')
        if dt_hint is not None:
            detail += f' The gyro-guided search covered ±4.00 s around Δt={dt_hint:.2f}s.'
        else:
            detail += ' Gyro synchronization was unavailable; the full temporal overlap was searched.'
        raise ValueError('No majority pose consensus: timelapse reconstruction is unreliable. '
                         + detail)

    mask = best_mask
    for _ in range(3):
        def cost(dt):
            valid = (tc-dt >= ts[0]) & (tc-dt <= ts[-1])
            use = mask & valid
            if use.sum() < minimum_consensus:
                return 1e6
            target = targets(dt)
            fit = similarity(c[use], target[use])
            return float(np.mean(residual(fit, target)[use] ** 2))
        grid = np.linspace(best_dt - 8., best_dt + 8., 81)
        grid_costs = np.asarray([cost(dt) for dt in grid])
        if np.all(grid_costs >= 1e6):
            raise ValueError('Time refinement lost majority overlap with the SLAM trajectory')
        coarse = grid[np.argmin(grid_costs)]
        best_dt = minimize_scalar(cost, bounds=(coarse-.3, coarse+.3), method='bounded').x
        target = targets(best_dt)
        valid = (tc-best_dt >= ts[0]) & (tc-best_dt <= ts[-1])
        use = mask & valid
        fit = similarity(c[use], target[use])
        mask = valid & (residual(fit, target) < .5)
        if mask.sum() < minimum_consensus:
            raise ValueError(
                f'Insufficient consistent timelapse poses after refinement: '
                f'{int(mask.sum())}/{len(c)} poses are within 0.50 m '
                f'(need {minimum_consensus}); the initial 1.20 m consensus was '
                f'{int(best_mask.sum())}/{len(c)} at Δt={best_dt:.2f}s')

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


def _pose_targets(capture_times, slam_times, positions, rotations, dt, arm):
    """Interpolate LiDAR body poses and apply a body-frame camera lever arm."""
    ts = np.asarray(slam_times, dtype=float) - float(slam_times[0])
    capture = np.asarray(capture_times, dtype=float)
    query = capture - float(dt)
    valid = (query >= ts[0]) & (query <= ts[-1])
    clipped = np.clip(query, ts[0], ts[-1])
    body = np.column_stack([
        np.interp(clipped, ts, np.asarray(positions)[:, axis])
        for axis in range(3)
    ])
    body_rotation = Slerp(ts, rotations)(clipped)
    return body + body_rotation.apply(arm), valid


def estimate_lever_arm(centers, world_to_camera, capture_times, slam_times,
                       positions, rotations, dt_hint=None, initial_arm=None,
                       arm_length_m=CAMERA_LEVER_LENGTH_M):
    """Estimate a fixed-length body-frame lever arm using only supplied poses.

    Similarity, clock offset and lever direction are fit jointly. The initial
    RANSAC consensus is computed from these same poses and is used only for the
    training objective; callers must score held-out poses without trimming.
    """
    c = np.asarray(centers, dtype=float)
    rcw = np.asarray(world_to_camera, dtype=float)
    tc = np.asarray(capture_times, dtype=float)
    if len(c) < 30 or len(c) != len(tc) or len(rcw) != len(c):
        raise ValueError('Lever refinement requires at least 30 matching poses')
    if initial_arm is None:
        initial_arm = rotations[0].inv().apply([0., 0., arm_length_m])
    initial_arm = np.asarray(initial_arm, dtype=float)
    if initial_arm.shape != (3,) or not np.all(np.isfinite(initial_arm)):
        raise ValueError('Initial lever arm must be a finite 3D vector')
    initial_norm = np.linalg.norm(initial_arm)
    if initial_norm < 1e-9 or arm_length_m <= 0:
        raise ValueError('Lever arm length must be positive')
    arm0 = initial_arm / initial_norm * arm_length_m

    initial_fit, initial_dt, train_inliers, _, _ = fit_timed_poses(
        c, rcw, tc, slam_times, positions, rotations, dt_hint,
        lever_arm_body_m=arm0)
    target0, valid0 = _pose_targets(tc, slam_times, positions, rotations,
                                    initial_dt, arm0)
    ts_relative = np.asarray(slam_times, dtype=float) - float(slam_times[0])
    use = (train_inliers & valid0
           & (tc - initial_dt >= ts_relative[0] + 1.0)
           & (tc - initial_dt <= ts_relative[-1] - 1.0))
    if use.sum() < max(30, int(.5 * len(c))):
        raise ValueError('Too few training inliers for lever refinement')

    azimuth = float(np.arctan2(arm0[1], arm0[0]))
    elevation = float(np.arcsin(np.clip(arm0[2] / arm_length_m, -1., 1.)))
    x0 = np.r_[Rotation.from_matrix(initial_fit[1]).as_rotvec(),
               np.log(initial_fit[0]), initial_fit[2],
               azimuth, elevation, initial_dt]

    def unpack(x):
        direction = np.array([
            np.cos(x[8]) * np.cos(x[7]),
            np.cos(x[8]) * np.sin(x[7]),
            np.sin(x[8]),
        ])
        return (np.exp(x[3]), Rotation.from_rotvec(x[:3]).as_matrix(),
                x[4:7], direction * arm_length_m, float(x[9]))

    def residual(x):
        scale, rotation, translation, arm, dt = unpack(x)
        target, valid = _pose_targets(tc[use], slam_times, positions,
                                      rotations, dt, arm)
        if not np.all(valid):
            # Keep all training consensus poses inside the trajectory domain.
            return np.full(int(use.sum()) * 3, 1000.0)
        predicted = scale * c[use] @ rotation.T + translation
        return (predicted - target).reshape(-1)

    lower = np.full(10, -np.inf)
    upper = np.full(10, np.inf)
    lower[3], upper[3] = -5., 5.
    lower[7], upper[7] = -np.pi, np.pi
    lower[8], upper[8] = -np.pi / 2, np.pi / 2
    lower[9], upper[9] = initial_dt - 1.0, initial_dt + 1.0
    solved = least_squares(
        residual, x0, bounds=(lower, upper), loss='soft_l1', f_scale=.15,
        x_scale='jac', max_nfev=180)
    scale, rotation, translation, arm, dt = unpack(solved.x)
    target, valid = _pose_targets(tc, slam_times, positions, rotations, dt, arm)
    errors = np.linalg.norm(scale * c @ rotation.T + translation - target, axis=1)
    inliers = valid & (errors < .5)
    if inliers.sum() < max(30, int(.5 * len(c))):
        raise ValueError('Refined lever arm failed the training consensus gate')
    rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    if not np.isfinite(rmse) or rmse > .25:
        raise ValueError(f'Refined calibration rejected: training RMSE {rmse:.3f} m')
    return ((float(scale), rotation, translation), dt, inliers, rmse, arm,
            {'optimizer_success': bool(solved.success),
             'optimizer_message': str(solved.message),
             'training_pose_count': int(use.sum()),
             'arm_length_m': float(arm_length_m)})


def _score_untrimmed(centers, capture_times, slam_times, positions, rotations,
                     fit, dt, arm, indices, shared_domain=None):
    target, valid = _pose_targets(np.asarray(capture_times)[indices], slam_times,
                                  positions, rotations, dt, arm)
    selected = np.flatnonzero(valid)
    if shared_domain is not None:
        selected = selected[np.asarray(shared_domain, dtype=bool)[selected]]
    if not len(selected):
        raise ValueError('Held-out block has no timestamps inside the SLAM path')
    scale, rotation, translation = fit
    predicted = (scale * np.asarray(centers)[indices[selected]] @ rotation.T
                 + translation)
    errors = np.linalg.norm(predicted - target[selected], axis=1)
    return {'count': int(len(errors)),
            'rmse_cm': float(np.sqrt(np.mean(errors ** 2)) * 100),
            'median_cm': float(np.median(errors) * 100),
            'p90_cm': float(np.percentile(errors, 90) * 100),
            'errors_m': errors}


def validate_lever_refinement(centers, world_to_camera, capture_times,
                              slam_times, positions, rotations, initial_arm=None,
                              folds=5, purge_frames=5, dt_hint=None):
    """Leave-one-contiguous-time-block-out validation with untrimmed test rows."""
    c = np.asarray(centers, dtype=float)
    rcw = np.asarray(world_to_camera, dtype=float)
    tc = np.asarray(capture_times, dtype=float)
    if len(c) < max(100, folds * 30) or len(c) != len(tc):
        raise ValueError('Blocked validation requires at least 100 matched poses')
    if initial_arm is None:
        initial_arm = rotations[0].inv().apply([0., 0., CAMERA_LEVER_LENGTH_M])
    order = np.argsort(tc, kind='stable')
    ordered_position = np.empty(len(order), dtype=int)
    ordered_position[order] = np.arange(len(order))
    blocks = np.array_split(order, folds)
    scores = []
    for fold_index, test_indices in enumerate(blocks):
        train_mask = np.ones(len(c), dtype=bool)
        train_mask[test_indices] = False
        # Purge temporal neighbors of the held-out block from training.
        for index in test_indices:
            pos = int(ordered_position[index])
            neighbors = order[max(0, pos - purge_frames):
                              min(len(order), pos + purge_frames + 1)]
            train_mask[neighbors] = False
        train_indices = np.flatnonzero(train_mask)
        fit_args = (c[train_indices], rcw[train_indices], tc[train_indices],
                    slam_times, positions, rotations)
        base_fit, base_dt, _, base_train_rmse, base_arm = fit_timed_poses(
            *fit_args, dt_hint=dt_hint, lever_arm_body_m=initial_arm)
        candidate_fit, candidate_dt, _, candidate_train_rmse, candidate_arm, opt = estimate_lever_arm(
            *fit_args, dt_hint=base_dt, initial_arm=initial_arm)
        _, base_valid = _pose_targets(tc[test_indices], slam_times, positions,
                                      rotations, base_dt, base_arm)
        _, candidate_valid = _pose_targets(tc[test_indices], slam_times, positions,
                                           rotations, candidate_dt, candidate_arm)
        shared_domain = base_valid & candidate_valid
        baseline = _score_untrimmed(c, tc, slam_times, positions, rotations,
                                    base_fit, base_dt, base_arm, test_indices,
                                    shared_domain)
        candidate = _score_untrimmed(c, tc, slam_times, positions, rotations,
                                     candidate_fit, candidate_dt, candidate_arm,
                                     test_indices, shared_domain)
        prior = np.asarray(initial_arm, dtype=float)
        direction_change = float(np.degrees(np.arccos(np.clip(
            np.dot(prior / np.linalg.norm(prior), candidate_arm / np.linalg.norm(candidate_arm)),
            -1., 1.))))
        opt['direction_change_from_prior_deg'] = direction_change
        scores.append({'fold': fold_index + 1,
                       'train_count': int(len(train_indices)),
                       'test_count': baseline['count'],
                       'baseline_train_rmse_cm': float(base_train_rmse * 100),
                       'candidate_train_rmse_cm': float(candidate_train_rmse * 100),
                       'baseline': {k: v for k, v in baseline.items() if k != 'errors_m'},
                       'candidate': {k: v for k, v in candidate.items() if k != 'errors_m'},
                       'candidate_dt_sync_seconds': float(candidate_dt),
                       'candidate_lever_arm_body_m': candidate_arm.tolist(),
                       'optimizer': opt})
    total_count = sum(row['baseline']['count'] for row in scores)
    baseline_rmse = float(np.sqrt(sum(row['baseline']['rmse_cm']**2 * row['baseline']['count']
                                      for row in scores) / total_count))
    candidate_rmse = float(np.sqrt(sum(row['candidate']['rmse_cm']**2 * row['candidate']['count']
                                       for row in scores) / total_count))
    fold_wins = sum(row['candidate']['rmse_cm'] < row['baseline']['rmse_cm']
                    for row in scores)
    counts_match = all(row['baseline']['count'] == row['candidate']['count']
                       for row in scores)
    optimizer_ok = all(row['optimizer']['optimizer_success'] for row in scores)
    direction_ok = all(row['optimizer']['direction_change_from_prior_deg'] <= 45.
                       for row in scores)
    accepted = bool(counts_match and optimizer_ok and direction_ok
                    and candidate_rmse <= baseline_rmse * .97
                    and fold_wins >= folds - 1)
    return {'folds': scores, 'baseline_untrimmed_rmse_cm': baseline_rmse,
            'candidate_untrimmed_rmse_cm': candidate_rmse,
            'relative_rmse_reduction': float(1. - candidate_rmse / baseline_rmse),
            'fold_wins': int(fold_wins), 'fold_count': int(folds),
            'minimum_required_reduction': .03,
            'minimum_required_fold_wins': int(folds - 1),
            'shared_test_domain': bool(counts_match),
            'optimizer_success_all_folds': bool(optimizer_ok),
            'direction_change_limit_deg': 45.,
            'direction_plausibility_all_folds': bool(direction_ok),
            'accepted': accepted,
            'test_policy': 'all valid held-out camera poses; no residual trimming'}


def refined_rig_sidecar(calibration, alignment):
    """Return a copy with the validated cam0 lever arm; never writes source JSON."""
    result = copy.deepcopy(calibration)
    refinement = alignment.get('lever_arm_refinement', {})
    if not refinement.get('accepted'):
        return result
    arm = np.asarray(alignment['initial_camera_lever_body_m'], dtype=float)
    norm = float(np.linalg.norm(arm))
    if arm.shape != (3,) or not np.isfinite(norm) or abs(norm - CAMERA_LEVER_LENGTH_M) > .002:
        raise ValueError('Validated camera lever arm must have the physical 185 mm length')
    key = 'T_lidar_to_cam0_rigid_4x4'
    matrix = np.asarray(result[key], dtype=float)
    matrix[:3, 3] = arm
    result[key] = matrix.tolist()
    result['lever_arm_cam0_meters'] = {
        'dx': float(arm[0]), 'dy': float(arm[1]), 'dz': float(arm[2]),
        'norm_distance_cm': norm * 100.}
    result['timelapse_calibration_version'] = VERSION
    result['lever_arm_refinement'] = copy.deepcopy(refinement)
    return result


def align_dataset(dataset_dir, dt_hint=None, output_path=None, validate_refinement=True,
                  trajectory_path=None):
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    root = Path(dataset_dir)
    images = pipeline.load_colmap_images(root/'sparse/0/images.bin')
    front = sorted([im for im in images if im['name'].startswith('cam0/')], key=lambda im: im['name'])
    trajectory_path = trajectory_path or pipeline.get_slam_trajectory_path(root)
    ts, positions, rotations = pipeline.load_trajectory(trajectory_path)
    times = np.array([pipeline.frame_time(root, im['name']) for im in front])
    centers = np.array([im['C'] for im in front])
    rcw = np.array([im['R_cw'] for im in front])
    initial_arm = rotations[0].inv().apply([0., 0., CAMERA_LEVER_LENGTH_M])
    refinement = None
    if validate_refinement and len(centers) >= 100:
        try:
            refinement = validate_lever_refinement(
                centers, rcw, times, ts, positions, rotations,
                initial_arm=initial_arm, dt_hint=dt_hint)
        except (ValueError, np.linalg.LinAlgError) as exc:
            refinement = {'accepted': False, 'reason': str(exc),
                          'test_policy': 'all valid held-out camera poses; no residual trimming'}
    fit, dt, inliers, rmse, arm = fit_timed_poses(
        centers, rcw, times, ts, positions, rotations, dt_hint,
        lever_arm_body_m=initial_arm)
    if refinement and refinement.get('accepted'):
        fit, dt, inliers, rmse, arm, optimizer = estimate_lever_arm(
            centers, rcw, times, ts, positions, rotations,
            dt_hint=dt, initial_arm=initial_arm)
        refinement['production_fit'] = optimizer
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
        minimum_valid = int(np.ceil(max(20., int(valid.sum()) * .5)))
        good = valid & (errors < .5)
        if good.sum() < minimum_valid:
            raise ValueError(
                f'Too few consistent timelapse poses for {lens}: '
                f'{int(good.sum())}/{int(valid.sum())} overlapping poses are within 0.50 m '
                f'(need {minimum_valid})')
        position_inliers = good.copy()
        minimum_rotation = int(np.ceil(max(20., int(position_inliers.sum()) * .5)))
        angle = (local[position_inliers].mean().inv()*local).magnitude()*180/np.pi
        rotation_inliers = position_inliers & (angle < 3.)
        if rotation_inliers.sum() < minimum_rotation:
            raise ValueError(
                f'Timelapse calibration rejected: rotation consensus for {lens} is '
                f'{int(rotation_inliers.sum())}/{int(position_inliers.sum())} position-inlier '
                f'poses within 3 degrees (need {minimum_rotation})')
        good = rotation_inliers
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
    if refinement is not None:
        result['lever_arm_refinement'] = refinement
    target_path = Path(output_path) if output_path is not None else root/'colmap_to_lidar_alignment.json'
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[+] Timelapse calibration: dt={dt:.4f}s, trajectory RMSE={rmse*100:.2f} cm; '
          f'{len(accepted)} accepted / {len(rejected)} rejected SfM poses')
    return result


def rejected_pose_overrides(dataset_dir, alignment, images, calibration=None):
    """Replace rejected SfM poses with calibrated camera poses on the LiDAR path."""
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    root = Path(dataset_dir)
    if calibration is None:
        cfg = pipeline.recalibrate_from_sfm(root)
    elif isinstance(calibration, (str, Path)):
        cfg = json.loads(Path(calibration).read_text(encoding='utf-8'))
    else:
        cfg = calibration
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
