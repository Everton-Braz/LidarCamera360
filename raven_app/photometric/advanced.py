"""Compact PPISP-style homography and diagonal bilateral-grid correction.

The homography follows the no-CRF/no-vignetting PPISP color stage: transform
RGI chromaticity and renormalize to preserve RGB intensity. Exposure remains in
the existing per-view log gains. The small bilateral grid uses diagonal RGB
affines (three log gains per cell), rather than Spirula's full 12-value affine
grid. This keeps the runtime payload small while retaining XY/luminance
trilinear interpolation.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

from .model import _metrics, _pairs, to_linear

GRID_SHAPE = (4, 3, 3, 3)  # depth, height, width, diagonal RGB log gains
GRID_LIMIT = 0.35
IMAGE_SIZE = 3840.0
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
_IDENTITY_H = np.eye(3, dtype=np.float64)
ADVANCED_MODEL_REVISION = 2


def _homography(rgb, h):
    """Apply PPISP's intensity-preserving RGI homography to linear RGB."""
    rgb = np.asarray(rgb, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64).reshape(3, 3)
    intensity = np.sum(rgb, axis=-1)
    rgi = np.stack((rgb[..., 0], rgb[..., 1], intensity), axis=-1)
    warped = rgi @ h.T
    floor = 1e-4 * np.abs(intensity) + 1e-8
    denominator = np.maximum(warped[..., 2], floor)
    red = warped[..., 0] * intensity / denominator
    green = warped[..., 1] * intensity / denominator
    return np.stack((red, green, intensity - red - green), axis=-1)


def _apply_frame_homographies(rgb, frame_id, homographies):
    intensity = np.sum(rgb, axis=1)
    rgi = np.column_stack((rgb[:, 0], rgb[:, 1], intensity))
    warped = np.einsum('nij,nj->ni', homographies[frame_id], rgi,
                       optimize=True)
    denominator = np.maximum(warped[:, 2], 1e-4 * np.abs(intensity) + 1e-8)
    red = warped[:, 0] * intensity / denominator
    green = warped[:, 1] * intensity / denominator
    return np.clip(np.column_stack((red, green, intensity - red - green)), 0, 1)


def _grid_sample(grid, rgb, u, v):
    """Spirula-style XY/luminance guide, with trilinear cell interpolation."""
    grid = np.asarray(grid, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    u, v = np.broadcast_arrays(np.asarray(u, dtype=np.float64),
                               np.asarray(v, dtype=np.float64))
    depth, height, width, channels = grid.shape
    if (depth, height, width, channels) != GRID_SHAPE:
        raise ValueError('bilateral_grid must have shape (4, 3, 3, 3)')
    coords = np.stack((np.clip(v / IMAGE_SIZE, 0, 1) * (height - 1),
                       np.clip(u / IMAGE_SIZE, 0, 1) * (width - 1),
                       np.clip(rgb @ _LUMA, 0, 1) * (depth - 1)), axis=-1)
    y0 = np.floor(coords[..., 0]).astype(np.int64)
    x0 = np.floor(coords[..., 1]).astype(np.int64)
    z0 = np.floor(coords[..., 2]).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    z1 = np.minimum(z0 + 1, depth - 1)
    fy = coords[..., 0] - y0
    fx = coords[..., 1] - x0
    fz = coords[..., 2] - z0
    out = np.zeros(rgb.shape, dtype=np.float64)
    for zz, wz in ((z0, 1 - fz), (z1, fz)):
        for yy, wy in ((y0, 1 - fy), (y1, fy)):
            for xx, wx in ((x0, 1 - fx), (x1, fx)):
                out += grid[zz, yy, xx] * (wz * wy * wx)[..., None]
    return out


def apply_linear(rgb_linear, u, v, coeff):
    """Apply optional PPISP H then bounded diagonal bilateral-grid gains."""
    out = np.asarray(rgb_linear, dtype=np.float64)
    if 'ppisp_h' in coeff:
        out = np.clip(_homography(out, coeff['ppisp_h']), 0, 1)
    if 'bilateral_grid' in coeff:
        log_gain = _grid_sample(coeff['bilateral_grid'], np.clip(out, 0, 1), u, v)
        out = out * np.exp(np.clip(log_gain, -GRID_LIMIT, GRID_LIMIT))
    return np.clip(out, 0, 1)


def _split(point_id, seed):
    ids, inverse, counts = np.unique(point_id, return_inverse=True,
                                     return_counts=True)
    if len(ids) < 30 or np.any(counts < 3):
        raise ValueError('Need at least 30 points with three observations each')
    order = np.random.default_rng(seed).permutation(len(ids))
    groups = np.zeros(len(ids), dtype=np.uint8)
    groups[order[int(.6 * len(ids)):int(.8 * len(ids))]] = 1
    groups[order[int(.8 * len(ids)):]] = 2
    return groups[inverse]


def _loo_targets(point_id, rgb, weight, selected):
    ix = np.flatnonzero(selected)
    _, inverse, counts = np.unique(point_id[ix], return_inverse=True,
                                   return_counts=True)
    mass = np.bincount(inverse, weights=weight[ix])
    sums = np.zeros((len(mass), 3), dtype=np.float64)
    np.add.at(sums, inverse, rgb[ix] * weight[ix, None])
    denom = mass[inverse] - weight[ix]
    keep = (counts[inverse] >= 3) & (denom > 1e-6)
    target = np.zeros((len(ix), 3), dtype=np.float64)
    target[keep] = ((sums[inverse[keep]] - rgb[ix[keep]] * weight[ix[keep], None]) /
                    denom[keep, None])
    result = np.full((len(point_id), 3), np.nan, dtype=np.float64)
    result[ix[keep]] = target[keep]
    return result


def _fit_homography(src, dst, weights, ridge=10.0):
    """Ridge-regularized 2-D projective chromaticity fit, anchored at identity."""
    total = np.sum(src, axis=1)
    target_total = np.sum(dst, axis=1)
    good = (np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) &
            (total > 1e-5) & (target_total > 1e-5))
    if np.count_nonzero(good) < 16:
        return _IDENTITY_H.copy()
    q = src[good, :2] / total[good, None]
    target = dst[good, :2] / target_total[good, None]
    base_w = np.maximum(weights[good], 1e-4)
    x, y = q[:, 0], q[:, 1]
    tx, ty = target[:, 0], target[:, 1]
    n = len(q)
    a = np.zeros((2 * n, 8), dtype=np.float64)
    b = np.empty(2 * n, dtype=np.float64)
    a[:n, 0:3] = np.column_stack((x, y, np.ones(n)))
    a[:n, 6:8] = -np.column_stack((tx * x, tx * y))
    b[:n] = tx
    a[n:, 3:6] = np.column_stack((x, y, np.ones(n)))
    a[n:, 6:8] = -np.column_stack((ty * x, ty * y))
    b[n:] = ty
    row_w = np.tile(base_w, 2)
    identity = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    params = identity.copy()
    for _ in range(3):
        residual = a @ params - b
        robust = np.minimum(1.0, 0.035 / np.maximum(np.abs(residual), 1e-8))
        w = row_w * robust
        lhs = a.T @ (a * w[:, None]) + ridge * np.eye(8)
        rhs = a.T @ (b * w) + ridge * identity
        try:
            params = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            return _IDENTITY_H.copy()
    h = np.array([[params[0], params[1], params[2]],
                  [params[3], params[4], params[5]],
                  [params[6], params[7], 1.0]])
    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-5:
        return _IDENTITY_H.copy()
    # Reject extreme chroma warps before they can create clipping or color halos.
    if np.max(np.abs(h - _IDENTITY_H)) > 0.18:
        h = _IDENTITY_H + np.clip(h - _IDENTITY_H, -0.18, 0.18)
        h[2, 2] = 1.0
    return h


def _bilinear_design(obs, rgb, indices, target, lens_for_frame, nlenses):
    """Sparse trilinear diagonal-grid design for one shared grid per lens."""
    depth, height, width, _ = GRID_SHAPE
    u, v = obs['u'][indices] / IMAGE_SIZE, obs['v'][indices] / IMAGE_SIZE
    yy = np.clip(v, 0, 1) * (height - 1)
    xx = np.clip(u, 0, 1) * (width - 1)
    zz = np.clip(rgb @ _LUMA, 0, 1) * (depth - 1)
    low = np.floor(np.stack((zz, yy, xx), axis=1)).astype(np.int64)
    frac = np.stack((zz, yy, xx), axis=1) - low
    high = np.minimum(low + 1, np.array([depth - 1, height - 1, width - 1]))
    lens = lens_for_frame[obs['frame_id'][indices]]
    n, channels = len(indices), 3
    rows = np.repeat(np.arange(n * channels), 8)
    cols = np.empty(n * channels * 8, dtype=np.int64)
    vals = np.empty(n * channels * 8, dtype=np.float64)
    rgb_index = np.tile(np.arange(3), n)
    lens_index = lens.repeat(3)
    for corner in range(8):
        zhi, yhi, xhi = ((corner >> 2) & 1, (corner >> 1) & 1, corner & 1)
        z = high[:, 0] if zhi else low[:, 0]
        y = high[:, 1] if yhi else low[:, 1]
        x = high[:, 2] if xhi else low[:, 2]
        wz = frac[:, 0] if zhi else 1 - frac[:, 0]
        wy = frac[:, 1] if yhi else 1 - frac[:, 1]
        wx = frac[:, 2] if xhi else 1 - frac[:, 2]
        cell = ((z * height + y) * width + x)
        slot = (np.arange(n * channels) * 8) + corner
        cols[slot] = (lens_index * (depth * height * width * 3) +
                      cell.repeat(3) * 3 + rgb_index)
        vals[slot] = (wz * wy * wx).repeat(3)
    rows2 = np.repeat(np.arange(n), channels)
    target_log = np.log(np.maximum(target, 1e-5)) - np.log(np.maximum(rgb, 1e-5))
    return sparse.coo_matrix((vals, (rows, cols)),
                             shape=(n * channels, nlenses * depth * height * width * 3)).tocsr(), \
        target_log.reshape(-1), rows2


def _fit_shared_grids(obs, train_ix, source, targets, lens_for_frame, nlenses,
                      smoothness=2.0):
    keep = np.isfinite(targets[train_ix]).all(axis=1)
    ix = train_ix[keep]
    x, y = source[ix], targets[ix]
    if len(ix) < 100:
        return np.zeros((nlenses,) + GRID_SHAPE, dtype=np.float64)
    design, values, sample_rows = _bilinear_design(obs, x, ix, y,
                                                    lens_for_frame, nlenses)
    sample_weight = np.sqrt(np.maximum(obs['weight'][ix], .02))
    row_weight = np.repeat(sample_weight, 3)
    weighted = design.multiply(row_weight[:, None]).tocsr()
    rhs = values * row_weight
    # First-difference smoothness plus a zero-mean color anchor per lens.
    shape = GRID_SHAPE[:3]
    edge_rows, edge_cols, edge_vals = [], [], []
    row = 0
    for lens in range(nlenses):
        for z in range(shape[0]):
            for y0 in range(shape[1]):
                for x0 in range(shape[2]):
                    cell = (z * shape[1] + y0) * shape[2] + x0
                    for axis, limit in ((0, shape[0]), (1, shape[1]), (2, shape[2])):
                        coord = [z, y0, x0]
                        if coord[axis] + 1 >= limit:
                            continue
                        coord[axis] += 1
                        next_cell = (coord[0] * shape[1] + coord[1]) * shape[2] + coord[2]
                        for c in range(3):
                            edge_rows.extend((row, row))
                            edge_cols.extend((lens * 108 + cell * 3 + c,
                                              lens * 108 + next_cell * 3 + c))
                            edge_vals.extend((1.0, -1.0))
                            row += 1
    smooth = sparse.coo_matrix((edge_vals, (edge_rows, edge_cols)),
                               shape=(row, nlenses * 108)).tocsr()
    # Keep the average correction at zero: the existing gain fit owns exposure/WB.
    anchor_rows, anchor_cols, anchor_vals = [], [], []
    ar = 0
    for lens in range(nlenses):
        for channel in range(3):
            anchor_rows.extend([ar] * 36)
            anchor_cols.extend([lens * 108 + cell * 3 + channel for cell in range(36)])
            anchor_vals.extend([1.0 / 36.0] * 36)
            ar += 1
    anchor = sparse.coo_matrix((anchor_vals, (anchor_rows, anchor_cols)),
                               shape=(ar, nlenses * 108)).tocsr()
    reg = sparse.vstack((smooth * float(smoothness), sparse.eye(nlenses * 108) * 0.15,
                         anchor * 16.0), format='csr')
    solution = lsqr(sparse.vstack((weighted, reg), format='csr'),
                    np.r_[rhs, np.zeros(reg.shape[0])], atol=1e-5,
                    btol=1e-5, iter_lim=180)[0]
    grids = solution.reshape(nlenses, *GRID_SHAPE)
    # Enforce zero mean on the actual train samples, not merely the grid nodes.
    train_lens = lens_for_frame[obs['frame_id'][ix]]
    for lens in range(nlenses):
        local = np.flatnonzero(train_lens == lens)
        if not len(local):
            continue
        sampled = _grid_sample(grids[lens], np.clip(source[ix[local]], 0, 1),
                               obs['u'][ix[local]], obs['v'][ix[local]])
        mean = np.average(sampled, axis=0,
                          weights=np.maximum(obs['weight'][ix[local]], .02))
        grids[lens] -= mean[None, None, None, :]
        peak = np.max(np.abs(grids[lens]))
        if peak > GRID_LIMIT:
            grids[lens] *= GRID_LIMIT / peak
    return grids


def _base_correct(obs, views, names):
    linear = to_linear(obs['rgb'])
    frame = obs['frame_id']
    gains = np.zeros((len(names), 3), dtype=np.float64)
    vignette = np.zeros((len(names), 3), dtype=np.float64)
    support = np.zeros(len(names), dtype=np.int64)
    for i, name in enumerate(names):
        coeff = views.get(str(name), {})
        gains[i] = coeff.get('log_gain', (0, 0, 0))
        vignette[i] = coeff.get('vignette', (0, 0, 0))
        support[i] = int(coeff.get('support', 0))
    radial = obs['radius'][:, None] ** np.array([2, 4, 6])
    log_factor = gains[frame] + np.sum(vignette[frame] * radial, axis=1)[:, None]
    corrected = linear * np.exp(-np.clip(log_factor, -np.log(2), np.log(2)))
    corrected[support[frame] < 12] = linear[support[frame] < 12]
    return np.clip(corrected, 0, 1)


def _apply_shared_grids(obs, source, grids, lens_for_frame, supported_frames):
    """Apply per-lens grids only to frames with enough training support."""
    output = np.asarray(source, dtype=np.float64).copy()
    frame = obs['frame_id']
    eligible = supported_frames[frame]
    for lens in range(len(grids)):
        ix = np.flatnonzero(eligible & (lens_for_frame[frame] == lens))
        if not len(ix):
            continue
        correction = _grid_sample(grids[lens], np.clip(source[ix], 0, 1),
                                  obs['u'][ix], obs['v'][ix])
        output[ix] = np.clip(source[ix] * np.exp(
            np.clip(correction, -GRID_LIMIT, GRID_LIMIT)), 0, 1)
    return output


def _choose_candidate(baseline, candidates):
    """Choose an advanced model using validation RMSE and Delta E only."""
    selected = ('gains-only', baseline, None, None, None)
    for candidate in candidates:
        name, score, corrected, homographies, grids = candidate
        incumbent = selected[1]
        if (score['linear_rmse'] < baseline['linear_rmse'] * .995 and
                score['delta_e76_mean'] < baseline['delta_e76_mean'] and
                score['linear_rmse'] < incumbent['linear_rmse'] * .995 and
                score['delta_e76_mean'] < incumbent['delta_e76_mean']):
            selected = candidate
    return selected


def fit_advanced(observations, names, lens_names, baseline, seed=20260929,
                 grid_smoothness=2.0):
    """Fit/select PPISP H and shared per-lens grids using a fixed point split.

    Metrics include independent PPISP-only and grid-only fits on the same
    point-level train/validation/test split, so the two components can be
    compared without allowing either to see held-out points.
    """
    started = time.perf_counter()
    obs = {k: np.asarray(v) for k, v in observations.items()}
    names = list(map(str, names))
    lens_names = list(map(str, lens_names))
    n = len(obs['point_id'])
    if not n or len(obs['rgb']) != n or obs['rgb'].shape != (n, 3):
        raise ValueError('Empty/inconsistent photometric observations')
    split = _split(obs['point_id'], seed)
    train, validation, test = split == 0, split == 1, split == 2
    x = _base_correct(obs, baseline['views'], names)
    lens_for_frame = np.zeros(len(names), dtype=np.int64)
    frame_lens = {}
    for f, l in zip(obs['frame_id'], obs['lens_id']):
        frame_lens.setdefault(int(f), int(l))
    for frame, lens in frame_lens.items():
        lens_for_frame[frame] = lens
    nlenses = len(lens_names)
    support = np.bincount(obs['frame_id'][train], minlength=len(names))
    target = _loo_targets(obs['point_id'], x, obs['weight'], train)
    h_matrices = np.broadcast_to(_IDENTITY_H, (len(names), 3, 3)).copy()
    for frame in np.unique(obs['frame_id'][train]):
        ix = np.flatnonzero(train & (obs['frame_id'] == frame) &
                            np.isfinite(target).all(axis=1))
        if len(ix) < 16:
            continue
        h_matrices[frame] = _fit_homography(x[ix], target[ix], obs['weight'][ix])
    # Anchor gauge separately in each previously connected image component.
    supported_frames = support >= 16
    component_for_frame = np.zeros(len(names), dtype=np.int64)
    for frame, name in enumerate(names):
        component_for_frame[frame] = int(baseline['views'].get(name, {}).get(
            'component', lens_for_frame[frame]))
    anchor = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=np.float64)
    for component in np.unique(component_for_frame[supported_frames]):
        frames = np.flatnonzero(supported_frames &
                                (component_for_frame == component))
        if not len(frames):
            continue
        params = h_matrices[frames].reshape(-1, 9)[:, :8]
        delta = np.median(params, axis=0) - anchor
        for frame in frames:
            h = h_matrices[frame].reshape(-1).copy()
            h[:8] -= delta
            h[8] = 1.0
            h_matrices[frame] = h.reshape(3, 3)
    after_h = _apply_frame_homographies(x, obs['frame_id'], h_matrices)
    validation_before = _metrics(obs, validation, x)
    validation_h = _metrics(obs, validation, after_h)
    # Fit two camera grids jointly with strong smoothness and zero-mean anchors.
    train_ix = np.flatnonzero(train)
    target_h = _loo_targets(obs['point_id'], after_h, obs['weight'], train)
    grids = _fit_shared_grids(obs, train_ix, after_h, target_h,
                              lens_for_frame, nlenses,
                              smoothness=grid_smoothness)
    grid_after = _apply_shared_grids(obs, after_h, grids, lens_for_frame,
                                     supported_frames)
    validation_grid = _metrics(obs, validation, grid_after)

    # Fit a second grid directly on the log-linear output for a true grid-only
    # ablation. This fit uses the same train points and held-out partitions.
    target_base = _loo_targets(obs['point_id'], x, obs['weight'], train)
    grids_only = _fit_shared_grids(obs, train_ix, x, target_base,
                                   lens_for_frame, nlenses,
                                   smoothness=grid_smoothness)
    grid_only_after = _apply_shared_grids(obs, x, grids_only, lens_for_frame,
                                          supported_frames)
    validation_grid_only = _metrics(obs, validation, grid_only_after)
    # Select among independent candidates using validation only. Every
    # advanced candidate must beat log-linear by at least 0.5% in RMSE and
    # improve mean Delta E; tie-breaking keeps the incumbent model simpler.
    candidates = (
        ('ppisp-homography', validation_h, after_h, h_matrices, np.zeros_like(grids)),
        ('bilateral-grid-only', validation_grid_only, grid_only_after,
         np.broadcast_to(_IDENTITY_H, (len(names), 3, 3)).copy(), grids_only),
        ('ppisp-bilateral-grid', validation_grid, grid_after, h_matrices, grids),
    )
    selected, validation_after, best, chosen_h, chosen_grids = _choose_candidate(
        validation_before, candidates)
    if selected == 'gains-only':
        best = x
        chosen_h = np.broadcast_to(_IDENTITY_H, (len(names), 3, 3)).copy()
        chosen_grids = np.zeros_like(grids)
    views = {}
    for frame, name in enumerate(names):
        base_coeff = dict(baseline['views'].get(name, {}))
        if support[frame] >= 16 and selected != 'gains-only':
            if selected in ('ppisp-homography', 'ppisp-bilateral-grid'):
                base_coeff['ppisp_h'] = chosen_h[frame].reshape(-1).tolist()
            if selected in ('bilateral-grid-only', 'ppisp-bilateral-grid'):
                base_coeff['bilateral_grid'] = chosen_grids[lens_for_frame[frame]].tolist()
        views[name] = base_coeff
    report = dict(baseline)
    report.update({
        'schema': 2,
        'mode': 'ppisp-bilateral',
        'advanced_model_revision': ADVANCED_MODEL_REVISION,
        'selected': selected,
        'variants': {
            'ppisp': 'no_crf_no_vig: exposure is provided by existing per-view gains; intensity-preserving RGI homography',
            'bilateral_grid': 'diagonal RGB log-gain grid, D4xH3xW3, trilinear XY/Rec.709-luminance sampling; shared per lens',
            'spirula_parity': 'reduced: Spirula default uses 12 affine values per grid cell and Rec.601 guidance',
            'grid_smoothness': float(grid_smoothness),
        },
        'views': views,
        'metrics': {
            **baseline.get('metrics', {}),
            'validation_before': validation_before,
            'validation_after': validation_after,
            'validation_homography': validation_h,
            'validation_bilateral': validation_grid,
            'test_before': _metrics(obs, test, x),
            'test_after': _metrics(obs, test, best),
            'points': int(len(np.unique(obs['point_id']))),
            'observations': int(n),
            'train_points': int(len(np.unique(obs['point_id'][train]))),
            'validation_points': int(len(np.unique(obs['point_id'][validation]))),
            'test_points': int(len(np.unique(obs['point_id'][test]))),
            'supported_views': int(np.count_nonzero(support >= 16)),
        },
        'timing': {
            **baseline.get('timing', {}),
            'advanced_fit_seconds': round(time.perf_counter() - started, 3),
        },
    })
    report['metrics'].update({
        'validation_off': _metrics(obs, validation, to_linear(obs['rgb'])),
        'test_off': _metrics(obs, test, to_linear(obs['rgb'])),
        'validation_loglinear': validation_before,
        'test_loglinear': _metrics(obs, test, x),
        'validation_grid_only': validation_grid_only,
        'test_grid_only': _metrics(obs, test, grid_only_after),
        'test_homography': _metrics(obs, test, after_h),
        'test_bilateral': _metrics(obs, test, grid_after),
    })
    return report
