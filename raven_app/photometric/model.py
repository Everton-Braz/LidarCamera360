"""Log-linear image gains with optional shared radial lens falloff.

Only repeated observations of the same scene point constrain the fit. A
point-level train/validation/test split prevents radiance leakage. The model
is anchored separately in each connected image component.
"""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

VERSION = 1
MAX_LOG_CORRECTION = float(np.log(2.0))


def to_linear(rgb):
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def to_srgb(rgb):
    x = np.clip(rgb, 0, 1)
    return 255 * np.where(x <= .0031308, 12.92 * x,
                          1.055 * x ** (1 / 2.4) - .055)


def apply_rgb(rgb, u, v, camera_params, coeff=None):
    """Inverse correction before consensus; r uses the colorizer's useful circle."""
    if coeff is None:
        return rgb
    r2 = ((np.asarray(u) - camera_params[2]) ** 2 +
          (np.asarray(v) - camera_params[3]) ** 2) / 1620.0 ** 2
    k = coeff['vignette']
    log_factor = (np.asarray(coeff['log_gain']) +
                  (r2 * (k[0] + r2 * (k[1] + r2 * k[2])))[..., None])
    corrected = to_linear(rgb) * np.exp(-np.clip(log_factor,
                                                -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION))
    if 'ppisp_h' in coeff or 'bilateral_grid' in coeff:
        from .advanced import apply_linear
        corrected = apply_linear(np.clip(corrected, 0, 1), u, v, coeff)
    return np.clip(np.rint(to_srgb(corrected)), 0, 255).astype(np.uint8)


def load_model(path):
    model = json.loads(Path(path).read_text(encoding='utf-8'))
    if model.get('schema') not in (VERSION, 2) or model.get('color_space') != 'srgb':
        raise ValueError('Unsupported photometric model schema/color space')
    if not isinstance(model.get('views'), dict):
        raise ValueError('Missing photometric views')
    for name, coeff in model['views'].items():
        if not isinstance(name, str) or '\\' in name or '..' in name.split('/'):
            raise ValueError('Invalid photometric image name')
        for key in ('log_gain', 'vignette'):
            values = np.asarray(coeff.get(key), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError('Invalid photometric coefficients')
            if np.max(np.abs(values)) > 4:
                raise ValueError('Photometric coefficients exceed safe limits')
        if 'ppisp_h' in coeff:
            h = np.asarray(coeff['ppisp_h'], dtype=float)
            if (h.shape != (9,) or not np.isfinite(h).all() or
                    np.max(np.abs(h)) > 4 or abs(h[8] - 1) > 1e-5 or
                    abs(np.linalg.det(h.reshape(3, 3))) < 1e-5):
                raise ValueError('Invalid PPISP homography')
        if 'bilateral_grid' in coeff:
            grid = np.asarray(coeff['bilateral_grid'], dtype=float)
            if (grid.shape != (4, 3, 3, 3) or not np.isfinite(grid).all() or
                    np.max(np.abs(grid)) > .350001):
                raise ValueError('Invalid bilateral grid')
        if model.get('schema') == VERSION and ('ppisp_h' in coeff or 'bilateral_grid' in coeff):
            raise ValueError('Advanced coefficients require photometric schema 2')
    return model


def _pairs(point, select):
    indices = np.flatnonzero(select)
    indices = indices[np.argsort(point[indices], kind='stable')]
    # A cycle supplies one edge per observation, avoids privileging a noisy
    # reference observation, and keeps the least-squares matrix sparse.
    boundaries = np.r_[0, np.flatnonzero(np.diff(point[indices])) + 1, len(indices)]
    following = np.roll(indices, -1)
    if len(indices):
        following[boundaries[1:] - 1] = indices[boundaries[:-1]]
    keep = indices != following
    return indices[keep], following[keep]


def _design(frame, lens, radius, a, b, nframes, nlenses, vignette):
    n = len(a)
    rows = [np.arange(n), np.arange(n)]
    cols = [frame[a], frame[b]]
    values = [np.ones(n), -np.ones(n)]
    if vignette:
        for power in range(1, 4):
            for ix, sign in ((a, 1), (b, -1)):
                rows.append(np.arange(n))
                cols.append(nframes + 3 * lens[ix] + power - 1)
                values.append(sign * radius[ix] ** (2 * power))
    return sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, nframes + (3 * nlenses if vignette else 0))).tocsr()


def _solve(obs, train, vignette, nframes, nlenses):
    a, b = _pairs(obs['point_id'], train)
    frame, lens, radius = obs['frame_id'], obs['lens_id'], obs['radius']
    design = _design(frame, lens, radius, a, b, nframes, nlenses, vignette)
    target = np.log(np.maximum(to_linear(obs['rgb'][a]), 1e-5)) - np.log(
        np.maximum(to_linear(obs['rgb'][b]), 1e-5))
    weights = np.minimum(obs['weight'][a], obs['weight'][b])
    support = np.bincount(frame[train], minlength=nframes)
    adjacency = sparse.coo_matrix((np.ones(len(a)), (frame[a], frame[b])),
                                  shape=(nframes, nframes)).tocsr()
    _, components = connected_components(adjacency, directed=False)
    # Lens falloff is achromatic, shared by all three channels. Solve the
    # luminance-log average first, then fit residual per-channel image gains.
    k = np.zeros((nlenses, 3))
    if vignette:
        d = design
        penalty = np.r_[np.full(nframes, .03), np.full(3 * nlenses, 2.0)]
        reg = sparse.diags(penalty)
        y = target.mean(axis=1)
        rw = weights.copy()
        solution = np.zeros(d.shape[1])
        for _ in range(4):
            sw = np.sqrt(rw)
            matrix = sparse.vstack((d.multiply(sw[:, None]), reg)).tocsr()
            solution = lsqr(matrix, np.r_[y * sw, np.zeros(d.shape[1])],
                            atol=1e-6, btol=1e-6, iter_lim=150)[0]
            residual = d @ solution - y
            rw = weights * np.minimum(1, .08 / np.maximum(np.abs(residual), 1e-8))
        k = solution[nframes:].reshape(nlenses, 3)
        k = np.clip(k, -.5, .5)
    falloff = np.sum(k[lens] * radius[:, None] ** np.array([2, 4, 6]), axis=1)
    target -= (falloff[a] - falloff[b])[:, None]
    d = design[:, :nframes]
    gain = np.zeros((nframes, 3))
    reg = sparse.eye(nframes, format='csr') * .03
    for channel in range(3):
        y = target[:, channel]
        rw = weights.copy()
        for _ in range(4):
            sw = np.sqrt(rw)
            matrix = sparse.vstack((d.multiply(sw[:, None]), reg)).tocsr()
            gain[:, channel] = lsqr(matrix, np.r_[y * sw, np.zeros(nframes)],
                                    atol=1e-6, btol=1e-6, iter_lim=150)[0]
            residual = d @ gain[:, channel] - y
            rw = weights * np.minimum(1, .08 / np.maximum(np.abs(residual), 1e-8))
    for component in np.unique(components):
        ix = (components == component) & (support >= 12)
        if ix.any():
            gain[ix] -= np.median(gain[ix], axis=0)
    gain = np.clip(gain, -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION)
    gain[support < 12] = 0
    return gain, k, support, components


def _correct(obs, gain, k, support):
    supported = support[obs['frame_id']] >= 12
    factor = gain[obs['frame_id']] + (np.sum(
        k[obs['lens_id']] * obs['radius'][:, None] ** np.array([2, 4, 6]), axis=1) * supported)[:, None]
    return np.clip(to_linear(obs['rgb']) * np.exp(-np.clip(
        factor, -MAX_LOG_CORRECTION, MAX_LOG_CORRECTION)), 0, 1)


def _metrics(obs, select, linear):
    import cv2
    a, b = _pairs(obs['point_id'], select)
    if not len(a):
        return {'pairs': 0, 'linear_rmse': None, 'delta_e76_mean': None}
    # OpenCV float RGB -> Lab uses sRGB and the D65 reference white.
    srgb = (to_srgb(linear) / 255).astype(np.float32)
    lab = cv2.cvtColor(srgb.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    de = np.linalg.norm(lab[a] - lab[b], axis=1)
    return {'pairs': len(a), 'linear_rmse': float(np.sqrt(np.mean((linear[a] - linear[b]) ** 2))),
            'delta_e76_mean': float(de.mean()), 'delta_e76_p90': float(np.percentile(de, 90))}


def fit_observations(observations, names, lens_names, seed=20260929):
    """Fit on 60% of points; select model on 20%; report untouched 20%."""
    obs = {key: np.asarray(value) for key, value in observations.items()}
    required = ('point_id', 'frame_id', 'lens_id', 'radius', 'rgb', 'weight')
    n = len(obs['point_id'])
    if n == 0 or any(len(obs[key]) != n for key in required):
        raise ValueError('Empty/inconsistent photometric observations')
    if (obs['rgb'].shape != (n, 3) or
        any(not np.isfinite(obs[k]).all() for k in required) or
        np.any(obs['weight'] <= 0) or np.any(obs['radius'] < 0) or
        np.any(obs['radius'] > 1) or np.any(obs['rgb'] < 0) or np.any(obs['rgb'] > 255)):
        raise ValueError('Invalid photometric observations')
    for key, bound in (('frame_id', len(names)), ('lens_id', len(lens_names))):
        if np.any(obs[key] < 0) or np.any(obs[key] >= bound) or np.any(obs[key] != obs[key].astype(int)):
            raise ValueError('Invalid observation indices')
        obs[key] = obs[key].astype(np.int64)
    ids, inverse, counts = np.unique(obs['point_id'], return_inverse=True, return_counts=True)
    if len(ids) < 30 or np.any(counts < 3):
        raise ValueError('Need at least 30 points with three observations each')
    order = np.random.default_rng(seed).permutation(len(ids))
    groups = np.zeros(len(ids), dtype=np.uint8)
    groups[order[int(.6 * len(ids)):int(.8 * len(ids))]] = 1
    groups[order[int(.8 * len(ids)):]] = 2
    split = groups[inverse]
    train, validation, test = split == 0, split == 1, split == 2
    baseline = _metrics(obs, validation, to_linear(obs['rgb']))
    gain, k, support, components = _solve(obs, train, False, len(names), len(lens_names))
    score = _metrics(obs, validation, _correct(obs, gain, k, support))
    selected = 'gains'
    # Model selection never reads the final test partition.
    if score['linear_rmse'] >= baseline['linear_rmse'] * .99 or score['delta_e76_mean'] >= baseline['delta_e76_mean']:
        gain[:] = 0
        selected = 'identity'
        score = baseline
    radial_coverage = all(np.ptp(obs['radius'][train & (obs['lens_id'] == i)]) > .45
                          if np.count_nonzero(train & (obs['lens_id'] == i)) >= 100 else False
                          for i in range(len(lens_names)))
    if radial_coverage:
        vg, vk, vs, vc = _solve(obs, train, True, len(names), len(lens_names))
        candidate = _metrics(obs, validation, _correct(obs, vg, vk, vs))
        if (candidate['linear_rmse'] < score['linear_rmse'] * .98 and
                candidate['delta_e76_mean'] < score['delta_e76_mean'] * .99):
            gain, k, support, components = vg, vk, vs, vc
            score, selected = candidate, 'gains+vignette'
    views = {}
    for i, name in enumerate(names):
        lens = obs['lens_id'][np.flatnonzero(obs['frame_id'] == i)[0]] if np.any(obs['frame_id'] == i) else 0
        views[name] = {'log_gain': gain[i].tolist(),
                       'vignette': (k[lens] if support[i] >= 12 else np.zeros(3)).tolist(),
                       'support': int(support[i]), 'component': int(components[i])}
    return {'schema': VERSION, 'color_space': 'srgb', 'radius_px': 1620.0,
            'selected': selected, 'views': views, 'lens_names': list(lens_names),
            'metrics': {'validation_before': baseline, 'validation_after': score,
                        'test_before': _metrics(obs, test, to_linear(obs['rgb'])),
                        'test_after': _metrics(obs, test, _correct(obs, gain, k, support)),
                        'points': len(ids), 'observations': n,
                        'train_points': int(np.count_nonzero(groups == 0)),
                        'validation_points': int(np.count_nonzero(groups == 1)),
                        'test_points': int(np.count_nonzero(groups == 2)),
                        'supported_views': int(np.count_nonzero(support >= 12))}}
