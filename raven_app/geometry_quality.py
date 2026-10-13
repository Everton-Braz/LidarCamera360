"""Reproducible local surface precision metrics (not surveyed accuracy)."""
import numpy as np
from scipy.spatial import cKDTree


def rigid_registration(source, target):
    """Map paired positions to the reference frame without changing scale."""
    source, target = np.asarray(source), np.asarray(target)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError('At least three paired XYZ positions are required')
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    sign = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ sign @ u.T
    return rotation, b - rotation @ a


def plane_fit(points):
    center = np.mean(points, axis=0)
    delta = points - center
    values, vectors = np.linalg.eigh(delta.T @ delta / len(points))
    return center, vectors[:, 0], values


def select_surface_patches(points, radius=.35, limit=200, seed=42):
    """Select spatially spread flat reference patches using training points only."""
    points = np.asarray(points)
    tree = cKDTree(points)
    _, indices = np.unique(np.floor(points / (2 * radius)).astype(np.int64),
                           axis=0, return_index=True)
    np.random.default_rng(seed).shuffle(indices)
    centers = []
    for index in indices:
        neighbors = tree.query_ball_point(points[index], radius)
        if len(neighbors) < 80:
            continue
        center, _, eigenvalues = plane_fit(points[neighbors])
        if eigenvalues[1] < .003 or eigenvalues[0] > .015 * eigenvalues[1]:
            continue
        if any(np.linalg.norm(center - previous) < 2 * radius for previous in centers):
            continue
        centers.append(center)
        if len(centers) == limit:
            break
    if not centers:
        raise ValueError('No sufficiently supported planar reference patches')
    return np.asarray(centers)


def score_surface_patches(training, heldout, centers, radius=.35):
    """Fit each local plane on training points; score every held-out neighbor."""
    train_tree, test_tree = cKDTree(training), cKDTree(heldout)
    rows = []
    for index, center in enumerate(centers):
        train_ids = train_tree.query_ball_point(center, radius)
        test_ids = test_tree.query_ball_point(center, radius)
        row = {'patch': index, 'train_points': len(train_ids), 'test_points': len(test_ids)}
        if min(len(train_ids), len(test_ids)) >= 40:
            origin, normal, _ = plane_fit(training[train_ids])
            errors = np.abs((heldout[test_ids] - origin) @ normal)
            row.update(rmse_mm=float(np.sqrt(np.mean(errors ** 2)) * 1000),
                       p90_mm=float(np.percentile(errors, 90) * 1000),
                       normal=normal.tolist())
        rows.append(row)
    return rows


def summarize_paired_surfaces(reference, candidate):
    """Equal patch weights prevent dense surfaces from dominating comparison."""
    pairs = [(a, b) for a, b in zip(reference, candidate)
             if a['patch'] == b['patch'] and 'rmse_mm' in a and 'rmse_mm' in b]
    if not pairs:
        raise ValueError('No common supported surface patches')
    base = np.array([a['rmse_mm'] for a, _ in pairs])
    after = np.array([b['rmse_mm'] for _, b in pairs])
    coverage = len(pairs) / max(1, sum('rmse_mm' in a for a in reference))
    normal_dots = np.array([abs(np.dot(a['normal'], b['normal'])) for a, b in pairs])
    normals_consistent_fraction = float(np.mean(normal_dots >= np.cos(np.deg2rad(15))))
    rotations_ok = bool(normals_consistent_fraction >= 0.90)
    reference_rmse, candidate_rmse = np.sqrt(np.mean(base ** 2)), np.sqrt(np.mean(after ** 2))
    normal_z = np.abs([a['normal'][2] for a, _ in pairs])
    by_orientation = {}
    for name, mask in (('horizontal', normal_z > .9), ('vertical', normal_z < .2),
                       ('other', (normal_z >= .2) & (normal_z <= .9))):
        if np.any(mask):
            by_orientation[name] = {'patches': int(mask.sum()),
                'reference_rmse_mm': float(np.sqrt(np.mean(base[mask] ** 2))),
                'candidate_rmse_mm': float(np.sqrt(np.mean(after[mask] ** 2)))}
    p90_improved = float(np.percentile(after, 90)) <= float(np.percentile(base, 90)) + 1e-4
    median_improved = float(np.median(after)) <= float(np.median(base)) + 1e-4
    patch_win = float(np.mean(after < base - 1e-6))
    improved = bool(coverage >= .90 and rotations_ok and
                    (p90_improved or median_improved or (candidate_rmse < reference_rmse and patch_win >= 0.45)))
    return {'common_patches': len(pairs), 'reference_patch_coverage': coverage,
            'reference_rmse_mm': float(reference_rmse),
            'candidate_rmse_mm': float(candidate_rmse),
            'relative_rmse_reduction': float(1 - candidate_rmse / max(reference_rmse, 1e-12)),
            'reference_patch_p90_mm': float(np.percentile(base, 90)),
            'candidate_patch_p90_mm': float(np.percentile(after, 90)),
            'patch_win_fraction': patch_win,
            'normals_consistent': bool(rotations_ok),
            'normals_consistent_fraction': normals_consistent_fraction,
            'by_orientation': by_orientation,
            'improved': improved,
            'interpretation': 'Local surface precision proxy, not ground-truth geometry accuracy.'}
