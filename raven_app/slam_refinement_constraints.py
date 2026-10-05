"""Static-plane bundle constraints and independent same-record validation.

Plane parameters are eliminated locally; trajectory corrections are evaluated
at each record's scan time. No dataset coordinates or manual revisits are used.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .geometry_quality import plane_fit, select_surface_patches, summarize_paired_surfaces


def sample_records(pcd, seed, maximum=2_000_000):
    ids = np.sort(np.random.default_rng(seed).choice(pcd.count, min(pcd.count, maximum), False))
    np.random.default_rng(17).shuffle(ids)
    xyz = np.column_stack([pcd.points[f][ids] for f in pcd.xyz_fields]).astype(float)
    if not np.isfinite(xyz).all():
        raise ValueError('Refinement sample contains non-finite coordinates')
    return ids, xyz


def plane_constraints(pcd, scans, nodes, radius=.35, sigma=.008):
    ids, xyz = sample_records(pcd, 42)
    ids, xyz = ids[:len(ids)//2], xyz[:len(xyz)//2]
    scan_times = np.array([s.timestamp for s in scans])
    starts = np.array([s.start for s in scans])
    times = scan_times[np.searchsorted(starts, ids, side='right') - 1]
    knots = np.array([n.timestamp for n in nodes])
    if len(knots) < 2:
        raise ValueError('At least two trajectory knots are required')
    hi = np.clip(np.searchsorted(knots, times, side='right'), 1, len(knots)-1)
    lo = hi-1
    alpha = np.clip((times-knots[lo])/(knots[hi]-knots[lo]), 0, 1)
    centers = select_surface_patches(xyz, radius=radius)
    tree = cKDTree(xyz)
    rows, columns, values, residuals = [], [], [], []
    row_offset = 0
    for center in centers:
        indices = np.asarray(tree.query_ball_point(center, radius))
        points = xyz[indices]
        origin, normal, _ = plane_fit(points)
        axis = np.eye(3)[np.argmin(abs(normal))]
        u = np.cross(normal, axis)
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        nuisance = np.column_stack([np.ones(len(points)), (points-origin)@u, (points-origin)@v])
        basis, _ = np.linalg.qr(nuisance)
        involved = np.unique(np.concatenate([lo[indices], hi[indices]]))
        involved = involved[involved > 0]
        if not len(involved):
            continue
        cols = np.concatenate([np.arange((node-1)*6, node*6) for node in involved])
        positions = {int(node): i*6 for i, node in enumerate(involved)}
        jac = np.zeros((len(points), len(cols)))
        for node_ids, weights in [(lo[indices], 1-alpha[indices]), (hi[indices], alpha[indices])]:
            for index in np.unique(node_ids):
                if index == 0:
                    continue
                mask = node_ids == index
                node = nodes[index]
                rotation = node.rotation.as_matrix()
                angular = np.cross(points[mask]-node.position, normal) @ rotation
                linear = np.broadcast_to(normal @ rotation, (int(mask.sum()), 3))
                block = np.column_stack([angular, linear]) * weights[mask, None]
                offset = positions[int(index)]
                jac[mask, offset:offset+6] += block
        errors = (points-origin)@normal
        jac -= basis@(basis.T@jac)
        errors -= basis@(basis.T@errors)
        scale = sigma*np.sqrt(len(points))
        jac /= scale
        errors /= scale
        rr, cc = np.nonzero(abs(jac) > 1e-12)
        rows.extend(row_offset+rr)
        columns.extend(cols[cc])
        values.extend(jac[rr, cc])
        residuals.extend(errors)
        row_offset += len(points)
    matrix = coo_matrix((values, (rows, columns)), shape=(row_offset, (len(nodes)-1)*6)).tocsr()
    return matrix, np.asarray(residuals), {'patches': len(centers), 'training_rows': row_offset,
                                         'sigma_m': sigma, 'radius_m': radius}


def validate_records(reference, candidate, seed=20261003, patch_seed=901, radius=.35):
    """Freeze reference membership before reading candidate geometry."""
    if reference.count != candidate.count or reference.dtype != candidate.dtype:
        raise ValueError('Validation requires identical record count and layout')
    ids, before = sample_records(reference, seed)
    after = np.column_stack([candidate.points[f][ids] for f in candidate.xyz_fields]).astype(float)
    if not np.isfinite(after).all():
        raise ValueError('Corrected sample contains non-finite coordinates')
    split = len(before)//2
    centers = select_surface_patches(before[:split], radius=radius, seed=patch_seed)
    train_tree, test_tree = cKDTree(before[:split]), cKDTree(before[split:])
    results = [[], []]
    for index, center in enumerate(centers):
        train = train_tree.query_ball_point(center, radius)
        test = test_tree.query_ball_point(center, radius)
        for xyz, output in zip((before, after), results):
            row = {'patch': index, 'train_points': len(train), 'test_points': len(test)}
            if min(len(train), len(test)) >= 40:
                origin, normal, _ = plane_fit(xyz[:split][train])
                errors = abs((xyz[split:][test]-origin)@normal)
                row.update(rmse_mm=float(np.sqrt(np.mean(errors**2))*1000), normal=normal.tolist())
            output.append(row)
    report = summarize_paired_surfaces(*results)
    report.update(sample_seed=seed, patch_seed=patch_seed,
                  protocol='Fixed reference records; independent plane fitting and held-out scoring')
    return report
