"""Static-plane bundle constraints and independent same-record validation.

Plane parameters are eliminated locally; trajectory corrections are evaluated
at each record's scan time. No dataset coordinates or manual revisits are used.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from . import slam_refinement_geometry as geo
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
    # the correction does not cross a pose jump: a point follows the submaps on its own side
    segments, firsts = geo.segment_starts(nodes)
    owner = np.searchsorted(firsts, times, side='right')-1
    node_owner = np.zeros(len(nodes), dtype=int)
    for number, members in enumerate(segments):
        node_owner[members] = number
    alpha[node_owner[hi] != owner] = 0.
    alpha[node_owner[lo] != owner] = 1.
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


def doubled_surfaces(nodes, parameters=None, seconds_apart=25., reach_m=8., gate_m=.3, stride=2):
    """Share of plane sightings that disagree between submaps recorded far apart in time.

    Where two submaps see the same plane from the same side, the median signed distance between them is how far
    that surface is doubled. Nothing is fitted, so it judges the map as it stands, and it needs no reference.
    The per-point surface validation cannot see this: moving whole submaps leaves each surface as flat as it was.
    """
    poses = [geo.transform_from_correction(node, np.zeros(6) if parameters is None or i == 0 else
                                           parameters[(i-1)*6:i*6]) for i, node in enumerate(nodes)]
    clouds = [rotation.apply(node.points[::stride])+position for node, (rotation, position) in zip(nodes, poses)]
    trees = [cKDTree(cloud) for cloud in clouds]
    centers = np.array([position for _, position in poses])
    fitted = []
    for tree, cloud in zip(trees, clouds):
        _, neighbours = tree.query(cloud, k=min(12, len(cloud)), workers=-1)
        patch = cloud[neighbours]
        patch = patch-patch.mean(axis=1, keepdims=True)
        values, vectors = np.linalg.eigh(np.einsum('nki,nkj->nij', patch, patch)/patch.shape[1])
        flatness = np.sqrt(np.maximum(values[:, 0], 0)/np.maximum(values[:, 1], 1e-12))
        fitted.append((vectors[:, :, 0], flatness))
    offsets = []
    for j, (normals, flatness) in enumerate(fitted):
        for i in range(len(nodes)):
            if i == j or abs(nodes[i].timestamp-nodes[j].timestamp) < seconds_apart or \
                    np.linalg.norm(centers[i]-centers[j]) > reach_m:
                continue
            distances, index = trees[j].query(clouds[i], distance_upper_bound=gate_m, workers=-1)
            hit = np.isfinite(distances)
            hit[hit] = flatness[index[hit]] < .12
            if hit.sum() < 300:
                continue
            normal, source, target = normals[index[hit]], clouds[i][hit], clouds[j][index[hit]]
            # the two faces of a wall are not a doubled wall
            same_side = np.einsum('ij,ij->i', normal, centers[i]-source)*np.einsum('ij,ij->i', normal, centers[j]-target) > 0
            if same_side.sum() < 300:
                continue
            normal, source, target = normal[same_side], source[same_side], target[same_side]
            axis = np.abs(normal).argmax(axis=1)
            signed = np.einsum('ij,ij->i', normal, source-target)*np.sign(normal[np.arange(len(normal)), axis])
            offsets.extend(abs(np.median(signed[axis == k])) for k in range(3) if (axis == k).sum() >= 200)
    offsets = np.asarray(offsets)
    if not len(offsets):
        return {'plane_pairs': 0, 'share_over_3cm': None, 'share_over_6cm': None}
    return {'plane_pairs': int(len(offsets)), 'share_over_3cm': float(np.mean(offsets > .03)),
            'share_over_6cm': float(np.mean(offsets > .06)), 'p90_mm': float(np.quantile(offsets, .9)*1000)}


def validate_records(reference, candidate, seed=20261003, patch_seed=901, radius=.35):
    """Freeze reference membership before reading candidate geometry."""
    if reference.count != candidate.count or reference.dtype != candidate.dtype:
        raise ValueError('Validation requires identical record count and layout')
    ids, before = sample_records(reference, seed)
    after = np.column_stack([candidate.points[f][ids] for f in candidate.xyz_fields]).astype(float)
    finite_mask = np.isfinite(after).all(axis=1)
    if finite_mask.sum() < len(after) * 0.5:
        raise ValueError('Corrected sample contains too many non-finite coordinates')
    if not finite_mask.all():
        ids = ids[finite_mask]
        before = before[finite_mask]
        after = after[finite_mask]
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
