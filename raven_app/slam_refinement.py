"""Automatic revisit refinement, isolated from raw SLAM and camera calibration.

Candidates remain separate until independent geometry checks accept them.
No correction is inferred when revisits lack sufficient geometric support.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, vstack
from scipy.spatial.transform import Rotation, Slerp

from . import slam_refinement_geometry as geo
from .slam_refinement_constraints import plane_constraints, validate_records

VERSION = 'revisit-plane-bundle-v1'
PARAMETERS = {'submap_seconds': 10., 'voxel_m': .08, 'max_submap_points': 40000,
              'max_loops_per_node': 4, 'plane_radius_m': .35, 'plane_sigma_m': .008}


def source_signature(paths):
    signature = {'version': VERSION, 'parameters': PARAMETERS}
    for name, path in paths.items():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        signature[name] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}
    return signature


def graph_jacobian(parameters, nodes, edges):
    """Colored finite differences using public NumPy/SciPy APIs only."""
    adjacency = [set() for _ in nodes]
    incident = [[] for _ in nodes]
    for edge_id, (i, j, *_) in enumerate(edges):
        adjacency[i].add(j)
        adjacency[j].add(i)
        incident[i].append(edge_id)
        incident[j].append(edge_id)
    colors = {}
    for index in sorted(range(1, len(nodes)), key=lambda i: -len(adjacency[i])):
        used = {colors[j] for j in adjacency[index] if j in colors}
        color = 0
        while color in used:
            color += 1
        colors[index] = color
    base = geo.graph_residual(parameters, nodes, edges)
    rows, cols, values = [], [], []
    for color in sorted(set(colors.values())):
        group = [index for index in colors if colors[index] == color]
        for axis in range(6):
            changed = parameters.copy()
            steps = {}
            for index in group:
                column = (index-1)*6+axis
                step = np.sqrt(np.finfo(float).eps)*max(1., abs(parameters[column]))
                changed[column] += step
                steps[index] = changed[column]-parameters[column]
            difference = geo.graph_residual(changed, nodes, edges)-base
            for index in group:
                rr = np.concatenate([np.arange(e*6, (e+1)*6) for e in incident[index]])
                rows.extend(rr)
                cols.extend([(index-1)*6+axis]*len(rr))
                values.extend(difference[rr]/steps[index])
    return coo_matrix((values, (rows, cols)), shape=(len(base), len(parameters))).tocsr()


def solve(nodes, loops, point_jac, point_errors):
    edges = [(i, i+1, *geo.relative_pose(nodes[i], nodes[i+1]), 'odometry')
             for i in range(len(nodes)-1)] + loops
    anchors = []
    for i, j, relative_r, relative_t, _ in loops:
        early, late = nodes[i], nodes[j]
        delta = early.rotation*relative_r.inv()*late.rotation.inv()
        position = early.position-early.rotation.apply(relative_r.inv().apply(relative_t))
        anchors.append((late.timestamp, delta, position-delta.apply(late.position)))
    reference_time = min(nodes[i].timestamp for i, *_ in loops)
    times, rotations, translations = [nodes[0].timestamp], [Rotation.identity()], [np.zeros(3)]
    if reference_time > times[0]:
        times.append(reference_time)
        rotations.append(Rotation.identity())
        translations.append(np.zeros(3))
    for timestamp in sorted(set(a[0] for a in anchors)):
        if timestamp <= times[-1]:
            continue
        group = [a for a in anchors if a[0] == timestamp]
        times.append(timestamp)
        rotations.append(Rotation.concatenate([a[1] for a in group]).mean())
        translations.append(np.mean([a[2] for a in group], axis=0))
    query = np.clip([n.timestamp for n in nodes], times[0], times[-1])
    dr = Slerp(times, Rotation.concatenate(rotations))(query)
    dt = np.column_stack([np.interp(query, times, np.asarray(translations)[:, k]) for k in range(3)])
    initial = []
    for node, rotation, translation in zip(nodes[1:], dr[1:], dt[1:]):
        initial.extend((node.rotation.inv()*rotation*node.rotation).as_rotvec())
        initial.extend(node.rotation.inv().apply(rotation.apply(node.position)+translation-node.position))
    initial = np.asarray(initial)

    def residual(x):
        return np.concatenate([geo.graph_residual(x, nodes, edges), point_errors+point_jac@x])

    def jacobian(x):
        return vstack([graph_jacobian(x, nodes, edges), point_jac]).tocsr()

    result = least_squares(residual, initial, jac=jacobian, tr_solver='lsmr', x_scale='jac',
                           tr_options={'atol': 1e-10, 'btol': 1e-10}, loss='soft_l1',
                           max_nfev=500, xtol=1e-7, ftol=1e-7, gtol=1e-7)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Refinement did not converge: {result.message}')
    changes = np.vstack([np.zeros(6), result.x.reshape(-1, 6)])
    if np.max(np.linalg.norm(changes[:, 3:], axis=1)) > 3. or np.max(np.linalg.norm(changes[:, :3], axis=1)) > np.deg2rad(10):
        raise ValueError('Optimized correction exceeds 3 m or 10 degrees')
    report = {'success': True, 'nfev': result.nfev, 'cost': float(result.cost),
              'optimality': float(result.optimality), 'message': result.message,
              'initial_cost': float(np.sum(residual(initial)**2)/2)}
    return result.x, edges, report


def refine(slam_dir: Path, output: Path, log=print):
    """Return a report; accepted outputs include both cloud and trajectory paths."""
    slam_dir, output = Path(slam_dir).resolve(), Path(output).resolve()
    if output == slam_dir or slam_dir in output.parents:
        raise ValueError('Refinement output must be separate from raw SLAM')
    trajectories = list((slam_dir/'result').glob('*.txt'))
    if len(trajectories) != 1:
        raise ValueError('Exactly one raw SLAM trajectory is required')
    paths = {'pcd': slam_dir/'pcd/all_raw_points.pcd', 'trajectory': trajectories[0],
             'scan_ranges': slam_dir/'scan_ranges.csv'}
    signature = source_signature(paths)
    report_path = output/'refinement_report.json'
    if report_path.exists():
        cached = json.loads(report_path.read_text(encoding='utf-8'))
        if cached.get('source_signature') == signature and cached.get('status') in {
                'no_supported_revisits', 'accepted', 'rejected_geometry'}:
            if cached['status'] != 'accepted':
                log(f"Reusing SLAM revisit audit: {cached['status']}.")
                return cached
            generated = {name: Path(cached[f'output_{name}']) for name in ('pcd', 'trajectory', 'scan_ranges')}
            if cached.get('output_signature') == source_signature(generated):
                log('Reusing validated global SLAM refinement.')
                return cached
        raise FileExistsError('Existing refinement differs or is incomplete; choose a fresh output directory')
    if output.exists():
        raise FileExistsError('Refinement output already exists')
    pcd = geo.read_pcd(paths['pcd'])
    trajectory = geo.read_trajectory(paths['trajectory'])
    scans = geo.read_scan_ranges(paths['scan_ranges'], pcd.count)
    if scans[0].timestamp < trajectory.times[0] or scans[-1].timestamp > trajectory.times[-1]:
        raise ValueError('Scan provenance timestamps extend outside the SLAM trajectory domain')
    nodes = geo.build_nodes(pcd, scans, trajectory, window_seconds=10., voxel_m=.08, maximum=40000)
    selected, skipped, selection = geo.select_loop_pairs(nodes, max_neighbors=4)
    loops, evidence, rejected = [], [], []
    log(f'Global SLAM refinement: {len(nodes)} submaps, {len(selected)} revisit candidates.')
    for number, (i, j, _, _) in enumerate(selected):
        try:
            rotation, translation, row = geo.propose_loop(nodes[i], nodes[j], 42+31*i+j)
            loops.append((i, j, rotation, translation, 'loop'))
            evidence.append(row)
        except ValueError as exc:
            rejected.append({'source_node': i, 'target_node': j, 'reason': str(exc)})
        if number % 10 == 0:
            log(f'Revisit validation {number+1}/{len(selected)}; accepted {len(loops)}.')
    report = {'version': VERSION, 'source_signature': signature, 'accepted_loops': len(loops),
              'selection': selection, 'rejected_candidates': rejected, 'skipped_by_degree_cap': skipped}
    if not loops:
        output.mkdir(parents=True)
        report['status'] = 'no_supported_revisits'
    else:
        log('Optimizing trajectory with scan-time static-plane constraints.')
        jac, errors, planes = plane_constraints(pcd, scans, nodes)
        parameters, edges, graph = solve(nodes, loops, jac, errors)
        before_loop = geo.graph_residual(np.zeros_like(parameters), nodes, loops)
        after_loop = geo.graph_residual(parameters, nodes, loops)
        closure_improved = np.linalg.norm(after_loop) < .90*np.linalg.norm(before_loop)
        try:
            geo.write_candidate(output, pcd, scans, trajectory, nodes, parameters, paths['scan_ranges'],
                                graph, edges, evidence, selection, PARAMETERS)
            log('Validating corrected geometry on independent reference records.')
            quality = validate_records(pcd, geo.read_pcd(output/'pcd'/pcd.path.name))
        except Exception:
            # The directory was proven absent above and contains only this
            # incomplete candidate; never leave a cache-shaped partial result.
            if output.is_dir():
                shutil.rmtree(output)
            raise
        report.update(status='accepted' if quality['improved'] and closure_improved else 'rejected_geometry',
                      validation=quality, plane_constraints=planes, graph=graph,
                      closure_improved=bool(closure_improved),
                      output_pcd=str(output/'pcd'/pcd.path.name),
                      output_trajectory=str(output/'result'/trajectory.path.name),
                      output_scan_ranges=str(output/'scan_ranges.csv'))
        report['output_signature'] = source_signature({name: Path(report[f'output_{name}'])
                                                       for name in ('pcd', 'trajectory', 'scan_ranges')})
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
