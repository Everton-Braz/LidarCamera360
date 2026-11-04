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

from . import slam_refinement_geometry as geo
from .slam_refinement_constraints import doubled_surfaces, plane_constraints, validate_records

VERSION = 'revisit-plane-bundle-v3'
PARAMETERS = {'submap_seconds': 10., 'voxel_m': .08, 'max_submap_points': 40000,
              'max_loops_per_node': 8, 'plane_radius_m': .35, 'plane_sigma_m': .015,
              'jump_threshold_m': .055, 'edge_sigmas': geo.EDGE_SIGMAS}


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


def correction_limits(nodes):
    """Largest believable correction, from the distance walked: drift, not a misplaced map."""
    path = float(sum(np.linalg.norm(b.position-a.position) for a, b in zip(nodes, nodes[1:])))
    return max(1., .01*path), max(5., .02*path)


def loop_limits(nodes, i, j):
    """Largest believable error of the odometry between two submaps; 2 % of the walk plus a margin."""
    path = float(sum(np.linalg.norm(b.position-a.position) for a, b in zip(nodes[i:j], nodes[i+1:j+1])))
    return .5+.02*path, 3.+.02*path


def solve(nodes, loops, point_jac, point_errors, method='rootslam', config=None):
    max_m, max_deg = correction_limits(nodes)
    if method == 'rootslam':
        try:
            from .rootslam import rootslam_solve, RootSLAMConfig
            cfg = config or RootSLAMConfig(precision='float64', max_translation_correction_m=max_m,
                                           max_rotation_correction_deg=max_deg, use_analytical_jacobians=True)
            return rootslam_solve(nodes, loops, point_jac, point_errors, config=cfg)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"RootSLAM solver fallback to LSMR: {exc}")

    edges = geo.odometry_edges(nodes) + loops
    initial = np.zeros((len(nodes)-1)*6)

    def residual(x):
        return np.concatenate([geo.graph_residual(x, nodes, edges), point_errors+point_jac@x])

    def jacobian(x):
        return vstack([graph_jacobian(x, nodes, edges), point_jac]).tocsr()

    result = least_squares(residual, initial, jac=jacobian, tr_solver='lsmr', x_scale='jac',
                           tr_options={'atol': 1e-10, 'btol': 1e-10}, loss='soft_l1',
                           max_nfev=2500, xtol=1e-5, ftol=1e-5, gtol=1e-5)
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f'Refinement did not converge: {result.message}')
    changes = np.vstack([np.zeros(6), result.x.reshape(-1, 6)])
    if np.max(np.linalg.norm(changes[:, 3:], axis=1)) > max_m or \
            np.max(np.linalg.norm(changes[:, :3], axis=1)) > np.deg2rad(max_deg):
        raise ValueError(f'Optimized correction exceeds {max_m:.1f} m or {max_deg:.1f} degrees')
    report = {'success': True, 'nfev': result.nfev, 'cost': float(result.cost),
              'optimality': float(result.optimality), 'message': result.message,
              'initial_cost': float(np.sum(residual(initial)**2)/2)}
    return result.x, edges, report


#: A closure-only acceptance may cost at most this fraction of local surface precision (validation noise).
MAX_SURFACE_LOSS = 0.05


def accept_refinement(quality, closure_improved):
    """Whether a refined trajectory may replace the raw SLAM.

    The independent surface validation decides. A better loop closure alone proves little: the optimizer
    minimizes that same residual, so it nearly always "improves", and moving whole submaps barely changes
    surface normals. It may only carry a candidate whose surfaces are no worse than the raw ones
    (CASA 374 Home3: closure improved, normals 99.5 % consistent, surfaces 9.9 -> 19.2 mm RMSE, accepted).
    """
    before, after = (quality.get('doubled_surfaces') or {}), (quality.get('doubled_surfaces_after') or {})
    if before.get('share_over_6cm') is not None and after.get('share_over_6cm') is not None and \
            after['share_over_6cm'] > before['share_over_6cm']:
        return False  # the whole point of closing loops; surfaces can stay flat while walls double
    if quality.get('improved'):
        return True
    reduction = quality.get('relative_rmse_reduction')
    not_worse = reduction is not None and np.isfinite(reduction) and reduction >= -MAX_SURFACE_LOSS
    return bool(closure_improved and quality.get('normals_consistent', False) and not_worse)


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
            if not accept_refinement(cached.get('validation') or {}, cached.get('closure_improved', False)):
                # accepted under an older, looser rule: judge it again, keep the evidence
                v = cached.get('validation') or {}
                cached.update(status='rejected_geometry', failure_reason=(
                    f"surfaces {v.get('reference_rmse_mm', float('nan')):.1f} -> "
                    f"{v.get('candidate_rmse_mm', float('nan')):.1f} mm RMSE; loop closure alone does not justify it"))
                report_path.write_text(json.dumps(cached, indent=2), encoding='utf-8')
                log(f"Cached SLAM refinement re-judged: {cached['failure_reason']}.")
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

    all_scans = scans
    # Detect stationary initialization and exclude stationary warm-up scans universally
    onset_time = geo.detect_motion_onset(trajectory)
    if onset_time > scans[0].timestamp:
        valid_scans = [s for s in scans if s.timestamp >= onset_time]
        if valid_scans and len(valid_scans) < len(scans):
            log(f'Stationary warm-up detected ({onset_time - scans[0].timestamp:.2f}s). Filtering pre-mission scans.')
            scans = valid_scans

    jumps = geo.detect_pose_jumps(trajectory, slam_dir/'imu.txt', threshold_m=PARAMETERS['jump_threshold_m'])
    jumps = jumps[jumps > scans[0].timestamp]
    if len(jumps):
        log(f'SLAM pose jumps found at {", ".join(f"{t-trajectory.times[0]:.1f}" for t in jumps)} s.')
    nodes = geo.build_nodes(pcd, scans, trajectory, window_seconds=10., voxel_m=.08, maximum=40000, breaks=jumps)
    max_loops = int(PARAMETERS.get('max_loops_per_node', 8))
    selected, skipped, selection = geo.select_loop_pairs(nodes, max_neighbors=max_loops)
    loops, evidence, rejected = [], [], []
    log(f'Global SLAM refinement: {len(nodes)} submaps, {len(selected)} revisit candidates.')
    for number, (i, j, _, _) in enumerate(selected):
        try:
            rotation, translation, row = geo.propose_loop(nodes[i], nodes[j], 42+31*i+j, *loop_limits(nodes, i, j))
            loops.append((i, j, rotation, translation, 'loop'))
            evidence.append(row)
        except ValueError as exc:
            rejected.append({'source_node': i, 'target_node': j, 'reason': str(exc)})
        if number % 10 == 0:
            log(f'Revisit validation {number+1}/{len(selected)}; accepted {len(loops)}.')
    report = {'version': VERSION, 'source_signature': signature, 'accepted_loops': len(loops),
              'pose_jumps_s': [float(t-trajectory.times[0]) for t in jumps],
              'selection': selection, 'rejected_candidates': rejected, 'skipped_by_degree_cap': skipped}
    if not loops:
        output.mkdir(parents=True)
        report['status'] = 'no_supported_revisits'
    else:
        log('Optimizing trajectory with scan-time static-plane constraints.')
        try:
            jac, errors, planes = plane_constraints(pcd, scans, nodes)
            parameters, edges, graph = solve(nodes, loops, jac, errors)
        except Exception as exc:
            log(f'Global SLAM refinement optimization did not converge ({exc}); retaining raw SLAM geometry.')
            output.mkdir(parents=True, exist_ok=True)
            report.update(status='rejected_geometry', failure_reason=str(exc))
            report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
            return report
        before_loop = geo.graph_residual(np.zeros_like(parameters), nodes, loops)
        after_loop = geo.graph_residual(parameters, nodes, loops)
        closure_improved = np.linalg.norm(after_loop) < .98*np.linalg.norm(before_loop)
        try:
            geo.write_candidate(output, pcd, all_scans, trajectory, nodes, parameters, paths['scan_ranges'],
                                graph, edges, evidence, selection, PARAMETERS)
            log('Validating corrected geometry on independent reference records.')
            quality = validate_records(pcd, geo.read_pcd(output/'pcd'/pcd.path.name))
            quality['doubled_surfaces'] = doubled_surfaces(nodes)
            quality['doubled_surfaces_after'] = doubled_surfaces(nodes, parameters)
            log(f"Doubled surfaces over 6 cm: {quality['doubled_surfaces']['share_over_6cm']} -> "
                f"{quality['doubled_surfaces_after']['share_over_6cm']}.")
        except Exception:
            # The directory was proven absent above and contains only this
            # incomplete candidate; never leave a cache-shaped partial result.
            if output.is_dir():
                shutil.rmtree(output)
            raise
        is_accepted = accept_refinement(quality, closure_improved)
        report.update(status='accepted' if is_accepted else 'rejected_geometry',
                      validation=quality, plane_constraints=planes, graph=graph,
                      closure_improved=bool(closure_improved),
                      output_pcd=str(output/'pcd'/pcd.path.name),
                      output_trajectory=str(output/'result'/trajectory.path.name),
                      output_scan_ranges=str(output/'scan_ranges.csv'))
        report['output_signature'] = source_signature({name: Path(report[f'output_{name}'])
                                                       for name in ('pcd', 'trajectory', 'scan_ranges')})
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
