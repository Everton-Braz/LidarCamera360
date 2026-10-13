"""Reusable, dataset-independent geometry and provenance I/O for SLAM refinement."""
from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

@dataclass
class PCD:
    path: Path
    header: bytes
    data_offset: int
    count: int
    dtype: np.dtype
    points: np.memmap
    xyz_fields: tuple[str, str, str]

@dataclass
class Trajectory:
    path: Path
    rows: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    rotations: Rotation
    slerp: Slerp

@dataclass
class Scan:
    timestamp: float
    start: int
    count: int

@dataclass
class Node:
    index: int
    start_time: float
    end_time: float
    timestamp: float
    rotation: Rotation
    position: np.ndarray
    points: np.ndarray
    scan_ids: tuple[int, ...]

def read_pcd(path: Path) -> PCD:
    header_lines: list[bytes] = []
    with path.open("rb") as stream:
        parsed: dict[str, list[str]] = {}
        for _ in range(128):
            raw = stream.readline()
            if not raw:
                break
            header_lines.append(raw)
            fields = raw.decode("ascii", errors="strict").strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            key = fields[0].upper()
            parsed[key] = fields[1:]
            if key == "DATA":
                break
        else:
            raise ValueError(f"PCD header is too long: {path}")
        if parsed.get("DATA") != ["binary"]:
            raise ValueError(f"Only uncompressed binary PCD is supported: {path}")
        names = parsed.get("FIELDS", [])
        sizes = parsed.get("SIZE", [])
        kinds = parsed.get("TYPE", [])
        counts = parsed.get("COUNT", ["1"] * len(names))
        if not names or not (len(names) == len(sizes) == len(kinds) == len(counts)):
            raise ValueError("Malformed PCD FIELDS/SIZE/TYPE/COUNT header")
        type_codes = {"F": "f", "I": "i", "U": "u"}
        entries = []
        for name, size_text, kind, count_text in zip(names, sizes, kinds, counts):
            if kind not in type_codes:
                raise ValueError(f"Unsupported PCD scalar type {kind!r}")
            size, count = int(size_text), int(count_text)
            if size not in (1, 2, 4, 8) or count < 1:
                raise ValueError(f"Unsupported PCD field layout for {name}")
            spec = "<" + type_codes[kind] + str(size)
            entries.append((name, spec) if count == 1 else (name, spec, (count,)))
        point_count = int(parsed.get("POINTS", ["-1"])[0])
        if point_count < 0:
            width = int(parsed.get("WIDTH", ["-1"])[0])
            height = int(parsed.get("HEIGHT", ["1"])[0])
            point_count = width * height
        data_offset = stream.tell()
        header = b"".join(header_lines)
    dtype = np.dtype(entries, align=False)
    if path.stat().st_size != data_offset + point_count * dtype.itemsize:
        raise ValueError("PCD binary payload size does not match its header")
    field_map = {name.lower(): name for name in names}
    if not all(name in field_map for name in ("x", "y", "z")):
        raise ValueError("PCD must contain x, y, z fields")
    xyz = tuple(field_map[name] for name in ("x", "y", "z"))
    for name in xyz:
        if dtype.fields[name][0].subdtype is not None or dtype.fields[name][0].kind != "f":
            raise ValueError(f"PCD coordinate field {name} must be a scalar float")
    points = np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=(point_count,))
    return PCD(path, header, data_offset, point_count, dtype, points, xyz)

def read_trajectory(path: Path) -> Trajectory:
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if rows.ndim != 2 or rows.shape[1] < 8 or len(rows) < 2:
        raise ValueError(f"Trajectory must contain timestamp, xyz, and xyzw quaternion: {path}")
    times = rows[:, 0]
    if not np.isfinite(rows[:, :8]).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Trajectory timestamps must be finite and strictly increasing")
    quaternions = rows[:, 4:8]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("Trajectory contains a zero quaternion")
    rotations = Rotation.from_quat(quaternions / norms[:, None])
    return Trajectory(path, rows, times, rows[:, 1:4].copy(), rotations,
                      Slerp(times, rotations))

def read_scan_ranges(path: Path, point_count: int) -> list[Scan]:
    scans: list[Scan] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp", "point_start", "point_count"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError("scan_ranges.csv must have timestamp,point_start,point_count columns")
        expected_start = 0
        previous_time = -math.inf
        for line_number, row in enumerate(reader, start=2):
            try:
                timestamp = float(row["timestamp"])
                start_float, count_float = float(row["point_start"]), float(row["point_count"])
                start, count = int(start_float), int(count_float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid scan provenance row {line_number}") from exc
            if not math.isfinite(timestamp) or start_float != start or count_float != count:
                raise ValueError(f"Non-finite timestamp or non-integer range at row {line_number}")
            if timestamp <= previous_time:
                raise ValueError(f"Scan timestamps are not strictly increasing at row {line_number}")
            if start != expected_start or count < 0:
                raise ValueError(
                    f"Scan range mismatch at row {line_number}: expected start {expected_start}, "
                    f"got start={start}, count={count}")
            scans.append(Scan(timestamp, start, count))
            expected_start += count
            previous_time = timestamp
    if not scans:
        raise ValueError("scan_ranges.csv contains no scans")
    if expected_start != point_count:
        raise ValueError(f"Provenance covers {expected_start:,} points; PCD contains {point_count:,}")
    return scans

def pose_at(traj: Trajectory, timestamp: float) -> tuple[Rotation, np.ndarray]:
    if timestamp < traj.times[0] or timestamp > traj.times[-1]:
        raise ValueError(
            f"Scan timestamp {timestamp:.9f} lies outside trajectory domain "
            f"[{traj.times[0]:.9f}, {traj.times[-1]:.9f}]")
    rotation = traj.slerp([timestamp])[0]
    position = np.array([np.interp(timestamp, traj.times, traj.positions[:, axis])
                         for axis in range(3)])
    return rotation, position

def voxel_downsample(points: np.ndarray, voxel_m: float, maximum: int) -> np.ndarray:
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return points.reshape(0, 3)
    keys = np.floor(points / voxel_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    reduced = sums / counts[:, None]
    if len(reduced) > maximum:
        # Deterministic random cap avoids bias from lexicographic voxel ordering.
        selected = np.sort(np.random.default_rng(0).choice(len(reduced), maximum, replace=False))
        reduced = reduced[selected]
    return reduced

def build_nodes(pcd: PCD, scans: list[Scan], traj: Trajectory,
                window_seconds: float, voxel_m: float, maximum: int) -> list[Node]:
    # Use temporal windows while preserving the exact contiguous scan order.
    groups: list[list[int]] = []
    current: list[int] = []
    bucket = None
    origin = scans[0].timestamp
    for scan_id, scan in enumerate(scans):
        this_bucket = int(math.floor((scan.timestamp - origin) / window_seconds))
        if current and this_bucket != bucket:
            groups.append(current)
            current = []
        current.append(scan_id)
        bucket = this_bucket
    if current:
        groups.append(current)

    nodes: list[Node] = []
    for scan_ids in groups:
        first, last = scans[scan_ids[0]], scans[scan_ids[-1]]
        center_time = (first.timestamp + last.timestamp) * 0.5
        center_rotation, center_position = pose_at(traj, center_time)
        chunks = []
        for scan_id in scan_ids:
            scan = scans[scan_id]
            if not scan.count:
                continue
            raw = pcd.points[scan.start:scan.start + scan.count]
            points = np.column_stack([raw[name] for name in pcd.xyz_fields]).astype(np.float64)
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if len(points):
                # PCD points are in the SLAM world frame; express each submap in
                # the local frame of its midpoint trajectory pose.
                local = center_rotation.inv().apply(points - center_position)
                chunks.append(local)
        if not chunks:
            raise ValueError(f"Submap {len(nodes)} has no finite points")
        downsampled = voxel_downsample(np.concatenate(chunks), voxel_m, maximum)
        if len(downsampled) < 200:
            raise ValueError(f"Submap {len(nodes)} has only {len(downsampled)} points")
        nodes.append(Node(len(nodes), first.timestamp, last.timestamp, center_time,
                          center_rotation, center_position, downsampled, tuple(scan_ids)))
    if len(nodes) < 2:
        raise ValueError("Need at least two temporal submaps")
    return nodes

def relative_pose(source: Node, target: Node) -> tuple[Rotation, np.ndarray]:
    """Return coordinates source-local -> target-local using original poses."""
    rotation = target.rotation.inv() * source.rotation
    translation = target.rotation.inv().apply(source.position - target.position)
    return rotation, translation

def estimate_normals(points: np.ndarray, k: int = 20):
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=min(k, len(points)), workers=-1)
    del distances
    neighborhoods = points[neighbors]
    centered = neighborhoods - neighborhoods.mean(axis=1, keepdims=True)
    covariances = np.einsum("nki,nkj->nij", centered, centered) / neighborhoods.shape[1]
    values, vectors = np.linalg.eigh(covariances)
    denom = np.maximum(values.sum(axis=1), 1e-12)
    planar = (values[:, 0] / denom) < 0.18
    normals = vectors[:, :, 0]
    normals[~planar] = np.nan
    good = normals[np.isfinite(normals).all(axis=1)]
    if len(good) < 200:
        raise ValueError("Too few locally planar target points for ICP normals")
    information = (good.T @ good) / len(good)
    eig = np.linalg.eigvalsh(information)
    if eig[0] < 0.025 or eig[-1] / max(eig[0], 1e-12) > 35:
        raise ValueError(f"Degenerate planar-only normal distribution (eigenvalues={eig.tolist()})")
    return tree, normals, {"count": int(len(good)), "normal_information_eigenvalues": eig.tolist()}

def _correspondences(source, target, tree, normals, rotation, translation, threshold):
    moved = rotation.apply(source) + translation
    distances, indices = tree.query(moved, k=1, workers=-1)
    valid = (distances <= threshold) & np.isfinite(normals[indices]).all(axis=1)
    return moved, distances, indices, valid

def _conditioned_update(points, targets, normals, residuals):
    design = np.column_stack((np.cross(points, normals), normals))
    # Balance angular and translational columns using the scene's lever arm.
    scale = max(float(np.median(np.linalg.norm(points, axis=1))), 1.0)
    scaled = design.copy()
    scaled[:, :3] /= scale
    singular = np.linalg.svd(scaled, compute_uv=False)
    if len(singular) < 6 or singular[-1] <= 0 or singular[0] / singular[-1] > 1e4:
        raise ValueError(f"Poorly-conditioned point-to-plane constraint (condition={singular[0] / max(singular[-1], 1e-15):.3g})")
    # Huber influence limits transient/dynamic points without trimming the output cloud.
    sigma = max(float(np.median(np.abs(residuals))) * 1.4826, 0.01)
    cutoff = max(2.5 * sigma, 0.03)
    weights = np.ones_like(residuals)
    large = np.abs(residuals) > cutoff
    weights[large] = cutoff / np.abs(residuals[large])
    root = np.sqrt(weights)
    solution, *_ = np.linalg.lstsq(scaled * root[:, None], -residuals * root, rcond=None)
    return solution[:3] / scale, solution[3:]

def icp_point_to_plane(source: np.ndarray, target: np.ndarray,
                       initial_rotation: Rotation, initial_translation: np.ndarray):
    tree, normals, normal_report = estimate_normals(target)
    rotation, translation = initial_rotation, initial_translation.copy()
    iterations = []
    for gate in (1.5, 0.7, 0.3):
        stage_count = 0
        for iteration in range(35):
            moved, distances, indices, valid = _correspondences(
                source, target, tree, normals, rotation, translation, gate)
            if int(valid.sum()) < max(200, int(0.25 * len(source))):
                raise ValueError(f"Only {int(valid.sum())} correspondences at {gate:.2f}m gate")
            pts = moved[valid]
            nrms = normals[indices[valid]]
            residuals = np.einsum("ij,ij->i", nrms, pts - target[indices[valid]])
            omega, shift = _conditioned_update(pts, target[indices[valid]], nrms, residuals)
            delta = Rotation.from_rotvec(omega)
            translation = delta.apply(translation) + shift
            rotation = delta * rotation
            stage_count = iteration + 1
            if np.linalg.norm(omega) < 2e-5 and np.linalg.norm(shift) < 2e-4:
                break
        iterations.append({"gate_m": gate, "iterations": stage_count,
                           "correspondences": int(valid.sum())})
    moved, distances, indices, valid = _correspondences(
        source, target, tree, normals, rotation, translation, 0.3)
    fitness = float(valid.sum() / len(source))
    rms = float(np.sqrt(np.mean(np.square(distances[valid])))) if valid.any() else math.inf
    if fitness < 0.5:
        raise ValueError(f"Final ICP fitness {fitness:.3f} is below 0.50")
    plane_errors = np.einsum("ij,ij->i", normals[indices[valid]], moved[valid] - target[indices[valid]])
    plane_rms = float(np.sqrt(np.mean(plane_errors ** 2)))
    if plane_rms >= 0.15 or rms >= 0.25:
        raise ValueError(f"Final ICP surface RMS {plane_rms:.4f}m / nearest RMS {rms:.4f}m failed 0.15m / 0.25m gates")
    return rotation, translation, {"fitness": fitness, "rms_m": rms, "plane_rms_m": plane_rms,
                                  "iterations": iterations, **normal_report}

def directed_metrics(source: np.ndarray, target: np.ndarray, threshold: float):
    distances, _ = cKDTree(target).query(source, k=1, workers=-1)
    inliers = distances <= threshold
    return {"count": int(len(source)), "overlap": float(inliers.mean()),
            "rms_inlier_m": float(np.sqrt(np.mean(distances[inliers] ** 2))) if inliers.any() else None,
            "rms_all_capped_m": float(np.sqrt(np.mean(np.minimum(distances, threshold) ** 2)))}

def heldout_metrics(source: np.ndarray, target: np.ndarray,
                    rotation: Rotation, translation: np.ndarray, threshold: float = 0.5):
    forward = directed_metrics(rotation.apply(source) + translation, target, threshold)
    # Symmetric reverse direction uses only the held-out source and target points.
    reverse = directed_metrics(rotation.inv().apply(target - translation), source, threshold)
    return {"source_to_target": forward, "target_to_source": reverse,
            "symmetric_overlap": 0.5 * (forward["overlap"] + reverse["overlap"]),
            "symmetric_rms_inlier_m": (
                None if forward["rms_inlier_m"] is None or reverse["rms_inlier_m"] is None else
                0.5 * (forward["rms_inlier_m"] + reverse["rms_inlier_m"]))}

def propose_loop(source_node: Node, target_node: Node, seed: int):
    initial_rotation, initial_translation = relative_pose(source_node, target_node)
    source, target = source_node.points, target_node.points
    rng = np.random.default_rng(seed)
    src_order = rng.permutation(len(source))
    tgt_order = rng.permutation(len(target))
    src_cut, tgt_cut = int(0.8 * len(source)), int(0.8 * len(target))
    source_fit, source_test = source[src_order[:src_cut]], source[src_order[src_cut:]]
    target_fit, target_test = target[tgt_order[:tgt_cut]], target[tgt_order[tgt_cut:]]
    if len(source_test) < 100 or len(target_test) < 100:
        raise ValueError("Too few independent held-out points")

    base = heldout_metrics(source_test, target_test, initial_rotation, initial_translation)
    before_fit = directed_metrics(initial_rotation.apply(source_fit) + initial_translation,
                                  target_fit, 0.3)
    rotation, translation, icp = icp_point_to_plane(
        source_fit, target_fit, initial_rotation, initial_translation)
    fit = directed_metrics(rotation.apply(source_fit) + translation, target_fit, 0.3)
    if fit["overlap"] < 0.45:
        raise ValueError(f"Training fitness {fit['overlap']:.3f} is below 0.45")
    if fit["rms_inlier_m"] is None or fit["rms_inlier_m"] >= 0.25:
        raise ValueError("Training final RMS is not below 0.25m")
    before_rms = before_fit["rms_all_capped_m"]
    after_rms = fit["rms_all_capped_m"]
    if after_rms > before_rms * 1.02 and (fit["rms_inlier_m"] or 0) > 0.15:
        raise ValueError(f"Training residual improvement is not substantial ({before_rms:.4f}->{after_rms:.4f}m)")

    after = heldout_metrics(source_test, target_test, rotation, translation)
    before_sym = base["symmetric_rms_inlier_m"]
    after_sym = after["symmetric_rms_inlier_m"]
    if before_sym is not None and after_sym is not None and after_sym > before_sym * 1.05:
        raise ValueError(f"Held-out symmetric residual degraded ({before_sym}->{after_sym})")
    if after["symmetric_overlap"] + 0.05 < base["symmetric_overlap"]:
        raise ValueError("Held-out symmetric overlap decreased")
    for direction in ("source_to_target", "target_to_source"):
        old, new = base[direction], after[direction]
        if new["overlap"] + 0.05 < old["overlap"]:
            raise ValueError(f"Held-out {direction} overlap decreased")
        if old["rms_inlier_m"] is not None and new["rms_inlier_m"] is not None and new["rms_inlier_m"] > old["rms_inlier_m"] * 1.05:
            raise ValueError(f"Held-out {direction} residual did not improve")

    delta_rotation = rotation * initial_rotation.inv()
    delta_translation = translation - delta_rotation.apply(initial_translation)
    delta_degrees = float(np.degrees(delta_rotation.magnitude()))
    delta_m = float(np.linalg.norm(delta_translation))
    if delta_m > 8.0 or delta_degrees > 20.0:
        raise ValueError(f"Relative correction too large ({delta_m:.3f}m, {delta_degrees:.2f}deg)")
    return (rotation, translation, {
        "accepted": True, "source_node": source_node.index, "target_node": target_node.index,
        "source_start_s": source_node.start_time, "target_start_s": target_node.start_time,
        "source_mid_s": source_node.timestamp, "target_mid_s": target_node.timestamp,
        "source_points": int(len(source)), "target_points": int(len(target)),
        "initial_relative_translation_m": initial_translation.tolist(),
        "relative_correction_m": delta_m, "relative_correction_deg": delta_degrees,
        "training_before": before_fit, "training_after": fit,
        "heldout_before": base, "heldout_after": after, "icp": icp,
    })

def select_loop_pairs(nodes: list[Node], max_neighbors: int):
    pairs = []
    for i, source in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            separation = target.timestamp - source.timestamp
            distance = float(np.linalg.norm(target.position - source.position))
            if separation >= 25.0 and distance <= 16.0:
                pairs.append((distance, -separation, i, j))
    pairs.sort()
    selected, degrees, skipped = [], [0] * len(nodes), []
    for distance, neg_sep, i, j in pairs:
        if degrees[i] >= max_neighbors or degrees[j] >= max_neighbors:
            skipped.append({"source_node": i, "target_node": j,
                            "source_start_s": nodes[i].start_time,
                            "target_start_s": nodes[j].start_time,
                            "spatial_distance_m": distance,
                            "temporal_separation_s": -neg_sep,
                            "accepted": False, "rejected_reason": "per-node candidate cap"})
            continue
        selected.append((i, j, distance, -neg_sep))
        degrees[i] += 1
        degrees[j] += 1
    return selected, skipped, {"spatial_temporal_candidates": len(pairs),
                               "selected_for_icp": len(selected),
                               "skipped_by_degree_cap": len(skipped)}

def transform_from_correction(node: Node, correction: np.ndarray):
    # Right-multiply original local-to-world pose by a local SE(3) correction.
    delta_rotation = Rotation.from_rotvec(correction[:3])
    world_rotation = node.rotation * delta_rotation
    world_position = node.position + node.rotation.apply(correction[3:])
    return world_rotation, world_position

def graph_residual(parameters, nodes, edges):
    corrected = [transform_from_correction(node, np.zeros(6) if i == 0 else
                 parameters[(i - 1) * 6:i * 6]) for i, node in enumerate(nodes)]
    chunks = []
    for edge in edges:
        i, j, measured_rotation, measured_translation, kind = edge
        ri, ti = corrected[i]
        rj, tj = corrected[j]
        predicted_rotation = rj.inv() * ri
        predicted_translation = rj.inv().apply(ti - tj)
        error_rotation = measured_rotation.inv() * predicted_rotation
        error_translation = measured_rotation.inv().apply(predicted_translation - measured_translation)
        if kind == 'odometry':
            trans_sigma, rot_sigma = (0.10, np.deg2rad(0.8))
        else:
            trans_sigma, rot_sigma = (0.02, np.deg2rad(0.2))
        chunks.extend((error_translation / trans_sigma).tolist())
        chunks.extend((error_rotation.as_rotvec() / rot_sigma).tolist())
    return np.asarray(chunks, dtype=np.float64)

def correction_curve(nodes: list[Node], parameters: np.ndarray, query_times: np.ndarray):
    rotations, translations = [], []
    for index, node in enumerate(nodes):
        correction = np.zeros(6) if index == 0 else parameters[(index - 1) * 6:index * 6]
        corrected_rotation, corrected_position = transform_from_correction(node, correction)
        # World-left correction D = corrected pose * inverse(original pose).
        delta_rotation = corrected_rotation * node.rotation.inv()
        delta_translation = corrected_position - delta_rotation.apply(node.position)
        rotations.append(delta_rotation)
        translations.append(delta_translation)
    node_times = np.asarray([node.timestamp for node in nodes])
    clipped = np.clip(query_times, node_times[0], node_times[-1])
    interpolator = Slerp(node_times, Rotation.concatenate(rotations))
    d_rotations = interpolator(clipped)
    d_translations = np.column_stack([
        np.interp(clipped, node_times, np.asarray(translations)[:, axis]) for axis in range(3)])
    return d_rotations, d_translations

def write_candidate(output: Path, pcd: PCD, scans: list[Scan], traj: Trajectory,
                    nodes: list[Node], parameters: np.ndarray, source_ranges: Path,
                    graph_report: dict, edges: list[tuple], loop_report: list[dict],
                    selection_report: dict, run_config: dict):
    if output.exists():
        raise FileExistsError(f"Output directory already exists; choose a new path: {output}")
    output.mkdir(parents=True)
    pcd_out = output / "pcd" / pcd.path.name
    pcd_out.parent.mkdir(parents=True)
    shutil.copyfile(pcd.path, pcd_out)
    output_points = np.memmap(pcd_out, dtype=pcd.dtype, mode="r+", offset=pcd.data_offset,
                              shape=(pcd.count,))

    field_map = {name.lower(): name for name in (pcd.dtype.names or ())}
    normal_fields = tuple(field_map.get(name) for name in ("normal_x", "normal_y", "normal_z"))
    if not all(normal_fields):
        normal_fields = ()
    elif any(pcd.dtype.fields[name][0].subdtype is not None or
             pcd.dtype.fields[name][0].kind != "f" for name in normal_fields):
        normal_fields = ()

    scan_times = np.asarray([scan.timestamp for scan in scans])
    scan_rotations, scan_translations = correction_curve(nodes, parameters, scan_times)
    for scan, d_rotation, d_translation in zip(scans, scan_rotations, scan_translations):
        if not scan.count:
            continue
        sl = slice(scan.start, scan.start + scan.count)
        xyz = np.column_stack([output_points[name][sl] for name in pcd.xyz_fields]).astype(np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        if finite.any():
            transformed = d_rotation.apply(xyz[finite]) + d_translation
            for axis, name in enumerate(pcd.xyz_fields):
                output_points[name][sl][finite] = transformed[:, axis]
        if normal_fields:
            normals = np.column_stack([output_points[name][sl] for name in normal_fields]).astype(np.float64)
            valid_normals = np.isfinite(normals).all(axis=1)
            valid_normals &= np.einsum("ij,ij->i", normals, normals) > 1e-24
            if valid_normals.any():
                rotated_normals = d_rotation.apply(normals[valid_normals])
                for axis, name in enumerate(normal_fields):
                    output_points[name][sl][valid_normals] = rotated_normals[:, axis]
    output_points.flush()
    del output_points

    d_rot, d_trans = correction_curve(nodes, parameters, traj.times)
    corrected_rows = traj.rows.copy()
    corrected_rows[:, 1:4] = d_rot.apply(traj.positions) + d_trans
    corrected_rot = d_rot * traj.rotations
    corrected_rows[:, 4:8] = corrected_rot.as_quat()
    trajectory_out = output / "result" / traj.path.name
    trajectory_out.parent.mkdir(parents=True)
    np.savetxt(trajectory_out, corrected_rows, fmt="%.9f")
    shutil.copyfile(source_ranges, output / "scan_ranges.csv")

    correction_vectors = []
    for i, node in enumerate(nodes):
        correction = np.zeros(6) if i == 0 else parameters[(i - 1) * 6:i * 6]
        r_corr, t_corr = transform_from_correction(node, correction)
        d_r = r_corr * node.rotation.inv()
        d_t = t_corr - d_r.apply(node.position)
        correction_vectors.append({"node": i, "start_s": node.start_time, "end_s": node.end_time,
                                  "mid_s": node.timestamp,
                                  "translation_m": d_t.tolist(),
                                  "rotation_deg": float(np.degrees(d_r.magnitude()))})
    graph_edges = []
    residuals = graph_residual(parameters, nodes, edges).reshape(-1, 6)
    for index, edge in enumerate(edges):
        graph_edges.append({"source_node": edge[0], "target_node": edge[1], "kind": edge[4],
                            "weighted_residual_norm": float(np.linalg.norm(residuals[index]))})
    report = {
        "status": "candidate_generated",
        "notice": "Experimental candidate only. Inspect against surveyed/reference evidence; no accuracy claim.",
        "source_pcd": str(pcd.path), "source_trajectory": str(traj.path),
        "source_scan_ranges": str(source_ranges),
        "point_count_input": pcd.count, "point_count_output": pcd.count,
        "scan_count": len(scans), "trajectory_pose_count": len(traj.rows),
        "time_domain_s": [float(scans[0].timestamp), float(scans[-1].timestamp)],
        "submap_count": len(nodes), "capture_span_s": float(nodes[-1].end_time - nodes[0].start_time),
        "submap_points_max_actual": int(max(len(n.points) for n in nodes)),
        "run_config": run_config,
        "candidate_selection": selection_report,
        "loop_candidates": loop_report,
        "graph": graph_report, "graph_edges": graph_edges,
        "node_world_corrections": correction_vectors,
        "output_pcd": str(pcd_out), "output_trajectory": str(trajectory_out),
        "point_xyz_policy": "All input records retained in original order; no XYZ scale or point removal.",
        "correction_policy": "Rigid SE(3) correction interpolated from optimized submap poses at each scan timestamp; same curve applied to trajectory.",
    }
    (output / "loop_graph_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
