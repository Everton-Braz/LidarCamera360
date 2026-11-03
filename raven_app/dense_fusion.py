"""Additive, adaptive densification of the LiDAR seed with RoMa v2 dense matching (Spirula `dense`).

Spirula's `dense` command matches the registered images of a COLMAP dataset with RoMa v2 at fixed poses
and writes dense/roma.ply. Run on the 3DGS export, whose poses are the metric SLAM + rig ones, its points
are already in the LiDAR frame. They never replace the LiDAR cloud:

* additive: every LiDAR point is kept; RoMa points are only appended;
* adaptive: a RoMa point is added only where the LiDAR is locally sparser than it, i.e. its nearest LiDAR
  point is farther than `gap_factor` x the LiDAR's own spacing there (never less than `min_gap_m`), so
  scanned walls get nothing and gaps, thin parts and far surfaces get filled;
* bounded: nothing farther than `max_gap_m` from any LiDAR point (no floaters in empty space), and the
  additions are thinned to `voxel_m` so they do not outweigh the scan.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

_PLY_TYPES = {'char': 'i1', 'uchar': 'u1', 'int8': 'i1', 'uint8': 'u1', 'short': '<i2', 'ushort': '<u2',
              'int16': '<i2', 'uint16': '<u2', 'int': '<i4', 'uint': '<u4', 'int32': '<i4', 'uint32': '<u4',
              'float': '<f4', 'float32': '<f4', 'double': '<f8', 'float64': '<f8'}


def read_ply(path):
    """(xyz float64 (N,3), rgb uint8 (N,3) or None) of a binary little-endian PLY vertex element."""
    data = Path(path).read_bytes()
    end = data.index(b'end_header') + len(b'end_header')
    end += 2 if data[end:end + 2] == b'\r\n' else 1
    header = data[:end].decode('ascii', 'replace').splitlines()
    if not any(line.startswith('format binary_little_endian') for line in header):
        raise ValueError(f'{path}: only binary little-endian PLY is supported')
    count, fields, in_vertex = 0, [], False
    for line in header:
        words = line.split()
        if words[:1] == ['element']:
            in_vertex = words[1] == 'vertex'
            if in_vertex:
                count = int(words[2])
        elif words[:1] == ['property'] and in_vertex:
            if words[1] == 'list':
                raise ValueError(f'{path}: list properties in the vertex element are not supported')
            fields.append((words[2], _PLY_TYPES[words[1]]))
    vertices = np.frombuffer(data, dtype=np.dtype(fields), count=count, offset=end)
    xyz = np.column_stack([vertices[a] for a in ('x', 'y', 'z')]).astype(np.float64)
    names = vertices.dtype.names
    channels = next((c for c in (('red', 'green', 'blue'), ('r', 'g', 'b')) if all(n in names for n in c)), None)
    rgb = np.column_stack([vertices[c] for c in channels]).astype(np.uint8) if channels else None
    return xyz, rgb


def write_ply(path, xyz, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    for i, a in enumerate('xyz'):
        arr[a] = xyz[:, i]
    for i, a in enumerate('rgb'):
        arr[a] = rgb[:, i]
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {len(xyz)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')
    with open(path, 'wb') as handle:
        handle.write(header)
        handle.write(arr.tobytes())
    return path


def _voxel_thin(xyz, size):
    keys = np.floor(xyz / size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


def additive_fusion(lidar_xyz, dense_xyz, *, gap_factor=1.5, min_gap_m=0.015, max_gap_m=0.30,
                    voxel_m=0.01, spacing_k=4, query_voxel_m=0.01):
    """Indices of the dense points to append to the LiDAR cloud, and a report.

    The LiDAR is thinned to `query_voxel_m` for the neighbour queries (its spacing there is the k-th
    neighbour distance), which keeps a 30 M point cloud tractable without changing where it has gaps.
    """
    from scipy.spatial import cKDTree
    lidar = lidar_xyz[_voxel_thin(lidar_xyz, query_voxel_m)] if query_voxel_m else lidar_xyz
    tree = cKDTree(lidar)
    spacing = tree.query(lidar, k=spacing_k + 1, workers=-1)[0][:, -1]
    gap, nearest = tree.query(dense_xyz, k=1, distance_upper_bound=max_gap_m, workers=-1)
    near = np.isfinite(gap)
    threshold = np.full(len(dense_xyz), np.inf)
    threshold[near] = np.maximum(min_gap_m, gap_factor * spacing[nearest[near]])
    candidates = np.flatnonzero(near & (gap > threshold))
    keep = candidates[_voxel_thin(dense_xyz[candidates], voxel_m)] if len(candidates) else candidates
    report = {'dense_points': int(len(dense_xyz)), 'lidar_points': int(len(lidar_xyz)),
              'lidar_query_points': int(len(lidar)), 'within_max_gap': int(near.sum()),
              'in_lidar_gaps': int(len(candidates)), 'added': int(len(keep)),
              'lidar_spacing_median_m': float(np.median(spacing)),
              'parameters': {'gap_factor': gap_factor, 'min_gap_m': min_gap_m, 'max_gap_m': max_gap_m,
                             'voxel_m': voxel_m, 'spacing_k': spacing_k, 'query_voxel_m': query_voxel_m}}
    return keep, report


def run_roma_dense(colmap_dir, spirula_bin, preset='balanced', log=print):
    """Run `spirula dense` on a COLMAP dataset (images/, masks/, sparse/0) and return dense/roma.ply."""
    from raven_app.subprocess_utils import run_hidden_stream
    colmap_dir = Path(colmap_dir)
    cmd = [str(spirula_bin), 'dense', str(colmap_dir), '--preset', preset]
    log(f"[*] RoMa v2 dense matching: {' '.join(cmd)}")
    t0 = time.time()
    code = run_hidden_stream(cmd)
    output = colmap_dir / 'dense' / 'roma.ply'
    if code != 0 or not output.is_file():
        raise RuntimeError(f'Spirula dense exited with code {code}' + ('' if output.is_file() else '; no dense/roma.ply'))
    log(f'[+] RoMa v2 dense cloud in {time.time() - t0:.0f} s: {output}')
    return output


def densify_colmap_export(colmap_dir, spirula_bin, preset='balanced', log=print, **fusion):
    """RoMa v2 on the 3DGS export, then additive fusion. Writes dense/roma_additions.ply (only the added
    points), dense/lidar_plus_roma.ply (LiDAR seed + additions, a Spirula `--seed-pointcloud`) and
    dense/additive_fusion.json. The LiDAR seed files in sparse/0 are left as they are."""
    colmap_dir = Path(colmap_dir)
    roma = colmap_dir / 'dense' / 'roma.ply'
    if not roma.is_file():
        roma = run_roma_dense(colmap_dir, spirula_bin, preset, log)
    seed = colmap_dir / 'sparse' / '0' / 'points3D.ply'
    lidar_xyz, lidar_rgb = read_ply(seed)
    dense_xyz, dense_rgb = read_ply(roma)
    if dense_rgb is None:
        raise ValueError(f'{roma} has no colours')
    t0 = time.time()
    keep, report = additive_fusion(lidar_xyz, dense_xyz, **fusion)
    write_ply(colmap_dir / 'dense' / 'roma_additions.ply', dense_xyz[keep], dense_rgb[keep])
    fused = write_ply(colmap_dir / 'dense' / 'lidar_plus_roma.ply', np.vstack([lidar_xyz, dense_xyz[keep]]),
                      np.vstack([lidar_rgb, dense_rgb[keep]]))
    report.update(roma=str(roma), seed=str(seed), fused=str(fused), seconds=round(time.time() - t0, 1))
    (colmap_dir / 'dense' / 'additive_fusion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    log(f"[+] Additive densification: +{report['added']:,} RoMa points in LiDAR gaps "
        f"({report['dense_points']:,} dense, {report['within_max_gap']:,} near the scan) -> {fused.name}")
    return report
