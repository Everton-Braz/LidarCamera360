"""Level a LiDAR-inertial SLAM map to the gravity measured over the whole walk.

FAST-LIVO2 takes "down" from the first ~2 s of IMU data. A recording that starts while the operator walks
(a trimmed clip, a chunk) mixes walking acceleration into that average and the whole map comes out tilted by a
constant few degrees: the floor slopes metres over a building and a top view is not a top view.

Over a whole walk the operator's own acceleration averages out (it starts and ends at walking pace), so the
mean specific force, rotated into the map by the SLAM attitudes, points straight up. Rotating the map so that
it does fixes the tilt without changing heading.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np

from raven_app.slam_refinement_geometry import read_pcd, read_trajectory

LEVEL_FILE = 'gravity_level.json'
MIN_TILT_DEG = 0.5  # static starts land within ~0.3 deg; leave them alone
MAX_TILT_DEG = 25.0  # more than this is not an initialization error
MIN_SECONDS = 20.0
MIN_GRAVITY_SHARE = 0.9  # |mean specific force| / median |specific force|: gravity dominates the average
MIN_SYNC_CORRELATION = 0.5
VELOCITY_HALF_WINDOW_S = 0.5
CHUNK_POINTS = 2_000_000


def _pearson(a, b):
    """Correlation without np.corrcoef, whose np.dot is BLAS (see _rotate_columns)."""
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _imu_on_trajectory_clock(imu, traj):
    """imu.txt counts from the first LiDAR frame; find its offset to the trajectory by matching rotation rates."""
    since_start = imu[:, 0] - imu[0, 0]
    rate = np.linalg.norm((traj.rotations[:-1].inv() * traj.rotations[1:]).as_rotvec(), axis=1) / np.diff(traj.times)
    elapsed = traj.times[:-1] - traj.times[0]
    magnitude = np.linalg.norm(imu[:, 1:4], axis=1)
    offsets = np.arange(-5.0, 5.0, 0.02)
    with np.errstate(invalid='ignore', divide='ignore'):
        scores = np.nan_to_num([_pearson(rate, np.interp(elapsed + o, since_start, magnitude))
                                for o in offsets], nan=-1.0)
    best = int(np.argmax(scores))
    offset = offsets[best] if scores[best] >= MIN_SYNC_CORRELATION else 0.0
    return since_start - offset + traj.times[0], float(scores[best])


def measure_tilt(imu_path, trajectory_path):
    """Gravity direction in the map from the IMU log and the trajectory (both in the IMU body frame)."""
    traj = read_trajectory(Path(trajectory_path))
    imu = np.loadtxt(imu_path, ndmin=2)
    if imu.ndim != 2 or imu.shape[1] < 7 or len(imu) < 100:
        return {'status': 'no_imu_log'}
    times, correlation = _imu_on_trajectory_clock(imu, traj)
    inside = (times > traj.times[0] + VELOCITY_HALF_WINDOW_S) & (times < traj.times[-1] - VELOCITY_HALF_WINDOW_S)
    seconds = float(times[inside][-1] - times[inside][0]) if inside.sum() > 1 else 0.0
    if seconds < MIN_SECONDS:
        return {'status': 'too_short', 'seconds': seconds}
    world = traj.slerp(times[inside]).apply(imu[inside, 4:7])
    # The walk's own acceleration averages to (v_end - v_start) / T, not zero: a 60 s clip that starts walking
    # one way and ends walking another is off by ~0.5 deg without this.
    first, last = times[inside][0], times[inside][-1]
    motion = (_velocity(traj, last) - _velocity(traj, first)) / (last - first)
    mean = world.mean(axis=0) - motion
    share = float(np.linalg.norm(mean) / np.median(np.linalg.norm(imu[inside, 4:7], axis=1)))
    up = mean / np.linalg.norm(mean)
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return {'status': 'measured', 'up': up.tolist(), 'tilt_deg': round(tilt, 3),
            'toward_azimuth_deg': round(float(np.degrees(np.arctan2(up[1], up[0]))), 1),
            'seconds': round(seconds, 1), 'gravity_share': round(share, 4),
            'motion_correction_m_s2': round(float(np.linalg.norm(motion)), 4),
            'imu_sync_correlation': round(correlation, 3)}


def _velocity(traj, time):
    """Map-frame velocity averaged over a second (smooths the gait)."""
    h = VELOCITY_HALF_WINDOW_S
    return np.array([np.interp(time + h, traj.times, traj.positions[:, axis])
                     - np.interp(time - h, traj.times, traj.positions[:, axis]) for axis in range(3)]) / (2 * h)


def leveling_rotation(up):
    """Smallest rotation taking ``up`` to +Z (heading unchanged)."""
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    axis = np.cross(up, [0.0, 0.0, 1.0])
    sine, cosine = np.linalg.norm(axis), up[2]
    if sine < 1e-12:
        return np.eye(3)
    k = axis / sine
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    # Rodrigues; skew @ skew == outer(k, k) - I for a unit axis, written without BLAS
    return np.eye(3) + sine * skew + (1 - cosine) * (np.outer(k, k) - np.eye(3))


def _rotate_columns(columns, rotation):
    """Element-wise 3x3 product (no BLAS: the workflow runs beside other numpy threads)."""
    x, y, z = (np.asarray(c, dtype=np.float64) for c in columns)
    return [rotation[i, 0] * x + rotation[i, 1] * y + rotation[i, 2] * z for i in range(3)]


def _rotate_pcd(source, target, rotation):
    pcd = read_pcd(Path(source))
    shutil.copyfile(source, target)
    points = np.memmap(target, dtype=pcd.dtype, mode='r+', offset=pcd.data_offset, shape=(pcd.count,))
    names = {name.lower(): name for name in pcd.dtype.names}
    triples = [pcd.xyz_fields]
    normals = tuple(names.get(f'normal_{axis}') for axis in 'xyz')
    if all(normals) and all(pcd.dtype.fields[n][0].kind == 'f' and pcd.dtype.fields[n][0].subdtype is None
                            for n in normals):
        triples.append(normals)
    for start in range(0, pcd.count, CHUNK_POINTS):
        block = slice(start, min(start + CHUNK_POINTS, pcd.count))
        for fields in triples:
            rotated = _rotate_columns([points[name][block] for name in fields], rotation)
            for name, values in zip(fields, rotated):
                points[name][block] = values
    points.flush()
    del points


def _rotate_trajectory(source, target, rotation):
    from scipy.spatial.transform import Rotation
    rows = np.loadtxt(source, dtype=np.float64, ndmin=2)
    rows[:, 1:4] = np.column_stack(_rotate_columns(rows[:, 1:4].T, rotation))
    rows[:, 4:8] = (Rotation.from_matrix(rotation) * Rotation.from_quat(rows[:, 4:8])).as_quat()
    np.savetxt(target, rows, fmt='%.6f')


def _write_marker(slam_dir, record):
    temporary = Path(slam_dir) / (LEVEL_FILE + '.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    temporary.replace(Path(slam_dir) / LEVEL_FILE)


def level_interrupted(slam_dir):
    """True when an earlier leveling stopped half-way: the SLAM output must be recomputed."""
    try:
        record = json.loads((Path(slam_dir) / LEVEL_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return record.get('status') == 'pending'


def level_slam_output(slam_dir, pcd_path, trajectory_path, log=print):
    """Rotate a raw SLAM output (cloud, normals, trajectory) so gravity is -Z. Runs once per SLAM output."""
    slam_dir, pcd_path, trajectory_path = Path(slam_dir), Path(pcd_path), Path(trajectory_path)
    marker = slam_dir / LEVEL_FILE
    try:
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record.get('status') != 'pending':
            return record
    except (OSError, ValueError):
        pass
    imu_path = slam_dir / 'imu.txt'
    try:
        record = measure_tilt(imu_path, trajectory_path) if imu_path.is_file() else {'status': 'no_imu_log'}
    except (OSError, ValueError) as exc:
        record = {'status': 'unmeasured', 'reason': str(exc)}
    if record['status'] != 'measured':
        _write_marker(slam_dir, record)
        return record
    tilt = record['tilt_deg']
    if record['gravity_share'] < MIN_GRAVITY_SHARE or tilt > MAX_TILT_DEG:
        record['status'] = 'unreliable'
        log(f'[!] Gravity check of the SLAM map is unreliable (tilt {tilt:.1f} deg, '
            f'gravity share {record["gravity_share"]:.2f}); the map is kept as computed.')
    elif tilt < MIN_TILT_DEG:
        record['status'] = 'level'
        log(f'[*] SLAM map is level: gravity within {tilt:.2f} deg of vertical over {record["seconds"]:.0f} s.')
    else:
        rotation = leveling_rotation(record['up'])
        record.update(status='pending', rotation=rotation.round(9).tolist())
        _write_marker(slam_dir, record)
        log(f'[*] SLAM map is tilted {tilt:.2f} deg (the recording starts in motion): leveling the cloud and '
            f'trajectory to the gravity measured over {record["seconds"]:.0f} s.')
        files = [(pcd_path, _rotate_pcd), (trajectory_path, _rotate_trajectory)]
        lidar_poses = pcd_path.parent / 'lidar_poses.txt'
        if lidar_poses.is_file():
            files.append((lidar_poses, _rotate_trajectory))
        staged = []
        for path, rotate in files:
            temporary = path.with_name(path.name + '.leveling')
            rotate(path, temporary, rotation)
            staged.append((temporary, path))
        for temporary, path in staged:
            os.replace(temporary, path)
        record['status'] = 'leveled'
    _write_marker(slam_dir, record)
    return record
