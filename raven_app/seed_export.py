"""Exact 3DGS seed budgets and inexpensive cloud-header previews."""

import math
from pathlib import Path


def validate_seed_percent(value):
    percent = float(value)
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise ValueError('Seed percentage must be greater than 0 and at most 100')
    return percent


def seed_point_count(total, percent=100):
    percent = validate_seed_percent(percent)
    total = int(total)
    return min(total, max(1, math.floor(total * percent / 100))) if total > 0 else 0


def sample_seed_points(xyz, rgb, percent=100, max_points=None):
    import numpy as np
    count = seed_point_count(len(xyz), percent)
    if max_points is not None and max_points > 0:
        count = min(count, int(max_points))
    if count == len(xyz):
        return xyz, rgb
    indices = np.linspace(0, len(xyz) - 1, count, dtype=np.int64)
    return xyz[indices], rgb[indices]


def cloud_header_point_count(path):
    """Read only a bounded PLY/PCD header, never the cloud payload."""
    try:
        with Path(path).open('rb') as stream:
            for _ in range(256):
                line = stream.readline(1024).decode('ascii', errors='ignore').strip()
                if line.startswith('element vertex '):
                    return max(0, int(line.split()[-1]))
                if line.startswith('POINTS '):
                    return max(0, int(line.split()[-1]))
                if not line or line == 'end_header' or line.startswith('DATA '):
                    break
    except (OSError, ValueError):
        pass
    return None


def dataset_seed_point_count(dataset_dir):
    """Match the loader's preference for a colored PLY, then raw PCD."""
    root = Path(dataset_dir)
    candidates = sorted((root / 'deliverables').glob('lidar_colored_*.ply'))
    if not candidates:
        candidates = sorted(root.glob('*.ply'))
    for path in reversed(candidates):
        if not path.name.startswith('points3D'):
            count = cloud_header_point_count(path)
            if count:
                return count
    candidates = sorted((root / 'deliverables').glob('*.pcd'))
    candidates += [root / 'slam_out/pcd/all_raw_points.pcd', root / 'all_raw_points.pcd']
    for path in candidates:
        count = cloud_header_point_count(path)
        if count:
            return count
    return None
