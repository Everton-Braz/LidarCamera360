"""Per-point scan times for time-gated colorization of dynamic scenes.

The accumulated SLAM map holds every state of a changing scene at once (cars that
left or arrived between passes, people walking).  A view may only colour points the
LiDAR scanned around that view's own time; the same subset is the scene state used
for its occlusion test.
"""
from pathlib import Path

import numpy as np

DEFAULT_TIME_WINDOW_S = 6.0


def point_scan_times(slam_dir, finite_mask):
    """Return the scan time (SLAM clock, seconds) of every point kept by ``load_pcd``.

    ``scan_ranges.csv`` (timestamp, point_start, point_count) maps contiguous point
    blocks of ``pcd/all_raw_points.pcd`` to their scan.  ``finite_mask`` is the
    per-raw-point mask ``load_pcd`` applied.  Returns None when the table is missing
    or does not describe this cloud, so callers fall back to ungated colorization.
    """
    table = Path(slam_dir) / 'scan_ranges.csv'
    if not table.is_file():
        print(f'[!] Time gating disabled: {table.name} not found in {Path(slam_dir).name}')
        return None
    try:
        rows = np.loadtxt(table, delimiter=',', skiprows=1, ndmin=2)
    except ValueError as exc:
        print(f'[!] Time gating disabled: unreadable {table.name} ({exc})')
        return None
    starts, counts = rows[:, 1].astype(np.int64), rows[:, 2].astype(np.int64)
    total = len(finite_mask)
    contiguous = len(rows) and starts[0] == 0 and np.array_equal(starts[1:], np.cumsum(counts)[:-1])
    if not contiguous or counts.sum() != total or (counts < 0).any():
        print(f'[!] Time gating disabled: {table.name} does not match the point cloud '
              f'({int(counts.sum()):,} listed vs {total:,} points)')
        return None
    times = np.repeat(rows[:, 0], counts)[np.asarray(finite_mask, dtype=bool)]
    if len(times) > 1 and np.any(np.diff(times) < 0):
        print(f'[!] Time gating disabled: {table.name} timestamps are not monotonic')
        return None
    return times


def window_slice(point_times, view_time, window):
    """Contiguous index range of points scanned within ±window of ``view_time``."""
    lo = np.searchsorted(point_times, view_time - window, side='left')
    hi = np.searchsorted(point_times, view_time + window, side='right')
    return slice(int(lo), int(hi))
