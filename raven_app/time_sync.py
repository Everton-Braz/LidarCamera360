"""Confidence-gated temporal alignment of camera and LiDAR gyro signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class GyroSyncEstimate:
    """Time-offset estimate plus evidence needed to decide whether to use it.

    ``dt_seconds`` follows ``camera_time - lidar_time``.  The LiDAR time axis
    is measured from its first packet, while camera times retain their supplied
    origin (normally the camera exposure epoch).  A camera clip captured later
    in a long LiDAR recording therefore has a negative offset.
    """

    dt_seconds: float | None
    accepted: bool
    score: float | None
    overlap_seconds: float
    overlap_fraction: float
    second_best_score: float | None
    ambiguity_margin: float | None
    reason: str
    sample_rate_hz: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report."""
        return asdict(self)


def _time_signal(times: Any, values: Any, label: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 1 or t.size != x.size:
        raise ValueError(f"{label} times and gyro must be equal-length 1D arrays")
    finite = np.isfinite(t) & np.isfinite(x)
    t, x = t[finite], x[finite]
    if t.size < 2:
        raise ValueError(f"{label} gyro needs at least two finite samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"{label} timestamps must be strictly increasing")
    return t, x


def _resample(times: np.ndarray, values: np.ndarray,
              sample_rate_hz: float) -> tuple[np.ndarray, float]:
    origin = float(times[0])
    local_t = times - origin
    duration = float(local_t[-1])
    count = int(np.floor(duration * sample_rate_hz + 1e-9)) + 1
    grid = np.arange(count, dtype=np.float64) / sample_rate_hz
    return np.interp(grid, local_t, values), origin


def estimate_gyro_time_sync(
    camera_times: Any,
    camera_gyro: Any,
    lidar_times: Any,
    lidar_gyro: Any,
    *,
    sample_rate_hz: float = 100.0,
    min_overlap_seconds: float = 5.0,
    min_overlap_fraction: float = 0.50,
    min_correlation: float = 0.55,
    min_peak_margin: float = 0.03,
    peak_exclusion_seconds: float = 2.0,
) -> GyroSyncEstimate:
    """Find a camera clip anywhere in a longer LiDAR gyro trace.

    Uses sliding, overlap-local Pearson correlation, so unmatched recording
    tails do not dilute the score. Correlation is evaluated on regular samples;
    the default acceptance threshold of 0.55 rejects weak, accidental matches.
    At least five seconds and half of the shorter recording must overlap.

    ``camera_times`` must be relative to its camera clock origin. LiDAR
    timestamps may be relative or epoch-based; their first packet is treated
    as time zero. The returned ``dt_seconds`` is camera time minus LiDAR time.
    Rejected estimates have ``dt_seconds=None``; callers should not silently
    substitute them with a guessed offset.
    """
    params = (sample_rate_hz, min_overlap_seconds, min_overlap_fraction,
              min_correlation, min_peak_margin, peak_exclusion_seconds)
    if not all(np.isfinite(v) for v in params):
        raise ValueError("sync thresholds must be finite")
    if sample_rate_hz <= 0 or min_overlap_seconds <= 0:
        raise ValueError("sample rate and minimum overlap must be positive")
    if not 0 <= min_overlap_fraction <= 1:
        raise ValueError("minimum overlap fraction must be between 0 and 1")
    if not -1 <= min_correlation <= 1:
        raise ValueError("minimum correlation must be between -1 and 1")
    if min_peak_margin < 0 or peak_exclusion_seconds < 0:
        raise ValueError("ambiguity thresholds cannot be negative")

    cam_t, cam_x = _time_signal(camera_times, camera_gyro, "camera")
    lid_t, lid_x = _time_signal(lidar_times, lidar_gyro, "LiDAR")
    cam, cam_origin = _resample(cam_t, cam_x, sample_rate_hz)
    # LiDAR absolute clock values are deliberately normalized to bag start.
    lid, _ = _resample(lid_t, lid_x, sample_rate_hz)

    shorter = min(cam.size, lid.size)
    required = max(
        int(np.ceil(min_overlap_seconds * sample_rate_hz)),
        int(np.ceil(min_overlap_fraction * shorter)),
    )
    lags = signal.correlation_lags(cam.size, lid.size, mode="full")
    # For lag k, corresponding indices satisfy lidar_index = camera_index - k.
    cam_start = np.maximum(0, lags)
    cam_end = np.minimum(cam.size, lid.size + lags)
    overlap = np.maximum(0, cam_end - cam_start)
    allowed = overlap >= required
    if not np.any(allowed):
        return GyroSyncEstimate(None, False, None, 0.0, 0.0, None, None,
                                "insufficient_overlap", sample_rate_hz)

    # A single FFT cross-correlation plus prefix sums gives local Pearson scores
    # for every lag without repeatedly allocating overlap windows.
    sum_xy = signal.correlate(cam, lid, mode="full", method="fft")
    lid_start = cam_start - lags
    lid_end = cam_end - lags

    def interval_sums(values: np.ndarray, starts: np.ndarray,
                      ends: np.ndarray) -> np.ndarray:
        prefix = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        return prefix[ends] - prefix[starts]

    sum_x = interval_sums(cam, cam_start, cam_end)
    sum_x2 = interval_sums(cam * cam, cam_start, cam_end)
    sum_y = interval_sums(lid, lid_start, lid_end)
    sum_y2 = interval_sums(lid * lid, lid_start, lid_end)
    n = np.maximum(overlap, 1).astype(np.float64)
    covariance = sum_xy - sum_x * sum_y / n
    var_x = np.maximum(0.0, sum_x2 - sum_x * sum_x / n)
    var_y = np.maximum(0.0, sum_y2 - sum_y * sum_y / n)
    denom = np.sqrt(var_x * var_y)
    scores = np.full(lags.shape, np.nan, dtype=np.float64)
    usable = allowed & (denom > np.finfo(np.float64).eps)
    scores[usable] = np.clip(covariance[usable] / denom[usable], -1.0, 1.0)
    if not np.any(np.isfinite(scores)):
        return GyroSyncEstimate(None, False, None, 0.0, 0.0, None, None,
                                "insufficient_signal_variation", sample_rate_hz)

    best_idx = int(np.nanargmax(scores))
    best_lag = int(lags[best_idx])
    best_score = float(scores[best_idx])
    best_overlap = int(overlap[best_idx])
    candidates = np.isfinite(scores) & (
        np.abs(lags - best_lag) >= int(np.ceil(peak_exclusion_seconds * sample_rate_hz))
    )
    second_score = float(np.max(scores[candidates])) if np.any(candidates) else None
    margin = best_score - second_score if second_score is not None else None
    overlap_fraction = best_overlap / shorter

    if best_score < min_correlation:
        accepted, reason = False, "weak_correlation"
    elif margin is not None and margin < min_peak_margin:
        accepted, reason = False, "ambiguous_peak"
    else:
        accepted, reason = True, "accepted"

    dt = cam_origin + best_lag / sample_rate_hz
    return GyroSyncEstimate(
        float(dt) if accepted else None,
        accepted,
        best_score,
        best_overlap / sample_rate_hz,
        float(overlap_fraction),
        second_score,
        margin,
        reason,
        float(sample_rate_hz),
    )
