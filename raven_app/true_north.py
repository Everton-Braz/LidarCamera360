"""Solve where a gravity-aligned scan sits on the map and which way is north.

A scan has no compass: SLAM fixes gravity but leaves heading free, and a phone-grade
GPS track of a few fixes over a few metres cannot supply it. This module combines
the evidence a capture does have, strongest first:

1. a heading the operator typed in;
2. the true bearing of a street or wall (``street_bearing_deg``), matched to the
   scan's own wall axes;
3. astronomical solar compass from unoccluded sun disks in calibrated 360-degree
   frames, checked against the known solar elevation for that UTC time and place;
4. GPS fixes matched to the trajectory by time, when they span enough ground;
5. the scan's walls registered against aerial imagery, which recovers heading and
   position together (prior-guided when solar or GPS fixes the rough quadrant,
   or full-360 when no prior is available).

Imagery registration cannot certify itself. Run against imagery of the wrong
neighbourhood, a street grid passes the same score gates more often than not, so a
heading resting on imagery alone comes back as ``needs_review`` with an overlay for a
person to check. It becomes ``resolved`` when solar compass or GPS independently agrees.
When nothing qualifies the result says ``unresolved`` instead of inventing a north, and
the scan is still placed on the GPS anchor.

Angles: ``yaw_grid_deg`` is the counter-clockwise rotation from the local XY axes to
the projected CRS' (Easting, Northing) axes, ``E,N = R(yaw) @ xy + t``. Bearings are
clockwise from north. Grid north and true north differ by the meridian convergence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import urllib.request

import numpy as np
import pyproj

TILE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
IMAGERY_SOURCE = 'Esri World Imagery'
IMAGERY_ZOOMS = (19, 18, 17)
IMAGERY_GSD_M = 0.3
MAX_SCENE_RADIUS_M = 80.0
GPS_SEARCH_M = 20.0
FIX_TIME_MARGIN_S = 5.0

# Heading gates for unconstrained 360-degree image registration, measured against every other heading.
MIN_PEAK_Z = 5.0
MIN_RIVAL_MARGIN = 0.25
RIVAL_EXCLUSION_DEG = 12.0
# Gates for prior-guided image registration (when solar compass or synced GPS bounds the heading).
MIN_PRIOR_TRANS_Z = 4.0
MIN_PRIOR_PEAK_Z = 1.5
PRIOR_TRANS_EXCLUSION_M = 3.5
# Placements this close to the best one count as equally good. Long straight walls
# leave a ridge of them, and GPS decides where along the ridge the scan sits.
PLATEAU_FRACTION = 0.9
# GPS alone needs this much baseline to mean anything for heading.
MIN_GPS_INLIERS = 6
MIN_GPS_EXTENT_M = 10.0
# A GPS heading this uncertain may still back up another one; only below the second value does it stand alone.
MAX_GPS_SUPPORT_SIGMA_DEG = 20.0
MAX_GPS_ALONE_SIGMA_DEG = 10.0
# Phone-grade GPS errors drift slowly, so fixes a second apart are not separate measurements.
GPS_FIX_SIGMA_M = 5.0
GPS_RELATIVE_SIGMA_M = 2.0
GPS_CORRELATION_S = 60.0
MIN_POSITION_SIGMA_M = 3.0
# Two independent headings this close confirm each other.
GPS_AGREEMENT_DEG = 10.0
# Position uncertainty when the fixes cannot be paired with the scan by time.
UNTIMED_ANCHOR_SIGMA_M = 15.0
# A fix this close in time to the scan's start or end still places it; the operator walks meanwhile.
MAX_FIX_GAP_S = 60.0
WALK_DRIFT_MPS = 0.7


def wrap_deg(angle: float) -> float:
    return float(angle) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def rotation_2d(yaw_deg: float) -> np.ndarray:
    rad = math.radians(yaw_deg)
    return np.array([[math.cos(rad), -math.sin(rad)], [math.sin(rad), math.cos(rad)]], dtype=np.float64)


def grid_convergence_deg(lat: float, lon: float, crs: Any) -> float:
    """Angle from true north to grid north, positive when grid north lies east of it."""
    transform = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform
    e0, n0 = transform(lon, lat)
    e1, n1 = transform(lon, lat + 1e-4)
    return float(-math.degrees(math.atan2(e1 - e0, n1 - n0)))


def true_bearing_of_axis(axis_local_deg: float, yaw_grid_deg: float, convergence_deg: float) -> float:
    """True bearing of a local direction given as a math angle from local +X."""
    return wrap_deg(90.0 - (axis_local_deg + yaw_grid_deg) + convergence_deg)


def load_timed_trajectory(path: Any) -> Optional[np.ndarray]:
    """(t, x, y, z) rows from a TUM-style trajectory, or None when unreadable."""
    if not path or not Path(path).is_file():
        return None
    try:
        rows = np.loadtxt(str(path), ndmin=2)
    except (OSError, ValueError):
        return None
    return rows[:, :4] if rows.shape[1] >= 4 and len(rows) else None


def _timed_fixes(gps_info: Dict[str, Any], dt_sync: float,
                 capture_origin_s: float) -> List[Tuple[float, np.ndarray]]:
    """(seconds since the trajectory's first pose, E,N) of every fix that is a measurement of its own.

    ``t_video = t_lidar + dt_sync`` is the dataset's sync, and the first GPS record is the start of the
    capture. A position repeated to the last digit is the receiver's previous answer, not a new one.
    """
    records = gps_info.get('records') or []
    utm = np.asarray(gps_info.get('utm_coords', []), dtype=np.float64)
    if not len(records) or len(utm) != len(records) or any('timestamp_epoch' not in r for r in records):
        return []
    epochs = np.array([r['timestamp_epoch'] for r in records], dtype=np.float64)
    origin = float(epochs.min())
    seen, fixes = set(), []
    for i in np.argsort(epochs, kind='stable'):
        position = tuple(utm[i].tolist())
        if position not in seen:
            seen.add(position)
            fixes.append((float(epochs[i] - origin - capture_origin_s - dt_sync), utm[i, :2]))
    return fixes


def match_fixes_to_trajectory(gps_info: Dict[str, Any], trajectory: np.ndarray, dt_sync: float,
                              capture_origin_s: float = 0.0) -> List[Dict[str, Any]]:
    """Pair each GPS fix with the scan position at that moment; fixes outside the trajectory are dropped."""
    times = trajectory[:, 0] - trajectory[0, 0]
    matches = []
    for t, en in _timed_fixes(gps_info, dt_sync, capture_origin_s):
        if t < -FIX_TIME_MARGIN_S or t > times[-1] + FIX_TIME_MARGIN_S:
            continue
        clamped = min(max(t, 0.0), float(times[-1]))
        local = [float(np.interp(clamped, times, trajectory[:, axis])) for axis in (1, 2)]
        matches.append({'t': t, 'local_xy': local, 'utm_en': en.tolist()})
    return matches


def _nearest_outside_fix(gps_info: Dict[str, Any], trajectory: np.ndarray, dt_sync: float,
                         capture_origin_s: float) -> Optional[Dict[str, Any]]:
    """The fix closest in time before or after the scan, paired with the end of the scan it is next to."""
    end = float(trajectory[-1, 0] - trajectory[0, 0])
    best = None
    for t, en in _timed_fixes(gps_info, dt_sync, capture_origin_s):
        gap = -t if t < 0 else t - end
        if best is None or gap < best['gap_s']:
            best = {'gap_s': float(gap), 'local': trajectory[0 if t < 0 else -1, 1:3].copy(), 'en': en.copy()}
    return best if best is not None and best['gap_s'] <= MAX_FIX_GAP_S else None


def wall_points(points: np.ndarray, *, cell_m: float = 5.0, band_m: Tuple[float, float] = (1.0, 3.0),
                voxel_m: float = 0.15) -> np.ndarray:
    """XY of everything standing 1-3 m above the local ground: what a plan view sees.

    Height is measured against the lowest point in each cell so a sloping site does
    not push its walls out of the band. Points are thinned to a plan voxel so a wall
    counts by its length, not by how long the scanner looked at it.
    """
    xyz = np.asarray(points, dtype=np.float64)
    keys = np.floor(xyz[:, :2] / cell_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    ground = np.full(int(inverse.max()) + 1, np.inf)
    np.minimum.at(ground, inverse, xyz[:, 2])
    height = xyz[:, 2] - ground[inverse]
    kept = xyz[(height >= band_m[0]) & (height <= band_m[1]), :2]
    if not len(kept):
        return np.zeros((0, 2), dtype=np.float64)
    return np.unique(np.floor(kept / voxel_m).astype(np.int64), axis=0) * voxel_m + voxel_m / 2.0


def wall_directions(xy: np.ndarray, *, neighbours: int = 16,
                    min_linearity: float = 6.0) -> Tuple[np.ndarray, np.ndarray]:
    """Keep plan points on straight runs of wall, with each run's direction in radians."""
    from scipy.spatial import cKDTree
    points = np.asarray(xy, dtype=np.float64)
    if len(points) < neighbours:
        return points[:0], np.zeros(0)
    _, index = cKDTree(points).query(points, k=neighbours)
    near = points[index] - points[index].mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', near, near) / neighbours
    values, vectors = np.linalg.eigh(covariance)
    straight = values[:, 1] >= min_linearity * np.maximum(values[:, 0], 1e-6)
    return points[straight], np.arctan2(vectors[straight, 1, 1], vectors[straight, 0, 1])


def wall_axes(xy: np.ndarray, *, count: int = 4, trials: int = 3000, inlier_m: float = 0.08,
              seed: int = 0) -> List[Dict[str, float]]:
    """Longest straight wall lines, longest first, each as a local math angle in [0, 180)."""
    rng = np.random.default_rng(seed)
    remaining = np.asarray(xy, dtype=np.float64)
    if len(remaining) > 20000:
        remaining = remaining[rng.choice(len(remaining), 20000, replace=False)]
    axes = []
    for _ in range(count):
        if len(remaining) < 20:
            break
        pairs = rng.integers(0, len(remaining), size=(trials, 2))
        delta = remaining[pairs[:, 1]] - remaining[pairs[:, 0]]
        length = np.hypot(delta[:, 0], delta[:, 1])
        usable = length >= 1.0
        if not usable.any():
            break
        normals = np.stack([-delta[usable, 1], delta[usable, 0]], axis=1) / length[usable, None]
        anchors = remaining[pairs[usable, 0]]
        best, best_mask = 0, None
        for normal, anchor in zip(normals, anchors):
            mask = np.abs((remaining - anchor) @ normal) < inlier_m
            total = int(mask.sum())
            if total > best:
                best, best_mask = total, mask
        line = remaining[best_mask]
        centre = line.mean(axis=0)
        direction = np.linalg.eigh(np.cov((line - centre).T))[1][:, 1]
        along = (line - centre) @ direction
        occupied = len(np.unique(np.floor(along)))
        axes.append({'axis_deg': float(math.degrees(math.atan2(direction[1], direction[0])) % 180.0),
                     'length_m': float(occupied), 'points': len(line)})
        remaining = remaining[~best_mask]
    axes.sort(key=lambda a: -a['length_m'])
    return axes


def _independent_fixes(matches: List[Dict[str, Any]], keep: np.ndarray) -> float:
    """How many separate measurements the kept fixes amount to, given how slowly GPS errors drift."""
    times = np.array([m['t'] for m in matches], dtype=np.float64)[keep]
    return min(float(len(times)), 1.0 + float(np.ptp(times)) / GPS_CORRELATION_S)


def _extent(points: np.ndarray) -> float:
    """Length of the points' longest direction (four standard deviations)."""
    return float(4.0 * np.sqrt(max(np.linalg.eigvalsh(np.cov(points.T))[1], 0.0))) if len(points) > 1 else 0.0


def _yaw_from_gps_matches(matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Rigid 2D fit of matched fixes, kept only when the fixes can determine heading."""
    if len(matches) < 2:
        return None
    local = np.array([m['local_xy'] for m in matches], dtype=np.float64)
    target = np.array([m['utm_en'] for m in matches], dtype=np.float64)
    inliers = np.ones(len(matches), dtype=bool)
    for _ in range(3):
        if inliers.sum() < 2:
            return None
        a = local[inliers] - local[inliers].mean(axis=0)
        b = target[inliers] - target[inliers].mean(axis=0)
        dot = float((a * b).sum())
        cross = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
        yaw = wrap_deg(math.degrees(math.atan2(cross, dot)))
        rot = rotation_2d(yaw)
        offset = target[inliers].mean(axis=0) - rot @ local[inliers].mean(axis=0)
        residual = np.linalg.norm(local @ rot.T + offset - target, axis=1)
        limit = max(3.0 * 1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 2.0)
        inliers = residual <= limit
    # The scan says how far the operator really walked; the fixes also spread by their own noise.
    extent = min(_extent(local[inliers]), _extent(target[inliers]))
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    radius = float(np.sqrt(np.mean(((local[inliers] - local[inliers].mean(axis=0)) ** 2).sum(axis=1))))
    noise = max(rms, GPS_RELATIVE_SIGMA_M) / math.sqrt(_independent_fixes(matches, inliers))
    sigma = float(min(math.degrees(noise / max(radius, 0.1)), 180.0))
    return {'yaw_grid_deg': yaw, 'inliers': int(inliers.sum()), 'fixes': len(matches), 'extent_m': extent,
            'rms_m': rms, 'sigma_deg': sigma,
            'qualifies': bool(inliers.sum() >= MIN_GPS_INLIERS and extent >= MIN_GPS_EXTENT_M
                              and sigma <= MAX_GPS_SUPPORT_SIGMA_DEG)}


@dataclass
class EdgeMap:
    """A north-up complex edge field (see edge_orientation) on the projected CRS grid."""
    image: np.ndarray
    east_min: float
    north_max: float
    gsd_m: float
    source: str = ''
    zoom: int = 0
    photo: Optional[np.ndarray] = None

    def to_pixels(self, en: np.ndarray) -> np.ndarray:
        en = np.asarray(en, dtype=np.float64)
        return np.stack([(en[:, 0] - self.east_min) / self.gsd_m, (self.north_max - en[:, 1]) / self.gsd_m], axis=1)


def edge_orientation(gray: np.ndarray) -> np.ndarray:
    """Edge strength and direction as a double-angle complex field, in E/N axes.

    Doubling the angle makes opposite gradients agree, so a wall matches whichever
    side of it is the brighter. Texture has no consistent direction and averages
    to zero, which plain edge density does not: that one prefers busy roofs.
    """
    import cv2
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    grad_e = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    grad_n = -cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    magnitude = np.hypot(grad_e, grad_n)
    scale = float(np.percentile(magnitude, 98))
    if scale <= 0:
        return np.zeros_like(magnitude, dtype=np.complex64)
    # An edge runs perpendicular to its gradient, hence the sign flip on the double angle.
    field = -np.minimum(magnitude / scale, 1.0) * np.exp(2j * np.arctan2(grad_n, grad_e))
    return (cv2.GaussianBlur(field.real.astype(np.float32), (0, 0), 1.5)
            + 1j * cv2.GaussianBlur(field.imag.astype(np.float32), (0, 0), 1.5)).astype(np.complex64)


def _tile_pixel(lat: np.ndarray, lon: np.ndarray, zoom: int) -> Tuple[np.ndarray, np.ndarray]:
    n = 2.0 ** zoom
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
    return x * 256.0, y * 256.0


def _fetch_tile(zoom: int, x: int, y: int, timeout: float) -> Optional[np.ndarray]:
    import cv2
    url = TILE_URL.format(z=zoom, x=x, y=y)
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'LidarCamera360'})
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            data = reply.read()
        tile = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None
    if tile is None:
        return None
    # Missing coverage comes back as a flat grey placard, not as an HTTP error.
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return None if np.median(np.abs(gray - np.median(gray))) < 1.5 else tile


def _build_edge_map(keys: List[Tuple[int, int]], tiles: List[Optional[np.ndarray]],
                    x0: int, y0: int, x1: int, y1: int, px: np.ndarray, py: np.ndarray,
                    east_min: float, north_max: float, zoom: int) -> EdgeMap:
    import cv2
    mh, mw = (y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256
    mosaic = np.zeros((mh, mw, 3), dtype=np.uint8)
    valid = np.zeros((mh, mw), dtype=np.uint8)
    has_missing = False
    for (x, y), tile in zip(keys, tiles):
        top, left = (y - y0) * 256, (x - x0) * 256
        if tile is not None:
            mosaic[top:top + 256, left:left + 256] = tile
            valid[top:top + 256, left:left + 256] = 255
        else:
            has_missing = True
    map_x = (px - x0 * 256).astype(np.float32)
    map_y = (py - y0 * 256).astype(np.float32)
    photo = cv2.remap(mosaic, map_x, map_y, cv2.INTER_LINEAR)
    image = edge_orientation(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY))
    if has_missing:
        # Suppress Sobel artifacts along the boundary of any missing tile.
        eroded = cv2.erode(valid, np.ones((11, 11), dtype=np.uint8))
        vmask = cv2.remap(eroded, map_x, map_y, cv2.INTER_NEAREST) > 0
        image = np.where(vmask, image, 0.0j).astype(np.complex64)
    return EdgeMap(image, east_min, north_max, IMAGERY_GSD_M, IMAGERY_SOURCE, zoom, photo)


def fetch_edge_map(center_en: np.ndarray, half_size_m: float, crs: Any, *, timeout: float = 8.0,
                   log: Callable[[str], None] = print) -> Optional[EdgeMap]:
    """Download aerial tiles around a point and resample them onto the CRS grid.

    Prefers a zoom level with complete tile coverage so a missing tile never introduces
    artificial black-square edges into the correlation field.
    """
    size = int(round(2.0 * half_size_m / IMAGERY_GSD_M))
    east_min = float(center_en[0] - half_size_m)
    north_max = float(center_en[1] + half_size_m)
    cols, rows = np.meshgrid(np.arange(size) + 0.5, np.arange(size) + 0.5)
    to_wgs84 = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True).transform
    lon, lat = to_wgs84(east_min + cols * IMAGERY_GSD_M, north_max - rows * IMAGERY_GSD_M)
    partial_fallback = None
    for zoom in IMAGERY_ZOOMS:
        px, py = _tile_pixel(lat, lon, zoom)
        x0, x1 = int(px.min() // 256), int(px.max() // 256)
        y0, y1 = int(py.min() // 256), int(py.max() // 256)
        keys = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
        with ThreadPoolExecutor(max_workers=6) as pool:
            tiles = list(pool.map(lambda key: _fetch_tile(zoom, key[0], key[1], timeout), keys))
        missing = sum(t is None for t in tiles)
        if missing == 0:
            return _build_edge_map(keys, tiles, x0, y0, x1, y1, px, py, east_min, north_max, zoom)
        log(f'[north] Imagery zoom {zoom}: {missing}/{len(tiles)} tiles unavailable.')
        if missing <= 0.25 * len(tiles) and partial_fallback is None:
            partial_fallback = (keys, tiles, x0, y0, x1, y1, px, py, zoom)
    if partial_fallback is not None:
        keys, tiles, x0, y0, x1, y1, px, py, zoom = partial_fallback
        return _build_edge_map(keys, tiles, x0, y0, x1, y1, px, py, east_min, north_max, zoom)
    return None


class _Correlator:
    """Best plan shift for a set of rotated wall segments, via FFT correlation."""

    def __init__(self, edges: EdgeMap, search_m: float):
        self.edges = edges
        self.spectrum = np.fft.fft2(edges.image)
        h, w = edges.image.shape
        self.rows = np.fft.fftfreq(h, 1.0 / h)
        self.cols = np.fft.fftfreq(w, 1.0 / w)
        limit = search_m / edges.gsd_m
        self.window = (self.rows[:, None] ** 2 + self.cols[None, :] ** 2) <= limit * limit

    def surface(self, en: np.ndarray, angle_rad: np.ndarray) -> Optional[np.ndarray]:
        h, w = self.edges.image.shape
        pixels = np.round(self.edges.to_pixels(en)).astype(np.int64)
        inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
        total = int(inside.sum())
        if total == 0:
            return None
        raster = np.zeros((h, w), dtype=np.complex64)
        raster[pixels[inside, 1], pixels[inside, 0]] = np.exp(2j * angle_rad[inside])
        occupied = int(np.count_nonzero(raster))
        if occupied == 0:
            return None
        correlation = np.fft.ifft2(self.spectrum * np.conj(np.fft.fft2(raster))).real / occupied
        correlation[~self.window] = -np.inf
        return correlation

    def plateau(self, correlation: np.ndarray) -> Dict[str, Any]:
        """Where the score stays near its peak, and the placement there closest to GPS."""
        peak = float(correlation.max())
        threshold = PLATEAU_FRACTION * peak if peak > 0.0 else peak - 1e-9
        rows, cols = np.nonzero(correlation >= threshold)
        if not len(rows):
            r, c = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
            rows, cols = np.array([r]), np.array([c])
        gsd = self.edges.gsd_m
        shifts = np.stack([self.cols[cols] * gsd, -self.rows[rows] * gsd], axis=1)
        nearest = shifts[int(np.argmin(np.hypot(shifts[:, 0], shifts[:, 1])))]
        centred = shifts - shifts.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov(centred.T)) if len(shifts) > 2 else (np.zeros(2), np.eye(2))
        major = vectors[:, 1]
        return {'score': peak, 'shift_en_m': [float(nearest[0]), float(nearest[1])],
                'sigma_along_m': float(np.sqrt(max(values[1], 0.0))),
                'sigma_across_m': float(np.sqrt(max(values[0], 0.0))),
                'along_bearing_deg': float(math.degrees(math.atan2(major[0], major[1])) % 180.0)}


def _translation_stats(correlator: _Correlator, correlation: np.ndarray,
                       shift_en: Sequence[float]) -> Tuple[float, float]:
    """Peak Z-score and rival margin across the translation search window."""
    peak = float(correlation.max())
    gsd = correlator.edges.gsd_m
    de = correlator.cols[None, :] * gsd - float(shift_en[0])
    dn = -correlator.rows[:, None] * gsd - float(shift_en[1])
    other_mask = correlator.window & ((de * de + dn * dn) > PRIOR_TRANS_EXCLUSION_M ** 2)
    others = correlation[other_mask]
    if not len(others):
        return 0.0, 0.0
    med = float(np.median(others))
    mad = 1.4826 * float(np.median(np.abs(others - med)))
    trans_z = (peak - med) / max(mad, 1e-9)
    trans_margin = (peak - float(others.max())) / max(peak - med, 1e-9)
    return float(trans_z), float(trans_margin)


def register_walls(walls_xy: np.ndarray, directions_rad: np.ndarray, anchor_local: np.ndarray, anchor_en: np.ndarray,
                   edges: EdgeMap, *, search_m: float = GPS_SEARCH_M,
                   yaw_prior_deg: Optional[float] = None,
                   yaw_prior_half_deg: Optional[float] = None,
                   yaw_prior_sigma_deg: Optional[float] = None) -> Dict[str, Any]:
    """Scan headings for the placement that lays the walls along imagery edges.

    When ``yaw_prior_deg`` and ``yaw_prior_half_deg`` are provided (e.g. from solar compass
    or a synced GPS track), the search stays inside that angular window with a gentle
    Gaussian heading prior so symmetric street grids or tail noise cannot hijack the match.
    Acceptance still checks the unweighted translation peak sharpness and its margin above
    random full-360 background headings.
    """
    walls = np.asarray(walls_xy, dtype=np.float64) - np.asarray(anchor_local, dtype=np.float64)[:2]
    anchor = np.asarray(anchor_en, dtype=np.float64)[:2]
    correlator = _Correlator(edges, search_m)

    def surface(yaw: float) -> Optional[np.ndarray]:
        return correlator.surface(walls @ rotation_2d(yaw).T + anchor, directions_rad + math.radians(yaw))

    def peak(yaw: float) -> float:
        correlation = surface(yaw)
        return 0.0 if correlation is None else float(correlation.max())

    prior_guided = yaw_prior_deg is not None and yaw_prior_half_deg is not None and yaw_prior_half_deg > 0.0
    if prior_guided:
        half_deg = float(yaw_prior_half_deg)
        prior_scale = max(2.0 * float(yaw_prior_sigma_deg if yaw_prior_sigma_deg is not None else half_deg / 3.0), 5.0)
        offsets = np.arange(-half_deg, half_deg + 1e-6, 0.5)
        headings = (float(yaw_prior_deg) + offsets) % 360.0
        scores = np.array([peak(float(h)) for h in headings])
        weights = np.exp(-0.5 * (offsets / prior_scale) ** 2)
        weighted = np.maximum(scores, 0.0) * weights if float(scores.max()) > 0.0 else scores
        best_off = float(offsets[int(np.argmax(weighted))])
        fine_lo = max(-half_deg, best_off - 1.5)
        fine_hi = min(half_deg, best_off + 1.5)
        fine_offsets = np.arange(fine_lo, fine_hi + 1e-6, 0.25)
        fine_headings = (float(yaw_prior_deg) + fine_offsets) % 360.0
        # Fine local peak refinement stays unweighted so the prior picks the right lobe without biasing wall alignment.
        fine_scores = np.array([peak(float(y)) for y in fine_headings])
        yaw = wrap_deg(float(fine_headings[int(np.argmax(fine_scores))]))
    else:
        headings = np.arange(0.0, 360.0, 1.0)
        scores = np.array([peak(float(y)) for y in headings])
        coarse = float(headings[int(np.argmax(scores))])
        fine = np.arange(coarse - 2.0, coarse + 2.01, 0.25)
        yaw = wrap_deg(float(fine[int(np.argmax([peak(float(y)) for y in fine]))]))

    best_surface = surface(yaw)
    if best_surface is None:
        return {'yaw_grid_deg': yaw, 'score': 0.0, 'shift_en_m': [0.0, 0.0],
                'sigma_along_m': search_m, 'sigma_across_m': search_m, 'along_bearing_deg': 0.0,
                'median_score': 0.0, 'peak_z': 0.0, 'trans_z': 0.0, 'trans_margin': 0.0,
                'rival_yaw_deg': 0.0, 'rival_margin': 0.0,
                'sigma_deg': 180.0, 'offset_from_gps_m': 0.0, 'wall_points': len(walls),
                'prior_guided': prior_guided, 'accepted': False}
    place = correlator.plateau(best_surface)
    score = place['score']
    trans_z, trans_margin = _translation_stats(correlator, best_surface, place['shift_en_m'])
    offset_m = float(math.hypot(*place['shift_en_m']))

    if prior_guided:
        bg_headings = np.arange(0.0, 360.0, 6.0)
        separated = np.array([angle_diff_deg(float(h), yaw) > RIVAL_EXCLUSION_DEG for h in bg_headings])
        bg_scores = np.array([peak(float(h)) if keep else -np.inf for h, keep in zip(bg_headings, separated)])
        others = bg_scores[separated]
        median = float(np.median(others))
        mad = float(np.median(np.abs(others - median)))
        rival = int(np.argmax(bg_scores))
        rival_yaw = float(bg_headings[rival])
        margin = (score - float(bg_scores[rival])) / max(score - median, 1e-9)
        peak_z = (score - median) / max(1.4826 * mad, 1e-9)
        half_height = median + 0.5 * (score - median)
        width_deg = 0.5 * sum(1 for s in scores if s >= half_height)
        at_boundary = angle_diff_deg(yaw, float(yaw_prior_deg)) >= max(float(yaw_prior_half_deg) - 0.5, 0.5)
        at_trans_boundary = offset_m >= max(float(search_m) - 1.0, 1.0)
        accepted = bool(
            score > 0.0
            and trans_z >= MIN_PRIOR_TRANS_Z
            and peak_z >= MIN_PRIOR_PEAK_Z
            and not at_boundary
            and not at_trans_boundary
        )
        return {'yaw_grid_deg': yaw, **place, 'median_score': median, 'peak_z': float(peak_z),
                'trans_z': trans_z, 'trans_margin': trans_margin,
                'rival_yaw_deg': rival_yaw, 'rival_margin': float(margin),
                'sigma_deg': float(max(width_deg, 0.5) / 2.355),
                'offset_from_gps_m': offset_m, 'wall_points': len(walls),
                'prior_guided': True, 'prior_yaw_deg': wrap_deg(float(yaw_prior_deg)),
                'prior_half_deg': float(yaw_prior_half_deg), 'at_boundary': at_boundary,
                'at_trans_boundary': at_trans_boundary,
                'accepted': accepted}

    separated = np.array([angle_diff_deg(float(h), yaw) > RIVAL_EXCLUSION_DEG for h in headings])
    others = scores[separated]
    median = float(np.median(others))
    mad = float(np.median(np.abs(others - median)))
    rival = int(np.argmax(np.where(separated, scores, -np.inf)))
    margin = (score - float(scores[rival])) / max(score - median, 1e-9)
    peak_z = (score - median) / max(1.4826 * mad, 1e-9)
    half_height = median + 0.5 * (score - median)
    width = sum(1 for s, h in zip(scores, headings)
                if s >= half_height and angle_diff_deg(float(h), yaw) <= RIVAL_EXCLUSION_DEG)
    return {'yaw_grid_deg': yaw, **place, 'median_score': median, 'peak_z': float(peak_z),
            'trans_z': trans_z, 'trans_margin': trans_margin,
            'rival_yaw_deg': float(headings[rival]), 'rival_margin': float(margin),
            'sigma_deg': float(max(width, 1) / 2.355),
            'offset_from_gps_m': float(math.hypot(*place['shift_en_m'])), 'wall_points': len(walls),
            'prior_guided': False,
            'accepted': bool(peak_z >= MIN_PEAK_Z and margin >= MIN_RIVAL_MARGIN)}


def write_review_image(path: Path, edges: EdgeMap, walls_xy: np.ndarray, rotation: np.ndarray, translation: np.ndarray,
                       trajectory_xy: Optional[np.ndarray], matches: List[Dict[str, Any]]) -> Optional[str]:
    """Overlay for a person to check the placement against the photo."""
    import cv2
    if edges.photo is None:
        return None
    canvas = cv2.resize(edges.photo, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    def place(xy: np.ndarray) -> np.ndarray:
        return np.round(edges.to_pixels(np.asarray(xy) @ rotation.T + translation) * 2).astype(np.int32)

    for x, y in place(walls_xy):
        if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]:
            canvas[y, x] = (0, 255, 255)
    if trajectory_xy is not None and len(trajectory_xy):
        cv2.polylines(canvas, [place(trajectory_xy).reshape(-1, 1, 2)], False, (255, 160, 0), 1, cv2.LINE_AA)
    for match in matches:
        x, y = np.round(edges.to_pixels(np.array([match['utm_en']])) * 2).astype(int)[0]
        cv2.circle(canvas, (int(x), int(y)), 5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return str(path)


def _matched_anchor(matches: List[Dict[str, Any]], yaw_deg: Optional[float]) -> Dict[str, Any]:
    """Anchor from time-matched fixes; with a heading, outliers are judged against the scan it places.

    Both sides average the same fixes, so the pair holds whatever the heading. A median taken in each
    frame on its own would join two different moments of the walk.
    """
    local = np.array([m['local_xy'] for m in matches], dtype=np.float64)
    target = np.array([m['utm_en'] for m in matches], dtype=np.float64)
    offsets = target if yaw_deg is None else target - local @ rotation_2d(yaw_deg).T
    center = np.median(offsets, axis=0)
    for _ in range(3):
        residual = np.linalg.norm(offsets - center, axis=1)
        middle = float(np.median(residual))
        keep = residual <= max(middle + 3.0 * 1.4826 * float(np.median(np.abs(residual - middle))), 2.0)
        center = offsets[keep].mean(axis=0)
    scatter = float(np.sqrt(np.mean(((offsets[keep] - center) ** 2).sum(axis=1))))
    sigma = max(scatter, GPS_FIX_SIGMA_M) / math.sqrt(_independent_fixes(matches, keep))
    return {'local': local[keep].mean(axis=0), 'en': target[keep].mean(axis=0), 'method': 'time_matched_fixes',
            'fixes': int(keep.sum()), 'sigma_m': float(max(sigma, MIN_POSITION_SIGMA_M))}


def _anchor(gps_info: Dict[str, Any], trajectory: Optional[np.ndarray], local_xy: Optional[np.ndarray],
            matches: List[Dict[str, Any]], yaw_deg: Optional[float] = None,
            nearest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One point known in both frames: where the GPS and the scan agree the rig was."""
    utm = np.asarray(gps_info['utm_coords'], dtype=np.float64)[:, :2]
    if matches:
        return _matched_anchor(matches, yaw_deg)
    if nearest is not None:
        return {'local': nearest['local'], 'en': nearest['en'], 'method': 'nearest_fix_in_time', 'fixes': 1,
                'sigma_m': float(math.hypot(GPS_FIX_SIGMA_M, WALK_DRIFT_MPS * nearest['gap_s']))}
    if trajectory is not None and len(trajectory):
        local = trajectory[:, 1:3]
    elif local_xy is not None and len(local_xy):
        local = np.asarray(local_xy, dtype=np.float64)[:, :2]
    else:
        return {'local': np.zeros(2), 'en': utm.mean(axis=0), 'method': 'gps_mean_only', 'fixes': len(utm),
                'sigma_m': UNTIMED_ANCHOR_SIGMA_M}
    # Untimed: the fixes and the scan cover different spans, so this is a rough anchor.
    return {'local': local.mean(axis=0), 'en': np.median(utm, axis=0), 'method': 'untimed_centroids',
            'fixes': len(utm), 'sigma_m': UNTIMED_ANCHOR_SIGMA_M}


def _plain(anchor: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in anchor.items()}


def _weighted_circular_mean(a_deg: float, a_sigma: float, b_deg: float, b_sigma: float) -> Tuple[float, float]:
    wa = 1.0 / max(float(a_sigma), 0.5) ** 2
    wb = 1.0 / max(float(b_sigma), 0.5) ** 2
    delta = (float(b_deg) - float(a_deg) + 180.0) % 360.0 - 180.0
    return wrap_deg(float(a_deg) + (wb / (wa + wb)) * delta), float(1.0 / math.sqrt(wa + wb))


def _choose_heading_prior(
    solar: Optional[Dict[str, Any]],
    gps_fit: Optional[Dict[str, Any]],
    synced: bool,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[str]]:
    """Select a trustworthy heading prior (yaw, half_window, sigma, source), or Nones."""
    solar_ok = bool(solar and solar.get('qualifies'))
    gps_ok = bool(gps_fit and gps_fit.get('qualifies') and synced)
    gps_alone = gps_ok and float(gps_fit['sigma_deg']) <= MAX_GPS_ALONE_SIGMA_DEG
    if solar_ok and gps_ok:
        tol = max(15.0, 2.5 * math.hypot(float(solar['sigma_deg']), float(gps_fit['sigma_deg'])))
        if angle_diff_deg(solar['yaw_grid_deg'], gps_fit['yaw_grid_deg']) <= tol:
            yaw, sigma = _weighted_circular_mean(solar['yaw_grid_deg'], solar['sigma_deg'],
                                                 gps_fit['yaw_grid_deg'], gps_fit['sigma_deg'])
            return yaw, float(min(18.0, max(8.0, 3.0 * sigma))), sigma, 'solar_and_gps'
        if gps_alone:
            s = float(gps_fit['sigma_deg'])
            return float(gps_fit['yaw_grid_deg']), float(min(18.0, max(8.0, 3.0 * s))), s, 'gps'
        if solar.get('strong'):
            s = float(solar['sigma_deg'])
            return float(solar['yaw_grid_deg']), float(min(18.0, max(10.0, 3.0 * s))), s, 'solar'
        return None, None, None, None
    if gps_alone:
        s = float(gps_fit['sigma_deg'])
        return float(gps_fit['yaw_grid_deg']), float(min(18.0, max(8.0, 3.0 * s))), s, 'gps'
    if solar_ok:
        s = float(solar['sigma_deg'])
        return float(solar['yaw_grid_deg']), float(min(18.0, max(10.0, 3.0 * s))), s, 'solar'
    if gps_ok:
        s = float(gps_fit['sigma_deg'])
        return float(gps_fit['yaw_grid_deg']), float(min(22.0, max(12.0, 2.0 * s))), s, 'gps_weak'
    return None, None, None, None


def solve_north(gps_info: Dict[str, Any], *, points: Optional[np.ndarray] = None,
                trajectory: Optional[np.ndarray] = None, local_xy: Optional[np.ndarray] = None,
                dt_sync: Optional[float] = None, capture_origin_s: Optional[float] = None,
                manual_yaw_deg: Optional[float] = None, street_bearing_deg: Optional[float] = None,
                allow_network: bool = True, edges: Optional[EdgeMap] = None,
                output_dir: Optional[Path] = None,
                colmap_dir: Optional[Path] = None,
                search_paths: Sequence[Any] = (),
                trajectory_path: Optional[Path] = None,
                solar: Optional[Dict[str, Any]] = None,
                log: Callable[[str], None] = print) -> Dict[str, Any]:
    """Combine the capture's evidence into a heading and a position on the grid.

    Without ``dt_sync`` the fixes are paired assuming both recordings started together.
    That is enough to try a heading on a long track, but not to trust it unseen.
    """
    crs = gps_info['crs_info']['horizontal_crs']
    mean = gps_info['mean_wgs84']
    convergence = grid_convergence_deg(mean['latitude_deg'], mean['longitude_deg'], crs)
    synced = dt_sync is not None
    origin = gps_info.get('capture_origin_s', 0.0) if capture_origin_s is None else capture_origin_s
    matches = (match_fixes_to_trajectory(gps_info, trajectory, float(dt_sync or 0.0), origin)
               if trajectory is not None else [])
    gps_fit = _yaw_from_gps_matches(matches)
    nearest = (_nearest_outside_fix(gps_info, trajectory, float(dt_sync), origin)
               if synced and trajectory is not None and not matches else None)

    def place(yaw_deg: Optional[float]) -> Dict[str, Any]:
        return _anchor(gps_info, trajectory, local_xy, matches if synced else [], yaw_deg, nearest)

    if solar is None and manual_yaw_deg is None and (colmap_dir is not None or search_paths or trajectory_path is not None):
        from raven_app.solar_compass import solve_solar_north
        solar = solve_solar_north(
            gps_info, convergence,
            colmap_dir=Path(colmap_dir) if colmap_dir is not None else None,
            search_paths=search_paths,
            trajectory_path=Path(trajectory_path) if trajectory_path is not None else None,
            dt_sync=dt_sync,
            capture_origin_s=origin,
            log=log,
        )

    # Before a heading is chosen, use solar or the track's own fit to reject outlier GPS fixes.
    solar_qualifies = bool(solar and solar.get('qualifies'))
    anchor_yaw = (solar['yaw_grid_deg'] if solar_qualifies
                  else gps_fit['yaw_grid_deg'] if gps_fit and gps_fit['fixes'] >= 3 else None)
    anchor = place(anchor_yaw)
    walls = wall_points(points) if points is not None and len(points) else None
    evidence: Dict[str, Any] = {'anchor': _plain(anchor),
                                'matched_fixes': len(matches), 'time_synchronized': synced, 'gps_fit': gps_fit}
    if solar is not None:
        evidence['solar'] = solar

    prior_yaw, prior_half, prior_sigma, prior_source = _choose_heading_prior(solar, gps_fit, synced)
    sensor_conflict = None
    if solar_qualifies and bool(gps_fit and gps_fit.get('qualifies') and synced):
        conflict_tol = max(15.0, 2.5 * math.hypot(float(solar['sigma_deg']), float(gps_fit['sigma_deg'])))
        diff_deg = angle_diff_deg(solar['yaw_grid_deg'], gps_fit['yaw_grid_deg'])
        if diff_deg > conflict_tol:
            sensor_conflict = {
                'solar_yaw_deg': float(solar['yaw_grid_deg']),
                'gps_yaw_deg': float(gps_fit['yaw_grid_deg']),
                'diff_deg': float(diff_deg),
                'tolerance_deg': float(conflict_tol),
            }
            evidence['sensor_conflict'] = sensor_conflict
    registration = None
    wants_imagery = manual_yaw_deg is None and walls is not None and len(walls) >= 200
    search_m = max(GPS_SEARCH_M, min(2.0 * anchor['sigma_m'], 35.0))
    if wants_imagery and edges is None and allow_network:
        offsets = np.linalg.norm(walls - anchor['local'], axis=1)
        walls = walls[offsets <= min(float(np.percentile(offsets, 99)), MAX_SCENE_RADIUS_M)]
        half = float(np.linalg.norm(walls - anchor['local'], axis=1).max()) + search_m + 5.0
        log(f'[north] Fetching aerial imagery around the GPS anchor ({half:.0f} m radius)...')
        edges = fetch_edge_map(anchor['en'], half, crs, log=log)
        if edges is None:
            evidence['imagery'] = 'unavailable'
            log('[north] Aerial imagery is unavailable here; continuing without it.')
    if wants_imagery and edges is not None:
        straight, directions = wall_directions(walls)
        registration = register_walls(straight, directions, anchor['local'], anchor['en'], edges,
                                      search_m=search_m,
                                      yaw_prior_deg=prior_yaw,
                                      yaw_prior_half_deg=prior_half,
                                      yaw_prior_sigma_deg=prior_sigma)
        registration['source'] = edges.source
        registration['zoom'] = edges.zoom
        if prior_source is not None:
            registration['prior_source'] = prior_source
        evidence['imagery'] = registration
        mode_tag = f'prior={prior_source} {prior_yaw:.1f}+/-{prior_half:.0f} deg, ' if prior_source else ''
        log(f"[north] Imagery registration ({mode_tag}zoom {edges.zoom}): yaw {registration['yaw_grid_deg']:.2f} deg, "
            f"peak z {registration['peak_z']:.1f}, trans z {registration.get('trans_z', 0.0):.1f}, "
            f"{registration['offset_from_gps_m']:.1f} m from GPS, "
            f"position +/-{registration['sigma_across_m']:.1f} m across and "
            f"+/-{registration['sigma_along_m']:.1f} m along bearing {registration['along_bearing_deg']:.0f}, "
            f"{'accepted' if registration['accepted'] else 'rejected'}.")

    gps_qualifies = bool(gps_fit and gps_fit['qualifies'])
    gps_alone = gps_qualifies and gps_fit['sigma_deg'] <= MAX_GPS_ALONE_SIGMA_DEG
    coarse_reliable = False
    yaw, method, sigma = None, None, None
    translation = None
    if manual_yaw_deg is not None:
        yaw, method, sigma = wrap_deg(manual_yaw_deg), 'manual_user_specified', None
    if yaw is None and street_bearing_deg is not None and walls is not None and len(walls) >= 200:
        track_reliable = gps_qualifies and synced
        imagery_reliable = bool(registration and registration['accepted'])
        coarse_reliable = track_reliable or imagery_reliable or solar_qualifies
        coarse = (registration['yaw_grid_deg'] if registration and imagery_reliable
                  else solar['yaw_grid_deg'] if solar_qualifies
                  else registration['yaw_grid_deg'] if registration and not track_reliable
                  else gps_fit['yaw_grid_deg'] if gps_fit else None)
        axes = wall_axes(walls)
        evidence['wall_axes'] = axes
        if coarse is not None and axes:
            grid_bearing = street_bearing_deg - convergence
            candidates = [wrap_deg(90.0 - grid_bearing - axis['axis_deg'] + k * 90.0)
                          for axis in axes[:2] for k in range(4)]
            yaw = min(candidates, key=lambda c: angle_diff_deg(c, coarse))
            method, sigma = 'street_bearing_and_wall_axis', 1.0
        else:
            evidence['street_bearing'] = 'needs a coarse heading to pick the wall'
    if yaw is None and registration and registration['accepted'] and registration.get('prior_guided'):
        yaw, sigma = registration['yaw_grid_deg'], registration['sigma_deg']
        method = ('solar_gps_and_aerial_imagery' if prior_source == 'solar_and_gps'
                  else 'solar_and_aerial_imagery' if prior_source == 'solar'
                  else 'gps_and_aerial_imagery')
        placed = anchor['en'] + np.asarray(registration['shift_en_m'], dtype=np.float64)
        translation = placed - rotation_2d(yaw) @ anchor['local']
        evidence['position'] = 'aerial_imagery_registration'
    if yaw is None and gps_alone and synced:
        yaw, method, sigma = gps_fit['yaw_grid_deg'], 'gps_time_matched_rigid_fit', gps_fit['sigma_deg']
    if yaw is None and solar_qualifies:
        yaw, method, sigma = solar['yaw_grid_deg'], 'solar_compass', solar['sigma_deg']
    if registration and registration['accepted']:
        if yaw is None:
            yaw, method, sigma = registration['yaw_grid_deg'], 'aerial_imagery_registration', registration['sigma_deg']
        if translation is None and angle_diff_deg(yaw, registration['yaw_grid_deg']) <= 5.0:
            # The registration's position only holds at its own heading: re-rotate about the anchor.
            placed = anchor['en'] + np.asarray(registration['shift_en_m'], dtype=np.float64)
            translation = placed - rotation_2d(yaw) @ anchor['local']
            evidence['position'] = 'aerial_imagery_registration'
    if yaw is None and gps_qualifies:
        method = 'gps_time_matched_rigid_fit' if synced else 'gps_assumed_sync_rigid_fit'
        yaw, sigma = gps_fit['yaw_grid_deg'], gps_fit['sigma_deg']

    status = 'resolved' if yaw is not None else 'unresolved'
    if method == 'gps_assumed_sync_rigid_fit':
        status = 'needs_review'
    elif method == 'gps_time_matched_rigid_fit' and not gps_alone:
        status = 'needs_review'
        log(f'[north] The GPS track is too short for a firm heading (+/-{sigma:.0f} deg). Check it on the map, or '
            'confirm it with a street bearing measured on a map.')
    elif method == 'street_bearing_and_wall_axis' and not coarse_reliable:
        status = 'needs_review'
        log('[north] The street bearing was matched to a wall, but nothing says firmly which of the four ways '
            'round the scan is. Check it on the map.')
    elif method == 'solar_compass':
        confirmed_by_gps = bool(
            gps_qualifies and synced
            and angle_diff_deg(yaw, gps_fit['yaw_grid_deg']) <= max(12.0, 2.5 * math.hypot(sigma, gps_fit['sigma_deg']))
        )
        evidence['confirmed_by_gps'] = confirmed_by_gps
        if not (solar.get('strong') or confirmed_by_gps):
            status = 'needs_review'
            log(f'[north] Solar compass heading (+/-{sigma:.1f} deg) has no imagery or GPS confirmation. '
                'Check it on the map.')
    elif method == 'solar_and_aerial_imagery' and not solar.get('strong') and registration['peak_z'] < 4.0:
        status = 'needs_review'
        log('[north] Heading rests on a weak solar compass match and moderate imagery peak. Check north_alignment_review.jpg.')
    elif method == 'gps_and_aerial_imagery' and prior_source == 'gps_weak' and registration['peak_z'] < 4.0:
        status = 'needs_review'
        log('[north] Heading rests on a short GPS track and moderate imagery peak. Check north_alignment_review.jpg.')
    elif method == 'aerial_imagery_registration':
        confirmed = gps_qualifies and angle_diff_deg(yaw, gps_fit['yaw_grid_deg']) <= GPS_AGREEMENT_DEG
        evidence['confirmed_by_gps'] = confirmed
        if not confirmed:
            status = 'needs_review'
            log('[north] This heading rests on aerial imagery alone. Check north_alignment_review.jpg, or confirm it '
                'with a street bearing measured on a map.')
    if sensor_conflict is not None and status == 'resolved' and method not in ('manual_user_specified', 'street_bearing_and_wall_axis'):
        status = 'needs_review'
        log(f"[north] Solar compass ({sensor_conflict['solar_yaw_deg']:.1f} deg) and synced GPS "
            f"({sensor_conflict['gps_yaw_deg']:.1f} deg) disagree by {sensor_conflict['diff_deg']:.1f} deg. "
            'Check the alignment on the map before trusting it.')
    if yaw is None:
        yaw, method = 0.0, 'unresolved'
        log('[north] North could not be determined; the scan is placed on the GPS anchor without a heading.')
    if translation is None:
        if method != 'unresolved':
            anchor = place(yaw)
            evidence['anchor'] = _plain(anchor)
        translation = anchor['en'] - rotation_2d(yaw) @ anchor['local']
        evidence['position'] = anchor['method']

    # Along a straight street the imagery has no opinion, so GPS still bounds that direction.
    # When imagery locks both orthogonal axes tightly (<= 2.5 m), report the imagery precision.
    position_sigma = anchor['sigma_m']
    if evidence['position'] != anchor['method']:
        img_sigma = max(registration['sigma_along_m'], registration['sigma_across_m'])
        if img_sigma <= 2.5:
            position_sigma = max(1.0, img_sigma)
        else:
            position_sigma = max(position_sigma, img_sigma)
    result = {'status': status, 'method': method, 'yaw_grid_deg': float(yaw), 'yaw_sigma_deg': sigma,
              'translation_en': [float(translation[0]), float(translation[1])],
              'position_sigma_m': float(position_sigma),
              'grid_convergence_deg': convergence, 'evidence': evidence}
    if output_dir is not None and edges is not None and walls is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        review = write_review_image(Path(output_dir) / 'north_alignment_review.jpg', edges, walls, rotation_2d(yaw),
                                    np.asarray(translation), trajectory[:, 1:3] if trajectory is not None else None,
                                    matches)
        if review:
            result['review_image'] = review
    return result
