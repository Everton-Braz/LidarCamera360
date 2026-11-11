"""Remove moving and standing people (and other transient objects) from the SLAM map.

Two independent tests, both applied before colorization so every exported cloud
(deliverables and the 3DGS seed built from them) is already clean:

* Ray carving (``carve_dynamic``): the accumulated map holds every scan, so a
  person walking by leaves a trail of points in mid-air.  A voxel is dynamic when
  other scans saw *through* it: their rays cross it, at times well away from the
  voxel's own hits, and end behind it.  Needs only the SLAM output.
* Person-mask voting (``person_vote_keep``): someone standing still is never seen
  through, so carving keeps them.  The RF-DETR person masks mark them in the
  photos; points that most views (with a z-buffer occlusion test) show as
  "person" are removed, except points on locally planar surfaces.

Time gating (``time_gating``) stays complementary: it limits which points each
view colours, while this module decides which points exist at all.
"""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

FILTER_DIR = 'dynamic_filter'
KEEP_FILE = 'static_keep.npy'
REPORT_FILE = 'report.json'
SCHEMA = 1


@dataclass(frozen=True)
class CarveParams:
    voxel: float = 0.05          # voxel size (m)
    margin: float = 0.15         # stop marching this far before each endpoint (m)
    window: float = 2.0          # ignore passes within this many seconds of a hit of the voxel
    min_free: int = 2            # minimum free passes for a dynamic voxel
    ratio: float = 0.25          # minimum free passes per hit
    grazing: float = 0.3         # ignore rays with |n.d| below this on planar voxels
    normal_cell: float = 0.15    # cell size of the plane fit (m)
    planarity: float = 0.1       # planar when the smallest eigenvalue < planarity * trace
    max_pass_range: float = 10.0  # only count passes this close to the sensor (m, 0 = any)
    max_height: float = 2.5      # only carve voxels up to this height above the local floor (m, 0 = any)
    max_span: float = 20.0       # only carve voxels whose hits span at most this (s, 0 = any)
    floor_cell: float = 1.0      # column size of the local floor estimate (m)
    coarse: int = 4              # coarse occupancy cell, in voxels, used to skip free space
    batch_samples: int = 12_000_000

    def validate(self):
        for name in ('voxel', 'normal_cell', 'floor_cell'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'Dynamic filter {name} must be positive')
        for name in ('margin', 'window', 'ratio', 'grazing', 'planarity',
                     'max_pass_range', 'max_height', 'max_span'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'Dynamic filter {name} must be zero or positive')
        if self.min_free < 1 or self.coarse < 1 or self.batch_samples < 1:
            raise ValueError('Dynamic filter min_free, coarse and batch_samples must be at least 1')
        return self


def _group_reduce(sorted_keys):
    """Start index of each run of equal values in a sorted array."""
    if not len(sorted_keys):
        return np.zeros(0, dtype=np.int64)
    return np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])


def _cell_planes(points, cell, planarity):
    """Normal and planarity of the cell (size ``cell``) holding each point."""
    ijk = np.floor((points - points.min(0)) / cell).astype(np.int64)
    key = (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]
    _, cid, count = np.unique(key, return_inverse=True, return_counts=True)
    nc = len(count)
    mean = np.stack([np.bincount(cid, points[:, a], nc) for a in range(3)], 1) / count[:, None]
    dp = points - mean[cid]
    cov = np.empty((nc, 3, 3))
    for a in range(3):
        for b in range(a, 3):
            cov[:, a, b] = cov[:, b, a] = np.bincount(cid, dp[:, a] * dp[:, b], nc) / count
    evals, evecs = np.linalg.eigh(cov)
    planar = (evals[:, 0] < planarity * np.maximum(evals.sum(1), 1e-12)) & (count >= 8)
    return cid, evecs[:, :, 0].astype(np.float32), planar


def _local_height(points, cell):
    """Height of each point above its column's floor (2nd-percentile z)."""
    col = np.floor((points[:, :2] - points[:, :2].min(0)) / cell).astype(np.int64)
    _, cinv, ccnt = np.unique((col[:, 0] << 32) | col[:, 1], return_inverse=True, return_counts=True)
    order = np.lexsort((points[:, 2], cinv))
    cstart = np.cumsum(ccnt) - ccnt
    floor = points[order[cstart + (ccnt * 0.02).astype(np.int64)], 2]
    return points[:, 2] - floor[cinv]


def carve_dynamic(points, scan_index, scan_times, scan_origins, params=None, log=print):
    """Return a per-point bool mask of points in dynamic voxels, and a stats dict.

    ``points`` (N, 3) world-frame returns; ``scan_index`` (N,) scan of each point;
    ``scan_times`` (S,) seconds; ``scan_origins`` (S, 3) sensor position per scan.
    """
    p = (params or CarveParams()).validate()
    started = time.perf_counter()
    pts = np.ascontiguousarray(points, dtype=np.float32)
    scan_index = np.asarray(scan_index, dtype=np.int64)
    times = np.asarray(scan_times, dtype=np.float64)
    times = (times - times.min()).astype(np.float32) if len(times) else times.astype(np.float32)
    origins = np.asarray(scan_origins, dtype=np.float32)
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=bool), {'points': 0, 'dynamic_points': 0}
    v = float(p.voxel)
    lo = np.minimum(pts.min(0), origins.min(0)) - 2 * v
    hi = np.maximum(pts.max(0), origins.max(0)) + 2 * v
    dims = (np.ceil((hi - lo) / v).astype(np.int64) + 1)

    def keys_of(q):
        ijk = np.floor((q - lo) / v).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < dims)).all(1)
        np.maximum(ijk, 0, out=ijk)
        return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2], inside, ijk

    # Occupied voxels: hits, first/last hit time, and the plane of their cell.
    pkey, _, pijk = keys_of(pts)
    vkeys, vid, hits = np.unique(pkey, return_inverse=True, return_counts=True)
    nv = len(vkeys)
    pt_time = times[scan_index]
    order = np.lexsort((pt_time, vid))
    starts = _group_reduce(vid[order])
    tmin = pt_time[order[starts]]
    tmax = pt_time[order[np.r_[starts[1:], n] - 1]]
    cid, cnormal, cplanar = _cell_planes(pts.astype(np.float64), p.normal_cell, p.planarity)
    vcell = np.zeros(nv, dtype=np.int64)
    vcell[vid] = cid
    normal, planar = cnormal[vcell], cplanar[vcell]
    # Coarse occupancy bitmap: almost every ray sample lies in empty space, and
    # one bit test per sample skips the exact voxel lookup for those.
    c = int(p.coarse)
    cdims = dims // c + 1
    ncells = int(cdims.prod())
    while ncells > 8_000_000_000:  # keep the bitmap under 1 GB
        c *= 2
        cdims = dims // c + 1
        ncells = int(cdims.prod())
    cijk = pijk // c
    bits = np.zeros((ncells + 7) // 8, dtype=np.uint8)
    ckey = np.unique((cijk[:, 0] * cdims[1] + cijk[:, 1]) * cdims[2] + cijk[:, 2])
    np.bitwise_or.at(bits, ckey >> 3, (1 << (ckey & 7)).astype(np.uint8))
    del pkey, pijk, cijk, ckey, order, cid
    log(f'[*] Dynamic filter: {nv:,} occupied {v * 100:.0f} cm voxels, {planar.mean():.0%} planar')

    # March every ray from its sensor origin and count free passes.
    d = pts - origins[scan_index]
    rng = np.linalg.norm(d, axis=1)
    d /= np.maximum(rng, 1e-9)[:, None]
    nsamp = np.floor((rng - p.margin) / v)
    if p.max_pass_range > 0:
        # Far rays carry more angular pose error and clip thin static
        # structures (scaffolding, fences, hedges) seen from afar.
        nsamp = np.minimum(nsamp, math.floor(p.max_pass_range / v + 0.5))
    nsamp = np.maximum(nsamp, 0).astype(np.int64)
    csum = np.cumsum(nsamp)
    total = int(csum[-1])
    free = np.zeros(nv, dtype=np.int64)
    start = 0
    while start < n:
        base = int(csum[start - 1]) if start else 0
        end = int(np.searchsorted(csum, base + p.batch_samples, side='right'))
        end = min(max(end, start + 1), n)
        ns = nsamp[start:end]
        m = int(ns.sum())
        if m:
            ray = np.repeat(np.arange(start, end), ns)
            first = np.cumsum(ns) - ns
            k = np.arange(m, dtype=np.int64) - np.repeat(first, ns)
            dist = (k.astype(np.float32) + 0.5) * v
            del k
            pos = origins[scan_index[ray]] + d[ray] * dist[:, None]
            ijk = np.floor((pos - lo) / v).astype(np.int64)
            inside = ((ijk >= 0) & (ijk < dims)).all(1)
            np.maximum(ijk, 0, out=ijk)
            cc = ijk // c
            ck = (cc[:, 0] * cdims[1] + cc[:, 1]) * cdims[2] + cc[:, 2]
            sel = inside & ((bits[np.minimum(ck >> 3, len(bits) - 1)] >> (ck & 7)) & 1).astype(bool)
            del cc, ck, pos
            ray, ijk = ray[sel], ijk[sel]
            key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
            idx = np.minimum(np.searchsorted(vkeys, key), nv - 1)
            hit = vkeys[idx] == key
            ray, idx = ray[hit], idx[hit]
            t = times[scan_index[ray]]
            away = (t < tmin[idx] - p.window) | (t > tmax[idx] + p.window)
            steep = ~planar[idx] | (np.abs((normal[idx] * d[ray]).sum(1)) >= p.grazing)
            ok = away & steep
            free += np.bincount(idx[ok], minlength=nv)
        start = end
    del d, rng
    dynamic = (free >= p.min_free) & (free >= p.ratio * hits)
    if p.max_span > 0:
        # A passer-by is hit for a few seconds; porous static structures
        # (hedges, mesh) are hit across most of the capture.
        dynamic &= tmax - tmin <= p.max_span
    if p.max_height > 0:
        vh = np.bincount(vid, _local_height(pts.astype(np.float64), p.floor_cell), nv) / hits
        dynamic &= vh <= p.max_height
    point_dynamic = dynamic[vid]
    stats = {'points': n, 'occupied_voxels': nv, 'dynamic_voxels': int(dynamic.sum()),
             'dynamic_points': int(point_dynamic.sum()), 'samples': total,
             'seconds': round(time.perf_counter() - started, 1)}
    log(f'[*] Dynamic filter: {stats["dynamic_voxels"]:,} moving-object voxels, '
        f'{stats["dynamic_points"]:,}/{n:,} points ({stats["dynamic_points"] / n:.2%}), '
        f'{total:,} ray samples in {stats["seconds"]:.0f} s')
    return point_dynamic, stats


def _signature(pcd_path):
    stat = Path(pcd_path).stat()
    return {'pcd': Path(pcd_path).name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _scan_origins(slam_dir, scan_stamps, trajectory_path):
    """Sensor position at each scan time, linearly interpolated on the trajectory."""
    traj = np.loadtxt(trajectory_path, ndmin=2)
    t = traj[:, 0]
    gap = np.abs(np.clip(scan_stamps, t[0], t[-1]) - scan_stamps).max() if len(scan_stamps) else 0.0
    if gap > 0.5:
        raise ValueError(f'SLAM scans extend {gap:.2f} s beyond the trajectory in {slam_dir}')
    return np.stack([np.interp(scan_stamps, t, traj[:, 1 + a]) for a in range(3)], 1)


def run_dynamic_filter(dataset_dir, params=None, force=False, log=print):
    """Carve the SLAM map of ``dataset_dir`` and cache a per-raw-point keep mask.

    The mask lives in ``<dataset>/dynamic_filter/static_keep.npy`` (one bool per
    point of ``all_raw_points.pcd``, NaN rows included) next to ``report.json``.
    It is reused while the PCD and the parameters are unchanged.
    """
    from raven_app.slam_source import active_slam_dir
    from scripts.pipeline_auto_calibrator_and_colorizer import get_slam_trajectory_path, load_pcd
    p = (params or CarveParams()).validate()
    dataset_dir = Path(dataset_dir)
    slam_dir = active_slam_dir(dataset_dir)
    pcd = slam_dir / 'pcd' / 'all_raw_points.pcd'
    out = dataset_dir / FILTER_DIR
    report_path, keep_path = out / REPORT_FILE, out / KEEP_FILE
    settings = {'schema': SCHEMA, 'source': _signature(pcd), 'params': asdict(p)}
    if not force and report_path.is_file() and keep_path.is_file():
        try:
            previous = json.loads(report_path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            previous = {}
        if all(previous.get(k) == v for k, v in settings.items()):
            log(f'[*] Reusing moving-object filter: {previous.get("dynamic_points", 0):,} points removed')
            return keep_path
    table = slam_dir / 'scan_ranges.csv'
    if not table.is_file():
        raise FileNotFoundError(f'Moving-object removal needs {table.name} in {slam_dir}')
    rows = np.loadtxt(table, delimiter=',', skiprows=1, ndmin=2)
    counts = rows[:, 2].astype(np.int64)
    pts, finite = load_pcd(pcd, return_mask=True)
    if counts.sum() != len(finite) or (len(rows) and rows[0, 1] != 0):
        raise ValueError(f'{table.name} lists {int(counts.sum()):,} points, {pcd.name} has {len(finite):,}')
    scan_index = np.repeat(np.arange(len(rows)), counts)[finite]
    origins = _scan_origins(slam_dir, rows[:, 0], get_slam_trajectory_path(dataset_dir))
    dynamic, stats = carve_dynamic(pts, scan_index, rows[:, 0], origins, p, log=log)
    keep = np.ones(len(finite), dtype=bool)
    keep[np.flatnonzero(finite)[dynamic]] = False
    out.mkdir(parents=True, exist_ok=True)
    report_path.unlink(missing_ok=True)
    np.save(keep_path, keep)
    settings.update(stats)
    temporary = report_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    temporary.replace(report_path)
    return keep_path


def load_static_keep(dataset_dir, finite_mask, log=print):
    """Keep mask over the finite points ``load_pcd`` returned, or None if unavailable.

    Refuses (returns None with a notice) a mask computed for another PCD.
    """
    dataset_dir = Path(dataset_dir)
    keep_path = dataset_dir / FILTER_DIR / KEEP_FILE
    report_path = dataset_dir / FILTER_DIR / REPORT_FILE
    if not keep_path.is_file() or not report_path.is_file():
        log('[!] Moving-object removal requested but no filter result exists; run the dynamic filter first')
        return None
    from raven_app.slam_source import active_slam_dir
    report = json.loads(report_path.read_text(encoding='utf-8'))
    pcd = active_slam_dir(dataset_dir) / 'pcd' / 'all_raw_points.pcd'
    keep = np.load(keep_path)
    if report.get('source') != _signature(pcd) or len(keep) != len(finite_mask):
        log('[!] Moving-object filter result belongs to another SLAM cloud; ignoring it')
        return None
    keep = keep[np.asarray(finite_mask, dtype=bool)]
    log(f'[*] Moving-object removal: dropping {int((~keep).sum()):,}/{len(keep):,} SLAM points')
    return keep


def removal_fringe_keep(points, removed, radius=0.3, min_removed=3, ratio=2.0, log=print):
    """False for kept points left stranded by a removal.

    A few points of a removed person or car survive every filter. Alone in
    the air they look like sparse distant surface: a splat exporter that sizes
    its tolerance from the local LiDAR spacing then keeps floaters around them.
    A kept point is dropped when at least ``min_removed`` removed points, and
    ``ratio`` times as many removed as kept ones, lie within ``radius``.
    """
    from scipy.spatial import cKDTree
    keep = np.ones(len(points), bool)
    if len(removed) < min_removed or not len(points):
        return keep
    removed_tree = cKDTree(removed)
    near = np.flatnonzero(np.isfinite(removed_tree.query(points, distance_upper_bound=radius, workers=-1)[0]))
    if len(near):
        n_removed = removed_tree.query_ball_point(points[near], radius, return_length=True, workers=-1)
        n_kept = cKDTree(points).query_ball_point(points[near], radius, return_length=True, workers=-1) - 1
        keep[near[(n_removed >= min_removed) & (n_removed > ratio * n_kept)]] = False
    log(f'[*] Dropped {int((~keep).sum()):,} points stranded by the removals')
    return keep


def _structure_points(points, candidates, cell=0.4, planarity=0.03, residual=0.03, floor=0.05):
    """True for candidates on the local floor or on a plane spanning a ``cell``.

    The 10 cm test of ``protect_planar`` also accepts a torso, so standing
    people are judged at a scale where a body is not planar but a wall,
    door or table top is.
    """
    cid, normal, planar = _cell_planes(points, cell, planarity)
    cand = points[candidates]
    count = np.bincount(cid)
    center = np.stack([np.bincount(cid, points[:, a]) for a in range(3)], 1) / count[:, None]
    c = cid[candidates]
    on_plane = planar[c] & (np.abs(((cand - center[c]) * normal[c]).sum(1)) < residual)
    return on_plane | (_local_height(points, 1.0)[candidates] < floor)


def _wide_groups(points, link, max_extent):
    """True for points whose ``link``-connected group spans more than ``max_extent`` in plan."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    n = len(points)
    pairs = cKDTree(points).query_pairs(link, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    groups, label = connected_components(graph, directed=False)
    lo = np.full((groups, 2), np.inf)
    hi = np.full((groups, 2), -np.inf)
    np.minimum.at(lo, label, points[:, :2])
    np.maximum.at(hi, label, points[:, :2])
    return (np.linalg.norm(hi - lo, axis=1) > max_extent)[label]


def person_vote_keep(points, views, images_dir, masks_dir, project, max_range=12.0,
                     raster_scale=0.25, occlusion_margin=0.4, persistent_fraction=0.25,
                     max_person_height=2.2, max_person_extent=2.0, log=print):
    """Remove points that most unoccluded views label as person (standing people).

    ``views`` is a list of ``(image_path, R_cw, C, params)`` in the LiDAR frame,
    the same tuples ``masked_operator_keep`` takes; ``project(P_cam, params)``
    returns ``(u, v, r)``.  Each view considers points within ``max_range`` of
    its centre and votes only for points that win a coarse z-buffer, so a wall
    behind a person is not blamed for the person's silhouette.  At least two
    person votes and a strict majority are required; floors and large planes
    are kept.  Pixels masked in at least ``persistent_fraction`` of a camera's
    frames (the operator, fixed exclusions) neither vote nor count as seen.
    Points higher than ``max_person_height`` above the local floor are kept, and
    so are connected groups whose footprint diagonal exceeds ``max_person_extent``
    (a facade masked behind a passer-by the carving already removed).
    """
    if not math.isfinite(max_range) or max_range <= 0:
        raise ValueError('Person vote range must be finite and positive')
    if not math.isfinite(occlusion_margin) or occlusion_margin < 0:
        raise ValueError('Occlusion margin must be finite and non-negative')
    if not math.isfinite(max_person_height) or max_person_height <= 0:
        raise ValueError('Maximum person height must be finite and positive')
    if not math.isfinite(max_person_extent) or max_person_extent <= 0:
        raise ValueError('Maximum person extent must be finite and positive')
    if not 0 < persistent_fraction <= 1:
        raise ValueError('Persistent mask fraction must be in (0, 1]')
    import cv2
    from scipy.ndimage import minimum_filter
    from raven_app.person_masks import fixed_keep_samples, load_keep_mask, mask_evidence
    started = time.perf_counter()
    points = np.asarray(points, dtype=np.float64)
    evidence_dir, config = mask_evidence(masks_dir)

    # Pass 1: person masks at z-buffer resolution, and how often each pixel is
    # masked per camera.  The operator (and fixed exclusions) sit in the same
    # pixels of nearly every frame; whatever is behind them is not a person.
    images_root = Path(images_dir).resolve()
    reduced, frequency, frames = [], {}, {}
    for image, *_ in views:
        mask = load_keep_mask(image, images_dir, evidence_dir)
        h, w = mask.shape
        zw, zh = max(1, int(w * raster_scale)), max(1, int(h * raster_scale))
        masked = cv2.resize((mask < 255).astype(np.uint8), (zw, zh), interpolation=cv2.INTER_AREA) > 0
        camera_name = Path(image).resolve().relative_to(images_root).parts[0]
        if camera_name not in frequency:
            frequency[camera_name], frames[camera_name] = np.zeros((zh, zw), np.uint32), 0
        if frequency[camera_name].shape == masked.shape:
            frequency[camera_name] += masked
            frames[camera_name] += 1
        reduced.append((camera_name, (h, w), np.packbits(masked.ravel())))
    persistent = {name: (count >= max(2, persistent_fraction * frames[name])).ravel()
                  for name, count in frequency.items()}
    log(f'[*] Persistent mask area (operator, fixed regions): '
        + ', '.join(f'{name} {100 * p.mean():.1f}%' for name, p in sorted(persistent.items())))

    person = np.zeros(len(points), dtype=np.uint16)
    observed = np.zeros(len(points), dtype=np.uint16)
    pts32 = points.astype(np.float32)
    for (image, rotation, center, params), (camera_name, (h, w), packed) in zip(views, reduced):
        center = np.asarray(center, dtype=np.float32)
        near = np.flatnonzero((np.square(pts32 - center).sum(1)) <= max_range * max_range)
        if not len(near):
            continue
        cam = (np.asarray(rotation, dtype=np.float64) @ (points[near] - center).T).T
        front = cam[:, 2] > .15
        near, cam = near[front], cam[front]
        if not len(near):
            continue
        u, v, _ = project(cam, params)
        valid = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        if params[-1] >= -997.0:
            valid &= np.hypot(u - params[2], v - params[3]) < 1620.0 * min(h, w) / 3840.0
        if config is not None:
            valid &= fixed_keep_samples((h, w), u, v, camera_name, config)
        near, u, v, dist = near[valid], u[valid], v[valid], np.linalg.norm(cam[valid], axis=1)
        if not len(near):
            continue
        zw, zh = max(1, int(w * raster_scale)), max(1, int(h * raster_scale))
        flat = (np.minimum((v * zh / h).astype(np.int64), zh - 1) * zw
                + np.minimum((u * zw / w).astype(np.int64), zw - 1))
        depth = np.full(zw * zh, np.inf, dtype=np.float32)
        np.minimum.at(depth, flat, dist.astype(np.float32))
        nearest = minimum_filter(depth.reshape(zh, zw), size=3, mode='constant', cval=np.inf).ravel()[flat]
        # A person's flanks lie up to a body depth behind the front-most point
        # of the same raster neighbourhood; what is farther is occluded.
        visible = dist <= nearest + np.maximum(occlusion_margin, 0.03 * nearest)
        near, flat = near[visible], flat[visible]
        stable = persistent[camera_name]
        if len(stable) == zw * zh:
            usable = ~stable[flat]
            near, flat = near[usable], flat[usable]
        masked = np.unpackbits(packed, count=zw * zh).astype(bool)
        observed[near] += 1
        person[near[masked[flat]]] += 1
    remove = (person >= 2) & (person.astype(np.uint32) * 2 > observed)
    candidates = np.flatnonzero(remove)
    if len(candidates):
        # A standing person is within reach of the floor.  Facades and ceilings
        # behind a person that the carving already removed are not.
        high = _local_height(points, 1.0)[candidates] > max_person_height
        remove[candidates[high]] = False
        candidates = candidates[~high]
        log(f'[*] Kept {int(high.sum()):,} person-voted points more than {max_person_height:g} m above the floor')
    if len(candidates):
        wide = _wide_groups(points[candidates], link=0.15, max_extent=max_person_extent)
        remove[candidates[wide]] = False
        candidates = candidates[~wide]
        log(f'[*] Kept {int(wide.sum()):,} person-voted points in groups wider than {max_person_extent:g} m')
    if len(candidates):
        planar = _structure_points(points, candidates)
        remove[candidates[planar]] = False
        log(f'[*] Protected {int(planar.sum()):,} person-voted points on floors and large planes')
    log(f'[*] Standing-person removal ({len(views)} views, {max_range:g} m): '
        f'{int(remove.sum()):,}/{len(points):,} points in {time.perf_counter() - started:.0f} s')
    if remove.all():
        raise ValueError('Person masks would remove the entire cloud')
    return ~remove
