"""Headless combined foreground masks: 255 keeps a pixel, 0 excludes it.

Camera-relative names retain the source extension (cam0/frame.jpg.png), so
the same files can be consumed by COLMAP/Spirula without collisions.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

from raven_app.config import get_app_root, get_config_dir
from raven_app.mask_resources import (MODEL_SHA256, download_progress, ensure_model,
                                      ensure_runtime, masker_environment,
                                      resolve_masker_executable, resolve_model)


def mask_path(image_path, images_dir, masks_dir):
    relative = Path(image_path).resolve().relative_to(Path(images_dir).resolve())
    return Path(masks_dir) / (relative.as_posix() + '.png')


def load_keep_mask(image_path, images_dir, masks_dir):
    path = mask_path(image_path, images_dir, masks_dir)
    if not path.is_file():
        raise ValueError(f'Missing person mask: {path}')
    mask = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Invalid person mask: {path}')
    return mask


def keep_samples(mask, u, v):
    """Reject if ANY of the four bilinear color taps overlaps a person."""
    x, y = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    valid = (x >= 0) & (y >= 0) & (x + 1 < mask.shape[1]) & (y + 1 < mask.shape[0])
    result = np.zeros(len(x), dtype=bool)
    xv, yv = x[valid], y[valid]
    result[valid] = ((mask[yv, xv] == 255) & (mask[yv, xv + 1] == 255)
                     & (mask[yv + 1, xv] == 255) & (mask[yv + 1, xv + 1] == 255))
    return result


DEFAULT_MASK_CONFIG = {'schema': 1, 'fisheye_border_percent': 0.0,
                       'rectangles': {}, 'ellipses': {}, 'polygons': {}}


def normalize_mask_config(config=None):
    """Validate portable, normalized per-camera mask settings."""
    config = DEFAULT_MASK_CONFIG if config is None else config
    if not isinstance(config, dict) or set(config) - {'schema', 'fisheye_border_percent', 'rectangles', 'ellipses', 'polygons'}:
        raise ValueError('Mask settings must contain only schema, border percentage and per-camera shapes')
    if config.get('schema', 1) != 1:
        raise ValueError('Unsupported mask settings schema')
    border = config.get('fisheye_border_percent', 0.0)
    if not isinstance(border, (int, float)) or not math.isfinite(border) or not 0 <= border <= 25:
        raise ValueError('Fisheye border cutoff must be between 0 and 25 percent')
    result = {'schema': 1, 'fisheye_border_percent': float(border)}
    for shape in ('rectangles', 'ellipses'):
        groups = config.get(shape, {})
        if not isinstance(groups, dict):
            raise ValueError(f'Mask {shape} must be grouped by camera')
        cleaned = {}
        for camera, boxes in groups.items():
            if not isinstance(camera, str) or not camera or '/' in camera or '\\' in camera:
                raise ValueError(f'Mask {shape} camera must be a camera folder name')
            if not isinstance(boxes, list):
                raise ValueError(f'Mask {shape} for {camera} must be a list')
            cleaned[camera] = []
            for box in boxes:
                if not isinstance(box, (tuple, list)) or len(box) != 4:
                    raise ValueError(f'Each mask {shape} shape needs four normalized coordinates')
                if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
                    raise ValueError(f'Mask {shape} coordinates must be finite numbers')
                x0, y0, x1, y1 = map(float, box)
                if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                    raise ValueError(f'Mask {shape} coordinates must satisfy 0 <= start < end <= 1')
                cleaned[camera].append([x0, y0, x1, y1])
        result[shape] = cleaned
    polygons = config.get('polygons', {})
    if not isinstance(polygons, dict):
        raise ValueError('Mask polygons must be grouped by camera')
    result['polygons'] = {}
    for camera, shapes in polygons.items():
        if not isinstance(camera, str) or not camera or '/' in camera or '\\' in camera:
            raise ValueError('Mask polygon camera must be a camera folder name')
        if not isinstance(shapes, list):
            raise ValueError(f'Mask polygons for {camera} must be a list')
        result['polygons'][camera] = []
        for polygon in shapes:
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError('Each mask polygon needs at least three normalized points')
            points = []
            for point in polygon:
                if (not isinstance(point, (tuple, list)) or len(point) != 2
                        or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in point)):
                    raise ValueError('Mask polygon points need two finite normalized coordinates')
                x, y = map(float, point)
                if not 0 <= x <= 1 or not 0 <= y <= 1:
                    raise ValueError('Mask polygon points must be between 0 and 1')
                points.append([x, y])
            result['polygons'][camera].append(points)
    return result


def read_mask_config(config=None):
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding='utf-8'))
    return normalize_mask_config(config)


def fisheye_cutoff_radius(shape, percent):
    """Radius of a centered circle excluding approximately `percent` of pixels.

    The circle-square intersection area is solved analytically so 1% is a
    gentle corner cleanup rather than an immediate jump to an inscribed circle.
    """
    if percent <= 0:
        return None
    height, width = shape[:2]
    half_w, half_h = width * .5, height * .5

    def kept_area(radius):
        full_to = min(half_w, math.sqrt(max(radius * radius - half_h * half_h, 0.0)))
        limit = min(half_w, radius)

        def integral(x):
            return .5 * (x * math.sqrt(max(radius * radius - x * x, 0.0))
                         + radius * radius * math.asin(min(1.0, x / radius)))

        return 4 * (half_h * full_to + integral(limit) - integral(full_to))

    low, high = 0.0, math.hypot(half_w, half_h)
    target = width * height * (1 - percent / 100)
    for _ in range(48):
        middle = (low + high) * .5
        if kept_area(middle) < target:
            low = middle
        else:
            high = middle
    return (low + high) * .5


def static_keep_mask(shape, camera, config):
    """255 outside user-defined fixed exclusions, 0 inside them."""
    config = normalize_mask_config(config)
    height, width = shape[:2]
    keep = np.full((height, width), 255, np.uint8)
    radius = fisheye_cutoff_radius(shape, config['fisheye_border_percent'])
    if radius is not None:
        yy, xx = np.ogrid[:height, :width]
        keep[(xx - (width - 1) * .5) ** 2 + (yy - (height - 1) * .5) ** 2 > radius ** 2] = 0
    for x0, y0, x1, y1 in config['rectangles'].get(camera, []):
        keep[int(y0 * height):math.ceil(y1 * height), int(x0 * width):math.ceil(x1 * width)] = 0
    for x0, y0, x1, y1 in config['ellipses'].get(camera, []):
        cx, cy = (x0 + x1) * width * .5, (y0 + y1) * height * .5
        rx, ry = (x1 - x0) * width * .5, (y1 - y0) * height * .5
        left, right = max(0, int(x0 * width)), min(width, math.ceil(x1 * width))
        top, bottom = max(0, int(y0 * height)), min(height, math.ceil(y1 * height))
        yy, xx = np.ogrid[top:bottom, left:right]
        keep[top:bottom, left:right][((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1] = 0
    for polygon in config['polygons'].get(camera, []):
        points = np.asarray([[round(x * (width - 1)), round(y * (height - 1))]
                             for x, y in polygon], dtype=np.int32)
        cv2.fillPoly(keep, [points], 0)
    return keep


def apply_mask_exclusions(mask, camera, config):
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError('Expected a grayscale uint8 mask')
    result = mask.copy()
    result[static_keep_mask(mask.shape, camera, config) == 0] = 0
    return result


def fixed_keep_samples(shape, u, v, camera, config):
    """Exclude four-tap color samples that touch a user-defined fixed region."""
    config = normalize_mask_config(config)
    height, width = shape[:2]
    x, y = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    keep = (x >= 0) & (y >= 0) & (x + 1 < width) & (y + 1 < height)
    radius = fisheye_cutoff_radius(shape, config['fisheye_border_percent'])
    if radius is not None:
        dx = np.maximum(abs(x - (width - 1) * .5), abs(x + 1 - (width - 1) * .5))
        dy = np.maximum(abs(y - (height - 1) * .5), abs(y + 1 - (height - 1) * .5))
        keep &= dx * dx + dy * dy <= radius * radius
    for x0, y0, x1, y1 in config['rectangles'].get(camera, []):
        left, top = int(x0 * width), int(y0 * height)
        right, bottom = math.ceil(x1 * width), math.ceil(y1 * height)
        keep &= ~((x + 1 >= left) & (x < right) & (y + 1 >= top) & (y < bottom))
    for x0, y0, x1, y1 in config['ellipses'].get(camera, []):
        cx, cy = (x0 + x1) * width * .5, (y0 + y1) * height * .5
        rx, ry = (x1 - x0) * width * .5, (y1 - y0) * height * .5
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            keep &= ((x + du - cx) / rx) ** 2 + ((y + dv - cy) / ry) ** 2 > 1
    for polygon in config['polygons'].get(camera, []):
        points = np.asarray([[px * (width - 1), py * (height - 1)] for px, py in polygon], dtype=float)
        taps_x = np.stack((x, x + 1, x, x + 1))
        taps_y = np.stack((y, y, y + 1, y + 1))
        inside = np.zeros(taps_x.shape, dtype=bool)
        for i in range(len(points)):
            x0, y0 = points[i]
            x1, y1 = points[(i + 1) % len(points)]
            if y0 == y1:
                continue
            crosses = ((y0 > taps_y) != (y1 > taps_y)) & (
                taps_x < (x1 - x0) * (taps_y - y0) / (y1 - y0) + x0)
            inside ^= crosses
        keep &= ~inside.any(axis=0)
    return keep


def masked_operator_keep(points, views, images_dir, masks_dir, radius, project):
    """Remove near-camera geometry only with repeated person-mask evidence.

    At least two masked observations and a strict majority are required.
    Clear nearby views provide contrary evidence. Dense planar geometry is also
    retained: a silhouette alone cannot distinguish a person from a door behind
    one. This is a conservative filter, not a dynamic SLAM classifier.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Operator radius must be finite and positive')
    from scipy.spatial import cKDTree
    person_dir = Path(masks_dir) / 'persons'
    manifest = Path(masks_dir) / 'manifest.json'
    metadata = json.loads(manifest.read_text(encoding='utf-8')) if manifest.is_file() else {}
    if metadata.get('schema') == 2 and not person_dir.is_dir():
        raise ValueError('Legacy combined masks need their person-only layer for geometry removal')
    evidence_dir = person_dir if metadata.get('schema') != 3 and person_dir.is_dir() else masks_dir
    config = normalize_mask_config(metadata.get('mask_config')) if metadata.get('schema') == 3 else None
    tree = cKDTree(points)
    person = np.zeros(len(points), dtype=np.uint32)
    observed = np.zeros(len(points), dtype=np.uint32)
    for image, rotation, center, params in views:
        indices = np.asarray(tree.query_ball_point(center, radius), dtype=np.int64)
        if not len(indices):
            continue
        camera = (np.asarray(rotation) @ (points[indices] - center).T).T
        forward = camera[:, 2] > .15
        indices, camera = indices[forward], camera[forward]
        if not len(indices):
            continue
        mask = load_keep_mask(image, images_dir, evidence_dir)
        u, v, _ = project(camera, params)
        valid = ((u >= 0) & (v >= 0) & (u < mask.shape[1] - 1) & (v < mask.shape[0] - 1)
                 & (np.hypot(u - params[2], v - params[3]) < 1620.0 * min(mask.shape) / 3840.0))
        if config is not None:
            camera_name = Path(image).resolve().relative_to(Path(images_dir).resolve()).parts[0]
            valid &= fixed_keep_samples(mask.shape, u, v, camera_name, config)
        indices, u, v = indices[valid], u[valid], v[valid]
        observed[indices] += 1
        person[indices[~keep_samples(mask, u, v)]] += 1
    remove = (person >= 2) & (person.astype(np.uint64) * 2 > observed)
    candidates = np.flatnonzero(remove)
    if len(candidates):
        # A person mask labels image pixels, not the depth of the object at that
        # pixel. Protect locally supported planes such as doors and walls before
        # deleting any LiDAR return. Process only mask-voted points to keep this
        # inexpensive for multi-million-point clouds.
        planar = np.zeros(len(candidates), dtype=bool)
        for reach, minimum in ((.10, 16), (.30, 12)):
            pending = np.flatnonzero(~planar)
            if not len(pending):
                break
            neighbors = tree.query(points[candidates[pending]], k=32,
                                   distance_upper_bound=reach, workers=-1)[1]
            for i, ids in enumerate(neighbors):
                local = points[ids[ids < len(points)]]
                if len(local) < minimum:
                    continue
                center = local.mean(axis=0)
                centered = local - center
                eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(local))
                candidate = points[candidates[pending[i]]]
                residual = abs((candidate - center) @ eigenvectors[:, 0])
                if eigenvalues[1] > 0.000025 and eigenvalues[0] < .04 * eigenvalues[1] and residual < .025:
                    planar[pending[i]] = True
                    continue
                # Trim isolated foreground returns before fitting the nearby
                # fixed surface. A person in front of a sparse floor otherwise
                # makes its covariance look non-planar.
                for _ in range(2):
                    center = local.mean(axis=0)
                    centered = local - center
                    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(local))
                    distances = np.abs(centered @ eigenvectors[:, 0])
                    local = local[distances <= np.quantile(distances, .8)]
                center = local.mean(axis=0)
                centered = local - center
                eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / len(local))
                # The queried point must lie on the plane. This stops a person
                # standing in front of a door/floor from inheriting its support.
                residual = abs((candidate - center) @ eigenvectors[:, 0])
                planar[pending[i]] = (len(local) >= 8 and eigenvalues[1] > 0.000025
                                      and eigenvalues[0] < .04 * eigenvalues[1]
                                      and residual < .025)
        remove[candidates[planar]] = False
        print(f'[*] Protected {int(planar.sum()):,} mask-voted points on planar surfaces (including floors)')
    print(f'[*] Mask-supported operator removal ({radius:g} m): {int(remove.sum()):,}/{len(points):,} points')
    if remove.all():
        raise ValueError('Person masks would remove the entire cloud')
    return ~remove


def generate_person_masks(dataset, model_path=None, threshold=0.5, margin=0.03, mask_config=None):
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Person score threshold must be between 0 and 1')
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError('Person mask margin must be between 0 and 1')
    config = read_mask_config(mask_config)
    executable = resolve_masker_executable()
    if executable is None:
        raise RuntimeError('Built-in RF-DETR runtime missing: install the masking-enabled app build')
    dataset = Path(dataset)
    images = dataset / 'images'
    sources = sorted(p for p in images.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff') and p.is_file())
    if not sources:
        raise ValueError(f'No source images in {images}')
    model = resolve_model(model_path)
    if model is None:
        fingerprint = MODEL_SHA256
    else:
        with model.open('rb') as stream:
            fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    masks = dataset / 'masks'
    masks.mkdir(parents=True, exist_ok=True)
    manifest = masks / 'manifest.json'
    settings = {'schema': 3, 'model_sha256': fingerprint, 'threshold': threshold, 'margin': margin,
                'polarity': 'white_keep_black_foreground', 'naming': 'image_relative_name_plus_png',
                'mask_config': config,
                'sources': {p.relative_to(images).as_posix(): [p.stat().st_size, p.stat().st_mtime_ns] for p in sources}}
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding='utf-8'))
        if (all(previous.get(k) == v for k, v in settings.items())
                and not (masks / 'persons').exists()
                and all(mask_path(p, images, masks).is_file() for p in sources)):
            print(f'[*] Reusing {len(sources)} combined foreground masks')
            return masks
    # Only fetch resources after confirming this dataset needs fresh inference.
    # A completed mask manifest can therefore be reused without a network call.
    if model is None:
        model = ensure_model(download_progress)
        with model.open('rb') as stream:
            settings['model_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    ensure_runtime(download_progress)
    # Invalidate completion before running: interrupted/failed batches cannot be reused.
    manifest.unlink(missing_ok=True)
    started = time.perf_counter()
    root = get_app_root()
    cache = Path(os.environ['RAVEN_RFDETR_CACHE']) if os.environ.get('RAVEN_RFDETR_CACHE') else (
        get_config_dir() / 'rfdetr-cache' if getattr(sys, 'frozen', False) else root / 'models')
    cache.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(executable), '--input', str(images.resolve()), '--output', str(masks.resolve()),
                    '--model', str(model.resolve()), '--cache-dir', str(cache.resolve()),
                    '--threshold', str(threshold), '--margin', str(margin)], check=True,
                   env=masker_environment(executable))
    static_masks = {}
    for source in sources:
        mask = load_keep_mask(source, images, masks)
        with Image.open(source) as image:
            dimensions = (image.height, image.width)
        if mask.shape != dimensions or np.any((mask != 0) & (mask != 255)):
            raise ValueError(f'Mask dimensions or values invalid for {source}')
        camera = source.relative_to(images).parts[0]
        key = (camera, dimensions)
        if key not in static_masks:
            static_masks[key] = static_keep_mask(dimensions, camera, config)
        mask[static_masks[key] == 0] = 0
        success, encoded = cv2.imencode('.png', mask)
        if not success:
            raise OSError(f'Cannot encode combined mask for {source}')
        encoded.tofile(mask_path(source, images, masks))
    # Schema 2 kept a duplicate person-only PNG tree. Remove it only after
    # every combined mask is valid; schema 3 reconstructs fixed exclusions from
    # the manifest and therefore needs just one PNG per frame.
    old_person_dir = masks / 'persons'
    if old_person_dir.exists():
        resolved_masks = masks.resolve()
        resolved_old = old_person_dir.resolve()
        if resolved_old == resolved_masks or not resolved_old.is_relative_to(resolved_masks):
            raise ValueError('Refusing to remove person masks outside the mask directory')
        shutil.rmtree(old_person_dir)
    settings.update(image_count=len(sources), elapsed_seconds=time.perf_counter() - started)
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(settings, indent=2), encoding='utf-8')
    temporary.replace(manifest)
    print(f'[+] {len(sources)} combined foreground masks ready: {masks}')
    return masks
