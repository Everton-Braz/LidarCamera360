"""Motion-blur detection for extracted frames.

A timelapse INSV keeps one shot per second, so there is no sharper neighbour to swap in at extraction: a
blurred frame can only be left out. Sharpness is the variance of the Laplacian over the central part of the
fisheye (away from the rim and the rig), and a frame counts as blurred when it falls below `ratio` times the
median of its `window` neighbours on the same lens. The relative test matters: a dark or bare room is never
as "sharp" as a cluttered one, but it is as sharp as the frames taken beside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = 'sharpness.json'
DEFAULT_RATIO = 0.5
DEFAULT_WINDOW = 21


def _score(path):
    import cv2
    gray = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if gray is None:
        return None
    h, w = gray.shape
    return float(cv2.Laplacian(gray[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.CV_64F).var())


def sharpness_scores(dataset_dir, names=None):
    """{image name: sharpness}, cached in images/sharpness.json against each file's size and mtime."""
    images = Path(dataset_dir) / 'images'
    if names is None:
        names = list(json.loads((images / 'frames.json').read_text(encoding='utf-8')).get('timestamps', {}))
    cache_path = images / CACHE
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    scores, changed = {}, False
    for name in names:
        path = images / name
        try:
            stat = path.stat()
        except OSError:
            continue
        stamp = [stat.st_size, stat.st_mtime_ns]
        entry = cache.get(name)
        if not entry or entry.get('file') != stamp:
            value = _score(path)
            if value is None:
                continue
            entry = cache[name] = {'file': stamp, 'sharpness': value}
            changed = True
        scores[name] = entry['sharpness']
    if changed:
        cache_path.write_text(json.dumps(cache), encoding='utf-8')
    return scores


def blurred_frames(dataset_dir, ratio=DEFAULT_RATIO, window=DEFAULT_WINDOW, names=None):
    """Names whose sharpness is below `ratio` x the median of their `window` same-lens neighbours."""
    from scipy.ndimage import median_filter
    scores = sharpness_scores(dataset_dir, names)
    by_lens = {}
    for name in scores:
        by_lens.setdefault(name.replace('\\', '/').split('/')[0], []).append(name)
    blurred = set()
    for lens_names in by_lens.values():
        lens_names.sort()
        values = np.array([scores[n] for n in lens_names])
        if len(values) < 3:
            continue
        local = median_filter(values, size=min(window, len(values)), mode='nearest')
        blurred.update(n for n, v, m in zip(lens_names, values, local) if v < ratio * m)
    return blurred
