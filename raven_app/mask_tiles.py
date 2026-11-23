"""Upright, tiled views of a frame for the RF-DETR person detector.

The detector takes 432x432 pixels and learned from upright photos. A whole 3K
fisheye squeezed into one pass leaves a person a few metres away only 10-20 px
tall, and a rig that carries the camera upside down shows every person standing
on their head: both make it miss people. Each frame is therefore turned upright
and cut into overlapping tiles, and the tile masks are merged in full size.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import shutil

import cv2
import numpy as np

INPUT = 432
VERSION = 1
# (tile = shorter side / divisor, overlap, drop the top row of the upright frame)
LEVELS = {'tensorrt': ((2, .5, False), (3, 1 / 3, True)),
          'vulkan': ((2, .5, True),)}
ORIENTATION_LEVELS = ((2, .5, True),)
ORIENTATION_SAMPLES = 6
MIN_TILE_KEPT = .03
CHUNK = 40


def _spans(length, tile, step):
    if tile >= length:
        return [0]
    # A frame a little off square (2944x2880) must not cost one more column.
    count = max(2, math.ceil((length - tile) / step - .15) + 1)
    return [round(i * (length - tile) / (count - 1)) for i in range(count)]


def tile_boxes(height, width, levels, keep=None):
    """(y, x, height, width) views of an upright frame: the whole frame, then the tiles of each level.

    Tiles that show almost nothing but fixed exclusions (`keep` == 0) are left out.
    """
    boxes = [(0, 0, height, width)]
    shorter = min(height, width)
    for divisor, overlap, skip_top in levels:
        tile = round(shorter / divisor)
        if tile < INPUT:
            continue
        step = tile * (1 - overlap)
        rows, columns = _spans(height, tile, step), _spans(width, tile, step)
        if skip_top and len(rows) > 2:
            rows = rows[1:]
        boxes.extend((y, x, tile, tile) for y in rows for x in columns
                     if keep is None or np.count_nonzero(keep[y:y + tile, x:x + tile]) >= MIN_TILE_KEPT * tile * tile)
    return boxes


def read_image(path):
    # Masks follow the stored pixel grid, as COLMAP and the colorizer read it.
    image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def rotate(image, quarter_turns):
    return np.ascontiguousarray(np.rot90(image, quarter_turns))


def merge_tiles(shape, boxes, masks):
    """Union of the people in every tile: 255 keeps a pixel, 0 is a person."""
    keep = np.full(shape, 255, np.uint8)
    for (y, x, height, width), mask in zip(boxes, masks):
        if mask.all():
            continue
        tile = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
        keep[y:y + height, x:x + width][tile < 128] = 0
    return keep


def person_blobs(keep):
    """How many separate people-sized areas a mask excludes; slivers do not count."""
    _, _, stats, _ = cv2.connectedComponentsWithStats((keep == 0).astype(np.uint8))
    smallest = (min(keep.shape) / 100) ** 2
    return int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] >= smallest))


def pick_rotation(counts):
    """Quarter turns that show clearly more people than the stored orientation, else 0."""
    best = max(range(len(counts)), key=lambda turns: counts[turns])
    return best if counts[best] >= 3 + 1.25 * counts[0] else 0


def sample_evenly(items, count):
    if len(items) <= count:
        return list(items)
    return [items[round(i * (len(items) - 1) / (count - 1))] for i in range(count)]


def write_views(image, turns, levels, fixed, folder, prefix):
    """Write the detector inputs of one frame; returns the boxes `read_views` needs."""
    image = rotate(image, turns)
    boxes = tile_boxes(*image.shape[:2], levels, None if fixed is None else rotate(fixed, turns))
    for number, (y, x, height, width) in enumerate(boxes):
        tile = image[y:y + height, x:x + width]
        size = (min(width, INPUT), min(height, INPUT))
        if size != (width, height):
            tile = cv2.resize(tile, size, interpolation=cv2.INTER_AREA)
        done, data = cv2.imencode('.jpg', tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not done:
            raise OSError('Cannot write detector view')
        data.tofile(Path(folder) / f'{prefix}_{number:03d}.jpg')
    return boxes


def read_views(folder, prefix, shape, turns, boxes):
    """Keep mask (255 keeps, 0 is a person) of one stored frame from the detector outputs."""
    masks = []
    for number in range(len(boxes)):
        path = Path(folder) / f'{prefix}_{number:03d}.jpg.png'
        mask = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE) if path.is_file() else None
        if mask is None or np.any((mask != 0) & (mask != 255)):
            raise ValueError(f'Person detector returned no valid mask: {path.name}')
        masks.append(mask)
    return rotate(merge_tiles(shape[::-1] if turns % 2 else shape, boxes, masks), -turns)


class TiledDetector:
    """Runs a directory-to-directory detector on the tiles of frames.

    `detect(input_dir, output_dir)` must write `<name>.png` (255 keeps, 0 is a
    person) for every image in `input_dir`. `keep_for(shape, source)` gives the
    fixed exclusions of a frame, or None.
    """

    def __init__(self, work_dir, detect, levels, keep_for=lambda shape, source: None):
        self.input, self.output = Path(work_dir) / 'in', Path(work_dir) / 'out'
        self.detect, self.levels, self.keep_for = detect, levels, keep_for
        self.workers = max(2, min(8, (os.cpu_count() or 4) // 2))
        self.passes = 0

    def run(self, jobs, finish, levels=None):
        """Detect people in `jobs` = [(source, quarter turns)]; returns finish(source, keep mask) for each."""
        for folder in (self.input, self.output):
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)

        def prepare(index):
            source, turns = jobs[index]
            image = read_image(source)
            return image.shape[:2], write_views(image, turns, self.levels if levels is None else levels,
                                                self.keep_for(image.shape[:2], source), self.input, f'{index:04d}')

        def collect(index):
            source, turns = jobs[index]
            shape, boxes = views[index]
            return finish(source, read_views(self.output, f'{index:04d}', shape, turns, boxes))

        with ThreadPoolExecutor(self.workers) as pool:
            views = list(pool.map(prepare, range(len(jobs))))
            self.passes += sum(len(boxes) for _, boxes in views)
            self.detect(self.input, self.output)
            return list(pool.map(collect, range(len(jobs))))

    def upright_rotation(self, sources):
        """Quarter turns (counter-clockwise) that put people of this camera on their feet."""
        samples = sample_evenly(sources, ORIENTATION_SAMPLES)

        def count(source, keep):
            fixed = self.keep_for(keep.shape, source)
            if fixed is not None:
                keep = np.where(fixed == 0, 255, keep).astype(np.uint8)
            return person_blobs(keep)

        found = self.run([(source, turns) for turns in range(4) for source in samples], count, ORIENTATION_LEVELS)
        counts = [sum(found[turns * len(samples):(turns + 1) * len(samples)]) for turns in range(4)]
        return pick_rotation(counts), counts
