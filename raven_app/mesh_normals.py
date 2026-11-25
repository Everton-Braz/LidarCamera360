"""Cleaner mesh surfaces from the normal maps of the photos (MoGe2 and the like).

A normal map of one photo is an estimate, but where several photos of a wall agree on a normal it is about as good
as a plane fitted to the LiDAR. Triangles turn towards that normal by a few millimetres: walls lose their bumps and
keep their steps and edges. The place of the surface stays what the photos and the LiDAR measured.

The maps are the ones Spirula Studio writes next to a dataset: <normals>/<photo>.png, 8-bit RGB = (n + 1) * 127.5,
n in the camera frame of the photo (x right, y down, z forward), towards the camera.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from raven_app import openmvs_dense as od

MIN_VIEWS = 3           # photos that see a triangle
MIN_AGREE = 0.985       # length of the mean of their unit normals: about 10 degrees apart
MIN_COS = 0.8           # to the triangle's own normal, 37 degrees: further off, the map sees something else
MAX_EDGE = 0.10         # m; longer triangles span gaps of the points
MAX_MOVE = 0.01         # m, of any vertex
ANCHOR = 0.03           # share of its way back a vertex goes each time: keeps shapes larger than ~10 cm
STEPS = 30
MAX_RANGE = 20.0        # m from the photo
MIN_INCIDENCE = 0.15    # cosine between the ray and the surface


def normal_map_path(folder, name):
    return (Path(folder) / name).with_suffix('.png')


def find_normals(colmap_dir, images, folder=None):
    """The folder with the photos' normal maps, or None. `folder` is checked instead of <colmap>/normals."""
    folder = Path(folder) if folder else Path(colmap_dir) / 'normals'
    found = sum(normal_map_path(folder, i[0]).is_file() for i in images)
    return folder if found >= MIN_VIEWS else None


def _read_map(path, width, height):
    """(h, w, 3) float32 unit normals, 0 where there is none; None if the map is not of this photo."""
    import cv2
    raw = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
    if raw is None or abs(raw.shape[1] * height / (raw.shape[0] * width) - 1) > 0.02:
        return None
    n = raw[..., ::-1].astype(np.float32) / 127.5 - 1
    length = np.sqrt((n * n).sum(axis=2))
    valid = np.abs(length - 1) < 0.1
    return np.where(valid[..., None], n / np.maximum(length, 1e-6)[..., None], 0)


def triangle_normals(vertices, faces):
    """Unit normals and areas."""
    a, b, c = (vertices[faces[:, k]].astype(np.float64) for k in range(3))
    n = np.cross(b - a, c - a)
    length = np.sqrt((n * n).sum(axis=1))
    return n / np.maximum(length, 1e-20)[:, None], length / 2


def _photo_normals(centres, normals, vertices, camera, pose, mask, path):
    """(triangles, weights, world normals) of one photo, or None. No BLAS: this runs in threads."""
    model, width, height, params = camera
    R, C = np.asarray(pose[0], np.float32), np.asarray(pose[1], np.float32)
    image = _read_map(path, width, height)
    if image is None:
        return None
    size = od.VIS_SIZE

    def see(points):
        near = np.flatnonzero((np.abs(points - C) < MAX_RANGE).all(axis=1))
        d = points[near] - C
        r = np.sqrt((d * d).sum(axis=1))
        X = np.stack([d[:, 0] * R[a, 0] + d[:, 1] * R[a, 1] + d[:, 2] * R[a, 2] for a in range(3)], axis=1)
        px, theta = od.equidistant_project(X, size)
        q = np.rint(px).astype(np.int32)
        ok = (theta < od.MAX_THETA) & (r > 0.2) & (r < MAX_RANGE) & (q >= 0).all(axis=1) & (q < size).all(axis=1)
        return near[ok], X[ok], r[ok], q[ok, 1] * size + q[ok, 0]

    z = np.full(size * size, np.inf, np.float32)
    for points in (vertices, centres):
        _, _, r, pixel = see(points)
        np.minimum.at(z, pixel, r)
    pad = np.pad(z.reshape(size, size), 1, constant_values=np.inf)      # closes the gaps between points
    z = np.min([pad[1 + a:size + 1 + a, 1 + b:size + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0).ravel()
    index, X, r, pixel = see(centres)
    facing = -(normals[index] * (centres[index] - C)).sum(axis=1) / r
    ok = mask.ravel()[pixel] & (r <= z[pixel] * 1.03 + 0.05) & (facing > MIN_INCIDENCE)
    index, X, r, facing = index[ok], X[ok], r[ok], facing[ok]
    px, valid = od.project_rays(model, params, X / r[:, None])
    u = np.floor(px[:, 0] * image.shape[1] / width).astype(np.int64)
    v = np.floor(px[:, 1] * image.shape[0] / height).astype(np.int64)
    ok = valid & (u >= 0) & (v >= 0) & (u < image.shape[1]) & (v < image.shape[0])
    index, r, facing, seen = index[ok], r[ok], facing[ok], image[v[ok], u[ok]]
    ok = (seen != 0).any(axis=1)
    index, r, facing, seen = index[ok], r[ok], facing[ok], seen[ok]
    world = np.stack([seen[:, 0] * R[0, a] + seen[:, 1] * R[1, a] + seen[:, 2] * R[2, a] for a in range(3)], axis=1)
    return index, facing / (r * r), world


def average_normals(vertices, faces, cameras, images, masks, folder, *, log=print, workers=4):
    """Mean normal of every triangle over the photos that see it: (unit normals, agreement 0..1, photos)."""
    vertices = np.asarray(vertices, np.float32)
    centres = vertices[faces].mean(axis=1)
    own = triangle_normals(vertices, faces)[0].astype(np.float32)
    total, weight = np.zeros((len(faces), 3)), np.zeros(len(faces))
    views = np.zeros(len(faces), np.int32)
    used = 0

    def job(k):
        name, camera_id, R, C = images[k]
        return _photo_normals(centres, own, vertices, cameras[camera_id], (R, C), masks[k],
                              normal_map_path(folder, name))

    with ThreadPoolExecutor(workers) as pool:
        for first in range(0, len(images), workers):        # a few at a time: each result is large
            for result in pool.map(job, range(first, min(first + workers, len(images)))):
                if result is None:
                    continue
                index, w, n = result
                for a in range(3):
                    total[:, a] += np.bincount(index, w * n[:, a], len(faces))
                weight += np.bincount(index, w, len(faces))
                views[index] += 1
                used += 1
    length = np.sqrt((total * total).sum(axis=1))
    log(f'[*] Normal maps of {used} photos')
    return total / np.maximum(length, 1e-12)[:, None], length / np.maximum(weight, 1e-12), views


def turn_triangles(vertices, faces, target, area, *, anchor=ANCHOR, steps=STEPS, max_move=MAX_MOVE):
    """Vertices moved so that the triangles face `target`, none further than `max_move`."""
    start = np.asarray(vertices, np.float64)
    x, count = start.copy(), len(start)
    around = np.maximum(np.bincount(faces.ravel(), np.repeat(area, 3), count), 1e-12)
    for _ in range(steps):
        centre = x[faces].mean(axis=1)
        move = np.zeros_like(x)
        for k in range(3):
            corner = faces[:, k]
            off = area * ((centre - x[corner]) * target).sum(axis=1)     # to the plane the triangle turns into
            for a in range(3):
                move[:, a] += np.bincount(corner, off * target[:, a], count)
        x += move / around[:, None] + anchor * (start - x)
    d = x - start
    length = np.sqrt((d * d).sum(axis=1))
    return start + d * np.minimum(1, max_move / np.maximum(length, 1e-12))[:, None]


def smooth_by_normals(vertices, faces, cameras, images, masks, folder, *, log=print):
    """(vertices, report) of the mesh turned to the photos' normal maps."""
    target, agree, views = average_normals(vertices, faces, cameras, images, masks, folder, log=log)
    own, area = triangle_normals(vertices, faces)
    corners = np.asarray(vertices, np.float64)[faces]
    longest = np.sqrt(((corners - np.roll(corners, 1, axis=1)) ** 2).sum(axis=2)).max(axis=1)
    guided = ((views >= MIN_VIEWS) & (agree > MIN_AGREE) & (longest < MAX_EDGE)
              & ((own * target).sum(axis=1) > MIN_COS))
    report = {'folder': str(folder), 'guided_share': round(float(guided.mean()), 3)}
    if not guided.any():
        log('[!] The normal maps fit no triangle of the mesh; it is left as it is')
        return vertices, report
    moved = turn_triangles(vertices, faces, np.where(guided[:, None], target, own), area)
    turned = triangle_normals(moved, faces)[0]
    report['median_move_mm'] = round(float(np.median(np.sqrt(((moved - vertices) ** 2).sum(axis=1)))) * 1000, 2)
    report['angle_to_maps_deg'] = [
        round(float(np.degrees(np.median(np.arccos(np.clip((n * target).sum(axis=1)[guided], -1, 1))))), 2)
        for n in (own, turned)]
    log(f"[+] Normal maps guided {100 * report['guided_share']:.0f} % of the triangles")
    return moved, report
