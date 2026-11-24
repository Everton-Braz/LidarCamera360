"""Dense photogrammetry from the fisheye photos with OpenMVS, guided by the LiDAR.

Runs on the 3DGS export (metric poses, masks, LiDAR seed) and writes only to its own folder:

    <colmap_3dgs>/dense_openmvs/
        dense_photo.ply     what the photos alone reconstructed
        dense_fused.ply     every photo point + the LiDAR points where the photos have none (mode 'fused')
        report.json         settings, counts, timings, and the accuracy measured against the LiDAR
        work/               images, masks, scene and depth maps (deleted unless keep_work)

The photos are never cut into pinhole faces. Each one stays a single fisheye: it is resampled once to the
ideal equidistant fisheye (pixel radius = f * theta), which only removes the calibrated lens distortion.
OpenMVS has no fisheye model, so this needs the patched DensifyPointCloud (third_party/openmvs): there, the
camera type OpenMVS calls SPHERICAL is this fisheye, on a square image spanning FISHEYE_FOV radians. A stock
OpenMVS would read the same file as a 360 panorama, so `check_binary` refuses a program without the patch.

The LiDAR guides it: points seen by each photo give OpenMVS the neighbour photos and the starting depths.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from raven_app.dense_fusion import _voxel_thin, read_ply, write_ply

FISHEYE_FOV = 3.5                  # radians across the image width; the same constant is in the patch
PATCH_MARK = b'LC360-FISHEYE-3.5'  # string the patched DensifyPointCloud carries
MAX_THETA = 1.5                    # rays farther from the axis are masked: the rim is blurred and flared
QUALITY = {'draft': 1024, 'balanced': 1664, 'high': 2496, 'full': 3328}   # fisheye image width
MODES = ('fused', 'photo')
MESH_LIDAR_VIEWS = 8               # photos kept per LiDAR point that fills a gap of the mesh


# --- COLMAP cameras -------------------------------------------------------------------------------
def _radial_fisheye(u, v, coeffs):
    r2 = u * u + v * v
    d, p = np.zeros_like(r2), r2
    for k in coeffs:
        d, p = d + k * p, p * r2
    return d


def project_rays(model, params, d):
    """Pixels (COLMAP convention, first pixel centre at 0.5) of unit rays `d` (N,3), and which are valid."""
    p = list(params)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    if model.endswith('FISHEYE'):
        s = np.where(rho > 1e-9, theta / np.maximum(rho, 1e-9), 1.0)
        u, v, valid = x * s, y * s, np.ones(len(d), bool)
    else:
        valid = z > 0.05
        zz = np.where(valid, z, 1.0)
        u, v = x / zz, y / zz
    r2 = u * u + v * v
    if model in ('SIMPLE_PINHOLE', 'SIMPLE_RADIAL', 'RADIAL', 'SIMPLE_RADIAL_FISHEYE', 'RADIAL_FISHEYE'):
        fx = fy = p[0]
        cx, cy, rest = p[1], p[2], p[3:]
        rad = _radial_fisheye(u, v, rest)
        du, dv = u * rad, v * rad
    else:
        fx, fy, cx, cy = p[:4]
        rest = p[4:]
        if model in ('PINHOLE',):
            du = dv = 0.0
        elif model == 'OPENCV_FISHEYE':
            rad = _radial_fisheye(u, v, rest)
            du, dv = u * rad, v * rad
        elif model in ('OPENCV', 'THIN_PRISM_FISHEYE'):
            k1, k2, p1, p2, *more = rest
            k3, k4, sx, sy = more if more else (0.0, 0.0, 0.0, 0.0)
            rad = _radial_fisheye(u, v, (k1, k2, k3, k4))
            du = u * rad + 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + sx * r2
            dv = v * rad + 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + sy * r2
        else:
            raise ValueError(f'camera model {model} is not supported for dense photogrammetry')
    return np.column_stack([fx * (u + du) + cx, fy * (v + dv) + cy]), valid


def equidistant_rays(size):
    """Unit rays (size*size,3) and angle to the axis of every pixel of the square fisheye image."""
    c = (np.arange(size) - (size - 1) / 2) * FISHEYE_FOV / size
    u, v = np.meshgrid(c, c)
    theta = np.hypot(u, v)
    s = np.sinc(theta / math.pi)            # sin(theta)/theta
    return np.column_stack([(u * s).ravel(), (v * s).ravel(), np.cos(theta).ravel()]), theta


def equidistant_project(X, size):
    """Pixels (integer pixel centres) and angle to the axis of camera-frame points."""
    rho = np.hypot(X[:, 0], X[:, 1])
    theta = np.arctan2(rho, X[:, 2])
    k = theta / np.maximum(rho, 1e-12) * (size / FISHEYE_FOV)
    return X[:, :2] * k[:, None] + (size - 1) / 2, theta


def _rotation(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_colmap_text(sparse):
    """({camera id: (model, w, h, params)}, [(name, camera id, R world->camera, centre)]) of a text model."""
    sparse = Path(sparse)
    if not (sparse / 'images.txt').is_file():
        raise FileNotFoundError(f'{sparse} has no images.txt')
    cameras, images = {}, []
    for line in (sparse / 'cameras.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            t = line.split()
            cameras[int(t[0])] = (t[1], int(t[2]), int(t[3]), [float(a) for a in t[4:]])
    lines = [l for l in (sparse / 'images.txt').read_text(encoding='utf-8').splitlines() if not l.startswith('#')]
    for line in lines[0::2]:
        t = line.split(maxsplit=9)
        if len(t) < 10:
            continue
        R = _rotation([float(a) for a in t[1:5]])
        images.append((t[9].strip(), int(t[8]), R, -(R.T * np.array([float(a) for a in t[5:8]])).sum(axis=1)))
    return cameras, images


# --- images ---------------------------------------------------------------------------------------
def stereographic_rays(size):
    """Same as equidistant_rays for a stereographic fisheye that reaches MAX_THETA at the image border."""
    c = (np.arange(size) - (size - 1) / 2) / stereographic_focal(size)
    u, v = np.meshgrid(c, c)
    r = np.hypot(u, v)
    theta = 2 * np.arctan(r / 2)
    s = np.where(r > 1e-9, np.sin(theta) / np.maximum(r, 1e-9), 1.0)
    return np.column_stack([(u * s).ravel(), (v * s).ravel(), np.cos(theta).ravel()]), theta


def stereographic_focal(size):
    return size / (4 * math.tan(MAX_THETA / 2))


def build_remap(model, w, h, params, size, rays_of=equidistant_rays):
    """(map_x, map_y, inside) resampling a photo to the ideal fisheye; `inside` is where it has pixels."""
    rays, theta = rays_of(size)
    px, valid = project_rays(model, params, rays)
    px -= 0.5
    inside = valid & (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] <= w - 1) & (px[:, 1] <= h - 1)
    inside = inside.reshape(size, size) & (theta <= MAX_THETA)
    return (px[:, 0].reshape(size, size).astype(np.float32), px[:, 1].reshape(size, size).astype(np.float32), inside)


def _read_image(path, flag):
    import cv2
    data = cv2.imdecode(np.fromfile(str(path), np.uint8), flag)      # cv2.imread fails on accented paths
    if data is None:
        raise ValueError(f'cannot read {path}')
    return data


def _write_image(path, image, *args):
    import cv2
    ok, data = cv2.imencode(Path(path).suffix, image, list(args))
    if not ok:
        raise ValueError(f'cannot encode {path}')
    data.tofile(str(path))


def find_mask(colmap_dir, name):
    for candidate in (colmap_dir / 'masks' / f'{name}.png', colmap_dir / 'masks' / Path(name).with_suffix('.png')):
        if candidate.is_file():
            return candidate
    return None


def prepare_images(colmap_dir, cameras, images, size, work, threads=8, log=print, *, rays_of=equidistant_rays,
                   focal=None, names=None, mask_dir='masks', mask_name='{}.mask.png'):
    """Write work/images/NNNNN.jpg + work/masks/NNNNN.mask.png (255 = use, 0 = ignore).

    Returns the masks at VIS_SIZE, for lidar_views."""
    names = names or [f'{i:05d}' for i in range(len(images))]
    import cv2
    (work / 'images').mkdir(parents=True, exist_ok=True)
    (work / mask_dir).mkdir(parents=True, exist_ok=True)
    maps = {cid: build_remap(*cameras[cid], size, rays_of) for cid in {i[1] for i in images}}
    # photos finer than the working size are shrunk first, or the resampling would skip pixels
    shrink = {cid: min(1.0, (focal or size / FISHEYE_FOV) / cameras[cid][3][0]) for cid in maps}

    def one(index):
        name, cid, _, _ = images[index]
        mx, my, inside = maps[cid]
        photo, s = _read_image(colmap_dir / 'images' / name, cv2.IMREAD_COLOR), shrink[cid]
        h, w = photo.shape[:2]
        if (w, h) != (cameras[cid][1], cameras[cid][2]):
            raise ValueError(f'{name} is {w}x{h} but its camera is {cameras[cid][1]}x{cameras[cid][2]}')
        sx = sy = 1.0
        if s < 0.9:
            small = cv2.resize(photo, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            sx, sy, photo = small.shape[1] / w, small.shape[0] / h, small
        out = cv2.remap(photo, (mx + 0.5) * sx - 0.5, (my + 0.5) * sy - 0.5, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        keep = inside
        mask_path = find_mask(colmap_dir, name)
        if mask_path is not None:
            mask = _read_image(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            keep = keep & (cv2.remap(mask, mx, my, cv2.INTER_NEAREST, borderValue=0) > 127)
        # the matching window must not straddle a person and the wall behind it
        keep = cv2.erode(keep.astype(np.uint8), np.ones((5, 5), np.uint8), borderValue=0)
        out[~inside] = 0
        _write_image(work / 'images' / f'{names[index]}.jpg', out, cv2.IMWRITE_JPEG_QUALITY, 96)
        _write_image(work / mask_dir / mask_name.format(names[index]), keep * 255)
        return cv2.resize(keep, (VIS_SIZE, VIS_SIZE), interpolation=cv2.INTER_AREA) > 0

    with ThreadPoolExecutor(threads) as pool:
        masks = list(pool.map(one, range(len(images))))
    log(f'[+] {len(images)} fisheye photos prepared at {size}x{size} px (one resampling, no pinhole faces)')
    return masks


# --- LiDAR seed -----------------------------------------------------------------------------------
VIS_SIZE = 512


def find_lidar_seed(colmap_dir):
    """The export's LiDAR cloud; the LiDAR-only copy when the seed was densified."""
    sparse = colmap_dir / 'sparse' / '0'
    return next((p for p in (sparse / 'points3D_lidar_only.ply', sparse / 'points3D.ply', colmap_dir / 'points3D.ply')
                 if p.is_file()), None)


def lidar_views(xyz, images, masks, occluder_xyz=None, max_views=24, max_range=60.0, tol=0.03, log=print):
    """Which photos see each LiDAR point: in the mask and not behind the cloud's own z-buffer.

    Returns (N, max_views) image indices, -1 padded, nearest photos first. `occluder_xyz` is a denser cloud
    for the z-buffer than the points themselves."""
    occ = xyz if occluder_xyz is None else occluder_xyz
    n = len(xyz)
    best_d = np.full((n, max_views), np.inf, np.float32)
    best_i = np.full((n, max_views), -1, np.int32)
    for index, (_, _, R, C) in enumerate(images):
        def see(points):
            d = points - C
            X = d[:, 0:1] * R[:, 0] + d[:, 1:2] * R[:, 1] + d[:, 2:3] * R[:, 2]
            px, theta = equidistant_project(X, VIS_SIZE)
            rng = np.sqrt((d * d).sum(axis=1))
            ok = (theta < MAX_THETA) & (rng > 0.2) & (rng < max_range)
            sel = np.flatnonzero(ok)
            q = np.rint(px[sel]).astype(np.int32)
            inside = (q >= 0).all(axis=1) & (q < VIS_SIZE).all(axis=1)
            return sel[inside], q[inside], rng[sel[inside]]
        _, oq, orng = see(occ)
        z = np.full(VIS_SIZE * VIS_SIZE, np.inf, np.float32)
        np.minimum.at(z, oq[:, 1] * VIS_SIZE + oq[:, 0], orng.astype(np.float32))
        z = z.reshape(VIS_SIZE, VIS_SIZE)
        pad = np.pad(z, 1, constant_values=np.inf)        # a 3x3 minimum closes the holes between points
        zmin = np.min([pad[1 + a:VIS_SIZE + 1 + a, 1 + b:VIS_SIZE + 1 + b] for a in (-1, 0, 1) for b in (-1, 0, 1)], axis=0)
        sel, q, rng = see(xyz)
        ok = masks[index][q[:, 1], q[:, 0]] & (rng <= zmin[q[:, 1], q[:, 0]] * (1 + tol) + 0.05)
        sel, rng = sel[ok], rng[ok].astype(np.float32)
        worst = best_d[sel].argmax(axis=1)
        better = rng < best_d[sel, worst]
        sel, worst, rng = sel[better], worst[better], rng[better]
        best_d[sel, worst], best_i[sel, worst] = rng, index
    order = np.argsort(best_d, axis=1)
    return np.take_along_axis(best_i, order, axis=1)


def stack_views(*views):
    """Padded view arrays of several clouds, one under the other."""
    width = max(v.shape[1] for v in views)
    return np.vstack([np.pad(v, ((0, 0), (0, width - v.shape[1])), constant_values=-1) for v in views])


# --- OpenMVS scene --------------------------------------------------------------------------------
def write_mvs(path, size, images, xyz, rgb, views):
    """OpenMVS interface file (version 8): one fisheye camera, the photos' poses, the seed with its views."""
    def text(s):
        b = s.encode('utf-8')
        return struct.pack('<Q', len(b)) + b

    def doubles(a):
        return np.asarray(a, '<f8').tobytes()

    f, c = size / FISHEYE_FOV, (size - 1) / 2
    count = (views >= 0).sum(axis=1)
    order = np.argsort(count, kind='stable')                  # equal view counts are written as one block
    order = order[count[order] >= 2]
    with open(path, 'wb') as out:
        out.write(b'MVSI' + struct.pack('<II', 8, 0))
        out.write(struct.pack('<Q', 1) + text('rig') + struct.pack('<Q', 1))
        out.write(text('fisheye') + text('') + struct.pack('<II', size, size))
        out.write(doubles([[f, 0, c], [0, f, c], [0, 0, 1]]) + doubles(np.eye(3)) + doubles(np.zeros(3)))
        out.write(struct.pack('<B', 2))                       # SPHERICAL = the fisheye of the patched build
        out.write(struct.pack('<Q', len(images)))
        for _, _, R, C in images:
            out.write(doubles(R) + doubles(C))
        out.write(struct.pack('<Q', len(images)))
        for i in range(len(images)):
            out.write(text(f'images/{i:05d}.jpg') + text(f'masks/{i:05d}.mask.png') + struct.pack('<IIII', 0, 0, i, i))
            out.write(struct.pack('<fff', 0, 0, 0) + struct.pack('<Q', 0))
        out.write(struct.pack('<B', 0))
        out.write(struct.pack('<Q', len(order)))
        for n in np.unique(count[order]):
            rows = order[count[order] == n]
            block = np.zeros(len(rows), np.dtype([('X', '<f4', 3), ('n', '<u8'),
                                                  ('v', np.dtype([('i', '<u4'), ('c', '<f4')]), int(n))]))
            block['X'], block['n'] = xyz[rows], n
            block['v']['i'], block['v']['c'] = views[rows, :n], 1.0
            out.write(block.tobytes())
        out.write(struct.pack('<Q', 0))
        out.write(struct.pack('<Q', len(order)) + np.ascontiguousarray(rgb[order][:, ::-1], np.uint8).tobytes())
        out.write(struct.pack('<QQQ', 0, 0, 0))
        out.write(doubles(np.eye(4)) + doubles(np.eye(3)) + doubles(np.zeros(6)))
    return len(order)


def check_binary(binary, name='DensifyPointCloud'):
    """The path, if it is the patched program; a stock one would misread the fisheye scene."""
    binary = Path(binary) if binary else None
    if binary is None or not binary.is_file():
        where = f' at {binary}.' if binary else '; set its path in Settings.'
        raise FileNotFoundError(f'{name} (OpenMVS, fisheye build) was not found{where} '
                                'third_party/openmvs/README.md explains how to build it.')
    if PATCH_MARK not in binary.read_bytes():
        raise RuntimeError(f'{binary} is not the fisheye build of OpenMVS (third_party/openmvs/README.md).')
    return binary


def run_openmvs(binary, work, size, *, neighbors=8, min_views=3, geometric_iters=2, log=print):
    from raven_app.subprocess_utils import run_hidden_stream
    cmd = [str(binary), '-w', str(work), '-i', 'scene.mvs', '-o', 'dense.mvs',
           '--ignore-mask-label', '0', '--resolution-level', '0', '--max-resolution', str(size),
           '--min-resolution', '320', '--number-views', str(neighbors), '--number-views-fuse', str(min_views),
           '--geometric-iters', str(geometric_iters), '--estimate-roi', '0', '--crop-to-roi', '0', '--tower-mode', '0',
           '--remove-dmaps', '0', '-v', '2']
    log('[*] OpenMVS: ' + ' '.join(cmd))
    code = run_hidden_stream(cmd)
    output = work / 'dense.ply'
    if code != 0 or not output.is_file():
        logs = sorted(work.glob('DensifyPointCloud*.log'))
        raise RuntimeError(f'OpenMVS exited with code {code}' + (f'; see {logs[-1]}' if logs else ''))
    return output


# --- fusion and accuracy --------------------------------------------------------------------------
def accuracy_vs_lidar(lidar_xyz, dense_xyz, *, sample=200_000, radius=0.10, seed=0):
    """Point-to-plane distance of photo points to the LiDAR, where it has a surface within `radius`."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    tree = cKDTree(lidar_xyz)
    pts = dense_xyz[rng.choice(len(dense_xyz), min(sample, len(dense_xyz)), replace=False)]
    d, idx = tree.query(pts, k=12, distance_upper_bound=radius, workers=-1)
    ok = np.isfinite(d).all(axis=1)
    if not ok.any():
        return {'compared': 0}
    nb = lidar_xyz[idx[ok]]
    mean = nb.mean(axis=1)
    q = nb - mean[:, None]
    cov = np.einsum('nki,nkj->nij', q, q) / q.shape[1]
    w, v = np.linalg.eigh(cov)
    flat = w[:, 0] < 0.05 * w[:, 1]
    dist = np.abs(((pts[ok] - mean) * v[:, :, 0]).sum(axis=1))[flat]
    if not len(dist):
        return {'compared': 0}
    return {'compared': int(len(dist)), 'near_lidar_share': float(ok.mean()),
            'median_m': float(np.median(dist)), 'p90_m': float(np.percentile(dist, 90)),
            'within_1cm': float((dist < 0.01).mean()), 'within_2cm': float((dist < 0.02).mean()),
            'within_5cm': float((dist < 0.05).mean())}


def fuse_with_lidar(lidar_xyz, dense_xyz, *, gap_factor=1.5, min_gap_m=0.03, query_voxel_m=0.01):
    """Indices of the LiDAR points to add to the photo points: those farther from them than their local spacing.

    The photo cloud is the sharper one, so it wins where both see a surface; the LiDAR fills foliage, glass and
    dark or plain walls, where photo matching finds nothing."""
    from scipy.spatial import cKDTree
    photo = dense_xyz[_voxel_thin(dense_xyz, query_voxel_m)]
    tree = cKDTree(photo)
    spacing = tree.query(photo, k=min(5, len(photo)), workers=-1)[0].reshape(len(photo), -1)[:, -1]
    gap, nearest = tree.query(lidar_xyz, k=1, workers=-1)
    # spacing is capped: an isolated photo point must not keep the LiDAR out of the space around it
    return np.flatnonzero(gap > np.maximum(min_gap_m, gap_factor * np.minimum(spacing[nearest], 0.10)))


def read_dense_ply(path):
    """(xyz, rgb, views) of OpenMVS's dense.ply; views is (N, V) photo indices, -1 padded, or None.

    Its vertices carry lists (the photos that see each point), so the rows have no fixed size."""
    from raven_app.dense_fusion import _PLY_TYPES
    data = Path(path).read_bytes()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    count, scalars, lists, inside = 0, [], [], False
    for line in data[:end].decode('ascii', 'replace').splitlines():
        t = line.split()
        if t[:1] == ['element']:
            inside = t[1] == 'vertex'
            count = int(t[2]) if inside else count
        elif t[:1] == ['property'] and inside:
            if t[1] == 'list':
                lists.append((t[4], np.dtype(_PLY_TYPES[t[2]]), np.dtype(_PLY_TYPES[t[3]])))
            elif lists:
                raise ValueError(f'{path}: scalar after a list is not supported')
            else:
                scalars.append((t[2], _PLY_TYPES[t[1]]))
    head = np.dtype(scalars)
    starts = np.empty((len(lists) + 1, count), np.int64)     # of the scalars, then of each list's values
    if lists:
        sizes = [(n.itemsize, v.itemsize) for _, n, v in lists]
        pos, whole, step = end, int.from_bytes, head.itemsize
        for i in range(count):
            starts[0, i] = pos
            pos += step
            for k, (n_size, v_size) in enumerate(sizes, 1):
                n = data[pos] if n_size == 1 else whole(data[pos:pos + n_size], 'little')
                pos += n_size
                starts[k, i] = pos
                pos += n * v_size
    else:
        starts[0] = end + head.itemsize * np.arange(count)
    raw = np.frombuffer(data, np.uint8)

    def gather(offsets, size):
        rows = np.empty((len(offsets), size), np.uint8)
        for c in range(0, len(offsets), 1_000_000):
            rows[c:c + 1_000_000] = raw[offsets[c:c + 1_000_000, None] + np.arange(size)]
        return rows

    rows = gather(starts[0], head.itemsize).view(head)[:, 0] if count else np.empty(0, head)
    xyz = np.column_stack([rows[a] for a in 'xyz']).astype(np.float64)
    rgb = np.column_stack([rows[a] for a in ('red', 'green', 'blue')]) if 'red' in head.names else None
    views = None
    for k, (name, n_type, v_type) in enumerate(lists, 1):
        if name == 'view_indices' and count:
            n = gather(starts[k] - n_type.itemsize, n_type.itemsize).view(n_type)[:, 0].astype(np.int64)
            views = np.full((count, max(1, int(n.max()))), -1, np.int32)
            for m in np.unique(n[n > 0]):
                at = np.flatnonzero(n == m)
                views[at, :m] = gather(starts[k][at], m * v_type.itemsize).view(v_type)
    return xyz, rgb, views


def densify_openmvs(colmap_dir, binary, *, out_dir=None, quality='balanced', mode='fused', use_lidar=True,
                    mesh=None, seed_voxel_m=0.06, keep_work=False, log=print):
    """Dense photogrammetry of a 3DGS export. Never changes the export's own files. Returns the report.

    `mesh` ('glb', 'obj', or 'ply' for a surface without texture) also meshes the cloud (openmvs_mesh); a mesh
    that fails is reported and leaves the clouds in place.

    use_lidar=False still needs the LiDAR poses but gives OpenMVS only a coarse seed (0.5 m), enough to
    choose the neighbour photos: the depths then come from the photos alone.
    """
    if quality not in QUALITY:
        raise ValueError(f'quality must be one of {", ".join(QUALITY)}')
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}')
    binary = check_binary(binary)
    if mesh:
        from raven_app import openmvs_mesh
        openmvs_mesh.check_programs(binary, mesh)            # before the long run, not after it
    colmap_dir = Path(colmap_dir)
    out = Path(out_dir) if out_dir else colmap_dir / 'dense_openmvs'
    work = out / 'work'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    size, timing, t0 = QUALITY[quality], {}, time.time()

    def lap(name):
        nonlocal t0
        timing[name], t0 = round(time.time() - t0, 1), time.time()

    cameras, images = read_colmap_text(colmap_dir / 'sparse' / '0')
    images = [i for i in images if (colmap_dir / 'images' / i[0]).is_file()]
    if len(images) < 3:
        raise ValueError(f'{colmap_dir}: {len(images)} registered photos; dense photogrammetry needs at least 3')
    # depth 4 + normal 12 + confidence 4 + views 4 bytes per pixel, and the photos themselves
    need, free = len(images) * size * size * 26, shutil.disk_usage(work).free
    if need > free:
        raise RuntimeError(f'dense photogrammetry at {quality} quality needs about {need / 1e9:.0f} GB of disk and '
                           f'{free / 1e9:.0f} GB are free in {out}; choose a lower quality or free some space')
    masks = prepare_images(colmap_dir, cameras, images, size, work, log=log)
    lap('images')
    seed = find_lidar_seed(colmap_dir)
    if seed is None:
        raise FileNotFoundError(f'{colmap_dir} has no LiDAR seed (points3D.ply)')
    lidar_xyz, lidar_rgb = read_ply(seed)
    if lidar_rgb is None:
        lidar_rgb = np.full((len(lidar_xyz), 3), 128, np.uint8)
    pick = _voxel_thin(lidar_xyz, seed_voxel_m if use_lidar else 0.5)
    occluder = lidar_xyz[_voxel_thin(lidar_xyz, 0.04)]
    views = lidar_views(lidar_xyz[pick], images, masks, occluder, log=log)
    seeded = write_mvs(work / 'scene.mvs', size, images, lidar_xyz[pick], lidar_rgb[pick], views)
    log(f'[+] LiDAR guidance: {seeded:,} points seen by 2 or more photos')
    if not seeded:
        raise RuntimeError('no LiDAR point is seen by two photos; check the poses and masks of the export')
    lap('lidar_guidance')
    dense_xyz, dense_rgb, dense_views = read_dense_ply(run_openmvs(binary, work, size, log=log))
    lap('openmvs')
    if not len(dense_xyz):
        raise RuntimeError('OpenMVS reconstructed no point')
    if dense_rgb is None:
        dense_rgb = np.full((len(dense_xyz), 3), 128, np.uint8)
    write_ply(out / 'dense_photo.ply', dense_xyz, dense_rgb)
    report = {'engine': 'OpenMVS (fisheye build)', 'quality': quality, 'image_width': size, 'mode': mode,
              'lidar_guided': bool(use_lidar), 'photos': len(images), 'seed_points': seeded,
              'lidar_points': int(len(lidar_xyz)), 'photo_points': int(len(dense_xyz)),
              'photo_cloud': str(out / 'dense_photo.ply'),
              'accuracy_vs_lidar': accuracy_vs_lidar(lidar_xyz, dense_xyz)}
    keep = fuse_with_lidar(lidar_xyz, dense_xyz) if mode == 'fused' else np.zeros(0, np.int64)
    if mode == 'fused':
        write_ply(out / 'dense_fused.ply', np.vstack([dense_xyz, lidar_xyz[keep]]),
                  np.vstack([dense_rgb, lidar_rgb[keep]]))
        report.update(lidar_points_added=int(len(keep)), fused_cloud=str(out / 'dense_fused.ply'))
    lap('fusion')
    report['seconds'] = timing
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if mesh:
        try:
            gap_views = lidar_views(lidar_xyz[keep], images, masks, occluder, max_views=MESH_LIDAR_VIEWS, log=log)
            report['mesh'] = openmvs_mesh.build_mesh(
                binary, work, out, size, images, np.vstack([dense_xyz, lidar_xyz[keep]]),
                np.vstack([dense_rgb, lidar_rgb[keep]]), stack_views(dense_views, gap_views), fmt=mesh,
                lidar_xyz=lidar_xyz, log=log)
        except Exception as exc:
            report['mesh'] = {'error': str(exc)}
            log(f'[!] The mesh was not built: {exc}')
        lap('mesh')
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    acc = report['accuracy_vs_lidar']
    log(f"[+] Dense photogrammetry: {len(dense_xyz):,} photo points"
        + (f", {report['lidar_points_added']:,} of the {len(lidar_xyz):,} LiDAR points added where the photos have none" if mode == 'fused' else '')
        + (f"; median {acc['median_m'] * 100:.1f} cm from the LiDAR surface" if acc.get('compared') else '')
        + f". Saved in {out}")
    if report.get('mesh', {}).get('faces'):
        log(f"[+] Mesh: {report['mesh']['faces']:,} triangles, {report['mesh']['area_m2']:,.0f} m2")
    return report
