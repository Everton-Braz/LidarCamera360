"""Sample verified SfM feature tracks, without projecting the full LiDAR cloud.

Track observations are visibility correspondences established by SfM, not a
depth buffer made from sparse points. Masks and reprojection checks still
reject contaminated correspondences. Original images are never rewritten.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import time

import cv2
import numpy as np

from .model import VERSION, fit_observations, load_model


def fingerprint(dataset, masks_dir=None):
    root = Path(dataset)
    digest = hashlib.sha256()
    files = [root / 'sparse/0/points3D.bin', root / 'sparse/0/images.bin',
             root / 'sparse/0/cameras.bin', root / 'images/frames.json']
    files += sorted((root / 'images').glob('*/*.jpg'))
    if masks_dir is not None:
        files += sorted(Path(masks_dir).glob('*/*.png'))
    digest.update(f'photometric-{VERSION}-tracks-v1'.encode())
    for path in files:
        digest.update(str(path.resolve()).encode())
        if path.is_file():
            stat = path.stat()
            digest.update(struct.pack('<QQ', stat.st_size, stat.st_mtime_ns))
    return digest.hexdigest()


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError('Truncated COLMAP model')
    return data


def _sample_tracks(path, sample_points):
    rng = np.random.default_rng(20260929)
    with path.open('rb') as stream:
        count, = struct.unpack('<Q', _read_exact(stream, 8))
        chosen = set(rng.choice(count, min(count, sample_points * 4), replace=False).tolist())
        records = []
        for index in range(count):
            pid, x, y, z, _, _, _, error, length = struct.unpack('<QdddBBBdQ', _read_exact(stream, 51))
            if index in chosen and length >= 3 and np.isfinite(error) and error <= 1.5:
                tracks = np.frombuffer(_read_exact(stream, length * 8), dtype='<i4').reshape(-1, 2)
                records.append((pid, (x, y, z), tracks.copy()))
            else:
                stream.seek(length * 8, 1)
    if not records:
        raise ValueError('No reliable SfM tracks with three views')
    xyz = np.array([r[1] for r in records])
    # Equal cells in the model's bounding box: choose one per cell then fill
    # deterministically. This limits domination by one highly textured patch.
    cells = np.floor((xyz - xyz.min(0)) / np.maximum(np.ptp(xyz, axis=0), 1e-9) * 31).astype(int)
    _, first = np.unique(cells, axis=0, return_index=True)
    rng.shuffle(first)
    remaining = np.setdiff1d(np.arange(len(records)), first)
    rng.shuffle(remaining)
    selected = np.r_[first, remaining][:sample_points]
    return [records[i] for i in selected]


def collect_observations(dataset, masks_dir=None, sample_points=50000, max_observations=500000):
    if sample_points < 30 or max_observations < 90:
        raise ValueError('Need at least 30 sample points and 90 observations')
    root = Path(dataset)
    from scripts.pipeline_auto_calibrator_and_colorizer import load_colmap_cameras, project_thin_prism
    from scipy.spatial.transform import Rotation
    from raven_app.person_masks import load_keep_mask, keep_samples
    cameras = load_colmap_cameras(root / 'sparse/0/cameras.bin')
    records = _sample_tracks(root / 'sparse/0/points3D.bin', min(sample_points, max_observations // 3))
    per_point = max(3, min(16, max_observations // len(records)))
    wanted = {}
    xyz_by_id = {}
    for pid, xyz, track in records:
        xyz_by_id[pid] = xyz
        track = track[np.linspace(0, len(track) - 1, min(per_point, len(track)), dtype=int)]
        for image_id, point_index in track:
            wanted.setdefault(int(image_id), {})[int(point_index)] = pid
    jobs = []
    names, lens_ids = [], []
    lens_names = sorted(str(cid) for cid in cameras)
    with (root / 'sparse/0/images.bin').open('rb') as stream:
        count, = struct.unpack('<Q', _read_exact(stream, 8))
        for _ in range(count):
            values = struct.unpack('<i7di', _read_exact(stream, 64))
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = values
            name = bytearray()
            while True:
                char = _read_exact(stream, 1)
                if char == b'\0':
                    break
                name.extend(char)
            name = name.decode('utf-8').replace('\\', '/')
            length, = struct.unpack('<Q', _read_exact(stream, 8))
            frame = len(names)
            names.append(name)
            lens = lens_names.index(str(camera_id))
            lens_ids.append(lens)
            if image_id not in wanted:
                stream.seek(length * 24, 1)
                continue
            features = np.frombuffer(_read_exact(stream, length * 24),
                                     dtype=[('u', '<f8'), ('v', '<f8'), ('pid', '<i8')])
            indices = np.array(sorted(wanted[image_id]), dtype=int)
            indices = indices[(indices >= 0) & (indices < length)]
            expected = np.array([wanted[image_id][int(i)] for i in indices])
            valid = features['pid'][indices] == expected
            indices, pids = indices[valid], expected[valid]
            if not len(pids):
                continue
            points = np.array([xyz_by_id[int(pid)] for pid in pids])
            rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            camera_points = points @ rotation.T + [tx, ty, tz]
            params = cameras[camera_id]['params']
            pu, pv, _ = project_thin_prism(camera_points, params)
            uv = np.column_stack((features['u'][indices], features['v'][indices]))
            radius = np.hypot(uv[:, 0] - params[2], uv[:, 1] - params[3]) / 1620.0
            good = ((np.linalg.norm(uv - np.column_stack((pu, pv)), axis=1) <= 2) &
                    (camera_points[:, 2] > 0) & (radius < .98))
            if good.any():
                jobs.append((name, frame, lens, pids[good], uv[good], radius[good]))

    def sample(job):
        name, frame, lens, pids, uv, radius = job
        path = root / 'images' / name
        # JPEG quarter decode reduces I/O/decode cost; patch averaging reduces
        # subpixel feature noise. Coordinates account for pixel-center scaling.
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        u, v = (uv[:, 0] + .5) / 4 - .5, (uv[:, 1] + .5) / 4 - .5
        x, y = np.floor(u).astype(int), np.floor(v).astype(int)
        good = (x >= 1) & (y >= 1) & (x + 2 < image.shape[1]) & (y + 2 < image.shape[0])
        if masks_dir is not None:
            mask = load_keep_mask(path, root / 'images', masks_dir)
            # Footprint covers the reduced decoder support, not just one tap.
            mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
            good &= keep_samples(mask, uv[:, 0], uv[:, 1])
        ix = np.flatnonzero(good)
        if not len(ix):
            return None
        x, y, u, v = x[ix], y[ix], u[ix], v[ix]
        dx, dy = (u - x)[:, None], (v - y)[:, None]
        rgb = ((1-dx)*(1-dy)*image[y, x] + dx*(1-dy)*image[y, x+1] +
               (1-dx)*dy*image[y+1, x] + dx*dy*image[y+1, x+1])[:, ::-1]
        patch = np.stack([image[y+oy, x+ox] for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1))])
        variation = np.max(np.ptp(patch.astype(float), axis=0), axis=1)
        valid = (rgb.min(1) > 12) & (rgb.max(1) < 243) & (variation < 45)
        ix, rgb = ix[valid], rgb[valid]
        return {'point_id': pids[ix], 'frame_id': np.full(len(ix), frame),
                'lens_id': np.full(len(ix), lens), 'radius': radius[ix],
                'rgb': rgb.astype(np.float32), 'weight': (1 - .5 * radius[ix] ** 2).astype(np.float32),
                'u': uv[ix, 0], 'v': uv[ix, 1]}

    chunks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, result in enumerate(executor.map(sample, jobs), 1):
            if result is not None:
                chunks.append(result)
            if i % 250 == 0:
                print(f'[photometric] sampled {i}/{len(jobs)} images', flush=True)
    if not chunks:
        raise ValueError('No usable photometric image samples')
    obs = {key: np.concatenate([item[key] for item in chunks]) for key in chunks[0]}
    _, inverse, counts = np.unique(obs['point_id'], return_inverse=True, return_counts=True)
    keep = counts[inverse] >= 3
    return {key: value[keep] for key, value in obs.items()}, names, lens_names


def fit_dataset(dataset, output=None, masks_dir=None, sample_points=50000, max_observations=500000,
                mode='loglinear'):
    if mode not in ('loglinear', 'ppisp-bilateral'):
        raise ValueError('Unsupported photometric mode')
    root = Path(dataset)
    filename = 'ppisp_bilateral_params.json' if mode == 'ppisp-bilateral' else 'photometric_params.json'
    destination = Path(output) if output else root / 'photometric' / filename
    if destination.suffix.lower() != '.json':
        destination = destination / 'photometric_params.json'
    start = time.perf_counter()
    obs, names, lens_names = collect_observations(root, masks_dir, sample_points, max_observations)
    sampled = time.perf_counter()
    model = fit_observations(obs, names, lens_names)
    if mode == 'ppisp-bilateral':
        from .advanced import fit_advanced
        model = fit_advanced(obs, names, lens_names, model)
    model['mode'] = mode
    model['fingerprint'] = fingerprint(root, masks_dir)
    model['settings'] = {'sample_points': sample_points, 'max_observations': max_observations,
                         'sampler': 'verified_sfm_tracks_quarter_jpeg', 'seed': 20260929}
    model['timing'] = {'sample_seconds': sampled - start, 'fit_seconds': time.perf_counter() - sampled}
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination.with_suffix('.observations.npz'), **obs,
                        names=np.asarray(names), lens_names=np.asarray(lens_names))
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(destination)
    from .report import write_report
    write_report(model, destination.with_suffix('.html'))
    print(f'[photometric] {model["selected"]}: {json.dumps(model["metrics"])}', flush=True)
    return {'parameters': str(destination), **model}


def prepare_model(dataset, params=None, masks_dir=None, mode=None):
    filename = 'ppisp_bilateral_params.json' if mode == 'ppisp-bilateral' else 'photometric_params.json'
    path = Path(params) if params else Path(dataset) / 'photometric' / filename
    signature = fingerprint(dataset, masks_dir)
    if path.is_file():
        model = load_model(path)
        if mode is not None and model.get('mode', 'loglinear') != mode:
            raise ValueError('Photometric parameter mode does not match the requested correction')
        if model.get('fingerprint') == signature:
            print(f'[photometric] reusing validated parameters: {path}', flush=True)
            return model
        if params is not None:
            raise ValueError('Photometric parameters belong to different images/model/masks; refit')
    elif params is not None:
        raise FileNotFoundError(path)
    return fit_dataset(dataset, path, masks_dir, mode=mode or 'loglinear')
