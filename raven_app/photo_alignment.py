"""Aligned fisheye photos for the point cloud colours and the 3DGS export.

`photo_refine` finds the rolling shutter, the lens, the lever arm and the pose of every photo. Both the colours of
the point cloud and 3DGS training need one pose per photo and one lens model, so the rolling shutter has to be taken
out of the photos themselves. `align_photos` does this once per run, before the colours, in `photo_alignment/` of
the run folder:

    images/<camera>/<name>.jpg   the same lens and size, as if the whole photo was taken at once; photos that need
                                 no change (other cameras) are links to the run's own
    masks/<camera>/...           the person masks, moved with their photos
    alignment.json               lenses, poses and how well they fit

The run's own photos are never changed. When the alignment is skipped or does not fit better, there is no aligned
set and everything works on the run's photos with the calibrated poses.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from raven_app import photo_refine as pr

FOLDER, VERSION = 'photo_alignment', 1
MAX_HELD_OUT_PX = 2.0                              # of a lens with REFERENCE_FOCAL; worse is not worth an image resampling
GRID_STEP = 16                                     # px of the photo per node of the shutter's map
JPEG_QUALITY = 97


@dataclass
class AlignedPhotos:
    """Photos with one true pose each. Names are 'cam0/frame_000001.jpg'."""
    images_dir: Path
    masks_dir: Path | None
    cameras: dict                                  # camera name -> (model, width, height, params)
    poses: dict                                    # photo name -> (R world to camera, centre), in the map's frame
    report: dict = field(default_factory=dict)

    def camera_of(self, name):
        return self.cameras[name.split('/', 1)[0]]


def rig_poses(dataset_dir, calib, fps, dt_sync):
    """Cameras {id: (model, w, h, params)}, photos [(name, id, R, C)] and {name: seconds after the trajectory's start}
    of the rig's fisheyes, from the trajectory and the calibration. Empty when the rig is not a pair of fisheyes."""
    from raven_app.video import frame_time
    from raven_app.workflow import _colmap_model_parameters, _validate_rigid_transform
    from scripts.pipeline_auto_calibrator_and_colorizer import get_slam_trajectory_path
    dataset_dir = Path(dataset_dir)
    rows = np.loadtxt(get_slam_trajectory_path(dataset_dir), ndmin=2)
    t = rows[:, 0] - rows[0, 0]
    slerp = Slerp(t, Rot.from_quat(rows[:, 4:8]))
    cameras, images, when = {}, [], {}
    for camera_id, name, key in ((1, 'cam0', 'cam0_front_intrinsics'), (2, 'cam1', 'cam1_rear_intrinsics')):
        intrinsics, transform = calib.get(key), calib.get(f'T_lidar_to_{name}_rigid_4x4')
        if not intrinsics or transform is None:
            continue
        camera = _colmap_model_parameters(intrinsics, calib.get('lens_model'), camera_id)
        if pr.thin_prism_params(camera[0], camera[3]) is None:
            continue
        T = _validate_rigid_transform(transform, f'LiDAR-to-{name} calibration')
        cameras[camera_id] = camera
        for path in sorted((dataset_dir / 'images' / name).glob('*.jpg')):
            photo = f'{name}/{path.name}'
            try:
                q = frame_time(dataset_dir, photo, fps) - float(dt_sync)
            except (KeyError, OSError, ValueError):
                continue
            if not 0.0 <= q <= t[-1]:
                continue
            R = slerp(q).as_matrix()
            centre = np.array([np.interp(q, t, rows[:, k]) for k in (1, 2, 3)]) + R @ T[:3, 3]
            images.append((photo, camera_id, (R @ T[:3, :3]).T, centre))
            when[photo] = q
    return cameras, images, when


def _unproject(params, px):
    """Unit rays of pixels under THIN_PRISM_FISHEYE parameters, and which pixels the lens really shows."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx, sy = params
    ud, vd = (px[:, 0] - cx) / fx, (px[:, 1] - cy) / fy
    u, v = ud.copy(), vd.copy()
    with np.errstate(over='ignore', invalid='ignore'):
        for _ in range(30):
            r2 = u * u + v * v
            rad = r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
            eu = u * (1 + rad) + 2 * p1 * u * v + p2 * (r2 + 2 * u * u) + sx * r2 - ud
            ev = v * (1 + rad) + 2 * p2 * u * v + p1 * (r2 + 2 * v * v) + sy * r2 - vd
            slope = np.maximum(1 + r2 * (3 * k1 + r2 * (5 * k2 + r2 * (7 * k3 + r2 * 9 * k4))), 0.2)
            u, v = u - eu / slope, v - ev / slope
        theta = np.hypot(u, v)
        ok = np.isfinite(theta) & (theta < 2.0) & (np.hypot(eu, ev) < 1e-5)
    theta = np.where(ok, theta, 0.0)
    s = np.where(theta > 1e-9, np.sin(theta) / np.maximum(theta, 1e-9), 1.0)
    u, v = np.where(ok, u, 0.0), np.where(ok, v, 0.0)
    return np.stack([u * s, v * s, np.cos(theta)], 1), ok


def lens_shutter_maps(camera, motion, inverse):
    """(map_x, map_y) that take the rolling shutter out of a photo: where each pixel of the still photo is found in
    the one that was read row by row. Same lens, same size. `motion` and `inverse` as for `shutter_maps`."""
    import cv2
    from scipy.ndimage import distance_transform_edt
    from raven_app.openmvs_dense import equidistant_project
    _, w, h, params = camera
    readout, omega, speed = motion
    gw, gh = max(8, w // GRID_STEP), max(8, h // GRID_STEP)
    gx, gy = np.meshgrid((np.arange(gw) + 0.5) * w / gw, (np.arange(gh) + 0.5) * h / gh)
    rays, ok = _unproject(params, np.column_stack([gx.ravel(), gy.ravel()]))
    near = equidistant_project(rays, inverse.shape[0])[0].astype(np.float32).reshape(gh, gw, 2)
    inv = cv2.remap(cv2.medianBlur(inverse, 5), near[..., 0], near[..., 1], cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE).ravel()
    K = np.tile(np.asarray(params, float), (len(rays), 1))
    first = px = pr.project_thin_prism(rays, K)
    for _ in range(3):
        tau = (readout[0] * (px[:, 0] - w / 2) + readout[1] * (px[:, 1] - h / 2)) / h
        d = rays - speed * (tau * inv)[:, None]
        turn = np.cross(omega, d)
        d = d - turn * tau[:, None] + 0.5 * np.cross(omega, turn) * (tau * tau)[:, None]
        px = pr.project_thin_prism(d, K)
    move = np.where(ok[:, None], px - first, 0).reshape(gh, gw, 2)
    if ok.any() and not ok.all():                  # carry the move of the last pixels of the lens outwards
        move = move[tuple(distance_transform_edt(~ok.reshape(gh, gw), return_distances=False, return_indices=True))]
    move = cv2.resize(move.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return x + move[..., 0], y + move[..., 1]


def _link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _signature(dataset_dir, cameras, images, masks_dir):
    """Changes whenever the alignment or a file of its photo and mask trees could come out differently."""
    h = hashlib.sha1(json.dumps([VERSION, cameras, str(masks_dir)], sort_keys=True).encode())
    for name, camera_id, R, C in images:
        h.update(name.encode() + np.round(np.r_[camera_id, np.ravel(R), C], 9).tobytes())
    tracks = [f for f in (dataset_dir / 'sparse/0/images.bin', dataset_dir / 'sparse/images.bin') if f.is_file()][:1]
    for root in (dataset_dir / 'images', masks_dir):
        tracks += sorted(p for p in root.rglob('*') if p.is_file()) if root is not None else []
    for path in tracks:
        s = path.stat()
        h.update(json.dumps([str(path), s.st_size, s.st_mtime_ns]).encode())
    return h.hexdigest()


def _lidar_points(dataset_dir):
    from scripts.pipeline_auto_calibrator_and_colorizer import active_slam_dir, load_pcd
    return load_pcd(active_slam_dir(dataset_dir) / 'pcd' / 'all_raw_points.pcd', return_mask=True)[0]


def _write_photos(dataset_dir, out, cameras, images, shutter, masks_dir, lidar_xyz):
    """The photo and mask trees. Returns how many photos had their rolling shutter removed."""
    import cv2
    from raven_app.dense_fusion import _voxel_thin
    from raven_app.openmvs_dense import _read_image
    written = set()
    if shutter is not None:
        lidar = np.asarray(lidar_xyz, np.float64)
        surface = lidar[_voxel_thin(lidar, 0.05)]

        def resample(k):
            name, camera_id, R, C = images[k]
            photo = _read_image(dataset_dir / 'images' / name, cv2.IMREAD_COLOR)
            _, w, h, _ = cameras[camera_id]
            if photo.shape[:2] != (h, w):
                return
            mx, my = lens_shutter_maps(cameras[camera_id], shutter[k], pr.inverse_range(surface, R, C))

            def save(tree, file, ending, pixels, how, *options):
                target = out / tree / file
                target.parent.mkdir(parents=True, exist_ok=True)
                moved = cv2.remap(pixels, mx, my, how, borderMode=cv2.BORDER_REPLICATE)
                cv2.imencode(ending, moved, *options)[1].tofile(target)

            mask = masks_dir / f'{name}.png' if masks_dir is not None else None
            if mask is not None and mask.is_file():
                keep = _read_image(mask, cv2.IMREAD_UNCHANGED)
                if keep.shape[:2] != (h, w):       # a photo and its mask move together or not at all
                    return
                save('masks', f'{name}.png', '.png', keep, cv2.INTER_NEAREST)
            save('images', name, '.jpg', photo, cv2.INTER_LANCZOS4, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            written.add(name)

        with ThreadPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as pool:
            list(pool.map(resample, [k for k in range(len(images)) if shutter[k] is not None]))
    for tree, source in (('images', dataset_dir / 'images'), ('masks', masks_dir)):
        for path in source.rglob('*') if source is not None else ():
            target = out / tree / path.relative_to(source)
            if path.is_file() and not target.exists():
                _link(path, target)
    return len(written)


def _load(out, signature):
    try:
        data = json.loads((out / 'alignment.json').read_text(encoding='utf-8'))
        poses = {n: (np.array(R), np.array(C)) for n, (R, C) in data['poses'].items()}
        if data['signature'] != signature or not all((out / 'images' / n).is_file() for n in poses):
            return None
        return AlignedPhotos(out / 'images', out / 'masks' if data['masks'] else None,
                             {n: tuple(c) for n, c in data['cameras'].items()}, poses, data['report'])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def align_photos(dataset_dir, calib_path, fps, dt_sync, masks_dir=None, log=print):
    """`AlignedPhotos` of the run, or None when its photos are used as they are (the reason is logged)."""
    from raven_app.workflow import timelapse_alignment_dt
    dataset_dir = Path(dataset_dir)
    out = dataset_dir / FOLDER
    masks_dir = Path(masks_dir) if masks_dir is not None else None
    try:
        calib = json.loads(Path(calib_path).read_text(encoding='utf-8'))
        dt = timelapse_alignment_dt(dataset_dir)
        dt = next((v for v in (dt, dt_sync, calib.get('dt_sync_seconds')) if v is not None), None)
        if dt is None:
            log('[*] Photo alignment skipped: the photos have no time offset to the scan')
            return None
        cameras, images, when = rig_poses(dataset_dir, calib, fps, dt)
        if not images:
            log('[*] Photo alignment skipped: it needs fisheye photos and their calibration')
            return None
        signature = _signature(dataset_dir, cameras, images, masks_dir)
        ready = _load(out, signature)
        if ready is not None:
            log(f"[+] Photo alignment: the aligned photos of this run are ready ({ready.report['held_out_px']:.2f} px)")
            return ready
        if not any((dataset_dir / f / 'images.bin').is_file() for f in ('sparse/0', 'sparse')):
            log('[*] Photo alignment skipped: the run has no feature tracks')
            return None
        lidar = _lidar_points(dataset_dir)
        cameras, images, shutter, report = pr.refine_photos(None, cameras, images, run_dir=dataset_dir,
                                                            lidar_xyz=lidar, when=when, log=log)
        if report['status'] != 'accepted':
            return None
        limit = MAX_HELD_OUT_PX * cameras[images[0][1]][3][0] / pr.REFERENCE_FOCAL
        if report['held_out_px'] > limit:
            log(f"[*] Photo alignment not used: {report['held_out_px']:.2f} px is more than {limit:.2f} px")
            return None
        if out.exists():
            shutil.rmtree(out)
        log('[*] Photo alignment: taking the rolling shutter out of the photos...' if shutter else
            '[*] Photo alignment: without a known motion the photos keep their pixels')
        report['photos'] = len(images)
        report['rolling_shutter_removed'] = _write_photos(dataset_dir, out, cameras, images, shutter, masks_dir, lidar)
        (out / 'alignment.json').write_text(json.dumps({
            'version': VERSION, 'signature': signature, 'report': report, 'masks': masks_dir is not None,
            'cameras': {f'cam{c - 1}': list(cameras[c]) for c in cameras},
            'poses': {n: [np.asarray(R).tolist(), np.asarray(C).tolist()] for n, _, R, C in images}}), encoding='utf-8')
        return _load(out, signature)
    except Exception as exc:                       # an optional improvement: the run's own photos always work
        log(f"[!] Photo alignment failed, the run's own photos are used: {exc}")
        return None
