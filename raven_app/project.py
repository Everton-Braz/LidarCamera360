"""The LidarCamera360 project: one per dataset, kept in the output folder, updated by every run.

A run produces *deliverables* (by default only the coloured PLY and the 3DGS/COLMAP dataset). The project is what stays
behind: `project.json` (schema `lc360-project` 1.x, the same file LidarCamera360 Pro opens), the camera poses derived
from the COLMAP export, and a list of the runs. Every other format (PCD, LAZ/LAS, GeoJSON, GPS tracks, trajectories) is
exported from the project afterwards with `export_from_project`, so a run never has to guess what the user will want.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from raven_app.project_validate import validate_project

SCHEMA = 'lc360-project'
VERSION = '1.0'
PROJECT_FILE = 'project.json'
EXPORT_FORMATS = ('pcd', 'laz', 'las', 'georef', 'gps', 'trajectory')
EXPORT_LABELS = {
    'pcd': 'Coloured cloud (.PCD)', 'laz': 'Coloured cloud (.LAZ)', 'las': 'Coloured cloud (.LAS)',
    'georef': 'Georeferenced cloud (LAZ + GeoJSON, needs INSV GPS)', 'gps': 'GPS track (GeoJSON, GPX, CSV)',
    'trajectory': 'SLAM trajectory (TUM + CSV)'}


_UUID4 = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')


class ManifestError(ValueError):
    """The manifest would not pass the v1 validator; nothing was written. `.findings` lists the findings."""

    def __init__(self, findings):
        self.findings = list(findings)
        super().__init__('project.json not written: ' + '; '.join(f'{f.code} {f.path}' for f in self.findings))


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def active_slam_dir(project_dir: Path) -> Path | None:
    project_dir = Path(project_dir)
    try:
        rel = json.loads((project_dir / 'active_slam.json').read_text(encoding='utf-8')).get('relative_directory')
        if rel and (project_dir / rel).is_dir():
            return project_dir / rel
    except (OSError, ValueError):
        pass
    found = sorted(project_dir.glob('slam_raw_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def find_trajectory(project_dir: Path) -> Path | None:
    slam = active_slam_dir(project_dir)
    if slam is None:
        return None
    files = sorted((slam / 'result').glob('*.txt')) if (slam / 'result').is_dir() else []
    return files[0] if files else None


def find_colored_cloud(project_dir: Path) -> Path | None:
    """Newest coloured PLY among the deliverables (the georeferenced copies are not it)."""
    plys = [p for p in (Path(project_dir) / 'deliverables').glob('*.ply') if p.is_file()]
    return max(plys, key=lambda p: p.stat().st_mtime) if plys else None


def poses_from_colmap(colmap_dir: Path, frames_json: Path | None = None, dt_sync: float = 0.0,
                      trajectory: Path | None = None) -> dict:
    """Pro's `camera_poses` document from a COLMAP text model (cameras.txt + images.txt)."""
    from scipy.spatial.transform import Rotation
    colmap_dir = Path(colmap_dir)
    cameras = {}
    for line in (colmap_dir / 'cameras.txt').read_text(encoding='utf-8').splitlines():
        w = line.split()
        if w and not line.startswith('#'):
            if w[1] != 'THIN_PRISM_FISHEYE':
                raise ValueError(f'camera model {w[1]} is not supported by the project exporter')
            p = [float(v) for v in w[4:]]
            cameras[int(w[0])] = dict(zip(('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1'), p),
                                      width=int(w[2]), height=int(w[3]))
    times = {}
    if frames_json is not None and Path(frames_json).is_file():
        times = json.loads(Path(frames_json).read_text(encoding='utf-8')).get('timestamps', {})
    epoch0 = None
    if trajectory is not None and Path(trajectory).is_file():
        with open(trajectory, encoding='utf-8') as f:
            first = next((l for l in f if l.strip() and not l.startswith('#')), None)
        epoch0 = float(first.split()[0]) if first else None
    lens_of, images = {}, []
    rows = [l for l in (colmap_dir / 'images.txt').read_text(encoding='utf-8').splitlines()
            if l.strip() and not l.startswith('#')]
    for line in rows[::2]:
        w = line.split()
        if len(w) < 10:
            continue
        q = np.array(w[1:5], float)
        t = np.array(w[5:8], float)
        name = w[9]
        cam, base = name.split('/')[0], Path(name).stem
        lens_of[cam] = cameras[int(w[8])]
        rot_cw = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        rot_wc = rot_cw.inv()
        position = -(rot_wc.apply(t))
        forward = rot_wc.apply([0, 0, 1])
        frame = int(''.join(c for c in base if c.isdigit()) or 0)
        seconds = times.get(name)
        stamp = (epoch0 + float(seconds) - dt_sync) if (seconds is not None and epoch0 is not None) else \
            float(seconds) if seconds is not None else float(frame)
        images.append(dict(image=name, cam=cam, frame=frame, t=stamp,
                           position=[round(float(v), 4) for v in position],
                           q_world_cam=[round(float(v), 6) for v in rot_wc.as_quat()],
                           forward=[round(float(v), 4) for v in forward],
                           yaw_deg=round(float(math.degrees(math.atan2(forward[0], forward[1])) % 360), 1),
                           pitch_deg=round(float(math.degrees(math.asin(max(-1, min(1, forward[2]))))), 2)))
    images.sort(key=lambda im: (im['frame'], im['cam']))
    return dict(frame='LiDAR SLAM world (m)', camera_axes='COLMAP: +Z forward, +X right, +Y down',
                lens_model='THIN_PRISM_FISHEYE', intrinsics=lens_of, chain='COLMAP export of the run',
                time='epoch seconds (LiDAR clock)' if epoch0 is not None else 'video seconds', images=images)


def _package_rig_profile(root: Path, rig_profile) -> str:
    """Return a package-relative path for the rig profile, copying an outside profile into the package.

    Callers pass whatever profile the run used, often an absolute path in the app's resources (a PyInstaller temp
    folder in the portable build). The manifest may only hold paths inside the package, so the file is copied to
    `rig_profile/<name>`. A profile that cannot be found is left out rather than failing the whole manifest.
    """
    if not rig_profile:
        return ''
    src = Path(rig_profile)
    if not src.is_absolute():
        src = root / src
    if not src.is_file():
        return ''
    try:
        return _rel(root, src)
    except ValueError:
        pass
    dest = root / 'rig_profile' / src.name
    dest.parent.mkdir(exist_ok=True)
    if not (dest.is_file() and dest.read_bytes() == src.read_bytes()):
        shutil.copyfile(src, dest)
    return _rel(root, dest)


def write_project(project_dir, *, dt_sync: float = 0.0, scanner: str = '', rig_profile: str = '', insv: str = '',
                  run: dict | None = None, app_version: str = '') -> dict:
    """Create or refresh `project.json` in the output folder; returns the manifest."""
    root = Path(project_dir)
    manifest_path = root / PROJECT_FILE
    try:
        old = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    except (OSError, ValueError):
        old = {}
    cloud = find_colored_cloud(root)
    assets = dict(old.get('assets', {}))
    if cloud is not None:
        assets['cloud'] = _rel(root, cloud)
    elif 'cloud' not in assets:
        raise ValueError('no coloured cloud in deliverables/ to build the project from')
    trajectory = find_trajectory(root)
    if trajectory is not None:
        assets['trajectory'] = _rel(root, trajectory)
    colmap = root / 'colmap_3dgs'
    if (colmap / 'sparse' / '0' / 'images.txt').is_file():
        assets['colmap'] = _rel(root, colmap / 'sparse' / '0')
        if (colmap / 'images').is_dir():
            assets['images'] = _rel(root, colmap / 'images')
        poses = root / 'cameras' / 'poses.json'
        poses.parent.mkdir(exist_ok=True)
        poses.write_text(json.dumps(poses_from_colmap(
            colmap / 'sparse' / '0', root / 'images' / 'frames.json', dt_sync, trajectory)), encoding='utf-8')
        assets['camera_poses'] = _rel(root, poses)
    if (colmap / 'points3D.ply').is_file():
        assets['gaussian_seed'] = _rel(root, colmap / 'points3D.ply')
    rig_profile = _package_rig_profile(root, rig_profile) or (old.get('capture') or {}).get('rig_profile', '')
    runs = list(old.get('runs', []))
    if run:
        runs = [r for r in runs if r.get('run_id') != run.get('run_id')] + [run]
    old_id = old.get('id')
    manifest = {
        'schema': SCHEMA, 'version': VERSION,
        'id': old_id if isinstance(old_id, str) and _UUID4.match(old_id) else str(uuid.uuid4()),
        'created_by': {'app': 'LidarCamera360', 'version': app_version or 'dev'},
        'units': 'm', 'frame': old.get('frame') or {'name': 'lidar_slam_world', 'up': '+Z'},
        'capture': {k: v for k, v in {'scanner': scanner, 'rig_profile': rig_profile, 'insv': str(insv) if insv else
                                                    (old.get('capture') or {}).get('insv', '')}.items() if v},
        'assets': assets, 'dt_sync_seconds': float(dt_sync),
        'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'runs': runs,
    }
    for key in ('crs', 'quality'):
        if key in old:
            manifest[key] = old[key]
    errors = validate_project(root, manifest).errors
    if errors:
        raise ManifestError(errors)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def export_from_project(project_dir, formats, dest=None, log=print) -> dict:
    """Write extra formats from the project into `dest` (default: project/exports). Returns {format: [files]}."""
    root = Path(project_dir)
    dest = Path(dest) if dest else root / 'exports'
    unknown = [f for f in formats if f not in EXPORT_FORMATS]
    if unknown:
        raise ValueError(f'unknown export formats {unknown}; choose from {EXPORT_FORMATS}')
    dest.mkdir(parents=True, exist_ok=True)
    try:
        manifest = json.loads((root / PROJECT_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        manifest = {}
    insv = manifest.get('capture', {}).get('insv') or ''
    made: dict[str, list[str]] = {}
    cloud = find_colored_cloud(root)
    cloud_data = None
    for fmt in (f for f in formats if f in ('pcd', 'laz', 'las')):
        if cloud is None:
            raise ValueError('the project has no coloured cloud to convert')
        from raven_app.cloud_export import save_cloud
        from raven_app.cloud_io import load_cloud
        cloud_data = cloud_data or load_cloud(cloud)
        out = dest / f'{cloud.stem}.{fmt}'
        save_cloud(cloud_data, out, formats=fmt, overwrite=True)
        made[fmt] = [str(out)]
        log(f'[+] {out}')
    if 'trajectory' in formats:
        traj = find_trajectory(root)
        if traj is None:
            raise ValueError('the project has no SLAM trajectory')
        out_tum = dest / 'trajectory_tum.txt'
        shutil.copyfile(traj, out_tum)
        out_csv = dest / 'trajectory.csv'
        np.savetxt(out_csv, np.loadtxt(traj, comments='#', ndmin=2), delimiter=',',
                   header='t,x,y,z,qx,qy,qz,qw', comments='', fmt='%.9f')
        made['trajectory'] = [str(out_tum), str(out_csv)]
        log(f'[+] {out_tum}')
    if 'gps' in formats or 'georef' in formats:
        if not insv or not Path(insv).is_file():
            raise ValueError('GPS and georeferencing need the original INSV video; it was not recorded in this '
                             'project or is no longer at its place')
    if 'gps' in formats:
        from raven_app.workflow import process_gps_metadata
        report = process_gps_metadata(insv, dest / 'gps', ('geojson', 'gpx', 'csv'))
        if report.get('status') != 'complete':
            raise ValueError('the INSV carries no valid GPS fixes')
        made['gps'] = [str(p) for p in (dest / 'gps').rglob('*') if p.is_file()]
        log(f"[+] GPS: {report.get('unique_fixes')} fixes in {dest / 'gps'}")
    if 'georef' in formats:
        if cloud is None:
            raise ValueError('the project has no coloured cloud to georeference')
        from raven_app.automatic_georeference import automatic_georeference
        from raven_app.georeference import calculate_insv_gps
        out = dest / 'georeferenced'
        automatic_georeference(insv, out, candidate_clouds=[cloud], formats=('laz', 'geojson'),
                               trajectory_path=find_trajectory(root), gps_info=calculate_insv_gps(insv),
                               dt_sync=manifest.get('dt_sync_seconds'))
        made['georef'] = [str(p) for p in out.rglob('*') if p.is_file()]
        log(f'[+] Georeferenced outputs in {out}')
    return made
