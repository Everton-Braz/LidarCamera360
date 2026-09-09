"""Vulkan subprocess bridge. Poses are prepared once by the existing calibration code."""
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from raven_app.config import get_app_root


def get_vulkan_bin():
    root = get_app_root()
    candidates = [root / p for p in ('bin/vulkan_colorizer.exe',
        'build/native/vulkan_colorizer/Release/vulkan_colorizer.exe',
        'build/vulkan/Release/vulkan_colorizer.exe', 'native/vulkan_colorizer.exe')]
    return next((p for p in candidates if p.is_file()), candidates[0])


def vulkan_status():
    executable = get_vulkan_bin()
    try:
        result = subprocess.run([str(executable), '--probe'], capture_output=True,
                                text=True, timeout=15)
        return {'path': str(executable), 'ready': result.returncode == 0,
                'detail': (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'path': str(executable), 'ready': False, 'detail': str(exc)}


def colorize_views(points, views, work_dir, device_id=-1):
    """Return RGB8, or None on native failure so callers can run the CPU fallback.

    Protocol RVC1: uint32 point/view counts, packed float4 points; each view is
    a UTF-8 path prefixed by uint32 length followed by 24 float32 values
    (row-major Rcw, Cworld, COLMAP's 12 intrinsic parameters).
    """
    if not views or not len(points):
        return None
    try:
        with tempfile.TemporaryDirectory(prefix='vulkan-', dir=work_dir) as tmp:
            job, out = Path(tmp) / 'job.bin', Path(tmp) / 'rgb.bin'
            with job.open('wb') as stream:
                stream.write(struct.pack('<4sII', b'RVC1', len(points), len(views)))
                # Shift large survey coordinates before float32 conversion.
                origin = np.asarray(points[0], dtype=np.float64)
                packed = np.zeros((len(points), 4), dtype='<f4')
                packed[:, :3] = points - origin
                packed.tofile(stream)
                del packed
                for path, rotation, center, params in views:
                    if len(params) != 12:
                        raise ValueError('Vulkan requires THIN_PRISM_FISHEYE (12 parameters)')
                    name = str(Path(path).resolve()).encode('utf-8')
                    values = np.concatenate((np.asarray(rotation).ravel(), np.asarray(center)-origin, params))
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite camera parameters')
                    stream.write(struct.pack('<I', len(name)) + name)
                    values.astype('<f4').tofile(stream)
            result = subprocess.run([str(get_vulkan_bin()), '--job', str(job), '--out', str(out),
                                     '--device', str(device_id)])
            if result.returncode != 0 or not out.is_file() or out.stat().st_size != len(points)*4:
                raise RuntimeError(f'Vulkan failed or returned incomplete output ({result.returncode})')
            return np.fromfile(out, dtype=np.uint8).reshape(-1, 4)[:, :3].copy()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'[!] Vulkan unavailable: {exc}. Falling back to CPU.')
        return None


def run_vulkan_colorizer(pcd_path, colmap_dir, images_dir, output_ply,
                         alignment_json=None, device_id=-1):
    from scripts.pipeline_auto_calibrator_and_colorizer import (
        load_pcd, load_colmap_cameras, load_colmap_images, write_ply, write_pcd)
    cameras = load_colmap_cameras(Path(colmap_dir) / 'cameras.bin')
    images = load_colmap_images(Path(colmap_dir) / 'images.bin')
    alignment = json.loads(Path(alignment_json).read_text()) if alignment_json else {
        'scale': 1, 'R': np.eye(3).tolist(), 't': [0, 0, 0]}
    rotation, translation = np.array(alignment['R']), np.array(alignment['t'])
    points = load_pcd(pcd_path)
    views = [(Path(images_dir)/im['name'], im['R_cw'] @ rotation.T,
              alignment['scale'] * (rotation @ im['C']) + translation,
              cameras[im['cam_id']]['params']) for im in sorted(images, key=lambda im: im['name'])]
    output_ply = Path(output_ply)
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    rgb = colorize_views(points, views, output_ply.parent, device_id)
    if rgb is None:
        return False
    write_ply(output_ply, points, rgb)
    write_pcd(output_ply.with_suffix('.pcd'), points, rgb)
    return True
