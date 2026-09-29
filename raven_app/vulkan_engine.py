"""Vulkan subprocess bridge. Poses are prepared once by the existing calibration code."""
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from raven_app.config import get_app_root
from raven_app.subprocess_utils import hidden_window_options, run_hidden_stream


def get_vulkan_bin():
    root = get_app_root()
    candidates = [root / p for p in ('build/vulkan/Release/vulkan_colorizer.exe',
        'build/native/vulkan_colorizer/Release/vulkan_colorizer.exe',
        'bin/vulkan_colorizer.exe', 'native/vulkan_colorizer.exe')]
    return next((p for p in candidates if p.is_file()), candidates[0])


def vulkan_status():
    executable = get_vulkan_bin()
    try:
        result = subprocess.run([str(executable), '--probe'], capture_output=True,
                                text=True, timeout=15, **hidden_window_options())
        return {'path': str(executable), 'ready': result.returncode == 0,
                'detail': (result.stdout + result.stderr).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'path': str(executable), 'ready': False, 'detail': str(exc)}


def _photo_key(value):
    """Canonicalize a camera-relative image name to camN/frame.ext."""
    parts = [part for part in str(value).replace('\\', '/').split('/')
             if part not in ('', '.', '..')]
    camera_index = next((i for i in range(len(parts) - 1, -1, -1)
                         if parts[i].casefold() in ('cam0', 'cam1')), None)
    if camera_index is not None:
        parts = parts[camera_index:]
    elif len(parts) > 2:
        parts = parts[-2:]
    return '/'.join(parts).casefold()


def _view_photometric(path, photometric, images_dir):
    """Return (gain RGB, vignette k1/k2/k3) for one normalized view path."""
    image = Path(path)
    if images_dir is not None:
        try:
            name = image.resolve().relative_to(Path(images_dir).resolve()).as_posix()
        except ValueError:
            name = image.as_posix()
    else:
        name = image.as_posix()
    key = _photo_key(name)
    entry = photometric.get(key)
    if entry is None:
        return np.zeros(6, dtype='<f4')
    if not isinstance(entry, dict):
        raise ValueError(f'Invalid photometric coefficients for {key}')
    gain = np.asarray(entry.get('log_gain', [0, 0, 0]), dtype=np.float64)
    vignette = np.asarray(entry.get('vignette', [0, 0, 0]), dtype=np.float64)
    values = np.concatenate((gain, vignette))
    if (gain.shape != (3,) or vignette.shape != (3,) or
            not np.isfinite(values).all() or np.max(np.abs(values)) > 4):
        raise ValueError(f'Invalid photometric coefficients for {key}')
    return values.astype('<f4')


def colorize_views(points, views, work_dir, device_id=-1, masks_dir=None, images_dir=None,
                   photometric=None):
    """Return RGB8, or None on native failure so callers can run the CPU fallback.

    Protocol RVC1/RVC2: uint32 point/view counts, packed float4 points; each view is
    a UTF-8 path prefixed by uint32 length followed by 24 float32 values
    (row-major Rcw, Cworld, COLMAP's 12 intrinsic parameters). RVC3 adds six
    photometric values per view; RVC4 combines those values with person masks.
    The six values are three log gains and three achromatic log-vignette coefficients.
    """
    if masks_dir is not None and images_dir is None:
        raise ValueError('images_dir is required for person masks')
    if photometric is not None and not isinstance(photometric, dict):
        raise ValueError('photometric must map normalized camera-relative image names to coefficients')
    if not views or not len(points):
        return None
    try:
        photo_values = ([_view_photometric(view[0], photometric, images_dir)
                         for view in views] if photometric is not None else None)
        with tempfile.TemporaryDirectory(prefix='vulkan-', dir=work_dir) as tmp:
            job, out = Path(tmp) / 'job.bin', Path(tmp) / 'rgb.bin'
            with job.open('wb') as stream:
                magic = (b'RVC4' if photometric is not None and masks_dir is not None else
                         b'RVC3' if photometric is not None else
                         b'RVC2' if masks_dir is not None else b'RVC1')
                stream.write(struct.pack('<4sII', magic, len(points), len(views)))
                # Shift large survey coordinates before float32 conversion.
                origin = np.asarray(points[0], dtype=np.float64)
                packed = np.zeros((len(points), 4), dtype='<f4')
                packed[:, :3] = points - origin
                packed.tofile(stream)
                del packed
                for view_i, (path, rotation, center, params) in enumerate(views):
                    if len(params) != 12:
                        raise ValueError('Vulkan requires THIN_PRISM_FISHEYE (12 parameters)')
                    name = str(Path(path).resolve()).encode('utf-8')
                    values = np.concatenate((np.asarray(rotation).ravel(), np.asarray(center)-origin, params))
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite camera parameters')
                    stream.write(struct.pack('<I', len(name)) + name)
                    values.astype('<f4').tofile(stream)
                    if masks_dir is not None:
                        from raven_app.person_masks import mask_path
                        mask = mask_path(path, images_dir, masks_dir)
                        if not mask.is_file():
                            raise ValueError(f'Missing person mask: {mask}')
                        encoded = str(mask.resolve()).encode('utf-8')
                        stream.write(struct.pack('<I', len(encoded)) + encoded)
                    if photo_values is not None:
                        photo_values[view_i].tofile(stream)
            code = run_hidden_stream([str(get_vulkan_bin()), '--job', str(job), '--out', str(out),
                                      '--device', str(device_id)])
            if code != 0 or not out.is_file() or out.stat().st_size != len(points)*4:
                raise RuntimeError(f'Vulkan failed or returned incomplete output ({code})')
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
