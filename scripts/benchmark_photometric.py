"""Compare native colorization on identical XYZ, preserving the input cloud."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--cloud', type=Path, required=True, help='Application binary XYZ/RGB PLY')
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--alignment', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--masks-dir', type=Path)
    args = parser.parse_args()
    from raven_app.photometric.dataset import prepare_model
    from raven_app.vulkan_engine import colorize_views
    from scipy.spatial.transform import Slerp
    from scripts.pipeline_auto_calibrator_and_colorizer import (load_colmap_cameras,
        load_colmap_images, write_ply, frame_time, get_slam_trajectory_path,
        load_trajectory)
    args.output.mkdir(parents=True, exist_ok=True)
    model = prepare_model(args.dataset, args.params, args.masks_dir)
    with args.cloud.open('rb') as stream:
        header = []
        for _ in range(50):
            line = stream.readline().decode('ascii').strip()
            header.append(line)
            if line == 'end_header':
                break
        properties = [s for s in header if s.startswith('property ')]
        if properties != ['property float x', 'property float y', 'property float z',
                          'property uchar red', 'property uchar green', 'property uchar blue']:
            raise ValueError('Expected the application binary XYZ/RGB PLY layout')
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Expected binary little endian PLY')
        count = int(next(s.split()[-1] for s in header if s.startswith('element vertex ')))
        data = np.fromfile(stream, dtype=[('xyz', '<f4', (3,)), ('rgb', 'u1', (3,))], count=count)
        if len(data) != count:
            raise ValueError('Truncated input PLY')
    points = data['xyz'].astype(np.float64)
    del data
    alignment = json.loads((args.alignment or args.dataset / 'colmap_to_lidar_alignment.json').read_text())
    rotation, translation = np.array(alignment['R']), np.array(alignment['t'])
    cameras = load_colmap_cameras(args.dataset / 'sparse/0/cameras.bin')
    images = load_colmap_images(args.dataset / 'sparse/0/images.bin')
    # Do not call recalibrate_from_sfm here: that production helper writes
    # rig_calibration.json inside the source dataset. The benchmark is read
    # only and uses the already exported calibration sidecar instead.
    overrides = {}
    if alignment.get('frame_time_source') == 'insv_timelapse':
        calibration_candidates = [args.dataset / 'rig_calibration.json']
        calibration_candidates += sorted((args.dataset / 'deliverables').glob('rig_calibration_*.json'))
        calibration_path = next((path for path in calibration_candidates if path.is_file()), None)
        if calibration_path is None:
            raise FileNotFoundError('No exported rig calibration sidecar for read-only benchmark')
        calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
        trajectory_times, trajectory_positions, trajectory_rotations = load_trajectory(
            get_slam_trajectory_path(args.dataset))
        interpolate = Slerp(trajectory_times - trajectory_times[0], trajectory_rotations)
        rejected = set(alignment.get('rejected_image_names', []))
        for image in images:
            if image['name'] not in rejected:
                continue
            query = frame_time(args.dataset, image['name']) - alignment['dt_sync_seconds']
            if not 0 <= query <= trajectory_times[-1] - trajectory_times[0]:
                overrides[image['name']] = None
                continue
            lens = 0 if image['name'].startswith('cam0/') else 1
            transform = np.asarray(calibration[f'T_lidar_to_cam{lens}_rigid_4x4'])
            body_rotation = interpolate(query).as_matrix()
            body = np.array([np.interp(query, trajectory_times - trajectory_times[0],
                                       trajectory_positions[:, axis]) for axis in range(3)])
            overrides[image['name']] = ((body_rotation @ transform[:3, :3]).T,
                                        body + body_rotation @ transform[:3, 3])
    views = []
    for im in sorted(images, key=lambda im: im['name']):
        if im['name'] in overrides:
            if overrides[im['name']] is None:
                continue
            r, c = overrides[im['name']]
        else:
            r = im['R_cw'] @ rotation.T
            c = alignment['scale'] * (rotation @ im['C']) + translation
        views.append((args.dataset / 'images' / im['name'], r, c, cameras[im['cam_id']]['params']))
    results = {'points': len(points), 'views': len(views), 'alignment': str(args.alignment),
               'photometric_metrics': model['metrics'], 'fit_timing': model['timing'],
               'note': 'Identical input XYZ for off/on; timings include native I/O, exclude PLY export.'}
    for name, coefficients in (('before', None), ('after', model['views'])):
        started = time.perf_counter()
        colors = colorize_views(points, views, args.output, masks_dir=args.masks_dir,
                                images_dir=args.dataset / 'images', photometric=coefficients)
        if colors is None:
            raise RuntimeError('Native benchmark failed; refusing incomparable CPU fallback')
        results[name + '_seconds'] = time.perf_counter() - started
        write_ply(args.output / (name + '.ply'), points, colors)
        del colors
        (args.output / 'benchmark.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
