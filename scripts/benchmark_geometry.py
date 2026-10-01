"""Compare SLAM surfaces on fixed patches and camera trajectory agreement."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raven_app.geometry_quality import (rigid_registration, select_surface_patches,
    score_surface_patches, summarize_paired_surfaces)


def sample_pcd(path, maximum=2000000):
    """Memory-map native binary PCD and sample without loading the full cloud."""
    with Path(path).open('rb') as stream:
        header = {}
        for _ in range(100):
            line = stream.readline().decode('ascii').strip().split()
            if not line or line[0].startswith('#'):
                continue
            header[line[0]] = line[1:]
            if line[0] == 'DATA':
                break
        if header.get('DATA') != ['binary']:
            raise ValueError('Expected binary uncompressed PCD')
        types = {'F': 'f', 'I': 'i', 'U': 'u'}
        fields = []
        counts = header.get('COUNT', ['1'] * len(header['FIELDS']))
        for name, size, kind, count in zip(header['FIELDS'], header['SIZE'], header['TYPE'], counts):
            dtype = '<' + types[kind] + size
            fields.append((name, dtype) if int(count) == 1 else (name, dtype, (int(count),)))
        count, offset = int(header['POINTS'][0]), stream.tell()
    dtype = np.dtype(fields)
    if Path(path).stat().st_size != offset + count * dtype.itemsize:
        raise ValueError('Invalid PCD payload length')
    data = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=(count,))
    indices = np.sort(np.random.default_rng(42).choice(count, min(count, maximum), replace=False))
    points = np.column_stack([data[name][indices] for name in ('x', 'y', 'z')]).astype(float)
    points = points[np.isfinite(points).all(axis=1)]
    # Deterministic random partition; neighboring acquisition rows are not forced together.
    np.random.default_rng(17).shuffle(points)
    return points, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--reference-slam', type=Path, required=True)
    parser.add_argument('--candidate-slam', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-points', type=int, default=2000000)
    args = parser.parse_args()
    from scripts import pipeline_auto_calibrator_and_colorizer as pipeline
    from raven_app.timelapse_calibration import validate_lever_refinement
    images = sorted([im for im in pipeline.load_colmap_images(args.dataset / 'sparse/0/images.bin')
                     if im['name'].startswith('cam0/')], key=lambda im: im['name'])
    centers = np.array([im['C'] for im in images])
    rotations = np.array([im['R_cw'] for im in images])
    times = np.array([pipeline.frame_time(args.dataset, im['name']) for im in images])
    def trajectory(root):
        return pipeline.load_trajectory(next((root / 'result').glob('*.txt')))
    reference_ts, reference_xyz, _ = trajectory(args.reference_slam)
    points, count = sample_pcd(args.reference_slam / 'pcd/all_raw_points.pcd', args.sample_points)
    half = len(points) // 2
    patches = select_surface_patches(points[:half])
    reference_rows = score_surface_patches(points[:half], points[half:], patches)
    del points
    report = {'dataset': str(args.dataset.resolve()), 'reference': str(args.reference_slam.resolve()),
              'reference_point_count': count, 'patch_centers_m': patches.tolist(),
              'radius_m': .35, 'sample_points': args.sample_points, 'results': {},
              'policy': 'Same reference patches; train/test points disjoint; no residual trimming; rigid frame alignment without scale. Camera CV is trajectory agreement, not surveyed accuracy.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for root in [args.reference_slam, *args.candidate_slam]:
        ts, xyz, rots = trajectory(root)
        cv = validate_lever_refinement(centers, rotations, times, ts, xyz, rots, dt_hint=3.6)
        if root == args.reference_slam:
            rows, total = reference_rows, count
        else:
            shared = (ts >= reference_ts[0]) & (ts <= reference_ts[-1])
            target = np.column_stack([np.interp(ts[shared], reference_ts, reference_xyz[:, i]) for i in range(3)])
            r, t = rigid_registration(xyz[shared], target)
            points, total = sample_pcd(root / 'pcd/all_raw_points.pcd', args.sample_points)
            points = points @ r.T + t
            half = len(points) // 2
            rows = score_surface_patches(points[:half], points[half:], patches)
            del points
        result = {'slam': str(root.resolve()), 'points': total,
                  'surface_comparison': summarize_paired_surfaces(reference_rows, rows),
                  'surface_patches': rows, 'trajectory_validation': cv,
                  'camera_cv_rmse_cm': cv['candidate_untrimmed_rmse_cm'] if cv['accepted']
                      else cv['baseline_untrimmed_rmse_cm']}
        if root == args.reference_slam:
            reference_cv = result['camera_cv_rmse_cm']
        result['eligible_for_promotion'] = bool(root != args.reference_slam and
            result['surface_comparison']['improved'] and total >= .99 * count and
            result['camera_cv_rmse_cm'] <= reference_cv)
        report['results'][str(root.resolve())] = result
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(root.name, json.dumps(result['surface_comparison']), flush=True)
        print('Camera trajectory blocked CV (cm):', result['camera_cv_rmse_cm'], flush=True)
    eligible = [value for value in report['results'].values() if value['eligible_for_promotion']]
    report['recommended_slam'] = (min(eligible, key=lambda value:
        value['surface_comparison']['candidate_rmse_mm'])['slam'] if eligible else str(args.reference_slam.resolve()))
    report['promotion_policy'] = 'At least 3% lower surface RMSE, non-worse patch P90 and camera CV, >=95% patch coverage, consistent normals and >=99% total point count.'
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
