"""Read back filtered cloud geometry and quantify color changes against a baseline."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raven_app.cloud_io import load_cloud


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--filtered', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    base, filtered = load_cloud(args.baseline), load_cloud(args.filtered)
    base_keys = np.ascontiguousarray(base.points, dtype='<f4').view('V12').ravel()
    filtered_keys = np.ascontiguousarray(filtered.points, dtype='<f4').view('V12').ravel()
    _, base_indices, filtered_indices = np.intersect1d(base_keys, filtered_keys, return_indices=True)
    if not np.isin(filtered_keys, base_keys).all():
        raise ValueError('Output contains geometry absent from the source cloud')
    delta = np.max(np.abs(base.colors[base_indices] - filtered.colors[filtered_indices]), axis=1) * 255
    colors = np.rint(filtered.colors * 255).astype(np.uint8)
    report = {
        'baseline': str(args.baseline.resolve()), 'filtered': str(args.filtered.resolve()),
        'input_points': len(base.points), 'output_points': len(filtered.points),
        'removed_points': len(base.points) - len(filtered.points),
        'matched_unique_positions': len(base_indices),
        'matched_positions_with_color_change_over_10_levels': int(np.count_nonzero(delta > 10)),
        'unobserved_gray_points': int(np.count_nonzero(np.all(colors == 180, axis=1))),
        'all_output_positions_present_in_input': True,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
