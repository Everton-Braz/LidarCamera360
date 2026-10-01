"""Fit PPISP/grid ablations from one cached, point-split observation set."""
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
    parser.add_argument('--observations', type=Path, required=True,
                        help='Cached NPZ from photometric dataset sampling')
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Log-linear parameter JSON for the same observations')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260929)
    parser.add_argument('--grid-smoothness', type=float, default=2.0)
    args = parser.parse_args()

    from raven_app.photometric.advanced import fit_advanced
    from raven_app.photometric.model import load_model
    from raven_app.photometric.report import write_report

    required = ('point_id', 'frame_id', 'lens_id', 'radius', 'rgb', 'weight',
                'u', 'v', 'names', 'lens_names')
    with np.load(args.observations, allow_pickle=False) as archive:
        missing = sorted(set(required) - set(archive.files))
        if missing:
            parser.error(f'observation file is missing fields: {", ".join(missing)}')
        observations = {key: archive[key] for key in required
                        if key not in ('names', 'lens_names')}
        names = archive['names'].tolist()
        lens_names = archive['lens_names'].tolist()
    baseline = load_model(args.baseline)
    started = time.perf_counter()
    model = fit_advanced(observations, names, lens_names, baseline,
                         seed=args.seed, grid_smoothness=args.grid_smoothness)
    model['ablation_source'] = {
        'observations': str(args.observations.resolve()),
        'baseline_model': str(args.baseline.resolve()),
        'point_split_seed': args.seed,
    }
    model['timing']['ablation_total_seconds'] = round(time.perf_counter() - started, 3)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params_path = args.output_dir / 'ablation_params.json'
    summary_path = args.output_dir / 'ablation_summary.json'
    report_path = args.output_dir / 'ablation_report.html'
    params_path.write_text(json.dumps(model, separators=(',', ':')), encoding='utf-8')
    write_report(model, report_path)
    summary = {
        'selected': model['selected'],
        'advanced_model_revision': model['advanced_model_revision'],
        'grid_smoothness': args.grid_smoothness,
        'metrics': model['metrics'],
        'timing': model['timing'],
        'params': str(params_path.resolve()),
        'report': str(report_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
