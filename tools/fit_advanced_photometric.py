"""Fit PPISP-style homographies and a small bilateral grid from cached tracks."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raven_app.photometric.advanced import fit_advanced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='schema-1 gains/vignette model JSON')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observations_path = args.observations.resolve()
    baseline_path = args.baseline.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with np.load(observations_path, allow_pickle=False) as packed:
        observations = {key: packed[key] for key in packed.files
                        if key not in ('names', 'lens_names')}
        names = packed['names'].astype(str).tolist()
        lens_names = packed['lens_names'].astype(str).tolist()
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    model = fit_advanced(observations, names, lens_names, baseline)
    model['advanced_source'] = {
        'observations': str(observations_path),
        'baseline_model': str(baseline_path),
    }
    args.output.write_text(json.dumps(model, separators=(',', ':')),
                           encoding='utf-8')
    metrics = model['metrics']
    print(json.dumps({
        'output': str(args.output.resolve()),
        'selected': model['selected'],
        'validation_before': metrics['validation_before'],
        'validation_after': metrics['validation_after'],
        'test_before': metrics['test_before'],
        'test_after': metrics['test_after'],
        'timing': model['timing'],
    }, indent=2))


if __name__ == '__main__':
    main()
