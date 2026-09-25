"""Validate mask pairs and produce a compact visual QA contact sheet."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raven_app.person_masks import load_keep_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = args.dataset / 'images'
    rows, summary = [], []
    for camera in ('cam0', 'cam1'):
        paths = sorted((images / camera).glob('*.jpg'))
        selected = set(np.linspace(0, len(paths) - 1, min(4, len(paths)), dtype=int))
        for index, path in enumerate(paths):
            mask = load_keep_mask(path, images, args.dataset/'masks')
            image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            if image is None or mask.shape != image.shape[:2] or not np.isin(mask, [0, 255]).all():
                raise ValueError(f'Invalid mask pair: {path}')
            summary.append({'image': path.relative_to(images).as_posix(),
                            'width': mask.shape[1], 'height': mask.shape[0],
                            'excluded_percent': 100 * float(np.mean(mask == 0))})
            if index in selected:
                small = cv2.resize(image, (480, 480))
                hit = cv2.resize(mask, (480, 480), interpolation=cv2.INTER_NEAREST) == 0
                overlay = small.copy()
                overlay[hit] = (.4 * overlay[hit] + .6 * np.array([0, 0, 255])).astype(np.uint8)
                cv2.putText(overlay, f'{camera}/{path.name}', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                rows.append(np.hstack((small, overlay)))
    if rows:
        cv2.imwrite(str(args.output / 'person_masks_preview.jpg'), np.vstack(rows))
    report = {'image_count': len(summary), 'polarity': 'white_keep_black_person', 'images': summary}
    (args.output / 'mask_quality_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Validated {len(summary)} masks. Preview: {args.output / "person_masks_preview.jpg"}')


if __name__ == '__main__':
    main()
