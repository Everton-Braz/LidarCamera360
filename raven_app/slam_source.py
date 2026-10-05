"""Resolve a consistently selected cloud/trajectory pair for downstream stages."""
import json
from pathlib import Path


def active_slam_dir(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    manifest = dataset_dir/'active_slam.json'
    if not manifest.is_file():
        return dataset_dir/'slam_out'
    data = json.loads(manifest.read_text(encoding='utf-8'))
    selected = (dataset_dir/data['relative_directory']).resolve()
    if dataset_dir not in selected.parents:
        raise ValueError('Active SLAM directory must be inside the dataset output')
    if not (selected/'pcd/all_raw_points.pcd').is_file() or not list((selected/'result').glob('*.txt')):
        raise FileNotFoundError('Selected SLAM cloud/trajectory pair is incomplete')
    return selected
