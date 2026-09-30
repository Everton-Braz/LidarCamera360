"""Bounded, robust photometric calibration without a training runtime."""
from .model import apply_rgb, fit_observations, load_model


def fit_dataset(dataset, output=None, masks_dir=None, sample_points=50000,
                max_observations=500000, mode='loglinear'):
    from .dataset import fit_dataset as run
    return run(dataset, output, masks_dir, sample_points, max_observations, mode=mode)


def prepare(dataset, mode='off', params=None, masks_dir=None):
    if mode == 'off':
        return None
    if mode not in ('loglinear', 'ppisp-bilateral'):
        raise ValueError('Unsupported photometric mode')
    from .dataset import prepare_model
    return prepare_model(dataset, params, masks_dir, mode=mode)['views']
