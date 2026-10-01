import numpy as np
import unittest
from scipy.spatial.transform import Rotation

from raven_app.geometry_quality import (rigid_registration, score_surface_patches,
                                        summarize_paired_surfaces)


def test_rigid_registration_preserves_metric_scale():
    points = np.random.default_rng(1).normal(size=(100, 3))
    r = Rotation.from_euler('xyz', [.2, -.3, .4]).as_matrix()
    target = points @ r.T + [1, 2, 3]
    found_r, found_t = rigid_registration(points, target)
    np.testing.assert_allclose(points @ found_r.T + found_t, target, atol=1e-12)
    assert np.linalg.det(found_r) > .99999


def test_heldout_surface_noise_and_coverage_gate():
    rng = np.random.default_rng(3)
    clean = np.column_stack((rng.uniform(-.2, .2, (2000, 2)), rng.normal(0, .002, 2000)))
    noisy = clean.copy()
    noisy[:, 2] += rng.normal(0, .015, len(noisy))
    centers = np.array([[0, 0, 0]])
    a = score_surface_patches(noisy[:1000], noisy[1000:], centers)
    b = score_surface_patches(clean[:1000], clean[1000:], centers)
    score = summarize_paired_surfaces(a, b)
    assert score['improved']
    assert 1.5 < score['candidate_rmse_mm'] < 2.5
    # Losing half the supported reference patches cannot pass by scoring only the rest.
    reference = a + [dict(a[0], patch=1)]
    candidate = b + [{'patch': 1, 'train_points': 0, 'test_points': 0}]
    assert not summarize_paired_surfaces(reference, candidate)['improved']


def test_test_outliers_are_not_trimmed():
    rng = np.random.default_rng(4)
    train = np.column_stack((rng.uniform(-.15, .15, (500, 2)), np.zeros(500)))
    heldout = train.copy()
    heldout[:50, 2] = .1
    row = score_surface_patches(train, heldout, np.array([[0, 0, 0]]))[0]
    assert abs(row['rmse_mm'] - np.sqrt(.1) * 100) < 1e-10


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith('test_') and callable(fn))
