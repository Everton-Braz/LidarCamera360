import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raven_app.photometric.model import fit_observations, to_srgb, apply_rgb, load_model


class PhotometricTests(unittest.TestCase):
    def synthetic(self, identity=False, vignette=False):
        rng = np.random.default_rng(15)
        n, frames = 700, 12
        point = np.repeat(np.arange(n), 5)
        frame = np.concatenate([rng.choice(frames, 5, replace=False) for _ in range(n)])
        lens = frame % 2
        radius = rng.uniform(.02, .98, len(point))
        gain = rng.normal(0, .13, (frames, 3))
        gain -= np.median(gain, axis=0)
        if identity:
            gain[:] = 0
        k = np.array([-.3, -.1]) if vignette else np.zeros(2)
        radiance = rng.uniform(.08, .5, (n, 3))
        rgb = to_srgb(radiance[point] * np.exp(gain[frame] + (k[lens] * radius**2)[:, None]))
        obs = dict(point_id=point, frame_id=frame, lens_id=lens, radius=radius,
                   rgb=rgb, weight=np.ones(len(point)))
        names = [f'cam{i%2}/frame_{i}.jpg' for i in range(frames)]
        return obs, names, gain

    def test_recovers_gains_and_reduces_unseen_point_error(self):
        obs, names, truth = self.synthetic()
        model = fit_observations(obs, names, ['0', '1'])
        recovered = np.array([model['views'][name]['log_gain'] for name in names])
        np.testing.assert_allclose(recovered, truth, atol=.012)
        self.assertLess(model['metrics']['test_after']['linear_rmse'],
                        .1 * model['metrics']['test_before']['linear_rmse'])

    def test_identity_and_json_roundtrip(self):
        obs, names, _ = self.synthetic(identity=True)
        model = fit_observations(obs, names, ['0', '1'])
        self.assertEqual(model['selected'], 'identity')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.json'
            path.write_text(json.dumps(model))
            self.assertEqual(load_model(path), model)
            model['views'][names[0]]['log_gain'][0] = float('nan')
            path.write_text(json.dumps(model))
            with self.assertRaises(ValueError):
                load_model(path)

    def test_vignette_requires_validation_improvement(self):
        obs, names, _ = self.synthetic(vignette=True)
        model = fit_observations(obs, names, ['0', '1'])
        self.assertEqual(model['selected'], 'gains+vignette')
        self.assertLess(model['metrics']['test_after']['linear_rmse'],
                        .25 * model['metrics']['test_before']['linear_rmse'])

    def test_inverse_and_identity(self):
        rgb = np.array([[70, 120, 180]], dtype=np.uint8)
        camera = [1, 1, 1920, 1920]
        coeff = {'log_gain': [0, 0, 0], 'vignette': [0, 0, 0]}
        np.testing.assert_array_equal(apply_rgb(rgb, [1920], [1920], camera, coeff), rgb)
        coeff['log_gain'] = [np.log(2)] * 3
        self.assertTrue(np.all(apply_rgb(rgb, [1920], [1920], camera, coeff) < rgb))

    def test_invalid_indices_and_sparse_points_fail(self):
        obs, names, _ = self.synthetic()
        obs['frame_id'][0] = len(names)
        with self.assertRaises(ValueError):
            fit_observations(obs, names, ['0', '1'])

    def test_disconnected_components_are_anchored(self):
        obs, names, _ = self.synthetic()
        other = {k: v.copy() for k, v in obs.items()}
        other['point_id'] += 1000
        other['frame_id'] += len(names)
        both = {k: np.concatenate((v, other[k])) for k, v in obs.items()}
        all_names = names + ['other/' + n for n in names]
        model = fit_observations(both, all_names, ['0', '1'])
        self.assertEqual(len(set(v['component'] for v in model['views'].values())), 2)
        for component in (0, 1):
            values = [v['log_gain'] for v in model['views'].values() if v['component'] == component]
            np.testing.assert_allclose(np.median(values, axis=0), 0, atol=1e-8)

    def test_cache_invalidates_images_masks_and_explicit_stale_model(self):
        from raven_app.photometric.dataset import fingerprint, prepare_model
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images/cam0').mkdir(parents=True)
            image = root / 'images/cam0/a.jpg'
            image.write_bytes(b'first')
            first = fingerprint(root)
            image.write_bytes(b'changed-image')
            self.assertNotEqual(first, fingerprint(root))
            masks = root / 'masks/cam0'
            masks.mkdir(parents=True)
            first = fingerprint(root, masks.parent)
            (masks / 'a.jpg.png').write_bytes(b'mask')
            self.assertNotEqual(first, fingerprint(root, masks.parent))
            path = root / 'params.json'
            path.write_text(json.dumps({'schema': 1, 'color_space': 'srgb',
                                        'views': {}, 'fingerprint': 'stale'}))
            with self.assertRaisesRegex(ValueError, 'refit'):
                prepare_model(root, path)

    def test_advanced_model_rejects_invalid_grid_homography_and_mode(self):
        from raven_app.photometric.dataset import prepare_model, fingerprint
        coeff = {'log_gain': [0] * 3, 'vignette': [0] * 3,
                 'ppisp_h': np.eye(3).reshape(-1).tolist(),
                 'bilateral_grid': np.zeros((4, 3, 3, 3)).tolist()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'params.json'
            model = {'schema': 2, 'color_space': 'srgb', 'mode': 'ppisp-bilateral',
                     'views': {'cam0/a.jpg': coeff}, 'fingerprint': fingerprint(root)}
            path.write_text(json.dumps(model))
            self.assertEqual(load_model(path), model)
            with self.assertRaisesRegex(ValueError, 'mode'):
                prepare_model(root, path, mode='loglinear')
            coeff['bilateral_grid'][0][0][0][0] = .36
            path.write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, 'bilateral'):
                load_model(path)
            coeff['bilateral_grid'][0][0][0][0] = 0
            coeff['ppisp_h'] = [0.] * 9
            path.write_text(json.dumps(model))
            with self.assertRaisesRegex(ValueError, 'homography'):
                load_model(path)


if __name__ == '__main__':
    unittest.main()
