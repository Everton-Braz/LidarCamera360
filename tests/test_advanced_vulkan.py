import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from raven_app.vulkan_engine import (_view_advanced, colorize_views, get_vulkan_bin)
from raven_app.photometric.model import apply_rgb
from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism


def _coefficients():
    d, y, x = np.mgrid[0:4, 0:3, 0:3]
    grid = np.stack((.08 * x - .025 * y - .012 * d,
                     -.05 * x + .03 * y + .014 * d,
                     .035 * y - .009 * d), axis=-1)
    return {
        'log_gain': [-.12, .06, -.03],
        'vignette': [.12, -.02, .01],
        # This intentionally pushes red above one so the post-H clamp is tested.
        'ppisp_h': [1.45, .2, 0.0, -.03, .96, .01, 0.0, 0.0, 1.0],
        'bilateral_grid': grid.tolist(),
    }


class AdvancedProtocolTests(unittest.TestCase):
    def test_per_view_payload_uses_identity_defaults_and_validates(self):
        path = Path('images/cam0/frame.jpg')
        h, grid = _view_advanced(path, {}, Path('images'))
        np.testing.assert_array_equal(h, np.eye(3, dtype='<f4').ravel())
        np.testing.assert_array_equal(grid, np.zeros(108, dtype='<f4'))

        h, grid = _view_advanced(path, {'cam0/frame.jpg': _coefficients()}, Path('images'))
        self.assertEqual(h.shape, (9,))
        self.assertEqual(grid.shape, (108,))
        with self.assertRaisesRegex(ValueError, 'PPISP homography'):
            _view_advanced(path, {'cam0/frame.jpg': {**_coefficients(), 'ppisp_h': [0] * 9}},
                           Path('images'))


@unittest.skipUnless(get_vulkan_bin().is_file(), 'Native Vulkan build unavailable')
class AdvancedVulkanParityTests(unittest.TestCase):
    def test_rvc5_and_rvc6_match_cpu_ppisp_grid_reference(self):
        """Advanced corrections run per observation and match the CPU formula."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images'
            cam = images / 'cam0'
            cam.mkdir(parents=True)
            path = cam / 'frame.png'
            rgb = np.array([232, 210, 188], dtype=np.uint8)
            cv2.imwrite(str(path), np.broadcast_to(rgb[::-1], (3840, 3840, 3)).copy())
            masks = root / 'masks'
            mask_path = masks / 'cam0' / 'frame.png.png'
            mask_path.parent.mkdir(parents=True)
            cv2.imwrite(str(mask_path), np.full((3840, 3840), 255, np.uint8))

            params = [1080, 1080, 1920, 1920] + [0] * 8
            radii = np.array([0.0, 210.0, 680.0, 1130.0, 1510.0])
            angles = np.array([0.0, .25, .8, 1.65, 2.55])
            normalized = np.tan(radii / params[0])
            points = np.column_stack((normalized * np.cos(angles),
                                      normalized * np.sin(angles),
                                      np.ones(len(radii))))
            views = [(path, np.eye(3), np.zeros(3), params)]
            coeff = _coefficients()
            model = {'cam0/frame.png': coeff}

            u, v, _ = project_thin_prism(points, params)
            expected = np.stack([apply_rgb(rgb, px, py, params, coeff)
                                 for px, py in zip(u, v)])
            unmasked = colorize_views(points, views, root, images_dir=images,
                                      photometric=model)
            masked = colorize_views(points, views, root, images_dir=images,
                                    masks_dir=masks, photometric=model)
            self.assertIsNotNone(unmasked)
            self.assertIsNotNone(masked)
            np.testing.assert_allclose(unmasked, expected, atol=1)
            np.testing.assert_allclose(masked, expected, atol=1)

            identity = {'log_gain': [0, 0, 0], 'vignette': [0, 0, 0],
                        'ppisp_h': np.eye(3).ravel().tolist(),
                        'bilateral_grid': np.zeros((4, 3, 3, 3)).tolist()}
            legacy = colorize_views(points, views, root)
            advanced_identity = colorize_views(points, views, root, images_dir=images,
                                               photometric={'cam0/frame.png': identity})
            self.assertIsNotNone(legacy)
            self.assertIsNotNone(advanced_identity)
            np.testing.assert_array_equal(advanced_identity, legacy)


if __name__ == '__main__':
    unittest.main()
