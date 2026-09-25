import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from raven_app.person_masks import keep_samples, mask_path, load_keep_mask, masked_operator_keep, apply_mask_exclusions
from raven_app.vulkan_engine import colorize_views, get_vulkan_bin


class PersonMaskTests(unittest.TestCase):
    def test_bilinear_boundary_and_polarity(self):
        mask = np.full((5, 5), 255, np.uint8)
        mask[2, 2] = 0
        np.testing.assert_array_equal(keep_samples(mask, np.array([0., 1.1, 2., -1., 4.]),
                                                       np.array([0., 1.1, 2., 0., 0.])),
                                      [True, False, False, False, False])

    def test_camera_names_and_missing_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(mask_path(root/'images/cam1/a.jpg', root/'images', root/'masks'), root/'masks/cam1/a.jpg.png')
            with self.assertRaises(ValueError):
                load_keep_mask(root/'images/cam0/a.jpg', root/'images', root/'masks')

    def test_pipeline_rejects_unmasked_radius_before_loading_cloud(self):
        from scripts.pipeline_auto_calibrator_and_colorizer import colorize_via_direct_rigid, colorize_via_spirula_sfm
        for method in (colorize_via_direct_rigid, colorize_via_spirula_sfm):
            with self.assertRaisesRegex(ValueError, 'requires person masks'):
                method(Path('nonexistent'), operator_radius=1.)

    def test_geometry_requires_repeated_majority_and_near_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root/'images', root/'masks'
            images.mkdir(); masks.mkdir()
            views = []
            for i in range(4):
                path = images/f'{i}.jpg'
                mask = np.full((64, 64), 255, np.uint8)
                if i < 3:
                    mask[31:34, 31:34] = 0
                cv2.imwrite(str(mask_path(path, images, masks)), mask)
                views.append((path, np.eye(3), np.zeros(3), [18, 18, 32, 32] + [0]*8))
            from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism
            points = np.array([[0., 0., .7], [0., 0., 2.], [.2, 0., .7]])
            keep = masked_operator_keep(points, views, images, masks, 1., project_thin_prism)
            np.testing.assert_array_equal(keep, [False, True, True])
            keep = masked_operator_keep(points, views[:1], images, masks, 1., project_thin_prism)
            np.testing.assert_array_equal(keep, [True, True, True])

    def test_masked_door_plane_survives_with_person_in_front(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root/'images', root/'masks'
            images.mkdir(); masks.mkdir()
            views = []
            for i in range(3):
                path = images/f'{i}.jpg'
                cv2.imwrite(str(mask_path(path, images, masks)), np.zeros((64, 64), np.uint8))
                views.append((path, np.eye(3), np.zeros(3), [18, 18, 32, 32] + [0]*8))
            xx, yy = np.meshgrid(np.linspace(-.05, .05, 11), np.linspace(-.05, .05, 11))
            door = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, .85)))
            points = np.vstack((door, [0., 0., .65]))
            from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism
            keep = masked_operator_keep(points, views, images, masks, 1., project_thin_prism)
            self.assertTrue(np.all(keep[:-1]))
            self.assertFalse(keep[-1])

    def test_configured_rig_rectangle_masks_color_but_not_geometry_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root/'images', root/'masks'
            images.mkdir(); masks.mkdir()
            (images/'cam0').mkdir(); (masks/'cam0').mkdir()
            sources = []
            config = {'schema': 1, 'fisheye_border_percent': 0,
                      'rectangles': {'cam0': [[.375, .875, .625, 1.0]]}}
            import json
            (masks/'manifest.json').write_text(json.dumps({'schema': 3, 'mask_config': config}), encoding='utf-8')
            for i in range(3):
                path = images/'cam0'/f'{i:02d}.png'
                cv2.imwrite(str(path), np.full((64, 64, 3), 80, np.uint8))
                combined = apply_mask_exclusions(np.full((64, 64), 255, np.uint8), 'cam0', config)
                cv2.imwrite(str(mask_path(path, images, masks)), combined)
                sources.append(path)
            combined = load_keep_mask(sources[0], images, masks)
            self.assertEqual(int(combined[61, 31]), 0)
            self.assertEqual(int(combined[57, 31]), 0)
            self.assertEqual(int(combined[20, 31]), 255)
            self.assertFalse((masks/'persons').exists())
            from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism
            views = [(path, np.eye(3), np.zeros(3), [18, 31, 32, 32] + [0]*8)
                     for path in sources[:3]]
            self.assertTrue(masked_operator_keep(np.array([[0., .60, .75]]), views,
                                                 images, masks, 1., project_thin_prism)[0])

    def test_sparse_ground_plane_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root/'images', root/'masks'
            images.mkdir(); masks.mkdir()
            views = []
            for i in range(3):
                path = images/f'{i}.jpg'
                cv2.imwrite(str(mask_path(path, images, masks)), np.zeros((64, 64), np.uint8))
                views.append((path, np.eye(3), np.zeros(3), [18, 18, 32, 32] + [0]*8))
            xx, yy = np.meshgrid(np.linspace(-.24, .24, 7), np.linspace(-.24, .24, 7))
            ground = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, .85)))
            points = np.vstack((ground, [0., 0., .60]))
            from scripts.pipeline_auto_calibrator_and_colorizer import project_thin_prism
            keep = masked_operator_keep(points, views, images, masks, 1., project_thin_prism)
            self.assertTrue(np.all(keep[:-1]))
            self.assertFalse(keep[-1])

    @unittest.skipUnless(get_vulkan_bin().is_file(), 'Vulkan build unavailable')
    def test_gpu_mask_rejects_red_view_and_uses_clean_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, masks = root/'images', root/'masks'
            images.mkdir(); masks.mkdir()
            views = []
            for index, color in enumerate(([255, 0, 0], [10, 80, 120])):
                path = images/f'{index}.png'
                cv2.imwrite(str(path), np.full((64, 64, 3), color[::-1], np.uint8))
                mask = np.full((64, 64), 255, np.uint8)
                if index == 0:
                    mask[32, 33] = 0  # only the neighboring interpolation tap
                cv2.imwrite(str(mask_path(path, images, masks)), mask)
                views.append((path, np.eye(3), np.zeros(3), [18, 18, 32, 32] + [0] * 8))
            points = np.array([[0., 0, 1.]])
            rgb = colorize_views(points, views, root, masks_dir=masks, images_dir=images)
            self.assertIsNotNone(rgb)
            np.testing.assert_allclose(rgb, [[10, 80, 120]], atol=1)
            rgb = colorize_views(points, views[:1], root, masks_dir=masks, images_dir=images)
            np.testing.assert_array_equal(rgb, [[180, 180, 180]])


if __name__ == '__main__':
    unittest.main()
