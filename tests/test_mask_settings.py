import unittest

import numpy as np

from raven_app.person_masks import (
    apply_mask_exclusions,
    fixed_keep_samples,
    normalize_mask_config,
)


class MaskSettingsTests(unittest.TestCase):
    def test_defaults_and_normalized_schema(self):
        self.assertEqual(normalize_mask_config(None), {
            'schema': 1,
            'fisheye_border_percent': 0.0,
            'rectangles': {},
            'ellipses': {},
            'polygons': {},
        })
        self.assertEqual(normalize_mask_config({
            'schema': 1,
            'fisheye_border_percent': 7,
            'rectangles': {'cam0': [[0.1, 0.2, 0.3, 0.4]]},
            'ellipses': {},
            'polygons': {},
        }), {
            'schema': 1,
            'fisheye_border_percent': 7.0,
            'rectangles': {'cam0': [[0.1, 0.2, 0.3, 0.4]]},
            'ellipses': {},
            'polygons': {},
        })

    def test_rejects_invalid_percent_and_rectangles(self):
        invalid = (
            {'fisheye_border_percent': -0.1},
            {'fisheye_border_percent': 25.1},
            {'fisheye_border_percent': float('nan')},
            {'rectangles': {'cam/2': [[0.1, 0.1, 0.2, 0.2]]}},
            {'rectangles': {'cam0': [[-0.1, 0.1, 0.2, 0.2]]}},
            {'rectangles': {'cam0': [[0.4, 0.1, 0.2, 0.2]]}},
            {'rectangles': {'cam1': [[0.1, 0.2, 0.3]]}},
            {'ellipses': {'cam0': [[0.5, 0.5, 0.5, 0.9]]}},
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_mask_config(config)

    def test_fisheye_border_percentage_tracks_excluded_image_area(self):
        source = np.full((1000, 1000), 255, dtype=np.uint8)
        for percent in (0, 1, 5, 20):
            result = apply_mask_exclusions(source, 'cam0', {'fisheye_border_percent': percent})
            excluded = np.count_nonzero(result == 0) / result.size * 100
            self.assertAlmostEqual(excluded, percent, delta=0.12)
            self.assertEqual(int(result[500, 500]), 255)
            self.assertEqual(set(np.unique(result)), {255} if percent == 0 else {0, 255})

    def test_camera_rectangles_apply_to_every_frame_of_that_camera(self):
        config = normalize_mask_config({
            'rectangles': {'cam0': [[0.2, 0.2, 0.4, 0.4]], 'cam1': []},
        })
        first_frame = np.full((100, 100), 255, dtype=np.uint8)
        second_frame = np.full((100, 100), 255, dtype=np.uint8)

        first_result = apply_mask_exclusions(first_frame, 'cam0', config)
        second_result = apply_mask_exclusions(second_frame, 'cam0', config)
        other_camera = apply_mask_exclusions(first_frame, 'cam1', config)

        self.assertEqual(int(first_result[25, 25]), 0)
        self.assertEqual(int(second_result[25, 25]), 0)
        self.assertEqual(int(other_camera[25, 25]), 255)
        self.assertEqual(int(first_result[50, 50]), 255)

    def test_static_keep_samples_combines_camera_rectangle_and_border(self):
        config = normalize_mask_config({
            'fisheye_border_percent': 20,
            'rectangles': {'cam1': [[0.2, 0.2, 0.4, 0.4]]},
        })
        keep = fixed_keep_samples(
            (100, 100),
            np.array([30.0, 50.0, 0.0]),
            np.array([30.0, 50.0, 0.0]),
            'cam1',
            config,
        )

        np.testing.assert_array_equal(keep, [False, True, False])

    def test_stretched_ellipse_is_combined_and_not_geometry_evidence(self):
        config = normalize_mask_config({'ellipses': {'cam0': [[0.2, 0.4, 0.8, 0.6]]}})
        source = np.full((100, 100), 255, dtype=np.uint8)
        result = apply_mask_exclusions(source, 'cam0', config)
        self.assertEqual(int(result[50, 50]), 0)
        self.assertEqual(int(result[40, 20]), 255)  # corner of bounding box stays clear
        self.assertEqual(int(result[20, 50]), 255)
        self.assertEqual(int(apply_mask_exclusions(source, 'cam1', config)[50, 50]), 255)
        samples = fixed_keep_samples((100, 100), np.array([50., 20.]),
                                     np.array([50., 40.]), 'cam0', config)
        np.testing.assert_array_equal(samples, [False, True])

if __name__ == '__main__':
    unittest.main()
