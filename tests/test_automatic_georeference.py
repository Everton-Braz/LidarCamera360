import unittest
import numpy as np

from raven_app.automatic_georeference import build_automatic_transform


class AutomaticGeoreferenceTests(unittest.TestCase):
    def test_static_gps_anchor_preserves_scale_and_local_z(self):
        gps = {
            'utm_coords': np.array([[500000., 7000000., 99.], [500010., 7000000., 99.]]),
            'crs_info': {'horizontal_crs': 'EPSG:31984', 'vertical_crs': 'EPSG:3855',
                         'compound_crs': 'EPSG:31984+3855', 'crs_name': 'SIRGAS'},
            'mean_wgs84': {'latitude_deg': -26., 'longitude_deg': -49., 'altitude_m': 99.},
            'records': [{}, {}],
        }
        result = build_automatic_transform(gps, np.array([[0., 0., 4.], [10., 0., 8.]]))
        matrix = np.asarray(result['matrix_4x4'])
        self.assertEqual(result['scale'], 1.0)
        self.assertEqual(result['quality'], 'estimated')
        self.assertEqual(matrix[2, 3], 0.0)
        self.assertNotIn('target_compound_crs', result)


if __name__ == '__main__':
    unittest.main()
