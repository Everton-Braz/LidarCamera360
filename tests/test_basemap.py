import unittest
import numpy as np
from pyproj import Transformer
from raven_app.basemap_layer import tile_geometry, WORLD


class BasemapGeometryTests(unittest.TestCase):
    def test_tiles_cover_projected_cloud_and_reproject_to_slippy_grid(self):
        points = np.array([[500000., 9590000., 10.], [500120., 9590090., 12.]])
        tiles = tile_geometry(points, 'EPSG:31984')
        self.assertTrue(0 < len(tiles) <= 36)
        convert = Transformer.from_crs(31984, 3857, always_xy=True)
        for tile in tiles:
            z, x, y = tile['id']
            width = 2 * WORLD / (1 << z)
            corners = tile['corners']
            mx, my = convert.transform(corners[:, 0], corners[:, 1])
            np.testing.assert_allclose(mx, [x*width-WORLD, (x+1)*width-WORLD]*2, atol=1e-6)
            np.testing.assert_allclose(my, [WORLD-y*width]*2+[WORLD-(y+1)*width]*2, atol=1e-6)
            self.assertTrue(np.all(corners[:, 2] < points[:, 2].min()+.1))

    def test_geographic_coordinates_are_not_treated_as_metres(self):
        with self.assertRaisesRegex(ValueError, 'projected'):
            tile_geometry(np.array([[-39., -3., 0.]]), 'EPSG:4326')

if __name__ == '__main__':
    unittest.main()
