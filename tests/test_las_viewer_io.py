import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyproj import CRS

laspy = None
try:
    import laspy
except ImportError:  # pragma: no cover - portable env is expected to provide laspy
    laspy = None

from raven_app.cloud_io import load_cloud


@unittest.skipUnless(laspy is not None, "laspy is required for LAS/LAZ I/O")
class LasViewerIoTests(unittest.TestCase):
    def test_las_preserves_float64_xyz_rgb_intensity_and_crs(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.las"
            header = laspy.LasHeader(point_format=7, version="1.4")
            header.scales = np.array([0.001, 0.001, 0.001])
            header.offsets = np.array([1000.0, 2000.0, 3000.0])
            header.add_crs(CRS.from_epsg(4326))
            las = laspy.LasData(header)
            xyz = np.array([[1000.123, 2000.456, 3000.789], [1001.234, 2001.567, 3001.890]])
            las.x, las.y, las.z = xyz.T
            las.red, las.green, las.blue = [65535, 1000], [0, 2000], [32768, 3000]
            las.intensity = [12, 34]
            las.write(path)

            cloud = load_cloud(path)
            self.assertEqual(cloud.points.dtype, np.float64)
            np.testing.assert_allclose(cloud.points, xyz, atol=0.001)
            np.testing.assert_allclose(cloud.colors[0], [1.0, 0.0, 32768 / 65535], atol=1e-6)
            np.testing.assert_array_equal(cloud.intensities, [12, 34])
            self.assertTrue(cloud.has_rgb)
            self.assertIn("4326", cloud.crs_wkt)

    def test_las_extension_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.LAS"
            las = laspy.create(point_format=3, file_version="1.2")
            las.x, las.y, las.z = [1.0], [2.0], [3.0]
            las.write(path)
            self.assertEqual(load_cloud(path).points.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
