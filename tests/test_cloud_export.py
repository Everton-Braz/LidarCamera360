import tempfile
import unittest
from pathlib import Path

import numpy as np
from pyproj import CRS

from raven_app.cloud_io import CloudData, load_cloud
from raven_app.cloud_export import save_cloud


class CloudExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cloud = CloudData(
            self.root / "source.pcd",
            np.array([[123456789.123, -9000000.5, 0.125], [123456790.123, -9000001.5, 1.125]], dtype=np.float64),
            np.array([[1.0, 0.0, 0.5], [0.1, 0.2, 0.3]], dtype=np.float32),
            2,
            np.array([10.5, 20.25], dtype=np.float32),
            True,
            CRS.from_epsg(4326).to_wkt(),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_ply_and_pcd_roundtrip_with_sidecar(self):
        for suffix in (".ply", ".pcd"):
            result = save_cloud(self.cloud, self.root / ("roundtrip" + suffix))
            self.assertEqual(result["points_written"], 2)
            self.assertEqual(result["crs_wkt"], self.cloud.crs_wkt)
            self.assertTrue(Path(result["metadata_sidecar"]).is_file())
            got = load_cloud(self.root / ("roundtrip" + suffix))
            np.testing.assert_array_equal(got.points, self.cloud.points)
            np.testing.assert_array_equal(got.intensities, self.cloud.intensities)
            np.testing.assert_allclose(got.colors, self.cloud.colors, atol=1 / 255)
            self.assertEqual(got.crs_wkt, self.cloud.crs_wkt)

    def test_rejects_overwrite_and_source_path(self):
        target = self.root / "new.ply"
        save_cloud(self.cloud, target)
        with self.assertRaises(FileExistsError):
            save_cloud(self.cloud, target)
        source = self.root / "source.pcd"
        source.touch()
        same = CloudData(source, self.cloud.points, self.cloud.colors, 2)
        with self.assertRaises(ValueError):
            save_cloud(same, source)

    def test_formats_must_match_single_output(self):
        with self.assertRaises(ValueError):
            save_cloud(self.cloud, self.root / "bad.ply", formats=["ply", "pcd"])
        with self.assertRaises(ValueError):
            save_cloud(self.cloud, self.root / "bad.ply", formats="pcd")

    def test_las_precision_and_crs_when_laspy_available(self):
        try:
            import laspy  # noqa: F401
        except ImportError:
            self.skipTest("laspy is required")
        result = save_cloud(self.cloud, self.root / "roundtrip.las")
        got = load_cloud(self.root / "roundtrip.las")
        np.testing.assert_allclose(got.points, self.cloud.points, atol=0.001)
        np.testing.assert_allclose(got.colors, self.cloud.colors, atol=1 / 255)
        self.assertEqual(result["metadata_sidecar"], None)


if __name__ == "__main__":
    unittest.main()
