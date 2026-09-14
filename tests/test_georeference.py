import json
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import pyproj

from raven_app.georeference import (
    auto_detect_utm_crs,
    calculate_insv_gps,
    export_geojson,
    fit_georeference,
    georeference_colmap,
    georeference_pointcloud,
    parse_colmap_images_txt,
    run_georeference,
)
from raven_app.insv_gps import GPS_RECORD, MAGIC


def make_fixes(times, xy, altitude=2.0):
    return [dict(timestamp_epoch=float(t), latitude_deg=float(y),
                 longitude_deg=float(x), altitude_m=float(altitude))
            for t, (x, y) in zip(times, xy)]


def write_insv(path, fixes):
    payload = b''.join(GPS_RECORD.pack(int(r['timestamp_epoch']), 0, 0, b'A',
                                       r['latitude_deg'], b'N', r['longitude_deg'],
                                       b'E', 1.5, 90.0, r['altitude_m']) for r in fixes)
    body = payload + struct.pack('<BBI', 0, 7, len(payload))
    directory = struct.pack('<BBII', 7, 0, len(payload), 0)
    body += directory + struct.pack('<BBI', 0, 0, len(directory))
    path.write_bytes(b'video' + body + bytes(32) + struct.pack('<II', len(body) + 72, 3) + MAGIC)


class GeoReferenceTests(unittest.TestCase):
    def _registered_case(self, count=5):
        local = np.array([[0., 0., 0.], [20., 0., 0.], [40., 0., 0.],
                          [40., 20., 0.], [0., 20., 0.]])[:count]
        angle = np.deg2rad(20.)
        rot = np.array([[np.cos(angle), -np.sin(angle), 0.],
                        [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        target = local @ rot.T + [500000., 9590000., 0.]
        inv = pyproj.Transformer.from_crs('EPSG:32724', 'EPSG:4326', always_xy=True)
        lon, lat = inv.transform(target[:, 0], target[:, 1])
        fixes = make_fixes(range(1000, 1000 + len(local)), zip(lon, lat))
        return local, target, fixes

    def test_auto_detect_utm_crs(self):
        # Brazil Fortaleza coordinates -> SIRGAS 2000 UTM Zone 24S
        crs_br = auto_detect_utm_crs(-3.7, -38.6)
        self.assertEqual(crs_br["epsg_h"], 31984)
        self.assertEqual(crs_br["zone"], 24)
        self.assertEqual(crs_br["hemisphere"], "S")
        self.assertIn("SIRGAS 2000", crs_br["crs_name"])

        # Northern hemisphere (e.g. Paris) -> WGS 84 UTM Zone 31N
        crs_fr = auto_detect_utm_crs(48.8, 2.3)
        self.assertEqual(crs_fr["epsg_h"], 32631)
        self.assertEqual(crs_fr["hemisphere"], "N")

    def test_calculate_insv_gps(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            insv = root / "test.insv"
            _, _, fixes = self._registered_case()
            write_insv(insv, fixes)

            info = calculate_insv_gps(insv)
            self.assertEqual(info["total_fixes"], len(fixes))
            self.assertIn("utm_coords", info)
            self.assertEqual(info["utm_coords"].shape, (len(fixes), 3))
            self.assertAlmostEqual(info["mean_wgs84"]["latitude_deg"], np.mean([f["latitude_deg"] for f in fixes]), places=4)

    def test_export_geojson_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            insv = root / "test.insv"
            _, _, fixes = self._registered_case()
            write_insv(insv, fixes)

            info = calculate_insv_gps(insv)
            colmap_cams = [
                {"name": "cam0/001.jpg", "frame_num": 1, "latitude": -3.7, "longitude": -38.6, "altitude_m": 10.0, "heading_deg": 45.0},
                {"name": "cam0/002.jpg", "frame_num": 2, "latitude": -3.7001, "longitude": -38.6001, "altitude_m": 10.0, "heading_deg": 45.0},
            ]
            bounds = {
                "min_easting_m": 500000.0, "max_easting_m": 500100.0,
                "min_northing_m": 9590000.0, "max_northing_m": 9590100.0,
                "min_altitude_m": 0.0, "max_altitude_m": 10.0,
                "num_points": 1000,
            }

            files = export_geojson(root, gps_records=info["records"], colmap_cameras=colmap_cams, cloud_bounds_utm=bounds, crs_info=info["crs_info"])
            self.assertIn("gps_trajectory_geojson", files)
            self.assertIn("colmap_cameras_geojson", files)
            self.assertIn("pointcloud_extent_geojson", files)
            self.assertIn("georeferenced_scene_geojson", files)

            # Verify valid RFC 7946 GeoJSON
            gps_geo = json.loads(Path(files["gps_trajectory_geojson"]).read_text(encoding="utf-8"))
            self.assertEqual(gps_geo["type"], "FeatureCollection")
            self.assertEqual(gps_geo["features"][0]["geometry"]["type"], "LineString")

            cam_geo = json.loads(Path(files["colmap_cameras_geojson"]).read_text(encoding="utf-8"))
            self.assertEqual(cam_geo["type"], "FeatureCollection")

            cloud_geo = json.loads(Path(files["pointcloud_extent_geojson"]).read_text(encoding="utf-8"))
            self.assertEqual(cloud_geo["features"][0]["geometry"]["type"], "Polygon")

            scene_geo = json.loads(Path(files["georeferenced_scene_geojson"]).read_text(encoding="utf-8"))
            self.assertGreater(len(scene_geo["features"]), 3)

    def test_fit_georeference_with_known_yaw(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            insv = root / "test.insv"
            _, _, fixes = self._registered_case()
            write_insv(insv, fixes)
            info = calculate_insv_gps(insv)

            res = fit_georeference(info, manual_yaw_deg=45.0)
            self.assertEqual(res["status"], "success")
            self.assertEqual(res["yaw_heading_deg"], 45.0)
            self.assertEqual(res["scale"], 1.0)
            self.assertIn("matrix_4x4", res)

    def test_pointcloud_georeference_and_export(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cloud = root / "test_cloud.pcd"
            cloud.write_text(
                "VERSION .7\nFIELDS x y z r g b intensity\nSIZE 4 4 4 1 1 1 4\nTYPE F F F U U U F\nCOUNT 1 1 1 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n"
                "1.0 2.0 3.0 255 0 0 10.0\n"
                "4.0 5.0 6.0 0 255 0 20.0\n",
                encoding="ascii"
            )
            transform_dict = {
                "rotation_matrix": np.eye(3).tolist(),
                "translation_utm_m": [500000.0, 9590000.0, 10.0],
                "target_compound_crs": "EPSG:32724+5773",
                "target_horizontal_crs": "EPSG:32724",
            }
            res = georeference_pointcloud(cloud, transform_dict, root / "out")
            self.assertEqual(res["points_count"], 2)
            self.assertTrue(Path(res["laz_path"]).is_file())
            self.assertTrue(Path(res["las_path"]).is_file())
            self.assertTrue(Path(res["pcd_path"]).is_file())
            self.assertTrue(Path(res["prj_path"]).is_file())

            # Verify LAZ coordinates
            import laspy
            las = laspy.read(res["laz_path"])
            np.testing.assert_allclose(las.x, [500001.0, 500004.0], atol=1e-3)
            np.testing.assert_allclose(las.y, [9590002.0, 9590005.0], atol=1e-3)
            np.testing.assert_allclose(las.z, [13.0, 16.0], atol=1e-3)

    def test_run_georeference_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            insv = root / "test.insv"
            _, _, fixes = self._registered_case()
            write_insv(insv, fixes)

            cloud = root / "cloud.pcd"
            cloud.write_text(
                "VERSION .7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n"
                "0 0 0 16711680\n1 1 1 65280\n",
                encoding="ascii"
            )

            report = run_georeference(insv, root / "deliverables", cloud=cloud)
            self.assertEqual(report["status"], "accepted")
            self.assertIn("gps_trajectory_geojson", report["deliverables"]["geojson_files"])
            self.assertIn("pointcloud_extent_geojson", report["deliverables"]["geojson_files"])
            self.assertIn("georeferenced_scene_geojson", report["deliverables"]["geojson_files"])
            self.assertTrue(Path(report["deliverables"]["cloud_export"]["laz_path"]).is_file())

    def test_georeference_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            insv = root / "test.insv"
            _, _, fixes = self._registered_case()
            write_insv(insv, fixes)

            from raven_app.cli import main
            code = main(["--headless", "georeference", "--insv", str(insv), "--output", str(root / "out")])
            self.assertEqual(code, 0)
            self.assertTrue((root / "out" / "gps_trajectory.geojson").is_file())
            self.assertTrue((root / "out" / "georeference_report.json").is_file())

    def test_orthophoto_generation(self):
        from raven_app.orthophoto import generate_orthophoto
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # Create a simple synthetic point cloud with XYZ and RGB
            pts = np.array([
                [500000.0, 9590000.0, 10.0],
                [500005.0, 9590000.0, 12.0],
                [500000.0, 9590005.0, 11.0],
                [500005.0, 9590005.0, 14.0],
            ], dtype=np.float64)
            colors = np.array([
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [255, 255, 255],
            ], dtype=np.uint8)

            res = generate_orthophoto(pts, root / "ortho", colors=colors, gsd_m=0.5, crs_epsg=31984)
            self.assertEqual(res["status"], "success")
            self.assertTrue(Path(res["orthophoto_tif"]).is_file())
            self.assertTrue(Path(res["orthophoto_png"]).is_file())
            self.assertTrue(Path(res["world_file_tfw"]).is_file())
            self.assertTrue(Path(res["dsm_tif"]).is_file())
            self.assertGreater(res["width_px"], 0)
            self.assertGreater(res["height_px"], 0)


if __name__ == "__main__":
    unittest.main()
