import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

from raven_app.insv_gps import GPS_RECORD, MAGIC, export_gps


def _fix(epoch, lat, lon, altitude=16.0, status=b"A"):
    return GPS_RECORD.pack(epoch, 0, 0, status, lat, b"N", lon, b"E", 0.0, 0.0, altitude)


def _insv(path, payload):
    body = payload + struct.pack("<BBI", 0, 7, len(payload))
    directory = struct.pack("<BBII", 7, 0, len(payload), 0)
    body += directory + struct.pack("<BBI", 0, 0, len(directory))
    header = bytes(32) + struct.pack("<II", len(body) + 72, 3) + MAGIC
    path.write_bytes(b"video" + body + header)


class GpsExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.insv = Path(self.tmp.name) / "capture.insv"
        _insv(self.insv, b"".join(_fix(1788370677 + i, 3.69 + i / 1000, 38.59 + i / 1000) for i in range(3)))

    def test_default_export_keeps_csv_and_gpx_compatibility(self):
        output = Path(self.tmp.name) / "out"
        report = export_gps(self.insv, output)
        self.assertTrue((output / "gps_raw.csv").is_file())
        self.assertTrue((output / "gps.csv").is_file())
        self.assertTrue((output / "gps.gpx").is_file())
        self.assertTrue((output / "gps_report.json").is_file())
        self.assertEqual(Path(report["output_files"]["gps.csv"]), (output / "gps.csv").resolve())

    def test_geojson_is_feature_collection_with_track_and_no_nan(self):
        output = Path(self.tmp.name) / "geo"
        report = export_gps(self.insv, output, formats=("geojson",))
        self.assertEqual(set(report["output_files"]), {"gps.geojson"})
        self.assertTrue((output / "gps_report.json").is_file())
        data = json.loads((output / "gps.geojson").read_text(encoding="utf-8"))
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 4)  # 3 fixes plus one LineString track
        self.assertTrue(all(feature["geometry"]["type"] in ("Point", "LineString") for feature in data["features"]))
        self.assertFalse("NaN" in (output / "gps.geojson").read_text(encoding="utf-8"))

    def test_invalid_or_empty_formats_fail_before_writing(self):
        for formats in ((), ("kml",), ("csv", "bad")):
            output = Path(self.tmp.name) / ("bad-" + str(len(formats)))
            with self.assertRaises(ValueError):
                export_gps(self.insv, output, formats=formats)
            self.assertFalse(output.exists())


class GpsWorkflowCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.insv = Path(self.tmp.name) / "capture.insv"
        _insv(self.insv, b"".join(_fix(1788370677 + i, 3.69 + i / 1000, 38.59 + i / 1000) for i in range(3)))

    def test_cli_exposes_gps_options_without_qt_import(self):
        from raven_app.cli import parse
        before = "PyQt6" in sys.modules
        args = parse(["workflow", "--bag", "x", "--insv", "x.insv", "--output", "out",
                      "--process-gps", "--gps-formats", "geojson", "gpx"])
        self.assertTrue(args.process_gps)
        self.assertEqual(args.gps_formats, ["geojson", "gpx"])
        args = parse(["extract-gps", "--insv", "x.insv", "--output", "out", "--formats", "csv"])
        self.assertEqual(args.formats, ["csv"])
        self.assertEqual("PyQt6" in sys.modules, before)

    def test_workflow_gps_helper_writes_selected_metadata(self):
        from raven_app.workflow import process_gps_metadata
        output = Path(self.tmp.name) / "workflow-gps"
        report = process_gps_metadata(self.insv, output, ("geojson",))
        self.assertEqual(report["status"], "complete")
        self.assertTrue((output / "gps.geojson").is_file())
        self.assertFalse(report["geometry_transformed"])

    def test_workflow_reports_no_valid_gps_without_geometry(self):
        from raven_app.workflow import process_gps_metadata
        invalid = Path(self.tmp.name) / "invalid.insv"
        _insv(invalid, _fix(1788370677, 3.69, 38.59, status=b"V"))
        report = process_gps_metadata(invalid, Path(self.tmp.name) / "invalid-gps", ("csv",))
        self.assertEqual(report["status"], "no_valid_gps")
        self.assertEqual(report["georeferencing_status"], "no_valid_gps")
        self.assertFalse(report["geometry_transformed"])

    def test_workflow_reports_unavailable_for_unsupported_metadata(self):
        from raven_app.workflow import process_gps_metadata
        unsupported = Path(self.tmp.name) / "unsupported.insv"
        unsupported.write_bytes(b"not-an-insv")
        report = process_gps_metadata(unsupported, Path(self.tmp.name) / "unsupported-gps", ("gpx",))
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["georeferencing_status"], "gps_metadata_unavailable")
        self.assertEqual(report["output_files"], {})
        self.assertFalse(report["geometry_transformed"])


if __name__ == "__main__":
    unittest.main()
