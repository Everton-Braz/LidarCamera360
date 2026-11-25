import json
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from raven_app.insv_gps import GPS_RECORD, MAGIC, _info_fields, drop_stale_start, export_gps, extract_gps


def fix(epoch=1788370677, lat=3.69549376, lon=38.59972568, status=b'A'):
    return GPS_RECORD.pack(epoch, 0, 673, status, lat, b'S', lon, b'W', 0., 0., 16.)


def insv(path, payload, directory=True):
    body = payload + struct.pack('<BBI', 0, 7, len(payload))
    if directory:
        table = struct.pack('<BBII', 7, 0, len(payload), 0)
        body += table + struct.pack('<BBI', 0, 0, len(table))
    header = bytes(32) + struct.pack('<II', len(body) + 72, 3) + MAGIC
    path.write_bytes(b'fake-video' + body + header)


class InsvGpsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'Escritório.insv'

    def test_repeated_fix_is_not_a_trajectory(self):
        insv(self.path, fix() * 4 + fix(status=b'V'))
        rows, fixes, report = extract_gps(self.path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(fixes), 1)
        self.assertEqual(report['duplicate_valid_records'], 3)
        self.assertEqual(report['invalid_records'], 1)
        self.assertEqual(report['georeferencing_status'], 'location_anchor_only')
        self.assertEqual(fixes[0]['timestamp_utc'], '2026-09-02T17:37:57.673Z')
        self.assertAlmostEqual(fixes[0]['latitude_deg'], -3.69549376)
        self.assertAlmostEqual(fixes[0]['longitude_deg'], -38.59972568)

    def test_sequential_trailer_and_equator(self):
        insv(self.path, fix(lat=0, lon=0), directory=False)
        self.assertEqual(extract_gps(self.path)[2]['valid_records'], 1)

    def test_export_preserves_raw(self):
        insv(self.path, fix() * 2)
        output = Path(self.tmp.name) / 'gps'
        export_gps(self.path, output)
        self.assertEqual(len((output / 'gps_raw.csv').read_text().splitlines()), 3)
        self.assertEqual(len((output / 'gps.csv').read_text().splitlines()), 2)
        self.assertEqual(len(ET.parse(output / 'gps.gpx').getroot().findall('.//{*}trkpt')), 1)
        self.assertEqual(json.loads((output / 'gps_report.json').read_text())['unique_fixes'], 1)

    def test_bad_magic_and_record_size(self):
        self.path.write_bytes(bytes(100))
        with self.assertRaisesRegex(ValueError, 'magic'):
            extract_gps(self.path)
        insv(self.path, fix() + b'x')
        with self.assertRaisesRegex(ValueError, '53-byte'):
            extract_gps(self.path)

    def test_directory_bounds(self):
        insv(self.path, fix())
        data = bytearray(self.path.read_bytes())
        struct.pack_into('<I', data, len(data) - 82, 2**31)
        self.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'outside trailer'):
            extract_gps(self.path)

    def test_void_nan_and_missing_gps(self):
        insv(self.path, fix(status=b'V') + fix(lat=float('nan')))
        self.assertEqual(extract_gps(self.path)[2]['georeferencing_status'], 'no_valid_gps')
        insv(self.path, b'')
        self.assertEqual(extract_gps(self.path)[2]['gps_records'], 0)

    def test_candidate_track_still_requires_alignment(self):
        insv(self.path, b''.join(fix(epoch=1788370677 + i, lat=3.69 + i / 10000) for i in range(4)))
        report = extract_gps(self.path)[2]
        self.assertTrue(report['has_candidate_trajectory'])
        self.assertEqual(report['georeferencing_status'], 'requires_time_sync_and_alignment_validation')

    def test_a_fix_left_from_before_the_recording_is_dropped(self):
        track = [{'timestamp_epoch': t} for t in (991.0, 1000.0, 1050.0, 1100.0)]
        self.assertEqual([f['timestamp_epoch'] for f in drop_stale_start(track, 1000.0, 100.0)], [1000.0, 1050.0, 1100.0])
        # A camera clock that runs ahead makes every fix look early; the track still fits the recording.
        self.assertEqual(drop_stale_start(track[1:], 1020.0, 100.0), track[1:])
        self.assertEqual(drop_stale_start(track, None, 100.0), track)
        self.assertEqual(drop_stale_start(track, 1000.0, None), track)

    def test_info_fields_skip_what_they_do_not_need_and_survive_damage(self):
        payload = bytes([8, 7]) + bytes([18, 2, 1, 2]) + bytes([0xA0, 2, 0x96, 1]) + bytes([29, 0, 0, 0, 0]) + bytes([80, 5])
        self.assertEqual(_info_fields(payload, (10, 36)), {36: 150, 10: 5})
        self.assertEqual(_info_fields(payload[:-1], (10, 36)), {36: 150})


if __name__ == '__main__':
    unittest.main()
