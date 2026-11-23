import io
import tempfile
import unittest
from pathlib import Path

from raven_app.slam_engine_output import forward_engine_output

TABLE = [
    '\x1b[1;34m+-------------------------------------------------------------+\x1b[0m',
    '\x1b[1;34m|                         LIO Mapping Time                    |\x1b[0m',
    '\x1b[1;34m+-------------------------------------------------------------+\x1b[0m',
    '\x1b[1;34m| Algorithm Stage               | Time (secs)                 |\x1b[0m',
    '\x1b[1;36m| DownSample                    | 0.000095                    |\x1b[0m',
    '\x1b[1;36m| ICP                           | 0.003577                    |\x1b[0m',
    '\x1b[1;36m| Current Total Time            | 0.005781                    |\x1b[0m',
    '\x1b[1;36m| Average Total Time            | 0.012600                    |\x1b[0m',
    '\x1b[1;34m+-------------------------------------------------------------+\x1b[0m',
]
SCAN = (['[ LIO ] Raw feature num: 3789, downsampled feature num:3767 effective feature num: 48 average residual: 0.04'] * 7
        + ['[ LIO ] Update Voxel Map', '\x1b[31m[DEBUG]: Last sliding length 0.0135343\x1b[0m'] + TABLE)
START = ['scale: 0.25', 'FIRST LIDAR FRAME!', 'Gravity Alignment Starts', 'Gravity Alignment Finished']
END = ['\x1b[31m[DEBUG]: Map sliding using 0.5 secs\x1b[0m', '\x1b[31m[DEBUG]: Delete 12 root voxels\x1b[0m',
       '\x1b[1;32m[FAST-LIVO2] savePCD() invoked! Points to save: 5535566\x1b[0m',
       'Completed: 5535566 points in 22.0967 seconds']


def _engine(scans):
    return [(line + '\n').encode() for line in START + SCAN * scans + END]


class EngineOutputTests(unittest.TestCase):
    def test_keeps_events_drops_per_scan_noise_and_logs_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'fastlivo2.log'
            out = io.StringIO()
            self.assertEqual(forward_engine_output(iter(_engine(100)), log, total_scans=100, out=out), 100)
            lines = out.getvalue().splitlines()
            self.assertEqual(lines[:4], START)
            self.assertIn('[FAST-LIVO2] savePCD() invoked! Points to save: 5535566', lines)
            self.assertIn('Completed: 5535566 points in 22.0967 seconds', lines)
            progress = [line for line in lines if 'SLAM scans mapped' in line]
            self.assertEqual(len(progress), 20)
            self.assertEqual(progress[-1], '[100/100] SLAM scans mapped, 13 ms per scan')
            self.assertTrue(lines[-1].startswith('[*] FAST-LIVO2 mapped 100 scans (13 ms per scan)'))
            self.assertLess(len(lines), 35)
            self.assertFalse(any('\x1b' in line or 'Raw feature' in line or 'DEBUG' in line for line in lines))
            self.assertEqual(len(log.read_text(encoding='utf-8').splitlines()), len(_engine(100)))

    def test_unknown_scan_count_reports_every_250_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            forward_engine_output(iter(_engine(600)), Path(tmp) / 'fastlivo2.log', out=out)
            progress = [line for line in out.getvalue().splitlines() if 'scans mapped' in line and '[*] FAST-LIVO2:' in line]
            self.assertEqual(progress, ['[*] FAST-LIVO2: 250 scans mapped, 13 ms per scan',
                                        '[*] FAST-LIVO2: 500 scans mapped, 13 ms per scan'])


if __name__ == '__main__':
    unittest.main()
