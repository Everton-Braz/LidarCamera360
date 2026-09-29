import tempfile
import unittest
from pathlib import Path

from raven_app.workflow_progress import (
    estimate_stage_seconds,
    load_stage_history,
    parse_progress_line,
    save_stage_history,
    stage_duration_estimates,
)


class WorkflowProgressTests(unittest.TestCase):
    def test_parses_stages_native_progress_and_completion(self):
        self.assertEqual(parse_progress_line('[STAGE 3/7] Running FAST-LIVO2 Native SLAM...'), {
            'kind': 'stage', 'stage': 3, 'total': 7, 'name': 'Running FAST-LIVO2 Native SLAM'
        })
        self.assertEqual(parse_progress_line('Vulkan frames 25/100'), {
            'kind': 'stage_fraction', 'fraction': .25
        })
        self.assertEqual(parse_progress_line('[+] 3624 combined foreground masks ready: path'), {
            'kind': 'masks_ready'
        })
        self.assertEqual(parse_progress_line('[*] Reusing 3624 combined foreground masks'), {
            'kind': 'masks_ready'
        })
        self.assertEqual(parse_progress_line('[SUCCESS] Unified Workflow Complete in 20s!'), {
            'kind': 'complete'
        })
        self.assertIsNone(parse_progress_line('ordinary log output'))

    def test_eta_uses_cached_files_and_dataset_workload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for camera in ('cam0', 'cam1'):
                directory = root / 'images' / camera
                directory.mkdir(parents=True)
                for index in range(2):
                    (directory / f'frame_{index:04d}.jpg').write_bytes(b'')
            pcd = root / 'slam_out' / 'pcd' / 'all_raw_points.pcd'
            pcd.parent.mkdir(parents=True)
            pcd.write_bytes(b'FIELDS x y z rgb\nWIDTH 30301126\nHEIGHT 1\nPOINTS 30301126\nDATA binary\n')
            trajectory = root / 'slam_out' / 'result' / 'Raven_3DMakerPro_Scan.txt'
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text('')
            sparse = root / 'sparse' / '0' / 'points3D.bin'
            sparse.parent.mkdir(parents=True)
            sparse.write_bytes(b'present')

            estimates = estimate_stage_seconds(root, None, None)
            self.assertEqual(estimates[0], 1.)
            self.assertEqual(estimates[2], 1.)
            self.assertEqual(estimates[3], 30.)
            self.assertGreater(estimates[4], 0.)
            self.assertGreater(estimates[5], 0.)
            self.assertAlmostEqual(estimates[6], 363.01126)

    def test_stage_timing_history_is_limited_and_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = [[1, 2, 3, 4, 5, 6], [], [], [], [], [], [10, 20]]
            save_stage_history(tmp, history)
            loaded = load_stage_history(tmp)
            self.assertEqual(loaded[0], [2., 3., 4., 5., 6.])
            self.assertEqual(len(loaded), 7)
            self.assertEqual(loaded[6], [10., 20.])
            self.assertEqual(stage_duration_estimates([10.] * 7, loaded)[0], 4.)

    def test_disabled_mask_stage_does_not_use_old_mask_timing(self):
        defaults = [1., 20., 1., 30., 0., 60., 300.]
        history = [[], [], [], [], [900.], [], []]
        estimates = stage_duration_estimates(defaults, history)
        self.assertEqual(estimates[4], 0.)

    def test_legacy_combined_history_does_not_bias_split_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'logs' / 'workflow_stage_times.json'
            path.parent.mkdir()
            path.write_text('{"stages":{"1":[1],"2":[2],"3":[3],"4":[4],"5":[50],"6":[60]}}')
            loaded = load_stage_history(tmp)
            self.assertEqual(loaded, [[1.], [2.], [3.], [4.], [], [], [60.]])


if __name__ == '__main__':
    unittest.main()
