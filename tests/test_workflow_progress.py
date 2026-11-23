import json
import tempfile
import unittest
from pathlib import Path

from raven_app.workflow_progress import (
    add_stage_ratio,
    dataset_times_path,
    estimate_stage_seconds,
    load_stage_history,
    load_stage_ratios,
    parse_progress_line,
    save_stage_history,
    save_stage_ratios,
    stage_duration_estimates,
)


def write_bag(path, seconds):
    """A ROS1 bag whose index spans ``seconds`` (two IMU messages)."""
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    store = get_typestore(Stores.ROS1_NOETIC)
    Imu = store.types['sensor_msgs/msg/Imu']
    Header, Time = store.types['std_msgs/msg/Header'], store.types['builtin_interfaces/msg/Time']
    Vector3, Quaternion = store.types['geometry_msgs/msg/Vector3'], store.types['geometry_msgs/msg/Quaternion']
    import numpy as np
    cov = np.zeros(9)
    with Writer(path) as writer:
        conn = writer.add_connection('/imu', Imu.__msgtype__, typestore=store)
        for t in (100.0, 100.0 + seconds):
            msg = Imu(Header(0, Time(int(t), 0), 'imu'), Quaternion(0, 0, 0, 1), cov,
                      Vector3(0, 0, 0), cov, Vector3(0, 0, 9.81), cov)
            writer.write(conn, int(t * 1e9), store.serialize_ros1(msg, Imu.__msgtype__))


class WorkflowProgressTests(unittest.TestCase):
    def test_parses_stages_native_progress_and_completion(self):
        self.assertEqual(parse_progress_line('[STAGE 3/7] Running FAST-LIVO2 Native SLAM...'), {
            'kind': 'stage', 'stage': 3, 'total': 7, 'name': 'Running FAST-LIVO2 Native SLAM'
        })
        self.assertEqual(parse_progress_line('Vulkan frames 25/100'), {
            'kind': 'stage_fraction', 'fraction': .25
        })
        self.assertEqual(parse_progress_line('[700/1400] SLAM scans mapped, 13 ms per scan'), {
            'kind': 'stage_fraction', 'fraction': .5
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
            self.assertAlmostEqual(estimates[6], 20 + 30_301_126 / 200_000 + 0.05 * 4)

    def test_image_count_before_extraction_comes_from_the_recording_length(self):
        # A 1 GB video of a 140 s walk used to be read as ~6,400 frames (about 1 h of ETA for 8 min of work).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag = root / 'scan.bag'
            write_bag(bag, seconds=140.0)
            kwargs = dict(method='direct', use_masks=True, fps=1.0, lenses=2)
            estimates = estimate_stage_seconds(root, bag, None, **kwargs)
            self.assertAlmostEqual(estimates[4], 280 * 1.0, delta=2)  # Vulkan masks for 280 frames
            self.assertAlmostEqual(estimate_stage_seconds(root, bag, None, mask_backend='tensorrt', **kwargs)[4],
                                   280 * 0.3, delta=1)
            self.assertAlmostEqual(estimate_stage_seconds(root, bag, None, **{**kwargs, 'fps': 2.0})[4],
                                   560 * 1.0, delta=4)
            self.assertGreater(estimates[2], 1.0)  # SLAM still to run

    def test_slam_cache_of_another_recording_is_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bag = root / 'scan.bag'
            write_bag(bag, seconds=10.0)
            slam = root / 'slam_raw_abc'
            (slam / 'pcd').mkdir(parents=True)
            (slam / 'result').mkdir()
            (slam / 'pcd' / 'all_raw_points.pcd').write_bytes(b'POINTS 2000000\nDATA binary\n')
            (slam / 'result' / 'trajectory.txt').write_text('')
            (root / 'active_slam.json').write_text(json.dumps({'relative_directory': 'slam_raw_abc'}))
            (slam / 'input_config.json').write_text(json.dumps({'size': bag.stat().st_size}))
            self.assertEqual(estimate_stage_seconds(root, bag, None)[2], 1.0)
            (slam / 'input_config.json').write_text(json.dumps({'size': bag.stat().st_size + 1}))
            self.assertGreater(estimate_stage_seconds(root, bag, None)[2], 1.0)

    def test_optional_work_adds_time_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = estimate_stage_seconds(tmp, None, None, method='direct', use_masks=False)
            dynamic = estimate_stage_seconds(tmp, None, None, method='direct', use_masks=False,
                                             remove_dynamic=True)
            refined = estimate_stage_seconds(tmp, None, None, method='direct', use_masks=False,
                                             rig_refine='photometric')
            voted = estimate_stage_seconds(tmp, None, None, method='direct', use_masks=True,
                                           remove_dynamic=True)
            masked = estimate_stage_seconds(tmp, None, None, method='direct', use_masks=True)
            self.assertEqual(base[4], 0.0)
            self.assertGreater(dynamic[4], 0.0)
            self.assertAlmostEqual(refined[5] - base[5], 80.0)
            self.assertEqual(dynamic[5], base[5])  # no masks, no standing-person vote
            self.assertGreater(voted[5], masked[5])

    def test_stage_ratios_are_learned_clamped_and_kept_beside_the_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            ratios = [[] for _ in range(7)]
            self.assertFalse(add_stage_ratio(ratios, 1, 0.01, 5.0))  # trivial stage
            self.assertFalse(add_stage_ratio(ratios, 4, 30.0, 0.0))  # skipped stage
            self.assertTrue(add_stage_ratio(ratios, 2, 70.0, 66.0))
            add_stage_ratio(ratios, 5, 1000.0, 100.0)
            add_stage_ratio(ratios, 5, 1.0, 100.0)
            self.assertAlmostEqual(ratios[2][0], 70 / 66)
            self.assertEqual(ratios[5], [5.0, 0.2])
            for _ in range(12):
                add_stage_ratio(ratios, 0, 30.0, 60.0)
            self.assertEqual(len(ratios[0]), 8)

            save_stage_history(tmp, [[1.0]] * 7)
            path = dataset_times_path(tmp)
            save_stage_ratios(path, ratios)
            save_stage_history(tmp, [[2.0]] * 7)
            self.assertEqual(load_stage_ratios(path)[5], [5.0, 0.2])
            self.assertEqual(load_stage_history(tmp)[0], [2.0])
            self.assertEqual(load_stage_ratios(Path(tmp) / 'missing.json'), [[] for _ in range(7)])

    def test_estimates_prefer_dataset_ratios_then_history_then_this_computer(self):
        defaults = [100.0] * 7
        own = [[0.5]] + [[] for _ in range(6)]
        legacy = [[70.0], [70.0]] + [[] for _ in range(5)]
        machine = [[2.0], [2.0], [2.0]] + [[] for _ in range(4)]
        estimates = stage_duration_estimates(defaults, legacy, own, machine)
        self.assertEqual(estimates[:4], [50.0, 70.0, 200.0, 100.0])
        cached = stage_duration_estimates([1.0] * 7, legacy, own, machine)
        self.assertEqual(cached, [1.0] * 7)

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
