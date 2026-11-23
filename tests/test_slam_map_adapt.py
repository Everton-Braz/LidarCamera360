"""Moving-start detection and SLAM runaway detection (adaptive FAST-LIVO2 map voxel)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from raven_app.workflow import (
    COARSE_MAP_VOXEL_M,
    SLAM_RUNAWAY_CAP_SHARE,
    _yaml_number,
    slam_velocity_cap_share,
    starts_in_motion,
)


def write_trajectory(path, positions, rate_hz=10.0):
    t = 1.79e9 + np.arange(len(positions)) / rate_hz
    rows = np.column_stack([t, positions, np.tile([0.0, 0.0, 0.0, 1.0], (len(positions), 1))])
    np.savetxt(path, rows, fmt='%.6f')


class StartMotionTest(unittest.TestCase):
    def test_still_starts_are_not_moving(self):
        # Measured on full recordings: gyro 0.008–0.069 rad/s, acceleration spread 0.012–0.066 m/s².
        for gyro, acc in [(0.008, 0.012), (0.069, 0.066), (0.052, 0.048)]:
            self.assertFalse(starts_in_motion({'gyro_mean_rad_s': gyro, 'acc_std_m_s2': acc, 'samples': 600}))

    def test_walking_starts_are_moving(self):
        # Measured on trimmed clips and chunks: gyro 0.35–0.61 rad/s, acceleration spread 0.25–0.57 m/s².
        for gyro, acc in [(0.562, 0.567), (0.354, 0.306), (0.609, 0.252), (0.05, 0.30), (0.30, 0.05)]:
            self.assertTrue(starts_in_motion({'gyro_mean_rad_s': gyro, 'acc_std_m_s2': acc, 'samples': 600}))

    def test_unknown_motion_is_not_moving(self):
        self.assertFalse(starts_in_motion(None))
        self.assertFalse(starts_in_motion({}))


class RunawayTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / 'trajectory.txt'

    def tearDown(self):
        self.dir.cleanup()

    def test_walking_trajectory_is_not_at_the_cap(self):
        rng = np.random.default_rng(1)
        steps = np.column_stack([np.full(600, 0.055), rng.normal(0, 0.01, 600), rng.normal(0, 0.002, 600)])
        write_trajectory(self.path, np.cumsum(steps, axis=0))
        self.assertEqual(slam_velocity_cap_share(self.path, 2.0), 0.0)

    def test_diverged_trajectory_is_pinned_at_the_cap(self):
        # 30 % of the run walks, then the filter falls away at the 2 m/s cap.
        walk = np.column_stack([np.full(300, 0.05), np.zeros(300), np.zeros(300)])
        fall = np.column_stack([np.full(700, 0.03), np.zeros(700), np.full(700, -0.1975)])
        write_trajectory(self.path, np.cumsum(np.vstack([walk, fall]), axis=0))
        share = slam_velocity_cap_share(self.path, 2.0)
        self.assertAlmostEqual(share, 0.7, delta=0.01)
        self.assertGreater(share, SLAM_RUNAWAY_CAP_SHARE)

    def test_missing_or_short_inputs_give_none(self):
        self.assertIsNone(slam_velocity_cap_share(self.path, 2.0))
        write_trajectory(self.path, np.zeros((5, 3)))
        self.assertIsNone(slam_velocity_cap_share(self.path, 2.0))
        write_trajectory(self.path, np.zeros((50, 3)))
        self.assertIsNone(slam_velocity_cap_share(self.path, None))
        self.assertIsNone(slam_velocity_cap_share(self.path, 0.0))


class ConfigTest(unittest.TestCase):
    def test_raven_config_needs_the_coarse_voxel_and_eagle_does_not(self):
        root = Path(__file__).resolve().parents[1] / 'FAST-LIVO2' / 'config'
        raven = (root / 'raven.yaml').read_text(encoding='utf-8')
        eagle = (root / 'eagle.yaml').read_text(encoding='utf-8')
        self.assertLess(_yaml_number(raven, 'voxel_size'), COARSE_MAP_VOXEL_M)
        self.assertGreaterEqual(_yaml_number(eagle, 'voxel_size'), COARSE_MAP_VOXEL_M)
        self.assertGreater(_yaml_number(raven, 'max_velocity'), 0)


class ImuStartMotionTest(unittest.TestCase):
    def write_bag(self, path, gyro_amp, acc_amp):
        from rosbags.rosbag1 import Writer
        from rosbags.typesys import Stores, get_typestore
        store = get_typestore(Stores.ROS1_NOETIC)
        Imu, Header, Time = store.types['sensor_msgs/msg/Imu'], store.types['std_msgs/msg/Header'], store.types['builtin_interfaces/msg/Time']
        Vector3, Quaternion = store.types['geometry_msgs/msg/Vector3'], store.types['geometry_msgs/msg/Quaternion']
        cov = np.zeros(9)
        with Writer(path) as writer:
            conn = writer.add_connection('/imu', Imu.__msgtype__, typestore=store)
            for i in range(1000):  # 5 s at 200 Hz
                t = 100.0 + i / 200.0
                phase = 2 * np.pi * 1.8 * t
                msg = Imu(Header(0, Time(int(t), int((t % 1) * 1e9)), 'imu'), Quaternion(0, 0, 0, 1), cov,
                          Vector3(gyro_amp * np.sin(phase), 0.0, gyro_amp * np.cos(phase)), cov,
                          Vector3(0.0, 0.0, 9.81 + acc_amp * np.sin(phase)), cov)
                writer.write(conn, int(t * 1e9), store.serialize_ros1(msg, Imu.__msgtype__))

    def test_reads_motion_over_the_first_seconds(self):
        from raven_app.bag_io import imu_motion_at_start
        with tempfile.TemporaryDirectory() as tmp:
            still, walking = Path(tmp) / 'still.bag', Path(tmp) / 'walking.bag'
            self.write_bag(still, 0.0, 0.0)
            self.write_bag(walking, 0.5, 0.7)
            m_still = imu_motion_at_start(still, '/imu')
            m_walk = imu_motion_at_start(walking, '/imu')
            self.assertEqual(m_still['samples'], 601)  # 3 s window, both ends included
            self.assertFalse(starts_in_motion(m_still))
            self.assertAlmostEqual(m_walk['gyro_mean_rad_s'], 0.5, places=3)
            self.assertAlmostEqual(m_walk['acc_std_m_s2'], 0.7 / np.sqrt(2), delta=0.03)
            self.assertTrue(starts_in_motion(m_walk))
            self.assertIsNone(imu_motion_at_start(walking, '/other'))


if __name__ == '__main__':
    unittest.main()
