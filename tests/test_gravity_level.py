import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from raven_app.gravity_level import (
    LEVEL_FILE,
    level_interrupted,
    level_slam_output,
    leveling_rotation,
    measure_tilt,
)

G = 9.81
TILT = Rotation.from_rotvec(np.radians(4.9) * np.array([np.cos(0.8), np.sin(0.8), 0.0]))


def _walk(times):
    """A 60 s walk around a 6 m loop: varying turn rate, body sway, head bob."""
    heading = 0.5 * times + 0.6 * np.sin(0.7 * times) + 0.15 * np.sin(2.3 * times)
    attitude = (Rotation.from_euler('z', heading[:, None]) * Rotation.from_euler('x', 0.03 * np.sin(11 * times)[:, None])
                * Rotation.from_euler('y', 0.03 * np.sin(9 * times)[:, None]))
    angle = 0.5 * times
    position = np.column_stack([6 * np.cos(angle), 6 * np.sin(angle), 1.2 + 0.03 * np.sin(2 * np.pi * 1.8 * times)])
    return attitude, position


def _write_dataset(root, world=Rotation.identity(), imu_lead=0.3):
    """SLAM output whose map frame is ``world`` times the true level frame (imu.txt is in the body frame)."""
    root = Path(root)
    (root / 'pcd').mkdir(parents=True)
    (root / 'result').mkdir()
    pose_times = 1000.0 + np.arange(0.0, 60.0, 0.1)
    attitude, position = _walk(pose_times - 1000.0)
    rows = np.column_stack([pose_times, world.apply(position), (world * attitude).as_quat()])
    trajectory = root / 'result' / 'Raven_3DMakerPro_Scan.txt'
    np.savetxt(trajectory, rows, fmt='%.6f')
    np.savetxt(root / 'pcd' / 'lidar_poses.txt', rows, fmt='%.6f')

    step = 0.005
    imu_times = np.arange(-imu_lead, 60.0, step)
    attitude, position = _walk(imu_times)
    after, _ = _walk(imu_times + step)
    gyro = (attitude.inv() * after).as_rotvec() / step
    velocity = np.gradient(position, step, axis=0)
    acceleration = np.gradient(velocity, step, axis=0)
    specific_force = attitude.inv().apply(acceleration + [0.0, 0.0, G])
    np.savetxt(root / 'imu.txt', np.column_stack([imu_times + 0.042, gyro, specific_force]), fmt='%.6f')

    grid = np.stack(np.meshgrid(np.linspace(-8, 8, 120), np.linspace(-8, 8, 120)), -1).reshape(-1, 2)
    floor = np.column_stack([grid, np.zeros(len(grid))])
    normals = np.tile([0.0, 0.0, 1.0], (len(grid), 1))
    data = np.zeros(len(grid), dtype=[(name, '<f4') for name in
                                      ('x', 'y', 'z', 'normal_x', 'normal_y', 'normal_z', 'intensity', 'curvature')])
    for names, values in ((('x', 'y', 'z'), world.apply(floor)), (('normal_x', 'normal_y', 'normal_z'),
                                                                   world.apply(normals))):
        for index, name in enumerate(names):
            data[name] = values[:, index]
    data['intensity'] = 7.0
    pcd = root / 'pcd' / 'all_raw_points.pcd'
    pcd.write_bytes((f'# .PCD v0.7\nFIELDS x y z normal_x normal_y normal_z intensity curvature\n'
                     f'SIZE 4 4 4 4 4 4 4 4\nTYPE F F F F F F F F\nCOUNT 1 1 1 1 1 1 1 1\n'
                     f'WIDTH {len(data)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(data)}\nDATA binary\n'
                     ).encode('ascii') + data.tobytes())
    return pcd, trajectory


def _points(pcd):
    header, payload = pcd.read_bytes().split(b'DATA binary\n', 1)
    return np.frombuffer(payload, dtype=[(name, '<f4') for name in
                                         ('x', 'y', 'z', 'nx', 'ny', 'nz', 'intensity', 'curvature')])


class GravityLevelTests(unittest.TestCase):
    def test_measures_the_tilt_of_a_map_started_in_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, trajectory = _write_dataset(tmp, world=TILT)
            record = measure_tilt(Path(tmp) / 'imu.txt', trajectory)
            self.assertEqual(record['status'], 'measured')
            self.assertAlmostEqual(record['tilt_deg'], 4.9, delta=0.1)
            self.assertGreater(record['imu_sync_correlation'], 0.9)
            self.assertGreater(record['gravity_share'], 0.95)

    def test_leveling_rotation_keeps_heading(self):
        up = TILT.apply([0.0, 0.0, 1.0])
        rotation = leveling_rotation(up)
        np.testing.assert_allclose(rotation @ up, [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(rotation @ np.cross(up, [0, 0, 1]), np.cross(up, [0, 0, 1]), atol=1e-12)
        np.testing.assert_allclose(leveling_rotation([0, 0, 2]), np.eye(3))

    def test_levels_cloud_normals_and_trajectories_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcd, trajectory = _write_dataset(root, world=TILT)
            before = _points(pcd).copy()
            self.assertGreater(np.ptp(before['z']), 1.0)
            messages = []
            record = level_slam_output(root, pcd, trajectory, log=messages.append)
            self.assertEqual(record['status'], 'leveled')
            self.assertIn('tilted 4.9', messages[0])
            after = _points(pcd)
            self.assertLess(np.abs(after['z']).max(), 0.08)  # a 16 m floor, flat within 0.3 deg
            self.assertGreater(after['nz'].min(), np.cos(np.radians(0.3)))
            np.testing.assert_array_equal(after['intensity'], before['intensity'])
            for path in (trajectory, root / 'pcd' / 'lidar_poses.txt'):
                rows = np.loadtxt(path)
                self.assertLess(np.ptp(rows[:, 3]), 0.1)  # head height on a level floor
            self.assertLess(measure_tilt(root / 'imu.txt', trajectory)['tilt_deg'], 0.3)
            self.assertFalse(list(root.rglob('*.leveling')))

            stamp = pcd.stat().st_mtime_ns
            self.assertEqual(level_slam_output(root, pcd, trajectory, log=messages.append), record)
            self.assertEqual(pcd.stat().st_mtime_ns, stamp)
            self.assertFalse(level_interrupted(root))

    def test_level_map_is_left_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcd, trajectory = _write_dataset(root)
            content = pcd.read_bytes(), trajectory.read_bytes()
            record = level_slam_output(root, pcd, trajectory, log=lambda message: None)
            self.assertEqual(record['status'], 'level')
            self.assertEqual((pcd.read_bytes(), trajectory.read_bytes()), content)

    def test_short_or_missing_imu_log_is_recorded_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pcd, trajectory = _write_dataset(root, world=TILT)
            (root / 'imu.txt').unlink()
            content = pcd.read_bytes()
            self.assertEqual(level_slam_output(root, pcd, trajectory)['status'], 'no_imu_log')
            self.assertEqual(pcd.read_bytes(), content)

    def test_interrupted_leveling_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(level_interrupted(root))
            (root / LEVEL_FILE).write_text(json.dumps({'status': 'pending'}))
            self.assertTrue(level_interrupted(root))


if __name__ == '__main__':
    unittest.main()
