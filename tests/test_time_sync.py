import unittest

import numpy as np

from raven_app.time_sync import estimate_gyro_time_sync


class GyroTimeSyncTests(unittest.TestCase):
    def test_finds_short_camera_clip_late_in_long_lidar_trace(self):
        rng = np.random.default_rng(47)
        rate = 20.0
        lidar = rng.normal(size=780 * int(rate))
        clip_start_seconds = 631.25
        clip_samples = 28 * int(rate)
        start = round(clip_start_seconds * rate)
        camera = lidar[start:start + clip_samples] + rng.normal(
            0.0, 0.025, clip_samples
        )

        result = estimate_gyro_time_sync(
            np.arange(camera.size) / rate,
            camera,
            1_790_000_000.0 + np.arange(lidar.size) / rate,
            lidar,
            sample_rate_hz=rate,
        )

        self.assertTrue(result.accepted, result)
        self.assertAlmostEqual(result.dt_seconds, -clip_start_seconds, delta=1 / rate)
        self.assertGreater(result.score, 0.95)
        self.assertAlmostEqual(result.overlap_seconds, 28.0, delta=1 / rate)
        self.assertGreater(result.ambiguity_margin, 0.1)
        self.assertEqual(result.reason, "accepted")

    def test_weak_noise_match_is_rejected_without_fallback_offset(self):
        rate = 20.0
        camera_rng = np.random.default_rng(101)
        lidar_rng = np.random.default_rng(202)
        camera = camera_rng.normal(size=24 * int(rate))
        lidar = lidar_rng.normal(size=700 * int(rate))

        result = estimate_gyro_time_sync(
            np.arange(camera.size) / rate,
            camera,
            np.arange(lidar.size) / rate,
            lidar,
            sample_rate_hz=rate,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.dt_seconds)
        self.assertEqual(result.reason, "weak_correlation")
        self.assertLess(result.score, 0.55)
        self.assertGreaterEqual(result.overlap_seconds, 5.0)

    def test_repeated_motion_is_rejected_as_ambiguous(self):
        rng = np.random.default_rng(9)
        rate = 20.0
        camera = rng.normal(size=15 * int(rate))
        lidar = np.concatenate((
            rng.normal(size=4 * int(rate)),
            camera,
            rng.normal(size=11 * int(rate)),
            camera,
            rng.normal(size=5 * int(rate)),
        ))

        result = estimate_gyro_time_sync(
            np.arange(camera.size) / rate,
            camera,
            np.arange(lidar.size) / rate,
            lidar,
            sample_rate_hz=rate,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.dt_seconds)
        self.assertEqual(result.reason, "ambiguous_peak")
        self.assertGreater(result.score, 0.99)
        self.assertLess(result.ambiguity_margin, 0.01)

    def test_rejects_overlap_shorter_than_minimum(self):
        rate = 20.0
        camera = np.sin(np.arange(160) * 0.2)
        lidar = np.sin(np.arange(80) * 0.2)

        result = estimate_gyro_time_sync(
            np.arange(camera.size) / rate,
            camera,
            np.arange(lidar.size) / rate,
            lidar,
            sample_rate_hz=rate,
            min_overlap_seconds=5.0,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.dt_seconds)
        self.assertEqual(result.reason, "insufficient_overlap")

    def test_constant_signal_cannot_produce_a_sync(self):
        rate = 20.0
        camera = np.ones(12 * int(rate))
        lidar = np.ones(40 * int(rate))

        result = estimate_gyro_time_sync(
            np.arange(camera.size) / rate,
            camera,
            np.arange(lidar.size) / rate,
            lidar,
            sample_rate_hz=rate,
        )

        self.assertFalse(result.accepted)
        self.assertIsNone(result.dt_seconds)
        self.assertEqual(result.reason, "insufficient_signal_variation")


if __name__ == "__main__":
    unittest.main()
