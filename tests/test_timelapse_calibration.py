import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from raven_app.timelapse_calibration import (
    estimate_lever_arm,
    fit_timed_poses,
    validate_lever_refinement,
)


class TimelapseCalibrationTests(unittest.TestCase):
    def test_searches_full_slam_recording_for_late_camera_clip_without_gyro_hint(self):
        ts = np.linspace(0, 1000, 5001)
        positions = np.column_stack((
            24*np.sin(ts/47), 11*np.cos(ts/29), .8*np.sin(ts/17)))
        rotations = Rotation.from_euler(
            'zyx', np.column_stack((
                .65*np.sin(ts/12), .17*np.sin(ts/21), .1*np.cos(ts/16))))
        tc = np.linspace(8, 88, 41)
        dt = -631.25
        query = tc-dt
        body_rotation = Slerp(ts, rotations)(query)
        arm = rotations[0].inv().apply([0, 0, .185])
        target = np.column_stack([
            np.interp(query, ts, positions[:, axis]) for axis in range(3)
        ]) + body_rotation.apply(arm)
        alignment = Rotation.from_euler('xyz', [.04, -.03, .6])
        translation = np.array([7., -5., 1.])
        scale = 4.6
        centers = alignment.inv().apply(target-translation)/scale
        rig = Rotation.from_euler('xyz', [.1, -.2, .7])
        world_to_camera = (alignment.inv()*body_rotation*rig).inv().as_matrix()

        fit, found_dt, accepted, rmse, _ = fit_timed_poses(
            centers, world_to_camera, tc, ts, positions, rotations)

        self.assertAlmostEqual(found_dt, dt, delta=.03)
        self.assertAlmostEqual(fit[0], scale, delta=.01)
        self.assertLess(rmse, .01)
        self.assertEqual(int(accepted.sum()), len(tc))

    def test_searches_majority_overlap_when_camera_clip_is_longer_than_slam(self):
        ts = np.linspace(0, 100, 1001)
        positions = np.column_stack((
            18*np.sin(ts/15), 9*np.cos(ts/19), .5*np.sin(ts/11)))
        rotations = Rotation.from_euler(
            'zyx', np.column_stack((
                .55*np.sin(ts/10), .16*np.sin(ts/17), .09*np.cos(ts/13))))
        tc = np.linspace(.1, 104.7, 205)
        dt = 2.45
        query = np.clip(tc-dt, ts[0], ts[-1])
        body_rotation = Slerp(ts, rotations)(query)
        arm = rotations[0].inv().apply([0, 0, .185])
        target = np.column_stack([
            np.interp(query, ts, positions[:, axis]) for axis in range(3)
        ]) + body_rotation.apply(arm)
        alignment = Rotation.from_euler('xyz', [.04, -.03, .6])
        translation = np.array([7., -5., 1.])
        scale = 4.6
        centers = alignment.inv().apply(target-translation)/scale
        rig = Rotation.from_euler('xyz', [.1, -.2, .7])
        world_to_camera = (alignment.inv()*body_rotation*rig).inv().as_matrix()

        fit, found_dt, accepted, rmse, _ = fit_timed_poses(
            centers, world_to_camera, tc, ts, positions, rotations)

        self.assertAlmostEqual(found_dt, dt, delta=.03)
        self.assertAlmostEqual(fit[0], scale, delta=.01)
        self.assertLess(rmse, .01)
        self.assertGreaterEqual(int(accepted.sum()), 190)

    def test_recovers_clock_and_rig_despite_repeated_facade_matches(self):
        ts = np.linspace(0, 200, 2001)
        positions = np.column_stack((20*np.sin(ts/37), 12*np.cos(ts/23), .2*np.sin(ts/10)))
        rotations = Rotation.from_euler('zyx', np.column_stack((.8*np.sin(ts/8), .2*np.sin(ts/13), .1*np.cos(ts/11))))
        tc = np.linspace(8, 193, 94)
        dt = 3.25
        query = tc-dt
        body_rotation = Slerp(ts, rotations)(query)
        arm = rotations[0].inv().apply([0, 0, .185])
        target = np.column_stack([np.interp(query,ts,positions[:,axis]) for axis in range(3)])+body_rotation.apply(arm)
        alignment = Rotation.from_euler('xyz', [.05, -.02, .7])
        offset = np.array([12., -8., 2.])
        scale = 4.2
        centers = alignment.inv().apply(target-offset)/scale
        rig = Rotation.from_euler('xyz', [.2, -.4, 1.])
        rcw = (alignment.inv()*body_rotation*rig).inv().as_matrix()
        bad = np.zeros(len(tc), dtype=bool)
        bad[30:52] = True
        centers[bad] += np.array([12., -4., 0.])
        fit, found_dt, accepted, rmse, _ = fit_timed_poses(centers,rcw,tc,ts,positions,rotations,4.4)
        self.assertAlmostEqual(found_dt, dt, delta=.01)
        self.assertLess(rmse,.01)
        self.assertFalse(np.any(accepted & bad))
        self.assertGreater(accepted.sum(),65)
        self.assertAlmostEqual(fit[0],scale,delta=.01)

    def test_refines_gyro_hint_inside_local_window_before_consensus_gate(self):
        rng = np.random.default_rng(27)
        ts = np.linspace(0, 100, 10001)
        positions = np.vstack((
            np.zeros(3), np.cumsum(rng.normal(0, .018, size=(len(ts)-1, 3)), axis=0)))
        rotations = Rotation.identity(len(ts))
        tc = np.arange(10., 90.)
        dt = 6.25
        query = tc - dt
        target = np.column_stack([
            np.interp(query, ts, positions[:, axis]) for axis in range(3)
        ]) + np.array([0., 0., .185])
        alignment = Rotation.from_euler('xyz', [.04, -.03, .6])
        translation = np.array([7., -5., 1.])
        scale = 4.6
        centers = alignment.inv().apply(target-translation)/scale

        fit, found_dt, accepted, rmse, _ = fit_timed_poses(
            centers, Rotation.identity(len(tc)).as_matrix(), tc,
            ts, positions, rotations, dt_hint=7.75)

        self.assertAlmostEqual(found_dt, dt, delta=.01)
        self.assertAlmostEqual(fit[0], scale, delta=.01)
        self.assertLess(rmse, .01)
        self.assertEqual(int(accepted.sum()), len(tc))

    def test_rejects_unrelated_reconstruction(self):
        rng=np.random.default_rng(5)
        ts=np.linspace(0,100,101)
        positions=rng.normal(size=(101,3))*30
        with self.assertRaisesRegex(ValueError,'consensus'):
            fit_timed_poses(rng.normal(size=(80,3))*10,
                            Rotation.identity(80).as_matrix(),np.linspace(10,90,80),
                            ts,positions,Rotation.identity(101),0)

    def test_uses_explicit_camera_lever_arm_in_body_frame(self):
        ts = np.linspace(0, 120, 1201)
        positions = np.column_stack((8*np.sin(ts/17), 6*np.cos(ts/21), .1*np.sin(ts/9)))
        rotations = Rotation.from_euler('zyx', np.column_stack((.7*np.sin(ts/7), .15*np.sin(ts/11), .08*np.cos(ts/9))))
        tc = np.linspace(6, 114, 60)
        dt = 2.75
        query = tc-dt
        body_rotation = Slerp(ts, rotations)(query)
        arm = np.array([.01145, .14886, .11122])
        target = np.column_stack([np.interp(query, ts, positions[:,axis]) for axis in range(3)]) + body_rotation.apply(arm)
        alignment = Rotation.from_euler('xyz', [.03, -.04, .5])
        offset = np.array([-4., 3., 1.])
        scale = 5.1
        centers = alignment.inv().apply(target-offset)/scale
        rig = Rotation.from_euler('xyz', [-.2, .25, .7])
        rcw = (alignment.inv()*body_rotation*rig).inv().as_matrix()

        fit, found_dt, accepted, rmse, found_arm = fit_timed_poses(
            centers, rcw, tc, ts, positions, rotations, dt+0.4,
            lever_arm_body_m=arm)

        self.assertAlmostEqual(found_dt, dt, delta=.01)
        self.assertAlmostEqual(fit[0], scale, delta=.01)
        self.assertLess(rmse, .01)
        self.assertEqual(accepted.sum(), len(tc))
        np.testing.assert_allclose(found_arm, arm)

    def _lever_dataset(self, count=120):
        ts = np.linspace(0, 120, 1201)
        positions = np.column_stack((
            10*np.sin(ts/19), 7*np.cos(ts/23), .3*np.sin(ts/8)))
        rotations = Rotation.from_euler(
            'zyx', np.column_stack((
                .75*np.sin(ts/8), .18*np.sin(ts/13), .13*np.cos(ts/11))))
        tc = np.linspace(7, 113, count)
        dt = 3.2
        query = tc-dt
        body_rotation = Slerp(ts, rotations)(query)
        prior = rotations[0].inv().apply([0., 0., .185])
        true_arm = Rotation.from_euler('xyz', [.10, -.08, .03]).apply(prior)
        target = np.column_stack([
            np.interp(query, ts, positions[:, axis]) for axis in range(3)
        ]) + body_rotation.apply(true_arm)
        world_alignment = Rotation.from_euler('xyz', [.04, -.03, .65])
        offset = np.array([8., -5., 1.5])
        scale = 4.8
        centers = world_alignment.inv().apply(target-offset)/scale
        rig = Rotation.from_euler('xyz', [.15, -.3, .8])
        rcw = (world_alignment.inv()*body_rotation*rig).inv().as_matrix()
        return centers, rcw, tc, ts, positions, rotations, prior, true_arm, dt

    def test_refines_lever_direction_with_fixed_physical_length(self):
        (centers, rcw, tc, ts, positions, rotations,
         prior, true_arm, dt) = self._lever_dataset()
        fit, found_dt, accepted, rmse, found_arm, details = estimate_lever_arm(
            centers, rcw, tc, ts, positions, rotations,
            dt_hint=dt+.2, initial_arm=prior)
        self.assertTrue(details['optimizer_success'])
        self.assertAlmostEqual(np.linalg.norm(found_arm), .185, places=8)
        self.assertAlmostEqual(found_dt, dt, delta=.02)
        self.assertLess(np.linalg.norm(found_arm-true_arm), .01)
        self.assertLess(rmse, .01)
        self.assertEqual(accepted.sum(), len(tc))

    def test_blocked_validation_scores_all_shared_valid_test_poses(self):
        (centers, rcw, tc, ts, positions, rotations,
         prior, _, _) = self._lever_dataset(count=160)
        result = validate_lever_refinement(
            centers, rcw, tc, ts, positions, rotations,
            initial_arm=prior, folds=5, purge_frames=2, dt_hint=3.2)
        self.assertEqual(result['fold_count'], 5)
        self.assertTrue(result['shared_test_domain'])
        self.assertEqual(result['test_policy'],
                         'all valid held-out camera poses; no residual trimming')
        self.assertEqual(sum(fold['baseline']['count'] for fold in result['folds']),
                         sum(fold['candidate']['count'] for fold in result['folds']))


if __name__ == '__main__':
    unittest.main()
