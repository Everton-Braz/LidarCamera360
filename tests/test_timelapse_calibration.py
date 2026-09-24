import unittest

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from raven_app.timelapse_calibration import fit_timed_poses


class TimelapseCalibrationTests(unittest.TestCase):
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

    def test_rejects_unrelated_reconstruction(self):
        rng=np.random.default_rng(5)
        ts=np.linspace(0,100,101)
        positions=rng.normal(size=(101,3))*30
        with self.assertRaisesRegex(ValueError,'consensus'):
            fit_timed_poses(rng.normal(size=(80,3))*10,
                            Rotation.identity(80).as_matrix(),np.linspace(10,90,80),
                            ts,positions,Rotation.identity(101),0)


if __name__ == '__main__':
    unittest.main()
