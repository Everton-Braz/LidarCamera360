"""Hand-eye start of a new rig: clock and mount rotation from motion."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from raven_app.camera_models import CameraModel
from raven_app.rig_handeye import bearings, estimate_clock, estimate_mount_rotation

FISHEYE = dict(fx=1080.0, fy=1080.0, cx=1920.0, cy=1920.0, k1=0.083, k2=-0.0295, p1=0.0, p2=0.0,
               k3=0.0109, k4=-0.00278, sx1=0.0, sy1=0.0, width=3840, height=3840)
OPENCV = dict(fx=2646.0, fy=2650.0, cx=1920.0, cy=1080.0, k1=0.05, k2=-0.01, p1=0.001, p2=-0.0005,
              width=3840, height=2160)


@pytest.mark.parametrize('model,intrinsics,max_angle', [
    ('THIN_PRISM_FISHEYE', FISHEYE, 95.0), ('OPENCV_FISHEYE', FISHEYE, 95.0), ('OPENCV', OPENCV, 30.0),
    ('PINHOLE', OPENCV, 30.0), ('EQUIRECTANGULAR', {'width': 7680, 'height': 3840}, 170.0)])
def test_bearings_invert_the_projection(model, intrinsics, max_angle):
    camera = CameraModel.from_intrinsics(model, intrinsics)
    rng = np.random.default_rng(0)
    rays = rng.normal(size=(4000, 3))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    rays = rays[np.degrees(np.arccos(rays[:, 2])) < max_angle]
    u, v, valid, _ = camera.project(rays * 7.0)
    back = bearings(camera, u[valid], v[valid])
    assert np.degrees(np.arccos(np.clip(np.sum(back * rays[valid], axis=1), -1, 1))).max() < 0.05


def _trajectory(seconds=200.0, rate=10.0, seed=1):
    """A walk with turns: yaw sweeps plus some pitch/roll wobble."""
    t = np.arange(0.0, seconds, 1.0 / rate)
    yaw = np.cumsum(np.sin(t / 7.0) * 0.6 + np.sin(t / 2.3) * 0.25) / rate
    pitch = 0.15 * np.sin(t / 3.1) + 0.05 * np.sin(t * 1.3)
    roll = 0.1 * np.sin(t / 4.7) + 0.04 * np.sin(t * 0.9 + 1.0)
    rotations = Rot.from_euler('zyx', np.column_stack([yaw, pitch, roll]))
    return t, np.zeros((len(t), 3)), rotations


def _camera_pairs(trajectory, mount, dt, frame_times, noise_deg=0.3, outliers=0.05, seed=2):
    t, _, rotations = trajectory
    rng = np.random.default_rng(seed)
    out = []
    for a, b in zip(frame_times[:-1], frame_times[1:]):
        ra = rotations[np.argmin(np.abs(t - (t[0] + a - dt)))]
        rb = rotations[np.argmin(np.abs(t - (t[0] + b - dt)))]
        body = (ra.inv() * rb).as_matrix()
        camera = mount.T @ body @ mount                     # camera-frame relative rotation
        rotvec = Rot.from_matrix(camera).as_rotvec() + np.deg2rad(noise_deg) * rng.normal(size=3) / np.sqrt(3)
        if rng.random() < outliers:
            rotvec = rng.normal(size=3) * 0.3
        out.append((a, b, rotvec, 100))
    return out


def test_clock_and_mount_are_recovered_from_a_wrong_start():
    trajectory = _trajectory()
    mount = Rot.from_euler('xyz', [-31.0, 4.0, 92.0], degrees=True).as_matrix() @ np.array(
        [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    true_dt = -7.0
    frame_times = np.arange(20.0, 185.0, 0.5)    # 2 fps camera clock
    pairs = _camera_pairs(trajectory, mount, true_dt, frame_times)

    dt, clock = estimate_clock(trajectory, pairs, dt_init=-4.0, window=10.0)
    assert abs(dt - true_dt) < 0.1 and clock['contrast'] > 5
    dt, _ = estimate_clock(trajectory, pairs, dt, 0.5, step=0.005)
    assert abs(dt - true_dt) < 0.06

    R, report = estimate_mount_rotation(trajectory, pairs, dt)
    error = np.degrees(np.linalg.norm(Rot.from_matrix(R.T @ mount).as_rotvec()))
    assert error < 1.0
    assert report['median_residual_deg'] < 1.0


def test_a_rig_that_never_turns_is_refused():
    t = np.arange(0.0, 100.0, 0.1)
    still = (t, np.zeros((len(t), 3)), Rot.identity(len(t)))
    pairs = [(a, a + 0.5, np.zeros(3), 50) for a in np.arange(10.0, 80.0, 0.5)]
    with pytest.raises(ValueError, match='turn the rig'):
        estimate_mount_rotation(still, pairs, 0.0)
