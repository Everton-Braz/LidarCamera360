import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

import raven_app.slam_refinement_geometry as geo
from raven_app import slam_refinement as sr
from raven_app.slam_refinement_constraints import doubled_surfaces
from raven_app.slam_refinement_geometry import Node


def trajectory(tmp_path, jump_at=None, jump=.15):
    times = 1000. + np.arange(0., 60., .1)
    positions = np.column_stack([.4*(times-1000.), np.sin((times-1000.)/5.), np.full_like(times, 1.)])
    if jump_at is not None:
        positions[times >= 1000.+jump_at, 1] += jump
    rows = np.column_stack([times, positions, np.tile([0., 0., 0., 1.], (len(times), 1))])
    path = tmp_path/'t.txt'
    np.savetxt(path, rows, fmt='%.9f')
    return geo.read_trajectory(path)


def test_a_step_in_the_pose_is_found_and_smooth_motion_is_not(tmp_path):
    assert len(geo.detect_pose_jumps(trajectory(tmp_path))) == 0
    found = geo.detect_pose_jumps(trajectory(tmp_path, 30.))
    assert len(found) == 1 and abs(found[0]-1030.) < .3


def test_imu_explains_real_acceleration(tmp_path):
    traj = trajectory(tmp_path, 30.)
    times = traj.times[0]-.3+np.arange(0., 60.6, .005)
    gravity = np.tile([0., 0., 9.8], (len(times), 1))
    gyro = np.zeros((len(times), 3))
    gyro[:, 2] = .5+.4*np.sin(times/3.)  # something for the clock match to lock onto
    imu = np.column_stack([times-times[0]+.3, gyro, gravity])
    path = tmp_path/'imu.txt'
    np.savetxt(path, imu, fmt='%.6f')
    rotations = Rotation.from_rotvec(np.column_stack([np.zeros((len(traj.times), 2)),
                                                      np.cumsum(np.interp(traj.times, times, gyro[:, 2])*.1)]))
    rows = np.column_stack([traj.times, traj.positions, rotations.as_quat()])
    source = tmp_path/'t2.txt'
    np.savetxt(source, rows, fmt='%.9f')
    moved, acceleration = geo.read_imu_world_acceleration(path, geo.read_trajectory(source))
    assert len(moved) and abs(moved[0]-traj.times[0]) < .2
    assert np.abs(acceleration).max() < .05  # gravity removed, on the trajectory's clock


def chain(count=12, segments=None):
    return [Node(i, 10.*i, 10.*i+10., 10.*i+5., Rotation.identity(), np.array([float(i), 0., 0.]), np.zeros((0, 3)), (),
                 0 if segments is None else segments[i]) for i in range(count)]


def test_correction_steps_at_a_jump_and_nowhere_else():
    nodes = chain(4, [0, 0, 1, 1])
    parameters = np.zeros(18)
    parameters[6+3:6+6] = [0., .14, 0.]   # node 2
    parameters[12+3:12+6] = [0., .14, 0.]  # node 3
    rotations, translations = geo.correction_curve(nodes, parameters, np.array([14., 19.9, 20.1, 30.]))
    assert np.allclose(translations[:2], 0.)
    assert np.allclose(translations[2:], [[0., .14, 0.]]*2)
    assert np.allclose(rotations.magnitude(), 0.)


def test_one_wrong_loop_does_not_bend_the_map():
    nodes = chain()
    truth = {(i, j): (Rotation.identity(), np.array([float(i-j), 0., 0.]))
             for i, j in [(0, 4), (1, 6), (2, 8), (3, 9), (5, 11), (0, 10), (4, 11), (2, 7)]}
    loops = [(i, j, *pose, 'loop') for (i, j), pose in truth.items()]
    loops.append((3, 7, Rotation.from_euler('z', 15., degrees=True), np.array([2., 6., 0.]), 'loop'))
    size = (len(nodes)-1)*6
    parameters, _, report = sr.solve(nodes, loops, csr_matrix((0, size)), np.zeros(0))
    assert report['engine'] == 'RootSLAM-SquareRootBA'
    assert report['loops_dropped'] == 1 and report['loop_weights'][-1] < .01
    assert np.abs(parameters).max() < .01


def test_correction_bounds_grow_with_the_walk():
    short, long = chain(5), chain(600)
    assert sr.correction_limits(short)[0] == 1.
    assert sr.correction_limits(long)[0] > 5.
    assert sr.loop_limits(long, 0, 500)[0] > sr.loop_limits(long, 0, 5)[0]


def wall_nodes(offset):
    rng = np.random.default_rng(3)
    wall = lambda: np.column_stack([rng.uniform(-4, 4, 4000), np.full(4000, 2.)+rng.normal(0, .002, 4000), rng.uniform(0, 3, 4000)])
    return [Node(0, 0., 10., 5., Rotation.identity(), np.zeros(3), wall(), ()),
            Node(1, 40., 50., 45., Rotation.identity(), np.array([0., offset, 0.]), wall(), ())]


def test_doubled_wall_is_measured_and_a_correction_removes_it():
    nodes = wall_nodes(.14)
    assert doubled_surfaces(nodes)['share_over_6cm'] == 1.
    assert doubled_surfaces(nodes, np.array([0., 0., 0., 0., -.14, 0.]))['share_over_6cm'] == 0.


def test_more_doubling_rejects_even_with_better_surfaces():
    quality = {'improved': True, 'doubled_surfaces': {'share_over_6cm': .02}, 'doubled_surfaces_after': {'share_over_6cm': .03}}
    assert not sr.accept_refinement(quality, True)
    quality['doubled_surfaces_after']['share_over_6cm'] = .01
    assert sr.accept_refinement(quality, True)
