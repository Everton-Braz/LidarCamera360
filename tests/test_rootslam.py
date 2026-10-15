import pytest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation

from raven_app.rootslam import (
    SquareRootLandmarkMarginalizer,
    RootSLAMConfig,
    RootSLAMSolver,
    analytical_graph_residual_and_jacobian,
    se3_adjoint,
    so3_right_jacobian_inv,
    skew,
    rootslam_solve
)
from raven_app.slam_refinement_geometry import Node
import raven_app.slam_refinement_geometry as geo
from raven_app.slam_refinement import graph_jacobian


def test_skew_and_adjoint():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    sk = skew(v)
    assert np.allclose(sk + sk.T, 0.0)
    assert np.allclose(sk @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 3.0, -2.0]))

    r = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    t = np.array([0.5, -0.2, 1.2])
    ad = se3_adjoint(r, t)
    assert ad.shape == (6, 6)
    assert np.allclose(ad[:3, :3], r)
    assert np.allclose(ad[3:, 3:], r)
    assert np.allclose(ad[3:, :3], skew(t) @ r)


def test_so3_right_jacobian_inv():
    w_zero = np.zeros(3)
    j_zero = so3_right_jacobian_inv(w_zero)
    assert np.allclose(j_zero, np.eye(3))

    w = np.array([0.1, -0.2, 0.05])
    j = so3_right_jacobian_inv(w)
    assert j.shape == (3, 3)
    assert np.linalg.det(j) > 0.9


def test_square_root_landmark_marginalization():
    # Simulate 5 observations of a 3D landmark from 2 camera poses (each 6-DoF, dp=12 total)
    m = 10  # 5 2D pixel residuals
    dp = 12
    dl = 3   # 3D point parameters

    rng = np.random.default_rng(42)
    jl = rng.standard_normal((m, dl)).astype(np.float32)
    jp = rng.standard_normal((m, dp)).astype(np.float32)
    r = rng.standard_normal(m).astype(np.float32)

    q2_r, q2_jp, q1_r, r1 = SquareRootLandmarkMarginalizer.marginalize_block(r, jp, jl)

    # Check dimensions
    assert q2_r.shape == (m - dl,)
    assert q2_jp.shape == (m - dl, dp)
    assert q1_r.shape == (dl,)
    assert r1.shape == (dl, dl)
    assert np.allclose(np.tril(r1, -1), 0.0)  # Upper triangular

    # Verify exact nullspace orthogonality: Q2^T * Jl must be identically 0
    q, _ = np.linalg.qr(jl, mode='complete')
    q2 = q[:, dl:]
    assert np.allclose(q2.T @ jl, 0.0, atol=1e-5)

    # Verify landmark back-substitution
    delta_xp = rng.standard_normal(dp).astype(np.float32)
    # Recovery formula: delta_xl = -R1^{-1} (q1_r + Q1^T * jp * delta_xp)
    q1 = q[:, :dl]
    delta_xl = -np.linalg.solve(r1, q1_r + (q1.T @ jp) @ delta_xp)
    full_residual = r + jp @ delta_xp + jl @ delta_xl
    reduced_residual = q2_r + q2_jp @ delta_xp
    # The norm squared of the full system equals ||reduced||^2 + ||upper factor||^2
    # and since upper factor is solved to 0 by delta_xl, ||full||^2 == ||reduced||^2!
    assert np.isclose(np.linalg.norm(full_residual), np.linalg.norm(reduced_residual), rtol=1e-4)


def test_rootslam_solver_convergence_synthetic_loop():
    # 4-node closed loop trajectory (origin, +x, +x+y, +y)
    nodes = [
        Node(0, 0.0, 10.0, 5.0, Rotation.identity(), np.array([0.0, 0.0, 0.0]), np.zeros(0), ()),
        Node(1, 10.0, 20.0, 15.0, Rotation.identity(), np.array([5.0, 0.0, 0.0]), np.zeros(0), ()),
        Node(2, 20.0, 30.0, 25.0, Rotation.identity(), np.array([5.0, 5.0, 0.0]), np.zeros(0), ()),
        # Node 3 with slight drift: 0.10m and 0.5 deg
        Node(3, 30.0, 40.0, 35.0, Rotation.from_euler('z', 0.5, degrees=True), np.array([0.10, 5.05, 0.0]), np.zeros(0), ())
    ]

    # True relative pose between node 0 and node 3 is: translation = [0, -5, 0] in node 3 frame
    # rotation = I * I = I
    # relative_pose(source=node 0, target=node 3): target.rotation.inv() * source.rotation = I
    # target.rotation.inv().apply(source.pos - target.pos) = [ -0.10, -5.05, 0 ]
    # If loop measurement closes to ground truth [0, -5, 0]:
    loops = [
        (0, 3, Rotation.identity(), np.array([0.0, -5.0, 0.0]), 'loop')
    ]

    num_params = (len(nodes) - 1) * 6
    point_jac = csr_matrix((0, num_params), dtype=np.float32)
    point_errors = np.zeros(0, dtype=np.float32)

    cfg = RootSLAMConfig(precision='float32', max_iterations=30, use_analytical_jacobians=False)
    params, edges, report = rootslam_solve(nodes, loops, point_jac, point_errors, config=cfg)

    assert report['success'] is True
    assert report['engine'] == 'RootSLAM-SquareRootBA'
    assert report['precision'] == 'float32'
    assert report['cost'] < report['initial_cost']
    assert report['cost_reduction_ratio'] > 0.05
    assert len(params) == num_params
    assert np.all(np.isfinite(params))
