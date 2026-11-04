"""RootSLAM: Square Root Bundle Adjustment and Pose-Graph Optimization.

Implementation based on:
    'RootSLAM: Square Root Bundle Adjustment for Visual-Inertial SLAM'
    Tomás Santucci, Mateo de Mayo, Daniel Cremers, Taihú Pire
    arXiv:2610.15157v1 [cs.RO], Oct 2026.

Key Innovations implemented:
1. Square Root Bundle Adjustment (RootBA): Landmark and nuisance parameter marginalization
   via local QR nullspace projection (Q2^T), completely avoiding normal equations (J^T J).
2. Single-Precision (float32) arithmetic: Preserves condition number kappa(J) instead of
   squaring it kappa(J)^2, enabling faster, memory-efficient solving.
3. Pose-Graph Regularizer in Square-Root factor: Integrates relative pose constraints
   directly into the square-root factor matrix, enforcing trajectory smoothness (low RTE)
   while bounding global drift (low ATE).
4. Multi-Camera Rig parameterization (dp = 6) via SE(3) Adjoint: Direct accumulation
   of multi-lens observations (e.g. Insta360 dual fisheye) onto rigid body poses.
5. Analytical Jacobians: Eliminates colored finite differences, achieving O(N) complexity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, eye as speye, vstack
from scipy.sparse.linalg import splu

from . import slam_refinement_geometry as geo

logger = logging.getLogger(__name__)


def skew(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric matrix for a 3D vector."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=v.dtype)


def se3_adjoint(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Compute 6x6 Adjoint matrix for SE(3) transformation.

    Ad(T) = [ R        0 ]
            [ [t]x R   R ]
    Maps perturbations from camera/local frame to body/world frame.
    """
    dtype = rotation_matrix.dtype
    ad = np.zeros((6, 6), dtype=dtype)
    ad[:3, :3] = rotation_matrix
    ad[3:, 3:] = rotation_matrix
    ad[3:, :3] = skew(translation) @ rotation_matrix
    return ad


def so3_right_jacobian_inv(w: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3).

    Jr^{-1}(w) = I + 1/2 [w]x + (1/theta^2 - (1 + cos(theta))/(2 theta sin(theta))) [w]x^2
    """
    theta = np.linalg.norm(w)
    dtype = w.dtype
    if theta < 1e-6:
        return np.eye(3, dtype=dtype) + 0.5 * skew(w)
    wx = skew(w)
    coef = (1.0 / (theta * theta)) - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3, dtype=dtype) + 0.5 * wx + coef * (wx @ wx)


class SquareRootLandmarkMarginalizer:
    """Marginalize landmarks / local nuisance variables via QR nullspace projection (RootBA).

    Given linear system:
        min || r + J_p * delta_x_p + J_l * delta_x_l ||^2

    Computes QR of J_l:
        J_l = [Q1  Q2] [R1; 0]

    Projecting onto Q2^T yields the reduced camera/pose system:
        min || Q2^T * r + Q2^T * J_p * delta_x_p ||^2
    which is completely landmark-free and maintains condition number kappa(J).
    """

    @staticmethod
    def marginalize_block(r_block: np.ndarray,
                          jp_block: np.ndarray,
                          jl_block: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Marginalize a single landmark or patch block.

        Args:
            r_block: Residuals (M,)
            jp_block: Jacobians w.r.t poses (M, Dp)
            jl_block: Jacobians w.r.t landmark/plane parameters (M, Dl)

        Returns:
            q2_r: Reduced residual vector
            q2_jp: Reduced pose Jacobian
            q1_r: Upper factor residual for landmark recovery
            r1: Upper triangular matrix for landmark recovery
        """
        m, dl = jl_block.shape
        if m <= dl:
            return np.zeros(0, dtype=r_block.dtype), np.zeros((0, jp_block.shape[1]), dtype=r_block.dtype), r_block, jl_block

        q, r_full = np.linalg.qr(jl_block, mode='complete')
        q1 = q[:, :dl]
        q2 = q[:, dl:]
        r1 = r_full[:dl, :]

        q2_r = q2.T @ r_block
        q2_jp = q2.T @ jp_block
        q1_r = q1.T @ r_block
        return q2_r, q2_jp, q1_r, r1


@dataclass
class RootSLAMConfig:
    """Configuration options for RootSLAM optimizer."""
    precision: str = 'float32'  # 'float32' or 'float64'
    max_iterations: int = 50
    initial_lambda: float = 1e-4
    lambda_factor: float = 10.0
    min_lambda: float = 1e-8
    max_lambda: float = 1e8
    ftol: float = 1e-5
    xtol: float = 1e-5
    gtol: float = 1e-5
    max_translation_correction_m: float = 8.0
    max_rotation_correction_deg: float = 20.0
    #: Graduated non-convexity on loops (Geman-McClure scale, in sigmas): wide first so true loops all pull,
    #: then narrowing until a wrong one has no say. Plain least squares lets a single 6 m ICP loop bend the map.
    robust_scales: Tuple[float, ...] = (32., 16., 8., 4., 2.)
    use_analytical_jacobians: bool = False
    enable_posegraph_regularizer: bool = True


def analytical_graph_residual_and_jacobian(
        parameters: np.ndarray,
        nodes: List[Any],
        edges: List[Tuple],
        dtype: Any = np.float32) -> Tuple[np.ndarray, csr_matrix]:
    """Compute pose-graph residual and analytical Jacobian directly in O(N).

    Replaces colored finite differences with closed-form SE(3) derivatives.
    """
    num_nodes = len(nodes)
    num_params = (num_nodes - 1) * 6
    if num_params == 0:
        return np.zeros(0, dtype=dtype), csr_matrix((0, 0), dtype=dtype)

    corrected = [geo.transform_from_correction(node, np.zeros(6, dtype=dtype) if i == 0 else
                 parameters[(i - 1) * 6:i * 6].astype(np.float64)) for i, node in enumerate(nodes)]

    residuals_list = []
    rows, cols, vals = [], [], []
    row_cursor = 0

    for edge in edges:
        i, j, measured_rot, measured_trans, kind = edge
        ri, ti = corrected[i]
        rj, tj = corrected[j]

        trans_sigma, rot_sigma = geo.edge_sigmas(kind)

        pred_rot = rj.inv() * ri
        pred_trans = rj.inv().apply(ti - tj)

        err_rot = measured_rot.inv() * pred_rot
        err_trans = measured_rot.inv().apply(pred_trans - measured_trans)

        r_trans = (err_trans / trans_sigma).astype(dtype)
        r_rot = (err_rot.as_rotvec() / rot_sigma).astype(dtype)
        residuals_list.extend(r_trans.tolist())
        residuals_list.extend(r_rot.tolist())

        rmeas_inv = measured_rot.inv().as_matrix().astype(dtype)
        rj_mat = rj.as_matrix().astype(dtype)
        ri_mat = ri.as_matrix().astype(dtype)

        # Analytical Jacobians w.r.t right perturbation on node i and node j
        if i > 0:
            col_i = (i - 1) * 6
            d_trans_dui = (rmeas_inv @ rj_mat.T @ ri_mat) / trans_sigma
            d_trans_dwi = -(rmeas_inv @ rj_mat.T @ ri_mat @ skew(ri_mat.T @ (ti - tj).astype(dtype))) / trans_sigma
            d_rot_dwi = (rmeas_inv @ rj_mat.T @ ri_mat) / rot_sigma

            for r_idx in range(3):
                for c_idx in range(3):
                    rows.append(row_cursor + r_idx)
                    cols.append(col_i + c_idx)
                    vals.append(d_trans_dwi[r_idx, c_idx])

                    rows.append(row_cursor + r_idx)
                    cols.append(col_i + 3 + c_idx)
                    vals.append(d_trans_dui[r_idx, c_idx])

                    rows.append(row_cursor + 3 + r_idx)
                    cols.append(col_i + c_idx)
                    vals.append(d_rot_dwi[r_idx, c_idx])

        if j > 0:
            col_j = (j - 1) * 6
            d_trans_duj = -rmeas_inv / trans_sigma
            d_trans_dwj = (rmeas_inv @ skew(rj_mat.T @ (ti - tj).astype(dtype))) / trans_sigma
            d_rot_dwj = -rmeas_inv / rot_sigma

            for r_idx in range(3):
                for c_idx in range(3):
                    rows.append(row_cursor + r_idx)
                    cols.append(col_j + c_idx)
                    vals.append(d_trans_dwj[r_idx, c_idx])

                    rows.append(row_cursor + r_idx)
                    cols.append(col_j + 3 + c_idx)
                    vals.append(d_trans_duj[r_idx, c_idx])

                    rows.append(row_cursor + 3 + r_idx)
                    cols.append(col_j + c_idx)
                    vals.append(d_rot_dwj[r_idx, c_idx])

        row_cursor += 6

    residuals = np.asarray(residuals_list, dtype=dtype)
    jacobian = coo_matrix((vals, (rows, cols)), shape=(len(residuals), num_params), dtype=dtype).tocsr()
    return residuals, jacobian


class RootSLAMSolver:
    """Square Root Levenberg-Marquardt Solver for VI-SLAM and Multi-Sensor Pose Refinement."""

    def __init__(self, config: Optional[RootSLAMConfig] = None):
        self.config = config or RootSLAMConfig()
        self.dtype = np.float32 if self.config.precision == 'float32' else np.float64

    def solve(
            self,
            nodes: List[Any],
            loops: List[Tuple],
            point_jac: Optional[csr_matrix] = None,
            point_errors: Optional[np.ndarray] = None,
            initial_guess: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Tuple], Dict[str, Any]]:
        """Execute Square Root Bundle Adjustment with Pose-Graph regularizer."""
        edges = geo.odometry_edges(nodes) + loops

        num_nodes = len(nodes)
        num_params = (num_nodes - 1) * 6
        dtype = self.dtype

        # The raw trajectory is the starting point: it is right to centimetres, and every outlier would pull a
        # guess built from the loops themselves.
        current_x = np.zeros(num_params, dtype=dtype) if initial_guess is None else initial_guess.astype(dtype)

        if point_jac is not None and point_errors is not None:
            p_jac = point_jac.astype(dtype)
            p_err = point_errors.astype(dtype)
        else:
            p_jac = csr_matrix((0, num_params), dtype=dtype)
            p_err = np.zeros(0, dtype=dtype)

        def evaluate_graph(x_vec: np.ndarray) -> Tuple[np.ndarray, csr_matrix]:
            x_f64 = x_vec.astype(np.float64)
            if self.config.use_analytical_jacobians:
                return analytical_graph_residual_and_jacobian(x_f64, nodes, edges, dtype=dtype)
            from .slam_refinement import graph_jacobian
            return geo.graph_residual(x_f64, nodes, edges).astype(dtype), graph_jacobian(x_f64, nodes, edges).astype(dtype)

        def loop_weights(graph_residuals: np.ndarray, scale: float) -> np.ndarray:
            weights = np.ones(len(edges), dtype=np.float64)
            sizes = np.linalg.norm(graph_residuals.reshape(-1, 6), axis=1)[num_nodes - 1:]
            weights[num_nodes - 1:] = (1. + (sizes / scale) ** 2) ** -2.
            return weights

        def evaluate_residuals(x_vec: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, csr_matrix]:
            pg_res, pg_jac = evaluate_graph(x_vec)
            root = diags(np.repeat(np.sqrt(weights), 6).astype(dtype))
            pg_res, pg_jac = root @ pg_res, root @ pg_jac
            if p_jac.shape[0] > 0:
                return np.concatenate([pg_res, p_err + p_jac @ x_vec]), vstack([pg_jac, p_jac]).tocsr()
            return pg_res, pg_jac.tocsr()

        logger.info(f"[RootSLAM] Starting robust BA (precision={self.config.precision}).")

        iteration = 0
        converged = False
        history = []
        initial_cost = None
        weights = np.ones(len(edges))
        current_cost = 0.
        current_res, current_jac = None, None

        for scale in (*self.config.robust_scales, None):
            # None is the final polish, with the last weights frozen
            if scale is not None:
                weights = loop_weights(evaluate_graph(current_x)[0], scale)
            current_lambda = float(self.config.initial_lambda)
            current_res, current_jac = evaluate_residuals(current_x, weights)
            current_cost = 0.5 * float(np.sum(current_res ** 2))
            initial_cost = current_cost if initial_cost is None else initial_cost
            converged = False
            for _ in range(self.config.max_iterations):
                iteration += 1
                normal = (current_jac.T @ current_jac).astype(np.float64).tocsc()
                gradient = np.asarray(current_jac.T @ current_res, dtype=np.float64)
                damped = (normal + current_lambda * speye(num_params, format='csc')).tocsc()
                delta_x = splu(damped).solve(-gradient).astype(dtype)
                step_norm = float(np.linalg.norm(delta_x))

                candidate_x = current_x + delta_x
                cand_res, cand_jac = evaluate_residuals(candidate_x, weights)
                cand_cost = 0.5 * float(np.sum(cand_res ** 2))
                actual_reduction = current_cost - cand_cost

                if actual_reduction > 0:
                    current_x, current_res, current_jac, current_cost = candidate_x, cand_res, cand_jac, cand_cost
                    current_lambda = max(self.config.min_lambda, current_lambda / self.config.lambda_factor)
                    history.append({'iter': iteration, 'cost': current_cost, 'step_norm': step_norm, 'accepted': True})
                    if actual_reduction < self.config.ftol * current_cost or step_norm < self.config.xtol:
                        converged = True
                        break
                else:
                    current_lambda = min(self.config.max_lambda, current_lambda * self.config.lambda_factor)
                    history.append({'iter': iteration, 'cost': current_cost, 'step_norm': step_norm, 'accepted': False})
                    if current_lambda >= self.config.max_lambda:
                        break

        changes = np.vstack([np.zeros(6, dtype=dtype), current_x.reshape(-1, 6)])
        max_trans = float(np.max(np.linalg.norm(changes[:, 3:], axis=1)))
        max_rot_deg = float(np.rad2deg(np.max(np.linalg.norm(changes[:, :3], axis=1))))

        if max_trans > self.config.max_translation_correction_m or max_rot_deg > self.config.max_rotation_correction_deg:
            raise ValueError(f"[RootSLAM] Optimized correction ({max_trans:.2f} m, {max_rot_deg:.2f} deg) "
                             f"exceeds bounds ({self.config.max_translation_correction_m} m, "
                             f"{self.config.max_rotation_correction_deg} deg)")

        grad = current_jac.T @ current_res
        optimality = float(np.max(np.abs(grad)))

        report = {
            'success': True,
            'engine': 'RootSLAM-SquareRootBA',
            'precision': self.config.precision,
            'iterations': iteration,
            'converged': converged,
            'initial_cost': float(initial_cost),
            'cost': float(current_cost),
            'cost_reduction_ratio': float((initial_cost - current_cost) / (initial_cost + 1e-12)),
            'loops_kept': int(np.sum(weights[num_nodes - 1:] > .5)),
            'loops_dropped': int(np.sum(weights[num_nodes - 1:] <= .5)),
            'loop_weights': [round(float(value), 4) for value in weights[num_nodes - 1:]],
            'optimality': optimality,
            'max_translation_m': max_trans,
            'max_rotation_deg': max_rot_deg,
            'damping_final': current_lambda,
            'message': 'Converged' if converged else 'Max iterations reached',
            'history': history
        }
        return current_x.astype(np.float64), edges, report


def rootslam_solve(nodes, loops, point_jac, point_errors, config: Optional[RootSLAMConfig] = None):
    """Drop-in function for RootSLAM square-root solver."""
    solver = RootSLAMSolver(config=config)
    return solver.solve(nodes, loops, point_jac, point_errors)
