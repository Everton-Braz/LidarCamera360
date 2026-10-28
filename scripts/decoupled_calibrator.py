#!/usr/bin/env python3
"""
Robust Decoupled LiDAR-Camera Calibrator
Zero-hardcode, drift-immune multi-tier estimator:
  Tier 1: Boresight SO(3) + temporal refinement dt_sync (scale & translation invariant)
  Tier 2: Lever-arm (translation) via high-rotational excitation segments (Fisher Information)
"""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional

def huber_loss(r: np.ndarray, delta: float = 0.05) -> np.ndarray:
    abs_r = np.abs(r)
    linear = abs_r > delta
    loss = np.where(linear, delta * (abs_r - 0.5 * delta), 0.5 * (r ** 2))
    return np.sum(loss)

def cauchy_loss(r: np.ndarray, c: float = 0.05) -> float:
    return float(np.sum(0.5 * (c ** 2) * np.log(1.0 + (r / c) ** 2)))

def calibrate_decoupled_extrinsics(
    paired_data: List[Tuple[np.ndarray, np.ndarray, float]],
    slerp_slam: Slerp,
    slam_times: np.ndarray,
    slam_positions: np.ndarray,
    nominal_R_lc: np.ndarray,
    nominal_t_lc: np.ndarray,
    dt_init: float = 0.0,
    max_window_seconds: float = 5.0,
    min_rot_deg: float = 1.2,
    t_offset_bound: float = 0.08
) -> Tuple[np.ndarray, np.ndarray, float, Dict]:
    """
    Calibrate (R_lc, t_lc, dt_sync) without scale or long-distance trajectory drift contamination.
    
    Parameters:
      paired_data: list of (T_w_c, _, t_rel)
      slerp_slam: Slerp interpolator over SLAM SO(3) orientations
      slam_times: array of SLAM timestamps
      slam_positions: array of SLAM 3D positions
      nominal_R_lc: 3x3 initial rotation estimate (from mechanical prior)
      nominal_t_lc: 3D vector initial translation estimate
      dt_init: initial time offset (from gyro cross-correlation)
    """
    t_min, t_max = slam_times[0], slam_times[-1]
    N = len(paired_data)
    if N < 8:
        raise ValueError(f"Insufficient paired frames ({N}) for decoupled calibration")

    # Extract camera poses and relative timestamps
    cam_times = np.array([p[2] for p in paired_data])
    T_w_c_all = [p[0] for p in paired_data]
    R_w_c_all = [T[:3, :3] for T in T_w_c_all]
    t_w_c_all = [T[:3, 3] for T in T_w_c_all]

    # Helper to interpolate SLAM pose at any time (t_rel - dt)
    def get_slam_pose(t_query: float):
        t_q = np.clip(t_query, t_min, t_max)
        rot = slerp_slam(t_q).as_matrix()
        idx = np.clip(np.searchsorted(slam_times, t_q), 1, len(slam_times) - 1)
        alpha = (t_q - slam_times[idx - 1]) / (slam_times[idx] - slam_times[idx - 1] + 1e-9)
        pos = (1 - alpha) * slam_positions[idx - 1] + alpha * slam_positions[idx]
        return rot, pos

    # Build pairs of frames with suitable motion (within short windows to avoid drift)
    pairs = []
    for i in range(N):
        for j in range(i + 1, min(i + 25, N)):
            dt_ij = cam_times[j] - cam_times[i]
            if 0.4 <= dt_ij <= max_window_seconds:
                # Relative rotation in camera frame
                R_ci_cj = R_w_c_all[i].T @ R_w_c_all[j]
                rot_angle = np.degrees(np.linalg.norm(R.from_matrix(R_ci_cj).as_rotvec()))
                if rot_angle >= min_rot_deg:
                    pairs.append((i, j, dt_ij, R_ci_cj, rot_angle))

    print(f"    Selected {len(pairs)} dynamic pair-frames with rotation >= {min_rot_deg:.1f} deg in local windows (<= {max_window_seconds}s)")

    if len(pairs) < 5:
        # Fallback to lower threshold if motion was very slow
        for i in range(N):
            for j in range(i + 1, min(i + 15, N)):
                dt_ij = cam_times[j] - cam_times[i]
                if 0.3 <= dt_ij <= max_window_seconds:
                    R_ci_cj = R_w_c_all[i].T @ R_w_c_all[j]
                    rot_angle = np.degrees(np.linalg.norm(R.from_matrix(R_ci_cj).as_rotvec()))
                    pairs.append((i, j, dt_ij, R_ci_cj, rot_angle))
        print(f"    Relaxed threshold: {len(pairs)} pairs collected")

    # =========================================================================
    # TIER 1: Boresight R_lc and Fine dt Optimization (Pure SO(3), Zero Scale/Translation Coupling)
    # Fundamental equation: R_li_lj(dt) * R_lc = R_lc * R_ci_cj
    # Residual: Log( R_lc^T * R_li_lj(dt)^T * R_lc * R_ci_cj )
    # =========================================================================
    init_rotvec = R.from_matrix(nominal_R_lc).as_rotvec()
    x0_tier1 = np.concatenate([init_rotvec, [0.0]])  # [rx, ry, rz, delta_dt]

    def cost_tier1(params):
        r_vec = params[:3]
        delta_dt = params[3]
        dt_eval = dt_init + delta_dt
        R_candidate = R.from_rotvec(r_vec).as_matrix()

        residuals = []
        for i, j, _, R_ci_cj, weight in pairs:
            t_i_slam = t_min + (cam_times[i] - dt_eval)
            t_j_slam = t_min + (cam_times[j] - dt_eval)
            if t_i_slam < t_min or t_j_slam > t_max:
                continue
            
            R_li, _ = get_slam_pose(t_i_slam)
            R_lj, _ = get_slam_pose(t_j_slam)
            R_li_lj = R_li.T @ R_lj

            # Hand-eye rotation mismatch
            R_err = R_candidate.T @ R_li_lj.T @ R_candidate @ R_ci_cj
            rot_err = R.from_matrix(R_err).as_rotvec()
            residuals.append(rot_err)

        if not residuals:
            return 1e6
        
        res_arr = np.concatenate(residuals)
        # Soft prior penalty to avoid unobservable null-space spin if motion is planar
        reg_rot = 0.05 * np.sum((r_vec - init_rotvec) ** 2)
        reg_dt = 10.0 * (delta_dt ** 2)
        return cauchy_loss(res_arr, c=0.03) + reg_rot + reg_dt

    bounds_tier1 = [
        (init_rotvec[0] - 0.2, init_rotvec[0] + 0.2),
        (init_rotvec[1] - 0.2, init_rotvec[1] + 0.2),
        (init_rotvec[2] - 0.2, init_rotvec[2] + 0.2),
        (-t_offset_bound, t_offset_bound)
    ]

    res_tier1 = minimize(cost_tier1, x0_tier1, bounds=bounds_tier1, method='L-BFGS-B')
    best_R_lc = R.from_rotvec(res_tier1.x[:3]).as_matrix()
    opt_delta_dt = float(res_tier1.x[3])
    refined_dt = dt_init + opt_delta_dt

    print(f"    [Tier 1] Boresight converged. Delta dt: {opt_delta_dt*1000:+.2f} ms (Total dt: {refined_dt:.6f}s)")

    # =========================================================================
    # TIER 2: Lever-arm (t_lc) Optimization via High-Excitation Segments
    # Linear relation per pair: (R_li_lj - I_3) * t_lc + s_ij * R_lc * t_ci_cj = t_li_lj
    # s_ij is eliminated or jointly estimated locally
    # =========================================================================
    # Filter pairs where ||R_li_lj - I|| is sufficiently large (rotational excitation)
    tier2_pairs = []
    for i, j, _, R_ci_cj, rot_deg in pairs:
        if rot_deg >= 2.0:  # significant angular excitation
            t_i_slam = t_min + (cam_times[i] - refined_dt)
            t_j_slam = t_min + (cam_times[j] - refined_dt)
            if t_min <= t_i_slam <= t_max and t_min <= t_j_slam <= t_max:
                R_li, p_li = get_slam_pose(t_i_slam)
                R_lj, p_lj = get_slam_pose(t_j_slam)
                R_li_lj = R_li.T @ R_lj
                t_li_lj = R_li.T @ (p_lj - p_li)
                t_ci_cj = t_w_c_all[j] - t_w_c_all[i]  # relative in world/SfM
                t_ci_cj_cam = R_w_c_all[i].T @ t_ci_cj
                tier2_pairs.append((R_li_lj, t_li_lj, t_ci_cj_cam))

    print(f"    [Tier 2] Pairs with high rotational observability: {len(tier2_pairs)}")

    x0_tier2 = nominal_t_lc.copy()

    def cost_tier2(t_cand):
        residuals = []
        for R_rel, t_rel_slam, t_rel_cam in tier2_pairs:
            # Predict relative translation
            lhs = (R_rel - np.eye(3)) @ t_cand
            # Estimate optimal local scale s_ij analytically
            cam_dir = best_R_lc @ t_rel_cam
            denom = np.dot(cam_dir, cam_dir) + 1e-9
            target = t_rel_slam - lhs
            s_ij = max(0.01, float(np.dot(target, cam_dir) / denom))
            diff = target - s_ij * cam_dir
            residuals.append(diff)

        if not residuals:
            return 1e6
        res_vec = np.concatenate(residuals)
        # Regularization: elastic prior towards CAD physical mounting
        # Prevents long corridors from inverting signs
        reg_arm = 15.0 * np.sum((t_cand - nominal_t_lc) ** 2)
        return huber_loss(res_vec, delta=0.03) + reg_arm

    bounds_tier2 = [
        (nominal_t_lc[0] - 0.06, nominal_t_lc[0] + 0.06),
        (nominal_t_lc[1] - 0.08, nominal_t_lc[1] + 0.08),
        (nominal_t_lc[2] - 0.08, nominal_t_lc[2] + 0.08)
    ]

    res_tier2 = minimize(cost_tier2, x0_tier2, bounds=bounds_tier2, method='L-BFGS-B')
    best_t_lc = res_tier2.x
    arm_len = float(np.linalg.norm(best_t_lc))

    print(f"    [Tier 2] Lever-arm converged: [{best_t_lc[0]:+.4f}, {best_t_lc[1]:+.4f}, {best_t_lc[2]:+.4f}] m (norm: {arm_len*100:.2f} cm)")

    diagnostics = {
        "pairs_count": len(pairs),
        "high_rot_pairs_count": len(tier2_pairs),
        "delta_dt_seconds": opt_delta_dt,
        "refined_dt_seconds": refined_dt,
        "arm_length_meters": arm_len,
        "tier1_fun": float(res_tier1.fun),
        "tier2_fun": float(res_tier2.fun)
    }

    return best_R_lc, best_t_lc, refined_dt, diagnostics
