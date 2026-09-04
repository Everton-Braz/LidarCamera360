import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import minimize
import cv2

class DirectNIDCalibrator:
    """
    Direct Target-less LiDAR-Camera Extrinsic Calibration via Normalized Information Distance (NID),
    inspired by koide3/direct_visual_lidar_calibration.
    
    Computes statistical mutual information between LiDAR geometric attributes (surface normal dot ray
    or reflectivity) and camera radiometric attributes (image grayscale intensity or gradient).
    """
    def __init__(self, projector, physical_lever_arm_m=0.185, gravity_up_l=None, n_bins=32):
        self.projector = projector
        self.n_bins = n_bins
        if gravity_up_l is None:
            self.u_L = np.array([0.003102, -0.504937, -0.863152])
            self.u_L /= np.linalg.norm(self.u_L)
        else:
            self.u_L = np.array(gravity_up_l) / np.linalg.norm(gravity_up_l)
        self.c_L = physical_lever_arm_m * self.u_L

    def compute_nid(self, x_samples, y_samples):
        """
        Computes Normalized Information Distance (NID) between two random variables.
        NID = 1 - I(X; Y) / H(X, Y)
        Returns a value in [0, 1] where 0 means maximal alignment.
        """
        if len(x_samples) < 500:
            return 1.0

        # Discretize into 2D histogram bins
        hist2d, _, _ = np.histogram2d(x_samples, y_samples, bins=self.n_bins,
                                      range=[[0, 1], [0, 1]])

        # Joint and marginal probability distributions
        p_xy = hist2d / np.sum(hist2d)
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        # Filter out zero probabilities for numerical stability
        eps = 1e-12
        p_xy_nz = p_xy[p_xy > eps]
        p_x_nz = p_x[p_x > eps]
        p_y_nz = p_y[p_y > eps]

        # Shannon Entropies
        H_xy = -np.sum(p_xy_nz * np.log2(p_xy_nz))
        H_x = -np.sum(p_x_nz * np.log2(p_x_nz))
        H_y = -np.sum(p_y_nz * np.log2(p_y_nz))

        # Mutual Information
        MI = max(0.0, H_x + H_y - H_xy)

        # Normalized Information Distance
        if H_xy < eps:
            return 1.0

        nid = 1.0 - (MI / H_xy)
        return float(np.clip(nid, 0.0, 1.0))

    def evaluate_cost(self, R, pts, refl, img_gray, img_grad):
        """
        Projects LiDAR points into camera image and calculates NID cost.
        """
        # Transform points relative to physical camera center
        pts_c = (R @ (pts - self.c_L).T).T
        u, v, valid = self.projector.project(pts_c)

        if not np.any(valid) or np.sum(valid) < 500:
            return 1.0

        u_val = np.round(u[valid]).astype(np.int32)
        v_val = np.round(v[valid]).astype(np.int32)

        # Attribute 1 (Camera): Normalized grayscale intensity & gradient
        img_h, img_w = img_gray.shape[:2]
        in_bounds = (u_val >= 0) & (u_val < img_w) & (v_val >= 0) & (v_val < img_h)

        if np.sum(in_bounds) < 500:
            return 1.0

        u_val = u_val[in_bounds]
        v_val = v_val[in_bounds]

        cam_intensity = img_gray[v_val, u_val].astype(np.float32) / 255.0
        cam_grad = img_grad[v_val, u_val].astype(np.float32) / 255.0

        # Attribute 2 (LiDAR): Normalized reflectivity
        refl_val = refl[valid][in_bounds].astype(np.float32)
        p99 = np.percentile(refl_val, 99.0) if len(refl_val) > 0 else 255.0
        refl_norm = np.clip(refl_val / max(p99, 1.0), 0.0, 1.0)

        # Attribute 3 (LiDAR Geometry): Surface viewing angle cos(theta) = Z / r
        pts_valid = pts_c[valid][in_bounds]
        dist = np.linalg.norm(pts_valid, axis=1)
        view_angle = np.clip(pts_valid[:, 2] / np.maximum(dist, 1e-6), 0.0, 1.0)

        # Combine NID on intensity and geometric edges
        nid_refl = self.compute_nid(refl_norm, cam_intensity)
        nid_geom = self.compute_nid(view_angle, cam_grad)

        total_nid = 0.6 * nid_refl + 0.4 * nid_geom
        return total_nid

    def calibrate(self, R_init, pts, refl, img_bgr, max_evals=150):
        """
        Optimizes [pitch, yaw, roll] trims using Nelder-Mead to minimize NID.
        """
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_grad = np.hypot(cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3),
                            cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3))
        p99_g = np.percentile(img_grad, 99.0)
        img_grad = np.clip(img_grad / max(p99_g, 1.0) * 255.0, 0, 255).astype(np.uint8)

        # Downsample points for speed in iterative optimization (15,000 points is plenty for histogram)
        if len(pts) > 20000:
            sub_idx = np.random.choice(len(pts), 20000, replace=False)
            pts_sub = pts[sub_idx]
            refl_sub = refl[sub_idx]
        else:
            pts_sub = pts
            refl_sub = refl

        def cost_fun(params):
            p, y, r = params
            R_trim = Rot.from_euler("xyz", [p, y, r], degrees=True).as_matrix()
            R = R_trim @ R_init

            nid = self.evaluate_cost(R, pts_sub, refl_sub, img_gray, img_grad)
            # Add mild upright gravity penalty
            up_in_cam = R @ self.u_L
            gravity_err = np.linalg.norm(np.cross(up_in_cam, np.array([0.0, -1.0, 0.0])))
            return nid + 0.05 * gravity_err

        print(f"[*] Custo NID inicial (sem ajustes): {cost_fun([0.0, 0.0, 0.0]):.5f}")
        # Use Powell method which works exceptionally well for multi-modal mutual information surfaces
        res = minimize(cost_fun, [0.0, 0.0, 0.0], method="Powell",
                       options={"maxiter": max_evals, "disp": False, "xtol": 0.05, "ftol": 1e-5})

        p_opt, y_opt, r_opt = res.x
        R_trim = Rot.from_euler("xyz", [p_opt, y_opt, r_opt], degrees=True).as_matrix()
        R_opt = R_trim @ R_init

        return {
            "R": R_opt,
            "t": -R_opt @ self.c_L,
            "c_L": self.c_L,
            "pitch_trim": float(p_opt),
            "yaw_trim": float(y_opt),
            "roll_trim": float(r_opt),
            "initial_nid": float(cost_fun([0.0, 0.0, 0.0])),
            "final_nid": float(res.fun),
            "iterations": int(res.nit)
        }
