import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares

class PlanarKabschSolver:
    """
    Solves rigid body rotation and translation between LiDAR and Camera using
    planar surface correspondences (normals and centroids) via Kabsch SVD
    and non-linear refinement with true physical mount constraints.
    """
    def __init__(self, physical_lever_arm_m=0.185, gravity_up_l=None):
        if gravity_up_l is None:
            self.u_L = np.array([0.003102, -0.504937, -0.863152])
            self.u_L /= np.linalg.norm(self.u_L)
        else:
            self.u_L = np.array(gravity_up_l) / np.linalg.norm(gravity_up_l)
        
        self.c_L = physical_lever_arm_m * self.u_L

    def solve_kabsch_normals(self, normals_lidar, normals_cam, weights=None):
        """
        Closed-form optimal rotation matrix via Kabsch (Wahba's) algorithm.
        Finds R that minimizes sum_i w_i || n_cam,i - R n_lidar,i ||^2.
        """
        N = len(normals_lidar)
        if N == 0:
            raise ValueError("No normals provided to Kabsch solver")

        if weights is None:
            weights = np.ones(N)

        H = np.zeros((3, 3))
        for i in range(N):
            n_l = normals_lidar[i] / np.linalg.norm(normals_lidar[i])
            n_c = normals_cam[i] / np.linalg.norm(normals_cam[i])
            w = weights[i]
            H += w * np.outer(n_c, n_l)

        # Singular Value Decomposition
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(U @ Vt)
        D = np.diag([1.0, 1.0, np.sign(d)])
        R_optimal = U @ D @ Vt
        return R_optimal

    def refine_extrinsics(self, R_init, correspondences, optimize_lever_arm=False):
        """
        Refines extrinsics using non-linear least squares.
        correspondences: list of dicts with keys:
          - 'normal_lidar': (3,)
          - 'centroid_lidar': (3,)
          - 'normal_cam': (3,)
          - 'center_ray_cam': (3,)
          - 'inlier_points_lidar': (M, 3) (optional)
        """
        def residual_fun(params):
            p, y, r = params[:3]
            R_trim = Rot.from_euler("xyz", [p, y, r], degrees=True).as_matrix()
            R = R_trim @ R_init

            resids = []
            for corr in correspondences:
                n_l = corr["normal_lidar"]
                n_c = corr["normal_cam"]
                c_l = corr["centroid_lidar"]
                r_c = corr["center_ray_cam"]

                # 1. Normal alignment residual: 1 - cos(theta)
                n_l_in_cam = R @ n_l
                normal_err = 1.0 - np.clip(np.dot(n_c, n_l_in_cam), -1.0, 1.0)
                resids.append(normal_err * 20.0) # weight normal alignment

                # 2. Centroid ray distance residual
                p_cam = R @ (c_l - self.c_L)
                dist_along_ray = np.dot(p_cam, r_c)
                ortho_vec = p_cam - dist_along_ray * r_c
                resids.extend(ortho_vec)

                # 3. Points-to-plane residual if available
                if "inlier_points_lidar" in corr:
                    pts_l = corr["inlier_points_lidar"]
                    pts_cam = (R @ (pts_l - self.c_L).T).T
                    # Distance of each point to the target plane in camera frame
                    pt_dists = np.dot(pts_cam, n_c)
                    resids.append(np.std(pt_dists) * 5.0)

            # 4. Enforce upright gravity alignment penalty
            up_in_cam = R @ self.u_L
            gravity_err = np.cross(up_in_cam, np.array([0.0, -1.0, 0.0]))
            resids.extend(gravity_err * 10.0)

            return np.array(resids)

        init_params = [0.0, 0.0, 0.0]
        opt = least_squares(residual_fun, init_params, method="lm")
        p_opt, y_opt, r_opt = opt.x
        R_trim = Rot.from_euler("xyz", [p_opt, y_opt, r_opt], degrees=True).as_matrix()
        R_refined = R_trim @ R_init

        res_final = residual_fun(opt.x)
        rmse = np.sqrt(np.mean(res_final**2))

        return {
            "R": R_refined,
            "t": -R_refined @ self.c_L,
            "c_L": self.c_L,
            "pitch_trim": float(p_opt),
            "yaw_trim": float(y_opt),
            "roll_trim": float(r_opt),
            "rmse": float(rmse),
            "cost": float(opt.cost)
        }
