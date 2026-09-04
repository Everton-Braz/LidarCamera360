import numpy as np

class LidarPlanarExtractor:
    """
    Extracts planar target boards from 3D LiDAR point clouds using reflectivity filtering
    and RANSAC plane fitting, as proposed in ARVCUMH/fisheye_lidar_calibration.
    """
    def __init__(self, refl_threshold=100.0, distance_threshold=0.03, max_iterations=1000):
        self.refl_threshold = refl_threshold
        self.dist_thresh = distance_threshold
        self.max_iter = max_iterations

    def extract_candidate_clusters(self, points, min_cluster_size=15, max_cluster_radius=0.45):
        """
        Filters high-reflectivity points and groups them into spatial clusters.
        points: (N, 4+) [X, Y, Z, Intensity, ...]
        """
        I = points[:, 3]
        # Filter high reflectivity (retroreflective target board / tape / AprilTag border)
        mask = I >= self.refl_threshold
        high_refl_pts = points[mask, :3]
        high_refl_vals = I[mask]

        if len(high_refl_pts) < min_cluster_size:
            # Fallback to top 5% intensity if threshold is too strict
            p95 = np.percentile(I, 95.0)
            mask = I >= p95
            high_refl_pts = points[mask, :3]
            high_refl_vals = I[mask]

        # Simple Euclidean clustering
        clusters = []
        visited = np.zeros(len(high_refl_pts), dtype=bool)

        for i in range(len(high_refl_pts)):
            if visited[i]:
                continue
            pt = high_refl_pts[i]
            dists = np.linalg.norm(high_refl_pts - pt, axis=1)
            in_cluster = dists < max_cluster_radius
            visited[in_cluster] = True

            cluster_pts = high_refl_pts[in_cluster]
            cluster_refl = high_refl_vals[in_cluster]

            if len(cluster_pts) >= min_cluster_size:
                clusters.append({
                    "points": cluster_pts,
                    "intensities": cluster_refl,
                    "centroid": np.mean(cluster_pts, axis=0),
                    "count": len(cluster_pts)
                })

        return clusters

    def fit_plane_ransac(self, pts):
        """
        RANSAC 3D plane fitting: returns (normal, d, inliers, centroid).
        Plane equation: normal . p + d = 0, with ||normal|| = 1.
        """
        N = len(pts)
        if N < 3:
            return None, None, None, None

        best_inliers = []
        best_plane = None

        rng = np.random.default_rng(42)
        for _ in range(self.max_iter):
            sample_idx = rng.choice(N, 3, replace=False)
            p1, p2, p3 = pts[sample_idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal /= norm

            d = -np.dot(normal, p1)
            dists = np.abs(pts @ normal + d)
            inliers = np.where(dists < self.dist_thresh)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)

        if len(best_inliers) < 8:
            return None, None, None, None

        # SVD refinement on all inliers
        inlier_pts = pts[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1] # normal is the eigenvector corresponding to smallest singular value
        normal /= np.linalg.norm(normal)

        # Orient normal towards LiDAR origin [0,0,0]
        if np.dot(normal, -centroid) < 0:
            normal = -normal

        d = -np.dot(normal, centroid)
        return normal, d, inlier_pts, centroid
