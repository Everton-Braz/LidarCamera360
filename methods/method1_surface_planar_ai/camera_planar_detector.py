import numpy as np
import cv2

class CameraPlanarDetector:
    """
    Detects planar target boards in raw dual-fisheye camera images and computes
    their 3D surface normals and directional rays in the camera coordinate frame.
    """
    def __init__(self, width=3840, height=3840, fov_deg=196.0):
        self.W = width
        self.H = height
        self.cx = width / 2.0
        self.cy = height / 2.0
        fov_rad = np.radians(fov_deg)
        self.f = (width / 2.0) / (fov_rad / 2.0)

    def pixel_to_ray(self, u, v):
        """
        Projects a 2D pixel (u, v) into a 3D unit ray in camera frame.
        Convention: +X = Right, +Y = Down, +Z = Forward.
        """
        dx = u - self.cx
        dy = v - self.cy
        r_pix = np.hypot(dx, dy)
        theta = r_pix / self.f
        phi = np.arctan2(dy, dx)
        ray = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        return ray / np.linalg.norm(ray)

    def detect_planar_targets(self, img_bgr, tag_size_m=0.150):
        """
        Detects planar targets using AprilTag fiducial boundaries and computes
        their 3D normal vector and directional rays.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)

        targets = []
        if ids is None or len(ids) == 0:
            return targets

        for i, tid in enumerate(ids.flatten()):
            pts_2d = corners[i][0] # 4 corners: c0 (top-left), c1 (top-right), c2 (bottom-right), c3 (bottom-left)
            center_2d = np.mean(pts_2d, axis=0)

            # Convert 4 corners to 3D unit rays
            rays = np.array([self.pixel_to_ray(pt[0], pt[1]) for pt in pts_2d])
            center_ray = self.pixel_to_ray(center_2d[0], center_2d[1])

            # Board edge vectors in 3D direction
            diag1 = rays[2] - rays[0]
            diag2 = rays[3] - rays[1]
            normal = np.cross(diag1, diag2)
            norm_val = np.linalg.norm(normal)
            if norm_val > 1e-6:
                normal /= norm_val

            # Orient normal facing the camera (Z_cam dot normal < 0)
            if np.dot(normal, center_ray) > 0:
                normal = -normal

            targets.append({
                "tag_id": int(tid),
                "corners_2d": pts_2d,
                "center_2d": center_2d,
                "corner_rays": rays,
                "center_ray": center_ray,
                "plane_normal": normal
            })

        return targets
