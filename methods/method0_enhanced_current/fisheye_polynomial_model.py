import numpy as np

class PolynomialFisheyeCamera:
    """
    Advanced Fisheye Camera Model implementing the Kannala-Brandt (KB4) polynomial
    projection and unprojection, as documented in Section 3.1 of the scientific research.
    
    Equations:
      theta = arctan(r_xy / Z)
      theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
      r_img = f * theta_d
      u = cx + r_img * (X / r_xy)
      v = cy + r_img * (Y / r_xy)
    """
    def __init__(self, width=3840, height=3840, f=1123.5, cx=1920.0, cy=1920.0,
                 k1=-0.04268, k2=0.00242, k3=-0.00087, k4=0.00012):
        self.W = width
        self.H = height
        self.f = f
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k4 = k4
        self.max_radius = (min(width, height) / 2.0) * 0.985

    @classmethod
    def from_fov(cls, width=3840, height=3840, fov_deg=196.0, k1=-0.04268, k2=0.00242):
        """
        Initializes from nominal FOV and distortion parameters.
        """
        fov_rad = np.radians(fov_deg)
        f_nom = (width / 2.0) / (fov_rad / 2.0)
        return cls(width=width, height=height, f=f_nom, cx=width / 2.0, cy=height / 2.0,
                   k1=k1, k2=k2, k3=0.0, k4=0.0)

    def project(self, P_cam):
        """
        Projects 3D points (N, 3) in camera frame to 2D pixels (u, v).
        """
        X = P_cam[:, 0]
        Y = P_cam[:, 1]
        Z = P_cam[:, 2]
        dist = np.linalg.norm(P_cam, axis=1)
        r_xy = np.hypot(X, Y)

        theta = np.arctan2(r_xy, Z)
        th2 = theta**2
        th4 = th2**2
        th6 = th2 * th4
        th8 = th4**2

        # Kannala-Brandt distortion expansion
        theta_d = theta * (1.0 + self.k1 * th2 + self.k2 * th4 + self.k3 * th6 + self.k4 * th8)
        r_img = self.f * theta_d

        valid = (Z > 0.05) & (dist > 0.20) & (dist < 30.0) & (r_img <= self.max_radius)

        cos_phi = np.where(r_xy > 1e-7, X / np.maximum(r_xy, 1e-7), 1.0)
        sin_phi = np.where(r_xy > 1e-7, Y / np.maximum(r_xy, 1e-7), 0.0)

        u = self.cx + r_img * cos_phi
        v = self.cy + r_img * sin_phi
        circle_valid = valid & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)

        return u, v, circle_valid

    def pixel_to_ray(self, u, v):
        """
        Unprojects a 2D pixel (u, v) into a 3D unit ray using Newton-Raphson inversion.
        """
        dx = u - self.cx
        dy = v - self.cy
        r_img = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)

        if r_img < 1e-6:
            return np.array([0.0, 0.0, 1.0])

        theta_d = r_img / self.f

        # Invert Kannala-Brandt theta_d = theta * (1 + k1*theta^2 + k2*theta^4) via Newton-Raphson
        theta = theta_d # Initial guess
        for _ in range(8):
            th2 = theta**2
            th4 = th2**2
            f_val = theta * (1.0 + self.k1 * th2 + self.k2 * th4) - theta_d
            f_prime = 1.0 + 3.0 * self.k1 * th2 + 5.0 * self.k2 * th4
            if abs(f_prime) < 1e-9:
                break
            delta = f_val / f_prime
            theta -= delta
            if abs(delta) < 1e-8:
                break

        sin_th = np.sin(theta)
        cos_th = np.cos(theta)
        ray = np.array([sin_th * np.cos(phi), sin_th * np.sin(phi), cos_th])
        return ray / np.linalg.norm(ray)
