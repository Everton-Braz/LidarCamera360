"""Camera models shared by calibration: projection, validity and usable image area.

Parameters use the same 12-value packing as colorization (``fx fy cx cy k1 k2 p1 p2
k3 k4 sx1 sy1``; PINHOLE / OPENCV mark ``sy1`` with -999 / -998, OPENCV_FISHEYE puts
its k3/k4 in the thin-prism k3/k4 slots), so a calibration solved here projects
exactly like the colorizer that uses it.  EQUIRECTANGULAR needs only width/height.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

FISHEYE_MODELS = ('THIN_PRISM_FISHEYE', 'OPENCV_FISHEYE')
PERSPECTIVE_MODELS = ('PINHOLE', 'OPENCV')
SUPPORTED_MODELS = FISHEYE_MODELS + PERSPECTIVE_MODELS + ('EQUIRECTANGULAR',)
INTRINSIC_KEYS = ('fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1')
PINHOLE_MARKER, OPENCV_MARKER = -999.0, -998.0


@dataclass
class CameraModel:
    model: str
    width: int
    height: int
    params: Tuple[float, ...] = field(default_factory=tuple)
    theta_max_deg: float = 96.0           # fisheye: usable field of view half-angle
    valid_radius_px: Optional[float] = None  # fisheye: usable image circle (full resolution)

    def __post_init__(self):
        self.model = str(self.model).upper()
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f'Unsupported camera model: {self.model}')
        if self.model != 'EQUIRECTANGULAR' and len(self.params) != 12:
            raise ValueError(f'{self.model} needs 12 packed parameters')

    @classmethod
    def from_intrinsics(cls, model, intrinsics, width=None, height=None):
        """Build from an intrinsics dict (calibration JSON / auxiliary camera config)."""
        model = str(model or intrinsics.get('camera_model') or 'THIN_PRISM_FISHEYE').upper()
        width = int(width or intrinsics.get('width') or 0)
        height = int(height or intrinsics.get('height') or 0)
        if model == 'EQUIRECTANGULAR':
            return cls(model, width, height)
        get = lambda key: float(intrinsics.get(key, 0.0))
        base = [get('fx'), get('fy'), get('cx'), get('cy')]
        if model == 'PINHOLE':
            params = base + [0.0] * 7 + [PINHOLE_MARKER]
        elif model == 'OPENCV':
            params = base + [get('k1'), get('k2'), get('p1'), get('p2'), 0.0, 0.0, 0.0, OPENCV_MARKER]
        elif model == 'OPENCV_FISHEYE':
            params = base + [get('k1'), get('k2'), 0.0, 0.0, get('k3'), get('k4'), 0.0, 0.0]
        else:
            params = [get(key) for key in INTRINSIC_KEYS]
        if min(params[:2]) <= 0:
            raise ValueError(f'Invalid focal length for {model}')
        return cls(model, width, height, tuple(params))

    @property
    def is_fisheye(self):
        return self.model in FISHEYE_MODELS

    def project(self, P):
        """Camera-frame points -> (u, v, valid, depth) in full-resolution pixels."""
        P = np.asarray(P, dtype=np.float64)
        depth = np.linalg.norm(P, axis=1)
        if self.model == 'EQUIRECTANGULAR':
            lon = np.arctan2(P[:, 0], P[:, 2])
            lat = np.arcsin(np.clip(P[:, 1] / np.maximum(depth, 1e-12), -1.0, 1.0))
            u = (lon / (2.0 * np.pi) + 0.5) * self.width
            v = (lat / np.pi + 0.5) * self.height
            return u, v, depth > 1e-9, depth
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = self.params
        if self.model in PERSPECTIVE_MODELS:
            z = P[:, 2]
            valid = z > 0.1
            zs = np.where(valid, z, 1.0)
            x, y = P[:, 0] / zs, P[:, 1] / zs
            r2 = x * x + y * y
            radial = 1 + r2 * (k1 + r2 * (k2 + r2 * (k3 + r2 * k4)))
            xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            return fx * xd + cx, fy * yd + cy, valid, depth
        r = np.hypot(P[:, 0], P[:, 1])
        theta = np.arctan2(r, P[:, 2])
        scale = np.divide(theta, r, out=np.zeros_like(theta), where=r > 1e-12)
        xd, yd = P[:, 0] * scale, P[:, 1] * scale
        rd2 = theta ** 2
        radial = 1 + rd2 * (k1 + rd2 * (k2 + rd2 * (k3 + rd2 * k4)))
        du = 2 * p1 * xd * yd + p2 * (rd2 + 2 * xd ** 2) + sx1 * rd2
        dv = p1 * (rd2 + 2 * yd ** 2) + 2 * p2 * xd * yd + sy1 * rd2
        valid = theta < np.deg2rad(self.theta_max_deg)
        return fx * (xd * radial + du) + cx, fy * (yd * radial + dv) + cy, valid, depth

    def image_mask(self, height, width):
        """Usable area of an image of this camera resized to (height, width)."""
        mask = np.ones((height, width), bool)
        if self.is_fisheye:
            yy, xx = np.mgrid[0:height, 0:width]
            scale = width / self.width if self.width else 1.0
            radius = (self.valid_radius_px * scale if self.valid_radius_px
                      else 0.985 * min(width, height) / 2)
            mask &= np.hypot(xx - width / 2, yy - height / 2) < radius
        return mask
