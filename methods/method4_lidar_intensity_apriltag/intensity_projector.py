import os
import sys
import numpy as np
import cv2
from pathlib import Path
from rosbags.highlevel import AnyReader

# Add root directory to path to import core if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from core import _pointcloud2

class LidarIntensityProjector:
    """
    Projects 3D LiDAR point clouds (XYZ + Reflectivity) into a 2D synthetic intensity image.
    Supports both equirectangular/cylindrical panoramic projections and target-centric ROI views.
    """
    def __init__(self, width=2048, height=512, fov_v_deg=(-16.0, 16.0)):
        self.W = width
        self.H = height
        self.fov_v_min = np.radians(fov_v_deg[0])
        self.fov_v_max = np.radians(fov_v_deg[1])

    def project_to_panorama(self, points, normalize_intensity=True):
        """
        Converts (N, 4+) [X, Y, Z, Intensity] into a (H, W) 2D intensity raster.
        """
        X = points[:, 0]
        Y = points[:, 1]
        Z = points[:, 2]
        I = points[:, 3]

        r_xy = np.hypot(X, Y)
        r = np.sqrt(r_xy**2 + Z**2)
        valid = (r > 0.2) & (r < 30.0)

        X = X[valid]
        Y = Y[valid]
        Z = Z[valid]
        I = I[valid]
        r_xy = r_xy[valid]

        azimuth = np.arctan2(Y, X) # [-pi, pi]
        elevation = np.arctan2(Z, r_xy) # [fov_v_min, fov_v_max]

        # Map to pixel coordinates
        u = ((azimuth + np.pi) / (2.0 * np.pi) * self.W).astype(np.int32)
        v = ((self.fov_v_max - elevation) / (self.fov_v_max - self.fov_v_min) * self.H).astype(np.int32)

        u = np.clip(u, 0, self.W - 1)
        v = np.clip(v, 0, self.H - 1)

        # Create sparse intensity image
        img_intensity = np.zeros((self.H, self.W), dtype=np.float32)
        img_counts = np.zeros((self.H, self.W), dtype=np.int32)
        img_range = np.full((self.H, self.W), np.nan, dtype=np.float32)

        # Accumulate
        np.add.at(img_intensity, (v, u), I)
        np.add.at(img_counts, (v, u), 1)

        mask = img_counts > 0
        img_intensity[mask] /= img_counts[mask]

        if normalize_intensity:
            p99 = np.percentile(img_intensity[mask], 99.0) if mask.any() else 255.0
            p1 = np.percentile(img_intensity[mask], 1.0) if mask.any() else 0.0
            scale = max(p99 - p1, 1.0)
            img_normalized = np.clip((img_intensity - p1) / scale * 255.0, 0, 255).astype(np.uint8)
        else:
            img_normalized = np.clip(img_intensity, 0, 255).astype(np.uint8)

        return img_normalized, mask

    def dense_interpolated_panorama(self, img_sparse, mask, kernel_size=(5, 5)):
        """
        Applies fast morphological dilation and closing to bridge gaps between scan rings.
        """
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        dilated = cv2.dilate(img_sparse, kernel_v, iterations=1)
        dilated = cv2.dilate(dilated, kernel_h, iterations=1)
        kernel_cross = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_cross)
        return closed

def load_bag_points(bag_path, max_frames=50):
    """
    Reads LiDAR PointCloud2 messages from a ROS bag.
    """
    points_list = []
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if "722z" in c.topic.lower() or "pointcloud2" in c.msgtype.lower()]
        count = 0
        for conn, ts, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            arr = _pointcloud2(msg)
            if arr is not None and len(arr) > 0:
                points_list.append(arr)
            count += 1
            if max_frames and count >= max_frames:
                break
    if not points_list:
        raise ValueError(f"No points extracted from {bag_path}")
    return np.concatenate(points_list, axis=0)
