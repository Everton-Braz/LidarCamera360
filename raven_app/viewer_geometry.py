"""Float64 camera and measurement geometry shared by rendering and picking."""
import numpy as np


def camera_matrix(target, yaw, elevation, half_height, aspect, scene_radius):
    yaw, elevation = np.radians([yaw, elevation])
    direction = np.array([np.cos(elevation)*np.cos(yaw), np.cos(elevation)*np.sin(yaw), np.sin(elevation)])
    right = np.cross(-direction, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    radius = max(float(scene_radius), 1e-3)
    eye = np.asarray(target) + direction * radius * 3
    view = np.eye(4); view[:3, :3] = np.stack((right, up, direction)); view[:3, 3] = -view[:3, :3] @ eye
    near, far = radius*.001, radius*10
    projection = np.diag([1/(half_height*aspect), 1/half_height, -2/(far-near), 1.])
    projection[2, 3] = -(far+near)/(far-near)
    return projection @ view, right, up


def project_points(points, matrix, width, height):
    clip = np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]
    screen = np.column_stack(((clip[:, 0]+1)*width/2, (1-clip[:, 1])*height/2))
    return screen, clip[:, 2]


def measurement_value(kind, points):
    p = np.asarray(points, dtype=np.float64)
    if kind in ('distance', 'polyline'):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 'm'
    if kind == 'angle':
        a, b = p[0]-p[1], p[2]-p[1]
        product = np.linalg.norm(a)*np.linalg.norm(b)
        if product <= 1e-15:
            raise ValueError('An angle needs three distinct points; the middle point is its vertex.')
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b)/product, -1, 1)))), 'deg'
    return None, 'm'
