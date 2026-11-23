import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from raven_app.workflow import (
    UPRIGHT_FILE,
    make_colmap_export_upright,
    rotate_colmap_cameras_180,
    upside_down_cameras,
)

W, H = 64, 48
PARAMS = [30.0, 31.0, 33.5, 22.25, 0.03, -0.01, 0.002, -0.001, 0.0007, -0.0002, 0.004, -0.003]


def project(params, rotation_cw, t_cw, point):
    """COLMAP THIN_PRISM_FISHEYE projection."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1 = params
    x, y, z = rotation_cw @ point + t_cw
    u, v = x / z, y / z
    r = np.hypot(u, v)
    theta = np.arctan(r)
    uu, vv = u * theta / r, v * theta / r
    r2 = uu * uu + vv * vv
    radial = k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3 + k4 * r2 ** 4
    du = uu * radial + 2 * p1 * uu * vv + p2 * (r2 + 2 * uu * uu) + sx1 * r2
    dv = vv * radial + 2 * p2 * uu * vv + p1 * (r2 + 2 * vv * vv) + sy1 * r2
    return np.array([fx * (uu + du) + cx, fy * (vv + dv) + cy])


def read_model(sparse):
    cameras = {}
    for line in (sparse / 'cameras.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            w = line.split()
            cameras[int(w[0])] = [float(v) for v in w[4:]]
    images = {}
    rows = [l for l in (sparse / 'images.txt').read_text().splitlines() if not l.startswith('#')]
    for i in range(0, len(rows) - 1, 2):
        w = rows[i].split()
        if len(w) != 10:
            continue
        q = [float(v) for v in w[1:5]]
        images[w[9]] = (int(w[8]), Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(),
                        np.array([float(v) for v in w[5:8]]), rows[i + 1].split())
    return cameras, images


def pose_line(image_id, rotation_cw, t_cw, camera_id, name):
    q = Rotation.from_matrix(rotation_cw).as_quat()
    return (f'{image_id} {float(q[3])!r} {float(q[0])!r} {float(q[1])!r} {float(q[2])!r} '
            f'{float(t_cw[0])!r} {float(t_cw[1])!r} {float(t_cw[2])!r} {camera_id} {name}')


class UprightExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'colmap_3dgs'
        self.sparse = self.root / 'sparse' / '0'
        self.sparse.mkdir(parents=True)
        # Camera looking along world +x. Upright: image down (+y) is world -z. Upside down: +z.
        upright = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
        upside_down = np.diag([-1.0, -1.0, 1.0]) @ upright
        self.poses = {'cam0/a.jpg': (1, upside_down), 'cam1/a.jpg': (2, upright)}
        self.point = np.array([4.0, 0.7, 0.4])
        self.t = np.array([0.1, -0.2, 0.3])
        cams = ''.join(f'{i} THIN_PRISM_FISHEYE {W} {H} {" ".join(map(repr, PARAMS))}\n' for i in (1, 2))
        (self.sparse / 'cameras.txt').write_text('# cameras\n' + cams)
        rows = ['# images']
        self.raw_pixel = {}
        for image_id, (name, (camera_id, rotation)) in enumerate(self.poses.items(), start=1):
            pixel = project(PARAMS, rotation, self.t, self.point)
            self.raw_pixel[name] = pixel
            rows += [pose_line(image_id, rotation, self.t, camera_id, name), f'{float(pixel[0])!r} {float(pixel[1])!r} 1']
        (self.sparse / 'images.txt').write_text('\n'.join(rows) + '\n')
        self.raw_images = {}
        for name in self.poses:
            yy, xx = np.mgrid[0:H, 0:W]
            image = np.dstack([xx * 3, yy * 5, (xx + yy) * 2]).astype(np.uint8)
            self.raw_images[name] = image
            path = self.root / 'images' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path.with_suffix('.png')), image)
            path.with_suffix('.png').rename(path)  # lossless content under a .jpg name is fine for cv2
            mask = np.full((H, W), 255, np.uint8)
            mask[:5, :7] = 0
            (self.root / 'masks' / name).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.root / 'masks' / f'{name}.png'), mask)

    def tearDown(self):
        self.tmp.cleanup()

    def test_only_the_lens_whose_up_points_down_is_found(self):
        self.assertEqual(upside_down_cameras(self.sparse), [1])

    def test_turned_model_projects_the_world_onto_the_turned_pixel(self):
        rotate_colmap_cameras_180(self.sparse, [1])
        cameras, images = read_model(self.sparse)
        camera_id, rotation, translation, observations = images['cam0/a.jpg']
        pixel = project(cameras[camera_id], rotation, translation, self.point)
        expected = np.array([W, H]) - self.raw_pixel['cam0/a.jpg']
        np.testing.assert_allclose(pixel, expected, atol=1e-9)
        np.testing.assert_allclose([float(observations[0]), float(observations[1])], expected, atol=1e-9)
        # Camera centre in the world does not move, and the upright lens is untouched.
        np.testing.assert_allclose(-rotation.T @ translation,
                                   -self.poses['cam0/a.jpg'][1].T @ self.t, atol=1e-12)
        self.assertEqual(cameras[2], PARAMS)
        self.assertEqual(upside_down_cameras(self.sparse), [])

    def test_turning_twice_restores_the_model(self):
        before = read_model(self.sparse)
        rotate_colmap_cameras_180(self.sparse, [1])
        rotate_colmap_cameras_180(self.sparse, [1])
        after = read_model(self.sparse)
        np.testing.assert_allclose(after[0][1], before[0][1], atol=1e-12)
        for name in self.poses:
            np.testing.assert_allclose(after[1][name][1], before[1][name][1], atol=1e-12)
            np.testing.assert_allclose(after[1][name][2], before[1][name][2], atol=1e-12)

    def test_export_turns_images_masks_and_records_it_once(self):
        source = Path(self.tmp.name) / 'dataset_frame.jpg'
        target = self.root / 'images' / 'cam0' / 'a.jpg'
        target.replace(source)
        import os
        os.link(source, target)  # the export hard-links dataset frames
        self.assertEqual(make_colmap_export_upright(self.root), ['cam0'])
        turned = cv2.imread(str(target))
        self.assertEqual(turned.shape, (H, W, 3))
        self.assertLess(np.abs(turned.astype(int) - np.rot90(self.raw_images['cam0/a.jpg'], 2)).mean(), 3)
        np.testing.assert_array_equal(cv2.imread(str(source)), self.raw_images['cam0/a.jpg'])
        mask = cv2.imread(str(self.root / 'masks' / 'cam0' / 'a.jpg.png'), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(int(mask[-1, -1]), 0)
        self.assertEqual(int(mask[0, 0]), 255)
        untouched = cv2.imread(str(self.root / 'masks' / 'cam1' / 'a.jpg.png'), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(int(untouched[0, 0]), 0)
        record = json.loads((self.root / UPRIGHT_FILE).read_text())
        self.assertEqual(record['camera_ids'], [1])
        self.assertEqual(make_colmap_export_upright(self.root), [])
        self.assertTrue((self.sparse / 'images.bin').is_file())

    def test_pose_adjustment_sees_raw_frames_and_leaves_the_export_upright(self):
        from unittest import mock
        from raven_app import workflow
        make_colmap_export_upright(self.root)
        upright_model = read_model(self.sparse)
        seen = {}

        def fake_refine(sparse_dir, tracks_dir, backup_path=None, **_):
            seen['model'] = read_model(Path(sparse_dir))
            return {'status': 'skipped', 'reason': 'test'}

        with mock.patch('raven_app.pose_ba.refine_poses', fake_refine):
            workflow.refine_colmap_3dgs_poses(self.root, self.root)
        raw = seen['model'][1]['cam0/a.jpg']
        np.testing.assert_allclose(raw[1], self.poses['cam0/a.jpg'][1], atol=1e-12)
        np.testing.assert_allclose(float(raw[3][0]), self.raw_pixel['cam0/a.jpg'][0], atol=1e-9)
        after = read_model(self.sparse)
        np.testing.assert_allclose(after[1]['cam0/a.jpg'][1], upright_model[1]['cam0/a.jpg'][1], atol=1e-12)
        np.testing.assert_allclose(after[0][1], upright_model[0][1], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
