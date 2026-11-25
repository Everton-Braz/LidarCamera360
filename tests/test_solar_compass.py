import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from raven_app.solar_compass import (
    cluster_solar_observations,
    find_colmap_dir,
    solar_position,
    solve_solar_north,
    unproject_pixels,
)
from raven_app.true_north import (
    EdgeMap,
    angle_diff_deg,
    edge_orientation,
    fetch_edge_map,
    grid_convergence_deg,
    register_walls,
    rotation_2d,
    solve_north,
)


def _project_thin_prism_simple(b_cam, fx, fy, cx, cy):
    """Exact projection for a zero-distortion THIN_PRISM_FISHEYE lens."""
    theta = math.acos(max(-1.0, min(1.0, float(b_cam[2]))))
    r_xy = math.hypot(float(b_cam[0]), float(b_cam[1]))
    if r_xy < 1e-12:
        return cx, cy
    return cx + fx * theta * (b_cam[0] / r_xy), cy + fy * theta * (b_cam[1] / r_xy)


def _render_sky_frame(path, width, height, sun_uv, *, sky_bgr=(215, 195, 185), halo_bgr=(230, 225, 220),
                      core_r=14, halo_r=38):
    """Draw a synthetic sky image with a bright sun core and circumsolar aureole."""
    img = np.full((height, width, 3), sky_bgr, dtype=np.uint8)
    if sun_uv is not None:
        cx, cy = int(round(sun_uv[0])), int(round(sun_uv[1]))
        cv2.circle(img, (cx, cy), halo_r, halo_bgr, -1)
        cv2.circle(img, (cx, cy), core_r, (255, 255, 255), -1)
    cv2.imwrite(str(path), img)


def _render_equirect_sky_frame(path, width, height, sun_cam_dir, *, sky_bgr=(215, 195, 185),
                               halo_bgr=(230, 225, 220), core_deg=2.2, halo_deg=6.0):
    """Render a physically accurate spherical sun disk onto an equirectangular image."""
    u = np.arange(width, dtype=np.float64)[None, :]
    v = np.arange(height, dtype=np.float64)[:, None]
    lon = (u / float(width) - 0.5) * 2.0 * math.pi
    lat = (0.5 - v / float(height)) * math.pi
    bx = np.cos(lat) * np.sin(lon)
    by = np.broadcast_to(-np.sin(lat), bx.shape)
    bz = np.cos(lat) * np.cos(lon)
    d = np.asarray(sun_cam_dir, dtype=np.float64)
    d = d / np.linalg.norm(d)
    dots = bx * d[0] + by * d[1] + bz * d[2]
    img = np.full((height, width, 3), sky_bgr, dtype=np.uint8)
    img[dots >= math.cos(math.radians(halo_deg))] = halo_bgr
    img[dots >= math.cos(math.radians(core_deg))] = (255, 255, 255)
    cv2.imwrite(str(path), img)


def _gps_info_at(lat, lon, epochs, crs='EPSG:31984'):
    import pyproj
    tf = pyproj.Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    e, n = tf.transform(lon, lat)
    utm = np.tile([e, n, 10.0], (len(epochs), 1))
    return {
        'utm_coords': utm,
        'crs_info': {'horizontal_crs': crs, 'crs_name': crs},
        'records': [{'timestamp_epoch': float(t)} for t in epochs],
        'mean_wgs84': {'latitude_deg': float(lat), 'longitude_deg': float(lon), 'altitude_m': 10.0},
    }


class SolarEphemerisTests(unittest.TestCase):
    def test_equinox_noon_near_equator_is_overhead(self):
        # 2026-03-20 12:07 UTC at (0, 0) is near solar noon on the vernal equinox.
        alt, _ = solar_position(0.0, 0.0, 1774008420.0)
        self.assertGreater(alt, 88.0)

    def test_northern_and_southern_midday_azimuths(self):
        # 2026-06-21 12:00 UTC at lon=0: summer solstice noon -> sun south from 45N, north from 35S.
        epoch = 1782043200.0
        alt_n, az_n = solar_position(45.0, 0.0, epoch)
        alt_s, az_s = solar_position(-35.0, 0.0, epoch)
        self.assertGreater(alt_n, 65.0)
        self.assertLess(angle_diff_deg(az_n, 180.0), 3.0)
        self.assertGreater(alt_s, 28.0)
        self.assertLess(angle_diff_deg(az_s, 0.0), 3.0)

    def test_midnight_is_below_horizon(self):
        # 2026-08-21 03:00 UTC in Fortaleza (~00:00 local) -> sun well below horizon.
        alt, _ = solar_position(-3.70, -38.60, 1787281200.0)
        self.assertLess(alt, -50.0)


class SolarCompassEndToEndTests(unittest.TestCase):
    def _build_colmap_dataset(self, root, true_yaw_deg, lat, lon, epoch, n_frames=14,
                              sky_bgr=(215, 195, 185), halo_bgr=(230, 225, 220),
                              elev_offset_deg=0.0):
        cdir = root / 'colmap_3dgs'
        sparse = cdir / 'sparse' / '0'
        cam0_dir = cdir / 'images' / 'cam0'
        sparse.mkdir(parents=True)
        cam0_dir.mkdir(parents=True)

        w = h = 480
        fx = fy = 140.0
        cx = cy = 240.0
        (sparse / 'cameras.txt').write_text(
            f'1 THIN_PRISM_FISHEYE {w} {h} {fx} {fy} {cx} {cy} 0 0 0 0 0 0 0 0\n',
            encoding='utf-8',
        )

        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, az_deg = solar_position(lat, lon, epoch)
        sun_en_math = 90.0 - (az_deg - conv)
        loc_math = math.radians(sun_en_math - true_yaw_deg)
        alt_rad = math.radians(alt_deg + elev_offset_deg)
        sun_local = np.array([
            math.cos(alt_rad) * math.cos(loc_math),
            math.cos(alt_rad) * math.sin(loc_math),
            math.sin(alt_rad),
        ], dtype=np.float64)

        lines = ['# images\n']
        for i in range(n_frames):
            # Camera looks mostly upward with varying yaw and slight tilt across frames.
            R_cw = Rotation.from_euler('xyz', [10.0 * math.sin(i), 8.0 * math.cos(i), i * 25.0], degrees=True).as_matrix()
            # Orient camera +Z to point roughly upward (+Z world) so the sun is well inside the fisheye FOV.
            b_cam = R_cw @ sun_local
            if b_cam[2] < 0.25:
                R_cw = Rotation.from_euler('z', i * 25.0, degrees=True).as_matrix()
                b_cam = R_cw @ sun_local
            uv = _project_thin_prism_simple(b_cam, fx, fy, cx, cy)
            name = f'cam0/frame_{i + 1:06d}.png'
            _render_sky_frame(cdir / 'images' / name, w, h, uv, sky_bgr=sky_bgr, halo_bgr=halo_bgr)
            q = Rotation.from_matrix(R_cw).as_quat()  # qx, qy, qz, qw
            lines.append(f'{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {float(i)} 0 0 1 {name}\n\n')
        (sparse / 'images.txt').write_text(''.join(lines), encoding='utf-8')
        return cdir

    def test_synthetic_colmap_recovers_true_north_accurately(self):
        # Fortaleza afternoon: 2026-10-22 17:51 UTC (sun alt ~36.6 deg).
        lat, lon, epoch, true_yaw = -3.7714, -38.6273, 1792691460.0, 123.4
        with tempfile.TemporaryDirectory() as d:
            cdir = self._build_colmap_dataset(Path(d), true_yaw, lat, lon, epoch, n_frames=14)
            gps = _gps_info_at(lat, lon, [epoch - 10.0, epoch + 10.0])
            conv = grid_convergence_deg(lat, lon, 'EPSG:31984')
            res = solve_solar_north(gps, conv, colmap_dir=cdir, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'])
            self.assertTrue(res['strong'])
            self.assertEqual(res['inliers'], 14)
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.0)
            self.assertLess(abs(res['elev_err_med_deg']), 1.0)

    def test_dark_ceiling_indoor_lights_are_rejected(self):
        lat, lon, epoch = -3.7714, -38.6273, 1792691460.0
        with tempfile.TemporaryDirectory() as d:
            # Bright white bulbs surrounded by a dark indoor ceiling -> ring_p10 << 180 -> rejected.
            cdir = self._build_colmap_dataset(
                Path(d), 77.0, lat, lon, epoch, n_frames=12,
                sky_bgr=(60, 60, 60), halo_bgr=(90, 90, 90),
            )
            gps = _gps_info_at(lat, lon, [epoch])
            conv = grid_convergence_deg(lat, lon, 'EPSG:31984')
            res = solve_solar_north(gps, conv, colmap_dir=cdir, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertFalse(res['qualifies'])
            self.assertEqual(res['inliers'], 0)

    def test_wrong_elevation_bright_object_is_rejected(self):
        lat, lon, epoch = -3.7714, -38.6273, 1792691460.0
        with tempfile.TemporaryDirectory() as d:
            cdir = self._build_colmap_dataset(
                Path(d), 77.0, lat, lon, epoch, n_frames=12, elev_offset_deg=12.0,
            )
            gps = _gps_info_at(lat, lon, [epoch])
            conv = grid_convergence_deg(lat, lon, 'EPSG:31984')
            res = solve_solar_north(gps, conv, colmap_dir=cdir, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertFalse(res['qualifies'])
            self.assertEqual(res['inliers'], 0)

    def test_nighttime_capture_is_rejected_without_reading_images(self):
        gps = _gps_info_at(-3.70, -38.60, [1787281200.0])  # local midnight
        res = solve_solar_north(gps, 0.0, log=lambda _: None)
        self.assertIsNotNone(res)
        self.assertFalse(res['qualifies'])
        self.assertEqual(res['reason'], 'sun_below_horizon_or_too_low')

    def test_competing_bright_clusters_are_rejected(self):
        v1 = np.array([math.cos(math.radians(40.0)), 0.0, math.sin(math.radians(40.0))])
        v2 = np.array([-math.cos(math.radians(40.0)), 0.0, math.sin(math.radians(40.0))])
        obs = ([{'yaw_deg': 50.0, 'elev_err_deg': 0.1, 'vec_mid': v1} for _ in range(6)]
               + [{'yaw_deg': 230.0, 'elev_err_deg': 0.1, 'vec_mid': v2} for _ in range(5)])
        res = cluster_solar_observations(obs, 40.0, 260.0)
        self.assertFalse(res['qualifies'])
        self.assertEqual(res['reason'], 'competing_bright_directions')

    def test_direct_rig_fallback_without_colmap_recovers_north(self):
        lat, lon, epoch, true_yaw = -3.7714, -38.6273, 1792691460.0, 212.5
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, az_deg = solar_position(lat, lon, epoch)
        loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
        alt_rad = math.radians(alt_deg)
        sun_local = np.array([math.cos(alt_rad) * math.cos(loc_math),
                              math.cos(alt_rad) * math.sin(loc_math),
                              math.sin(alt_rad)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cam0 = root / 'images' / 'cam0'
            cam0.mkdir(parents=True)
            w = h = 480
            fx = fy = 140.0
            cx = cy = 240.0
            (root / 'rig_calibration.json').write_text(json.dumps({
                'lens_model': 'THIN_PRISM_FISHEYE',
                'T_lidar_to_cam0_rigid_4x4': np.eye(4).tolist(),
                'cam0_front_intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': w, 'height': h},
            }), encoding='utf-8')
            (root / 'workflow_config.json').write_text(json.dumps({'fps': 1.0}), encoding='utf-8')
            traj_rows = []
            for i in range(12):
                R_wb = Rotation.from_euler('z', i * 15.0, degrees=True).as_matrix()
                q = Rotation.from_matrix(R_wb).as_quat()
                traj_rows.append([100.0 + i, float(i), 0.0, 0.0, q[0], q[1], q[2], q[3]])
                b_cam = R_wb.T @ sun_local
                uv = _project_thin_prism_simple(b_cam, fx, fy, cx, cy)
                _render_sky_frame(cam0 / f'frame_{i + 1:06d}.png', w, h, uv)
            traj_path = root / 'slam' / 'result' / 'track.txt'
            traj_path.parent.mkdir(parents=True)
            np.savetxt(traj_path, traj_rows)
            gps = _gps_info_at(lat, lon, [epoch, epoch + 11.0], crs=crs)
            res = solve_solar_north(gps, conv, search_paths=(traj_path,), trajectory_path=traj_path,
                                    dt_sync=0.0, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'])
            self.assertEqual(res['source'], 'rig_trajectory')
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.0)

    def test_unproject_supports_multiple_camera_models(self):
        uv = np.array([[240.0, 240.0]])
        for model, params in (
            ('THIN_PRISM_FISHEYE', np.array([140, 140, 240, 240, 0, 0, 0, 0, 0, 0, 0, 0], float)),
            ('OPENCV_FISHEYE', np.array([140, 140, 240, 240, 0, 0, 0, 0], float)),
            ('PINHOLE', np.array([500, 500, 240, 240], float)),
            ('SIMPLE_PINHOLE', np.array([500, 240, 240], float)),
            ('EQUIRECTANGULAR', np.zeros(0)),
        ):
            b = unproject_pixels(uv, model, params, width=480, height=480)[0]
            self.assertTrue(np.allclose(b, [0.0, 0.0, 1.0], atol=1e-6), msg=model)


class SolarAndImageryFusionTests(unittest.TestCase):
    def _square_scene(self, true_yaw, origin=np.array([541000.0, 9583000.0]), gsd=0.3, half=50.0):
        """A 4-fold rotationally symmetric square room: full-360 imagery cannot disambiguate 90 deg multiples!"""
        s = 12.0
        xs = np.arange(-s, s, 0.1)
        plan = np.vstack([
            np.c_[xs, np.full_like(xs, -s)], np.c_[xs, np.full_like(xs, s)],
            np.c_[np.full_like(xs, -s), xs], np.c_[np.full_like(xs, s), xs],
        ])
        points = np.vstack([np.c_[plan, np.full(len(plan), z)] for z in (0.0, 1.5, 2.5)])
        size = int(2 * half / gsd)
        east_min, north_max = origin[0] - half, origin[1] + half
        east, north = np.meshgrid(east_min + (np.arange(size) + 0.5) * gsd,
                                  north_max - (np.arange(size) + 0.5) * gsd)
        local = (np.c_[east.ravel(), north.ravel()] - origin) @ rotation_2d(true_yaw)
        inside = (np.abs(local[:, 0]) <= s) & (np.abs(local[:, 1]) <= s)
        gray = np.where(inside.reshape(east.shape), 220.0, 20.0)
        return points, EdgeMap(edge_orientation(gray), east_min, north_max, gsd)

    def test_solar_prior_breaks_90_degree_symmetry_and_recovers_exact_shift(self):
        origin = np.array([541000.0, 9583000.0])
        true_yaw = 359.0  # Also tests 0/360 wrap-around!
        points, edges = self._square_scene(true_yaw, origin)
        # GPS is shifted 8 m east, 2.5 m south of truth!
        gps_anchor = origin + np.array([8.0, -2.5])
        gps = {
            'utm_coords': np.array([[gps_anchor[0], gps_anchor[1], 0.0]]),
            'crs_info': {'horizontal_crs': 'EPSG:31984', 'crs_name': 'SIRGAS'},
            'records': [{'timestamp_epoch': 1787320500.0}],
            'mean_wgs84': {'latitude_deg': -3.70, 'longitude_deg': -38.60, 'altitude_m': 0.0},
        }
        # Without solar prior, 4-fold symmetry makes full-360 imagery fail rival margin.
        no_solar = solve_north(gps, points=points, edges=edges, local_xy=np.zeros((1, 2)),
                               allow_network=False, log=lambda _: None)
        self.assertEqual(no_solar['status'], 'unresolved')

        # With solar compass ~2 deg off (wrapping across 0 deg), prior-guided imagery locks both yaw & position!
        solar = {'yaw_grid_deg': 1.2, 'sigma_deg': 3.0, 'inliers': 20, 'qualifies': True, 'strong': True}
        with_solar = solve_north(gps, points=points, edges=edges, local_xy=np.zeros((1, 2)),
                                 solar=solar, allow_network=False, log=lambda _: None)
        self.assertEqual((with_solar['status'], with_solar['method']), ('resolved', 'solar_and_aerial_imagery'))
        self.assertLess(angle_diff_deg(with_solar['yaw_grid_deg'], true_yaw), 0.75)
        self.assertLess(np.linalg.norm(np.subtract(with_solar['translation_en'], origin)), 0.8)

    def test_offline_strong_solar_resolves_while_weak_solar_requests_review(self):
        gps = _gps_info_at(-3.70, -38.60, [1787320500.0])
        strong = {'yaw_grid_deg': 76.0, 'sigma_deg': 3.2, 'inliers': 30, 'qualifies': True, 'strong': True}
        res_strong = solve_north(gps, solar=strong, allow_network=False, log=lambda _: None)
        self.assertEqual((res_strong['status'], res_strong['method']), ('resolved', 'solar_compass'))
        self.assertAlmostEqual(res_strong['yaw_grid_deg'], 76.0)

        weak = {'yaw_grid_deg': 76.0, 'sigma_deg': 8.0, 'inliers': 5, 'qualifies': True, 'strong': False}
        res_weak = solve_north(gps, solar=weak, allow_network=False, log=lambda _: None)
        self.assertEqual((res_weak['status'], res_weak['method']), ('needs_review', 'solar_compass'))

    def test_fetch_edge_map_prefers_complete_zoom_over_missing_tile(self):
        good_tile = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        calls = []

        def fake_fetch(zoom, x, y, timeout):
            calls.append((zoom, x, y))
            # Zoom 19 misses one tile; zoom 18 has complete coverage.
            if zoom == 19 and len([c for c in calls if c[0] == 19]) == 1:
                return None
            return good_tile

        with mock.patch('raven_app.true_north._fetch_tile', side_effect=fake_fetch):
            em = fetch_edge_map(np.array([544465.0, 9591506.0]), 30.0, 'EPSG:31984', log=lambda _: None)
        self.assertIsNotNone(em)
        self.assertEqual(em.zoom, 18)

    def test_indoor_bright_ceiling_skylight_rejected_by_parallax(self):
        lat, lon, epoch = -3.7714, -38.6273, 1792691460.0
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, _ = solar_position(lat, lon, epoch)
        alt_rad = math.radians(alt_deg)
        with tempfile.TemporaryDirectory() as d:
            cdir = Path(d) / 'colmap_3dgs'
            (cdir / 'sparse' / '0').mkdir(parents=True)
            (cdir / 'images' / 'cam0').mkdir(parents=True)
            w = h = 480
            fx = fy = 140.0
            cx = cy = 240.0
            (cdir / 'sparse' / '0' / 'cameras.txt').write_text(
                f'1 THIN_PRISM_FISHEYE {w} {h} {fx} {fy} {cx} {cy} 0 0 0 0 0 0 0 0\n', encoding='utf-8'
            )
            # Finite skylight 3 m above camera over a 1.5 m walk on a bright white ceiling (ring_p10 >= 180).
            light_z = 3.0
            light_pos = np.array([light_z / math.tan(alt_rad), 0.0, light_z])
            lines = ['# images\n']
            for i in range(16):
                cam_pos = np.array([0.0, -0.75 + 1.5 * i / 15.0, 0.0])
                vec = light_pos - cam_pos
                vec /= np.linalg.norm(vec)
                R_cw = Rotation.from_euler('z', i * 15.0, degrees=True).as_matrix()
                uv = _project_thin_prism_simple(R_cw @ vec, fx, fy, cx, cy)
                name = f'cam0/frame_{i + 1:06d}.png'
                _render_sky_frame(cdir / 'images' / name, w, h, uv,
                                  sky_bgr=(205, 205, 205), halo_bgr=(235, 235, 235), core_r=12, halo_r=32)
                t_cw = -R_cw @ cam_pos
                q = Rotation.from_matrix(R_cw).as_quat()
                lines.append(f'{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {t_cw[0]} {t_cw[1]} {t_cw[2]} 1 {name}\n\n')
            (cdir / 'sparse' / '0' / 'images.txt').write_text(''.join(lines), encoding='utf-8')
            gps = _gps_info_at(lat, lon, [epoch - 10.0, epoch + 10.0], crs=crs)
            res = solve_solar_north(gps, conv, colmap_dir=cdir, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertFalse(res['qualifies'])
            self.assertEqual(res['reason'], 'finite_distance_light_source')
            json.dumps(res, allow_nan=False)

    def test_chunked_colmap_reads_frames_json_and_capture_origin(self):
        # Full capture spans 24 minutes near noon (sun alt rises >6 deg); chunk is minutes 0..3 with origin=15s.
        lat, lon, t0, true_yaw = -3.7714, -38.6273, 1790688000.0, 175.0
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        origin_s = 15.0
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cdir = root / 'colmap_3dgs'
            (cdir / 'sparse' / '0').mkdir(parents=True)
            (cdir / 'images' / 'cam0').mkdir(parents=True)
            (root / 'images').mkdir(parents=True)
            w = h = 480
            fx = fy = 140.0
            cx = cy = 240.0
            (cdir / 'sparse' / '0' / 'cameras.txt').write_text(
                f'1 THIN_PRISM_FISHEYE {w} {h} {fx} {fy} {cx} {cy} 0 0 0 0 0 0 0 0\n', encoding='utf-8'
            )
            ts_map = {}
            lines = ['# images\n']
            for i in range(14):
                vt = float(i * 12.0)
                alt_deg, az_deg = solar_position(lat, lon, t0 + origin_s + vt)
                loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
                alt_rad = math.radians(alt_deg)
                sun_local = np.array([math.cos(alt_rad) * math.cos(loc_math),
                                      math.cos(alt_rad) * math.sin(loc_math),
                                      math.sin(alt_rad)])
                R_cw = Rotation.from_euler('z', i * 20.0, degrees=True).as_matrix()
                uv = _project_thin_prism_simple(R_cw @ sun_local, fx, fy, cx, cy)
                name = f'cam0/frame_{i + 1:06d}.png'
                ts_map[name] = vt
                _render_sky_frame(cdir / 'images' / name, w, h, uv)
                q = Rotation.from_matrix(R_cw).as_quat()
                lines.append(f'{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {float(i)} 0 0 1 {name}\n\n')
            (cdir / 'sparse' / '0' / 'images.txt').write_text(''.join(lines), encoding='utf-8')
            (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')
            gps = _gps_info_at(lat, lon, [t0, t0 + 1440.0], crs=crs)
            res = solve_solar_north(gps, conv, colmap_dir=cdir, capture_origin_s=origin_s, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'])
            self.assertTrue(res['strong'])
            self.assertLess(abs(res['elev_err_med_deg']), 0.6)
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.5)

    def test_direct_rig_reads_real_workflow_schema_photometric_calib_and_non_identity_rotation(self):
        lat, lon, epoch, true_yaw = -3.7714, -38.6273, 1792691460.0, 212.5
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, az_deg = solar_position(lat, lon, epoch)
        loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
        alt_rad = math.radians(alt_deg)
        sun_local = np.array([math.cos(alt_rad) * math.cos(loc_math),
                              math.cos(alt_rad) * math.sin(loc_math),
                              math.sin(alt_rad)])
        # Non-identity camera-to-body rotation tests exact T_lidar_to_cam0 convention!
        R_lc = Rotation.from_euler('xyz', [18.0, -12.0, 65.0], degrees=True).as_matrix()
        T_lc = np.eye(4)
        T_lc[:3, :3] = R_lc
        T_lc[:3, 3] = [0.02, 0.10, 0.12]
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            # Put a decoy colmap_3dgs in parent folder to ensure run_dir does NOT leak upward!
            decoy = parent / 'colmap_3dgs'
            (decoy / 'sparse' / '0').mkdir(parents=True)
            (decoy / 'images').mkdir(parents=True)
            (decoy / 'sparse' / '0' / 'cameras.txt').write_text('', encoding='utf-8')
            (decoy / 'sparse' / '0' / 'images.txt').write_text('', encoding='utf-8')

            run_dir = parent / 'run_output'
            cam0 = run_dir / 'images' / 'cam0'
            cam0.mkdir(parents=True)
            w = h = 480
            fx = fy = 140.0
            cx = cy = 240.0
            cal_path = run_dir / 'rig_calibration_photometric.json'
            cal_path.write_text(json.dumps({
                'lens_model': 'THIN_PRISM_FISHEYE',
                'T_lidar_to_cam0_rigid_4x4': T_lc.tolist(),
                'cam0_front_intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': w, 'height': h},
            }), encoding='utf-8')
            (run_dir / 'workflow_config.json').write_text(json.dumps({
                'schema_version': 1,
                'runs': [{
                    'calibration_used': {'path': str(cal_path)},
                    'image_extraction': {'camera_signatures': {'cam0': {'fps': 1.0}}},
                }],
            }), encoding='utf-8')
            traj_rows = []
            ts_map = {}
            for i in range(12):
                R_wb = Rotation.from_euler('z', i * 15.0, degrees=True).as_matrix()
                q = Rotation.from_matrix(R_wb).as_quat()
                traj_rows.append([100.0 + i, float(i), 0.0, 0.0, q[0], q[1], q[2], q[3]])
                R_cw = (R_wb @ R_lc).T
                uv = _project_thin_prism_simple(R_cw @ sun_local, fx, fy, cx, cy)
                name = f'cam0/frame_{i + 1:06d}.png'
                ts_map[name] = float(i)
                _render_sky_frame(run_dir / 'images' / name, w, h, uv)
            (run_dir / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')
            traj_path = run_dir / 'slam_refined' / 'result' / 'track.txt'
            traj_path.parent.mkdir(parents=True)
            np.savetxt(traj_path, traj_rows)

            self.assertIsNone(find_colmap_dir((traj_path, run_dir)))
            gps = _gps_info_at(lat, lon, [epoch, epoch + 11.0], crs=crs)
            res = solve_solar_north(gps, conv, search_paths=(traj_path, run_dir), trajectory_path=traj_path,
                                    dt_sync=0.0, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'])
            self.assertEqual(res['source'], 'rig_trajectory')
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.0)

    def test_conflicting_solar_and_synced_gps_requests_review(self):
        origin = np.array([541000.0, 9583000.0])
        times = np.arange(14, dtype=np.float64)
        local = np.c_[times * 4.0, np.zeros(14)]
        en = local @ rotation_2d(120.0).T + origin
        gps = {
            'utm_coords': np.c_[en, np.zeros(14)],
            'crs_info': {'horizontal_crs': 'EPSG:31984', 'crs_name': 'SIRGAS'},
            'records': [{'timestamp_epoch': 1787320500.0 + t} for t in times],
            'mean_wgs84': {'latitude_deg': -3.77, 'longitude_deg': -38.63, 'altitude_m': 0.0},
        }
        traj = np.c_[times + 500.0, local, np.zeros(14)]
        # Strong solar disagrees with strong synced GPS by 65 deg -> must NOT silently resolve!
        bad_solar = {'yaw_grid_deg': 185.0, 'sigma_deg': 2.5, 'inliers': 30, 'qualifies': True, 'strong': True}
        res = solve_north(gps, trajectory=traj, dt_sync=0.0, solar=bad_solar, allow_network=False, log=lambda _: None)
        self.assertEqual(res['status'], 'needs_review')
        self.assertIn('sensor_conflict', res['evidence'])

    def test_weak_solar_with_marginal_imagery_requests_review(self):
        origin = np.array([541000.0, 9583000.0])
        points, edges = self._square_scene(359.0, origin)
        gps = {
            'utm_coords': np.array([[origin[0] + 4.0, origin[1] - 2.0, 0.0]]),
            'crs_info': {'horizontal_crs': 'EPSG:31984', 'crs_name': 'SIRGAS'},
            'records': [{'timestamp_epoch': 1787320500.0}],
            'mean_wgs84': {'latitude_deg': -3.70, 'longitude_deg': -38.60, 'altitude_m': 0.0},
        }
        weak_solar = {'yaw_grid_deg': 1.0, 'sigma_deg': 8.0, 'inliers': 5, 'qualifies': True, 'strong': False}
        marginal_reg = {
            'yaw_grid_deg': 359.2, 'score': 0.12, 'shift_en_m': [-4.0, 2.0],
            'sigma_along_m': 0.5, 'sigma_across_m': 0.5, 'along_bearing_deg': 0.0,
            'median_score': 0.08, 'peak_z': 2.2, 'trans_z': 4.5, 'trans_margin': 0.15,
            'rival_yaw_deg': 90.0, 'rival_margin': 0.05, 'sigma_deg': 2.0,
            'offset_from_gps_m': 4.47, 'wall_points': 400, 'prior_guided': True, 'accepted': True,
        }
        with mock.patch('raven_app.true_north.register_walls', return_value=marginal_reg):
            res = solve_north(gps, points=points, edges=edges, local_xy=np.zeros((1, 2)),
                              solar=weak_solar, allow_network=False, log=lambda _: None)
        self.assertEqual(res['status'], 'needs_review')
        self.assertEqual(res['method'], 'solar_and_aerial_imagery')

    def test_recorded_dt_sync_reads_photometric_and_workflow_config(self):
        from raven_app.georeference import recorded_dt_sync
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            traj = root / 'slam_refined' / 'result' / 'track.txt'
            (root / 'workflow_config.json').write_text(json.dumps({
                'schema_version': 1,
                'runs': [{
                    'paths': {'insv': {'path': r'D:\data\scan.insv'}},
                    'resolved_dt_sync_seconds': -3.25,
                }],
            }), encoding='utf-8')
            self.assertEqual(recorded_dt_sync((traj,), root / 'scan.insv'), -3.25)
            self.assertIsNone(recorded_dt_sync((traj,), root / 'other.insv'))
            (root / 'rig_calibration_photometric.json').write_text(
                json.dumps({'dt_sync_seconds': -7.75}), encoding='utf-8'
            )
            self.assertEqual(recorded_dt_sync((traj,), root / 'scan.insv'), -7.75)

    def test_negative_correlation_scores_rejected_cleanly(self):
        origin = np.array([541000.0, 9583000.0])
        # Build a purely negative edge field at yaw=0 so all wall correlations are negative.
        size = 200
        gsd = 0.3
        half = 0.5 * size * gsd
        edges = EdgeMap(np.full((size, size), -1.0 + 0.0j, dtype=np.complex64), origin[0] - half, origin[1] + half, gsd)
        walls = np.c_[np.linspace(-5.0, 5.0, 80), np.zeros(80)]
        dirs = np.zeros(80)
        reg = register_walls(walls, dirs, np.zeros(2), origin, edges,
                             yaw_prior_deg=0.0, yaw_prior_half_deg=10.0, yaw_prior_sigma_deg=3.0)
        self.assertFalse(reg['accepted'])

    def test_long_25_minute_straight_walk_compensates_solar_azimuth_and_altitude_drift(self):
        # Fortaleza late morning: sun climbs >5 deg in altitude and swings >3 deg in azimuth over 25 min.
        lat, lon, t0, true_yaw = -3.7714, -38.6273, 1787316900.0, 142.3
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt0, az0 = solar_position(lat, lon, t0)
        alt1, az1 = solar_position(lat, lon, t0 + 1500.0)
        self.assertGreater(abs(alt1 - alt0), 5.0)
        self.assertGreater(angle_diff_deg(az1, az0), 3.0)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cdir = root / 'colmap_3dgs'
            (cdir / 'sparse' / '0').mkdir(parents=True)
            (cdir / 'images' / 'cam0').mkdir(parents=True)
            (root / 'images').mkdir(parents=True)
            w = h = 480
            fx = fy = 140.0
            cx = cy = 240.0
            (cdir / 'sparse' / '0' / 'cameras.txt').write_text(
                f'1 THIN_PRISM_FISHEYE {w} {h} {fx} {fy} {cx} {cy} 0 0 0 0 0 0 0 0\n', encoding='utf-8'
            )
            ts_map = {}
            lines = ['# images\n']
            n_frames = 16
            for i in range(n_frames):
                vt = float(i * 100.0)  # 0 .. 1500 s (25 minutes)
                alt_deg, az_deg = solar_position(lat, lon, t0 + vt)
                loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
                alt_rad = math.radians(alt_deg)
                sun_local = np.array([
                    math.cos(alt_rad) * math.cos(loc_math),
                    math.cos(alt_rad) * math.sin(loc_math),
                    math.sin(alt_rad),
                ], dtype=np.float64)
                # Straight-line 150 m walk so uncompensated solar motion would mimic finite-distance parallax.
                cam_pos = np.array([float(i * 10.0), 0.0, 0.0], dtype=np.float64)
                R_cw = Rotation.from_euler('z', i * 20.0, degrees=True).as_matrix()
                uv = _project_thin_prism_simple(R_cw @ sun_local, fx, fy, cx, cy)
                name = f'cam0/frame_{i + 1:06d}.png'
                ts_map[name] = vt
                _render_sky_frame(cdir / 'images' / name, w, h, uv)
                t_cw = -R_cw @ cam_pos
                q = Rotation.from_matrix(R_cw).as_quat()
                lines.append(f'{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {t_cw[0]} {t_cw[1]} {t_cw[2]} 1 {name}\n\n')
            (cdir / 'sparse' / '0' / 'images.txt').write_text(''.join(lines), encoding='utf-8')
            (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')

            gps = _gps_info_at(lat, lon, [t0, t0 + 1500.0], crs=crs)
            res = solve_solar_north(gps, conv, colmap_dir=cdir, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'], msg=res)
            self.assertTrue(res['strong'], msg=res)
            self.assertEqual(res['reason'], 'ok')
            self.assertEqual(res['inliers'], n_frames)
            self.assertLess(res['rms_3d_deg'], 0.8)
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.0)

    def test_slam_profile_marker_workflow_config_does_not_trap_traversal(self):
        lat, lon, epoch, true_yaw = -3.7714, -38.6273, 1792691460.0, 212.5
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, az_deg = solar_position(lat, lon, epoch)
        loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
        alt_rad = math.radians(alt_deg)
        sun_local = np.array([
            math.cos(alt_rad) * math.cos(loc_math),
            math.cos(alt_rad) * math.sin(loc_math),
            math.sin(alt_rad),
        ])
        with tempfile.TemporaryDirectory() as d:
            parent = Path(d)
            # Parent decoy colmap_3dgs must NOT be leaked into from run_dir.
            decoy = parent / 'colmap_3dgs'
            (decoy / 'sparse' / '0').mkdir(parents=True)
            (decoy / 'images').mkdir(parents=True)
            (decoy / 'sparse' / '0' / 'cameras.txt').write_text('', encoding='utf-8')
            (decoy / 'sparse' / '0' / 'images.txt').write_text('', encoding='utf-8')

            run_dir = parent / 'chunk_01'
            slam_dir = run_dir / 'slam_raw_abc123'
            traj_path = slam_dir / 'result' / 'Raven_3DMakerPro_Scan.txt'
            traj_path.parent.mkdir(parents=True)
            # Timelapse SLAM profile marker named workflow_config.json inside slam_raw_*:
            (slam_dir / 'workflow_config.json').write_text(json.dumps({
                'name': 'raven_timelapse_stable_v1',
                'filter_size_surf_m': 0.06,
                'voxel_size_m': 0.2,
            }), encoding='utf-8')

            # True run-level workflow_config.json (fps only, no frames.json) + rig_calibration.json.
            w = h = 480
            fx = fy = 140.0
            cx = cy = 240.0
            cal_path = run_dir / 'rig_calibration.json'
            cal_path.write_text(json.dumps({
                'lens_model': 'THIN_PRISM_FISHEYE',
                'T_lidar_to_cam0_rigid_4x4': np.eye(4).tolist(),
                'cam0_front_intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': w, 'height': h},
            }), encoding='utf-8')
            (run_dir / 'workflow_config.json').write_text(json.dumps({
                'schema_version': 1,
                'runs': [{
                    'calibration_used': {'path': str(cal_path)},
                    'image_extraction': {'camera_signatures': {'cam0': {'fps': 1.0}}},
                }],
            }), encoding='utf-8')
            cam0 = run_dir / 'images' / 'cam0'
            cam0.mkdir(parents=True)
            traj_rows = []
            for i in range(12):
                R_wb = Rotation.from_euler('z', i * 15.0, degrees=True).as_matrix()
                q = Rotation.from_matrix(R_wb).as_quat()
                traj_rows.append([100.0 + i, float(i), 0.0, 0.0, q[0], q[1], q[2], q[3]])
                uv = _project_thin_prism_simple(R_wb.T @ sun_local, fx, fy, cx, cy)
                _render_sky_frame(cam0 / f'frame_{i + 1:06d}.png', w, h, uv)
            np.savetxt(traj_path, traj_rows)

            # 1) Without run_dir/colmap_3dgs, find_colmap_dir passes slam_raw_abc123 but stops at chunk_01 (no parent leak).
            self.assertIsNone(find_colmap_dir((traj_path,)))
            # Direct rig fallback skips slam_raw_abc123/workflow_config.json and reads chunk_01/workflow_config.json fps.
            gps = _gps_info_at(lat, lon, [epoch, epoch + 11.0], crs=crs)
            res = solve_solar_north(gps, conv, search_paths=(traj_path,), trajectory_path=traj_path,
                                    dt_sync=0.0, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'])
            self.assertEqual(res['source'], 'rig_trajectory')
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.0)

            # 2) When run_dir/colmap_3dgs exists, find_colmap_dir must walk past slam_raw_abc123 to find it!
            run_colmap = run_dir / 'colmap_3dgs'
            (run_colmap / 'sparse' / '0').mkdir(parents=True)
            (run_colmap / 'images').mkdir(parents=True)
            (run_colmap / 'sparse' / '0' / 'cameras.txt').write_text('', encoding='utf-8')
            (run_colmap / 'sparse' / '0' / 'images.txt').write_text('', encoding='utf-8')
            self.assertEqual(find_colmap_dir((traj_path,)), run_colmap.resolve())

    def test_equirectangular_high_elevation_sun_and_seam_crossing_recovers_north(self):
        # 2026-08-21 13:55 UTC in Fortaleza -> sun alt ~71 deg (1/cos(lat) > 3x horizontal stretch in equirect).
        lat, lon, epoch, true_yaw = -3.7714, -38.6273, 1787320500.0, 77.0
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, az_deg = solar_position(lat, lon, epoch)
        self.assertGreater(alt_deg, 65.0)
        loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
        alt_rad = math.radians(alt_deg)
        sun_local = np.array([
            math.cos(alt_rad) * math.cos(loc_math),
            math.cos(alt_rad) * math.sin(loc_math),
            math.sin(alt_rad),
        ])
        # Upright equirectangular rig: camera -Y points up (+Z world), matching real Insta360 mount.
        R_lc = Rotation.from_euler('xyz', [-90.0, -25.0, -90.0], degrees=True).as_matrix()
        T_lc = np.eye(4)
        T_lc[:3, :3] = R_lc
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cam0 = root / 'images' / 'cam0'
            cam0.mkdir(parents=True)
            w, h = 960, 480
            (root / 'rig_calibration_photometric.json').write_text(json.dumps({
                'lens_model': 'EQUIRECTANGULAR',
                'valid': True,
                'dt_sync_seconds': 0.0,
                'T_lidar_to_cam0_rigid_4x4': T_lc.tolist(),
                'cam0_front_intrinsics': {'camera_model': 'EQUIRECTANGULAR', 'width': w, 'height': h},
            }), encoding='utf-8')
            traj_rows = []
            ts_map = {}
            for i in range(12):
                # Rotate across all 360 deg so the sun crosses the lon = +/-180 deg equirectangular seam.
                R_wb = Rotation.from_euler('z', i * 30.0, degrees=True).as_matrix()
                q = Rotation.from_matrix(R_wb).as_quat()
                traj_rows.append([100.0 + i, float(i), 0.0, 0.0, q[0], q[1], q[2], q[3]])
                R_cw = (R_wb @ R_lc).T
                name = f'cam0/frame_{i + 1:06d}.png'
                ts_map[name] = float(i)
                _render_equirect_sky_frame(root / 'images' / name, w, h, R_cw @ sun_local)
            (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')
            traj_path = root / 'slam_raw_1' / 'result' / 'track.txt'
            traj_path.parent.mkdir(parents=True)
            np.savetxt(traj_path, traj_rows)

            gps = _gps_info_at(lat, lon, [epoch, epoch + 11.0], crs=crs)
            res = solve_solar_north(gps, conv, search_paths=(traj_path,), trajectory_path=traj_path,
                                    dt_sync=0.0, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'], msg=res)
            self.assertTrue(res['strong'], msg=res)
            self.assertEqual(res['inliers'], 12)
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.5)

    def test_equirectangular_indoor_dark_ceiling_and_skylight_are_rejected(self):
        lat, lon, epoch = -3.7714, -38.6273, 1787320500.0
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        alt_deg, _ = solar_position(lat, lon, epoch)
        alt_rad = math.radians(alt_deg)
        R_lc = Rotation.from_euler('x', -90.0, degrees=True).as_matrix()
        T_lc = np.eye(4)
        T_lc[:3, :3] = R_lc
        w, h = 960, 480

        # 1) Dark indoor ceiling with bright bulb at matching solar elevation -> rejected by halo gate.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'images' / 'cam0').mkdir(parents=True)
            (root / 'rig_calibration.json').write_text(json.dumps({
                'lens_model': 'EQUIRECTANGULAR',
                'T_lidar_to_cam0_rigid_4x4': T_lc.tolist(),
                'cam0_front_intrinsics': {'camera_model': 'EQUIRECTANGULAR', 'width': w, 'height': h},
            }), encoding='utf-8')
            sun_local = np.array([math.cos(alt_rad), 0.0, math.sin(alt_rad)])
            traj_rows, ts_map = [], {}
            for i in range(10):
                R_wb = Rotation.from_euler('z', i * 36.0, degrees=True).as_matrix()
                q = Rotation.from_matrix(R_wb).as_quat()
                traj_rows.append([100.0 + i, float(i), 0.0, 0.0, q[0], q[1], q[2], q[3]])
                name = f'cam0/frame_{i + 1:06d}.png'
                ts_map[name] = float(i)
                _render_equirect_sky_frame(
                    root / 'images' / name, w, h, (R_wb @ R_lc).T @ sun_local,
                    sky_bgr=(50, 50, 50), halo_bgr=(80, 80, 80),
                )
            (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')
            traj_path = root / 'slam_raw_1' / 'result' / 'track.txt'
            traj_path.parent.mkdir(parents=True)
            np.savetxt(traj_path, traj_rows)
            gps = _gps_info_at(lat, lon, [epoch, epoch + 9.0], crs=crs)
            res = solve_solar_north(gps, conv, search_paths=(traj_path,), trajectory_path=traj_path,
                                    dt_sync=0.0, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertFalse(res['qualifies'])
            self.assertEqual(res['inliers'], 0)

        # 2) Bright indoor ceiling with finite-distance skylight -> rejected by parallax.
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'images' / 'cam0').mkdir(parents=True)
            (root / 'rig_calibration.json').write_text(json.dumps({
                'lens_model': 'EQUIRECTANGULAR',
                'T_lidar_to_cam0_rigid_4x4': T_lc.tolist(),
                'cam0_front_intrinsics': {'camera_model': 'EQUIRECTANGULAR', 'width': w, 'height': h},
            }), encoding='utf-8')
            light_z = 3.0
            light_pos = np.array([light_z / math.tan(alt_rad), 0.0, light_z])
            traj_rows, ts_map = [], {}
            for i in range(14):
                cam_pos = np.array([0.0, -0.4 + 0.8 * i / 13.0, 0.0])
                vec = light_pos - cam_pos
                vec /= np.linalg.norm(vec)
                R_wb = Rotation.from_euler('z', i * 20.0, degrees=True).as_matrix()
                q = Rotation.from_matrix(R_wb).as_quat()
                traj_rows.append([100.0 + i, cam_pos[0], cam_pos[1], cam_pos[2], q[0], q[1], q[2], q[3]])
                name = f'cam0/frame_{i + 1:06d}.png'
                ts_map[name] = float(i)
                _render_equirect_sky_frame(
                    root / 'images' / name, w, h, (R_wb @ R_lc).T @ vec,
                    sky_bgr=(205, 205, 205), halo_bgr=(235, 235, 235),
                )
            (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')
            traj_path = root / 'slam_raw_1' / 'result' / 'track.txt'
            traj_path.parent.mkdir(parents=True)
            np.savetxt(traj_path, traj_rows)
            gps = _gps_info_at(lat, lon, [epoch, epoch + 13.0], crs=crs)
            res = solve_solar_north(gps, conv, search_paths=(traj_path,), trajectory_path=traj_path,
                                    dt_sync=0.0, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertFalse(res['qualifies'])
            self.assertEqual(res['reason'], 'finite_distance_light_source')

    def test_recorded_dt_sync_prefers_valid_photometric_and_skips_invalid(self):
        from raven_app.georeference import recorded_dt_sync
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            traj = root / 'slam_refined' / 'result' / 'track.txt'
            (root / 'rig_calibration.json').write_text(json.dumps({'dt_sync_seconds': -7.65}), encoding='utf-8')
            (root / 'rig_calibration_photometric.json').write_text(
                json.dumps({'dt_sync_seconds': -7.74, 'valid': True}), encoding='utf-8'
            )
            self.assertEqual(recorded_dt_sync((traj,), root / 'scan.insv'), -7.74)
            (root / 'rig_calibration_photometric.json').write_text(
                json.dumps({'dt_sync_seconds': -9.99, 'valid': False}), encoding='utf-8'
            )
            self.assertEqual(recorded_dt_sync((traj,), root / 'scan.insv'), -7.65)

    def test_review_required_message_matches_actual_review_reason(self):
        from raven_app.views.georeference_view import review_required_message

        conflict_msg = review_required_message({
            'north': {
                'method': 'gps_time_matched_rigid_fit',
                'evidence': {'sensor_conflict': {'solar_yaw_deg': 185.0, 'gps_yaw_deg': 120.0, 'diff_deg': 65.0}},
            }
        })
        self.assertIn('disagree', conflict_msg)
        self.assertIn('65.0', conflict_msg)
        self.assertNotIn('aerial imagery alone', conflict_msg)
        self.assertNotIn('north_alignment_review.jpg', conflict_msg)

        solar_offline_msg = review_required_message({'north': {'method': 'solar_compass', 'evidence': {}}})
        self.assertIn('solar compass', solar_offline_msg)
        self.assertNotIn('aerial imagery alone', solar_offline_msg)
        self.assertNotIn('north_alignment_review.jpg', solar_offline_msg)

        short_gps_msg = review_required_message({'north': {'method': 'gps_time_matched_rigid_fit', 'evidence': {}}})
        self.assertIn('GPS track is too short', short_gps_msg)
        self.assertNotIn('aerial imagery alone', short_gps_msg)

        unsynced_gps_msg = review_required_message({'north': {'method': 'gps_assumed_sync_rigid_fit', 'evidence': {}}})
        self.assertIn('time sync', unsynced_gps_msg)
        self.assertNotIn('aerial imagery alone', unsynced_gps_msg)

        street_msg = review_required_message({'north': {'method': 'street_bearing_and_wall_axis', 'evidence': {}}})
        self.assertIn('90°', street_msg)
        self.assertNotIn('aerial imagery alone', street_msg)

        imagery_msg = review_required_message({
            'north': {'method': 'aerial_imagery_registration', 'review_image': 'north_alignment_review.jpg', 'evidence': {}}
        })
        self.assertIn('aerial imagery alone', imagery_msg)
        self.assertIn('north_alignment_review.jpg', imagery_msg)

        imagery_no_img = review_required_message({'north': {'method': 'aerial_imagery_registration', 'evidence': {}}})
        self.assertIn('aerial imagery alone', imagery_no_img)
        self.assertNotIn('north_alignment_review.jpg', imagery_no_img)

        none_conflict = review_required_message({
            'north': {'evidence': {'sensor_conflict': {'solar_yaw_deg': None, 'gps_yaw_deg': None, 'diff_deg': None}}}
        })
        self.assertIn('disagree', none_conflict)

        fallback_method = review_required_message({
            'transform': {'yaw_estimation_method': 'solar_compass'}
        })
        self.assertIn('solar compass', fallback_method)

    def test_colmap_zero_param_equirectangular_2k_jpeg_low_sun_and_mixed_pinhole(self):
        # Low sun near horizon (~6.5 deg) in 2K (1920x960) JPEG equirectangular COLMAP with 0 params
        # and a secondary PINHOLE camera (wide camera should be preferred, and capture_origin_s=None allowed).
        lat, lon, true_yaw = -3.7714, -38.6273, 145.0
        crs = 'EPSG:31984'
        conv = grid_convergence_deg(lat, lon, crs)
        base_ep = 1792691460.0
        ep = base_ep
        for dt in range(0, 14400, 60):
            a, _ = solar_position(lat, lon, base_ep + dt)
            if 5.8 <= a <= 6.8:
                ep = base_ep + dt
                break
        alt_deg, az_deg = solar_position(lat, lon, ep)
        self.assertTrue(5.8 <= alt_deg <= 6.8)
        loc_math = math.radians(90.0 - (az_deg - conv) - true_yaw)
        alt_rad = math.radians(alt_deg)
        sun_local = np.array([
            math.cos(alt_rad) * math.cos(loc_math),
            math.cos(alt_rad) * math.sin(loc_math),
            math.sin(alt_rad),
        ])
        R_lc = Rotation.from_euler('x', -90.0, degrees=True).as_matrix()
        w, h = 1920, 960
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cdir = root / 'colmap_3dgs'
            (cdir / 'sparse' / '0').mkdir(parents=True)
            (cdir / 'images' / 'cam0').mkdir(parents=True)
            (cdir / 'images' / 'cam2').mkdir(parents=True)
            # Camera 1: 0-parameter EQUIRECTANGULAR; Camera 2: narrow PINHOLE decoy.
            (cdir / 'sparse' / '0' / 'cameras.txt').write_text(
                f'1 EQUIRECTANGULAR {w} {h}\n'
                '2 PINHOLE 640 480 500 500 320 240\n',
                encoding='utf-8',
            )
            lines = ['# images\n']
            ts_map = {}
            for i in range(12):
                R_wb = Rotation.from_euler('z', i * 30.0, degrees=True).as_matrix()
                R_cw = (R_wb @ R_lc).T
                name = f'cam0/frame_{i + 1:06d}.jpg'
                ts_map[name] = float(i)
                _render_equirect_sky_frame(
                    cdir / 'images' / name, w, h, R_cw @ sun_local, core_deg=1.5, halo_deg=5.0,
                )
                q = Rotation.from_matrix(R_cw).as_quat()
                lines.append(f'{i + 1} {q[3]} {q[0]} {q[1]} {q[2]} {float(i)} 0 0 1 {name}\n\n')
            # Add dark pinhole decoy image to confirm wide camera selection keeps equirectangular frames.
            _render_sky_frame(cdir / 'images' / 'cam2' / 'frame_000001.jpg', 640, 480, None, sky_bgr=(20, 20, 20))
            lines.append('100 1 0 0 0 0 0 0 2 cam2/frame_000001.jpg\n\n')
            (cdir / 'sparse' / '0' / 'images.txt').write_text(''.join(lines), encoding='utf-8')
            (root / 'images').mkdir(parents=True)
            (root / 'images' / 'frames.json').write_text(json.dumps({'timestamps': ts_map}), encoding='utf-8')

            gps = _gps_info_at(lat, lon, [ep, ep + 11.0], crs=crs)
            res = solve_solar_north(gps, conv, colmap_dir=cdir, capture_origin_s=None, log=lambda _: None)
            self.assertIsNotNone(res)
            self.assertTrue(res['qualifies'], msg=res)
            self.assertTrue(res['strong'], msg=res)
            self.assertEqual(res['inliers'], 12)
            self.assertLess(angle_diff_deg(res['yaw_grid_deg'], true_yaw), 1.0)

    def test_recorded_dt_sync_matches_equirect_mp4_stem_and_rejects_trimmed_folder(self):
        from raven_app.georeference import recorded_dt_sync
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            traj = root / 'slam_raw_1' / 'result' / 'track.txt'
            # Stitched equirectangular .mp4 shares time base with same-stem .insv.
            (root / 'workflow_config.json').write_text(json.dumps({
                'schema_version': 1,
                'runs': [{
                    'paths': {'insv': {'path': r'D:\data\VID_20260821_105436_00_269.mp4'}},
                    'resolved_dt_sync_seconds': -7.74,
                }],
            }), encoding='utf-8')
            self.assertEqual(recorded_dt_sync((traj,), root / 'VID_20260821_105436_00_269.insv'), -7.74)

            # Trimmed run's rig_calibration.json must NOT be reused for the untrimmed .insv.
            (root / 'workflow_config.json').write_text(json.dumps({
                'schema_version': 1,
                'runs': [{
                    'paths': {'insv': {'path': r'D:\data\VID_20260617_trimmed.insv'}},
                    'resolved_dt_sync_seconds': -0.80,
                }],
            }), encoding='utf-8')
            (root / 'rig_calibration.json').write_text(json.dumps({'dt_sync_seconds': -0.80}), encoding='utf-8')
            self.assertIsNone(recorded_dt_sync((traj,), root / 'VID_20260617.insv'))
            # Passing insv=None (caller accounts for capture_origin_s) still reads the folder's sync.
            self.assertEqual(recorded_dt_sync((traj,), None), -0.80)


if __name__ == '__main__':
    unittest.main()
