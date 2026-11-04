import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from raven_app.georeference import apply_transform_adjustment, calculate_insv_gps, fit_georeference
from raven_app.insv_gps import read_gps_sidecar, write_gps_sidecar
from raven_app.true_north import (
    EdgeMap, angle_diff_deg, edge_orientation, grid_convergence_deg, load_timed_trajectory,
    match_fixes_to_trajectory, rotation_2d, solve_north, true_bearing_of_axis,
)

YAW = 118.0
ORIGIN = np.array([541000.0, 9583000.0])
CRS_INFO = {'horizontal_crs': 'EPSG:31984', 'crs_name': 'SIRGAS 2000 / UTM zone 24S'}
GSD = 0.3
# Lots of different sizes, so the plan has one way to fit and no symmetry.
BLOCKS = ((-18, -11, 9, 7), (-7, -14, 13, 6), (8, -9, 5, 12), (-16, 2, 7, 5), (-5, 5, 16, 8), (14, 6, 6, 3))


def block_outline(step=0.1):
    edges = []
    for x, y, w, h in BLOCKS:
        xs, ys = np.arange(x, x + w, step), np.arange(y, y + h, step)
        edges += [np.c_[xs, np.full_like(xs, y)], np.c_[xs, np.full_like(xs, y + h)],
                  np.c_[np.full_like(ys, x), ys], np.c_[np.full_like(ys, x + w), ys]]
    return np.vstack(edges)


def scan_points():
    plan = block_outline()
    return np.vstack([np.c_[plan, np.full(len(plan), z)] for z in (0.0, 1.5, 2.5)])


def scene_edges(yaw=YAW, origin=ORIGIN, half=60.0):
    """A north-up picture of the blocks, as the imagery would show their roofs."""
    size = int(2 * half / GSD)
    east_min, north_max = origin[0] - half, origin[1] + half
    east, north = np.meshgrid(east_min + (np.arange(size) + .5) * GSD, north_max - (np.arange(size) + .5) * GSD)
    local = (np.c_[east.ravel(), north.ravel()] - origin) @ rotation_2d(yaw)
    gray = np.zeros(east.shape)
    for x, y, w, h in BLOCKS:
        inside = (local[:, 0] >= x) & (local[:, 0] <= x + w) & (local[:, 1] >= y) & (local[:, 1] <= y + h)
        gray[inside.reshape(gray.shape)] = 200.0
    return EdgeMap(edge_orientation(gray), east_min, north_max, GSD)


def gps_info(en, epochs=None):
    en = np.atleast_2d(np.asarray(en, dtype=np.float64))
    epochs = np.arange(len(en), dtype=np.float64) + 1000.0 if epochs is None else epochs
    return {'utm_coords': np.c_[en, np.zeros(len(en))], 'crs_info': CRS_INFO,
            'records': [{'timestamp_epoch': float(t)} for t in epochs],
            'mean_wgs84': {'latitude_deg': -3.77, 'longitude_deg': -38.63, 'altitude_m': 0.0}}


def walk_case(yaw=YAW, count=12, step_m=4.0, noise=None):
    """A walk along local X sampled once a second, with its fixes on the grid."""
    times = np.arange(count, dtype=np.float64)
    local = np.c_[times * step_m, np.zeros(count)]
    en = local @ rotation_2d(yaw).T + ORIGIN
    if noise is not None:
        en = en + noise
    trajectory = np.c_[times + 500.0, local, np.zeros(count)]
    return gps_info(en), trajectory


class AngleConventionTests(unittest.TestCase):
    def test_true_bearing_adds_convergence_to_the_grid_bearing(self):
        self.assertAlmostEqual(true_bearing_of_axis(90.0, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(true_bearing_of_axis(0.0, 0.0, 0.0), 90.0)
        self.assertAlmostEqual(true_bearing_of_axis(90.0, -30.0, 0.0), 30.0)
        self.assertAlmostEqual(true_bearing_of_axis(90.0, 0.0, 1.5), 1.5)

    def test_convergence_changes_sign_across_the_central_meridian(self):
        # EPSG:31984 has its central meridian at 39W.
        self.assertLess(abs(grid_convergence_deg(0.0, -39.0, 'EPSG:31984')), 1e-6)
        west = grid_convergence_deg(-3.77, -40.0, 'EPSG:31984')
        east = grid_convergence_deg(-3.77, -38.0, 'EPSG:31984')
        self.assertAlmostEqual(west, -east, places=4)
        self.assertLess(east, 0.0)  # grid north sits west of true north here, in the south, east of the meridian

    def test_fit_reports_every_form_of_the_heading(self):
        result = fit_georeference(gps_info([ORIGIN]), manual_yaw_deg=YAW)
        self.assertEqual(result['north_status'], 'resolved')
        self.assertAlmostEqual(result['yaw_heading_deg'], YAW)
        self.assertAlmostEqual(result['local_y_true_bearing_deg'],
                               true_bearing_of_axis(90.0, YAW, result['grid_convergence_deg']))
        self.assertAlmostEqual(result['yaw_true_deg'], YAW - result['grid_convergence_deg'])


class TimeMatchingTests(unittest.TestCase):
    def test_trimmed_capture_and_sync_shift_the_match(self):
        gps, trajectory = walk_case()
        matches = match_fixes_to_trajectory(gps, trajectory, dt_sync=2.0, capture_origin_s=1.0)
        first = next(m for m in matches if m['t'] >= 0)
        # The fix at capture second 3 is at video 2, lidar 0: the start of the walk.
        self.assertAlmostEqual(first['t'], 0.0)
        self.assertAlmostEqual(first['local_xy'][0], 0.0)
        self.assertEqual(first['utm_en'], gps['utm_coords'][3, :2].tolist())

    def test_fixes_outside_the_trajectory_are_dropped(self):
        gps, trajectory = walk_case()
        self.assertLess(len(match_fixes_to_trajectory(gps, trajectory[:3], dt_sync=0.0)), 12)

    def test_records_without_time_match_nothing(self):
        gps = gps_info([ORIGIN, ORIGIN + 1.0])
        gps['records'] = [{}, {}]
        self.assertEqual(match_fixes_to_trajectory(gps, walk_case()[1], dt_sync=0.0), [])

    def test_timed_trajectory_reader(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'trajectory.txt'
            path.write_text('1.0 2 3 4 0 0 0 1\n2.0 5 6 7 0 0 0 1\n', encoding='ascii')
            self.assertEqual(load_timed_trajectory(path).tolist(), [[1, 2, 3, 4], [2, 5, 6, 7]])
            self.assertIsNone(load_timed_trajectory(Path(d) / 'missing.txt'))


class SolveNorthTests(unittest.TestCase):
    def solve(self, gps, **kwargs):
        return solve_north(gps, allow_network=False, log=lambda _: None, **kwargs)

    def test_nothing_to_go_on_says_unresolved(self):
        north = self.solve(gps_info([ORIGIN, ORIGIN + [1.0, 1.0]]))
        self.assertEqual(north['status'], 'unresolved')
        self.assertTrue(np.allclose(north['translation_en'], ORIGIN + 0.5))
        json.dumps(north, allow_nan=False)

    def test_short_gps_track_cannot_set_heading(self):
        gps, trajectory = walk_case(count=4, step_m=1.5)
        north = self.solve(gps, trajectory=trajectory, dt_sync=0.0)
        self.assertEqual(north['status'], 'unresolved')

    def test_long_synced_gps_track_sets_heading(self):
        gps, trajectory = walk_case()
        north = self.solve(gps, trajectory=trajectory, dt_sync=0.0)
        self.assertEqual((north['status'], north['method']), ('resolved', 'gps_time_matched_rigid_fit'))
        self.assertLess(angle_diff_deg(north['yaw_grid_deg'], YAW), 0.01)
        self.assertTrue(np.allclose(rotation_2d(north['yaw_grid_deg']) @ [0, 0] + north['translation_en'], ORIGIN))

    def test_gps_track_without_a_time_sync_asks_for_review(self):
        gps, trajectory = walk_case()
        north = self.solve(gps, trajectory=trajectory)
        self.assertEqual((north['status'], north['method']), ('needs_review', 'gps_assumed_sync_rigid_fit'))
        self.assertFalse(north['evidence']['time_synchronized'])

    def test_manual_heading_wins_over_everything(self):
        gps, trajectory = walk_case()
        north = self.solve(gps, trajectory=trajectory, dt_sync=0.0, manual_yaw_deg=45.0)
        self.assertEqual((north['status'], north['yaw_grid_deg']), ('resolved', 45.0))

    def test_imagery_alone_recovers_the_pose_but_asks_for_review(self):
        anchor = ORIGIN + [2.0, -1.5]
        north = self.solve(gps_info([anchor]), points=scan_points(), edges=scene_edges(), local_xy=np.zeros((1, 2)))
        self.assertEqual((north['status'], north['method']), ('needs_review', 'aerial_imagery_registration'))
        self.assertLess(angle_diff_deg(north['yaw_grid_deg'], YAW), 1.0)
        self.assertLess(np.linalg.norm(np.subtract(north['translation_en'], ORIGIN)), 1.0)
        self.assertFalse(north['evidence']['confirmed_by_gps'])

    def test_imagery_confirmed_by_gps_is_resolved(self):
        gps, trajectory = walk_case(yaw=YAW + 5.0, count=8)
        north = self.solve(gps, points=scan_points(), edges=scene_edges(), trajectory=trajectory, dt_sync=0.0)
        self.assertEqual(north['status'], 'resolved')

    def test_street_bearing_picks_the_axis_nearest_the_imagery(self):
        convergence = grid_convergence_deg(-3.77, -38.63, 'EPSG:31984')
        street = true_bearing_of_axis(0.0, YAW, convergence)  # the blocks' long sides
        anchor = ORIGIN + [1.0, 1.0]
        north = self.solve(gps_info([anchor]), points=scan_points(), edges=scene_edges(), local_xy=np.zeros((1, 2)),
                           street_bearing_deg=street + 1.0)
        self.assertEqual((north['status'], north['method']), ('resolved', 'street_bearing_and_wall_axis'))
        self.assertLess(angle_diff_deg(north['yaw_grid_deg'], YAW - 1.0), 0.5)

    def test_review_image_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            edges = scene_edges()
            edges.photo = np.zeros((*edges.image.shape, 3), dtype=np.uint8)
            north = self.solve(gps_info([ORIGIN]), points=scan_points(), edges=edges, local_xy=np.zeros((1, 2)),
                               output_dir=d)
            self.assertTrue(Path(north['review_image']).is_file())


class AdjustmentTests(unittest.TestCase):
    def test_changing_the_heading_answers_the_review(self):
        north = {'status': 'needs_review', 'method': 'aerial_imagery_registration', 'yaw_grid_deg': 30.0,
                 'yaw_sigma_deg': 2.0, 'position_sigma_m': 3.0, 'translation_en': ORIGIN.tolist(),
                 'grid_convergence_deg': -0.02}
        base = fit_georeference(gps_info([ORIGIN]), north=north)
        shifted = apply_transform_adjustment(base, delta_easting_m=2.0)
        self.assertEqual(shifted['north_status'], 'needs_review')
        turned = apply_transform_adjustment(base, delta_yaw_deg=3.0)
        self.assertEqual((turned['north_status'], turned['yaw_estimation_method']),
                         ('resolved', 'interactive_user_adjustment'))
        self.assertAlmostEqual(turned['local_y_true_bearing_deg'], true_bearing_of_axis(90.0, 33.0, -0.02))


class SidecarTests(unittest.TestCase):
    FIXES = [{'timestamp_epoch': 1000.0 + i, 'timestamp_utc': 'x', 'latitude_deg': -3.77 - i * 1e-5,
              'longitude_deg': -38.63, 'altitude_m': 10.0} for i in range(3)]

    def test_trimming_a_trim_accumulates_the_origin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            first = root / 'a_trimmed.insv'
            Path(str(first) + '.gps.json').write_text(
                json.dumps({'fixes': self.FIXES, 'capture_origin_s': 2.5}), encoding='utf-8')
            second = root / 'a_trimmed_again.insv'
            write_gps_sidecar(first, second, 4.0)
            fixes, origin = read_gps_sidecar(second)
            self.assertEqual(fixes, self.FIXES)
            self.assertAlmostEqual(origin, 6.5)

    def test_a_video_without_gps_leaves_no_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'plain.mp4'
            source.write_bytes(b'not a capture')
            self.assertIsNone(write_gps_sidecar(source, root / 'out.mp4', 1.0))
            self.assertFalse((root / 'out.mp4.gps.json').exists())

    def test_trimmed_video_reads_gps_from_its_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d) / 'clip.insv'
            video.write_bytes(b'no trailer here')
            Path(str(video) + '.gps.json').write_text(
                json.dumps({'fixes': self.FIXES, 'capture_origin_s': 1.25}), encoding='utf-8')
            gps = calculate_insv_gps(video)
            self.assertEqual(gps['capture_origin_s'], 1.25)
            self.assertEqual(gps['total_fixes'], 3)


if __name__ == '__main__':
    unittest.main()
