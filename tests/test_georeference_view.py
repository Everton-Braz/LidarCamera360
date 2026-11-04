import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtWidgets import QApplication

from raven_app.georeference import fit_georeference
from raven_app.views.georeference_view import GeoreferenceView, _export_adjustment

ORIGIN = [541000.0, 9583000.0]
CRS_INFO = {'horizontal_crs': 'EPSG:31984', 'crs_name': 'SIRGAS 2000 / UTM zone 24S'}


def transform(status='needs_review', yaw=30.0):
    gps = {'utm_coords': np.array([[*ORIGIN, 0.0]]), 'crs_info': CRS_INFO, 'records': [],
           'mean_wgs84': {'latitude_deg': -3.77, 'longitude_deg': -38.63, 'altitude_m': 0.0}}
    north = {'status': status, 'method': 'aerial_imagery_registration', 'yaw_grid_deg': yaw, 'yaw_sigma_deg': 2.0,
             'position_sigma_m': 3.0, 'translation_en': ORIGIN, 'grid_convergence_deg': -0.02}
    return fit_georeference(gps, north=north)


def report(tf):
    return {'status': 'review_required', 'transform': tf, 'gps_summary': {},
            'deliverables': {'cloud_export': {'bounds_utm': {
                'min_easting_m': ORIGIN[0], 'max_easting_m': ORIGIN[0] + 20.0,
                'min_northing_m': ORIGIN[1], 'max_northing_m': ORIGIN[1] + 10.0}}}}


class GeoreferenceViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = GeoreferenceView()
        self.addCleanup(self.view.deleteLater)

    def test_yaw_is_left_to_the_solver_until_someone_chooses_it(self):
        view = self.view
        captured = {}

        class Worker:
            def __init__(self, params, operation=None):
                captured.update(params)
                self.result = self.error = self.finished = type('S', (), {'connect': lambda *_: None})()

            def start(self):
                pass

            def isRunning(self):
                return False

        import raven_app.views.georeference_view as module
        original, module.GeoreferenceWorker = module.GeoreferenceWorker, Worker
        self.addCleanup(setattr, module, 'GeoreferenceWorker', original)
        with tempfile.TemporaryDirectory() as d:
            insv = Path(d) / 'a.insv'
            insv.write_bytes(b'x')
            view.insv_input.setText(str(insv))
            view._run()
            self.assertIsNone(captured['yaw_deg'])
            view._worker = None
            view.street_input.setText('16,5')
            view.yaw_spin.setValue(45.0)
            view._run()
            self.assertEqual(captured['yaw_deg'], 45.0)
            self.assertEqual(captured['street_bearing_deg'], 16.5)

    def test_a_result_absorbs_the_shifts_and_clears_the_choice(self):
        view = self.view
        view.shift_e_spin.setValue(5.0)
        view.yaw_spin.setValue(10.0)
        view._on_result(report(transform()))
        self.assertEqual(view.shift_e_spin.value(), 0.0)
        self.assertEqual(view.yaw_spin.value(), 30.0)
        self.assertFalse(view._yaw_chosen)

    def test_rotation_turns_about_the_scan_rather_than_its_origin(self):
        view = self.view
        tf = transform()
        view._on_result(report(tf))
        centre = view._cloud_local_corners.mean(axis=0)
        before = np.array(tf['rotation_matrix'])[:2, :2] @ centre + tf['translation_utm_m'][:2]
        view.yaw_spin.setValue(75.0)
        after_tf = view._adjusted(tf)
        after = np.array(after_tf['rotation_matrix'])[:2, :2] @ centre + after_tf['translation_utm_m'][:2]
        self.assertTrue(np.allclose(before, after))
        self.assertEqual(after_tf['north_status'], 'resolved')

    def test_elevation_control_updates_the_preview(self):
        view = self.view
        view._on_result(report(transform()))
        calls = []
        view.map_widget.set_footprint = lambda *a, **k: calls.append(k)
        view.shift_z_spin.setValue(3.0)
        self.assertEqual(len(calls), 1)

    def test_saving_a_correction_settles_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            result = _export_adjustment(transform('resolved'), d, '', '', False, False, True,
                                        report(transform('resolved')))
            self.assertEqual(result['status'], 'accepted')
            saved = json.loads((Path(d) / 'georeference_report.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['transform']['north_status'], 'resolved')


if __name__ == '__main__':
    unittest.main()
