import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QCloseEvent
    from PyQt6.QtWidgets import QApplication
    from raven_app.cloud_io import CloudData
    from raven_app.cloud_view import CloudView
    from raven_app.viewer_window import PointCloudViewerWindow
    QT_AVAILABLE = True
except ImportError:  # pragma: no cover
    QT_AVAILABLE = False


@unittest.skipUnless(QT_AVAILABLE, "PyQt6 is required for viewer integration tests")
class ViewerGpsIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        cls.app = QApplication.instance() or QApplication([])

    @staticmethod
    def cloud(name, offset):
        points = (np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float64) + offset)
        return CloudData(Path(name), points, np.ones((2, 3), dtype=np.float32), 2)

    def test_replacing_only_cloud_a_rebases_float64_coordinates(self):
        view = CloudView()
        try:
            view.set_cloud(0, self.cloud("local.pcd", np.array([0, 0, 0])))
            first_origin = view.origin.copy()
            replacement = self.cloud("utm.las", np.array([500000, 9600000, 100]))
            view.set_cloud(0, replacement)
            np.testing.assert_allclose(view.origin, replacement.points.mean(axis=0))
            self.assertFalse(np.allclose(first_origin, view.origin))
            self.assertEqual(view.clouds[0].points.dtype, np.float64)
        finally:
            view.deleteLater()

    def test_viewer_is_single_scene_with_edit_panel(self):
        viewer = PointCloudViewerWindow()
        try:
            self.assertFalse(hasattr(viewer, "_tabs"))
            self.assertTrue(hasattr(viewer, "edit_panel"))
            viewer.set_dataset_context(insv="capture.insv", trajectory="traj.txt", output="out", colmap_dir="colmap")
            self.assertEqual(viewer._dataset_context["trajectory"], "traj.txt")
        finally:
            viewer.close()
            viewer.deleteLater()

    def test_close_is_blocked_while_export_runs(self):
        viewer = PointCloudViewerWindow()
        try:
            viewer._edit_worker = type("Busy", (), {"isRunning": lambda self: True})()
            event = QCloseEvent()
            viewer.closeEvent(event)
            self.assertFalse(event.isAccepted())
        finally:
            viewer._edit_worker = None
            viewer.close()
            viewer.deleteLater()

    def test_loading_cloud_preserves_local_source_context(self):
        viewer = PointCloudViewerWindow()
        try:
            xyz = np.array([[0., 0., 0.], [1., 1., 1.]])
            local = CloudData(Path('local.pcd'), xyz, np.ones((2, 3)), 2)
            viewer._on_cloud_loaded(0, local)
            self.assertEqual(viewer._dataset_context['cloud'], 'local.pcd')
        finally:
            viewer.close()
            viewer.deleteLater()

    def test_main_workflow_passes_actual_slam_trajectory(self):
        from raven_app.gui import RavenMainWindow
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory = root / 'slam_out/result/Raven_3DMakerPro_Scan.txt'
            trajectory.parent.mkdir(parents=True)
            trajectory.touch()
            main = RavenMainWindow()
            try:
                main.workflow_view.out_input.setText(str(root))
                main.workflow_view.insv_input.setText('capture.insv')
                main._open_viewer()
                self.assertFalse(hasattr(main, 'georeference_view'))
                self.assertEqual(main.viewer_window._dataset_context['trajectory'], str(trajectory))
            finally:
                if main.viewer_window:
                    main.viewer_window.close()
                main.close()
                main.deleteLater()

    def test_save_and_orthophoto_workers_write_modified_cloud(self):
        from PyQt6.QtWidgets import QFileDialog
        from raven_app.viewer_window import PointCloudViewerWindow
        try:
            import pyproj
        except ImportError:
            self.skipTest("pyproj is required")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cloud = self.cloud("tiny.ply", np.array([0.0, 0.0, 0.0]))
            cloud.crs_wkt = pyproj.CRS.from_epsg(31984).to_wkt()
            viewer = PointCloudViewerWindow()
            try:
                viewer.renderer.set_cloud(0, cloud)
                viewer.edit_panel.yaw.setValue(10.0)
                viewer.edit_panel.apply_button.click()
                save_path = root / "modified.ply"
                ortho_path = root / "ortho.tif"
                with patch.object(QFileDialog, "getSaveFileName", side_effect=[(str(save_path), ""), (str(ortho_path), "")]):
                    viewer._save_modified_cloud(0)
                    while viewer._edit_worker is not None:
                        self.app.processEvents()
                    viewer._generate_orthophoto(0, 1.0)
                    while viewer._edit_worker is not None:
                        self.app.processEvents()
                self.assertTrue(save_path.is_file())
                self.assertTrue(ortho_path.is_file())
                self.assertTrue((root / "modified.ply.geo.json").is_file())
            finally:
                viewer.close()
                viewer.deleteLater()


if __name__ == "__main__":
    unittest.main()
