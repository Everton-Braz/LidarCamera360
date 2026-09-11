"""Tests for 3D Viewer features: icons, clipping dialog, independent colors."""
import unittest
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from raven_app.view_icons import create_cube_icon, PRESETS
from raven_app.clipping_panel import ClippingBoxDialog
from raven_app.cloud_io import CloudData
from raven_app.cloud_view import CloudView

app = QApplication.instance() or QApplication(["test", "-platform", "offscreen"])


class ViewerFeaturesTests(unittest.TestCase):
    def test_cubemap_icons(self):
        for preset in PRESETS:
            icon = create_cube_icon(preset, size=24)
            self.assertIsInstance(icon, QIcon)
            self.assertFalse(icon.isNull())
            pix = icon.pixmap(24, 24)
            self.assertFalse(pix.isNull())
            self.assertEqual(pix.width(), 24)
            self.assertEqual(pix.height(), 24)

    def test_clipping_box_dialog(self):
        diag = ClippingBoxDialog()
        self.assertFalse(diag.switch_btn.isChecked())
        diag.switch_btn.setChecked(True)
        self.assertTrue(diag.switch_btn.isChecked())

        # Test set bounds from clouds
        pts = np.array([[-10.0, -20.0, -5.0], [10.0, 20.0, 15.0]], dtype=np.float64)
        colors = np.ones((2, 3), dtype=np.float32)
        cloud = CloudData(path=Path("dummy.pcd"), points=pts, colors=colors, original_count=2)
        diag.set_bounds_from_clouds([cloud])

        c_min, c_max, enabled = diag.get_clipping_bounds()
        self.assertTrue(enabled)
        self.assertLessEqual(c_min[0], -10.0)
        self.assertGreaterEqual(c_max[0], 10.0)

        # Test reset
        diag.reset_bounds()
        c_min2, c_max2, _ = diag.get_clipping_bounds()
        self.assertEqual(c_min2[0], diag._cloud_bounds['x'][0])
        self.assertEqual(c_max2[0], diag._cloud_bounds['x'][1])
        diag.deleteLater()

    def test_cloud_view_settings(self):
        view = CloudView()
        self.assertEqual(view.color_modes, ['rgb', 'rgb'])
        self.assertEqual(view.colormaps, ['turbo', 'viridis'])

        # Individual slot color modes
        view.set_color_mode(0, 'height')
        view.set_color_mode(1, 'intensity')
        self.assertEqual(view.color_modes[0], 'height')
        self.assertEqual(view.color_modes[1], 'intensity')

        # Individual slot colormaps
        view.set_colormap(0, 'turbo')
        view.set_colormap(1, 'viridis')
        self.assertEqual(view.colormaps[0], 'turbo')
        self.assertEqual(view.colormaps[1], 'viridis')

        # Point size
        view.set_point_size(3.5)
        self.assertAlmostEqual(view.point_size, 3.5)

        # Background color
        view.set_background_color('#ffffff')
        self.assertEqual(view.bg_color.name().lower(), '#ffffff')

        # Clipping bounds
        c_min = np.array([-5.0, -5.0, 0.0], dtype=np.float32)
        c_max = np.array([5.0, 5.0, 10.0], dtype=np.float32)
        view.set_clipping_bounds(c_min, c_max, enabled=True)
        self.assertTrue(view.clipping_enabled)
        np.testing.assert_allclose(view.clip_min, c_min)
        np.testing.assert_allclose(view.clip_max, c_max)
        view.deleteLater()


if __name__ == '__main__':
    unittest.main()
