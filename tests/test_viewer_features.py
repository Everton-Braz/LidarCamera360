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

    def test_viewport_clipping_box(self):
        view = CloudView()
        self.assertFalse(view.clipping_box_mode)
        view.set_clipping_box_mode(True)
        self.assertTrue(view.clipping_box_mode)
        self.assertTrue(view.clipping_enabled)

        # Invert toggle
        self.assertFalse(view.clip_invert)
        view.set_clip_invert(True)
        self.assertTrue(view.clip_invert)

        # Load dummy cloud and test reset_clipping_to_clouds and get_clipped_points
        pts = np.array([
            [-10.0, -10.0, 0.0],
            [0.0, 0.0, 5.0],
            [10.0, 10.0, 10.0],
        ], dtype=np.float64)
        colors = np.ones((3, 3), dtype=np.float32)
        cloud = CloudData(path=Path("test.pcd"), points=pts, colors=colors, original_count=3)
        view.set_cloud(0, cloud)

        view.reset_clipping_to_clouds()
        self.assertLessEqual(view.clip_min[0], -10.0)
        self.assertGreaterEqual(view.clip_max[0], 10.0)

        # Set tight clipping bounds around center point [0, 0, 5]
        view.set_clip_invert(False)
        view.set_clipping_bounds(np.array([-1.0, -1.0, 4.0], dtype=np.float32), np.array([1.0, 1.0, 6.0], dtype=np.float32), enabled=True)
        clipped_pts, _, _ = view.get_clipped_points(0)
        self.assertEqual(len(clipped_pts), 1)
        np.testing.assert_allclose(clipped_pts[0], [0.0, 0.0, 5.0])

        # Test inverted clip
        view.set_clip_invert(True)
        inv_pts, _, _ = view.get_clipped_points(0)
        self.assertEqual(len(inv_pts), 2)

        # Test persistent sliced view: hiding box keeps clipping enabled
        view.set_clipping_box_mode(True)
        self.assertTrue(view.clipping_box_mode)
        self.assertTrue(view.clipping_enabled)
        view.set_clipping_box_visible(False)
        self.assertFalse(view.clipping_box_visible)
        self.assertTrue(view.clipping_enabled)

        # Disabling clipping turns off both
        view.set_clipping_enabled(False)
        self.assertFalse(view.clipping_enabled)
        self.assertFalse(view.clipping_box_mode)
        view.deleteLater()

    def test_rotatable_obb_clipping(self):
        view = CloudView()
        pts = np.array([
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ], dtype=np.float64)
        colors = np.ones((2, 3), dtype=np.float32)
        cloud = CloudData(path=Path("test_obb.pcd"), points=pts, colors=colors, original_count=2)
        view.set_cloud(0, cloud)

        # Local box: extents [2.0, 8.0, 2.0] at center [0, 0, 0]
        # At yaw = 0, pt [5, 0, 0] is outside (5 > 2), pt [0, 5, 0] is inside (5 <= 8)
        view.set_clip_box(np.array([0.0, 0.0, 0.0]), np.array([2.0, 8.0, 2.0]), yaw_deg=0.0, enabled=True)
        clipped, _, _ = view.get_clipped_points(0)
        self.assertEqual(len(clipped), 1)
        np.testing.assert_allclose(clipped[0], [0.0, 5.0, 0.0])

        # Rotate box by 90 degrees: local X aligns with world Y, local Y aligns with -world X
        # Now pt [5, 0, 0] is along local Y (dist 5 <= 8 -> inside!)
        # and pt [0, 5, 0] is along local X (dist 5 > 2 -> outside!)
        view.set_clip_yaw(90.0)
        clipped, _, _ = view.get_clipped_points(0)
        self.assertEqual(len(clipped), 1)
        np.testing.assert_allclose(clipped[0], [5.0, 0.0, 0.0])

        view.deleteLater()

    def test_viewport_transform_tools(self):
        view = CloudView()
        self.assertIsNone(view.transform_mode)

        # Mode toggles
        view.set_transform_mode('translate')
        self.assertEqual(view.transform_mode, 'translate')

        view.set_transform_mode('rotate')
        self.assertEqual(view.transform_mode, 'rotate')

        view.set_transform_mode(None)
        self.assertIsNone(view.transform_mode)

        # Cloud adjustment preview and full apply
        pts = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ], dtype=np.float64)
        cloud = CloudData(path=Path("dummy.ply"), points=pts, colors=np.ones((2, 3), dtype=np.float32), original_count=2)
        view.set_cloud(0, cloud)

        # Apply 90 degree yaw rotation with preview
        view.apply_cloud_adjustment(0, 90.0, (5.0, 0.0, 0.0), is_preview=True)
        yaw, shift = view._adjustments[0]
        self.assertAlmostEqual(yaw, 90.0)
        np.testing.assert_allclose(shift, [5.0, 0.0, 0.0])

        # Full apply
        view.apply_cloud_adjustment(0, 90.0, (5.0, 0.0, 0.0), is_preview=False)
        # Center is (0, 0, 0). Point [1, 0, 0] rotated 90 deg yaw is [0, 1, 0] + shift [5, 0, 0] -> [5, 1, 0]
        np.testing.assert_allclose(view.clouds[0].points[0], [5.0, 1.0, 0.0], atol=1e-5)

        # Reset
        view.reset_cloud_adjustment(0)
        np.testing.assert_allclose(view.clouds[0].points[0], [1.0, 0.0, 0.0], atol=1e-5)
        view.deleteLater()


if __name__ == '__main__':
    unittest.main()
