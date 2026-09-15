"""Tests for high-resolution 3D view export and resolution selection dialog."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPageSize, QPageLayout, QPdfWriter
from PyQt6.QtWidgets import QApplication

from raven_app.cloud_io import CloudData
from raven_app.cloud_view import CloudView
from raven_app.export_view_dialog import ExportViewDialog, PRESETS, FORMATS

app = QApplication.instance() or QApplication(["test"])


class ExportViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.view = CloudView()
        cls.view.resize(800, 600)
        cls.view.show()

        # Seed sample point cloud
        N = 500
        pts = np.random.uniform(-5.0, 5.0, (N, 3)).astype(np.float64)
        colors = np.ones((N, 3), dtype=np.float32)
        cloud = CloudData(path=Path("test_cloud.ply"), points=pts, colors=colors, original_count=N)
        cls.view.set_cloud(0, cloud)
        cls.view.set_color_mode(0, "height")
        app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.view.deleteLater()

    def test_render_to_image_dimensions(self):
        # Full HD
        img_fhd = self.view.render_to_image(1920, 1080, include_overlays=True)
        self.assertIsInstance(img_fhd, QImage)
        self.assertFalse(img_fhd.isNull())
        self.assertEqual(img_fhd.width(), 1920)
        self.assertEqual(img_fhd.height(), 1080)

        # 4K UHD
        img_4k = self.view.render_to_image(3840, 2160, include_overlays=True)
        self.assertIsInstance(img_4k, QImage)
        self.assertFalse(img_4k.isNull())
        self.assertEqual(img_4k.width(), 3840)
        self.assertEqual(img_4k.height(), 2160)

    def test_render_to_image_without_overlays(self):
        img_clean = self.view.render_to_image(
            1280, 720,
            include_overlays=False,
            point_size_multiplier=1.5
        )
        self.assertFalse(img_clean.isNull())
        self.assertEqual(img_clean.width(), 1280)
        self.assertEqual(img_clean.height(), 720)

    def test_pdf_export_at_resolution(self):
        img = self.view.render_to_image(1920, 1080, include_overlays=True)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "output_view.pdf"
            writer = QPdfWriter(str(pdf_path))
            writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
            writer.setPageOrientation(QPageLayout.Orientation.Landscape)
            writer.setResolution(300)

            painter = QPainter(writer)
            page_rect = writer.pageLayout().paintRectPixels(writer.resolution())
            scaled = img.scaled(
                page_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x = (page_rect.width() - scaled.width()) // 2
            y = (page_rect.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
            painter.end()

            self.assertTrue(pdf_path.is_file())
            self.assertGreater(pdf_path.stat().st_size, 10_000)

    def test_export_dialog_presets_and_ratio(self):
        diag = ExportViewDialog(self.view)
        # Default should be 4K UHD (3840 x 2160)
        self.assertEqual(diag.spin_w.value(), 3840)
        self.assertEqual(diag.spin_h.value(), 2160)
        self.assertTrue(diag.cb_lock_aspect.isChecked())
        self.assertIn("3840 × 2160", diag.lbl_stats.text())

        # Select Full HD preset
        fhd_idx = next(i for i, p in enumerate(PRESETS) if p[0] == "fhd")
        diag.preset_combo.setCurrentIndex(fhd_idx)
        self.assertEqual(diag.spin_w.value(), 1920)
        self.assertEqual(diag.spin_h.value(), 1080)

        # Select PDF format
        pdf_idx = next(i for i, f in enumerate(FORMATS) if f[0] == "pdf")
        diag.fmt_combo.setCurrentIndex(pdf_idx)
        self.assertFalse(diag.pdf_container.isHidden())

        # Switch back to PNG
        png_idx = next(i for i, f in enumerate(FORMATS) if f[0] == "png")
        diag.fmt_combo.setCurrentIndex(png_idx)
        self.assertTrue(diag.pdf_container.isHidden())

        diag.deleteLater()


if __name__ == "__main__":
    unittest.main()
