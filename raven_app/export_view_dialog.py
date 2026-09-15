"""Dialog for exporting 3D point cloud snapshots and PDFs with selectable resolution."""
from __future__ import annotations

import math
from pathlib import Path
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import (
    QColor, QFont, QImage, QPainter, QPageLayout, QPageSize, QPdfWriter
)
from PyQt6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QVBoxLayout, QWidget, QFileDialog, QMessageBox
)
from qfluentwidgets import (
    BodyLabel, CaptionLabel, CardWidget, CheckBox, ComboBox,
    FluentIcon, PrimaryPushButton, PushButton, SpinBox, SubtitleLabel
)
from raven_app.i18n import tr


PRESETS = [
    ("4k", tr("4K UHD (3840 × 2160) — Recommended"), 3840, 2160),
    ("2k", tr("2K QHD (2560 × 1440)"), 2560, 1440),
    ("fhd", tr("Full HD 1080p (1920 × 1080)"), 1920, 1080),
    ("8k", tr("8K UHD (7680 × 4320) — Ultra High Res"), 7680, 4320),
    ("a4_land", tr("Print A4 Landscape (3508 × 2480 @ 300 DPI)"), 3508, 2480),
    ("a4_port", tr("Print A4 Portrait (2480 × 3508 @ 300 DPI)"), 2480, 3508),
    ("viewport", tr("Current Viewport ({w} × {h})"), None, None),
    ("2x", tr("2× Viewport ({w} × {h})"), 2.0, 2.0),
    ("3x", tr("3× Viewport ({w} × {h})"), 3.0, 3.0),
    ("4x", tr("4× Viewport ({w} × {h})"), 4.0, 4.0),
    ("custom", tr("Custom Resolution"), None, None),
]

FORMATS = [
    ("png", tr("PNG Image (*.png)"), ".png"),
    ("jpg", tr("JPEG Image (*.jpg *.jpeg)"), ".jpg"),
    ("pdf", tr("PDF Document (*.pdf)"), ".pdf"),
    ("tif", tr("TIFF Image (*.tif *.tiff)"), ".tif"),
    ("bmp", tr("Bitmap (*.bmp)"), ".bmp"),
]


class ExportViewDialog(QDialog):
    """Dialog enabling the user to configure and export high-resolution views as images or PDFs."""
    export_completed = pyqtSignal(str, str, int, int)  # file_path, format, width, height

    def __init__(self, renderer, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self.setWindowTitle(tr("Export High-Resolution View"))
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setFixedWidth(520)

        # Baseline viewport resolution
        self.vp_w = max(self.renderer.width(), 400)
        self.vp_h = max(self.renderer.height(), 300)
        self.aspect_ratio = self.vp_w / self.vp_h
        self._updating_dimensions = False

        self._build_ui()
        self._set_initial_values()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # Header Title
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_box.addWidget(SubtitleLabel(tr("Export High-Resolution View"), self))
        desc = CaptionLabel(
            tr("Save point cloud renders and PDFs with custom resolution, aspect ratio, and HUD overlays."),
            self
        )
        desc.setWordWrap(True)
        title_box.addWidget(desc)
        layout.addLayout(title_box)

        # -------------------------------------------------------------
        # Section 1: Resolution & Dimensions Card
        # -------------------------------------------------------------
        res_card = CardWidget(self)
        rc_layout = QVBoxLayout(res_card)
        rc_layout.setContentsMargins(14, 14, 14, 14)
        rc_layout.setSpacing(10)

        rc_layout.addWidget(BodyLabel(tr("Resolution Preset"), res_card))
        self.preset_combo = ComboBox(res_card)
        for key, label, w_val, h_val in PRESETS:
            if key == "viewport":
                display_label = label.format(w=self.vp_w, h=self.vp_h)
            elif key in ("2x", "3x", "4x"):
                mult = int(w_val)
                display_label = label.format(w=self.vp_w * mult, h=self.vp_h * mult)
            else:
                display_label = label
            self.preset_combo.addItem(display_label, userData=key)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        rc_layout.addWidget(self.preset_combo)

        dim_row = QHBoxLayout()
        dim_row.setSpacing(10)

        # Width
        w_box = QVBoxLayout()
        w_box.setSpacing(3)
        w_box.addWidget(CaptionLabel(tr("Width (px)"), res_card))
        self.spin_w = SpinBox(res_card)
        self.spin_w.setRange(100, 16384)
        self.spin_w.setSingleStep(100)
        self.spin_w.setValue(3840)
        self.spin_w.valueChanged.connect(self._on_width_changed)
        w_box.addWidget(self.spin_w)
        dim_row.addLayout(w_box)

        # Height
        h_box = QVBoxLayout()
        h_box.setSpacing(3)
        h_box.addWidget(CaptionLabel(tr("Height (px)"), res_card))
        self.spin_h = SpinBox(res_card)
        self.spin_h.setRange(100, 16384)
        self.spin_h.setSingleStep(100)
        self.spin_h.setValue(2160)
        self.spin_h.valueChanged.connect(self._on_height_changed)
        h_box.addWidget(self.spin_h)
        dim_row.addLayout(h_box)

        rc_layout.addLayout(dim_row)

        # Lock Aspect Ratio
        self.cb_lock_aspect = CheckBox(tr("Lock aspect ratio"), res_card)
        self.cb_lock_aspect.setChecked(True)
        self.cb_lock_aspect.stateChanged.connect(self._on_lock_aspect_changed)
        rc_layout.addWidget(self.cb_lock_aspect)

        # Informative Stats Label
        self.lbl_stats = CaptionLabel("", res_card)
        self.lbl_stats.setStyleSheet("color: #009faa; font-weight: 500;")
        rc_layout.addWidget(self.lbl_stats)

        layout.addWidget(res_card)

        # -------------------------------------------------------------
        # Section 2: Display & Overlay Options Card
        # -------------------------------------------------------------
        opt_card = CardWidget(self)
        oc_layout = QVBoxLayout(opt_card)
        oc_layout.setContentsMargins(14, 14, 14, 14)
        oc_layout.setSpacing(8)
        oc_layout.addWidget(BodyLabel(tr("Elements & Quality"), opt_card))

        self.cb_hud = CheckBox(tr("Include scale ruler and colorbar legends"), opt_card)
        self.cb_hud.setChecked(True)
        oc_layout.addWidget(self.cb_hud)

        self.cb_labels = CheckBox(tr("Include point cloud filenames and info"), opt_card)
        self.cb_labels.setChecked(True)
        oc_layout.addWidget(self.cb_labels)

        self.cb_measurements = CheckBox(tr("Include 3D measurements"), opt_card)
        self.cb_measurements.setChecked(True)
        oc_layout.addWidget(self.cb_measurements)

        self.cb_scale_points = CheckBox(tr("Scale point size with resolution (maintains point cloud density)"), opt_card)
        self.cb_scale_points.setChecked(True)
        oc_layout.addWidget(self.cb_scale_points)

        layout.addWidget(opt_card)

        # -------------------------------------------------------------
        # Section 3: File Format & PDF Options Card
        # -------------------------------------------------------------
        fmt_card = CardWidget(self)
        fc_layout = QVBoxLayout(fmt_card)
        fc_layout.setContentsMargins(14, 14, 14, 14)
        fc_layout.setSpacing(10)

        fc_layout.addWidget(BodyLabel(tr("Export Format"), fmt_card))
        self.fmt_combo = ComboBox(fmt_card)
        for key, label, _ in FORMATS:
            self.fmt_combo.addItem(label, userData=key)
        self.fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        fc_layout.addWidget(self.fmt_combo)

        # PDF Specific Controls Container
        self.pdf_container = QWidget(fmt_card)
        pdf_layout = QVBoxLayout(self.pdf_container)
        pdf_layout.setContentsMargins(0, 4, 0, 0)
        pdf_layout.setSpacing(8)

        pdf_row = QHBoxLayout()
        pdf_row.setSpacing(10)

        # Page Size
        ps_box = QVBoxLayout()
        ps_box.setSpacing(3)
        ps_box.addWidget(CaptionLabel(tr("Page Size"), self.pdf_container))
        self.pdf_pagesize = ComboBox(self.pdf_container)
        self.pdf_pagesize.addItem("A4", userData=QPageSize.PageSizeId.A4)
        self.pdf_pagesize.addItem("Letter", userData=QPageSize.PageSizeId.Letter)
        self.pdf_pagesize.addItem("A3", userData=QPageSize.PageSizeId.A3)
        ps_box.addWidget(self.pdf_pagesize)
        pdf_row.addLayout(ps_box)

        # Orientation
        ori_box = QVBoxLayout()
        ori_box.setSpacing(3)
        ori_box.addWidget(CaptionLabel(tr("Orientation"), self.pdf_container))
        self.pdf_orientation = ComboBox(self.pdf_container)
        self.pdf_orientation.addItem(tr("Auto (Matches aspect ratio)"), userData="auto")
        self.pdf_orientation.addItem(tr("Landscape"), userData="landscape")
        self.pdf_orientation.addItem(tr("Portrait"), userData="portrait")
        ori_box.addWidget(self.pdf_orientation)
        pdf_row.addLayout(ori_box)

        # DPI
        dpi_box = QVBoxLayout()
        dpi_box.setSpacing(3)
        dpi_box.addWidget(CaptionLabel(tr("Print DPI"), self.pdf_container))
        self.pdf_dpi = ComboBox(self.pdf_container)
        self.pdf_dpi.addItem("300 DPI (High Quality Print)", userData=300)
        self.pdf_dpi.addItem("150 DPI (Screen / Draft)", userData=150)
        self.pdf_dpi.addItem("600 DPI (Ultra Quality)", userData=600)
        dpi_box.addWidget(self.pdf_dpi)
        pdf_row.addLayout(dpi_box)

        pdf_layout.addLayout(pdf_row)
        fc_layout.addWidget(self.pdf_container)
        self.pdf_container.setVisible(False)

        layout.addWidget(fmt_card)

        # -------------------------------------------------------------
        # Action Buttons
        # -------------------------------------------------------------
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_clipboard = PushButton(FluentIcon.COPY, tr("Copy to Clipboard"), self)
        self.btn_clipboard.clicked.connect(self._on_copy_clipboard)
        btn_row.addWidget(self.btn_clipboard)

        btn_row.addStretch(1)

        self.btn_cancel = PushButton(tr("Cancel"), self)
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_cancel)

        self.btn_export = PrimaryPushButton(FluentIcon.SAVE, tr("Export / Save..."), self)
        self.btn_export.clicked.connect(self._on_export_clicked)
        btn_row.addWidget(self.btn_export)

        layout.addLayout(btn_row)

    def _set_initial_values(self):
        # Default: 4K UHD
        self.preset_combo.setCurrentIndex(0)
        self._update_stats_label()

    def _on_preset_changed(self, idx: int):
        if idx < 0 or idx >= len(PRESETS):
            return
        key, _, w_val, h_val = PRESETS[idx]
        if key == "custom":
            return

        self._updating_dimensions = True
        try:
            if key == "viewport":
                w = self.vp_w
                h = self.vp_h
            elif key in ("2x", "3x", "4x"):
                mult = int(w_val)
                w = self.vp_w * mult
                h = self.vp_h * mult
            else:
                w = int(w_val)
                h = int(h_val)
            self.spin_w.setValue(w)
            self.spin_h.setValue(h)
            if self.cb_lock_aspect.isChecked() and h > 0:
                self.aspect_ratio = w / h
        finally:
            self._updating_dimensions = False
        self._update_stats_label()

    def _on_width_changed(self, w: int):
        if self._updating_dimensions:
            return
        if self.cb_lock_aspect.isChecked() and self.aspect_ratio > 0:
            self._updating_dimensions = True
            new_h = max(100, round(w / self.aspect_ratio))
            self.spin_h.setValue(new_h)
            self._updating_dimensions = False
        self._mark_custom_preset()
        self._update_stats_label()

    def _on_height_changed(self, h: int):
        if self._updating_dimensions:
            return
        if self.cb_lock_aspect.isChecked() and self.aspect_ratio > 0:
            self._updating_dimensions = True
            new_w = max(100, round(h * self.aspect_ratio))
            self.spin_w.setValue(new_w)
            self._updating_dimensions = False
        self._mark_custom_preset()
        self._update_stats_label()

    def _on_lock_aspect_changed(self, state: int):
        if self.cb_lock_aspect.isChecked():
            h = self.spin_h.value()
            if h > 0:
                self.aspect_ratio = self.spin_w.value() / h

    def _mark_custom_preset(self):
        # If user manually changed spinboxes, change combo to "Custom"
        current_w = self.spin_w.value()
        current_h = self.spin_h.value()
        cur_preset_idx = self.preset_combo.currentIndex()
        if 0 <= cur_preset_idx < len(PRESETS):
            key, _, w_val, h_val = PRESETS[cur_preset_idx]
            if key not in ("custom",):
                expected_w, expected_h = None, None
                if key == "viewport":
                    expected_w, expected_h = self.vp_w, self.vp_h
                elif key in ("2x", "3x", "4x"):
                    mult = int(w_val)
                    expected_w, expected_h = self.vp_w * mult, self.vp_h * mult
                elif w_val and h_val:
                    expected_w, expected_h = int(w_val), int(h_val)
                if (expected_w, expected_h) != (current_w, current_h):
                    custom_idx = next(i for i, p in enumerate(PRESETS) if p[0] == "custom")
                    self.preset_combo.setCurrentIndex(custom_idx)

    def _update_stats_label(self):
        w = self.spin_w.value()
        h = self.spin_h.value()
        mp = (w * h) / 1_000_000.0
        # Physical print size at 300 DPI in cm (1 inch = 2.54 cm)
        w_cm = (w / 300.0) * 2.54
        h_cm = (h / 300.0) * 2.54
        self.lbl_stats.setText(
            f"{w} × {h} px • {mp:.2f} MP • {w_cm:.1f} × {h_cm:.1f} cm @ 300 DPI"
        )

    def _on_format_changed(self, idx: int):
        fmt_key = self.fmt_combo.itemData(idx)
        self.pdf_container.setVisible(fmt_key == "pdf")

    def _on_copy_clipboard(self):
        """Render high-resolution image and copy to system clipboard."""
        w = self.spin_w.value()
        h = self.spin_h.value()
        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
            img = self.renderer.render_to_image(
                width=w,
                height=h,
                include_overlays=self.cb_hud.isChecked(),
                point_size_multiplier=1.0 if self.cb_scale_points.isChecked() else (self.vp_h / max(h, 1)),
                show_labels=self.cb_labels.isChecked(),
                show_hud=self.cb_hud.isChecked(),
                show_measurements=self.cb_measurements.isChecked(),
            )
            QApplication.clipboard().setImage(img)
            self.export_completed.emit("clipboard", "image", w, h)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, tr("Export Error"), str(exc))
        finally:
            self.unsetCursor()

    def _on_export_clicked(self):
        w = self.spin_w.value()
        h = self.spin_h.value()
        fmt_idx = self.fmt_combo.currentIndex()
        fmt_key, _, default_ext = FORMATS[fmt_idx]

        filter_map = {
            "png": tr("PNG Image (*.png)"),
            "jpg": tr("JPEG Image (*.jpg *.jpeg)"),
            "pdf": tr("PDF Document (*.pdf)"),
            "tif": tr("TIFF Image (*.tif *.tiff)"),
            "bmp": tr("Bitmap (*.bmp)"),
        }
        all_filters = ";;".join([
            filter_map[fmt_key],
            tr("All Files (*.*)")
        ])

        default_name = f"point_cloud_{w}x{h}{default_ext}"
        path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Save High-Resolution View"),
            default_name,
            all_filters
        )
        if not path:
            return

        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
            # Render offscreen image at chosen resolution
            img = self.renderer.render_to_image(
                width=w,
                height=h,
                include_overlays=self.cb_hud.isChecked(),
                point_size_multiplier=1.0 if self.cb_scale_points.isChecked() else (self.vp_h / max(h, 1)),
                show_labels=self.cb_labels.isChecked(),
                show_hud=self.cb_hud.isChecked(),
                show_measurements=self.cb_measurements.isChecked(),
            )

            if path.lower().endswith(".pdf"):
                self._save_to_pdf(path, img)
            else:
                success = img.save(path)
                if not success:
                    raise RuntimeError(tr("Failed to save image to {path}.").format(path=path))

            self.export_completed.emit(path, fmt_key, w, h)
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, tr("Export Error"), str(exc))
        finally:
            self.unsetCursor()

    def _save_to_pdf(self, path: str, img: QImage):
        writer = QPdfWriter(path)
        ps_id = self.pdf_pagesize.currentData()
        writer.setPageSize(QPageSize(ps_id))

        dpi = int(self.pdf_dpi.currentData() or 300)
        writer.setResolution(dpi)

        ori_mode = self.pdf_orientation.currentData()
        if ori_mode == "landscape" or (ori_mode == "auto" and img.width() >= img.height()):
            writer.setPageOrientation(QPageLayout.Orientation.Landscape)
        else:
            writer.setPageOrientation(QPageLayout.Orientation.Portrait)

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
