"""Standalone Fluent point cloud comparison and measurement window."""
from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import (QIcon, QPalette, QColor, QKeySequence,
    QPixmap, QPainter, QPdfWriter, QPageSize, QShortcut)
from PyQt6.QtPrintSupport import QPrinter, QPrintDialog
from PyQt6.QtWidgets import (QApplication, QColorDialog, QFileDialog, QHBoxLayout,
    QMainWindow, QScrollArea, QSplitter, QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CardWidget, CaptionLabel, ComboBox, FluentIcon,
    DoubleSpinBox, ListWidget, Slider, SubtitleLabel, ToolButton,
    ToggleToolButton, RoundMenu, Action, MenuAnimationType, qconfig, isDarkTheme)
from raven_app.i18n import tr
from raven_app.view_icons import create_cube_icon
from raven_app.clipping_panel import ClippingBoxDialog


COLOR_MODES = [
    ("rgb", tr("RGB / File")),
    ("height", tr("Height (Z)")),
    ("intensity", tr("Intensity")),
    ("x", tr("X Axis")),
    ("y", tr("Y Axis")),
]

COLORMAPS = [
    ("turbo", "Turbo"),
    ("viridis", "Viridis"),
    ("rainbow", "Rainbow"),
    ("heat", "Heat"),
    ("grayscale", "Grayscale"),
]

VIEW_PRESETS = [
    ("iso", tr("Isometric (3D)")),
    ("top", tr("Top (Z+)")),
    ("bottom", tr("Bottom (Z-)")),
    ("front", tr("Front (Y-)")),
    ("back", tr("Back (Y+)")),
    ("left", tr("Left (X-)")),
    ("right", tr("Right (X+)")),
]

BG_PRESETS = [
    (tr("Dark Navy (Default)"), "#0b1017"),
    (tr("Pure Black"), "#000000"),
    (tr("Deep Slate"), "#161b22"),
    (tr("Studio Gray"), "#2a2e33"),
    (tr("Light Neutral"), "#e5e9f0"),
    (tr("Pure White"), "#ffffff"),
]


class _CloudLoadWorker(QThread):
    loaded = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, slot: int, path: str):
        super().__init__(); self.slot, self.path = slot, path

    def run(self):
        try:
            from raven_app.cloud_io import load_cloud
            self.loaded.emit(self.slot, load_cloud(self.path))
        except Exception as exc:
            self.failed.emit(self.slot, str(exc))


class _ViewerExportWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, operation, *args, **kwargs):
        super().__init__(); self.operation, self.args, self.kwargs = operation, args, kwargs

    def run(self):
        try: self.completed.emit(self.operation(*self.args, **self.kwargs))
        except Exception as exc: self.failed.emit(str(exc))


class PointCloudViewerWindow(QMainWindow):
    """Parentless top-level A/B viewer isolated from FluentWindow's GL surface."""
    def __init__(self, parent=None):
        super().__init__(None)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(tr("LidarCamera360 - 3D Viewer"))
        self.resize(1220, 800)
        self.setMinimumSize(980, 640)
        if parent is not None:
            icon = parent.windowIcon()
            if isinstance(icon, QIcon) and not icon.isNull():
                self.setWindowIcon(icon)
        self._workers = set()
        self._closing = False
        self._build_ui()
        self._setup_shortcuts()

    @staticmethod
    def _tool(icon, text, parent):
        button = ToolButton(icon, parent)
        button.setToolTip(text)
        button.setAccessibleName(text)
        button.setFixedSize(38, 38)
        return button

    @staticmethod
    def _cube_tool(preset: str, text: str, parent):
        icon = create_cube_icon(preset, size=24)
        button = ToolButton(icon, parent)
        button.setToolTip(text)
        button.setAccessibleName(text)
        button.setFixedSize(38, 38)
        return button

    def _build_ui(self):
        from raven_app.cloud_view import CloudView
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 12, 14, 14)
        root_layout.setSpacing(10)

        # 2-Tier Compact Toolbar (Replaces old header title + single cramped row)
        tools = CardWidget(root)
        tools_layout = QVBoxLayout(tools)
        tools_layout.setContentsMargins(12, 10, 12, 10)
        tools_layout.setSpacing(8)

        # --- Tier 1: Cloud Layers, Split Comparison & Export ---
        row1 = QHBoxLayout()
        row1.setSpacing(6)

        # Cloud A controls
        row1.addWidget(CaptionLabel("A:", tools))
        self.load_a = self._tool(FluentIcon.FOLDER, tr("Load cloud A"), tools)
        self.load_a.clicked.connect(lambda: self._choose_cloud(0))
        row1.addWidget(self.load_a)

        self.color_combo_a = ComboBox(tools)
        for _, label in COLOR_MODES:
            self.color_combo_a.addItem(label)
        self.color_combo_a.setFixedWidth(115)
        self.color_combo_a.currentIndexChanged.connect(lambda idx: self._on_color_combo_changed(0, idx))
        row1.addWidget(self.color_combo_a)

        self.cmap_combo_a = ComboBox(tools)
        for _, label in COLORMAPS:
            self.cmap_combo_a.addItem(label)
        self.cmap_combo_a.setFixedWidth(95)
        self.cmap_combo_a.setEnabled(False)
        self.cmap_combo_a.currentIndexChanged.connect(lambda idx: self._on_cmap_combo_changed(0, idx))
        row1.addWidget(self.cmap_combo_a)

        row1.addSpacing(10)
        row1.addWidget(CaptionLabel(tr("Split"), tools))
        self.split_slider = Slider(Qt.Orientation.Horizontal, tools)
        self.split_slider.setRange(0, 1000)
        self.split_slider.setValue(500)
        self.split_slider.setMinimumWidth(80)
        self.split_slider.setMaximumWidth(160)
        row1.addWidget(self.split_slider)
        row1.addSpacing(10)

        # Cloud B controls
        row1.addWidget(CaptionLabel("B:", tools))
        self.load_b = self._tool(FluentIcon.FOLDER, tr("Load cloud B"), tools)
        self.load_b.clicked.connect(lambda: self._choose_cloud(1))
        row1.addWidget(self.load_b)

        self.color_combo_b = ComboBox(tools)
        for _, label in COLOR_MODES:
            self.color_combo_b.addItem(label)
        self.color_combo_b.setFixedWidth(115)
        self.color_combo_b.currentIndexChanged.connect(lambda idx: self._on_color_combo_changed(1, idx))
        row1.addWidget(self.color_combo_b)

        self.cmap_combo_b = ComboBox(tools)
        for _, label in COLORMAPS:
            self.cmap_combo_b.addItem(label)
        self.cmap_combo_b.setFixedWidth(95)
        self.cmap_combo_b.setEnabled(False)
        self.cmap_combo_b.currentIndexChanged.connect(lambda idx: self._on_cmap_combo_changed(1, idx))
        row1.addWidget(self.cmap_combo_b)

        row1.addStretch(1)

        # Export Tools
        self.save_button = self._tool(FluentIcon.SAVE, tr("Save image or PDF (Ctrl+S)"), tools)
        self.save_button.clicked.connect(self._save_view)
        row1.addWidget(self.save_button)

        self.copy_button = self._tool(FluentIcon.COPY, tr("Copy image to clipboard (Ctrl+C)"), tools)
        self.copy_button.clicked.connect(self._copy_view)
        row1.addWidget(self.copy_button)

        self.print_button = self._tool(FluentIcon.PRINT, tr("Print view (Ctrl+P)"), tools)
        self.print_button.clicked.connect(self._print_view)
        row1.addWidget(self.print_button)

        tools_layout.addLayout(row1)

        # --- Tier 2: Point Size, Background Color, Clipping Box, View Presets & Fit ---
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        row2.addWidget(CaptionLabel(tr("Point Size"), tools))
        self.point_size = DoubleSpinBox(tools)
        self.point_size.setRange(0.1, 20.0)
        self.point_size.setSingleStep(0.2)
        self.point_size.setValue(2.0)
        self.point_size.setFixedWidth(125)  # comfortably fits number and buttons without squishing
        self.point_size.valueChanged.connect(self._on_spin_point_size_changed)
        row2.addWidget(self.point_size)

        row2.addSpacing(8)

        # Background Color Picker Button with RoundMenu
        self.bg_button = self._tool(FluentIcon.PALETTE, tr("Change Background Color"), tools)
        self.bg_button.clicked.connect(self._show_bg_menu)
        row2.addWidget(self.bg_button)

        # 3D Clipping Box Button (CloudCompare style in viewport)
        self.clip_button = ToggleToolButton(FluentIcon.CUT, tools)
        self.clip_button.setToolTip(tr("3D Clipping Box (X/Y/Z Crop)"))
        self.clip_button.setAccessibleName(tr("3D Clipping Box"))
        self.clip_button.setFixedSize(38, 38)
        self.clip_button.clicked.connect(self._toggle_clipping_box)
        row2.addWidget(self.clip_button)

        row2.addSpacing(4)

        # Move & Rotate Transform buttons in Toolbar
        self.move_tool_button = ToggleToolButton(FluentIcon.MOVE, tools)
        self.move_tool_button.setToolTip(tr("Move tool: Drag cloud or axes in viewport"))
        self.move_tool_button.setAccessibleName(tr("Move tool"))
        self.move_tool_button.setFixedSize(38, 38)
        self.move_tool_button.clicked.connect(self._toggle_move_tool)
        row2.addWidget(self.move_tool_button)

        self.rotate_tool_button = ToggleToolButton(FluentIcon.ROTATE, tools)
        self.rotate_tool_button.setToolTip(tr("Rotate tool: Drag cloud or ring in viewport"))
        self.rotate_tool_button.setAccessibleName(tr("Rotate tool"))
        self.rotate_tool_button.setFixedSize(38, 38)
        self.rotate_tool_button.clicked.connect(self._toggle_rotate_tool)
        row2.addWidget(self.rotate_tool_button)

        row2.addSpacing(8)
        row2.addWidget(CaptionLabel(tr("View"), tools))
        self.view_combo = ComboBox(tools)
        for _, label in VIEW_PRESETS:
            self.view_combo.addItem(label)
        self.view_combo.setFixedWidth(130)
        self.view_combo.currentIndexChanged.connect(self._on_view_combo_changed)
        row2.addWidget(self.view_combo)

        # Cubemap 3D viewpoint buttons
        self.iso_button = self._cube_tool("iso", tr("Isometric view (1)"), tools)
        self.top_button = self._cube_tool("top", tr("Top view (2)"), tools)
        self.bottom_button = self._cube_tool("bottom", tr("Bottom view (3)"), tools)
        self.front_button = self._cube_tool("front", tr("Front view (4)"), tools)
        self.back_button = self._cube_tool("back", tr("Back view (5)"), tools)
        self.left_button = self._cube_tool("left", tr("Left view (6)"), tools)
        self.right_button = self._cube_tool("right", tr("Right view (7)"), tools)

        cube_buttons = [
            (self.iso_button, "iso"),
            (self.top_button, "top"),
            (self.bottom_button, "bottom"),
            (self.front_button, "front"),
            (self.back_button, "back"),
            (self.left_button, "left"),
            (self.right_button, "right"),
        ]
        for btn, preset in cube_buttons:
            btn.clicked.connect(lambda checked=False, p=preset: self.renderer.set_preset(p))
            row2.addWidget(btn)

        self.fit_button = self._tool(FluentIcon.FIT_PAGE, tr("Fit all (F)"), tools)
        self.fit_button.clicked.connect(lambda: self.renderer.fit_all())
        row2.addWidget(self.fit_button)

        self.measure_toggle = self._tool(FluentIcon.RIGHT_ARROW, tr("Show or hide measurements"), tools)
        self.measure_toggle.clicked.connect(self._toggle_measurements)
        row2.addWidget(self.measure_toggle)

        row2.addStretch(1)
        tools_layout.addLayout(row2)

        root_layout.addWidget(tools)

        # Main Central Area: single 3D scene with compact tools and measurements.
        split = QSplitter(Qt.Orientation.Horizontal, root)
        split.setChildrenCollapsible(False)
        render_card = CardWidget(split)
        render_layout = QVBoxLayout(render_card)
        render_layout.setContentsMargins(6, 6, 6, 6)
        self.renderer = CloudView(render_card)
        render_layout.addWidget(self.renderer)

        side = QWidget(split)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        from raven_app.viewer_edit_panel import ViewerEditPanel
        self.edit_panel = ViewerEditPanel(self.renderer)
        self.edit_panel.save_requested.connect(self._save_modified_cloud)
        self.edit_panel.orthophoto_requested.connect(self._generate_orthophoto)
        edit_scroll = QScrollArea(side)
        edit_scroll.setWidgetResizable(True)
        edit_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        edit_scroll.setWidget(self.edit_panel)
        side_layout.addWidget(edit_scroll)

        measure_card = CardWidget(side)
        self.measure_card = measure_card
        measure_layout = QVBoxLayout(measure_card)
        measure_layout.setContentsMargins(14, 14, 14, 14)
        measure_layout.setSpacing(8)
        measure_card.setMinimumWidth(260)
        measure_layout.addWidget(SubtitleLabel(tr("Measurements"), measure_card))
        help_label = CaptionLabel(tr("Choose a tool, then click points in the scene."), measure_card)
        help_label.setWordWrap(True)
        measure_layout.addWidget(help_label)

        mode_row = QHBoxLayout()
        self._mode_buttons = {}
        for mode, icon, label in (
            ("point", FluentIcon.PIN, tr("Point coordinates")),
            ("distance", FluentIcon.ALIGNMENT, tr("Distance")),
            ("polyline", FluentIcon.EDIT, tr("Polyline length")),
            ("angle", FluentIcon.ROTATE, tr("Angle"))
        ):
            button = ToggleToolButton(icon, measure_card)
            button.setToolTip(label)
            button.setAccessibleName(label)
            button.setFixedSize(38, 38)
            button.setProperty("measurementMode", mode)
            button.clicked.connect(lambda checked=False, value=mode: self._set_mode(value, checked))
            self._mode_buttons[mode] = button
            mode_row.addWidget(button)
        mode_row.addStretch(1)
        measure_layout.addLayout(mode_row)

        self.measurements = ListWidget(measure_card)
        self.measurements.setAlternatingRowColors(True)
        self.measurements.setMaximumHeight(100)
        measure_layout.addWidget(self.measurements, 1)

        action_row = QHBoxLayout()
        self.undo_button = self._tool(FluentIcon.RETURN, tr("Undo last measurement"), measure_card)
        self.clear_button = self._tool(FluentIcon.DELETE, tr("Clear measurements"), measure_card)
        self.export_csv = self._tool(FluentIcon.SAVE, tr("Export measurements as CSV"), measure_card)
        self.undo_button.clicked.connect(self.renderer.undo_measurement)
        self.clear_button.clicked.connect(self.renderer.clear_measurements)
        self.export_csv.clicked.connect(self._export_measurements)
        for button in (self.undo_button, self.clear_button, self.export_csv):
            action_row.addWidget(button)
        action_row.addStretch(1)
        measure_layout.addLayout(action_row)

        self.status = BodyLabel(tr("Open a PCD, PLY, LAS or LAZ cloud. Enable the base map for georeferenced files."), measure_card)
        self.status.setWordWrap(True)
        measure_layout.addWidget(self.status)

        hints = CaptionLabel(tr("Left drag: orbit  •  Right/middle drag: pan  •  Wheel: zoom  •  F: fit  •  1-7: views  •  +/-: point size  •  C: color  •  Enter: polyline  •  Esc: cancel"), measure_card)
        hints.setWordWrap(True)
        measure_layout.addWidget(hints)

        side_layout.addWidget(measure_card, 1)
        measure_card.hide()
        split.addWidget(render_card)
        split.addWidget(side)
        split.setSizes([840, 360])
        root_layout.addWidget(split, 1)
        root_layout.addWidget(self.status)
        self._dataset_context = dict(insv=None, trajectory=None, output=None, colmap_dir=None)
        self._edit_worker = None
        self.setCentralWidget(root)

        # Compatibility attributes
        self.color_combo = self.color_combo_a
        self.cmap_combo = self.cmap_combo_a
        self.mode_combo = None

        # Wire Signals
        self.split_slider.valueChanged.connect(lambda value: self.renderer.set_split(value / 1000.0))
        self.renderer.split_changed.connect(self._sync_split)
        self.renderer.point_size_changed.connect(self._sync_point_size)
        self.renderer.view_preset_changed.connect(self._sync_view_preset)
        self.renderer.color_mode_changed.connect(self._sync_color_mode)
        self.renderer.colormap_changed.connect(self._sync_colormap)
        self.renderer.measurement_added.connect(self._replace_measurements)
        self.renderer.status_changed.connect(self.status.setText)

        # Clipping Box & Transform signals
        self.renderer.clipping_box_mode_changed.connect(self._on_clipping_box_mode_changed)
        self.renderer.transform_mode_changed.connect(self._on_transform_mode_changed)
        self.renderer.export_clipped_requested.connect(self._export_clipped_cloud)

        # Retain clipping dialog instance for headless/legacy compatibility
        self.clip_dialog = ClippingBoxDialog(self)
        self.clip_dialog.clipping_changed.connect(self._on_clipping_changed)
        self.clip_dialog.finished.connect(lambda _: self.clip_button.setChecked(False))

        qconfig.themeChanged.connect(self._sync_theme)
        self._sync_theme()

    def _toggle_measurements(self):
        visible = not self.measure_card.isVisible()
        self.measure_card.setVisible(visible)
        self.measure_toggle.setToolTip(tr("Hide measurements") if visible else tr("Show measurements"))

    def set_dataset_context(self, *, insv=None, trajectory=None, output=None, colmap_dir=None):
        cloud = self._dataset_context.get('cloud')
        self._dataset_context = dict(insv=insv, trajectory=trajectory, output=output, colmap_dir=colmap_dir)
        if cloud:
            self._dataset_context['cloud'] = cloud
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self, self._save_view)
        QShortcut(QKeySequence("Ctrl+C"), self, self._copy_view)
        QShortcut(QKeySequence("Ctrl+P"), self, self._print_view)

    def _on_color_combo_changed(self, slot: int, index: int):
        if 0 <= index < len(COLOR_MODES):
            mode = COLOR_MODES[index][0]
            self.renderer.set_color_mode(slot, mode)
            cmap_combo = self.cmap_combo_a if slot == 0 else self.cmap_combo_b
            cmap_combo.setEnabled(mode != 'rgb')

    def _on_cmap_combo_changed(self, slot: int, index: int):
        if 0 <= index < len(COLORMAPS):
            cmap = COLORMAPS[index][0]
            self.renderer.set_colormap(slot, cmap)

    def _on_view_combo_changed(self, index: int):
        if 0 <= index < len(VIEW_PRESETS):
            name = VIEW_PRESETS[index][0]
            self.renderer.set_preset(name)

    def _sync_view_preset(self, name: str):
        for i, (key, _) in enumerate(VIEW_PRESETS):
            if key == name:
                self.view_combo.blockSignals(True)
                self.view_combo.setCurrentIndex(i)
                self.view_combo.blockSignals(False)
                break

    def _sync_color_mode(self, *args):
        for slot, (combo, cmap_combo) in enumerate([(self.color_combo_a, self.cmap_combo_a), (self.color_combo_b, self.cmap_combo_b)]):
            mode = self.renderer.color_modes[slot]
            for i, (key, _) in enumerate(COLOR_MODES):
                if key == mode:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    cmap_combo.setEnabled(mode != 'rgb')
                    break

    def _sync_colormap(self, *args):
        for slot, cmap_combo in enumerate([self.cmap_combo_a, self.cmap_combo_b]):
            cmap = self.renderer.colormaps[slot]
            for i, (key, _) in enumerate(COLORMAPS):
                if key == cmap:
                    cmap_combo.blockSignals(True)
                    cmap_combo.setCurrentIndex(i)
                    cmap_combo.blockSignals(False)
                    break

    def _on_spin_point_size_changed(self, value: float):
        self.renderer.set_point_size(value)

    def _sync_point_size(self, value: float):
        self.point_size.blockSignals(True)
        self.point_size.setValue(value)
        self.point_size.blockSignals(False)

    def _set_mode(self, mode, checked=True):
        if not checked:
            self.renderer.set_mode("navigate")
            return
        for name, button in self._mode_buttons.items():
            button.setChecked(name == mode)
        self.renderer.set_mode(mode)

    def _show_bg_menu(self):
        menu = RoundMenu(parent=self)
        for name, hex_code in BG_PRESETS:
            act = Action(name, menu)
            act.triggered.connect(lambda _, c=hex_code: self.renderer.set_background_color(c))
            menu.addAction(act)
        menu.addSeparator()
        custom_act = Action(FluentIcon.PALETTE, tr("Custom Color..."), menu)
        custom_act.triggered.connect(self._choose_custom_bg_color)
        menu.addAction(custom_act)
        menu.exec(self.bg_button.mapToGlobal(self.bg_button.rect().bottomLeft()), aniType=MenuAnimationType.DROP_DOWN)

    def _choose_custom_bg_color(self):
        current = self.renderer.bg_color
        col = QColorDialog.getColor(current, self, tr("Choose Background Color"))
        if col.isValid():
            self.renderer.set_background_color(col)

    def _toggle_clipping_box(self, checked: bool = False):
        if checked:
            self.renderer.set_clipping_box_mode(True)
        else:
            self.renderer.set_clipping_box_visible(False)

    def _on_clipping_box_mode_changed(self, enabled: bool):
        self.clip_button.blockSignals(True)
        self.clip_button.setChecked(enabled)
        self.clip_button.blockSignals(False)

    def _toggle_move_tool(self, checked: bool = False):
        if checked:
            self.renderer.set_transform_mode('translate')
        else:
            self.renderer.set_transform_mode(None)

    def _toggle_rotate_tool(self, checked: bool = False):
        if checked:
            self.renderer.set_transform_mode('rotate')
        else:
            self.renderer.set_transform_mode(None)

    def _on_transform_mode_changed(self, mode: str):
        m = str(mode).lower()
        self.move_tool_button.blockSignals(True)
        self.rotate_tool_button.blockSignals(True)
        self.move_tool_button.setChecked(m == 'translate')
        self.rotate_tool_button.setChecked(m == 'rotate')
        self.move_tool_button.blockSignals(False)
        self.rotate_tool_button.blockSignals(False)

    def _export_clipped_cloud(self):
        slot = self.renderer.active_transform_slot
        cloud = self.renderer.clouds[slot]
        if cloud is None:
            slot = 0 if self.renderer.clouds[0] is not None else 1
            cloud = self.renderer.clouds[slot]
        if cloud is None:
            self.status.setText(tr("Load a cloud before exporting."))
            return

        pts, cols, ints = self.renderer.get_clipped_points(slot)
        if len(pts) == 0:
            self.status.setText(tr("No points inside current clipping box to export."))
            return

        out_path, _ = QFileDialog.getSaveFileName(
            self,
            tr("Export Clipped Cloud"),
            f"{cloud.path.stem}_clipped.ply",
            tr("Point clouds (*.ply *.las *.laz *.pcd)")
        )
        if not out_path:
            return

        from raven_app.cloud_io import CloudData
        from raven_app.cloud_export import save_cloud
        clipped_cloud = CloudData(
            path=Path(out_path),
            points=pts,
            colors=cols if cols is not None else np.zeros((len(pts), 3), dtype=np.float32),
            intensities=ints,
            original_count=len(pts),
            crs_wkt=cloud.crs_wkt
        )
        self._start_edit_worker(_ViewerExportWorker(save_cloud, clipped_cloud, out_path), tr("Exporting clipped cloud..."))

    def _toggle_clipping_dialog(self, checked: bool = False):
        if checked:
            self.clip_dialog.set_bounds_from_clouds(self.renderer.clouds)
            self.clip_dialog.show()
            self.clip_dialog.raise_()
            self.clip_dialog.activateWindow()
        else:
            self.clip_dialog.hide()

    def _on_clipping_changed(self):
        c_min, c_max, enabled = self.clip_dialog.get_clipping_bounds()
        self.renderer.set_clipping_bounds(c_min, c_max, enabled)
        self.clip_button.setChecked(self.clip_dialog.isVisible())

    def _save_view(self):
        from raven_app.export_view_dialog import ExportViewDialog
        dialog = ExportViewDialog(self.renderer, self)
        dialog.export_completed.connect(self._on_export_completed)
        dialog.exec()

    def _on_export_completed(self, path: str, fmt: str, w: int, h: int):
        if path == "clipboard":
            self.status.setText(tr("Snapshot ({w} × {h}) copied to clipboard.").format(w=w, h=h))
        elif fmt == "pdf":
            self.status.setText(tr("Exported view to PDF ({w} × {h}): {path}").format(w=w, h=h, path=Path(path).name))
        else:
            self.status.setText(tr("Saved snapshot ({w} × {h}) to {path}").format(w=w, h=h, path=Path(path).name))

    def _copy_view(self):
        try:
            img = self.renderer.grabFramebuffer()
            QApplication.clipboard().setImage(img)
            self.status.setText(tr("Snapshot copied to clipboard."))
        except Exception as exc:
            self.status.setText(tr("Could not copy snapshot: {error}").format(error=exc))

    def _print_view(self):
        try:
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            dialog = QPrintDialog(printer, self)
            if dialog.exec() == QPrintDialog.DialogCode.Accepted:
                painter = QPainter(printer)
                rect = painter.viewport()
                target_w, target_h = rect.width(), rect.height()
                try:
                    img = self.renderer.render_to_image(
                        target_w, target_h,
                        include_overlays=True,
                        point_size_multiplier=1.0,
                    )
                except Exception:
                    img = self.renderer.grabFramebuffer()
                scaled = img.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                x = (rect.width() - scaled.width()) // 2
                y = (rect.height() - scaled.height()) // 2
                painter.drawImage(x, y, scaled)
                painter.end()
                self.status.setText(tr("3D view printed successfully."))
        except Exception as exc:
            self.status.setText(tr("Could not print view: {error}").format(error=exc))

    def _sync_theme(self, *args):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('#202020' if isDarkTheme() else '#f0f4f9'))
        palette.setColor(QPalette.ColorRole.WindowText, QColor('#ffffff' if isDarkTheme() else '#1a1a1a'))
        self.setPalette(palette)
        self.centralWidget().setAutoFillBackground(True)
        self.centralWidget().setPalette(palette)
        self.renderer.update()

    def _choose_cloud(self, slot):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open point cloud"), "", tr("Point clouds (*.pcd *.ply *.las *.laz);;All files (*.*)"))
        if path:
            self._load_cloud(slot, path)

    def _load_cloud(self, slot, path):
        self.load_a.setEnabled(False)
        self.load_b.setEnabled(False)
        worker = _CloudLoadWorker(slot, path)
        self._workers.add(worker)
        worker.loaded.connect(self._on_cloud_loaded)
        worker.failed.connect(self._on_cloud_failed)
        worker.finished.connect(lambda: self._worker_done(worker))
        worker.start()
        self.status.setText(tr("Loading {path}...").format(path=Path(path).name))

    def _worker_done(self, worker):
        self._workers.discard(worker)
        worker.deleteLater()
        self.load_a.setEnabled(True)
        self.load_b.setEnabled(True)
        if self._closing and not self._workers:
            self._closing = False
            self.close()

    def _on_cloud_loaded(self, slot, cloud):
        self.renderer.set_cloud(slot, cloud)
        if slot == 0:
            self._dataset_context['cloud'] = str(cloud.path)
        if hasattr(self, 'clip_dialog'):
            self.clip_dialog.set_bounds_from_clouds(self.renderer.clouds)

    def _on_cloud_failed(self, slot, message):
        self.status.setText(tr("Could not load cloud: {error}").format(error=message))

    def _replace_measurements(self, text):
        self.measurements.clear()
        if text:
            self.measurements.addItems(str(text).splitlines())

    def _sync_split(self, value):
        self.split_slider.blockSignals(True)
        self.split_slider.setValue(round(value * 1000))
        self.split_slider.blockSignals(False)

    def _export_measurements(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Export measurements"), "measurements.csv", "CSV (*.csv)")
        if path:
            try:
                self.renderer.export_measurements(path)
                self.status.setText(tr("Measurements exported to {path}").format(path=Path(path).name))
            except Exception as exc:
                self.status.setText(tr("Could not export measurements: {error}").format(error=exc))

    def _start_edit_worker(self, worker, message):
        self._edit_worker = worker
        self.edit_panel.set_busy(True)
        worker.completed.connect(self._edit_worker_done)
        worker.failed.connect(self._edit_worker_failed)
        worker.finished.connect(lambda: self._edit_worker_finished(worker))
        self.status.setText(message)
        worker.start()

    def _edit_worker_finished(self, worker):
        if self._edit_worker is worker:
            self._edit_worker = None
        self.edit_panel.set_busy(False)
        worker.deleteLater()

    def _edit_worker_done(self, result):
        self.status.setText(tr("Operation completed: {path}").format(path=str((result or {}).get("output", result))[:180]))

    def _edit_worker_failed(self, message):
        self.status.setText(tr("Operation failed: {error}").format(error=message))

    def _save_modified_cloud(self, slot):
        cloud = self.renderer.clouds[int(slot)]
        if cloud is None:
            self.status.setText(tr("Load a cloud before saving.")); return
        path, _ = QFileDialog.getSaveFileName(self, tr("Save modified cloud"), cloud.path.stem + ".ply",
                                              tr("Point clouds (*.las *.laz *.ply *.pcd)"))
        if not path: return
        from raven_app.cloud_export import save_cloud
        snapshot = replace(cloud)
        self._start_edit_worker(_ViewerExportWorker(save_cloud, snapshot, path), tr("Saving modified cloud..."))

    def _generate_orthophoto(self, slot, gsd):
        cloud = self.renderer.clouds[int(slot)]
        if cloud is None:
            self.status.setText(tr("Load a cloud before generating an orthophoto.")); return
        path, _ = QFileDialog.getSaveFileName(self, tr("Save orthophoto"), cloud.path.stem + "_orthophoto.tif", tr("GeoTIFF (*.tif *.tiff)"))
        if not path: return
        from raven_app.orthophoto import generate_orthophoto
        snapshot = replace(cloud)
        epsg = None
        if snapshot.crs_wkt:
            try:
                import pyproj
                epsg = pyproj.CRS.from_wkt(snapshot.crs_wkt).to_epsg()
            except Exception:
                epsg = None
        worker = _ViewerExportWorker(generate_orthophoto, snapshot, path, gsd_m=float(gsd), crs_wkt=snapshot.crs_wkt, crs_epsg=epsg)
        self._start_edit_worker(worker, tr("Generating orthophoto..."))

    def closeEvent(self, event):
        if self._edit_worker is not None and self._edit_worker.isRunning():
            self.status.setText(tr("Waiting for export or orthophoto generation to finish..."))
            event.ignore()
            return
        if hasattr(self, 'clip_dialog') and self.clip_dialog.isVisible():
            self.clip_dialog.close()
        if self._workers:
            self._closing = True
            for worker in tuple(self._workers):
                worker.requestInterruption()
            event.ignore()
            self.status.setText(tr("Waiting for cloud loading to finish..."))
            return
        super().closeEvent(event)


ViewerWindow = PointCloudViewerWindow
