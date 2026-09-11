"""Standalone Fluent point cloud comparison and measurement window."""
from __future__ import annotations

from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QIcon, QPalette, QColor
from PyQt6.QtWidgets import (QFileDialog, QHBoxLayout, QMainWindow, QSplitter,
    QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CardWidget, CaptionLabel, FluentIcon,
    DoubleSpinBox, ListWidget, Slider, SubtitleLabel, TitleLabel, ToolButton, ToggleToolButton,
    qconfig, isDarkTheme)
from raven_app.i18n import tr


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


class PointCloudViewerWindow(QMainWindow):
    """Parentless top-level A/B viewer isolated from FluentWindow's GL surface."""
    def __init__(self, parent=None):
        # Keep the native GL surface opaque and independent of FluentWindow.
        super().__init__(None)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowTitle(tr("LidarCamera360 - 3D Viewer")); self.resize(1180, 760)
        self.setMinimumSize(960, 620)
        if parent is not None:
            icon = parent.windowIcon()
            if isinstance(icon, QIcon) and not icon.isNull(): self.setWindowIcon(icon)
        self._workers = set(); self._closing = False; self._build_ui()

    @staticmethod
    def _tool(icon, text, parent):
        button = ToolButton(icon, parent); button.setToolTip(text); button.setAccessibleName(text)
        button.setFixedSize(38, 38); return button

    def _build_ui(self):
        from raven_app.cloud_view import CloudView
        root = QWidget(self); root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 14, 18, 18); root_layout.setSpacing(12)

        header = CardWidget(root); header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 12, 18, 12); header_layout.setSpacing(2)
        header_layout.addWidget(TitleLabel(tr("3D Viewer"), header))
        header_layout.addWidget(SubtitleLabel(tr("Compare two point clouds and inspect measurements"), header))
        root_layout.addWidget(header)

        tools = CardWidget(root); tools_layout = QHBoxLayout(tools)
        tools_layout.setContentsMargins(10, 8, 10, 8); tools_layout.setSpacing(6)
        self.load_a = self._tool(FluentIcon.FOLDER, tr("Load cloud A"), tools)
        self.load_b = self._tool(FluentIcon.FOLDER, tr("Load cloud B"), tools)
        self.load_a.clicked.connect(lambda: self._choose_cloud(0)); self.load_b.clicked.connect(lambda: self._choose_cloud(1))
        tools_layout.addWidget(CaptionLabel("A", tools)); tools_layout.addWidget(self.load_a)
        tools_layout.addWidget(CaptionLabel("B", tools)); tools_layout.addWidget(self.load_b); tools_layout.addSpacing(8)
        tools_layout.addWidget(CaptionLabel(tr("Split"), tools))
        self.split_slider = Slider(Qt.Orientation.Horizontal, tools); self.split_slider.setRange(0, 1000); self.split_slider.setValue(500)
        self.split_slider.setMinimumWidth(150); self.split_slider.setMaximumWidth(280); tools_layout.addWidget(self.split_slider); tools_layout.addSpacing(8)
        tools_layout.addWidget(CaptionLabel(tr("Point size"), tools)); self.point_size = DoubleSpinBox(tools)
        self.point_size.setRange(.1, 20); self.point_size.setSingleStep(.1); self.point_size.setValue(2.0); self.point_size.setFixedWidth(130); tools_layout.addWidget(self.point_size); tools_layout.addStretch(1)
        self.fit_button = self._tool(FluentIcon.FIT_PAGE, tr("Fit all"), tools)
        self.top_button = self._tool(getattr(FluentIcon, "UP", FluentIcon.SPEED_HIGH), tr("Top view"), tools)
        self.front_button = self._tool(FluentIcon.VIEW, tr("Front view"), tools)
        self.right_button = self._tool(FluentIcon.ROTATE, tr("Right view"), tools)
        self.iso_button = self._tool(FluentIcon.FULL_SCREEN, tr("Isometric view"), tools)
        for button in (self.fit_button, self.top_button, self.front_button, self.right_button, self.iso_button): tools_layout.addWidget(button)
        root_layout.addWidget(tools)

        split = QSplitter(Qt.Orientation.Horizontal, root); split.setChildrenCollapsible(False)
        render_card = CardWidget(split); render_layout = QVBoxLayout(render_card); render_layout.setContentsMargins(6, 6, 6, 6)
        self.renderer = CloudView(render_card); render_layout.addWidget(self.renderer)
        measure_card = CardWidget(split); measure_layout = QVBoxLayout(measure_card); measure_layout.setContentsMargins(14, 14, 14, 14); measure_layout.setSpacing(8)
        measure_card.setMinimumWidth(260)
        measure_layout.addWidget(SubtitleLabel(tr("Measurements"), measure_card))
        help_label = CaptionLabel(tr("Choose a tool, then click points in the scene."), measure_card)
        help_label.setWordWrap(True); measure_layout.addWidget(help_label)
        mode_row = QHBoxLayout(); self._mode_buttons = {}
        for mode, icon, label in (("point", FluentIcon.PIN, tr("Point coordinates")), ("distance", FluentIcon.ALIGNMENT, tr("Distance")), ("polyline", FluentIcon.EDIT, tr("Polyline length")), ("angle", FluentIcon.ROTATE, tr("Angle"))):
            button = ToggleToolButton(icon, measure_card)
            button.setToolTip(label); button.setAccessibleName(label); button.setFixedSize(38, 38)
            button.setProperty("measurementMode", mode)
            button.clicked.connect(lambda checked=False, value=mode: self._set_mode(value, checked)); self._mode_buttons[mode] = button; mode_row.addWidget(button)
        mode_row.addStretch(1); measure_layout.addLayout(mode_row)
        self.measurements = ListWidget(measure_card); self.measurements.setAlternatingRowColors(True); measure_layout.addWidget(self.measurements, 1)
        action_row = QHBoxLayout(); self.undo_button = self._tool(FluentIcon.RETURN, tr("Undo last measurement"), measure_card); self.clear_button = self._tool(FluentIcon.DELETE, tr("Clear measurements"), measure_card); self.export_csv = self._tool(FluentIcon.SAVE, tr("Export measurements as CSV"), measure_card)
        self.undo_button.clicked.connect(self.renderer.undo_measurement); self.clear_button.clicked.connect(self.renderer.clear_measurements); self.export_csv.clicked.connect(self._export_measurements)
        for button in (self.undo_button, self.clear_button, self.export_csv): action_row.addWidget(button)
        action_row.addStretch(1); measure_layout.addLayout(action_row)
        self.status = BodyLabel(tr("Load two PCD or PLY clouds to compare. Files must share the same coordinate system."), measure_card); self.status.setWordWrap(True); measure_layout.addWidget(self.status)
        hints = CaptionLabel(tr("Left drag: orbit  •  Right/middle drag: pan  •  Wheel: zoom  •  F: fit  •  Enter: finish polyline  •  Esc: cancel"), measure_card)
        hints.setWordWrap(True); measure_layout.addWidget(hints)
        split.addWidget(render_card); split.addWidget(measure_card); split.setSizes([820, 300]); root_layout.addWidget(split, 1); self.setCentralWidget(root)

        self.mode_combo = None  # compatibility attribute for older integrations
        self.split_slider.valueChanged.connect(lambda value: self.renderer.set_split(value / 1000.0)); self.point_size.valueChanged.connect(self._set_point_size)
        self.fit_button.clicked.connect(self.renderer.fit_all)
        for button, preset in ((self.top_button, "top"), (self.front_button, "front"), (self.right_button, "right"), (self.iso_button, "iso")): button.clicked.connect(lambda checked=False, name=preset: self.renderer.set_preset(name))
        self.renderer.measurement_added.connect(self._replace_measurements); self.renderer.status_changed.connect(self.status.setText); self.renderer.split_changed.connect(self._sync_split)
        qconfig.themeChanged.connect(self._sync_theme)
        self._sync_theme()

    def _set_mode(self, mode, checked=True):
        if not checked:
            self.renderer.set_mode("navigate")
            return
        for name, button in self._mode_buttons.items(): button.setChecked(name == mode)
        self.renderer.set_mode(mode)

    def _sync_theme(self, *args):
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor('#202020' if isDarkTheme() else '#f0f4f9'))
        palette.setColor(QPalette.ColorRole.WindowText, QColor('#ffffff' if isDarkTheme() else '#1a1a1a'))
        self.setPalette(palette)
        self.centralWidget().setAutoFillBackground(True)
        self.centralWidget().setPalette(palette)
        self.renderer.update()

    def _choose_cloud(self, slot):
        path, _ = QFileDialog.getOpenFileName(self, tr("Open point cloud"), "", tr("Point clouds (*.pcd *.ply);;All files (*.*)"))
        if path: self._load_cloud(slot, path)

    def _load_cloud(self, slot, path):
        self.load_a.setEnabled(False); self.load_b.setEnabled(False); worker = _CloudLoadWorker(slot, path); self._workers.add(worker)
        worker.loaded.connect(self._on_cloud_loaded); worker.failed.connect(self._on_cloud_failed); worker.finished.connect(lambda: self._worker_done(worker)); worker.start(); self.status.setText(tr("Loading {path}...").format(path=Path(path).name))

    def _worker_done(self, worker):
        self._workers.discard(worker); worker.deleteLater(); self.load_a.setEnabled(True); self.load_b.setEnabled(True)
        if self._closing and not self._workers: self._closing = False; self.close()

    def _on_cloud_loaded(self, slot, cloud): self.renderer.set_cloud(slot, cloud)
    def _on_cloud_failed(self, slot, message): self.status.setText(tr("Could not load cloud: {error}").format(error=message))

    def _replace_measurements(self, text):
        self.measurements.clear()
        if text: self.measurements.addItems(str(text).splitlines())

    def _sync_split(self, value):
        self.split_slider.blockSignals(True); self.split_slider.setValue(round(value * 1000)); self.split_slider.blockSignals(False)

    def _set_point_size(self, value): self.renderer.point_size = value; self.renderer.update()

    def _export_measurements(self):
        path, _ = QFileDialog.getSaveFileName(self, tr("Export measurements"), "measurements.csv", "CSV (*.csv)")
        if path:
            try:
                self.renderer.export_measurements(path); self.status.setText(tr("Measurements exported to {path}").format(path=Path(path).name))
            except Exception as exc: self.status.setText(tr("Could not export measurements: {error}").format(error=exc))

    def closeEvent(self, event):
        if self._workers:
            self._closing = True
            for worker in tuple(self._workers): worker.requestInterruption()
            event.ignore(); self.status.setText(tr("Waiting for cloud loading to finish...")); return
        super().closeEvent(event)


ViewerWindow = PointCloudViewerWindow
