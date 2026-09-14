"""Compact editing and export controls for the single-scene cloud viewer."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QWidget
from qfluentwidgets import (CaptionLabel, ComboBox, DoubleSpinBox,
    SubtitleLabel, PrimaryPushButton, PushButton, SwitchButton)
from raven_app.i18n import tr


class ViewerEditPanel(QWidget):
    save_requested = pyqtSignal(int)
    orthophoto_requested = pyqtSignal(int, float)

    def __init__(self, renderer, parent=None):
        super().__init__(parent)
        self.renderer = renderer
        self._busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        layout.addWidget(SubtitleLabel(tr("Scene tools"), self))

        row = QHBoxLayout()
        row.addWidget(CaptionLabel(tr("Active cloud"), self))
        self.slot_combo = ComboBox(self)
        self.slot_combo.addItems([tr("Cloud A"), tr("Cloud B")])
        self.slot_combo.currentIndexChanged.connect(self._sync_adjustment)
        row.addWidget(self.slot_combo, 1)
        layout.addLayout(row)

        self.basemap_toggle = SwitchButton(tr("Base map: Off"), self)
        self.basemap_toggle.checkedChanged.connect(self._basemap_changed)
        layout.addWidget(self.basemap_toggle)
        row = QHBoxLayout()
        row.addWidget(CaptionLabel(tr("Provider"), self))
        self.provider_combo = ComboBox(self)
        self.provider_combo.addItems([tr("Satellite"), tr("Street")])
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        row.addWidget(self.provider_combo, 1)
        layout.addLayout(row)

        layout.addWidget(SubtitleLabel(tr("Transform preview"), self))
        self.yaw = self._spin(-360.0, 360.0, 0.1, tr("Yaw (deg)"), layout)
        self.dx = self._spin(-1e9, 1e9, 0.01, tr("dX"), layout)
        self.dy = self._spin(-1e9, 1e9, 0.01, tr("dY"), layout)
        self.dz = self._spin(-1e9, 1e9, 0.01, tr("dZ"), layout)
        row = QHBoxLayout()
        self.apply_button = PrimaryPushButton(tr("Apply"), self)
        self.reset_button = PushButton(tr("Reset"), self)
        self.apply_button.clicked.connect(self._apply)
        self.reset_button.clicked.connect(self._reset)
        row.addWidget(self.apply_button)
        row.addWidget(self.reset_button)
        layout.addLayout(row)

        self.save_button = PushButton(tr("Save modified cloud"), self)
        self.save_button.clicked.connect(lambda: self.save_requested.emit(self.active_slot))
        layout.addWidget(self.save_button)
        layout.addWidget(SubtitleLabel(tr("Orthophoto"), self))
        row = QHBoxLayout()
        row.addWidget(CaptionLabel(tr("Resolution (m/px)"), self))
        self.gsd = DoubleSpinBox(self)
        self.gsd.setRange(0.001, 100.0)
        self.gsd.setSingleStep(0.01)
        self.gsd.setValue(0.05)
        row.addWidget(self.gsd, 1)
        layout.addLayout(row)
        self.ortho_button = PushButton(tr("Generate orthophoto"), self)
        self.ortho_button.clicked.connect(lambda: self.orthophoto_requested.emit(self.active_slot, float(self.gsd.value())))
        layout.addWidget(self.ortho_button)
        layout.addStretch(1)

    @property
    def active_slot(self) -> int:
        return int(self.slot_combo.currentIndex())

    def _spin(self, lo, hi, step, label, layout):
        row = QHBoxLayout()
        row.addWidget(CaptionLabel(label, self))
        spin = DoubleSpinBox(self)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        row.addWidget(spin, 1)
        layout.addLayout(row)
        return spin

    def _basemap_changed(self, checked):
        self.basemap_toggle.setText(tr("Base map: On") if checked else tr("Base map: Off"))
        self.renderer.set_basemap_enabled(bool(checked))

    def _provider_changed(self, index):
        self.renderer.set_basemap_provider("satellite" if index == 0 else "street")

    def _values(self):
        return float(self.yaw.value()), (float(self.dx.value()), float(self.dy.value()), float(self.dz.value()))

    def _sync_adjustment(self, slot=None):
        values = getattr(self.renderer, "_adjustments", [None, None])[self.active_slot]
        if not values:
            values = (0.0, (0.0, 0.0, 0.0))
        yaw, translation = values
        for spin, value in zip((self.yaw, self.dx, self.dy, self.dz), (yaw, *translation)):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    def _preview(self):
        if not self._busy:
            yaw, translation = self._values()
            self.renderer.apply_cloud_adjustment(self.active_slot, yaw, translation)

    def _apply(self):
        self._preview()

    def _reset(self):
        self.renderer.reset_cloud_adjustment(self.active_slot)
        for spin in (self.yaw, self.dx, self.dy, self.dz):
            spin.blockSignals(True)
            spin.setValue(0.0)
            spin.blockSignals(False)

    def set_busy(self, busy: bool):
        self._busy = bool(busy)
        self.setEnabled(not self._busy)
