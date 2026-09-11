"""3D Clipping Box control dialog with real-time axis slicing."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CaptionLabel, CardWidget, DoubleSpinBox,
    FluentIcon, PrimaryPushButton, PushButton, Slider, SubtitleLabel, SwitchButton)
from raven_app.i18n import tr


class ClippingBoxDialog(QDialog):
    clipping_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("3D Clipping Box"))
        self.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowCloseButtonHint)
        self.setFixedWidth(420)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        
        self._cloud_bounds = {
            'x': (-50.0, 50.0),
            'y': (-50.0, 50.0),
            'z': (-20.0, 30.0),
        }
        self._updating = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header with Switch
        top_row = QHBoxLayout()
        top_row.addWidget(SubtitleLabel(tr("Clipping Box"), self))
        top_row.addStretch(1)
        self.switch_btn = SwitchButton(self)
        self.switch_btn.setOnText(tr("ON"))
        self.switch_btn.setOffText(tr("OFF"))
        self.switch_btn.checkedChanged.connect(self._on_switch_toggled)
        top_row.addWidget(self.switch_btn)
        layout.addLayout(top_row)

        desc = CaptionLabel(tr("Slice point clouds along X, Y, and Z axes in real time."), self)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Sliders Card
        card = CardWidget(self)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(10)

        self.axis_controls = {}
        for axis, label_text, color in (
            ('x', tr("X Axis (Width)"), '#e05656'),
            ('y', tr("Y Axis (Depth)"), '#48bb78'),
            ('z', tr("Z Axis (Height)"), '#4299e1'),
        ):
            sec = QVBoxLayout(); sec.setSpacing(3)
            hdr = QHBoxLayout()
            dot = CaptionLabel("●", card)
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            hdr.addWidget(dot)
            hdr.addWidget(BodyLabel(label_text, card))
            hdr.addStretch(1)
            sec.addLayout(hdr)

            # Spinboxes row
            row = QHBoxLayout(); row.setSpacing(6)
            row.addWidget(CaptionLabel(tr("Min:"), card))
            spin_min = DoubleSpinBox(card); spin_min.setRange(-1000000.0, 1000000.0)
            spin_min.setDecimals(1); spin_min.setSingleStep(0.5); spin_min.setFixedWidth(130)
            row.addWidget(spin_min)

            row.addWidget(CaptionLabel(tr("Max:"), card))
            spin_max = DoubleSpinBox(card); spin_max.setRange(-1000000.0, 1000000.0)
            spin_max.setDecimals(1); spin_max.setSingleStep(0.5); spin_max.setFixedWidth(130)
            row.addWidget(spin_max)
            sec.addLayout(row)

            # Sliders row
            slider_min = Slider(Qt.Orientation.Horizontal, card)
            slider_min.setRange(0, 1000); slider_min.setValue(0)
            slider_max = Slider(Qt.Orientation.Horizontal, card)
            slider_max.setRange(0, 1000); slider_max.setValue(1000)

            sec.addWidget(slider_min)
            sec.addWidget(slider_max)
            card_layout.addLayout(sec)

            self.axis_controls[axis] = {
                'spin_min': spin_min,
                'spin_max': spin_max,
                'slider_min': slider_min,
                'slider_max': slider_max,
            }

            # Wire signals
            slider_min.valueChanged.connect(lambda v, a=axis: self._on_slider_changed(a, 'min', v))
            slider_max.valueChanged.connect(lambda v, a=axis: self._on_slider_changed(a, 'max', v))
            spin_min.valueChanged.connect(lambda v, a=axis: self._on_spin_changed(a, 'min', v))
            spin_max.valueChanged.connect(lambda v, a=axis: self._on_spin_changed(a, 'max', v))

        layout.addWidget(card)

        # Buttons row
        btn_row = QHBoxLayout()
        self.reset_btn = PushButton(FluentIcon.SYNC, tr("Reset Full Volume"), self)
        self.reset_btn.clicked.connect(self.reset_bounds)
        btn_row.addWidget(self.reset_btn)
        layout.addLayout(btn_row)

    def set_bounds_from_clouds(self, clouds: list):
        valid_clouds = [c for c in clouds if c is not None]
        if not valid_clouds:
            return
        all_lo = np.min([c.points.min(axis=0) for c in valid_clouds], axis=0)
        all_hi = np.max([c.points.max(axis=0) for c in valid_clouds], axis=0)
        margin = np.maximum((all_hi - all_lo) * 0.05, 0.5)

        self._cloud_bounds = {
            'x': (float(all_lo[0] - margin[0]), float(all_hi[0] + margin[0])),
            'y': (float(all_lo[1] - margin[1]), float(all_hi[1] + margin[1])),
            'z': (float(all_lo[2] - margin[2]), float(all_hi[2] + margin[2])),
        }
        self.reset_bounds()

    def reset_bounds(self):
        self._updating = True
        for axis, (b_lo, b_hi) in self._cloud_bounds.items():
            ctrl = self.axis_controls[axis]
            ctrl['spin_min'].setValue(b_lo)
            ctrl['spin_max'].setValue(b_hi)
            ctrl['slider_min'].setValue(0)
            ctrl['slider_max'].setValue(1000)
        self._updating = False
        self.clipping_changed.emit()

    def get_clipping_bounds(self) -> tuple[np.ndarray, np.ndarray, bool]:
        enabled = self.switch_btn.isChecked()
        c_min = np.array([
            self.axis_controls['x']['spin_min'].value(),
            self.axis_controls['y']['spin_min'].value(),
            self.axis_controls['z']['spin_min'].value(),
        ], dtype=np.float32)
        c_max = np.array([
            self.axis_controls['x']['spin_max'].value(),
            self.axis_controls['y']['spin_max'].value(),
            self.axis_controls['z']['spin_max'].value(),
        ], dtype=np.float32)
        return c_min, c_max, enabled

    def _on_switch_toggled(self, checked: bool):
        self.clipping_changed.emit()

    def _on_slider_changed(self, axis: str, kind: str, val: int):
        if self._updating:
            return
        b_lo, b_hi = self._cloud_bounds[axis]
        span = max(b_hi - b_lo, 1e-4)
        float_val = b_lo + (val / 1000.0) * span
        self._updating = True
        ctrl = self.axis_controls[axis]
        if kind == 'min':
            if float_val > ctrl['spin_max'].value():
                float_val = ctrl['spin_max'].value()
                ctrl['slider_min'].setValue(int((float_val - b_lo) / span * 1000))
            ctrl['spin_min'].setValue(float_val)
        else:
            if float_val < ctrl['spin_min'].value():
                float_val = ctrl['spin_min'].value()
                ctrl['slider_max'].setValue(int((float_val - b_lo) / span * 1000))
            ctrl['spin_max'].setValue(float_val)
        self._updating = False
        self.clipping_changed.emit()

    def _on_spin_changed(self, axis: str, kind: str, val: float):
        if self._updating:
            return
        b_lo, b_hi = self._cloud_bounds[axis]
        span = max(b_hi - b_lo, 1e-4)
        pos = int(np.clip((val - b_lo) / span * 1000.0, 0, 1000))
        self._updating = True
        ctrl = self.axis_controls[axis]
        if kind == 'min':
            ctrl['slider_min'].setValue(pos)
        else:
            ctrl['slider_max'].setValue(pos)
        self._updating = False
        self.clipping_changed.emit()
