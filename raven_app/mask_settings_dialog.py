"""Visual editor for per-camera fixed-area and fisheye mask settings."""
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QRectF, QPointF, QProcess, QProcessEnvironment, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF, QIcon, QPalette
from PyQt6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout,
                             QSlider, QVBoxLayout, QWidget)
from qfluentwidgets import (BodyLabel, CaptionLabel, ComboBox, DoubleSpinBox,
                            FluentIcon, ToolButton)

from raven_app.i18n import tr
from raven_app.config import get_app_root, get_config_dir
from raven_app.mask_resources import (masker_environment, resolve_masker_executable,
                                      model_is_ready, resolve_model,
                                      resolve_runtime_dir,
                                      resolve_vulkan_executable,
                                      resolve_vulkan_schedule,
                                      resolve_vulkan_shaders_dir)
from raven_app.person_masks import (apply_mask_exclusions, fisheye_cutoff_radius,
                                    normalize_mask_config, static_keep_mask)


def _shape_icon(kind):
    image = QPixmap(24, 24)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QApplication.palette().color(QPalette.ColorRole.ButtonText), 2))
    if kind == 'rectangles':
        painter.drawRect(4, 5, 16, 14)
    elif kind == 'ellipses':
        painter.drawEllipse(4, 5, 16, 14)
    else:
        painter.drawPolygon(QPolygonF([QPointF(4, 19), QPointF(6, 5), QPointF(20, 9), QPointF(17, 19)]))
    painter.end()
    return QIcon(image)


class MaskCanvas(QWidget):
    shapes_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)
        self.image = QImage()
        self.mask_image = QImage()
        self.show_mask = False
        self.rectangles = []
        self.ellipses = []
        self.polygons = []
        self.shape_kind = 'rectangles'
        self.selected = None
        self.border_percent = 0.0
        self.drag_start = None
        self.drag_end = None
        self.drag_action = None
        self.drag_original = None
        self.polygon_draft = []
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self._pan_start = None
        self._pan_origin = None
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, path):
        self.image = QImage(str(path)) if path else QImage()
        self.mask_image = QImage()
        self.show_mask = False
        self._clamp_pan()
        self.update()
        return not self.image.isNull()

    def set_mask(self, mask):
        image = np.ascontiguousarray(mask, dtype=np.uint8)
        self.mask_image = QImage(image.data, image.shape[1], image.shape[0], image.strides[0],
                                 QImage.Format.Format_Grayscale8).copy()
        self.update()

    def set_shapes(self, rectangles, ellipses, polygons):
        self.rectangles = rectangles
        self.ellipses = ellipses
        self.polygons = polygons
        self.selected = None
        self.polygon_draft.clear()
        self.update()

    def _shapes(self, kind):
        return {'rectangles': self.rectangles, 'ellipses': self.ellipses,
                'polygons': self.polygons}[kind]

    def _box_rect(self, box):
        image = self._image_rect()
        x0, y0, x1, y1 = box
        return QRectF(image.left() + x0 * image.width(), image.top() + y0 * image.height(),
                      (x1 - x0) * image.width(), (y1 - y0) * image.height())

    def _handles(self, box):
        rect = self._box_rect(box)
        return {'left': QPointF(rect.left(), rect.center().y()),
                'right': QPointF(rect.right(), rect.center().y()),
                'top': QPointF(rect.center().x(), rect.top()),
                'bottom': QPointF(rect.center().x(), rect.bottom())}

    def _image_point(self, point):
        rect = self._image_rect()
        return QPointF(rect.left() + point[0] * rect.width(), rect.top() + point[1] * rect.height())

    def _polygon_at(self, position, polygon):
        return QPolygonF([self._image_point(point) for point in polygon]).containsPoint(
            position, Qt.FillRule.OddEvenFill)

    def _hit_shape(self, position):
        if self.selected is not None:
            kind, index = self.selected
            if kind == 'polygons':
                for vertex, point in enumerate(self._shapes(kind)[index]):
                    p = self._image_point(point)
                    if abs(position.x() - p.x()) <= 10 and abs(position.y() - p.y()) <= 10:
                        return kind, index, f'vertex:{vertex}'
            else:
                box = self._shapes(kind)[index]
                for handle, point in self._handles(box).items():
                    if abs(position.x() - point.x()) <= 10 and abs(position.y() - point.y()) <= 10:
                        return kind, index, handle
        for kind in ('polygons', 'ellipses', 'rectangles'):
            for index in reversed(range(len(self._shapes(kind)))):
                shape = self._shapes(kind)[index]
                contains = (self._polygon_at(position, shape) if kind == 'polygons'
                            else self._box_rect(shape).contains(position))
                if contains:
                    return kind, index, 'move'
        return None

    def _image_rect(self):
        if self.image.isNull():
            return QRectF()
        scale = min(self.width() / self.image.width(), self.height() / self.image.height())
        width, height = self.image.width() * scale, self.image.height() * scale
        width, height = width * self.zoom, height * self.zoom
        return QRectF((self.width() - width) / 2 + self.pan.x(),
                      (self.height() - height) / 2 + self.pan.y(), width, height)

    def _normalized(self, position):
        rect = self._image_rect()
        if rect.isEmpty():
            return None
        return QPointF(max(0., min(1., (position.x() - rect.left()) / rect.width())),
                       max(0., min(1., (position.y() - rect.top()) / rect.height())))

    def _clamp_pan(self):
        if self.image.isNull():
            self.pan = QPointF(0, 0)
            return
        rect = self._image_rect()
        x_limit = max(0., (rect.width() - self.width()) * .5)
        y_limit = max(0., (rect.height() - self.height()) * .5)
        self.pan.setX(max(-x_limit, min(x_limit, self.pan.x())))
        self.pan.setY(max(-y_limit, min(y_limit, self.pan.y())))

    def zoom_by(self, factor, anchor=None):
        if self.image.isNull():
            return
        anchor = anchor or QPointF(self.width() * .5, self.height() * .5)
        old = self._image_rect()
        u = (anchor.x() - old.left()) / old.width()
        v = (anchor.y() - old.top()) / old.height()
        old_zoom = self.zoom
        self.zoom = max(1.0, min(8.0, self.zoom * factor))
        ratio = self.zoom / old_zoom
        new_width = old.width() * ratio
        new_height = old.height() * ratio
        self.pan = QPointF(anchor.x() - u * new_width - (self.width() - new_width) * .5,
                           anchor.y() - v * new_height - (self.height() - new_height) * .5)
        self._clamp_pan()
        self.update()

    def reset_zoom(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def wheelEvent(self, event):
        if event.angleDelta().y():
            self.zoom_by(1.2 ** (event.angleDelta().y() / 120), event.position())
            event.accept()

    def _finish_polygon(self):
        if len(self.polygon_draft) >= 3:
            points = [[point.x(), point.y()] for point in self.polygon_draft]
            area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1]
                           - points[(i + 1) % len(points)][0] * points[i][1]
                           for i in range(len(points))))
            if area >= .0001:
                self.polygons.append(points)
                self.selected = 'polygons', len(self.polygons) - 1
                self.shapes_changed.emit()
        self.polygon_draft.clear()
        self.update()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._finish_polygon()
            event.accept()
        elif event.key() == Qt.Key.Key_Escape and self.polygon_draft:
            self.polygon_draft.clear()
            self.update()
            event.accept()
        elif event.key() == Qt.Key.Key_Backspace and self.polygon_draft:
            self.polygon_draft.pop()
            self.update()
            event.accept()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 23, 28))
        rect = self._image_rect()
        if rect.isEmpty():
            painter.setPen(QColor(225, 225, 225))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, tr('Choose a preview frame'))
            return
        painter.drawImage(rect, self.mask_image if self.show_mask and not self.mask_image.isNull() else self.image)
        if self.show_mask:
            return
        if self.border_percent:
            radius = (fisheye_cutoff_radius((self.image.height(), self.image.width()), self.border_percent)
                      * rect.width() / self.image.width())
            center = rect.center()
            painter.setPen(QPen(QColor(255, 190, 50), 3))
            painter.drawEllipse(center, radius, radius)
        preview = None
        if self.drag_start is not None and self.drag_end is not None:
            preview = [min(self.drag_start.x(), self.drag_end.x()),
                       min(self.drag_start.y(), self.drag_end.y()),
                       max(self.drag_start.x(), self.drag_end.x()),
                       max(self.drag_start.y(), self.drag_end.y())]
        for kind, color in (('rectangles', QColor(255, 75, 75)), ('ellipses', QColor(80, 220, 255))):
            painter.setPen(QPen(color, 2))
            painter.setBrush(QColor(color.red(), color.green(), color.blue(), 70))
            boxes = list(self._shapes(kind))
            if preview is not None and kind == self.shape_kind:
                boxes.append(preview)
            for box in boxes:
                if kind == 'rectangles':
                    painter.drawRect(self._box_rect(box))
                else:
                    painter.drawEllipse(self._box_rect(box))
        painter.setPen(QPen(QColor(255, 95, 210), 2))
        painter.setBrush(QColor(255, 95, 210, 70))
        for polygon in self.polygons:
            painter.drawPolygon(QPolygonF([self._image_point(point) for point in polygon]))
        if self.polygon_draft:
            painter.setPen(QPen(QColor(255, 95, 210), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            points = [self._image_point((point.x(), point.y())) for point in self.polygon_draft]
            if len(points) == 1:
                painter.drawEllipse(points[0], 3, 3)
            else:
                painter.drawPolyline(QPolygonF(points))
            painter.setBrush(QColor(255, 95, 210))
            for point in points:
                painter.drawEllipse(point, 4, 4)
        if self.selected is not None:
            kind, index = self.selected
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setBrush(QColor(255, 255, 255))
            if kind == 'polygons':
                for point in self._shapes(kind)[index]:
                    point = self._image_point(point)
                    painter.drawRect(QRectF(point.x() - 4, point.y() - 4, 8, 8))
            else:
                for point in self._handles(self._shapes(kind)[index]).values():
                    painter.drawRect(QRectF(point.x() - 4, point.y() - 4, 8, 8))

    def mousePressEvent(self, event):
        if event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._pan_start = event.position()
            self._pan_origin = QPointF(self.pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self.show_mask:
            return
        if event.button() == Qt.MouseButton.LeftButton and self._image_rect().contains(event.position()):
            if self.shape_kind == 'polygons':
                point = self._normalized(event.position())
                if self.polygon_draft and len(self.polygon_draft) >= 3:
                    first = self._image_point((self.polygon_draft[0].x(), self.polygon_draft[0].y()))
                    if (event.position() - first).manhattanLength() <= 12:
                        self._finish_polygon()
                        return
                if not self.polygon_draft:
                    hit = self._hit_shape(event.position())
                    if hit is not None and hit[0] == 'polygons':
                        _, index, self.drag_action = hit
                        self.selected = 'polygons', index
                        self.drag_origin = point
                        self.drag_original = [list(p) for p in self.polygons[index]]
                        return
                if not self.polygon_draft:
                    self.selected = None
                self.polygon_draft.append(point)
                self.setFocus()
                self.update()
                return
            hit = self._hit_shape(event.position())
            self.drag_origin = self._normalized(event.position())
            if hit is not None:
                kind, index, self.drag_action = hit
                self.selected = kind, index
                self.drag_original = list(self._shapes(kind)[index])
            else:
                self.selected = None
                self.drag_action = 'draw'
                self.drag_start = self.drag_origin
                self.drag_end = self.drag_start
            self.update()

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self.pan = self._pan_origin + delta
            self._clamp_pan()
            self.update()
            return
        if self.drag_action is None:
            return
        point = self._normalized(event.position())
        if self.drag_action == 'draw':
            self.drag_end = point
        else:
            kind, index = self.selected
            dx, dy = point.x() - self.drag_origin.x(), point.y() - self.drag_origin.y()
            if kind == 'polygons':
                original = self.drag_original
                if self.drag_action == 'move':
                    dx = max(-min(p[0] for p in original), min(1 - max(p[0] for p in original), dx))
                    dy = max(-min(p[1] for p in original), min(1 - max(p[1] for p in original), dy))
                    self.polygons[index] = [[x + dx, y + dy] for x, y in original]
                else:
                    vertex = int(self.drag_action.split(':', 1)[1])
                    updated = [list(p) for p in original]
                    updated[vertex] = [point.x(), point.y()]
                    self.polygons[index] = updated
            else:
                x0, y0, x1, y1 = self.drag_original
                if self.drag_action == 'move':
                    dx = max(-x0, min(1 - x1, dx))
                    dy = max(-y0, min(1 - y1, dy))
                    box = [x0 + dx, y0 + dy, x1 + dx, y1 + dy]
                else:
                    box = [x0, y0, x1, y1]
                    for handle, position, lower, upper in (
                        ('left', 0, 0, x1 - .005), ('right', 2, x0 + .005, 1),
                        ('top', 1, 0, y1 - .005), ('bottom', 3, y0 + .005, 1)):
                        if self.drag_action == handle:
                            delta = dx if handle in ('left', 'right') else dy
                            box[position] = max(lower, min(upper, box[position] + delta))
                self._shapes(kind)[index] = box
        self.update()

    def mouseReleaseEvent(self, event):
        if self._pan_start is not None:
            self._pan_start = self._pan_origin = None
            self.unsetCursor()
            return
        if self.drag_action is None:
            return
        self.mouseMoveEvent(event)
        if self.drag_action == 'draw':
            x0, x1 = sorted((self.drag_start.x(), self.drag_end.x()))
            y0, y1 = sorted((self.drag_start.y(), self.drag_end.y()))
            if x1 - x0 >= .005 and y1 - y0 >= .005:
                shapes = self._shapes(self.shape_kind)
                shapes.append([x0, y0, x1, y1])
                self.selected = self.shape_kind, len(shapes) - 1
                self.shapes_changed.emit()
        else:
            self.shapes_changed.emit()
        self.drag_action = self.drag_original = None
        self.drag_start = self.drag_end = None
        self.update()

    def mouseDoubleClickEvent(self, event):
        if self.shape_kind == 'polygons' and event.button() == Qt.MouseButton.LeftButton:
            self._finish_polygon()


class MaskSettingsDialog(QDialog):
    """Draw fixed rig exclusions on a preview; apply each to every frame of its lens."""

    def __init__(self, dataset_dir, config=None, parent=None, model_path=None,
                 backend='vulkan', insv_path=None, fps=2.0, camera_names=(),
                 auxiliary_sources=()):
        super().__init__(parent)
        self.insv_path = Path(insv_path) if insv_path else None
        self.fps = float(fps)
        self._frames_temp = None
        if dataset_dir:
            self.dataset_dir = Path(dataset_dir)
        elif self.insv_path:
            self._frames_temp = tempfile.TemporaryDirectory(prefix='raven-mask-frames-')
            self.dataset_dir = Path(self._frames_temp.name)
        else:
            self.dataset_dir = None
        from raven_app.video import has_cached_frame_extraction
        self._lazy_video_preview = bool(
            self.insv_path is not None
            and not has_cached_frame_extraction(self.insv_path, self.dataset_dir, self.fps)
        )
        self.config = normalize_mask_config(config)
        self.auxiliary_sources = {}
        for source in auxiliary_sources or ():
            camera_name = (source.get('camera_name') if isinstance(source, dict)
                           else getattr(source, 'camera_name', None))
            if camera_name:
                self.auxiliary_sources[str(camera_name)] = source
        self.model_path = model_path
        self.backend = backend
        self.preview_paths = {}
        self.frames_by_camera = {}
        self._current_source = None
        self._person_masks = {}
        self._preview_process = None
        self._preview_temp = None
        self._frame_extraction_process = None
        self._frame_extraction_output = []
        self._frame_extraction_index = None
        self._pending_frame_extraction_index = None
        self._video_preview_timeline = ()
        self._video_preview_paths = {'cam0': {}, 'cam1': {}}
        self._aux_video_preview_paths = {}
        self._aux_video_preview_info = {}
        self._aux_video_preview_temp = None
        self._video_preview_temp = None
        self._video_preview_root = None
        self._frame_extraction_timer = QTimer(self)
        self._frame_extraction_timer.setSingleShot(True)
        self._frame_extraction_timer.setInterval(140)
        self._frame_extraction_timer.timeout.connect(self._start_selected_video_frame_extraction)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._start_detection)
        self.setWindowTitle(tr('Mask settings'))
        self.resize(880, 900)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        description = CaptionLabel(tr('Drag to draw rectangles or ovals. For polygons, click each corner and double-click or press Enter to finish; Backspace removes a corner and Esc cancels. Use the wheel to zoom and right or middle drag to pan. Shapes apply to every frame of the selected camera.'))
        description.setWordWrap(True)
        layout.addWidget(description)

        controls = QHBoxLayout()
        controls.addWidget(BodyLabel(tr('Camera:')))
        self.camera_combo = ComboBox()
        cameras = ['cam0', 'cam1']
        discovered_cameras = set()
        if self.dataset_dir and (self.dataset_dir / 'images').is_dir():
            discovered_cameras = {
                p.name for p in (self.dataset_dir / 'images').iterdir() if p.is_dir()
            }
        for camera in camera_names or ():
            name = str(camera).strip()
            if name and name not in cameras:
                cameras.append(name)
        cameras.extend(sorted(
            discovered_cameras - set(cameras),
            key=lambda name: tuple(
                (0, int(part)) if part.isdigit() else (1, part.lower())
                for part in re.split(r'(\d+)', name)
            ),
        ))
        for camera in cameras:
            self.camera_combo.addItem(camera)
        self.camera_combo.currentTextChanged.connect(self._camera_changed)
        controls.addWidget(self.camera_combo)
        self.preview_button = ToolButton(FluentIcon.PHOTO, self)
        self.preview_button.setToolTip(tr('Choose a preview frame'))
        self.preview_button.setAccessibleName(tr('Choose a preview frame'))
        self.preview_button.clicked.connect(self._browse_preview)
        controls.addWidget(self.preview_button)
        self.show_mask_button = ToolButton(FluentIcon.VIEW, self)
        self.show_mask_button.setToolTip(tr('Preview mask'))
        self.show_mask_button.setAccessibleName(tr('Preview mask'))
        self.show_mask_button.clicked.connect(self._toggle_mask_preview)
        controls.addWidget(self.show_mask_button)
        controls.addStretch()
        for icon, tooltip, callback in (
            (FluentIcon.ZOOM_IN, tr('Zoom in'), lambda _checked=False: self.canvas.zoom_by(1.2)),
            (FluentIcon.ZOOM_OUT, tr('Zoom out'), lambda _checked=False: self.canvas.zoom_by(1 / 1.2)),
            (FluentIcon.FIT_PAGE, tr('Fit image to view'), lambda _checked=False: self.canvas.reset_zoom()),
        ):
            button = ToolButton(icon, self)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.clicked.connect(callback)
            controls.addWidget(button)
        layout.addLayout(controls)

        self.canvas = MaskCanvas(self)
        self.canvas.shapes_changed.connect(self._refresh_mask_preview)
        layout.addWidget(self.canvas, 1)

        self.preview_status = CaptionLabel('')
        self.preview_status.setWordWrap(True)
        layout.addWidget(self.preview_status)

        timeline = QHBoxLayout()
        timeline.addWidget(BodyLabel(tr('Frame:')))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setToolTip(tr('Choose a camera frame to preview'))
        self.frame_slider.valueChanged.connect(self._frame_changed)
        timeline.addWidget(self.frame_slider, 1)
        self.frame_label = CaptionLabel(tr('No frames'))
        self.frame_label.setMaximumWidth(260)
        timeline.addWidget(self.frame_label)
        layout.addLayout(timeline)

        edit_row = QHBoxLayout()
        edit_row.addWidget(BodyLabel(tr('Draw:')))
        self.shape_buttons = {}
        for kind, tooltip in (('rectangles', tr('Draw rectangle')),
                              ('ellipses', tr('Draw oval or circle')),
                              ('polygons', tr('Draw free-form polygon'))):
            button = ToolButton(_shape_icon(kind), self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.clicked.connect(lambda checked, shape=kind: self._set_shape_kind(shape))
            self.shape_buttons[kind] = button
            edit_row.addWidget(button)
        self.shape_buttons['rectangles'].setChecked(True)
        self.undo_button = ToolButton(FluentIcon.DELETE, self)
        self.undo_button.setToolTip(tr('Remove selected shape'))
        self.undo_button.setAccessibleName(tr('Remove selected shape'))
        self.undo_button.clicked.connect(self._undo)
        self.clear_button = ToolButton(FluentIcon.BROOM, self)
        self.clear_button.setToolTip(tr('Clear all shapes for this camera'))
        self.clear_button.setAccessibleName(tr('Clear all shapes for this camera'))
        self.clear_button.clicked.connect(self._clear)
        edit_row.addWidget(self.undo_button)
        edit_row.addWidget(self.clear_button)
        edit_row.addStretch()
        layout.addLayout(edit_row)

        border_row = QHBoxLayout()
        border_row.addWidget(BodyLabel(tr('Cut fisheye border:')))
        self.border_spin = DoubleSpinBox()
        self.border_spin.setRange(0, 25)
        self.border_spin.setDecimals(2)
        self.border_spin.setSingleStep(0.1)
        self.border_spin.setSuffix(' %')
        self.border_spin.setValue(self.config['fisheye_border_percent'])
        self.border_spin.valueChanged.connect(self._border_changed)
        self.canvas.border_percent = self.border_spin.value()
        border_row.addWidget(self.border_spin)
        border_hint = CaptionLabel(tr('0% off; 1% excludes about 1% of image pixels. Saved separately for each camera.'))
        border_hint.setWordWrap(True)
        border_hint.setMaximumWidth(360)
        border_row.addWidget(border_hint)
        border_row.addStretch()
        layout.addLayout(border_row)

        actions = QHBoxLayout()
        actions.addStretch()
        cancel = ToolButton(FluentIcon.CANCEL, self)
        cancel.setToolTip(tr('Cancel'))
        cancel.setAccessibleName(tr('Cancel'))
        cancel.clicked.connect(self.reject)
        self.save_button = ToolButton(FluentIcon.ACCEPT, self)
        self.save_button.setToolTip(tr('Use these masks'))
        self.save_button.setAccessibleName(tr('Use these masks'))
        self.save_button.clicked.connect(self.accept)
        actions.addWidget(cancel)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self._camera_changed(self.camera_combo.currentText())
        self._start_frame_extraction_if_needed()

    def _camera_changed(self, camera):
        self._stop_detection()
        self._person_masks.clear()
        boxes = self.config['rectangles'].setdefault(camera, [])
        ellipses = self.config['ellipses'].setdefault(camera, [])
        polygons = self.config['polygons'].setdefault(camera, [])
        self.canvas.set_shapes(boxes, ellipses, polygons)
        border = self.config['fisheye_border_percent_by_camera'].get(
            camera, self.config['fisheye_border_percent'])
        self.border_spin.blockSignals(True)
        self.border_spin.setValue(border)
        self.border_spin.blockSignals(False)
        self.canvas.border_percent = border
        if self._lazy_video_preview and camera in ('cam0', 'cam1'):
            index = self.frame_slider.value()
            count = len(self._video_preview_timeline)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, max(0, count - 1))
            index = min(index, max(0, count - 1))
            self.frame_slider.setValue(index)
            self.frame_slider.setEnabled(count > 1)
            self.frame_slider.blockSignals(False)
            self._load_video_preview_frame(index)
            if count and self._current_source is None:
                self._schedule_video_frame_extraction(index)
            self.show_mask_button.setToolTip(tr('Preview mask'))
            self.show_mask_button.setAccessibleName(tr('Preview mask'))
            self.show_mask_button.setIcon(FluentIcon.VIEW)
            return
        folder = self.dataset_dir / 'images' / camera if self.dataset_dir else None
        frames = []
        if folder and folder.is_dir():
            frames = sorted((p for p in folder.iterdir() if p.is_file()
                             and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')),
                            key=self._natural_key)
        if self._frame_extraction_process is not None:
            frames = []
        if camera in self.preview_paths:
            frames = [self.preview_paths[camera]]
        if not frames and camera in self.auxiliary_sources:
            source_path = Path(self._source_value(camera, 'source_path') or '')
            if source_path.is_file() and source_path.suffix.lower() in (
                    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'):
                frames = [source_path]
            elif source_path.is_dir():
                frames = sorted((p for p in source_path.iterdir() if p.is_file()
                                 and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')),
                                key=self._natural_key)
        self.frames_by_camera[camera] = frames
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, len(frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(len(frames) > 1)
        self.frame_slider.blockSignals(False)
        self._load_frame(0)
        if not frames and camera in self.auxiliary_sources:
            self._prepare_auxiliary_video_preview(camera)
        self.show_mask_button.setToolTip(tr('Preview mask'))
        self.show_mask_button.setAccessibleName(tr('Preview mask'))
        self.show_mask_button.setIcon(FluentIcon.VIEW)
        if self._frame_extraction_process is not None:
            self.frame_label.setText(tr('Extracting frames...'))
            self.preview_status.setText(tr('Extracting video frames for mask preview. This may take a few minutes.'))
        elif not frames and self.insv_path is None:
            self.preview_status.setText(tr('No extracted frames found. Choose a preview image to see the mask.'))
        else:
            self.preview_status.setText('')

    def _source_value(self, camera, key, default=None):
        source = self.auxiliary_sources.get(camera)
        return source.get(key, default) if isinstance(source, dict) else getattr(source, key, default)

    def _prepare_auxiliary_video_preview(self, camera):
        source_path = Path(self._source_value(camera, 'source_path') or '')
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            capture.release()
            self.preview_status.setText(tr('Cannot load auxiliary camera preview. Check its source file.'))
            return
        native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        if total <= 0:
            self.preview_status.setText(tr('Cannot read the auxiliary video timeline.'))
            return
        target_fps = float(self._source_value(camera, 'fps', self.fps) or self.fps or 1.0)
        step = max(1, int(round(native_fps / target_fps)))
        count = max(1, (total + step - 1) // step)
        self._aux_video_preview_info[camera] = (source_path, step, count)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, count - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(count > 1)
        self.frame_slider.blockSignals(False)
        self._load_auxiliary_video_preview(camera, 0)

    def _load_auxiliary_video_preview(self, camera, index):
        source_path, step, count = self._aux_video_preview_info[camera]
        cached = self._aux_video_preview_paths.setdefault(camera, {}).get(index)
        if cached is not None and cached.is_file():
            self._current_source = cached
            self.canvas.set_image(cached)
            self.frame_label.setText(f'{index + 1}/{count}  {cached.name}')
            self.preview_status.setText('')
            return
        capture = cv2.VideoCapture(str(source_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index * step)
        ok, frame = capture.read()
        capture.release()
        if not ok or frame is None:
            self._current_source = None
            self.canvas.set_image(None)
            self.frame_label.setText(tr('Frame unavailable'))
            self.preview_status.setText(tr('Could not decode this auxiliary video frame.'))
            return
        if self._aux_video_preview_temp is None:
            self._aux_video_preview_temp = tempfile.TemporaryDirectory(prefix='raven-mask-aux-preview-')
        target = Path(self._aux_video_preview_temp.name) / camera / f'frame_{index + 1:06d}.jpg'
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            self.preview_status.setText(tr('Could not save the auxiliary camera preview frame.'))
            return
        self._aux_video_preview_paths[camera][index] = target
        self._current_source = target
        self.canvas.set_image(target)
        self.frame_label.setText(f'{index + 1}/{count}  {target.name}')
        self.preview_status.setText('')

    def _start_frame_extraction_if_needed(self):
        if self.insv_path is None:
            return
        if not self._lazy_video_preview:
            return
        if not self.insv_path.is_file():
            self.preview_status.setText(tr('Cannot extract preview frames: video file not found.'))
            return

        try:
            from raven_app.video import video_preview_timeline
            self._video_preview_timeline = video_preview_timeline(self.insv_path, self.fps)
        except Exception as exc:
            self.preview_status.setText(tr('Cannot read video timeline: {error}').format(error=str(exc)))
            return
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, len(self._video_preview_timeline) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(len(self._video_preview_timeline) > 1)
        self.frame_slider.blockSignals(False)
        self._video_selected_index = 0
        self.preview_status.setText(tr('Loading the first frame for mask preview...'))
        self._load_video_preview_frame(0)
        self._schedule_video_frame_extraction(0, immediate=True)

    def _load_video_preview_frame(self, index):
        camera = self.camera_combo.currentText()
        path = self._video_preview_paths.get(camera, {}).get(index)
        if path is not None and path.is_file():
            self._current_source = path
            self.canvas.set_image(path)
            self.frame_label.setText(f'{index + 1}/{len(self._video_preview_timeline)}  {path.name}')
            self.preview_status.setText('')
            return
        self._current_source = None
        self.canvas.set_image(None)
        self.frame_label.setText(
            tr('Loading frame {current}/{total}...').format(
                current=index + 1, total=len(self._video_preview_timeline)))

    def _schedule_video_frame_extraction(self, index, *, immediate=False):
        if not self._video_preview_timeline or self._video_preview_paths.get(
                self.camera_combo.currentText(), {}).get(index, Path()).is_file():
            return
        self._frame_extraction_timer.stop()
        if self._frame_extraction_process is not None:
            if self._frame_extraction_index == index:
                return
            self._stop_frame_extraction()
        self._pending_frame_extraction_index = index
        self.preview_status.setText(
            tr('Extracting preview frame {current}/{total}...').format(
                current=index + 1, total=len(self._video_preview_timeline)))
        self._frame_extraction_timer.start(0 if immediate else self._frame_extraction_timer.interval())

    def _start_selected_video_frame_extraction(self):
        if self._frame_extraction_process is not None or self.insv_path is None:
            return
        index = self._pending_frame_extraction_index
        if index is None or index >= len(self._video_preview_timeline):
            return
        if self._video_preview_temp is None:
            self._video_preview_temp = tempfile.TemporaryDirectory(prefix='raven-mask-frame-previews-')
            self._video_preview_root = Path(self._video_preview_temp.name)

        arguments = []
        if not getattr(sys, 'frozen', False):
            entry = Path(__file__).resolve().parents[1] / 'raven.py'
            if entry.is_file():
                arguments.append(str(entry))
            else:
                arguments.extend(['-m', 'raven_app.cli'])
        arguments.extend([
            '--headless', 'extract-insv-frame', '--insv', str(self.insv_path),
            '--output', str(self._video_preview_root), '--fps', format(self.fps, '.12g'),
            '--index', str(index),
        ])
        process = QProcess(self)
        self._frame_extraction_process = process
        self._frame_extraction_index = index
        self._frame_extraction_output = []
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(
            lambda current=process: self._read_frame_extraction_output(current))
        process.finished.connect(
            lambda code, _status, current=process: self._frame_extraction_finished(current, code))
        process.errorOccurred.connect(
            lambda error, current=process: self._frame_extraction_error(current, error))
        process.start(sys.executable, arguments)
        if not process.waitForStarted(1500):
            self._finish_frame_extraction_error(process, process.errorString())

    def _read_frame_extraction_output(self, process):
        data = bytes(process.readAllStandardOutput()).decode(errors='replace')
        for line in data.splitlines():
            line = line.strip()
            if line:
                self._frame_extraction_output.append(line)

    def _frame_extraction_error(self, process, error):
        if process is self._frame_extraction_process and error == QProcess.ProcessError.FailedToStart:
            self._finish_frame_extraction_error(process, process.errorString())

    def _finish_frame_extraction_error(self, process, detail):
        if process is not self._frame_extraction_process:
            return
        self._frame_extraction_process = None
        self._frame_extraction_index = None
        process.deleteLater()
        self.save_button.setEnabled(True)
        self.preview_status.setText(tr('Frame extraction failed: {error}').format(error=detail))

    def _frame_extraction_finished(self, process, code):
        if process is not self._frame_extraction_process:
            return
        self._read_frame_extraction_output(process)
        index = self._frame_extraction_index
        self._frame_extraction_process = None
        self._frame_extraction_index = None
        process.deleteLater()
        self.save_button.setEnabled(True)
        paths = {}
        if index is not None and self._video_preview_root is not None:
            for camera in ('cam0', 'cam1'):
                path = self._video_preview_root / 'images' / camera / f'frame_{index + 1:06d}.jpg'
                if path.is_file():
                    paths[camera] = path
        if code == 0 and len(paths) == 2:
            for camera, path in paths.items():
                self._video_preview_paths[camera][index] = path
            if index == getattr(self, '_video_selected_index', 0):
                self._load_video_preview_frame(index)
                self.preview_status.setText(tr('Frame ready. Select Preview mask to inspect the RF-DETR mask.'))
                if self.canvas.show_mask:
                    self._refresh_mask_preview()
                    self.preview_timer.start()
            return
        detail = next((line for line in reversed(self._frame_extraction_output)
                       if line.startswith('RavenCalibrator:') or 'failed' in line.lower()), None)
        self.preview_status.setText(tr('Frame extraction failed: {error}').format(
            error=detail or (self._frame_extraction_output[-1] if self._frame_extraction_output else str(code))))

    def _stop_frame_extraction(self):
        self._frame_extraction_timer.stop()
        self._pending_frame_extraction_index = None
        process = self._frame_extraction_process
        self._frame_extraction_process = None
        self._frame_extraction_index = None
        if process is not None:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.terminate()
                if not process.waitForFinished(1200):
                    process.kill()
                    process.waitForFinished(1200)
            process.deleteLater()

    @staticmethod
    def _natural_key(path):
        return tuple((0, int(part)) if part.isdigit() else (1, part.lower())
                     for part in re.split(r'(\d+)', path.name))

    def _load_frame(self, index):
        frames = self.frames_by_camera.get(self.camera_combo.currentText(), [])
        if not frames:
            self._current_source = None
            self.canvas.set_image(None)
            self.frame_label.setText(tr('No frames'))
            return
        index = max(0, min(index, len(frames) - 1))
        self._current_source = frames[index]
        self.canvas.set_image(self._current_source)
        self.frame_label.setText(f'{index + 1}/{len(frames)}  {self._current_source.name}')

    def _frame_changed(self, index):
        was_mask = self.canvas.show_mask
        self.preview_timer.stop()
        self._stop_detection()
        self._person_masks.clear()
        camera = self.camera_combo.currentText()
        if self._lazy_video_preview and camera in ('cam0', 'cam1'):
            self._video_selected_index = index
            if self._frame_extraction_process is not None and self._frame_extraction_index != index:
                self._stop_frame_extraction()
            self.canvas.show_mask = was_mask
            self._load_video_preview_frame(index)
            if self._current_source is None:
                self._schedule_video_frame_extraction(index)
                if was_mask:
                    self.show_mask_button.setToolTip(tr('Show image'))
                    self.show_mask_button.setAccessibleName(tr('Show image'))
                    self.show_mask_button.setIcon(FluentIcon.HIDE)
                return
            self.preview_status.setText('')
            if was_mask:
                self.show_mask_button.setToolTip(tr('Show image'))
                self.show_mask_button.setAccessibleName(tr('Show image'))
                self.show_mask_button.setIcon(FluentIcon.HIDE)
                self._refresh_mask_preview()
                self.preview_timer.start()
            else:
                self.show_mask_button.setToolTip(tr('Preview mask'))
                self.show_mask_button.setAccessibleName(tr('Preview mask'))
                self.show_mask_button.setIcon(FluentIcon.VIEW)
            return
        if camera in self._aux_video_preview_info:
            self._load_auxiliary_video_preview(camera, index)
            self.canvas.show_mask = was_mask
            if was_mask and self._current_source is not None:
                self._refresh_mask_preview()
                self.preview_timer.start()
            return
        self._load_frame(index)
        self.canvas.show_mask = was_mask
        if was_mask:
            self.show_mask_button.setToolTip(tr('Show image'))
            self.show_mask_button.setAccessibleName(tr('Show image'))
            self.show_mask_button.setIcon(FluentIcon.HIDE)
            self._refresh_mask_preview()
            self.preview_timer.start()
        else:
            self.show_mask_button.setToolTip(tr('Preview mask'))
            self.show_mask_button.setAccessibleName(tr('Preview mask'))
            self.show_mask_button.setIcon(FluentIcon.VIEW)
            self.canvas.update()

    def _browse_preview(self):
        path, _ = QFileDialog.getOpenFileName(self, tr('Choose camera preview frame'), '',
                                               'Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)')
        if path:
            self._stop_detection()
            camera = self.camera_combo.currentText()
            self.preview_paths[camera] = Path(path)
            self.frames_by_camera[camera] = [Path(path)]
            self.frame_slider.blockSignals(True)
            self.frame_slider.setRange(0, 0)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(False)
            self.frame_slider.blockSignals(False)
            self._load_frame(0)
            self.show_mask_button.setToolTip(tr('Preview mask'))
            self.show_mask_button.setAccessibleName(tr('Preview mask'))
            self.show_mask_button.setIcon(FluentIcon.VIEW)
            self.preview_status.setText('')

    def _undo(self):
        if self.canvas.selected is not None:
            kind, index = self.canvas.selected
            self.canvas._shapes(kind).pop(index)
            self.canvas.selected = None
            self.canvas.update()
            self._refresh_mask_preview()

    def _clear(self):
        self.canvas.rectangles.clear()
        self.canvas.ellipses.clear()
        self.canvas.polygons.clear()
        self.canvas.selected = None
        self.canvas.update()
        self._refresh_mask_preview()

    def _set_shape_kind(self, kind):
        self.canvas.shape_kind = kind
        self.canvas.polygon_draft.clear()
        self.canvas.update()

    def _border_changed(self, value):
        camera = self.camera_combo.currentText()
        self.config['fisheye_border_percent_by_camera'][camera] = float(value)
        self.canvas.border_percent = value
        self.canvas.update()
        self._refresh_mask_preview()

    def _preview_key(self):
        if self._current_source is None:
            return None
        return self.camera_combo.currentText(), self._current_source.resolve()

    def _toggle_mask_preview(self):
        if self.canvas.show_mask:
            self.preview_timer.stop()
            self._stop_detection()
            self.canvas.show_mask = False
            self.canvas.update()
            self.show_mask_button.setToolTip(tr('Preview mask'))
            self.show_mask_button.setAccessibleName(tr('Preview mask'))
            self.show_mask_button.setIcon(FluentIcon.VIEW)
            return
        if self.canvas.image.isNull() or self._current_source is None:
            self.preview_status.setText(tr('Choose a preview frame first.'))
            return
        self.canvas.show_mask = True
        self.show_mask_button.setToolTip(tr('Show image'))
        self.show_mask_button.setAccessibleName(tr('Show image'))
        self.show_mask_button.setIcon(FluentIcon.HIDE)
        self._refresh_mask_preview()
        if self._preview_key() not in self._person_masks:
            self._start_detection()

    def _refresh_mask_preview(self):
        if not self.canvas.show_mask:
            return
        camera = self.camera_combo.currentText()
        shape = (self.canvas.image.height(), self.canvas.image.width())
        base = self._person_masks.get(self._preview_key())
        if base is None:
            mask = static_keep_mask(shape, camera, self.settings())
        else:
            mask = apply_mask_exclusions(base, camera, self.settings())
            self.preview_status.setText(tr('Combined preview: RF-DETR people, fixed shapes, and fisheye cutoff. White keeps pixels; black excludes them.'))
        self.canvas.set_mask(mask)

    def _start_detection(self):
        if self._preview_process is not None:
            return
        root = get_app_root()
        executable = (resolve_vulkan_executable() if self.backend == 'vulkan'
                      else resolve_masker_executable())
        try:
            model = resolve_model(self.model_path)
        except FileNotFoundError as exc:
            self.preview_status.setText(tr('RF-DETR model unavailable: {error}').format(error=str(exc)))
            return
        inference_model = model
        shaders = None
        if self.backend == 'vulkan':
            inference_model = resolve_vulkan_schedule(model)
            shaders = resolve_vulkan_shaders_dir(executable)
            backend_ready = inference_model is not None and shaders is not None
        else:
            backend_ready = resolve_runtime_dir() is not None
        if executable is None or model is None or not model_is_ready(model) or not backend_ready:
            self.preview_status.setText(tr('Fixed-area preview only. Use “Download model RF-DETR” on the main screen to prepare the selected mask backend.'))
            return
        key = self._preview_key()
        camera, source = key
        parent = self.dataset_dir if self.dataset_dir and self.dataset_dir.is_dir() else get_config_dir()
        parent.mkdir(parents=True, exist_ok=True)
        self._preview_temp = tempfile.TemporaryDirectory(prefix='raven-mask-preview-', dir=parent)
        temporary = Path(self._preview_temp.name)
        input_dir = temporary / 'images' / camera
        input_dir.mkdir(parents=True)
        (temporary / 'masks').mkdir()
        try:
            shutil.copy2(source, input_dir / source.name)
        except OSError as exc:
            self._preview_temp.cleanup()
            self._preview_temp = None
            self.preview_status.setText(tr('Cannot prepare mask preview: {error}').format(error=str(exc)))
            return
        process = QProcess(self)
        self._preview_process = process
        process.finished.connect(lambda code, _status: self._detection_finished(process, key, code))
        process.setProgram(str(executable))
        arguments = ['--input', str(temporary / 'images'), '--output', str(temporary / 'masks'),
                     '--model', str(inference_model), '--threshold', '0.5', '--margin', '0.03']
        environment = QProcessEnvironment.systemEnvironment()
        if self.backend == 'vulkan':
            arguments.extend(['--shaders', str(shaders)])
        else:
            cache = Path(os.environ['RAVEN_RFDETR_CACHE']) if os.environ.get('RAVEN_RFDETR_CACHE') else (
                get_config_dir() / 'rfdetr-cache' if getattr(sys, 'frozen', False) else root / 'models')
            cache.mkdir(parents=True, exist_ok=True)
            arguments.extend(['--cache-dir', str(cache)])
            environment.insert('PATH', masker_environment(executable).get('PATH', ''))
        process.setArguments(arguments)
        process.setProcessEnvironment(environment)
        process.start()
        if not process.waitForStarted(1000):
            error = process.errorString()
            self._stop_detection()
            self.preview_status.setText(tr('RF-DETR preview could not start: {error}').format(error=error))
            return
        self.preview_status.setText(tr('Showing fixed areas while RF-DETR detects people for this frame...'))

    def _detection_finished(self, process, key, code):
        if process is not self._preview_process:
            return
        camera, source = key
        path = Path(self._preview_temp.name) / 'masks' / camera / (source.name + '.png')
        mask = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE) if code == 0 and path.is_file() else None
        error = bytes(process.readAllStandardError()).decode(errors='replace').strip()
        self._preview_process = None
        process.deleteLater()
        self._preview_temp.cleanup()
        self._preview_temp = None
        if mask is None:
            self.preview_status.setText(tr('RF-DETR preview failed: {error}').format(error=error or str(code)))
            return
        if mask.shape != (self.canvas.image.height(), self.canvas.image.width()):
            self.preview_status.setText(tr('RF-DETR preview size does not match the frame.'))
            return
        self._person_masks[key] = mask
        if self.canvas.show_mask and key == self._preview_key():
            self._refresh_mask_preview()

    def _stop_detection(self):
        process = self._preview_process
        self._preview_process = None
        if process is not None:
            if process.state() != QProcess.ProcessState.NotRunning:
                process.kill()
                process.waitForFinished(3000)
            process.deleteLater()
        if self._preview_temp is not None:
            self._preview_temp.cleanup()
            self._preview_temp = None

    def done(self, result):
        self._stop_detection()
        self._stop_frame_extraction()
        if self._video_preview_temp is not None:
            self._video_preview_temp.cleanup()
            self._video_preview_temp = None
            self._video_preview_root = None
        if self._aux_video_preview_temp is not None:
            self._aux_video_preview_temp.cleanup()
            self._aux_video_preview_temp = None
        if self._frames_temp is not None:
            self._frames_temp.cleanup()
            self._frames_temp = None
        super().done(result)

    def accept(self):
        if self._frame_extraction_process is not None or self._frame_extraction_timer.isActive():
            self.preview_status.setText(tr('Wait for frame extraction to finish before saving mask settings.'))
            return
        super().accept()

    def settings(self):
        self.config['fisheye_border_percent_by_camera'][self.camera_combo.currentText()] = self.border_spin.value()
        return normalize_mask_config(self.config)
