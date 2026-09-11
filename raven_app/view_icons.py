"""Vector 3D Isometric ViewCube icons with highlighted faces for CAD/LiDAR navigation."""
from __future__ import annotations

import math
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

PRESETS = ('iso', 'top', 'bottom', 'front', 'back', 'left', 'right')
_ICON_CACHE: dict[tuple[str, int], QIcon] = {}


def create_cube_icon(view: str, size: int = 36) -> QIcon:
    """Generate a high-DPI 3D isometric cube icon with the selected face highlighted."""
    view = view.lower()
    key = (view, size)
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]

    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    cx, cy = size / 2.0, size / 2.0
    r = size * 0.38
    dx = r * math.cos(math.radians(30))
    dy = r * math.sin(math.radians(30))

    # 7 key vertices of the isometric projection:
    C = QPointF(cx, cy)
    T = QPointF(cx, cy - r)
    B = QPointF(cx, cy + r)
    TL = QPointF(cx - dx, cy - dy)
    TR = QPointF(cx + dx, cy - dy)
    BL = QPointF(cx - dx, cy + dy)
    BR = QPointF(cx + dx, cy + dy)

    top_poly = QPolygonF([TL, T, TR, C])
    front_poly = QPolygonF([TL, C, B, BL])   # Front face (Y-)
    right_poly = QPolygonF([C, TR, BR, B])   # Right face (X+)

    accent = QColor('#00a2ff')
    accent_dark = QColor('#0078d4')
    accent_light = QColor('#38b6ff')
    base_top = QColor(200, 212, 224, 210)
    base_front = QColor(160, 175, 190, 210)
    base_right = QColor(128, 142, 156, 210)
    wire_pen = QPen(QColor(45, 60, 75, 230), 1.4)
    accent_pen = QPen(QColor(255, 255, 255, 240), 1.6)

    if view == 'iso':
        p.setPen(wire_pen)
        p.setBrush(accent_light); p.drawPolygon(top_poly)
        p.setBrush(accent); p.drawPolygon(front_poly)
        p.setBrush(accent_dark); p.drawPolygon(right_poly)

    elif view == 'top':
        p.setPen(wire_pen)
        p.setBrush(base_front); p.drawPolygon(front_poly)
        p.setBrush(base_right); p.drawPolygon(right_poly)
        p.setPen(accent_pen)
        p.setBrush(accent); p.drawPolygon(top_poly)

    elif view == 'front':
        p.setPen(wire_pen)
        p.setBrush(base_top); p.drawPolygon(top_poly)
        p.setBrush(base_right); p.drawPolygon(right_poly)
        p.setPen(accent_pen)
        p.setBrush(accent); p.drawPolygon(front_poly)

    elif view == 'right':
        p.setPen(wire_pen)
        p.setBrush(base_top); p.drawPolygon(top_poly)
        p.setBrush(base_front); p.drawPolygon(front_poly)
        p.setPen(accent_pen)
        p.setBrush(accent); p.drawPolygon(right_poly)

    elif view == 'bottom':
        # Base cube wireframe
        p.setPen(wire_pen)
        p.setBrush(base_top); p.drawPolygon(top_poly)
        p.setBrush(base_front); p.drawPolygon(front_poly)
        p.setBrush(base_right); p.drawPolygon(right_poly)
        # Highlighted bottom indicator face beneath
        p.setPen(accent_pen)
        p.setBrush(accent)
        bot_poly = QPolygonF([BL, B, BR, QPointF(cx, cy + r * 1.36)])
        p.drawPolygon(bot_poly)

    elif view == 'back':
        # Base cube wireframe
        p.setPen(wire_pen)
        p.setBrush(base_top); p.drawPolygon(top_poly)
        p.setBrush(base_front); p.drawPolygon(front_poly)
        p.setBrush(base_right); p.drawPolygon(right_poly)
        # Highlighted back indicator face extending behind
        p.setPen(accent_pen)
        p.setBrush(accent)
        back_poly = QPolygonF([T, TR, QPointF(cx + dx * 1.25, cy - dy * 1.3), QPointF(cx + dx * 0.25, cy - r * 1.3)])
        p.drawPolygon(back_poly)

    elif view == 'left':
        # Base cube wireframe
        p.setPen(wire_pen)
        p.setBrush(base_top); p.drawPolygon(top_poly)
        p.setBrush(base_front); p.drawPolygon(front_poly)
        p.setBrush(base_right); p.drawPolygon(right_poly)
        # Highlighted left indicator face extending behind left
        p.setPen(accent_pen)
        p.setBrush(accent)
        left_poly = QPolygonF([TL, T, QPointF(cx - dx * 0.25, cy - r * 1.3), QPointF(cx - dx * 1.25, cy - dy * 1.3)])
        p.drawPolygon(left_poly)

    else:
        # Generic fallback
        p.setPen(wire_pen)
        p.setBrush(base_top); p.drawPolygon(top_poly)
        p.setBrush(base_front); p.drawPolygon(front_poly)
        p.setBrush(base_right); p.drawPolygon(right_poly)

    p.end()
    icon = QIcon(pix)
    _ICON_CACHE[key] = icon
    return icon
