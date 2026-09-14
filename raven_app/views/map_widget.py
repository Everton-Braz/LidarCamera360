"""Interactive desktop satellite and street map widget with free tile providers.

Pure PyQt6 implementation using QNetworkAccessManager and QPainter.
Supports Esri/ArcGIS World Imagery (Aerial Satellite), OpenStreetMap, OpenTopoMap,
and overlays georeferenced point cloud footprints, camera poses, and GPS tracks.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import (
    QPoint, QPointF, QRect, QRectF, QSize, Qt, QUrl, pyqtSignal,
)
from PyQt6.QtGui import (
    QColor, QFont, QImage, QPainter, QPen, QPolygonF, QBrush,
)
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkRequest, QNetworkReply
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QToolButton, QLabel,
)
from qfluentwidgets import ComboBox, ToolButton, FluentIcon, CaptionLabel

TILE_PROVIDERS = {
    "ArcGIS Aerial (Esri Satellite)": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "max_zoom": 19,
        "attribution": "Esri World Imagery",
    },
    "OpenStreetMap (Standard)": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "max_zoom": 19,
        "attribution": "© OpenStreetMap",
    },
    "OpenTopoMap (Terrain)": {
        "url": "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        "max_zoom": 17,
        "attribution": "© OpenTopoMap",
    },
    "CartoDB Positron": {
        "url": "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png",
        "max_zoom": 19,
        "attribution": "© CARTO",
    },
}


def latlon_to_tile_xy(lat: float, lon: float, zoom: int) -> Tuple[float, float]:
    """Convert WGS 84 (lat, lon) to continuous tile coordinates at given zoom level."""
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_xy_to_latlon(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates back to WGS 84 (lat, lon)."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


class TileMapWidget(QWidget):
    """Interactive tile map widget displaying point cloud footprints and cameras over satellite imagery."""
    center_moved = pyqtSignal(float, float)
    footprint_nudged = pyqtSignal(float, float)  # delta_easting_m, delta_northing_m

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 260)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        # Map state
        self.center_lat = -3.695812
        self.center_lon = -38.599353
        self.zoom = 18
        self.provider_name = "ArcGIS Aerial (Esri Satellite)"

        # Data overlays
        self.footprint_latlons: List[Tuple[float, float]] = []
        self.camera_points: List[Dict[str, Any]] = []
        self.gps_points: List[Tuple[float, float]] = []
        self.heading_deg: float = 0.0

        # Tile cache (tile_key -> QImage)
        self._tile_cache: OrderedDict[str, QImage] = OrderedDict()
        self._max_cached_tiles = 300
        self._pending_requests: set = set()

        # Network manager
        self._nam = QNetworkAccessManager(self)
        self._nam.finished.connect(self._on_tile_reply)

        # Mouse interaction
        self._dragging_map = False
        self._last_mouse_pos = QPoint()
        self._interactive_shift_active = False

        self._init_controls()

    def _init_controls(self):
        """Overlay controls for layer switching and zoom buttons."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Top bar with layer switcher
        top_bar = QHBoxLayout()
        self.layer_combo = ComboBox(self)
        for name in TILE_PROVIDERS.keys():
            self.layer_combo.addItem(name)
        self.layer_combo.setCurrentText(self.provider_name)
        self.layer_combo.currentTextChanged.connect(self._on_provider_changed)
        self.layer_combo.setFixedWidth(240)
        top_bar.addWidget(self.layer_combo)
        top_bar.addStretch(1)

        # Zoom in/out buttons
        self.btn_zoom_in = ToolButton(self)
        self.btn_zoom_in.setIcon(FluentIcon.ZOOM_IN)
        self.btn_zoom_in.setFixedSize(32, 32)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        top_bar.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = ToolButton(self)
        self.btn_zoom_out.setIcon(FluentIcon.ZOOM_OUT)
        self.btn_zoom_out.setFixedSize(32, 32)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        top_bar.addWidget(self.btn_zoom_out)

        layout.addLayout(top_bar)
        layout.addStretch(1)

        # Bottom attribution bar
        bottom_bar = QHBoxLayout()
        self.attribution_label = QLabel(self)
        self.attribution_label.setStyleSheet("color: rgba(255, 255, 255, 0.85); background: rgba(0, 0, 0, 0.6); padding: 2px 6px; border-radius: 4px; font-size: 10px;")
        self._update_attribution()
        bottom_bar.addWidget(self.attribution_label)
        bottom_bar.addStretch(1)
        layout.addLayout(bottom_bar)

    def _update_attribution(self):
        prov = TILE_PROVIDERS.get(self.provider_name, {})
        self.attribution_label.setText(f"Map: {prov.get('attribution', '')}")

    def _on_provider_changed(self, name: str):
        if name in TILE_PROVIDERS:
            self.provider_name = name
            self._update_attribution()
            self._tile_cache.clear()
            self.update()

    def set_center(self, lat: float, lon: float, zoom: Optional[int] = None):
        """Center map at given coordinates."""
        self.center_lat = float(lat)
        self.center_lon = float(lon)
        if zoom is not None:
            max_z = TILE_PROVIDERS[self.provider_name]["max_zoom"]
            self.zoom = max(1, min(max_z, zoom))
        self.update()

    def zoom_in(self):
        max_z = TILE_PROVIDERS[self.provider_name]["max_zoom"]
        if self.zoom < max_z:
            self.zoom += 1
            self.update()

    def zoom_out(self):
        if self.zoom > 1:
            self.zoom -= 1
            self.update()

    def set_footprint(self, latlons: List[Tuple[float, float]], heading_deg: float = 0.0):
        """Update point cloud bounding polygon footprint."""
        self.footprint_latlons = list(latlons)
        self.heading_deg = float(heading_deg)
        if self.footprint_latlons:
            # Auto center on footprint
            c_lat = np.mean([p[0] for p in self.footprint_latlons])
            c_lon = np.mean([p[1] for p in self.footprint_latlons])
            self.center_lat = float(c_lat)
            self.center_lon = float(c_lon)
        self.update()

    def set_cameras(self, cameras: List[Dict[str, Any]]):
        """Update COLMAP camera positions."""
        self.camera_points = list(cameras)
        self.update()

    def set_gps_track(self, records: List[Dict[str, Any]]):
        """Update GPS track."""
        self.gps_points = [(r["latitude"], r["longitude"]) for r in records if "latitude" in r and "longitude" in r]
        self.update()

    # Tile fetching & network
    def _tile_key(self, z: int, x: int, y: int) -> str:
        return f"{self.provider_name}:{z}:{x}:{y}"

    def _fetch_tile(self, z: int, x: int, y: int):
        key = self._tile_key(z, x, y)
        if key in self._tile_cache or key in self._pending_requests:
            return

        prov = TILE_PROVIDERS.get(self.provider_name, {})
        url_tmpl = prov.get("url", "")
        # Format URL (handling {s} if present)
        url_str = url_tmpl.format(z=z, x=x, y=y, s="a")

        req = QNetworkRequest(QUrl(url_str))
        req.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, "LidarCamera360/1.0 (GIS Desktop)")
        reply = self._nam.get(req)
        reply.setProperty("tile_key", key)
        self._pending_requests.add(key)

    def _on_tile_reply(self, reply: QNetworkReply):
        key = reply.property("tile_key")
        if key in self._pending_requests:
            self._pending_requests.remove(key)

        if reply.error() == QNetworkReply.NetworkError.NoError:
            data = reply.readAll()
            img = QImage()
            if img.loadFromData(data):
                if len(self._tile_cache) >= self._max_cached_tiles:
                    self._tile_cache.popitem(last=False)
                self._tile_cache[key] = img
                self.update()
        reply.deleteLater()

    # Coordinate conversions between screen pixels and geographical coordinates
    def _geo_to_pixel(self, lat: float, lon: float) -> QPointF:
        center_tx, center_ty = latlon_to_tile_xy(self.center_lat, self.center_lon, self.zoom)
        tx, ty = latlon_to_tile_xy(lat, lon, self.zoom)
        dx = (tx - center_tx) * 256.0
        dy = (ty - center_ty) * 256.0
        return QPointF(self.width() / 2.0 + dx, self.height() / 2.0 + dy)

    def _pixel_to_geo(self, px: float, py: float) -> Tuple[float, float]:
        center_tx, center_ty = latlon_to_tile_xy(self.center_lat, self.center_lon, self.zoom)
        dx = px - self.width() / 2.0
        dy = py - self.height() / 2.0
        tx = center_tx + dx / 256.0
        ty = center_ty + dy / 256.0
        return tile_xy_to_latlon(tx, ty, self.zoom)

    # Mouse handling
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_map = True
            self._last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging_map:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            # Pan map by moving center in tile space
            dx_tiles = delta.x() / 256.0
            dy_tiles = delta.y() / 256.0
            ctx, cty = latlon_to_tile_xy(self.center_lat, self.center_lon, self.zoom)
            new_lat, new_lon = tile_xy_to_latlon(ctx - dx_tiles, cty - dy_tiles, self.zoom)
            self.center_lat = new_lat
            self.center_lon = new_lon
            self.center_moved.emit(self.center_lat, self.center_lon)
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_map = False
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        deg = event.angleDelta().y()
        if deg > 0:
            self.zoom_in()
        elif deg < 0:
            self.zoom_out()

    def mouseDoubleClickEvent(self, event):
        # Center map on clicked position
        lat, lon = self._pixel_to_geo(event.pos().x(), event.pos().y())
        self.set_center(lat, lon)
        self.center_moved.emit(lat, lon)

    # Paint event
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # 1. Fill background with dark map color
        painter.fillRect(self.rect(), QColor("#111827"))

        # 2. Render visible map tiles
        w, h = self.width(), self.height()
        ctx, cty = latlon_to_tile_xy(self.center_lat, self.center_lon, self.zoom)

        num_tiles_x = int(math.ceil(w / 256.0)) + 2
        num_tiles_y = int(math.ceil(h / 256.0)) + 2

        start_tile_x = int(math.floor(ctx)) - num_tiles_x // 2
        start_tile_y = int(math.floor(cty)) - num_tiles_y // 2

        max_tile_idx = (1 << self.zoom) - 1

        for tx in range(start_tile_x, start_tile_x + num_tiles_x + 1):
            if tx < 0 or tx > max_tile_idx:
                continue
            for ty in range(start_tile_y, start_tile_y + num_tiles_y + 1):
                if ty < 0 or ty > max_tile_idx:
                    continue

                screen_x = (tx - ctx) * 256.0 + w / 2.0
                screen_y = (ty - cty) * 256.0 + h / 2.0

                key = self._tile_key(self.zoom, tx, ty)
                if key in self._tile_cache:
                    painter.drawImage(int(screen_x), int(screen_y), self._tile_cache[key])
                else:
                    self._fetch_tile(self.zoom, tx, ty)
                    # Draw subtle placeholder grid
                    painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
                    painter.drawRect(int(screen_x), int(screen_y), 256, 256)

        # 3. Draw GPS Trajectory Track (golden line)
        if len(self.gps_points) > 1:
            pen_gps = QPen(QColor("#f59e0b"), 3)
            pen_gps.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen_gps)
            pts_screen = [self._geo_to_pixel(p[0], p[1]) for p in self.gps_points]
            for i in range(len(pts_screen) - 1):
                painter.drawLine(pts_screen[i], pts_screen[i + 1])

        # 4. Draw Point Cloud Bounding Footprint (cyan polygon with semi-transparent fill)
        if len(self.footprint_latlons) >= 3:
            poly = QPolygonF([self._geo_to_pixel(p[0], p[1]) for p in self.footprint_latlons])
            brush_footprint = QBrush(QColor(0, 240, 255, 75))
            pen_footprint = QPen(QColor("#00f0ff"), 2.5)
            painter.setBrush(brush_footprint)
            painter.setPen(pen_footprint)
            painter.drawPolygon(poly)

            # Draw heading orientation arrow from center
            c_x = np.mean([p.x() for p in poly])
            c_y = np.mean([p.y() for p in poly])
            c_pt = QPointF(c_x, c_y)
            arrow_len = 35.0
            # Heading 0 deg = North (-Y in pixel coordinates)
            rad_h = math.radians(self.heading_deg)
            arrow_dx = arrow_len * math.sin(rad_h)
            arrow_dy = -arrow_len * math.cos(rad_h)
            tip_pt = QPointF(c_x + arrow_dx, c_y + arrow_dy)

            painter.setPen(QPen(QColor("#ef4444"), 3))
            painter.drawLine(c_pt, tip_pt)
            painter.setBrush(QBrush(QColor("#ef4444")))
            painter.drawEllipse(tip_pt, 4, 4)

        # 5. Draw COLMAP camera points
        if self.camera_points:
            pen_cam = QPen(QColor("#ef4444"), 1)
            brush_cam = QBrush(QColor("#f87171"))
            painter.setPen(pen_cam)
            painter.setBrush(brush_cam)
            for c in self.camera_points[::3]:  # sample every 3rd camera for fast redraw
                pt = self._geo_to_pixel(c["latitude"], c["longitude"])
                painter.drawEllipse(pt, 3, 3)

        # 6. Scale bar (in meters)
        meters_per_pixel = 156543.03392 * math.cos(math.radians(self.center_lat)) / (2.0 ** self.zoom)
        bar_len_px = 80
        bar_meters = bar_len_px * meters_per_pixel
        bar_label = f"{bar_meters:.1f} m" if bar_meters < 1000 else f"{bar_meters/1000:.1f} km"

        sb_x, sb_y = 16, self.height() - 36
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(sb_x, sb_y, sb_x + bar_len_px, sb_y)
        painter.drawLine(sb_x, sb_y - 4, sb_x, sb_y + 4)
        painter.drawLine(sb_x + bar_len_px, sb_y - 4, sb_x + bar_len_px, sb_y + 4)
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(sb_x + 6, sb_y - 6, bar_label)

        painter.end()
