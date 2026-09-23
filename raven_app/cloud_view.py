"""GPU point-cloud swipe comparison with a shared orthographic camera."""
import csv
import math
from itertools import product
import numpy as np
from raven_app.i18n import tr
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush, QMatrix4x4, QSurfaceFormat, QLinearGradient, QFont, QPolygonF, QImage, QPainterPath, QCursor
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLFunctions_2_1, QOpenGLFramebufferObject
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from raven_app.viewer_geometry import camera_matrix, project_points, measurement_value


PRESETS = {
    'iso': (-55., 25.),
    'top': (-90., 89.9999),
    'bottom': (-90., -89.9999),
    'front': (-90., 0.),
    'back': (90., 0.),
    'left': (180., 0.),
    'right': (0., 0.),
}

_LUT_CACHE: dict[str, np.ndarray] = {}


def _generate_lut(name: str) -> np.ndarray:
    """Generate a 256x3 float32 colormap lookup table."""
    try:
        import matplotlib
        cmap_name = {'rainbow': 'jet', 'heat': 'hot', 'grayscale': 'gray'}.get(name, name)
        cmap = matplotlib.colormaps.get(cmap_name)
        if cmap is not None:
            return cmap(np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    except Exception:
        pass

    x = np.linspace(0.0, 1.0, 256)
    if name == 'grayscale':
        return np.repeat(x[:, None], 3, axis=1).astype(np.float32)
    if name == 'heat':
        r = np.clip(x * 3.0, 0.0, 1.0)
        g = np.clip(x * 3.0 - 1.0, 0.0, 1.0)
        b = np.clip(x * 3.0 - 2.0, 0.0, 1.0)
        return np.column_stack([r, g, b]).astype(np.float32)
    if name == 'rainbow':
        r = np.clip(1.5 - np.abs(x * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(x * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(x * 4.0 - 1.0), 0.0, 1.0)
        return np.column_stack([r, g, b]).astype(np.float32)

    # Google Turbo polynomial approximation (perceptually uniform)
    k_r = [0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943]
    k_g = [0.09140261, 2.19418839, 4.84296658, -14.18503327, 4.27729857, 2.82956604]
    k_b = [0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973]
    r = np.clip(np.polyval(k_r[::-1], x), 0.0, 1.0)
    g = np.clip(np.polyval(k_g[::-1], x), 0.0, 1.0)
    b = np.clip(np.polyval(k_b[::-1], x), 0.0, 1.0)
    return np.column_stack([r, g, b]).astype(np.float32)


def get_lut(name: str) -> np.ndarray:
    name = name.lower()
    if name not in _LUT_CACHE:
        _LUT_CACHE[name] = _generate_lut(name)
    return _LUT_CACHE[name]


class CloudView(QOpenGLWidget):
    measurement_added = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    split_changed = pyqtSignal(float)
    point_size_changed = pyqtSignal(float)
    view_preset_changed = pyqtSignal(str)
    color_mode_changed = pyqtSignal(str)
    colormap_changed = pyqtSignal(str)
    clipping_box_mode_changed = pyqtSignal(bool)
    clipping_bounds_changed = pyqtSignal(object, object, bool)
    transform_mode_changed = pyqtSignal(str)
    cloud_transformed = pyqtSignal(int, float, object)
    export_clipped_requested = pyqtSignal()
    MAX_DRAW_POINTS = 5_000_000

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat(); fmt.setVersion(2, 1); fmt.setDepthBufferSize(24); fmt.setSamples(0)
        self.setFormat(fmt)
        self.setMinimumSize(400, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.clouds = [None, None]; self._buffers = [None, None]; self._indices = [None, None]
        self._dirty = set(); self._gl = None; self._program = None; self.error = ''
        self.origin = np.zeros(3); self.target = np.zeros(3)
        self.yaw, self.elevation, self.half_height, self.scene_radius = -55., 25., 5., 5.
        self.split = .5; self.point_size = 2.; self.mode = 'navigate'
        self.color_modes = ['rgb', 'rgb']
        self.colormaps = ['turbo', 'viridis']
        self.bg_color = QColor('#0b1017')
        self.clipping_enabled = False
        self.clipping_box_mode = False
        self.clipping_box_visible = False
        self.clip_invert = False
        self.clip_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.clip_extents = np.array([1e9, 1e9, 1e9], dtype=np.float32)
        self.clip_yaw = 0.0
        self.clip_min = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
        self.clip_max = np.array([1e9, 1e9, 1e9], dtype=np.float32)
        self._active_clipping_handle: str | None = None
        self._hovered_clipping_handle: str | None = None
        self._clipping_drag_start_pos = None
        self._clipping_drag_start_center = None
        self._clipping_drag_start_extents = None
        self._clipping_drag_start_yaw = 0.0
        self._clipping_drag_start_bounds = None
        self._clipping_box_screen_center = (0.0, 0.0)
        self._clipping_handle_screens: dict = {}
        self._clipping_hud_rects: dict = {}
        self._clipping_min_badge_rects: dict = {}
        self._hovered_hud_btn: str | None = None

        self.transform_mode: str | None = None  # None, 'translate', 'rotate'
        self.active_transform_slot: int = 0
        self._active_gizmo_axis: str | None = None  # 'x', 'y', 'z', 'free', 'yaw'
        self._hovered_gizmo_axis: str | None = None
        self._transform_drag_start_pos = None
        self._transform_drag_start_val = None
        self._gizmo_handle_screens: dict = {}

        self._scalar_ranges: dict[str, tuple[float, float]] = {}
        self.current_preset = 'iso'
        self.measurements = []; self.pending = []; self._last = None; self._press = None; self._drag_split = False
        self._dragged = False
        self._original_points = [None, None]
        self._adjustments = [(0.0, np.zeros(3)), (0.0, np.zeros(3))]
        self.basemap_enabled = False
        self.basemap_provider = 'satellite'
        from raven_app.basemap_layer import BaseMapLayer, MapTileRenderer
        self.basemap_layer = BaseMapLayer(self)
        self.basemap_layer.changed.connect(self.update)
        self.basemap_layer.status.connect(self.status_changed.emit)
        self._map_renderer = MapTileRenderer()

    @property
    def color_mode(self) -> str:
        return self.color_modes[0]

    @property
    def colormap(self) -> str:
        return self.colormaps[0]

    def set_cloud(self, slot, cloud):
        # Rebase again when replacing the only loaded cloud (e.g. local -> UTM).
        # GPU vertices stay near zero while measurements retain original doubles.
        first = self.clouds[1 - slot] is None
        if first:
            lo, hi = cloud.points.min(axis=0), cloud.points.max(axis=0)
            self.origin = lo + (hi-lo)*.5
        self.clouds[slot] = cloud
        self._original_points[slot] = cloud.points.copy()
        self._adjustments[slot] = (0.0, np.zeros(3, dtype=np.float64))
        self._scalar_ranges.clear()
        self.error = ''
        step = max(1, math.ceil(len(cloud.points)/self.MAX_DRAW_POINTS))
        self._indices[slot] = np.arange(0, len(cloud.points), step, dtype=np.int64)
        self._dirty.update(i for i, c in enumerate(self.clouds) if c is not None)
        if self.measurements or self.pending:
            self.clear_measurements()
        if first: self.fit_all()
        else:
            bounds = np.concatenate([c.points[[np.argmin(c.points[:, j]), np.argmax(c.points[:, j])]] for c in self.clouds if c is not None for j in range(3)])
            self.scene_radius = max(self.scene_radius, float(np.linalg.norm(bounds-self.origin-self.target, axis=1).max()))
        suffix = f'; preview sampled to {len(self._indices[slot]):,} points' if step > 1 else ''
        self.status_changed.emit(f"{'AB'[slot]}: {cloud.path.name} | {len(cloud.points):,} valid points{suffix}. Coordinates unchanged.")
        if slot == 0 or self.clouds[0] is None:
            self.basemap_layer.configure(cloud)
        self.update()

    def set_point_size(self, size: float):
        new_size = float(np.clip(size, 0.1, 20.0))
        if abs(self.point_size - new_size) > 1e-4:
            self.point_size = new_size
            self.point_size_changed.emit(self.point_size)
            self.update()

    def clear_cloud(self, slot):
        """Remove a comparison cloud and release its GPU buffer on the next paint."""
        self.clouds[slot] = None
        self._original_points[slot] = None
        self._adjustments[slot] = (0.0, np.zeros(3, dtype=np.float64))
        self._indices[slot] = None
        self._dirty.add(slot)
        self._scalar_ranges.clear()
        self.clear_measurements()
        other = self.clouds[1 - slot]
        if other is not None:
            self.fit_all()
        self.basemap_layer.configure(other)
        self.update()

    def set_basemap_enabled(self, enabled: bool):
        self.basemap_enabled = bool(enabled)
        self.basemap_layer.set_enabled(enabled)
        self.update()

    def set_basemap_provider(self, provider: str):
        provider = str(provider).lower()
        if provider in ('satellite', 'street'):
            self.basemap_provider = provider
            self.basemap_layer.set_provider(provider)
            self.update()

    def apply_cloud_adjustment(self, slot: int, yaw_deg: float, translation_xyz, is_preview: bool = False, refit: bool = False):
        """Preview or apply an adjustment relative to the loaded original cloud."""
        if not 0 <= int(slot) < 2 or self.clouds[slot] is None or self._original_points[slot] is None:
            return
        slot = int(slot)
        original = self._original_points[slot]
        center = original.mean(axis=0)
        angle = math.radians(float(yaw_deg))
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        shift = np.asarray(translation_xyz, dtype=np.float64).reshape(3)
        if not np.isfinite([yaw_deg, *shift]).all():
            raise ValueError('Adjustment values must be finite')

        self._adjustments[slot] = (float(yaw_deg), shift.copy())

        if is_preview:
            # Fast preview during active dragging: only transform the sampled indices used for GPU drawing
            ids = self._indices[slot]
            if ids is not None and len(ids) > 0:
                sampled_orig = original[ids]
                self.clouds[slot].points[ids] = (sampled_orig - center) @ rot.T + center + shift
        else:
            # Full transform on release or explicit apply
            adjusted = np.empty_like(original)
            for start in range(0, len(original), 500000):
                adjusted[start:start+500000] = (original[start:start+500000] - center) @ rot.T + center + shift
            self.clouds[slot].points = adjusted

        self.clear_measurements()
        self._scalar_ranges.clear()
        self._dirty.add(slot)
        if refit:
            self.fit_all()
        self.update()

    def reset_cloud_adjustment(self, slot: int):
        if not 0 <= int(slot) < 2 or self.clouds[int(slot)] is None or self._original_points[int(slot)] is None:
            return
        slot = int(slot)
        self.clouds[slot].points = self._original_points[slot].copy()
        self._adjustments[slot] = (0.0, np.zeros(3, dtype=np.float64))
        self.clear_measurements()
        self._scalar_ranges.clear()
        self._dirty.add(slot)
        self.update()

    def set_background_color(self, color: QColor | str):
        self.bg_color = QColor(color)
        self.update()

    def set_clipping_enabled(self, enabled: bool):
        """Toggle point cloud clipping shader effect."""
        self.clipping_enabled = bool(enabled)
        if not self.clipping_enabled:
            self.clipping_box_mode = False
            self.clipping_box_visible = False
            self._active_clipping_handle = None
            self._hovered_clipping_handle = None
            self.clipping_box_mode_changed.emit(False)
        self.update()

    def set_clipping_bounds(self, c_min: np.ndarray, c_max: np.ndarray, enabled: bool = True):
        self.clip_min = np.asarray(c_min, dtype=np.float32)
        self.clip_max = np.asarray(c_max, dtype=np.float32)
        self.clip_center = (self.clip_min + self.clip_max) * 0.5
        self.clip_extents = np.maximum((self.clip_max - self.clip_min) * 0.5, 0.05)
        self.clipping_enabled = bool(enabled)
        self.clipping_bounds_changed.emit(self.clip_min, self.clip_max, self.clipping_enabled)
        self.update()

    def set_clip_box(self, center: np.ndarray, extents: np.ndarray, yaw_deg: float = 0.0, enabled: bool | None = None):
        self.clip_center = np.asarray(center, dtype=np.float32)
        self.clip_extents = np.maximum(np.asarray(extents, dtype=np.float32), 0.05)
        self.clip_yaw = float(yaw_deg)
        self.clip_min = self.clip_center - self.clip_extents
        self.clip_max = self.clip_center + self.clip_extents
        if enabled is not None:
            self.clipping_enabled = bool(enabled)
        self.clipping_bounds_changed.emit(self.clip_min, self.clip_max, self.clipping_enabled)
        self.update()

    def set_clip_yaw(self, yaw_deg: float):
        self.clip_yaw = float(yaw_deg)
        self.clipping_bounds_changed.emit(self.clip_min, self.clip_max, self.clipping_enabled)
        self.update()

    def set_clipping_box_mode(self, enabled: bool):
        self.clipping_box_mode = bool(enabled)
        self.clipping_box_visible = bool(enabled)
        if self.clipping_box_mode:
            if np.any(self.clip_extents > 1e8) or self.clip_extents[0] <= 0:
                self.reset_clipping_to_clouds()
            else:
                self.clipping_enabled = True
            self.status_changed.emit(tr("3D Clipping Box active. Drag arrows to resize, rings to rotate, or center to move."))
        else:
            self._active_clipping_handle = None
            self._hovered_clipping_handle = None
        self.clipping_box_mode_changed.emit(self.clipping_box_mode)
        self.update()

    def set_clipping_box_visible(self, visible: bool):
        """Toggle 3D box wireframe/handles/HUD visibility while keeping shader slice active."""
        self.clipping_box_visible = bool(visible)
        self.clipping_box_mode = bool(visible)
        self._active_clipping_handle = None
        self._hovered_clipping_handle = None
        self.clipping_box_mode_changed.emit(self.clipping_box_mode)
        self.update()

    def align_clipping_box_to_cloud(self):
        """Snap clipping box yaw to the active transform cloud's rotation angle."""
        slot = self.active_transform_slot
        if 0 <= slot < len(self._adjustments):
            yaw, _ = self._adjustments[slot]
            self.set_clip_yaw(yaw)
            self.status_changed.emit(tr(f"Clipping box aligned to Cloud {'AB'[slot]} (Yaw: {yaw:+.2f}°)."))

    def set_clip_invert(self, invert: bool):
        self.clip_invert = bool(invert)
        self.update()

    def reset_clipping_to_clouds(self):
        valid = [c for c in self.clouds if c is not None and c.points is not None and len(c.points) > 0]
        if not valid:
            self.clip_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.clip_extents = np.array([10.0, 10.0, 10.0], dtype=np.float32)
            self.clip_yaw = 0.0
            self.clip_min = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
            self.clip_max = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        else:
            all_lo = np.min([c.points.min(axis=0) for c in valid], axis=0)
            all_hi = np.max([c.points.max(axis=0) for c in valid], axis=0)
            margin = np.maximum((all_hi - all_lo) * 0.02, 0.2)
            center = (all_lo + all_hi) * 0.5
            extents = (all_hi - all_lo) * 0.5 + margin
            self.clip_center = np.asarray(center, dtype=np.float32)
            self.clip_extents = np.asarray(extents, dtype=np.float32)
            self.clip_yaw = 0.0
            self.clip_min = np.asarray(all_lo - margin, dtype=np.float32)
            self.clip_max = np.asarray(all_hi + margin, dtype=np.float32)
        self.clipping_enabled = True
        self.clipping_box_visible = True
        self.clipping_box_mode = True
        self.clipping_bounds_changed.emit(self.clip_min, self.clip_max, True)
        self.clipping_box_mode_changed.emit(True)
        self.update()

    def get_clipped_points(self, slot: int = 0) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return (points, colors, intensities) inside current clipping box for the specified slot."""
        if not 0 <= slot < 2 or self.clouds[slot] is None:
            return np.empty((0, 3)), None, None
        cloud = self.clouds[slot]
        pts = cloud.points
        center = self.clip_center
        ext = self.clip_extents
        rad = math.radians(self.clip_yaw)
        c, s = math.cos(rad), math.sin(rad)
        dx = pts[:, 0] - center[0]
        dy = pts[:, 1] - center[1]
        dz = pts[:, 2] - center[2]
        loc_x = c * dx + s * dy
        loc_y = -s * dx + c * dy
        loc_z = dz
        mask = (np.abs(loc_x) <= ext[0]) & (np.abs(loc_y) <= ext[1]) & (np.abs(loc_z) <= ext[2])
        if self.clip_invert:
            mask = ~mask
        clipped_pts = pts[mask]
        clipped_cols = cloud.colors[mask] if cloud.colors is not None else None
        clipped_ints = cloud.intensities[mask] if cloud.intensities is not None else None
        return clipped_pts, clipped_cols, clipped_ints

    def set_transform_mode(self, mode: str | None):
        m = str(mode).lower() if mode else None
        if m in ('translate', 'move'):
            self.transform_mode = 'translate'
            self.mode = 'navigate'
            self.status_changed.emit(tr("Move mode: Click and drag cloud in viewport or drag 3D axis arrows."))
        elif m in ('rotate', 'rot'):
            self.transform_mode = 'rotate'
            self.mode = 'navigate'
            self.status_changed.emit(tr("Rotate mode: Click and drag rotation ring or cloud to rotate."))
        else:
            self.transform_mode = None
            self._active_gizmo_axis = None
            self._hovered_gizmo_axis = None
            self.status_changed.emit(tr("Navigate mode: Drag to orbit; right drag to pan; wheel to zoom."))
        self.transform_mode_changed.emit(self.transform_mode or '')
        self.update()

    def set_active_transform_slot(self, slot: int):
        if 0 <= int(slot) < 2:
            self.active_transform_slot = int(slot)
            self.update()

    def set_color_mode(self, slot_or_mode, mode: str | None = None):
        if mode is None:
            m = str(slot_or_mode).lower()
            slots = [0, 1]
        else:
            m = mode.lower()
            slots = [int(slot_or_mode)]
        changed = False
        for s in slots:
            if self.color_modes[s] != m:
                self.color_modes[s] = m
                changed = True
                if self.clouds[s] is not None:
                    self._dirty.add(s)
        if changed:
            self.error = ''
            self.color_mode_changed.emit(m)
            self.update()

    def set_colormap(self, slot_or_cmap, cmap: str | None = None):
        if cmap is None:
            c = str(slot_or_cmap).lower()
            slots = [0, 1]
        else:
            c = cmap.lower()
            slots = [int(slot_or_cmap)]
        changed = False
        for s in slots:
            if self.colormaps[s] != c:
                self.colormaps[s] = c
                changed = True
                if self.clouds[s] is not None and self.color_modes[s] != 'rgb':
                    self._dirty.add(s)
        if changed:
            self.error = ''
            self.colormap_changed.emit(c)
            self.update()

    def _get_scalar_range(self, field_name: str) -> tuple[float, float]:
        if field_name in self._scalar_ranges:
            return self._scalar_ranges[field_name]
        vals_list = []
        for slot, c in enumerate(self.clouds):
            if c is None:
                continue
            ids = self._indices[slot]
            if field_name == 'z':
                vals_list.append(c.points[ids, 2])
            elif field_name == 'x':
                vals_list.append(c.points[ids, 0])
            elif field_name == 'y':
                vals_list.append(c.points[ids, 1])
            elif field_name == 'intensity' and c.intensities is not None:
                vals_list.append(c.intensities[ids])

        if not vals_list:
            res = (0.0, 1.0)
        else:
            all_vals = np.concatenate(vals_list) if len(vals_list) > 1 else vals_list[0]
            if len(all_vals) == 0:
                res = (0.0, 1.0)
            else:
                p_lo = float(np.percentile(all_vals, 0.5))
                p_hi = float(np.percentile(all_vals, 99.5))
                if p_hi <= p_lo:
                    p_hi = p_lo + 1.0
                res = (p_lo, p_hi)

        self._scalar_ranges[field_name] = res
        return res

    def _get_colors(self, slot: int) -> np.ndarray:
        cloud = self.clouds[slot]
        ids = self._indices[slot]
        mode = self.color_modes[slot]
        cmap = self.colormaps[slot]
        if mode == 'rgb' or not any(c is not None for c in self.clouds):
            return cloud.colors[ids]

        lut = get_lut(cmap)
        if mode in ('height', 'z'):
            vals = cloud.points[ids, 2]
            field_name = 'z'
        elif mode == 'x':
            vals = cloud.points[ids, 0]
            field_name = 'x'
        elif mode == 'y':
            vals = cloud.points[ids, 1]
            field_name = 'y'
        elif mode == 'intensity':
            if cloud.intensities is not None:
                vals = cloud.intensities[ids]
                field_name = 'intensity'
            else:
                return cloud.colors[ids]
        else:
            return cloud.colors[ids]

        vmin, vmax = self._get_scalar_range(field_name)
        span = max(float(vmax - vmin), 1e-6)
        u = np.clip((vals - vmin) / span, 0.0, 1.0)
        indices = np.clip((u * 255.0).astype(np.int32), 0, 255)
        return lut[indices]

    def fit_all(self):
        clouds = [c for c in self.clouds if c is not None]
        if not clouds: return
        lo = np.min([c.points.min(axis=0) for c in clouds], axis=0)
        hi = np.max([c.points.max(axis=0) for c in clouds], axis=0)
        self.target = lo+(hi-lo)*.5-self.origin
        self.scene_radius = max(float(np.linalg.norm(hi-lo))*.5, .001)
        # Fit projected bounds, so a long depth axis does not shrink a front
        # elevation into the middle of the window.
        _, right, up = self._matrix()
        corners = np.array(list(product(*zip(lo, hi)))) - self.origin - self.target
        aspect = max(self.width(), 1)/max(self.height(), 1)
        self.half_height = max(float(np.abs(corners @ up).max()),
                               float(np.abs(corners @ right).max())/aspect, .001)*1.08
        self.update()

    def set_split(self, value):
        self.split = float(np.clip(value, 0, 1)); self.split_changed.emit(self.split); self.update()

    def set_preset(self, name):
        if name not in PRESETS:
            return
        self.yaw, self.elevation = PRESETS[name]
        self.current_preset = name
        self.view_preset_changed.emit(name)
        self.update()

    def set_mode(self, mode):
        self.mode = mode; self.pending = []; self.setFocus(); self.update()
        hint = {'navigate':'Drag to orbit; right/middle drag to pan; wheel to zoom.', 'point':'Click a cloud point to read its original XYZ coordinates.', 'distance':'Click two points to measure 3D distance and XYZ differences.', 'polyline':'Click vertices; Enter finishes the polyline; Escape cancels.', 'angle':'Click three points; the second point is the angle vertex.'}
        self.status_changed.emit(hint[mode])

    def clear_measurements(self):
        self.measurements.clear(); self.pending.clear(); self.measurement_added.emit(''); self.update()

    def undo_measurement(self):
        if self.pending: self.pending.pop()
        elif self.measurements: self.measurements.pop()
        self._emit_measurements(); self.update()

    def _emit_measurements(self):
        self.measurement_added.emit('\n'.join(m['label'] for m in self.measurements))

    def _matrix(self):
        return camera_matrix(self.target, self.yaw, self.elevation, self.half_height, max(self.width(),1)/max(self.height(),1), self.scene_radius)

    def initializeGL(self):
        try:
            self._gl = QOpenGLFunctions_2_1()
            if not self._gl.initializeOpenGLFunctions(): raise RuntimeError('OpenGL 2.1 is unavailable on this display.')
            self._program = QOpenGLShaderProgram(self)
            vertex = '''#version 120
attribute vec3 position;
attribute vec3 color;
uniform mat4 mvp;
varying vec3 rgb;
varying vec3 world_pos;
void main(){
    world_pos = position;
    gl_Position = mvp * vec4(position, 1.0);
    rgb = color;
}
'''
            fragment = '''#version 120
varying vec3 rgb;
varying vec3 world_pos;
uniform vec3 clip_center_rel;
uniform vec3 clip_extents;
uniform float clip_cos;
uniform float clip_sin;
uniform int clip_enabled;
uniform int clip_invert;
void main(){
    if (clip_enabled != 0) {
        vec3 d = world_pos - clip_center_rel;
        vec3 p_local = vec3(
            clip_cos * d.x + clip_sin * d.y,
            -clip_sin * d.x + clip_cos * d.y,
            d.z
        );
        bool outside = (abs(p_local.x) > clip_extents.x ||
                        abs(p_local.y) > clip_extents.y ||
                        abs(p_local.z) > clip_extents.z);
        if (clip_invert != 0 ? !outside : outside) {
            discard;
        }
    }
    gl_FragColor = vec4(rgb, 1.0);
}
'''
            if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vertex): raise RuntimeError(self._program.log())
            if not self._program.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, fragment): raise RuntimeError(self._program.log())
            self._program.bindAttributeLocation('position', 0); self._program.bindAttributeLocation('color', 1)
            if not self._program.link(): raise RuntimeError(self._program.log())
            self._dirty.update(i for i,c in enumerate(self.clouds) if c is not None)
            self.context().aboutToBeDestroyed.connect(self._cleanup)
        except Exception as exc:
            self.error = str(exc); self.status_changed.emit('Renderer error: '+self.error)

    def _cleanup(self):
        self.makeCurrent()
        self._map_renderer.cleanup()
        for buf in self._buffers:
            if buf is not None: buf.destroy()
        self._buffers = [None, None]
        if self._program is not None: self._program.removeAllShaders()
        self._program = None
        self.doneCurrent()

    def _upload(self, slot):
        cloud = self.clouds[slot]; ids = self._indices[slot]
        if cloud is None:
            if self._buffers[slot] is not None:
                self._buffers[slot].destroy()
                self._buffers[slot] = None
            return
        packed = np.empty((len(ids),6),dtype=np.float32)
        packed[:,:3] = cloud.points[ids]-self.origin; packed[:,3:] = self._get_colors(slot)
        if self._buffers[slot] is not None: self._buffers[slot].destroy()
        buf = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer); buf.create(); buf.bind()
        buf.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        buf.allocate(packed.ctypes.data, packed.nbytes); buf.release()
        self._buffers[slot] = buf

    def _render_gl_scene(self, gl, width: int, height: int, point_size_scale: float = 1.0):
        gl.glViewport(0, 0, width, height)
        gl.glDisable(0x0C11)
        gl.glClearColor(self.bg_color.redF(), self.bg_color.greenF(), self.bg_color.blueF(), 1.0)
        gl.glClear(0x4000 | 0x0100)
        aspect = max(width, 1) / max(height, 1)
        matrix = camera_matrix(self.target, self.yaw, self.elevation, self.half_height, aspect, self.scene_radius)[0]
        try:
            self._map_renderer.draw(self.basemap_layer, gl, matrix, self.origin)
        except Exception as exc:
            self.basemap_layer.enabled = False
            self.status_changed.emit('Map renderer: ' + str(exc))
        gl.glEnable(0x0B71)
        gl.glDepthFunc(0x0201)
        gl.glDisable(0x0BE2)
        gl.glPointSize(float(self.point_size * point_size_scale))
        for slot in sorted(self._dirty):
            self._upload(slot)
        self._dirty.clear()
        self._program.bind()
        self._program.setUniformValue('mvp', QMatrix4x4(matrix.ravel().tolist()))
        self._program.setUniformValue('clip_enabled', 1 if self.clipping_enabled else 0)
        self._program.setUniformValue('clip_invert', 1 if self.clip_invert else 0)
        rel_center = self.clip_center - self.origin
        self._program.setUniformValue('clip_center_rel', float(rel_center[0]), float(rel_center[1]), float(rel_center[2]))
        self._program.setUniformValue('clip_extents', float(self.clip_extents[0]), float(self.clip_extents[1]), float(self.clip_extents[2]))
        rad = math.radians(self.clip_yaw)
        self._program.setUniformValue('clip_cos', float(math.cos(rad)))
        self._program.setUniformValue('clip_sin', float(math.sin(rad)))

        both = all(c is not None for c in self.clouds)
        cut = round(width * self.split)
        for slot, cloud in enumerate(self.clouds):
            if cloud is None:
                continue
            if both:
                gl.glEnable(0x0C11)
                gl.glScissor(0 if slot == 0 else cut, 0, cut if slot == 0 else width - cut, height)
            else:
                gl.glDisable(0x0C11)
            gl.glClear(0x0100)
            self._buffers[slot].bind()
            self._program.enableAttributeArray(0)
            self._program.enableAttributeArray(1)
            self._program.setAttributeBuffer(0, 0x1406, 0, 3, 24)
            self._program.setAttributeBuffer(1, 0x1406, 12, 3, 24)
            gl.glDrawArrays(0x0000, 0, len(self._indices[slot]))
            self._program.disableAttributeArray(0)
            self._program.disableAttributeArray(1)
            self._buffers[slot].release()
        self._program.release()
        gl.glDisable(0x0C11)
        gl.glDisable(0x0B71)

    def paintGL(self):
        painter = QPainter(self)
        painter.beginNativePainting()
        try:
            if self._gl is not None and self._program is not None and not self.error:
                ratio = self.devicePixelRatioF()
                width = round(self.width() * ratio)
                height = round(self.height() * ratio)
                self._render_gl_scene(self._gl, width, height, point_size_scale=ratio)
        except Exception as exc:
            self.error = str(exc)
            self.status_changed.emit('Renderer error: ' + self.error)
        finally:
            painter.endNativePainting()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            self._overlay(painter)
        except Exception as exc:
            painter.drawText(20, 40, f"Overlay error: {exc}")
        finally:
            painter.end()

    def _draw_clipping_min_badge(self, p: QPainter, w: int, h: int, s: float):
        """Draw minimal floating pill badge indicating active clipping when 3D box is hidden."""
        self._clipping_min_badge_rects.clear()
        p.save()
        badge_w = round(260 * s)
        badge_h = round(34 * s)
        hx = (w - badge_w) // 2
        hy = round(14 * s)

        p.setPen(QPen(QColor(56, 75, 96, 220), max(1.0, 1.2 * s)))
        p.setBrush(QColor(14, 20, 28, 235))
        p.drawRoundedRect(QRectF(hx, hy, badge_w, badge_h), 17 * s, 17 * s)

        f = p.font(); f.setPointSize(max(7, round(8.5 * s))); f.setBold(True); p.setFont(f)
        p.setPen(QColor('#00d2df'))
        p.drawText(round(hx + 14 * s), round(hy + 22 * s), "✂ " + tr("Clipping Active"))

        # Edit Box button
        edit_text = tr("Edit Box")
        ew = round((p.fontMetrics().horizontalAdvance(edit_text) + 16) * s)
        eh = round(24 * s)
        ey = hy + (badge_h - eh) // 2
        ex = hx + badge_w - ew - round(36 * s)
        edit_rect = QRectF(ex, ey, ew, eh)
        self._clipping_min_badge_rects['edit_box'] = edit_rect

        is_edit_hover = (self._hovered_hud_btn == 'edit_box')
        p.setPen(QPen(QColor('#00d2df' if is_edit_hover else '#3b4b5e'), 1.0))
        p.setBrush(QColor(0, 210, 223, 50) if is_edit_hover else QColor(255, 255, 255, 18))
        p.drawRoundedRect(edit_rect, 12 * s, 12 * s)
        p.setPen(QColor('#ffffff' if is_edit_hover else '#c2d6ea'))
        p.drawText(edit_rect, Qt.AlignmentFlag.AlignCenter, edit_text)

        # Close button
        cx_btn = ex + ew + round(6 * s)
        close_rect = QRectF(cx_btn, ey, round(24 * s), eh)
        self._clipping_min_badge_rects['close'] = close_rect
        is_close_hover = (self._hovered_hud_btn == 'close_min')
        p.setPen(QPen(QColor('#ff4757' if is_close_hover else '#3b4b5e'), 1.0))
        p.setBrush(QColor(255, 71, 87, 50) if is_close_hover else QColor(255, 255, 255, 18))
        p.drawRoundedRect(close_rect, 12 * s, 12 * s)
        p.setPen(QColor('#ff4757' if is_close_hover else '#c2d6ea'))
        p.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "✕")

        p.restore()

    def _draw_clipping_box(self, p: QPainter, w: int, h: int, s: float, matrix: np.ndarray):
        if not (self.clipping_enabled or self.clipping_box_mode or self.clipping_box_visible) or not any(c is not None for c in self.clouds):
            return

        if not self.clipping_box_visible:
            if self.clipping_enabled:
                self._draw_clipping_min_badge(p, w, h, s)
            return

        C = self.clip_center
        E = self.clip_extents
        ex, ey, ez = float(E[0]), float(E[1]), float(E[2])
        rad = math.radians(self.clip_yaw)
        cos_y, sin_y = math.cos(rad), math.sin(rad)

        # 8 rotated corner points in world coordinates
        loc_corners = np.array([
            [-ex, -ey, -ez],
            [ ex, -ey, -ez],
            [ ex,  ey, -ez],
            [-ex,  ey, -ez],
            [-ex, -ey,  ez],
            [ ex, -ey,  ez],
            [ ex,  ey,  ez],
            [-ex,  ey,  ez],
        ], dtype=np.float32)

        rot_x = cos_y * loc_corners[:, 0] - sin_y * loc_corners[:, 1]
        rot_y = sin_y * loc_corners[:, 0] + cos_y * loc_corners[:, 1]
        rot_corners = np.column_stack([rot_x + C[0], rot_y + C[1], loc_corners[:, 2] + C[2]])

        screen_pts, depths = project_points(rot_corners - self.origin, matrix, w, h)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Draw wireframe edges in signature CloudCompare yellow
        p.save()
        box_color = QColor('#fbc531')
        p.setPen(QPen(box_color, max(1.2, 1.8 * s), Qt.PenStyle.SolidLine))
        for i1, i2 in edges:
            if -1.0 <= depths[i1] <= 1.0 or -1.0 <= depths[i2] <= 1.0:
                p.drawLine(QPointF(*screen_pts[i1]), QPointF(*screen_pts[i2]))

        # Horizontal Yaw Rotation Ring around the box waist
        sc_c, d_c = project_points((C - self.origin).reshape(1, 3), matrix, w, h)
        if -1.0 <= d_c[0] <= 1.0:
            cx, cy = float(sc_c[0, 0]), float(sc_c[0, 1])
            self._clipping_box_screen_center = (cx, cy)
            r_ring_world = max(ex, ey) * 1.25 + 0.3
            ang = np.linspace(0, 2 * math.pi, 48)
            ring_w_pts = np.column_stack([
                C[0] + r_ring_world * np.cos(ang),
                C[1] + r_ring_world * np.sin(ang),
                np.full_like(ang, C[2])
            ])
            sc_ring, d_ring = project_points(ring_w_pts - self.origin, matrix, w, h)
            ring_valid = np.all((-1.0 <= d_ring) & (d_ring <= 1.0))
            if ring_valid:
                ring_path = QPainterPath()
                ring_path.moveTo(QPointF(*sc_ring[0]))
                for pt in sc_ring[1:]:
                    ring_path.lineTo(QPointF(*pt))
                ring_path.closeSubpath()
                is_rot_hover = (self._hovered_clipping_handle == 'rot_yaw')
                p.setPen(QPen(QColor(0, 210, 223, 220 if is_rot_hover else 120), max(1.2, (2.0 if is_rot_hover else 1.4) * s), Qt.PenStyle.DashLine))
                p.drawPath(ring_path)

        # 6 Interactive Face Handles
        self._clipping_handle_screens.clear()
        pixel_scale = 2.0 * self.half_height / max(1, h)
        L_world = 30.0 * s * pixel_scale

        faces = [
            ('x_min', np.array([-cos_y, -sin_y, 0.0]), QColor('#ff4757'), 'X Min', 0, -1),
            ('x_max', np.array([ cos_y,  sin_y, 0.0]), QColor('#ff4757'), 'X Max', 0, 1),
            ('y_min', np.array([ sin_y, -cos_y, 0.0]), QColor('#2ed573'), 'Y Min', 1, -1),
            ('y_max', np.array([-sin_y,  cos_y, 0.0]), QColor('#2ed573'), 'Y Max', 1, 1),
            ('z_min', np.array([0.0, 0.0, -1.0]), QColor('#1e90ff'), 'Z Min', 2, -1),
            ('z_max', np.array([0.0, 0.0,  1.0]), QColor('#1e90ff'), 'Z Max', 2, 1),
        ]

        stem_len = 24.0 * s
        for key, normal, color, label, axis_idx, sign in faces:
            face_pt = C + normal * E[axis_idx]
            sc_pt, d_pt = project_points((face_pt - self.origin).reshape(1, 3), matrix, w, h)
            if not (-1.0 <= d_pt[0] <= 1.0):
                continue
            sx, sy = float(sc_pt[0, 0]), float(sc_pt[0, 1])
            sc_tip, _ = project_points((face_pt + normal * L_world - self.origin).reshape(1, 3), matrix, w, h)
            tx, ty = float(sc_tip[0, 0]), float(sc_tip[0, 1])
            vx, vy = tx - sx, ty - sy
            dist = math.hypot(vx, vy)
            if dist < 1e-2:
                continue
            ux, uy = vx / dist, vy / dist
            perp_x, perp_y = -uy, ux

            ax = sx + ux * stem_len
            ay = sy + uy * stem_len

            is_active = (self._active_clipping_handle == key)
            is_hover = (self._hovered_clipping_handle == key)
            is_ring_hover = (self._hovered_clipping_handle == key + '_rot')

            # Store for resizing: key -> (sx, sy, ax, ay, ux, uy, axis_idx, sign, normal)
            self._clipping_handle_screens[key] = (sx, sy, ax, ay, ux, uy, axis_idx, sign, normal)
            # Store face ring for rotation: key + '_rot' -> (sx, sy, ring_radius)
            self._clipping_handle_screens[key + '_rot'] = (sx, sy, 14.0 * s)

            if is_active or is_hover or is_ring_hover:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 90))
                p.drawEllipse(QPointF(sx, sy), 16.0 * s, 16.0 * s)

            # Draw Ring / Torus at face center
            p.save()
            p.translate(sx, sy)
            p.rotate(math.degrees(math.atan2(uy, ux)) + 90.0)
            ring_w = (15.0 if is_ring_hover else 12.0) * s
            ring_h = (7.0 if is_ring_hover else 5.0) * s
            p.setPen(QPen(QColor('white') if is_ring_hover else color.lighter(130), max(1.5, 2.0 * s)))
            p.setBrush(QColor(color.red(), color.green(), color.blue(), 220 if is_ring_hover else 180))
            p.drawEllipse(QPointF(0, 0), ring_w, ring_h)
            p.restore()

            # Draw Arrow Stem
            p.setPen(QPen(color, max(2.0, (3.2 if is_hover else 2.6) * s)))
            p.drawLine(QPointF(sx, sy), QPointF(ax, ay))

            # Draw Arrowhead Cone
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color.lighter(120) if is_hover else color)
            cone_tip = QPointF(ax + ux * (8.0 * s), ay + uy * (8.0 * s))
            cone_b1 = QPointF(ax - ux * (2.0 * s) + perp_x * (5.5 * s), ay - uy * (2.0 * s) + perp_y * (5.5 * s))
            cone_b2 = QPointF(ax - ux * (2.0 * s) - perp_x * (5.5 * s), ay - uy * (2.0 * s) - perp_y * (5.5 * s))
            p.drawPolygon(QPolygonF([cone_tip, cone_b1, cone_b2]))

            if is_hover or is_active:
                dim_extent = 2.0 * E[axis_idx]
                tag_text = f"{label}: {dim_extent:.2f} m"
                f = p.font(); f.setPointSize(max(7, round(8.5 * s))); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(tag_text) + 12 * s
                th = 20.0 * s
                p.setPen(QPen(color, 1.0))
                p.setBrush(QColor(15, 23, 34, 235))
                p.drawRoundedRect(QRectF(ax + ux * 10 * s - tw / 2, ay + uy * 10 * s - th / 2, tw, th), 4 * s, 4 * s)
                p.setPen(QColor('white'))
                p.drawText(QRectF(ax + ux * 10 * s - tw / 2, ay + uy * 10 * s - th / 2, tw, th), Qt.AlignmentFlag.AlignCenter, tag_text)
            elif is_ring_hover:
                tag_text = f"{tr('Drag ring to rotate box')} ({self.clip_yaw:+.1f}°)"
                f = p.font(); f.setPointSize(max(7, round(8.5 * s))); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(tag_text) + 12 * s
                th = 20.0 * s
                p.setPen(QPen(QColor('#00d2df'), 1.0))
                p.setBrush(QColor(15, 23, 34, 235))
                p.drawRoundedRect(QRectF(sx - tw / 2, sy - th - 12 * s, tw, th), 4 * s, 4 * s)
                p.setPen(QColor('#00d2df'))
                p.drawText(QRectF(sx - tw / 2, sy - th - 12 * s, tw, th), Qt.AlignmentFlag.AlignCenter, tag_text)

        # Center 4-way translation widget
        if -1.0 <= d_c[0] <= 1.0:
            cx, cy = float(sc_c[0, 0]), float(sc_c[0, 1])
            is_c_active = (self._active_clipping_handle == 'center')
            is_c_hover = (self._hovered_clipping_handle == 'center')
            self._clipping_handle_screens['center'] = (cx, cy, 18.0 * s)

            if is_c_active or is_c_hover:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 234, 0, 90))
                p.drawEllipse(QPointF(cx, cy), 18.0 * s, 18.0 * s)

            p.setPen(QPen(QColor('#0b1017'), max(1.0, 1.5 * s)))
            p.setBrush(QColor('#f1c40f'))
            cr = 6.0 * s
            p.drawEllipse(QPointF(cx, cy), cr, cr)

            arr_dist = 14.0 * s
            for dx_dir, dy_dir in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p.drawLine(QPointF(cx + dx_dir * cr, cy + dy_dir * cr), QPointF(cx + dx_dir * arr_dist, cy + dy_dir * arr_dist))
                tip = QPointF(cx + dx_dir * (arr_dist + 4 * s), cy + dy_dir * (arr_dist + 4 * s))
                b1 = QPointF(cx + dx_dir * arr_dist - dy_dir * 3 * s, cy + dy_dir * arr_dist + dx_dir * 3 * s)
                b2 = QPointF(cx + dx_dir * arr_dist + dy_dir * 3 * s, cy + dy_dir * arr_dist - dx_dir * 3 * s)
                p.drawPolygon(QPolygonF([tip, b1, b2]))

            if is_c_hover or is_c_active:
                lbl = tr("Move entire box")
                f = p.font(); f.setPointSize(max(7, round(8.5 * s))); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(lbl) + 12 * s
                th = 20.0 * s
                p.setPen(QPen(QColor('#f1c40f'), 1.0))
                p.setBrush(QColor(15, 23, 34, 235))
                p.drawRoundedRect(QRectF(cx - tw / 2, cy - th - 12 * s, tw, th), 4 * s, 4 * s)
                p.setPen(QColor('#f1c40f'))
                p.drawText(QRectF(cx - tw / 2, cy - th - 12 * s, tw, th), Qt.AlignmentFlag.AlignCenter, lbl)

        # In-Viewport Floating HUD Pill
        self._clipping_hud_rects.clear()
        hud_w = round(560 * s)
        hud_h = round(38 * s)
        hx = (w - hud_w) // 2
        hy = round(14 * s)

        p.setPen(QPen(QColor(56, 75, 96, 220), max(1.0, 1.2 * s)))
        p.setBrush(QColor(14, 20, 28, 240))
        p.drawRoundedRect(QRectF(hx, hy, hud_w, hud_h), 19 * s, 19 * s)

        f = p.font(); f.setPointSize(max(7, round(9 * s))); f.setBold(True); p.setFont(f)
        p.setPen(QColor('#fbc531'))
        p.drawText(round(hx + 14 * s), round(hy + 24 * s), "✂ " + tr("3D Clip"))

        f.setBold(False); f.setPointSize(max(6, round(8 * s))); p.setFont(f)
        p.setPen(QColor('#8ec5fc'))
        dim_str = f"{2*ex:.1f} × {2*ey:.1f} × {2*ez:.1f} m"
        p.drawText(round(hx + 76 * s), round(hy + 24 * s), dim_str)

        # Yaw indicator
        f.setBold(True); p.setFont(f)
        p.setPen(QColor('#00d2df'))
        yaw_str = f"∡ {self.clip_yaw:+.1f}°"
        p.drawText(round(hx + 195 * s), round(hy + 24 * s), yaw_str)

        btn_defs = [
            ('align', tr("Align"), False),
            ('reset', tr("Reset Box"), False),
            ('invert', tr("Invert Clip"), self.clip_invert),
            ('export', tr("Export"), False),
            ('hide', "👁 " + tr("Hide Box"), False),
            ('close', "✕", False),
        ]
        bx = hx + round(262 * s)
        for b_name, b_text, b_active in btn_defs:
            f.setPointSize(max(6, round(8 * s)))
            f.setBold(b_active or b_name == 'close')
            p.setFont(f)
            bw = round((26 if b_name == 'close' else p.fontMetrics().horizontalAdvance(b_text) + 12) * s)
            bh = round(24 * s)
            by = hy + (hud_h - bh) // 2
            btn_rect = QRectF(bx, by, bw, bh)
            self._clipping_hud_rects[b_name] = btn_rect

            is_btn_hover = (self._hovered_hud_btn == b_name)
            p.setPen(QPen(QColor('#00d2df' if b_active else ('#68829e' if is_btn_hover else '#3b4b5e')), 1.0))
            if b_active:
                p.setBrush(QColor(0, 210, 223, 70))
            elif is_btn_hover:
                p.setBrush(QColor(255, 255, 255, 35))
            else:
                p.setBrush(QColor(255, 255, 255, 15))
            p.drawRoundedRect(btn_rect, 12 * s, 12 * s)

            p.setPen(QColor('#00d2df') if b_active else (QColor('#ffffff') if is_btn_hover else QColor('#c2d6ea')))
            p.drawText(btn_rect, Qt.AlignmentFlag.AlignCenter, b_text)
            bx += bw + round(5 * s)

        p.restore()

    def _draw_transform_gizmo(self, p: QPainter, w: int, h: int, s: float, matrix: np.ndarray):
        if not self.transform_mode or self.clouds[self.active_transform_slot] is None or self._original_points[self.active_transform_slot] is None:
            return

        slot = self.active_transform_slot
        original = self._original_points[slot]
        center = original.mean(axis=0)
        yaw, shift = self._adjustments[slot]
        pivot = center + shift

        sc_p, d_p = project_points((pivot - self.origin).reshape(1, 3), matrix, w, h)
        if not (-1.0 <= d_p[0] <= 1.0):
            return

        px, py = float(sc_p[0, 0]), float(sc_p[0, 1])
        pixel_scale = 2.0 * self.half_height / max(1, h)
        self._gizmo_handle_screens.clear()
        self._gizmo_handle_screens['pivot'] = (px, py)

        p.save()

        slot_lbl = 'A' if slot == 0 else 'B'
        if self.transform_mode == 'translate':
            mode_txt = f"↔ " + tr("Move Cloud {slot}").format(slot=slot_lbl)
        else:
            mode_txt = f"🔄 " + tr("Rotate Cloud {slot}").format(slot=slot_lbl)

        f = p.font(); f.setPointSize(max(7, round(8.5 * s))); f.setBold(True); p.setFont(f)
        badge_w = p.fontMetrics().horizontalAdvance(mode_txt) + 16 * s
        badge_h = 26 * s
        bx = w - badge_w - 14 * s
        by = 14 * s
        p.setPen(QPen(QColor('#00d2df'), 1.2 * s))
        p.setBrush(QColor(14, 22, 32, 235))
        p.drawRoundedRect(QRectF(bx, by, badge_w, badge_h), 13 * s, 13 * s)
        p.setPen(QColor('#ffffff'))
        p.drawText(QRectF(bx, by, badge_w, badge_h), Qt.AlignmentFlag.AlignCenter, mode_txt)

        if self.transform_mode == 'translate':
            arrow_len = 70.0 * s
            L_w = arrow_len * pixel_scale
            axes = [
                ('x', np.array([1.0, 0.0, 0.0]), QColor('#ff4757'), 'X'),
                ('y', np.array([0.0, 1.0, 0.0]), QColor('#2ed573'), 'Y'),
                ('z', np.array([0.0, 0.0, 1.0]), QColor('#1e90ff'), 'Z'),
            ]

            is_center_active = (self._active_gizmo_axis == 'free')
            is_center_hover = (self._hovered_gizmo_axis == 'center')
            self._gizmo_handle_screens['center'] = (px, py, 11.0 * s)

            p.setPen(QPen(QColor('#ffffff') if is_center_hover else QColor('#ffd166'), max(1.5, 2.0 * s)))
            p.setBrush(QColor('#ffd166') if not is_center_hover else QColor('#ffeaa7'))
            p.drawEllipse(QPointF(px, py), 8.0 * s, 8.0 * s)

            for key, u_vec, color, label in axes:
                tip_w = pivot + u_vec * L_w
                sc_t, _ = project_points((tip_w - self.origin).reshape(1, 3), matrix, w, h)
                tx, ty = float(sc_t[0, 0]), float(sc_t[0, 1])
                vx, vy = tx - px, ty - py
                dist = math.hypot(vx, vy)
                if dist < 1e-2:
                    continue
                ux, uy = vx / dist, vy / dist
                perp_x, perp_y = -uy, ux

                is_active = (self._active_gizmo_axis == key)
                is_hover = (self._hovered_gizmo_axis == key)
                self._gizmo_handle_screens[key] = (px, py, tx, ty)

                if is_active or is_hover:
                    p.setPen(QPen(QColor(255, 255, 255, 120), max(4.0, 6.0 * s)))
                    p.drawLine(QPointF(px, py), QPointF(tx, ty))

                p.setPen(QPen(color, max(2.2, 3.0 * s)))
                p.drawLine(QPointF(px, py), QPointF(tx, ty))

                cone_tip = QPointF(tx + ux * (8.0 * s), ty + uy * (8.0 * s))
                b1 = QPointF(tx - ux * (2.0 * s) + perp_x * (5.5 * s), ty - uy * (2.0 * s) + perp_y * (5.5 * s))
                b2 = QPointF(tx - ux * (2.0 * s) - perp_x * (5.5 * s), ty - uy * (2.0 * s) - perp_y * (5.5 * s))
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                p.drawPolygon(QPolygonF([cone_tip, b1, b2]))

                f.setPointSize(max(7, round(8.5 * s))); f.setBold(True); p.setFont(f)
                p.setPen(color.lighter(140))
                p.drawText(QPointF(tx + ux * (14.0 * s) - 4 * s, ty + uy * (14.0 * s) + 4 * s), label)

        elif self.transform_mode == 'rotate':
            ring_r_px = 85.0 * s
            L_w = ring_r_px * pixel_scale
            N = 48
            ring_pts = []
            for k in range(N + 1):
                ang = 2.0 * math.pi * (k % N) / N
                pt_w = pivot + np.array([L_w * math.cos(ang), L_w * math.sin(ang), 0.0])
                sc_k, _ = project_points((pt_w - self.origin).reshape(1, 3), matrix, w, h)
                ring_pts.append(QPointF(float(sc_k[0, 0]), float(sc_k[0, 1])))

            is_active = (self._active_gizmo_axis == 'yaw')
            is_hover = (self._hovered_gizmo_axis == 'yaw')
            self._gizmo_handle_screens['yaw'] = (px, py, ring_r_px)

            ring_col = QColor('#00d2df')
            if is_active or is_hover:
                p.setPen(QPen(QColor(0, 210, 223, 100), max(5.0, 7.0 * s)))
                p.drawPolyline(ring_pts)

            p.setPen(QPen(ring_col, max(2.2, 3.0 * s)))
            p.drawPolyline(ring_pts)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(ring_col)
            p.drawEllipse(QPointF(px, py), 5.0 * s, 5.0 * s)

            if is_active:
                yaw_txt = f"Yaw: {yaw:+.1f}°"
                f.setPointSize(max(8, round(10 * s))); f.setBold(True); p.setFont(f)
                tw = p.fontMetrics().horizontalAdvance(yaw_txt) + 16 * s
                th = 24 * s
                p.setPen(QPen(ring_col, 1.2 * s))
                p.setBrush(QColor(14, 20, 28, 240))
                p.drawRoundedRect(QRectF(px - tw / 2, py - ring_r_px - th - 8 * s, tw, th), 6 * s, 6 * s)
                p.setPen(QColor('#ffffff'))
                p.drawText(QRectF(px - tw / 2, py - ring_r_px - th - 8 * s, tw, th), Qt.AlignmentFlag.AlignCenter, yaw_txt)

        p.restore()

    def _overlay(self, p, width=None, height=None, scale=1.0, show_labels=True, show_hud=True, show_measurements=True):
        w = self.width() if width is None else int(width)
        h = self.height() if height is None else int(height)
        s = float(scale)

        is_light = self.bg_color.lightnessF() > 0.55
        text_pen = QColor('#18222d') if is_light else QColor('#d8e7f2')
        p.setPen(text_pen)

        if self.basemap_layer.enabled and any(t.get('image') is not None for t in self.basemap_layer.tiles):
            label = self.basemap_layer.attribution
            attr_font = p.font()
            attr_font.setPointSize(max(6, round(8 * s)))
            p.setFont(attr_font)
            attr_w = p.fontMetrics().horizontalAdvance(label) + round(16 * s)
            attr_h = round(23 * s)
            p.fillRect(w - attr_w - round(8 * s), h - round(27 * s), attr_w, attr_h, QColor(0, 0, 0, 180))
            p.setPen(QColor('white'))
            p.drawText(w - attr_w, h - round(11 * s), label)
            p.setPen(text_pen)

        if self.error:
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, '3D renderer unavailable\n' + self.error)
            return
        if not any(c is not None for c in self.clouds):
            p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                       tr('Open a point cloud to explore in 3D') + '\n' +
                       tr('Add a second cloud to compare • PCD / PLY / LAS / LAZ'))
            return

        if show_labels:
            lbl_font = p.font()
            lbl_font.setPointSize(max(7, round(9 * s)))
            p.setFont(lbl_font)
            for slot, c in enumerate(self.clouds):
                if c is not None:
                    label = p.fontMetrics().elidedText(f"{'AB'[slot]}  {c.path.name}", Qt.TextElideMode.ElideMiddle, max(round(80 * s), w // 2 - round(32 * s)))
                    x = round(14 * s) if slot == 0 else max(round(14 * s), w - p.fontMetrics().horizontalAdvance(label) - round(14 * s))
                    p.drawText(int(x), int(round(24 * s)), label)

        # Redesigned Modern Split Slider
        if all(c is not None for c in self.clouds):
            x = w * self.split
            p.setPen(QPen(QColor(0, 0, 0, 70), max(1.0, 3.0 * s)))
            p.drawLine(QPointF(x, 26 * s), QPointF(x, h))
            p.setPen(QPen(QColor('#00d2df'), max(1.0, 1.8 * s)))
            p.drawLine(QPointF(x, 26 * s), QPointF(x, h))

            handle_w, handle_h = round(32 * s), round(46 * s)
            hy = h // 2 - handle_h // 2
            hx = int(x) - handle_w // 2

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 80))
            p.drawRoundedRect(hx - 1, hy + round(2 * s), handle_w + 2, handle_h, round(16 * s), round(16 * s))

            pill_grad = QLinearGradient(hx, hy, hx, hy + handle_h)
            pill_grad.setColorAt(0.0, QColor(32, 46, 62, 245))
            pill_grad.setColorAt(0.5, QColor(20, 30, 42, 250))
            pill_grad.setColorAt(1.0, QColor(14, 22, 32, 255))
            p.setBrush(pill_grad)
            p.setPen(QPen(QColor('#00d2df'), max(1.0, 1.5 * s)))
            p.drawRoundedRect(hx, hy, handle_w, handle_h, round(15 * s), round(15 * s))

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor('#ffffff'))
            arr_sz = round(5 * s)
            p.drawPolygon(QPolygonF([QPointF(x - 3 * s, hy + handle_h // 2), QPointF(x - 8 * s, hy + handle_h // 2 - arr_sz), QPointF(x - 8 * s, hy + handle_h // 2 + arr_sz)]))
            p.drawPolygon(QPolygonF([QPointF(x + 3 * s, hy + handle_h // 2), QPointF(x + 8 * s, hy + handle_h // 2 - arr_sz), QPointF(x + 8 * s, hy + handle_h // 2 + arr_sz)]))

        aspect = max(w, 1) / max(h, 1)
        matrix = camera_matrix(self.target, self.yaw, self.elevation, self.half_height, aspect, self.scene_radius)[0]

        # 3D Clipping Box & Interactive Handles (CloudCompare style)
        self._draw_clipping_box(p, w, h, s, matrix)

        # 3D Viewport Transform Gizmo (Move / Rotate)
        self._draw_transform_gizmo(p, w, h, s, matrix)

        if show_measurements:
            m_font = p.font()
            m_font.setPointSize(max(7, round(9 * s)))
            p.setFont(m_font)
            for record in self.measurements + ([{'points': self.pending, 'label': 'Pending'}] if self.pending else []):
                xyz = np.array([r['xyz'] for r in record['points']])
                screen, depth = project_points(xyz - self.origin, matrix, w, h)
                p.setPen(QPen(QColor('#ffd166'), max(1.0, 2.0 * s)))
                p.setBrush(QColor('#ffd166'))
                dot_r = max(2.0, 4.0 * s)
                for i, (sx, sy) in enumerate(screen):
                    if -1 <= depth[i] <= 1:
                        p.drawEllipse(QPointF(sx, sy), dot_r, dot_r)
                        if i and -1 <= depth[i - 1] <= 1:
                            p.drawLine(QPointF(*screen[i - 1]), QPointF(sx, sy))
                if len(screen) and -1 <= depth[-1] <= 1:
                    p.drawText(QPointF(*screen[-1]) + QPointF(9 * s, -9 * s), record['label'].split(' | ')[0])

        if show_hud:
            # Orthographic ruler, independent of the file coordinate origin.
            per_pixel = 2 * self.half_height / max(1, h)
            raw = per_pixel * (100 * s)
            magnitude = 10 ** math.floor(math.log10(max(raw, 1e-15)))
            value = next(v * magnitude for v in (1, 2, 5, 10) if v * magnitude >= raw)
            length = value / per_pixel
            ry = h - round(24 * s)
            p.setPen(QPen(text_pen, max(1.0, 2.0 * s)))
            p.drawLine(QPointF(round(16 * s), ry), QPointF(round(16 * s) + length, ry))
            ruler_font = p.font()
            ruler_font.setPointSize(max(7, round(9 * s)))
            p.setFont(ruler_font)
            p.drawText(round(16 * s), ry - round(7 * s), f'{value:g} m')

            # Scalar / Elevation HUD legend card(s)
            scalar_slots = [sl for sl, cl in enumerate(self.clouds) if cl is not None and self.color_modes[sl] != 'rgb']
            if scalar_slots:
                card_w, card_h = round(78 * s), round(142 * s)
                base_cx = w - card_w - round(14 * s)
                cy = h - card_h - round(14 * s)

                slots_to_draw = []
                for sl in scalar_slots:
                    key = (self.color_modes[sl], self.colormaps[sl])
                    if not any(key == (self.color_modes[x], self.colormaps[x]) for x in slots_to_draw):
                        slots_to_draw.append(sl)

                for idx, slot in enumerate(reversed(slots_to_draw)):
                    cx = base_cx - idx * (card_w + round(8 * s))
                    smode = self.color_modes[slot]
                    scmap = self.colormaps[slot]
                    field_name = 'z' if smode in ('height', 'z') else smode
                    vmin, vmax = self._get_scalar_range(field_name)

                    p.save()
                    p.setPen(QPen(QColor('#384b60'), max(1.0, 1.0 * s)))
                    p.setBrush(QColor(18, 26, 36, 220))
                    p.drawRoundedRect(cx, cy, card_w, card_h, round(6 * s), round(6 * s))

                    title_map = {'height': 'Height', 'z': 'Height', 'intensity': 'Intensity', 'x': 'X Axis', 'y': 'Y Axis'}
                    title = title_map.get(smode, smode.title())
                    if len(slots_to_draw) > 1:
                        title = f"{'AB'[slot]}: {title}"
                    p.setPen(QColor('#9ac3e6'))
                    f = p.font()
                    f.setBold(True)
                    f.setPointSize(max(6, round(8 * s)))
                    p.setFont(f)
                    p.drawText(cx, cy + round(4 * s), card_w, round(16 * s), Qt.AlignmentFlag.AlignHCenter, title)

                    bar_x = cx + round(8 * s)
                    bar_y = cy + round(24 * s)
                    bar_w = round(12 * s)
                    bar_h = round(104 * s)
                    grad = QLinearGradient(bar_x, bar_y + bar_h, bar_x, bar_y)
                    lut = get_lut(scmap)
                    for i in range(8):
                        stop = i / 7.0
                        idx_val = int(stop * 255)
                        c = lut[idx_val]
                        grad.setColorAt(stop, QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2])))

                    p.setPen(QPen(QColor('#556b82'), max(1.0, 1.0 * s)))
                    p.setBrush(grad)
                    p.drawRect(bar_x, bar_y, bar_w, bar_h)

                    f.setBold(False)
                    f.setPointSize(max(5, round(7 * s)))
                    p.setFont(f)
                    p.setPen(QColor('#e2eef8'))
                    unit = 'm' if smode in ('height', 'z', 'x', 'y') else ''
                    t_top = f"{vmax:+.1f}{unit}" if unit else f"{vmax:.0f}"
                    t_mid = f"{(vmax+vmin)*0.5:+.1f}{unit}" if unit else f"{(vmax+vmin)*0.5:.0f}"
                    t_bot = f"{vmin:+.1f}{unit}" if unit else f"{vmin:.0f}"
                    p.drawText(bar_x + bar_w + round(4 * s), bar_y + round(9 * s), t_top)
                    p.drawText(bar_x + bar_w + round(4 * s), bar_y + bar_h // 2 + round(4 * s), t_mid)
                    p.drawText(bar_x + bar_w + round(4 * s), bar_y + bar_h - round(1 * s), t_bot)
                    p.restore()

    def render_to_image(
        self,
        width: int,
        height: int,
        include_overlays: bool = True,
        point_size_multiplier: float = 1.0,
        show_labels: bool = True,
        show_hud: bool = True,
        show_measurements: bool = True,
    ) -> QImage:
        """Render the 3D point cloud scene to an off-screen QImage at arbitrary resolution."""
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid render dimensions: {width}x{height}")
        self.makeCurrent()
        try:
            if self._gl is None or self._program is None:
                self.initializeGL()
            fbo = QOpenGLFramebufferObject(width, height, QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
            if not fbo.isValid():
                fbo = QOpenGLFramebufferObject(width, height, QOpenGLFramebufferObject.Attachment.Depth)
            if not fbo.isValid():
                raise RuntimeError(tr("OpenGL cannot create framebuffer of size {w}x{h} on this hardware.").format(w=width, h=height))
            fbo.bind()
            screen_h = max(self.height(), 1)
            scale = max(width / max(self.width(), 1), height / screen_h)
            pt_scale = scale * point_size_multiplier
            self._render_gl_scene(self._gl, width, height, point_size_scale=pt_scale)
            fbo.release()
            img = fbo.toImage()
            if img.isNull():
                raise RuntimeError(tr("Failed to read image from framebuffer object."))
            if include_overlays:
                p = QPainter(img)
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
                self._overlay(
                    p,
                    width=width,
                    height=height,
                    scale=scale,
                    show_labels=show_labels,
                    show_hud=show_hud,
                    show_measurements=show_measurements,
                )
                p.end()
            return img
        finally:
            self.doneCurrent()

    def pick_point(self, x, y):
        loaded=[i for i,c in enumerate(self.clouds) if c is not None]
        if not loaded: return None
        slot=loaded[0] if len(loaded)==1 else (0 if x<self.width()*self.split else 1)
        cloud=self.clouds[slot]; ids=self._indices[slot]; matrix=self._matrix()[0]
        candidates=[]
        # Same common-origin transform as the GPU. Pick from the displayed
        # original vertices, never from a resampled depth coordinate.
        for start in range(0,len(ids),200000):
            indices=ids[start:start+200000]; screen,z=project_points(cloud.points[indices]-self.origin,matrix,self.width(),self.height())
            delta=np.sum((screen-[x,y])**2,axis=1)
            valid=(delta<=64)&(z>=-1)&(z<=1)
            if np.any(valid):
                local=np.flatnonzero(valid); candidates.extend((float(delta[j]),float(z[j]),int(indices[j])) for j in local)
        if not candidates: return None
        closest=min(r[0] for r in candidates)
        chosen=min((r for r in candidates if r[0]<=closest+4),key=lambda r:r[1])
        index=chosen[2]
        return {'source':str(cloud.path),'cloud':'AB'[slot],'index':index,'xyz':cloud.points[index].tolist()}

    def _add_pick(self, record):
        if record is None:
            self.status_changed.emit('No visible point near the cursor. Zoom in and click the cloud.'); return
        self.pending.append(record)
        count={'point':1,'distance':2,'angle':3}.get(self.mode)
        if count is not None and len(self.pending)==count: self._finish_measurement()
        self.update()

    def _finish_measurement(self):
        if not self.pending or (self.mode=='polyline' and len(self.pending)<2): return
        points=[r['xyz'] for r in self.pending]
        try: value,unit=measurement_value(self.mode,points)
        except ValueError as exc:
            self.status_changed.emit(str(exc)); self.pending.pop(); return
        number=len(self.measurements)+1
        if self.mode=='point':
            label=f"{number}. XYZ: "+', '.join(f'{v:.6f}' for v in points[0])+' m'
        else: label=f'{number}. {self.mode.title()}: {value:.6f} {unit}'
        if self.mode=='distance':
            d=np.array(points[1])-points[0]; label+=' | Delta XYZ '+', '.join(f'{v:+.6f}' for v in d)+' m'
        label+=' | '+', '.join(r['cloud']+':'+str(r['index']) for r in self.pending)
        self.measurements.append({'type':self.mode,'points':list(self.pending),'value':value,'unit':unit,'label':label})
        self.pending=[]; self._emit_measurements(); self.update()

    def export_measurements(self, path):
        with open(path,'w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream); writer.writerow(['measurement','type','value','unit','vertex','cloud','source','point_index','x','y','z'])
            for n,m in enumerate(self.measurements,1):
                for vertex,r in enumerate(m['points'],1):
                    writer.writerow([n,m['type'],m['value'],m['unit'],vertex,r['cloud'],r['source'],r['index'],*r['xyz']])

    def mousePressEvent(self, event):
        self.setFocus()
        pos = event.position()
        self._press = pos
        self._last = pos
        self._dragged = False

        # 1. Check Minimal Floating Badge Buttons click (when box is hidden)
        if self.clipping_enabled and not self.clipping_box_visible and self._clipping_min_badge_rects:
            for b_name, rect in self._clipping_min_badge_rects.items():
                if rect.contains(pos):
                    if b_name == 'edit_box':
                        self.set_clipping_box_visible(True)
                    elif b_name == 'close':
                        self.set_clipping_enabled(False)
                    return

        # 2. Check Floating Clipping HUD Buttons click (when box is visible)
        if self.clipping_box_visible and self._clipping_hud_rects:
            for b_name, rect in self._clipping_hud_rects.items():
                if rect.contains(pos):
                    if b_name == 'align':
                        self.align_clipping_box_to_cloud()
                    elif b_name == 'reset':
                        self.reset_clipping_to_clouds()
                    elif b_name == 'invert':
                        self.set_clip_invert(not self.clip_invert)
                    elif b_name == 'export':
                        self.export_clipped_requested.emit()
                    elif b_name == 'hide':
                        self.set_clipping_box_visible(False)
                    elif b_name == 'close':
                        self.set_clipping_enabled(False)
                    return

        # 3. Check 3D Clipping Box Handles click
        if self.clipping_box_visible and event.button() == Qt.MouseButton.LeftButton:
            hit_handle = None

            # 3a. Check Face Ring Rotation Handles (near face center sx, sy)
            for key in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
                r_info = self._clipping_handle_screens.get(key + '_rot')
                if r_info:
                    sx, sy, r_radius = r_info
                    if math.hypot(pos.x() - sx, pos.y() - sy) <= 15:
                        hit_handle = 'rot_yaw'
                        break

            # 3b. Check Center 4-way translation widget
            if hit_handle is None:
                c_info = self._clipping_handle_screens.get('center')
                if c_info:
                    cx, cy, _ = c_info
                    if math.hypot(pos.x() - cx, pos.y() - cy) <= 18:
                        hit_handle = 'center'

            # 3c. Check Face Resize Handles (near arrow tip ax, ay or stem)
            if hit_handle is None:
                for key in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
                    info = self._clipping_handle_screens.get(key)
                    if info:
                        sx, sy, ax, ay, ux, uy, axis_idx, sign, normal = info
                        # Check arrow tip
                        if math.hypot(pos.x() - ax, pos.y() - ay) <= 18:
                            hit_handle = key
                            break
                        # Check along stem line
                        vx, vy = ax - sx, ay - sy
                        L2 = vx * vx + vy * vy
                        if L2 > 0:
                            t = max(0.0, min(1.0, ((pos.x() - sx) * vx + (pos.y() - sy) * vy) / L2))
                            nx, ny = sx + t * vx, sy + t * vy
                            if math.hypot(pos.x() - nx, pos.y() - ny) <= 12:
                                hit_handle = key
                                break

            # 3d. Check Horizontal Rotation Ring
            if hit_handle is None and self._hovered_clipping_handle == 'rot_yaw':
                hit_handle = 'rot_yaw'

            if hit_handle is not None:
                self._active_clipping_handle = hit_handle
                self._clipping_drag_start_pos = pos
                self._clipping_drag_start_center = self.clip_center.copy()
                self._clipping_drag_start_extents = self.clip_extents.copy()
                self._clipping_drag_start_yaw = float(self.clip_yaw)
                self.update()
                return

        # 4. Check Viewport Transform Gizmo & Cloud Click
        if self.transform_mode in ('translate', 'rotate') and event.button() == Qt.MouseButton.LeftButton:
            slot = self.active_transform_slot
            if self.clouds[slot] is not None and self._original_points[slot] is not None:
                hit_gizmo = None
                if self.transform_mode == 'translate':
                    c_info = self._gizmo_handle_screens.get('center')
                    if c_info and math.hypot(pos.x() - c_info[0], pos.y() - c_info[1]) <= c_info[2]:
                        hit_gizmo = 'free'
                    else:
                        for axis in ('x', 'y', 'z'):
                            info = self._gizmo_handle_screens.get(axis)
                            if info:
                                px, py, tx, ty = info
                                vx, vy = tx - px, ty - py
                                L2 = vx * vx + vy * vy
                                if L2 > 0:
                                    t = max(0.0, min(1.0, ((pos.x() - px) * vx + (pos.y() - py) * vy) / L2))
                                    nx, ny = px + t * vx, py + t * vy
                                    if math.hypot(pos.x() - nx, pos.y() - ny) <= 14:
                                        hit_gizmo = axis
                                        break
                elif self.transform_mode == 'rotate':
                    r_info = self._gizmo_handle_screens.get('yaw')
                    if r_info:
                        px, py, r_px = r_info
                        dist = math.hypot(pos.x() - px, pos.y() - py)
                        if abs(dist - r_px) <= 16 or dist <= 12:
                            hit_gizmo = 'yaw'

                # If gizmo handle wasn't hit directly, check if clicking directly on the point cloud
                if hit_gizmo is None:
                    picked = self.pick_point(pos.x(), pos.y())
                    if picked is not None and (all(c is None for c in self.clouds[1:]) or picked.get('cloud') == 'AB'[slot]):
                        hit_gizmo = 'free' if self.transform_mode == 'translate' else 'yaw'

                if hit_gizmo is not None:
                    self._active_gizmo_axis = hit_gizmo
                    self._transform_drag_start_pos = pos
                    self._transform_drag_start_val = (float(self._adjustments[slot][0]), self._adjustments[slot][1].copy())
                    self.update()
                    return

        # 5. Fallback to normal navigation / split / picking
        self._drag_split = event.button() == Qt.MouseButton.LeftButton and all(c is not None for c in self.clouds) and abs(pos.x() - self.width() * self.split) < 10

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._last is None:
            self._handle_hover(pos)
            return

        if self._press is not None and (pos - self._press).manhattanLength() >= 4:
            self._dragged = True
        delta = pos - self._last
        self._last = pos

        # 1. Dragging Clipping Handle
        if self._active_clipping_handle is not None:
            total_delta = pos - self._clipping_drag_start_pos
            pixel_scale = 2.0 * self.half_height / max(1, self.height())
            _, right, up = self._matrix()

            if self._active_clipping_handle == 'center':
                d_shift = (right * total_delta.x() - up * total_delta.y()) * pixel_scale
                self.clip_center = np.asarray(self._clipping_drag_start_center + d_shift, dtype=np.float32)
                self.clip_min = self.clip_center - self.clip_extents
                self.clip_max = self.clip_center + self.clip_extents
            elif self._active_clipping_handle == 'rot_yaw':
                cx, cy = self._clipping_box_screen_center
                start_angle = math.atan2(self._clipping_drag_start_pos.y() - cy, self._clipping_drag_start_pos.x() - cx)
                curr_angle = math.atan2(pos.y() - cy, pos.x() - cx)
                d_angle = math.degrees(curr_angle - start_angle)
                if self.elevation < 0:
                    d_angle = -d_angle
                new_yaw = (self._clipping_drag_start_yaw - d_angle + 180.0) % 360.0 - 180.0
                self.clip_yaw = float(new_yaw)
            else:
                info = self._clipping_handle_screens.get(self._active_clipping_handle)
                if info:
                    sx, sy, ax, ay, ux, uy, axis_idx, sign, normal = info
                    proj_dist = (total_delta.x() * ux + total_delta.y() * uy) * pixel_scale
                    start_ext = self._clipping_drag_start_extents
                    start_c = self._clipping_drag_start_center
                    new_half = max(0.05, float(start_ext[axis_idx] + proj_dist * 0.5))
                    delta_half = new_half - float(start_ext[axis_idx])
                    new_ext = start_ext.copy()
                    new_ext[axis_idx] = new_half
                    new_center = start_c + normal * delta_half
                    self.clip_extents = np.asarray(new_ext, dtype=np.float32)
                    self.clip_center = np.asarray(new_center, dtype=np.float32)
                    self.clip_min = self.clip_center - self.clip_extents
                    self.clip_max = self.clip_center + self.clip_extents

            self.clipping_bounds_changed.emit(self.clip_min, self.clip_max, True)
            self.update()
            return

        # 2. Dragging Transform Gizmo / Cloud Object
        if self._active_gizmo_axis is not None:
            slot = self.active_transform_slot
            total_delta = pos - self._transform_drag_start_pos
            start_yaw, start_shift = self._transform_drag_start_val
            pixel_scale = 2.0 * self.half_height / max(1, self.height())
            _, right, up = self._matrix()

            if self._active_gizmo_axis == 'free':
                d_shift = (right * total_delta.x() - up * total_delta.y()) * pixel_scale
                new_shift = start_shift + d_shift
                self.apply_cloud_adjustment(slot, start_yaw, new_shift, is_preview=True)
                self.cloud_transformed.emit(slot, start_yaw, new_shift)
            elif self._active_gizmo_axis in ('x', 'y', 'z'):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}[self._active_gizmo_axis]
                u_world = np.zeros(3)
                u_world[axis_idx] = 1.0
                sx = float(np.dot(u_world, right))
                sy = float(-np.dot(u_world, up))
                mag = math.hypot(sx, sy)
                if mag > 1e-3:
                    ux, uy = sx / mag, sy / mag
                    proj_dist = (total_delta.x() * ux + total_delta.y() * uy) * pixel_scale
                    new_shift = start_shift.copy()
                    new_shift[axis_idx] += proj_dist
                    self.apply_cloud_adjustment(slot, start_yaw, new_shift, is_preview=True)
                    self.cloud_transformed.emit(slot, start_yaw, new_shift)
            elif self._active_gizmo_axis == 'yaw':
                pivot_info = self._gizmo_handle_screens.get('pivot')
                if pivot_info:
                    px, py = pivot_info
                    start_angle = math.atan2(self._transform_drag_start_pos.y() - py, self._transform_drag_start_pos.x() - px)
                    curr_angle = math.atan2(pos.y() - py, pos.x() - px)
                    d_angle = math.degrees(curr_angle - start_angle)
                    if self.elevation < 0:
                        d_angle = -d_angle
                    new_yaw = (start_yaw - d_angle + 180.0) % 360.0 - 180.0
                    self.apply_cloud_adjustment(slot, new_yaw, start_shift, is_preview=True)
                    self.cloud_transformed.emit(slot, new_yaw, start_shift)
            self.update()
            return

        # 3. Normal navigation / split
        if self._drag_split:
            self.set_split(pos.x() / max(1, self.width()))
            return

        if event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton) or (event.buttons() & Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            _, right, up = self._matrix()
            scale = 2 * self.half_height / max(self.height(), 1)
            self.target += (-right * delta.x() + up * delta.y()) * scale
            self.update()
        elif event.buttons() & Qt.MouseButton.LeftButton and self._dragged:
            self.yaw -= delta.x() * 0.4
            self.elevation = float(np.clip(self.elevation + delta.y() * 0.4, -89.9999, 89.9999))
            self.current_preset = 'custom'
            self.view_preset_changed.emit('custom')
            self.update()

        self._handle_hover(pos)

    def _handle_hover(self, pos):
        hover_changed = False
        new_cursor = Qt.CursorShape.ArrowCursor

        # 1. Minimal Badge hover
        badge_btn = None
        if self.clipping_enabled and not self.clipping_box_visible and self._clipping_min_badge_rects:
            for b_name, rect in self._clipping_min_badge_rects.items():
                if rect.contains(pos):
                    badge_btn = b_name
                    new_cursor = Qt.CursorShape.PointingHandCursor
                    break
        if badge_btn != self._hovered_hud_btn and not self.clipping_box_visible:
            self._hovered_hud_btn = badge_btn
            hover_changed = True

        # 2. Floating HUD hover
        hud_btn = None
        if self.clipping_box_visible and self._clipping_hud_rects:
            for b_name, rect in self._clipping_hud_rects.items():
                if rect.contains(pos):
                    hud_btn = b_name
                    new_cursor = Qt.CursorShape.PointingHandCursor
                    break
        if self.clipping_box_visible and hud_btn != self._hovered_hud_btn:
            self._hovered_hud_btn = hud_btn
            hover_changed = True

        # 3. Clipping handles hover
        if self.clipping_box_visible and hud_btn is None:
            clip_hover = None

            # Check face rings (rotation)
            for key in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
                r_info = self._clipping_handle_screens.get(key + '_rot')
                if r_info:
                    sx, sy, r_radius = r_info
                    if math.hypot(pos.x() - sx, pos.y() - sy) <= 15:
                        clip_hover = key + '_rot'
                        new_cursor = Qt.CursorShape.PointingHandCursor
                        break

            # Check center 4-way translation
            if clip_hover is None:
                c_info = self._clipping_handle_screens.get('center')
                if c_info:
                    cx, cy, _ = c_info
                    if math.hypot(pos.x() - cx, pos.y() - cy) <= 18:
                        clip_hover = 'center'
                        new_cursor = Qt.CursorShape.SizeAllCursor

            # Check face resize arrows
            if clip_hover is None:
                for key in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
                    info = self._clipping_handle_screens.get(key)
                    if info:
                        sx, sy, ax, ay, ux, uy, axis_idx, sign, normal = info
                        if math.hypot(pos.x() - ax, pos.y() - ay) <= 18:
                            clip_hover = key
                            new_cursor = Qt.CursorShape.PointingHandCursor
                            break
                        vx, vy = ax - sx, ay - sy
                        L2 = vx * vx + vy * vy
                        if L2 > 0:
                            t = max(0.0, min(1.0, ((pos.x() - sx) * vx + (pos.y() - sy) * vy) / L2))
                            nx, ny = sx + t * vx, sy + t * vy
                            if math.hypot(pos.x() - nx, pos.y() - ny) <= 12:
                                clip_hover = key
                                new_cursor = Qt.CursorShape.PointingHandCursor
                                break

            # Check rotation ring proximity around center
            if clip_hover is None:
                cx, cy = self._clipping_box_screen_center
                d_c = math.hypot(pos.x() - cx, pos.y() - cy)
                ex, ey = float(self.clip_extents[0]), float(self.clip_extents[1])
                pixel_scale = 2.0 * self.half_height / max(1, self.height())
                r_ring_px = (max(ex, ey) * 1.25 + 0.3) / max(1e-4, pixel_scale)
                if abs(d_c - r_ring_px) <= 16:
                    clip_hover = 'rot_yaw'
                    new_cursor = Qt.CursorShape.PointingHandCursor

            if clip_hover != self._hovered_clipping_handle:
                self._hovered_clipping_handle = clip_hover
                hover_changed = True

        # 4. Gizmo handles hover
        if self.transform_mode in ('translate', 'rotate') and hud_btn is None and self._hovered_clipping_handle is None:
            gizmo_hover = None
            if self.transform_mode == 'translate':
                c_info = self._gizmo_handle_screens.get('center')
                if c_info and math.hypot(pos.x() - c_info[0], pos.y() - c_info[1]) <= c_info[2]:
                    gizmo_hover = 'center'
                    new_cursor = Qt.CursorShape.SizeAllCursor
                else:
                    for axis in ('x', 'y', 'z'):
                        info = self._gizmo_handle_screens.get(axis)
                        if info:
                            px, py, tx, ty = info
                            vx, vy = tx - px, ty - py
                            L2 = vx * vx + vy * vy
                            if L2 > 0:
                                t = max(0.0, min(1.0, ((pos.x() - px) * vx + (pos.y() - py) * vy) / L2))
                                nx, ny = px + t * vx, py + t * vy
                                if math.hypot(pos.x() - nx, pos.y() - ny) <= 14:
                                    gizmo_hover = axis
                                    new_cursor = Qt.CursorShape.SizeAllCursor
                                    break
            elif self.transform_mode == 'rotate':
                r_info = self._gizmo_handle_screens.get('yaw')
                if r_info:
                    px, py, r_px = r_info
                    dist = math.hypot(pos.x() - px, pos.y() - py)
                    if abs(dist - r_px) <= 16:
                        gizmo_hover = 'yaw'
                        new_cursor = Qt.CursorShape.PointingHandCursor

            if gizmo_hover is None:
                slot = self.active_transform_slot
                if self.clouds[slot] is not None:
                    p_info = self._gizmo_handle_screens.get('pivot')
                    if p_info and math.hypot(pos.x() - p_info[0], pos.y() - p_info[1]) <= 60:
                        new_cursor = Qt.CursorShape.OpenHandCursor if self.transform_mode == 'translate' else Qt.CursorShape.PointingHandCursor

            if gizmo_hover != self._hovered_gizmo_axis:
                self._hovered_gizmo_axis = gizmo_hover
                hover_changed = True

        if hover_changed:
            self.setCursor(new_cursor)
            self.update()

    def mouseReleaseEvent(self, event):
        if self._active_clipping_handle is not None:
            self._active_clipping_handle = None
            self._clipping_drag_start_pos = None
            self._clipping_drag_start_center = None
            self._clipping_drag_start_extents = None
            self._clipping_drag_start_bounds = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return

        if self._active_gizmo_axis is not None:
            slot = self.active_transform_slot
            yaw, shift = self._adjustments[slot]
            self.apply_cloud_adjustment(slot, yaw, shift, is_preview=False)
            self.cloud_transformed.emit(slot, yaw, shift)
            self._active_gizmo_axis = None
            self._transform_drag_start_pos = None
            self._transform_drag_start_val = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return

        if self._press is not None and not self._dragged and not self._drag_split and event.button() == Qt.MouseButton.LeftButton and self.mode != 'navigate' and (event.position() - self._press).manhattanLength() < 5 and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._add_pick(self.pick_point(event.position().x(), event.position().y()))

        self._press = None
        self._last = None
        self._drag_split = False

    def wheelEvent(self,event):
        old_height = self.half_height
        self.half_height=float(np.clip(old_height*math.exp(-event.angleDelta().y()/120*.15),self.scene_radius*1e-6,self.scene_radius*1000))
        _,right,up=self._matrix()
        mouse=event.position()
        self.target += (right*(mouse.x()-self.width()/2)+up*(self.height()/2-mouse.y()))*2*(old_height-self.half_height)/max(self.height(),1)
        self.update()

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter) and self.mode=='polyline': self._finish_measurement()
        elif event.key()==Qt.Key.Key_Escape: self.pending=[]; self.update()
        elif event.key() in (Qt.Key.Key_Backspace,Qt.Key.Key_Delete): self.undo_measurement()
        elif event.key()==Qt.Key.Key_F: self.fit_all()
        elif event.key()==Qt.Key.Key_1: self.set_preset('iso')
        elif event.key()==Qt.Key.Key_2: self.set_preset('top')
        elif event.key()==Qt.Key.Key_3: self.set_preset('bottom')
        elif event.key()==Qt.Key.Key_4: self.set_preset('front')
        elif event.key()==Qt.Key.Key_5: self.set_preset('back')
        elif event.key()==Qt.Key.Key_6: self.set_preset('left')
        elif event.key()==Qt.Key.Key_7: self.set_preset('right')
        elif event.key()==Qt.Key.Key_C:
            modes = ['rgb', 'height', 'intensity', 'x', 'y']
            next_mode = modes[(modes.index(self.color_mode) + 1) % len(modes)]
            self.set_color_mode(next_mode)
        elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal, Qt.Key.Key_BracketRight):
            self.set_point_size(self.point_size + 0.5)
            self.status_changed.emit(f"Point size: {self.point_size:.1f}")
        elif event.key() in (Qt.Key.Key_Minus, Qt.Key.Key_Underscore, Qt.Key.Key_BracketLeft):
            self.set_point_size(self.point_size - 0.5)
            self.status_changed.emit(f"Point size: {self.point_size:.1f}")
        else: super().keyPressEvent(event)
