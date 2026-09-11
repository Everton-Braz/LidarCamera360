"""GPU point-cloud swipe comparison with a shared orthographic camera."""
import csv
import math
from itertools import product
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QMatrix4x4, QSurfaceFormat, QLinearGradient, QFont, QPolygonF
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLFunctions_2_1
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
        self.clip_min = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
        self.clip_max = np.array([1e9, 1e9, 1e9], dtype=np.float32)
        self._scalar_ranges: dict[str, tuple[float, float]] = {}
        self.current_preset = 'iso'
        self.measurements = []; self.pending = []; self._last = None; self._press = None; self._drag_split = False
        self._dragged = False

    @property
    def color_mode(self) -> str:
        return self.color_modes[0]

    @property
    def colormap(self) -> str:
        return self.colormaps[0]

    def set_cloud(self, slot, cloud):
        first = not any(c is not None for c in self.clouds)
        if first:
            lo, hi = cloud.points.min(axis=0), cloud.points.max(axis=0)
            self.origin = lo + (hi-lo)*.5
        self.clouds[slot] = cloud
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
        self.update()

    def set_point_size(self, size: float):
        new_size = float(np.clip(size, 0.1, 20.0))
        if abs(self.point_size - new_size) > 1e-4:
            self.point_size = new_size
            self.point_size_changed.emit(self.point_size)
            self.update()

    def set_background_color(self, color: QColor | str):
        self.bg_color = QColor(color)
        self.update()

    def set_clipping_enabled(self, enabled: bool):
        self.clipping_enabled = bool(enabled)
        self.update()

    def set_clipping_bounds(self, c_min: np.ndarray, c_max: np.ndarray, enabled: bool = True):
        self.clip_min = np.asarray(c_min, dtype=np.float32)
        self.clip_max = np.asarray(c_max, dtype=np.float32)
        self.clipping_enabled = bool(enabled)
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
uniform vec3 clip_min;
uniform vec3 clip_max;
uniform int clip_enabled;
void main(){
    if (clip_enabled != 0) {
        if (world_pos.x < clip_min.x || world_pos.x > clip_max.x ||
            world_pos.y < clip_min.y || world_pos.y > clip_max.y ||
            world_pos.z < clip_min.z || world_pos.z > clip_max.z) {
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
        for buf in self._buffers:
            if buf is not None: buf.destroy()
        self._buffers = [None, None]
        if self._program is not None: self._program.removeAllShaders()
        self._program = None
        self.doneCurrent()

    def _upload(self, slot):
        cloud = self.clouds[slot]; ids = self._indices[slot]
        packed = np.empty((len(ids),6),dtype=np.float32)
        packed[:,:3] = cloud.points[ids]-self.origin; packed[:,3:] = self._get_colors(slot)
        if self._buffers[slot] is not None: self._buffers[slot].destroy()
        buf = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer); buf.create(); buf.bind()
        buf.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        buf.allocate(packed.ctypes.data, packed.nbytes); buf.release()
        self._buffers[slot] = buf

    def paintGL(self):
        painter = QPainter(self)
        painter.beginNativePainting()
        try:
            if self._gl is not None and self._program is not None and not self.error:
                gl = self._gl; ratio = self.devicePixelRatioF(); width, height = round(self.width()*ratio),round(self.height()*ratio)
                gl.glViewport(0,0,width,height); gl.glDisable(0x0C11)
                gl.glClearColor(self.bg_color.redF(), self.bg_color.greenF(), self.bg_color.blueF(), 1.)
                gl.glClear(0x4000|0x0100)
                gl.glEnable(0x0B71); gl.glDepthFunc(0x0201); gl.glDisable(0x0BE2)
                gl.glPointSize(float(self.point_size*ratio))
                for slot in sorted(self._dirty): self._upload(slot)
                self._dirty.clear()
                self._program.bind()
                self._program.setUniformValue('mvp', QMatrix4x4(self._matrix()[0].ravel().tolist()))
                self._program.setUniformValue('clip_enabled', 1 if self.clipping_enabled else 0)
                rel_min = self.clip_min - self.origin
                rel_max = self.clip_max - self.origin
                self._program.setUniformValue('clip_min', float(rel_min[0]), float(rel_min[1]), float(rel_min[2]))
                self._program.setUniformValue('clip_max', float(rel_max[0]), float(rel_max[1]), float(rel_max[2]))

                both = all(c is not None for c in self.clouds); cut=round(width*self.split)
                for slot,cloud in enumerate(self.clouds):
                    if cloud is None: continue
                    if both:
                        gl.glEnable(0x0C11); gl.glScissor(0 if slot==0 else cut,0,cut if slot==0 else width-cut,height)
                    else: gl.glDisable(0x0C11)
                    gl.glClear(0x0100)
                    self._buffers[slot].bind()
                    self._program.enableAttributeArray(0); self._program.enableAttributeArray(1)
                    self._program.setAttributeBuffer(0,0x1406,0,3,24); self._program.setAttributeBuffer(1,0x1406,12,3,24)
                    gl.glDrawArrays(0x0000,0,len(self._indices[slot]))
                    self._program.disableAttributeArray(0); self._program.disableAttributeArray(1); self._buffers[slot].release()
                self._program.release(); gl.glDisable(0x0C11); gl.glDisable(0x0B71)
        except Exception as exc:
            self.error = str(exc); self.status_changed.emit('Renderer error: '+self.error)
        finally:
            painter.endNativePainting()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            self._overlay(painter)
        except Exception as exc:
            painter.drawText(20, 40, f"Overlay error: {exc}")
        finally:
            painter.end()

    def _overlay(self, p):
        is_light = self.bg_color.lightnessF() > 0.55
        text_pen = QColor('#18222d') if is_light else QColor('#d8e7f2')
        p.setPen(text_pen)

        if self.error:
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'3D renderer unavailable\n'+self.error); return
        if not any(c is not None for c in self.clouds):
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'Open cloud A and cloud B to compare\nPCD / PLY | shared coordinates | measurements in metres'); return
        for slot,c in enumerate(self.clouds):
            if c is not None:
                label=p.fontMetrics().elidedText(f"{'AB'[slot]}  {c.path.name}",Qt.TextElideMode.ElideMiddle,max(80,self.width()//2-32))
                x=14 if slot==0 else max(14,self.width()-p.fontMetrics().horizontalAdvance(label)-14)
                p.drawText(x,24,label)

        # Redesigned Modern Split Slider
        if all(c is not None for c in self.clouds):
            x = self.width() * self.split
            p.setPen(QPen(QColor(0, 0, 0, 70), 3))
            p.drawLine(QPointF(x, 26), QPointF(x, self.height()))
            p.setPen(QPen(QColor('#00d2df'), 1.8))
            p.drawLine(QPointF(x, 26), QPointF(x, self.height()))

            handle_w, handle_h = 32, 46
            hy = self.height() // 2 - handle_h // 2
            hx = int(x) - handle_w // 2

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 80))
            p.drawRoundedRect(hx - 1, hy + 2, handle_w + 2, handle_h, 16, 16)

            pill_grad = QLinearGradient(hx, hy, hx, hy + handle_h)
            pill_grad.setColorAt(0.0, QColor(32, 46, 62, 245))
            pill_grad.setColorAt(0.5, QColor(20, 30, 42, 250))
            pill_grad.setColorAt(1.0, QColor(14, 22, 32, 255))
            p.setBrush(pill_grad)
            p.setPen(QPen(QColor('#00d2df'), 1.5))
            p.drawRoundedRect(hx, hy, handle_w, handle_h, 15, 15)

            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor('#ffffff'))
            p.drawPolygon(QPolygonF([QPointF(x - 3, hy + handle_h//2), QPointF(x - 8, hy + handle_h//2 - 5), QPointF(x - 8, hy + handle_h//2 + 5)]))
            p.drawPolygon(QPolygonF([QPointF(x + 3, hy + handle_h//2), QPointF(x + 8, hy + handle_h//2 - 5), QPointF(x + 8, hy + handle_h//2 + 5)]))

        matrix = self._matrix()[0]

        # 3D Clipping Box wireframe guide
        if self.clipping_enabled and any(c is not None for c in self.clouds):
            x0, y0, z0 = self.clip_min
            x1, y1, z1 = self.clip_max
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
            ])
            screen_pts, depths = project_points(corners - self.origin, matrix, self.width(), self.height())
            edges = [
                (0,1), (1,2), (2,3), (3,0),
                (4,5), (5,6), (6,7), (7,4),
                (0,4), (1,5), (2,6), (3,7)
            ]
            p.save()
            p.setPen(QPen(QColor('#26cad3'), 1.3, Qt.PenStyle.DashLine))
            for i1, i2 in edges:
                if -1 <= depths[i1] <= 1 or -1 <= depths[i2] <= 1:
                    p.drawLine(QPointF(*screen_pts[i1]), QPointF(*screen_pts[i2]))
            p.restore()

        for record in self.measurements + ([{'points':self.pending,'label':'Pending'}] if self.pending else []):
            xyz=np.array([r['xyz'] for r in record['points']]); screen,depth=project_points(xyz-self.origin,matrix,self.width(),self.height())
            p.setPen(QPen(QColor('#ffd166'),2)); p.setBrush(QColor('#ffd166'))
            for i,(x,y) in enumerate(screen):
                if -1<=depth[i]<=1:
                    p.drawEllipse(QPointF(x,y),4,4)
                    if i and -1<=depth[i-1]<=1: p.drawLine(QPointF(*screen[i-1]),QPointF(x,y))
            if len(screen) and -1<=depth[-1]<=1:
                p.drawText(QPointF(*screen[-1])+QPointF(9,-9),record['label'].split(' | ')[0])

        # Orthographic ruler, independent of the file coordinate origin.
        per_pixel=2*self.half_height/max(1,self.height()); raw=per_pixel*100
        magnitude=10**math.floor(math.log10(max(raw,1e-15))); value=next(v*magnitude for v in (1,2,5,10) if v*magnitude>=raw)
        length=value/per_pixel; y=self.height()-24
        p.setPen(QPen(text_pen, 2)); p.drawLine(QPointF(16,y),QPointF(16+length,y)); p.drawText(16,y-7,f'{value:g} m')

        # Scalar / Elevation HUD legend card(s)
        scalar_slots = [s for s, c in enumerate(self.clouds) if c is not None and self.color_modes[s] != 'rgb']
        if scalar_slots:
            card_w, card_h = 78, 142
            base_cx = self.width() - card_w - 14
            cy = self.height() - card_h - 14

            # Dedup if both slots have identical mode and colormap
            slots_to_draw = []
            for s in scalar_slots:
                key = (self.color_modes[s], self.colormaps[s])
                if not any(key == (self.color_modes[x], self.colormaps[x]) for x in slots_to_draw):
                    slots_to_draw.append(s)

            for idx, slot in enumerate(reversed(slots_to_draw)):
                cx = base_cx - idx * (card_w + 8)
                smode = self.color_modes[slot]
                scmap = self.colormaps[slot]
                field_name = 'z' if smode in ('height', 'z') else smode
                vmin, vmax = self._get_scalar_range(field_name)

                p.save()
                p.setPen(QPen(QColor('#384b60'), 1))
                p.setBrush(QColor(18, 26, 36, 220))
                p.drawRoundedRect(cx, cy, card_w, card_h, 6, 6)

                title_map = {'height': 'Height', 'z': 'Height', 'intensity': 'Intensity', 'x': 'X Axis', 'y': 'Y Axis'}
                title = title_map.get(smode, smode.title())
                if len(slots_to_draw) > 1:
                    title = f"{'AB'[slot]}: {title}"
                p.setPen(QColor('#9ac3e6'))
                f = p.font()
                f.setBold(True); f.setPointSize(8); p.setFont(f)
                p.drawText(cx, cy + 4, card_w, 16, Qt.AlignmentFlag.AlignHCenter, title)

                bar_x, bar_y, bar_w, bar_h = cx + 8, cy + 24, 12, 104
                grad = QLinearGradient(bar_x, bar_y + bar_h, bar_x, bar_y)
                lut = get_lut(scmap)
                for i in range(8):
                    stop = i / 7.0
                    idx_val = int(stop * 255)
                    c = lut[idx_val]
                    grad.setColorAt(stop, QColor.fromRgbF(float(c[0]), float(c[1]), float(c[2])))

                p.setPen(QPen(QColor('#556b82'), 1))
                p.setBrush(grad)
                p.drawRect(bar_x, bar_y, bar_w, bar_h)

                f.setBold(False); f.setPointSize(7); p.setFont(f)
                p.setPen(QColor('#e2eef8'))
                unit = 'm' if smode in ('height', 'z', 'x', 'y') else ''
                t_top = f"{vmax:+.1f}{unit}" if unit else f"{vmax:.0f}"
                t_mid = f"{(vmax+vmin)*0.5:+.1f}{unit}" if unit else f"{(vmax+vmin)*0.5:.0f}"
                t_bot = f"{vmin:+.1f}{unit}" if unit else f"{vmin:.0f}"
                p.drawText(bar_x + bar_w + 4, bar_y + 9, t_top)
                p.drawText(bar_x + bar_w + 4, bar_y + bar_h // 2 + 4, t_mid)
                p.drawText(bar_x + bar_w + 4, bar_y + bar_h - 1, t_bot)
                p.restore()

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

    def mousePressEvent(self,event):
        self.setFocus(); self._press=event.position(); self._last=event.position()
        self._dragged = False
        self._drag_split=event.button()==Qt.MouseButton.LeftButton and all(c is not None for c in self.clouds) and abs(event.position().x()-self.width()*self.split)<10

    def mouseMoveEvent(self,event):
        if self._last is None: return
        if self._press is not None and (event.position()-self._press).manhattanLength() >= 5:
            self._dragged = True
        delta=event.position()-self._last; self._last=event.position()
        if self._drag_split: self.set_split(event.position().x()/max(1,self.width())); return
        if event.buttons() & (Qt.MouseButton.RightButton|Qt.MouseButton.MiddleButton) or (event.buttons() & Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            _,right,up=self._matrix(); scale=2*self.half_height/max(self.height(),1)
            self.target+=(-right*delta.x()+up*delta.y())*scale; self.update()
        elif event.buttons() & Qt.MouseButton.LeftButton and self._dragged:
            self.yaw-=delta.x()*.4; self.elevation=float(np.clip(self.elevation+delta.y()*.4,-89.9999,89.9999))
            self.current_preset = 'custom'
            self.view_preset_changed.emit('custom')
            self.update()

    def mouseReleaseEvent(self,event):
        if self._press is not None and not self._dragged and not self._drag_split and event.button()==Qt.MouseButton.LeftButton and self.mode!='navigate' and (event.position()-self._press).manhattanLength()<5 and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            self._add_pick(self.pick_point(event.position().x(),event.position().y()))
        self._press=None; self._last=None; self._drag_split=False

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
