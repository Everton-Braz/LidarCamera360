"""GPU point-cloud swipe comparison with a shared orthographic camera."""
import csv
import math
from itertools import product
import numpy as np
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QMatrix4x4, QSurfaceFormat
from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLFunctions_2_1
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from raven_app.viewer_geometry import camera_matrix, project_points, measurement_value


class CloudView(QOpenGLWidget):
    measurement_added = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    split_changed = pyqtSignal(float)
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
        self.measurements = []; self.pending = []; self._last = None; self._press = None; self._drag_split = False
        self._dragged = False

    def set_cloud(self, slot, cloud):
        first = not any(c is not None for c in self.clouds)
        if first:
            lo, hi = cloud.points.min(axis=0), cloud.points.max(axis=0)
            self.origin = lo + (hi-lo)*.5
        self.clouds[slot] = cloud
        step = max(1, math.ceil(len(cloud.points)/self.MAX_DRAW_POINTS))
        self._indices[slot] = np.arange(0, len(cloud.points), step, dtype=np.int64)
        self._dirty.add(slot)
        if self.measurements or self.pending:
            self.clear_measurements()
        if first: self.fit_all()
        else:
            bounds = np.concatenate([c.points[[np.argmin(c.points[:, j]), np.argmax(c.points[:, j])]] for c in self.clouds if c is not None for j in range(3)])
            self.scene_radius = max(self.scene_radius, float(np.linalg.norm(bounds-self.origin-self.target, axis=1).max()))
        suffix = f'; preview sampled to {len(self._indices[slot]):,} points' if step > 1 else ''
        self.status_changed.emit(f"{'AB'[slot]}: {cloud.path.name} | {len(cloud.points):,} valid points{suffix}. Coordinates unchanged.")
        self.update()

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
        self.yaw, self.elevation = {'top':(-90.,89.9999), 'front':(-90.,0.), 'right':(0.,0.), 'iso':(-55.,25.)}[name]
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
void main(){ gl_Position=mvp*vec4(position,1.0); rgb=color; }
'''
            fragment = '''#version 120
varying vec3 rgb;
void main(){ gl_FragColor=vec4(rgb,1.0); }
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
            if buf: buf.destroy()
        self._buffers = [None, None]
        if self._program: self._program.removeAllShaders()
        self._program = None
        self.doneCurrent()

    def _upload(self, slot):
        cloud = self.clouds[slot]; ids = self._indices[slot]
        packed = np.empty((len(ids),6),dtype=np.float32)
        packed[:,:3] = cloud.points[ids]-self.origin; packed[:,3:] = cloud.colors[ids]
        if self._buffers[slot]: self._buffers[slot].destroy()
        buf = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer); buf.create(); buf.bind()
        buf.setUsagePattern(QOpenGLBuffer.UsagePattern.StaticDraw)
        buf.allocate(packed.ctypes.data, packed.nbytes); buf.release()
        self._buffers[slot] = buf

    def paintGL(self):
        painter = QPainter(self)
        painter.beginNativePainting()
        try:
            if self._gl and self._program and not self.error:
                gl = self._gl; ratio = self.devicePixelRatioF(); width, height = round(self.width()*ratio),round(self.height()*ratio)
                gl.glViewport(0,0,width,height); gl.glDisable(0x0C11)
                gl.glClearColor(.045,.065,.09,1.); gl.glClear(0x4000|0x0100)
                gl.glEnable(0x0B71); gl.glDepthFunc(0x0201); gl.glDisable(0x0BE2)
                gl.glPointSize(float(self.point_size*ratio))
                for slot in sorted(self._dirty): self._upload(slot)
                self._dirty.clear()
                self._program.bind(); self._program.setUniformValue('mvp',QMatrix4x4(self._matrix()[0].ravel().tolist()))
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
        finally: painter.endNativePainting()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._overlay(painter); painter.end()

    def _overlay(self, p):
        p.setPen(QColor('#d8e7f2'))
        if self.error:
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'3D renderer unavailable\n'+self.error); return
        if not any(c is not None for c in self.clouds):
            p.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,'Open cloud A and cloud B to compare\nPCD / PLY | shared coordinates | measurements in metres'); return
        for slot,c in enumerate(self.clouds):
            if c is not None:
                label=p.fontMetrics().elidedText(f"{'AB'[slot]}  {c.path.name}",Qt.TextElideMode.ElideMiddle,max(80,self.width()//2-32))
                x=14 if slot==0 else max(14,self.width()-p.fontMetrics().horizontalAdvance(label)-14)
                p.drawText(x,24,label)
        if all(c is not None for c in self.clouds):
            x=self.width()*self.split; p.setPen(QPen(QColor('#26cad3'),2)); p.drawLine(QPointF(x,32),QPointF(x,self.height()))
            p.setBrush(QColor('#26cad3')); p.drawRoundedRect(int(x)-17,self.height()//2-20,34,40,7,7)
            p.setPen(QColor('#052026')); p.drawText(int(x)-12,self.height()//2+5,'<>')
        matrix = self._matrix()[0]
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
        p.setPen(QPen(QColor('#d8e7f2'),2)); p.drawLine(QPointF(16,y),QPointF(16+length,y)); p.drawText(16,y-7,f'{value:g} m')

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
            self.yaw-=delta.x()*.4; self.elevation=float(np.clip(self.elevation+delta.y()*.4,-89.9999,89.9999)); self.update()

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
        else: super().keyPressEvent(event)
