"""Asynchronous map tiles located in the loaded cloud's coordinate system."""
from collections import OrderedDict
import math

import numpy as np
from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6.QtGui import QImage

WORLD = 20037508.342789244
PROVIDERS = {
    'satellite': ('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 'Esri World Imagery'),
    'street': ('https://tile.openstreetmap.org/{z}/{x}/{y}.png', '© OpenStreetMap contributors'),
}

_DOWNLOAD_POOL = QThreadPool()
_DOWNLOAD_POOL.setMaxThreadCount(4)


class _TileDownload(QRunnable):
    def __init__(self, key, signal):
        super().__init__()
        self.key, self.signal = key, signal

    def run(self):
        from urllib.request import build_opener, ProxyHandler, Request
        provider, z, x, y = self.key
        url = PROVIDERS[provider][0].format(z=z, x=x, y=y)
        try:
            # Keep DNS, TLS and Windows proxy discovery away from the GUI thread.
            opener = build_opener(ProxyHandler({}))
            request = Request(url, headers={'User-Agent': 'LidarCamera360/1.0 (desktop map viewer)'})
            with opener.open(request, timeout=10) as response:
                data = response.read(2_000_000)
            error = ''
        except Exception as exc:
            data, error = b'', str(exc)
        try:
            self.signal.emit(self.key, data, error)
        except RuntimeError:
            pass  # Viewer was closed while the request finished.


def tile_geometry(points, crs_wkt, max_tiles=36):
    """Return a bounded tile grid and ground-plane corners in source coordinates."""
    from pyproj import CRS, Transformer
    crs = CRS.from_user_input(crs_wkt)
    if not crs.is_projected:
        raise ValueError('Map display requires a projected point-cloud CRS')
    sample = np.asarray(points)[::max(1, len(points) // 20000)]
    lo, hi = sample.min(axis=0), sample.max(axis=0)
    forward = Transformer.from_crs(crs, 3857, always_xy=True)
    reverse = Transformer.from_crs(3857, crs, always_xy=True)
    xs, ys = forward.transform([lo[0], hi[0], lo[0], hi[0]],
                               [lo[1], lo[1], hi[1], hi[1]], errcheck=True)
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    span = max(xmax - xmin, ymax - ymin, 20.)
    pad = span * .5
    xmin, xmax = max(-WORLD, xmin-pad), min(WORLD, xmax+pad)
    ymin, ymax = max(-WORLD, ymin-pad), min(WORLD, ymax+pad)
    zoom = min(19, max(1, int(math.log2(2 * WORLD / span)) + 1))
    while True:
        width = 2 * WORLD / (1 << zoom)
        tx0, tx1 = int((xmin+WORLD)//width), min((1<<zoom)-1, int((xmax+WORLD)//width))
        ty0, ty1 = int((WORLD-ymax)//width), min((1<<zoom)-1, int((WORLD-ymin)//width))
        if (tx1-tx0+1)*(ty1-ty0+1) <= max_tiles or zoom == 1:
            break
        zoom -= 1
    ground = float(np.quantile(sample[:, 2], .01)) - .05
    tiles = []
    for x in range(tx0, tx1+1):
        for y in range(ty0, ty1+1):
            west, north = x*width-WORLD, WORLD-y*width
            xx, yy = reverse.transform([west, west+width, west, west+width],
                                       [north, north, north-width, north-width], errcheck=True)
            tiles.append({'id': (zoom, x, y), 'corners': np.column_stack((xx, yy, np.full(4, ground)))})
    return tiles


class BaseMapLayer(QObject):
    changed = pyqtSignal()
    status = pyqtSignal(str)
    downloaded = pyqtSignal(object, bytes, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = False
        self.provider = 'satellite'
        self.tiles = []
        self.revision = 0
        self._cache = OrderedDict()
        self._pending = {}
        self._failed = set()
        self._cloud = None
        self.downloaded.connect(self._received)

    @property
    def attribution(self):
        return PROVIDERS[self.provider][1]

    def configure(self, cloud):
        self._cloud = cloud
        self.tiles = []
        self.revision += 1
        if not self.enabled:
            self.changed.emit()
            return
        if cloud is None or not cloud.crs_wkt:
            self.status.emit('Map unavailable: this cloud has no geographic CRS. Open a georeferenced output.')
            self.changed.emit()
            return
        try:
            self.tiles = tile_geometry(cloud.points, cloud.crs_wkt)
            for tile in self.tiles:
                tile['key'] = (self.provider, *tile['id'])
                tile['image'] = self._cache.get(tile['key'])
                if tile['image'] is None:
                    self._request(tile['key'])
            self.status.emit('Loading map layer…' if any(t['image'] is None for t in self.tiles) else self.attribution)
        except (ValueError, RuntimeError) as exc:
            self.status.emit(str(exc))
        self.changed.emit()

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)
        if self.enabled:
            self._failed.clear()
        self.configure(self._cloud)

    def set_provider(self, provider):
        if provider not in PROVIDERS:
            raise ValueError('Unknown map provider')
        self.provider = provider
        self.configure(self._cloud)

    def _request(self, key):
        if key in self._pending or key in self._failed:
            return
        self._pending[key] = True
        _DOWNLOAD_POOL.start(_TileDownload(key, self.downloaded))

    def _received(self, key, data, error):
        self._pending.pop(key, None)
        if not error:
            image = QImage.fromData(data)
            if not image.isNull():
                self._cache[key] = image
                while len(self._cache) > 128:
                    self._cache.popitem(last=False)
                for tile in self.tiles:
                    if tile['key'] == key:
                        tile['image'] = image
                self.changed.emit()
        else:
            self._failed.add(key)
            if self.enabled and key[0] == self.provider:
                self.status.emit('Map tiles unavailable: ' + error)


class MapTileRenderer:
    """Textured ground tiles share the point cloud's OpenGL camera."""
    def __init__(self):
        self.program = None
        self.buffer = None
        self.textures = {}

    def cleanup(self):
        for texture in self.textures.values():
            texture.destroy()
        self.textures.clear()
        if self.buffer is not None:
            self.buffer.destroy()
        if self.program is not None:
            self.program.removeAllShaders()
        self.buffer = self.program = None

    def draw(self, layer, gl, matrix, origin):
        from PyQt6.QtOpenGL import QOpenGLBuffer, QOpenGLShader, QOpenGLShaderProgram, QOpenGLTexture
        from PyQt6.QtGui import QMatrix4x4
        if not layer.enabled or not any(t.get('image') is not None for t in layer.tiles):
            return
        if self.program is None:
            self.program = QOpenGLShaderProgram()
            vertex = '''#version 120
attribute vec3 position; attribute vec2 uv; uniform mat4 mvp; varying vec2 texcoord;
void main() { gl_Position = mvp * vec4(position, 1.0); texcoord = uv; }
'''
            fragment = '''#version 120
uniform sampler2D tile; varying vec2 texcoord;
void main() { gl_FragColor = texture2D(tile, texcoord); }
'''
            for kind, source in ((QOpenGLShader.ShaderTypeBit.Vertex, vertex), (QOpenGLShader.ShaderTypeBit.Fragment, fragment)):
                if not self.program.addShaderFromSourceCode(kind, source):
                    raise RuntimeError(self.program.log())
            self.program.bindAttributeLocation('position', 0)
            self.program.bindAttributeLocation('uv', 1)
            if not self.program.link():
                raise RuntimeError(self.program.log())
            self.buffer = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self.buffer.create()
        active = {t['key'] for t in layer.tiles}
        for key in list(self.textures):
            if key not in active:
                self.textures.pop(key).destroy()
        self.program.bind()
        self.program.setUniformValue('mvp', QMatrix4x4(matrix.ravel().tolist()))
        self.program.setUniformValue('tile', 0)
        gl.glDisable(0x0B71)
        gl.glDepthMask(False)
        try:
            for tile in layer.tiles:
                image = tile.get('image')
                if image is None:
                    continue
                key = tile['key']
                if key not in self.textures:
                    texture = QOpenGLTexture(image)
                    texture.setMinificationFilter(QOpenGLTexture.Filter.Linear)
                    texture.setMagnificationFilter(QOpenGLTexture.Filter.Linear)
                    texture.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
                    self.textures[key] = texture
                texture = self.textures[key]
                vertices = np.column_stack((tile['corners']-origin, [[0,0],[1,0],[0,1],[1,1]])).astype(np.float32)
                texture.bind(0)
                self.buffer.bind()
                self.buffer.allocate(vertices.ctypes.data, vertices.nbytes)
                self.program.enableAttributeArray(0)
                self.program.enableAttributeArray(1)
                self.program.setAttributeBuffer(0, 0x1406, 0, 3, 20)
                self.program.setAttributeBuffer(1, 0x1406, 12, 2, 20)
                gl.glDrawArrays(0x0005, 0, 4)
                self.program.disableAttributeArray(0)
                self.program.disableAttributeArray(1)
                self.buffer.release()
                texture.release()
        finally:
            self.program.release()
            gl.glDepthMask(True)
            gl.glEnable(0x0B71)
