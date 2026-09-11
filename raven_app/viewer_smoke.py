"""Small real-OpenGL integration check, also run by the packaged GUI smoke test."""
from pathlib import Path
import tempfile
import numpy as np
from raven_app.cloud_io import CloudData
from raven_app.viewer_geometry import project_points


def prepare_viewer_smoke(main):
    native_id = int(main.winId())
    main._open_viewer()
    window = main.viewer_window
    assert window.parentWidget() is None, 'OpenGL viewer must not join the translucent main window hierarchy'
    window._smoke_main = main
    window._smoke_main_id = native_id
    view = window.renderer
    origin = np.array([600000., 7500000., 100.])
    x,z=np.meshgrid(np.linspace(-2,2,101),np.linspace(-2,2,101))
    points=np.column_stack([x.ravel(),np.zeros(x.size),z.ravel()])+origin
    for slot,color in enumerate(([1.,.1,.1],[.1,.3,1.])):
        view.set_cloud(slot,CloudData(Path('smoke_'+str(slot)+'.pcd'),points,np.tile(color,(len(points),1)).astype('f4'),len(points)))
    view.set_preset('front'); view.point_size=5
    return window


def verify_viewer_smoke(window):
    from PyQt6.QtCore import Qt, QPointF, QEvent
    from PyQt6.QtGui import QMouseEvent
    view=window.renderer
    assert window.isWindow() and window.isVisible()
    assert int(window._smoke_main.winId()) == window._smoke_main_id
    assert not window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert view.isValid() and not view.error,view.error
    point=np.array([[599999.,7500000.,100.]])
    screen,_=project_points(point-view.origin,view._matrix()[0],view.width(),view.height())
    x,y=screen[0]; matrix=view._matrix()[0].copy(); samples=[]; picked=[]
    for split in (1.,0.):
        view.set_split(split)
        image=view.grabFramebuffer(); ratio=view.devicePixelRatioF()
        color=image.pixelColor(round(x*ratio),round(y*ratio))
        samples.append([color.red(),color.green(),color.blue()])
        record=view.pick_point(x,y); assert record is not None
        picked.append(record)
        np.testing.assert_allclose(view._matrix()[0],matrix,atol=1e-12)
    assert samples[0][0]>200 and samples[0][2]<60,samples
    assert samples[1][2]>200 and samples[1][0]<60,samples
    assert picked[0]['cloud']=='A' and picked[1]['cloud']=='B'
    np.testing.assert_allclose(picked[0]['xyz'],picked[1]['xyz'],atol=1e-12)
    np.testing.assert_allclose(picked[0]['xyz'],point[0],atol=1e-8)
    window._mode_buttons['distance'].click()
    assert view.mode=='distance' and window._mode_buttons['distance'].isChecked()
    records=[dict(picked[0],xyz=[600000.,7500000.,100.]),dict(picked[1],xyz=[600003.,7500000.,104.])]
    for record in records:view._add_pick(record)
    assert view.measurements[0]['value']==5.
    assert window.measurements.count()==1
    view.set_mode('angle')
    for xyz in ([1.,0.,0.],[0.,0.,0.],[0.,1.,0.]):view._add_pick(dict(picked[0],xyz=xyz))
    assert view.measurements[-1]['value']==90.
    assert window.measurements.count()==2
    with tempfile.TemporaryDirectory() as tmp:
        export=Path(tmp)/'measurements.csv';view.export_measurements(export)
        text=export.read_text(encoding='utf-8-sig');assert '5.0,m' in text and '90.0,deg' in text
    view.undo_measurement();assert window.measurements.count()==1
    view.clear_measurements();assert window.measurements.count()==0
    # A measurement remains active while dragging, including a drag that returns
    # to its starting pixel. That gesture must never add an accidental vertex.
    view.set_mode('distance')
    view._add_pick(picked[0])
    yaw=view.yaw
    def mouse(kind, x, y, button, buttons):
        return QMouseEvent(kind,QPointF(x,y),QPointF(x,y),button,buttons,Qt.KeyboardModifier.NoModifier)
    left=Qt.MouseButton.LeftButton; none=Qt.MouseButton.NoButton
    view.mousePressEvent(mouse(QEvent.Type.MouseButtonPress,80,80,left,left))
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove,120,90,none,left))
    assert view.yaw != yaw and len(view.pending)==1 and view.mode=='distance'
    view.mouseMoveEvent(mouse(QEvent.Type.MouseMove,80,80,none,left))
    view.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease,80,80,left,none))
    assert len(view.pending)==1 and not view.measurements
    view.clear_measurements()
    window._mode_buttons['distance'].click()
    assert view.mode=='navigate' and not window._mode_buttons['distance'].isChecked()
    view.set_split(.5)
    return {'viewer_ready':True,'main_window_preserved':True,'navigate_while_measuring':True,'swipe_pixels':samples,'same_world_point':True,'measurement_distance_m':5.,'measurement_angle_deg':90.,'csv_export':True}
