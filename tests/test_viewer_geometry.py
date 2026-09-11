import unittest
import numpy as np
from raven_app.viewer_geometry import camera_matrix, measurement_value, project_points

class ViewerGeometryTests(unittest.TestCase):
    def test_distance_large(self): self.assertAlmostEqual(measurement_value('distance',[[1e12,2e12,3e12],[1e12+3,2e12+4,3e12]])[0],5)
    def test_angle_and_degenerate(self):
        self.assertAlmostEqual(measurement_value('angle',[[1,0,0],[0,0,0],[0,2,0]])[0],90)
        with self.assertRaises(ValueError): measurement_value('angle',[[0,0,0],[0,0,0],[1,0,0]])
    def test_polyline(self): self.assertAlmostEqual(measurement_value('polyline',[[0,0,0],[3,0,0],[3,4,0]])[0],7)
    def test_origin_shift(self):
        pts=np.array([[1e6,2e6,3e6],[1e6+2,2e6+1,3e6+4]]); o=np.array([1e6,2e6,3e6]); t=np.array([1e6+1,2e6,3e6+2]); a,*_=camera_matrix(t,25,10,5,1.5,10); b,*_=camera_matrix(t-o,25,10,5,1.5,10); x=project_points(pts,a,800,600); y=project_points(pts-o,b,800,600); np.testing.assert_allclose(x[0],y[0],atol=1e-10); np.testing.assert_allclose(x[1],y[1],atol=1e-10)
if __name__=='__main__': unittest.main()
