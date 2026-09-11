import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from raven_app.cloud_io import load_cloud

class CloudIoTests(unittest.TestCase):
    def setUp(self): self.t = tempfile.TemporaryDirectory(); self.d = Path(self.t.name)
    def tearDown(self): self.t.cleanup()
    def test_ascii_rgb_filter(self):
        p=self.d/'a.pcd'; p.write_text('FIELDS x y z r g b\nSIZE 4 4 4 1 1 1\nTYPE F F F U U U\nCOUNT 1 1 1 1 1 1\nPOINTS 2\nDATA ascii\n1 2 3 255 0 128\nnan 4 5 0 255 0\n')
        c=load_cloud(p); self.assertEqual(c.original_count,2); np.testing.assert_allclose(c.colors,[[1,0,128/255]])
    def test_binary_packed_float(self):
        p=self.d/'a.pcd'; h=b'FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\nPOINTS 1\nDATA binary\n'; p.write_bytes(h+struct.pack('<ffff',1,2,3,struct.unpack('<f',struct.pack('<I',0x80402010))[0])); np.testing.assert_allclose(load_cloud(p).colors,[[64/255,32/255,16/255]])
    def test_ply_big_endian_intensity(self):
        p=self.d/'a.ply'; h='ply\nformat binary_big_endian 1.0\nelement vertex 2\nproperty double x\nproperty float y\nproperty short z\nproperty uchar intensity\nend_header\n'; p.write_bytes(h.encode()+struct.pack('>dfhBdfhB',1,2,3,64,4,5,6,255)); c=load_cloud(p); np.testing.assert_allclose(c.points,[[1,2,3],[4,5,6]])
    def test_errors(self):
        p=self.d/'bad.pcd'; p.write_bytes(b'FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 1\nDATA binary_compressed\n'); self.assertRaisesRegex(ValueError,'binary_compressed',load_cloud,p)
        q=self.d/'bad.ply'; q.write_text('ply\nformat ascii 1.0\nelement vertex 1\nproperty list uchar int x\nend_header\n'); self.assertRaisesRegex(ValueError,'list',load_cloud,q)
if __name__=='__main__': unittest.main()
