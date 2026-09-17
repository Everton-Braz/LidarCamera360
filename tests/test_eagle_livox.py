"""Unit tests for Eagle LiDAR (Livox CustomMsg) and Insta360 X4/X6 camera integration."""
import json
import struct
import unittest
from pathlib import Path
import yaml
import numpy as np

from raven_app.bag_io import cloud_payload, detect_bag_topics
from raven_app.config import get_app_root


class TestEagleLivoxIntegration(unittest.TestCase):

    def test_livox_custom_msg_binary_unpacking(self):
        """Test high-performance direct buffer unpacking of Livox CustomMsg points."""
        point_count = 100
        points_data = bytearray()
        for i in range(point_count):
            offset_ns = i * 10000  # 10 microseconds step
            x = float(i * 0.1)
            y = float(-i * 0.05)
            z = float(i * 0.02)
            reflectivity = min(255, i)
            tag = 0
            line = i % 4
            # Struct: <u4 offset_time, <f4 x, <f4 y, <f4 z, u1 reflectivity, u1 tag, u1 line
            points_data += struct.pack('<IfffBBB', offset_ns, x, y, z, reflectivity, tag, line)

        frame_id_bytes = b"livox_frame"
        buf = bytearray()
        # Header: seq(4), stamp_sec(4), stamp_nsec(4) = 12 bytes
        buf += struct.pack('<III', 1, 1000, 500000)
        # frame_id: len(4) + string bytes
        buf += struct.pack('<I', len(frame_id_bytes)) + frame_id_bytes
        # timebase(8), point_num(4), lidar_id(1), rsvd(3) = 16 bytes
        buf += struct.pack('<QIB3s', 1000000500000, point_count, 0, b'\x00\x00\x00')
        # points array vector length (4)
        buf += struct.pack('<I', point_count)
        # array of point structs (point_count * 19 bytes)
        buf += points_data

        # Test cloud_payload with raw bytes
        cloud_bytes, count = cloud_payload(None, raw=bytes(buf), msgtype='livox_ros_driver2/msg/CustomMsg')
        self.assertEqual(count, point_count)
        self.assertEqual(len(cloud_bytes), point_count * 20)  # 5 floats per point

        pts = np.frombuffer(cloud_bytes, dtype='<f4').reshape(-1, 5)
        # Check first point: x, y, z, intensity, curvature
        self.assertAlmostEqual(float(pts[0, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(pts[0, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(pts[0, 2]), 0.0, places=4)
        self.assertAlmostEqual(float(pts[0, 4]), 0.0, places=4)

        # Check last point
        self.assertAlmostEqual(float(pts[-1, 0]), (point_count - 1) * 0.1, places=4)
        self.assertAlmostEqual(float(pts[-1, 4]), (99 * 10000) * 1e-6, places=4)

    def test_eagle_config_files_exist_and_valid(self):
        """Verify that eagle.yaml and rig_profile_eagle.json exist and are well-formed."""
        root = get_app_root()

        # Check FAST-LIVO2 config
        flv2_eagle = root / "FAST-LIVO2" / "config" / "eagle.yaml"
        self.assertTrue(flv2_eagle.is_file(), f"Missing {flv2_eagle}")
        with open(flv2_eagle, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.assertEqual(cfg.get('preprocess', {}).get('lidar_type'), 1)
        self.assertEqual(cfg.get('common', {}).get('lid_topic'), '/livox/lidar')
        self.assertEqual(cfg.get('common', {}).get('imu_topic'), '/livox/imu')

        # Check Rig Profile config
        rig_eagle = root / "configs" / "rig_profile_eagle.json"
        self.assertTrue(rig_eagle.is_file(), f"Missing {rig_eagle}")
        with open(rig_eagle, 'r', encoding='utf-8') as f:
            rig = json.load(f)
        self.assertEqual(rig.get('scanner'), 'Eagle_LiDAR')
        self.assertEqual(rig.get('topics', {}).get('lidar'), '/livox/lidar')
        self.assertEqual(rig.get('topics', {}).get('imu'), '/livox/imu')
        self.assertIn('T_lidar_to_cam0_rigid_4x4', rig)
        self.assertIn('T_lidar_to_cam1_rigid_4x4', rig)


if __name__ == '__main__':
    unittest.main()
