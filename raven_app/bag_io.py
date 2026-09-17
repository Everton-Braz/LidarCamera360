"""Stream ROS1 bags into the native estimator without a ROS installation.

FLV2 v1: 8-byte magic; records <uint8 kind, float64 seconds, uint32 count>.
IMU=1: six float64 (gyro XYZ, acceleration XYZ); cloud=2: count * five
float32 (XYZ, intensity, relative milliseconds); image=3: count encoded bytes.
All values are little endian. Input starts at the beginning for IMU initialization.
"""
from pathlib import Path
import struct
import numpy as np

MAGIC = b'FLV2\x01\0\0\0'
HEADER = struct.Struct('<BdI')

_LIVOX_POINT_DTYPE = np.dtype([
    ('offset_time', '<u4'),
    ('x', '<f4'),
    ('y', '<f4'),
    ('z', '<f4'),
    ('reflectivity', 'u1'),
    ('tag', 'u1'),
    ('line', 'u1')
])


def stamp(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def point_array(msg):
    types = {1:'i1', 2:'u1', 3:'i2', 4:'u2', 5:'i4', 6:'u4', 7:'f4', 8:'f8'}
    endian = '>' if msg.is_bigendian else '<'
    fields = [f for f in msg.fields if f.count == 1]
    dtype = np.dtype({'names':[f.name for f in fields],
                     'formats':[endian+types[f.datatype] for f in fields],
                     'offsets':[f.offset for f in fields], 'itemsize':msg.point_step})
    if len(msg.data) < msg.row_step * msg.height:
        raise ValueError('Truncated PointCloud2 data')
    return np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                      strides=(msg.row_step,msg.point_step)).reshape(-1)


def cloud_payload(msg, time_field='timestamp', time_unit='seconds', time_origin='first-point',
                  raw=None, msgtype=''):
    """Extract (x, y, z, intensity, curvature_in_ms) from PointCloud2 or Livox CustomMsg."""
    # Check if message is a Livox CustomMsg (Eagle / Mid-360 / Avia / Livox ROS driver)
    is_livox = ('CustomMsg' in msgtype) or hasattr(msg, 'points') or ('livox' in type(msg).__name__.lower())
    if is_livox:
        if raw is not None:
            try:
                frame_len = struct.unpack_from('<I', raw, 12)[0]
                pos = 16 + frame_len
                timebase, point_num, lidar_id = struct.unpack_from('<QIB', raw, pos)
                pos += 8 + 4 + 1 + 3  # 3 reserved bytes
                points_len = struct.unpack_from('<I', raw, pos)[0]
                pos += 4
                raw_pts = np.frombuffer(raw, dtype=_LIVOX_POINT_DTYPE, count=point_num, offset=pos)
                out = np.zeros((point_num, 5), dtype='<f4')
                out[:, 0] = raw_pts['x']
                out[:, 1] = raw_pts['y']
                out[:, 2] = raw_pts['z']
                out[:, 3] = raw_pts['reflectivity']
                out[:, 4] = raw_pts['offset_time'].astype(np.float32) * 1e-6  # ns to ms (0..1000)
                return out.tobytes(), point_num
            except Exception:
                pass  # Fall back to object-based unpacking

        pts = getattr(msg, 'points', [])
        n = len(pts)
        out = np.zeros((n, 5), dtype='<f4')
        for i, p in enumerate(pts):
            out[i, 0] = p.x
            out[i, 1] = p.y
            out[i, 2] = p.z
            out[i, 3] = getattr(p, 'reflectivity', 0)
            out[i, 4] = getattr(p, 'offset_time', 0) * 1e-6
        return out.tobytes(), n

    # Standard sensor_msgs/msg/PointCloud2
    a = point_array(msg)
    for name in ('x','y','z',time_field):
        if name not in a.dtype.names:
            raise ValueError(f'PointCloud2 lacks {name!r}; set --time-field/--time-unit for this sensor')
    out = np.zeros((len(a),5),dtype='<f4')
    for i,name in enumerate(('x','y','z')):
        out[:,i] = a[name]
    if 'intensity' in a.dtype.names:
        out[:,3] = a['intensity']
    t = a[time_field].astype(np.float64)
    scale = {'seconds':1000.,'milliseconds':1.,'microseconds':.001,'nanoseconds':.000001}[time_unit]
    if len(t):
        if time_origin == 'first-point':
            t -= t[0]
        elif time_origin == 'header':
            t -= stamp(msg)*1000./scale
    out[:,4] = t*scale
    return out.tobytes(),len(a)


def image_payload(msg):
    import cv2
    if hasattr(msg,'format'):
        return bytes(msg.data)
    enc=msg.encoding.lower()
    channels={'bgr8':3,'rgb8':3,'mono8':1,'bgra8':4,'rgba8':4}.get(enc)
    if not channels:
        raise ValueError(f'Unsupported image encoding {enc!r}')
    if len(msg.data)<msg.step*msg.height:
        raise ValueError('Truncated image')
    img=np.ndarray((msg.height,msg.width,channels),dtype=np.uint8,buffer=msg.data,
                   strides=(msg.step,channels,1))
    if enc=='rgb8': img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    elif enc=='rgba8': img=cv2.cvtColor(img,cv2.COLOR_RGBA2BGRA)
    # PNG is lossless; do not introduce JPEG artifacts into photometric estimation.
    ok,encoded=cv2.imencode('.png',img,[cv2.IMWRITE_PNG_COMPRESSION,1])
    if not ok: raise ValueError('Image encoding failed')
    return encoded.tobytes()


def inspect_bags(paths):
    from rosbags.highlevel import AnyReader
    with AnyReader([Path(p) for p in paths]) as reader:
        return {'duration_seconds':reader.duration/1e9,
                'messages':reader.message_count,
                'connections':[{'topic':c.topic,'type':c.msgtype,'messages':c.msgcount} for c in reader.connections]}


def detect_bag_topics(paths, default_lidar=None, default_imu=None, default_cam=None):
    """Inspect bag connections and return best detected topics and scanner profile name."""
    info = inspect_bags(paths)
    connections = info.get('connections', [])
    best_lidar = None
    best_imu = None
    best_cam = None
    scanner_type = 'generic'

    for conn in connections:
        topic = conn.get('topic', '')
        msgtype = conn.get('type', '')
        # LiDAR detection
        if 'CustomMsg' in msgtype or '/livox/lidar' in topic:
            best_lidar = topic
            scanner_type = 'eagle'
        elif ('PointCloud2' in msgtype or 'vanjee_722z' in topic or 'lidar' in topic.lower() or 'points' in topic.lower()) and not best_lidar:
            best_lidar = topic
            if 'vanjee' in topic.lower():
                scanner_type = 'raven'

        # IMU detection
        if 'Imu' in msgtype or 'imu' in topic.lower():
            if not best_imu or '/livox/imu' in topic or '/vanjee_imu_packets' in topic:
                best_imu = topic

        # Camera detection
        if 'Image' in msgtype or 'camera' in topic.lower() or 'image' in topic.lower():
            if not best_cam:
                best_cam = topic

    if not best_lidar and default_lidar:
        best_lidar = default_lidar
    if not best_imu and default_imu:
        best_imu = default_imu
    if not best_cam and default_cam:
        best_cam = default_cam

    return {
        'lidar_topic': best_lidar or '/vanjee_722z',
        'imu_topic': best_imu or '/vanjee_imu_packets',
        'image_topic': best_cam or '/camera_front/image/compressed',
        'scanner_type': scanner_type,
        'connections': connections,
        'duration_seconds': info.get('duration_seconds', 0.0),
        'messages': info.get('messages', 0)
    }


def export_bags(paths, stream, *, lidar_topic=None, imu_topic=None,
                image_topic=None, lio=False, time_field='timestamp',
                time_unit='seconds', time_origin='first-point'):
    from rosbags.highlevel import AnyReader
    import math
    last={}
    counts={1:0,2:0,3:0}
    with AnyReader([Path(p) for p in paths]) as reader:
        conn_topics = {c.topic for c in reader.connections}

        # Auto-resolve topics if not provided or if defaults aren't in bag
        if lidar_topic is None or imu_topic is None or (lidar_topic not in conn_topics and imu_topic not in conn_topics):
            detected = detect_bag_topics(paths, default_lidar=lidar_topic, default_imu=imu_topic, default_cam=image_topic)
            if lidar_topic is None or lidar_topic not in conn_topics:
                lidar_topic = detected['lidar_topic']
            if imu_topic is None or imu_topic not in conn_topics:
                imu_topic = detected['imu_topic']
            if not lio and (image_topic is None or image_topic not in conn_topics):
                image_topic = detected['image_topic']

        topics = {lidar_topic: 2, imu_topic: 1}
        if not lio and image_topic:
            topics[image_topic] = 3

        connections = [c for c in reader.connections if c.topic in topics]
        missing = set(topics) - {c.topic for c in connections}
        if missing:
            raise ValueError('Missing bag topics: ' + ', '.join(sorted(missing)))

        stream.write(MAGIC)
        for c, _, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, c.msgtype)
            kind = topics[c.topic]
            t = stamp(msg)
            if not math.isfinite(t) or t < last.get(kind, float('-inf')):
                raise ValueError(f'Non-monotonic header timestamp on {c.topic}; inspect overlapping/split bags')
            last[kind] = t
            if kind == 1:
                g = msg.angular_velocity
                a = msg.linear_acceleration
                data = struct.pack('<6d', g.x, g.y, g.z, a.x, a.y, a.z)
                n = 6
            elif kind == 2:
                data, n = cloud_payload(msg, time_field, time_unit, time_origin, raw=raw, msgtype=c.msgtype)
            else:
                data = image_payload(msg)
                n = len(data)
            stream.write(HEADER.pack(kind, t, n))
            stream.write(data)
            counts[kind] += 1
        stream.flush()
    return {'imu': counts[1], 'scans': counts[2], 'images': counts[3]}

