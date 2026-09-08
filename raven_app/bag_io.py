"""Stream ROS1 bags into the native estimator without a ROS installation.

FLV2 v1: 8-byte magic; records <uint8 kind, float64 seconds, uint32 count>.
IMU=1: six float64 (gyro XYZ, acceleration XYZ); cloud=2: count * five
float32 (XYZ, intensity, relative milliseconds); image=3: count encoded bytes.
All values are little endian. Input starts at the beginning for IMU initialization.
"""
from pathlib import Path
import struct

MAGIC = b'FLV2\x01\0\0\0'
HEADER = struct.Struct('<BdI')


def stamp(msg):
    s = msg.header.stamp
    return s.sec + s.nanosec * 1e-9


def point_array(msg):
    import numpy as np
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


def cloud_payload(msg, time_field='timestamp', time_unit='seconds', time_origin='first-point'):
    import numpy as np
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
    import numpy as np
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


def export_bags(paths, stream, *, lidar_topic='/vanjee_722z', imu_topic='/vanjee_imu_packets',
                image_topic='/camera_front/image_raw', lio=False, time_field='timestamp',
                time_unit='seconds', time_origin='first-point'):
    from rosbags.highlevel import AnyReader
    import math
    last={}
    counts={1:0,2:0,3:0}
    with AnyReader([Path(p) for p in paths]) as reader:
        topics={lidar_topic:2,imu_topic:1}
        if not lio: topics[image_topic]=3
        connections=[c for c in reader.connections if c.topic in topics]
        missing=set(topics)-{c.topic for c in connections}
        if missing: raise ValueError('Missing bag topics: '+', '.join(sorted(missing)))
        stream.write(MAGIC)
        for c,_,raw in reader.messages(connections=connections):
            msg=reader.deserialize(raw,c.msgtype)
            kind=topics[c.topic];t=stamp(msg)
            if not math.isfinite(t) or t<last.get(kind,float('-inf')):
                raise ValueError(f'Non-monotonic header timestamp on {c.topic}; inspect overlapping/split bags')
            last[kind]=t
            if kind==1:
                g=msg.angular_velocity;a=msg.linear_acceleration
                data=struct.pack('<6d',g.x,g.y,g.z,a.x,a.y,a.z);n=6
            elif kind==2:
                data,n=cloud_payload(msg,time_field,time_unit,time_origin)
            else:
                data=image_payload(msg);n=len(data)
            stream.write(HEADER.pack(kind,t,n));stream.write(data);counts[kind]+=1
        stream.flush()
    return {'imu':counts[1],'scans':counts[2],'images':counts[3]}
