"""Stream ROS1 bags into the native estimator without a ROS installation.

FLV2 v1: 8-byte magic; records <uint8 kind, float64 seconds, uint32 count>.
IMU=1: six float64 (gyro XYZ, acceleration XYZ); cloud=2: count * five
float32 (XYZ, intensity, relative milliseconds); image=3: count encoded bytes.
All values are little endian. Input starts at the beginning for IMU initialization.
"""
import re
from pathlib import Path
import struct
from typing import Optional, Union, List, Dict, Any, Tuple
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


def rotate_image(img, rotation_deg):
    """Rotate a decoded frame clockwise by 0, 90, 180 or 270 degrees."""
    import cv2
    rotation_deg = int(rotation_deg or 0) % 360
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if rotation_deg and rotation_deg not in codes:
        raise ValueError(f'Image rotation must be a multiple of 90 degrees, got {rotation_deg}')
    return cv2.rotate(img, codes[rotation_deg]) if rotation_deg else img


def image_payload(msg, rotation_deg=0, max_side=1200, scale=0.25):
    """Encode an image message for the VIO engine, rotated and downscaled like its camera model.

    Frames wider than ``max_side`` are reduced by ``scale``: prepare_fastlivo_config
    scales the camera model the same way, which keeps the C++ VIO within memory.
    """
    import cv2
    if hasattr(msg, 'format'):
        data = bytes(msg.data)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('Image decoding failed')
        if not rotation_deg and max(img.shape[:2]) <= max_side:
            return data
    else:
        enc = msg.encoding.lower()
        channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1, 'bgra8': 4, 'rgba8': 4}.get(enc)
        if not channels:
            raise ValueError(f'Unsupported image encoding {enc!r}')
        if len(msg.data) < msg.step * msg.height:
            raise ValueError('Truncated image')
        img = np.ndarray((msg.height, msg.width, channels), dtype=np.uint8, buffer=msg.data,
                         strides=(msg.step, channels, 1))
        if enc == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif enc == 'rgba8': img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    img = rotate_image(img, rotation_deg)
    if max(img.shape[:2]) > max_side:
        img = cv2.resize(img, (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    if hasattr(msg, 'format'):
        ok, encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        # PNG is lossless; do not introduce JPEG artifacts into photometric estimation.
        ok, encoded = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok: raise ValueError('Image encoding failed')
    return encoded.tobytes()


def inspect_bags(paths):
    from rosbags.highlevel import AnyReader
    try:
        with AnyReader([Path(p) for p in paths]) as reader:
            return {'duration_seconds': reader.duration / 1e9,
                    'messages': reader.message_count,
                    'connections': [{'topic': c.topic, 'type': c.msgtype, 'messages': c.msgcount} for c in reader.connections]}
    except Exception:
        return {'duration_seconds': 0.0, 'messages': 0, 'connections': []}


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

    if connections:
        raw_has_lidar = bool(best_lidar)
        raw_has_imu = bool(best_imu)
        raw_has_cam = bool(best_cam)
    else:
        # Mock/empty file: do not treat as an image bag
        raw_has_lidar = True
        raw_has_imu = True
        raw_has_cam = True

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
        'has_lidar': raw_has_lidar,
        'has_imu': raw_has_imu,
        'has_image': raw_has_cam,
        'detected_lidar': best_lidar if raw_has_lidar else None,
        'detected_imu': best_imu if raw_has_imu else None,
        'detected_cam': best_cam if raw_has_cam else None,
        'scanner_type': scanner_type,
        'connections': connections,
        'duration_seconds': info.get('duration_seconds', 0.0),
        'messages': info.get('messages', 0)
    }


def export_bags(paths, stream, *, lidar_topic=None, imu_topic=None,
                image_topic=None, lio=False, time_field='timestamp',
                time_unit='seconds', time_origin='first-point', image_rotation_deg=0):
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
        import gc
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
                data = image_payload(msg, image_rotation_deg)
                n = len(data)
            stream.write(HEADER.pack(kind, t, n))
            stream.write(data)
            counts[kind] += 1
            if counts[2] > 0 and counts[2] % 500 == 0 and kind == 2:
                del data, msg
                gc.collect(1)
        stream.flush()
    return {'imu': counts[1], 'scans': counts[2], 'images': counts[3]}


def find_adjacent_image_bag(bag_path: Path) -> Optional[Path]:
    """Find candidate IMAGE_*.bag adjacent to a LiDAR bag."""
    if not bag_path:
        return None
    bag_path = Path(bag_path).resolve()
    parent = bag_path.parent
    if not parent.is_dir():
        return None
    name = bag_path.name

    # 1. Direct prefix replacement: LIDAR_xxx -> IMAGE_xxx, PCL_xxx -> IMAGE_xxx
    if name.startswith("LIDAR_"):
        cand = parent / ("IMAGE_" + name[6:])
        if cand.is_file() and cand.resolve() != bag_path:
            return cand
    elif name.startswith("PCL_"):
        cand = parent / ("IMAGE_" + name[4:])
        if cand.is_file() and cand.resolve() != bag_path:
            return cand
    elif "lidar" in name.lower():
        cand_name = re.sub(r'lidar', 'image', name, flags=re.IGNORECASE)
        cand = parent / cand_name
        if cand.is_file() and cand.resolve() != bag_path:
            return cand

    # 2. Search parent directory for IMAGE_*.bag (excluding bag_path itself)
    image_candidates = [p for p in sorted(parent.glob("IMAGE_*.bag")) if p.resolve() != bag_path]
    if image_candidates:
        return image_candidates[0]
    image_candidates_lower = [
        p for p in sorted(parent.glob("*.bag"))
        if "image" in p.name.lower() and p.resolve() != bag_path
    ]
    if image_candidates_lower:
        return image_candidates_lower[0]

    return None


def find_adjacent_lidar_bag(bag_path: Path) -> Optional[Path]:
    """Find candidate LIDAR_*.bag adjacent to an Image bag."""
    if not bag_path:
        return None
    bag_path = Path(bag_path).resolve()
    parent = bag_path.parent
    if not parent.is_dir():
        return None
    name = bag_path.name

    # 1. Direct prefix replacement: IMAGE_xxx -> LIDAR_xxx, IMAGE_xxx -> PCL_xxx
    if name.startswith("IMAGE_"):
        for prefix in ("LIDAR_", "PCL_"):
            cand = parent / (prefix + name[6:])
            if cand.is_file() and cand.resolve() != bag_path:
                return cand
    elif "image" in name.lower():
        for repl in ("lidar", "pcl"):
            cand_name = re.sub(r'image', repl, name, flags=re.IGNORECASE)
            cand = parent / cand_name
            if cand.is_file() and cand.resolve() != bag_path:
                return cand

    # 2. Search parent directory for LIDAR_*.bag or PCL_*.bag (excluding bag_path itself)
    lidar_candidates = [p for p in sorted(parent.glob("LIDAR_*.bag")) if p.resolve() != bag_path]
    if lidar_candidates:
        return lidar_candidates[0]
    pcl_candidates = [p for p in sorted(parent.glob("PCL_*.bag")) if p.resolve() != bag_path]
    if pcl_candidates:
        return pcl_candidates[0]
    lidar_candidates_lower = [
        p for p in sorted(parent.glob("*.bag"))
        if any(k in p.name.lower() for k in ("lidar", "pcl", "raven", "eagle")) and p.resolve() != bag_path
    ]
    if lidar_candidates_lower:
        return lidar_candidates_lower[0]

    return None


def discover_bag_image_topics(bag_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Discover image topics within a ROS1 bag file.

    Returns:
        List of dicts: [{'topic': str, 'msgtype': str, 'msgcount': int}, ...]
    """
    p = Path(bag_path)
    if not p.is_file():
        return []
    from rosbags.highlevel import AnyReader
    results = []
    try:
        with AnyReader([p]) as reader:
            for c in reader.connections:
                if 'Image' in c.msgtype or 'image' in c.topic.lower() or 'camera' in c.topic.lower():
                    results.append({
                        'topic': c.topic,
                        'msgtype': c.msgtype,
                        'msgcount': getattr(c, 'msgcount', 0),
                    })
    except Exception:
        pass
    return results


def first_image_size(bag_path: Union[str, Path], topic: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """(width, height) of the first decoded frame of an image topic, as stored in the bag."""
    import cv2
    from rosbags.highlevel import AnyReader
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections
                 if (c.topic == topic if topic else 'Image' in c.msgtype)]
        for c, _, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, c.msgtype)
            if hasattr(msg, 'format'):
                img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
                return (int(img.shape[1]), int(img.shape[0])) if img is not None else None
            return int(msg.width), int(msg.height)
    return None


def get_bag_start_time(bag_path: Union[str, Path], topic: Optional[str] = None) -> Optional[float]:
    """Quickly read the first timestamp from a bag file in seconds."""
    p = Path(bag_path)
    if not p.is_file():
        return None
    from rosbags.highlevel import AnyReader
    try:
        with AnyReader([p]) as reader:
            conns = reader.connections
            if topic:
                conns = [c for c in conns if c.topic == topic]
            for c, _, raw in reader.messages(connections=conns):
                msg = reader.deserialize(raw, c.msgtype)
                return stamp(msg)
    except Exception:
        pass
    return None


def extract_images(bag_path: Path, output_dir: Path, *, image_bag_path: Optional[Path] = None,
                   image_topic: Optional[str] = None, fps: float = 2.0,
                   camera_name: str = "cam0", ref_start_time: Optional[float] = None) -> int:
    """Extract front camera frames from an IMAGE_*.bag or LiDAR bag into output_dir.

    Creates output_dir/<camera_name> and output_dir/frames.json for workflow compatibility.
    """
    from rosbags.highlevel import AnyReader
    import json
    import cv2

    bag_path = Path(bag_path)
    output_dir = Path(output_dir)

    target_bag = None
    if image_bag_path is not None and Path(image_bag_path).is_file():
        target_bag = Path(image_bag_path)
    else:
        target_bag = find_adjacent_image_bag(bag_path) or bag_path

    out_images = output_dir / camera_name
    out_images.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    timestamps = {}
    last_t = -1e9
    min_interval = 1.0 / max(0.1, fps) if fps > 0 else 0.0
    first_t = None

    with AnyReader([target_bag]) as reader:
        conn_topics = {c.topic: c for c in reader.connections}
        target_topic = image_topic
        if not target_topic or target_topic not in conn_topics:
            candidates = ["/camera_front/image_raw", "/camera_front/image/compressed"]
            for cand in candidates:
                if cand in conn_topics:
                    target_topic = cand
                    break
            if not target_topic:
                for c in reader.connections:
                    if 'Image' in c.msgtype or 'image' in c.topic.lower() or 'camera' in c.topic.lower():
                        target_topic = c.topic
                        break

        if not target_topic or target_topic not in conn_topics:
            return 0

        connections = [conn_topics[target_topic]]
        frame_idx = 1
        for c, _, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, c.msgtype)
            t = stamp(msg)
            if first_t is None:
                first_t = t
            if t - last_t < min_interval:
                continue
            last_t = t

            if hasattr(msg, 'format'):
                data = bytes(msg.data)
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                enc = msg.encoding.lower()
                channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1, 'bgra8': 4, 'rgba8': 4}.get(enc, 3)
                img = np.ndarray((msg.height, msg.width, channels), dtype=np.uint8, buffer=msg.data,
                                 strides=(msg.step, channels, 1))
                if enc == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif enc == 'rgba8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = out_images / frame_filename
            cv2.imwrite(str(frame_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rel_name = f"{camera_name}/{frame_filename}"
            # If reference start time provided (e.g. from LiDAR bag), compute relative seconds
            t_val = (t - ref_start_time) if ref_start_time is not None else t
            timestamps[rel_name] = float(t_val)
            frame_idx += 1
            extracted_count += 1

    manifest_path = output_dir / "frames.json"
    existing_manifest = {}
    if manifest_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}
    merged_timestamps = existing_manifest.get("timestamps", {})
    merged_timestamps.update(timestamps)
    manifest = {
        "schema": 1,
        "signature": f"bag_{target_bag.stem}_{extracted_count}",
        "time_source": "ros_bag",
        "timestamps": merged_timestamps
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return extracted_count


def extract_bag_auxiliary_frames(
    config: Any,
    dataset_dir: Union[str, Path],
    target_fps: Optional[float] = None,
    sharpness_window: int = 3,
    jpeg_quality: int = 95,
) -> Tuple[List[Path], Dict[str, float]]:
    """Extract frames from a ROS bag auxiliary camera into dataset/images/<camera_name>.

    Returns:
        (extracted_paths, timestamp_dict) mapping e.g. 'cam2/frame_000001.jpg' to relative seconds.
    """
    import json
    import cv2
    from rosbags.highlevel import AnyReader

    dataset_dir = Path(dataset_dir)
    images_root = dataset_dir / "images"
    cam_dir = images_root / config.camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)

    source_p = Path(config.source_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Bag camera source does not exist: {source_p}")

    fps = float(config.fps if target_fps is None else target_fps)
    if fps <= 0:
        fps = 2.0
    rotation_deg = int(getattr(config, 'image_rotation_deg', 0) or 0) % 360

    source_stat = source_p.stat()
    aux_sig_key = f"signature_{config.camera_name}"
    curr_sig = {
        "source": str(source_p.resolve()),
        "size": source_stat.st_size,
        "mtime_ns": source_stat.st_mtime_ns,
        "fps": fps,
        "camera_name": config.camera_name,
        "rotation_deg": rotation_deg,
        "version": 2,
    }

    manifest_path = images_root / "frames.json"
    existing_data: Dict[str, Any] = {"schema": 1, "timestamps": {}}
    if manifest_path.is_file():
        try:
            existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "timestamps" not in existing_data or not isinstance(existing_data["timestamps"], dict):
                existing_data["timestamps"] = {}
        except Exception:
            existing_data = {"schema": 1, "timestamps": {}}

    existing_timestamps = existing_data["timestamps"]
    if existing_data.get(aux_sig_key) == curr_sig:
        prefix = f"{config.camera_name}/"
        matched = {k: v for k, v in existing_timestamps.items() if k.startswith(prefix)}
        if matched and all((images_root / k).is_file() for k in matched):
            extracted = sorted([images_root / k for k in matched])
            print(f"[*] Reusing completed bag frame extraction for '{config.camera_name}' ({len(extracted)} frames; cached).")
            return extracted, matched

    # Identify target topic
    topics_info = discover_bag_image_topics(source_p)
    target_topic = None
    if getattr(config, 'source_stream_index', None) is not None and 0 <= config.source_stream_index < len(topics_info):
        target_topic = topics_info[config.source_stream_index]['topic']
    elif topics_info:
        # Match by camera name index if multiple topics
        digits = ''.join(filter(str.isdigit, config.camera_name))
        cam_idx = (int(digits) - 2) if digits and int(digits) >= 2 else 0
        if 0 <= cam_idx < len(topics_info):
            target_topic = topics_info[cam_idx]['topic']
        else:
            target_topic = topics_info[0]['topic']

    # Find reference start time from LiDAR bag if available
    ref_start_time = None
    cand_lidar_bags = list(source_p.parent.glob("LIDAR_*.bag")) or list(dataset_dir.glob("*.bag"))
    if cand_lidar_bags:
        ref_start_time = get_bag_start_time(cand_lidar_bags[0])

    extracted: List[Path] = []
    timestamps: Dict[str, float] = {}
    last_t = -1e9
    min_interval = 1.0 / max(0.1, fps)
    first_t = None

    with AnyReader([source_p]) as reader:
        conn_topics = {c.topic: c for c in reader.connections}
        if not target_topic or target_topic not in conn_topics:
            candidates = ["/camera_front/image_raw", "/camera_front/image/compressed"]
            for cand in candidates:
                if cand in conn_topics:
                    target_topic = cand
                    break
            if not target_topic and topics_info:
                target_topic = topics_info[0]['topic']

        if not target_topic or target_topic not in conn_topics:
            raise RuntimeError(f"No valid image topic found in bag: {source_p}")

        connections = [conn_topics[target_topic]]
        frame_idx = 1
        for c, _, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, c.msgtype)
            t = stamp(msg)
            if first_t is None:
                first_t = t
            if t - last_t < min_interval:
                continue
            last_t = t

            if hasattr(msg, 'format'):
                data = bytes(msg.data)
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                enc = msg.encoding.lower()
                channels = {'bgr8': 3, 'rgb8': 3, 'mono8': 1, 'bgra8': 4, 'rgba8': 4}.get(enc, 3)
                img = np.ndarray((msg.height, msg.width, channels), dtype=np.uint8, buffer=msg.data,
                                 strides=(msg.step, channels, 1))
                if enc == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif enc == 'rgba8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)

            frame_filename = f"frame_{frame_idx:06d}.jpg"
            frame_path = cam_dir / frame_filename
            img = rotate_image(img, rotation_deg)
            cv2.imwrite(str(frame_path), img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            rel_name = f"{config.camera_name}/{frame_filename}"
            t_val = (t - ref_start_time) if ref_start_time is not None else (t - first_t)
            timestamps[rel_name] = float(t_val)
            extracted.append(frame_path)
            frame_idx += 1

    existing_timestamps.update(timestamps)
    existing_data[aux_sig_key] = curr_sig
    existing_data["timestamps"] = existing_timestamps
    existing_data.setdefault("time_sources", {})[config.camera_name] = "bag"
    # Frame times count from this bag-clock instant; the workflow turns it into the exact
    # offset against the SLAM trajectory (see bag_time_origin).
    origin = ref_start_time if ref_start_time is not None else first_t
    if origin is not None:
        existing_data.setdefault("time_origins", {})[config.camera_name] = float(origin)
    manifest_path.write_text(json.dumps(existing_data, indent=2), encoding="utf-8")
    return extracted, timestamps


def bag_time_origin(dataset_dir, camera_name):
    """Absolute bag-clock time (s) that ``camera_name``'s frame times in frames.json count from.

    Older manifests only kept the first frame's absolute stamp; the origin is then
    recovered from that frame's relative time.
    """
    import json
    manifest = json.loads((Path(dataset_dir) / 'images' / 'frames.json').read_text(encoding='utf-8'))
    origin = (manifest.get('time_origins') or {}).get(camera_name)
    if origin is not None:
        return float(origin)
    first = manifest.get('first_timestamp_sec')
    times = [t for name, t in (manifest.get('timestamps') or {}).items() if name.startswith(f'{camera_name}/')]
    if first is None or not times:
        raise ValueError(f'frames.json has no bag time origin for {camera_name}; re-extract its frames')
    return float(first) - min(times)
