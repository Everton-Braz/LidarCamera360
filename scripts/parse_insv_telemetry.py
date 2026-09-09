import struct
import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Inspect the telemetry trailer of an Insta360 INSV file.")
parser.add_argument("--insv", required=True, help="Path to the INSV file.")
insv_path = parser.parse_args().insv
file_size = os.path.getsize(insv_path) if os.path.exists(insv_path) else 0
HEADER_SIZE = 32 + 4 + 4 + 32 # 72

RECORD_NAMES = {
    0: "Offsets",
    1: "Metadata",
    2: "Thumbnail",
    3: "Gyro",
    4: "Exposure",
    5: "ThumbnailExt",
    6: "TimelapseTimestamp",
    7: "Gps",
    8: "StarNum",
    9: "AAAData",
    10: "Anchors",
    11: "AAASimulation",
    12: "ExposureSecondary",
    13: "Magnetic",
    14: "Euler",
    15: "SecGyro",
    16: "Speed",
    17: "TBox",
    18: "Quaternions",
    128: "TimeMap"
}

with open(insv_path, "rb") as fin:
    fin.seek(-HEADER_SIZE, 2)
    header = fin.read(HEADER_SIZE)
    magic = header[HEADER_SIZE-32:]
    print("Magic check:", magic == b"8db42d694ccc418790edff439fe026bf")
    
    extra_size = struct.unpack('<I', header[32:36])[0]
    version = struct.unpack('<I', header[36:40])[0]
    extra_start = file_size - extra_size
    print(f"extra_size = {extra_size:,} bytes, version = {version}, extra_start = {extra_start:,}")
    
    # Check offset table
    # let mut offset = (HEADER_SIZE + 4+1+1) as i64; // = 78
    # seek to End(-offset + 1) -> reads first_id (1 byte), format (1 byte), size (u32)
    fin.seek(-(HEADER_SIZE + 6), 2)
    tag_buf = fin.read(6)
    rec_id, rec_format = struct.unpack('<BB', tag_buf[:2])
    rec_size = struct.unpack('<I', tag_buf[2:6])[0]
    print(f"End tag: id={rec_id} ({RECORD_NAMES.get(rec_id, 'Unknown')}), format={rec_format}, size={rec_size}")
    
    if rec_id == 0: # Offsets
        fin.seek(-(HEADER_SIZE + 6 + rec_size), 2)
        offsets_data = fin.read(rec_size)
        print(f"Read {len(offsets_data)} bytes of offsets table.")
        
        # Parse entries: id (u8), format (u8), size (u32), offset (u32) -> 10 bytes per entry
        offsets = {}
        for i in range(0, len(offsets_data), 10):
            oid, ofmt, osize, ooff = struct.unpack('<BBII', offsets_data[i:i+10])
            name = RECORD_NAMES.get(oid, f"Unknown_{oid}")
            offsets[oid] = (name, ofmt, osize, ooff)
            print(f"  Record {oid:2d} ({name:18s}): format={ofmt}, size={osize:9,d} bytes, offset={ooff:9,d}")
            
