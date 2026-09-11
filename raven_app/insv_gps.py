"""Read native INSV GPS trailers without decoding video or requiring ExifTool.

53-byte GPS layout verified against ExifTool's QuickTimeStream.pl ProcessINSV.
GPS UTC is not a video presentation timestamp. Repeated fixes must not be
interpreted as a timed trajectory, and altitude datum is not declared here.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

MAGIC = b"8db42d694ccc418790edff439fe026bf"
GPS_RECORD = struct.Struct("<IIHcdcdcddd")
FIELDS = ("record_index", "timestamp_utc", "timestamp_epoch", "latitude_deg",
          "longitude_deg", "altitude_m", "speed_mps", "track_deg", "status",
          "valid", "unknown_uint32", "source_insv")


def gps_blocks(path):
    """Yield bounded GPS payloads from the trailer's variable-sized directory."""
    path = Path(path)
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        if size < 78:
            raise ValueError("INSV file is too short for a trailer")
        stream.seek(size - 72)
        header = stream.read(72)
        if header[-32:] != MAGIC:
            raise ValueError("Missing INSV trailer magic; unsupported or truncated file")
        length = struct.unpack_from("<I", header, 32)[0]
        if not 78 <= length <= size:
            raise ValueError("Invalid INSV trailer length")
        start, end = size - length, size - 72
        stream.seek(end - 6)
        fmt, kind, count = struct.unpack("<BBI", stream.read(6))
        if kind == 0 and fmt == 0:
            if count % 10 or count > length - 78:
                raise ValueError("Malformed INSV trailer directory")
            stream.seek(end - 6 - count)
            directory = stream.read(count)
            seen = set()
            for kind, fmt, count, offset in struct.iter_unpack("<BBII", directory):
                if not kind or not count:
                    continue
                if offset + count + 6 > length - 72:
                    raise ValueError("INSV directory entry lies outside trailer")
                if kind != 7:
                    continue
                if fmt != 0:
                    raise ValueError(f"Unsupported INSV GPS format {fmt}")
                if (offset, count) in seen:
                    continue
                seen.add((offset, count))
                stream.seek(start + offset)
                payload = stream.read(count)
                if stream.read(6) != struct.pack("<BBI", fmt, kind, count):
                    raise ValueError("INSV GPS directory/footer mismatch")
                yield payload
        else:
            # Older captures can have sequential records without a directory.
            while end > start:
                if end - start < 6:
                    raise ValueError("Truncated INSV record footer")
                stream.seek(end - 6)
                fmt, kind, count = struct.unpack("<BBI", stream.read(6))
                begin = end - 6 - count
                if begin < start:
                    raise ValueError("INSV record lies outside trailer")
                if kind == 7:
                    if fmt != 0:
                        raise ValueError(f"Unsupported INSV GPS format {fmt}")
                    stream.seek(begin)
                    yield stream.read(count)
                end = begin


def extract_gps(path):
    path = Path(path).resolve()
    rows = []
    for payload in gps_blocks(path):
        if len(payload) % GPS_RECORD.size:
            raise ValueError("Unsupported/truncated GPS payload: expected 53-byte records")
        for epoch, unknown, ms, status, lat, ns, lon, ew, speed, track, altitude in GPS_RECORD.iter_unpack(payload):
            lat = -abs(lat) if ns == b"S" else lat
            lon = -abs(lon) if ew in (b"W", b"O") else lon
            valid = (status == b"A" and ns in (b"N", b"S") and
                     ew in (b"E", b"W", b"O") and epoch > 0 and ms < 1000 and
                     all(math.isfinite(v) for v in (lat, lon, altitude, speed, track)) and
                     -90 <= lat <= 90 and -180 <= lon <= 180)
            stamp = epoch + ms / 1000
            utc = datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            rows.append(dict(zip(FIELDS, (len(rows), utc, stamp, lat, lon, altitude,
                                         speed, track, status.hex() if not status.isalpha() else status.decode("ascii", "replace"),
                                         valid, unknown, str(path)))))
    valid = [r for r in rows if r["valid"]]
    unique = {}
    for row in valid:
        key = tuple(row[k] for k in ("timestamp_epoch", "latitude_deg", "longitude_deg", "altitude_m"))
        unique.setdefault(key, row)
    fixes = sorted(unique.values(), key=lambda r: r["timestamp_epoch"])
    positions = {(r["latitude_deg"], r["longitude_deg"], r["altitude_m"]) for r in fixes}
    times = {r["timestamp_epoch"] for r in fixes}
    timed_positions = len(times) >= 3 and len(positions) >= 3
    report = {
        "source_insv": str(path), "gps_records": len(rows),
        "valid_records": len(valid), "invalid_records": len(rows) - len(valid),
        "unique_fixes": len(fixes), "duplicate_valid_records": len(valid) - len(fixes),
        "unique_timestamps": len(times), "unique_positions": len(positions),
        "gps_time_span_seconds": max(times) - min(times) if times else 0,
        "horizontal_crs": "EPSG:4326", "altitude_datum": "unspecified; retained as recorded",
        "video_time_mapping": "unverified; GPS UTC must be synchronized to video timestamps",
        "has_candidate_trajectory": timed_positions,
        "georeferencing_status": ("requires_time_sync_and_alignment_validation" if timed_positions else
                                  "location_anchor_only" if fixes else "no_valid_gps"),
        "reason": ("Validate timing, spatial extent and fit residuals before transforming geometry." if timed_positions else
                   "GPS does not supply a moving timed trajectory; heading/scale cannot be solved from these fixes." if fixes else
                   "No valid GPS fixes were decoded."),
    }
    if fixes:
        report["first_fix"] = {k: fixes[0][k] for k in ("timestamp_utc", "latitude_deg", "longitude_deg", "altitude_m")}
    return rows, fixes, report


def export_gps(path, output):
    """Export raw records, deduplicated valid fixes, GPX and an honest quality report."""
    rows, fixes, report = extract_gps(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in (("gps_raw.csv", rows), ("gps.csv", fixes)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(records)
    namespace = "http://www.topografix.com/GPX/1/1"
    ET.register_namespace("", namespace)
    def tag(name):
        return f"{{{namespace}}}{name}"
    root = ET.Element(tag("gpx"), version="1.1", creator="LidarCamera360")
    if fixes:
        track = ET.SubElement(root, tag("trk"))
        ET.SubElement(track, tag("name")).text = Path(path).name
        segment = ET.SubElement(track, tag("trkseg"))
        for row in fixes:
            point = ET.SubElement(segment, tag("trkpt"), lat=str(row["latitude_deg"]), lon=str(row["longitude_deg"]))
            ET.SubElement(point, tag("ele")).text = str(row["altitude_m"])
            ET.SubElement(point, tag("time")).text = row["timestamp_utc"]
    ET.ElementTree(root).write(output / "gps.gpx", encoding="utf-8", xml_declaration=True)
    (output / "gps_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
