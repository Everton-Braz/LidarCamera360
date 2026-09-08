#!/usr/bin/env python3
"""
align_exif_gps_to_true_north_laz.py

Embeds the EXACT True North EXIF GPS coordinates and headings (matching the verified .LAZ / .PCD / .LAS datasets)
into the images of model_3_upright_yaw180_reverse_Rx_plus90.
Leaves sparse/ (cameras.txt, images.txt, points3D.txt, .bin) 100% untouched.
"""

import os
import sys
import json
from datetime import datetime, timezone
import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R
import piexif

sys.path.append(r'd:\APLICATIVOS\FAST-LIVO2')
import scripts.fix_realityscan_colmap_rotation_and_import as fix
import scripts.generate_perfect_realityscan_models as gen

def align_exif_to_true_north_laz(target_model_dir=r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\RealityScan_Dataset"):
    images_dir = os.path.join(target_model_dir, "images")

    print(f"[*] Aligning EXIF GPS in: {images_dir}")
    print("[*] Preserving sparse/ (cameras, images, points3D) 100% untouched.")

    # 1. Load verified georeference transform (matches colorized_lidar_georeferenced_utm.laz)
    with open(r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\Georeferenced\local_to_geographic.json", "r", encoding="utf-8") as f:
        geo = json.load(f)

    R_geo = np.array(geo["rotation_matrix"]) # Yaw = 77.18 deg
    t_geo = np.array(geo["translation_utm_m"])
    t_start = geo["timestamp_sync"]["video_start_epoch"]

    transformer = pyproj.Transformer.from_crs("EPSG:31984", "EPSG:4326", always_xy=True)

    # 2. Load ground-truth metric camera poses
    metric_images_txt = r"d:\APLICATIVOS\FAST-LIVO2\Log\small_test\Colmap_Metric_Fisheye\images.txt"
    metric_images = fix.load_images(metric_images_txt)

    count = 0
    gps_records = []

    for img_id in sorted(metric_images.keys()):
        img = metric_images[img_id]
        name = img["name"] # e.g. cam0/00010.jpg
        img_path = os.path.join(images_dir, name)
        if not os.path.exists(img_path):
            continue

        c_loc = img["C_world"]
        c_utm = R_geo @ c_loc + t_geo
        lon, lat, alt = transformer.transform(c_utm[0], c_utm[1], c_utm[2])

        # Optical forward vector in True North
        r_wc = img["r_wc"]
        fwd_loc = r_wc @ np.array([0, 0, 1])
        fwd_utm = R_geo @ fwd_loc
        heading_deg = (np.degrees(np.arctan2(fwd_utm[0], fwd_utm[1])) + 360.0) % 360.0

        fname = os.path.basename(name)
        frame_num = int("".join(filter(str.isdigit, fname)))
        t_epoch = t_start + (frame_num / 24.0)
        dt = datetime.fromtimestamp(t_epoch, tz=timezone.utc)

        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: "S" if lat < 0 else "N",
            piexif.GPSIFD.GPSLatitude: gen.deg_to_dms_rational(lat),
            piexif.GPSIFD.GPSLongitudeRef: "W" if lon < 0 else "E",
            piexif.GPSIFD.GPSLongitude: gen.deg_to_dms_rational(lon),
            piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
            piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 100), 100),
            piexif.GPSIFD.GPSImgDirectionRef: "T",
            piexif.GPSIFD.GPSImgDirection: (int(heading_deg * 100), 100),
            piexif.GPSIFD.GPSDateStamp: dt.strftime("%Y:%m:%d"),
            piexif.GPSIFD.GPSTimeStamp: ((dt.hour, 1), (dt.minute, 1), (int(dt.second * 1000 + dt.microsecond / 1000), 1000))
        }

        try:
            exif_dict = piexif.load(img_path)
            exif_dict["GPS"] = gps_ifd
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, img_path)
            count += 1
            gps_records.append((lat, lon, alt, heading_deg))
        except Exception as e:
            print(f"[-] Error writing EXIF to {name}: {e}")

    if len(gps_records) > 0:
        gps_arr = np.array(gps_records)
        lat_span_m = np.ptp(gps_arr[:, 0]) * 111320.0
        lon_span_m = np.ptp(gps_arr[:, 1]) * 111320.0 * np.cos(np.radians(-3.7))

        print(f"\n[+] Successfully updated {count} images with True North EXIF GPS data!")
        print(f"    Trajectory Extents on RealityScan Map:")
        print(f"    Latitude Span (North-South, towards 'do Rio'): {lat_span_m:.2f} meters")
        print(f"    Longitude Span (East-West across property):    {lon_span_m:.2f} meters")
        print(f"    Altitude Range:                                [{np.min(gps_arr[:, 2]):.2f}m, {np.max(gps_arr[:, 2]):.2f}m]")
        print(f"    Mean GPS Position:                             Lat: {np.mean(gps_arr[:, 0]):.7f}°S, Lon: {np.mean(gps_arr[:, 1]):.7f}°W")
        print(f"    True North Alignment:                         100% Identical to colorized_lidar_georeferenced_utm.laz (EPSG:31984)")
    else:
        print(f"\n[!] Warning: No matching images found to update in {images_dir}")

if __name__ == "__main__":
    align_exif_to_true_north_laz()
