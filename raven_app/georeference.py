"""Complete GPS Georeferencing, Point Cloud & COLMAP Alignment, and GeoJSON Engine.

Computes gravity-preserving horizontal (Yaw) rigid transformation between local LiDAR/COLMAP
metric coordinates and UTM/SIRGAS 2000 coordinates, and exports LAZ/LAS/PCD deliverables,
geotagged image metadata, and standard RFC 7946 GeoJSON layers.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R

from raven_app.cloud_io import load_cloud
from raven_app.insv_gps import extract_gps

try:
    import piexif
except ImportError:
    piexif = None


def auto_detect_utm_crs(mean_lat: float, mean_lon: float, prefer_sirgas: bool = True) -> Dict[str, Any]:
    """Determine the optimal projected UTM CRS (preferring SIRGAS 2000 in Brazil/South America)."""
    zone = int((mean_lon + 180) / 6) + 1
    is_south = mean_lat < 0
    # In South America (e.g. Brazil), SIRGAS 2000 is standard
    if prefer_sirgas and is_south and -90 <= mean_lon <= -30 and -60 <= mean_lat <= 15:
        epsg_h = 31960 + zone  # e.g. 31984 for UTM Zone 24S
        epsg_v = 3855          # EGM2008 height
        name_h = f"SIRGAS 2000 / UTM zone {zone}S"
    else:
        epsg_h = 32700 + zone if is_south else 32600 + zone
        epsg_v = 5773          # EGM96 height
        name_h = f"WGS 84 / UTM zone {zone}{'S' if is_south else 'N'}"

    compound_str = f"EPSG:{epsg_h}+{epsg_v}"
    return {
        "zone": zone,
        "hemisphere": "S" if is_south else "N",
        "epsg_h": epsg_h,
        "epsg_v": epsg_v,
        "horizontal_crs": f"EPSG:{epsg_h}",
        "vertical_crs": f"EPSG:{epsg_v}",
        "compound_crs": compound_str,
        "crs_name": name_h,
    }


def calculate_insv_gps(insv_path: str | Path, target_crs: Optional[str] = None) -> Dict[str, Any]:
    """Extract embedded GNSS telemetry from INSV and compute projected metric UTM coordinates."""
    path = Path(insv_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"INSV file not found: {path}")

    _, fixes, report = extract_gps(path)
    if not fixes:
        raise ValueError(f"No valid GPS fixes could be extracted from: {path.name}")

    gps_records = []
    for f in fixes:
        gps_records.append({
            "timestamp_epoch": float(f["timestamp_epoch"]),
            "datetime_utc": f["timestamp_utc"],
            "latitude": float(f["latitude_deg"]),
            "longitude": float(f["longitude_deg"]),
            "altitude_m": float(f["altitude_m"]),
            "speed_mps": float(f.get("speed_mps", 0.0)),
            "track_deg": float(f.get("track_deg", 0.0)),
            "source": path.name,
        })

    gps_records.sort(key=lambda r: r["timestamp_epoch"])
    mean_lat = float(np.mean([r["latitude"] for r in gps_records]))
    mean_lon = float(np.mean([r["longitude"] for r in gps_records]))
    mean_alt = float(np.mean([r["altitude_m"] for r in gps_records]))

    if target_crs:
        crs_obj = pyproj.CRS.from_user_input(target_crs)
        epsg_code = crs_obj.to_epsg() or 0
        crs_info = {
            "zone": int((mean_lon + 180) / 6) + 1,
            "hemisphere": "S" if mean_lat < 0 else "N",
            "epsg_h": epsg_code,
            "epsg_v": 3855 if mean_lat < 0 else 5773,
            "horizontal_crs": target_crs if target_crs.upper().startswith("EPSG:") else f"EPSG:{epsg_code}",
            "vertical_crs": "EPSG:3855" if mean_lat < 0 else "EPSG:5773",
            "compound_crs": f"EPSG:{epsg_code}+{3855 if mean_lat < 0 else 5773}",
            "crs_name": crs_obj.name,
        }
    else:
        crs_info = auto_detect_utm_crs(mean_lat, mean_lon)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", crs_info["horizontal_crs"], always_xy=True)
    utm_coords = np.zeros((len(gps_records), 3), dtype=np.float64)
    for i, r_pt in enumerate(gps_records):
        e, n, u = transformer.transform(r_pt["longitude"], r_pt["latitude"], r_pt["altitude_m"])
        utm_coords[i] = [e, n, u]
        r_pt["easting_m"] = float(e)
        r_pt["northing_m"] = float(n)

    # Compute trajectory statistics
    lats = [r["latitude"] for r in gps_records]
    lons = [r["longitude"] for r in gps_records]
    diffs = np.diff(utm_coords[:, :2], axis=0) if len(utm_coords) > 1 else np.zeros((1, 2))
    dist_m = float(np.sum(np.linalg.norm(diffs, axis=1)))
    duration_s = float(gps_records[-1]["timestamp_epoch"] - gps_records[0]["timestamp_epoch"]) if len(gps_records) > 1 else 0.0

    return {
        "source_insv": str(path),
        "total_fixes": len(gps_records),
        "duration_seconds": duration_s,
        "total_distance_m": dist_m,
        "mean_wgs84": {
            "latitude_deg": mean_lat,
            "longitude_deg": mean_lon,
            "altitude_m": mean_alt,
        },
        "bounding_box_wgs84": {
            "min_latitude": float(min(lats)),
            "max_latitude": float(max(lats)),
            "min_longitude": float(min(lons)),
            "max_longitude": float(max(lons)),
        },
        "bounding_box_utm": {
            "min_easting_m": float(np.min(utm_coords[:, 0])),
            "max_easting_m": float(np.max(utm_coords[:, 0])),
            "min_northing_m": float(np.min(utm_coords[:, 1])),
            "max_northing_m": float(np.max(utm_coords[:, 1])),
        },
        "crs_info": crs_info,
        "records": gps_records,
        "utm_coords": utm_coords,
    }


def parse_colmap_images_txt(images_txt_path: str | Path) -> List[Dict[str, Any]]:
    """Parse camera center positions from COLMAP images.txt."""
    path = Path(images_txt_path)
    if not path.is_file():
        return []

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frames = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        name = parts[9]
        if not (name.startswith("cam") or name.lower().endswith((".jpg", ".jpeg", ".png"))):
            continue
        try:
            qw, qx, qy, qz = [float(x) for x in parts[1:5]]
            tx, ty, tz = [float(x) for x in parts[5:8]]
            cam_id = int(parts[8])
        except ValueError:
            continue

        R_w2c = R.from_quat([qx, qy, qz, qw]).as_matrix()
        C_loc = -R_w2c.T @ np.array([tx, ty, tz])
        fname = Path(name).name
        digits = "".join(filter(str.isdigit, fname))
        frame_num = int(digits) if digits else len(frames)
        frames.append({
            "name": name,
            "frame_num": frame_num,
            "xyz": C_loc,
            "R_w2c": R_w2c,
            "cam_id": cam_id,
        })

    frames.sort(key=lambda f: f["frame_num"])
    return frames


def fit_georeference(
    gps_info: Dict[str, Any],
    colmap_frames: Optional[List[Dict[str, Any]]] = None,
    trajectory_xyz: Optional[np.ndarray] = None,
    manual_yaw_deg: Optional[float] = None,
    delta_easting_m: float = 0.0,
    delta_northing_m: float = 0.0,
    delta_elevation_m: float = 0.0,
) -> Dict[str, Any]:
    """Compute the optimal gravity-preserving rigid transformation (Yaw + UTM shift)."""
    gps_utm = gps_info["utm_coords"]
    crs_info = gps_info["crs_info"]
    mean_lat = gps_info["mean_wgs84"]["latitude_deg"]
    mean_lon = gps_info["mean_wgs84"]["longitude_deg"]
    mean_alt = gps_info["mean_wgs84"]["altitude_m"]

    # 1. Determine local XYZ trajectory from COLMAP or SLAM trajectory
    local_xyz = None
    if colmap_frames:
        local_xyz = np.array([cf["xyz"] for cf in colmap_frames], dtype=np.float64)
    elif trajectory_xyz is not None and len(trajectory_xyz) > 0:
        local_xyz = np.asarray(trajectory_xyz, dtype=np.float64)

    # 2. Determine True North Yaw
    if manual_yaw_deg is not None:
        yaw_deg = float(manual_yaw_deg)
        yaw_method = "manual_user_specified"
    elif local_xyz is not None and len(local_xyz) >= 2 and len(gps_utm) >= 2:
        # Compute principal orientation of local trajectory vs GPS UTM trajectory via PCA
        cov_loc = np.cov(local_xyz[:, :2].T)
        eigvals_l, eigvecs_l = np.linalg.eigh(cov_loc)
        v_loc = eigvecs_l[:, 1]
        ang_loc = float(np.degrees(np.arctan2(v_loc[1], v_loc[0])))

        cov_gps = np.cov(gps_utm[:, :2].T)
        eigvals_g, eigvecs_g = np.linalg.eigh(cov_gps)
        v_gps = eigvecs_g[:, 1]
        ang_gps = float(np.degrees(np.arctan2(v_gps[1], v_gps[0])))

        candidate_1 = (ang_gps - ang_loc + 360.0) % 360.0
        candidate_2 = (ang_gps - ang_loc + 180.0 + 360.0) % 360.0

        # Disambiguate forward direction using start-to-end vector correlation
        d_loc = local_xyz[-1, :2] - local_xyz[0, :2]
        d_gps = gps_utm[-1, :2] - gps_utm[0, :2]
        if np.linalg.norm(d_loc) > 1.0 and np.linalg.norm(d_gps) > 1.0:
            ang_d_loc = float(np.degrees(np.arctan2(d_loc[1], d_loc[0])))
            ang_d_gps = float(np.degrees(np.arctan2(d_gps[1], d_gps[0])))
            motion_yaw = (ang_d_gps - ang_d_loc + 360.0) % 360.0
            # Pick candidate closest to motion_yaw
            diff1 = abs((candidate_1 - motion_yaw + 180.0) % 360.0 - 180.0)
            diff2 = abs((candidate_2 - motion_yaw + 180.0) % 360.0 - 180.0)
            yaw_deg = candidate_1 if diff1 <= diff2 else candidate_2
        else:
            yaw_deg = candidate_1
        yaw_method = "trajectory_pca_alignment"
    else:
        # Fallback to GPS track heading or initial displacement
        diffs = gps_utm[-1, :2] - gps_utm[0, :2] if len(gps_utm) > 1 else np.array([0.0, 1.0])
        if np.linalg.norm(diffs) > 0.5:
            yaw_deg = float((np.degrees(np.arctan2(diffs[1], diffs[0])) + 360.0) % 360.0)
        else:
            valid_tracks = [r["track_deg"] for r in gps_info["records"] if r.get("track_deg", 0) > 0]
            yaw_deg = float(np.median(valid_tracks)) if valid_tracks else 0.0
        yaw_method = "gps_motion_vector_heading"

    # 3. Construct 2D rotation matrix
    rad = np.radians(yaw_deg)
    R_2d = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)],
    ], dtype=np.float64)

    # 4. Compute translation shift
    if local_xyz is not None and len(local_xyz) > 0:
        loc_xy_rot = (R_2d @ local_xyz[:, :2].T).T
        t_2d = np.mean(gps_utm[:, :2], axis=0) - np.mean(loc_xy_rot, axis=0)
        t_z = float(mean_alt - np.mean(local_xyz[:, 2]))
    else:
        t_2d = np.mean(gps_utm[:, :2], axis=0)
        t_z = float(mean_alt)

    # Apply user-specified delta translation nudges
    t_3d = np.array([
        t_2d[0] + delta_easting_m,
        t_2d[1] + delta_northing_m,
        t_z + delta_elevation_m,
    ], dtype=np.float64)

    R_3d = np.eye(3, dtype=np.float64)
    R_3d[:2, :2] = R_2d

    T_4x4 = np.eye(4, dtype=np.float64)
    T_4x4[:3, :3] = R_3d
    T_4x4[:3, 3] = t_3d

    return {
        "status": "success",
        "source_frame": "local_metric",
        "target_horizontal_crs": crs_info["horizontal_crs"],
        "target_vertical_crs": crs_info["vertical_crs"],
        "target_compound_crs": crs_info["compound_crs"],
        "crs_name": crs_info["crs_name"],
        "reference_origin_wgs84": {
            "latitude_deg": mean_lat,
            "longitude_deg": mean_lon,
            "altitude_m": mean_alt,
        },
        "yaw_heading_deg": float(yaw_deg),
        "yaw_estimation_method": yaw_method,
        "rotation_matrix": R_3d.tolist(),
        "translation_utm_m": t_3d.tolist(),
        "matrix_4x4": T_4x4.tolist(),
        "scale": 1.0,
        "gravity_alignment": "100% Upright (Pitch=0.00°, Roll=0.00°)",
    }


def apply_transform_adjustment(
    current_transform: Dict[str, Any],
    *,
    delta_yaw_deg: float = 0.0,
    absolute_yaw_deg: Optional[float] = None,
    delta_easting_m: float = 0.0,
    delta_northing_m: float = 0.0,
    delta_elevation_m: float = 0.0,
    reference_centroid_local: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Apply interactive rotation and translation adjustments to an existing georeferencing transform.

    Rotates cleanly around the scene centroid so geometry does not orbit across the map.
    """
    updated = dict(current_transform)
    old_yaw = float(current_transform.get("yaw_heading_deg", 0.0))
    new_yaw = float(absolute_yaw_deg) if absolute_yaw_deg is not None else float((old_yaw + delta_yaw_deg) % 360.0)

    rad_new = np.radians(new_yaw)
    R_new_2d = np.array([
        [np.cos(rad_new), -np.sin(rad_new)],
        [np.sin(rad_new),  np.cos(rad_new)],
    ], dtype=np.float64)

    R_new_3d = np.eye(3, dtype=np.float64)
    R_new_3d[:2, :2] = R_new_2d

    old_t = np.array(current_transform["translation_utm_m"], dtype=np.float64)
    old_R = np.array(current_transform["rotation_matrix"], dtype=np.float64)

    if reference_centroid_local is not None:
        c_loc = np.asarray(reference_centroid_local, dtype=np.float64)[:2]
        c_utm_2d = (old_R[:2, :2] @ c_loc) + old_t[:2]
        new_t_2d = c_utm_2d - (R_new_2d @ c_loc) + np.array([delta_easting_m, delta_northing_m])
    else:
        new_t_2d = old_t[:2] + np.array([delta_easting_m, delta_northing_m])

    new_t_z = old_t[2] + delta_elevation_m
    new_t = np.array([new_t_2d[0], new_t_2d[1], new_t_z], dtype=np.float64)

    T_4x4 = np.eye(4, dtype=np.float64)
    T_4x4[:3, :3] = R_new_3d
    T_4x4[:3, 3] = new_t

    updated["yaw_heading_deg"] = new_yaw
    updated["yaw_estimation_method"] = "interactive_user_adjustment"
    updated["rotation_matrix"] = R_new_3d.tolist()
    updated["translation_utm_m"] = new_t.tolist()
    updated["matrix_4x4"] = T_4x4.tolist()
    return updated


def export_interactive_html_map(
    output_html_path: str | Path,
    center_lat: float,
    center_lon: float,
    footprint_latlons: Optional[List[Tuple[float, float]]] = None,
    cameras: Optional[List[Dict[str, Any]]] = None,
    gps_records: Optional[List[Dict[str, Any]]] = None,
    heading_deg: float = 0.0,
    title: str = "LidarCamera360 - Georeferenced Scene Map",
) -> str:
    """Export a standalone interactive Leaflet HTML map using free satellite and street basemaps."""
    out_path = Path(output_html_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    footprint_json = json.dumps([[p[1], p[0]] for p in footprint_latlons] if footprint_latlons else [])
    cam_json = json.dumps([
        {"lat": c["latitude"], "lon": c["longitude"], "name": c.get("name", ""), "heading": c.get("heading_deg", 0.0)}
        for c in (cameras or [])[::2]  # sample every 2nd camera for smooth web performance
    ])
    gps_json = json.dumps([
        {"lat": r["latitude"], "lon": r["longitude"], "speed": r.get("speed_mps", 0.0)}
        for r in (gps_records or [])
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body, html {{ margin: 0; padding: 0; height: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    #map {{ width: 100%; height: 100%; }}
    .floating-card {{
      position: absolute; top: 16px; right: 16px; z-index: 1000;
      background: rgba(16, 24, 38, 0.92); color: #fff; padding: 14px 18px;
      border-radius: 10px; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.12);
      max-width: 320px; font-size: 13px;
    }}
    .floating-card h4 {{ margin: 0 0 8px 0; color: #00f0ff; font-size: 15px; }}
    .metric {{ display: flex; justify-content: space-between; margin-bottom: 4px; color: #cbd5e1; }}
    .metric b {{ color: #f8fafc; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; background: #0284c7; color: #fff; font-size: 11px; margin-top: 6px; }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="floating-card">
    <h4>Georeferenced Scene</h4>
    <div class="metric"><span>Center Lat:</span> <b>{center_lat:.6f}°</b></div>
    <div class="metric"><span>Center Lon:</span> <b>{center_lon:.6f}°</b></div>
    <div class="metric"><span>True North Yaw:</span> <b>{heading_deg:.2f}°</b></div>
    <div class="metric"><span>Cameras Loaded:</span> <b>{len(cameras or [])}</b></div>
    <div class="badge">ArcGIS Satellite Aerial Active</div>
  </div>
  <script>
    const map = L.map('map', {{ zoomControl: true }}).setView([{center_lat}, {center_lon}], 18);

    // Free tile layer providers
    const arcgisAerial = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 19, attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics'
    }}).addTo(map);

    const osm = L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19, attribution: '&copy; OpenStreetMap contributors'
    }});

    const topo = L.tileLayer('https://tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 17, attribution: '&copy; OpenTopoMap'
    }});

    L.control.layers({{
      "ArcGIS Satellite Aerial": arcgisAerial,
      "OpenStreetMap (Standard)": osm,
      "OpenTopoMap (Terrain)": topo
    }}).addTo(map);

    // Point cloud extent footprint polygon
    const footprint = {footprint_json};
    if (footprint.length > 0) {{
      const poly = L.polygon(footprint, {{
        color: '#00f0ff', fillColor: '#00f0ff', fillOpacity: 0.35, weight: 3
      }}).addTo(map).bindPopup("<b>Point Cloud Footprint</b><br>True North Yaw: {heading_deg:.2f}°");
      map.fitBounds(poly.getBounds(), {{ padding: [40, 40] }});
    }}

    // GPS Trajectory Track
    const gpsPoints = {gps_json};
    if (gpsPoints.length > 0) {{
      const latlngs = gpsPoints.map(p => [p.lat, p.lon]);
      L.polyline(latlngs, {{ color: '#f59e0b', weight: 4, opacity: 0.9 }}).addTo(map).bindPopup("<b>INSV GPS Track</b>");
    }}

    // COLMAP Camera Poses
    const cameras = {cam_json};
    cameras.forEach(c => {{
      L.circleMarker([c.lat, c.lon], {{
        radius: 4, color: '#ef4444', fillColor: '#f87171', fillOpacity: 0.8, weight: 1
      }}).addTo(map).bindTooltip(c.name);
    }});
  </script>
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    return str(out_path)


def export_geojson(
    output_dir: str | Path,
    gps_records: Optional[List[Dict[str, Any]]] = None,
    colmap_cameras: Optional[List[Dict[str, Any]]] = None,
    cloud_bounds_utm: Optional[Dict[str, float]] = None,
    crs_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Export standard RFC 7946 GeoJSON files for GPS track, camera trajectory, and cloud extent."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    all_scene_features = []

    # 1. GPS Trajectory GeoJSON
    if gps_records:
        coords = [[r["longitude"], r["latitude"], round(r.get("altitude_m", 0.0), 2)] for r in gps_records]
        track_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords,
                },
                "properties": {
                    "name": "GPS Trajectory Track",
                    "source": gps_records[0].get("source", "Insta360 INSV"),
                    "num_fixes": len(gps_records),
                    "start_time": gps_records[0]["datetime_utc"],
                    "end_time": gps_records[-1]["datetime_utc"],
                },
            }
        ]
        for i, r in enumerate(gps_records):
            track_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r["longitude"], r["latitude"], round(r.get("altitude_m", 0.0), 2)],
                },
                "properties": {
                    "fix_index": i,
                    "timestamp_utc": r["datetime_utc"],
                    "altitude_m": r.get("altitude_m", 0.0),
                    "speed_mps": r.get("speed_mps", 0.0),
                    "track_deg": r.get("track_deg", 0.0),
                },
            })

        gps_geojson_path = out_dir / "gps_trajectory.geojson"
        fc_gps = {
            "type": "FeatureCollection",
            "name": "GPS Trajectory",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": track_features,
        }
        gps_geojson_path.write_text(json.dumps(fc_gps, indent=2), encoding="utf-8")
        generated_files["gps_trajectory_geojson"] = str(gps_geojson_path)
        all_scene_features.extend(track_features)

    # 2. COLMAP Camera Poses GeoJSON
    if colmap_cameras:
        cam_coords = [[c["longitude"], c["latitude"], round(c["altitude_m"], 2)] for c in colmap_cameras]
        cam_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": cam_coords,
                },
                "properties": {
                    "name": "COLMAP Camera Path",
                    "total_cameras": len(colmap_cameras),
                },
            }
        ]
        for c in colmap_cameras:
            cam_features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [c["longitude"], c["latitude"], round(c["altitude_m"], 2)],
                },
                "properties": {
                    "image": c["name"],
                    "frame_num": c.get("frame_num", 0),
                    "heading_deg": round(c.get("heading_deg", 0.0), 2),
                    "altitude_m": round(c["altitude_m"], 2),
                    "timestamp_utc": c.get("timestamp_utc", ""),
                },
            })

        cam_geojson_path = out_dir / "colmap_cameras.geojson"
        fc_cam = {
            "type": "FeatureCollection",
            "name": "COLMAP Cameras",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": cam_features,
        }
        cam_geojson_path.write_text(json.dumps(fc_cam, indent=2), encoding="utf-8")
        generated_files["colmap_cameras_geojson"] = str(cam_geojson_path)
        all_scene_features.extend(cam_features)

    # 3. Point Cloud Extent Bounding Box GeoJSON
    if cloud_bounds_utm and crs_info:
        h_crs = crs_info["horizontal_crs"]
        transformer_to_wgs84 = pyproj.Transformer.from_crs(h_crs, "EPSG:4326", always_xy=True)
        min_e, max_e = cloud_bounds_utm["min_easting_m"], cloud_bounds_utm["max_easting_m"]
        min_n, max_n = cloud_bounds_utm["min_northing_m"], cloud_bounds_utm["max_northing_m"]

        corners_utm = [
            (min_e, min_n),
            (max_e, min_n),
            (max_e, max_n),
            (min_e, max_n),
            (min_e, min_n),
        ]
        corners_wgs84 = [list(transformer_to_wgs84.transform(e, n)) for e, n in corners_utm]

        footprint_features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [corners_wgs84],
                },
                "properties": {
                    "name": "Georeferenced Point Cloud Extent",
                    "num_points": cloud_bounds_utm.get("num_points", 0),
                    "horizontal_crs": h_crs,
                    "min_elevation_m": cloud_bounds_utm.get("min_altitude_m", 0.0),
                    "max_elevation_m": cloud_bounds_utm.get("max_altitude_m", 0.0),
                    "easting_span_m": round(max_e - min_e, 2),
                    "northing_span_m": round(max_n - min_n, 2),
                },
            }
        ]
        cloud_geojson_path = out_dir / "pointcloud_extent.geojson"
        fc_cloud = {
            "type": "FeatureCollection",
            "name": "Point Cloud Extent",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
            "features": footprint_features,
        }
        cloud_geojson_path.write_text(json.dumps(fc_cloud, indent=2), encoding="utf-8")
        generated_files["pointcloud_extent_geojson"] = str(cloud_geojson_path)
        all_scene_features.extend(footprint_features)

    # 4. Master Unified Scene GeoJSON
    master_geojson_path = out_dir / "georeferenced_scene.geojson"
    fc_master = {
        "type": "FeatureCollection",
        "name": "Georeferenced Scene (GPS, Cameras, LiDAR Cloud)",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": all_scene_features,
    }
    master_geojson_path.write_text(json.dumps(fc_master, indent=2), encoding="utf-8")
    generated_files["georeferenced_scene_geojson"] = str(master_geojson_path)

    return generated_files


def georeference_pointcloud(
    cloud_path: str | Path,
    transform_dict: Dict[str, Any],
    output_dir: str | Path,
) -> Dict[str, Any]:
    """Transform point cloud coordinates to UTM and export LAZ, LAS, PCD, and PRJ deliverables."""
    import laspy

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_file = Path(cloud_path)

    cloud = load_cloud(cloud_file)
    R_mat = np.array(transform_dict["rotation_matrix"], dtype=np.float64)
    t_vec = np.array(transform_dict["translation_utm_m"], dtype=np.float64)

    # Apply 3D transformation
    xyz_utm = (cloud.points @ R_mat.T) + t_vec
    num_pts = len(xyz_utm)

    bounds = {
        "min_easting_m": float(np.min(xyz_utm[:, 0])),
        "max_easting_m": float(np.max(xyz_utm[:, 0])),
        "min_northing_m": float(np.min(xyz_utm[:, 1])),
        "max_northing_m": float(np.max(xyz_utm[:, 1])),
        "min_altitude_m": float(np.min(xyz_utm[:, 2])),
        "max_altitude_m": float(np.max(xyz_utm[:, 2])),
        "num_points": num_pts,
    }

    compound_crs_str = transform_dict.get("target_compound_crs", "EPSG:31984+3855")
    crs_compound = pyproj.CRS.from_string(compound_crs_str)

    # 1. Export Compressed LAZ
    out_laz = out_dir / "colorized_lidar_georeferenced_utm.laz"
    out_las = out_dir / "colorized_lidar_georeferenced_utm.las"

    header = laspy.LasHeader(point_format=7 if cloud.has_rgb else 6, version="1.4")
    header.offsets = [
        float(np.floor(bounds["min_easting_m"])),
        float(np.floor(bounds["min_northing_m"])),
        float(np.floor(bounds["min_altitude_m"])),
    ]
    header.scales = [0.001, 0.001, 0.001]
    header.add_crs(crs_compound, keep_compatibility=False)

    las_data = laspy.LasData(header)
    las_data.x = xyz_utm[:, 0]
    las_data.y = xyz_utm[:, 1]
    las_data.z = xyz_utm[:, 2]
    if cloud.has_rgb:
        las_data.red = (cloud.colors[:, 0] * 65535).astype(np.uint16)
        las_data.green = (cloud.colors[:, 1] * 65535).astype(np.uint16)
        las_data.blue = (cloud.colors[:, 2] * 65535).astype(np.uint16)
    if cloud.intensities is not None:
        las_data.intensity = np.clip(cloud.intensities, 0, 65535).astype(np.uint16)

    las_data.write(str(out_laz))
    las_data.write(str(out_las))

    # 2. Export ESRI PRJ Definition
    out_prj = out_dir / "colorized_lidar_georeferenced_utm.prj"
    out_prj.write_text(crs_compound.to_wkt(), encoding="utf-8")

    # 3. Export Binary PCD
    out_pcd = out_dir / "colorized_lidar_georeferenced_utm.pcd"
    rgb_uint8 = (cloud.colors * 255.0).astype(np.uint8) if cloud.has_rgb else np.full((num_pts, 3), 200, dtype=np.uint8)
    rgb_packed = (rgb_uint8[:, 0].astype(np.uint32) << 16) | (rgb_uint8[:, 1].astype(np.uint32) << 8) | rgb_uint8[:, 2].astype(np.uint32)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<u4")])
    pcd_arr = np.zeros(num_pts, dtype=dt)
    pcd_arr["x"] = xyz_utm[:, 0].astype(np.float32)
    pcd_arr["y"] = xyz_utm[:, 1].astype(np.float32)
    pcd_arr["z"] = xyz_utm[:, 2].astype(np.float32)
    pcd_arr["rgb"] = rgb_packed

    header_str = (
        f"# .PCD v0.7 - Point Cloud Data\n"
        f"VERSION 0.7\n"
        f"FIELDS x y z rgb\n"
        f"SIZE 4 4 4 4\n"
        f"TYPE F F F U\n"
        f"COUNT 1 1 1 1\n"
        f"WIDTH {num_pts}\n"
        f"HEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {num_pts}\n"
        f"DATA binary\n"
    )
    with out_pcd.open("wb") as f_pcd:
        f_pcd.write(header_str.encode("utf-8"))
        f_pcd.write(pcd_arr.tobytes())

    return {
        "laz_path": str(out_laz),
        "las_path": str(out_las),
        "prj_path": str(out_prj),
        "pcd_path": str(out_pcd),
        "points_count": num_pts,
        "bounds_utm": bounds,
    }


def georeference_colmap(
    colmap_dir: str | Path,
    transform_dict: Dict[str, Any],
    output_dir: str | Path,
    geotag_images: bool = True,
) -> Dict[str, Any]:
    """Transform COLMAP camera poses to UTM, reproject to WGS84, and export manifest and geotags."""
    base_dir = Path(colmap_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images_txt = base_dir / "sparse" / "images.txt"
    if not images_txt.is_file():
        images_txt = base_dir / "sparse" / "0" / "images.txt"
    if not images_txt.is_file():
        images_txt = base_dir / "images.txt"

    frames = parse_colmap_images_txt(images_txt)
    if not frames:
        return {"cameras_count": 0, "cameras": []}

    h_crs = transform_dict["target_horizontal_crs"]
    transformer_to_wgs84 = pyproj.Transformer.from_crs(h_crs, "EPSG:4326", always_xy=True)
    R_mat = np.array(transform_dict["rotation_matrix"], dtype=np.float64)
    t_vec = np.array(transform_dict["translation_utm_m"], dtype=np.float64)

    geotagged_cameras = []
    colmap_images_dir = base_dir / "images"
    exif_updated_count = 0

    for f in frames:
        C_loc = f["xyz"]
        C_utm = R_mat @ C_loc + t_vec
        lon, lat, alt = transformer_to_wgs84.transform(C_utm[0], C_utm[1], C_utm[2])

        forward_loc = f["R_w2c"].T @ np.array([0, 0, 1])
        forward_utm = R_mat @ forward_loc
        heading_deg = float((np.degrees(np.arctan2(forward_utm[0], forward_utm[1])) + 360.0) % 360.0)

        cam_rec = {
            "name": f["name"],
            "frame_num": f["frame_num"],
            "latitude": float(lat),
            "longitude": float(lon),
            "altitude_m": float(alt),
            "heading_deg": heading_deg,
            "easting_m": float(C_utm[0]),
            "northing_m": float(C_utm[1]),
        }
        geotagged_cameras.append(cam_rec)

        # Inject EXIF tags if piexif is available and images exist
        if geotag_images and piexif is not None:
            img_path = colmap_images_dir / f["name"]
            if img_path.is_file():
                try:
                    deg_lat, deg_lon = abs(lat), abs(lon)
                    d_lat, m_lat = int(deg_lat), int((deg_lat - int(deg_lat)) * 60)
                    s_lat = int(round((deg_lat - d_lat - m_lat / 60.0) * 3600 * 1000))
                    d_lon, m_lon = int(deg_lon), int((deg_lon - int(deg_lon)) * 60)
                    s_lon = int(round((deg_lon - d_lon - m_lon / 60.0) * 3600 * 1000))

                    gps_ifd = {
                        piexif.GPSIFD.GPSLatitudeRef: "S" if lat < 0 else "N",
                        piexif.GPSIFD.GPSLatitude: ((d_lat, 1), (m_lat, 1), (s_lat, 1000)),
                        piexif.GPSIFD.GPSLongitudeRef: "W" if lon < 0 else "E",
                        piexif.GPSIFD.GPSLongitude: ((d_lon, 1), (m_lon, 1), (s_lon, 1000)),
                        piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
                        piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 100), 100),
                        piexif.GPSIFD.GPSImgDirectionRef: "T",
                        piexif.GPSIFD.GPSImgDirection: (int(heading_deg * 100), 100),
                    }
                    exif_dict = piexif.load(str(img_path))
                    exif_dict["GPS"] = gps_ifd
                    piexif.insert(piexif.dump(exif_dict), str(img_path))
                    exif_updated_count += 1
                except Exception:
                    pass

    # Export images_geotagged.csv
    out_csv = out_dir / "images_geotagged.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["image", "latitude", "longitude", "altitude_m", "heading_deg", "easting_m", "northing_m"])
        for c in geotagged_cameras:
            writer.writerow([
                c["name"], f"{c['latitude']:.8f}", f"{c['longitude']:.8f}",
                f"{c['altitude_m']:.3f}", f"{c['heading_deg']:.2f}",
                f"{c['easting_m']:.3f}", f"{c['northing_m']:.3f}",
            ])

    return {
        "cameras_count": len(geotagged_cameras),
        "cameras": geotagged_cameras,
        "images_geotagged_csv": str(out_csv),
        "exif_updated_count": exif_updated_count,
    }


def run_georeference(
    insv: str | Path,
    output: str | Path,
    *,
    cloud: Optional[str | Path] = None,
    colmap_dir: Optional[str | Path] = None,
    trajectory: Optional[str | Path] = None,
    crs: Optional[str] = None,
    yaw_deg: Optional[float] = None,
    delta_easting_m: float = 0.0,
    delta_northing_m: float = 0.0,
    delta_elevation_m: float = 0.0,
    generate_orthophoto_file: bool = True,
    ortho_gsd_m: float = 0.05,
    export_geojson_files: bool = True,
    export_cloud_files: bool = True,
    export_colmap_files: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute complete end-to-end GPS georeferencing pipeline."""
    out_dir = Path(output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calculate GPS telemetry from INSV
    gps_info = calculate_insv_gps(insv, target_crs=crs)

    # 2. Parse COLMAP frames if available
    colmap_frames = None
    if colmap_dir and Path(colmap_dir).is_dir():
        imgs_txt = Path(colmap_dir) / "sparse" / "images.txt"
        if not imgs_txt.is_file():
            imgs_txt = Path(colmap_dir) / "sparse" / "0" / "images.txt"
        if not imgs_txt.is_file():
            imgs_txt = Path(colmap_dir) / "images.txt"
        if imgs_txt.is_file():
            colmap_frames = parse_colmap_images_txt(imgs_txt)

    # 3. Parse SLAM trajectory if available
    traj_xyz = None
    if trajectory and Path(trajectory).is_file():
        try:
            arr = np.loadtxt(str(trajectory), ndmin=2)
            if arr.shape[1] >= 4:
                traj_xyz = arr[:, 1:4]
        except Exception:
            pass

    # 4. Fit Georeferencing Transformation
    transform_dict = fit_georeference(
        gps_info,
        colmap_frames=colmap_frames,
        trajectory_xyz=traj_xyz,
        manual_yaw_deg=yaw_deg,
        delta_easting_m=delta_easting_m,
        delta_northing_m=delta_northing_m,
        delta_elevation_m=delta_elevation_m,
    )

    # 5. Georeference Point Cloud if provided
    cloud_result = None
    cloud_bounds = None
    if cloud and Path(cloud).is_file() and export_cloud_files:
        cloud_result = georeference_pointcloud(cloud, transform_dict, out_dir)
        cloud_bounds = cloud_result["bounds_utm"]

    # 6. Georeference COLMAP Dataset if provided
    colmap_result = None
    colmap_cameras = None
    if colmap_dir and Path(colmap_dir).is_dir() and export_colmap_files:
        colmap_result = georeference_colmap(colmap_dir, transform_dict, out_dir)
        colmap_cameras = colmap_result["cameras"]

    # 7. Export GeoJSON files
    geojson_files = {}
    if export_geojson_files:
        geojson_files = export_geojson(
            out_dir,
            gps_records=gps_info["records"],
            colmap_cameras=colmap_cameras,
            cloud_bounds_utm=cloud_bounds,
            crs_info=gps_info["crs_info"],
        )

    # 8. Generate Orthophoto if requested and point cloud is provided
    ortho_result = None
    if generate_orthophoto_file and cloud and Path(cloud).is_file():
        try:
            from raven_app.orthophoto import generate_orthophoto
            ortho_target = out_dir / "georeferenced_orthophoto.png"
            epsg_code = int(gps_info["crs_info"]["epsg_h"]) if "epsg_h" in gps_info["crs_info"] else None
            ortho_result = generate_orthophoto(
                cloud,
                ortho_target,
                gsd_m=ortho_gsd_m,
                transform_matrix=np.array(transform_dict["matrix_4x4"]),
                crs_epsg=epsg_code,
            )
        except Exception as e:
            ortho_result = {"status": "error", "message": str(e)}

    # 9. Export Standalone Interactive HTML Leaflet Map (ArcGIS Aerial + OSM)
    html_map_path = None
    try:
        transformer = pyproj.Transformer.from_crs(gps_info["crs_info"]["horizontal_crs"], "EPSG:4326", always_xy=True)
        t_utm = transform_dict["translation_utm_m"]
        c_lon, c_lat, _ = transformer.transform(t_utm[0], t_utm[1], t_utm[2])
        
        footprint_pts = []
        if cloud_bounds:
            # 4 corners in UTM
            corners_utm = [
                (cloud_bounds["min_easting_m"], cloud_bounds["min_northing_m"]),
                (cloud_bounds["max_easting_m"], cloud_bounds["min_northing_m"]),
                (cloud_bounds["max_easting_m"], cloud_bounds["max_northing_m"]),
                (cloud_bounds["min_easting_m"], cloud_bounds["max_northing_m"]),
            ]
            for ce, cn in corners_utm:
                flon, flat, _ = transformer.transform(ce, cn, 0)
                footprint_pts.append((flon, flat))

        html_map_path = export_interactive_html_map(
            out_dir / "georeference_map.html",
            center_lat=c_lat,
            center_lon=c_lon,
            footprint_latlons=footprint_pts,
            cameras=colmap_cameras,
            gps_records=gps_info["records"],
            heading_deg=transform_dict["yaw_heading_deg"],
        )
    except Exception:
        pass

    # 10. Save local_to_geographic.json & georeference_report.json
    transform_json_path = out_dir / "local_to_geographic.json"
    transform_json_path.write_text(json.dumps(transform_dict, indent=2), encoding="utf-8")

    report = {
        "status": "accepted",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_insv": str(Path(insv).resolve()),
        "gps_summary": {
            "total_fixes": gps_info["total_fixes"],
            "duration_seconds": gps_info["duration_seconds"],
            "total_distance_m": gps_info["total_distance_m"],
            "reference_origin_wgs84": gps_info["mean_wgs84"],
            "crs_info": gps_info["crs_info"],
        },
        "transform": transform_dict,
        "deliverables": {
            "transform_json": str(transform_json_path),
            "geojson_files": geojson_files,
            "cloud_export": cloud_result,
            "colmap_export": colmap_result,
            "orthophoto_export": ortho_result,
            "interactive_html_map": html_map_path,
        },
    }

    report_path = out_dir / "georeference_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

