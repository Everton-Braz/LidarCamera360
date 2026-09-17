"""Automatic, conservative GPS anchoring of workflow-produced point clouds."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from pyproj import CRS

from raven_app.cloud_io import load_cloud, CloudData
from raven_app.georeference import calculate_insv_gps, fit_georeference, export_geojson
from raven_app.cloud_export import save_cloud

FORMATS = ('laz', 'las', 'ply', 'pcd', 'geojson')


def _horizontal_wkt(crs_info):
    return CRS.from_user_input(crs_info['horizontal_crs']).to_wkt()


def build_automatic_transform(gps_info, trajectory_xyz=None):
    """Fit yaw/origin while retaining local elevation and metric scale."""
    result = fit_georeference(gps_info, trajectory_xyz=trajectory_xyz)
    matrix = np.asarray(result['matrix_4x4'], dtype=np.float64)
    matrix[2, 3] = 0.0
    gps_span = float(np.linalg.norm(np.asarray(gps_info['utm_coords'])[-1, :2] - np.asarray(gps_info['utm_coords'])[0, :2])) if len(gps_info.get('utm_coords', [])) > 1 else 0.0
    estimated_track = trajectory_xyz is not None and len(trajectory_xyz) >= 2 and gps_span >= 1.0 and result.get('yaw_estimation_method') == 'trajectory_pca_alignment'
    result.update({
        'matrix_4x4': matrix.tolist(), 'translation_utm_m': matrix[:3, 3].tolist(),
        'scale': 1.0, 'quality': 'estimated' if estimated_track else 'location_anchor_only',
        'validation': 'not_survey_validated', 'vertical_transform': 'none_local_Z_retained',
        'horizontal_only': True,
    })
    result.pop('target_vertical_crs', None)
    result.pop('target_compound_crs', None)
    result['altitude_datum'] = 'unspecified; local Z retained'
    return result


def _trajectory_xyz(path):
    if not path or not Path(path).is_file():
        return None
    try:
        arr = np.loadtxt(path, ndmin=2)
        return arr[:, 1:4] if arr.shape[1] >= 4 else None
    except (OSError, ValueError):
        return None


def automatic_georeference(insv_path, deliverables_dir, candidate_clouds=None,
                           formats=('laz', 'geojson'), trajectory_path=None,
                           trajectory_xyz=None, gps_info=None):
    """Transform the explicitly produced clouds and write a reviewable report."""
    out = Path(deliverables_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selected = tuple(dict.fromkeys(str(x).lower().lstrip('.') for x in formats))
    if not selected or any(x not in FORMATS for x in selected):
        raise ValueError(f'geo formats must be selected from {FORMATS}')
    gps = gps_info or calculate_insv_gps(insv_path)
    if not gps.get('records'):
        report = {'status': 'skipped', 'reason': 'no_valid_gps', 'quality': 'location_anchor_only', 'outputs': []}
        (out / 'automatic_georeference_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return report
    trajectory_xyz = trajectory_xyz if trajectory_xyz is not None else _trajectory_xyz(trajectory_path)
    transform = build_automatic_transform(gps, trajectory_xyz)
    R = np.asarray(transform['matrix_4x4'], dtype=np.float64)[:3, :3]
    t = np.asarray(transform['matrix_4x4'], dtype=np.float64)[:3, 3]
    sources = [Path(p) for p in (candidate_clouds or [])]
    sources = [p for p in sources if p.is_file() and p.suffix.lower() in {'.las', '.laz', '.ply', '.pcd'}]
    geo_dir = out / 'georeferenced'
    geo_dir.mkdir(exist_ok=True)
    exported = []
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    bounds_count = 0
    for source in sources:
        cloud = load_cloud(source)
        points = np.empty_like(cloud.points, dtype=np.float64)
        for start in range(0, len(points), 500000):
            points[start:start+500000] = cloud.points[start:start+500000] @ R.T + t
        transformed = CloudData(source, points, cloud.colors,
                                 cloud.original_count, cloud.intensities, cloud.has_rgb,
                                 _horizontal_wkt(gps['crs_info']))
        bounds_min = np.minimum(bounds_min, transformed.points.min(axis=0))
        bounds_max = np.maximum(bounds_max, transformed.points.max(axis=0))
        bounds_count += len(transformed.points)
        for fmt in selected:
            if fmt == 'geojson':
                continue
            target = geo_dir / f'{source.stem}_georeferenced.{fmt}'
            save_cloud(transformed, target, overwrite=True)
            sidecar = target.with_suffix(target.suffix + '.georef.json')
            sidecar.write_text(json.dumps({'source': str(source.resolve()), 'transform': transform,
                                            'quality': transform['quality'], 'validation': transform['validation']}, indent=2) + '\n', encoding='utf-8')
            exported.append(str(target))
    bounds = None
    if bounds_count:
        bounds = {'min_easting_m': float(bounds_min[0]), 'max_easting_m': float(bounds_max[0]),
                  'min_northing_m': float(bounds_min[1]), 'max_northing_m': float(bounds_max[1]),
                  'min_altitude_m': float(bounds_min[2]), 'max_altitude_m': float(bounds_max[2]), 'num_points': bounds_count}
    # GeoJSON footprint only needs the horizontal CRS; do not propagate a
    # guessed/unspecified vertical datum into automatic deliverable metadata.
    horizontal_crs_info = {'horizontal_crs': gps['crs_info']['horizontal_crs']}
    geojson = export_geojson(geo_dir, gps_records=gps['records'], cloud_bounds_utm=bounds, crs_info=horizontal_crs_info) if 'geojson' in selected else {}
    gps_summary = {k: gps[k] for k in ('source_insv', 'total_fixes', 'duration_seconds', 'total_distance_m',
                                       'mean_wgs84', 'bounding_box_wgs84') if k in gps}
    report = {'status': 'complete', 'quality': transform['quality'], 'validation': transform['validation'],
              'source_insv': str(Path(insv_path).resolve()), 'transform': transform,
              'outputs': exported, 'geojson': geojson, 'gps_summary': gps_summary}
    (out / 'automatic_georeference_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return report


# Stable descriptive aliases for callers integrating the workflow engine.
georeference_clouds = automatic_georeference


def transform_cloud_data(cloud: CloudData, transform):
    """Return a float64, horizontally transformed CloudData object."""
    matrix = np.asarray(transform['matrix_4x4'], dtype=np.float64)
    points = cloud.points @ matrix[:3, :3].T + matrix[:3, 3]
    return CloudData(cloud.path, points, cloud.colors, cloud.original_count,
                     cloud.intensities, cloud.has_rgb, cloud.crs_wkt)
