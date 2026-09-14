"""High-resolution georeferenced orthophoto generator from LiDAR point clouds.

Produces metric top-down orthomosaics (orthophotos) and Digital Surface Models (DSM)
with configurable ground sampling distance (GSD), top-surface occlusion filtering,
and exports standard GeoTIFF (.tif), PNG with World File (.tfw/.pgw), and metadata.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    tifffile = None

from raven_app.cloud_io import load_cloud, CloudData


def generate_orthophoto(
    cloud_input: Union[str, Path, CloudData, np.ndarray],
    output_path: Union[str, Path],
    *,
    gsd_m: float = 0.05,
    colors: Optional[np.ndarray] = None,
    mode: str = "rgb",
    crs_wkt: Optional[str] = None,
    crs_epsg: Optional[int] = None,
    bounds_utm: Optional[Tuple[float, float, float, float]] = None,
    transform_matrix: Optional[np.ndarray] = None,
    fill_holes: bool = True,
    max_hole_radius_px: int = 3,
) -> Dict[str, Any]:
    """Generate a high-resolution georeferenced orthophoto from point cloud data.

    Args:
        cloud_input: File path (PCD, PLY, LAS, LAZ) or CloudData or (N, 3) point array.
        output_path: Target output path (will produce .tif, .png, .tfw, .json).
        gsd_m: Ground Sampling Distance in meters per pixel (default: 0.05m = 5cm/px).
        colors: Optional (N, 3) RGB color array in range [0, 1] or [0, 255].
        mode: Render mode: 'rgb', 'dsm' (elevation), or 'intensity'.
        crs_wkt: Coordinate reference system WKT string (optional).
        crs_epsg: Horizontal EPSG code (e.g. 31984).
        bounds_utm: Optional (min_e, max_e, min_n, max_n) bounding box to crop.
        transform_matrix: Optional 4x4 matrix to transform local points into georeferenced coordinates.
        fill_holes: Whether to interpolate small empty cells.
        max_hole_radius_px: Maximum radius in pixels to interpolate empty cells.

    Returns:
        Dict with paths to generated orthophoto deliverables, metadata, and dimensions.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    base_stem = out_file.stem

    # 1. Load point coordinates and colors
    if isinstance(cloud_input, (str, Path)):
        p = Path(cloud_input)
        if not p.is_file():
            raise FileNotFoundError(f"Point cloud not found: {p}")
        cloud_data = load_cloud(str(p))
        pts = cloud_data.points
        pt_colors = cloud_data.colors if cloud_data.has_rgb else None
        intensities = cloud_data.intensities
    elif isinstance(cloud_input, CloudData):
        pts = cloud_input.points
        pt_colors = cloud_input.colors if cloud_input.has_rgb else None
        intensities = cloud_input.intensities
    elif isinstance(cloud_input, np.ndarray):
        pts = cloud_input
        pt_colors = colors
        intensities = None
    else:
        raise TypeError(f"Unsupported cloud input type: {type(cloud_input)}")

    if len(pts) == 0:
        raise ValueError("Point cloud contains no points.")

    # 2. Apply optional 4x4 coordinate transformation (e.g. local to UTM)
    if transform_matrix is not None:
        T = np.asarray(transform_matrix, dtype=np.float64)
        if T.shape == (4, 4):
            pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
            pts = (T @ pts_h.T).T[:, :3]

    # 3. Determine bounding box and raster grid dimensions
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    if bounds_utm is not None:
        min_x, max_x, min_y, max_y = bounds_utm
        mask = (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
        if np.any(mask):
            pts = pts[mask]
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            if pt_colors is not None and len(pt_colors) == len(mask):
                pt_colors = pt_colors[mask]
            if intensities is not None and len(intensities) == len(mask):
                intensities = intensities[mask]
    else:
        min_x, max_x = float(np.min(x)), float(np.max(x))
        min_y, max_y = float(np.min(y)), float(np.max(y))

    width_m = max_x - min_x
    height_m = max_y - min_y

    if gsd_m <= 0:
        gsd_m = 0.05

    cols = int(math.ceil(width_m / gsd_m)) + 1
    rows = int(math.ceil(height_m / gsd_m)) + 1

    # Clamp raster dimensions to prevent memory overflow (max 16384x16384)
    max_dim = 16384
    if cols > max_dim or rows > max_dim:
        scale_down = max(cols / max_dim, rows / max_dim)
        gsd_m = gsd_m * scale_down
        cols = int(math.ceil(width_m / gsd_m)) + 1
        rows = int(math.ceil(height_m / gsd_m)) + 1

    # Coordinate mapping:
    # Raster row 0 is at max_y (North), row (rows-1) is at min_y (South)
    # Raster col 0 is at min_x (West), col (cols-1) is at max_x (East)
    col_idx = np.clip(np.floor((x - min_x) / gsd_m).astype(np.int32), 0, cols - 1)
    row_idx = np.clip(np.floor((max_y - y) / gsd_m).astype(np.int32), 0, rows - 1)
    pixel_idx = row_idx * cols + col_idx

    # 4. Top-surface occlusion filter:
    # Sort points by Z ascending so the highest point (Z_max) overwrites any point underneath
    sort_order = np.argsort(z)
    sorted_pixel_idx = pixel_idx[sort_order]
    sorted_z = z[sort_order]

    # Initialize raster buffers
    dsm_grid = np.full((rows, cols), np.nan, dtype=np.float32)
    rgb_grid = np.zeros((rows, cols, 4), dtype=np.uint8)  # RGBA with alpha channel

    # If RGB colors available, prepare uint8 array
    if pt_colors is not None and len(pt_colors) == len(pts):
        c_arr = np.asarray(pt_colors, dtype=np.float32)
        if np.nanmax(c_arr) <= 1.0:
            c_uint8 = np.clip(c_arr * 255.0, 0, 255).astype(np.uint8)
        else:
            c_uint8 = np.clip(c_arr, 0, 255).astype(np.uint8)
    else:
        # Generate synthetic height ramp if no RGB
        z_norm = (z - np.min(z)) / max(np.ptp(z), 1e-4)
        c_uint8 = np.column_stack([
            np.clip(255 * z_norm, 0, 255).astype(np.uint8),
            np.clip(255 * (1.0 - np.abs(z_norm - 0.5) * 2), 0, 255).astype(np.uint8),
            np.clip(255 * (1.0 - z_norm), 0, 255).astype(np.uint8),
        ])

    sorted_colors = c_uint8[sort_order]

    # Vectorized assignment: later points in sorted order (highest Z) take precedence
    dsm_flat = dsm_grid.reshape(-1)
    dsm_flat[sorted_pixel_idx] = sorted_z.astype(np.float32)

    rgb_flat = rgb_grid.reshape(-1, 4)
    rgb_flat[sorted_pixel_idx, :3] = sorted_colors
    rgb_flat[sorted_pixel_idx, 3] = 255  # Fully opaque occupied pixels

    # 5. Hole filling (morphological/nearest neighbor interpolation for small gaps)
    if fill_holes and max_hole_radius_px > 0:
        alpha_mask = rgb_grid[:, :, 3] == 0
        if np.any(alpha_mask):
            from scipy.ndimage import distance_transform_edt
            dist, indices = distance_transform_edt(alpha_mask, return_indices=True)
            close_holes = (dist > 0) & (dist <= max_hole_radius_px)
            if np.any(close_holes):
                near_r = indices[0][close_holes]
                near_c = indices[1][close_holes]
                rgb_grid[close_holes, :3] = rgb_grid[near_r, near_c, :3]
                rgb_grid[close_holes, 3] = 255
                dsm_grid[close_holes] = dsm_grid[near_r, near_c]

    # 6. Save deliverables
    # A. High-resolution PNG (RGBA)
    png_path = out_file.parent / f"{base_stem}.png"
    pil_img = Image.fromarray(rgb_grid, mode="RGBA")
    pil_img.save(str(png_path), optimize=True)

    # B. World file (.pgw and .tfw)
    # Standard 6-line ESRI World File:
    # Line 1: Pixel size in X direction (GSD)
    # Line 2: Rotation term (0.0 for unrotated raster)
    # Line 3: Rotation term (0.0 for unrotated raster)
    # Line 4: Negative pixel size in Y direction (-GSD)
    # Line 5: Center of upper-left pixel X (min_x + gsd_m/2)
    # Line 6: Center of upper-left pixel Y (max_y - gsd_m/2)
    x_center_ul = min_x + (gsd_m / 2.0)
    y_center_ul = max_y - (gsd_m / 2.0)
    world_content = (
        f"{gsd_m:.8f}\n"
        f"0.00000000\n"
        f"0.00000000\n"
        f"{-gsd_m:.8f}\n"
        f"{x_center_ul:.8f}\n"
        f"{y_center_ul:.8f}\n"
    )

    tfw_path = out_file.parent / f"{base_stem}.tfw"
    pgw_path = out_file.parent / f"{base_stem}.pgw"
    tfw_path.write_text(world_content, encoding="utf-8")
    pgw_path.write_text(world_content, encoding="utf-8")

    # C. GeoTIFF (.tif)
    tif_path = out_file.parent / f"{base_stem}.tif"
    saved_tif = False

    # Try writing GeoTIFF via tifffile with ModelTiepoint and PixelScale
    if tifffile is not None:
        try:
            # Tiepoint: (0, 0, 0, min_x, max_y, 0)
            # PixelScale: (gsd_m, gsd_m, 0)
            extratags = [
                (33550, 'd', 3, (gsd_m, gsd_m, 0.0), False),  # ModelPixelScaleTag
                (33922, 'd', 6, (0.0, 0.0, 0.0, min_x, max_y, 0.0), False),  # ModelTiepointTag
            ]
            if crs_epsg:
                # Standard GeoKeyDirectoryTag for Projected CRS
                # Key 1024 (GTModelTypeGeoKey) = 1 (ModelTypeProjected)
                # Key 1025 (GTRasterTypeGeoKey) = 1 (RasterPixelIsArea)
                # Key 3072 (ProjectedCSTypeGeoKey) = crs_epsg
                geo_keys = (
                    1, 1, 0, 3,  # Header: KeyDirectoryVersion, KeyRevision, MinorRevision, NumberOfKeys
                    1024, 0, 1, 1,  # GTModelTypeGeoKey: 1 = Projected
                    1025, 0, 1, 1,  # GTRasterTypeGeoKey: 1 = PixelIsArea
                    3072, 0, 1, crs_epsg,  # ProjectedCSTypeGeoKey
                )
                extratags.append((34735, 's', len(geo_keys), geo_keys, False))

            tifffile.imwrite(
                str(tif_path),
                rgb_grid,
                photometric='rgb',
                extratags=extratags,
                compression='zlib',
            )
            saved_tif = True
        except Exception:
            saved_tif = False

    if not saved_tif:
        # Fallback to Pillow TIFF saving with companion .tfw
        pil_img.save(str(tif_path), format="TIFF", compression="tiff_deflate")

    # D. Digital Surface Model (DSM) 32-bit Float TIFF
    dsm_path = out_file.parent / f"{base_stem}_dsm.tif"
    if tifffile is not None:
        try:
            tifffile.imwrite(
                str(dsm_path),
                dsm_grid,
                extratags=[
                    (33550, 'd', 3, (gsd_m, gsd_m, 0.0), False),
                    (33922, 'd', 6, (0.0, 0.0, 0.0, min_x, max_y, 0.0), False),
                ],
                compression='zlib',
            )
        except Exception:
            pass

    # E. Metadata JSON
    metadata = {
        "status": "success",
        "orthophoto_png": str(png_path),
        "orthophoto_tif": str(tif_path),
        "world_file_tfw": str(tfw_path),
        "world_file_pgw": str(pgw_path),
        "dsm_tif": str(dsm_path) if dsm_path.is_file() else None,
        "width_px": cols,
        "height_px": rows,
        "gsd_m": float(gsd_m),
        "ground_coverage_m": {
            "width_m": float(width_m),
            "height_m": float(height_m),
            "area_m2": float(width_m * height_m),
        },
        "bounding_box_utm": {
            "min_easting_m": float(min_x),
            "max_easting_m": float(max_x),
            "min_northing_m": float(min_y),
            "max_northing_m": float(max_y),
        },
        "elevation_m": {
            "min_elevation_m": float(np.nanmin(dsm_grid)) if np.any(np.isfinite(dsm_grid)) else 0.0,
            "max_elevation_m": float(np.nanmax(dsm_grid)) if np.any(np.isfinite(dsm_grid)) else 0.0,
        },
        "crs_epsg": crs_epsg,
        "total_source_points": len(pts),
    }

    meta_path = out_file.parent / f"{base_stem}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
