"""Streaming point-cloud export for LAS/LAZ, PLY and PCD."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .cloud_io import CloudData


_FORMATS = {".las", ".laz", ".ply", ".pcd"}
_CHUNK = 262_144


def _validate(cloud: CloudData, output: Path, formats) -> str:
    suffix = output.suffix.lower()
    if suffix not in _FORMATS:
        raise ValueError(f"unsupported output format: {output.suffix or output.name}")
    try:
        if cloud.path and Path(cloud.path).resolve() == output.resolve():
            raise ValueError("output path must differ from cloud source path")
    except OSError:
        pass
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if formats is not None:
        if isinstance(formats, str):
            selected = [formats]
        else:
            selected = list(formats)
        selected = [str(x).lower() for x in selected]
        selected = [x if x.startswith(".") else "." + x for x in selected]
        if len(selected) != 1 or selected[0] != suffix:
            raise ValueError("formats must select exactly the output suffix")
    xyz = np.asarray(cloud.points)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("cloud points must have shape (N, 3)")
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("cloud contains no finite points")
    if len(np.asarray(cloud.colors)) != len(xyz):
        raise ValueError("point/color count mismatch")
    if cloud.intensities is not None and len(np.asarray(cloud.intensities)) != len(xyz):
        raise ValueError("point/intensity count mismatch")
    return suffix


def _rgb(cloud: CloudData, start: int, end: int) -> np.ndarray:
    c = np.asarray(cloud.colors[start:end], dtype=np.float64)
    if c.size and np.nanmax(np.abs(c)) <= 1.0:
        c *= 255.0
    return np.clip(np.rint(np.nan_to_num(c, nan=0.0)), 0, 255).astype(np.uint8)


def _write_pcd(cloud: CloudData, output: Path) -> None:
    n = len(cloud.points)
    fields = "x y z r g b" + (" intensity" if cloud.intensities is not None else "")
    sizes = "8 8 8 1 1 1" + (" 8" if cloud.intensities is not None else "")
    types = "F F F U U U" + (" F" if cloud.intensities is not None else "")
    counts = "1 1 1 1 1 1" + (" 1" if cloud.intensities is not None else "")
    header = (f"VERSION .7\nFIELDS {fields}\nSIZE {sizes}\nTYPE {types}\nCOUNT {counts}\n"
              f"WIDTH {n}\nHEIGHT 1\nPOINTS {n}\nDATA binary\n").encode("ascii")
    dt = np.dtype([(x, "<f8") for x in ("x", "y", "z")] +
                  [(x, "u1") for x in ("r", "g", "b")] +
                  ([('intensity', '<f8')] if cloud.intensities is not None else []))
    with output.open("xb") as f:
        f.write(header)
        for start in range(0, n, _CHUNK):
            end = min(n, start + _CHUNK)
            rec = np.empty(end - start, dtype=dt)
            rec["x"], rec["y"], rec["z"] = np.asarray(cloud.points[start:end], dtype=np.float64).T
            rec["r"], rec["g"], rec["b"] = _rgb(cloud, start, end).T
            if cloud.intensities is not None:
                rec["intensity"] = np.asarray(cloud.intensities[start:end], dtype=np.float64)
            f.write(rec.tobytes())


def _write_ply(cloud: CloudData, output: Path) -> None:
    n = len(cloud.points)
    intensity = cloud.intensities is not None
    props = "property double x\nproperty double y\nproperty double z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n"
    if intensity:
        props += "property double intensity\n"
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n{props}end_header\n".encode("ascii")
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")] +
                  ([('intensity', '<f8')] if intensity else []))
    with output.open("xb") as f:
        f.write(header)
        for start in range(0, n, _CHUNK):
            end = min(n, start + _CHUNK)
            rec = np.empty(end - start, dtype=dt)
            rec["x"], rec["y"], rec["z"] = np.asarray(cloud.points[start:end], dtype=np.float64).T
            rec["red"], rec["green"], rec["blue"] = _rgb(cloud, start, end).T
            if intensity:
                rec["intensity"] = np.asarray(cloud.intensities[start:end], dtype=np.float64)
            f.write(rec.tobytes())


def _write_las(cloud: CloudData, output: Path) -> None:
    import laspy
    from laspy import LasHeader, LasData

    xyz = np.asarray(cloud.points, dtype=np.float64)
    header = LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.floor(np.nanmin(xyz, axis=0) / 1000.0) * 1000.0
    if cloud.crs_wkt:
        try:
            from pyproj import CRS
            header.add_crs(CRS.from_wkt(cloud.crs_wkt))
        except Exception as exc:
            raise ValueError("invalid CRS WKT") from exc
    las = LasData(header)
    las.points = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
    las.x, las.y, las.z = xyz.T
    for name, values in zip(("red", "green", "blue"), _rgb(cloud, 0, len(xyz)).T):
        setattr(las, name, values.astype(np.uint16) * 257)
    if cloud.intensities is not None:
        las.intensity = np.clip(np.rint(np.asarray(cloud.intensities, dtype=np.float64)), 0, 65535).astype(np.uint16)
    las.write(output)


def save_cloud(cloud: CloudData, output: Path, formats=None) -> dict:
    """Write *cloud* to a new file and return paths/count/CRS metadata."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = _validate(cloud, output, formats)
    if suffix == ".pcd":
        _write_pcd(cloud, output)
    elif suffix == ".ply":
        _write_ply(cloud, output)
    else:
        _write_las(cloud, output)
    sidecar = None
    if suffix in (".pcd", ".ply"):
        sidecar_path = output.with_suffix(output.suffix + ".geo.json")
        metadata = {"crs_wkt": cloud.crs_wkt, "coordinate_frame": "projected" if cloud.crs_wkt else "local"}
        sidecar_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        sidecar = str(sidecar_path)
    return {"output": str(output), "path": str(output), "points_written": len(cloud.points),
            "crs_wkt": cloud.crs_wkt, "metadata_sidecar": sidecar}
