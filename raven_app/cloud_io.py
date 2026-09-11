"""Small, dependency-free (apart from NumPy) PCD/PLY point-cloud loader."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO

import numpy as np


@dataclass
class CloudData:
    path: Path
    points: np.ndarray
    colors: np.ndarray
    original_count: int

    @property
    def name(self) -> str:
        return self.path.name


def _finish(path: Path, xyz: np.ndarray, colors: np.ndarray, count: int) -> CloudData:
    xyz = np.asarray(xyz, dtype=np.float64).reshape((-1, 3))
    colors = np.asarray(colors, dtype=np.float32).reshape((-1, 3))
    if len(xyz) != len(colors):
        raise ValueError(f"{path}: point/color count mismatch")
    valid = np.isfinite(xyz).all(axis=1)
    xyz, colors = xyz[valid], np.clip(colors[valid], 0.0, 1.0)
    if not len(xyz):
        raise ValueError(f"{path}: no finite points")
    colors = np.nan_to_num(colors, nan=0.0, posinf=1.0, neginf=0.0)
    return CloudData(path, np.ascontiguousarray(xyz), np.ascontiguousarray(colors), int(count))


def _color_values(values: np.ndarray, integer: bool = False) -> np.ndarray:
    raw = np.asarray(values)
    a = raw.astype(np.float64)
    if integer or (a.size and np.nanmax(np.abs(a)) > 1.0):
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def _packed_color(values: np.ndarray, integer: bool = False) -> np.ndarray:
    a = np.asarray(values)
    if not integer:
        u = a.astype(np.float32, copy=False).view(np.uint32)
    else:
        u = a.astype(np.uint32, copy=False)
    return np.column_stack(((u >> 16) & 255, (u >> 8) & 255, u & 255)).astype(np.float32) / 255.0


_PCD_TYPES = {"F": "f", "I": "i", "U": "u"}
_PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _read_pcd(path: Path) -> CloudData:
    with path.open("rb") as f:
        headers = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PCD header missing DATA")
            text = line.decode("ascii", "strict").strip()
            headers.append(text)
            if text.upper().startswith("DATA "):
                break
        h: dict[str, list[str]] = {}
        for line in headers:
            p = line.split()
            if p:
                h[p[0].upper()] = p[1:]
        fields = h.get("FIELDS") or h.get("FIELD")
        if not fields:
            raise ValueError(f"{path}: PCD missing FIELDS")
        sizes = [int(x) for x in h.get("SIZE", [])]
        types = h.get("TYPE", [])
        counts = [int(x) for x in h.get("COUNT", ["1"] * len(fields))]
        if len(sizes) != len(fields) or len(types) != len(fields) or len(counts) != len(fields):
            raise ValueError(f"{path}: inconsistent PCD field metadata")
        n = int((h.get("POINTS") or [str(int((h.get("WIDTH") or ["0"])[0]) * int((h.get("HEIGHT") or ["1"])[0]))])[0])
        if n <= 0 or n > 1_000_000_000:
            raise ValueError(f"{path}: PCD contains no points")
        if any(s <= 0 for s in sizes) or any(c <= 0 for c in counts):
            raise ValueError(f"{path}: PCD field sizes/counts must be positive")
        mode = headers[-1].split(None, 1)[1].lower()
        if mode == "binary_compressed":
            raise ValueError(f"{path}: PCD binary_compressed is unsupported")
        if mode == "ascii":
            raw = f.read()
            try:
                data = np.fromstring(raw.decode("ascii"), sep=" ", dtype=np.float64)
            except UnicodeDecodeError as e:
                raise ValueError(f"{path}: invalid PCD ASCII payload") from e
            stride = sum(counts)
            if stride <= 0 or data.size < n * stride:
                raise ValueError(f"{path}: truncated PCD ASCII payload")
            data = data[:n * stride].reshape(n, stride)
            offsets = np.cumsum([0] + counts)
            get = lambda name: data[:, offsets[fields.index(name)]:offsets[fields.index(name)] + 1].ravel()
        elif mode == "binary":
            descr = []
            for name, size, typ, count in zip(fields, sizes, types, counts):
                code = _PCD_TYPES.get(typ.upper())
                if not code or size not in (1, 2, 4, 8):
                    raise ValueError(f"{path}: unsupported PCD field type {typ}{size}")
                descr.append((name, f"<{code}{size}", count))
            dt = np.dtype(descr)
            payload = f.read()
            if len(payload) < n * dt.itemsize:
                raise ValueError(f"{path}: truncated PCD binary payload")
            rec = np.frombuffer(payload, dtype=dt, count=n)
            get = lambda name: rec[name][:, 0] if rec[name].ndim > 1 else rec[name]
        else:
            raise ValueError(f"{path}: unsupported PCD DATA mode {mode!r}")
        try:
            xyz = np.column_stack([get(x) for x in ("x", "y", "z")])
        except (KeyError, ValueError):
            raise ValueError(f"{path}: PCD must contain x, y and z fields")
        field_types = {name: typ.upper() for name, typ in zip(fields, types)}
        if all(x in fields for x in ("r", "g", "b")):
            colors = np.column_stack([_color_values(get(x), field_types[x] in ("I", "U")) for x in ("r", "g", "b")])
        elif "rgb" in fields or "rgba" in fields:
            key = "rgb" if "rgb" in fields else "rgba"
            colors = _packed_color(get(key), field_types[key] in ("I", "U"))
        elif "intensity" in fields:
            v = _color_values(get("intensity"), field_types["intensity"] in ("I", "U")); colors = np.repeat(v[:, None], 3, axis=1)
        else:
            colors = np.full((n, 3), 0.7, dtype=np.float32)
    return _finish(path, xyz, colors, n)


def _read_ply(path: Path) -> CloudData:
    with path.open("rb") as f:
        first = f.readline().decode("ascii", "strict").strip()
        if first != "ply":
            raise ValueError(f"{path}: missing PLY magic")
        fmt = None; elements: list[tuple[str, int, list[tuple]]] = []; current = None
        while True:
            line = f.readline()
            if not line: raise ValueError(f"{path}: PLY header missing end_header")
            p = line.decode("ascii", "strict").strip().split()
            if not p: continue
            if p[0] == "format": fmt = p[1]
            elif p[0] == "element":
                current = [p[1], int(p[2]), []]; elements.append(current)
            elif p[0] == "property" and current is not None:
                if p[1] == "list":
                    if current[0] == "vertex": raise ValueError(f"{path}: vertex list properties are unsupported")
                    current[2].append(("list", p[2], p[3], p[4]))
                else:
                    if p[1] not in _PLY_TYPES: raise ValueError(f"{path}: unsupported PLY type {p[1]}")
                    current[2].append((p[2], p[1]))
            elif p[0] == "end_header": break
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian") or not elements:
            raise ValueError(f"{path}: unsupported or incomplete PLY header")
        vertex = next((e for e in elements if e[0] == "vertex"), None)
        if vertex is None: raise ValueError(f"{path}: PLY has no vertex element")
        if elements.index(vertex) != 0:
            raise ValueError(f"{path}: vertex element must precede other PLY elements")
        n, props = vertex[1], vertex[2]
        if n <= 0: raise ValueError(f"{path}: PLY contains no vertices")
        names = [p[0] for p in props]
        if not all(x in names for x in ("x", "y", "z")): raise ValueError(f"{path}: PLY must contain x, y and z properties")
        if fmt == "ascii":
            rows = []
            for _ in range(n):
                line = f.readline()
                if not line: raise ValueError(f"{path}: truncated PLY vertex data")
                vals = line.split()
                if len(vals) < len(props): raise ValueError(f"{path}: malformed PLY vertex row")
                rows.append(vals[:len(props)])
            arr = np.asarray(rows, dtype=np.float64)
            get = lambda name: arr[:, names.index(name)]
        else:
            endian = "<" if fmt.endswith("little_endian") else ">"
            dt = np.dtype([(p[0], endian + _PLY_TYPES[p[1]]) for p in props])
            raw = f.read(n * dt.itemsize)
            if len(raw) < n * dt.itemsize: raise ValueError(f"{path}: truncated PLY binary vertex data")
            rec = np.frombuffer(raw, dtype=dt, count=n); get = lambda name: rec[name]
        xyz = np.column_stack([get(x) for x in ("x", "y", "z")])
        ptypes = {p[0]: p[1] for p in props}
        if all(x in names for x in ("red", "green", "blue")):
            colors = np.column_stack([_color_values(get(x), ptypes[x] in ("char", "int8", "uchar", "uint8", "short", "int16", "ushort", "uint16", "int", "int32", "uint", "uint32")) for x in ("red", "green", "blue")])
        elif "intensity" in names:
            v = _color_values(get("intensity"), ptypes["intensity"] not in ("float", "float32", "double", "float64")); colors = np.repeat(v[:, None], 3, axis=1)
        else: colors = np.full((n, 3), 0.7, dtype=np.float32)
    return _finish(path, xyz, colors, n)


def load_cloud(path: str | Path) -> CloudData:
    p = Path(path)
    if not p.is_file(): raise ValueError(f"cloud file does not exist: {p}")
    if p.suffix.lower() == ".pcd": return _read_pcd(p)
    if p.suffix.lower() == ".ply": return _read_ply(p)
    raise ValueError(f"unsupported point-cloud format: {p.suffix or p.name}")
