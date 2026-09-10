#!/usr/bin/env python3
"""Compatibility CLI for the unified SfM consensus colorizer."""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.pipeline_auto_calibrator_and_colorizer import colorize_via_spirula_sfm

def main(argv=None):
    parser = argparse.ArgumentParser(description="Colorize a LiDAR cloud from aligned COLMAP/Spirula views.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None, help="Output directory or PLY/PCD prefix.")
    args = parser.parse_args(argv)
    ply, pcd = map(Path, colorize_via_spirula_sfm(args.dataset.resolve(), fps=args.fps))
    if args.output is None: return 0
    out = args.output
    if out.suffix.lower() in {".ply", ".pcd"}:
        prefix = out.with_suffix("")
        targets = (out if out.suffix.lower() == ".ply" else prefix.with_suffix(".ply"), out if out.suffix.lower() == ".pcd" else prefix.with_suffix(".pcd"))
    else:
        out.mkdir(parents=True, exist_ok=True)
        targets = (out / ply.name, out / pcd.name)
    shutil.copy2(ply, targets[0]); shutil.copy2(pcd, targets[1])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
