# -*- coding: utf-8 -*-
"""Headless run on a set of captures. Edit PAIRS to point at your data."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from core import Prior
from solver import Solver

B = r"C:\scans\калибровка самосборного"
PAIRS = [
    (os.path.join(B, r"scan_00004_2026-08-20_1122_sharestudio\scan_00004_2026-08-20_1122.bag"),
     os.path.join(B, "VID_20260820_114514_00_012.mp4")),
    (os.path.join(B, r"scan_00005_2026-08-20_1123_sharestudio\scan_00005_2026-08-20_1123.bag"),
     os.path.join(B, "VID_20260820_114609_00_013.mp4")),
    (os.path.join(B, r"scan_00006_2026-08-20_1125_sharestudio\scan_00006_2026-08-20_1125.bag"),
     os.path.join(B, "VID_20260820_114805_00_014.mp4")),
]

if __name__ == "__main__":
    t0 = time.time()
    s = Solver(PAIRS, Prior(up=0.15, back=0.08, right=0.0, tol=0.10),
               quality=sys.argv[1] if len(sys.argv) > 1 else "Normal",
               log=lambda m: print(m, flush=True))
    ok, bad = s.run()
    print("\n=== summary (%.0f s) ===" % (time.time() - t0))
    for r in s.summary_rows():
        print("  %-30s %-9s rig=%-18s score=%s" % (r[0], r[1], r[4], r[5]))
        print("      %s" % r[6])
    print("\nrig:", s.stats["rig"])
    print("uncertainty:", s.stats["uncertainty"])
    if "lsq_residual_deg" in s.stats:
        print("lsq residual: %.2f deg / %.0f mm"
              % (s.stats["lsq_residual_deg"], 1000 * s.stats["lsq_residual_m"]))
    json.dump(s.json(), open("selftest_output.json", "w"), indent=1)
