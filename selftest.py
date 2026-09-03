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

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-test")
PAIRS = [
    (os.path.join(B, "LIDAR_20260828094048Fabrica2.bag"),
     os.path.join(B, "VID_20260828_094637_00_270.mp4")),
]

if __name__ == "__main__":
    t0 = time.time()
    s = Solver(PAIRS, Prior(up=0.18, back=0.07, right=0.0, tol=0.15),
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
