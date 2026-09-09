# Historical implementation validation

This document records development validation. It is a historical engineering report, not a promise of performance or accuracy for every sensor, capture, GPU, or dataset. Re-run checks for the exact release and hardware you intend to publish.

The validated Windows workflow covered native FAST-LIVO2 ingestion, malformed FLV2 handling, isolated working-directory package startup, headless execution, GUI startup, and Ctrl+C cancellation. The reference Raven run processed 20,597 records and 981 scan updates, producing 3,858,286 points in 10.12 seconds on the test machine. Camera coverage ended before LiDAR coverage in that run, leaving 25 pending scans; current metadata reports incomplete camera coverage explicitly.

```powershell
python -m unittest discover -s tests
python tests/check_native.py
python tests/smoke_packaged.py
python tests/check_cancel.py
```

These checks validate tested code paths. They do not establish universal throughput, reconstruction accuracy, or color agreement. Dataset-specific comparisons under `docs/images/` are retained as historical visual material.
