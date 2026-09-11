# Embedded INSV GPS

Extract directly from native trailers, without ExifTool or video decoding:

```powershell
python lidarcamera360.py extract-gps --insv "D:\capture\video.insv" --output "D:\capture\gps"
```

Pass multiple paths after `--insv` for batch processing. Each file gets its own
subfolder with `gps_raw.csv` (including invalid records), `gps.csv` (unique valid
fixes), `gps.gpx`, and `gps_report.json`. Rerunning replaces those GPS exports;
capture files remain untouched. Exit status is 2 if any file has no valid fixes.
Static GPS exports successfully but reports `location_anchor_only`.

The native 53-byte layout was checked against
[ExifTool's INSV implementation](https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/QuickTimeStream.pl).
Variable-sized directories and sequential trailers are supported. Unsupported
layouts and invalid bounds are rejected. No video-to-GPS time offset is guessed.

## Georeferencing

This command extracts GPS; it does not transform point clouds or COLMAP models.
Repeated identical fixes provide one location, not a trajectory. For alignment,
match video presentation timestamps (`images/frames.json`) to GPS UTC before
pairing camera centers with fixes. Reject extrapolation and long GPS gaps.
Validate motion, geometry and fit residuals: the candidate flag only checks for
at least three distinct positions and timestamps.

LiDAR is metric: preserve its scale. Independent SfM may need alignment to LiDAR
first. Apply a shared geographic transform only when both already share a frame.
Transform COLMAP poses and sparse points together; EXIF geotags alone do not
georeference a model. Camera centers are `-R.T @ t`, per the
[COLMAP format](https://colmap.github.io/format.html).

Altitude datum is unspecified in this payload; do not label it EGM96/EGM2008.
Keep a local model plus its geographic transform for applications that lose
precision at large projected coordinates.

## Escritório validation

`VID_20260902_143757_00_277.insv` contains 921 records: 920 valid copies of one
fix and one invalid record. The unique fix is:

- UTC: `2026-09-02T17:37:57.673Z`
- Latitude/longitude: `-3.69549376, -38.59972568`
- Recorded altitude: `16.04937744140625 m` (datum unspecified)

This cannot determine geographic heading. A metric, gravity-aligned scan needs
an independent heading and an identified local position corresponding to the
GPS antenna (including sensor offset), or suitable surveyed control points.
The test folder contains a ROS bag and INSV only, with no existing point cloud
or COLMAP reconstruction. The legacy `FAST-LIVO2/georeference.py` guesses timing
and heading and should not be used automatically for this static-fix capture.
