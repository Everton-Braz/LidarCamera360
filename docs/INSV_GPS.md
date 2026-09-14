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

The unified desktop workflow can run the same metadata extraction during a
workflow with `--process-gps`. Select one or more `--gps-formats` from
`geojson`, `gpx`, and `csv`; the UI enables all three by default. The main
workflow also has an **Automatic GPS georeferencing** checkbox and separate
output-format checkboxes for LAS, LAZ, PLY, and PCD. When enabled, the cloud is
georeferenced only after synchronization and spatial checks pass. Missing or
insufficient GPS leaves the cloud in its local frame and records the reason.

The native 53-byte layout was checked against
[ExifTool's INSV implementation](https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/QuickTimeStream.pl).
Variable-sized directories and sequential trailers are supported. Unsupported
layouts and invalid bounds are rejected. No video-to-GPS time offset is guessed.

## Georeferencing

`georeference` assesses a timed TUM trajectory against INSV GPS and writes
`georeference_report.json`. With `--cloud`, it exports a `.laz` only after the
assessment is accepted:

The command below runs the same assessment headlessly. The desktop workflow
exposes this through its main automatic GPS option; there is no separate
georeferencing tab in the 3D viewer.

```powershell
python lidarcamera360.py georeference --insv capture.insv `
  --trajectory slam.txt --output assessment `
  --time-offset 0.0 --confirm-time-sync --gravity-aligned `
  --cloud all_raw_points.pcd
```

The transform is horizontal only: WGS84 UTM by default (or an explicitly
supplied projected CRS), in metres, with scale `1.0`. Z remains the local SLAM
height; recorded GPS altitude is retained as evidence but is never applied and
no vertical datum is inferred. The fit solves yaw and XY translation only.
`--time-offset` means `GPS UTC = trajectory timestamp + offset`; it must come
from independently verified clock synchronization. `--confirm-time-sync` and
`--gravity-aligned` are explicit declarations, not estimates. Supply
`--lever-arm X Y Z` for the body-frame trajectory-origin to GPS-antenna offset.
No fit result guarantees GPS accuracy beyond the supplied residual and coverage
checks. Exported LAS/LAZ files embed the valid CRS WKT; PLY/PCD files carry a
`.geo.json` sidecar with the CRS and coordinate-frame metadata. The 3D viewer
can load the resulting cloud, edit/save it, generate an orthophoto, and show an
optional satellite/street map layer under the cloud.

Every rejected assessment still writes its report and exits with status 2; no
cloud is exported. Repeated fixes or poor horizontal coverage cannot determine
heading. Reject extrapolation and long trajectory gaps. Keep a local model plus
its geographic transform where large projected coordinates could reduce numeric
precision.

### SESSION-01 evidence

The assessment at
`D:\ARQUIVOS_TESTE_2\GALPAO-ALUGADO-FABRICA\SESSION-01` is recorded in
`build/gps-validation/session-01-assessment/georeference_report.json`. With
`--time-offset 0` (strictly diagnostic and unverified), it decoded 8,661 GPS
rows, 38 unique fixes over 71.655 s, and a 61 m recorded height span. The fit
was rejected: only 11/38 matches were inliers (28.9%) and horizontal extent was
9.10 m, below the 10 m minimum; independent clock synchronization and gravity
alignment were also unverified. Consequently no cloud export was produced.

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
