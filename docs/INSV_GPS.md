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

`georeference` places a scan on a projected CRS (SIRGAS 2000 UTM in Brazil, WGS84
UTM elsewhere, or `--crs`). The fit is yaw and XY translation at scale `1.0`. Z
stays the local SLAM height: recorded GPS altitude has neither the accuracy nor a
known datum to anchor it, so use `--shift-z` when the site's elevation is known.

```powershell
python lidarcamera360.py georeference --insv capture.insv `
  --trajectory slam.txt --cloud colored.ply --output georeferenced `
  --dt-sync 3.107 --street-bearing 16
```

North comes from the strongest evidence the capture offers:

| Source | How | Status |
| --- | --- | --- |
| `--yaw-deg` | Known rotation from local XY to grid E/N, counter-clockwise | `resolved` |
| `--street-bearing` | True bearing of a street or long wall, measured on a map, matched to the scan's wall directions | `resolved` |
| GPS fixes matched by time | Needs `--dt-sync`, at least 6 fixes and 10 m of baseline | `resolved` |
| Aerial imagery | The scan's walls registered against Esri World Imagery | `needs_review` |
| Imagery and GPS within 10° | Both of the above, independently | `resolved` |
| None of these | The scan is placed on the GPS anchor, its rotation arbitrary | `unresolved` |

Imagery cannot certify itself. In a negative control on `trimtest`, the same walls
were registered against imagery of 12 wrong places 250-1800 m away, and 7 passed the
same score gates, because a street grid fits other street grids. The run therefore
writes `north_alignment_review.jpg`, the scan drawn over the imagery, for a person to
check. Imagery also leaves position loose along a straight street, where GPS decides
it, and the imagery may carry its own offset. None of this is survey grade.

The imagery request reveals the site's location to the tile provider, so `--offline`
is there to skip it. The desktop workflow's automatic stage uses imagery too.

The report's `status` is `accepted`, `review_required` or `location_only`. The CLI
exits 0 for the first two and 2 for the last, which still writes its deliverables
but with an arbitrary rotation. `yaw_heading_deg` is relative to grid north;
`yaw_true_deg` and `local_y_true_bearing_deg` include the grid convergence, as do
the headings written to camera EXIF and CSV. EXIF altitude is left out, since local
Z is not an elevation.

Trimming a video drops its INSV trailer, so `trim` keeps the fixes in
`<video>.gps.json` with the trimmed clip's start in the capture. A clip trimmed
before this existed has no sidecar: pass the original INSV with `--capture-origin`
set to the `origin_capture_sec` of its `.timelapse.json`.

LAS/LAZ files embed the CRS WKT, and PCD stores float64 coordinates. The 3D viewer
can load the resulting cloud, edit/save it, generate an orthophoto, and show an
optional satellite/street map layer under the cloud.

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
