# RF-DETR validation — 2026-09-24

Dataset: `D:\ARQUIVOS_TESTE_2\SMALL-DATASET-TEST` (89 dual-lens pairs,
178 images at 3840 × 3840; 3,786,042 LiDAR points).

The native Windows worker loads RF-DETR Seg Medium once through TensorRT 10.
The packaged application contains its ONNX model, worker, runtime libraries,
and engine builder. No Python installation, PyTorch, or Spirula GUI is needed
for native inference. The app distribution is the entire
`dist/LidarCamera360` folder; launch `LidarCamera360.exe` inside it.

## Measured results

Hardware: NVIDIA GeForce RTX 5070 Ti. The desktop and existing GPU applications
remained open (roughly 6.8 GB VRAM already occupied in the initial observation).
No simultaneous inference job from this task ran during the measured batches.

| Check | Result |
| --- | --- |
| Native full batch, including PNG writing | 178 masks in 13.47 s (13.21 images/s) |
| Packaged masking command, including validation | 19.08 s, cached engine load 0.64 s |
| Empty-cache engine compilation and load | 90.98 s on this GPU |
| SfM masked colorization and geometry filtering | 18.0 s; 2,866 points removed |
| Direct masked colorization and geometry filtering | 17.6 s; 2,870 points removed |
| Direct output geometry read-back | 3,783,172 points; all positions belong to the original cloud |
| Matched positions with a color change exceeding 10/255 in any channel | 15,003 |

Settings: person threshold 0.5, detection-box margin 0.03, no tiling, operator
radius 1 m. Geometry removal requires at least two nearby masked observations
and a strict majority over clear observations. This is a conservative filter;
it does not establish that every operator return has been removed. Masks reject
person pixels for color sampling at every depth, including all four bilinear
taps. Surfaces without an accepted color observation keep the existing gray
fallback. Shadows and scanner hardware are not separate segmentation classes.

## Checks

- The packaged executable exited 0 from an isolated working directory and a
  PATH containing only Windows directories, including a new engine build.
- All 178 masks were read back and checked for matching dimensions and binary
  white-keep / black-person polarity. A contact sheet was visually inspected.
- Parallel and serial native decoding produced identical PNGs across all 178
  images. A separate full-frame reference test covers cropped interpolation,
  Euclidean expansion, empty/border masks, thresholding, and nonfinite outputs.
- 52 focused Python tests passed across masking, GPU projection, calibration,
  CLI, workflow, and 3DGS export; the native decoder test passed.
- On an 18,931-point, eight-image subset, CPU and Vulkan retained identical
  positions. Both masked and unmasked comparisons had six points differing
  by more than 2/255 in a color channel (maximum 19/255, mean 0.0133/255).
  Color outputs are therefore not claimed to be byte-identical across backends.
- Both updated GUI panels were rendered and inspected at 1280 × 900.

Raw logs, previews, cloud comparison JSON, and the test copy are under
`build/person-mask-validation/`. Final masks and colorized clouds are in the
original dataset's `masks/` and `deliverables/` directories. Mask names retain
the camera-relative source extension, for example `cam0/frame_000001.jpg.png`.
The workflow also exports these masks into `colmap_3dgs/masks/` when COLMAP
export is selected; trainer mask configuration remains trainer-specific.

## Fixed-surface regression — 2026-09-25

The user supplied a CloudCompare view of a door with a hole in
`GALPAO-ALUGADO-FABRICA/SESSION-02/output3/deliverables/` after selecting a
1 m operator radius. Read-back of the PCD headers gives 5,355,259 points in
the raw `output3` cloud and 5,128,382 in that result: **226,877 removed**.
The same session's `output2` run has RF-DETR masks and retains 5,355,057
points: only **202 removed** relative to its 5,355,259-point raw cloud.
These counts show the unmasked trajectory corridor is the destructive path;
they do not by themselves locate the door in 3D.

The CLI and both colorization functions now reject a positive radius without
person masks, before writing an output cloud. The GUI enables person masking
when a positive radius is selected. Mask-supported removal additionally
protects points with a dense or sparse local planar neighborhood, even when a
person silhouette overlaps them. Synthetic door and ground regressions test
this distinction. A read-only dry run on `output2` with its existing person
masks and the session's calibration retained 5,355,132 of 5,355,259 points
(127 mask-supported removals), without writing deliverables. The test
calibration was taken from the session root, so this count is a safety check,
not an exact recreation of the earlier `output2` run. Reprocess from the
original raw PCD to repair an existing corridor-filtered deliverable;
filtering cannot restore deleted points from that deliverable alone.

The `output2` camera images also show the scanner housing fixed at the bottom
of both lenses while RF-DETR masks only the people. On 158 frames per lens,
the camera-attached detector marked 2,704 (`cam0`) and 2,783 (`cam1`) pixels
at 192 x 192 sampling resolution. The combined image masks exclude these
pixels from colorization and 3DGS export; their separate person-only layer
remains the sole source of geometry-removal votes. This does not establish
complete scanner coverage or classify arbitrary moving non-person objects.
Reduced-resolution JPEG decoding took 8.6 s for all 316 images in the
diagnostic scan, compared with 19.9 s for full-resolution decoding.

The rebuilt onedir executable was tested from an isolated working directory
with `PATH` limited to Windows directories. It generated two masks through
the bundled native worker and exited 0; read-back confirmed schema 2,
combined-foreground polarity, and two separate person-only masks. The same
executable rejected an unmasked 1 m radius at argument parsing with exit 2.
All 114 Python tests passed after the changes.

## Configurable single-mask format

The next mask format removes the hard-coded bottom-of-lens scanner heuristic.
The Mask settings dialog lets users draw normalized rectangles separately for
each camera and choose a 0–25% centered circular fisheye-border cutoff. Its
settings are stored in `mask_settings.json` and in schema-3 `manifest.json`.
RF-DETR person detections, selected rectangles, and border cutoff are combined
into **one binary PNG per source frame**; there is no `persons/` mask tree.
For geometry removal, the manifest identifies fixed exclusions so those black
pixels are never mistaken for person evidence. The same combined PNG serves
colorization and 3DGS export.

A two-frame native smoke test with camera-specific rectangles and a 4% border
produced two binary PNGs, schema 3, and no `persons/` directory. Focused tests
cover geometry protection inside a rig rectangle, per-camera application,
and border math. The user-provided `output05` remains unchanged until a new
masking run uses the revised settings.

## Area-based border and mask preview

The border percentage now means the approximate fraction of source pixels
excluded by the centered circle. At 0% the border mask is disabled; at 1% a
3840 x 3840 frame excluded 1.000% of pixels in a local check. The dialog's
Preview mask button runs RF-DETR on the selected frame in a temporary folder
and shows its person mask combined with the current fixed shapes and border.
The preview can toggle back to the source image without changing the dataset.

The 121-test Python suite passed. The rebuilt Windows onedir app passed an
isolated-PATH two-frame native masking run (one binary PNG per frame, no
`persons/` duplicate) and its GUI smoke test exited 0. The release was built
side by side under `dist/mask-preview-release/` because a running process held
the previous distribution directory open.

The mask editor now supports mouse-wheel zoom, right/middle-button panning,
rectangles, ellipses, and normalized freehand polygons. Polygon vertices can be
adjusted after drawing. Its camera timeline uses natural filename ordering and
starts on the first extracted frame. Editing commands use icon-only controls
with hover tooltips; save and cancel retain text labels.
