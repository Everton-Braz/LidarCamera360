# Changelog

All notable changes to **LidarCamera360** are documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [v0.4.0] - 2026-11-12

### 📦 Project package, people removal and georeferencing
- **`.lc360` v1 project package**: each dataset keeps a `project.json` (schema `lc360-project`) that names its cloud,
  images, camera poses and trajectory. `spec/SPEC.md` defines the contract; `raven_app/project_validate.py` checks
  a package in two levels (`E_JSON`, `E_SCHEMA`, `E_VERSION`, `E_PATH_ESCAPE`, `E_MISSING_ASSET`, `E_POSES`,
  `W_NO_ID`) and `write_project` refuses to write a manifest with an error. Conformance examples are in
  `spec/conformance/`. `jsonschema` is now a dependency and the packager refuses to build without it.
- **`export` subcommand** writes extra formats (PCD, LAZ/LAS, GeoJSON, GPS tracks, trajectories) from a project
  after the run, so a run no longer guesses which formats are wanted. A rig profile outside the package is copied
  under `rig_profile/` instead of failing with `E_PATH_ESCAPE`.
- **`--remove-dynamic`** removes moving and standing people from exported clouds: LiDAR points that later scans see
  through are carved, then points the person masks vote as a person across views are dropped. Facades, planar
  structure and points high above the floor are kept. Available in the CLI, workflow, GUI and MCP server.
- **Operator removal radius defaults to 1 m** whenever person masks are used (0 still disables it).
- **Per-profile mask settings**: rig profiles can carry `mask_settings` (fisheye border cutoff, fixed shapes such as
  the scanner body); they apply when no `--mask-config` or dataset `mask_settings.json` is given and prefill the
  mask dialog. `mask-persons` takes `--profile`.
- **Georeferencing solves north from evidence** and reports how far to trust the result. Imagery-derived north
  still needs a manual check; do not accept it unreviewed.
- **SLAM loop closure** survives pose jumps and rejects wrong ICP loops.

### 🧭 Timelapse and rig safety
- The workflow and the COLMAP export use the **calibrated timelapse clock offset**; timelapse frames carry capture
  times, so the video clock's gyro offset does not apply. The IMU offset stays in the run record.
- **Rig calibrations made for another frame size are refused**: the refinement now starts from a calibration that
  matches the frames (for example 2880 px against a 3840 px profile) and skips mismatched files in later stages.
- `--slam-dir` imports an existing SLAM and `--slam-profile` picks the timelapse settings.

### 🛠 Workflow options
- `--trim-end-at` and `--chunks` for datasets, and `--sfm-matcher` (`sift` or `aliked-lightglue`).
- The initial time offset is read from existing calibration files.
- Raven One RS rig profile and rig bootstrap gates.

### 🏠 Timelapse captures end to end (CASA 374 Home3)
- **Rig recalibration from SfM** no longer collapses the lever arm on timelapse captures: the 35-frame window-local
  Sim(3) absorbed R_body @ arm on a straight walk (2.6 cm instead of 17 cm → "implausible camera lever arm").
  A lever-aware timelapse alignment now supplies the fit and every accepted pose (18.4 / 18.3 cm on Home3).
- **SLAM revisit refinement** is accepted only when the independent surface check does not get worse. A better
  loop closure alone accepted candidates whose surfaces went 9.9 → 19.2 mm RMSE (trajectory moved up to 7.8 m),
  which also broke the SfM alignment (26 cm RMSE). The workflow no longer uses a rejected refinement, and
  cached reports accepted under the old rule are judged again.
- **Colorization covers the whole capture**: SfM colorization adds the frames sparse/0 never registered, with
  SLAM + calibrated-rig poses and the nearest fitted frame's PPISP/bilateral correction (Home3: uncoloured
  points 60.7 % → 4.4 %).
- **Motion-blurred frames are left out** of colorization and the 3DGS export (sharpness below half of the
  same-lens neighbours' median; `raven_app/frame_quality.py`, cached in `images/sharpness.json`).
- **3DGS pose adjustment** no longer fails with "Singular matrix": observations of points out of 0.15–100 m range
  are dropped before each normal-equation step (Home3: 1.97 → 0.88 px median reprojection).
- **Spirula Studio v2026.10.30**: person masks are generated before SfM and passed to it (the operator walks with
  the rig), and dark captures get `--image-exposure auto` when the build supports it.
- `workflow --rebuild slam sfm` recomputes those stages while reusing the extracted frames; the previous
  `sparse/` is moved to `sparse.previous_<time>`.
- `workflow --densify roma`: RoMa v2 dense matching (`spirula dense`) on the 3DGS export, added to the LiDAR seed
  only in its gaps (`raven_app/dense_fusion.py`). The fused cloud becomes the export's seed
  (`sparse/0/points3D.*`, LiDAR-only seed kept as `points3D_lidar_only.*`), and Spirula's RoMa run moves to
  `densify/roma_run/` so its auto-detected `dense/roma.ply` (RoMa alone) cannot replace the 30 M point seed.
  A dense point is added only on the LiDAR surface around it (within 1 cm of the plane of its LiDAR neighbours
  within 20 cm): without that test, Home3's RoMa points (a median 11 cm off the walls) formed a loose shell.
  On that dark capture only 330 of 48,937 RoMa points qualify, so densification stays off by default.
- The COLMAP export retries its final folder swap, which Windows can refuse while freshly written files are open.

### 🧰 Rig profiles v2, new-rig calibration and a faster refinement (Phases 3–5)
- **Profile v2**: one file describes any setup — scanner plus bodies of cameras (main camera `cam0` [+ `cam1`], auxiliary cameras with their own clock), each with its model, intrinsics and mount (`raven_app/rig_profile.py`).
  - v1 files (every existing calibration JSON) migrate on load. Saved v2 files keep the v1 keys, so every colorizer reads them unchanged. The bundled Raven + X4 and Eagle profiles are now v2; their transforms and intrinsics are identical.
  - Your profiles live in `%APPDATA%\LidarCamera360\profiles`, so app updates keep them. A user profile with the same file name replaces a bundled one; `is_default` picks the default.
  - Auxiliary cameras in a profile fill in a capture's `--third-camera` entries that only name a source: no sample SfM is needed for a camera the profile already calibrates.
- **New rig without a validated profile (Phase 3)**:
  - `profile new` (CLI), `new_profile` (MCP) and the **New rig…** wizard in the GUI build a starting profile. You give:
    - a camera from the catalogue: Insta360 X3/X4/X5/ONE RS 1-inch, any equirectangular 360, GoPro MAX, phone main / ultra-wide, GoPro Wide, generic pinhole;
    - the measured camera offset;
    - an orientation preset.
  - The scanner tilt is built in, so the presets reproduce the validated Raven and Eagle mounts within 1.4°.
  - `calibrate --coarse` first estimates the clock and the mount rotation of every camera from motion (hand-eye):
    - the camera's rotation between frames (SIFT + essential matrix) is matched against the SLAM trajectory;
    - their angles give the clock without any mount guess, and the rotation axes give the mount.
    - Hand-eye accuracy against the accepted calibrations:

      | Dataset | Clock | Mount |
      |---|---|---|
      | PIER | 2 ms | 0.19° |
      | HarmonyMall | 2.5 ms | 0.17° |
      | Equirect video | 2 ms | 0.16° |
      | TESTE_3 X4 | 4 ms | 0.26° |
      | TESTE_3 phone | 14 ms | 0.8° |
    - The photometric refinement then finishes.
  - Exit test on PIER: starting 15° / 12 cm / 2.8 s away from the accepted calibration, it ended 0.02° / 0.85 cm / 0.06 ms away in 195 s.
  - A camera whose rotations disagree with the scanner's by more than 2° (median) is rejected: it is not rigidly mounted or has wrong intrinsics.
  - `--save-profile NAME` stores an accepted result as a profile, with the capture in `validated_on`.
- **CLI**: `profile list | show | validate | import | export | create --from-calibration | new | set-default | cameras`; `calibrate --coarse --coarse-dt-window --save-profile --report`.
- **MCP**: `validate_profile`, `save_profile`, `camera_catalogue`, `new_profile`; `calibrate_rig` takes `coarse`, `coarse_dt_window` and `save_profile`; `get_profile` returns v2.
- **GUI — Rig Calibration page rebuilt as a profile editor**:
  - profile list (yours and bundled, default first);
  - a form for the name, scanner and description;
  - a camera table: model drop-down, image size, focal length / centre, position in cm, and rotation relative to an upright forward lens;
  - add or remove auxiliary cameras;
  - save (a bundled profile is saved as your own copy), save as, import, export, use as default, and the New rig wizard.
  - **Validate on a capture** runs `calibrate` (optionally with the coarse search), shows the held-out verdict and saves an accepted result as a profile.
  - The raw JSON editor is under Advanced.
- **Speed (Phase 5)**:
  - Frames are projected and sampled in parallel threads, in float32, with cached frame poses and pre-gathered points. Images load and blur in parallel.
  - One cost evaluation went from 0.55 s to 0.15 s and a visibility freeze from 3.0 s to 0.5 s, with the same cost value to 5 decimals.
  - PIER refinement: 349 s → 121 s, same result (Δt −7.0255 s, held-out +29.8 %).
  - The 3×3 rotation is written out instead of a BLAS matmul, because concurrent OpenBLAS calls crashed.

### 🌐 Equirectangular exports and wide auxiliary cameras
- **Gyro sync for exported MP4s**: an equirectangular MP4 exported by Insta360 Studio has no gyro trailer. The sibling INSV with the same name and duration (same time base) is now used for the IMU cross-correlation. On SMALL-DATASET-TEST: Δt −7.746 s, correlation 0.999.
- **360 main camera in the rig refinement**: an equirectangular dataset is refined as a single EQUIRECTANGULAR camera at the profile's cam0 mount.
  - A coarse search over the 24 axis-aligned orientations finds how the export is oriented. The held-out check compares against the profile's orientation.
  - The written calibration is a proper equirectangular one (no cam1 keys).
- **Narrow cameras (phones)**: the clock scan re-freezes visibility for every candidate offset. Points leaving the frame no longer make far-off offsets look better.
- **Sample SfM**: when the sample's main rig fails the plausibility gates, the hardware profile is kept as the main rig and auxiliary cameras are still calibrated. Before, every attempt crashed. An exploratory solve no longer marks its alignment as rejected.
- **Refinement results are keyed on the solver version**, so an updated solver never reuses a stale cached result.
- TESTE_3_CAMERAS with the 0.6× phone as OPENCV: sample SfM now calibrates it (fx ≈ 2646 px, 37/37 poses). The photometric refinement is accepted for both the X4 (+19.2 %) and the phone (+3.1 %).

### 🧭 Any camera model and auxiliary cameras in the rig refinement (Phase 2)
- **`raven_app/camera_models.py`** is one projection and validity layer for THIN_PRISM_FISHEYE, OPENCV_FISHEYE, PINHOLE, OPENCV and EQUIRECTANGULAR cameras. It uses the colorizer's own 12-parameter packing.
- **The refinement is generic over *bodies*** (cameras sharing one housing and one clock). The main rig is cam0 + cam1, or a single equirectangular cam0. Each auxiliary camera (`third_camera.json`) is a body with its own clock.
  - Bodies are refined one after another: the main rig first, then each auxiliary camera against it.
  - An auxiliary camera refines its rotation and clock, keeping the measured lever arm; the clock gets a two-level ±3 s scan.
  - Each body has its own held-out acceptance.
- **Robustness**:
  - Points leaving the image or field of view keep a penalty, so the mount cannot "look away" to lower the cost.
  - Bounded mount search.
  - Clock and rotation are re-centred alternately: a yaw error trades off against the clock while the rig turns.
  - Held-out scoring compares both states on the same observations.
  - Correct `cv2.resize` pixel-centre mapping.
- **Workflow**: in direct mode, an auxiliary camera that failed the sample-SfM calibration but has calibrated intrinsics is deferred to the photometric refinement. It is coloured if accepted; otherwise the workflow stops before colorization, as before. Results are saved to `third_camera_calibrated.json`.

### 🤖 MCP calibration and processing tools
- New tools: `list_profiles`, `get_profile`, `validate_calibration`, `calibrate_rig`, `run_workflow`, `colorize`, `get_job_status`, `cancel_job` and `list_jobs`.
- Long runs are background CLI jobs (`raven_app/mcp_jobs.py`) with a persisted log and state. Their status reports the stage, outputs and rig-refinement verdicts.
- CLI errors without a message (e.g. `MemoryError`) now print their type instead of an empty line.

### 📐 Per-capture photometric rig refinement (Phase 1)
- **`raven_app/photometric_rig.py`** refines the time offset and the mount of the dual-fisheye rig by multi-view colour consistency, starting from the hardware profile. It always runs a dense dt scan, then coarse-to-fine Powell with a weak profile prior.
- **Acceptance rule**: a result is used only if it lowers the cost on held-out frames (never seen by the fit) by ≥ 1 % and passes the rig plausibility gates. Otherwise the profile and the gyro dt are kept, and the rejected attempt is recorded.
- **Workflow and `colorize`**: refinement runs before direct colorization (`--rig-refine photometric|dt|none`, default `photometric`). The result is written to `rig_calibration_photometric.json` and reused while its inputs are unchanged. A manual `--dt` stays fixed (mount only). HarmonyMall: about 4 min, held-out cost −5.4 % vs the profile.
- **`lidarcamera360 calibrate`** runs the refinement alone and prints the verdict and held-out metrics. It exits 3 when the result is rejected.
- **GUI**: the workflow page has a rig profile selector, a rig refinement mode and the colorization time window. They are passed as `--profile`, `--rig-refine` and `--time-window`, with pt-BR translations.

### 🎯 Colorization of dynamic scenes and calibration safety (Phase 0)
- **Time-gated colorization**: each camera view only colours, and occludes with, LiDAR points scanned within ±6 s of its capture (`--time-window`, 0 disables). Cars that left or arrived between passes and people walking no longer paint the other pass. It is implemented in the Vulkan engine (protocol `RVT1`..`RVT6`: point `w` = scan time, per-view time and window), the CPU fallback and the reconstruction (SfM) path. Per-point times come from the SLAM `scan_ranges.csv`.
- **Direct mode uses SLAM poses only**: removed the hidden SfM "trajectory drift correction" that replaced SLAM poses whenever an SfM alignment existed, even a poor one.
- **Locked time offset**: a manual `--dt` or an accepted gyro sync is final. The SfM alignment is solved at that offset and never moves it. The sample-SfM rig no longer overrides it either.
- **Rig plausibility gates** (`raven_app/rig_gates.py`): SfM and sample-SfM rigs are rejected when they leave the hardware profile, by lens baseline ±1.5 cm, angle between lenses ±2°, 10° rotation or 8 cm position per lens, or a 40 cm lever arm. They are quarantined to `rig_calibration.rejected.json`, and the workflow falls back to the profile. Stale implausible per-dataset calibrations are ignored by the resolver.
- **Hardware profile selection**: `--profile <id|path>` on `workflow` and `colorize`. The default is the profile marked `is_default`, and Eagle captures use the Eagle profile. `colorize` without `--calib` now resolves a validated calibration.
- **Mask/frames consistency**: `--masks-dir` from another extraction (different fps or frame times) is refused with a clear message instead of being applied to the wrong frames by file name.
- **No silent loss of the colorized cloud**: with no export format selected, the run's PLY is kept. Format filtering only deletes clouds produced by this run, never older deliverables.

---

## [v0.3.0] - 2026-10-15

### 🌐 360° Equirectangular Video Support
- **Single-Camera 360° Pipeline**: Full native pipeline for stitched equirectangular videos (`.mp4`), enabling single-track 360 camera ingestion alongside dual-fisheye INSV containers.
- **COLMAP Equirectangular Model (ID 17)**: Complete support for COLMAP camera model 17 (`EQUIRECTANGULAR`), including cameras.bin binary export, parameter counts, and optical geometry.
- **Spherical LiDAR Colorization**: Direct spherical projection mapping cartesian 3D LiDAR points $[X, Y, Z]$ into spherical longitude/latitude coordinates $[u, v]$ on equirectangular frames.
- **CLI Video Flag Alias**: Extended CLI parsers (`workflow`, `colorize`, `extract-insv`, `georeference`) with `--video` as an alias for `--insv`.

### 🔄 RootSLAM & Pose-Graph Revisit Optimization
- **Square-Root Information Smoothing**: Implemented `RootSLAM` pose graph solver with square-root landmark marginalization, right Jacobian SO(3) updates, and synthetic/real loop closure convergence.
- **Trajectory Refinement Constraints**: Added geometric constraints (`slam_refinement*.py`) mitigating tilt degeneracy and drift across multi-pass revisits.

### ✂️ Synchronized Dataset Trim & Chunking
- **Synchronized Temporal Trim**: Subcommand `trim` and workflow flags `--trim-start` / `--trim-end` for synchronized cutting of both ROS1 bags and 360 video containers with sub-frame time accuracy.
- **Overlapping Dataset Chunks**: Subcommand `chunk` to divide massive datasets into overlapping manageable chunks with temporal interpolation for downstream 3DGS training.
- **UI Integration**: Integrated trim spinboxes, export dialogs, and chunking workflows directly into `UnifiedWorkflowView`.

### 🛡️ 100% Privacy-First Architecture (Telemetry-Free)
- **Complete Telemetry Removal**: Completely eliminated all anonymous telemetry, heartbeat HTTP calls, and installation UUID tracking from the application, settings interface, and CLI/MCP layers.
- **Zero Background Network Activity**: The application does not transmit any usage data, telemetry payloads, or identifiers.
- **Automated Configuration Sanitization**: Automatically purges any legacy telemetry configuration keys and persistent client ID files from user directories.

### 🤖 Native Model Context Protocol (MCP) Server
- **AI Agent Automation**: Built-in JSON-RPC 2.0 stdio MCP server (`python lidarcamera360.py mcp`) exposing 7 specialized tools to AI assistants (Claude Desktop, Antigravity, Cursor):
  - `doctor`: Automated diagnostic inspection of native MSVC engines, Vulkan compute devices, and Spirula SfM status.
  - `inspect_bag`: Fast, zero-ROS ROS1 bag inspection for LiDAR/IMU/Image topics and message frequencies.
  - `extract_gps`: High-frequency GNSS extraction from Insta360 INSV containers into GeoJSON, GPX, and CSV.
  - `extract_insv_frames`: Hardware-accelerated dual-fisheye frame extraction via PyAV.
  - `georeference`: Full georeferencing and CRS reprojection pipeline for point clouds and COLMAP datasets.
  - `check_updates`: Remote GitHub release checking and changelog inspection.
  - `submit_feedback`: Manual, voluntary bug reporting with packaged diagnostics.

### 🔧 Core Engine Quality & Mathematical Precision
- **Optical Model Selection Preservation**: Enforced contract in `calibrate_third_camera_intrinsics` to preserve explicit user model choice on SfM model mismatch.
- **Dual-Camera Recalibration Robustness**: Guaranteed `T_lidar_to_cam1_rigid_4x4` generation via rigid baseline in dual-fisheye setups without misdetecting single-camera mode.
- **Non-Finite Point Filtering**: Added strict `np.isfinite` filtering on point clouds prior to colorization and projection.
- **FAST-LIVO2 VIO Memory Safety**: Prevented memory crashes during heavy visual feature tracking by enforcing strict boundary validation on image patch buffers.
- **Comprehensive Test Suite**: 100% passing test suite across 282 unit, integration, and UI tests (15 subtests).

---

## [v0.2.0] - 2026-09-23

### 🦅 3DMakerPro Eagle LiDAR & Insta360 X6 Hardware Support
- **Full Eagle Scanner Integration**: Added native parsing and processing for the **3DMakerPro Eagle LiDAR Scanner** via Livox CustomMsg (`livox_ros_driver2/msg/CustomMsg`) and standard PointCloud2 protocols.
- **Insta360 X6 Support**: Fully calibrated dual-fisheye optical parameters and extrinsics profile for the new Insta360 X6 8K panoramic camera.
- **Automated IMU Time Synchronization**: Programmatic gyro cross-correlation between camera telemetry and Eagle IMU data, achieving sub-millisecond clock sync ($\Delta t$).
- **Dual-Lens Extrinsic Rig Calibration**: Pre-calibrated extrinsic profiles (`configs/rig_profile_eagle.json`, `FAST-LIVO2/config/eagle.yaml`) with Cam0 & Cam1 lever arms and orientation optimization.
- **Direct Rigid & Consensus Colorization**: Verified top-3 multi-view consensus and trajectory-based colorization for Eagle LiDAR scans.

### ✂️ Interactive 3D Oriented Bounding Box (OBB) & Viewport Gizmos
- **Oriented Bounding Box (OBB)**: Full 3D rotation around the vertical axis (Yaw) allowing precise slicing aligned with building facades, roads, or terrain features.
- **Interactive Transform Gizmos**: Added intuitive face pull handles with anchored opposite faces, planar translation gizmo, and horizontal rotation ring.
- **Persistent Sliced View**: Decoupled clipping logic from bounding box visibility—inspect pristine sliced point clouds with gizmos hidden while the shader cut remains active.
- **Quick Alignment Tool**: Instant one-click alignment (`[Alinhar]`) to match box yaw with cloud orientation.
- **Sub-Volume Cloud Export**: Crop and export clipped point clouds directly to PLY, PCD, LAS, and XYZ.

### 🌟 Metric LiDAR 3DGS Seed Initialization (Gaussian Splatting)
- **Direct Metric Seeding**: Export colorized metric LiDAR point clouds directly as initial seeds (`points3D.ply`, `points3D.bin`, `points3D.txt`) into COLMAP sparse workspaces.
- **Multi-Framework Compatibility**: Seamlessly trains with Inria 3DGS, Nerfstudio (`splatfacto`), PostShot, Spirula Studio, and LichtFeld Studio.
- **Intelligent Density Striding**: Auto-subsamples ultra-dense clouds down to optimal target densities (~1.5M points) to maximize 3DGS training convergence without CUDA VRAM exhaustion.

### 🗺️ Georeferencing, GCPs & GeoTIFF Orthophotos
- **Trajectory Georeferencing**: Automated alignment of SLAM trajectories to GNSS fixes extracted directly from native INSV video trailers.
- **Ground Control Points (GCP)**: Support for GCP constraints and survey benchmarks.
- **Orthophoto Generation**: High-resolution GeoTIFF orthomosaic export with embedded georeferencing and auxiliary world files (`.tfw`).
- **Standardized Survey Export**: Export georeferenced point clouds in LAS 1.4 / LAZ with Compound CRS support (SIRGAS 2000 / UTM zones and ellipsoidal/orthometric heights).

### 🖥️ Viewer & Export Enhancements
- **High-Resolution Viewport Render**: Export high-DPI screenshots and raster/vector PDFs with presets up to 8K UHD.
- **Multi-Modal Shading**: Independent A/B viewport color modes (True RGB, LiDAR Intensity, Height Colormap, Surface Normals).
- **Measurement Tools**: Point-to-point 3D distance and elevation inspection.

---

## [v0.1.0] - 2026-09-08

### Initial Release
- **Offline FAST-LIVO2 SLAM Engine**: Native C++17 Windows MSVC estimator without ROS or Linux dependencies.
- **Hardware Profile**: 3DMakerPro Raven LiDAR Scanner + Insta360 X4 360° camera (+18.5 cm vertical lever arm).
- **Vulkan GPU Compute**: Hardware-accelerated dual-fisheye point cloud colorization (`vulkan_colorizer.exe`).
- **Telemetry Extraction**: In-process INSV dual-fisheye decoding and gyro telemetry synchronization.
- **Modern Fluent UI**: WinUI 3 Fluent design system with Mica/Acrylic effects and multi-language support (English, Portuguese, Spanish, French, German, Chinese, Japanese).
