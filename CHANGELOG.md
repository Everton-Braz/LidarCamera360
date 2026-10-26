# Changelog

All notable changes to **LidarCamera360** are documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

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
