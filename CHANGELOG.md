# Changelog

All notable changes to **LidarCamera360** are documented in this file.

The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
