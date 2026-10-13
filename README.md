<div align="center">

# LidarCamera360

Native INSV GPS extraction: see [GPS export and georeferencing requirements](docs/INSV_GPS.md).

**Unified LiDAR-Inertial SLAM, 360° Camera Telemetry, Rig Calibration, and 3D Colorization Suite**

[![Release](https://img.shields.io/badge/Release-v0.2.0-brightgreen.svg)](https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![C++17 MSVC](https://img.shields.io/badge/C%2B%2B-17%20MSVC-orange.svg)](https://visualstudio.microsoft.com/)
[![Vulkan](https://img.shields.io/badge/GPU-Vulkan%20Compute-red.svg)](https://www.vulkan.org/)
[![PyQt6 Fluent](https://img.shields.io/badge/UI-PyQt6%20Fluent%20Widgets-teal.svg)](https://github.com/zhiyiYo/PyQt-Fluent-Widgets)
[![Tests](https://img.shields.io/badge/Tests-252%20passed-success.svg)](tests/)
[![License](https://img.shields.io/badge/License-Dual%20MIT%20%2F%20GPLv3-green.svg)](LICENSE)

<br/>

<img src="assets/logo.png" alt="LidarCamera360 Logo" width="220" />

<p>
  <b>LidarCamera360</b> is an open-source, high-performance desktop application and headless CLI suite for fusing 3D LiDAR point clouds with 360° panoramic cameras (dual-fisheye). It executes offline LiDAR-inertial SLAM without requiring ROS or Linux, performs hardware-accelerated video decoding, synchronizes sensor clocks via IMU gyro cross-correlation, estimates rigid extrinsic geometry, and colorizes dense point clouds using Vulkan GPU compute shaders.
</p>

</div>

> [!NOTE]
> **Hardware & Rig Compatibility:**  
> **LidarCamera360** comes pre-calibrated and out-of-the-box ready for:
> - **3DMakerPro Raven LiDAR Scanner + Insta360 X4** (+18.5 cm vertical lever arm, `configs/rig_profile.json`, `FAST-LIVO2/config/raven.yaml`)
> - **3DMakerPro Eagle LiDAR Scanner + Insta360 X6** (Livox CustomMsg protocol, `configs/rig_profile_eagle.json`, `FAST-LIVO2/config/eagle.yaml`)
>
> Its modular architecture is completely sensor-agnostic: it can be easily adapted to any mobile or handheld scanning setup (e.g., Livox, Ouster, Hesai, Velodyne, RoboSense, or VanJee LiDARs paired with dual-fisheye or perspective camera arrays) by customizing the rigid extrinsic geometry and camera model in [`configs/rig_profile.json`](configs/rig_profile.json) and adjusting topic mappings in [`FAST-LIVO2/config/`](FAST-LIVO2/config/).

---

## 🌟 Key Capabilities

1. **In-Process ROS Bag Inspection & Processing (Zero ROS Required)**
   - Operates natively on Windows without WSL, Docker, or ROS master daemons.
   - Reads ROS1 `.bag` files, parses `sensor_msgs/PointCloud2` (with per-point timing), `livox_ros_driver2/CustomMsg`, IMU packets, and compressed camera topics.

2. **Native Offline FAST-LIVO2 SLAM Engine**
   - Bundled MSVC C++17 offline estimator (`fastlivo2.exe`).
   - Supports standard point clouds and Livox/Eagle high-rate scan formats.
   - Produces high-accuracy LiDAR odometry, 6-DoF trajectories, and dense point cloud maps directly from raw packets.

3. **High-Throughput INSV Video Extraction & Multi-Source Auxiliary Ingestion**
   - In-process dual-track and single-stream INSV decoding via bundled PyAV / FFmpeg with hardware acceleration (`CUDA`, `D3D11VA`, `VideoToolbox`, `VAAPI`).
   - Native support for single-lens and auxiliary panoramic action cameras (Insta360 Ace Pro, GO 3, and auxiliary camera arrays) with dynamic stream selection in UI/CLI.
   - Fast SIMD-based grayscale Laplacian sharpness selection across configurable window intervals and GOP keyframe seeking.

4. **Programmatic IMU Gyro Cross-Correlation & Timelapse Auto-Calibration**
   - Decodes embedded binary telemetry trailers from Insta360 INSV files (Gyro and Exposure records) for video and timelapse captures.
   - Computes Pearson cross-correlation between camera gyro angular velocity norms and LiDAR IMU gyro streams to achieve sub-millisecond clock synchronization ($\Delta t$).
   - Automatic temporal sync adaptation for variable-rate timelapses and custom trajectory windows.

5. **Thin Prism Fisheye Optical Modeling & Multi-Modal Calibration**
   - Complete 12-parameter Thin Prism fisheye distortion model ($f_x, f_y, c_x, c_y, k_1..k_4, p_1, p_2, s_{x1}, s_{y1}$).
   - Horn/Umeyama Sim(3) metric scale recovery and robust point-to-plane Trimmed ICP fine registration.
   - Pre-configured profiles for **Raven + Insta360 X4** and **Eagle + Insta360 X6**.

6. **Dual-Method High-Resolution 3D Colorization**
   - **Method 1 (Reconstruction / SfM Spirula Consensus)**: Gold standard multi-view consensus projection, resolving occlusion via Z-Buffer and sharpness weighting with Vulkan compute shaders (`vulkan_colorizer.exe` with CPU fallback).
   - **Method 2 (Trajectory / Direct Rigid)**: High-speed projection directly along the SLAM trajectory with optional SfM trajectory drift correction.

7. **Interactive 3D OBB Clipping Box & Transform Gizmos**
   - Interactive 3D Oriented Bounding Box (OBB) rotation around Z (Yaw), axis pull arrows with anchored opposite face, and horizontal rotation ring.
   - Persistent sliced view: keep cuts active in the viewport while hiding box gizmos for pristine inspection.
   - Sub-volume point cloud cropping and export (PLY, PCD, LAS, XYZ).

8. **Full-Density Metric LiDAR 3DGS Seed Initialization & Instant Hardlink Export**
   - **Zero Downsampling Cap:** Feed 100% of dense metric LiDAR point clouds (tested up to 28.5M+ points!) directly into modern 3D Gaussian Splatting engines (Spirula Studio, LichtFeld Studio, PostShot, Nerfstudio).
   - **Faster Training Convergence:** Dense geometric scaffolding accelerates convergence and runs efficiently (~6.5 GiB VRAM on modern GPUs like RTX 5070 Ti) by eliminating iterative densification lag.
   - **Hybrid SfM + LiDAR Fusion:** Automatically merges SfM camera ray tracks with dense metric LiDAR geometry into a unified COLMAP model.
   - **Zero Duplicate Disk Storage:** Employs NTFS hardlinks and directory junctions to export COLMAP datasets in seconds with 0 additional disk space consumption.

9. **Automated Georeferencing, GCPs & GeoTIFF Orthophotos**
   - Automated GNSS/INSV trajectory georeferencing & Ground Control Points (GCP) alignment.
   - High-resolution GeoTIFF orthophoto export with world files (.tfw) and embedded spatial reference.
   - Native INSV GPS extraction and satellite track quality reporting (DOP, fix type, accuracy metrics).
   - Georeferenced LAZ / LAS 1.4 export with Compound CRS (SIRGAS 2000 / UTM).

10. **Multi-Camera SLAM Refinement & Geometric Pose Constraints**
    - Joint multi-camera bundle adjustment and trajectory refinement (`slam_refinement.py`, `slam_refinement_geometry.py`).
    - Enforces 6-DoF rigid extrinsic constraints and SLAM pose synthesis across auxiliary perspective and fisheye cameras (cam2, cam3, ...) for complete panoramic coverage.

11. **On-Device Vulkan RF-DETR Dynamic Person Masking**
    - GPU-accelerated person & operator detection/masking powered by native SPIR-V compute shaders (`native/vulkan_rfdetr/shaders/`).
    - Fully offline, deterministic, zero-cloud execution eliminating pedestrians and dynamic artifacts from 3D reconstruction and Gaussian splatting.

12. **Modern Fluent UI & Multilingual Localization (i18n)**
    - Microsoft WinUI 3 Fluent Design System with Mica and Acrylic materials, dark/light theme switching.
    - Runtime localization supporting **English**, **Português (Brasil)**, **Español**, **Français**, **Deutsch**, **简体中文**, and **日本語**.

---

## 🖥️ Application Screenshots

| **Unified Processing Studio** | **FAST-LIVO2 LiDAR SLAM** |
|:---:|:---:|
| <img src="docs/images/screenshot_unified_studio.png" alt="Unified LiDAR-Camera Studio" width="500"/> | <img src="docs/images/screenshot_slam.png" alt="FAST-LIVO2 SLAM Engine" width="500"/> |

| **Vulkan 3D Point Cloud Colorization** | **LiDAR-Camera Rig Calibration** |
|:---:|:---:|
| <img src="docs/images/screenshot_colorize.png" alt="Point Cloud Colorization Studio" width="500"/> | <img src="docs/images/screenshot_calibration.png" alt="Rig Calibration & Optical Modeling" width="500"/> |

<p align="center">
  <b>System Diagnostics & Hardware Acceleration Engine Doctor</b><br/>
  <img src="docs/images/screenshot_doctor.png" alt="System Diagnostics & Engine Doctor" width="800"/>
</p>

---

## 🚀 Quick Start

### Running via Python

```powershell
# Check bundled engines, GPU device, and dependencies
python lidarcamera360.py --headless doctor

# Inspect a ROS1 bag's topics and message rates
python lidarcamera360.py inspect-bag --bag D:\capture\scan.bag

# Run automated end-to-end processing (Bag + INSV -> SLAM -> Sync -> Colorize -> Deliverables)
python lidarcamera360.py workflow --bag D:\capture\scan.bag --insv D:\capture\video.insv --output D:\dataset --method direct

# Launch the desktop graphical interface
python lidarcamera360.py
```

### Running the Standalone Windows Executable (No Python Required)

Pre-compiled standalone packages and portable executables are available on [GitHub Releases](https://github.com/Everton-Braz/LidarCamera360/releases/tag/v0.2.0):
- 📥 **Direct Portable Download:** [LidarCamera360_portable.exe (v0.2.0)](https://github.com/Everton-Braz/LidarCamera360/releases/download/v0.2.0/LidarCamera360_portable.exe)

When using the portable executable or directory bundle:

```powershell
# Check bundled native SLAM and Vulkan engines
.\LidarCamera360.exe --headless doctor

# Run automated end-to-end processing directly
.\LidarCamera360.exe workflow --bag D:\capture\scan.bag --insv D:\capture\video.insv --output D:\dataset --method direct

# Or simply double-click LidarCamera360.exe to launch the GUI
```

---

## 🛠️ Installation & Building from Source

### Prerequisites
- **Windows 10 / 11 (64-bit)**
- **Python 3.10+**
- **Visual Studio 2022** (C++ Desktop Development workload with MSVC v143 and Windows SDK)
- **CMake 3.24+**
- **Vulkan SDK** (for GPU compute colorization)
- **vcpkg** (for PCL, OpenCV, Eigen3, Sophus, yaml-cpp dependencies)

### Setup Environment

```powershell
# Clone repository
git clone https://github.com/Everton-Braz/Lidar-camera-calibrator.git
cd Lidar-camera-calibrator

# Create virtual environment and install Python requirements
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Build Native SLAM and Vulkan Engines

```powershell
# Automated build using provided PowerShell tool
.\tools\build_windows.ps1 -VcpkgRoot D:\vcpkg -InstallDependencies
```

For complete build instructions and packaging steps, see [docs/BUILD.md](docs/BUILD.md).

---

## 📂 Project Structure

```text
LidarCamera360/
├── assets/                                 # Brand assets (logo, application icons)
│   ├── logo.png
│   └── app.ico
├── configs/                                # Sensor rig configurations & parameters
│   ├── rig_profile.json                    # Canonical 3DMakerPro Raven + Insta360 X4 profile
│   ├── camera.yaml                         # Camera intrinsic matrix
│   └── slam.yaml                           # FAST-LIVO2 LiDAR odometry parameters
├── docs/                                   # Architectural guides and technical specs
│   ├── BUILD.md                            # Native compilation and bundling instructions
│   ├── STANDALONE.md                       # Comprehensive operator manual
│   └── IMPLEMENTATION_VALIDATION.md        # Calibration accuracy & comparison benchmarks
├── FAST-LIVO2/                             # HKU MARS FAST-LIVO2 LiDAR-inertial-visual SLAM
├── locales/                                # Internationalization catalogs (JSON)
│   ├── en.json                             # Canonical English source
│   ├── pt-BR.json                          # Português (Brasil)
│   ├── es.json                             # Español
│   ├── fr.json                             # Français
│   ├── de.json                             # Deutsch
│   ├── zh-CN.json                          # 简体中文
│   └── ja.json                             # 日本語
├── native/                                 # C++17 standalone MSVC offline adapter
│   ├── vulkan_rfdetr/                      # Headless Vulkan RF-DETR segmentation & shaders
│   └── native_runtime.h                    # Native runtime bridge
├── raven_app/                              # Core Python application package
│   ├── auxiliary_insv.py                   # Single-stream & auxiliary INSV extraction
│   ├── auxiliary_pose_support.py           # Auxiliary camera pose estimation & validation
│   ├── branding.py                         # Application identity & version metadata
│   ├── cli.py                              # Command-line interface parser
│   ├── cloud_view.py                       # OpenGL point cloud viewer & 3D gizmos
│   ├── config.py                           # Settings and tool validation
│   ├── dataset.py                          # Dataset paths and schema normalization
│   ├── gui.py                              # Fluent UI main window and navigation
│   ├── i18n.py                             # Localization runtime translation layer
│   ├── process_runner.py                   # Async job runner with clean cancellation
│   ├── slam_refinement.py                  # Multi-camera SLAM trajectory refinement
│   ├── slam_refinement_geometry.py         # 6-DoF rigid transform geometry & interpolation
│   ├── slam_refinement_constraints.py      # Geometric & temporal optimization constraints
│   ├── third_camera.py                     # Auxiliary multi-camera config & calibration
│   ├── timelapse_calibration.py            # Binary INSV timelapse telemetry & time sync
│   ├── video.py                            # Accelerated dual/single-track INSV extraction
│   ├── vulkan_engine.py                    # Vulkan GPU compute shader bridge
│   ├── workflow.py                         # End-to-end pipeline orchestrator
│   └── views/                              # WinUI Fluent tab views
│       ├── add_source_dialog.py            # Multi-camera & auxiliary stream selector
│       ├── calibration_view.py             # Rig calibration editor
│       ├── colorize_view.py                # Standalone colorization studio
│       ├── doctor_view.py                  # System diagnostics & engine health
│       ├── inspect_view.py                 # ROS Bag inspector
│       ├── settings_view.py                # Themes, tools & language selection
│       ├── slam_view.py                    # FAST-LIVO2 mapping studio
│       └── unified_workflow_view.py        # Single-click unified workflow
├── scripts/                                # Standalone CLI processing modules
│   ├── align_colmap_to_lidar.py            # Umeyama Sim(3) metric trajectory alignment
│   ├── colorize_direct_rigid_method.py     # Trajectory direct projection
│   ├── colorize_lidar_multiview_fisheye.py # Multi-view calibrated colorizer
│   ├── colorize_sfm_spirula_method.py      # SfM consensus colorizer
│   ├── correlate_imu_gyro.py               # IMU gyro temporal synchronization
│   ├── parse_insv_telemetry.py             # INSV binary trailer telemetry parser
│   ├── pipeline_auto_calibrator_and_colorizer.py # Master unified script
│   └── run_automatic_icp_calibration.py    # Robust point-to-plane ICP fine calibration
├── tests/                                  # Automated unit and integration test suite
├── tools/                                  # Build and packaging automation
│   ├── build_windows.ps1                   # MSVC build script
│   └── package_app.py                      # Standalone PyInstaller packager
├── lidarcamera360.py                       # Primary application entry point
├── raven.py                                # Backward-compatible entry alias
├── LICENSE                                 # Dual-licensing terms
├── THIRD_PARTY_NOTICES.md                  # Detailed third-party license audit
└── requirements.txt                        # Python dependencies
```

---

## 📜 Open Source Licensing & Compliance

LidarCamera360 embraces an open, transparent licensing architecture:

- **Original Project Code**: Independently authored Python orchestration, mathematical routines, and Vulkan compute shaders (`colorizer.comp`) are released under the permissive **MIT License** in [LICENSE](LICENSE).
- **Packaged Application Suite & Desktop Binary**: To comply with upstream open-source copyleft licenses—notably **FAST-LIVO2** (GPL-2.0-only), **Spirula Studio** (GPL-3.0), and the **PyQt6 / PyQt6-Fluent-Widgets** runtime (GPL-3.0 / GPLv3)—the combined desktop application distribution is provided under the terms of the **GNU General Public License v3 (GPLv3)**.
- For complete dependency licenses, notices, and source distribution conditions, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

---

## 🤝 Contributing

Contributions, bug reports, and hardware profile additions are welcome! Please open an issue or submit a pull request on GitHub.
