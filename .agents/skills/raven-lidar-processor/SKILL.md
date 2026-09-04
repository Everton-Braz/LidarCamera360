---
name: raven-lidar-processor
description: >-
  Process 3DMakerPro Raven LiDAR Scanner ROS bag files with FAST-LIVO2 to generate
  dense colorized 3D point clouds (.pcd, .laz, .las), geographic UTM datasets (SIRGAS 2000),
  and full COLMAP datasets (images, cameras.txt, images.txt, points3D.txt) for Photogrammetry,
  3D Gaussian Splatting (3DGS), RealityScan, and LichtFeld Studio. Includes full environment
  setup, Sophus build patches, multi-bag merging, fisheye calibration, pinhole undistortion,
  orientation fixes, Insta360 X4 360° fisheye colorization with Spirula Studio SfM, Sim(3) ICP
  alignment, 6-DoF axis synchronization, and True North EXIF geotagging.
---

# 3DMakerPro Raven LiDAR Scanner & Insta360 Processing with FAST-LIVO2

This skill provides a complete guide, technical reference, and runbook to set up [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) and process raw datasets from the **3DMakerPro Raven LiDAR Scanner** and **Insta360 X4 360° Camera** into dense, colorized 3D point clouds (`.pcd`, `.laz`, `.las`), georeferenced GIS datasets (SIRGAS 2000 / UTM), and fully synchronized **COLMAP models** ready for **RealityScan**, **3D Gaussian Splatting (3DGS)**, **LichtFeld Studio**, and **PostShot**.

---

## 1. Overview & Official References

- **Upstream Algorithm**: [FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry](https://github.com/hku-mars/FAST-LIVO2) (HKU MaRS Lab)
- **Visual Frontend**: [rpg_vikit](https://github.com/xuankuzcr/rpg_vikit) (Customized for FAST-LIVO2 by Chunran Zheng)
- **Lie Group Library**: [Sophus (commit `a621ff`)](https://github.com/strasdat/Sophus)
- **Target Hardware**: **3DMakerPro Raven LiDAR Scanner**
  - **LiDAR**: Vanjee 722z (16-line spinning LiDAR, `XT32` compatible, topic `/vanjee_722z`).
  - **Camera**: JMK7 12MP Fisheye Camera (Kannala-Brandt model, topic `/camera_front/image_raw`).
  - **IMU**: Vanjee Internal 6-DoF IMU (topic `/vanjee_imu_packets`).
- **Auxiliary Camera**: **Insta360 X4 360° Camera** (8K Dual-Fisheye, 12MP per lens, GPS telemetry).

---

## 2. Prerequisites & Environment Setup (WSL2 Ubuntu 20.04)

### Step 2.1: WSL2 Installation on Custom Disk (e.g., Disk D:)
```powershell
# From Windows PowerShell (Admin):
wsl --install -d Ubuntu-20.04 --no-launch
wsl --export Ubuntu-20.04 D:\WSL\ubuntu2004_temp.tar
wsl --unregister Ubuntu-20.04
wsl --import Ubuntu-20.04 D:\WSL\Ubuntu-20.04 D:\WSL\ubuntu2004_temp.tar --version 2
del D:\WSL\ubuntu2004_temp.tar
```

### Step 2.2: Install ROS 1 Noetic & Build Dependencies
Run inside WSL2 (`wsl -d Ubuntu-20.04 -u root`):

```bash
apt-get update && apt-get install -y curl gnupg2 lsb-release ca-certificates build-essential cmake git

# Add ROS Noetic official repository
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list

# Install ROS Noetic & sensor packages
apt-get update && apt-get install -y \
  ros-noetic-desktop-full \
  ros-noetic-cv-bridge \
  ros-noetic-image-transport \
  ros-noetic-image-transport-plugins \
  ros-noetic-pcl-ros \
  ros-noetic-pcl-conversions \
  ros-noetic-eigen-conversions \
  ros-noetic-tf \
  ros-noetic-cmake-modules \
  libgoogle-glog-dev \
  libgflags-dev \
  libatlas-base-dev \
  libsuitesparse-dev \
  python3-catkin-tools
```

### Step 2.3: Build & Patch Sophus (`a621ff`) for GCC 9+
Sophus `a621ff` requires a patch for `std::complex` assignment compatibility on GCC 9+:

```bash
cd /tmp
git clone https://github.com/strasdat/Sophus.git
cd Sophus && git checkout a621ff

# Apply GCC 9+ std::complex fix
sed -i 's/unit_complex_\.real() = 1\.;/unit_complex_.real(1.);/g' sophus/so2.cpp
sed -i 's/unit_complex_\.imag() = 0\.;/unit_complex_.imag(0.);/g' sophus/so2.cpp

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
make install
cd /tmp && rm -rf Sophus

# Install FindSophus.cmake module globally for ROS and CMake
cat << 'EOF' > /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake
find_path(Sophus_INCLUDE_DIRS sophus/se3.h PATHS /usr/local/include /usr/include)
find_library(Sophus_LIBRARIES Sophus PATHS /usr/local/lib /usr/lib)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Sophus DEFAULT_MSG Sophus_LIBRARIES Sophus_INCLUDE_DIRS)
EOF

cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake /usr/share/cmake-3.16/Modules/FindSophus.cmake || true
```

### Step 2.4: Set up Catkin Workspace & Compile
```bash
mkdir -p /root/catkin_ws/src
cd /root/catkin_ws/src

# 1. Clone rpg_vikit frontend
git clone https://github.com/xuankuzcr/rpg_vikit.git
mkdir -p rpg_vikit/vikit_common/CMakeModules rpg_vikit/vikit_ros/CMakeModules
cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake rpg_vikit/vikit_common/CMakeModules/
cp /opt/ros/noetic/share/cmake_modules/cmake/Modules/FindSophus.cmake rpg_vikit/vikit_ros/CMakeModules/

# 2. Link FAST-LIVO2 repository
ln -s /mnt/d/APLICATIVOS/FAST-LIVO2 /root/catkin_ws/src/fast_livo

# 3. Build workspace
source /opt/ros/noetic/setup.bash
cd /root/catkin_ws
catkin_make -DCMAKE_BUILD_TYPE=Release

# Add setup to bashrc
echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc
```

---

## 3. Dataset Preparation: Merging Multi-Bag Scans

The Raven scanner saves captures in sequential bags (`IMAGE_*.bag`, `PCL_*.bag`, `OTHER_*.bag`). Merge them into a single time-synchronized file with `scripts/merge_raven_bags.py`:

```bash
python scripts/merge_raven_bags.py \
  --input-dir "DATA_TEST" \
  --output "DATA_TEST/raven_merged.bag" \
  --topics "/vanjee_722z" "/vanjee_imu_packets" "/camera_front/image_raw"
```

---

## 4. Camera Calibration & RayStudio-Compatible Pinhole Undistortion

The 12MP Raven camera uses a **Kannala-Brandt (Equidistant Fisheye)** lens model.

### 4.1. Raw Fisheye Configuration: `config/camera_raven.yaml`
```yaml
cam_model: EquidistantCamera
cam_width: 3000
cam_height: 4000
scale: 0.25
cam_fx: 1123.5155689157102
cam_fy: 1124.2448375847944
cam_cx: 1542.5973889895367
cam_cy: 2110.539590184509
k1: -0.04268357288416628
k2: 0.0024212974706268172
k3: -0.000872012773202397
k4: 0.00012160742067349935
```

### 4.2. Portrait Orientation Handling (`src/vio.cpp`)
Inside `VIOManager::processFrame`, raw $4000 \times 3000$ landscape frames must be rotated **Counter-Clockwise (CCW)** to align with the $3000 \times 4000$ portrait optical calibration:
```cpp
if (img.cols > img.rows && width < height) {
  cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
}
```
*Note: A Clockwise (CW) rotation flips the vertical axis, projecting blue sky onto the ground.*

### 4.3. Pinhole Undistortion for 3D Gaussian Splatting (3DGS)
Standard 3DGS training requires rectilinear perspective pinhole images without circular black borders.
We reverse-engineered and matched the exact undistortion parameters used by **RayStudio** (3DMakerPro's official software):

- **Target Resolution**: **$768 (\text{Width}) \times 1024 (\text{Height})$**
- **Rectified Pinhole Intrinsic Matrix ($K_{new}$)**:
  $$K_{new} = \begin{bmatrix} 457.633379 & 0.0 & 384.0 \\ 0.0 & 457.828465 & 512.0 \\ 0.0 & 0.0 & 1.0 \end{bmatrix}$$
- **Undistortion Pipeline in Code**:
  1. In `VIOManager::initializeVIO()`, `cv::fisheye::initUndistortRectifyMap` precomputes the rectification tables (`undist_map1`, `undist_map2`).
  2. In `VIOManager::dumpDataForColmap()`, `cv::remap` unwarps the native 12MP portrait image directly into standard perspective JPEG images in `Log/Colmap/images/` with **0.0% black borders**.
  3. `Log/Colmap/sparse/0/cameras.txt` is written as:
     ```
     1 PINHOLE 768 1024 457.633379 457.828465 384.000000 512.000000
     ```

---

## 5. Main Parameter Configuration (`config/raven.yaml`)

```yaml
common:
  img_topic: "/camera_front/image_raw"
  lid_topic: "/vanjee_722z"
  imu_topic: "/vanjee_imu_packets"
  img_en: 1
  lidar_en: 1
  ros_driver_bug_fix: false

extrin_calib:
  extrinsic_T: [0.0, 0.0, 0.0]
  extrinsic_R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  Rcl: [-0.99961077, 0.02221961, -0.01687019,
         0.00370246, -0.49369625, -0.86962653,
        -0.02765151, -0.86935051,  0.49342182]
  Pcl: [-0.00681037, -0.06977653, -0.00248979]

time_offset: 
  imu_time_offset: 0.0
  img_time_offset: 0.0
  exposure_time_init: 0.0

preprocess:
  point_filter_num: 1        # 1 = retain 100% of all LiDAR points (maximum density)
  filter_size_surf: 0.05     # 5cm surface voxel size
  lidar_type: 5              # XT32 / Vanjee 722z format
  scan_line: 16
  blind: 0.4                 # 40cm blind radius to filter scanner body/hand
  feature_extract_enabled: false

vio:
  max_iterations: 5
  outlier_threshold: 500
  img_point_cov: 500         # Balanced visual-inertial weighting
  patch_size: 8
  patch_pyrimid_level: 4
  normal_en: true
  raycast_en: false
  inverse_composition_en: false
  exposure_estimate_en: true
  inv_expo_cov: 0.1

imu:
  imu_en: true
  imu_int_frame: 30
  acc_cov: 0.2               # Tighter accel covariance for anti-drift
  gyr_cov: 0.05              # Tighter gyro covariance for anti-drift
  b_acc_cov: 0.00001
  b_gyr_cov: 0.00001

lio:
  max_iterations: 5
  dept_err: 0.02
  beam_err: 0.05
  min_eigen_value: 0.005     # Stricter plane threshold for sharp geometry
  voxel_size: 0.3            # 30cm voxel for tight geometric detail
  max_layer: 2
  max_points_num: 100        # Up to 100 points per voxel octree for planar fit
  layer_init_num: [5, 5, 5, 5, 5]

uav:
  imu_rate_odom: false
  gravity_align_en: true     # Active gravity alignment against tilt

publish:
  dense_map_en: true
  pub_scan_num: 1
  blind_rgb_points: 0.0

pcd_save:
  pcd_save_en: true
  type: 0                    # 0: World Frame
  colmap_output_en: true     # Generates COLMAP dataset
  filter_size_pcd: 0.02      # 2cm downsampling for filtered cloud
  interval: -1               # Save on node exit
```

---

## 6. Execution Runbook

### 6.1. Primary Execution (Full Dataset - Recommended):
Run the continuous full dataset against `DATA_TEST/raven_merged.bag`:

```bash
wsl.exe -d Ubuntu-20.04 -u root bash -c \
  "bash /mnt/d/APLICATIVOS/FAST-LIVO2/scripts/run_raven_dataset.sh 1.5 /mnt/d/APLICATIVOS/FAST-LIVO2/DATA_TEST/raven_merged.bag"
```

### 6.2. Clean Shutdown Protocol:
To avoid corrupted PCD headers:
1. Send `SIGINT` directly to `fastlivo_mapping`:
   ```bash
   NODE_PID=$(pgrep -f fastlivo_mapping)
   kill -INT $NODE_PID
   while kill -0 $NODE_PID 2>/dev/null; do sleep 3; done
   ```
2. Kill `roslaunch` only after `fastlivo_mapping` finishes writing to disk.

---

## 7. Optional Workflow: Chunk Partitioning (On User Request)

> [!WARNING]
> **Why Slicing Bags Before Running SLAM Fails**:
> FAST-LIVO2 requires the scanner to be **stationary** during the first ~30 frames ($0.15\text{s} - 0.3\text{s}$) to estimate the gravity vector and gyro biases.
> If a sub-bag is sliced mid-motion, the estimator assumes zero initial velocity and wrong gravity orientation, causing the trajectory to bend into a massive arc (e.g. 20,000m scale).
> **Correct Approach**: Run continuous SLAM on the full bag from $t=0$, and then partition the output points and images into chunks by timestamp.

```bash
# 1. Partition COLMAP images and trajectory poses
python scripts/partition_full_run_into_chunks.py

# 2. Slice binary PCD point clouds by time interval
python scripts/slice_pcd_chunks.py
```

---

## 8. Insta360 X4 360° Dual-Fisheye Calibration & Production Colorization

The calibrated sensor-to-sensor extrinsic transformation between the **3DMakerPro Raven LiDAR Scanner** and the **Insta360 X4 360° Camera** is stored in the centralized production configuration file:
👉 [`calibracao_rigida_raven_insta360.json`](file:///c:/Users/User/Documents/APLICATIVOS/Lidar-Camera-calibrator/calibracao_rigida_raven_insta360.json)

### 8.1. Confirmed Physical & Optical Invariants:
1. **Physical Lever Arm**: $\|\mathbf{t}_{LC}\| = \mathbf{18.50\text{ cm}}$ (coincides with the physical mount rod length to sub-millimeter precision).
2. **Optical Model**: **Thin Prism Fisheye** ($3840 \times 3840$):
   - Front Camera (`cam0`): $f_x = 1080.1874\text{ px}$, $f_y = 1079.9874\text{ px}$, $c_x = 1920.0$, $c_y = 1920.0$.
   - Rear Camera (`cam1`): $f_x = 1079.0045\text{ px}$, $f_y = 1078.4408\text{ px}$, $c_x = 1920.0$, $c_y = 1920.0$.
   - Useful aperture radius: $r_{\text{px}} < 1650.0\text{ px}$.
3. **Rigid Extrinsic Matrix ($T_{\text{LiDAR}\leftarrow\text{Cam0}}$)**:
   ```python
   T_LC0 = np.array([
       [-0.04816735, -0.00245086,  0.99883627,  0.01143877],
       [-0.85190651, -0.52197769, -0.04236267,  0.14067179],
       [ 0.52147408, -0.85295562,  0.02305438,  0.11961192],
       [ 0.0,         0.0,         0.0,         1.0       ]
   ])
   ```

---

### 8.2. Production Workflow: Dual-Path Colorization

Depending on operational requirements, two workflows are available:

#### Path A: Direct Rigid Colorization (Ultra-Fast — No SfM Needed)
* **When to use**: Whenever the camera remains mounted on the **same rigid rod/bracket** on the Raven scanner.
* **How it works**:
  1. FAST-LIVO2 generates the raw cloud (`all_raw_points.pcd`) and trajectory (`Raven_3DMakerPro_Scan.txt`).
  2. The video frames are extracted from `0:v:0` (`cam0/`) and `0:v:1` (`cam1/`).
  3. Programmatic time offset $\Delta t$ is synchronized via optical flow cross-correlation.
  4. The camera pose at video time $t$ is computed directly as:
     $$T_{\text{world}\leftarrow\text{cam}}(t) = T_{\text{world}\leftarrow\text{lidar}}(t - \Delta t) \cdot T_{\text{LiDAR}\leftarrow\text{cam}}$$
  5. The colorization engine projects all LiDAR points into the nearest front/rear camera frames with $960 \times 960$ Z-buffer occlusion culling, finishing in seconds.

#### Path B: SfM-Guided Sim(3) + Multimodal ICP (Autonomous Precision Guarantee)
* **When to use**: If the camera was detached, mechanical angle changed, or absolute 2-centimeter autonomous closure is required.
* **How it works**:
  1. Extract circular fisheye keyframes (`cam0/`, `cam1/` at $3840 \times 3840$) from the `.insv` video file.
  2. Run Structure from Motion (SfM) in [Spirula Studio](https://github.com/harry7557558/spirula-studio) using the `THIN_PRISM_FISHEYE` model.
  3. Run trajectory similarity alignment:
     ```bash
     python align_colmap_to_lidar.py
     ```
     Recovers the exact metric scale ($s = 4.8082$) and synchronizes timestamps ($\Delta t = 5.52\text{s}$) with $\text{RMSE} = 6.92\text{ cm}$.
  4. Run surface-level multimodal ICP fine registration:
     ```bash
     python run_automatic_icp_calibration.py
     ```
     Refines registration to **$\text{RMSE} = 3.55\text{ cm}$** and **$\text{median error} = 2.56\text{ cm}$** with $80.1\%$ inliers.
  5. Run high-performance multi-view colorization:
     ```bash
     python colorize_lidar_multiview_fisheye.py
     ```
     Paints **100.00% of the 2,336,721 LiDAR points** in **$\sim 81\text{ seconds}$** using 190 dual-fisheye keyframes ($3840 \times 3840$) with raster Z-buffer occlusion prevention.

---

## 9. Georeferencing Pipeline (UTM / SIRGAS 2000, Compound CRS & Stitch3D)

Automate the complete GIS georeferencing flow using [`georeference.py`](file:///d:/APLICATIVOS/FAST-LIVO2/georeference.py):

```bash
python georeference.py pipeline \
  --insv "SMALL-DATASET-TEST/videos/VID_20260821_105436_00_269.insv" \
  --colmap-dir "Log/small_test/Colmap_Metric_Fisheye" \
  --pcd-in "Log/small_test/pcd/colorized_insta360_fisheye_calibrated.pcd" \
  --output-dir "Log/small_test/Georeferenced" \
  --crs "EPSG:31984" \
  --compound-crs "EPSG:31984+3855" \
  --yaw-deg 77.18
```

### 9.1. Key Artifacts Generated:
1. **Compressed LAS 1.4 / LAZ (`colorized_lidar_georeferenced_utm.laz`)**:
   - Stores Compound CRS metadata via standard WKT VLR headers (`SIRGAS 2000 / UTM zone 24S` + `EGM2008 height`).
   - 100% compatible with **CloudCompare**, **QGIS**, and online cloud viewers like **Stitch3D.io** (resolves the *"Missing/Mismatching CRS"* warning).
2. **True North Heading Alignment**:
   - Automatically computes or manual-locks the True North azimuth ($\text{Yaw} = 77.18^\circ$), orienting the building footprint directly inside cadastral street boundaries.
3. **EXIF-Geotagged Image Dataset (`images_geotagged/`)**:
   - Embeds standard upright EXIF GPS coordinates (`GPSLatitude`, `GPSLongitude`, `GPSAltitude`, `GPSImgDirection`) into all JPEG frames.

---

## 10. Photogrammetry & 3DGS COLMAP Dataset Preparation (RealityScan, LichtFeld Studio, PostShot)

To prepare a COLMAP dataset from 360° dual-fisheye frames that opens seamlessly in **RealityScan**, **LichtFeld Studio**, and **3D Gaussian Splatting**:

### 10.1. Dual 3DGS Pipeline Exporters
1. **Raw Circular Dual-Fisheye 3DGS (`Colmap_Fisheye_3DGS/`)**:
   - Preserves 100% native circular dual-fisheye frames (no rectilinear stretching, unwarping artifacts, or FOV loss).
   - Generates calibrated `OPENCV_FISHEYE` / `THIN_PRISM_FISHEYE` camera parameters with 1:1 metric-scale camera poses in `sparse/0/` (both binary `.bin` and `.txt`).
   - Run `scripts/export_raw_fisheye_3dgs_colmap.py`.
2. **Full-Frame Square Rectilinear Undistortion ($1536 \times 1536$) (`Colmap_Undistorted_3DGS/`)**:
   - Unwarps circular fisheye lenses into square rectilinear perspective pinholes ($f=870.0\text{px}$, $1536 \times 1536$) with **0% black borders**:
   - Run `scripts/export_full_square_undistorted_colmap.py`.

### 10.2. Visual Metric Alignment Quality Reporting
Run `scripts/generate_metric_alignment_report.py` to generate:
- **`metric_alignment_report.html`**: Interactive report displaying quantitative error statistics (median inlier residual: $1.94\text{ cm}$, $84.3\%$ within $5\text{ cm}$, $92.6\%$ within $10\text{ cm}$, scale factor: $s = 5.367388$).
- **LiDAR-to-Camera Projection Overlays**: Renders multi-frame depth false-color projection maps across the full trajectory.

### 10.3. Synchronized 6-DoF Axis Rotation Standards (RealityScan Golden Dataset)
When exporting to **RealityScan**, the dataset must have an upright ground plane ($Z = \text{up}$), forward-facing cameras, self-contained images, and True North EXIF GPS tags:
- **Base Coordinate Transform**:
  $$R_{\text{base}} = \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.0 & 0.0 & 1.0 \\ 0.0 & -1.0 & 0.0 \end{bmatrix}, \quad R_{\text{pts}} = R_z(180^\circ) \cdot R_{\text{base}}$$
- **Camera Pose Transformation**:
  $$R_{\text{cam}} = R_x(0.0^\circ) \cdot R_{\text{pts}} = R_{\text{pts}}$$
  $$C_{\text{new}} = R_{\text{cam}} \cdot C_{\text{world}}, \quad R_{\text{wc,new}} = R_{\text{cam}} \cdot R_{\text{wc}}, \quad R_{\text{cw,new}} = R_{\text{wc,new}}^T, \quad t_{\text{cw,new}} = -R_{\text{cw,new}} \cdot C_{\text{new}}$$
- **True North EXIF GPS Geotagging**:
  $$R_{\text{world}\to\text{geo}} = R_{\text{geo}} \cdot R_{\text{pts}}^T, \quad t_{\text{geo}}$$
- **Pipeline Scripts**:
  ```bash
  python scripts/generate_perfect_realityscan_models.py
  python scripts/apply_camera_rotation_minus90.py 0.0
  python scripts/align_exif_gps_to_true_north_laz.py
  ```

---

## 11. Output File Structure

```
Log/
├── pcd/
│   ├── all_raw_points.pcd                         # Raw dense SLAM point cloud (2.33M points)
│   └── all_downsampled_points.pcd                 # Downsampled SLAM point cloud
├── <dataset_name>/
│   ├── pcd/
│   │   ├── 09_NUVEM_LIDAR_COLORIDA_MULTIVIEW_FISHEYE.ply # 100% Colorized binary PLY (35.05 MB)
│   │   ├── 09_NUVEM_LIDAR_COLORIDA_MULTIVIEW_FISHEYE.pcd # 100% Colorized binary PCD (37.39 MB)
│   │   └── colorized_insta360_fisheye_calibrated.pcd     # Standalone colorized point cloud
│   ├── Georeferenced/
│   │   ├── colorized_lidar_georeferenced_utm.laz         # Compound CRS LAZ (EPSG:31984+3855)
│   │   ├── colorized_lidar_georeferenced_utm.las         # Standard LAS 1.4 WKT Record 2112
│   │   ├── colorized_lidar_georeferenced_utm.prj         # Projection sidecar
│   │   ├── local_to_geographic.json                      # Transformation affine matrix & metadata
│   │   └── gps_trajectory.gpx                            # GPS trajectory track
│   ├── Colmap_Metric_Fisheye/                            # Metric-scale dual-fisheye dataset
│   │   ├── metric_alignment_report.html                  # Visual alignment quality report & overlays
│   │   └── sparse/0/ (cameras.bin, images.bin, points3D.bin)
│   ├── Colmap_Fisheye_3DGS/                              # Raw Circular Fisheye 3DGS dataset
│   │   ├── images/ (cam0/, cam1/ raw fisheye frames)
│   │   └── sparse/0/ (cameras.bin, images.bin, points3D.bin)
│   ├── Colmap_Undistorted_3DGS/                          # Rectilinear 1536x1536 Undistorted 3DGS
│   │   ├── images/ (cam0/, cam1/ 1536x1536 square pinholes)
│   │   └── sparse/0/ (cameras.bin, images.bin, points3D.bin)
│   ├── RealityScan_Dataset/                              # Production RealityScan Dataset (1-Click Import)
│   │   ├── images/ (cam0/, cam1/ with True North EXIF GPS)
│   │   └── sparse/ (0/cameras.bin, images.bin, points3D.bin, cameras.txt, images.txt, points3D.txt)
│   ├── calibracao_rigida_raven_insta360.json             # Central 6-DoF Rigid Extrinsic + Thin Prism JSON
│   └── calibracao_automatica_icp_resultado.json          # Multi-modal ICP Fine Registration Report
└── result/
    └── Raven_3DMakerPro_Scan.txt                         # 6-DoF trajectory in TUM format
```

---

## 12. Troubleshooting Reference

| Problem | Root Cause | Solution |
| :--- | :--- | :--- |
| **Door aligns on one wall, but window on opposite wall is displaced** | Nominal focal length ($1122.5\text{ px}$) creates severe lateral radial scale error | Use the calibrated **Thin Prism** model ($f_x=1080.19\text{ px}$, $f_y=1079.99\text{ px}$) from [`calibracao_rigida_raven_insta360.json`](file:///c:/Users/User/Documents/APLICATIVOS/Lidar-Camera-calibrator/calibracao_rigida_raven_insta360.json). |
| **Foreground wall color bleeding into points behind it** | Lack of occlusion culling in multi-view projection | Enable $960 \times 960$ raster Z-buffer depth test (`d_point <= d_buffer * 1.08 + 0.15`). |
| **Point cloud geometry has double-edges / drift** | Coarse voxel size, lack of gravity alignment, or noisy planar fitting. | Set `uav/gravity_align_en: true`, `lio/voxel_size: 0.3`, `lio/min_eigen_value: 0.005`, and `lio/max_points_num: 100`. |
| **Trajectory bends into an arc when running a sliced chunk** | Bag was sliced mid-motion; IMU initialization assumed stationary initial conditions and calculated wrong biases. | Run continuous SLAM from $t=0$, and use `partition_full_run_into_chunks.py` / `slice_pcd_chunks.py` to extract time slices. |
| **Ground plane tilted in long runs** | Gyro/Accel bias drift over 30+ minutes. | Enable `uav/gravity_align_en: true` and tighten IMU covariances (`acc_cov: 0.2`, `gyr_cov: 0.05`). |
| **Black circular borders on COLMAP images** | Fisheye lens projection not unwarped to rectilinear pinhole format. | Use `export_full_square_undistorted_colmap.py` to generate $1536 \times 1536$ square pinholes with 0% black borders. |
| **Stitch3D: "Missing and mismatching CRS"** | LAS file lacks standard Compound 3D CRS WKT VLR header. | Use `georeference.py pipeline` with `--compound-crs "EPSG:31984+3855"` to write standard EPSG WKT VLR headers. |
| **RealityScan model tilted 90° sideways** | World-to-camera matrix $R_{\text{cw}}$ transformed incorrectly or coordinate frame mismatch. | Apply synchronized 6-DoF rigid rotation with `fix_realityscan_colmap_rotation_and_import.py` ($\det(R_{\text{fix}}) = +1.0$). |
| **RealityScan repeatedly prompts to locate images** | Image paths in `images.txt` do not match relative folder hierarchy. | Use self-contained layout with `images/cam0/` and `images/cam1/` located beside `sparse/`. |
| **RealityScan 3D bounding box stretched 200m+ tall** | Outlier sky tie-points created during SfM unconstrained bundle adjustment. | Filter points with $Z \in [-3\text{m}, +15\text{m}]$ using `filter_clean_colmap_points()`. |
| **Map trajectory misaligned with satellite street** | EXIF GPS tags lack True North yaw calibration. | Embed calibrated True North heading ($\text{Yaw} = 77.18^\circ$) using `align_exif_gps_to_true_north_laz.py`. |
| **`all_downsampled_points.pcd` has same size as `all_raw_points.pcd`** | Mapping points are pre-filtered at $5\text{cm}$ (`filter_size_surf: 0.05`). Post-save voxel filter is $2\text{cm}$ (`filter_size_pcd: 0.02`), retaining all points. | Increase `filter_size_pcd` to $>0.05$ (e.g. $0.10$ for 10cm decimation) in `config/raven.yaml` if a sparser cloud is desired. |

