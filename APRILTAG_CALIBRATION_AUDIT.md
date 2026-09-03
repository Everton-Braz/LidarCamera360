# AprilTag LiDAR–Camera Calibration Audit

## 1. Codebase Overview
- **Primary Language**: Python 3.9+ (Windows & Linux compatible).
- **Core Modules**:
  - `core.py`: Mathematical foundations, spherical projections, IMU gravity alignment, ROS bag parsing (`rosbags`), edge extraction.
  - `solver.py`: Optimization pipeline, cross-validation across multiple static captures.
  - `colorize.py`: LiDAR point cloud colorization exporter (`.ply`, `.pcd`, `.las`, `.laz`) utilizing estimated extrinsic matrices.
  - `gui.py`: Graphical user interface built on `customtkinter`.
  - `generate_apriltags.py`: High-resolution AprilTag & dual-modality calibration board generator.
  - `apriltag_calib.py`: Independent AprilTag extrinsic calibration engine.
- **Fiducial Libraries**: OpenCV 4.12+ `cv2.aruco` (with `DICT_APRILTAG_36h11`, `ArucoDetector`, and subpixel corner refinement).

---

## 2. Data Interfaces

### LiDAR Data
- **Sources**: ROS 1 `.bag` files (`sensor_msgs/PointCloud2`, Livox `CustomMsg`, 3DMakerPro Raven `/vanjee_722z`), `.pcd`, `.ply`, and pre-parsed `.calib_cache.npz` files.
- **Fields**: $(X, Y, Z)$ coordinates in meters, plus `intensity` / `reflectivity` (0–255).
- **IMU**: Linear acceleration and angular velocity for gravity-vector reference.

### Camera Data
- **Format**: 2:1 Equirectangular 360° video (`.mp4`) from Insta360 X4 / X3 / ONE RS without in-camera stabilization or horizon lock.
- **Resolution**: Typically $7680 \times 3840$ or $5760 \times 2880$ pixels.
- **Spherical Mapping**:
  $$\text{lon} = \left(\frac{u}{W} - 0.5\right) 2\pi, \quad \text{lat} = \left(0.5 - \frac{v}{H}\right) \pi$$
  $$\mathbf{d} = \begin{bmatrix} \cos(\text{lat})\sin(\text{lon}) \\ -\sin(\text{lat}) \\ \cos(\text{lat})\cos(\text{lon}) \end{bmatrix}$$
  $$\begin{bmatrix} u \\ v \end{bmatrix} = \begin{bmatrix} \left(\frac{\text{atan2}(X, Z)}{2\pi} + 0.5\right) W \\ \left(0.5 - \frac{\text{asin}(-Y / r)}{\pi}\right) H \end{bmatrix}$$

---

## 3. Coordinate Frames & Conventions
- **Camera Frame ($\mathcal{F}_{\text{cam}}$)**: OpenCV convention ($X$ right, $Y$ down, $Z$ forward).
- **LiDAR Frame ($\mathcal{F}_{\text{lidar}}$)**: Sensor body frame.
- **Rig Frame**: Physical intuitive axes: `up` (along gravity), `back` (against camera optical axis), `right` (along camera right).
- **Extrinsic Transformation**:
  $$\mathbf{X}_{\text{cam}} = \mathbf{R} \mathbf{X}_{\text{lidar}} + \mathbf{t} \quad \Longleftrightarrow \quad \mathbf{T}_{\text{cam}\leftarrow\text{lidar}} = \begin{bmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}$$
  $$\mathbf{C}_{\text{cam\_in\_lidar}} = -\mathbf{R}^\top \mathbf{t}$$

---

## 4. Calibration Target & Strategy

### Target Design (Grammatikopoulos et al. 2022)
- **Central Marker**: AprilTag `tag36h11` of known size (150 mm / 0.150 m).
- **Dual-Modality Crosshairs**: Orthogonal high-reflectivity retroreflective stripes intersecting at the exact tag center.
- **Detection in Camera**: Subpixel detection of the 4 tag corners; tag center computed as average of 4 corners and reconstructed as 3D unit ray $\mathbf{d}_i$.
- **Detection in LiDAR**: Planar surface segmentation + intensity peak / line intersection at target center yielding $\mathbf{P}_{\text{lidar}, i}$.

### Solvers
1. **3D–2D Spherical PnP**: Minimizes angular reprojection error $\arccos(\mathbf{d}_i \cdot \frac{\mathbf{R}\mathbf{P}_i + \mathbf{t}}{\|\mathbf{R}\mathbf{P}_i + \mathbf{t}\|})$ and pixel distance $\|\pi(\mathbf{R}\mathbf{P}_i + \mathbf{t}) - \mathbf{p}_i\|$.
2. **3D–3D Closed-Form (Umeyama / Kabsch)**: When full 6D tag pose $\mathbf{P}_{\text{cam}, i}$ is estimated from tag size.
3. **Multi-Target Joint Optimization**: Solves over multiple tags and static viewpoints simultaneously with RANSAC outlier rejection.

---

## 5. Output Integration
- The calibration result is exported to `extrinsics_determined.json` and `.yaml` matching the exact format required by `colorize.py` and downstream 3D Gaussian Splatting / COLMAP pipelines.
