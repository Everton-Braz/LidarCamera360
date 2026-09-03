---
name: lidar-camera-apriltag-calibrator
description: Comprehensive calibration workflow for rigid LiDAR-Camera rigs (3DMakerPro Raven LiDAR + Insta360 dual-fisheye camera) using AprilTags, covering static target calibration, continuous moving-scan dynamic spatial-temporal bundle adjustment, programmatic time synchronization, and point cloud colorization.
---

# LiDAR-Camera AprilTag Calibrator (Static & Dynamic)

> [!WARNING]
> **Active Development Status**:
> The calibration methods documented here successfully achieve upright 3D room colorization, correct dual-stream mapping, programmatic video-to-LiDAR time synchronization, and multi-tag bundle adjustment. **However, the calibration is still actively being improved and is not yet mathematically perfect.** Residual angular trims ($\pm 1^\circ$ to $2^\circ$) and minor parallax displacements (such as picture frame boundary offsets on walls) remain subject to ongoing fine-tuning.

---

## 1. Physical Mount Geometry & Coordinate Invariants

When calibrating a rigid LiDAR-camera rig (specifically the 3DMakerPro Raven Vanjee 722z scanner coupled with an Insta360 X3/X4 dual-fisheye camera on the handle):

### A. Sensor Tilt & True Gravity Vectors
The Vanjee 722z LiDAR optical head is mounted with an intentional **$\sim 30.3^\circ$ forward tilt** relative to the vertical handle.
- **IMU Resting Acceleration Vector**: $\vec{a} = [-0.03,\ 4.9335,\ 8.4338]\text{ m/s}^2$
- **Gravity DOWN Vector in LiDAR Frame**:
  $$\vec{d}_L = \frac{\vec{a}}{\|\vec{a}\|} = [0.0,\ 0.5049,\ 0.8632]$$
- **Gravity UP Vector in LiDAR Frame**:
  $$\vec{u}_L = -\vec{d}_L = [0.0,\ -0.5049,\ -0.8632]$$
- **LiDAR Horizontal Right Vector**:
  $$\vec{r}_L = [1.0,\ 0.0,\ 0.0] - ([1.0,\ 0.0,\ 0.0] \cdot \vec{u}_L)\vec{u}_L \approx [1.0,\ 0.0,\ 0.0]$$

### B. Swapped Dual-Fisheye Stream Mapping
Physical testing and visual CloudCompare evaluation confirmed the lens-to-stream assignment:
- **Stream 0 (`0:v:0`)**: **Right Lens** pointing along $+X_{\text{lidar}}$ (towards $\vec{r}_L$).
- **Stream 1 (`0:v:1`)**: **Left Lens** pointing along $-X_{\text{lidar}}$ (towards $-\vec{r}_L$).

### C. True Physical Lever-Arm Offset ($\vec{t}_{\text{cam}}$)
The 1/4" camera screw mount is located $18.5\text{ cm}$ above the LiDAR optical center along the handle's vertical axis ($\vec{u}_L$).
> [!CAUTION]
> **Critical Lever-Arm Trap**:
> Setting $\vec{t}_{\text{cam}} = [0.0, -0.185, 0.0]$ assumes the mount is aligned with the LiDAR's tilted optical $Y$ axis. Because the sensor is tilted forward by $30.3^\circ$, this creates a **$16.0\text{ cm}$ vertical displacement error ($Z$) and $9.2\text{ cm}$ error ($Y$)**!
> 
> **Correct Physical Offset**:
> $$\vec{t}_{\text{cam}} = h \cdot \vec{u}_L = 0.185 \cdot [0.0,\ -0.5049,\ -0.8632] = [0.0006,\ -0.0934,\ -0.1597]\text{ meters}$$

### D. Nominal Base Rotation Matrices
Using the camera optical frame convention ($+X$ = Image Right, $+Y$ = Image Down, $+Z$ = Optical Forward):
```python
# Stream 0: Right Lens (+X_L)
Z0 = right_L
Y0 = u_L - np.dot(u_L, Z0) * Z0
Y0 /= np.linalg.norm(Y0)
X0 = np.cross(Y0, Z0)
R_stream0_base = np.stack([X0, Y0, Z0], axis=0)

# Stream 1: Left Lens (-X_L)
Z1 = -right_L
Y1 = u_L - np.dot(u_L, Z1) * Z1
Y1 /= np.linalg.norm(Y1)
X1 = np.cross(Y1, Z1)
R_stream1_base = np.stack([X1, Y1, Z1], axis=0)
```

---

## 2. Programmatic Time Synchronization ($\Delta t$)

The operator starts camera video recording and LiDAR ROS bag scanning independently; hence, the time offset $\Delta t$ varies per scan (typically $3\text{s}$ to $6\text{s}$).

### Cross-Correlation Method
Instead of guessing or manual trial-and-error:
1. **LiDAR SLAM Motion Curve**:
   From FAST-LIVO2 trajectory `Raven_3DMakerPro_Scan.txt`, compute the trajectory linear velocity:
   $$v_{\text{slam}}(t) = \frac{\|\mathbf{p}(t_{k+1}) - \mathbf{p}(t_k)\|}{t_{k+1} - t_k}$$
2. **Video Optical Flow Motion Curve**:
   Extract downsampled grayscale frames at 5 FPS from video stream 0 (`320x320` or `480x480`):
   $$v_{\text{video}}(t) = \text{mean}\left(\|\text{FarnebackFlow}(I_{k-1}, I_k)\|\right)$$
3. **Normalized Cross-Correlation**:
   $$\rho(\tau) = \sum_{t} \bar{v}_{\text{video}}(t) \cdot \bar{v}_{\text{slam}}(t - \tau)$$
   The global maximum peak $\tau^* = \arg\max \rho(\tau)$ directly yields the exact time synchronization offset $\Delta t$ in seconds.

---

## 3. Workflow 1: Static Target Calibration

Use static scans when zeroing out motion blur and verifying raw spatial extrinsics:
1. Place 2 to 4 AprilTags (family `tag36h11`, $150\text{ mm}$ or $200\text{ mm}$) firmly on walls.
2. Rest scanner and camera stationary on a tripod or desk.
3. Record a short LiDAR scan (`.bag`) and capture static dual-fisheye frames (`lens1_front.jpg`, `lens2_back.jpg`).
4. Detect tag centers in 2D image using `cv2.aruco.ArucoDetector(DICT_APRILTAG_36h11)`.
5. Segment the corresponding 3D retroreflective planar cluster in the LiDAR point cloud.
6. Optimize 6-DoF transformation using non-linear least squares (`scipy.optimize.minimize` or Levenberg-Marquardt).

---

## 4. Workflow 2: Dynamic Moving Calibration (Trajectory Bundle Adjustment)

When calibrating directly from a moving handheld scan:
1. **Merge ROS Bags**:
   Merge `LIDAR_*.bag` and `IMAGE_*.bag` (remapping `/camera_front/image/compressed` to `/camera/image_color/compressed`).
2. **Execute FAST-LIVO2 Direct SLAM**:
   Generate the dense registered point cloud (`all_raw_points.pcd`) and continuous 6-DoF trajectory (`Raven_3DMakerPro_Scan.txt`).
3. **Programmatic Time Sync**:
   Run optical flow cross-correlation to find baseline offset $\Delta t_0$.
4. **Moving AprilTag Detection**:
   Scan both video streams at regular intervals (e.g. 1.0s step) detecting all visible AprilTag occurrences $(t_k, \text{stream}_k, \text{tag\_id}_k, u_k, v_k)$.
5. **3D World Tag Triangulation**:
   For each tag observed from $\ge 2$ different trajectory poses, solve the linear least-squares intersection of back-projected world rays:
   $$\min_{P_w^i} \sum_j \| (I - \mathbf{d}_j \mathbf{d}_j^T)(P_w^i - \mathbf{o}_j) \|^2$$
6. **Joint Non-Linear Bundle Adjustment**:
   Jointly minimize reprojection error across all moving tag observations:
   $$\min_{\theta_{\text{pitch}}, \theta_{\text{yaw}}, \theta_{\text{roll}}, \Delta t} \frac{1}{M} \sum_{k=1}^M \left\| \pi\left( R(\boldsymbol{\theta}) \cdot R_{w,l}^T(t_k + \Delta t)(P_w^i - \mathbf{t}_{w,l}) + \vec{t}_{\text{cam}} \right) - \begin{bmatrix} u_k \\ v_k \end{bmatrix} \right\|^2$$

---

## 5. Equidistant Fisheye Projection Model

For the Insta360 $196^\circ$ fisheye lenses ($3840 \times 3840$):
- Focal length: $f = \frac{W / 2}{\text{FOV}_{\text{rad}} / 2} = \frac{1920}{\pi \cdot 196 / 360} \approx 1122.95\text{ px}$
- Principal point: $(c_x, c_y) = (1920.0, 1920.0)$
- Projection equations:
  $$r_{xy} = \sqrt{X^2 + Y^2},\quad \theta = \text{atan2}(r_{xy}, Z)$$
  $$r_{\text{img}} = f \cdot \theta$$
  $$u = c_x + r_{\text{img}} \frac{X}{r_{xy}},\quad v = c_y + r_{\text{img}} \frac{Y}{r_{xy}}$$

---

## 6. Known Gotchas & Troubleshooting

| Symptom | Root Cause | Solution |
|---|---|---|
| **Images projected upside-down** | Optical $Y$ axis pointing down relative to gravity | Invert camera optical $Y$: set $\vec{y}_{\text{cam}} = \vec{u}_L$ (LiDAR UP). |
| **Opposite wall texture projected** | Stream index swapped with lens orientation | Use Stream 0 = Right lens ($+X_L$), Stream 1 = Left lens ($-X_L$). |
| **Texture displaced vertically by 15-20 cm** | Lever arm offset assigned to optical $Y$ instead of handle UP axis | Use $\vec{t}_{\text{cam}} = 0.185 \cdot \vec{u}_L = [0.0006, -0.0934, -0.1597]\text{ m}$. |
| **Slanted / diagonally sheared rooms** | Non-orthogonal base rotation matrix | Enforce Gram-Schmidt orthogonality aligned to IMU gravity UP vector. |
| **Motion smear on moving scan** | Time offset mismatch between video and trajectory | Run programmatic optical flow cross-correlation to find $\Delta t^*$. |
