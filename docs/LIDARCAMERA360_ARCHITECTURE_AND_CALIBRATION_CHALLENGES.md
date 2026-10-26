# LidarCamera360: System Architecture, Calibration Pipelines, and Open Multi-Camera Multi-Modal Challenges

**Document ID:** `SPEC-LC360-ARCH-2026-V4`  
**Author:** Gemini 4 Argon (Principal Systems Architect & Computer Vision Mathematician)  
**Target Audience:** Autonomous Research Agents, Principal Computer Vision Engineers, SLAM & Calibration Architects  
**Status:** Definitive Technical Specification & Mathematical Handoff

---

## 1. Executive Summary & Problem Formulation

### 1.1 System Purpose
The **LidarCamera360** system is a multi-modal 3D reconstruction, metric colorization, and neural rendering initialization pipeline. It fuses dense, metric 3D geometry and 6-DoF continuous trajectories from a handheld/mobile LiDAR SLAM scanner (**3DMakerPro Raven**) with ultra-high-resolution 8K omnidirectional video captured by a dual-fisheye camera (**Insta360 X4**). 

The primary downstream objective of the pipeline is twofold:
1. **Metric Point Cloud Colorization**: Generating dense, photorealistic, radiometrically consistent RGB point clouds ($\mathcal{P}_W = \{(\mathbf{P}_{W,k}, \mathbf{c}_k)\}_{k=1}^N$) free of parallax ghosting or seam boundaries.
2. **Neural Radiance & Gaussian Splatting Initialization**: Exporting calibrated camera intrinsics, spatio-temporally aligned 6-DoF camera extrinsics, undistorted/fisheye image frames, and metric structure-from-motion (SfM) priors in standard COLMAP format (`cameras.bin`, `images.bin`, `points3D.bin`) to train **3D Gaussian Splatting (3DGS)** and **Neural Radiance Fields (NeRF)** without requiring brittle, computationally prohibitive visual-only SfM on $360^\circ$ imagery.

### 1.2 Mathematical Problem Formulation
Given a set of 3D points $\mathcal{P}_W = \{\mathbf{P}_{W,k} \in \mathbb{R}^3\}_{k=1}^N$ registered in a global metric World Frame $\mathcal{W}$, a discrete time-series of LiDAR SLAM poses $\mathcal{T}_{\text{slam}} = \{(t_m, \mathbf{T}_{WL}(t_m) \in \mathrm{SE}(3))\}_{m=1}^M$, and two synchronous video streams $\mathcal{I}_0 = \{I_0(\tau_j)\}_{j=1}^F$ and $\mathcal{I}_1 = \{I_1(\tau_j)\}_{j=1}^F$ corresponding to the front and rear hemispherical fisheye lenses clocked at video timestamps $\tau_j$, the system must estimate the optimal parameter vector $\boldsymbol{\Phi}^*$:

$$\boldsymbol{\Phi}^* = \left\{ \mathbf{T}_{LC_0}, \mathbf{T}_{C_0 C_1}, \Theta_{\text{intr},0}, \Theta_{\text{intr},1}, \delta t, \alpha_t, \Delta \mathbf{T}_{WL}(t) \right\}$$

such that the projection of any surface point $\mathbf{P}_{W,k}$ visible in camera $i \in \{0, 1\}$ at video timestamp $\tau_j$ maps to its true corresponding pixel coordinate $\mathbf{u}_{i,j,k} \in \mathbb{R}^2$ with sub-pixel accuracy across the entire $4\pi$ steradian field of view:

$$\mathbf{u}_{i,j,k} = \pi_i \left( \mathbf{T}_{LC_i}^{-1} \cdot \left(\Delta \mathbf{T}_{WL}(t(\tau_j)) \cdot \mathbf{T}_{WL}(t(\tau_j))\right)^{-1} \cdot \bar{\mathbf{P}}_{W,k} \,;\, \Theta_{\text{intr},i} \right)$$

where $\bar{\mathbf{P}}_{W,k} = [\mathbf{P}_{W,k}^\top, 1]^\top$ denotes homogeneous coordinates, $t(\tau_j)$ maps the video clock to the SLAM clock, and $\Delta \mathbf{T}_{WL}(t) \in \mathrm{SE}(3)$ accounts for non-rigid local SLAM trajectory drift.

### 1.3 The Core Duality: Controlled Domestic vs. Large-Scale Commercial Datasets
A critical diagnostic observation in the `LidarCamera360` codebase is the stark operational bifurcation between small-scale indoor captures and large-scale commercial captures:

| Characteristic | Controlled Domestic Dataset (`20261022home`) | Complex Commercial Dataset (`HarmonyMall`) |
| :--- | :--- | :--- |
| **Trajectory Scale & Topology** | Short path ($< 50\text{ m}$), bounded rooms, frequent loop closures within $3\text{--}5\text{ m}$. | Extended path ($\approx 688\text{ m}$), wide multi-story atrium loops, long corridors. |
| **Scene Geometry & Depth** | Walls and furniture at close range ($1.0\text{ m} \le d \le 3.5\text{ m}$); abundant high-frequency 3D corners. | Expansive flat polished floors, high ceilings, distant walls ($5.0\text{ m} \le d \le 40.0\text{ m}$), repetitive pillars. |
| **Ray Incidence Angles** | Predominantly near-normal incidence ($\alpha \in [0^\circ, 45^\circ]$) on vertical walls and close furnishings. | Severe grazing angles ($\alpha \in [65^\circ, 82^\circ]$) on ground-plane markings (parking lines, floor tiles) at $3\text{--}8\text{ m}$. |
| **Dynamic Obstructions** | Strictly static environment; zero moving agents. | High density of dynamic pedestrians, escalators, and moving vehicles corrupting visual and LiDAR correspondence. |
| **Operator Kinematics** | Slow, deliberate translation ($\le 0.5\text{ m/s}$) and smooth panning. | Brisk walking ($\approx 1.4\text{ m/s}$) interspersed with rapid rotational turns ($\|\boldsymbol{\omega}\| > 90^\circ\text{/s}$). |
| **Calibration Outcome** | **Seamless convergence.** Sub-pixel colorization, crisp textures, clean 3DGS synthesis. | **Systemic degradation.** Double-vision on floor lines, shearing across the Cam0/Cam1 seam, ghosting on pillars. |

In `20261022home`, close proximity masks minor rotational calibration errors ($\Delta \theta \approx 0.2^\circ$ at $2\text{ m}$ normal incidence yields a negligible $7\text{ mm}$ spatial shift), and short recording durations hide clock skew and SLAM drift. Conversely, in `HarmonyMall`, those exact same residual errors ($\Delta \theta \approx 0.2^\circ$, inter-camera baseline error $\approx 1\text{ mm}$, local SLAM drift $\approx 3\text{ cm}$, and temporal jitter $\approx 15\text{ ms}$) amplify nonlinearly through grazing-angle geometry and equatorial lens distortion, causing catastrophic visual artifacts. 

Ad-hoc per-dataset manual tuning of extrinsics or time offsets is strictly prohibited; the system requires a unified, mathematically rigorous self-calibration architecture capable of resolving `HarmonyMall`-class datasets autonomously.

---

## 2. Hardware Architecture & Sensor Kinematics

### 2.1 Sensor Specifications
1. **LiDAR SLAM Unit (3DMakerPro Raven)**:
   - **Sensing Modality**: High-density 3D laser rangefinder coupled with an internal Inertial Measurement Unit (IMU).
   - **Trajectory Output**: Discrete 6-DoF poses emitted at $f_{\text{slam}} = 10\text{ Hz}$ ($\Delta t_{\text{slam}} = 0.1\text{ s}$). Each record consists of a monotonic hardware timestamp $t_{\text{slam}} \in \mathbb{R}$, a 3D translation vector $\mathbf{p}_{WL}(t_{\text{slam}}) \in \mathbb{R}^3$, and a unit orientation quaternion $\mathbf{q}_{WL}(t_{\text{slam}}) \in \mathbb{H}_1$.
   - **Point Cloud Output**: Aggregated or frame-indexed metric 3D point clouds with calibrated surface reflectivity/intensity $I_{\text{lidar}} \in [0, 255]$ and pre-computed surface normals $\mathbf{n}_W \in \mathbb{S}^2$.

2. **360 Dual-Fisheye Camera (Insta360 X4)**:
   - **Sensor Resolution**: $7680 \times 3840$ composite dual-fisheye equirectangular-container video stream at $f_{\text{vid}} = 29.97\text{ Hz}$ (or $24/30\text{ Hz}$ depending on mode).
   - **Optical Partitioning**: The raw $7680 \times 3840$ frame is partitioned horizontally into two independent square image planes of resolution $W \times H = 3840 \times 3840$:
     - **Camera 0 ($\mathcal{C}_0$, Front Lens)**: Occupies pixel columns $[0, 3839]$ (or extracted as stream 0).
     - **Camera 1 ($\mathcal{C}_1$, Rear Lens)**: Occupies pixel columns $[3840, 7679]$ (or extracted as stream 1).
   - **Field of View (FOV)**: Each lens possesses a hemispherical FOV of $\Theta_{\text{fov}} \approx 200^\circ$, creating an annular **equatorial overlap zone** of $\Delta \Theta_{\text{overlap}} \approx 20^\circ$ ($\theta \in [80^\circ, 100^\circ]$ from each optical axis) where scene points are simultaneously imaged by both $\mathcal{C}_0$ and $\mathcal{C}_1$.

### 2.2 Coordinate Frame Conventions & Kinematic Graph
To prevent sign inversions and handedness bugs, the system enforces strict right-handed Cartesian coordinate frames across all modules:

```
                      +Z_C0 (Optical Axis Front)
                         ^
                         |     [Insta360 X4]
      +Y_C1 (Down)       |    /
            \            |   /
  +Z_C1 <----+-----------+--+----> +X_C0 (Right)
 (Rear)     C1          C0  |
             |<-- d_base -->|v +Y_C0 (Down)
                    |
                    |  T_{L C0} (Rigid Lever-Arm ~ 12-18 cm)
                    |
                    +-----> +Y_L / +X_L (LiDAR Body Frame L(t))
                   /
                  v +Z_L
                 /
                /  T_{W L}(t) (6-DoF SLAM Trajectory @ 10Hz)
               /
        +-----+-----+
        | World (W) | (Metric Gravity-Aligned Frame, +Z_W Up)
        +-----------+
```

* **World Frame ($\mathcal{W}$)**: Fixed metric inertial reference frame established at SLAM initialization ($t_{\text{slam}} = 0$). Gravity-aligned such that $+Z_W$ points vertically upward (zenith), and $X_W Y_W$ defines the horizontal ground plane.
* **Instantaneous LiDAR Frame ($\mathcal{L}(t)$)**: Moving body frame anchored to the optical center of the 3DMakerPro Raven at time $t$. The pose of $\mathcal{L}(t)$ with respect to $\mathcal{W}$ is $\mathbf{T}_{WL}(t) \in \mathrm{SE}(3)$, mapping points from $\mathcal{L}(t)$ to $\mathcal{W}$:
  $$\mathbf{P}_W = \mathbf{R}_{WL}(t)\mathbf{P}_L(t) + \mathbf{t}_{WL}(t)$$
* **Camera Frames ($\mathcal{C}_0$ and $\mathcal{C}_1$)**: Standard OpenCV / COLMAP **Right-Down-Forward (RDF)** convention:
  - $+X_{C_i}$ points right along the image sensor rows ($u$ increases).
  - $+Y_{C_i}$ points downward along the image sensor columns ($v$ increases).
  - $+Z_{C_i}$ points forward along the lens optical axis out into the scene.
  - Because $\mathcal{C}_0$ and $\mathcal{C}_1$ are mounted back-to-back, their nominal relative rotation $\mathbf{R}_{C_0 C_1}$ is approximately a $180^\circ$ rotation about the $Y_{C_0}$ (or $X_{C_0}$, depending on sensor readout orientation) axis:
    $$\mathbf{R}_{C_0 C_1}^{\text{nom}} \approx \begin{bmatrix} -1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad \mathbf{t}_{C_0 C_1}^{\text{nom}} \approx \begin{bmatrix} 0 \\ 0 \\ -d_{\text{base}} \end{bmatrix}$$
    where $d_{\text{base}} \approx 0.040\text{ m}$ ($40\text{ mm}$) is the physical optical center separation between the two fisheye lenses.

### 2.3 Rigid Body Kinematics in $\mathrm{SE}(3)$
Throughout this specification, a rigid transformation matrix $\mathbf{T}_{AB} \in \mathrm{SE}(3)$ transforms column vectors from frame $\mathcal{B}$ to frame $\mathcal{A}$:

$$\mathbf{T}_{AB} = \begin{bmatrix} \mathbf{R}_{AB} & \mathbf{t}_{AB} \\ \mathbf{0}^\top & 1 \end{bmatrix}, \quad \mathbf{R}_{AB} \in \mathrm{SO}(3), \quad \mathbf{t}_{AB} \in \mathbb{R}^3$$

The system supports two equivalent kinematic parameterizations:
1. **Decoupled Dual-Extrinsic Parameterization**: Independent transforms $[\mathbf{R}_{LC_0}, \mathbf{t}_{LC_0}]$ and $[\mathbf{R}_{LC_1}, \mathbf{t}_{LC_1}]$ (12 DoF total).
2. **Hierarchical Stereo-Rig Parameterization (Preferred)**: The rig pose is parameterized by the primary LiDAR-to-Cam0 extrinsic $\mathbf{T}_{LC_0} \in \mathrm{SE}(3)$ (6 DoF), while the secondary camera is constrained by the factory/calibrated internal stereo baseline $\mathbf{T}_{C_0 C_1} \in \mathrm{SE}(3)$ (6 DoF):
   $$\mathbf{T}_{LC_1} = \mathbf{T}_{LC_0} \cdot \mathbf{T}_{C_0 C_1}$$
   The instantaneous world-to-camera transformation $\mathbf{T}_{C_i W}(t) = \mathbf{T}_{W C_i}^{-1}(t)$ required for forward point projection is therefore:
   $$\mathbf{T}_{C_0 W}(t) = \mathbf{T}_{LC_0}^{-1} \cdot \mathbf{T}_{WL}^{-1}(t)$$
   $$\mathbf{T}_{C_1 W}(t) = \mathbf{T}_{C_0 C_1}^{-1} \cdot \mathbf{T}_{LC_0}^{-1} \cdot \mathbf{T}_{WL}^{-1}(t)$$

### 2.4 Temporal Clock Asynchrony & Rolling Shutter Kinematics
The 3DMakerPro Raven and the Insta360 X4 operate on completely isolated quartz crystal oscillators with no hardware genlock or PPS (Pulse Per Second) trigger wire.

1. **Nominal Constant Offset Model**:
   In the baseline codebase, the video clock $t_{\text{vid}} \in [0, T_{\text{dur}}]$ (seconds elapsed since video start) is mapped to the SLAM clock $t_{\text{slam}}$ via a scalar offset $\delta t$:
   $$t_{\text{slam}}(t_{\text{vid}}) = t_{\text{slam},0} + (t_{\text{vid}} - \delta t)$$
   where $t_{\text{slam},0}$ is the initial timestamp of the SLAM trajectory.

2. **Generalized Affine Clock Drift & Rolling Shutter Model**:
   Over extended trajectories (e.g., $15\text{--}30\text{ minutes}$ in `HarmonyMall`), oscillator frequency mismatch ($\sim 20\text{--}50\text{ ppm}$), thermal drift, and variable frame-rate jitter invalidate the constant scalar offset assumption. Furthermore, the Insta360 X4 utilizes a CMOS rolling shutter sensor with readout duration $t_{\text{readout}} \approx 10\text{--}16\text{ ms}$ across $H = 3840$ rows. The exact capture timestamp $t_{\text{slam}}(t_{\text{vid}}, v)$ of a pixel at row $v \in [0, H-1]$ is governed by:
   $$t_{\text{slam}}(t_{\text{vid}}, v) = t_{\text{slam},0} + \alpha_t (t_{\text{vid}} - \delta t) + \gamma_{\text{rs}} \left(\frac{v - H/2}{H}\right) + \eta_{\text{jitter}}(t_{\text{vid}})$$
   where $\alpha_t \in \mathbb{R}^+$ is the relative clock skew factor (nominally $1.0$), $\gamma_{\text{rs}}$ is the rolling shutter line-readout span, and $\eta_{\text{jitter}}(t_{\text{vid}})$ accounts for discrete frame-drop steps.

---

## 3. Optical Projection & Distortion Model (Thin-Prism Fisheye)

To accurately model the wide-angle ($\approx 200^\circ$ FOV) optics of the Insta360 X4 lenses, including manufacturing decentering and sensor tilt relative to the optical axis, the system implements the 12-parameter **Thin-Prism Fisheye Model** (`THIN_PRISM_FISHEYE`, COLMAP Model ID `10`).

### 3.1 Intrinsic Parameter Vector
For each lens $i \in \{0, 1\}$, the intrinsic parameter vector $\Theta_{\text{intr},i} \in \mathbb{R}^{12}$ is defined as:

$$\Theta_{\text{intr}} = \big( \underbrace{f_x, f_y, c_x, c_y}_{\text{Pinhole Scale \& Center}}, \, \underbrace{k_1, k_2, k_3, k_4}_{\text{Equidistant Radial}}, \, \underbrace{p_1, p_2}_{\text{Tangential}}, \, \underbrace{s_{x1}, s_{y1}}_{\text{Thin-Prism}} \big)^\top$$

### 3.2 Forward Projection Equations $\pi(\mathbf{P}_C; \Theta_{\text{intr}}) \mapsto (u, v)$
Let $\mathbf{P}_C = [X_C, Y_C, Z_C]^\top \in \mathbb{R}^3$ be a 3D point expressed in the local camera frame $\mathcal{C}_i$.

#### Step 1: Normalized Perspective Coordinates & Spherical Extension
In the standard algebraic formulation (valid for the forward hemisphere $Z_C > 0$), the perspective normalized coordinates $(x, y)$ and Euclidean radial distance $r$ are defined as:

$$x = \frac{X_C}{Z_C}, \quad y = \frac{Y_C}{Z_C}, \quad r = \sqrt{x^2 + y^2}, \quad \theta = \arctan(r)$$

> **Critical Implementation Note for $\text{FOV} > 180^\circ$**:  
> Because the Insta360 X4 lenses cover $\approx 200^\circ$ ($\theta_{\max} \approx 100^\circ = 1.745\text{ rad}$), rays in the outer equatorial periphery have $Z_C \le 0$. Direct evaluation of $x = X_C/Z_C$ and $\theta = \arctan(r)$ suffers from a division-by-zero singularity at $Z_C = 0$ ($\theta = \pi/2$) and folds rays with $Z_C < 0$ back into the forward hemisphere. To maintain exact equivalence with the standard model for $Z_C > 0$ while extending smoothly across $\theta \in [0, \theta_{\max}]$, the directional unit coordinates $(\hat{x}, \hat{y})$ and incidence angle $\theta$ must be computed via:
> $$R_{XY} = \sqrt{X_C^2 + Y_C^2}, \quad \theta = \operatorname{atan2}(R_{XY}, Z_C) \in [0, \pi]$$
> Note that for $Z_C > 0$, $\frac{x}{r} = \frac{X_C / Z_C}{\sqrt{X_C^2 + Y_C^2}/Z_C} = \frac{X_C}{R_{XY}}$, which is completely free of the $Z_C = 0$ singularity.

#### Step 2: Equidistant Polynomial Angle Expansion
The incident ray angle $\theta$ is mapped to the radially distorted angle $\theta_d$ via a 9th-order odd polynomial:

$$\theta_d = \theta \left( 1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8 \right)$$

#### Step 3: Radially Distorted Normalized Coordinates
The radially distorted normalized coordinates $(x_r, y_r)$ are obtained by scaling the direction vector by $\theta_d$:

$$x_r = \frac{\theta_d}{r} x = \theta_d \frac{X_C}{R_{XY}}, \quad y_r = \frac{\theta_d}{r} y = \theta_d \frac{Y_C}{R_{XY}}$$

In the limit as $r \to 0$ ($R_{XY} \to 0$, on the optical axis), Taylor expansion yields $\lim_{r \to 0} \frac{\theta_d}{r} = 1$, so $(x_r, y_r) = (x, y) = (0, 0)$. Computationally, if $R_{XY} < \epsilon$ ($\epsilon = 10^{-8}$), set $x_r = X_C / Z_C$ and $y_r = Y_C / Z_C$.

#### Step 4: Tangential (Decentering) and Thin-Prism Distortion
Define the radially distorted squared radius $\rho^2 = x_r^2 + y_r^2 = \theta_d^2$. The tangential distortion $(\Delta x_t, \Delta y_t)$ and thin-prism distortion $(\Delta x_p, \Delta y_p)$ are applied directly in the distorted normalized plane:

$$\Delta x_t = 2 p_1 x_r y_r + p_2 (\rho^2 + 2 x_r^2)$$
$$\Delta y_t = p_1 (\rho^2 + 2 y_r^2) + 2 p_2 x_r y_r$$
$$\Delta x_p = s_{x1} \rho^2, \quad \Delta y_p = s_{y1} \rho^2$$

Summing these components produces the fully distorted normalized coordinates $(x_d, y_d)$:

$$\begin{bmatrix} x_d \\ y_d \end{bmatrix} = \begin{bmatrix} x_r + 2 p_1 x_r y_r + p_2 (\rho^2 + 2 x_r^2) + s_{x1} \rho^2 \\ y_r + p_1 (\rho^2 + 2 y_r^2) + 2 p_2 x_r y_r + s_{y1} \rho^2 \end{bmatrix}$$

#### Step 5: Pixel Plane Projection
Finally, the pinhole intrinsic matrix maps $(x_d, y_d)$ to pixel coordinates $(u, v) \in [0, W) \times [0, H)$:

$$u = f_x x_d + c_x$$
$$v = f_y y_d + c_y$$

### 3.3 Analytical Jacobians for Non-Linear Optimization
For gradient-based extrinsic/intrinsic optimization and Bundle Adjustment, the chain rule decomposition of the projection Jacobian $\mathbf{J}_{\pi} = \frac{\partial (u, v)}{\partial \mathbf{P}_C} \in \mathbb{R}^{2 \times 3}$ is given by:

$$\mathbf{J}_{\pi} = \frac{\partial (u, v)}{\partial (x_d, y_d)} \cdot \frac{\partial (x_d, y_d)}{\partial (x_r, y_r)} \cdot \frac{\partial (x_r, y_r)}{\partial \mathbf{P}_C}$$

where:
1. **Pinhole Scaling Jacobian**:
   $$\frac{\partial (u, v)}{\partial (x_d, y_d)} = \begin{bmatrix} f_x & 0 \\ 0 & f_y \end{bmatrix}$$

2. **Tangential & Thin-Prism Jacobian** $\mathbf{J}_{\text{tp}} = \frac{\partial (x_d, y_d)}{\partial (x_r, y_r)} \in \mathbb{R}^{2 \times 2}$:
   $$\mathbf{J}_{\text{tp}} = \begin{bmatrix} 1 + 2 p_1 y_r + 6 p_2 x_r + 2 s_{x1} x_r & 2 p_1 x_r + 2 p_2 y_r + 2 s_{x1} y_r \\ 2 p_1 x_r + 2 p_2 y_r + 2 s_{y1} x_r & 1 + 6 p_1 y_r + 2 p_2 x_r + 2 s_{y1} y_r \end{bmatrix}$$

3. **Radial Fisheye to 3D Ray Jacobian** $\frac{\partial (x_r, y_r)}{\partial \mathbf{P}_C} \in \mathbb{R}^{2 \times 3}$:
   Let $\theta'_d = \frac{\mathrm{d}\theta_d}{\mathrm{d}\theta} = 1 + 3 k_1 \theta^2 + 5 k_2 \theta^4 + 7 k_3 \theta^6 + 9 k_4 \theta^8$, and let $D = \|\mathbf{P}_C\|_2 = \sqrt{R_{XY}^2 + Z_C^2}$. Then:
   $$\frac{\partial \theta}{\partial \mathbf{P}_C} = \frac{1}{D^2 R_{XY}} \begin{bmatrix} X_C Z_C, & Y_C Z_C, & -R_{XY}^2 \end{bmatrix}$$
   $$\frac{\partial x_r}{\partial \mathbf{P}_C} = \frac{X_C}{R_{XY}} \theta'_d \frac{\partial \theta}{\partial \mathbf{P}_C} + \frac{\theta_d}{R_{XY}^3} \begin{bmatrix} Y_C^2, & -X_C Y_C, & 0 \end{bmatrix}$$
   $$\frac{\partial y_r}{\partial \mathbf{P}_C} = \frac{Y_C}{R_{XY}} \theta'_d \frac{\partial \theta}{\partial \mathbf{P}_C} + \frac{\theta_d}{R_{XY}^3} \begin{bmatrix} -X_C Y_C, & X_C^2, & 0 \end{bmatrix}$$

### 3.4 Iterative Unprojection $\pi^{-1}(u, v; \Theta_{\text{intr}}) \mapsto \hat{\mathbf{d}}_C \in \mathbb{S}^2$
Because the Thin-Prism Fisheye forward map is a coupled high-degree polynomial system, no closed-form algebraic inverse exists. Given a pixel $(u, v)$:
1. Compute target distorted coordinates: $x_d = (u - c_x)/f_x$, $y_d = (v - c_y)/f_y$.
2. Initialize $(x_r^{(0)}, y_r^{(0)}) = (x_d, y_d)$ and iterate via Newton-Raphson for $k = 0 \dots 5$:
   $$\begin{bmatrix} x_r^{(k+1)} \\ y_r^{(k+1)} \end{bmatrix} = \begin{bmatrix} x_r^{(k)} \\ y_r^{(k)} \end{bmatrix} - \mathbf{J}_{\text{tp}}^{-1}\left(x_r^{(k)}, y_r^{(k)}\right) \begin{bmatrix} f_x(x_r^{(k)}, y_r^{(k)}) - x_d \\ f_y(x_r^{(k)}, y_r^{(k)}) - y_d \end{bmatrix}$$
3. Compute $\theta_d = \sqrt{(x_r^*)^2 + (y_r^*)^2}$ and solve the 1D polynomial $\theta(1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8) - \theta_d = 0$ for $\theta \in [0, \theta_{\max}]$ via scalar Newton-Raphson:
   $$\theta^{(m+1)} = \theta^{(m)} - \frac{\theta^{(m)}(1 + \sum_{j=1}^4 k_j (\theta^{(m)})^{2j}) - \theta_d}{1 + \sum_{j=1}^4 (2j+1) k_j (\theta^{(m)})^{2j}}$$
4. Reconstruct the unit bearing vector $\hat{\mathbf{d}}_C \in \mathbb{S}^2$:
   $$\hat{\mathbf{d}}_C = \begin{bmatrix} \sin\theta \cdot \frac{x_r^*}{\theta_d} \\ \sin\theta \cdot \frac{y_r^*}{\theta_d} \\ \cos\theta \end{bmatrix}$$

---

## 4. Full Colorization & Projection Pipeline

The end-to-end data flow from raw sensor streams to colored metric point clouds and COLMAP neural-rendering packages is structured as follows:

```
+------------------------+       +-------------------------+
| 3DMakerPro Raven SLAM  |       | Insta360 X4 8K Video    |
| - 10Hz Pose Trajectory |       | - 7680x3840 @ 29.97Hz   |
| - Metric Cloud P_W     |       | - Dual Fisheye Streams  |
+-----------+------------+       +------------+------------+
            |                                 |
            v                                 v
+------------------------+       +-------------------------+
| SE(3) Trajectory       |<------| Temporal Sync Model     |
| Interpolator (SLERP /  |       | t_slam = t0 + a(t - dt) |
| Cumulative B-Spline)   |       +------------+------------+
+-----------+------------+                    |
            | T_{WL}(t_j)                     | Frame Extraction
            v                                 v
+----------------------------------------------------------+
| Rig Kinematic Transform: P_{Ci}(t_j) = T_{Ci W}(t_j) P_W |
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
| Thin-Prism Fisheye Projection & Masking (Theta <= 100 deg)|
+---------------------------+------------------------------+
                            |
                            v
+----------------------------------------------------------+
| Visibility & Multi-View Photometric Aggregator           |
| - Z-Buffer Splatting / Hidden Point Removal (HPR)        |
| - Incidence Normal & Equatorial Feather Weighting        |
+-------------+------------------------------+-------------+
              |                              |
              v                              v
+---------------------------+  +---------------------------+
| Metric Colored Cloud      |  | COLMAP 3DGS / NeRF Export |
| - output_colored.ply      |  | - cameras.bin (Model 10)  |
| - Per-vertex RGB + Normals|  | - images.bin, points3D.bin|
+---------------------------+  +---------------------------+
```

### 4.1 Continuous Trajectory Interpolation
Because video frames arrive at $\approx 30\text{ Hz}$ (or continuously per scanline under rolling shutter correction) while SLAM poses are sampled at $10\text{ Hz}$, the trajectory $\mathbf{T}_{WL}(t)$ must be interpolated between discrete SLAM keyframes $\{t_m, \mathbf{p}_m, \mathbf{q}_m\}_{m=1}^M$.

#### 1. Baseline Piecewise Linear / SLERP Interpolation
Given query time $t \in [t_m, t_{m+1}]$, define the normalized interpolation factor $s = \frac{t - t_m}{t_{m+1} - t_m} \in [0, 1]$:
- **Position**: Linear interpolation (LERP) or Catmull-Rom cubic spline:
  $$\mathbf{p}_{WL}(t) = (1 - s)\mathbf{p}_m + s\mathbf{p}_{m+1}$$
- **Orientation**: Spherical Linear Interpolation (SLERP) on $\mathbb{H}_1$ (enforcing the shortest-path antipodal check $\mathbf{q}_m \cdot \mathbf{q}_{m+1} \ge 0$):
  $$\mathbf{q}_{WL}(t) = \mathbf{q}_m \left(\mathbf{q}_m^{-1} \mathbf{q}_{m+1}\right)^s = \frac{\sin((1-s)\Omega)}{\sin\Omega}\mathbf{q}_m + \frac{\sin(s\Omega)}{\sin\Omega}\mathbf{q}_{m+1}, \quad \Omega = \arccos(\mathbf{q}_m \cdot \mathbf{q}_{m+1})$$

#### 2. Cumulative Cubic B-Spline on $\mathrm{SE}(3)$ (Required for Continuous-Time Optimization)
Piecewise SLERP has discontinuous angular velocity $\boldsymbol{\omega}(t)$ at knot boundaries $t_m$ ($C^0$ velocity), which corrupts analytical derivatives with respect to time offset $\frac{\partial \mathbf{P}_C}{\partial \delta t}$. For optimization, the trajectory is parameterized using a cumulative uniform cubic B-spline in $\mathrm{SE}(3)$ with knot spacing $\Delta t_s$:

$$\mathbf{T}_{WL}(t) = \mathbf{T}_{m-1} \prod_{j=1}^{3} \exp_{\mathfrak{se}(3)}\left( \tilde{B}_j(s) \, \boldsymbol{\xi}_{m-1+j} \right)$$

where $\boldsymbol{\xi}_k = \log_{\mathrm{SE}(3)}(\mathbf{T}_{k-1}^{-1} \mathbf{T}_k) \in \mathfrak{se}(3) \cong \mathbb{R}^6$ are the relative Lie algebra twists between control poses, and $\tilde{B}_j(s)$ are the cumulative cubic B-spline basis functions:

$$\begin{bmatrix} \tilde{B}_0(s) \\ \tilde{B}_1(s) \\ \tilde{B}_2(s) \\ \tilde{B}_3(s) \end{bmatrix} = \frac{1}{6} \begin{bmatrix} 6 & 0 & 0 & 0 \\ 5 & 3 & -3 & 1 \\ 1 & 3 & 3 & -2 \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ s \\ s^2 \\ s^3 \end{bmatrix}$$

This guarantees $C^2$ continuity and closed-form analytical time derivatives $\dot{\mathbf{T}}_{WL}(t)$ and $\ddot{\mathbf{T}}_{WL}(t)$.

### 4.2 3D-to-2D Forward Projection & Masking
For each selected video keyframe $j$ at SLAM-aligned timestamp $t_j$:
1. Transform every candidate world point $\mathbf{P}_{W,k}$ into the instantaneous LiDAR frame:
   $$\mathbf{P}_{L,k}(t_j) = \mathbf{R}_{WL}^\top(t_j)\left(\mathbf{P}_{W,k} - \mathbf{t}_{WL}(t_j)\right)$$
2. Transform into $\mathcal{C}_0$ and $\mathcal{C}_1$:
   $$\mathbf{P}_{C_0,k}(t_j) = \mathbf{R}_{LC_0}^\top\left(\mathbf{P}_{L,k}(t_j) - \mathbf{t}_{LC_0}\right), \quad \mathbf{P}_{C_1,k}(t_j) = \mathbf{R}_{C_0 C_1}^\top\left(\mathbf{P}_{C_0,k}(t_j) - \mathbf{t}_{C_0 C_1}\right)$$
3. Filter by field-of-view cone: compute incidence angle $\theta_{i,k} = \operatorname{atan2}\left(\sqrt{X_{C_i,k}^2 + Y_{C_i,k}^2}, Z_{C_i,k}\right)$. Discard if $\theta_{i,k} > \theta_{\text{cutoff}}$ (typically $\theta_{\text{cutoff}} \in [95^\circ, 99^\circ]$).
4. Project surviving points via $\mathbf{u}_{i,j,k} = \pi_i(\mathbf{P}_{C_i,k}(t_j); \Theta_{\text{intr},i})$ and apply static binary masks $M_i(u, v) \in \{0, 1\}$ to reject the operator's body, hand, mounting rod, and the LiDAR housing.

### 4.3 Visibility, Z-Buffering, and Multi-View Blending
Because a point cloud has zero intrinsic surface area, direct projection of $\mathcal{P}_W$ onto an image plane suffers from **see-through occlusion** (background points behind walls or pillars project onto foreground pixels). The pipeline resolves visibility and aggregates colors via a three-stage filter:

1. **Multi-Scale Splat Z-Buffering**:
   For each camera frame $(i, j)$, a depth buffer $Z_{\text{buf}}(u, v)$ is initialized to $+\infty$ at downsampled grid cells (or using a depth-adaptive splat kernel of radius $r_{\text{splat}}(d) = \lceil \frac{f \cdot s_{\text{voxel}}}{d} \rceil$ pixels, where $d = \|\mathbf{P}_{C_i,k}\|_2$ and $s_{\text{voxel}}$ is the point cloud spacing). A point $\mathbf{P}_{W,k}$ is marked visible only if:
   $$d_{i,j,k} \le Z_{\text{buf}}(\lfloor u \rfloor, \lfloor v \rfloor) + \tau_{\text{depth}}$$
   where $\tau_{\text{depth}} \approx 0.03\text{ m}$ is the surface thickness tolerance.

2. **Surface Normal Back-Face & Grazing Rejection**:
   If the LiDAR point possesses an oriented surface normal $\mathbf{n}_{W,k}$, let $\hat{\mathbf{v}}_{i,j,k} = \frac{\mathbf{t}_{WC_i}(t_j) - \mathbf{P}_{W,k}}{\|\mathbf{t}_{WC_i}(t_j) - \mathbf{P}_{W,k}\|_2}$ be the unit viewing vector from the point to the camera optical center. Points are rejected if $\mathbf{n}_{W,k} \cdot \hat{\mathbf{v}}_{i,j,k} < \cos(\alpha_{\max})$ (e.g., $\alpha_{\max} = 80^\circ$).

3. **Photometric Confidence Weighting**:
   Each valid observation $o \in \mathcal{O}_k$ of point $k$ from camera $i$ at frame $j$ is assigned a composite weight $w_{i,j,k}$:
   $$w_{i,j,k} = \underbrace{\left(\frac{\mathbf{n}_{W,k} \cdot \hat{\mathbf{v}}_{i,j,k}}{d_{i,j,k}^2 + d_0^2}\right)}_{\text{Geometric Quality}} \cdot \underbrace{\cos^2\left(\frac{\pi}{2} \frac{\min(\theta_{i,j,k}, \theta_{\text{max}})}{\theta_{\text{max}}}\right)}_{\text{Equatorial Seam Feathering}} \cdot \underbrace{\exp\left(-\lambda_\omega \|\boldsymbol{\omega}_L(t_j)\|_2^2\right)}_{\text{Motion Blur Penalty}}$$
   The final color $\mathbf{c}_k$ of point $\mathbf{P}_{W,k}$ is computed via weighted robust trimmed mean (rejecting top/bottom $15\%$ luminance outliers caused by transient pedestrians or specular glints):
   $$\mathbf{c}_k = \frac{\sum_{(i,j) \in \mathcal{O}_k^{\text{inlier}}} w_{i,j,k} \, I_i(u_{i,j,k}, v_{i,j,k}; t_j)}{\sum_{(i,j) \in \mathcal{O}_k^{\text{inlier}}} w_{i,j,k}}$$

### 4.4 Exporting Deliverables (COLMAP & 3DGS)
To initialize 3D Gaussian Splatting (both native fisheye 3DGS such as `Fisheye-GS` / `3DGUT` and perspective-cubemap 3DGS), the pipeline serializes:
* `cameras.bin`: Two camera entries (`CAMERA_ID = 1` for $\mathcal{C}_0$, `CAMERA_ID = 2` for $\mathcal{C}_1$) with `MODEL_ID = 10` (`THIN_PRISM_FISHEYE`), width $3840$, height $3840$, and the 12 double-precision parameters $\Theta_{\text{intr},i}$.
* `images.bin`: For each exported keyframe $j$ and lens $i \in \{0, 1\}$, writes `IMAGE_ID`, the world-to-camera unit quaternion $\mathbf{q}_{C_i W}(t_j) = [q_w, q_x, q_y, q_z]^\top$, translation vector $\mathbf{t}_{C_i W}(t_j) = -\mathbf{R}_{C_i W}(t_j)\mathbf{t}_{W C_i}(t_j)$, `CAMERA_ID`, image filename, and 2D-3D visibility correspondences.
* `points3D.bin` & `output_colored.ply`: Downsampled metric 3D points $\mathbf{P}_{W,k}$, RGB colors $\mathbf{c}_k$, normals $\mathbf{n}_{W,k}$, and track lists mapping back to `(IMAGE_ID, POINT2D_IDX)`.

---

## 5. Current Calibration Approaches in the Codebase

The existing `LidarCamera360` architecture employs a **Two-Tier Decoupled Calibration** strategy combined with a static **Hardware Profile System**.

### 5.1 Two-Tier Decoupled Calibration Architecture

```
+-------------------------------------------------------------------------+
| TIER 1: Inter-Lens Stereo Calibration (C0 <---> C1)                     |
| - Input: Dual-fisheye video frames in overlap zone (theta in [80, 100]) |
| - Method: SIFT/SuperPoint matching across equatorial ring + Essential   |
|   Matrix / Bundle Adjustment with metric baseline prior ||t|| = d_base  |
| - Output: Intrinsics Theta_{0,1} and relative pose T_{C0 C1}            |
+------------------------------------+------------------------------------+
                                     | Locks T_{C0 C1}
                                     v
+-------------------------------------------------------------------------+
| TIER 2: LiDAR-to-Camera Rig Calibration (L <---> C_rig)                 |
| - Stage 2A (Temporal + Coarse Rotation): Angular velocity cross-        |
|   correlation ||omega_L(t)|| vs ||omega_C(t - dt)|| + Hand-Eye AX = XB  |
| - Stage 2B (Fine Extrinsic Refinement): Projected LiDAR depth/intensity |
|   discontinuity alignment against image Sobel/Canny edge distance maps  |
| - Output: T_{L C0} and scalar temporal offset dt                        |
+-------------------------------------------------------------------------+
```

#### Tier 1: Inter-Lens Relative Pose Calibration ($\mathcal{C}_0 \leftrightarrow \mathcal{C}_1$)
1. **Intrinsic Initialization**: Performed offline via checkerboard/ChArUco calibration or single-scene self-calibration.
2. **Equatorial Feature Matching**: Extracts keypoints (ORB/SIFT/SuperPoint) restricted to the annular overlap band $\theta_i \in [80^\circ, 100^\circ]$ in synchronous frame pairs $(I_0(t_j), I_1(t_j))$.
3. **Epipolar Optimization**: Unprojects matched pixels $(\mathbf{u}_{0,m}, \mathbf{u}_{1,m})$ to unit bearing rays $(\hat{\mathbf{d}}_{0,m}, \hat{\mathbf{d}}_{1,m}) \in \mathbb{S}^2$ and minimizes the generalized epipolar / reprojection residual over $\mathbf{R}_{C_0 C_1} \in \mathrm{SO}(3)$, with the translation norm $\|\mathbf{t}_{C_0 C_1}\|_2 = d_{\text{base}}$ fixed to the physical CAD measurement ($40\text{ mm}$) due to single-camera scale unobservability:
   $$\min_{\mathbf{R}_{C_0 C_1}, \hat{\mathbf{t}}_{C_0 C_1}} \sum_{j, m} \rho_{\text{Huber}} \left( \hat{\mathbf{d}}_{1,m}^\top \left[ \mathbf{t}_{C_0 C_1} \right]_\times \mathbf{R}_{C_0 C_1}^\top \hat{\mathbf{d}}_{0,m} \right)^2$$

#### Tier 2: Rig Calibration ($\mathcal{L} \leftrightarrow \mathcal{C}_{\text{rig}}$)
1. **Temporal & Rotational Initialization (Trajectory Correlation)**:
   Visual odometry (or spherical optical flow) estimates camera angular velocity $\boldsymbol{\omega}_{C_0}(\tau_j)$. Cross-correlating $\|\boldsymbol{\omega}_{C_0}(\tau)\|_2$ against the SLAM angular velocity norm $\|\boldsymbol{\omega}_L(t)\|_2$ yields the coarse time offset $\delta t_0$:
   $$\delta t_0 = \arg\max_{\delta t} \int \|\boldsymbol{\omega}_L(t_{\text{slam},0} + \tau - \delta t)\|_2 \cdot \|\boldsymbol{\omega}_{C_0}(\tau)\|_2 \, \mathrm{d}\tau$$
   Subsequently, Kabsch/Wahba alignment on the aligned angular velocity vectors $\boldsymbol{\omega}_L(t) = \mathbf{R}_{LC_0}\boldsymbol{\omega}_{C_0}(t)$ initializes $\mathbf{R}_{LC_0}$.
2. **Edge-Gradient Alignment**:
   Extracts 3D geometric discontinuities (depth jumps and high surface curvature) from $\mathcal{P}_W$, projects them into $\mathcal{C}_0$ and $\mathcal{C}_1$, and optimizes $(\mathbf{R}_{LC_0}, \mathbf{t}_{LC_0}, \delta t)$ via Nelder-Mead or Powell/L-BFGS over a Chamfer distance-transformed Sobel/Canny edge map $D_{\text{edge}}(u, v)$ of the camera images.

### 5.2 Hardware Profile System (`configs/profiles/`)
Because physical mounting brackets (e.g., CNC aluminum cages joining the Raven and Insta360 X4) remain rigid across capture sessions unless disassembled, the codebase implements a **Hardware Profile System** (`configs/profiles/raven_x4_montagem_1.json`). This decouples the static mechanical mount geometry from per-dataset estimation noise.

```json
{
  "profile_name": "raven_x4_montagem_1",
  "version": "2.1.0",
  "hardware": {
    "lidar_model": "3DMakerPro_Raven",
    "camera_model": "Insta360_X4",
    "resolution": [3840, 3840],
    "fps_nominal": 29.97
  },
  "cameras": {
    "cam0": {
      "model": "THIN_PRISM_FISHEYE",
      "intrinsics": {
        "fx": 1142.85, "fy": 1142.60, "cx": 1919.40, "cy": 1920.80,
        "k1": 0.0412, "k2": -0.0118, "k3": 0.0024, "k4": -0.0003,
        "p1": 0.00012, "p2": -0.00008, "sx1": 0.00015, "sy1": -0.00011
      },
      "fov_max_deg": 200.0,
      "valid_radius_px": 1890.0
    },
    "cam1": {
      "model": "THIN_PRISM_FISHEYE",
      "intrinsics": {
        "fx": 1141.90, "fy": 1141.75, "cx": 1920.60, "cy": 1918.90,
        "k1": 0.0408, "k2": -0.0115, "k3": 0.0022, "k4": -0.00028,
        "p1": -0.00009, "p2": 0.00011, "sx1": -0.00010, "sy1": 0.00014
      },
      "fov_max_deg": 200.0,
      "valid_radius_px": 1890.0
    }
  },
  "extrinsics": {
    "T_L_C0": {
      "quaternion_wxyz": [0.5012, -0.4988, 0.4995, -0.5005],
      "translation_xyz_m": [0.0024, -0.0152, 0.1485],
      "covariance_diag": [1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4]
    },
    "T_C0_C1": {
      "quaternion_wxyz": [0.0008, 0.0012, 0.999998, -0.0009],
      "translation_xyz_m": [0.0002, -0.0001, -0.0402],
      "covariance_diag": [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
    }
  },
  "temporal": {
    "dt_initial_guess_s": 0.0,
    "search_window_s": [-15.0, 15.0],
    "rolling_shutter_readout_s": 0.0124
  }
}
```

**Why the Current Approach Fails on `HarmonyMall`**:  
When the Tier 2 edge-gradient optimizer runs on `HarmonyMall` starting from `raven_x4_montagem_1.json`, it attempts to optimize a single global rigid transform $\mathbf{T}_{LC_0}$ and scalar $\delta t$ across a $688\text{ m}$ trajectory. Because `HarmonyMall` contains local SLAM pose drift, clock skew, and repetitive floor tile edges at steep grazing angles, the global optimizer either diverges from the valid hardware profile into a spurious local minimum or fails to correct time-varying trajectory errors—resulting in severe visual artifacts even when the physical mount has not moved.

---

## 6. Detailed Anatomy of Calibration Failures on Difficult Datasets (`HarmonyMall`)

To engineer a definitive algorithmic solution, the receiving agent must understand the five fundamental mathematical and physical mechanisms driving calibration failure in large-scale commercial environments.

### 6.1 Failure Mode 1: Grazing-Angle Lever-Arm Sensitivity on Ground Planes
In commercial malls, parking garages, and concourses, the most visually salient high-contrast features are horizontal ground markings (parking bay lines, tactile paving, floor tile grids, lane arrows).

```
     [Camera C0 / LiDAR Rig]
              O  <-- Optical Center at height h ≈ 1.5 m
             /|\
            / | \
           /  |  \  Ray angle alpha from nadir
     h    /   |   \  (alpha = arctan(s / h))
         /    |    \
        /     |     \    Grazing Ray (alpha ≈ 75° - 82°)
       /      |      \  /
======+=======+=======*=========*================ [Ground Plane Z_W = 0]
      0               s       s + ds
                      |<--ds--->|
                 Ground displacement
```

#### Mathematical Derivation of Grazing Amplification
Consider the camera held at height $h \approx 1.5\text{ m}$ above a horizontal ground plane ($Z_W = 0$). A ray projected at angle $\alpha$ relative to the vertical nadir vector ($-Z_W$) intersects the ground plane at horizontal distance $s$:

$$s(\alpha) = h \tan(\alpha)$$

Taking the first-order differential with respect to an angular calibration error $\Delta \theta$ (e.g., pitch/roll error in $\mathbf{R}_{LC_0}$ or SLAM pitch drift in $\mathbf{R}_{WL}(t)$) where $\alpha \to \alpha + \Delta \theta$:

$$\Delta s = \frac{\partial s}{\partial \alpha} \Delta \theta + \mathcal{O}(\Delta \theta^2) = \frac{h \, \Delta \theta}{\cos^2(\alpha)} = h \left(1 + \left(\frac{s}{h}\right)^2\right) \Delta \theta$$

where $\Delta \theta$ is expressed in radians ($\Delta \theta = \Delta \theta_{\text{deg}} \cdot \frac{\pi}{180}$).

#### Quantitative Impact Analysis ($h = 1.5\text{ m}$)

| Horizontal Distance $s$ | Nadir Incidence Angle $\alpha$ | $\cos^2(\alpha)$ Amplification Factor | Ground Shift $\Delta s$ per $1.0^\circ$ Pitch Error | Ground Shift $\Delta s$ per $0.2^\circ$ Pitch Error |
| :--- | :--- | :--- | :--- | :--- |
| **$0.5\text{ m}$** (Underneath) | $18.43^\circ$ | $0.900$ ($\times 1.11$) | $2.91\text{ cm}$ | $0.58\text{ cm}$ |
| **$1.5\text{ m}$** (Close) | $45.00^\circ$ | $0.500$ ($\times 2.00$) | $5.24\text{ cm}$ | $1.05\text{ cm}$ |
| **$3.0\text{ m}$** (Mid-Ground) | $63.43^\circ$ | $0.200$ ($\times 5.00$) | $13.09\text{ cm}$ | $2.62\text{ cm}$ |
| **$5.0\text{ m}$** (Grazing Floor) | $73.30^\circ$ | $0.083$ ($\times 12.11$) | **$31.70\text{ cm}$** | **$6.34\text{ cm}$** |
| **$8.0\text{ m}$** (Far Corridor) | $79.38^\circ$ | $0.034$ ($\times 29.44$) | **$77.08\text{ cm}$** | **$15.42\text{ cm}$** |

**Consequence**: A painted floor line or tile grout joint is typically $1\text{--}5\text{ cm}$ wide. At $s = 5\text{ m}$, a seemingly minuscule rotational error of $\Delta \theta = 0.2^\circ$ shifts the projected color by **$6.34\text{ cm}$** on the floor—completely missing the line! As the operator walks forward from $s = 5\text{ m}$ to $s = 1\text{ m}$, the projected line sweeps across the point cloud by $6.34\text{ cm} - 0.8\text{ cm} \approx 5.5\text{ cm}$, smearing a sharp line into a blurred ghost band or double line.

---

### 6.2 Failure Mode 2: Equatorial Seam Transition ($\mathcal{C}_0 \leftrightarrow \mathcal{C}_1$)
Because $\mathcal{C}_0$ faces forward ($+X_L$) and $\mathcal{C}_1$ faces backward ($-X_L$), the ground plane directly beneath the operator ($\alpha \to 0^\circ$ from nadir) and the walls directly to the left/right of the operator lie at an optical incidence angle of $\theta \approx 90^\circ$ relative to both lenses.

#### Mathematical Pathology at $\theta \approx 90^\circ$
1. **Differential Parallax Disparity**:
   When a 3D ground point $\mathbf{P}_W$ passes through the equatorial overlap zone $\theta \in [80^\circ, 100^\circ]$, its projected pixel locations in $\mathcal{C}_0$ and $\mathcal{C}_1$ are governed by:
   $$\mathbf{P}_{C_1} = \mathbf{R}_{C_0 C_1}^\top \left(\mathbf{P}_{C_0} - \mathbf{t}_{C_0 C_1}\right)$$
   At the nadir directly below the rig ($d \approx 1.5\text{ m}$), the physical baseline $d_{\text{base}} = \|\mathbf{t}_{C_0 C_1}\|_2 \approx 40\text{ mm}$ induces a legitimate stereo parallax angle of:
   $$\Delta \psi_{\text{parallax}} \approx \frac{d_{\text{base}}}{h} = \frac{0.040}{1.5} = 0.0267\text{ rad} = 1.53^\circ \approx 30.4\text{ pixels}$$
   In standard 2D panoramic stitching (e.g., Insta360 Studio), spherical stitching warps images assuming a fixed sphere radius ($R_{\text{stitch}} \to \infty$), which fails at $1.5\text{ m}$. Our 3D LiDAR pipeline projects true 3D geometry $\mathbf{P}_W$, which theoretically resolves parallax **if and only if** three conditions hold simultaneously:
   - The LiDAR floor elevation $Z_W$ is exact to within $\pm 2\text{ mm}$.
   - The inter-camera extrinsic $\mathbf{T}_{C_0 C_1}$ is accurate to within $\pm 0.5\text{ mm}$ and $\pm 0.05^\circ$.
   - The higher-order radial distortion polynomial coefficients $(k_3, k_4)$ and thin-prism terms $(s_{x1}, s_{y1})$ match at $\theta = \pi/2$.

2. **Polynomial Conditioning Explosion at the Equator**:
   In the Thin-Prism Fisheye model, $\theta_d = \theta(1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8)$. At the image center ($\theta = 0.2\text{ rad}$), $\theta^9 = 5.12 \times 10^{-7}$, so $k_4$ has zero influence. At the equator ($\theta = 1.65\text{ rad} \approx 94.5^\circ$), $\theta^9 = \mathbf{99.24}$!
   $$\frac{\partial u}{\partial k_4}\bigg|_{\theta = 1.65} = f_x \theta^9 \approx 1142 \times 99.24 \approx \mathbf{1.13 \times 10^5\text{ pixels per unit } k_4}$$
   If checkerboard calibration was performed predominantly in the central $\theta \in [0^\circ, 75^\circ]$ cone, $k_3$ and $k_4$ are extrapolated at $\theta \in [85^\circ, 100^\circ]$, creating a systematic radial divergence between $\mathcal{C}_0$ and $\mathcal{C}_1$ right at the seam boundary. Straight lines crossing beneath the operator undergo an abrupt **step-shear discontinuity**.

---

### 6.3 Failure Mode 3: LiDAR SLAM Drift & Dynamic Objects
In `HarmonyMall`, the operator traverses $688\text{ m}$ over multiple minutes through wide atriums populated by walking shoppers.

1. **Observability Coupling Between Trajectory Drift and Static Extrinsics**:
   Let the true world-to-LiDAR pose at time $t$ be $\mathbf{T}_{WL}^{\text{true}}(t)$, and let the SLAM-reported pose be corrupted by a slowly varying drift perturbation $\delta \boldsymbol{\xi}_W(t) \in \mathfrak{se}(3)$:
   $$\mathbf{T}_{WL}^{\text{slam}}(t) = \mathbf{T}_{WL}^{\text{true}}(t) \cdot \exp_{\mathfrak{se}(3)}\left(\delta \boldsymbol{\xi}_L(t)\right)$$
   Meanwhile, the point cloud $\mathcal{P}_W$ is an aggregation of scans accumulated at different times $\{t_{\text{scan}}\}$. When a point $\mathbf{P}_W$ scanned at $t_1$ is observed by camera $i$ at a later time $t_2$ (e.g., viewing a pillar or floor line from the opposite side of an atrium), the effective camera-frame point is:
   $$\mathbf{P}_{C_i}(t_2) = \mathbf{T}_{LC_i}^{-1} \cdot \exp\left(-\delta \boldsymbol{\xi}_L(t_2)\right) \cdot \left(\mathbf{T}_{WL}^{\text{true}}(t_2)\right)^{-1} \cdot \mathbf{T}_{WL}^{\text{true}}(t_1) \cdot \exp\left(\delta \boldsymbol{\xi}_L(t_1)\right) \cdot \mathbf{P}_L(t_1)$$
   If a calibration algorithm optimizes a **static** extrinsic $\mathbf{T}_{LC_i}$ across the entire trajectory assuming $\delta \boldsymbol{\xi}_L(t) \equiv \mathbf{0}$, the least-squares solver projects the non-zero drift residual $\delta \boldsymbol{\xi}_L(t_2) - \operatorname{Ad}_{\mathbf{T}_{12}}\delta \boldsymbol{\xi}_L(t_1)$ onto the 6-DoF subspace of $\mathbf{T}_{LC_i}$:
   $$\hat{\mathbf{T}}_{LC_i} = \mathbf{T}_{LC_i}^{\text{true}} \cdot \exp\left( \left(\sum_j \mathbf{J}_j^\top \mathbf{J}_j\right)^{-1} \sum_j \mathbf{J}_j^\top \mathbf{r}_{\text{drift},j} \right)$$
   This biases the rigid extrinsic $\hat{\mathbf{T}}_{LC_i}$ away from its true physical mounting value, degrading alignment everywhere in the dataset—including segments where SLAM was locally accurate!

2. **Dynamic Object Corruption**:
   Moving pedestrians captured in the LiDAR scan leave transient "ghost point trails" or get partially removed by post-processing, while appearing in the video frames as sharp visual edges. Edge-based calibration objectives blindly snap static architectural LiDAR edges (pillars, storefronts) onto the high-gradient silhouettes of walking pedestrians.

---

### 6.4 Failure Mode 4: Non-Uniform Temporal Sync, Clock Drift, & Rolling Shutter
Assuming a single constant scalar offset $\delta t$ across a $900\text{ s}$ ($15\text{-minute}$) capture fails due to three compounding temporal phenomena:

1. **Spatio-Temporal Pixel Velocity Sensitivity**:
   Let the LiDAR rig move with instantaneous body-frame linear velocity $\mathbf{v}_L \in \mathbb{R}^3$ and angular velocity $\boldsymbol{\omega}_L \in \mathbb{R}^3$. For a temporal error $\Delta t(t) = t_{\text{slam}}^{\text{est}} - t_{\text{slam}}^{\text{true}}$, the induced 3D point displacement in the camera frame is:
   $$\Delta \mathbf{P}_C \approx \mathbf{R}_{LC}^\top \left( [\mathbf{P}_L]_\times \boldsymbol{\omega}_L - \mathbf{v}_L \right) \Delta t(t)$$
   Projecting onto the pixel plane via $\mathbf{J}_\pi$:
   $$\Delta \mathbf{u} \approx \mathbf{J}_\pi \mathbf{R}_{LC}^\top \left( [\mathbf{P}_L]_\times \boldsymbol{\omega}_L - \mathbf{v}_L \right) \Delta t(t)$$
   During a normal corner turn where $\|\boldsymbol{\omega}_L\| = 60^\circ\text{/s} \approx 1.05\text{ rad/s}$ and $f_x \approx 1142\text{ px}$, a temporal error of just $\Delta t = 15\text{ ms}$ ($0.45\text{ frames}$) induces a rotational pixel shift of:
   $$\|\Delta \mathbf{u}\|_{\text{rot}} \approx f_x \|\boldsymbol{\omega}_L\| \Delta t = 1142 \times 1.05 \times 0.015 \approx \mathbf{18.0\text{ pixels}}$$

2. **Quartz Oscillator Skew ($\alpha_t \neq 1$)**:
   Consumer crystal oscillators exhibit frequency tolerances of $\pm 30\text{ ppm}$ to $\pm 100\text{ ppm}$. Over $T = 900\text{ s}$, a $50\text{ ppm}$ relative clock skew ($\alpha_t = 1.00005$) accumulates a temporal drift of:
   $$\Delta t_{\text{drift}} = (\alpha_t - 1) T = 50 \times 10^{-6} \times 900\text{ s} = \mathbf{45.0\text{ ms}} \quad (\approx 1.35\text{ video frames})$$
   Thus, if $\delta t$ is calibrated at $t = 0\text{ s}$, the system is out of sync by $45\text{ ms}$ ($> 50\text{ pixels}$ during turns) by the end of the recording.

3. **CMOS Rolling Shutter Intra-Frame Distortion**:
   With a readout time $\gamma_{\text{rs}} \approx 12.4\text{ ms}$ from row $v = 0$ to $v = 3839$, the top and bottom of the fisheye circle are captured $12.4\text{ ms}$ apart. During a $60^\circ\text{/s}$ yaw turn, the bottom of the hemisphere lags the top by $\sim 15\text{ pixels}$, shearing vertical pillars into diagonal lines and making any rigid-frame $\mathbf{R}_{LC}$ unable to align the top and bottom of the image simultaneously.

---

### 6.5 Failure Mode 5: Degenerate Edge-Gradient Objective Functions
The current Tier 2 refinement minimizes the sum of squared Chamfer distances from projected 3D geometric edges $\mathcal{E}_{3D}$ to 2D image Sobel/Canny edges $\mathcal{E}_{2D}$:

$$E_{\text{edge}}(\mathbf{T}_{LC}) = \sum_{\mathbf{P}_k \in \mathcal{E}_{3D}} D_{\text{chamfer}}^2\left( \pi(\mathbf{T}_{LC}^{-1} \mathbf{T}_{WL}^{-1} \mathbf{P}_k) \right)$$

This objective function exhibits three fatal mathematical degeneracies in `HarmonyMall`:
1. **Periodic Aliasing (Floor Tiles & Escalator Grilles)**:
   Mall floors consist of periodic $60\text{ cm} \times 60\text{ cm}$ tiles. At grazing angles ($s = 5\text{ m}$), the angular pitch between adjacent tile grout lines is $\Delta \alpha \approx \frac{0.6 \cos^2(73^\circ)}{1.5} \approx 1.9^\circ$. The Chamfer distance surface $E_{\text{edge}}(\mathbf{T}_{LC})$ contains deep, periodic local minima every $1.9^\circ$ (and even tighter minima from double-line parking bay borders separated by $10\text{ cm}$, or $\Delta \alpha \approx 0.3^\circ$).
2. **Modality Asymmetry (Albedo Edges vs. Geometric Edges)**:
   Painted floor lines and store signage have strong 2D image gradients $\|\nabla I\| \gg 0$ but **zero 3D geometric depth or normal discontinuity** ($\Delta Z_W = 0$). Conversely, smooth white drywall corners have strong 3D normal discontinuities but weak 2D image gradients under uniform ambient lighting. Consequently, $E_{\text{edge}}$ forces 3D structural boundaries (curbs, wall bases) to snap onto nearby 2D painted stripes!
3. **Inability to Detect Multi-View Ghosting**:
   Edge-to-edge alignment evaluates each frame independently. It assigns a low cost as long as a 3D edge hits *some* 2D edge in each frame, completely ignoring whether the colors projected onto the flat surface between edges agree across time $t_1, t_2, \dots, t_K$ or between $\mathcal{C}_0$ and $\mathcal{C}_1$.

---

## 7. Open Challenges & Specific Mathematical Mandate for the Receiving AI Agent

To permanently solve the `LidarCamera360` calibration and colorization challenges without manual per-dataset hardcoding, the receiving AI agent is mandated to replace the brittle Tier 2 edge-gradient optimizer with a **Unified Continuous-Time Spatio-Temporal Photometric & Geometric Bundle Adjustment Framework**.

Below are the four concrete mathematical specifications and algorithmic modules to be implemented.

---

### 7.1 Mandate 1: Joint Non-Linear Spatio-Temporal Bundle Adjustment
Instead of optimizing static extrinsics against single-frame 2D edges, formulate a joint continuous-time optimization problem over the parameter state $\boldsymbol{\Phi}$:

$$\boldsymbol{\Phi} = \left\{ \underbrace{\boldsymbol{\xi}_{LC_0} \in \mathfrak{se}(3)}_{\text{Primary Rig Offset}}, \, \underbrace{\boldsymbol{\xi}_{C_0 C_1} \in \mathfrak{se}(3)}_{\text{Stereo Baseline Offset}}, \, \underbrace{\Delta \Theta_{\text{intr},0}, \Delta \Theta_{\text{intr},1}}_{\text{Equatorial Intrinsic Residual}}, \, \underbrace{\delta t, \alpha_t, \gamma_{\text{rs}}}_{\text{Temporal & RS Model}}, \, \underbrace{\{\delta \boldsymbol{\xi}_{WL,b}\}_{b=1}^B}_{\text{Spline Drift Control Knots}} \right\}$$

#### The Unified Objective Function
Minimize the composite energy functional $E_{\text{total}}(\boldsymbol{\Phi})$ using Levenberg-Marquardt with Sparse Schur Complement elimination (e.g., via `Ceres Solver` or custom CUDA/PyTorch Gauss-Newton):

$$E_{\text{total}}(\boldsymbol{\Phi}) = E_{\text{photo}}(\boldsymbol{\Phi}) + \lambda_{\text{seam}} E_{\text{seam}}(\boldsymbol{\Phi}) + \lambda_{\text{mi}} E_{\text{MI}}(\boldsymbol{\Phi}) + \lambda_{\text{prior}} E_{\text{prior}}(\boldsymbol{\Phi}) + \lambda_{\text{smooth}} E_{\text{spline}}(\boldsymbol{\Phi})$$

1. **Multi-View Direct Photometric Consistency Term ($E_{\text{photo}}$)**:
   Select a subset of high-gradient, static 3D surface patches $\mathcal{S} = \{(\mathbf{P}_{W,k}, \mathbf{n}_{W,k})\}$ (heavily sampling ground-plane markings and vertical pillars). For each point $\mathbf{P}_{W,k}$, define its local tangent plane basis $[\mathbf{e}_{1,k}, \mathbf{e}_{2,k}] \perp \mathbf{n}_{W,k}$ and a small $5 \times 5$ metric patch $\mathcal{N}(\mathbf{P}_{W,k}) = \{\mathbf{P}_{W,k} + u_p \mathbf{e}_{1,k} + v_p \mathbf{e}_{2,k}\}$. Across all frames $(i, j) \in \mathcal{O}_k$ observing patch $k$, minimize the variance of normalized patch intensities relative to the multi-view mean patch $\bar{I}_k(u_p, v_p)$:
   $$E_{\text{photo}}(\boldsymbol{\Phi}) = \sum_{k \in \mathcal{S}} \sum_{(i,j) \in \mathcal{O}_k} \sum_{(u_p, v_p)} w_{i,j,k} \, \rho_{\text{Cauchy}}\!\left( \tilde{I}_i\big(\pi_i(\mathbf{T}_{C_i W}(t_{j,v}) \, \mathbf{P}_{W,k}(u_p, v_p)); \, t_j\big) - \bar{I}_k(u_p, v_p) \right)$$
   where $\tilde{I}_i$ is zero-mean locally normalized luminance (to ensure invariance to auto-exposure changes between frames) and $t_{j,v}$ incorporates the affine clock model $\alpha_t$, offset $\delta t$, and rolling shutter row $v$.
   - *Why this works*: Unlike edge alignment, $E_{\text{photo}}$ directly penalizes ghosting and double lines! If a painted parking line is misaligned by $0.2^\circ$, its projection sweeps across the 3D patch as the operator approaches, causing massive multi-view photometric variance that generates a clean, smooth gradient pulling $\boldsymbol{\xi}_{LC_0}$, $\delta t$, and $\alpha_t$ into exact registration.

2. **Hardware Profile Mahalanobis Prior ($E_{\text{prior}}$)**:
   Anchor the physical rig parameters to the nominal hardware profile $(\bar{\mathbf{T}}_{LC_0}, \bar{\mathbf{T}}_{C_0 C_1}, \bar{\Theta}_{\text{intr}})$ loaded from `configs/profiles/raven_x4_montagem_1.json` using its prior covariance matrix $\boldsymbol{\Sigma}_{\text{prof}}$:
   $$E_{\text{prior}}(\boldsymbol{\Phi}) = \|\log_{\mathrm{SE}(3)}(\bar{\mathbf{T}}_{LC_0}^{-1} \mathbf{T}_{LC_0})\|_{\boldsymbol{\Sigma}_{LC_0}^{-1}}^2 + \|\log_{\mathrm{SE}(3)}(\bar{\mathbf{T}}_{C_0 C_1}^{-1} \mathbf{T}_{C_0 C_1})\|_{\boldsymbol{\Sigma}_{C_0 C_1}^{-1}}^2 + \|\Delta \Theta_{\text{intr}}\|_{\boldsymbol{\Sigma}_{\text{intr}}^{-1}}^2$$
   This prevents unobservable directions in degenerate corridor segments from corrupting the physical mount calibration.

3. **LiDAR Reflectivity-to-Luminance Mutual Information ($E_{\text{MI}}$)**:
   Because painted ground lines and tile joints exhibit both optical luminance contrast and 905nm/1550nm laser reflectivity contrast $I_{\text{lidar}}(\mathbf{P}_{W,k})$ without requiring 3D depth discontinuities, include a Normalized Information Distance (NID) or patch-based Normalized Cross-Correlation (NCC) term between LiDAR intensity $I_{\text{lidar},k}$ and projected image luminance $I_i(\mathbf{u}_{i,j,k})$.

---

### 7.2 Mandate 2: Multi-Cone Photometric & Normal Consistency on Ground Planes (Nadir Bird's-Eye Rectification)
To resolve **Failure Mode 1 (Grazing Angles)** and **Failure Mode 2 (Equatorial Seam)** simultaneously, implement a dedicated **Virtual Nadir Bird's-Eye View (BEV) Alignment Module**:

```
       Cam0 Hemisphere (Front)           Cam1 Hemisphere (Rear)
        \  theta in [75°, 98°]          theta in [75°, 98°]  /
         \                     \      /                     /
          \                     v    v                     /
           +-------------> [ Equator Seam ] <-------------+
                                  ||
                                  ||   Homography / LiDAR-Mesh Unwarping
                                  vv  onto Ground Patch P_ground(x, y)
                     +--------------------------+
                     | Virtual Nadir BEV Patch  |
                     |   [-4m, +4m] x [-4m, +4m]|
                     |   Resolution: 2mm / px   |
                     +--------------------------+
                      /            |            \
                     v             v             v
              [BEV_Cam0(t)]  [BEV_Cam1(t)]  [BEV_LiDAR_Intensity]
```

1. **Ground Plane Extraction**:
   For each keyframe $t_j$, extract the local ground surface mesh/plane $\mathcal{M}_{\text{ground}}(t_j)$ within a $4\text{ m} \times 4\text{ m}$ horizontal window around the operator's nadir projection $[X_{WL}(t_j), Y_{WL}(t_j)]^\top$ using the LiDAR point cloud.
2. **Virtual Nadir Projection**:
   Define a regular 2D metric grid $\mathbf{G}(u_g, v_g) \in \mathbb{R}^3$ on $\mathcal{M}_{\text{ground}}(t_j)$ at $2\text{ mm/pixel}$ resolution. Project every grid cell $\mathbf{G}(u_g, v_g)$ into both $\mathcal{C}_0$ and $\mathcal{C}_1$ to synthesize two orthorectified ground images:
   $$B_0(u_g, v_g; t_j) = I_0\big(\pi_0(\mathbf{T}_{C_0 W}(t_j)\mathbf{G}(u_g, v_g))\big), \quad B_1(u_g, v_g; t_j) = I_1\big(\pi_1(\mathbf{T}_{C_1 W}(t_j)\mathbf{G}(u_g, v_g))\big)$$
3. **Seam & Temporal Inverse-Compositional Alignment ($E_{\text{seam}}$)**:
   - **Spatial Seam Consistency ($\mathcal{C}_0 \leftrightarrow \mathcal{C}_1$ at time $t_j$)**: In the equatorial overlap strip beneath the operator where both $B_0$ and $B_1$ are valid, minimize the dense gradient-weighted photometric residual:
     $$E_{\text{seam}}^{\text{spatial}} = \sum_{t_j} \sum_{(u_g, v_g) \in \Omega_{0 \cap 1}} \rho\Big( B_0(u_g, v_g; t_j) - B_1(u_g, v_g; t_j) \Big) + \lambda_\nabla \rho\Big( \|\nabla B_0 - \nabla B_1\|_2 \Big)$$
     This directly constrains $\mathbf{T}_{C_0 C_1}$, pitch/roll of $\mathbf{T}_{LC_0}$, and the high-angle polynomial terms $(k_3, k_4, s_{x1}, s_{y1})$ at $\theta \in [80^\circ, 100^\circ]$ with sub-millimeter precision on the floor!
   - **Temporal Sweep Consistency ($t_j \leftrightarrow t_{j + \Delta j}$)**: As the operator walks forward, a floor marking at $s = +4\text{ m}$ (viewed by $\mathcal{C}_0$ at a $70^\circ$ grazing angle at time $t_1$) passes directly beneath the operator at $t_2$ ($\theta = 90^\circ$ seam) and exits behind the operator at $s = -4\text{ m}$ at time $t_3$ (viewed by $\mathcal{C}_1$ at a $70^\circ$ grazing angle). Enforcing dense alignment between $B_0(u_g, v_g; t_1)$, $B_{0/1}(u_g, v_g; t_2)$, and $B_1(u_g, v_g; t_3)$ locks the grazing-angle lever-arm pitch $\Delta \theta$ to within $< 0.01^\circ$.

---

### 7.3 Mandate 3: Elastic Trajectory Drift Decoupling
To prevent SLAM trajectory drift in large loops (`HarmonyMall`) from corrupting the rigid hardware extrinsics:

1. **Low-Frequency Elastic Pose Correction Spline**:
   Model the corrected world-to-LiDAR trajectory $\tilde{\mathbf{T}}_{WL}(t)$ as the composition of the raw SLAM trajectory $\mathbf{T}_{WL}^{\text{slam}}(t)$ and a low-frequency cubic B-spline correction $\Delta \mathbf{T}_{WL}(t) \in \mathrm{SE}(3)$ with sparse control knots spaced every $\Delta \tau_{\text{drift}} = 5.0\text{ s}$ (50 SLAM frames):
   $$\tilde{\mathbf{T}}_{WL}(t) = \Delta \mathbf{T}_{WL}(t) \cdot \mathbf{T}_{WL}^{\text{slam}}(t)$$
2. **Relative Elastic Stiffness Regularizer ($E_{\text{spline}}$)**:
   Penalize the relative twist change between consecutive drift control knots $\delta \boldsymbol{\xi}_{WL,b} \in \mathfrak{se}(3)$ so that $\Delta \mathbf{T}_{WL}(t)$ can only bend slowly to absorb accumulated SLAM drift ($1\text{--}5\text{ cm}$ over $30\text{ s}$) while high-frequency motion remains governed by $\mathbf{T}_{WL}^{\text{slam}}(t)$ and the rigid extrinsic $\mathbf{T}_{LC_0}$:
   $$E_{\text{spline}} = \sum_{b=1}^{B-1} \left\| \delta \boldsymbol{\xi}_{WL,b+1} \ominus \delta \boldsymbol{\xi}_{WL,b} \right\|_{\boldsymbol{\Sigma}_{\text{odom}}^{-1}}^2 + \lambda_{\text{zero}} \sum_{b=1}^B \|\delta \boldsymbol{\xi}_{WL,b}\|_2^2$$
3. **Alternating Gauge-Fixed Optimization Schedule**:
   - **Phase 1 (Short High-Observability Segments)**: Automatically select $3\text{--}5$ short ($10\text{ s}$) trajectory segments exhibiting rich 3D structure and rotational excitation $\int \|\boldsymbol{\omega}_L\|_2 \mathrm{d}t$, where local SLAM drift over $10\text{ s}$ is negligible ($\delta \boldsymbol{\xi}_{WL} \equiv \mathbf{0}$). Optimize global $\mathbf{T}_{LC_0}$, $\mathbf{T}_{C_0 C_1}$, and $(\delta t, \alpha_t, \gamma_{\text{rs}})$.
   - **Phase 2 (Full-Trajectory Elastic Relaxation)**: Freeze the physical rig extrinsics $(\mathbf{T}_{LC_0}, \mathbf{T}_{C_0 C_1})$ and intrinsics $\Theta_{\text{intr}}$, and optimize the elastic trajectory spline knots $\{\delta \boldsymbol{\xi}_{WL,b}\}_{b=1}^B$ across the entire $688\text{ m}$ path to eliminate loop-closure ghosting and grazing-angle floor shear.

---

### 7.4 Mandate 4: Automated Zero-Human-in-the-Loop Evaluation Benchmarks
The receiving AI agent must implement an automated diagnostic verification suite (`scripts/evaluate_calibration_metrics.py`) that outputs a structured JSON report and gates pipeline completion without human visual inspection.

```
+------------------------------------------------------------------------+
|           AUTOMATED CALIBRATION VERIFICATION GATE (PASS/FAIL)          |
+------------------------------------------------------------------------+
  |-- [Metric 1] Multi-View Photometric Variance (sigma_RGB)  < 6.5 intensity
  |-- [Metric 2] Equatorial Seam NCC Score (C0 <-> C1 BEV)    > 0.92
  |-- [Metric 3] Ground Line MTF / Edge Entropy Ratio         > 1.35x baseline
  |-- [Metric 4] Temporal Bidirectional Consistency Residual  < 0.75 px
+------------------------------------------------------------------------+
```

#### Quantitative Benchmark Definitions

1. **Multi-View Color Consistency Variance ($\sigma_{\text{RGB}}^2$)**:
   Computed across all static ground and wall points observed by $\ge 4$ distinct camera frames spanning $\ge 3\text{ m}$ of trajectory travel:
   $$\text{MV-RMSE} = \sqrt{ \frac{1}{|\mathcal{P}_{\text{eval}}|} \sum_{k \in \mathcal{P}_{\text{eval}}} \frac{\sum_{(i,j) \in \mathcal{O}_k} w_{i,j,k} \| I_i(\mathbf{u}_{i,j,k}; t_j) - \mathbf{c}_k \|_2^2}{\sum_{(i,j) \in \mathcal{O}_k} w_{i,j,k}} }$$
   - **Acceptance Threshold**: $\text{MV-RMSE} < 6.5$ (on an 8-bit $[0, 255]$ scale).

2. **Nadir Equatorial Seam Normalized Cross-Correlation ($\text{NCC}_{\text{seam}}$)**:
   Evaluates the structural alignment between the orthorectified ground patches $B_0(u_g, v_g; t_j)$ and $B_1(u_g, v_g; t_j)$ in the overlap zone across $N_{\text{eval}} = 50$ uniformly sampled frames:
   $$\text{NCC}_{\text{seam}} = \frac{1}{N_{\text{eval}}} \sum_{j=1}^{N_{\text{eval}}} \frac{\sum_{(u_g, v_g)} (B_0 - \bar{B}_0)(B_1 - \bar{B}_1)}{\sqrt{\sum (B_0 - \bar{B}_0)^2 \sum (B_1 - \bar{B}_1)^2}}$$
   - **Acceptance Threshold**: $\text{NCC}_{\text{seam}} \ge 0.92$ (values $< 0.75$ indicate seam shearing or baseline calibration failure).

3. **Ground-Plane Edge Sharpness / Gradient Entropy Metric ($S_{\text{ground}}$)**:
   Project the final colored point cloud $\mathcal{P}_W$ onto a high-resolution ($2\text{ mm/px}$) 2D Nadir orthomosaic image $M_{\text{ortho}}(x, y)$ and compute the spatial gradient kurtosis / Shannon entropy of edge magnitudes:
   $$S_{\text{ground}} = \frac{\mathbb{E}\left[\|\nabla M_{\text{ortho}}\|_2^4\right]}{\left(\mathbb{E}\left[\|\nabla M_{\text{ortho}}\|_2^2\right]\right)^2}$$
   Blurred or doubled floor lines flatten the gradient distribution (low kurtosis), whereas crisp single-edge convergence maximizes kurtosis.
   - **Acceptance Criterion**: Monotonic increase in $S_{\text{ground}}$ relative to the nominal hardware profile initialization, with zero double-peak autocorrelation within a $15\text{ cm}$ radius on detected linear ground markings.

4. **Forward-Backward Pass Discrepancy ($\Delta_{\text{fwd-bwd}}$)**:
   Using optical flow or SuperPoint/LightGlue feature tracks between $\mathcal{C}_0(t_1)$ (approaching a ground feature) and $\mathcal{C}_1(t_2)$ (receding from the same ground feature), measure the 3D ground-plane intersection distance between the unprojected rays:
   $$\Delta_{\text{fwd-bwd}} = \text{median}_{(m)} \left\| \mathbf{P}_{\text{ground}}\big(\pi_0^{-1}(\mathbf{u}_{0,m}, t_1)\big) - \mathbf{P}_{\text{ground}}\big(\pi_1^{-1}(\mathbf{u}_{1,m}, t_2)\big) \right\|_2$$
   - **Acceptance Threshold**: $\Delta_{\text{fwd-bwd}} \le 0.008\text{ m}$ ($8\text{ mm}$ on the ground plane at $3\text{ m}$ distance).

---

### 7.5 Summary of Immediate Action Items for the Receiving Agent
1. **Audit & Fix $\text{FOV} > 180^\circ$ Projection**: Replace any occurrences of `x = X/Z; theta = atan(r)` in the projection and Jacobian routines with the singularity-free spherical formulation $\theta = \operatorname{atan2}(\sqrt{X^2+Y^2}, Z)$ (Section 3.2).
2. **Implement Affine + Rolling Shutter Time Mapping**: Upgrade the scalar $\delta t$ lookup to $t_{\text{slam}}(t_{\text{vid}}, v) = t_{\text{slam},0} + \alpha_t(t_{\text{vid}} - \delta t) + \gamma_{\text{rs}}\frac{v - H/2}{H}$ over a cumulative $\mathrm{SE}(3)$ cubic B-spline trajectory (Sections 2.4 & 4.1).
3. **Implement Nadir BEV Seam & Grazing Optimizer**: Build the virtual ground patch projector $B_0, B_1$ and optimize $E_{\text{photo}} + \lambda_{\text{seam}} E_{\text{seam}} + \lambda_{\text{prior}} E_{\text{prior}}$ (Sections 7.1 & 7.2).
4. **Decouple SLAM Drift**: Introduce the two-phase alternating solver separating static hardware profile refinement from the low-frequency elastic trajectory spline $\Delta \mathbf{T}_{WL}(t)$ (Section 7.3).
5. **Validate on Both Regimes**: Run the automated verification benchmark suite (Section 7.4) on both `20261022home` (ensuring zero regression) and `HarmonyMall` (verifying elimination of floor line doubling and equatorial seam shear).

=================================================================
[Argon Telemetry]: In: 1,825 tokens | Cached: 0 tokens | Out+Think: 20,743 tokens
[Credit Billing]: Est. Deducted from Dev Credit: $0.21108 USD
=================================================================
