# AprilTag-Based Camera–LiDAR Calibration for Raven + Insta360 X4

This document is a **guide for an LLM coding agent** that must implement an AprilTag-based extrinsic calibration between the Raven LiDAR scanner and the Insta360 X4 camera mounted on top of it. The goal is to estimate a precise rigid transform between the LiDAR frame and the camera frame, optionally including basic temporal calibration, using **AprilTags placed on walls and around the scene**.

The agent must **inspect the real project codebase and data layout first**, then design code that works with:

- Raven LiDAR point clouds (ROS or file-based).
- Insta360 X4 images/video (decoded to frames).
- AprilTag detections in camera images.
- AprilTag-related structures that can be detected in LiDAR.

The implementation should follow the principles from recent target-based camera–LiDAR calibration work that uses fiducial markers (AprilTags) and custom LiDAR targets.[web:90][web:92][web:93][web:96]

---

## 0. Overall Objectives

1. **Geometric (extrinsic) calibration**: Estimate the 3D rigid transform \(T_{\text{Lidar}\rightarrow\text{Cam}}\) between the Raven LiDAR frame and the Insta360 X4 camera frame, using AprilTag-based 2D–3D correspondences.
2. **Optional temporal calibration**: Estimate a constant time offset between LiDAR and camera streams by tracking a moving AprilTag target.[web:90][web:92]
3. **Validation**: Project LiDAR points onto camera images using the estimated transform and visually/quantitatively confirm correct alignment.
4. **Reusability**: Provide calibration outputs that can be consumed by existing colourisation and mapping code (e.g., ROS TF, YAML, JSON, COLMAP adapters).

The agent must **not** change existing SLAM/colourisation pipelines unless explicitly requested. Calibration is an independent module.

---

## 1. Inspect the Existing Project First

Before writing calibration code, the agent must produce an `APRILTAG_CALIBRATION_AUDIT.md` that answers:

1. **Codebase**
   - Languages used (Python, C++, ROS/ROS2 packages, etc.).
   - Where LiDAR and camera data are currently read and written.
   - Existing AprilTag or fiducial libraries in use (e.g., `apriltag`, `apriltag_ros`, `AprilTag_ROS`, `srrg2_apriltag_calibration`).[web:89][web:91][web:98]

2. **Data interfaces**
   - LiDAR data format: ROS `sensor_msgs/PointCloud2`, `.pcd`, `.ply`, `.las`, etc.
   - Camera data format: ROS `sensor_msgs/Image`, exported JPG/PNG, frame folder, or `.mp4/.insv` video.
   - How Insta360 frames are generated (cube-map faces, perspective views, equirectangular panoramas).

3. **Coordinate frames and conventions**
   - Current naming and hierarchy of frames (e.g., `raven_lidar`, `insta360_cam`, `world`, `odom`).
   - Axes conventions (right-hand system, units in meters).

4. **Available calibration utilities**
   - Any existing camera intrinsics calibration.
   - Any existing LiDAR–camera transform approximations.

5. **AprilTag configuration**
   - Whether tag IDs, sizes, and layout (positions on walls) are known or documented.
   - Whether there is an AprilTag bundle YAML or similar file.[web:102]

The agent must not assume missing pieces; identify them and, where needed, propose simple data formats to fill them.

---

## 2. Physical Calibration Setup with AprilTags

The code must be designed for the following physical setup:

1. **Rigid sensor rig**
   - Raven LiDAR scanner and Insta360 X4 mounted on a common rigid bracket (as in the provided photo), so their relative pose is fixed.

2. **AprilTags in the environment**
   - Print AprilTags of known size (e.g., 8–12 cm) and mount them on walls, poles, or planar boards at various heights and orientations.
   - Optionally create **calibration boards** that combine:
     - An AprilTag for camera detection.
     - A LiDAR marker (e.g., retroreflective tape strips in a cross pattern) so the target center can be detected in the LiDAR point cloud.[web:90][web:92]

3. **Calibration motion**
   - Perform one or more calibration sweeps:
     - Stand the rig still and capture several static frames (LiDAR + camera) with tags visible.
     - Move slowly around the scene so the rig sees tags from multiple viewpoints and distances.

4. **Assumptions**
   - LiDAR and camera have overlapping fields of view.
   - AprilTags are visible in camera images with sufficient resolution.
   - LiDAR can see the calibration boards or at least the planar surfaces where tags are mounted.

---

## 3. Camera Intrinsic Calibration with AprilTags

If the camera intrinsics are not yet calibrated, the agent should:

1. Use an AprilTag grid or custom planar AprilTag pattern to calibrate the Insta360 X4’s virtual perspective views.[web:101]
2. Steps:
   - Capture calibration images where the AprilTag pattern fills most of the field of view at various angles and distances.[web:101]
   - Detect AprilTag corners with a suitable detector (`apriltag` or `detectAprilGrid`).[web:98][web:101]
   - Generate world points for the pattern (known tag layout and size).[web:101]
   - Estimate camera parameters (focal lengths, principal point, distortion coefficients) using a standard camera calibration routine.
3. Store results in:
   - ROS `camera_info` messages.
   - YAML/JSON config used by the calibration pipeline.

If intrinsics are already correctly estimated by another method, this step can be skipped, but intrinsics must be available to the AprilTag pose estimator.

---

## 4. LiDAR Target Detection with AprilTag Boards

To build accurate 3D–2D correspondences, the code must detect **the same physical target** in both LiDAR and camera.[web:90][web:92]

### 4.1 Calibration board design

Follow the approach in Grammatikopoulos et al. (2022):[web:90][web:92]

- A planar board with:
  - A small AprilTag at the center (≈3 cm).[web:92]
  - Two orthogonal reflective stripes crossing at the center.

This design allows:

- Precise detection of the board center in LiDAR (intersection of stripes).[web:92]
- Precise detection of the AprilTag center in camera (AprilTag pose).[web:92]

### 4.2 Detecting board center in LiDAR

For each LiDAR scan used for calibration:

1. Extract points belonging to reflective stripes;
   - Use intensity thresholding if LiDAR has reflectivity, or region selection based on approximate board location.
   - Fit two lines to stripe point sets; compute their intersection as the board center.[web:90][web:92]

2. Represent the board center in LiDAR frame as \(P_{\text{LiDAR}} = (X, Y, Z)\).

### 4.3 Detecting AprilTag center in camera

For each camera image:

1. Run AprilTag detection (e.g., `apriltag_ros` or C++ AprilTag library).[web:91][web:98]
2. For each detected tag:
   - Get its ID, corner coordinates, and estimated 6D pose given camera intrinsics and tag size (AprilTag pose estimation API).[web:98]
   - Compute the tag center in image coordinates (average of four corners).
   - Optionally compute the tag center in camera coordinates from the pose (translation vector), giving \(P_{\text{Cam}}\) in 3D.

3. Match tags in camera images to boards in LiDAR based on tag ID or board position.

---

## 5. Building 2D–3D (or 3D–3D) Correspondences

The calibration algorithm requires multiple correspondences between LiDAR and camera observations of the same targets.[web:90][web:92][web:93][web:96]

### 5.1 3D–2D correspondences

Preferred formulation:

- Use LiDAR board center as **3D point** \(P_{\text{LiDAR}}\).
- Use AprilTag center in image as **2D pixel** \(p_{\text{Cam}} = (u, v)\).

For each calibration sample \(i\):

- \(P_{\text{LiDAR}, i}\) from LiDAR.
- \(p_{\text{Cam}, i}\) from camera.

### 5.2 3D–3D correspondences

Alternative formulation:

- Use LiDAR board center as **3D point in LiDAR frame**.
- Use AprilTag pose estimate as **3D point in camera frame** \(P_{\text{Cam}, i}\).

Then solve for the rigid transform \(T_{\text{LiDAR}\rightarrow\text{Cam}}\) that minimizes 3D–3D alignment error across many samples.

### 5.3 Data association

The agent must implement a robust data association step that:

- Matches board instances across LiDAR and camera by:
  - Tag ID.
  - Approximate spatial grouping.
- Rejects outliers where detection is poor or inconsistent.

At least **10–20** good correspondences from different viewpoints and distances are recommended for robust calibration.[web:90][web:93][web:96]

---

## 6. Solving for Extrinsics (LiDAR→Camera)

Once correspondences are built, solve for the 3D rigid transform using established methods.[web:90][web:92][web:96]

### 6.1 3D–2D PnP approach

If using 3D LiDAR points and 2D image pixels:

1. For each sample:
   - 3D: \(P_{\text{LiDAR}}\).
   - 2D: \(p_{\text{Cam}}\).

2. Use camera intrinsics and a PnP solver to estimate extrinsics that minimize reprojection error:

\[
\text{minimize} \sum_i \|\pi(K, T_{\text{LiDAR}\rightarrow\text{Cam}}, P_{\text{LiDAR}, i}) - p_{\text{Cam}, i}\|^2,
\]

where \(\pi\) is the projection function, \(K\) camera intrinsics.[web:90][web:92][web:96]

3. Use RANSAC-PnP or robust optimization to reject mismatched samples.[web:96]

### 6.2 3D–3D alignment approach

If using both LiDAR and camera 3D centers:

1. For each sample:
   - 3D LiDAR: \(P_{\text{LiDAR}, i}\).
   - 3D camera: \(P_{\text{Cam}, i}\).

2. Solve for rotation \(R\) and translation \(T\) minimizing:

\[
\text{minimize} \sum_i \|R P_{\text{LiDAR}, i} + T - P_{\text{Cam}, i}\|^2.
\]

3. Use closed-form methods (Kabsch/Umeyama) with outlier rejection.[web:96]

### 6.3 Implementation requirements

The agent must:

- Implement calibration as an offline script or node that reads a calibration bag or data file.
- Output \(T_{\text{LiDAR}\rightarrow\text{Cam}}\) as:
  - A 4×4 homogeneous transform matrix (YAML/JSON).
  - Translation vector + unit quaternion.
- Provide a CLI, e.g.:

```bash
rosrun raven_calib apriltag_lidar_calib \
  --bag calibration.bag \
  --tag-config config/tags.yaml \
  --out config/lidar_to_camera.yaml
```

Or a Python entry point if not using ROS.

---

## 7. Optional Temporal Calibration

If LiDAR and camera data streams are not perfectly synchronized, a constant **time offset** can be estimated using AprilTag motion.[web:90][web:92]

### 7.1 Moving target method

1. Record a sequence where a calibration board moves steadily through the scene.
2. For each timestamp:
   - Detect AprilTag center in image.
   - Compute 3D board center in LiDAR frame and project it into the image using current extrinsics and candidate time offset.
3. Adjust the time offset to minimize the difference between projected board position and detected AprilTag position over time.

### 7.2 Implementation notes

- Temporal calibration should be optional and gated by sufficient moving target data.
- The agent must report the estimated time offset and residual alignment error.

---

## 8. Validation and Visualization

The agent must provide tools to validate the calibration:[web:94][web:97][web:99]

1. **Reprojection visualization**
   - Take a LiDAR scan and corresponding camera image.
   - Transform LiDAR points to camera frame and project them into the image.
   - Overlay projected points and AprilTag detections and display or save visualization.

2. **Quantitative metrics**
   - Mean and median reprojection error in pixels.
   - 3D alignment error (for 3D–3D methods).

3. **Diagnostic outputs**
   - Highlight which correspondences were used and which rejected.
   - Save intermediate correspondences for debugging.

---

## 9. Integration with Colourisation and Mapping

Once calibration is complete:

1. Export \(T_{\text{LiDAR}\rightarrow\text{Cam}}\) to configuration files used by:
   - ROS TF tree.
   - Colourisation node that projects LiDAR points into camera images.
   - COLMAP/pose2colmap adapters.

2. Ensure that subsequent LiDAR colourisation and mapping code reads the new transform and uses it consistently with existing coordinate frames.

3. DO NOT modify SLAM algorithms or COLMAP reconstruction unless explicitly requested; extrinsics are an external input.

---

## 10. Implementation Checklist for the LLM Agent

1. Generate `APRILTAG_CALIBRATION_AUDIT.md` based on repository inspection.
2. Implement camera intrinsic calibration using AprilTag patterns if needed.[web:101]
3. Implement LiDAR board center detection using reflective stripes or planar segmentation.[web:90][web:92]
4. Integrate AprilTag detection on camera images using an appropriate library or ROS package.[web:91][web:98]
5. Build robust 3D–2D or 3D–3D correspondences between LiDAR and camera targets.[web:90][web:92][web:96]
6. Solve for \(T_{\text{LiDAR}\rightarrow\text{Cam}}\) using PnP or Kabsch/Umeyama.[web:90][web:96]
7. Implement optional temporal calibration using moving calibration board.[web:92]
8. Implement visualization tools to validate reprojection and alignment.[web:94][web:97]
9. Write calibration outputs (YAML/JSON matrix, ROS TF config) and integrate with existing colourisation code.
10. Document the exact calibration procedure and commands in the project README.

The agent should favour clarity, modularity, and robust error checking. It must never assume synthetic AprilTag placements or LiDAR visibility; all assumptions should be checked against the actual SMALL-DATASET-TEST and future calibration datasets.
