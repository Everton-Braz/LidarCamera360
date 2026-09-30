# Auxiliary Camera & Third Source Integration (Smartphone / Third Camera)

## Overview & Architecture

RavenCalibrator natively processes synchronized dual-fisheye imagery from the Insta360 X4 camera paired with 3D LiDAR scanners (Vanjee 722z / Raven or Livox Mid-360 / Eagle). While dual 180°+ fisheye lenses provide complete 360° spherical coverage, high-detail photogrammetry and 3D Gaussian Splatting (3DGS) benefit substantially from supplementary high-resolution, narrow-angle rectilinear (pinhole) perspectives.

The **Multiple Image Source / Third Camera** feature enables attaching a secondary sensor (e.g., a forward-facing smartphone, GoPro/action camera, or DSLR) to the rig. The pipeline automatically:
1. Extracts sharp, motion-filtered frames into `dataset/images/cam2/`.
2. Estimates or refines the rigid mounting extrinsics ($T_{\text{lidar}\to\text{cam2}}$) relative to the primary sensor rig.
3. Synthesizes continuous 6DoF metric camera trajectories along the FAST-LIVO2 SLAM motion path.
4. Injects Camera 3 and its synthesized poses into the COLMAP 3DGS dataset (`cameras.txt`, `images.txt`).

---

## Hardware Mounting & Nominal Lever Arm

In a typical scanning rig configuration, a smartphone or action camera is mounted facing forward directly above or below the primary 360° camera:

```
         [360° Camera (Insta360 X4)]
                     |
         [Third Camera (Smartphone)] (cam2)
                     |  +15 cm (forward, Y) / +20 cm (up, Z)
          [LiDAR Scanner (Raven/Eagle)]
```

### Coordinate Frame Convention (LiDAR Body Frame):
- **+X**: Right
- **+Y**: Forward
- **+Z**: Up
- **Nominal Lever Arm**: $T_{\text{lidar}\to\text{cam2}} = \begin{bmatrix} 1 & 0 & 0 & 0.00 \\ 0 & 1 & 0 & +0.15 \\ 0 & 0 & 1 & +0.20 \\ 0 & 0 & 0 & 1 \end{bmatrix}$ (meters)

---

## Configuration Schema (`configs/third_camera.json`)

The third camera configuration can be loaded or saved as a standard JSON file:

```json
{
  "camera_name": "cam2",
  "source_path": "path/to/auxiliary_video.mp4",
  "source_type": "video",
  "enabled": true,
  "fps": 2.0,
  "time_offset_s": 0.0,
  "camera_model": "PINHOLE",
  "intrinsics": {
    "width": 1920.0,
    "height": 1080.0,
    "fx": 1536.0,
    "fy": 1536.0,
    "cx": 960.0,
    "cy": 540.0,
    "k1": 0.0,
    "k2": 0.0,
    "p1": 0.0,
    "p2": 0.0
  },
  "T_lidar_to_cam2_rigid_4x4": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.15],
    [0.0, 0.0, 1.0, 0.20],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "description": "Frontal smartphone camera (auxiliary image/video source for 3DGS enhancement)"
}
```

---

## Pipeline Execution Workflow

### 1. Frame Extraction (`extract_third_camera_frames`)
- Extracts frames at the configured FPS (default `2.0`).
- Applies a 3-frame rolling Laplacian sharpness filter (`compute_laplacian_sharpness`) to discard motion-blurred frames.
- Stores frames as `images/cam2/frame_000001.jpg` with high JPEG quality (`95`).
- Appends relative presentation timestamps (PTS in seconds) to `images/frames.json`.

### 2. Extrinsics Alignment (`align_third_camera_to_rig`)
- **Strategy 1 (SfM Bundle Adjustment)**: If multi-camera SfM reconstructed `cam2` images, extrinsics are solved from the Sim(3) alignment with the LiDAR trajectory.
- **Strategy 2 (2D-2D SIFT Feature Matching)**: Extracts SIFT feature matches between `cam0` (primary front fisheye) and `cam2` (auxiliary camera), estimates the Essential Matrix $E$ with RANSAC, and recovers relative rotation $R_{02}$ and translation direction $t_{02}$.
- **Fallback**: Preserves nominal/calibrated rig lever arm vector.

### 3. Trajectory Pose Synthesis (`synthesize_third_camera_poses`)
- Evaluates the query timestamp $t_q = t_{\text{start}} + (t_{\text{video}} - (\Delta t_{\text{sync}} + \Delta t_{\text{offset}}))$.
- Interpolates LiDAR position via linear weighting and orientation via Slerp quaternions.
- Computes camera world position $C_{\text{world}} = p_L + R_L \cdot t_{LC2}$ and rotation $R_{\text{cw}} = (R_L \cdot R_{LC2})^T$.

### 4. COLMAP & 3DGS Injection (`inject_third_camera_into_colmap`)
- Injects Camera 3 into `sparse/0/cameras.txt`:
  ```
  3 PINHOLE 1920 1080 1536.0 1536.0 960.0 540.0
  ```
- Appends 6DoF camera poses and image entries for `cam2/frame_*.jpg` into `sparse/0/images.txt`.
- Fully recognized by Spirula Studio, Gaussian Splatting, and Nerfstudio without manual conversion.

---

## How to Use

### Via Desktop GUI (Unified Studio)
1. Open RavenCalibrator GUI:
   ```powershell
   python -m raven_app.cli gui
   ```
2. Navigate to **Unified Studio** (`unified_workflow_view.py`).
3. Under **1. Input Datasets & Destination**, click **"Add Source Image/Video..."**.
4. In the dialog:
   - Select your auxiliary video (`.mp4`, `.mov`) or image directory.
   - The dialog automatically detects resolution, duration, and frame rate, and populates default smartphone intrinsics (~68° HFOV).
   - Adjust extraction FPS (default 2.0 fps) and time offset if needed.
   - Click **Confirm**.
5. Click **"Run Unified Pipeline"**. The workflow automatically extracts, aligns, and injects `cam2` into the deliverables.

### Via Command-Line Interface (CLI)
You can provide either a JSON configuration file or directly point to a video file:

```powershell
# Using custom configuration JSON
python -m raven_app.cli workflow `
    --bag path/to/scan.bag `
    --insv path/to/video.insv `
    --output path/to/output_dir `
    --third-camera configs/third_camera.json

# Directly pointing to an auxiliary smartphone video
python -m raven_app.cli workflow `
    --bag path/to/scan.bag `
    --insv path/to/video.insv `
    --output path/to/output_dir `
    --third-camera path/to/smartphone_video.mp4
```
