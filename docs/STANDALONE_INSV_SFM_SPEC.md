# Implementation Specification: Standalone INSV Frame Extraction & Embedded Thin-Prism Fisheye SfM

**Target Codebase:** `C:\Users\Everton-PC\Documents\APLICATIVOS\Lidar-camera-calibrator`  
**Reference Codebase:** `D:\APLICATIVOS\spirula-studio`  
**Document Goal:** Provide autonomous LLM agents with an exhaustive, copy-paste ready technical blueprint to eliminate external runtime dependencies (`ffmpeg.exe` and external `spirula.exe`) and achieve a fully standalone, portable Windows application.

---

## Table of Contents
1. [Executive Overview & Requirements](#1-executive-overview--requirements)
2. [Mathematical Specification: Thin-Prism Fisheye Camera Model](#2-mathematical-specification-thin-prism-fisheye-camera-model)
3. [INSV Multi-Track Architecture & Sharpness Selection](#3-insv-multi-track-architecture--sharpness-selection)
4. [Approach A: Pythonic In-Process PyAV + Bundled Vulkan SfM Subprocess](#4-approach-a-pythonic-in-process-pyav--bundled-vulkan-sfm-subprocess)
5. [Approach B: Full Native C++ Engine Integration (CMake + C-API)](#5-approach-b-full-native-c-engine-integration-cmake--c-api)
6. [Comparative Trade-Off Analysis](#6-comparative-trade-off-analysis)
7. [Step-by-Step Implementation Guide for LLM Agents](#7-step-by-step-implementation-guide-for-llm-agents)
8. [Verification, Testing & Acceptance Criteria](#8-verification-testing--acceptance-criteria)

---

## 1. Executive Overview & Requirements

### 1.1 The Problem
Currently, `Lidar-camera-calibrator`:
1. Shells out to an external `ffmpeg.exe` binary via `subprocess.run` in `raven_app/workflow.py` to extract video frames from `.insv` files. If FFmpeg is missing from `PATH`, extraction fails completely.
2. Shells out to an external `spirula.exe` executable in `scripts/pipeline_auto_calibrator_and_colorizer.py` to run SfM. If Spirula Studio is not installed in a sibling directory or PATH, Method 1 (SfM gold standard) cannot run.

### 1.2 Target Objectives
1. **Zero External CLI Dependencies:** The packaged executable (`RavenCalibrator.exe`) must run on a clean Windows machine without needing external FFmpeg or an independent Spirula installation.
2. **Dual-Fisheye Stream Extraction:** Extract `0:v:0` (front lens `cam0`) and `0:v:1` (rear lens `cam1`) directly from `.insv` containers with intelligent blur rejection (Laplacian variance sharpness scoring).
3. **Thin-Prism Fisheye SfM:** Reconstruct high-precision camera poses using COLMAP camera model ID 10 (`THIN_PRISM_FISHEYE`, 12 parameters) using unit viewing rays (*bearings*) on $\mathbb{S}^2$.

---

## 2. Mathematical Specification: Thin-Prism Fisheye Camera Model

The camera model implemented in Spirula Studio (`src/sfm/core/Camera.h`) and COLMAP model ID `10` is defined by 12 intrinsic parameters:
$$\mathbf{\theta} = [f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, s_{x1}, s_{y1}]$$

### 2.1 Forward Projection (3D Point in Camera Frame $\mathbf{P}_c \to$ Pixel $(u, v)$)
Given $\mathbf{P}_c = [x, y, z]^T$:
1. Compute radial distance in the $xy$-plane and incidence angle $\theta$:
   $$r = \sqrt{x^2 + y^2}, \quad \theta = \text{atan2}(r, z)$$
2. Compute normalized equidistant coordinates $(u_f, v_f)$ where $\|(u_f, v_f)\| = \theta$:
   $$u_f = \begin{cases} \theta \frac{x}{r}, & r > 10^{-12} \\ 0, & r \le 10^{-12} \end{cases}, \quad v_f = \begin{cases} \theta \frac{y}{r}, & r > 10^{-12} \\ 0, & r \le 10^{-12} \end{cases}$$
3. Compute Kannala-Brandt 8th-order radial polynomial:
   $$r_f^2 = \theta^2$$
   $$\text{radial} = r_f^2 \cdot \left(k_1 + r_f^2 \cdot \left(k_2 + r_f^2 \cdot \left(k_3 + r_f^2 \cdot k_4\right)\right)\right)$$
4. Compute tangential distortion ($p_1, p_2$) and thin-prism distortion ($s_{x1}, s_{y1}$):
   $$du = u_f \cdot \text{radial} + 2 p_1 u_f v_f + p_2 (r_f^2 + 2 u_f^2) + s_{x1} r_f^2$$
   $$dv = v_f \cdot \text{radial} + p_1 (r_f^2 + 2 v_f^2) + 2 p_2 u_f v_f + s_{y1} r_f^2$$
5. Map to sensor pixels:
   $$u = f_x (u_f + du) + c_x, \quad v = f_y (v_f + dv) + c_y$$

### 2.2 Backward Unprojection to Unit Viewing Rays (*Bearings*) $\mathbf{b} \in \mathbb{S}^2$
> [!IMPORTANT]
> Because the Insta360 fisheye lens has a field of view of $\approx 200^\circ$, rays past $90^\circ$ have $z \le 0$. Standard pinhole normalization $(x/z, y/z)$ diverges to infinity at $90^\circ$ and inverts sign. All two-view verification, triangulation, and PnP must operate on 3D unit bearing vectors $\mathbf{b}$.

Algorithm for $(u, v) \to \mathbf{b} = [b_x, b_y, b_z]^T$:
1. Normalize by focal length and principal point:
   $$\bar{u} = \frac{u - c_x}{f_x}, \quad \bar{v} = \frac{v - c_y}{f_y}$$
2. Fixed-point / 1D Newton-Raphson iteration (up to 6 iterations):
   - Initialize distortion corrections $dt_x = 0, dt_y = 0$.
   - For iteration $k = 0 \dots 5$:
     $$u' = \bar{u} - dt_x, \quad v' = \bar{v} - dt_y, \quad r_d = \sqrt{u'^2 + v'^2}$$
     If $r_d < 10^{-12}$, then $\theta = 0, u_f = 0, v_f = 0$, break.
     Invert Kannala-Brandt radial profile $\theta_d \to \theta$ using 1D Newton:
     $$\theta_{i+1} = \theta_i - \frac{\theta_i (1 + k_1 \theta_i^2 + k_2 \theta_i^4 + k_3 \theta_i^6 + k_4 \theta_i^8) - r_d}{1 + 3k_1 \theta_i^2 + 5k_2 \theta_i^4 + 7k_3 \theta_i^6 + 9k_4 \theta_i^8}$$
     Compute $u_f = \theta \frac{u'}{r_d}, v_f = \theta \frac{v'}{r_d}, r_f^2 = \theta^2$.
     Update corrections:
     $$dt_x^{\text{new}} = 2 p_1 u_f v_f + p_2 (r_f^2 + 2 u_f^2) + s_{x1} r_f^2$$
     $$dt_y^{\text{new}} = p_1 (r_f^2 + 2 v_f^2) + 2 p_2 u_f v_f + s_{y1} r_f^2$$
     If $|dt_x^{\text{new}} - dt_x| < 10^{-12}$ and $|dt_y^{\text{new}} - dt_y| < 10^{-12}$, break.
     $dt_x = dt_x^{\text{new}}, dt_y = dt_y^{\text{new}}$.
3. Compute unit 3D bearing ray on the unit sphere:
   $$\mathbf{b} = \begin{bmatrix} \sin\theta \frac{u_f}{\theta} \\ \sin\theta \frac{v_f}{\theta} \\ \cos\theta \end{bmatrix}, \quad \|\mathbf{b}\|_2 = 1$$
   *(Note that $\cos\theta < 0$ naturally handles rays in the backward hemisphere).*

---

## 3. INSV Multi-Track Architecture & Sharpness Selection

### 3.1 Container Demuxing Logic
An `.insv` file contains two distinct H.264/H.265 video tracks inside its ISO-BMFF container:
- Track index 0 (`0:v:0`): Front Lens (`cam0`)
- Track index 1 (`0:v:1`): Rear Lens (`cam1`)

### 3.2 Sharp Frame Selection Algorithm
To avoid motion blur from vehicle/handheld motion, implement the exact metric from Spirula Studio (`src/app/gui/FrameSelect.cpp`):
1. Downsample the candidate image to $512 \times 512$ using area box-averaging.
2. Convert to luminance using BT.601 weights: $Y = 0.299 R + 0.587 G + 0.114 B$.
3. Zero-center the luminance: $Y \leftarrow Y - \text{mean}(Y)$.
4. Apply discrete $3 \times 3$ Laplacian filter:
   $$\Delta Y(x, y) = Y(x+1, y) + Y(x-1, y) + Y(x, y+1) + Y(x, y-1) - 4 Y(x, y)$$
5. Calculate sample variance of $\Delta Y$:
   $$\text{Score} = \text{Var}(\Delta Y) = \frac{1}{N} \sum (\Delta Y)^2 - \left(\frac{1}{N} \sum \Delta Y\right)^2$$
6. Across each window of $W$ candidate frames (e.g. $W = \text{round}(\text{fps}_{\text{native}} / \text{fps}_{\text{target}})$), write only the candidate with the highest variance.

---

## 4. Approach A: Pythonic In-Process PyAV + Bundled Vulkan SfM Subprocess

**Rationale:** Fastest to implement, highest reliability, zero build toolchain conflicts with MSVC/PCL/Vulkan. PyAV (`av`) vendors FFmpeg C libraries inside its wheels, requiring NO system `ffmpeg.exe`. Spirula's headless CLI is compiled once into a compact Vulkan binary and bundled by PyInstaller.

### 4.1 Dependency Updates
Add to `requirements.txt` and `requirements-build.txt`:
```txt
av>=11.0.0
```

### 4.2 INSV Decoder Implementation in `raven_app/workflow.py`
Replace the external `subprocess.run([ff_bin, ...])` in `extract_insv_frames` with the following:

```python
import av
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

def compute_laplacian_sharpness(img_bgr: np.ndarray) -> float:
    """
    Computes frame sharpness matching Spirula's FrameSelect.cpp:
    512x512 box-downsampling, BT.601 grayscale, 3x3 Laplacian variance.
    """
    small = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= np.mean(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    return float(np.var(lap))

def extract_insv_frames_pyav(
    insv_path: Path,
    output_dir: Path,
    fps: float = 1.0,
    sharp_window: int = 5,
    progress_cb: Optional[Callable[[int, int], None]] = None
) -> bool:
    """
    Extracts dual-fisheye frames (cam0 and cam1) from .insv without external ffmpeg.
    Applies sliding-window sharpness selection to reject motion blur.
    """
    images_dir = output_dir / "images"
    cam0_dir = images_dir / "cam0"
    cam1_dir = images_dir / "cam1"
    cam0_dir.mkdir(parents=True, exist_ok=True)
    cam1_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing extraction
    c0_existing = list(cam0_dir.glob("*.jpg"))
    c1_existing = list(cam1_dir.glob("*.jpg"))
    if len(c0_existing) > 0 and len(c1_existing) > 0:
        print(f"[*] Frames already extracted: {len(c0_existing)} in cam0, {len(c1_existing)} in cam1.")
        return True

    try:
        container = av.open(str(insv_path))
    except Exception as e:
        print(f"[!] Failed to open container {insv_path}: {e}")
        return False

    video_streams = [s for s in container.streams if s.type == "video"]
    if len(video_streams) < 2:
        container.close()
        print(f"[!] Expected >= 2 video streams in {insv_path.name}, found {len(video_streams)}")
        return False

    src_fps = float(video_streams[0].average_rate or 30.0)
    frame_step = max(1, int(round(src_fps / fps)))
    window_size = max(1, sharp_window)

    print(f"[*] Extracting INSV in-process via PyAV:")
    print(f"    Source: {insv_path.name} ({src_fps:.1f} FPS)")
    print(f"    Target: {fps:.1f} FPS (Sample step: {frame_step}, Sharpness window: {window_size})")

    for track_idx in (0, 1):
        stream = video_streams[track_idx]
        target_dir = cam0_dir if track_idx == 0 else cam1_dir
        saved_count = 0
        window_buffer = []

        # Seek container to start
        container.seek(0)
        frame_counter = 0

        for packet in container.demux(stream):
            for frame in packet.decode():
                frame_counter += 1
                if frame_counter % frame_step == 0:
                    img_rgb = frame.to_ndarray(format="bgr24")
                    score = compute_laplacian_sharpness(img_rgb)
                    window_buffer.append((score, img_rgb))

                    if len(window_buffer) >= window_size:
                        # Select sharpest candidate in window
                        best_score, best_frame = max(window_buffer, key=lambda x: x[0])
                        out_file = target_dir / f"frame_{saved_count:06d}.jpg"
                        cv2.imwrite(str(out_file), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        saved_count += 1
                        window_buffer.clear()
                        if progress_cb:
                            progress_cb(track_idx, saved_count)

        # Flush trailing window
        if window_buffer:
            best_score, best_frame = max(window_buffer, key=lambda x: x[0])
            out_file = target_dir / f"frame_{saved_count:06d}.jpg"
            cv2.imwrite(str(out_file), best_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            saved_count += 1
            window_buffer.clear()

        print(f"[+] Track {track_idx} (cam{track_idx}) complete: {saved_count} sharp frames saved.")

    container.close()
    return True
```

### 4.3 Bundling Headless Spirula SfM into PyInstaller Package
1. **Compile Headless Spirula SfM (One-Time Native Build):**
   In `d:\APLICATIVOS\spirula-studio`:
   ```cmd
   cmake -B build_headless -G "Visual Studio 17 2022" -A x64 -DSS_BACKEND=vulkan -DSS_BUILD_GUI=OFF -DSS_BUILD_CLI=ON
   cmake --build build_headless --config Release --target spirula --parallel 8
   ```
   This generates `build_headless/Release/spirula.exe` (a headless CLI that only requires standard Windows `vulkan-1.dll`).

2. **Update `tools/package_app.py`:**
   Configure PyInstaller to bundle `spirula_sfm.exe` into the app payload and collect all PyAV binaries:
   ```python
   # In tools/package_app.py:
   cmd += [
       '--collect-all', 'av',
   ]

   # Bundle headless spirula into bin/
   sfm_source = ROOT / 'spirula' / 'spirula.exe' # or compiled headless binary
   if sfm_source.is_file():
       cmd += ['--add-binary', f'{sfm_source};bin']
   ```

3. **Runtime Resolver in `raven_app/config.py`:**
   ```python
   def get_spirula_bin() -> Path:
       """Resolves bundled or local headless Spirula SfM binary."""
       # 1. PyInstaller frozen environment
       if getattr(sys, 'frozen', False):
           base_dir = Path(sys._MEIPASS)
           bundled = base_dir / 'bin' / 'spirula.exe'
           if bundled.is_file():
               return bundled

       # 2. Local workspace check
       local_bin = Path(__file__).resolve().parents[1] / 'bin' / 'spirula.exe'
       if local_bin.is_file():
           return local_bin

       # 3. System PATH fallback
       which_bin = shutil.which('spirula')
       if which_bin:
           return Path(which_bin)

       raise FileNotFoundError("Could not locate bundled spirula.exe binary.")
   ```

4. **SfM Invocation Parameters in `scripts/pipeline_auto_calibrator_and_colorizer.py`:**
   ```python
   cmd = [
       str(spirula_bin), "sfm", "auto",
       str(img_dir),
       "-o", str(dataset_dir),
       "--data-type", "video",
       "--camera-mode", "folder",
       "--camera-model", "thin-prism-fisheye",
       "--focal", "1080.19",
       "--overlap", "15"
   ]
   ```

---

## 5. Approach B: Full Native C++ Engine Integration (CMake + C-API)

**Rationale:** Zero child processes. `RavenCalibrator` executes SfM and frame demuxing directly inside the process or via a shared library (`spirula_sfm.dll`).

### 5.1 Architecture
```
RavenCalibrator (Python GUI)
  │
  ├── ctypes / C-API
  │
  ▼
spirula_sfm.dll (or native static library in RavenNative)
  ├── Pipeline.h / Pipeline.cpp (sfm::run_auto)
  ├── Vulkan Context (VkContext.h)
  └── CamModel::ThinPrismFisheye (Camera.h)
```

### 5.2 Creating the C-API Bridge in `native/sfm_bridge.h` and `.cpp`
Create `native/sfm_bridge.h`:
```cpp
#pragma once

#ifdef _WIN32
  #define SFM_API __declspec(dllexport)
#else
  #define SFM_API __attribute__((visibility("default")))
#endif

extern "C" {

struct SfmExecutionResult {
    int exit_code;          // 0 = ok, 2 = fail, 3 = partial
    int64_t registered;
    int64_t total_images;
    int64_t num_points;
    double mean_reprojection_error;
};

SFM_API SfmExecutionResult RunThinPrismSfM(
    const char* image_dir,
    const char* output_dir,
    float initial_focal,
    int overlap_window,
    int device_index
);

}
```

Create `native/sfm_bridge.cpp`:
```cpp
#include "sfm_bridge.h"
#include "sfm/Pipeline.h"
#include "sfm/SfmConfig.h"
#include "sfm/core/Camera.h"

SFM_API SfmExecutionResult RunThinPrismSfM(
    const char* image_dir,
    const char* output_dir,
    float initial_focal,
    int overlap_window,
    int device_index
) {
    sfm::SfmConfig cfg;
    cfg.camera.model = sfm::CamModel::ThinPrismFisheye;
    cfg.camera_mode = sfm::CameraMode::Folder;
    cfg.focal = initial_focal > 0.0f ? initial_focal : 1080.19f;
    cfg.overlap = overlap_window > 0 ? overlap_window : 15;
    cfg.device = device_index;
    cfg.data_type = "video";

    sfm::AutoInputs in;
    in.image_dir = image_dir;
    in.workspace = output_dir;

    sfm::RunContext ctx;
    sfm::AutoResult res = sfm::run_auto(cfg, in);

    SfmExecutionResult out{};
    out.exit_code = res.exit_code;
    out.registered = res.registered;
    out.total_images = res.images;
    out.num_points = res.points;
    out.mean_reprojection_error = res.mean_reproj;
    return out;
}
```

### 5.3 CMake Build Configuration (`native/CMakeLists.txt`)
Add the Spirula SfM sources and Vulkan SDK dependency:
```cmake
find_package(Vulkan REQUIRED)

set(SPIRULA_SFM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../spirula-studio/src")

add_library(spirula_sfm SHARED
    sfm_bridge.cpp
    "${SPIRULA_SFM_ROOT}/sfm/Pipeline.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/core/Camera.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/core/CameraSetup.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/core/Features.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/core/Matches.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/core/Model.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/map/Mapper.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/map/Assemble.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/map/MetricGauge.cpp"
    "${SPIRULA_SFM_ROOT}/sfm/vk/VkContext.cpp"
)

target_include_directories(spirula_sfm PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${SPIRULA_SFM_ROOT}"
    ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(spirula_sfm PRIVATE
    Vulkan::Vulkan
)

target_compile_definitions(spirula_sfm PRIVATE SS_BACKEND_VULKAN)
```

### 5.4 Python `ctypes` Wrapper (`raven_app/native_sfm.py`)
```python
import ctypes
from pathlib import Path
from dataclasses import dataclass

class CSfmResult(ctypes.Structure):
    _fields_ = [
        ("exit_code", ctypes.c_int),
        ("registered", ctypes.c_int64),
        ("total_images", ctypes.c_int64),
        ("num_points", ctypes.c_int64),
        ("mean_reprojection_error", ctypes.c_double),
    ]

def run_native_sfm(image_dir: Path, output_dir: Path, focal: float = 1080.19, overlap: int = 15) -> bool:
    dll_path = Path(__file__).resolve().parents[1] / "bin" / "spirula_sfm.dll"
    if not dll_path.is_file():
        raise FileNotFoundError(f"Native library not found at: {dll_path}")

    lib = ctypes.CDLL(str(dll_path))
    lib.RunThinPrismSfM.argtypes = [
        ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    lib.RunThinPrismSfM.restype = CSfmResult

    res = lib.RunThinPrismSfM(
        str(image_dir).encode('utf-8'),
        str(output_dir).encode('utf-8'),
        focal,
        overlap,
        -1 # best device
    )

    print(f"[+] Native SfM finished with code {res.exit_code}: {res.registered}/{res.total_images} registered, {res.num_points} points, reproj err: {res.mean_reprojection_error:.3f} px")
    return res.exit_code == 0
```

---

## 6. Comparative Trade-Off Analysis

| Criteria | Approach A (PyAV + Bundled Headless Binary) | Approach B (Full C++ DLL via CMake) |
| :--- | :--- | :--- |
| **Implementation Effort** | **Low (~2 hours)** | High (~2-3 days) |
| **Risk of Toolchain Divergence** | **Zero** (PyAV is prebuilt; binary is isolated) | Medium (MSVC + Slang/Vulkan compilation subtleties) |
| **Execution Performance** | Identical (both use GPU Vulkan) | Identical |
| **Process Model** | Subprocess invocation (`bin/spirula.exe`) | In-process execution (`.dll`) |
| **Debugging & Error Isolation** | **Simple** (CLI flags can be tested independently) | Complex (memory crashes in C++ can terminate GUI) |
| **Packaging Cleanliness** | Handled natively by PyInstaller hooks | Must bundle DLL + Slang SPIR-V shaders |
| **Recommendation** | **PRIMARY RECOMMENDED PATH** | Future Refactoring Option |

---

## 7. Step-by-Step Implementation Guide for LLM Agents

When acting as an agent to implement **Approach A** on `Lidar-camera-calibrator`:

### Step 1: Install and Verify `av`
```powershell
.\build\package-env\Scripts\python.exe -m pip install "av>=11.0.0"
```
Verify `av` can open `.insv` containers and inspect multi-track:
```python
import av
c = av.open("test.insv")
assert len([s for s in c.streams if s.type == 'video']) == 2
```

### Step 2: Update `raven_app/workflow.py`
1. Locate `extract_insv_frames`.
2. Replace the external `ffmpeg` subprocess calls with `extract_insv_frames_pyav`.
3. Retain the same folder structure: `output_dir / "images" / "cam0"` and `"cam1"`.
4. Retain standard frame formatting: `frame_%06d.jpg`.

### Step 3: Bundle Headless Spirula
1. Place the compiled `spirula.exe` inside `C:\Users\Everton-PC\Documents\APLICATIVOS\Lidar-camera-calibrator\spirula\spirula.exe`.
2. Update `tools/package_app.py` to add `'--collect-all', 'av'` and `--add-binary` for `spirula.exe;bin`.
3. Verify `raven_app/config.py` resolves `sys._MEIPASS / 'bin' / 'spirula.exe'`.

### Step 4: Validate SfM Configuration
In `scripts/pipeline_auto_calibrator_and_colorizer.py`:
Ensure `run_spirula_sfm_auto()` passes:
- `--camera-mode folder` (mandatory for multi-track front/rear cameras)
- `--camera-model thin-prism-fisheye`
- `--data-type video`

---

## 8. Verification, Testing & Acceptance Criteria

### Verification Commands
```powershell
# 1. Test in-process extraction
python -c "from raven_app.workflow import extract_insv_frames; import pathlib; extract_insv_frames(pathlib.Path('D:/capture/video.insv'), pathlib.Path('D:/test_out'), fps=1.0)"

# 2. Verify extracted frames
Get-ChildItem D:\test_out\images\cam0\*.jpg | Measure-Object
Get-ChildItem D:\test_out\images\cam1\*.jpg | Measure-Object

# 3. Test Headless SfM with Thin Prism model
.\dist\RavenCalibrator\bin\spirula.exe sfm auto D:\test_out\images -o D:\test_out --camera-mode folder --camera-model thin-prism-fisheye --focal 1080.19

# 4. Verify COLMAP Model Output
Test-Path D:\test_out\sparse\0\cameras.bin
Test-Path D:\test_out\sparse\0\images.bin
Test-Path D:\test_out\sparse\0\points3D.bin
```

### Acceptance Checklist
- [ ] No subprocess invokes `ffmpeg` or checks for `ffmpeg.exe`.
- [ ] PyInstaller build (`tools/build_windows.ps1`) completes without missing symbol errors.
- [ ] `dist/RavenCalibrator/RavenCalibrator.exe` runs end-to-end extraction and SfM on a fresh machine without external FFmpeg or Spirula installed.
- [ ] `sparse/0/cameras.bin` contains Model ID 10 with 12 parameters per camera folder (`cam0` and `cam1`).
