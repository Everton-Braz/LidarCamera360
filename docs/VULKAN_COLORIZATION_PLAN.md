# Technical Specification & Development Plan: Vulkan Compute Shader Colorizer

## Executive Summary
This document outlines the architectural blueprint, mathematical foundations, memory layout, and implementation roadmap for developing **Option B: Vulkan Compute Shader Colorizer** in [RavenCalibrator](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator).

### Problem Statement
Currently, multi-view point cloud colorization ([`colorize_via_spirula_sfm`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/scripts/pipeline_auto_calibrator_and_colorizer.py#L995) and [`colorize_via_direct_rigid`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/scripts/pipeline_auto_calibrator_and_colorizer.py#L1163)) is executed sequentially in Python on the CPU. For large LiDAR point clouds ($10\text{M}+$ points) across $200\text{--}400$ dual-fisheye 8K video frames:
- CPU runtime requires **2 to 5 minutes**.
- Image decoding and non-linear Thin Prism projection are constrained by Python GIL and CPU SIMD limits.

### Proposed Solution
Develop a high-performance **Vulkan 1.4 Compute Shader Engine** (`vulkan_colorizer`) running on the dedicated GPU (**NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM**):
- **Zero PCIe Roundtrips**: 3D LiDAR point coordinates are uploaded to GPU device-local memory once.
- **Asynchronous Double-Buffered Texture Streaming**: Camera JPEG frames are decoded on CPU worker threads in parallel with GPU compute execution.
- **Hardware-Accelerated Projections**: Thin Prism 12-parameter polynomial evaluation and bilinear image sampling mapped directly to GPU compute cores.
- **Target Performance**: Reduce total colorization time from ~180 seconds to **< 3 seconds** (over **50× speedup**).

---

## 1. Mathematical & Optical Formulation

### 1.1. Coordinate Transformation
Given a LiDAR point in world coordinates $P_{\text{world}} = (X_w, Y_w, Z_w)^T$ and camera pose $(R_{cw}, C_{\text{world}})$:
$$P_{\text{cam}} = \begin{pmatrix} X_c \\ Y_c \\ Z_c \end{pmatrix} = R_{cw} (P_{\text{world}} - C_{\text{world}})$$

Near-plane clipping: if $Z_c \le 0.15\text{ m}$, the point is behind or too close to the lens and discarded.

### 1.2. Thin Prism 12-Parameter Fisheye Projection
Normalized camera coordinates:
$$x = \frac{X_c}{Z_c}, \quad y = \frac{Y_c}{Z_c}, \quad r = \sqrt{x^2 + y^2}, \quad \theta = \arctan(r)$$

Equidistant radial distortion with 4 polynomial coefficients $(k_1, k_2, k_3, k_4)$:
$$\theta_d = \theta \left(1 + k_1 \theta^2 + k_2 \theta^4 + k_3 \theta^6 + k_4 \theta^8\right)$$

Distorted radial coordinates:
$$x_d = x \frac{\theta_d}{r}, \quad y_d = y \frac{\theta_d}{r}, \quad r_d^2 = x_d^2 + y_d^2$$

Thin Prism tangential and prism distortion with $(p_1, p_2, sx_1, sy_1)$:
$$x'' = x_d + 2 p_1 x_d y_d + p_2 (r_d^2 + 2 x_d^2) + sx_1 r_d^2$$
$$y'' = y_d + p_1 (r_d^2 + 2 y_d^2) + 2 p_2 x_d y_d + sy_1 r_d^2$$

Pixel coordinates via focal length and principal point $(fx, fy, cx, cy)$:
$$u = fx \cdot x'' + cx, \quad v = fy \cdot y'' + cy$$

Circular fisheye aperture boundary mask:
$$r_{\text{px}} = \sqrt{(u - cx)^2 + (v - cy)^2} < 1620.0\text{ px}, \quad 0 \le u < 3839, \quad 0 \le v < 3839$$

### 1.3. Sub-Millimeter Atomic Z-Buffer Occlusion Testing
To reject occluded points behind walls, furniture, or topography without full mesh rasterization, a $960 \times 960$ (or $1920 \times 1920$) depth grid is used:
$$u_g = \text{clamp}\left(\left\lfloor \frac{u}{\text{scale}} \right\rfloor, 0, W_z - 1\right), \quad v_g = \text{clamp}\left(\left\lfloor \frac{v}{\text{scale}} \right\rfloor, 0, H_z - 1\right)$$

Euclidean distance $d = \|P_{\text{cam}}\|$. Quantized to 32-bit unsigned integer depth:
$$\text{depth}_{\text{fixed}} = \text{uint}(d \cdot 1000.0f)$$

Atomic minimum in GLSL:
```glsl
imageAtomicMin(depthMap, ivec2(ug, vg), depth_fixed);
```

Visibility condition:
$$d \le \left(\frac{\text{depth}_{\text{min}}}{1000.0f}\right) \times 1.08 + 0.15\text{ m}$$

### 1.4. Angular & Distance Quality Metric
Images looking directly at a surface at close range provide superior texture sharpness compared to grazing angles or distant perspectives:
$$S(d, r_{\text{px}}) = \frac{1.0 - 0.5 \left(\frac{r_{\text{px}}}{1620.0}\right)^2}{\max(d, 0.5)^{1.5}}$$

### 1.5. Top-3 Consensus & Outlier Rejection
For each LiDAR point $i$, keep the top 3 best views:
$$\text{TopScores}[i] = (s_1, s_2, s_3), \quad \text{TopColors}[i] = (C_1, C_2, C_3)$$
If a new valid observation has $score > \min(\text{TopScores}[i])$, it replaces the lowest score slot.

Final color resolution:
- 1 view: $C_1$
- 2 views: weighted average $\frac{s_1 C_1 + s_2 C_2}{s_1 + s_2}$
- 3 views: color consensus with outlier rejection: if $\|C_k - \text{median}(C)\|_2 > \tau$, reject dynamic artifact and blend remaining inliers.

---

## 2. Vulkan Compute Shader Architecture

```
                       HOST MEMORY (CPU)                                   DEVICE VRAM (NVIDIA RTX 5070 Ti)
   +-------------------------------------------------------+      +-------------------------------------------------------+
   | LiDAR Point Cloud (XYZ + Intensity)                   | ---> | PointsBuffer (SSBO, Device-Local, ReadOnly)           |
   +-------------------------------------------------------+      +-------------------------------------------------------+
   | Camera Frame K (JPEG -> RGBA8 Staging)                | ---> | CameraTexture (VkImage, 2D Optimal, Hardware Sampler) |
   +-------------------------------------------------------+      +-------------------------------------------------------+
                                                                  | DepthMap (VkImage R32UI, Atomic Min Occlusion Grid)   |
                                                                  +-------------------------------------------------------+
                                                                  | TopScoresBuffer (SSBO, vec3 per point, ReadWrite)     |
                                                                  +-------------------------------------------------------+
                                                                  | TopColorsBuffer (SSBO, uvec3 per point, ReadWrite)    |
                                                                  +-------------------------------------------------------+
                                                                  | OutputColorsBuffer (SSBO, u8vec4 per point, WriteOnly)|
                                                                  +-------------------------------------------------------+
```

### 2.1. Compute Pass 1: `occlusion_zbuf.comp`
- **Workgroup Size**: `layout(local_size_x = 256) in;`
- **Dispatch**: $\lceil N / 256 \rceil$ invocations.
- **Task**:
  1. Fetch $P_{\text{world}} = \text{points}[idx]$.
  2. Compute $P_{\text{cam}} = R_{cw}(P_{\text{world}} - C)$.
  3. Evaluate Thin Prism projection $\to (u, v)$.
  4. If within lens bounds, quantize to $(u_g, v_g)$ and execute `imageAtomicMin(depthMap, ivec2(ug, vg), depth_fixed)`.

### 2.2. Compute Pass 2: `colorize_consensus.comp`
- **Workgroup Size**: `layout(local_size_x = 256) in;`
- **Dispatch**: $\lceil N / 256 \rceil$ invocations.
- **Task**:
  1. Fetch $P_{\text{world}} = \text{points}[idx]$ and reproject.
  2. Read minimum depth from `depthMap` at $(u_g, v_g)$.
  3. Execute visibility check: $d \le d_{\text{min}} \times 1.08 + 0.15$.
  4. Compute quality score $S$.
  5. Check if $S > \min(\text{topScores}[idx])$.
  6. Sample `texture(cameraSampler, vec2(u / 3840.0, v / 3840.0)).rgb`.
  7. Atomically or thread-exclusively update `topScores[idx]` and `topColors[idx]`.

### 2.3. Compute Pass 3: `resolve_consensus.comp` (Executed Once at End)
- **Workgroup Size**: `layout(local_size_x = 256) in;`
- **Dispatch**: $\lceil N / 256 \rceil$ invocations.
- **Task**:
  1. Reads `topScores[idx]` and `topColors[idx]`.
  2. Evaluates statistical consensus and outlier rejection.
  3. Writes packed RGB8 to `outputColors[idx]`.

---

## 3. Host Engine Implementation (`vulkan_colorizer`)

### 3.1. Directory Structure
```
native/
├── CMakeLists.txt
├── vulkan_colorizer/
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── vk_context.h         # Instance, physical device (RTX 5070 Ti), compute queue
│   │   ├── vk_buffer.h          # Staging and device-local SSBO allocations
│   │   ├── vk_texture.h         # Asynchronous image staging and optimal layout transitions
│   │   └── vk_pipeline.h        # Descriptor sets, compute pipelines, SPIR-V loader
│   ├── src/
│   │   ├── main.cpp             # CLI entrypoint (--pcd, --colmap, --images, --out)
│   │   ├── vk_context.cpp
│   │   ├── vk_buffer.cpp
│   │   ├── vk_texture.cpp
│   │   ├── vk_pipeline.cpp
│   │   └── colorizer_engine.cpp # Double-buffered execution loop
│   └── shaders/
│       ├── occlusion_zbuf.comp
│       ├── colorize_consensus.comp
│       └── resolve_consensus.comp
```

### 3.2. Double-Buffered CPU-GPU Concurrency
```
 Time ------------------------------------------------------------------------------------>
 CPU Thread: [ Decode Frame 0 ] [ Decode Frame 1 ] [ Decode Frame 2 ] [ Decode Frame 3 ] ...
 GPU Compute:                   [ ZBuf + Color 0 ] [ ZBuf + Color 1 ] [ ZBuf + Color 2 ] ...
```
Using dual `VkImage` texture buffers ($A$ and $B$):
- While GPU processes Frame $K$ using Texture $A$, CPU decodes Frame $K+1$ and copies to Texture $B$ staging memory.
- Synchronization is handled via `VkFence` and `VkSemaphore` with zero stall on CPU or GPU.

---

## 4. Integration into RavenCalibrator

### 4.1. Python Wrapper ([`raven_app/vulkan_engine.py`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/raven_app/vulkan_engine.py))
```python
def run_vulkan_colorizer(
    pcd_path: Path,
    colmap_dir: Path,
    images_dir: Path,
    output_ply: Path,
    alignment_json: Path = None,
    device_id: int = 1
) -> bool:
    """Invokes native vulkan_colorizer.exe with hardware GPU acceleration."""
```

### 4.2. Workflow Integration ([`raven_app/workflow.py`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/raven_app/workflow.py))
- Automatically detects `vulkan_colorizer.exe` in `bin/` or `native/`.
- If available, runs `[STAGE 5/5]` via Vulkan GPU compute.
- If unavailable or on failure, falls back transparently to CPU multi-threading.

### 4.3. User Interface ([`raven_app/views/unified_workflow_view.py`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/raven_app/views/unified_workflow_view.py))
- Adds toggle:
  `☑ Enable Vulkan 1.4 GPU Compute Acceleration (RTX 5070 Ti)` (checked by default).

### 4.4. Standalone Packaging ([`tools/package_app.py`](file:///c:/Users/Everton-PC/Documents/APLICATIVOS/Lidar-camera-calibrator/tools/package_app.py))
- Bundles `vulkan_colorizer.exe` and compiled `.spv` SPIR-V shaders into `bin/` inside `RavenCalibrator.exe`.

---

## 5. Implementation Roadmap & Milestones

| Milestone | Task | Deliverables | Estimated Effort |
|---|---|---|---|
| **M1: Shaders & Math** | Write `occlusion_zbuf.comp`, `colorize_consensus.comp`, `resolve_consensus.comp`. Test compilation with `glslc`. | Valid SPIR-V bytecode files (`.spv`). | 1 day |
| **M2: Vulkan Core** | Implement `vk_context`, memory allocators, staging buffers, and compute dispatch in C++17. | Clean Vulkan compute pipeline abstraction. | 2 days |
| **M3: Colorizer Engine** | Implement PCD loading, camera trajectory interpolation, double-buffered texture stream, and PLY writer. | Standalone `vulkan_colorizer.exe` CLI. | 2 days |
| **M4: Integration & GUI** | Wire into `workflow.py`, `pipeline_auto_calibrator_and_colorizer.py`, and Fluent UI settings/controls. | GUI toggles, config options, doctor reporting. | 1 day |
| **M5: Testing & Benchmarking** | Validate RGB numerical consistency against CPU ground truth. Benchmark 10M point dataset on RTX 5070 Ti. | Unit tests, benchmark report, GitHub commit. | 1 day |

---

## 6. Verification & Quality Assurance

1. **Reprojection Accuracy Test**:
   - Compare point-by-point pixel coordinates $(u, v)$ between CPU Python `project_thin_prism` and GLSL compute shader across $100,000$ points. Tolerable error: $< 0.05$ pixels.
2. **Depth Buffer Occlusion Test**:
   - Verify that occluded background points behind a foreground wall are rejected identically to CPU `np.minimum.at`.
3. **Consensus Color Fidelity**:
   - Compute mean color difference $\Delta E_{\text{CIEDE2000}}$ between CPU Top-3 consensus and Vulkan Top-3 consensus. Tolerable $\Delta E < 1.0$ (visually identical).
4. **Performance Benchmark**:
   - Record total processing time on real test dataset (`LIDAR_20260821105002Teste.bag`, 175 frames):
     - Baseline CPU: ~120 seconds.
     - Vulkan GPU Goal: **< 3.0 seconds**.
