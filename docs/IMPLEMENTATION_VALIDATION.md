# Standalone INSV and Vulkan implementation validation

## Implementation

- Approach A: PyAV decoding plus bundled Spirula subprocess. The alternate SfM DLL approach was not implemented.
- Dual-track sharpness selection preserves cadence and records PTS in a completion manifest. Real timestamps feed alignment, recalibration, drift correction and direct/3DGS poses.
- Spirula is invoked with folder camera grouping, video data type, 1080.19 initial focal length and THIN_PRISM_FISHEYE. Nonzero exit codes, missing sparse outputs and unsupported camera models fail validation.
- Vulkan uploads point positions once, uses two staging/texture slots, overlaps next-frame decoding with compute, clears an atomic depth buffer per view, accumulates Top-3 observations and resolves RGB once.
- Python retains PCD/COLMAP loading, calibrated pose interpolation and PLY/PCD output. The native subprocess uses a versioned RVC1 binary job (`--job`, `--out`, `--device`) to share exactly the same poses with CPU fallback; this is an intentional alternative to duplicating those readers/interpolators in C++.
- Thin Prism projection was corrected to apply tangential/prism terms in equidistant coordinates, matching the local Spirula `Camera.h`. CPU and GPU use bilinear sampling and millimeter depth quantization.
- Large survey coordinates are shifted in float64 before upload. Unsupported image dimensions, device limits and native failures return to CPU. Each job currently requires all camera images to have the same resolution. Existing CPU colorization remains optimized for the 3840-square Raven rig.

## Verification

- Native Release build and all three GLSL shaders compiled successfully with MSVC and Vulkan SDK 1.4.341.1.
- Native probe selected NVIDIA GeForce RTX 5070 Ti and created all three compute pipelines.
- 14 existing standalone tests passed, including GUI construction and COLMAP metric export.
- Six additional tests passed: paired PyAV extraction/cadence/PTS/resume, single-track failure, legacy timing, GPU occlusion/consensus at large world coordinates, missing-image fallback, and a 100,000-point distorted-projection color fixture.
- The 100,000-point fixture uses a coordinate-encoded texture; it is not a direct sub-0.05-pixel projection readback test or a CIEDE2000 certification.

## Performance

`python -m tools.benchmark_vulkan` on this machine processed 10,000,000 synthetic
points and 175 views of 3840 x 3840 JPEG imagery in **8.345 seconds**, including
Python/native job IO and image decoding. This is a synthetic workload and is
not the real scan acceptance benchmark. The proposed **under-3-second target
has not been achieved**. No 50x speedup is claimed without a matched CPU run.

The native API deliberately uses Vulkan 1.1-compatible compute operations and
works with the installed Vulkan 1.4 driver. No external `vulkaninfo` is needed
by the colorizer's device selection.

## Standalone and real-media checks

- PyInstaller onedir packaging completed successfully. Output:
  `dist/RavenCalibrator/RavenCalibrator.exe` (keep `_internal` alongside it).
- The packaged smoke test ran from an unrelated temporary directory with PATH
  restricted to Windows System32. Doctor, bundled Spirula and Vulkan probes,
  PyAV extraction, invalid-input handling and Vulkan direct colorization passed.
- The real `VID_20260821_105436_00_269.insv` (88.29 seconds, two 24 FPS video
  tracks) decoded into **89 synchronized frame pairs** in 287.80 seconds.
- The packaged Spirula binary, also with isolated PATH, reconstructed a 40-image
  crop of that extraction: **37/40 registered**, **6,733 points**, **two cameras
  with model ID 10 and 12 parameters**, mean reprojection error **1.740 px**.
  Its reported total SfM time was **5.91 seconds**, exit code 0.
- CPU/GPU comparison using those 37 real images and the reconstructed sparse
  points: mean absolute RGB-channel difference **0.02099/255**, maximum **14/255**.
  CPU took 2.140 s and GPU 2.284 s on this small fixture; GPU startup/decoding
  dominates at 6,733 points. This is not a full LiDAR benchmark or a Delta-E test.
- All **20 unit/integration tests** passed. No full bag-to-deliverables run on a
  clean physical machine was performed; the isolated packaged smoke test and
  real-media/SfM checks cover the new standalone dependencies.
