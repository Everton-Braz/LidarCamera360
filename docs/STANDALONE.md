# Standalone Windows app

Builds two Windows x64 executables: `RavenCalibrator.exe` (desktop controls and
headless CLI, bundled Python runtime) and `fastlivo2.exe` (native MSVC C++17
FAST-LIVO2 estimator). Neither needs ROS, WSL, or an installed Python interpreter.
The default portable directory avoids extracting large DLLs on every launch.
Keep the executable and `_internal` directory together when moving the app.
`-OneFile` builds `dist/single-file/RavenCalibrator.exe`, a self-extracting EXE
when a single file matters more than startup time.

## Use

Double-click RavenCalibrator.exe for the desktop application, designed with the
Microsoft UI XAML Fluent Design System (WinUI 3 controls, Mica/Acrylic backdrop materials,
navigation rail, and dark/light mode). Every processing command also runs headlessly;
`--headless` explicitly forbids opening a GUI.

```powershell
.\RavenCalibrator.exe --help
.\RavenCalibrator.exe --headless doctor
.\RavenCalibrator.exe inspect-bag --bag D:\capture\merged.bag
.\RavenCalibrator.exe slam --bag D:\capture\merged.bag --lio --output D:\dataset\slam_out --threads 4
.\RavenCalibrator.exe slam --bag D:\capture\merged.bag --image-topic /camera_front/image/compressed --output D:\dataset\slam_vio
.\RavenCalibrator.exe colorize --dataset D:\dataset --method direct --fps 1 --dt -4.2
```

SLAM writes `pcd/all_raw_points.pcd`, `result/Raven_3DMakerPro_Scan.txt` and
`run.json` under the chosen output directory. Existing map/trajectory results
are rejected, so use a fresh directory for another run. Return codes: 0 success,
2 invalid input/runtime failure, 130 cancellation. Native engine failures are
propagated. Ctrl+C flushes accumulated map data; interrupted results are partial.

Default topics and calibration are for the project's Raven rig. For other
calibrations pass `--config` and `--camera`. Camera fusion accepts raw bgr8/rgb8/
mono8/BGRA/RGBA images and compressed images; select their actual topic explicitly.
Use `inspect-bag` first. The offline adapter currently accepts standard
PointCloud2 with XYZ and per-point timing, not Livox CustomMsg packets.
For Raven, `timestamp` is in seconds and offsets are measured from the first
point, matching the supplied XT32 preprocessing. Other sensors must supply
`--time-field`, `--time-unit`, and `--time-origin` as appropriate. Missing timing
is an error. Feature extraction must be disabled in the offline config.

Pass split, non-overlapping bags together after `--bag`; do not pass a merged bag
and its source bags together. Header times must be monotonic within each sensor.
The reader streams records with bounded sensor queues and preserves startup IMU
initialization. It never slices away the beginning of a capture. Large maps still
require memory proportional to the accumulated points, as in the upstream engine.
Camera fusion only maps synchronized camera/LiDAR/IMU coverage. If the camera
ends before the other sensors, the remaining scans are reported in `run.json`
and the log; use `--lio` to map the full LiDAR/IMU overlap.

Colorization uses the existing project algorithms and requires prepared input:
`slam_out/pcd/all_raw_points.pcd`, `slam_out/result/Raven_3DMakerPro_Scan.txt`,
paired JPEG frames in `images/cam0` and `images/cam1`, and calibration JSON.
SfM requires `sparse/0` or `--run-spirula`; direct drift correction uses an
existing aligned SfM reconstruction. INSV extraction is not part of this UI.
The frame FPS and time offset must match the extraction/capture.

## Native engine independently

```powershell
.\RavenCalibrator.exe export-bag --bag D:\capture\merged.bag --lio --output D:\capture\input.flv2
.\_internal\bin\fastlivo2.exe --input D:\capture\input.flv2 --config .\_internal\FAST-LIVO2\config\raven.yaml --camera .\_internal\FAST-LIVO2\config\camera_raven.yaml --output D:\slam --lio
```

`slam` streams that same binary format through stdin, avoiding an intermediate
copy on disk. The estimator compiles the supplied FAST-LIVO2 IMU, voxel map and
visual estimator sources. `native/prepare_sources.py` creates a build-only
adaptation: middleware publication is disabled, parameter access reads YAML,
and outputs use the selected directory. The supplied ROS sources are preserved.
This is an offline port, not a live sensor driver or a ROS compatibility runtime.

## Build

Requirements for developers: Visual Studio C++ tools + Windows SDK, CMake 3.24+,
Python 3.10+, Git, and vcpkg. The delivered app does not require these tools.

```powershell
.\tools\build_windows.ps1 -VcpkgRoot D:\vcpkg -InstallDependencies
# Visual Studio 2022:
.\tools\build_windows.ps1 -VcpkgRoot D:\vcpkg -Generator 'Visual Studio 17 2022'
# Optional single file (slower first launch/extraction): add -OneFile
```

Sophus is pinned to `a621ff2e56c56c839a6c40418d42c3c254424b5c` and Vikit to
`6c886c8e5d83997806e00294826d528cea3581dd`. The source manifest in the package
records hashes of the supplied FAST-LIVO2 tree. C++ dependencies are PCL,
Eigen, OpenCV (calib3d/imgproc/imgcodecs), yaml-cpp and OpenMP. Release uses /O2
without machine-specific AVX flags so the package remains portable across x64 CPUs.

## Third-party source and notices

FAST-LIVO2 is GPL-2.0; its license is bundled. Sophus, Vikit, PCL, OpenCV and
other libraries retain their own notices. The existing Spirula binary is bundled
when present. Its distribution permissions and matching source/notices must be
checked before publishing a release. This local build does not publish anything.
When distributing the native binary, include the corresponding FAST-LIVO2 source,
native adaptation, build scripts and dependency license/source materials.

## INSV extraction and GPU colorization

Approach A is the default package: PyAV and its codec libraries decode video
in-process, and `bin/spirula.exe` performs headless Vulkan SfM. No external
FFmpeg or separate Spirula installation is required. Packaging fails if the
Spirula executable or the colorizer/shaders are missing.

```powershell
.\RavenCalibrator.exe extract-insv --insv D:\capture\video.insv --output D:\dataset --fps 1
.\RavenCalibrator.exe workflow --bag D:\capture\lidar.bag --insv D:\capture\video.insv --output D:\dataset --method sfm --export-ply --export-pcd --export-colmap
.\RavenCalibrator.exe colorize --dataset D:\dataset --method direct --no-vulkan
```

Extraction keeps synchronized front/rear pairs, selects the sharpest of up to
five candidates per output interval, and writes measured presentation times to
`images/frames.json`. Alignment, calibration, direct projection and 3DGS export
read these times. Older datasets without that manifest retain one-based frame
number/FPS timing. Both tracks must decode successfully before publication.

Vulkan acceleration is enabled by default in the unified GUI and CLI. It uses
the selected calibration/Sim(3)/drift-corrected poses for both colorization
methods. Missing hardware, an unsupported job or a failed subprocess invokes
the CPU path. `doctor` reports the device and shader probe. A Vulkan-capable
GPU driver is required for GPU processing; the runtime does not require the SDK.
The shaders use the Vulkan 1.1 compute subset and run on Vulkan 1.4 drivers.

To build the colorizer independently (Visual Studio generator as installed):

```powershell
cmake -S native/vulkan_colorizer -B build/vulkan -A x64 -DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build build/vulkan --config Release
```

The full native build includes this target. Building requires Vulkan SDK with
`glslc` and vcpkg OpenCV. CMake compiles shaders into the binary output directory;
packaging takes these compiled outputs rather than source-tree SPIR-V files.
See `docs/IMPLEMENTATION_VALIDATION.md` for measured validation and limitations.
