# OpenMVS, fisheye build (optional)

The dense photogrammetry stage runs OpenMVS `DensifyPointCloud` as a separate program. It is not bundled.
OpenMVS is AGPL-3.0; `fisheye.patch` in this folder modifies it and is under the same AGPL-3.0 license
(modified by the LidarCamera360 project in November 2026). The rest of this repository only exchanges files
with the program.

## What the patch changes

Upstream OpenMVS (commit `d235f83275a29a22e3c8ef0d91b6e15448c94aa5`) matches pinhole or full equirectangular
images. The patch replaces the equirectangular camera by an ideal equidistant fisheye: a square image,
3.5 rad wide, focal length `width / 3.5`, centre `(width - 1) / 2`, depth measured along the ray. A whole
fisheye photo is then matched as one image, with no pinhole faces. Only the CUDA path is supported, and a
patched binary no longer reads equirectangular scenes; the app refuses a binary without the patch. SFM and
the two programs that need Ceres are left out of the build.

## Build (Windows, NVIDIA GPU)

Needs Visual Studio 2022 Build Tools (v143), CUDA Toolkit, CMake 3.25 or newer, Git, about 15 GB of disk and
a checkout of [vcpkg](https://github.com/microsoft/vcpkg). In an empty folder (PowerShell):

```powershell
git clone https://github.com/cdcseacave/openMVS src
git -C src checkout d235f83275a29a22e3c8ef0d91b6e15448c94aa5
git -C src apply <this folder>\fisheye.patch
mkdir triplets; copy <this folder>\x64-windows-rel.cmake triplets
cmake -S src -B make -G "Visual Studio 17 2022" -A x64 -T "v143,cuda=<your CUDA version>" "-DCMAKE_TOOLCHAIN_FILE=$PWD\vcpkg\scripts\buildsystems\vcpkg.cmake" "-DVCPKG_OVERLAY_TRIPLETS=$PWD\triplets" -DVCPKG_TARGET_TRIPLET=x64-windows-rel -DVCPKG_HOST_TRIPLET=x64-windows-rel -DOpenMVS_BUILD_VIEWER=OFF -DOpenMVS_USE_PYTHON=OFF -DOpenMVS_USE_SIFTGPU=OFF -DOpenMVS_USE_CERES=OFF -DOpenMVS_ENABLE_TESTS=OFF -DOpenMVS_USE_BREAKPAD=OFF -DOpenMVS_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<120 for RTX 50xx, 89 for 40xx, 86 for 30xx>
cmake --build make --config Release --target DensifyPointCloud --parallel 4
```

The program is `make\bin\vc17\x64\Release\DensifyPointCloud.exe`; keep the DLLs beside it. Choose it in
Settings, or set `LIDARCAMERA360_OPENMVS`, or put that folder's files in `openmvs\` next to the app.
