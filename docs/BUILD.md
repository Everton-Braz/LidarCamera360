# Build and release notes

Developer builds need Visual Studio C++ tools and Windows SDK, CMake 3.24+, Python 3.10+, Git, and vcpkg.

```powershell
.\tools\build_windows.ps1 -VcpkgRoot D:\vcpkg -InstallDependencies
```

The helper builds the native estimator and Vulkan colorizer, creates the Python packaging environment, and writes `dist/LidarCamera360/`. Use `-Generator 'Visual Studio 17 2022'` on Visual Studio 2022. Use `-OneFile` only when a self-extracting executable is required.

Sophus is pinned to `a621ff2e56c56c839a6c40418d42c3c254424b5c` and Vikit to `6c886c8e5d83997806e00294826d528cea3581dd`. `native/prepare_sources.py` copies the upstream FAST-LIVO2, Sophus, and Vikit trees into a generated build tree and applies the portability layer; the supplied upstream source remains unchanged.

Native targets use C++17, PCL, Eigen, OpenCV, yaml-cpp, OpenMP, and Vulkan for GPU colorization. Release builds use portable x64 settings without machine-specific AVX requirements.

Run the checks listed in [the validation report](IMPLEMENTATION_VALIDATION.md). Inspect `_internal/licenses/` and update [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) whenever dependency versions or bundled binaries change. A redistributable FAST-LIVO2 build must include corresponding source and GPL-2.0-only materials. A Spirula binary needs a matching upstream license, source/provenance record, and notices before publication.
