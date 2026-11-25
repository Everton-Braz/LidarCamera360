# OpenMVS, fisheye build (bundled as `openmvs/*.exe`)

- License: AGPL-3.0 (`openmvs/licenses/OpenMVS-AGPL-3.0.txt`); authors in `OpenMVS-COPYRIGHT.md`
- Upstream: <https://github.com/cdcseacave/openMVS>, commit `d235f83275a29a22e3c8ef0d91b6e15448c94aa5`
- Modified by the LidarCamera360 project in November 2026 with `third_party/openmvs/fisheye.patch` (AGPL-3.0)
- Programs: `DensifyPointCloud.exe`, `ReconstructMesh.exe`, `TextureMesh.exe`
- Corresponding source: `OpenMVS-fisheye-source.zip`, attached to the same release. It holds the patched source, the
  build instructions, the vcpkg triplet, recipes and commit, and the sources of CGAL (GPL-3.0), GMP and libheif
  (LGPL-3.0). The commit and the patch are also public at the addresses above and in the release's tag.
- The other libraries and their licenses are in `openmvs/licenses/libraries.txt`.
- The NVIDIA CUDA runtime is linked into `MVS.dll` as NVIDIA's CUDA EULA allows; the Visual C++ runtime DLLs are
  redistributed under Microsoft's terms.

LidarCamera360 runs the programs as separate processes, exchanges files with them and does not link them. To use
another build, point Settings to its `DensifyPointCloud.exe` or replace the files in `openmvs/`.
