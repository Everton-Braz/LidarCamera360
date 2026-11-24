# Third-party notices

The MIT license in `LICENSE` applies only to original LidarCamera360 material independently authored for this repository. It does not relicense upstream source, bundled executables, or package dependencies.

| Component | Use | License/evidence | Distribution condition |
| --- | --- | --- | --- |
| FAST-LIVO2 | LiDAR-inertial-visual estimator and native adaptation | GPL-2.0-only; `FAST-LIVO2/LICENSE`; [upstream repository](https://github.com/hku-mars/FAST-LIVO2) | Preserve the GPL notice and provide corresponding source, including the build adaptation, for a covered binary. The upstream README requests a separate commercial license discussion for commercial use. |
| Spirula Studio | Optional headless SfM/reconstruction executable; bundled v2026.10.30, commit `6ad73ef`; official Windows Vulkan x86_64 archive SHA-256 `dfaf1376ab78acf4bfbaaeb7a3f12b7c06ab30bf1d9eee6082100dbc1e7b71dd` | GPL-3.0; [release](https://github.com/harry7557558/spirula-studio/releases/tag/v2026.10.30), [pinned source](https://github.com/harry7557558/spirula-studio/tree/6ad73ef), [license](https://github.com/harry7557558/spirula-studio/blob/6ad73ef/LICENSE) | Preserve this release's license and source/provenance materials when publishing a package containing `spirula.exe`. Review upstream patent notices for optional patented codecs. |
| OpenMVS (fisheye build) | Optional dense photogrammetry, run as a separate program the user builds; not bundled. `third_party/openmvs/fisheye.patch` modifies commit `d235f83` | AGPL-3.0; [upstream](https://github.com/cdcseacave/openMVS). The patch is AGPL-3.0 too | Do not bundle the binary without its AGPL-3.0 license and the patched source. |
| RootSLAM | Algorithmic formulation for square-root bundle adjustment & pose-graph loop refinement (`raven_app/rootslam.py`) | Independent implementation based on academic paper arXiv:2610.15157 (Santucci et al., Oct 2026) | Cite publication in academic and technical documentation. |
| MOLA / MRPT | Design reference for the pose-graph tools; `native/mola_tools/` is an independent implementation that links only Eigen and contains no MOLA/MRPT source | Reference only. MOLA is GPL-3.0, MRPT is BSD-3-Clause; [MOLAorg](https://github.com/MOLAorg) | Cite the projects. Copying MOLA source into this repository would make that part GPL-3.0. |
| Sophus | Native math dependency | Upstream license retained with the pinned source | Preserve upstream notices in build or release materials. |
| Vikit | Camera/math dependency used by FAST-LIVO2 | Upstream license retained with the pinned source | Preserve upstream notices in build or release materials. |
| PCL, Eigen, OpenCV, yaml-cpp, OpenMP, Vulkan SDK | Native build/runtime dependencies | Their own upstream licenses; generated notices under `dist/LidarCamera360/_internal/licenses/` when present | Keep generated notices with redistributed binaries and verify exact versions for each release. |
| jsonschema 4.26.0 | Validates `project.json` and the pose file against the v1 schemas (`raven_app/project_validate.py`) | MIT (`licenses/jsonschema-4.26.0-MIT.txt`) | Preserve the MIT notice. |
| attrs 26.1.0, referencing 0.37.0, rpds-py 2026.9.1, jsonschema-specifications 2025.9.1 | Dependencies of jsonschema | MIT (`licenses/attrs-26.1.0-MIT.txt`, `licenses/referencing-0.37.0-MIT.txt`, `licenses/rpds-py-2026.9.1-MIT.txt`, `licenses/jsonschema-specifications-2025.9.1-MIT.txt`) | Preserve the MIT notices. |
| Python packages | Bag I/O, video decoding, numerics, GUI, packaging | Package metadata and license files in the built environment | Keep the generated license directory with the portable distribution. |
| PyQt6 6.10.2 | GUI Python bindings | GPL-3.0-only (`licenses/PyQt6-6.10.2-GPL-3.0-only.txt`); package metadata declares SPDX `GPL-3.0-only` | A binary using PyQt6 under the GPL option must satisfy GPL obligations. Riverbank also offers a commercial license; obtain it if the intended product terms are incompatible with GPL. |
| Qt 6.10.2 | Qt runtime shipped with PyQt6 | LGPL-3.0-or-later metadata (`licenses/Qt6-6.10.2-LGPL-3.0.txt`) | Preserve Qt notices and LGPL relinking/installation information applicable to the distributed binary. |
| PyQt6-sip 13.11.0 | PyQt6 binding support | BSD-2-Clause (`licenses/PyQt6-sip-13.11.0-BSD-2-Clause.txt`) | Preserve the BSD notice. |
| PyQt6-Fluent-Widgets 1.11.3 | GUI widgets | GPLv3 in package metadata; exact SPDX variant is not declared (`licenses/PyQt6-Fluent-Widgets-1.11.3-GPLv3.txt`) | Treat the GUI distribution as GPLv3-covered unless a compatible commercial or other license is obtained from the upstream author. |
| PyQt6-Frameless-Window 0.8.2 | Frameless window controls | GPLv3 in package metadata; exact SPDX variant is not declared (`licenses/PyQt6-Frameless-Window-0.8.2-GPLv3.txt`) | Preserve the license and source/notice obligations for the GUI distribution. |

## How a release is put together

A portable release is an aggregate of separate programs that talk through files and command lines:

| Program | Built from | License of the program |
| --- | --- | --- |
| `LidarCamera360.exe` + `_internal/` | MIT project source, PyQt6, Qt, PyQt6-Fluent-Widgets, Python packages | GPL-3.0 (PyQt6 is GPL-3.0-only) |
| `bin/fastlivo2.exe` | `native/main.cpp` (MIT) + the FAST-LIVO2 port, Sophus, Vikit, PCL, OpenCV | GPL-2.0-only |
| `bin/spirula.exe` | Spirula Studio v2026.10.30, unmodified official binary | GPL-3.0 |
| `bin/rfdetr-masker.exe` | `native/rfdetr_masker/` | GPL-3.0 |
| `bin/vulkan_colorizer.exe`, `bin/mola_optimizer.exe` | Project source | MIT |

`fastlivo2.exe` cannot be relicensed under GPL-3.0 (FAST-LIVO2 is GPL-2.0-only), so it stays a
separate executable and is never imported or linked by the GPL-3.0 app. `tools/package_app.py`
copies every license text, these notices and `SOURCE.md` into `licenses/` of the package.
Corresponding source for each GPL program is this repository at the release tag (FAST-LIVO2 port,
native build files and Python source) and the pinned upstream commits linked above.

## Commercial use

The GPL allows selling copies and services. The FAST-LIVO2 README additionally asks commercial users
to contact its authors; do that before selling anything built around `fastlivo2.exe`. A separate,
proprietary product can reuse the MIT project source, but not the GPL components, and must not link
PyQt6 or the Fluent widgets without commercial licenses (PySide6, LGPL, is the usual alternative).

## Audit status

The repository cannot truthfully be advertised as an all-MIT distribution. Original Python orchestration, independently authored Vulkan colorizer code, and documentation may be offered under MIT when separable and independently authored. The native FAST-LIVO2 adaptation remains subject to GPL-2.0-only terms, and the current PyQt6 GUI stack includes GPL-3.0-only/GPLv3 components. A GUI binary therefore needs GPL-compliant distribution or commercial licenses for the relevant components. Combining components into a distributable application requires preserving applicable notices and source obligations; obtain legal review for a commercial release.

The upstream sources were checked on 2026-09-09. FAST-LIVO2 identifies its source as GPLv2 and links to `LICENSE`; Spirula Studio identifies its repository as GPL-3.0. The packaged Python metadata was also checked on 2026-09-09: PyQt6 6.10.2 declares SPDX `GPL-3.0-only`, Qt 6.10.2 declares LGPL v3, PyQt6-Fluent-Widgets 1.11.3 and PyQt6-Frameless-Window 0.8.2 declare GPLv3 without an SPDX variant, and PyQt6-sip 13.11.0 declares BSD-2-Clause. These links and copied license files are audit references, not substitutes for shipping required source materials.
