# Third-party notices

The MIT license in `LICENSE` applies only to original LidarCamera360 material independently authored for this repository. It does not relicense upstream source, bundled executables, or package dependencies.

| Component | Use | License/evidence | Distribution condition |
| --- | --- | --- | --- |
| FAST-LIVO2 | LiDAR-inertial-visual estimator and native adaptation | GPL-2.0-only; `FAST-LIVO2/LICENSE`; [upstream repository](https://github.com/hku-mars/FAST-LIVO2) | Preserve the GPL notice and provide corresponding source, including the build adaptation, for a covered binary. The upstream README requests a separate commercial license discussion for commercial use. |
| Spirula Studio | Optional headless SfM/reconstruction executable | GPL-3.0; [upstream repository and license marker](https://github.com/harry7557558/spirula-studio) | Before publishing a package containing `spirula.exe`, record its exact release/commit and include matching license and source/provenance materials. Review upstream patent notices for optional patented codecs. |
| Sophus | Native math dependency | Upstream license retained with the pinned source | Preserve upstream notices in build or release materials. |
| Vikit | Camera/math dependency used by FAST-LIVO2 | Upstream license retained with the pinned source | Preserve upstream notices in build or release materials. |
| PCL, Eigen, OpenCV, yaml-cpp, OpenMP, Vulkan SDK | Native build/runtime dependencies | Their own upstream licenses; generated notices under `dist/LidarCamera360/_internal/licenses/` when present | Keep generated notices with redistributed binaries and verify exact versions for each release. |
| Python packages | Bag I/O, video decoding, numerics, GUI, packaging | Package metadata and license files in the built environment | Keep the generated license directory with the portable distribution. |
| PyQt6 6.10.2 | GUI Python bindings | GPL-3.0-only (`licenses/PyQt6-6.10.2-GPL-3.0-only.txt`); package metadata declares SPDX `GPL-3.0-only` | A binary using PyQt6 under the GPL option must satisfy GPL obligations. Riverbank also offers a commercial license; obtain it if the intended product terms are incompatible with GPL. |
| Qt 6.10.2 | Qt runtime shipped with PyQt6 | LGPL-3.0-or-later metadata (`licenses/Qt6-6.10.2-LGPL-3.0.txt`) | Preserve Qt notices and LGPL relinking/installation information applicable to the distributed binary. |
| PyQt6-sip 13.11.0 | PyQt6 binding support | BSD-2-Clause (`licenses/PyQt6-sip-13.11.0-BSD-2-Clause.txt`) | Preserve the BSD notice. |
| PyQt6-Fluent-Widgets 1.11.3 | GUI widgets | GPLv3 in package metadata; exact SPDX variant is not declared (`licenses/PyQt6-Fluent-Widgets-1.11.3-GPLv3.txt`) | Treat the GUI distribution as GPLv3-covered unless a compatible commercial or other license is obtained from the upstream author. |
| PyQt6-Frameless-Window 0.8.2 | Frameless window controls | GPLv3 in package metadata; exact SPDX variant is not declared (`licenses/PyQt6-Frameless-Window-0.8.2-GPLv3.txt`) | Preserve the license and source/notice obligations for the GUI distribution. |

## Audit status

The repository cannot truthfully be advertised as an all-MIT distribution. Original Python orchestration, independently authored Vulkan colorizer code, and documentation may be offered under MIT when separable and independently authored. The native FAST-LIVO2 adaptation remains subject to GPL-2.0-only terms, and the current PyQt6 GUI stack includes GPL-3.0-only/GPLv3 components. A GUI binary therefore needs GPL-compliant distribution or commercial licenses for the relevant components. Combining components into a distributable application requires preserving applicable notices and source obligations; obtain legal review for a commercial release.

The upstream sources were checked on 2026-09-09. FAST-LIVO2 identifies its source as GPLv2 and links to `LICENSE`; Spirula Studio identifies its repository as GPL-3.0. The packaged Python metadata was also checked on 2026-09-09: PyQt6 6.10.2 declares SPDX `GPL-3.0-only`, Qt 6.10.2 declares LGPL v3, PyQt6-Fluent-Widgets 1.11.3 and PyQt6-Frameless-Window 0.8.2 declare GPLv3 without an SPDX variant, and PyQt6-sip 13.11.0 declares BSD-2-Clause. These links and copied license files are audit references, not substitutes for shipping required source materials.
