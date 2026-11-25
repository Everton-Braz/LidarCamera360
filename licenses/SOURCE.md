# Corresponding source

Every LidarCamera360 release is built from the public repository
<https://github.com/Everton-Braz/LidarCamera360> at the tag named after the release version
(for example `v0.4.0`). That tag holds the complete corresponding source of the GPL programs in
the package:

| Program | Source in the tagged repository |
| --- | --- |
| `LidarCamera360.exe` | `raven_app/`, `scripts/`, `lidarcamera360.py`, `tools/package_app.py`, `requirements.txt` |
| `bin/fastlivo2.exe` | `native/main.cpp`, `native/CMakeLists.txt`, `native/prepare_sources.py`, `FAST-LIVO2/` |
| `bin/rfdetr-masker.exe` | `native/rfdetr_masker/` |
| `bin/spirula.exe` | Unmodified upstream binary; see `Spirula-Studio-SOURCE.md` |
| `openmvs/*.exe` (AGPL-3.0) | `third_party/openmvs/`; see `OpenMVS-SOURCE.md` |

Build instructions are in `docs/BUILD.md`. If a tag is ever unavailable, write to the maintainer
through the repository's issue tracker and the source will be provided for at least three years
after the release.
