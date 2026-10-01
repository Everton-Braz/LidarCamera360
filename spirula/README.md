# Spirula Studio Binary Directory

Bundled executable: **Spirula Studio v2026.9.30** (upstream commit `1943eda`, Windows Vulkan x86_64).

- **Release**: [v2026.9.30](https://github.com/harry7557558/spirula-studio/releases/tag/v2026.9.30)
- **Official archive**: [spirula-2026.9.30-windows-vulkan-x86_64.zip](https://github.com/harry7557558/spirula-studio/releases/download/v2026.9.30/spirula-2026.9.30-windows-vulkan-x86_64.zip)
- **Archive SHA-256**: `79515ab186fc704af6eac22d65365d236a94afde9fa016511cfef2a234e154a9`
- **Bundled `spirula.exe` SHA-256**: `aaedc3079f640944114c17e748f3356716aa308d6ed1ce84d6794da880dda2fe`
- **Usage**: `pipeline_auto_calibrator_and_colorizer.py --run-spirula` uses this executable for SfM. In `sfm auto`, repeated per-folder overrides are supported, e.g. `--camera-model cam0=thin-prism-fisheye --camera-model cam2=pinhole`; `--focal` and `--distortion` accept the same `PREFIX=VALUE` form.
