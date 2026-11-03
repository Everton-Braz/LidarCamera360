# Spirula Studio Binary Directory

Bundled executable: **Spirula Studio v2026.10.30** (upstream commit `6ad73ef`, Windows Vulkan x86_64).

- **Release**: [v2026.10.30](https://github.com/harry7557558/spirula-studio/releases/tag/v2026.10.30)
- **Official archive**: [spirula-2026.10.30-windows-vulkan-x86_64.zip](https://github.com/harry7557558/spirula-studio/releases/download/v2026.10.30/spirula-2026.10.30-windows-vulkan-x86_64.zip)
- **Archive SHA-256**: `dfaf1376ab78acf4bfbaaeb7a3f12b7c06ab30bf1d9eee6082100dbc1e7b71dd`
- **Bundled `spirula.exe` SHA-256**: `5ea84267b276ea92e62bc1521977c836dfa7141eb3b9e2a12637cc13a2a859f2`
- **Usage**: `pipeline_auto_calibrator_and_colorizer.py --run-spirula` uses this executable for SfM. In `sfm auto`, repeated per-folder overrides are supported, e.g. `--camera-model cam0=thin-prism-fisheye --camera-model cam2=pinhole`; `--focal` and `--distortion` accept the same `PREFIX=VALUE` form.
