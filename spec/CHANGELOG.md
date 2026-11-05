# Package contract changelog

A major version changes when a key is removed, renamed or changes meaning. A minor version adds optional keys or
finding codes that old readers can ignore.

## 1.0

First published contract.

- Manifest schema `lc360-project` with required `schema`, `version`, `created_by`, `units`, `frame`, `assets.cloud`.
- Optional `id` (UUID v4) and `updated_utc`.
- Pose file schema `lc360-camera-poses-1.0` (`THIN_PRISM_FISHEYE`).
- Path rule: relative, inside the package, symbolic links followed.
- Finding codes `E_JSON`, `E_SCHEMA`, `E_VERSION`, `E_PATH_ESCAPE`, `E_MISSING_ASSET`, `E_POSES`, `W_NO_ID`,
  `W_DUP_ID`.
- Conformance examples under `spec/conformance/`.
