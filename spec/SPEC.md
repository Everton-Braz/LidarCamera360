# `.lc360` project package, version 1

A package is a folder. Its `project.json` names the files inside it. The open exporter writes packages; any reader
(including the paid Pro app) opens them. A reader never writes into a package: what it generates goes somewhere else.

The machine-readable parts of this contract are:

- `raven_app/schema/lc360-project.schema.json` (the manifest)
- `raven_app/schema/lc360-camera-poses-1.0.schema.json` (the pose file)
- `raven_app/project_validate.py` (the validator and the error codes)
- `spec/conformance/` (example packages with the findings each must produce)

If this text and those files disagree, the files win and this text is a bug.

## Manifest (`project.json`)

| Key | Required | Meaning |
| --- | --- | --- |
| `schema` | yes | The string `lc360-project`. |
| `version` | yes | `"1.<minor>"`. A reader accepts every minor of major 1 and ignores keys it does not know. |
| `id` | no | UUID version 4, lower case. Identifies the package across folders and copies. |
| `updated_utc` | no | ISO 8601 time of the last change to the package. Readers use it to decide whether a cache is stale. |
| `created_by` | yes | `{app, version}` of the exporter. |
| `units` | yes | The string `m`. |
| `frame` | yes | `{name, up}`; `up` is `+Z` or `+Y`. |
| `crs` | no | `{epsg, transform}`; `transform` is a path inside the package. |
| `capture` | no | `scanner`, `rig_profile` (a path), `insv`, `start_time`, `duration_s`. |
| `assets` | yes | Map from a name to a path inside the package. `cloud` is required; `images`, `camera_poses`, `trajectory`, `gaussian_seed` are known; other names are allowed. |
| `dt_sync_seconds`, `runs`, `quality` | no | `quality` is a path inside the package. |

Unknown keys are ignored at every level. Adding an optional key is a minor change; removing or renaming a key, or
changing its meaning, is a major change.

## Pose file (`assets.camera_poses`)

`lens_model` is `THIN_PRISM_FISHEYE`. `intrinsics` maps a camera name to `width`, `height`, `fx`, `fy`, `cx`, `cy`
and optional distortion terms. `images` lists one entry per photo with `image`, `cam`, `frame`, `t`, `position`,
`q_world_cam`, `forward`, `yaw_deg`, `pitch_deg`. The schema file is the full definition.

## Path rule

Every path in the manifest (`assets.*`, `crs.transform`, `capture.rig_profile`, `quality`) MUST be a relative path
that stays inside the package folder after symbolic links are followed. These are rejected as `E_PATH_ESCAPE`: an
empty value, a leading `/` or `\`, a drive letter (`C:`), a UNC path (`//host/share`), any backslash, any `..`
segment, and a path whose resolved location is outside the package folder (checked with a path-component test, not
a string prefix test). `capture.insv` is a record of the original recording and is not read as a path.

## Findings

| Code | Level | Meaning | Blocks a reader |
| --- | --- | --- | --- |
| `E_JSON` | 1 | `project.json` is missing or is not valid JSON. | yes |
| `E_SCHEMA` | 1 | The manifest does not fit the schema (every error is reported). | yes |
| `E_VERSION` | 1 | The major version is not 1. | yes |
| `E_PATH_ESCAPE` | 2 | A manifest path breaks the path rule. | yes |
| `E_MISSING_ASSET` | 2 | A listed file does not exist. | on `assets.cloud` only |
| `E_POSES` | 2 | The pose file does not fit its schema. | no |
| `W_NO_ID` | 2 | The manifest has no `id`. | no |
| `W_DUP_ID` | 2 | Two package folders carry one `id`. Needs two packages, so a reader raises it, not the validator. | no |

Level 2 runs only when level 1 found no error. The report's `level` is 0 when level 1 failed, 1 when level 2 found
an error, 2 when the package is clean (warnings do not lower it).

A reader decides what it needs from the package. A package without `images` or `camera_poses` can still be viewed;
a feature that needs a missing asset says which one is missing instead of failing.

## Reader rules

- A reader MUST NOT create, change or delete anything inside the package folder.
- A reader MUST refuse a package with a blocking finding and say which finding.
- A reader SHOULD keep its own derived data in a cache outside the package, keyed by `id` when present.

## Conformance

`spec/conformance/valid/` and `spec/conformance/invalid/` hold example packages. Each has an `expect.json` with the
error and warning codes the validator must report. Every error code has at least one invalid example; adding a code
means adding an example.
