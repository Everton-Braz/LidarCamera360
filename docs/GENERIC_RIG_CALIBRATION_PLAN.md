# Plan: rig calibration for any LiDAR + camera + mount

**Date:** 2026-10-26 · **Branch:** dev · **Status:** Phase 0 and Phase 1 (two-lens rigs, GUI profile/refinement selectors) implemented; phases 2–5 pending

## 1. Why: what PIER and HarmonyMall showed

Both datasets use the same rig (Raven + Insta360 X4, profile `raven_x4_montagem_1`). Both had the same root cause: the app trusts SfM over the hardware profile and the gyro.

| Dataset | What the app did | Held-out photometric cost | Fix |
|---|---|---|---|
| PIER | gyro dt −7.196 s + sample-SfM rig (19 cm lever arm, 6.6 cm lens baseline) | 0.919 | profile + photometric refine: dt −7.027 s, **0.648 (−30 %)** |
| HarmonyMall | SfM alignment **overwrote** the gyro dt (3.337 → 4.277 s) + SfM rig (23/26 cm lever arms, lens angle 179.0°) | 1.170 | profile + photometric refine: dt 3.354 s, **0.928 (−21 %)** |

Held-out = frames never used by the fit (82 on PIER, 128 on HarmonyMall).

Specifics:
- **HarmonyMall dt scan (±1.2 s):** a single minimum at +0.02 s from the gyro value. The SfM dt is 0.94 s away and costs more (up to 1.20).
- **HarmonyMall projection overlay** (`D:\ARQUIVOS_TESTE_2\HarmonyMall\compare_20261026\overlay_frame811.png`): with the SfM calibration the car outlines miss the cars; with dt ≈ 3.35 s they fit.

Calibration alone did not remove the HarmonyMall ghosting. Two more defects were found in the colourisers:

- **No time gating.** Each point takes its best views from the whole capture. HarmonyMall is a parking lot revisited about 700 s later, with cars that left, arrived or swapped spots, and with people walking. Colours from one pass landed on the other pass's geometry: the brown Duster was painted with the red car next to it.
  - `scan_ranges.csv` gives every point its scan time. Colouring each point only from frames within ±6 s of that time fixes it. The z-buffer then uses the scene as it was at that moment.
  - This is `scripts/time_gated_colorizer.py`: 98.4 % of points coloured in-window, about 21 min on CPU.
  - Proof: `overlay_frame811_split.png` projects LiDAR onto the real frame. The car the time-gated cloud colours red is the car that is red in the image.
- **Direct mode silently uses SfM poses.** `colorize_via_direct_rigid` applies a "trajectory drift correction" from SfM keyframes whenever `colmap_to_lidar_alignment.json` and `sparse/0/images.bin` exist, even when that alignment is poor. On HarmonyMall the trajectory RMSE was 640 cm.

The lessons generalise to any setup:
1. **The profile is a strong prior.** A rigid mount doesn't change between captures. A per-capture solve may only nudge it, by a few degrees and a few centimetres.
2. **dt is per capture and the most sensitive parameter.** The gyro is a good initial value. Images must confirm it (photometric dt scan). SfM must never silently overwrite it.
3. **Every calibration needs an objective acceptance test.** Held-out multi-view colour consistency works for any camera model, because it only compares pixel values of the same 3D point across frames.

## 2. Target architecture

### 2.1 Rig profile v2: one file describes any setup

`configs/profiles/<id>.json` for bundled profiles; `%APPDATA%/LidarCamera360/profiles/<id>.json` for user profiles (the packaged app cannot write into `_MEIPASS`).

```jsonc
{
  "schema": 2,
  "profile_id": "raven_x4_montagem_1",
  "name": "Raven + Insta360 X4 (mount 1)",
  "scanner": {
    "model": "3DMakerPro Raven",            // free text
    "preset": "raven",                       // raven | eagle | generic → topics + FAST-LIVO2 config
    "lidar_topic": "/vanjee_722z", "imu_topic": "/vanjee_imu_packets",
    "slam_config": "FAST-LIVO2/config/raven.yaml"
  },
  "bodies": [                                // a body = cameras sharing one clock and one rigid housing
    {
      "body_id": "x4",
      "source": "insv",                      // insv | insv_pair | video | image_folder | bag_topic
      "clock": {"sync": "gyro", "dt_reference": "slam_first_pose", "dt_tolerance_s": 0.6},
      "cameras": [
        {"name": "cam0", "model": "THIN_PRISM_FISHEYE", "width": 3840, "height": 3840,
         "intrinsics": {"fx": 0, "fy": 0, "cx": 0, "cy": 0, "k1": 0, "...": 0},
         "valid_fov_deg": 192, "valid_radius_px": 1620,
         "T_lidar_to_cam": [[...]]},
        {"name": "cam1", "...": "..."}
      ],
      "intra_body_prior": {"rot_sigma_deg": 0.5, "baseline_cm": 2.8, "baseline_tol_cm": 1.0}
    }
  ],
  "mount_prior": {"rot_sigma_deg": 3.0, "trans_sigma_cm": 5.0},
  "validated_on": [
    {"dataset": "PIER", "date": "2026-10-26", "holdout_cost_before": 0.919, "holdout_cost_after": 0.648}
  ]
}
```

- **Any camera model** the projection layer already supports: PINHOLE, OPENCV, OPENCV_FISHEYE, THIN_PRISM_FISHEYE and EQUIRECTANGULAR. They reuse the 12-parameter protocol in `pipeline._colorization_camera_params` / `project_thin_prism`.
- **Any number of cameras and bodies.** One body can be an X4 (2 lenses, one clock). A second body can be a GoPro (1 camera, its own clock). `ThirdCameraConfig` maps directly onto a body with one camera.
- **v1 migration.** The loader converts the current flat files (`cam0_front_intrinsics`, `T_lidar_to_cam{0,1}_rigid_4x4`, `rig_profile_eagle.json`, `calibracao_rigida_*.json`) into this structure in memory. During the transition, writers also emit the v1 keys so the existing colorizers keep working.

### 2.2 New package `raven_app/rig/`

| Module | Responsibility | Replaces / reuses |
|---|---|---|
| `profile.py` | `RigProfile`, `Body`, `CameraSpec` dataclasses; load / validate / migrate / save; list bundled and user profiles; the default comes from settings, not a filename | `profile_manager.py` (hardcoded default, no validation), dead code in `dataset.py` |
| `camera_models.py` | One `project(P, camera)` and `valid_mask(camera, shape)` per model, including the fisheye circle from `valid_radius_px` instead of the hardcoded 1620 px | `project_thin_prism`, `project_equirectangular`, the 1620 px constants in 6 files |
| `time_reference.py` | One place that converts dt between bag-IMU-origin and SLAM-first-pose origin; every dt carries `dt_reference` | the mixed references documented in `photometric_rig_calibration.py` and the wrong comment at `workflow.py:2830` |
| `photometric.py` | The calibrator, promoted from `scripts/photometric_rig_calibration.py` and generalised to N bodies × M cameras. Parameters: per-body mount delta (6) + per-camera intra-body delta (3, tight prior) + per-body dt. Includes a coarse mode for unknown mounts (see 2.4). Built-in held-out split. | `scripts/photometric_rig_calibration.py` |
| `gates.py` | Physical and statistical acceptance. A candidate is accepted only if all hold: (a) held-out cost improves on the profile by ≥ 2 %; (b) change from the profile ≤ `k·σ` (rotation, translation); (c) intra-body baseline and relative angle within tolerance; (d) dt within `dt_tolerance_s` of gyro, when gyro sync was accepted. Otherwise keep the profile and record why. | `validate_calibration_file` (only a 40 cm lever check), the timelapse-only gates |
| `resolver.py` | One precedence for every command: explicit `--calib` > accepted per-dataset calibration whose `source_signature` matches > selected profile. Rejected files never win. | `resolve_calibration_file` (hardcodes mount 1), the second precedence inside `colorize_via_direct_rigid` |

### 2.3 Workflow changes (`raven_app/workflow.py`)

1. **Selected profile, not a hardcoded file.** `execute_unified_workflow(profile_id=...)`. `resolve_calibration_file` goes through `rig.resolver`.
2. **dt policy.**
   - `--dt` given → it is final.
   - Gyro sync accepted → it is the initial value for the photometric dt scan (±`dt_tolerance_s`, about 30 cost evaluations, a few minutes).
   - SfM dt is only a fallback when gyro sync fails, and it must also pass the photometric scan.
   - Remove the unconditional overwrite at stages 4/5 (`align_colmap_to_lidar` → `dt_sync`).
3. **New stage "Rig refinement" after SLAM, before colorization.** `--rig-refine {photometric,sfm,none}`, default `photometric`. It runs `rig.photometric` with the profile as prior, applies `rig.gates`, then writes `rig_calibration.json` with `calibration_source`, `dt_reference`, the held-out report and `source_signature`.
4. **SfM recalibration becomes opt-in** (`--rig-refine sfm`). Its result must also pass `rig.gates`. With the PIER and HarmonyMall SfM rigs this would have caught the 6.6 cm baseline and the 22–26 cm lever arms.
5. **Reconstruction mode.** When the SfM→LiDAR alignment ICP RMSE exceeds a threshold (HarmonyMall: 19 cm), fall back to direct mode with the refined rig, and log it. This is the "colours from SfM poses" path that produced the ghosting.
6. **Time-gated colourisation, default on** (`--time-window 6`).
   - Every view of a point must be within ±window of the point's scan time (`scan_ranges.csv`). The z-buffer uses only the points scanned in that window.
   - Port it into `vulkan_engine.colorize_views`: per view, pass the point index range of the window instead of the whole cloud. This is also faster, because each frame touches about 2 % of the cloud.
   - Points with no in-window view fall back to a wider window and are flagged.
   - Remove the SfM "drift correction" from direct mode, or gate it on alignment quality (trajectory RMSE ≤ 10 cm) plus a held-out photometric check.
7. **Packaging.** Never delete the colorized cloud when no export flag was given; keep PLY by default. Today a run reports SUCCESS with only a JSON left in `deliverables/` (`workflow.py:3182-3189`).
8. **Masks.** Check that `--masks-dir` matches the images' `frames.json` (fps and timestamps) before use. Today, masks from a 1 fps extraction are silently applied by filename to 2 fps images (PIER).

### 2.4 Calibrating a brand-new setup (no validated profile yet)

The PIER and HarmonyMall fixes worked because a good prior existed. For a new user:

1. **Wizard input.**
   - Scanner preset (or topics) and camera type. The camera type fills the model, resolution and default intrinsics from a camera catalogue: Insta360 X3/X4/X5, ONE RS 1-inch, GoPro MAX, a generic equirectangular .mp4, a generic pinhole or fisheye.
   - The mount: the offset measured with a ruler (x/y/z in cm), plus an orientation chosen from pictures ("camera above scanner, lenses left/right", "lenses front/back", …).
   - That gives about ±5 cm and ±15°.
2. **Coarse search.**
   - dt from gyro if the camera has IMU telemetry (INSV); otherwise a wide photometric dt scan (±10 s at 0.1 s steps, then refine).
   - Rotation: a yaw/pitch/roll grid at heavy blur (σ = 16 → 8 px) on the photometric cost. A colour-consistency cost has a wide basin when images are blurred.
3. **Fine solve.** The same coarse-to-fine Powell as today, with weak priors.
4. **Intrinsics.** Use the catalogue values by default. Optionally refine them from the sample SfM (COLMAP `cameras.bin`, reusing `_third_camera_intrinsics_from_colmap`). The photometric solver does not touch intrinsics.
5. **Acceptance.** Run held-out gates on a second, independent capture (recommended) or on held-out frames.
6. **Save as profile.** Later captures only need the cheap dt scan plus an optional refine.

**Capture advice the wizard shows:** 2–3 minutes; walk past textured structures within 3–15 m; include turns; no other people close to the rig.

### 2.5 CLI

```text
lidarcamera360 profile list | show <id> | validate <file> | import <file> | export <id> <file>
lidarcamera360 profile create --from-calibration <rig_calibration.json> --name "My rig"
lidarcamera360 calibrate --dataset <output_dir> --profile <id> [--refine photometric|sfm|none]
                         [--dt-window 0.6] [--frames 40] [--coarse] [--save-profile "name"] [--report out.json]
lidarcamera360 workflow ... --profile <id> --rig-refine photometric|sfm|none --dt-source gyro|photometric|manual
lidarcamera360 colorize ... --profile <id>
```

- `--calib` stays and wins over `--profile`.
- `--recalibrate` / `--recalibrate-from-sfm` become aliases of `--rig-refine sfm`, with a deprecation note.

### 2.6 MCP (`raven_app/mcp_server.py`)

It has no processing tools today. Add:

| Tool | Notes |
|---|---|
| `list_profiles`, `get_profile`, `validate_profile`, `save_profile` | thin wrappers over `rig.profile` |
| `calibrate_rig` | starts a job and returns `job_id`; the result has the calibration, held-out metrics and gate verdict |
| `run_workflow`, `colorize` | start jobs with the same arguments as the CLI |
| `get_job_status`, `cancel_job` | progress lines, final status, output paths |

Long-running work goes through a job model, because the JSON-RPC handler is synchronous today. It reuses `ProcessRunner` semantics: spawn the CLI and parse stdout progress.

### 2.7 GUI (PyQt6 + qfluentwidgets)

- **"Rig Calibration" page → "Rig Profiles"** (`views/calibration_view.py`):
  - A profile list (bundled and user) and a form editor instead of the raw JSON editor: scanner block, bodies, a camera table with a model drop-down, and intrinsics import (file / COLMAP).
  - Mount offset in cm plus an orientation preset with a small 3D preview of the axes.
  - **"Validate on dataset…"**: runs `calibrate --refine none` (metrics only) and `--refine photometric`, then shows before/after held-out cost, the gate verdict, a projection overlay of one frame, and a colourised crop.
  - **"Save as new profile"**.
- **"New rig" wizard:** the steps in 2.4, with the capture advice.
- **Unified workflow page** (`views/unified_workflow_view.py`):
  - A profile combo, actually passed as `--profile` (today the selection is never sent).
  - Rig refinement: Photometric (recommended) / SfM / None. dt source: Gyro + photometric / Manual.
  - The "Auto-Recalibrate from SfM" checkbox moves under "Advanced".
  - After a run, a "Calibration verdict" card: dt used and its reference, mount delta vs profile, held-out improvement, and a link to the overlay image.

## 3. Phases

| Phase | Content | Exit criterion |
|---|---|---|
| **0. Stop the damage** (small, now) | time-gated colourisation (port of `scripts/time_gated_colorizer.py`); no SfM drift correction in direct mode; dt policy (no SfM overwrite when gyro is accepted or `--dt` is given); gates on SfM recalibration (lens baseline, relative angle, delta from profile); keep clouds when no export flag; mask fps/timestamp check; resolver uses the selected profile | PIER and HarmonyMall default workflows reproduce the profile+gyro quality (held-out ≤ 0.95 / ≤ 1.00) |
| **1. Photometric refine in the app** | `rig/photometric.py` (2-lens, today's maths) + `rig/gates.py` + `calibrate` CLI + workflow stage, default on | PIER ≤ 0.66, HarmonyMall ≤ 0.94 held-out; 20261022home no regression |
| **2. Profile v2 + N cameras** | `rig/profile.py` with migration, `camera_models.py`, N bodies × M cameras, `ThirdCameraConfig` folded in | Raven+X4, Eagle+X4 and an equirect .mp4 dataset all calibrate through the same code path |
| **3. New-rig bootstrap** | coarse search, camera catalogue, wizard | Recover a synthetic 15° / 10 cm / 3 s offset on PIER from the measured-mount prior |
| **4. GUI + MCP** | pages and tools above | GUI and MCP runs produce byte-identical calibration JSON to the CLI for the same inputs |
| **5. Speed** | move the cost to Vulkan or numba; today one refine takes 25–40 min on CPU | full refine < 5 min; dt scan < 1 min |

## 4. Tests

- **Profile:** schema validation and v1→v2 migration for every file in `configs/` and the repo root.
- **Resolver:** precedence and rejected-file handling (there are no tests for `resolve_calibration_file` or `validate_calibration_file` today).
- **`time_reference`:** IMU0 ↔ SLAM0 round trip.
- **Gates:** they reject the recorded PIER and HarmonyMall SfM rigs and accept the photometric ones.
- **Synthetic photometric test:** render a coloured synthetic cloud into fisheye, pinhole and equirect cameras with a known perturbation (2°, 3 cm, 80 ms). Recover it within 0.2°, 0.5 cm and 5 ms.
- **CLI and MCP:** argument round-trip tests for `profile`, `calibrate`, `run_workflow`.
- **Regression datasets:** PIER, HarmonyMall, 20261022home, with stored held-out costs as thresholds (manual and slow, not in CI).

## 5. Open questions for you

1. User profiles in `%APPDATA%` (survives updates) or next to the exe (portable)?
2. Default when a dataset has no IMU telemetry and no profile: run the wide dt scan automatically (slow), or ask for a manual dt?
3. Should the workflow refuse to colourise when every calibration candidate fails its gates, or colourise with the profile and flag the result?
