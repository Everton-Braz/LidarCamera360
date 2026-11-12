# PIER: LiDAR ↔ Insta360 X4 calibration (photometric)

**Date:** 2026-10-26 · **Dataset:** `D:\ARQUIVOS_TESTE_2\PIER` · **Script:** `scripts/photometric_rig_calibration.py`

## Result

| | Current `rig_calibration.json` (sample_sfm) | **Photometric (new)** |
|---|---|---|
| dt_sync (SLAM reference) | −7.196 s | **−7.027 s** |
| Cam0 lever arm | 19.3 cm (−2.2, 11.8, 15.1) | **17.9 cm (1.6, 15.1, 9.5)** |
| Lens baseline / relative angle | 6.6 cm / 179.1° | **2.8 cm / 179.92°** |
| Held-out photometric cost (82 frames never used in the fit) | 0.919 | **0.648 (−30 %)** |
| Held-out median \|residual\| (σ units) | 0.497 | **0.347** |
| Multi-view RGB RMSE, 1.34 M points, 164 frames | 35.5 | **27.8 (−22 %)** |

Output: `LidarCamera360_CLI_Output\rig_calibration_photometric.json`. It uses the same schema as `rig_calibration.json`. Its `dt_sync_seconds` is already in the SLAM-trajectory reference that `colorize_via_direct_rigid` uses.

Visual proof: `docs/images/pier_calibration_overlay.jpg` shows LiDAR projected onto fast-turn and slow frames, before and after. `docs/images/pier_calibration_topdown.jpg` shows the colorized deck from above, before and after.

Reproducibility: an independent run of the shipped script (3 blur stages, its own frame sampling) landed within 0.05°, 0.7 cm and 0.4 ms of these values.

## Diagnosis

1. **The time offset is the main error, not the mount.** A 1-D scan of dt with the hardware profile has a single sharp minimum at −7.03 s. The cost there is 0.70, against 0.88 at the gyro value. During PIER's 60–70°/s turns, 170 ms means about 11° of pointing error.
2. **Two clock references are mixed in the code.** `workflow.auto_sync_imu_gyro` measures dt from the first IMU packet in the bag. The colorizers apply it from `t_slam[0]`, which on PIER is **0.346 s later**. Fixing only this (dt = −6.850 s) makes things **worse** (held-out cost 0.983), so there is also an opposite camera-side offset of about −0.18 s (video PTS vs the INSV gyro/exposure clock). Without the INSV trailer I can't separate the two. Until they are separated, dt has to be refined from the images.
3. **The hardware profile `raven_x4_montagem_1` is nearly right.** Only 1.7° yaw, 3.4 cm and dt needed correcting. The sample-SfM recalibration (19–25 cm lever arm, 6.6 cm lens baseline) is physically wrong and scores no better than the profile. Earlier SfM/Sim(3) attempts (lever arm 54–130 cm, RMSE 8.9 m) were rightly rejected.

## Method (spec §7.1 E_photo + E_prior)

- 250k LiDAR points within 18 m of the trajectory. Full THIN_PRISM_FISHEYE projection (atan2 form, valid past 180°).
- Visibility is frozen per stage: z-buffer, person masks, fisheye circle, θ < 96°.
- Cost: mean Cauchy deviation of each point's exposure-normalized gray value from its multi-view mean. It penalizes ghosting directly, unlike edge distance.
- Steps: dense dt scan (±0.6 s), then Powell over [rotvec C0, t C0, rotvec C0→C1, dt] with blur σ = 4 → 2 → 1 → 0 px (at 960 px), plus a weak prior to the profile.
- Converged parameters were stable across all four blur levels: rotation within 0.07°, translation within 2 cm, dt within 0.4 ms.

## How to use

```bash
# Colorize PIER with the new calibration (no recalibration):
python scripts/pipeline_auto_calibrator_and_colorizer.py --dataset D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output \
    --method direct --calib D:/ARQUIVOS_TESTE_2/PIER/LidarCamera360_CLI_Output/rig_calibration_photometric.json --dt -7.0266
# (--dt is passed explicitly: colorize_via_direct_rigid lets dt_override win over the JSON,
#  and the GUI workflow passes the gyro dt as override.)
# Recalibrate any dataset (about 25 min on 2 CPU cores with 40 frames/lens):
python scripts/photometric_rig_calibration.py --dataset <output_dir> \
    --init configs/profiles/raven_x4_montagem_1.json --dt-init <gyro dt> \
    --masks <studio_output>/masks
```

## Recommended code changes

1. After gyro sync, always run the photometric dt scan (cheap: about 30 cost evaluations). Never pass the gyro dt straight to colorization.
2. Make the dt reference explicit (`dt_sync_reference`) and convert between bag-IMU-origin and SLAM-origin in one place.
3. Use the hardware profile + photometric refinement as the default calibration path. Use sample-SfM only as a fallback, and reject any result with a lens baseline above 4 cm or a lever arm more than 5 cm from the profile.
4. Check `20261022home` for regressions with the same script. Nothing there was run in this session.
