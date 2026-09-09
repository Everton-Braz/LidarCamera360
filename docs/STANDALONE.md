# Windows user guide

LidarCamera360 is a portable Windows x64 application. Keep `LidarCamera360.exe` and `_internal/` together when copying `dist/LidarCamera360/`. The single-file build is under `dist/single-file/` when extraction at startup is acceptable.

## Commands

```powershell
.\LidarCamera360.exe --help
.\LidarCamera360.exe --headless doctor
.\LidarCamera360.exe inspect-bag --bag D:\capture\merged.bag
.\LidarCamera360.exe export-bag --bag D:\capture\merged.bag --output D:\capture\input.flv2
.\LidarCamera360.exe slam --bag D:\capture\merged.bag --lio --output D:\dataset\slam_out --threads 4
.\LidarCamera360.exe extract-insv --insv D:\capture\video.insv --output D:\dataset --fps 1
.\LidarCamera360.exe colorize --dataset D:\dataset --method trajectory --fps 1 --dt -4.2
```

The GUI exposes the same operations. Headless commands do not create a Qt application and suit automation. Return code `0` means success, `2` means invalid input or processing failure, and `130` means cancellation. A cancelled SLAM run flushes the accumulated partial map.

## Bag requirements

Default Raven topics are `/vanjee_722z`, `/vanjee_imu_packets`, and optionally `/camera_front/image/compressed`. Use `inspect-bag` to discover topic names and fields. The reader accepts standard `PointCloud2` XYZ data with per-point timing; it does not accept Livox `CustomMsg` packets. Missing timing, non-monotonic timestamps, malformed records, and unsupported image encodings are rejected.

Pass split, non-overlapping bags after `--bag` when a capture is split. Do not pass a merged bag with its source bags. Use a fresh output directory for each SLAM run.

## Workflow data

`slam` writes a raw map, trajectory, and `run.json`. Camera fusion covers only synchronized data. If the camera ends first, pending LiDAR scans are recorded in the metadata and log; use `--lio` when the full LiDAR/IMU interval is required.

`extract-insv` writes paired front/rear JPEGs and `images/frames.json`, whose measured presentation times are used by alignment and colorization. Both lens tracks must decode successfully.

Colorization needs `pcd/all_raw_points.pcd`, `result/Raven_3DMakerPro_Scan.txt`, paired frames, and a calibration profile. Reconstruction mode uses Spirula Studio when `--run-spirula` is requested or its sparse reconstruction is missing. Trajectory mode uses the calibrated trajectory and optional reconstruction drift correction. Canonical files are `reconstruction_colorized.ply/.pcd`, `trajectory_colorized.ply/.pcd`, `trajectory.txt`, and `configs/rig_profile.json`; `sfm` and `direct` are legacy aliases.

## Native engine

```powershell
.\_internal\bin\fastlivo2.exe --input D:\capture\input.flv2 --config .\_internal\FAST-LIVO2\config\raven.yaml --camera .\_internal\FAST-LIVO2\config\camera_raven.yaml --output D:\slam --lio
```

`native/prepare_sources.py` makes a build-only adaptation of the supplied ROS estimator. Middleware publication is disabled, YAML supplies parameters, and the offline runner owns ingestion and output. The upstream tree remains available for audit and its license applies to the adapted estimator.

## Troubleshooting

Run `doctor` first. If Spirula is needed, configure its path in Settings or place `spirula.exe` in `spirula/` (or packaged `bin/`). Vulkan failures can be diagnosed with `--no-vulkan`; CPU processing is slower. Do not mix merged and component bags or reuse a completed output directory.
