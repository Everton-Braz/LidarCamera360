# FIEC pipeline log review — 2026-10-01

Source: `D:\ARQUIVOS_TESTE_2\FIEC\LIDAR_20260929Fiec_studio_output`. The run exited successfully (`0`) in `2746.2 s` (`45.8 min`). The resulting output was reported as acceptable; this review did not modify the test output.

| Stage | Logged time |
| --- | ---: |
| INSV and auxiliary frame extraction combined | 207.09 s |
| Temporal analysis | 6.40 s |
| SLAM | 207.07 s |
| SfM and alignment | 1035.69 s |
| Mask generation | 396.56 s |
| Colorization (internal log timer; stage-history entry is empty) | 264.50 s |
| Stage 7 packaging/export | 566.60 s |

The stage-history entries total `2419.41 s`; adding the colorizer's internal `264.50 s` timer gives `2683.91 s`, leaving `62.29 s` outside this approximate breakdown. The missing stage-6 history is also a progress/timing-reporting improvement opportunity. SfM/alignment is the largest logged stage. Packaging/export is next; it writes a seed of more than 32 million points in three formats and is a strong candidate for streamed or combined output work that preserves the current deliverables.

The run used `1386` INSV pairs and `1339` auxiliary-camera frames. The auxiliary camera registered in `1132/1339` frames (`84.5%`). The Add Source settings showed camera 2 as `PINHOLE` with `fx=3072`, but the final sparse reconstruction reports all three cameras with COLMAP model ID `10` (`THIN_PRISM_FISHEYE`). The log therefore does not show the selected auxiliary model reaching SfM; verify the per-folder camera-model override and inspect the resulting `cameras.bin` after the routing fix. Also compare scene-up orientation against the expected vertical before accepting the reconstruction orientation.

The auxiliary time offset was unknown or ambiguous, so rig-constraint injection was skipped. The log reports `15.69 cm` trajectory-alignment RMSE and `dt=16.3399 s`; this is not a cloud-geometry accuracy measurement. Treat these as diagnostics until synchronization and camera geometry are independently checked. Nine GPS fixes were detected, but the log alone does not validate their alignment to the reconstructed scene.

Mask-supported operator removal reduced the 3DGS seed from `32,621,799` raw points to `32,612,698`, removing `9,101` points (`0.028%`). Image masks separately exclude observations during colorization; this point-count reduction does not measure image-mask coverage.

In a separate INSV extraction comparison, parallel sharpness selection took `54.76 s` versus `67.48 s` (`18.8%` faster) and selected the same `1386` pairs. This supports the extraction change without changing frame selection. For further performance work, prioritize SfM/alignment first, then the large seed export; keep the accepted FIEC output unchanged while validating those changes.

Seed export now defaults to 100% and accepts `--seed-percent` or the UI slider/numeric field. The UI reads cloud headers to preview the count. For the current colored FIEC cloud, 50% exports 16,306,349 points and 25% exports 8,153,174 points. PLY/BIN/TXT receive the same selected seed; the source cloud remains unchanged.

Auxiliary extraction now seeks past unused video sections when GOP spacing permits, retaining the original three candidate frames per sampling window. A 60-second FIEC decode-and-sharpness comparison matched all 180 candidate timestamps and scores: median elapsed time was `5.38 s` for full decoding and `4.97 s` with seeking (about 7.7% less time). This excludes JPEG writing and is not an end-to-end pipeline speed measurement. Synthetic integration tests also verify identical exported JPEGs and full-decoding fallback on seek failure. Rotation metadata is applied before writing frames, including this video's 180-degree rotation.
