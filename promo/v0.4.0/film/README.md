# LidarCamera360 v0.4.0 release film

A 1920x1080, 30 fps, 39 s video made with [fframes](https://fframes.studio) 1.4.0 and
rendered with the Skia (Vulkan) backend. It has no narration and no music by default.

| file | what |
| --- | --- |
| `src/lib.rs` | the video: 7 scenes, animation, audio map |
| `src/main.rs` | the command line (`fframes::cli`) and the loading of `footage/` |
| `media/` | fonts, UI captures and the logo, embedded into the binary |
| `footage/` | optional real footage, read at runtime (not committed) |

## Scenes

| # | scene | length | content |
| --- | --- | --- | --- |
| 1 | Hook | 4 s | "Any rig. Calibrated per capture." over the 3DGS fly-through |
| 2 | Rig | 6 s | Rig profiles v2 (UI capture, zoom) |
| 3 | RsOne | 7 s | Insta360 ONE RS: the two-INSV checkbox is clicked in the Studio |
| 4 | Calib | 6 s | Self-calibration, 3 cards |
| 5 | Clean | 7 s | Before/after wipe of `--remove-dynamic` |
| 6 | Export | 5 s | `project.lc360` → PCD, LAZ/LAS, GeoJSON, GPS track, trajectory |
| 7 | EndCard | 4 s | Logo, version, GitHub link |

> The RsOne scene announces Insta360 ONE RS support. Render the final film only after
> a real ONE RS dataset has been processed successfully.

## Footage slots

Put these in `footage/` (create the folder). A missing file falls back to a procedural
point-cloud animation, so the film always renders.

| file | used in | what |
| --- | --- | --- |
| `3dgs_flythrough.mp4` | Hook | 3DGS or colored-cloud fly-through, ≥ 4 s, 1920x1080, no UI |
| `cloud_before.png` | Clean | cloud with people, 1920x1080 |
| `cloud_after.png` | Clean | same viewpoint after `--remove-dynamic` |
| `music.mp3` | whole film | optional; enable it in `audio()` in `src/lib.rs` |

## Build environment (Windows)

Set these in each PowerShell window first:

```powershell
$env:FFMPEG_DIR="$env:USERPROFILE\tools\ffmpeg-n9.0-latest-win64-gpl-shared-9.0"
$env:LIBCLANG_PATH="C:\Program Files\LLVM\bin"
$env:PATH="$env:FFMPEG_DIR\bin;$env:PATH"
```

`Cargo.lock` pins `fframes-media`, `fframes-media-dir-macro` and `svgr-macro` at 1.4.0.
Version 1.4.1 pulls `usvgr 0.50`, which clashes with fframes 1.4.0 (`usvgr 0.46`), and the
build fails with E0308. Do not run a plain `cargo update`.

## Work on it

```sh
cargo run --release -- timeline                  # scenes and duration
cargo run --release -- inspect                   # missing media/fonts, clipped text
cargo run --release -- frame Rig@3.5s RsOne@5s   # PNGs into frames/
cargo run --release -- strip Clean -n 8          # contact sheet
cargo run --release -- preview                   # real-time GPU player (space, h/l, q)
cargo run --release -- render -o release_v0.4.0.mp4
```

Until the footage exists, `inspect` reports the three missing footage files; that is expected.
