# Image masks

The built-in RF-DETR masker writes **one combined binary PNG for each source image** under `dataset/masks/`. White (`255`) keeps a pixel; black (`0`) excludes it. Person detections, optional fisheye-border cutoff, and user-defined rig rectangles and ellipses are combined in that same PNG. There are no separate person or rig-mask copies.

The mask settings dialog is available from the app's masking controls. It lets you set the fisheye border percentage and draw fixed rectangles, ellipses, or free-form polygons for each camera. Select a shape to move it; drag ellipse or rectangle handles to resize it, or drag a polygon vertex to adjust its outline. Polygon drawing uses clicks for corners, then double-click or Enter to finish; Backspace removes the last unfinished corner and Escape cancels. The mouse wheel zooms, and right or middle drag pans the preview. Each shape uses normalized image coordinates from 0 to 1 and applies to **every frame from that camera**.

The preview opens on the first naturally sorted extracted frame. Use the timeline to inspect other frames from that camera; choosing a frame updates the mask preview when mask preview mode is on.

The same settings can be saved in JSON and passed to the headless masker or colorizer with `--mask-config`:

```json
{
  "schema": 1,
  "fisheye_border_percent": 4.0,
  "rectangles": {
    "cam0": [[0.02, 0.78, 0.22, 0.99]],
    "cam1": []
  },
  "ellipses": {
    "cam0": [[0.38, 0.80, 0.62, 0.98]],
    "cam1": []
  },
  "polygons": {
    "cam0": [[[0.4, 0.8], [0.55, 0.78], [0.62, 0.98], [0.46, 0.96]]],
    "cam1": []
  }
}
```

For example:

```powershell
python raven.py mask-persons --dataset D:\ARQUIVOS_TESTE_2\SMALL-DATASET-TEST --mask-config D:\settings\mask-config.json
python raven.py colorize --dataset D:\ARQUIVOS_TESTE_2\SMALL-DATASET-TEST --mask-persons --mask-config D:\settings\mask-config.json
```

Fisheye cutoff removes pixels outside a centered circle. Its percentage is the approximate share of the whole image excluded: `0%` removes nothing, `1%` removes roughly 1% of pixels near the corners, and the range is 0–25%. The dialog's mask preview shows the exact result for the current frame. Keep the cutoff small enough that it does not discard useful scene pixels. Draw rig shapes tightly around the mount in each camera; an oversized shape removes valid scene content from every frame of that camera. These are image exclusions for colorization and 3DGS masks. They do not remove LiDAR points from the cloud.

In the desktop app, enable **Generate masks** in Colorization or Unified Workflow, then open **Mask settings...** to draw rig shapes or set the fisheye cutoff. Icon controls show their names when hovered. **Download model RF-DETR** fetches the verified model and any missing TensorRT runtime into `%APPDATA%\RavenCalibrator`; the portable executable does not contain these large assets. The model is about 139 MB. The TensorRT runtime download is about 1.3 GB and extraction may need up to 3 GB of temporary disk space. The app asks before starting this download. Mask generation can also fetch missing resources when needed, and completed masks can be reused without downloading them again. The dialog saves settings as `mask_settings.json` in the dataset/output folder when processing starts. **Operator removal radius** defaults to `0` (off), and setting it above zero requires person masks. A point near the camera is removed only with repeated person evidence outside the fixed exclusions, a strict majority over clear views, and no local LiDAR plane support. This conservative geometry filter helps preserve doors, walls, and floors, but cannot guarantee every moving-object return is removed. Inspect the result against the raw cloud.

The built-in runtime uses RF-DETR segmentation through TensorRT and requires a compatible NVIDIA GPU and driver. The model is downloaded from the [RF-DETR Seg Medium ONNX model card](https://huggingface.co/davidkodar/rf-detr-seg-medium-onnx) (Apache 2.0); TensorRT 10.13.3.9 runtime DLLs come from [NVIDIA's official package index](https://pypi.nvidia.com/tensorrt-cu13-libs/). Both downloads are pinned by size and SHA-256 before they are installed in the user cache. An ONNX model may need a cold engine build on its first run; later runs can reuse the engine cache. Normal packaging includes the small native worker only. `python tools/package_app.py --with-person-masker` is an optional large offline bundle and expects `build/rfdetr-masker/Release/rfdetr-masker.exe`, the complete TensorRT runtime DLL set, and `models/rfdetr-seg-medium.onnx` to be staged first.
