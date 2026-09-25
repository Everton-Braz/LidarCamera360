# Headless RF-DETR person masker

This standalone Windows CLI runs RF-DETR Seg Medium through TensorRT, without
the Spirula GUI, Vulkan runtime, Python, or PyTorch. It loads one engine and
one execution context for the entire directory, then performs one model call
per image.

```powershell
.\rfdetr-masker.exe `
  --input  C:\capture\images `
  --output C:\capture\person-masks `
  --model  C:\app\models\rfdetr-seg-medium.onnx `
  --cache-dir $env:APPDATA\RavenCalibrator\rfdetr-cache `
  --threshold 0.30 `
  --margin 0.05
```

The command walks input folders recursively. A source `cam0\frame.jpg` is
written as `cam0\frame.jpg.png`, preserving the source-relative camera path.
Output is an 8-bit grayscale PNG: white (`255`) means keep the pixel and black
(`0`) means remove a detected person. `--margin` is a signed ratio of the
detection box's mean side; positive values grow person masks, negative values
shrink them. `--threshold` is the person-class confidence threshold.

`--model` accepts `.onnx` or a serialized `.engine`. ONNX builds a TensorRT
engine on first use and caches it using the ONNX fingerprint, GPU UUID,
TensorRT version, and CUDA driver version. `--cache-dir` keeps that cache
outside the model directory. A compatible prebuilt engine avoids ONNX parsing
and engine building. Engine files are specific to their TensorRT/GPU setup.

Build with `build.ps1 -TensorRTRoot <TensorRT SDK root>`. The normal build
places `rfdetr-masker.exe` and the TensorRT runtime/parser DLLs together under
`build/rfdetr-masker/Release`. The optional
`-IncludeBuilderResource` copies TensorRT's builder resource DLL (over 1 GB in
the local SDK) when the output must build engines from uncached ONNX files.
The Windows NVIDIA driver and a CUDA-compatible GPU are required at runtime. In
the desktop app's normal portable build, the model and four TensorRT DLLs are
downloaded on demand into `%APPDATA%\RavenCalibrator`; the portable executable
does not bundle them. The optional `tools/package_app.py --with-person-masker`
build includes those large resources for offline use.

The TensorRT wrapper, RF-DETR decoder, and mask dilation are adapted from
Spirula Studio under GPL version 3. The complete upstream license is
included in this directory. The RF-DETR ONNX model is downloaded from its
model card and distributed under the terms linked from it. TensorRT DLLs are
fetched from NVIDIA's official `pypi.nvidia.com` package index and retain
NVIDIA's license terms.
