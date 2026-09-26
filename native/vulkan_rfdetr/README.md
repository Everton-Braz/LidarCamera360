# Headless Vulkan RF-DETR executor

`rfdetr-vulkan.exe` runs the fixed-shape RF-DETR Seg Medium segmentation model
with Vulkan compute shaders. It uses neither CUDA nor TensorRT at runtime.
This implementation currently targets Windows and a Vulkan compute device.
The native CLI decodes images, runs the graph, then uses the same person-mask
decoder as the TensorRT CLI. The original `vulkan_rfdetr_ops_test` softmax
prototype remains as a separate operator test. This backend is experimental
and is not selected by the desktop app or included in its portable package.

The ONNX export is an offline preparation step. It resolves the model's static
shapes, folds constants and shape calculations, and writes a little-endian
`RVK1` schedule containing FP32 weights. It rejects unsupported or dynamic
operations rather than running them on the CPU. Its `*.rvk.json` manifest
records the ONNX SHA-256, tensor and operation counts, and output shapes.
The supplied 432×432 model exports to 1,101 scheduled graph nodes and about 130 MB.
Do not commit the generated model or private test images.

```powershell
python tools/export_rfdetr_vulkan.py `
  --model models/rfdetr-seg-medium.onnx `
  --output build/vulkan-rfdetr/model.rvk
cmake -S native/vulkan_rfdetr -B build/vulkan-rfdetr -G "Visual Studio 18 2026" -A x64
cmake --build build/vulkan-rfdetr --config Release --target rfdetr-vulkan --parallel 4
build/vulkan-rfdetr/Release/rfdetr-vulkan.exe `
  --model build/vulkan-rfdetr/model.rvk `
  --input C:\capture\images `
  --output C:\capture\masks `
  --threshold 0.30 --margin 0.05
```

The CLI walks the input directory recursively and writes one 8-bit PNG for
each image, preserving the relative path and appending `.png`. White means
keep; black means remove a detected person. The output directory must be
outside the input tree. It loads the model and records the Vulkan command
buffer once, then reuses them for every image. Input normalization and final
mask composition run on the CPU; the full ONNX tensor graph runs on Vulkan.
`--device N` selects a Vulkan device by index. `--limit N` processes the first
N sorted images. `--dump-input DIR` saves normalized FP32 tensors for parity
diagnostics.

For raw tensor tests, use `--tensor-input input.f32 --tensor-output DIR` and
optionally `--repeat N`. The input is little-endian FP32 NCHW data matching the
exported graph shape. Outputs are `dets.f32`, `labels.f32`, and `masks.f32`.
The native executable, three compiled shaders in its adjacent `shaders/`
directory, and a generated `.rvk` model are needed at runtime.

The model uses a 241 MB reuse arena on the tested RTX 5070 Ti. The GPU output
is transferred to staging memory, preferably host-cached,
before the CPU decoder reads it. The executor validates model shapes, memory
limits, finite outputs, supported operator attributes, and input lengths.

Validation commands:

```powershell
python -m unittest tests.test_export_rfdetr_vulkan tests.test_compare_mask_sets
python tools/validate_vulkan_executor.py `
  --executable build/vulkan-rfdetr/Release/rfdetr-vulkan.exe `
  --work-dir build/vulkan-rfdetr/operator-validation
ctest --test-dir build/vulkan-rfdetr -C Release --output-on-failure
python tools/compare_mask_sets.py `
  --reference build/vulkan-rfdetr/tensorrt-reference `
  --candidate build/vulkan-rfdetr/vulkan-reference `
  --output build/vulkan-rfdetr/mask-parity.json
```

The nine compact ONNX cases compare native Vulkan output with ONNX Runtime
CPU, including repeated runs with the same GPU buffers. Full-mask comparison
uses representative images and the same score threshold and margin for both
backends. TensorRT may make different decisions for scores close to the
threshold because its current engine uses FP16, while this executor uses FP32.
Keep cold model preparation, model loading, GPU inference, and complete image
batch time separate when measuring speed.

On a local RTX 5070 Ti, 89 images at 3840×3840 took 38.59 seconds with Vulkan
(2.31 images/second, excluding the 1.59-second model load), versus 11.40
seconds with the existing TensorRT FP16 backend (7.81 images/second, excluding
its cold engine build/load). Their removal masks had 0.9885 aggregate IoU.
Two frames scored below 0.78 IoU because a low-confidence second detection
included both the operator's extended arm and some ground. Suppressing that
entire detection removed the arm in other frames, so this backend retains the
shared decoder's current behavior until a more selective edge correction is
validated. Do not assume exact mask parity or use it to remove 3D ground points
without inspecting those frames.
