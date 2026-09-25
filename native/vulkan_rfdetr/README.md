# Vulkan RF-DETR inference prototype

This experimental target starts a separate Vulkan compute path for RF-DETR; it
does not modify the production TensorRT masker or the point-cloud colorizer.
The first GPU primitive is stable row-wise softmax, used by transformer
attention. `vulkan_rfdetr_ops_test --self-test` compares its output against a
double-precision CPU reference on short and multi-workgroup-width rows.

Build it independently with the existing Vulkan SDK:

```powershell
cmake -S native/vulkan_rfdetr -B build/vulkan-rfdetr -G "Visual Studio 18 2026" -A x64
cmake --build build/vulkan-rfdetr --config Release --parallel 4
build/vulkan-rfdetr/Release/vulkan_rfdetr_ops_test.exe --self-test
ctest --test-dir build/vulkan-rfdetr -C Release --output-on-failure
```

This is an operator-level prototype, not yet an RF-DETR inference engine. The
next required pieces include model-weight loading, matrix multiplication,
normalization, the model's attention/decoder and segmentation heads, then mask
parity tests against the existing backend. Keep it out of the portable package
until it can generate source-resolution person masks with validated parity.
