# Native RF-DETR worker provenance

TensorRt.cpp/.h and the algorithms in RfDetrDecode.cpp/.h are adapted from
Spirula Studio (`src/nn/TensorRt.*`, `src/sam/NativeMaskDecode.*`,
`src/sam/MaskDilate.*`) in the local reference checkout. Its base commit was
17b5a0c51a897c8e8e040cbe4e12e6319945d786; the local integration files include
changes beyond that base. The complete adapted sources and build scripts
are shipped alongside this notice. The upstream GPL version 3 text is in
LICENSE. The independent WIC image codec and batch host are provided under
GPL version 3 as part of this native worker.

Reference: https://github.com/harry7557558/spirula-studio

RF-DETR Seg Medium ONNX is the 139,278,517-byte export from
https://huggingface.co/davidkodar/rf-detr-seg-medium-onnx . The model card
declares Apache 2.0. The model is packaged separately in models/.

NVIDIA TensorRT and CUDA are NVIDIA components and retain their own license
terms. The native worker loads TensorRT 10 libraries and uses the CUDA runtime.
The NVIDIA display driver is installed separately on the destination machine.
