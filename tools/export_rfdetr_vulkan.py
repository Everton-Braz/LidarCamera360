#!/usr/bin/env python3
"""Export a static ONNX graph to the RVK1 Vulkan schedule format.

This tool performs shape inference and constant folding only. It never runs
model inference on image/runtime tensors and never falls back to ONNX Runtime
inference.  The resulting .rvk contains float32-backed constants plus enough
ONNX type metadata for the native executor to implement Cast correctly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference


MAX_RANK = 8
EMPTY_OPTIONAL = 0xFFFFFFFF
_SUPPORTED_DOMAINS = {"", "ai.onnx"}


class ExportError(ValueError):
    """The model cannot be represented by the RVK1 contract."""


def _shape_and_type(value_info: onnx.ValueInfoProto) -> tuple[int, tuple[int, ...]] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        dims.append(int(dim.dim_value))
    return int(tensor_type.elem_type), tuple(dims)


def _value_info_map(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...]]]:
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        info = _shape_and_type(value)
        if info is not None:
            result[value.name] = info
    return result


def _validate_static_metadata(name: str, elem_type: int, shape: tuple[int, ...]) -> None:
    if not name:
        raise ExportError("empty tensor names are not supported")
    if len(shape) > MAX_RANK:
        raise ExportError(f"tensor {name!r} rank {len(shape)} exceeds RVK1 limit {MAX_RANK}")
    if any(d <= 0 for d in shape):
        raise ExportError(f"tensor {name!r} has an empty or dynamic dimension: {shape}")
    if elem_type <= TensorProto.UNDEFINED:
        raise ExportError(f"tensor {name!r} has undefined ONNX dtype")


def _known_shape(info: dict[str, tuple[int, tuple[int, ...]]], name: str) -> tuple[int, ...]:
    try:
        return info[name][1]
    except KeyError as exc:
        raise ExportError(f"shape inference did not produce a static shape for {name!r}") from exc


def _reference_fold_node(
    node: onnx.NodeProto,
    inputs: dict[str, np.ndarray],
    opset_imports: list[onnx.OperatorSetIdProto],
) -> list[np.ndarray]:
    input_names = [name for name in node.input if name]
    graph_inputs = [
        helper.make_tensor_value_info(
            name,
            helper.np_dtype_to_tensor_dtype(np.asarray(inputs[name]).dtype),
            list(np.asarray(inputs[name]).shape),
        )
        for name in input_names
    ]
    # Give outputs a declared type when symbolic inference has supplied one;
    # ReferenceEvaluator can otherwise infer it from the operator.
    graph_outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, []) for name in node.output if name]
    graph = helper.make_graph([node], "rvk1_constant_fold", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=opset_imports, producer_name="export_rfdetr_vulkan")
    evaluator = ReferenceEvaluator(model)
    feed = {name: inputs[name] for name in input_names}
    result = evaluator.run(None, feed)
    if len(result) != len(graph_outputs):
        raise ExportError(f"constant folding {node.op_type} returned an unexpected output count")
    arrays = [np.asarray(value) for value in result]
    if any(array.dtype == object for array in arrays):
        raise ExportError(f"constant folding {node.op_type} returned an unsupported object tensor")
    return arrays


def _fold_constant_node(node: onnx.NodeProto, opset_imports: list[onnx.OperatorSetIdProto]) -> list[np.ndarray]:
    return _reference_fold_node(node, {}, opset_imports)


def _shape_node_value(
    node: onnx.NodeProto,
    info: dict[str, tuple[int, tuple[int, ...]]],
) -> np.ndarray:
    dims = _known_shape(info, node.input[0])
    start = next((int(a.i) for a in node.attribute if a.name == "start"), 0)
    end = next((int(a.i) for a in node.attribute if a.name == "end"), len(dims))
    normalized_start = start if start >= 0 else len(dims) + start
    normalized_end = end if end >= 0 else len(dims) + end
    normalized_start = min(max(normalized_start, 0), len(dims))
    normalized_end = min(max(normalized_end, 0), len(dims))
    return np.asarray(dims[normalized_start:normalized_end], dtype=np.int64)


def _fold_model(
    model: onnx.ModelProto,
) -> tuple[list[onnx.NodeProto], dict[str, np.ndarray], dict[str, tuple[int, tuple[int, ...]]]]:
    info = _value_info_map(model)
    constants: dict[str, np.ndarray] = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
    }
    residual: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if node.domain not in _SUPPORTED_DOMAINS:
            raise ExportError(f"custom ONNX domain {node.domain!r} is not supported ({node.op_type})")
        if any(not output for output in node.output):
            raise ExportError(f"empty output slots are not supported ({node.op_type})")

        folded: list[np.ndarray] | None = None
        if node.op_type == "Constant":
            try:
                folded = _fold_constant_node(node, list(model.opset_import))
            except Exception as exc:
                raise ExportError(f"could not fold Constant node {node.name or node.output[0]!r}: {exc}") from exc
        elif node.op_type == "Shape" and node.input and node.input[0]:
            folded = [_shape_node_value(node, info)]
        elif node.input and all((not name) or name in constants for name in node.input):
            input_constants = {name: constants[name] for name in node.input if name}
            try:
                folded = _reference_fold_node(node, input_constants, list(model.opset_import))
            except Exception:
                # Keep unsupported reference operators in the executable graph.
                folded = None

        if folded is not None:
            if len(folded) != len(node.output):
                raise ExportError(f"folded {node.op_type} output count does not match its ONNX node")
            for name, array in zip(node.output, folded):
                array = np.asarray(array)
                if name in info:
                    inferred_type, inferred_shape = info[name]
                    if tuple(array.shape) != inferred_shape:
                        raise ExportError(
                            f"folded shape mismatch for {name!r}: inferred {inferred_shape}, got {array.shape}"
                        )
                    if inferred_type != helper.np_dtype_to_tensor_dtype(array.dtype):
                        raise ExportError(
                            f"folded dtype mismatch for {name!r}: inferred {inferred_type}, "
                            f"evaluated {array.dtype}"
                        )
                constants[name] = array
                info[name] = (helper.np_dtype_to_tensor_dtype(array.dtype), tuple(int(d) for d in array.shape))
        else:
            residual.append(node)

    return residual, constants, info


def _prune_graph(
    model: onnx.ModelProto,
    residual: list[onnx.NodeProto],
    constants: dict[str, np.ndarray],
) -> tuple[list[onnx.NodeProto], set[str], list[str], list[str]]:
    graph_inputs = [
        value.name for value in model.graph.input
        if value.name not in constants
    ]
    graph_outputs = [value.name for value in model.graph.output]
    needed = set(graph_outputs)
    kept_reversed: list[onnx.NodeProto] = []
    for node in reversed(residual):
        if any(name in needed for name in node.output):
            kept_reversed.append(node)
            needed.update(name for name in node.input if name)
    kept = list(reversed(kept_reversed))

    referenced = set(graph_inputs) | set(graph_outputs)
    for node in kept:
        referenced.update(name for name in node.input if name)
        referenced.update(name for name in node.output if name)
    referenced &= set(constants) | set(graph_inputs) | {n for node in kept for n in node.output}
    return kept, referenced, graph_inputs, graph_outputs


def _normalize_slice_sentinels(
    nodes: list[onnx.NodeProto],
    constants: dict[str, np.ndarray],
    info: dict[str, tuple[int, tuple[int, ...]]],
) -> None:
    """Replace INT64 Slice bounds only when a static axis proves equivalence.

    RVK1 stores integer constants in float32. ONNX uses INT64_MIN/MAX as
    effectively unbounded Slice bounds; those cannot be rounded into float32.
    With static tensor dimensions smaller than 2**24, +/-2**24 has identical
    clamping behavior and remains exact in float32.
    """
    users: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in nodes:
        for index, name in enumerate(node.input):
            if name:
                users.setdefault(name, []).append((node, index))

    limit = 1 << 24
    for name, array in list(constants.items()):
        source = np.asarray(array)
        if source.dtype.kind not in "iu" or not source.size:
            continue
        too_large = (source > limit) | (source < -limit)
        if not np.any(too_large):
            continue
        consumers = users.get(name, [])
        if not consumers or any(node.op_type != "Slice" or index not in (1, 2) for node, index in consumers):
            # Unused constants will be removed by the serializer's reference
            # pruning; any live non-Slice use is not safely normalizable.
            if consumers:
                raise ExportError(f"large integer constant {name!r} is not used only as a Slice bound")
            continue

        replacements = source.copy()
        for node, input_index in consumers:
            if input_index == 1 and np.any(source > limit):
                raise ExportError(f"INT64_MAX cannot be normalized as a Slice start in {node.name or node.op_type}")
            if input_index == 2 and np.any(source < -limit):
                raise ExportError(f"INT64_MIN cannot be normalized as a Slice end in {node.name or node.op_type}")
            data_shape = _known_shape(info, node.input[0])
            starts = np.asarray(constants[node.input[1]]).reshape(-1) if len(node.input) > 1 and node.input[1] in constants else None
            ends = np.asarray(constants[node.input[2]]).reshape(-1) if len(node.input) > 2 and node.input[2] in constants else None
            axes = (
                np.asarray(constants[node.input[3]]).reshape(-1).astype(np.int64)
                if len(node.input) > 3 and node.input[3] and node.input[3] in constants
                else None
            )
            bound = starts if input_index == 1 else ends
            if bound is None or axes is None and len(data_shape) < bound.size:
                raise ExportError(f"cannot prove the axis range for Slice sentinel in {node.name or node.op_type}")
            axes = np.arange(bound.size, dtype=np.int64) if axes is None else axes
            if axes.size != bound.size:
                raise ExportError(f"Slice axes and bounds have different lengths in {node.name or node.op_type}")
            for axis in axes:
                normalized_axis = int(axis) if int(axis) >= 0 else len(data_shape) + int(axis)
                if normalized_axis < 0 or normalized_axis >= len(data_shape):
                    raise ExportError(f"Slice axis is outside inferred rank in {node.name or node.op_type}")
                if data_shape[normalized_axis] >= limit:
                    raise ExportError(
                        f"cannot normalize Slice sentinel: dimension {data_shape[normalized_axis]} "
                        f"is not smaller than 2**24 in {node.name or node.op_type}"
                    )
            replacements[replacements == np.iinfo(source.dtype).max] = limit
            replacements[replacements == np.iinfo(source.dtype).min] = -limit
        constants[name] = replacements


def _remove_empty_concat_inputs(
    nodes: list[onnx.NodeProto],
    constants: dict[str, np.ndarray],
    info: dict[str, tuple[int, tuple[int, ...]]],
) -> None:
    """Drop provably empty static operands from Concat nodes.

    RF-DETR exports static empty slices as no-op concat operands. Removing
    those operands preserves the values while keeping zero dimensions out of
    RVK1's tensor table (which intentionally rejects empty tensors).
    """
    for node in nodes:
        if node.op_type != "Concat":
            continue
        axis_attr = next((int(attribute.i) for attribute in node.attribute if attribute.name == "axis"), None)
        if axis_attr is None:
            raise ExportError(f"Concat node {node.name or node.op_type} has no axis attribute")
        shapes = [_known_shape(info, name) for name in node.input]
        if not shapes:
            raise ExportError(f"Concat node {node.name or node.op_type} has no inputs")
        rank = len(shapes[0])
        axis = axis_attr if axis_attr >= 0 else rank + axis_attr
        if axis < 0 or axis >= rank or any(len(shape) != rank for shape in shapes):
            raise ExportError(f"Concat node {node.name or node.op_type} has incompatible ranks or axis")

        remove: set[int] = set()
        for index, (name, shape) in enumerate(zip(node.input, shapes)):
            if name not in constants or shape[axis] != 0:
                continue
            if any(dim == 0 for dim_index, dim in enumerate(shape) if dim_index != axis):
                continue
            if any(
                other_dim != shape[dim_index]
                for other_index, other_shape in enumerate(shapes)
                if other_index != index
                for dim_index, other_dim in enumerate(other_shape)
                if dim_index != axis
            ):
                continue
            remove.add(index)

        if not remove:
            continue
        remaining = [name for index, name in enumerate(node.input) if index not in remove]
        if not remaining:
            raise ExportError(f"Concat node {node.name or node.op_type} contains only empty operands")
        node.input[:] = remaining
        first_type, first_shape = info[remaining[0]]
        if len(remaining) == 1:
            node.op_type = "Identity"
            node.domain = ""
            node.attribute.clear()
            info[node.output[0]] = (first_type, first_shape)
            continue

        output_shape = list(first_shape)
        total_axis = 0
        for name in remaining:
            elem_type, shape = info[name]
            if elem_type != first_type:
                raise ExportError(f"Concat node {node.name or node.op_type} mixes tensor types")
            total_axis += shape[axis]
        output_shape[axis] = total_axis
        info[node.output[0]] = (first_type, tuple(output_shape))


def _to_f32_values(name: str, array: np.ndarray) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype.kind in "iu":
        # Float32 represents every integer in [-2**24, 2**24] exactly.
        limit = 1 << 24
        if source.size and (np.any(source > limit) or np.any(source < -limit)):
            raise ExportError(f"integer constant {name!r} exceeds exact float32 integer range")
    if source.dtype.kind not in "biuf":
        raise ExportError(f"constant {name!r} uses unsupported dtype {source.dtype}")
    converted = source.astype(np.float32, copy=False).reshape(-1)
    if not np.all(np.isfinite(converted)):
        raise ExportError(f"constant {name!r} contains non-finite or float32-overflow values")
    return converted


def _attribute_values(attribute: onnx.AttributeProto) -> tuple[int, Any]:
    if attribute.type == AttributeProto.INT:
        values = [int(attribute.i)]
        kind = 0
    elif attribute.type == AttributeProto.INTS:
        values = [int(value) for value in attribute.ints]
        kind = 0
    elif attribute.type == AttributeProto.FLOAT:
        values = [float(attribute.f)]
        kind = 1
    elif attribute.type == AttributeProto.FLOATS:
        values = [float(value) for value in attribute.floats]
        kind = 1
    elif attribute.type == AttributeProto.STRING:
        return 2, bytes(attribute.s).decode("utf-8")
    elif attribute.type == AttributeProto.TENSOR:
        raise ExportError(f"tensor attribute {attribute.name!r} remains after constant folding")
    else:
        raise ExportError(
            f"unsupported ONNX attribute type {attribute.type} for {attribute.name!r} "
            "(RVK1 supports integer lists, float lists, and strings)"
        )
    if kind == 0 and any(value < -(1 << 31) or value >= (1 << 31) for value in values):
        raise ExportError(f"integer attribute {attribute.name!r} exceeds int32 range")
    if kind == 1 and any(not math.isfinite(value) or not math.isfinite(np.float32(value)) for value in values):
        raise ExportError(f"float attribute {attribute.name!r} is non-finite or exceeds float32 range")
    return kind, values


class _Writer:
    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def raw(self, value: bytes) -> None:
        self.parts.append(value)

    def u32(self, value: int) -> None:
        if value < 0 or value > 0xFFFFFFFF:
            raise ExportError(f"integer {value} does not fit in RVK1 u32")
        self.parts.append(struct.pack("<I", value))

    def i32(self, value: int) -> None:
        self.parts.append(struct.pack("<i", value))

    def f32(self, value: float) -> None:
        self.parts.append(struct.pack("<f", value))

    def f32_array(self, values: np.ndarray) -> None:
        little_endian = np.asarray(values, dtype="<f4")
        self.parts.append(little_endian.tobytes(order="C"))

    def string(self, value: str) -> None:
        encoded = value.encode("utf-8")
        self.u32(len(encoded))
        self.raw(encoded)

    def finish(self) -> bytes:
        return b"".join(self.parts)


def _serialize(
    nodes: list[onnx.NodeProto],
    constants: dict[str, np.ndarray],
    info: dict[str, tuple[int, tuple[int, ...]]],
    referenced: set[str],
    input_names: list[str],
    output_names: list[str],
) -> tuple[bytes, list[dict[str, Any]], Counter[str]]:
    ordered_names: list[str] = []
    for node in nodes:
        for name in (*node.input, *node.output):
            if name and name in referenced and name not in ordered_names:
                ordered_names.append(name)
    for name in (*input_names, *output_names):
        if name not in ordered_names:
            ordered_names.append(name)
    tensor_ids = {name: index for index, name in enumerate(ordered_names)}

    writer = _Writer()
    writer.raw(b"RVK1")
    writer.u32(len(ordered_names))
    writer.u32(len(nodes))
    writer.u32(len(input_names))
    writer.u32(len(output_names))

    manifest_tensors: dict[str, dict[str, Any]] = {}
    for name in ordered_names:
        if name in constants:
            arr = np.asarray(constants[name])
            elem_type = helper.np_dtype_to_tensor_dtype(arr.dtype)
            shape = tuple(int(dim) for dim in arr.shape)
        else:
            if name not in info:
                raise ExportError(f"no inferred ONNX dtype or shape for tensor {name!r}")
            elem_type, shape = info[name]
            arr = None
        _validate_static_metadata(name, elem_type, shape)
        writer.string(name)
        writer.u32(elem_type)
        writer.u32(len(shape))
        for dim in shape:
            writer.u32(dim)
        if arr is None:
            writer.u32(0)
        else:
            values = _to_f32_values(name, arr)
            if values.size == 0:
                raise ExportError(f"constant tensor {name!r} has no values; RVK1 uses count 0 for runtime tensors")
            writer.u32(values.size)
            writer.f32_array(values)
        manifest_tensors[name] = {
            "name": name,
            "id": tensor_ids[name],
            "dtype": int(elem_type),
            "shape": list(shape),
            "constant": arr is not None,
        }

    for name in input_names:
        writer.u32(tensor_ids[name])
    for name in output_names:
        writer.u32(tensor_ids[name])

    op_counts: Counter[str] = Counter()
    for node in nodes:
        writer.string(node.op_type)
        writer.string(node.name)
        writer.u32(len(node.input))
        for name in node.input:
            writer.u32(EMPTY_OPTIONAL if not name else tensor_ids[name])
        writer.u32(len(node.output))
        for name in node.output:
            writer.u32(tensor_ids[name])
        writer.u32(len(node.attribute))
        for attribute in node.attribute:
            writer.string(attribute.name)
            kind, values = _attribute_values(attribute)
            writer.u32(kind)
            if kind == 2:
                writer.string(values)
            else:
                writer.u32(len(values))
                for value in values:
                    writer.i32(value) if kind == 0 else writer.f32(value)
        op_counts[node.op_type] += 1

    return writer.finish(), [manifest_tensors[name] for name in ordered_names], op_counts


def export_model(model_path: Path, output_path: Path) -> dict[str, Any]:
    model_path = Path(model_path)
    output_path = Path(output_path)
    if not model_path.is_file():
        raise ExportError(f"model does not exist: {model_path}")
    model = onnx.load(str(model_path), load_external_data=True)
    try:
        inferred = SymbolicShapeInference.infer_shapes(
            model,
            auto_merge=True,
            guess_output_rank=True,
            verbose=0,
        )
    except Exception as exc:
        raise ExportError(f"symbolic shape inference failed: {exc}") from exc
    if inferred is not None:
        model = inferred

    for value in (*model.graph.input, *model.graph.output):
        if value.name not in {init.name for init in model.graph.initializer}:
            metadata = _shape_and_type(value)
            if metadata is None:
                raise ExportError(f"graph input/output {value.name!r} has a dynamic or missing shape")
            _validate_static_metadata(value.name, *metadata)

    residual, constants, info = _fold_model(model)
    _remove_empty_concat_inputs(residual, constants, info)
    nodes, referenced, input_names, output_names = _prune_graph(model, residual, constants)
    _normalize_slice_sentinels(nodes, constants, info)
    if not output_names:
        raise ExportError("model has no graph outputs")
    data, manifest_tensors, op_counts = _serialize(
        nodes, constants, info, referenced, input_names, output_names
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    manifest = {
        "format": "RVK1",
        "model": str(model_path),
        "model_sha256": digest,
        "schedule": str(output_path),
        "schedule_bytes": len(data),
        "tensor_count": len(manifest_tensors),
        "node_count": len(nodes),
        "input_names": input_names,
        "output_names": output_names,
        "output_shapes": {name: list(info[name][1]) for name in output_names},
        "ops": dict(sorted(op_counts.items())),
        "tensors": manifest_tensors,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="input ONNX model")
    parser.add_argument("--output", required=True, type=Path, help="output RVK1 schedule")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = export_model(args.model, args.output)
    except Exception as exc:
        print(f"RVK1 export failed: {exc}", file=sys.stderr)
        return 2
    op_summary = ", ".join(f"{op}={count}" for op, count in manifest["ops"].items())
    print(
        f"Exported {manifest['node_count']} nodes, {manifest['tensor_count']} tensors, "
        f"{manifest['schedule_bytes']} bytes to {args.output}"
    )
    print(f"Residual ops: {op_summary or '(none)'}")
    print(f"Outputs: {json.dumps(manifest['output_shapes'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
