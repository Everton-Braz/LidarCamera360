#!/usr/bin/env python3
"""Compare the Vulkan RVK1 executor against ONNX Runtime CPU on compact graphs.

Requires ``onnx``, ``onnxruntime`` and ``numpy``. Each case has one float input
named ``input``; all other operands are model constants. Native results are
expected as float32 files named ``<output-name>.f32``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np


FLOAT_RTOL = 5e-4
FLOAT_ATOL = 5e-5
SEED = 20260925


@dataclass
class Case:
    name: str
    input_data: np.ndarray
    nodes: list[Any]
    initializers: list[Any]
    outputs: list[tuple[str, int, tuple[int, ...]]]


def _tensor(name: str, value: np.ndarray) -> Any:
    import onnx

    return onnx.numpy_helper.from_array(np.asarray(value), name=name)


def _cases() -> list[Case]:
    import onnx
    from onnx import TensorProto, helper

    rng = np.random.default_rng(SEED)
    cases: list[Case] = []

    # Rank-3 matmul, broadcast add, layer normalization, and softmax.
    data = rng.normal(0.0, 0.7, (2, 3, 4)).astype(np.float32)
    weights = rng.normal(0.0, 0.4, (4, 5)).astype(np.float32)
    bias = rng.normal(0.0, 0.2, (5,)).astype(np.float32)
    scale = rng.uniform(0.7, 1.3, (5,)).astype(np.float32)
    ln_bias = rng.normal(0.0, 0.1, (5,)).astype(np.float32)
    cases.append(Case(
        "matmul_add_layernorm_softmax_rank3", data,
        [
            helper.make_node("MatMul", ["input", "weights"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["biased"]),
            helper.make_node("LayerNormalization", ["biased", "scale", "ln_bias"], ["normalized"],
                             axis=-1, epsilon=1e-5),
            helper.make_node("Softmax", ["normalized"], ["probabilities"], axis=-1),
        ],
        [_tensor("weights", weights), _tensor("bias", bias), _tensor("scale", scale),
         _tensor("ln_bias", ln_bias)],
        [("probabilities", TensorProto.FLOAT, (2, 3, 5))],
    ))

    # Gemm with transposed B and a broadcast vector bias.
    data = rng.normal(0.0, 0.6, (3, 4)).astype(np.float32)
    weights = rng.normal(0.0, 0.5, (6, 4)).astype(np.float32)
    bias = rng.normal(0.0, 0.3, (6,)).astype(np.float32)
    cases.append(Case(
        "gemm_transb_bias", data,
        [helper.make_node("Gemm", ["input", "weights", "bias"], ["gemm"],
                          transB=1, alpha=0.75, beta=0.5)],
        [_tensor("weights", weights), _tensor("bias", bias)],
        [("gemm", TensorProto.FLOAT, (3, 6))],
    ))

    # Grouped NCHW convolution with asymmetric padding and stride.
    data = rng.normal(0.0, 0.5, (1, 4, 5, 6)).astype(np.float32)
    weights = rng.normal(0.0, 0.25, (4, 2, 3, 2)).astype(np.float32)
    bias = rng.normal(0.0, 0.1, (4,)).astype(np.float32)
    cases.append(Case(
        "conv_grouped_padding_stride", data,
        [helper.make_node("Conv", ["input", "weights", "bias"], ["convolved"],
                          group=2, strides=[2, 1], pads=[1, 0, 1, 1])],
        [_tensor("weights", weights), _tensor("bias", bias)],
        [("convolved", TensorProto.FLOAT, (1, 4, 3, 6))],
    ))

    # GridSample uses a fixed grid containing in-range and out-of-range points.
    data = rng.uniform(-1.0, 1.0, (1, 2, 3, 4)).astype(np.float32)
    grid = np.asarray([[
        [[-1.2, -0.8], [-0.35, -0.2], [0.65, 0.6]],
        [[-0.9, 1.15], [0.1, 0.0], [1.1, -0.4]],
    ]], dtype=np.float32)
    cases.append(Case(
        "gridsample_bilinear_zeros_align0", data,
        [helper.make_node("GridSample", ["input", "grid"], ["sampled"],
                          mode="bilinear", padding_mode="zeros", align_corners=0)],
        [_tensor("grid", grid)],
        [("sampled", TensorProto.FLOAT, (1, 2, 2, 3))],
    ))

    # Resize linear, half-pixel, using fixed output dimensions.
    data = rng.normal(0.0, 0.5, (1, 2, 3, 4)).astype(np.float32)
    sizes = np.asarray([1, 2, 6, 8], dtype=np.int64)
    cases.append(Case(
        "resize_linear_half_pixel", data,
        [helper.make_node("Resize", ["input", "", "", "sizes"], ["resized"],
                          mode="linear", coordinate_transformation_mode="half_pixel",
                          cubic_coeff_a=-0.75, exclude_outside=0, extrapolation_value=0.0,
                          nearest_mode="round_prefer_floor")],
        [_tensor("sizes", sizes)],
        [("resized", TensorProto.FLOAT, (1, 2, 6, 8))],
    ))

    # A compact layout/indexing graph exercises shape-preserving tensor plumbing.
    data = rng.normal(0.0, 0.8, (2, 3, 4)).astype(np.float32)
    gather_indices = np.asarray([0, 2], dtype=np.int64)
    element_indices = rng.integers(0, 2, (2, 2, 6), dtype=np.int64)
    cases.append(Case(
        "reshape_transpose_slice_concat_tile_gather", data,
        [
            helper.make_node("Transpose", ["input"], ["transposed"], perm=[0, 2, 1]),
            helper.make_node("Slice", ["transposed", "starts", "ends", "axes", "steps"], ["sliced"]),
            helper.make_node("Concat", ["sliced", "sliced"], ["joined"], axis=1),
            helper.make_node("Tile", ["joined", "repeats"], ["tiled"]),
            helper.make_node("Reshape", ["tiled", "target_shape"], ["reshaped"]),
            helper.make_node("Gather", ["reshaped", "gather_indices"], ["gathered"], axis=1),
            helper.make_node("GatherElements", ["gathered", "element_indices"], ["selected"], axis=1),
        ],
        [
            _tensor("starts", np.asarray([1], dtype=np.int64)),
            _tensor("ends", np.asarray([4], dtype=np.int64)),
            _tensor("axes", np.asarray([1], dtype=np.int64)),
            _tensor("steps", np.asarray([2], dtype=np.int64)),
            _tensor("repeats", np.asarray([1, 2, 1], dtype=np.int64)),
            _tensor("target_shape", np.asarray([2, 4, 6], dtype=np.int64)),
            _tensor("gather_indices", gather_indices),
            _tensor("element_indices", element_indices),
        ],
        [("selected", TensorProto.FLOAT, (2, 2, 6))],
    ))

    # TopK values and indices; values are unique to avoid tie-order ambiguity.
    data = rng.normal(0.0, 1.0, (3, 7)).astype(np.float32)
    cases.append(Case(
        "topk_values_and_indices", data,
        [helper.make_node("TopK", ["input", "k"], ["top_values", "top_indices"],
                          axis=1, largest=1, sorted=1)],
        [_tensor("k", np.asarray([3], dtype=np.int64))],
        [("top_values", TensorProto.FLOAT, (3, 3)),
         ("top_indices", TensorProto.INT64, (3, 3))],
    ))

    # Supported reductions plus common activation/transcendental paths.
    data = rng.normal(0.0, 0.55, (2, 3, 4)).astype(np.float32)
    cases.append(Case(
        "reductions_activation_math", data,
        [
            helper.make_node("ReduceSum", ["input", "axis2"], ["sum"], keepdims=1),
            helper.make_node("Sigmoid", ["sum"], ["activated"]),
            helper.make_node("ReduceMax", ["input"], ["reduced"], axes=[1, 2], keepdims=0),
            helper.make_node("Relu", ["input"], ["positive"]),
            helper.make_node("Exp", ["positive"], ["exponential"]),
            helper.make_node("Sqrt", ["positive"], ["square_root"]),
        ],
        [_tensor("axis2", np.asarray([2], dtype=np.int64))],
        [("activated", TensorProto.FLOAT, (2, 3, 1)),
         ("reduced", TensorProto.FLOAT, (2,)),
         ("exponential", TensorProto.FLOAT, (2, 3, 4)),
         ("square_root", TensorProto.FLOAT, (2, 3, 4))],
    ))

    # The RF-DETR-style contraction equation requested for the mask head.
    data = rng.normal(0.0, 0.4, (1, 3, 2, 2)).astype(np.float32)
    embeddings = rng.normal(0.0, 0.5, (1, 2, 3)).astype(np.float32)
    cases.append(Case(
        "einsum_bchw_bnc_bnhw", data,
        [helper.make_node("Einsum", ["input", "embeddings"], ["masks"],
                          equation="bchw,bnc->bnhw")],
        [_tensor("embeddings", embeddings)],
        [("masks", TensorProto.FLOAT, (1, 2, 2, 2))],
    ))

    return cases


def _build_model(case: Case) -> Any:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        case.nodes,
        case.name,
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(case.input_data.shape))],
        [helper.make_tensor_value_info(name, dtype, list(shape)) for name, dtype, shape in case.outputs],
        initializer=case.initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="validate_vulkan_executor",
    )
    # Keep compatible with runtimes which support ONNX opset 17 but not newer IR.
    model.ir_version = min(int(model.ir_version), 10)
    onnx.checker.check_model(model)
    return model


def _run(command: list[str], cwd: Path, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)


def _safe_output_file(output_dir: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
        raise ValueError(f"unsafe output name from model: {name!r}")
    return output_dir / f"{name}.f32"


def _compare_output(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    expected_f32 = np.asarray(expected).astype(np.float32, copy=False)
    actual = np.asarray(actual, dtype=np.float32)
    if actual.size != expected_f32.size:
        return {"passed": False, "reason": f"size mismatch: native {actual.size}, ORT {expected_f32.size}"}
    if not np.all(np.isfinite(actual)):
        return {"passed": False, "reason": "native output contains NaN or infinity"}

    if np.asarray(expected).dtype.kind in "iu":
        matches = np.array_equal(actual, expected_f32)
        return {
            "passed": bool(matches),
            "comparison": "exact_integer_values",
            "max_abs_error": float(np.max(np.abs(actual - expected_f32))) if actual.size else 0.0,
        }

    if not np.all(np.isfinite(expected_f32)):
        return {"passed": False, "reason": "ONNX Runtime output contains NaN or infinity"}
    difference = np.abs(actual - expected_f32)
    passed = bool(np.allclose(actual, expected_f32, rtol=FLOAT_RTOL, atol=FLOAT_ATOL))
    return {
        "passed": passed,
        "comparison": "allclose",
        "rtol": FLOAT_RTOL,
        "atol": FLOAT_ATOL,
        "max_abs_error": float(np.max(difference)) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))) if difference.size else 0.0,
    }


def _validate_case(case: Case, run_dir: Path, executable: Path, exporter: Path) -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    case_dir = run_dir / case.name
    case_dir.mkdir(parents=True, exist_ok=False)
    model_path = case_dir / "model.onnx"
    schedule_path = case_dir / "model.rvk"
    input_path = case_dir / "input.f32"
    output_dir = case_dir / "native-output"
    output_dir.mkdir()

    model = _build_model(case)
    onnx.save_model(model, str(model_path))
    input_data = np.ascontiguousarray(case.input_data, dtype=np.float32)
    input_data.tofile(input_path)

    expected_session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
        sess_options=_ort_options(ort),
    )
    output_names = [name for name, _, _ in case.outputs]
    expected_outputs = expected_session.run(output_names, {"input": input_data})

    export_result = _run(
        [sys.executable, str(exporter), "--model", str(model_path), "--output", str(schedule_path)],
        cwd=exporter.parent.parent,
    )
    if export_result.returncode != 0:
        raise RuntimeError("RVK1 export failed: " + (export_result.stderr or export_result.stdout).strip())

    native_result = _run(
        [str(executable), "--model", str(schedule_path), "--tensor-input", str(input_path),
         "--tensor-output", str(output_dir), "--repeat", "3"],
        cwd=case_dir,
    )
    if native_result.returncode != 0:
        raise RuntimeError("native executor failed: " + (native_result.stderr or native_result.stdout).strip())

    output_results: dict[str, Any] = {}
    passed = True
    for (name, _, shape), expected in zip(case.outputs, expected_outputs):
        result_path = _safe_output_file(output_dir, name)
        if not result_path.is_file():
            output_results[name] = {"passed": False, "reason": f"missing native output: {result_path.name}"}
            passed = False
            continue
        actual = np.fromfile(result_path, dtype=np.float32)
        if actual.size != int(np.prod(shape, dtype=np.int64)):
            output_results[name] = {
                "passed": False,
                "reason": f"wrong number of values: got {actual.size}, expected {int(np.prod(shape))}",
            }
            passed = False
            continue
        actual = actual.reshape(shape)
        comparison = _compare_output(actual, expected)
        output_results[name] = comparison
        passed &= bool(comparison.get("passed"))

    return {
        "name": case.name,
        "passed": passed,
        "input_shape": list(input_data.shape),
        "outputs": output_results,
        "native_stdout": native_result.stdout.strip(),
    }


def _ort_options(ort: Any) -> Any:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return options


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path, help="built native Vulkan executor CLI")
    parser.add_argument("--work-dir", required=True, type=Path, help="directory for generated models and reports")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    executable = args.executable.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    exporter = Path(__file__).resolve().with_name("export_rfdetr_vulkan.py")
    if not executable.is_file():
        print(f"Vulkan executor does not exist: {executable}", file=sys.stderr)
        return 2
    if not exporter.is_file():
        print(f"RVK1 exporter does not exist: {exporter}", file=sys.stderr)
        return 2

    try:
        import onnxruntime as ort

        work_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = work_dir / f"vulkan-validation-{stamp}-{os.getpid()}"
        run_dir.mkdir()
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        for case in _cases():
            case_started = time.perf_counter()
            try:
                result = _validate_case(case, run_dir, executable, exporter)
            except Exception as exc:
                result = {"name": case.name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
            result["seconds"] = round(time.perf_counter() - case_started, 4)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"[{status}] {case.name} ({result['seconds']:.2f}s)")

        passed_count = sum(bool(result["passed"]) for result in results)
        report = {
            "format": "vulkan-rfdetr-executor-validation-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "executable": str(executable),
            "onnxruntime_version": ort.__version__,
            "seed": SEED,
            "float_tolerance": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL},
            "cases_passed": passed_count,
            "cases_total": len(results),
            "seconds": round(time.perf_counter() - started, 4),
            "results": results,
        }
        report_path = run_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Report: {report_path}")
        print(f"Result: {passed_count}/{len(results)} cases passed")
        return 0 if passed_count == len(results) else 1
    except Exception as exc:
        print(f"Vulkan executor validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
