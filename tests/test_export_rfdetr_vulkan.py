import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.export_rfdetr_vulkan import ExportError, _serialize, export_model


def _make_model(graph: onnx.GraphProto) -> onnx.ModelProto:
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 18)],
        ir_version=9,
        producer_name="rvk1-export-test",
    )
    onnx.checker.check_model(model)
    return model


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    return data[offset:offset + length].decode("utf-8"), offset + length


class RfdetrVulkanExporterTests(unittest.TestCase):
    def test_exports_static_schedule_folds_shape_and_constants(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        shape_out = helper.make_tensor_value_info("shape_out", TensorProto.INT64, [2])
        one = numpy_helper.from_array(np.asarray(2.0, dtype=np.float32), name="one")
        two = numpy_helper.from_array(np.asarray(5.0, dtype=np.float32), name="two")
        graph = helper.make_graph(
            [
                helper.make_node("Constant", [], ["one"], value=one, name="const_one"),
                helper.make_node("Constant", [], ["two"], value=two, name="const_two"),
                helper.make_node("Add", ["one", "two"], ["sum"], name="folded_add"),
                helper.make_node("Add", ["x", "sum"], ["y"], name="runtime_add"),
                helper.make_node("Shape", ["x"], ["shape_out"], name="static_shape"),
            ],
            "exporter_test",
            [x],
            [y, shape_out],
        )
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "input.onnx"
            output_path = Path(temporary) / "model.rvk"
            onnx.save(_make_model(graph), model_path)
            manifest = export_model(model_path, output_path)

            self.assertEqual(output_path.read_bytes()[:4], b"RVK1")
            self.assertEqual(manifest["node_count"], 1)
            self.assertEqual(manifest["ops"], {"Add": 1})
            self.assertEqual(manifest["input_names"], ["x"])
            self.assertEqual(manifest["output_names"], ["y", "shape_out"])
            self.assertEqual(manifest["output_shapes"], {"y": [1, 2], "shape_out": [2]})
            self.assertEqual(manifest["schedule_bytes"], output_path.stat().st_size)
            self.assertTrue(output_path.with_suffix(".rvk.json").is_file())
            on_disk_manifest = json.loads(output_path.with_suffix(".rvk.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk_manifest["model_sha256"], manifest["model_sha256"])
            names = {tensor["name"]: tensor for tensor in manifest["tensors"]}
            # Manifest tensor records are currently compact and ids are stable.
            self.assertEqual(names["sum"]["constant"], True)

    def test_rejects_dynamic_dimensions(self):
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 3])
        graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "dynamic", [x], [y])
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "dynamic.onnx"
            onnx.save(_make_model(graph), model_path)
            with self.assertRaisesRegex(ExportError, "dynamic or missing shape|dynamic dimension"):
                export_model(model_path, Path(temporary) / "dynamic.rvk")

    def test_rejects_int64_constant_outside_exact_float32_range(self):
        big = numpy_helper.from_array(np.asarray(1 << 24 | 1, dtype=np.int64), name="big")
        output = helper.make_tensor_value_info("big", TensorProto.INT64, [])
        graph = helper.make_graph([], "large_integer", [], [output], initializer=[big])
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "large.onnx"
            onnx.save(_make_model(graph), model_path)
            with self.assertRaisesRegex(ExportError, "exceeds exact float32 integer range"):
                export_model(model_path, Path(temporary) / "large.rvk")

    def test_serializes_empty_optional_input_sentinel_and_attributes(self):
        node = helper.make_node("Gemm", ["x", "", "w"], ["y"], alpha=0.5, transA=1, name="gemm")
        weight = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        info = {
            "x": (TensorProto.FLOAT, (1, 2)),
            "w": (TensorProto.FLOAT, (2, 2)),
            "y": (TensorProto.FLOAT, (1, 2)),
        }
        data, tensors, ops = _serialize(
            [node],
            {"w": weight},
            info,
            {"x", "w", "y"},
            ["x"],
            ["y"],
        )
        self.assertEqual(data[:4], b"RVK1")
        self.assertEqual(ops, {"Gemm": 1})
        self.assertEqual([tensor["id"] for tensor in tensors], [0, 1, 2])
        # Header then three tensor records; parse far enough to reach input ids,
        # output ids, node header, and its three input IDs.
        offset = 20
        for _ in tensors:
            _, offset = _read_string(data, offset)
            _, rank = struct.unpack_from("<II", data, offset)
            offset += 8 + rank * 4
            (count,) = struct.unpack_from("<I", data, offset)
            offset += 4 + count * 4
        offset += 8  # one graph input id and one graph output id
        _, offset = _read_string(data, offset)  # op type
        _, offset = _read_string(data, offset)  # node name
        (input_count,) = struct.unpack_from("<I", data, offset)
        offset += 4
        input_ids = list(struct.unpack_from("<III", data, offset))
        self.assertEqual(input_count, 3)
        self.assertEqual(input_ids[1], 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
