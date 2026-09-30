import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from raven_app.cli import parse, run


class PhotometricCliTests(unittest.TestCase):
    def test_photometric_ui_options_translate_and_enable_params(self):
        from PyQt6.QtWidgets import QApplication
        from raven_app.gui import create_main_window
        from raven_app.i18n import set_language

        app = QApplication.instance() or QApplication([])
        set_language("en")
        window = create_main_window()
        try:
            set_language("pt-BR")
            app.processEvents()
            for view in (window.colorize_view, window.workflow_view):
                self.assertEqual(view.photometric_label.text() if hasattr(view, "photometric_label")
                                 else view.photometric_caption.text(), "Compensação fotométrica:")
                self.assertEqual(view.photometric_combo.itemText(0), "Desativada")
                self.assertEqual(view.photometric_combo.itemText(1), "Log-linear")
                self.assertEqual(view.photometric_combo.itemText(2), "PPISP + grade bilateral")
                self.assertFalse(view.photometric_params_input.isEnabled())
                view.photometric_combo.setCurrentIndex(1)
                self.assertTrue(view.photometric_params_input.isEnabled())
                self.assertEqual(view.photometric_combo.currentData(), "loglinear")
                view.photometric_combo.setCurrentIndex(2)
                self.assertTrue(view.photometric_params_input.isEnabled())
                self.assertEqual(view.photometric_combo.currentData(), "ppisp-bilateral")

            set_language("en")
            app.processEvents()
            self.assertEqual(window.colorize_view.photometric_label.text(), "Photometric compensation:")
            self.assertEqual(window.workflow_view.photometric_caption.text(), "Photometric compensation:")
        finally:
            window.close()
            set_language("en")

    def test_photometric_options_default_and_forwardable_on_colorize_and_workflow(self):
        color = parse(["colorize", "--dataset", "dataset"])
        self.assertEqual(color.photometric, "off")
        self.assertIsNone(color.photometric_params)

        for command, base in (
            ("colorize", ["--dataset", "dataset"]),
            ("workflow", ["--bag", "bag", "--insv", "video.insv", "--output", "out"]),
        ):
            args = parse([
                command, *base, "--photometric", "loglinear",
                "--photometric-params", "fit.json",
            ])
            self.assertEqual(args.photometric, "loglinear")
            self.assertEqual(args.photometric_params, Path("fit.json"))
            advanced = parse([
                command, *base, "--photometric", "ppisp-bilateral",
                "--photometric-params", "fit.json",
            ])
            self.assertEqual(advanced.photometric, "ppisp-bilateral")
            self.assertEqual(advanced.photometric_params, Path("fit.json"))

    def test_photometric_cli_validation(self):
        with self.assertRaises(SystemExit) as invalid_mode:
            parse(["colorize", "--dataset", "dataset", "--photometric", "bilateral"])
        self.assertEqual(invalid_mode.exception.code, 2)

        with self.assertRaises(SystemExit) as params_without_mode:
            parse(["colorize", "--dataset", "dataset", "--photometric-params", "fit.json"])
        self.assertEqual(params_without_mode.exception.code, 2)

    def test_photometric_fit_defaults_and_forwards_arguments_lazily(self):
        args = parse(["--headless", "photometric-fit", "--dataset", "dataset"])
        self.assertEqual(args.sample_points, 50000)
        self.assertEqual(args.max_observations, 500000)
        self.assertEqual(args.mode, "loglinear")
        self.assertIsNone(args.output)
        self.assertIsNone(args.masks_dir)

        fit = unittest.mock.Mock(return_value={"status": "fitted"})
        fake_module = types.ModuleType("raven_app.photometric")
        fake_module.fit_dataset = fit
        output = io.StringIO()
        with patch.dict(sys.modules, {"raven_app.photometric": fake_module}), contextlib.redirect_stdout(output):
            self.assertEqual(run(args), 0)
        fit.assert_called_once_with(
            Path("dataset"), output=None, masks_dir=None,
            sample_points=50000, max_observations=500000, mode="loglinear",
        )
        self.assertEqual(json.loads(output.getvalue()), {"status": "fitted"})

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            explicit = parse([
                "photometric-fit", "--dataset", str(base),
                "--output", str(base / "fit.json"),
                "--masks-dir", str(base / "masks"),
                "--sample-points", "1200", "--max-observations", "9000",
                "--mode", "ppisp-bilateral",
            ])
            with patch.dict(sys.modules, {"raven_app.photometric": fake_module}):
                self.assertEqual(run(explicit), 0)
        fit.assert_called_with(
            base, output=base / "fit.json", masks_dir=base / "masks",
            sample_points=1200, max_observations=9000, mode="ppisp-bilateral",
        )

    def test_colorize_passes_photometric_settings_to_both_methods(self):
        import scripts.pipeline_auto_calibrator_and_colorizer as pipeline

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset = Path(temp_dir)
            (dataset / "slam_out" / "pcd").mkdir(parents=True)
            (dataset / "slam_out" / "pcd" / "all_raw_points.pcd").touch()
            trajectory = dataset / "trajectory.txt"
            trajectory.touch()
            for camera in ("cam0", "cam1"):
                image_dir = dataset / "images" / camera
                image_dir.mkdir(parents=True)
                (image_dir / "frame.jpg").touch()

            args = parse([
                "colorize", "--dataset", str(dataset), "--method", "all",
                "--photometric", "ppisp-bilateral", "--photometric-params", "fit.json",
            ])
            with patch.object(pipeline, "get_slam_trajectory_path", return_value=trajectory), \
                 patch.object(pipeline, "colorize_via_spirula_sfm") as sfm, \
                 patch.object(pipeline, "colorize_via_direct_rigid") as direct:
                self.assertEqual(run(args), 0)

            for method in (sfm, direct):
                self.assertEqual(method.call_args.kwargs["photometric"], "ppisp-bilateral")
                self.assertEqual(method.call_args.kwargs["photometric_params"], Path("fit.json"))

    def test_workflow_passes_photometric_settings(self):
        args = parse([
            "workflow", "--bag", "bag", "--insv", "video.insv", "--output", "out",
            "--scanner", "raven", "--photometric", "ppisp-bilateral",
            "--photometric-params", "fit.json",
        ])
        with patch("raven_app.workflow.execute_unified_workflow", return_value=0) as execute:
            self.assertEqual(run(args), 0)
        self.assertEqual(execute.call_args.kwargs["photometric"], "ppisp-bilateral")
        self.assertEqual(execute.call_args.kwargs["photometric_params"], Path("fit.json"))


if __name__ == "__main__":
    unittest.main()
