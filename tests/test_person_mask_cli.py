import unittest
from pathlib import Path

from raven_app.cli import parse


class PersonMaskCliTests(unittest.TestCase):
    def test_colorize_person_mask_options(self):
        args = parse([
            "colorize", "--dataset", "dataset",
            "--mask-persons", "--mask-model", "model.engine",
            "--mask-threshold", "0.65", "--mask-margin", "0.04",
            "--mask-config", "mask_settings.json",
            "--operator-radius", "0.35",
        ])
        self.assertTrue(args.mask_persons)
        self.assertEqual(str(args.mask_model), "model.engine")
        self.assertEqual(args.mask_threshold, 0.65)
        self.assertEqual(args.mask_margin, 0.04)
        self.assertEqual(args.operator_radius, 0.35)
        self.assertEqual(args.mask_config, Path("mask_settings.json"))

    def test_existing_masks_directory_and_defaults(self):
        args = parse(["colorize", "--dataset", "dataset", "--masks-dir", "dataset/masks"])
        self.assertFalse(args.mask_persons)
        self.assertEqual(args.masks_dir, Path("dataset/masks"))
        self.assertEqual(args.mask_threshold, 0.5)
        self.assertEqual(args.mask_margin, 0.03)
        self.assertEqual(args.operator_radius, 0.0)

    def test_standalone_mask_command_is_headless_and_defaults(self):
        args = parse(["--headless", "mask-persons", "--dataset", "dataset"])
        self.assertEqual(args.command, "mask-persons")
        self.assertIsNone(args.model)
        self.assertEqual(args.threshold, 0.5)
        self.assertEqual(args.margin, 0.03)
        self.assertIsNone(args.mask_config)

    def test_config_requires_mask_generation(self):
        with self.assertRaises(SystemExit) as ctx:
            parse(["colorize", "--dataset", "dataset", "--mask-config", "mask_settings.json"])
        self.assertEqual(ctx.exception.code, 2)

    def test_mask_generation_and_reuse_are_exclusive(self):
        with self.assertRaises(SystemExit) as ctx:
            parse([
                "colorize", "--dataset", "dataset", "--mask-persons",
                "--masks-dir", "dataset/masks",
            ])
        self.assertEqual(ctx.exception.code, 2)

    def test_radius_without_masks_is_rejected_before_processing(self):
        with self.assertRaises(SystemExit) as ctx:
            parse(["colorize", "--dataset", "dataset", "--operator-radius", "1"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
