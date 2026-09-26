import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tools.compare_mask_sets import compare_mask_sets, main


class CompareMaskSetsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        self.candidate = self.root / "candidate"

    def tearDown(self):
        self.temp.cleanup()

    def write_mask(self, root, relative, array):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        self.assertTrue(cv2.imwrite(str(path), array))
        return path

    def test_reports_per_frame_and_pixel_weighted_overall_scores(self):
        # Frame one agrees exactly. In frame two, reference removes a 2x2
        # square while candidate removes three pixels, three in common.
        exact = np.full((3, 4), 255, np.uint8)
        changed_ref = np.full((3, 4), 255, np.uint8)
        changed_candidate = np.full((3, 4), 255, np.uint8)
        changed_ref[0:2, 0:2] = 0
        changed_candidate[0, 0:2] = 0
        changed_candidate[1, 0] = 0
        self.write_mask(self.reference, "cam0/000.png", exact)
        self.write_mask(self.candidate, "cam0/000.png", exact)
        self.write_mask(self.reference, "cam1/001.png", changed_ref)
        self.write_mask(self.candidate, "cam1/001.png", changed_candidate)

        report = compare_mask_sets(self.reference, self.candidate)

        self.assertEqual(report["frame_count"], 2)
        self.assertEqual(report["polarity"], "white_keep_black_remove")
        self.assertEqual(report["frames"][0]["path"], "cam0/000.png")
        self.assertEqual(report["frames"][0]["removal_iou"], 1.0)
        self.assertEqual(report["frames"][1]["removal_iou"], 3 / 4)
        self.assertAlmostEqual(report["frames"][1]["pixel_agreement"], 11 / 12)
        self.assertEqual(report["overall"]["pixels"], 24)
        self.assertEqual(report["overall"]["union_pixels"], 4)
        self.assertEqual(report["overall"]["intersection_pixels"], 3)
        self.assertEqual(report["overall"]["removal_iou"], 0.75)
        self.assertAlmostEqual(report["overall"]["pixel_agreement"], 23 / 24)

    def test_empty_removal_regions_have_iou_one(self):
        keep = np.full((2, 5), 255, np.uint8)
        self.write_mask(self.reference, "frame.png", keep)
        self.write_mask(self.candidate, "frame.png", keep)

        report = compare_mask_sets(self.reference, self.candidate)

        self.assertEqual(report["overall"]["removal_iou"], 1.0)
        self.assertEqual(report["overall"]["pixel_agreement"], 1.0)

    def test_rejects_missing_relative_paths(self):
        keep = np.full((2, 2), 255, np.uint8)
        self.write_mask(self.reference, "cam0/a.png", keep)
        self.write_mask(self.candidate, "cam0/b.png", keep)

        with self.assertRaisesRegex(ValueError, "Mask file sets differ"):
            compare_mask_sets(self.reference, self.candidate)

    def test_rejects_shape_mismatch(self):
        self.write_mask(self.reference, "frame.png", np.full((2, 3), 255, np.uint8))
        self.write_mask(self.candidate, "frame.png", np.full((3, 2), 255, np.uint8))

        with self.assertRaisesRegex(ValueError, "Mask shape mismatch for frame.png"):
            compare_mask_sets(self.reference, self.candidate)

    def test_rejects_nonbinary_masks(self):
        invalid = np.full((2, 3), 255, np.uint8)
        invalid[0, 0] = 127
        self.write_mask(self.reference, "frame.png", invalid)
        self.write_mask(self.candidate, "frame.png", np.full((2, 3), 255, np.uint8))

        with self.assertRaisesRegex(ValueError, r"only 0 \(remove\) and 255 \(keep\)"):
            compare_mask_sets(self.reference, self.candidate)

    def test_cli_writes_json_report(self):
        keep = np.full((2, 2), 255, np.uint8)
        self.write_mask(self.reference, "frame.png", keep)
        self.write_mask(self.candidate, "frame.png", keep)
        output = self.root / "reports" / "comparison.json"

        self.assertEqual(
            main(
                [
                    "--reference",
                    str(self.reference),
                    "--candidate",
                    str(self.candidate),
                    "--output",
                    str(output),
                ]
            ),
            0,
        )
        self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
