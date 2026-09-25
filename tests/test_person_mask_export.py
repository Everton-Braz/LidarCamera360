import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from raven_app.workflow import _copy_person_masks


class PersonMaskExportTests(unittest.TestCase):
    def test_exports_validated_masks_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            images = dataset / "images"
            masks = dataset / "masks"
            image_files = []
            for camera in ("cam0", "cam1"):
                image_dir = images / camera
                image_dir.mkdir(parents=True)
                image_path = image_dir / "frame.jpg"
                cv2.imwrite(str(image_path), np.full((12, 16, 3), 100, np.uint8))
                image_files.append(image_path)
                mask_path = masks / camera / "frame.jpg.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask = np.full((12, 16), 255, np.uint8)
                mask[2:4, 3:5] = 0
                cv2.imwrite(str(mask_path), mask)
            manifest = {"schema": 1, "polarity": "white_keep_black_person", "image_count": 2}
            (masks / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            output = root / "colmap_3dgs"
            stale = output / "masks" / "cam0" / "old.jpg.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old")
            count = _copy_person_masks(dataset, masks, output, image_files)

            self.assertEqual(count, 2)
            self.assertFalse(stale.exists())
            self.assertEqual(
                json.loads((output / "masks" / "manifest.json").read_text(encoding="utf-8")),
                manifest,
            )
            for camera in ("cam0", "cam1"):
                exported = cv2.imdecode(
                    np.fromfile(output / "masks" / camera / "frame.jpg.png", np.uint8),
                    cv2.IMREAD_GRAYSCALE,
                )
                self.assertEqual(exported.shape, (12, 16))
                self.assertEqual(int(exported[2, 3]), 0)
                self.assertEqual(int(exported[0, 0]), 255)

    def test_rejects_wrong_mask_polarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "dataset" / "images" / "cam0"
            image_dir.mkdir(parents=True)
            image_path = image_dir / "frame.jpg"
            cv2.imwrite(str(image_path), np.full((8, 8, 3), 80, np.uint8))
            masks = root / "source_masks"
            masks.mkdir()
            (masks / "manifest.json").write_text(
                json.dumps({"polarity": "white_person_black_keep"}), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _copy_person_masks(root / "dataset", masks, root / "out", [image_path])


if __name__ == "__main__":
    unittest.main()
