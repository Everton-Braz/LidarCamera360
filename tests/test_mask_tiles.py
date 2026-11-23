import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import raven_app.person_masks as person_masks
from raven_app import mask_tiles
from raven_app.workflow_progress import estimate_stage_seconds

PERSON = (40, 40, 220)


def fake_detector(input_dir, output_dir):
    """Finds red blocks, but only where they stand upright (a white head above the body) and large enough."""
    for view in Path(input_dir).iterdir():
        image = cv2.imread(str(view)).astype(int)
        red = ((image[..., 2] > 150) & (image[..., 0] < 110)).astype(np.uint8)
        mask = np.full(red.shape, 255, np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(red)
        for index, (x, y, width, height, _) in enumerate(stats[1:], 1):
            if height < 24 or width < 6 or y < 4:
                continue
            head = image[y - 4:y - 1, x + width // 2].min()
            if head > 200:
                mask[labels == index] = 0
        cv2.imwrite(str(Path(output_dir) / (view.name + '.png')), mask)


def frame(people, height=1440, width=1472, turns=0):
    image = np.full((height, width, 3), 90, np.uint8)
    for x, y, w, h in people:
        image[y:y + h, x:x + w] = PERSON
        image[y - 30:y, x:x + w] = 255
    return np.ascontiguousarray(np.rot90(image, turns))


class MaskTileTests(unittest.TestCase):
    def test_tiles_cover_the_frame_at_every_level_and_stay_inside_it(self):
        for shape in ((2880, 2944), (2880, 2880), (1920, 3840), (1080, 1920), (1920, 1080)):
            for tiles in ((2, .5, False), (3, 1 / 3, False)):
                boxes = mask_tiles.tile_boxes(*shape, (tiles,))
                self.assertEqual(boxes[0], (0, 0, *shape))
                seen = np.zeros(shape, bool)
                for y, x, height, width in boxes[1:]:
                    self.assertGreaterEqual(min(y, x), 0)
                    self.assertTrue(y + height <= shape[0] and x + width <= shape[1])
                    seen[y:y + height, x:x + width] = True
                self.assertTrue(seen.all() or len(boxes) == 1)

    def test_views_per_frame_stay_bounded(self):
        self.assertEqual(len(mask_tiles.tile_boxes(2880, 2944, mask_tiles.LEVELS['tensorrt'])), 22)
        self.assertEqual(len(mask_tiles.tile_boxes(2880, 2944, mask_tiles.LEVELS['vulkan'])), 7)
        self.assertEqual(mask_tiles.tile_boxes(400, 400, mask_tiles.LEVELS['tensorrt']), [(0, 0, 400, 400)])

    def test_tiles_of_only_fixed_exclusions_are_skipped(self):
        keep = np.full((2880, 2880), 255, np.uint8)
        keep[1440:] = 0
        boxes = mask_tiles.tile_boxes(2880, 2880, ((2, .5, False),), keep)
        self.assertEqual({box[0] for box in boxes}, {0, 720})

    def test_rotation_needs_clearly_more_people(self):
        self.assertEqual(mask_tiles.pick_rotation([19, 5, 41, 4]), 2)
        self.assertEqual(mask_tiles.pick_rotation([0, 0, 2, 0]), 0)
        self.assertEqual(mask_tiles.pick_rotation([20, 3, 23, 3]), 0)
        self.assertEqual(mask_tiles.pick_rotation([0, 0, 0, 0]), 0)

    def test_small_people_are_found_in_tiles_and_merged_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            for turns in range(4):  # how the camera is mounted
                path = Path(tmp) / f'{turns}.png'
                small = frame([(700, 600, 24, 60)], turns=turns)
                cv2.imwrite(str(path), small)
                detector = mask_tiles.TiledDetector(Path(tmp) / 'work', fake_detector, ((2, .5, False),))
                upright, mask = -turns % 4, {}
                detector.run([(path, upright)], lambda source, keep: mask.update(keep=keep))
                person = (small == PERSON).all(axis=2)
                self.assertEqual(mask['keep'].shape, person.shape)
                self.assertGreater((mask['keep'][person] == 0).mean(), .9)
                self.assertLess((mask['keep'][~person] == 0).mean(), .001)
                whole = mask_tiles.TiledDetector(Path(tmp) / 'whole', fake_detector, ())
                whole.run([(path, upright)], lambda source, keep: mask.update(keep=keep))
                self.assertTrue(mask['keep'].all())  # one squeezed pass misses them

    def test_upside_down_camera_is_detected_per_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            for turns in (0, 2):
                paths = []
                for i in range(4):
                    paths.append(Path(tmp) / f'{turns}_{i}.png')
                    cv2.imwrite(str(paths[-1]), frame([(300 + 60 * i, 700, 60, 160), (900, 700, 60, 160)], turns=turns))
                detector = mask_tiles.TiledDetector(Path(tmp) / f'work{turns}', fake_detector, ())
                self.assertEqual(detector.upright_rotation(paths)[0], turns)

    def test_missing_detector_output_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.png'
            cv2.imwrite(str(path), frame([]))
            detector = mask_tiles.TiledDetector(Path(tmp) / 'work', lambda i, o: None, ())
            with self.assertRaisesRegex(ValueError, 'no valid mask'):
                detector.run([(path, 0)], lambda source, keep: None)


class GenerateTests(unittest.TestCase):
    def generate(self, root, backend='vulkan', runs=None, returncode=0):
        model = root / 'rfdetr.onnx'
        model.write_bytes(b'model')

        def fake_run(command, **options):
            if runs is not None:
                runs.append(command)
            if not returncode:
                fake_detector(command[command.index('--input') + 1], command[command.index('--output') + 1])
            return person_masks.subprocess.CompletedProcess(command, returncode, stdout=b'device lost')

        with patch.object(person_masks, 'resolve_vulkan_executable', return_value=root / 'x.exe'), \
                patch.object(person_masks, 'resolve_vulkan_shaders_dir', return_value=root), \
                patch.object(person_masks, 'resolve_model', return_value=model), \
                patch.object(person_masks, 'ensure_resources', return_value=(model, model)), \
                patch.object(person_masks.subprocess, 'run', side_effect=fake_run):
            return person_masks.generate_person_masks(root, backend=backend)

    def dataset(self, root, turns):
        for camera in ('cam0', 'cam1'):
            (root / 'images' / camera).mkdir(parents=True)
            for i in range(4):
                image = frame([(300 + 60 * i, 700, 60, 160), (900, 700, 60, 160)], turns=turns[camera])
                cv2.imwrite(str(root / 'images' / camera / f'{i}.png'), image)

    def test_masks_people_of_an_upside_down_camera_and_reuses_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, runs = Path(tmp), []
            self.dataset(root, {'cam0': 2, 'cam1': 0})
            masks = self.generate(root, runs=runs)
            for camera in ('cam0', 'cam1'):
                for i in range(4):
                    person = (cv2.imread(str(root / 'images' / camera / f'{i}.png')) == PERSON).all(axis=2)
                    mask = cv2.imread(str(masks / camera / f'{i}.png.png'), 0)
                    self.assertGreater((mask[person] == 0).mean(), .9)
            manifest = json.loads((masks / 'manifest.json').read_text())
            self.assertEqual(manifest['camera_rotation_deg'], {'cam0': 180, 'cam1': 0})
            self.assertEqual(manifest['schema'], 4)
            self.assertEqual(list(root.glob('.mask-tiles-*')), [])
            count = len(runs)
            self.generate(root, runs=runs)
            self.assertEqual(len(runs), count)

            # Masks of an older detector pass are made again.
            del manifest['inference']
            (masks / 'manifest.json').write_text(json.dumps(manifest))
            self.generate(root, runs=runs)
            self.assertGreater(len(runs), count)

    def test_failed_executor_reports_its_output_and_leaves_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root, {'cam0': 0, 'cam1': 0})
            with self.assertRaisesRegex(RuntimeError, 'device lost'):
                self.generate(root, returncode=3)
            self.assertFalse((root / 'masks' / 'manifest.json').exists())
            self.assertEqual(list(root.glob('.mask-tiles-*')), [])

    def test_estimate_counts_masks_of_an_older_detector_as_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.dataset(root, {'cam0': 0, 'cam1': 0})
            masks = self.generate(root)
            self.assertEqual(estimate_stage_seconds(root, None, None, method='direct')[4], 0.0)
            (masks / 'manifest.json').write_text('{"schema": 4}')
            self.assertGreater(estimate_stage_seconds(root, None, None, method='direct')[4], 0.0)


if __name__ == '__main__':
    unittest.main()
