import json

import cv2
import numpy as np

from raven_app.frame_quality import blurred_frames, sharpness_scores


def test_blur_is_judged_against_the_neighbours_of_the_same_lens(tmp_path):
    rng = np.random.default_rng(0)
    images = tmp_path / 'images'
    names = []
    for lens in ('cam0', 'cam1'):
        (images / lens).mkdir(parents=True)
        for n in range(1, 31):
            texture = rng.integers(0, 255, (256, 256), dtype=np.uint8)
            if lens == 'cam1':
                texture = (texture // 4).astype(np.uint8)        # a dark lens: low sharpness everywhere
            if n == 12:
                texture = cv2.GaussianBlur(texture, (0, 0), 3)     # motion-blurred frame
            name = f'{lens}/frame_{n:06d}.jpg'
            cv2.imwrite(str(images / name), cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR))
            names.append(name)
    (images / 'frames.json').write_text(json.dumps({'timestamps': {n: 0.0 for n in names}}))
    assert blurred_frames(tmp_path) == {'cam0/frame_000012.jpg', 'cam1/frame_000012.jpg'}
    cached = json.loads((images / 'sharpness.json').read_text())
    assert set(cached) == set(names)
    assert sharpness_scores(tmp_path) == {n: cached[n]['sharpness'] for n in names}   # served from the cache
