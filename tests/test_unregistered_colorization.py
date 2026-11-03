import json

import numpy as np

from scripts import pipeline_auto_calibrator_and_colorizer as pipeline


def test_unregistered_frames_of_the_model_lenses_are_added_with_their_lens_camera(tmp_path):
    images = tmp_path / 'images'
    names = [f'cam{lens}/frame_{n:06d}.jpg' for n in (1, 2, 3) for lens in (0, 1)] + ['cam2/frame_000001.jpg']
    for name in names:
        (images / name).parent.mkdir(parents=True, exist_ok=True)
        (images / name).write_bytes(b'')
    (images / 'frames.json').write_text(json.dumps({'timestamps': {n: 0.0 for n in names}}))
    model = [{'name': 'cam0/frame_000001.jpg', 'cam_id': 1, 'C': np.ones(3), 'R_cw': np.eye(3)},
             {'name': 'cam1/frame_000002.jpg', 'cam_id': 2, 'C': np.ones(3), 'R_cw': np.eye(3)}]
    out, extra = pipeline._with_unregistered_views(tmp_path, model)
    assert sorted(extra) == sorted(n for n in names if n not in {'cam0/frame_000001.jpg', 'cam1/frame_000002.jpg',
                                                                   'cam2/frame_000001.jpg'})
    assert {im['name']: im['cam_id'] for im in out[2:]} == {n: 1 if n.startswith('cam0') else 2 for n in extra}
    assert len(out) == len(model) + len(extra)


def test_unfitted_views_borrow_the_nearest_fitted_frame_of_their_lens():
    views = {'cam0/frame_000010.jpg': {'g': 10}, 'cam0/frame_000020.jpg': {'g': 20}, 'cam1/frame_000012.jpg': {'g': 12}}
    out = pipeline._borrow_nearest_photometric(views, ['cam0/frame_000014.jpg', 'cam0/frame_000019.jpg',
                                                       'cam1/frame_000040.jpg', 'cam2/frame_000001.jpg'])
    assert out['cam0/frame_000014.jpg'] == {'g': 10}
    assert out['cam0/frame_000019.jpg'] == {'g': 20}
    assert out['cam1/frame_000040.jpg'] == {'g': 12}
    assert 'cam2/frame_000001.jpg' not in out
    assert views.keys() == {'cam0/frame_000010.jpg', 'cam0/frame_000020.jpg', 'cam1/frame_000012.jpg'}
