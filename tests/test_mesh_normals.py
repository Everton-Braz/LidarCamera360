"""Meshes smoothed by the normal maps of the photos (raven_app.mesh_normals)."""
import cv2
import numpy as np
import pytest

from raven_app import mesh_normals as mn
from raven_app import openmvs_dense as od

SIZE = 200
CAMERA = ('OPENCV_FISHEYE', SIZE, SIZE, [60.0, 60.0, SIZE / 2, SIZE / 2, 0.0, 0.0, 0.0, 0.0])
STEP = 0.03


def grid(height):
    """Triangles of the surface z = height(x, y), facing the cameras at z = 0."""
    axis = np.arange(-1.2, 1.2 + STEP / 2, STEP)
    x, y = np.meshgrid(axis, axis)
    n = len(axis)
    i = np.arange(n * n).reshape(n, n)
    a, b, c, d = (k.ravel() for k in (i[:-1, :-1], i[:-1, 1:], i[1:, :-1], i[1:, 1:]))
    faces = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    return np.column_stack([x.ravel(), y.ravel(), height(x, y).ravel()]), faces


def photos():
    turn = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
    return [(f'cam0/frame_{i:06d}.jpg', 1, turn if i else np.eye(3), np.array([0.3 * i - 0.5, 0.1 * i, 0.0]))
            for i in range(4)]


def write_maps(folder, images, normal_at):
    """A normal map of each photo of the plane z = 2, as Spirula Studio writes them."""
    v, u = np.mgrid[:SIZE, :SIZE]
    a = np.stack([(u + 0.5 - SIZE / 2) / 60.0, (v + 0.5 - SIZE / 2) / 60.0], axis=-1)
    theta = np.linalg.norm(a, axis=-1)
    rays = np.dstack([a * (np.sin(theta) / np.maximum(theta, 1e-9))[..., None], np.cos(theta)])
    for name, _, R, C in images:
        d = rays.reshape(-1, 3) @ R
        far = np.where(d[:, 2] > 0.2, 2 / np.maximum(d[:, 2], 0.2), 0)
        world = normal_at(d * far[:, None] + C)
        n = (world @ R.T).reshape(SIZE, SIZE, 3)
        n[(theta > 1.6) | (far.reshape(SIZE, SIZE) == 0)] = 1.0     # what "no normal" looks like
        path = mn.normal_map_path(folder, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.rint((n[..., ::-1] + 1) * 127.5).astype(np.uint8))


def masks_of(images, value=True):
    return [np.full((od.VIS_SIZE, od.VIS_SIZE), value) for _ in images]


def smooth(vertices, faces, images, folder, masks=None):
    return mn.smooth_by_normals(vertices, faces, {1: CAMERA}, images, masks or masks_of(images), folder,
                                log=lambda *_: None)


def inside(vertices):
    return (np.abs(vertices[:, :2]) < 0.9).all(axis=1)


def flat(points):
    return np.tile([0.0, 0.0, -1.0], (len(points), 1))


def test_bumps_of_a_wall_go_and_the_wall_stays_in_place(tmp_path):
    rng = np.random.default_rng(0)
    vertices, faces = grid(lambda x, y: 2 + rng.normal(0, 0.003, x.shape))
    images = photos()
    write_maps(tmp_path, images, flat)
    moved, report = smooth(vertices, faces, images, tmp_path)
    ok = inside(vertices)
    assert moved[ok, 2].std() < 0.4 * vertices[ok, 2].std()
    assert abs(moved[ok, 2].mean() - 2) < 0.0005
    assert report['guided_share'] > 0.8 and report['angle_to_maps_deg'][1] < report['angle_to_maps_deg'][0]


def test_a_step_keeps_its_edge(tmp_path):
    """A 20 degree fold stays sharp: each side has its own normals."""
    slope = np.tan(np.radians(20))
    rng = np.random.default_rng(1)
    vertices, faces = grid(lambda x, y: 2 + slope * np.maximum(x, 0) + rng.normal(0, 0.002, x.shape))
    images = photos()
    tilted = np.array([slope, 0, -1]) / np.hypot(slope, 1)
    # the maps are drawn on the plane z = 2: the fold is at x = 0 of it, as on the surface
    write_maps(tmp_path, images, lambda p: np.where((p[:, :1] > 0), tilted, [0.0, 0.0, -1.0]))
    moved, _ = smooth(vertices, faces, images, tmp_path)
    normals, _ = mn.triangle_normals(moved, faces)
    x = moved[faces][..., 0].mean(axis=1)
    near = inside(moved[faces].mean(axis=1)) & (np.abs(x) > STEP) & (np.abs(x) < 0.5)
    truth = np.where((x > 0)[:, None], tilted, [0.0, 0.0, -1.0])
    before, _ = mn.triangle_normals(vertices, faces)
    error = [np.degrees(np.arccos(np.clip((n * truth).sum(axis=1)[near], -1, 1))) for n in (before, normals)]
    assert np.median(error[1]) < 0.5 * np.median(error[0])


def test_no_vertex_moves_further_than_a_centimetre(tmp_path):
    vertices, faces = grid(lambda x, y: np.full(x.shape, 2.0))
    images = photos()
    tilted = np.array([np.sin(0.4), 0, -np.cos(0.4)])        # 23 degrees off: a map that is wrong everywhere
    write_maps(tmp_path, images, lambda p: np.tile(tilted, (len(p), 1)))
    moved, report = smooth(vertices, faces, images, tmp_path)
    assert report['guided_share'] > 0.5
    assert np.linalg.norm(moved - vertices, axis=1).max() <= mn.MAX_MOVE + 1e-9


def test_maps_that_disagree_or_contradict_the_mesh_are_not_used(tmp_path):
    vertices, faces = grid(lambda x, y: np.full(x.shape, 2.0))
    images = photos()
    write_maps(tmp_path / 'side', images, lambda p: np.tile([0.8, 0.0, -0.6], (len(p), 1)))
    moved, report = smooth(vertices, faces, images, tmp_path / 'side')
    assert report['guided_share'] == 0 and np.array_equal(moved, vertices)
    for k, image in enumerate(images):                       # 35 degrees apart from each other
        s = np.sin(0.31) * (-1) ** k
        write_maps(tmp_path / 'torn', [image], lambda p: np.tile([s, 0.0, -np.cos(0.31)], (len(p), 1)))
    assert smooth(vertices, faces, images, tmp_path / 'torn')[1]['guided_share'] < 0.05


def test_masked_photos_do_not_count(tmp_path):
    vertices, faces = grid(lambda x, y: np.full(x.shape, 2.0))
    images = photos()
    write_maps(tmp_path, images, flat)
    assert smooth(vertices, faces, images, tmp_path, masks_of(images, False))[1]['guided_share'] == 0


def test_a_hidden_surface_takes_no_normal_from_the_one_in_front(tmp_path):
    front, faces = grid(lambda x, y: np.full(x.shape, 2.0))
    back, _ = grid(lambda x, y: np.full(x.shape, 3.0))
    back[:, :2] *= 0.5
    images = photos()
    write_maps(tmp_path, images, flat)
    _, _, views = mn.average_normals(np.vstack([front, back]), np.vstack([faces, faces + len(front)]), {1: CAMERA},
                                     images, masks_of(images), tmp_path, log=lambda *_: None)
    assert np.median(views[:len(faces)]) == 4 and views[len(faces):].max() == 0


def test_the_maps_are_found_and_must_have_the_photos_shape(tmp_path):
    images = photos()
    assert mn.find_normals(tmp_path, images) is None
    write_maps(tmp_path / 'normals', images, flat)
    assert mn.find_normals(tmp_path, images) == tmp_path / 'normals'
    assert mn.find_normals(tmp_path, images, tmp_path / 'other') is None
    path = mn.normal_map_path(tmp_path / 'normals', images[0][0])
    assert mn._read_map(path, SIZE, SIZE).shape == (SIZE, SIZE, 3)
    assert mn._read_map(path, SIZE, SIZE // 2) is None
    assert mn._read_map(tmp_path / 'none.png', SIZE, SIZE) is None


def test_a_given_folder_without_maps_stops_before_the_long_run(tmp_path, monkeypatch):
    from raven_app import openmvs_mesh as om
    monkeypatch.setattr(od, 'check_binary', lambda b: b)
    monkeypatch.setattr(om, 'check_programs', lambda *_: None)
    monkeypatch.setattr(od, 'read_colmap_text', lambda _: ({1: CAMERA}, photos()))
    for name, *_ in photos():
        (tmp_path / 'images' / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / 'images' / name).write_bytes(b'')
    with pytest.raises(FileNotFoundError, match='normal maps'):
        od.densify_openmvs(tmp_path, 'x', mesh='ply', normals=tmp_path / 'nothing', log=lambda *_: None)
