import json
import math
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from raven_app import openmvs_dense as od
from raven_app import openmvs_mesh as om
from raven_app import photogrammetry_export as pe
from raven_app.dense_fusion import write_ply

W = 400
LENS = (W / 3.5, W / 3.5, W / 2, W / 2, 0.02, -0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture
def colmap(tmp_path):
    """Three fisheye photos, 0.5 m apart, in front of a coloured wall."""
    root = tmp_path / 'colmap_3dgs'
    sparse = root / 'sparse' / '0'
    sparse.mkdir(parents=True)
    (root / 'images' / 'cam0').mkdir(parents=True)
    (root / 'masks' / 'cam0').mkdir(parents=True)
    (sparse / 'cameras.txt').write_text(f'1 THIN_PRISM_FISHEYE {W} {W} ' + ' '.join(map(str, LENS)) + '\n')
    lines = []
    for i in range(3):
        name = f'cam0/frame_{i:06d}.jpg'
        lines += [f'{i + 1} 1 0 0 0 {-0.5 * i} 0 0 1 {name}', '']
        cv2.imwrite(str(root / 'images' / name), np.full((W, W, 3), 60 * i + 60, np.uint8))
        mask = np.full((W, W), 255, np.uint8)
        mask[:, :W // 4] = 0
        cv2.imwrite(str(root / 'masks' / (name + '.png')), mask)
    (sparse / 'images.txt').write_text('\n'.join(lines) + '\n')
    g = np.arange(-1, 1, 0.05)
    x, y = np.meshgrid(g, g)
    xyz = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 2.0)])
    write_ply(sparse / 'points3D.ply', xyz, np.full((len(xyz), 3), 120, np.uint8))
    return root


def test_the_stereographic_rays_are_realityscans_division_lens():
    size = 301
    rays, theta = od.stereographic_rays(size)
    f = od.stereographic_focal(size)
    c = (np.arange(size) - (size - 1) / 2) / f
    u, v = (a.ravel() for a in np.meshgrid(c, c))
    k = 1 + pe.STEREOGRAPHIC_K1 * (u * u + v * v)
    front = k > 1e-3                                  # undistorted = distorted / (1 + k1 r^2)
    np.testing.assert_allclose(rays[front, 0] / rays[front, 2], u[front] / k[front], atol=1e-6, rtol=1e-6)
    assert math.isclose(theta[size // 2, -1], od.MAX_THETA, abs_tol=0.02)


def test_equidistant_rays_project_back_on_their_pixel():
    size = 201
    rays, theta = od.equidistant_rays(size)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-9)
    xy = od.equidistant_project(rays, size)
    xy = xy[0] if isinstance(xy, tuple) else xy
    gx, gy = np.meshgrid(np.arange(size), np.arange(size))
    inside = theta.ravel() < od.MAX_THETA
    np.testing.assert_allclose(xy[inside], np.column_stack([gx.ravel(), gy.ravel()])[inside], atol=1e-5)


def test_metashape_export_keeps_whole_fisheyes_masks_and_poses(colmap):
    import json
    out = pe.export_metashape(colmap, size=256, log=lambda *_: None)
    data = json.loads((out / 'cameras.json').read_text())
    assert [c['image'] for c in data['cameras']] == [f'cam0_frame_{i:06d}.jpg' for i in range(3)]
    np.testing.assert_allclose([c['camera_to_world'][0][3] for c in data['cameras']], [0, 0.5, 1.0])
    assert math.isclose(data['focal'], 256 / od.FISHEYE_FOV)
    mask = cv2.imread(str(out / 'masks' / 'cam0_frame_000000.png'), 0)
    assert mask[128, 128] == 255 and mask[128, 20] == 0 and mask[0, 0] == 0
    compile((out / 'import_metashape.py').read_text(), 'import_metashape.py', 'exec')


def test_realityscan_export_locks_pose_and_lens(colmap):
    out = pe.export_realityscan(colmap, size=256, log=lambda *_: None)
    xmp = (out / 'images' / 'cam0_frame_000002.xmp').read_text()
    assert 'DistortionModel="division"' in xmp and '>-0.25 0 0' in xmp and 'PosePrior="locked"' in xmp
    assert '<xcr:Position>1 ' in xmp
    assert (out / 'images' / 'cam0_frame_000002.jpg.mask.png').is_file()
    assert (out / 'lidar.las').is_file()
    assert 'lasVcamsCameraPosesMethod" value="0"' in (out / 'lidar_import.xml').read_text()
    assert 'From camera pose priors' in (out / 'IMPORT_LIDAR.txt').read_text()


def test_exporting_again_keeps_the_users_project(colmap):
    out = pe.export_metashape(colmap, size=128, log=lambda *_: None)
    (out / 'mine.psx').write_text('x')
    pe.export_metashape(colmap, size=128, log=lambda *_: None)
    assert (out / 'mine.psx').is_file()


def test_realityscan_is_opened_with_the_division_lens(tmp_path):
    command, = pe.open_commands('realityscan', Path('RealityScan.exe'), tmp_path)
    assert 'sfmDistortionModel=Division' in command
    assert len(pe.open_commands('metashape', Path('metashape.exe'), tmp_path)) == 2


def test_fusion_keeps_the_photos_and_adds_lidar_only_in_their_gaps():
    g = np.arange(0, 1, 0.02)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    photo = np.column_stack([x, y, np.zeros(x.size)])
    lidar = np.vstack([photo[::7] + [0, 0, 0.015], [[3.0, 3, 0], [0.5, 0.5, 0.2]]])
    keep = od.fuse_with_lidar(lidar, photo)
    assert keep.tolist() == [len(lidar) - 2, len(lidar) - 1]


def test_stock_openmvs_is_refused(tmp_path):
    stock = tmp_path / 'DensifyPointCloud.exe'
    stock.write_bytes(b'not the fisheye build')
    with pytest.raises(Exception):
        od.check_binary(stock)


def test_mvs_scene_header(tmp_path, colmap):
    cameras, images = od.read_colmap_text(colmap / 'sparse' / '0')
    xyz = np.array([[0.0, 0, 2], [0.2, 0, 2]])
    path = tmp_path / 'scene.mvs'
    n = od.write_mvs(path, 256, images, xyz, np.zeros((2, 3), np.uint8), np.array([[0, 1], [1, 2]]))
    head = path.read_bytes()[:12]
    assert n == 2 and head[:4] == b'MVSI' and struct.unpack('<I', head[4:8])[0] == 8


def test_failed_dense_run_changes_nothing(colmap, tmp_path):
    before = sorted(p.relative_to(colmap) for p in colmap.rglob('*'))
    with pytest.raises(Exception):
        od.densify_openmvs(colmap, tmp_path / 'missing.exe', log=lambda *_: None)
    assert sorted(p.relative_to(colmap) for p in colmap.rglob('*')) == before


WALL = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2]], float), np.array([[0, 1, 2], [0, 2, 3]])


def test_mesh_ply_round_trip_and_distance_to_the_lidar(tmp_path):
    om.write_mesh_ply(tmp_path / 'm.ply', *WALL)
    vertices, faces = om.read_mesh_ply(tmp_path / 'm.ply')
    np.testing.assert_allclose(vertices, WALL[0])
    assert faces.tolist() == WALL[1].tolist()
    points, area = om.sample_surface(vertices, faces, 1000)
    assert math.isclose(area, 4.0) and np.abs(points[:, :2]).max() <= 1
    g = np.arange(-1, 1, 0.02)
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    report = om.mesh_vs_lidar(np.column_stack([x, y, np.full(x.size, 2.02)]), vertices, faces, sample=20_000)
    assert math.isclose(report['accuracy_vs_lidar']['median_m'], 0.02, abs_tol=2e-3)
    assert report['lidar_covered']['within_5cm'] > 0.99


def test_long_triangles_and_small_loose_parts_are_removed():
    vertices = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [5, 5, 0], [0.1, 0.1, 0], [3, 3, 0], [3.1, 3, 0], [3, 3.1, 0]])
    faces = np.array([[0, 1, 2], [1, 2, 3], [1, 4, 2], [5, 6, 7]])
    v, f = om.clean_mesh(vertices, faces, 0.3)
    assert len(f) == 3 and len(v) == 7
    v, f = om.clean_mesh(vertices, faces, 0.3, min_part=2)
    assert f.tolist() == [[0, 1, 2], [1, 3, 2]] and len(v) == 4
    assert all(len(a) == 0 for a in om.clean_mesh(vertices, faces, 0.01, min_part=2))


def test_a_glb_holds_its_own_texture(tmp_path):
    pixels = b'not really a jpeg'
    (tmp_path / 't.jpg').write_bytes(pixels)
    scene = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 6}], 'images': [{'uri': 't.jpg'}],
             'bufferViews': [{'buffer': 0, 'byteLength': 6}]}
    head = json.dumps(scene).encode()
    head += b' ' * (-len(head) % 4)
    glb = tmp_path / 'm.glb'
    glb.write_bytes(struct.pack('<4sII', b'glTF', 2, 36 + len(head)) + struct.pack('<I4s', len(head), b'JSON') + head
                    + struct.pack('<I4s', 8, b'BIN' + bytes(1)) + b'123456' + bytes(2))
    om.pack_glb(glb)
    data = glb.read_bytes()
    n = struct.unpack_from('<I', data, 12)[0]
    scene = json.loads(data[20:20 + n])
    view = scene['bufferViews'][scene['images'][0]['bufferView']]
    body = data[28 + n:]
    assert body[view['byteOffset']:view['byteOffset'] + view['byteLength']] == pixels and body[:6] == b'123456'
    assert struct.unpack_from('<I', data, 8)[0] == len(data) and scene['buffers'][0]['byteLength'] == len(body)
    assert not (tmp_path / 't.jpg').exists() and 'uri' not in scene['images'][0]
    om.pack_glb(glb)
    assert glb.read_bytes() == data


@pytest.fixture
def openmvs(tmp_path, monkeypatch):
    """Marked stand-ins for the three programs, and what they would write."""
    folder = tmp_path / 'openmvs'
    folder.mkdir()
    monkeypatch.setattr(om, 'MIN_PART', 1)       # WALL has two triangles
    for name in ('DensifyPointCloud', 'ReconstructMesh', 'TextureMesh'):
        (folder / f'{name}.exe').write_bytes(od.PATCH_MARK)

    def dense(binary, work, size, **_):
        return work

    def read(work):
        g = np.arange(-0.9, 0.9, 0.03)
        x, y = (a.ravel() for a in np.meshgrid(g, g))
        return (np.column_stack([x, y, np.full(x.size, 2.0)]), np.full((x.size, 3), 90, np.uint8),
                np.tile([0, 1, 2], (x.size, 1)))

    def step(program, work, args, output, log):
        if program.stem == 'ReconstructMesh':
            om.write_mesh_ply(output, *WALL)
        else:
            output.write_bytes(b'')
            assert '--ignore-mask-label' in args

    monkeypatch.setattr(od, 'run_openmvs', dense)
    monkeypatch.setattr(od, 'read_dense_ply', read)
    monkeypatch.setattr(om, 'run_step', step)
    monkeypatch.setattr(om, 'pack_glb', lambda path: None)
    monkeypatch.setattr(om, 'MAX_EDGE_M', 10.0)
    monkeypatch.setitem(od.QUALITY, 'draft', 256)
    return folder / 'DensifyPointCloud.exe'


def test_dense_run_also_writes_the_mesh(colmap, openmvs):
    report = od.densify_openmvs(colmap, openmvs, quality='draft', mesh='glb', log=lambda *_: None)
    out = colmap / 'dense_openmvs'
    assert report['mesh']['faces'] == 2 and (out / 'mesh.ply').is_file() and (out / 'mesh_textured.glb').is_file()
    assert json.loads((out / 'report.json').read_text())['mesh']['faces'] == 2 and not (out / 'work').exists()


@pytest.mark.parametrize('normals, uses', [(None, 'normals'), (False, None), ('mine', 'mine')])
def test_the_mesh_uses_the_normal_maps_of_the_photos(colmap, openmvs, monkeypatch, normals, uses):
    from raven_app import mesh_normals
    for folder in ('normals', 'mine'):
        for image in od.read_colmap_text(colmap / 'sparse' / '0')[1]:
            path = mesh_normals.normal_map_path(colmap / folder, image[0])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'')
    given = []

    def smooth(vertices, faces, cameras, images, masks, folder, *, log):
        given.append(folder)
        assert len(masks) == len(images) and masks[0].shape == (od.VIS_SIZE, od.VIS_SIZE)
        return vertices + [0, 0, 0.001], {'guided_share': 1.0}

    monkeypatch.setattr(mesh_normals, 'smooth_by_normals', smooth)
    report = od.densify_openmvs(colmap, openmvs, quality='draft', mesh='ply', log=lambda *_: None,
                                normals=colmap / normals if normals else normals)
    assert given == ([colmap / uses] if uses else [])
    vertices = om.read_mesh_ply(colmap / 'dense_openmvs' / 'mesh.ply')[0]
    assert np.allclose(vertices[:, 2], WALL[0][:, 2] + (0.001 if uses else 0))
    assert ('normal_maps' in report['mesh']) == bool(uses)


def test_a_failed_smoothing_keeps_the_mesh(colmap, openmvs, monkeypatch):
    from raven_app import mesh_normals

    def fail(*_, **__):
        raise MemoryError('no room')

    monkeypatch.setattr(mesh_normals, 'find_normals', lambda *_: colmap)
    monkeypatch.setattr(mesh_normals, 'smooth_by_normals', fail)
    report = od.densify_openmvs(colmap, openmvs, quality='draft', mesh='ply', log=lambda *_: None)
    assert report['mesh']['faces'] == 2 and report['mesh']['normal_maps'] == {'error': 'no room'}


def test_the_aligned_photos_go_to_the_matching_without_their_shutter(colmap, openmvs, monkeypatch):
    from raven_app import photo_refine
    before = {p: p.read_bytes() for p in (colmap / 'sparse').rglob('*') if p.is_file()}
    motion = (np.array([0.0, 0.03]), np.array([0.0, 2.0, 0.0]), np.zeros(3))
    seen = {}

    def aligned(folder, cameras, images, *, run_dir, lidar_xyz, log):
        seen.update(run_dir=run_dir, points=len(lidar_xyz))
        moved = [(n, c, R, C + [0, 0, 0.01]) for n, c, R, C in images]
        return cameras, moved, [motion] * len(images), {'status': 'accepted'}

    def scene(path, size, images, *args):
        seen['centre'] = images[1][3]
        return write(path, size, images, *args)

    write = od.write_mvs
    monkeypatch.setattr(photo_refine, 'refine_photos', aligned)
    monkeypatch.setattr(od, 'write_mvs', scene)
    report = od.densify_openmvs(colmap, openmvs, quality='draft', run_dir=colmap / 'run', keep_work=True, log=lambda *_: None)
    assert report['photo_alignment'] == {'status': 'accepted'}
    assert seen['run_dir'] == colmap / 'run' and seen['points'] == 1600
    np.testing.assert_allclose(seen['centre'], [0.5, 0, 0.01])
    moved = cv2.imread(str(colmap / 'dense_openmvs' / 'work' / 'masks' / '00000.mask.png'), 0)
    report = od.densify_openmvs(colmap, openmvs, quality='draft', refine=False, keep_work=True, log=lambda *_: None)
    assert report['photo_alignment'] == {'status': 'off'}
    still = cv2.imread(str(colmap / 'dense_openmvs' / 'work' / 'masks' / '00000.mask.png'), 0)
    assert (moved != still).any()             # the mask moves with its photo
    (colmap / 'colmap_export_report.json').write_text(json.dumps({'pose_source': 'aligned', 'photo_alignment': {'held_out_px': 0.7}}))
    monkeypatch.setattr(photo_refine, 'refine_photos', lambda *a, **k: pytest.fail('aligned twice'))
    report = od.densify_openmvs(colmap, openmvs, quality='draft', log=lambda *_: None)
    assert report['photo_alignment'] == {'held_out_px': 0.7}
    assert before == {p: p.read_bytes() for p in (colmap / 'sparse').rglob('*') if p.is_file()}


def test_a_failed_mesh_keeps_the_dense_cloud(colmap, openmvs, monkeypatch):
    def broken(*_):
        raise RuntimeError('out of memory')
    monkeypatch.setattr(om, 'run_step', broken)
    report = od.densify_openmvs(colmap, openmvs, quality='draft', mesh='ply', log=lambda *_: None)
    assert 'out of memory' in report['mesh']['error'] and (colmap / 'dense_openmvs' / 'dense_fused.ply').is_file()


def test_a_mesh_without_its_programs_stops_before_the_long_run(colmap, openmvs):
    openmvs.with_name('TextureMesh.exe').unlink()
    assert not om.has_programs(openmvs)
    with pytest.raises(FileNotFoundError, match='TextureMesh'):
        od.densify_openmvs(colmap, openmvs, quality='draft', mesh='glb', log=lambda *_: None)
    assert not (colmap / 'dense_openmvs').exists()
    assert om.check_programs(openmvs, 'ply')[1] is None


def test_views_of_two_clouds_are_stacked():
    assert od.stack_views(np.array([[1, 2]]), np.array([[3, -1, -1]])).tolist() == [[1, 2, -1], [3, -1, -1]]


def _street(size=600, wall=(235, 235, 235)):
    """A fisheye photo of a pale wall under a blue sky, and the wall's LiDAR points (camera level, z up)."""
    R = np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]])       # looks along +y
    rays = od.equidistant_rays(size)[0] @ R
    t = 5 / np.maximum(rays[:, 1], 1e-6)
    built = ((rays * t[:, None])[:, 2] < 3) & (rays[:, 1] > 0.02)
    ground = rays[:, 2] < 0
    photo = np.full((size * size, 3), (250, 210, 160), np.uint8)
    photo[built] = wall
    photo[ground] = (90, 95, 100)
    x, z = (a.ravel() for a in np.meshgrid(np.arange(-40, 40, 0.06), np.arange(-1.5, 3, 0.06)))
    return photo.reshape(size, size, 3), built.reshape(size, size), R, np.column_stack([x, np.full(x.size, 5.0), z])


def test_the_sky_is_masked_and_a_pale_wall_is_not():
    photo, built, R, surface = _street()
    rays, theta = od.equidistant_rays(od.SKY_SIZE)
    sky = od.sky_mask(photo, R, np.zeros(3), surface, rays, theta)
    inside = od.equidistant_rays(600)[1] < 1.35
    up = (od.equidistant_rays(600)[0] @ R)[:, 2].reshape(600, 600)
    open_sky = inside & ~built & (up > 0.1)
    assert sky[open_sky].mean() > 0.97 and sky[inside & built].mean() < 0.02
    assert not sky[up < 0].any()


def test_no_sky_is_claimed_where_the_lidar_sees_everything_or_nothing():
    photo, _, R, surface = _street()
    rays, theta = od.equidistant_rays(od.SKY_SIZE)
    x, y = (a.ravel() for a in np.meshgrid(np.arange(-20, 20, 0.05), np.arange(-20, 20, 0.05)))
    ceiling = np.column_stack([x, y, np.full(x.size, 2.0)])
    assert not od.sky_mask(photo, R, np.zeros(3), np.vstack([surface, ceiling]), rays, theta).any()
    assert not od.sky_mask(photo, R, np.zeros(3), np.zeros((0, 3)), rays, theta).any()
