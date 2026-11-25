"""Surface mesh of the dense cloud, textured from the whole fisheye photos, with OpenMVS.

Second, optional step of `openmvs_dense`: it needs what that step leaves in its work folder (the ideal fisheye
photos and their masks) and every dense point's photos, which carve the empty space in front of the surface.

    <dense_openmvs>/mesh.ply              the surface
    <dense_openmvs>/mesh_textured.glb     the same surface with the photos' texture (or .obj, .mtl, .jpg)

Like the dense step, this needs the patched OpenMVS programs (third_party/openmvs); no photo is cut into
pinhole faces. The texture of each triangle is copied from one fisheye photo. Normal maps of the photos, when
there are any, smooth the surface first (mesh_normals).
"""
from __future__ import annotations

import json
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from raven_app import openmvs_dense as od

FORMATS = ('glb', 'obj', 'ply')      # ply: the surface alone, without texture
TEXTURE_SIZE = 8192
POINT_SPACING = 1.5 / 1024          # of the photo width: closer points share a vertex, a few cm on a facade
SPURIOUS = 5                       # drops triangles longer than 5x the 95th percentile edge, and small loose parts
MAX_EDGE_M = 1.0                   # longer triangles bridge openings and the sky line
MIN_PART = 100                     # triangles; smaller loose parts float in the air
EMPTY_COLOR = 0x808080             # triangles no photo shows
MAX_FACES = 8_000_000              # a larger mesh is simplified before it is textured, which flips a few thin triangles


def companion(binary, name):
    """Path of another program of the same OpenMVS build (ReconstructMesh, TextureMesh)."""
    binary = Path(binary)
    return od.check_binary(binary.with_name(name + binary.suffix), name)


def check_programs(binary, fmt='glb'):
    """The programs this mesh format needs; raises when one is missing or is not the fisheye build."""
    if fmt not in FORMATS:
        raise ValueError(f'the mesh format must be one of {", ".join(FORMATS)}')
    return companion(binary, 'ReconstructMesh'), None if fmt == 'ply' else companion(binary, 'TextureMesh')


def has_programs(binary):
    try:
        check_programs(binary)
    except Exception:
        return False
    return True


def run_step(program, work, args, output, log):
    from raven_app.subprocess_utils import run_hidden_stream
    cmd = [str(program), '-w', str(work), *map(str, args), '-v', '2']
    log('[*] OpenMVS: ' + ' '.join(cmd))
    code = run_hidden_stream(cmd)
    if code != 0 or not output.is_file():
        logs = sorted(Path(work).glob(f'{program.stem}*.log'))
        raise RuntimeError(f'{program.stem} exited with code {code}' + (f'; see {logs[-1]}' if logs else ''))


def read_mesh_ply(path):
    """(vertices, faces) of a binary triangle mesh."""
    from raven_app.dense_fusion import _PLY_TYPES
    data = Path(path).read_bytes()
    end = data.index(b'end_header\n') + len(b'end_header\n')
    elements = []
    for line in data[:end].decode('ascii', 'replace').splitlines():
        t = line.split()
        if t[:1] == ['element']:
            elements.append((t[1], int(t[2]), []))
        elif t[:1] == ['property']:
            if t[1] == 'list':             # triangles only: a count of 3, then the 3 indices
                elements[-1][2].extend([(t[4] + '_n', _PLY_TYPES[t[2]]), (t[4], _PLY_TYPES[t[3]], 3)])
            else:
                elements[-1][2].append((t[2], _PLY_TYPES[t[1]]))
    found, pos = {}, end
    for name, count, fields in elements:
        dtype = np.dtype(fields)
        found[name] = np.frombuffer(data, dtype, count, pos)
        pos += count * dtype.itemsize
    vertex, face = found['vertex'], found['face']
    key = next(n for n in face.dtype.names if n in ('vertex_indices', 'vertex_index'))
    if len(face) and not (face[key + '_n'] == 3).all():
        raise ValueError(f'{path} is not a triangle mesh')
    return np.column_stack([vertex[a] for a in 'xyz']).astype(np.float64), face[key].astype(np.int64)


def write_mesh_ply(path, vertices, faces):
    with open(path, 'wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\n'
                 f'element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n'
                 f'element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n').encode())
        f.write(np.asarray(vertices, '<f4').tobytes())
        block = np.zeros(len(faces), np.dtype([('n', 'u1'), ('i', '<i4', 3)]))
        block['n'], block['i'] = 3, faces
        f.write(block.tobytes())


def clean_mesh(vertices, faces, max_edge, min_part=1):
    """The mesh without its triangles longer than `max_edge` m and then without its loose parts of fewer than
    `min_part` triangles; vertices no triangle uses go too."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    longest = np.max([np.linalg.norm(vertices[faces[:, a]] - vertices[faces[:, b]], axis=1)
                      for a, b in ((0, 1), (1, 2), (2, 0))], axis=0)
    faces = faces[longest <= max_edge]
    edges = np.vstack([faces[:, :2], faces[:, 1:]])
    part = connected_components(coo_matrix((np.ones(len(edges), bool), (edges[:, 0], edges[:, 1])),
                                           shape=(len(vertices), len(vertices))), directed=False)[1][faces[:, 0]]
    faces = faces[np.bincount(part, minlength=1)[part] >= min_part]
    used, index = np.unique(faces, return_inverse=True)
    return vertices[used], index.reshape(-1, 3)


def sample_surface(vertices, faces, count, seed=0):
    """`count` points spread evenly over the mesh's area, and that area in m2."""
    rng = np.random.default_rng(seed)
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    edges = np.cumsum(area)
    pick = np.searchsorted(edges, rng.random(count) * edges[-1]).clip(0, len(faces) - 1)
    u, v = rng.random((2, count, 1))
    flip = u + v > 1
    u, v = np.where(flip, 1 - u, u), np.where(flip, 1 - v, v)
    return a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick]), float(edges[-1])


def mesh_vs_lidar(lidar_xyz, vertices, faces, *, sample=200_000, seed=0):
    """How far the surface is from the LiDAR, and how much of the LiDAR the surface covers."""
    from scipy.spatial import cKDTree
    points, area = sample_surface(vertices, faces, min(8_000_000, max(sample, 4 * len(faces))), seed)
    report = {'faces': int(len(faces)), 'vertices': int(len(vertices)), 'area_m2': round(area, 1),
              'accuracy_vs_lidar': od.accuracy_vs_lidar(lidar_xyz, points, sample=sample, seed=seed)}
    rng = np.random.default_rng(seed)
    probe = lidar_xyz[rng.choice(len(lidar_xyz), min(sample, len(lidar_xyz)), replace=False)]
    d = cKDTree(points).query(probe, distance_upper_bound=1.0, workers=-1)[0]
    report['lidar_covered'] = {f'within_{c}cm': float((d < c / 100).mean()) for c in (2, 5, 10, 20)}
    return report


def pack_glb(path):
    """Put the image files a .glb points to inside it, so the one file is enough; the loose images are removed."""
    path = Path(path)
    data = path.read_bytes()
    length = struct.unpack_from('<I', data, 12)[0]
    scene = json.loads(data[20:20 + length])
    start = 20 + length
    body = bytearray(data[start + 8:start + 8 + struct.unpack_from('<I', data, start)[0]])
    loose = []
    for image in scene.get('images', []):
        if 'uri' not in image or image['uri'].startswith('data:'):
            continue
        source = path.parent / image.pop('uri')
        pixels = source.read_bytes()
        body += bytes(-len(body) % 4)
        scene['bufferViews'].append({'buffer': 0, 'byteOffset': len(body), 'byteLength': len(pixels)})
        image.update(bufferView=len(scene['bufferViews']) - 1,
                     mimeType='image/png' if source.suffix.lower() == '.png' else 'image/jpeg')
        body += pixels
        loose.append(source)
    if not loose:
        return
    body += bytes(-len(body) % 4)
    scene['buffers'][0]['byteLength'] = len(body)
    head = json.dumps(scene, separators=(',', ':')).encode()
    head += b' ' * (-len(head) % 4)
    with open(path, 'wb') as f:
        f.write(struct.pack('<4sII', b'glTF', 2, 28 + len(head) + len(body)))
        f.write(struct.pack('<I4s', len(head), b'JSON') + head)
        f.write(struct.pack('<I4s', len(body), b'BIN' + bytes(1)) + body)
    for source in loose:
        source.unlink()


def build_mesh(binary, work, out, size, images, xyz, rgb, views, *, fmt='glb', max_faces=MAX_FACES,
               lidar_xyz=None, smooth=None, log=print):
    """Mesh points that know their photos; `work` holds images/ and masks/. Returns the report.

    `smooth(vertices, faces)` returns the vertices to texture and a report of it; if it fails, the mesh stays as it
    is."""
    reconstruct, texturer = check_programs(binary, fmt)
    work, out, t0 = Path(work), Path(out), time.time()
    used = od.write_mvs(work / 'mesh_scene.mvs', size, images, xyz, rgb, views)
    log(f'[*] Meshing {used:,} points seen by 2 or more photos')
    surface = work / 'mesh.ply'
    # OpenMVS's own long-triangle filter works on the median edge, a few cm here, and opens the sparser facades
    run_step(reconstruct, work, ['-i', 'mesh_scene.mvs', '-o', 'mesh.ply', '--export-type', 'ply',
                                 '--min-point-distance', f'{POINT_SPACING * size:.2f}', '--max-edge-scale', '0',
                                 '--remove-spurious', SPURIOUS], surface, log)
    vertices, faces = clean_mesh(*read_mesh_ply(surface), MAX_EDGE_M, MIN_PART)
    if not len(faces):
        raise RuntimeError('OpenMVS reconstructed no surface')
    log(f'[+] Surface: {len(faces):,} triangles')
    report = {'points': int(used), 'seconds': {'surface': round(time.time() - t0, 1)}}
    if smooth is not None:
        t0 = time.time()
        try:
            vertices, report['normal_maps'] = smooth(vertices, faces)
        except Exception as exc:
            report['normal_maps'] = {'error': str(exc)}
            log(f'[!] The mesh was not smoothed with the normal maps: {exc}')
        report['seconds']['normal_maps'] = round(time.time() - t0, 1)
    write_mesh_ply(surface, vertices, faces)
    if lidar_xyz is not None:
        report.update(mesh_vs_lidar(lidar_xyz, vertices, faces))
    shutil.copyfile(surface, out / 'mesh.ply')
    report['mesh'] = str(out / 'mesh.ply')
    if texturer:
        t0 = time.time()
        for old in out.glob('mesh_textured*'):
            old.unlink()
        args = ['-i', 'mesh_scene.mvs', '-m', 'mesh.ply', '-o', 'mesh_textured.mvs', '--export-type', fmt,
                '--ignore-mask-label', '0', '--max-texture-size', TEXTURE_SIZE, '--export-texture-lossless', '0',
                '--empty-color', EMPTY_COLOR]
        if len(faces) > max_faces:
            args += ['--decimate', f'{max_faces / len(faces):.4f}']
        textured = work / f'mesh_textured.{fmt}'
        run_step(texturer, work, args, textured, log)
        for path in work.glob('mesh_textured*'):
            if path.suffix != '.mvs':
                shutil.move(str(path), out / path.name)
        if fmt == 'glb':
            pack_glb(out / textured.name)
        report['textured_mesh'] = str(out / textured.name)
        report['seconds']['texture'] = round(time.time() - t0, 1)
    return report
