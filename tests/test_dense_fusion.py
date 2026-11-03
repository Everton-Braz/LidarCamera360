import numpy as np

from raven_app.dense_fusion import additive_fusion, read_ply, write_ply


def _wall(step, hole=None):
    g = np.arange(0, 2, step)
    y, z = np.meshgrid(g, g)
    pts = np.column_stack([np.zeros(y.size), y.ravel(), z.ravel()])
    if hole is not None:
        (y0, y1), (z0, z1) = hole
        pts = pts[~((pts[:, 1] > y0) & (pts[:, 1] < y1) & (pts[:, 2] > z0) & (pts[:, 2] < z1))]
    return pts


def test_roma_points_fill_only_the_lidar_gaps_and_never_float_free():
    lidar = _wall(0.01, hole=((0.8, 1.2), (0.8, 1.2)))           # 1 cm scan with a 40 cm hole
    dense = np.vstack([_wall(0.005) + [0.002, 0, 0], [[3.0, 1.0, 1.0]]])   # RoMa everywhere + a floater
    keep, report = additive_fusion(lidar, dense, query_voxel_m=0)
    added = dense[keep]
    assert len(added) > 0 and report['added'] == len(added)
    inside = (added[:, 1] > 0.8) & (added[:, 1] < 1.2) & (added[:, 2] > 0.8) & (added[:, 2] < 1.2)
    assert inside.all(), 'nothing is added where the scan already has points'
    assert not (added[:, 0] > 1).any(), 'the floater 3 m away stays out'
    # thinned to ~1 cm: the 40 cm hole gets at most ~40 x 40 points, not RoMa's 0.5 cm density
    assert len(added) <= 41 * 41


def test_ply_round_trip(tmp_path):
    xyz = np.random.default_rng(1).normal(size=(50, 3))
    rgb = np.random.default_rng(2).integers(0, 255, (50, 3)).astype(np.uint8)
    path = write_ply(tmp_path / 'x.ply', xyz, rgb)
    got_xyz, got_rgb = read_ply(path)
    np.testing.assert_allclose(got_xyz, xyz.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(got_rgb, rgb)


def test_densify_export_makes_the_fused_cloud_the_seed_and_moves_roma_out_of_dense(tmp_path):
    from raven_app.dense_fusion import densify_colmap_export
    from raven_app.workflow import write_colmap_points3d
    lidar = _wall(0.01, hole=((0.8, 1.2), (0.8, 1.2)))
    sparse = tmp_path / 'sparse' / '0'
    write_colmap_points3d(sparse, lidar, np.full((len(lidar), 3), 100, np.uint8))
    dense = _wall(0.005) + [0.002, 0, 0]
    write_ply(tmp_path / 'dense' / 'roma.ply', dense, np.full((len(dense), 3), 200, np.uint8))
    report = densify_colmap_export(tmp_path, spirula_bin=None, query_voxel_m=0, log=lambda *_: None)
    assert not (tmp_path / 'dense').exists(), 'Spirula must not find RoMa alone as the dense seed'
    assert (tmp_path / 'densify' / 'roma_run' / 'roma.ply').is_file()
    seed_xyz, seed_rgb = read_ply(sparse / 'points3D.ply')
    assert len(seed_xyz) == len(lidar) + report['added'] and report['added'] > 0
    assert (seed_rgb[:len(lidar)] == 100).all() and (seed_rgb[len(lidar):] == 200).all()
    assert len(read_ply(sparse / 'points3D_lidar_only.ply')[0]) == len(lidar)
    assert (sparse / 'points3D.bin').is_file() and (sparse / 'points3D_lidar_only.bin').is_file()
    again = densify_colmap_export(tmp_path, spirula_bin=None, query_voxel_m=0, log=lambda *_: None)
    assert again['added'] == report['added'], 'a rerun fuses from the LiDAR-only seed, not the fused one'
    assert len(read_ply(sparse / 'points3D.ply')[0]) == len(lidar) + report['added']
