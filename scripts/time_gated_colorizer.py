#!/usr/bin/env python3
"""Time-gated LiDAR colourisation for dynamic scenes (standalone, no app changes).

Why this exists
---------------
The app's colourisers pick the best views of each LiDAR point from the WHOLE capture.
In scenes that change between passes (cars that leave / arrive, people walking) the
accumulated LiDAR map holds every state at once, so a car scanned on pass 1 gets the
colours of the empty spot (or another car) filmed on pass 2, and the ground under a car
that left gets the car's colours.  ``colorize_via_direct_rigid`` additionally replaces
SLAM poses with SfM-derived ones whenever an alignment file exists ("trajectory drift
correction"), which on HarmonyMall comes from a rejected-quality alignment (RMSE 640 cm).

Method
------
* Every point carries its scan time (``scan_ranges.csv``: timestamp, point_start, count).
* Each frame (both lenses) only colours points scanned within ``±window`` seconds of the
  frame time; that subset is also the scene state used for the z-buffer, so occluders
  that existed at that moment are honoured and objects that did not exist are ignored.
* Pure SLAM poses (active SLAM dir), rigid calibration + dt in the SLAM-first-pose reference.
* Person masks (white = keep), fisheye validity circle, theta < 95 deg, range gate.
* Per point keep the best 3 observations (score: distance, off-axis angle, |dt|) and output
  their per-channel median.  Points without any in-window view fall back to a wider window.
* Operator removal: points closer than ``--operator-radius`` to the scanner at their own
  scan time, above the scanner-height floor margin, are dropped.

Usage
-----
python scripts/time_gated_colorizer.py --dataset <output_dir> --calib <rig.json> --dt <s> \
    --masks <output_dir>/masks --window 6 --operator-radius 1.0
"""
from __future__ import annotations

import argparse, json, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

from photometric_rig_calibration import find_slam_dir, load_calibration, load_trajectory, project_thin_prism


def load_pcd_xyz(path):
    with open(path, 'rb') as f:
        header = {}
        while True:
            line = f.readline().decode('ascii').strip().split()
            if not line or line[0].startswith('#'):
                continue
            header[line[0]] = line[1:]
            if line[0] == 'DATA':
                break
        names = header['FIELDS']; sizes = list(map(int, header['SIZE'])); types = header['TYPE']
        dtype = np.dtype([(n, '<' + {'F': 'f', 'I': 'i', 'U': 'u'}[t] + str(s)) for n, s, t in zip(names, sizes, types)])
        data = np.fromfile(f, dtype=dtype, count=int(header['POINTS'][0]))
    return np.column_stack([data['x'], data['y'], data['z']]).astype(np.float32)


def write_ply(path, xyz, rgb):
    v = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    v['x'], v['y'], v['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    v['r'], v['g'], v['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii'))
        v.tofile(f)


class TopK:
    """Best-K observations per point, streamed frame by frame."""

    def __init__(self, n, k=3):
        self.score = np.full((n, k), np.inf, np.float32)
        self.rgb = np.zeros((n, k, 3), np.uint8)

    def add(self, idx, score, rgb):
        worst = self.score[idx].argmax(1)
        better = score < self.score[idx, worst]
        idx, worst = idx[better], worst[better]
        self.score[idx, worst] = score[better]
        self.rgb[idx, worst] = rgb[better]

    def colours(self):
        seen = np.isfinite(self.score)
        cnt = seen.sum(1)
        rgb = self.rgb.astype(np.float32)
        rgb[~seen] = np.nan
        with np.errstate(all='ignore'):
            med = np.nanmedian(rgb, axis=1)
        return np.nan_to_num(med).astype(np.uint8), cnt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--dt', type=float, required=True, help='dt in the SLAM-first-pose reference')
    ap.add_argument('--masks', help='Person mask dir (<cam>/<frame>.jpg.png, white = keep)')
    ap.add_argument('--window', type=float, default=6.0, help='± seconds between point scan time and frame time')
    ap.add_argument('--fallback-window', type=float, default=45.0)
    ap.add_argument('--max-range', type=float, default=25.0)
    ap.add_argument('--min-range', type=float, default=0.6)
    ap.add_argument('--theta-max', type=float, default=95.0)
    ap.add_argument('--valid-radius-frac', type=float, default=1620 / 1920, help='fisheye circle radius / (W/2)')
    ap.add_argument('--image-scale', type=int, default=2, choices=(1, 2, 4), help='decode images at 1/N resolution')
    ap.add_argument('--operator-radius', type=float, default=0.0)
    ap.add_argument('--max-frames', type=int, default=0, help='debug: limit frames per lens')
    ap.add_argument('--out', help='Output PLY (default deliverables/lidar_colored_timegated_<ts>.ply)')
    a = ap.parse_args()

    ds = Path(a.dataset); t_start = time.time()
    slam = find_slam_dir(ds)
    print(f'[*] SLAM: {slam.name}')
    t, p, R = load_trajectory(slam); slerp = Slerp(t, R)
    xyz = load_pcd_xyz(slam / 'pcd' / 'all_raw_points.pcd')
    sr = np.loadtxt(slam / 'scan_ranges.csv', delimiter=',', skiprows=1)
    scan_t, scan_start, scan_cnt = sr[:, 0], sr[:, 1].astype(np.int64), sr[:, 2].astype(np.int64)
    assert scan_start[-1] + scan_cnt[-1] == len(xyz), 'scan_ranges.csv does not match the PCD'
    pt_t = np.repeat(scan_t, scan_cnt)
    finite = np.isfinite(xyz).all(1)
    print(f'[*] {len(xyz):,} points, {len(scan_t):,} scans, {(~finite).sum():,} non-finite')

    _, T0, T1, K0, K1 = load_calibration(a.calib)
    cams = {'cam0': (T0, K0), 'cam1': (T1, K1)}
    ft = json.loads((ds / 'images' / 'frames.json').read_text(encoding='utf-8'))['timestamps']
    frames = []
    for cam in cams:
        names = sorted(n for n in ft if n.startswith(cam + '/') and (ds / 'images' / n).is_file())
        if a.max_frames:
            names = names[:a.max_frames]
        for n in names:
            tq = t[0] + ft[n] - a.dt
            if t[0] <= tq <= t[-1]:
                frames.append((n, cam, tq))
    print(f'[*] {len(frames)} frames inside the trajectory, window ±{a.window}s, dt={a.dt:.4f}s')

    def pose(tq):
        return np.array([np.interp(tq, t, p[:, k]) for k in range(3)]), slerp([tq]).as_matrix()[0]

    flags = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}[a.image_scale]

    def colourise(top, window, only=None):
        for i, (n, cam, tq) in enumerate(frames):
            lo, hi = np.searchsorted(scan_t, [tq - window, tq + window])
            if hi <= lo:
                continue
            i0, i1 = scan_start[lo], scan_start[hi - 1] + scan_cnt[hi - 1]
            idx = np.arange(i0, i1)
            idx = idx[finite[i0:i1]]
            if only is not None:
                idx = idx[only[idx]]
                if not len(idx):
                    continue
            img = cv2.imread(str(ds / 'images' / n), flags)
            H, W = img.shape[:2]; s = 1.0 / a.image_scale
            T, K = cams[cam]
            pL, RL = pose(tq)
            pc = pL + RL @ T[:3, 3]; Rwc = RL @ T[:3, :3]
            # z-buffer uses ALL points of the window (scene state at that moment), colour only `idx`
            zidx = np.arange(i0, i1); zidx = zidx[finite[i0:i1]]
            P = (xyz[zidx] - pc) @ Rwc
            d = np.linalg.norm(P, axis=1)
            u, v, th = project_thin_prism(P.astype(np.float64), K)
            u *= s; v *= s
            ok = (th < np.deg2rad(a.theta_max)) & (d > a.min_range) & (d < a.max_range)
            ok &= np.hypot(u - W / 2, v - H / 2) < a.valid_radius_frac * W / 2
            ok &= (u >= 0) & (v >= 0) & (u < W - 1) & (v < H - 1)
            if a.masks:
                mk = cv2.imread(str(Path(a.masks) / (n + '.png')), cv2.IMREAD_GRAYSCALE)
                if mk is not None:
                    mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST) > 127
                    ok[ok] &= mk[v[ok].astype(int), u[ok].astype(int)]
            k = np.flatnonzero(ok)
            ui, vi = u[k].astype(np.int32), v[k].astype(np.int32)
            cell = (vi // 4) * (W // 4 + 1) + ui // 4
            zb = np.full(cell.max() + 1 if len(cell) else 1, np.inf, np.float32)
            np.minimum.at(zb, cell, d[k])
            vis = d[k] <= zb[cell] * 1.03 + 0.05
            k, ui, vi = k[vis], ui[vis], vi[vis]
            gidx = zidx[k]
            if only is not None:
                keep = only[gidx]; k, ui, vi, gidx = k[keep], ui[keep], vi[keep], gidx[keep]
            score = d[k] * (1.0 + 0.5 * (th[k] / (np.pi / 2)) ** 2) + 0.05 * np.abs(pt_t[gidx] - tq)
            rgb = img[vi, ui][:, ::-1]
            top.add(gidx, score.astype(np.float32), rgb)
            if (i + 1) % 100 == 0:
                print(f'    [{i + 1}/{len(frames)}] {time.time() - t_start:.0f}s', flush=True)

    top = TopK(len(xyz))
    colourise(top, a.window)
    rgb, cnt = top.colours()
    miss = finite & (cnt == 0)
    print(f'[*] in-window colour: {(cnt[finite] > 0).mean() * 100:.1f}% of points; fallback for {miss.sum():,}')
    if miss.any() and a.fallback_window > a.window:
        top2 = TopK(len(xyz))
        colourise(top2, a.fallback_window, only=miss)
        rgb2, cnt2 = top2.colours()
        rgb[miss] = rgb2[miss]; cnt[miss] = cnt2[miss]
    keep = finite & (cnt > 0)
    print(f'[*] uncoloured after fallback: {(finite & (cnt == 0)).sum():,} points (dropped)')

    if a.operator_radius > 0:
        sc = np.repeat(np.arange(len(scan_t)), scan_cnt)
        tq = scan_t[sc]
        pos = np.column_stack([np.interp(tq, t, p[:, k]) for k in range(3)])
        near = np.linalg.norm(xyz - pos, axis=1) < a.operator_radius
        body = near & (xyz[:, 2] > pos[:, 2] - 1.2)
        keep &= ~body
        print(f'[*] operator removal ({a.operator_radius} m): {body.sum():,} points')

    out = Path(a.out) if a.out else ds / 'deliverables' / f'lidar_colored_timegated_{datetime.now():%Y%m%d_%H%M%S}.ply'
    out.parent.mkdir(parents=True, exist_ok=True)
    write_ply(out, xyz[keep], rgb[keep])
    print(f'[+] {keep.sum():,} points -> {out}  ({time.time() - t_start:.0f}s)')


if __name__ == '__main__':
    main()
