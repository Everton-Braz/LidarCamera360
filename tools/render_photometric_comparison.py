"""Render identical orthographic point samples for before/after inspection."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read(path):
    with path.open('rb') as stream:
        for _ in range(50):
            if stream.readline().strip() == b'end_header':
                break
        else:
            raise ValueError('Invalid PLY header')
        offset = stream.tell()
    return np.memmap(path, mode='r', offset=offset,
                     dtype=[('xyz', '<f4', 3), ('rgb', 'u1', 3)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--azimuth', type=float, action='append', help='Horizontal viewing angle in radians; repeat for multiple views')
    parser.add_argument('--elevation', type=float, default=.6, help='Vertical component of the view direction')
    parser.add_argument('--before-label', default='ANTES')
    parser.add_argument('--after-label', default='DEPOIS')
    parser.add_argument('--height', type=int, default=650)
    parser.add_argument('--along', type=float, nargs=2, metavar=('START', 'END'),
                        help='Crop to this fraction of the horizontal principal axis (0..1)')
    args = parser.parse_args()
    before, after = read(args.before), read(args.after)
    if len(before) != len(after):
        raise ValueError('Clouds must share identical vertex order and XYZ')
    indices = np.arange(0, len(before), max(1, len(before) // 4000000))
    xyz = before['xyz'][indices].astype(float)
    if not np.array_equal(xyz, after['xyz'][indices]):
        raise ValueError('Sample XYZ differ')
    low, high = np.percentile(xyz, [1, 99], axis=0)
    keep = np.all((xyz >= low) & (xyz <= high), axis=1)
    xyz, indices = xyz[keep], indices[keep]
    if args.along:
        if not 0 <= args.along[0] < args.along[1] <= 1:
            parser.error('--along requires 0 <= START < END <= 1')
        _, axes = np.linalg.eigh(np.cov(xyz[:, :2].T))
        axis = axes[:, -1]
        if axis[0] < 0:
            axis = -axis
        along = xyz[:, :2] @ axis
        fraction = (along - along.min()) / np.ptp(along)
        keep = (fraction >= args.along[0]) & (fraction <= args.along[1])
        xyz, indices = xyz[keep], indices[keep]
        low, high = xyz.min(0), xyz.max(0)
    xyz -= (low + high) / 2
    width, height = 1000, args.height
    if height < 100:
        parser.error('--height must be at least 100')
    azimuths = args.azimuth or (-.7, 2.4)
    canvas = Image.new('RGB', (width * 2, (height + 35) * len(azimuths)), '#eff2f5')
    draw = ImageDraw.Draw(canvas)
    for row, azimuth in enumerate(azimuths):
        right = np.array([np.cos(azimuth), np.sin(azimuth), 0])
        forward = np.array([-np.sin(azimuth), np.cos(azimuth), args.elevation])
        forward /= np.linalg.norm(forward)
        up = np.cross(forward, right)
        if up[2] < 0:
            up = -up
        projected = np.column_stack((xyz @ right, xyz @ up))
        scale = min((width-30) / np.ptp(projected[:, 0]), (height-30) / np.ptp(projected[:, 1]))
        padding = (np.array([width, height]) - np.ptp(projected, axis=0) * scale) / 2
        uv = np.rint((projected-projected.min(0)) * scale + padding).astype(int)
        uv[:, 1] = height - 1 - uv[:, 1]
        pixel = uv[:, 1] * width + uv[:, 0]
        order = np.argsort(xyz @ forward, kind='stable')
        _, first = np.unique(pixel[order], return_index=True)
        visible = order[first]
        for column, cloud in enumerate((before, after)):
            frame = np.full((height, width, 3), [20, 28, 38], dtype=np.uint8)
            frame[uv[visible, 1], uv[visible, 0]] = cloud['rgb'][indices[visible]]
            top = row * (height + 35)
            draw.text((column*width+20, top+10), args.before_label if column == 0 else args.after_label, fill='#202633')
            canvas.paste(Image.fromarray(frame), (column*width, top+35))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)


if __name__ == '__main__':
    main()
