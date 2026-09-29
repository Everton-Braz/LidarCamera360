"""Report RGB changes between two application-layout binary PLY clouds."""
import argparse
from pathlib import Path

import numpy as np


DTYPE = np.dtype([('xyz', '<f4', 3), ('rgb', 'u1', 3)])


def read(path):
    with Path(path).open('rb') as stream:
        header = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('Truncated PLY header')
            header.append(line)
            if line.strip() == b'end_header':
                break
        count = int(next(line.decode().split()[-1] for line in header
                         if line.startswith(b'element vertex ')))
        if b'format binary_little_endian 1.0\n' not in header:
            raise ValueError('Expected binary little-endian PLY')
        offset = stream.tell()
    return np.memmap(path, mode='r', offset=offset, dtype=DTYPE, shape=(count,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    args = parser.parse_args()
    before, after = read(args.before), read(args.after)
    if len(before) != len(after):
        raise ValueError('Point counts differ')
    if not np.array_equal(before['xyz'], after['xyz']):
        raise ValueError('XYZ coordinates differ')
    delta = after['rgb'].astype(np.int16) - before['rgb'].astype(np.int16)
    absolute = np.abs(delta)
    print(f'points={len(before)}')
    print(f'changed={np.count_nonzero(np.any(delta, axis=1))}')
    print(f'mean_abs_rgb={absolute.mean(axis=0).tolist()}')
    print(f'p95_abs_rgb={np.percentile(absolute, 95, axis=0).tolist()}')
    print(f'max_abs_rgb={absolute.max(axis=0).tolist()}')
    print(f'mean_rgb_distance={np.linalg.norm(delta, axis=1).mean():.5f}')


if __name__ == '__main__':
    main()
