"""Compare Manhattan (L1) and Euclidean (L2) coloring on synthetic N-D labels."""
import argparse
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ncolor


def make_mask(shape, n, dtype=np.int32, seed=0):
    """Paint local balls without allocating a full-image grid per label."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=dtype)
    radius = max(2, min(shape) // 16)
    for label in range(1, n + 1):
        center = [int(rng.integers(extent)) for extent in shape]
        lo = [max(0, c - radius) for c in center]
        hi = [min(extent, c + radius + 1) for extent, c in zip(shape, center)]
        coords = np.ogrid[tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, center))]
        inside = sum(axis * axis for axis in coords) <= radius * radius
        mask[tuple(slice(a, b) for a, b in zip(lo, hi))][inside] = label
    return mask


def time_call(call, runs, warmup=2):
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        call()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='use small smoke-test fixtures')
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--threads', type=int, default=-1)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    shapes = [(64, 64), (16, 16, 16)] if args.quick else [
        (512, 512), (1024, 1024), (2048, 2048), (4096, 4096),
        (64, 64, 64), (128, 128, 128), (256, 256, 256)]
    engine = ncolor.Engine(n_threads=args.threads)
    print('| Shape | Dtype | L2 ms | L1 ms | Different pixels |')
    print('|---|---|---|---|---|')
    for shape in shapes:
        for dtype in (np.uint16, np.int32):
            mask = make_mask(shape, max(20, int(np.prod(shape)) // 8000), dtype)
            outputs, timings = [], []
            for p in (2, 1):
                output, conflicts = engine.label(mask, p=p, return_conflicts=True)
                if conflicts:
                    raise RuntimeError(f'coloring has {conflicts} conflicts for {shape}, p={p}')
                outputs.append(output)
                timings.append(time_call(lambda: engine.label(mask, p=p), args.runs))
            different = np.count_nonzero(outputs[0] != outputs[1])
            name = 'x'.join(map(str, shape))
            print(f'| {name} | {dtype.__name__} | {timings[0]:.3f} | {timings[1]:.3f} | {different} |')


if __name__ == '__main__':
    main()
