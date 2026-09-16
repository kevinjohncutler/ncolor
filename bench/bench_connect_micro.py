"""Time adjacency extraction on expanded cell labels."""
import argparse

import numpy as np

from bench_lp_compare import make_mask, time_call
import ncolor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--threads', type=int, default=-1)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    shapes = [(64, 64), (16, 16, 16)] if args.quick else [
        (1024, 1024), (2048, 2048), (4096, 4096),
        (64, 64, 64), (128, 128, 128), (256, 256, 256)]
    engine = ncolor.Engine(n_threads=args.threads)
    print('| Shape | Label count | Pair count | Median ms |')
    print('|---|---|---|---|')
    for shape in shapes:
        mask = make_mask(shape, max(20, int(np.prod(shape)) // 8000))
        expanded = engine.expand_labels(mask, p=1)
        assert np.all(expanded > 0)
        np.testing.assert_array_equal(expanded[mask > 0], mask[mask > 0])
        pairs = engine.connect(expanded, conn=2)
        elapsed = time_call(lambda: engine.connect(expanded, conn=2), args.runs)
        name = 'x'.join(map(str, shape))
        print(f'| {name} | {len(np.unique(expanded))} | {len(pairs)} | {elapsed:.3f} |')


if __name__ == '__main__':
    main()
