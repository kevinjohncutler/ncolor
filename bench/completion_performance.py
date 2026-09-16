"""Paired production-path timings for retained layouts and prepared rendering."""
import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import ncolor
from feature_experiments import fingerprint


def paired(first, second, runs=31):
    expected = fingerprint(first())
    assert fingerprint(second()) == expected
    samples = [[], []]
    for iteration in range(runs + 2):
        for index in ((0, 1) if iteration % 2 else (1, 0)):
            start = time.perf_counter_ns()
            result = (first, second)[index]()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if iteration >= 2:
                samples[index].append(elapsed)
            assert fingerprint(result) == expected
    medians = [statistics.median(x) for x in samples]
    return dict(samples_ms=samples, median_ms=medians, speedup=medians[0] / medians[1], fingerprint=expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--rectangular', action='store_true')
    args = parser.parse_args()
    engine = ncolor.Engine(n_threads=args.threads)
    solver = engine._solver
    results = {}
    shapes = ([(128, 4096), (4096, 128), (257, 2048), (2048, 257)]
              if args.rectangular else [(512, 512), (2048, 2048), (96, 96, 96), (2, 513, 517)])
    for shape in shapes:
        for dense in (False, True):
            image = np.zeros(shape, np.int32)
            seeds = image[(slice(8, None, 32),) * len(shape)]
            seeds[:] = np.arange(1, seeds.size + 1).reshape(seeds.shape)
            if not seeds.size:
                image.flat[::997] = np.arange(1, image.flat[::997].size + 1)
            if dense:
                image = engine.expand_labels(image)
            for mode in ('standard', 'clean'):
                for wrap in (False, True):
                    options = dict(p=2, expand_mode=mode, wrap=wrap)
                    key = f't{args.threads}_{shape}_{dense}_{mode}_{wrap}'
                    results[key] = paired(
                        lambda: solver.label(image, _retain_layout=False, **options),
                        lambda: solver.label(image, **options))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
