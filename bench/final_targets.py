"""Whole-call checks for component, weighted-layout, and prepared-render changes."""
import argparse
import json
import os
from pathlib import Path
import sys

from release_comparison import cpu_model, fingerprint, timed
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import ncolor
    assert Path(ncolor.__file__).resolve().is_relative_to(args.source.resolve())
    from skimage.measure import label
    engine = ncolor.Engine(n_threads=args.threads)
    results = {}
    rng = np.random.default_rng(712)

    def record(name, call, check=None):
        expected = call()
        expected_hash = fingerprint(expected)
        def validate(actual):
            assert fingerprint(actual) == expected_hash
            if check:
                check(actual)
            return {'fingerprint': expected_hash}
        results[name] = timed(call, validate, 15)
        print(name, round(results[name]['median_ms'], 3), flush=True)

    for shape in [(2, 513, 517), (1024, 1024), (96, 96, 96)]:
        for density in (.01, .1, .7, 1.):
            image = rng.random(shape) < density
            for conn in sorted({1, image.ndim}):
                reference = label(image, connectivity=conn)
                record(f'components/{shape}/{density}/{conn}',
                       lambda: engine.connected_components(image, conn=conn)[0],
                       lambda actual: np.testing.assert_array_equal(actual, reference))

    for shape in [(512, 768), (1024, 1024), (96, 96, 96)]:
        image = np.zeros(shape, np.int32)
        for i in range(1, 33):
            center = rng.integers(4, min(shape)-4, len(shape))
            image[tuple(slice(int(c-3), int(c+3)) for c in center)] = i
        for mode in ('standard', 'clean'):
            for weight in ('min', 'max', 'mean', 'count', 'harmonic', 'mean_inv'):
                record(f'weighted/{shape}/{mode}/{weight}',
                       lambda: engine.label(image, n=32, expand_mode=mode,
                                            weight_objective=1, weight_mode=weight))

    for shape in [(512, 512), (2048, 2048), (96, 96, 96)]:
        image = np.zeros(shape, np.int32)
        seeds = image[(slice(8, None, 32),)*len(shape)]
        seeds[:] = np.arange(1, seeds.size+1).reshape(seeds.shape)
        dense = engine.expand_labels(image)
        for fraction in (.001, .01, .1, 1.):
            sparse = np.where(rng.random(shape) < fraction, dense, 0).astype(np.int32)
            # Repeated rendering includes output initialization and coloring.
            prepared = engine.prepare_labels(sparse)
            # A generous palette keeps the timed search out of the measurement.
            expected = engine.label(sparse, n=32)
            record(f'prepared/{shape}/{fraction}', lambda: prepared.color(n=32, engine=engine),
                   lambda actual: np.testing.assert_array_equal(actual, expected))
            results[f'prepared/{shape}/{fraction}']['snapshot_bytes'] = prepared.nbytes
            record(f'prepare/{shape}/{fraction}', lambda: np.asarray(
                [engine.prepare_labels(sparse).n_labels], np.int32))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(cpu=cpu_model(), threads=args.threads,
                                         affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                                         results=results), indent=2)+'\n')
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
