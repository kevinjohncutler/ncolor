"""Measure complete calls, preparation, and repeated prepared coloring."""
import argparse
import json
from pathlib import Path

import numpy as np
import ncolor
from feature_experiments import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    engine = ncolor.Engine(n_threads=4)
    results = {}
    for shape in [(512, 512), (2048, 2048), (96, 96, 96)]:
        image = np.zeros(shape, np.int32)
        seeds = image[(slice(8, None, 32),) * len(shape)]
        seeds[:] = np.arange(1, seeds.size + 1).reshape(seeds.shape)
        for mode in ('standard', 'clean'):
            for weighted in (False, True):
                options = dict(expand_mode=mode, weight_objective=int(weighted))
                prepared = engine.prepare_labels(image, **options)
                expected = engine.label(image, **options)
                np.testing.assert_array_equal(prepared.color(engine=engine), expected)
                key = f'{shape}_{mode}_weighted_{weighted}'
                def prepare():
                    snapshot = engine.prepare_labels(image, **options)
                    return np.array([snapshot.n_labels, snapshot.nbytes], dtype=np.int64)
                results[key] = {
                    'complete': measure(lambda: engine.label(image, **options), 15),
                    'prepare': measure(prepare, 15),
                    'recolor': measure(lambda: prepared.color(engine=engine), 15),
                    'lookup': measure(lambda: prepared.color(return_lut=True, engine=engine), 15),
                    'snapshot_bytes': prepared.nbytes,
                }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
