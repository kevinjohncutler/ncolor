"""Compare component partitioning on thin volumes with exact fingerprints."""
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
    rng = np.random.default_rng(920)
    results = {}
    for threads in (1, 4):
        engine = ncolor.Engine(n_threads=threads)
        for shape in [(1024, 1024), (2, 1024, 1024), (2, 2, 512, 512),
                      (2, 3, 128, 1024), (3, 512, 512)]:
            for density in (.1, .7):
                data = rng.random(shape) < density
                for conn in (1, len(shape)):
                    name = f'{threads}_{shape}_{density}_{conn}'
                    results[name] = measure(
                        lambda: engine.connected_components(data, conn=conn), 11)
                cells = rng.integers(0, 5, shape, dtype=np.int32)
                results[f'{threads}_{shape}_{density}_per_label'] = measure(
                    lambda: engine._expand.components_per_label(cells, conn=1), 11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
