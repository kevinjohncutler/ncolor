"""Check retained memory for prepared coloring and thin components."""
import argparse
import gc
import json
from pathlib import Path
import resource
import sys

import numpy as np
import psutil
import ncolor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    engine = ncolor.Engine(n_threads=4)
    rng = np.random.default_rng(764)
    mask = rng.random((2, 1024, 1024)) < 0.2
    labels = np.zeros((1024, 1024), np.int32)
    seeds = labels[8::32, 8::32]
    seeds[:] = np.arange(1, seeds.size + 1).reshape(seeds.shape)
    snapshot = engine.prepare_labels(labels, weight_objective=1)
    expected = engine.label(labels, weight_objective=1)
    process = psutil.Process()
    results = {}
    for case in ('recolor', 'rebuild', 'components'):
        history = []
        for _ in range(100):
            if case == 'components':
                result, count = engine.connected_components(mask)
                assert count > 0
                np.testing.assert_array_equal(result != 0, mask)
            else:
                current = snapshot if case == 'recolor' else engine.prepare_labels(labels, weight_objective=1)
                result = current.color(engine=engine)
                np.testing.assert_array_equal(result, expected)
                del current
            del result
            gc.collect()
            history.append(process.memory_info().rss)
        results[case] = dict(resident_bytes=history,
                             final_window_range=max(history[-20:]) - min(history[-20:]))
    results['snapshot_bytes'] = snapshot.nbytes
    results['peak_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(args.output.resolve())
    for case in ('recolor', 'rebuild', 'components'):
        assert results[case]['final_window_range'] < 8 * 1024**2, results[case]


if __name__ == '__main__':
    main()
