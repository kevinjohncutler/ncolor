"""Isolated peak and retained-memory checks for component labeling."""
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
    parser.add_argument('--case', choices=['sparse', 'dense', 'checkerboard'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 40:
        parser.error('--iterations must be at least 40 to separate warmup and measurement windows')
    rng = np.random.default_rng(812)
    if args.case == 'checkerboard':
        axis = np.arange(2048, dtype=np.uint8) % 2
        mask = axis[:, None] ^ axis[None, :]
    else:
        mask = (rng.integers(0, 100, (2048, 2048), dtype=np.uint8)
                < (10 if args.case == 'sparse' else 70))
    engine = ncolor.Engine(n_threads=4)
    call = engine.connected_components if hasattr(engine, 'connected_components') else ncolor.connected_components
    process = psutil.Process()
    gc.collect()
    initial = process.memory_info().rss
    samples = []
    for _ in range(args.iterations):
        out, count = call(mask, conn=1)
        if args.case == 'checkerboard':
            assert count == mask.size // 2
        assert np.array_equal(out != 0, mask != 0)
        del out
        gc.collect()
        samples.append(process.memory_info().rss)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != 'darwin':
        peak *= 1024
    result = {'case': args.case, 'initial_resident_bytes': initial,
              'peak_resident_bytes': peak, 'retained_resident_bytes': samples,
              'post_warm_range_bytes': max(samples[-20:]) - min(samples[-20:]),
              'early_range_bytes': max(samples[4:20]) - min(samples[4:20]),
              'iterations': args.iterations, 'measurement_window': 20,
              'component_count': count, 'threads': process.num_threads()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(args.output.resolve(), flush=True)
    assert result['post_warm_range_bytes'] < 8 * 1024 ** 2, result


if __name__ == '__main__':
    main()
