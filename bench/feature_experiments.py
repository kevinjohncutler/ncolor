"""Fixed feature/contact and component workloads, with output fingerprints.

Run separately against a saved baseline and the current checkout, setting
PYTHONPATH to select the package. Compare fingerprints before timings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import ncolor
from ncolor._backend import _impl


def fingerprint(value):
    digest = hashlib.sha256()
    for item in value if isinstance(value, tuple) else (value,):
        array = np.asarray(item)
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def measure(call, runs):
    result = call()
    expected = fingerprint(result)
    call()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    assert fingerprint(result) == expected
    return {'median_ms': statistics.median(samples), 'samples_ms': samples,
            'fingerprint': expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=9)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(1926)
    engine = ncolor.Engine(n_threads=args.threads)
    results = {}
    for shape in [(512, 512), (2048, 2048), (96, 96, 96)]:
        name = 'x'.join(map(str, shape))
        image = np.zeros(shape, np.int32)
        slices = (slice(8, None, 32),) * len(shape)
        count = image[slices].size
        image[slices] = np.arange(1, count + 1).reshape(image[slices].shape)
        for mode in ('standard', 'clean'):
            results[f'expand_{mode}_{name}'] = measure(
                lambda: engine.expand_labels(image, mode=mode), args.runs)
            results[f'label_{mode}_{name}'] = measure(
                lambda: engine.label(image, expand_mode=mode), args.runs)
        for density in (.1, .7):
            mask = (rng.random(shape) < density).astype(np.uint8)
            cc = (lambda: engine.connected_components(mask, conn=2)) if hasattr(
                engine, 'connected_components') else (lambda: ncolor.connected_components(mask, conn=2))
            results[f'components_{density}_{name}'] = measure(cc, args.runs)
        cells = engine.expand_labels(image)
        results[f'per_label_{name}'] = measure(
            lambda: _impl.cc_label_per_label(cells, conn=len(shape)), args.runs)
        results[f'format_clean_{name}'] = measure(
            lambda: engine.format_labels(cells, clean=True), args.runs)
        print(name, 'complete', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'threads': args.threads, 'results': results}, indent=2) + '\n')
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
