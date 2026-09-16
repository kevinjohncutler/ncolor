"""Compare dtype-bounded presence detection through public operations."""
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
    rng = np.random.default_rng(503)
    results = {}
    for threads in (1, 4):
        engine = ncolor.Engine(n_threads=threads)
        for shape in [(512, 512), (2048, 2048)]:
            for dtype in (np.bool_, np.uint8, np.int8, np.uint16, np.int16, np.int32):
                low = -16 if np.issubdtype(dtype, np.signedinteger) else 0
                high = 2 if dtype == np.bool_ else 32
                image = rng.integers(low, high, shape, dtype=np.int32).astype(dtype)
                key = f'format_{threads}_{shape}_{np.dtype(dtype).name}'
                results[key] = measure(lambda: engine.format_labels(image), 15)
            for dtype in (np.uint8, np.int8):
                for pattern in ('full', 'sparse'):
                    if pattern == 'full':
                        image = rng.integers(0, 256, shape, dtype=np.uint8).view(dtype)
                    else:
                        image = (rng.integers(0, 2, shape, dtype=np.uint8) * 127).view(dtype)
                    key = f'format_{threads}_{shape}_{np.dtype(dtype).name}_{pattern}'
                    results[key] = measure(lambda: engine.format_labels(image), 15)
            for dtype in (np.uint8, np.uint16, np.int32):
                image = np.zeros(shape, dtype=dtype)
                stride = max(1, shape[0] // (15 if dtype == np.uint8 else 32))
                seeds = image[4::stride, 4::stride]
                seeds[:] = np.arange(1, seeds.size + 1).reshape(seeds.shape)
                for mode in ('standard', 'clean'):
                    key = f'label_{threads}_{shape}_{np.dtype(dtype).name}_{mode}'
                    results[key] = measure(lambda: engine.label(image, expand_mode=mode), 9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
