"""End-to-end formatting check for label-presence strategies."""
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
    rng = np.random.default_rng(1926)
    results = {}
    for threads in (1, 4):
        engine = ncolor.Engine(n_threads=threads)
        for domain in (32, 4096, 65536):
            for dense in (False, True):
                image = rng.integers(0, domain // (1 if dense else 2), (2048, 2048), dtype=np.int32)
                if not dense:
                    image *= 2
                key = f'{threads}_threads_{domain}_labels_dense_{dense}'
                results[key] = measure(lambda: engine.format_labels(image), 21)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
