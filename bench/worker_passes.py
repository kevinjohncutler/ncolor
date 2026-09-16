"""Measure whole coloring calls around serial memory-pass thresholds."""
import argparse
import json
import os
from pathlib import Path
import sys
from release_comparison import cpu_model, fingerprint, timed
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--threads', type=int, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
import ncolor
assert Path(ncolor.__file__).resolve().is_relative_to(a.source.resolve())
engine = ncolor.Engine(n_threads=a.threads)
results = {}
for shape in [(64, 128), (128, 256), (256, 256), (256, 512)]:
    image = np.zeros(shape, np.int32)
    seeds = image[7::32, 7::32]
    seeds[:] = np.arange(1, seeds.size + 1).reshape(seeds.shape)
    for formatted in (False, True):
        call = lambda: engine.label(image, n=16, format_input=formatted)
        expected = call()
        def check(actual):
            np.testing.assert_array_equal(actual, expected)
            return dict(fingerprint=fingerprint(actual))
        results[f'{shape}/{formatted}'] = timed(call, check, 51)
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(cpu=cpu_model(), threads=a.threads,
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    results=results), indent=2)+'\n')
print(a.output.resolve(), flush=True)
