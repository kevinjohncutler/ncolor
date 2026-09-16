"""Paired whole-call timings of the shipped retained-layout switch."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
from release_comparison import cpu_model, fingerprint
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
import ncolor
assert Path(ncolor.__file__).resolve().is_relative_to(a.source.resolve())
solver = ncolor.Engine(n_threads=4)._solver
results = {}
rng = np.random.default_rng(719)
for shape in [(512, 768), (1024, 1024), (96, 96, 96)]:
    image = np.zeros(shape, np.int32)
    for i, center in enumerate(rng.integers(4, min(shape)-4, (32, len(shape))), 1):
        image[tuple(slice(int(c-3), int(c+3)) for c in center)] = i
    for mode in ('standard', 'clean'):
        for weight in range(1, 7):
            samples = {False: [], True: []}
            expected = None
            for rep in range(25):
                for retained in ((False, True) if rep % 2 == 0 else (True, False)):
                    start = time.perf_counter_ns()
                    result, count = solver.label(image, n_colors=32, expand_mode=mode,
                        weight_objective=1, weight_mode=weight, _retain_layout=retained)
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    digest = fingerprint(result)
                    if expected is None: expected = (digest, count)
                    assert expected == (digest, count)
                    if rep >= 4: samples[retained].append(elapsed)
            results[f'{shape}/{mode}/{weight}'] = dict(
                original_ms=statistics.median(samples[False]),
                retained_ms=statistics.median(samples[True]), samples_ms=samples,
                fingerprint=expected[0])
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(cpu=cpu_model(), threads=4,
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
    results=results), indent=2)+'\n')
print(a.output.resolve(), flush=True)
