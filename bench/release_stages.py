"""Diagnose release regressions using the shipped coloring stage timers.

Run after the timing suite, with no competing builds or benchmarks.
These instrumented timings explain whole-call changes; they do not replace
uninstrumented release measurements.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import statistics
import sys

from release_comparison import cpu_model, fingerprint
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--case', action='append', help='limit profiling to named corpus entries')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import ncolor
    assert Path(ncolor.__file__).resolve().is_relative_to(args.source.resolve())
    engine = ncolor.Engine(n_threads=args.threads)
    inputs = np.load(args.corpus)
    if args.case and any(name not in inputs.files for name in args.case):
        parser.error('--case must name an entry in the corpus')
    results = {}
    for name in inputs.files:
        if not name.startswith('labels_') or (args.case and name not in args.case):
            continue
        image = inputs[name]
        for mode in ('default', 'matched'):
            options = {} if mode == 'default' else dict(conn=1, format_input=False,
                expand_mode='standard', soft_conn=0, soft_radius=0)
            samples = {}
            expected = fingerprint(engine.label(image, **options))
            for iteration in range(25):
                with contextlib.redirect_stderr(io.StringIO()):
                    result = engine.label(image, verbose=True, **options)
                assert fingerprint(result) == expected
                if iteration >= 4:
                    for stage, ms in engine._solver.get_last_stages():
                        samples.setdefault(stage, []).append(ms)
            results[name+'/'+mode] = dict(fingerprint=expected, samples_ms=samples,
                median_ms={stage:statistics.median(values) for stage, values in samples.items()})
    output = dict(version=args.version, threads=args.threads, cpu=cpu_model(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                  results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(args.output.resolve(), flush=True)


if __name__ == '__main__':
    main()
