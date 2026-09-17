"""Alternate isolated builds to measure snapshot preparation and rendering."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from release_comparison import cpu_model, fingerprint
import numpy as np


def worker(source, focus=False, envelope=False, case_filter=None, batch=1, exact=False):
    sys.path.insert(0, str(source.resolve()))
    import ncolor
    assert Path(ncolor.__file__).resolve().is_relative_to(source.resolve())
    engine = ncolor.Engine(n_threads=4)
    cases = {}
    shapes = [(64, 64), (512, 512)] if focus else [(64, 64), (1, 32768), (512, 512), (2048, 2048), (96, 96, 96)]
    if envelope:
        shapes = [(32, 32), (64, 64), (64, 128), (65, 128), (128, 128), (16, 16, 32)]
    fractions = (.01, .3) if envelope else (1.,) if focus else (0., .001, .01, .03, .1, 1.)
    phases = ('expand_standard', 'expand_clean', 'expand_standard_wrap', 'expand_clean_wrap') if envelope else ('prepare', 'render', 'lookup') if focus else ('prepare', 'render')
    for shape_index, shape in enumerate(shapes):
        for fraction_index, fraction in enumerate(fractions):
            for phase in phases:
                cases[f'{phase}/{shape}/{fraction}'] = (shape, fraction, phase, 433+shape_index*6+fraction_index)
    if case_filter:
        cases = {key: value for key, value in cases.items() if any(key == part if exact else part in key for part in case_filter)}
        assert cases, "case filters matched no cases"
    print(json.dumps(list(cases)), flush=True)
    current = None
    snapshot = image = None
    for request in sys.stdin:
        shape, fraction, phase, seed = cases[json.loads(request)]
        key = (shape, fraction)
        if key != current:
            # Retain only the active fixture. Empty-map savings in an earlier
            # case must not change the cache footprint of every later case.
            snapshot = image = None
            seeds = np.zeros(shape, np.int32)
            view = seeds[tuple(slice(min(8, extent-1), None, 32) for extent in shape)]
            view[:] = np.arange(1, view.size+1).reshape(view.shape)
            dense = engine.expand_labels(seeds)
            assert dense.any()
            rng = np.random.default_rng(seed)
            if envelope:
                dense = rng.integers(1, 9, shape, dtype=np.int32)
            image = np.where(rng.random(shape) < fraction, dense, 0).astype(np.int32)
            snapshot = engine.prepare_labels(image)
            current = key
            del seeds, view, dense
        start = time.perf_counter_ns()
        for _ in range(batch):
            if phase == 'prepare':
                result = engine.prepare_labels(image)
            elif phase.startswith('expand_'):
                result = engine.expand_labels(image, p=2, mode='clean' if 'clean' in phase else 'standard',
                                              wrap=phase.endswith('_wrap'))
            else:
                result = snapshot.color(n=32, engine=engine, return_lut=phase == 'lookup')
        elapsed = (time.perf_counter_ns()-start)/1e6/batch
        size = result.nbytes if phase == 'prepare' else None
        # Rendering and hashing a newly prepared snapshot are outside timing.
        pixels = result.color(n=32, engine=engine) if phase == 'prepare' else result
        print(json.dumps(dict(ms=elapsed, fingerprint=fingerprint(pixels),
                              snapshot_bytes=size)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path)
    parser.add_argument('--after', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--settle-ms', type=float, default=2.,
                        help='idle time outside timing so the other process parks its workers')
    parser.add_argument('--batch', type=int, default=1, help='calls per timed sample for steady throughput checks')
    parser.add_argument('--exact', action='store_true', help='match full case names instead of substrings')
    parser.add_argument('--case', action='append', help='include matching case-name substrings (repeatable)')
    parser.add_argument('--envelope', action='store_true', help='measure tiny feature-transform passes around the cutoff')
    parser.add_argument('--focus', action='store_true', help='isolate dense rendering and lookup-only coloring')
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.focus and args.envelope:
        parser.error('focus and envelope select different case sets')
    if args.batch < 1:
        parser.error("batch must be positive")
    if args.worker:
        worker(args.worker, args.focus, args.envelope, args.case, args.batch, args.exact)
        return
    if not (args.before and args.after and args.output):
        parser.error('before, after, and output are required')
    if not np.isfinite(args.settle_ms) or args.settle_ms < 0:
        parser.error('settle-ms must be finite and nonnegative')
    children = []
    try:
        for source in (args.before, args.after):
            children.append(subprocess.Popen([sys.executable, '-u', '-S', __file__,
                '--worker', str(source), '--batch', str(args.batch)]+(['--exact'] if args.exact else [])+(['--focus'] if args.focus else ['--envelope'] if args.envelope else []) + [item for value in args.case or [] for item in ('--case', value)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
        names = [json.loads(child.stdout.readline()) for child in children]
        assert names[0] == names[1]
        results = {}
        for name in names[0]:
            samples = [[], []]
            sizes = [None, None]
            expected = None
            for rep in range(19):
                for index in ((0, 1) if rep % 2 == 0 else (1, 0)):
                    child = children[index]
                    child.stdin.write(json.dumps(name)+'\n')
                    child.stdin.flush()
                    result = json.loads(child.stdout.readline())
                    # Both children share CPU placement. Let the inactive pool
                    # finish spinning before the other child measures a call.
                    time.sleep(args.settle_ms / 1000)
                    if expected is None:
                        expected = result['fingerprint']
                    assert expected == result['fingerprint'], name
                    sizes[index] = result['snapshot_bytes']
                    if rep >= 4:
                        samples[index].append(result['ms'])
            results[name] = dict(before_ms=statistics.median(samples[0]),
                after_ms=statistics.median(samples[1]), samples_ms=samples,
                snapshot_bytes=sizes, fingerprint=expected)
            print(name, round(results[name]['before_ms']/results[name]['after_ms'], 3), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(cpu=cpu_model(), threads=4, batch=args.batch, settle_ms=args.settle_ms, case_isolation=True, envelope_probe=args.envelope,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            results=results), indent=2)+'\n')
        print(args.output.resolve(), flush=True)
    finally:
        for child in children:
            child.stdin.close()
        for child in children:
            child.wait(timeout=60)


if __name__ == '__main__':
    main()
