"""Alternate calls between two warmed, isolated builds to recheck noisy cases."""
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


def worker(source, sparse=False, render=False):
    sys.path.insert(0, str(source.resolve()))
    import ncolor
    assert Path(ncolor.__file__).resolve().is_relative_to(source.resolve())
    engine = ncolor.Engine(n_threads=4)
    rng = np.random.default_rng(851)
    calls = {}
    for shape in ([] if render else [(2, 513, 517), (1024, 1024), (96, 96, 96)]):
        for density in ((.01,) if sparse else (.1, .7, 1.)):
            mask = rng.random(shape) < density
            for conn in (1, len(shape)):
                calls[f'components/{shape}/{density}/{conn}'] = (
                    lambda mask=mask, conn=conn: engine.connected_components(mask, conn=conn)[0])
    for shape in ([] if render else [(1024, 1024), (2, 513, 517)]):
        mask = np.indices(shape).sum(axis=0) % 2 == 0
        calls[f'components/{shape}/checkerboard/{len(shape)}'] = (
            lambda mask=mask: engine.connected_components(mask, conn=mask.ndim)[0])
    for shape in ([] if sparse or render else [(512, 768), (1024, 1024)]):
        image = np.zeros(shape, np.int32)
        for i, center in enumerate(rng.integers(4, min(shape)-4, (32, len(shape))), 1):
            image[tuple(slice(int(c-3), int(c+3)) for c in center)] = i
        for mode in ('standard', 'clean'):
            for weight in ('min', 'count', 'mean_inv'):
                calls[f'weighted/{shape}/{mode}/{weight}'] = (
                    lambda image=image, mode=mode, weight=weight: engine.label(
                        image, n=32, weight_objective=1, weight_mode=weight, expand_mode=mode))
    if render:
        for shape in [(512, 512), (2048, 2048), (96, 96, 96)]:
            image = np.zeros(shape, np.int32)
            seeds = image[(slice(8, None, 32),)*len(shape)]
            seeds[:] = np.arange(1, seeds.size+1).reshape(seeds.shape)
            dense = engine.expand_labels(image)
            for fraction in (.001, .01, .1, 1.):
                image = np.where(rng.random(shape) < fraction, dense, 0).astype(np.int32)
                prepared = engine.prepare_labels(image)
                calls[f'prepared/{shape}/{fraction}'] = (
                    lambda prepared=prepared: prepared.color(n=32, engine=engine))
                calls[f'prepare/{shape}/{fraction}'] = (
                    lambda image=image: np.asarray([engine.prepare_labels(image).n_labels], np.int32))
    print(json.dumps(list(calls)), flush=True)
    for request in sys.stdin:
        call = calls[json.loads(request)]
        start = time.perf_counter_ns()
        result = call()
        elapsed = (time.perf_counter_ns()-start)/1e6
        print(json.dumps(dict(ms=elapsed, fingerprint=fingerprint(result))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before', type=Path)
    p.add_argument('--after', type=Path)
    p.add_argument('--output', type=Path)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--render', action='store_true', help='recheck prepared rendering and preparation')
    modes.add_argument('--sparse', action='store_true', help='recheck one-percent foreground masks')
    p.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker:
        worker(a.worker, a.sparse, a.render)
        return
    if not (a.before and a.after and a.output): p.error('before, after, and output are required')
    children = []
    try:
        for source in (a.before, a.after):
            children.append(subprocess.Popen([sys.executable, '-u', '-S', __file__,
                '--worker', str(source)]+(['--sparse'] if a.sparse else ['--render'] if a.render else []), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True))
        names = [json.loads(child.stdout.readline()) for child in children]
        assert names[0] == names[1]
        results = {}
        for name in names[0]:
            samples = [[], []]
            expected = None
            for rep in range(25):
                for index in ((0, 1) if rep % 2 == 0 else (1, 0)):
                    child = children[index]
                    child.stdin.write(json.dumps(name)+'\n')
                    child.stdin.flush()
                    result = json.loads(child.stdout.readline())
                    if expected is None: expected = result['fingerprint']
                    assert expected == result['fingerprint'], name
                    if rep >= 4: samples[index].append(result['ms'])
            results[name] = dict(before_ms=statistics.median(samples[0]),
                after_ms=statistics.median(samples[1]), samples_ms=samples, fingerprint=expected)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(dict(cpu=cpu_model(), threads=4, sparse_probe=a.sparse, render_probe=a.render,
            affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            results=results), indent=2)+'\n')
        print(a.output.resolve(), flush=True)
    finally:
        for child in children:
            child.stdin.close()
            child.wait(timeout=60)


if __name__ == '__main__':
    main()
