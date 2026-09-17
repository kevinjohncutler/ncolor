"""Throughput of four threads calling ncolor.label, as shown in the README.

Each process times one arrangement, chosen by NCOLOR_MAX_ENGINES at process
start: unset lets overlapping calls use narrower engines, 1 makes them take
turns on one full-width pool. ``compare`` alternates the two in fresh
processes and reports the median ratio per image size.

    python bench/concurrent_callers.py compare --sizes 512 1024 2048 4096 --rounds 5
"""
import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

CALLERS = 4
IMAGES_PER_CALLER = 4
REPEATS = 5


def cells(size, seed):
    """Square cells of side 6 to 23 at a density that does not depend on size."""
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), np.int32)
    count = size * size // 900
    corners = rng.integers(0, size - 24, (count, 2))
    sides = rng.integers(6, 24, count)
    for label, ((y, x), side) in enumerate(zip(corners, sides), 1):
        image[y:y + side, x:x + side] = label
    return image


def images_for(size):
    """Distinct images, so callers do real, varied work."""
    return [cells(size, seed) for seed in range(CALLERS * IMAGES_PER_CALLER)]


def measure(size):
    import ncolor
    images = images_for(size)
    with cf.ThreadPoolExecutor(CALLERS) as pool:
        list(pool.map(ncolor.label, images))
        list(pool.map(ncolor.label, images))
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            list(pool.map(ncolor.label, images))
            times.append(time.perf_counter() - start)
    print(json.dumps(dict(size=size, median_s=statistics.median(times))), flush=True)


def compare(sizes, rounds):
    arrangements = {"engines": {}, "turns": {"NCOLOR_MAX_ENGINES": "1"}}
    results = {size: {name: [] for name in arrangements} for size in sizes}
    for round_index in range(rounds):
        order = list(arrangements) if round_index % 2 == 0 else list(reversed(arrangements))
        for size in sizes:
            for name in order:
                env = dict(os.environ)
                env.pop("NCOLOR_MAX_ENGINES", None)
                env.update(arrangements[name])
                out = subprocess.run([sys.executable, __file__, "measure", "--size", str(size)],
                                     env=env, capture_output=True, text=True, check=True)
                results[size][name].append(json.loads(out.stdout.strip().splitlines()[-1])["median_s"])
    summary = {}
    for size in sizes:
        engines = statistics.median(results[size]["engines"])
        turns = statistics.median(results[size]["turns"])
        summary[size] = dict(engines_ms=engines * 1e3, turns_ms=turns * 1e3, speedup=turns / engines)
        print(f"{size:5d}  engines {engines * 1e3:9.1f} ms  turns {turns * 1e3:9.1f} ms  "
              f"speedup {turns / engines:.2f}x", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("measure")
    one.add_argument("--size", type=int, required=True)
    both = sub.add_parser("compare")
    both.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    both.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.size)
    else:
        compare(args.sizes, args.rounds)


if __name__ == "__main__":
    main()
