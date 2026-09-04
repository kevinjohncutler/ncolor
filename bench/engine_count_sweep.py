"""Throughput of 4 concurrent callers against the number of engines.

The library's choice is not really split-or-not: it is how many engines
overlapping callers share. k=1 is taking turns on one full-width pool,
which is what ncolor has always done. Threads per engine are W//k
throughout, so every arrangement uses about one machine's worth.
"""
import os, statistics, sys, threading, time
import numpy as np
import ncolor
from ncolor._backend import _smt

CALLERS, REPS, ROUNDS = 4, 2, 5
W = _smt.auto_threads()
size = int(os.environ.get("NCOLOR_SWEEP_SIZE", "1024"))
mask = _smt._make_calibration_mask(size)
host = os.uname().nodename.split(".")[0]
try:
    load = open("/proc/loadavg").read().split()[0]
except OSError:
    load = os.popen("sysctl -n vm.loadavg").read().strip().strip("{} ").split()[0]

def throughput(k):
    """Median wall time for CALLERS threads to each do REPS labels."""
    engines = [ncolor.Engine(n_threads=max(1, W // k)) for _ in range(k)]
    for e in engines:
        e.label(mask)

    def run():
        ts = [threading.Thread(
            target=lambda e=engines[i % k]: [e.label(mask) for _ in range(REPS)])
            for i in range(CALLERS)]
        for t in ts: t.start()
        for t in ts: t.join()

    run(); run()
    times = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter(); run(); times.append(time.perf_counter() - t0)
    for e in engines:
        e.release_buffers()
    return statistics.median(times)

results = {}
for k in (1, 2, 4):
    results[k] = throughput(k)
base = results[1]
cells = "  ".join(f"k={k}: {t*1e3:6.1f} ms ({base/t:4.2f}x)" for k, t in results.items())
best = min(results, key=results.get)
print(f"{host:13s} W={W:3d} n={size} load={load:>5s}  {cells}   best k={best}", flush=True)
