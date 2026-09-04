"""Does slicing thin hurt because the machine is small, or because the
engine is? Total width is held at W throughout; only the number of
slices changes. Each engine has its own caller, so throughput is images
per second and the arrangements are directly comparable.
"""
import os, statistics, threading, time
import ncolor
from ncolor._backend import _smt

REPS, ROUNDS = 3, 5
W = _smt.auto_threads()
size = int(os.environ.get("NCOLOR_SWEEP_SIZE", "1024"))
mask = _smt._make_calibration_mask(size)
host = os.uname().nodename.split(".")[0]

def rate(k):
    per = max(1, W // k)
    engines = [ncolor.Engine(n_threads=per) for _ in range(k)]
    for e in engines:
        e.label(mask)

    def run():
        ts = [threading.Thread(target=lambda e=e: [e.label(mask) for _ in range(REPS)])
              for e in engines]
        for t in ts: t.start()
        for t in ts: t.join()

    run(); run()
    times = [(lambda t0: (run(), time.perf_counter() - t0)[1])(time.perf_counter())
             for _ in range(ROUNDS)]
    for e in engines:
        e.release_buffers()
    return per, (k * REPS) / statistics.median(times)

ks = [k for k in (1, 2, 4, 8, 16, 32) if W // k >= 1]
base = None
out = []
for k in ks:
    per, r = rate(k)
    if base is None:
        base = r
    out.append(f"{per:2d}thr x{k:<2d}: {r:6.0f} img/s ({r/base:4.2f}x)")
print(f"{host:13s} W={W:3d} n={size}  " + "  ".join(out), flush=True)
