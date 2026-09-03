"""A/B benchmark for optional worker pinning (NCOLOR_PIN_THREADS).

Run once with the env var unset and once with it =1; compare. Pinning is
read once per process, so the two regimes must be separate invocations.
"""
import os, sys, time
import numpy as np
from ncolor._backend import Solver


def make_mask(H, seed=0, ndim=2):
    rng = np.random.default_rng(seed)
    if ndim == 2:
        mask = np.zeros((H, H), np.int32)
        n = max(50, H * H // 4000)
        c = rng.integers(10, H - 10, size=(n, 2))
        r = rng.integers(5, max(10, H // 40), size=n)
        for i, ((y, x), rr) in enumerate(zip(c, r), 1):
            mask[max(0, y - rr):y + rr, max(0, x - rr):x + rr] = i
        return mask
    mask = np.zeros((H, H, H), np.int32)
    n = max(50, H ** 3 // 8000)
    c = rng.integers(5, H - 5, size=(n, 3))
    r = rng.integers(3, max(6, H // 20), size=n)
    for i, ((z, y, x), rr) in enumerate(zip(c, r), 1):
        mask[max(0, z - rr):z + rr, max(0, y - rr):y + rr, max(0, x - rr):x + rr] = i
    return mask


def bench(sv, mask, warmup=3, iters=15):
    for _ in range(warmup):
        sv.label(mask)
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); sv.label(mask); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3, float(np.median(ts)) * 1e3


pin = os.environ.get("NCOLOR_PIN_THREADS", "") or "0"
cases = [("2D-1024", 1024, 2), ("2D-2048", 2048, 2),
         ("2D-4096", 4096, 2), ("3D-256", 256, 3)]
masks = {name: make_mask(H, ndim=nd) for name, H, nd in cases}
Ts = [int(a) for a in sys.argv[1:]] or [64, 32]
for T in Ts:
    sv = Solver(T)
    for name, _, _ in cases:
        try:
            mn, md = bench(sv, masks[name])
            print(f"PIN={pin} T={sv.n_threads} {name:8s}: min={mn:7.2f}ms med={md:7.2f}ms", flush=True)
        except Exception as e:
            print(f"PIN={pin} T={sv.n_threads} {name:8s}: ERROR {e}", flush=True)
    del sv
