"""Micro-bench just the connect step. Used to isolate scan_band_unpadded
perf from the rest of the ncolor.label pipeline."""
import time
import statistics as st
import numpy as np
import ncolor
from ncolor._backend import Solver


def make_2d(H, W, n, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=np.uint16)
    for i in range(1, n + 1):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = int(rng.integers(8, max(16, H // 30)))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def make_3d(D, H, W, n, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((D, H, W), dtype=np.uint16)
    for i in range(1, n + 1):
        cz, cy, cx = (rng.integers(8, D - 8),
                      rng.integers(8, H - 8),
                      rng.integers(8, W - 8))
        r = int(rng.integers(2, max(4, D // 8)))
        zz, yy, xx = np.ogrid[:D, :H, :W]
        m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def bench(fn, runs=30, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), st.median(ts), st.stdev(ts)


s = Solver(-1)
shapes = [
    ("2D 1024²",  make_2d(1024, 1024, 100)),
    ("2D 2048²",  make_2d(2048, 2048, 200)),
    ("2D 4096²",  make_2d(4096, 4096, 600)),
    ("3D 64³",    make_3d(64, 64, 64, 30)),
    ("3D 128³",   make_3d(128, 128, 128, 50)),
    ("3D 256³",   make_3d(256, 256, 256, 200)),
]

# Fully expand first so connect operates on a dense (no zeros) image.
expanded = [(name, ncolor.label(m, p=1, expand=True)) for name, m in shapes]

print(f"{'shape':<14} {'connect best':>14} {'med':>8} {'stdev':>8}")
for name, m in expanded:
    best, med, std = bench(lambda m=m: s.connect(m, conn=2))
    print(f"  {name:<12} {best:>13.3f}  {med:>7.3f}  {std:>7.3f}")
