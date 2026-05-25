"""Targeted benchmark for the L2 expand step alone.

Isolates the parabolic-envelope sweep cost from the surrounding ncolor.label
pipeline so we can iterate on Phase-1 (envelope build) optimisations and
measure the impact directly. Both 2D and 3D shapes; uint16 input.
"""
import os
import sys
import time
import statistics as st
from pathlib import Path
# Script-relative src path so the bench runs from any checkout location.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import ncolor
from ncolor._backend import ExpandEngine


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
        cz, cy, cx = (
            rng.integers(8, D - 8),
            rng.integers(8, H - 8),
            rng.integers(8, W - 8),
        )
        r = int(rng.integers(2, max(4, D // 8)))
        zz, yy, xx = np.ogrid[:D, :H, :W]
        m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def bench(fn, runs=20, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return min(ts), st.median(ts), st.stdev(ts)


eng = ExpandEngine(-1)
shapes = [
    ("2D 1024²",  make_2d(1024, 1024, 100)),
    ("2D 2048²",  make_2d(2048, 2048, 200)),
    ("2D 4096²",  make_2d(4096, 4096, 600)),
    ("3D 64³",    make_3d(64, 64, 64, 30)),
    ("3D 128³",   make_3d(128, 128, 128, 50)),
    ("3D 256³",   make_3d(256, 256, 256, 200)),
]
print(f"{'shape':<14} {'L2 best':>10} {'L2 med':>10} {'L2 stdev':>10}")
for name, m in shapes:
    arr = m.astype(np.int32)
    best, med, std = bench(lambda arr=arr: eng.expand_labels(arr, p=2))
    print(f"  {name:<12} {best:>9.2f}  {med:>9.2f}  {std:>9.2f}")
