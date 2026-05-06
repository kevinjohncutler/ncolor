"""End-to-end cpp vs numba comparison: 2D + 3D, multiple dtypes.

Runs the cpp pipeline in this process (default backend) and the numba
pipeline in a subprocess with NCOLOR_BACKEND=numba. Prints best & median
ms for each, then the speedup ratio.

Designed to run on any machine where the repo is checked out — resolves
its sibling ``src/`` directory automatically.
"""
import os
import sys
import time
import statistics as st
import subprocess
import textwrap

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "src")
sys.path.insert(0, _SRC)


def make_2d(H, W, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=dtype)
    for i in range(1, n + 1):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = int(rng.integers(8, max(16, H // 30)))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def make_3d(D, H, W, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((D, H, W), dtype=dtype)
    for i in range(1, n + 1):
        cz = rng.integers(D // 8, D - D // 8)
        cy = rng.integers(H // 8, H - H // 8)
        cx = rng.integers(W // 8, W - W // 8)
        r = int(rng.integers(2, max(4, D // 8)))
        zz, yy, xx = np.ogrid[:D, :H, :W]
        m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def bench(fn, runs=20, warmup=4):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return min(times), st.median(times)


import ncolor
print(f"backend module: {ncolor.label.__module__}")

shapes_2d = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
shapes_3d = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
dtypes = [np.uint16, np.int32]

# cpp times
cpp = {}
print(f"\n{'shape':<16} {'dtype':<8} {'cpp best':>10} {'cpp med':>10}")
print("-" * 50)
for H, W in shapes_2d:
    for dt in dtypes:
        m = make_2d(H, W, max(20, H * W // 8000), dt)
        b, med = bench(lambda: ncolor.label(m, return_n=True))
        cpp[("2d", H, W, dt.__name__)] = (b, med)
        print(f"2D {H}x{W:<10} {dt.__name__:<8} {b:>9.2f}  {med:>9.2f}")
for D, H, W in shapes_3d:
    for dt in dtypes:
        m = make_3d(D, H, W, max(20, D * H * W // 8000), dt)
        b, med = bench(lambda: ncolor.label(m, return_n=True))
        cpp[("3d", D, H, W, dt.__name__)] = (b, med)
        print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} {b:>9.2f}  {med:>9.2f}")


# Numba in a subprocess (one process per shape so each starts with a
# warm Numba cache). Done in subprocess because cpp is the default
# in-process backend; flipping NCOLOR_BACKEND only affects fresh imports.
SCRIPT = textwrap.dedent(
    f"""
    import os, sys, time, statistics as st
    sys.path.insert(0, {_SRC!r})
    import numpy as np
    import ncolor

    def make_2d(H, W, n, dtype, seed=0):
        rng = np.random.default_rng(seed)
        m = np.zeros((H, W), dtype=dtype)
        for i in range(1, n + 1):
            cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
            r = int(rng.integers(8, max(16, H // 30)))
            yy, xx = np.ogrid[:H, :W]
            m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
        return m

    def make_3d(D, H, W, n, dtype, seed=0):
        rng = np.random.default_rng(seed)
        m = np.zeros((D, H, W), dtype=dtype)
        for i in range(1, n + 1):
            cz = rng.integers(D // 8, D - D // 8)
            cy = rng.integers(H // 8, H - H // 8)
            cx = rng.integers(W // 8, W - W // 8)
            r = int(rng.integers(2, max(4, D // 8)))
            zz, yy, xx = np.ogrid[:D, :H, :W]
            m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
        return m

    def bench(fn, runs=20, warmup=4):
        for _ in range(warmup): fn()
        ts = []
        for _ in range(runs):
            t = time.perf_counter(); fn()
            ts.append((time.perf_counter() - t) * 1000)
        return min(ts), st.median(ts)

    kind = sys.argv[1]
    dt = getattr(np, sys.argv[-1])
    if kind == '2d':
        H, W = int(sys.argv[2]), int(sys.argv[3])
        m = make_2d(H, W, max(20, H * W // 8000), dt)
    else:
        D, H, W = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        m = make_3d(D, H, W, max(20, D * H * W // 8000), dt)
    b, med = bench(lambda: ncolor.label(m, return_n=True))
    print(f'{{med:.2f}} {{b:.2f}}')
    """
)

env = {**os.environ, "NCOLOR_BACKEND": "numba"}
nb = {}
print(f"\n{'shape':<16} {'dtype':<8} {'numba best':>11} {'numba med':>11}")
print("-" * 50)


def run_numba(args, key):
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT, *args],
        capture_output=True, text=True, env=env, timeout=300,
    )
    if p.returncode != 0 or not p.stdout.strip():
        print(f"FAIL: {key}: {p.stderr[:200].replace(chr(10), ' ')}")
        return
    med, b = map(float, p.stdout.strip().splitlines()[-1].split())
    nb[key] = (b, med)


for H, W in shapes_2d:
    for dt in dtypes:
        key = ("2d", H, W, dt.__name__)
        run_numba(["2d", str(H), str(W), dt.__name__], key)
        if key in nb:
            b, med = nb[key]
            print(f"2D {H}x{W:<10} {dt.__name__:<8} {b:>10.2f}  {med:>10.2f}")
for D, H, W in shapes_3d:
    for dt in dtypes:
        key = ("3d", D, H, W, dt.__name__)
        run_numba(["3d", str(D), str(H), str(W), dt.__name__], key)
        if key in nb:
            b, med = nb[key]
            print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} {b:>10.2f}  {med:>10.2f}")


print(f"\n{'shape':<16} {'dtype':<8} {'cpp med':>10} {'numba med':>11} {'speedup':>10}")
print("-" * 60)
for H, W in shapes_2d:
    for dt in dtypes:
        key = ("2d", H, W, dt.__name__)
        if key in nb:
            cpp_med, nb_med = cpp[key][1], nb[key][1]
            sp = nb_med / cpp_med
            print(f"2D {H}x{W:<10} {dt.__name__:<8} {cpp_med:>9.2f}  {nb_med:>10.2f}  {sp:>8.2f}×")
for D, H, W in shapes_3d:
    for dt in dtypes:
        key = ("3d", D, H, W, dt.__name__)
        if key in nb:
            cpp_med, nb_med = cpp[key][1], nb[key][1]
            sp = nb_med / cpp_med
            print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} {cpp_med:>9.2f}  {nb_med:>10.2f}  {sp:>8.2f}×")
