"""cpp vs numba comparison for both L1 and L2 expand metrics.

For each shape × dtype, times four pipelines:
  - cpp p=2 (Felzenszwalb parabolic envelope, default)
  - cpp p=1 (Saito-Toriwaki separable sweep)
  - numba L2 (numba._numba_legacy.color.label, default)
  - numba L1 (expand_labels(metric='l1') + label(expand=False))

Also reports whether cpp p=1 and cpp p=2 produce identical *colored*
outputs on the test inputs (they differ in Voronoi boundary placement
at corners, which can change the adjacency graph).
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
from ncolor.color import _get_solver


def cpp_label(m, p):
    """Direct cpp call with explicit p."""
    return _get_solver().label(np.asarray(m), p=int(p))[0]


shapes_2d = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
shapes_3d = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
dtypes = [np.uint16, np.int32]

# --- cpp times (L2 + L1) ---
cpp = {}
print(f"{'shape':<16} {'dtype':<8} {'cpp L2 med':>11} {'cpp L1 med':>11}")
print("-" * 50)
for H, W in shapes_2d:
    for dt in dtypes:
        m = make_2d(H, W, max(20, H * W // 8000), dt)
        _, m_l2 = bench(lambda: cpp_label(m, 2))
        _, m_l1 = bench(lambda: cpp_label(m, 1))
        cpp[("2d", H, W, dt.__name__)] = (m_l2, m_l1)
        print(f"2D {H}x{W:<10} {dt.__name__:<8} {m_l2:>10.2f}  {m_l1:>10.2f}")
for D, H, W in shapes_3d:
    for dt in dtypes:
        m = make_3d(D, H, W, max(20, D * H * W // 8000), dt)
        _, m_l2 = bench(lambda: cpp_label(m, 2))
        _, m_l1 = bench(lambda: cpp_label(m, 1))
        cpp[("3d", D, H, W, dt.__name__)] = (m_l2, m_l1)
        print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} {m_l2:>10.2f}  {m_l1:>10.2f}")


# --- numba in subprocess (L2 default, L1 = expand_labels(metric='l1') + label(expand=False)) ---
SCRIPT = textwrap.dedent(
    f"""
    import os, sys, time, statistics as st
    sys.path.insert(0, {_SRC!r})
    import numpy as np
    import ncolor
    from ncolor._numba_legacy.expand import expand_labels as nb_expand
    from ncolor._numba_legacy.color import label as nb_label

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

    kind, metric = sys.argv[1], sys.argv[2]
    dt = getattr(np, sys.argv[-1])
    if kind == '2d':
        H, W = int(sys.argv[3]), int(sys.argv[4])
        m = make_2d(H, W, max(20, H * W // 8000), dt)
    else:
        D, H, W = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        m = make_3d(D, H, W, max(20, D * H * W // 8000), dt)

    if metric == 'l2':
        fn = lambda: nb_label(m, return_n=True)
    else:  # 'l1' — expand with l1 metric, then run rest of pipeline
        def fn():
            exp = nb_expand(m, metric='l1')
            return nb_label(exp, expand=False, return_n=True)

    b, med = bench(fn)
    print(f'{{med:.2f}} {{b:.2f}}')
    """
)

env = {**os.environ, "NCOLOR_BACKEND": "numba"}
nb = {}
print(f"\n{'shape':<16} {'dtype':<8} {'numba L2 med':>13} {'numba L1 med':>13}")
print("-" * 56)


def run_numba(args, key):
    p = subprocess.run(
        [sys.executable, "-c", SCRIPT, *args],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if p.returncode != 0 or not p.stdout.strip():
        return None
    med, b = map(float, p.stdout.strip().splitlines()[-1].split())
    return (b, med)


for H, W in shapes_2d:
    for dt in dtypes:
        key = ("2d", H, W, dt.__name__)
        l2 = run_numba(["2d", "l2", str(H), str(W), dt.__name__], key)
        l1 = run_numba(["2d", "l1", str(H), str(W), dt.__name__], key)
        if l2 and l1:
            nb[key] = (l2[1], l1[1])
            print(f"2D {H}x{W:<10} {dt.__name__:<8} {l2[1]:>12.2f}  {l1[1]:>12.2f}")
        else:
            print(f"2D {H}x{W:<10} {dt.__name__:<8}   (numba failed)")
for D, H, W in shapes_3d:
    for dt in dtypes:
        key = ("3d", D, H, W, dt.__name__)
        l2 = run_numba(["3d", "l2", str(D), str(H), str(W), dt.__name__], key)
        l1 = run_numba(["3d", "l1", str(D), str(H), str(W), dt.__name__], key)
        if l2 and l1:
            nb[key] = (l2[1], l1[1])
            print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} {l2[1]:>12.2f}  {l1[1]:>12.2f}")
        else:
            print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8}   (numba failed)")


# --- speedup table ---
print(f"\n{'shape':<16} {'dtype':<8} "
      f"{'cpp L2':>8} {'numba L2':>10} {'L2 ×':>7}   "
      f"{'cpp L1':>8} {'numba L1':>10} {'L1 ×':>7}")
print("-" * 84)
for H, W in shapes_2d:
    for dt in dtypes:
        key = ("2d", H, W, dt.__name__)
        if key in nb:
            cpp_l2, cpp_l1 = cpp[key]
            nb_l2, nb_l1 = nb[key]
            print(f"2D {H}x{W:<10} {dt.__name__:<8} "
                  f"{cpp_l2:>7.2f} {nb_l2:>9.2f} {nb_l2/cpp_l2:>6.2f}×   "
                  f"{cpp_l1:>7.2f} {nb_l1:>9.2f} {nb_l1/cpp_l1:>6.2f}×")
for D, H, W in shapes_3d:
    for dt in dtypes:
        key = ("3d", D, H, W, dt.__name__)
        if key in nb:
            cpp_l2, cpp_l1 = cpp[key]
            nb_l2, nb_l1 = nb[key]
            print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} "
                  f"{cpp_l2:>7.2f} {nb_l2:>9.2f} {nb_l2/cpp_l2:>6.2f}×   "
                  f"{cpp_l1:>7.2f} {nb_l1:>9.2f} {nb_l1/cpp_l1:>6.2f}×")


# --- equivalence check: do cpp L1 and cpp L2 produce identical colored output? ---
print(f"\n{'shape':<16} {'dtype':<8} {'pixel match':>12} {'diff pixels':>12} {'L1 valid?':>10} {'L2 valid?':>10}")
print("-" * 64)


def is_valid_2d(colored, mask_input):
    """Check no two horizontally/vertically adjacent distinct nonzero
    labels in mask_input share a color in colored."""
    m = mask_input
    # horizontal
    nz_r = (m[:, :-1] != 0) & (m[:, 1:] != 0)
    diff_r = m[:, :-1] != m[:, 1:]
    bad_r = (colored[:, :-1] == colored[:, 1:]) & nz_r & diff_r
    nz_d = (m[:-1, :] != 0) & (m[1:, :] != 0)
    diff_d = m[:-1, :] != m[1:, :]
    bad_d = (colored[:-1, :] == colored[1:, :]) & nz_d & diff_d
    return not (bad_r.any() or bad_d.any())


def is_valid_3d(colored, mask_input):
    m = mask_input
    bad = False
    for ax in range(3):
        sl0 = [slice(None)] * 3; sl0[ax] = slice(None, -1)
        sl1 = [slice(None)] * 3; sl1[ax] = slice(1, None)
        sl0, sl1 = tuple(sl0), tuple(sl1)
        nz = (m[sl0] != 0) & (m[sl1] != 0)
        diff = m[sl0] != m[sl1]
        b = (colored[sl0] == colored[sl1]) & nz & diff
        bad = bad or b.any()
    return not bad


for H, W in shapes_2d:
    for dt in dtypes:
        m = make_2d(H, W, max(20, H * W // 8000), dt)
        out_l2 = cpp_label(m, 2)
        out_l1 = cpp_label(m, 1)
        diff_pix = int(np.sum(out_l1 != out_l2))
        match = (diff_pix == 0)
        valid_l1 = is_valid_2d(out_l1, m.astype(np.int32))
        valid_l2 = is_valid_2d(out_l2, m.astype(np.int32))
        match_str = "YES" if match else f"{(1 - diff_pix / out_l2.size) * 100:.4f}%"
        print(f"2D {H}x{W:<10} {dt.__name__:<8} {match_str:>11}  {diff_pix:>12} "
              f"{str(valid_l1):>10} {str(valid_l2):>10}")
for D, H, W in shapes_3d:
    for dt in dtypes:
        m = make_3d(D, H, W, max(20, D * H * W // 8000), dt)
        out_l2 = cpp_label(m, 2)
        out_l1 = cpp_label(m, 1)
        diff_pix = int(np.sum(out_l1 != out_l2))
        match = (diff_pix == 0)
        valid_l1 = is_valid_3d(out_l1, m.astype(np.int32))
        valid_l2 = is_valid_3d(out_l2, m.astype(np.int32))
        match_str = "YES" if match else f"{(1 - diff_pix / out_l2.size) * 100:.4f}%"
        print(f"3D {D}x{H}x{W:<8} {dt.__name__:<8} {match_str:>11}  {diff_pix:>12} "
              f"{str(valid_l1):>10} {str(valid_l2):>10}")
