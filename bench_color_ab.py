"""A/B benchmark: serial vs parallel color attempts in Solver.label.

Interleaves both modes call-by-call so thermal/scheduler effects average
out across the comparison. Reports median, p10, p90 wall-clock for each
mode. Run with the same input on different machines (m2 mac, threadripper,
intel, amd) to see whether the parallel-attempts win travels.

Usage:
  python bench_color_ab.py [--quick] [--shapes-3d D H W ...] [--shapes-2d ...]
"""
import argparse
import os
import sys
import time
import statistics as st
# Resolve repo path on whichever host this runs on (mac NAS mount,
# linux /home/kcutler/DataDrive mount, etc.)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
import numpy as np
from ncolor._backend import Solver


def make_balls_3d(D, H, W, n, dtype=np.uint16, seed=0):
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


def make_circles_2d(H, W, n, dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=dtype)
    for i in range(1, n + 1):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = int(rng.integers(8, max(16, H // 30)))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def bench_interleaved(solver, m, runs=40, warmup=8):
    """Alternate serial / parallel calls so they share thermal state."""
    # Warmup both paths
    for _ in range(warmup):
        solver.label(m, color_mode=0)
        solver.label(m, color_mode=1)
    serial_times, parallel_times = [], []
    serial_color, parallel_color = [], []
    # Interleave (S, P, S, P, ...) to distribute scheduling noise
    for _ in range(runs):
        t = time.perf_counter()
        solver.label(m, color_mode=0, capture_stages=True)
        serial_times.append((time.perf_counter() - t) * 1000)
        s = dict(solver.get_last_stages())
        serial_color.append(s.get("color", 0.0))

        t = time.perf_counter()
        solver.label(m, color_mode=1, capture_stages=True)
        parallel_times.append((time.perf_counter() - t) * 1000)
        s = dict(solver.get_last_stages())
        parallel_color.append(s.get("color", 0.0))
    return serial_times, parallel_times, serial_color, parallel_color


def report(label, st_v, pa_v, st_c, pa_c):
    sp = lambda v, q: sorted(v)[max(0, int(q * len(v)) - 1)]
    print(f"  {label:<14} "
          f"serial: med={st.median(st_v):.2f} p10={sp(st_v, 0.1):.2f} p90={sp(st_v, 0.9):.2f}  |  "
          f"parallel: med={st.median(pa_v):.2f} p10={sp(pa_v, 0.1):.2f} p90={sp(pa_v, 0.9):.2f}")
    print(f"  {'  └ color stage':<14} "
          f"serial: med={st.median(st_c):.2f}  |  parallel: med={st.median(pa_c):.2f}  "
          f"(speedup {st.median(st_c) / max(st.median(pa_c), 0.001):.2f}×)")
    delta = st.median(pa_v) - st.median(st_v)
    pct = 100.0 * delta / st.median(st_v)
    verdict = "PARALLEL WIN" if delta < -0.1 else ("PARALLEL LOSS" if delta > 0.1 else "TIE")
    print(f"  {'  └ TOTAL':<14} delta = {delta:+.2f} ms ({pct:+.1f}%) → {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer iterations")
    args = ap.parse_args()

    runs = 20 if args.quick else 40
    solver = Solver(-1)
    print(f"machine: n_threads={solver.n_threads}, runs={runs}\n")

    print("3D shapes:")
    for D, H, W in [(64, 64, 64), (128, 128, 128), (256, 256, 256)]:
        m = make_balls_3d(D, H, W, max(20, D * H * W // 8000))
        st_v, pa_v, st_c, pa_c = bench_interleaved(solver, m, runs=runs)
        report(f"{D}x{H}x{W}", st_v, pa_v, st_c, pa_c)

    print("\n2D shapes:")
    for H, W in [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]:
        m = make_circles_2d(H, W, max(20, H * W // 8000))
        st_v, pa_v, st_c, pa_c = bench_interleaved(solver, m, runs=runs)
        report(f"{H}x{W}", st_v, pa_v, st_c, pa_c)


if __name__ == "__main__":
    main()
