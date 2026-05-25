"""A/B benchmark: wrap=False vs wrap=True (both with balance=True).

Same interleaved-pair pattern as bench_balance_ab.py — calls alternate
between modes so thermal/scheduler noise averages out. Reports median +
p10/p90 wall-clock per call alongside slot-cell-count balance.

Usage:
  python bench_wrap_ab.py [--quick]
"""
import argparse
import os
import statistics as st
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))

import numpy as np
import skimage.io
import skimage.measure

import ncolor


def make_circles_2d(H, W, n, r_min=4, r_max=10, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=np.int32)
    for i in range(1, n + 1):
        cy, cx = rng.integers(r_max, H - r_max), rng.integers(r_max, W - r_max)
        r = int(rng.integers(r_min, r_max + 1))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def make_balls_3d(D, H, W, n, r_min=2, r_max=5, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((D, H, W), dtype=np.int32)
    for i in range(1, n + 1):
        cz = rng.integers(r_max, D - r_max)
        cy = rng.integers(r_max, H - r_max)
        cx = rng.integers(r_max, W - r_max)
        r = int(rng.integers(r_min, r_max + 1))
        zz, yy, xx = np.ogrid[:D, :H, :W]
        m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def make_tight_crop_2d(H, W, n, seed=0):
    """Cells crammed close to the borders — where wrap is *most* expected to
    change behaviour (many cells within reach of the toroidal seam)."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=np.int32)
    for i in range(1, n + 1):
        cy, cx = rng.integers(2, H - 2), rng.integers(2, W - 2)
        r = int(rng.integers(3, 6))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def load_example_png():
    p = _HERE / "test_files" / "example.png"
    return skimage.io.imread(p).astype(np.int32)


def slot_cell_counts(nc):
    return [int(skimage.measure.label((nc == s).astype(np.uint8)).max())
            for s in range(1, int(nc.max()) + 1)]


def spread(counts):
    return (max(counts) - min(counts)) if counts else 0


def stdev_pct(counts):
    if not counts:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    fracs = [100.0 * c / total for c in counts]
    return st.pstdev(fracs)


def time_call(fn, *args, **kwargs):
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return (time.perf_counter() - t) * 1000.0, out


def bench_one(label, m, runs, warmup):
    for _ in range(warmup):
        ncolor.label(m, balance=True, wrap=False)
        ncolor.label(m, balance=True, wrap=True)
    t_off, t_on = [], []
    last_off = last_on = None
    for _ in range(runs):
        dt, out = time_call(ncolor.label, m, balance=True, wrap=False)
        t_off.append(dt); last_off = out
        dt, out = time_call(ncolor.label, m, balance=True, wrap=True)
        t_on.append(dt);  last_on = out
    return {
        "label": label,
        "shape": m.shape,
        "t_off_ms": t_off,
        "t_on_ms":  t_on,
        "slots_off": slot_cell_counts(last_off),
        "slots_on":  slot_cell_counts(last_on),
    }


def fmt_times(ms):
    s = sorted(ms)
    med = s[len(s) // 2]
    p10 = s[max(0, int(len(s) * 0.1))]
    p90 = s[min(len(s) - 1, int(len(s) * 0.9))]
    return f"{med:7.2f}  [p10={p10:6.2f}, p90={p90:6.2f}]"


def fmt_slots(counts):
    if not counts:
        return "(no slots)"
    return "/".join(str(c) for c in counts) + f"  Δ={spread(counts):3d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    runs   = 10 if args.quick else 30
    warmup =  2 if args.quick else 6

    cases = [
        ("example.png (logo)",          load_example_png()),
        ("circles 2D 512² n=80",        make_circles_2d(512, 512, 80)),
        ("circles 2D 1024² n=300",      make_circles_2d(1024, 1024, 300)),
        ("tight-crop 256² n=200",       make_tight_crop_2d(256, 256, 200)),
        ("balls 3D 64³ n=80",           make_balls_3d(64, 64, 64, 80)),
    ]

    results = []
    for lbl, m in cases:
        print(f"running: {lbl}  shape={m.shape}", flush=True)
        results.append(bench_one(lbl, m, runs=runs, warmup=warmup))

    print()
    print("=" * 100)
    print(f"{'case':32s}  {'wrap':5s}  {'time (ms, median)':28s}  slot counts (cells)")
    print("-" * 100)
    for r in results:
        print(f"{r['label']:32s}  {'OFF':5s}  {fmt_times(r['t_off_ms']):28s}  "
              f"{fmt_slots(r['slots_off'])}")
        print(f"{'':32s}  {'ON':5s}  {fmt_times(r['t_on_ms']):28s}  "
              f"{fmt_slots(r['slots_on'])}")
        med_off = sorted(r['t_off_ms'])[len(r['t_off_ms']) // 2]
        med_on  = sorted(r['t_on_ms'])[len(r['t_on_ms'])  // 2]
        ratio = med_on / med_off if med_off > 0 else float('inf')
        print(f"{'':32s}  Δ      cost ON/OFF = {ratio:4.2f}×    "
              f"slot Δ {spread(r['slots_off']):3d}→{spread(r['slots_on']):3d}    "
              f"slot σ {stdev_pct(r['slots_off']):4.2f}%→{stdev_pct(r['slots_on']):4.2f}%")
        print()
    print("=" * 100)


if __name__ == "__main__":
    main()
