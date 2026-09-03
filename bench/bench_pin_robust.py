"""Robust paired A/B for NCOLOR_PIN_THREADS.

Pinning is read once per process, so unpinned vs pinned must be separate
processes.  This driver launches them back-to-back (paired) many times and
reports the full distribution + a paired sign test, so a small/noisy effect
can't masquerade as a real regression (or win).

Usage (driver):  bench_pin_robust.py T H ndim iters N
       (child):  bench_pin_robust.py --child T H ndim iters     [env NCOLOR_PIN_THREADS]
"""
import os, sys, subprocess, statistics, time
import numpy as np


def make_mask(H, ndim, seed=0):
    rng = np.random.default_rng(seed)
    if ndim == 2:
        m = np.zeros((H, H), np.int32)
        n = max(50, H * H // 4000)
        c = rng.integers(10, H - 10, size=(n, 2)); r = rng.integers(5, max(10, H // 40), size=n)
        for i, ((y, x), rr) in enumerate(zip(c, r), 1):
            m[max(0, y - rr):y + rr, max(0, x - rr):x + rr] = i
        return m
    m = np.zeros((H, H, H), np.int32)
    n = max(50, H ** 3 // 8000)
    c = rng.integers(5, H - 5, size=(n, 3)); r = rng.integers(3, max(6, H // 20), size=n)
    for i, ((z, y, x), rr) in enumerate(zip(c, r), 1):
        m[max(0, z - rr):z + rr, max(0, y - rr):y + rr, max(0, x - rr):x + rr] = i
    return m


def child(T, H, ndim, iters):
    from ncolor._backend import Solver
    sv = Solver(T)
    mask = make_mask(H, ndim)
    for _ in range(5):
        sv.label(mask)
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter(); sv.label(mask); dt = time.perf_counter() - t
        if dt < best:
            best = dt
    print(f"{best * 1e3:.4f}")


def summarize(xs):
    xs = sorted(xs)
    return (f"n={len(xs)} min={xs[0]:.2f} med={statistics.median(xs):.2f} "
            f"mean={statistics.mean(xs):.2f} std={statistics.pstdev(xs):.2f} max={xs[-1]:.2f}")


def driver(T, H, ndim, iters, N):
    env0 = dict(os.environ); env0["NCOLOR_PIN_THREADS"] = ""; env0["NCOLOR_NO_CALIBRATE"] = "1"
    env1 = dict(os.environ); env1["NCOLOR_PIN_THREADS"] = "1"; env1["NCOLOR_NO_CALIBRATE"] = "1"
    cmd = [sys.executable, os.path.abspath(__file__), "--child", str(T), str(H), str(ndim), str(iters)]
    p0, p1 = [], []
    wins = 0  # reps where pinned is FASTER (pin1 < pin0)
    for rep in range(N):
        # interleave order each rep to cancel any residual ordering bias
        order = [(env0, p0), (env1, p1)] if rep % 2 == 0 else [(env1, p1), (env0, p0)]
        vals = {}
        for env, bucket in order:
            out = subprocess.run(cmd, env=env, capture_output=True, text=True)
            v = float(out.stdout.strip().splitlines()[-1])
            bucket.append(v); vals[id(bucket)] = v
        if vals[id(p1)] < vals[id(p0)]:
            wins += 1
        print(f"  rep {rep+1:2d}/{N}: unpinned={p0[-1]:7.2f}  pinned={p1[-1]:7.2f}  "
              f"Δ={p1[-1]-p0[-1]:+6.2f}ms ({(p1[-1]/p0[-1]-1)*100:+5.1f}%)", flush=True)
    print(f"\n[{H}^{ndim} T={T}, {N} paired reps, min-of-{iters} each]")
    print(f"  unpinned: {summarize(p0)}")
    print(f"  pinned  : {summarize(p1)}")
    dmed = statistics.median(p1) - statistics.median(p0)
    print(f"  median Δ = {dmed:+.2f}ms ({(statistics.median(p1)/statistics.median(p0)-1)*100:+.1f}%); "
          f"pinned faster in {wins}/{N} reps")


if __name__ == "__main__":
    if sys.argv[1] == "--child":
        child(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    else:
        driver(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
