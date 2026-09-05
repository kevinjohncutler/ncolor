"""Time one ncolor build on the shared corpus, or check its outputs.

Deliberately standalone: it imports only numpy and whatever ``ncolor``
is on ``sys.path``, so the same file measures the baseline tree and the
candidate tree. Point ``PYTHONPATH`` at a tree's ``src`` and run it with
any interpreter that can import numpy.

Per-iteration times are written out, not a summary, so the driver can
pair the two builds and put an interval on the difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import threading
import time

import numpy as np

TARGET_S = 0.20          # wall time to spend per (case, op)
# Graphs for the picker itself. The corpus images are nearly all easy
# for it: a slot wins almost at once, so they say little about how the
# race behaves when it has to work. These are hard enough that the
# search runs, which is where a change to the race would show.
HARD_GRAPHS = ((800, 6, 0), (1500, 5, 1), (2000, 8, 2), (3000, 5, 3))
MIN_ITERS, MAX_ITERS = 5, 40


def ops_for(mod):
    return {
        "label": lambda m: mod.label(m),
        "label_noexpand": lambda m: mod.label(m, expand=False),
        "expand_labels": lambda m: mod.expand_labels(m),
        "connect": lambda m: mod.connect(m),
        "format_labels": lambda m: mod.format_labels(m),
    }


def time_op(fn, m):
    """Per-call times in ms. Iterations are chosen from a trial call so
    every case costs about the same wall time on any machine."""
    fn(m)                                        # warm buffers and pools
    t0 = time.perf_counter()
    fn(m)
    one = time.perf_counter() - t0
    iters = int(min(MAX_ITERS, max(MIN_ITERS, TARGET_S / max(one, 1e-6))))
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(m)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def concurrent_rate(mod, m, n_threads=4):
    """Images per second with n_threads callers, which is what the
    engine pool changes. A build that serializes them scores about the
    same as one caller. Calls per thread are chosen from a trial so a
    big image does not cost minutes."""
    t0 = time.perf_counter()
    mod.label(m)
    one = time.perf_counter() - t0
    reps = int(min(8, max(1, 0.25 / max(one, 1e-6))))

    def once():
        ts = [threading.Thread(target=lambda: [mod.label(m) for _ in range(reps)])
              for _ in range(n_threads)]
        t0 = time.perf_counter()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return (n_threads * reps) / (time.perf_counter() - t0)

    once()
    return [once() for _ in range(3)]


def rand_graph(n, deg, seed):
    """A deterministic random graph, same for every build."""
    rng = np.random.default_rng(seed)
    e = set()
    for u in range(n):
        for v in rng.integers(0, n, deg):
            if v != u:
                e.add((min(u, int(v)), max(u, int(v))))
    return np.array(sorted(e), np.int32)


def time_graphs(mod):
    out = {}
    # color_graph is newer than some of the builds this is pointed at
    # (it does not exist in 2.0.2), and a missing entry is simply not
    # compared, so say so and carry on rather than losing the run.
    if not hasattr(mod, "color_graph"):
        print("  (this build has no color_graph; skipping the hard graphs)",
              flush=True)
        return out
    for n, deg, seed in HARD_GRAPHS:
        edges = rand_graph(n, deg, seed)
        fn = lambda e=edges, k=n: mod.color_graph(e, k)
        fn()
        t0 = time.perf_counter(); fn(); one = time.perf_counter() - t0
        iters = int(min(20, max(3, 0.5 / max(one, 1e-6))))
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        out[f"g{n}_d{deg}"] = {"color_graph": ts}
    return out


def digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.blake2b(a.tobytes(), digest_size=12).hexdigest()


def edge_digest(pairs):
    """Order-independent digest of an adjacency list.

    ``connect`` returns an unordered edge list, and the order pairs come
    out in follows the scan, which is an implementation detail: the
    fused scan emits them in a different order than the old one while
    producing the same graph. Canonicalize before hashing so a reordering
    is not mistaken for a behavior change.
    """
    p = np.asarray(pairs)
    if p.ndim != 2 or p.size == 0:
        return digest(p)
    p = np.sort(p, axis=1)
    p = np.unique(p, axis=0)
    return digest(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default="?")
    ap.add_argument("--mode", choices=("serial", "concurrent", "verify", "graphs"),
                    default="serial")
    ap.add_argument("--cases", default="")
    ap.add_argument("--shuffle", type=int, default=-1,
                    help="seed to measure the cases in a random order; -1 keeps "
                         "corpus order")
    args = ap.parse_args()

    import ncolor
    if args.mode == "graphs":
        res = time_graphs(ncolor)
        meta = {"build": args.build, "mode": "graphs",
                "host": socket.gethostname().split(".")[0],
                "ncolor": getattr(ncolor, "__version__", "?")}
        with open(args.out, "w") as fh:
            json.dump({"meta": meta, "results": res}, fh)
        print(f"[{meta['host']}] {args.build:9s} graphs", flush=True)
        return
    npz = np.load(args.corpus)
    names = args.cases.split(",") if args.cases else list(npz.files)
    # Measuring in a fixed order lets one case sit in another's shadow.
    # Seen on an i9: format_labels on boxes2d_1024_s1 ran 45% slower than
    # the baseline, reproducibly and with a tight interval, but only when
    # it followed boxes2d_1024_s0; reversing the order made the two
    # builds identical. A different order per repetition turns that kind
    # of artifact into noise the interval can see, instead of a bias it
    # cannot.
    if args.shuffle >= 0:
        np.random.default_rng(args.shuffle).shuffle(names)
    ops = ops_for(ncolor)
    res = {}

    for name in names:
        m = npz[name]
        if args.mode == "serial":
            res[name] = {op: time_op(fn, m) for op, fn in ops.items()}
        elif args.mode == "concurrent":
            res[name] = {"label_rate": concurrent_rate(ncolor, m)}
        else:
            out, n = ncolor.label(m, return_n=True)
            out = np.asarray(out)
            conflicts = ncolor.label(m, return_conflicts=True)[1]
            res[name] = {
                # These are deterministic, so the two builds must agree
                # byte for byte.
                "expand_labels": digest(ncolor.expand_labels(m)),
                "format_labels": digest(ncolor.format_labels(m)),
                "connect": edge_digest(ncolor.connect(m)),
                # label races several searches under a time budget, so
                # only what it owes is comparable: same foreground, no
                # conflicts, colors dense from 1. A second foreground
                # says whether this build is even self-consistent here:
                # on an image of single-pixel labels neither build is,
                # so there is nothing to compare across them.
                "label_fg": digest((out != 0).astype(np.uint8)),
                "label_fg_again": digest(
                    (np.asarray(ncolor.label(m)) != 0).astype(np.uint8)),
                "label_n": int(n),
                "label_conflicts": int(conflicts),
                "label_colors": sorted(int(v) for v in np.unique(out) if v),
            }

    meta = {
        "build": args.build,
        "mode": args.mode,
        "host": socket.gethostname().split(".")[0],
        "ncolor": getattr(ncolor, "__version__", "?"),
        "file": getattr(ncolor, "__file__", "?"),
        "interpreter": platform.python_version(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("NCOLOR_")},
        "t": time.strftime("%H:%M:%S"),
    }
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "results": res}, fh)
    print(f"[{meta['host']}] {args.build:9s} {args.mode:10s} ncolor={meta['ncolor']}", flush=True)


if __name__ == "__main__":
    main()
