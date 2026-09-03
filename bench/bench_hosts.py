"""Cross-host regression benchmark for ncolor.

Records min / median wall time of the public API on a fixed set of
synthetic inputs, together with enough metadata (host, CPU, thread
count, compiler flags, git sha) to compare two builds on the same host.

Run on one host (writes ``<out>/<tag>__<host>.json``)::

    PYTHONPATH=src python bench/bench_hosts.py --tag baseline --out bench_outputs/hosts

Compare two tags across every host that has both::

    python bench/bench_hosts.py --compare baseline candidate --dir bench_outputs/hosts

``bench/run_host_bench.sh`` wraps the remote build + run + copy-back.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ inputs

def boxes_2d(H, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, H), dtype=np.int32)
    n = max(20, H * H // 8000)
    centers = rng.integers(20, H - 20, size=(n, 2))
    radii = rng.integers(8, max(16, H // 30), size=n)
    for i, ((cy, cx), r) in enumerate(zip(centers, radii), 1):
        m[max(0, cy - r):cy + r, max(0, cx - r):cx + r] = i
    return m


def boxes_3d(D, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((D, D, D), dtype=np.int32)
    n = max(50, D ** 3 // 20000)
    centers = rng.integers(6, D - 6, size=(n, 3))
    radii = rng.integers(3, max(5, D // 16), size=n)
    for i, ((cz, cy, cx), r) in enumerate(zip(centers, radii), 1):
        m[max(0, cz - r):cz + r, max(0, cy - r):cy + r, max(0, cx - r):cx + r] = i
    return m


def fixture_800():
    p = REPO / "test_files" / "synthetic_800.npz"
    if not p.exists():
        return None
    return np.load(p)["labels"].astype(np.int32)


def build_cases(quick=False):
    cases = [
        ("2d_1024", boxes_2d(1024), 15),
        ("2d_2048", boxes_2d(2048), 15),
        ("3d_128", boxes_3d(128), 15),
    ]
    fx = fixture_800()
    if fx is not None:
        cases.insert(0, ("synthetic_800", fx, 20))
    if not quick:
        cases += [
            ("2d_4096", boxes_2d(4096), 7),
            ("3d_256", boxes_3d(256), 5),
        ]
    return cases


# ------------------------------------------------------------------ timing

def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000.0)
    return {"min_ms": min(ts), "median_ms": statistics.median(ts), "n": iters}


def stage_breakdown(m):
    """Per-stage ms from the engine's own timers (one call)."""
    import ncolor
    from ncolor.color import _get_solver
    devnull = open(os.devnull, "w")
    old = sys.stderr
    sys.stderr = devnull
    try:
        ncolor.label(m, verbose=True)
    finally:
        sys.stderr = old
        devnull.close()
    return {name: round(ms, 3) for name, ms in _get_solver().get_last_stages()}


def run(tag, out_dir, quick=False, note=""):
    import ncolor
    from ncolor._backend import _smt

    ops = {
        "label": lambda m: ncolor.label(m),
        "label_noexpand": lambda m: ncolor.label(m, expand=False),
        "expand_labels": lambda m: ncolor.expand_labels(m),
        "connect": lambda m: ncolor.connect(m),
        "format_labels": lambda m: ncolor.format_labels(m),
    }
    results = {}
    for name, m, iters in build_cases(quick):
        entry = {"shape": list(m.shape), "n_cells": int(m.max())}
        for op, fn in ops.items():
            entry[op] = timeit(lambda: fn(m), iters)
        entry["label_stages"] = stage_breakdown(m)
        results[name] = entry
        print(f"{name:14s} label {entry['label']['min_ms']:8.2f} ms  "
              f"expand {entry['expand_labels']['min_ms']:8.2f}  "
              f"connect {entry['connect']['min_ms']:7.2f}  "
              f"format {entry['format_labels']['min_ms']:6.2f}", flush=True)

    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        sha = os.environ.get("NCOLOR_BENCH_SHA", "unknown")
    meta = {
        "tag": tag,
        "note": note,
        "host": socket.gethostname().split(".")[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _smt._cpu_model(),
        "physical_cores": _smt._physical_cores(),
        "logical_cores": os.cpu_count(),
        "auto_threads": _smt.auto_threads(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "ncolor": ncolor.__version__,
        "git_sha": sha,
        "compiler": os.environ.get("NCOLOR_BENCH_COMPILER") or sysconfig.get_config_var("CC"),
        "env": {k: v for k, v in os.environ.items() if k.startswith("NCOLOR_")},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}__{meta['host']}.json"
    path.write_text(json.dumps({"meta": meta, "results": results}, indent=1))
    print(f"wrote {path}  (threads={meta['auto_threads']}, cpu={meta['cpu']})")
    return path


# ----------------------------------------------------------------- compare

def load_tag(dir_, tag):
    out = {}
    for p in glob.glob(str(Path(dir_) / f"{tag}__*.json")):
        d = json.loads(Path(p).read_text())
        out[d["meta"]["host"]] = d
    return out


def compare(dir_, tag_a, tag_b, ops=("label", "expand_labels", "connect", "format_labels"),
            threshold=0.03):
    A, B = load_tag(dir_, tag_a), load_tag(dir_, tag_b)
    hosts = sorted(set(A) & set(B))
    if not hosts:
        print(f"no host has both tags {tag_a!r} and {tag_b!r} in {dir_}")
        return 1
    worst = 0.0
    print(f"| host | case | op | {tag_a} ms | {tag_b} ms | ratio | verdict |")
    print("|---|---|---|---|---|---|---|")
    for h in hosts:
        ma, mb = A[h]["meta"], B[h]["meta"]
        if ma["auto_threads"] != mb["auto_threads"]:
            print(f"| {h} | | | | | | thread count differs: {ma['auto_threads']} vs {mb['auto_threads']} |")
        for case in A[h]["results"]:
            if case not in B[h]["results"]:
                continue
            for op in ops:
                a = A[h]["results"][case][op]["min_ms"]
                b = B[h]["results"][case][op]["min_ms"]
                r = b / a if a else float("nan")
                worst = max(worst, r)
                verdict = "SLOWER" if r > 1 + threshold else ("faster" if r < 1 - threshold else "same")
                print(f"| {h} | {case} | {op} | {a:.2f} | {b:.2f} | {r:.3f} | {verdict} |")
    print(f"\nworst ratio {tag_b}/{tag_a}: {worst:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="label for this build (e.g. baseline)")
    ap.add_argument("--out", default=str(REPO / "bench_outputs" / "hosts"))
    ap.add_argument("--quick", action="store_true", help="skip the 4096^2 and 256^3 cases")
    ap.add_argument("--note", default="")
    ap.add_argument("--compare", nargs=2, metavar=("TAG_A", "TAG_B"))
    ap.add_argument("--dir", default=str(REPO / "bench_outputs" / "hosts"))
    ap.add_argument("--threshold", type=float, default=0.03)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(args.dir, *args.compare, threshold=args.threshold))
    if not args.tag:
        ap.error("--tag is required unless --compare is given")
    run(args.tag, args.out, quick=args.quick, note=args.note)


if __name__ == "__main__":
    main()
