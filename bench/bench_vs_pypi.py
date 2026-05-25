"""Head-to-head benchmark: local HEAD (C++ + auto-soft) vs ncolor 1.5.3
from PyPI (pure numba).

Runs each backend in a subprocess (a single process can only load one
ncolor module). Each image × backend × n is timed with 12 reps + 4
warmups, reporting min and median ms plus n_used (whether the picker
actually reached the requested ``n`` or had to fall back to ``n+1``).

PyPI 1.5.3 is the latest released ncolor on pip; install it into a
dedicated target directory before running:

    pip install ncolor==1.5.3 --target /tmp/ncolor_pypi/install --no-deps --force-reinstall
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
import numpy as np
import skimage.io
import ncolor as ncolor_local

# Script-relative repo root so the bench is portable across checkouts.
REPO = str(Path(__file__).resolve().parent.parent)
PYPI_PATH = os.environ.get("NCOLOR_PYPI_PATH", "/tmp/ncolor_pypi/install")


def load_logo():
    raw = skimage.io.imread(f"{REPO}/test_files/example.png").astype(np.int32)
    ys, xs = np.where(raw > 0)
    sub = raw[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.int32)
    _, inv = np.unique(sub, return_inverse=True)
    return inv.reshape(sub.shape).astype(np.int32)


def load_synth():
    img = np.load(f"{REPO}/test_files/synthetic_800.npy").astype(np.int32)
    _, inv = np.unique(img, return_inverse=True)
    return inv.reshape(img.shape).astype(np.int32)


def load_mm():
    """Load a dense 2k^2-class microscopy segmentation if the user has
    one. Set ``NCOLOR_MM_FIXTURE=/path/to/seg.npy`` (uint16/int32 label
    image with thousands of cells; see README for the kind of image
    used in the reported benchmark)."""
    p = os.environ.get("NCOLOR_MM_FIXTURE")
    if not p or not os.path.exists(p):
        return None
    img = np.load(p).astype(np.int32)
    _, inv = np.unique(img, return_inverse=True)
    return inv.reshape(img.shape).astype(np.int32)


def time_local(lab, n, reps=12, warmup=4):
    """Local HEAD: conn=1 + p=2 (L2 / Felzenszwalb) — the path the
    soft/bridge_free stack was designed around. p=1 (Manhattan, current
    default) leaves K_5 obstructions on mm-class data; p=2 places
    Voronoi boundaries to avoid them. Default auto-soft + bridge_free."""
    for _ in range(warmup):
        ncolor_local.label(lab, n=n, conn=1, p=2)
    ts = []
    n_used = 0
    for _ in range(reps):
        t0 = time.perf_counter()
        _, n_used = ncolor_local.label(lab, n=n, conn=1, p=2, return_n=True)
        ts.append((time.perf_counter() - t0) * 1000)
    return min(ts), float(np.median(ts)), int(n_used)


# PyPI numba ncolor matched at conn=1 for an apples-to-apples comparison
# of the hard kernel. (PyPI 1.5.3 has no soft/bridge_free knobs.)
PYPI_SCRIPT = textwrap.dedent("""
    import sys, time, json, numpy as np
    sys.path.insert(0, %r)
    import ncolor as nb_ncolor
    lab = np.load(sys.argv[1])
    n = int(sys.argv[2])
    reps, warmup = 12, 4
    for _ in range(warmup):
        nb_ncolor.label(lab, n=n, conn=1, expand=True)
    ts, last_n = [], 0
    for _ in range(reps):
        t0 = time.perf_counter()
        _, last_n = nb_ncolor.label(lab, n=n, conn=1, expand=True, return_n=True)
        ts.append((time.perf_counter() - t0) * 1000)
    print(json.dumps({'min': min(ts), 'med': float(np.median(ts)), 'n_used': int(last_n)}))
""") % PYPI_PATH


def time_pypi(lab, n):
    arr_path = "/tmp/ncolor_pypi/_lab_arg.npy"
    os.makedirs(os.path.dirname(arr_path), exist_ok=True)
    np.save(arr_path, lab.astype(np.int32))
    p = subprocess.run(
        [sys.executable, "-c", PYPI_SCRIPT, arr_path, str(n)],
        capture_output=True, text=True, timeout=600,
    )
    if p.returncode != 0:
        print(f"  PYPI FAIL: {p.stderr[-400:]}")
        return None, None, None
    d = json.loads(p.stdout.strip().splitlines()[-1])
    return d["min"], d["med"], d["n_used"]


def main():
    targets = [
        ("logo",      load_logo()),
        ("synth_800", load_synth()),
    ]
    mm = load_mm()
    if mm is not None:
        targets.append(("mm 2k^2", mm))
    else:
        print("# NCOLOR_MM_FIXTURE not set or path missing; skipping mm row.")
    print()
    for n in (4, 5):
        print(f"## n={n}")
        print(f"  {'image':<10}  {'cells':>5}  "
               f"{'PyPI 1.5.3 (numba)':<28}  "
               f"{'HEAD (C++ + auto-soft)':<26}  "
               f"{'speedup':>8}")
        print("  " + "-" * 90)
        for name, lab in targets:
            l_min, l_med, l_n = time_local(lab, n)
            p_min, p_med, p_n = time_pypi(lab, n)
            speedup = (p_min / l_min) if l_min and p_min else 0
            local_str = f"{l_min:>5.2f} / {l_med:>5.2f} ms  n={l_n}"
            pypi_str = (f"{p_min:>5.2f} / {p_med:>5.2f} ms  n={p_n}"
                         if p_min is not None else "FAILED")
            print(f"  {name:<10}  {int(lab.max()):>5}  "
                   f"{pypi_str:<28}  {local_str:<26}  "
                   f"{speedup:>6.1f}x")
        print()


if __name__ == "__main__":
    main()
