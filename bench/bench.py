"""Compare numba ncolor vs C++/threadpool prototype on the same inputs.

Run after building the C++ extension::

    cd <ncolor>/cpp_proto
    python setup.py build_ext --inplace
    python bench.py

The script:
  1. Reuses ncolor's `neighbors()` so connectivity offsets match exactly.
  2. Builds a synthetic 287x377 label image (matches the user-facing test
     input size used in the diagnosis bench).
  3. Times each implementation over warmup + N steady-state runs.
  4. Reports per-run cost AND scales the C++ version across thread counts
     by re-creating the engine each time.
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time

import numpy as np

# Ensure ncolor's NUMBA_NUM_THREADS env can vary.
print(f"NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS', 'default')}", file=sys.stderr)

# Force the numba reference for "numba ncolor.label" timing line — even when
# NCOLOR_BACKEND=cpp is set in the environment, this script wants to compare.
os.environ["NCOLOR_BACKEND"] = "numba"
import ncolor                                       # noqa: E402
from ncolor._numba_legacy.color import neighbors, _PARALLEL_THRESHOLD  # noqa: E402

try:
    from ncolor._backend import Solver, ConnectEngine, ExpandEngine, _smt
except ImportError:
    sys.exit(
        "ncolor._backend extension not found. Build with:\n"
        "  python setup.py build_ext --inplace"
    )

print(f"ncolor:        {ncolor.__file__}")
print(f"backend:       {Solver.__module__}")
print(f"auto_threads:  {_smt.auto_threads()}")


def make_mask(H=287, W=377, n=20, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=np.int32)
    for i in range(1, n + 1):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = int(rng.integers(8, max(16, H // 30)))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def prep_inputs(mask, conn=1):
    """Replicate ncolor.color.connect's preprocessing → (line, nbs, ht_size)."""
    n_labels = int(mask.max())
    nbs = neighbors(mask.shape, conn, True)
    ht_raw = len(nbs) * n_labels * 2
    ht_size = 1
    while ht_size < ht_raw:
        ht_size <<= 1
    buf = np.pad(mask, 1, mode="constant").astype(np.int32, copy=False)
    nbs_pad = neighbors(buf.shape, conn, True).astype(np.int64, copy=False)
    line = buf.ravel().astype(np.int32, copy=False)
    return line, nbs_pad, ht_size


def bench(fn, runs=20, warmup=3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return {
        "mean": st.mean(times),
        "median": st.median(times),
        "min": min(times),
        "max": max(times),
    }


def fmt(stats):
    return f"mean={stats['mean']:6.2f} med={stats['median']:6.2f} min={stats['min']:6.2f} max={stats['max']:6.2f} ms"


def main():
    sizes = [(287, 377), (512, 512), (1024, 1024), (2048, 2048)]
    # Auto-pick the upper bound from cpu_count so many-core hosts
    # actually exercise their full thread complement instead of capping at 32.
    cpu_max = os.cpu_count() or 8
    cpp_thread_grid = sorted(set(t for t in [1, 4, 8, 16, 32, 64, 128] if t <= cpu_max))

    for H, W in sizes:
        n_inst = max(20, (H * W) // 8000)
        mask = make_mask(H, W, n=n_inst)
        mask_int = mask.astype(np.int32, copy=False)
        line, nbs, ht_size = prep_inputs(mask)
        print(f"\n{'=' * 78}")
        print(f"shape={H}x{W}  pixels={H*W:>10}  ht_size={ht_size}  n_labels={n_inst}")
        print(f"{'-' * 78}")

        # numba — full ncolor.label (find pairs + expand + color + apply lut)
        s = bench(lambda: ncolor.label(mask, max_depth=20, expand=True, return_n=True, format_input=False))
        print(f"  numba ncolor.label   ({os.environ.get('NUMBA_NUM_THREADS','def')} threads): {fmt(s)}")

        # C++ end-to-end via Solver.
        for nt in cpp_thread_grid:
            solver = Solver(nt)
            s = bench(lambda: solver.label(mask_int, 4, 30, 10), runs=10, warmup=2)
            if nt == cpp_thread_grid[0]:
                cpp_out, cpp_n = solver.label(mask_int, 4, 30, 10)
                nb_out, nb_n = ncolor.label(
                    mask, max_depth=20, expand=True, return_n=True, format_input=False)
                # Element-wise identity test: are the two outputs the SAME
                # (not just both valid colorings, but bit-for-bit equal)?
                identical = (cpp_out.shape == nb_out.shape) and np.array_equal(cpp_out, nb_out)
                # If not identical, count diff and check each is still a valid coloring.
                diff = int(np.count_nonzero(cpp_out != nb_out)) if cpp_out.shape == nb_out.shape else -1
                cpp_valid = _is_valid_coloring(cpp_out, mask)
                nb_valid = _is_valid_coloring(nb_out, mask)
                tag = "IDENTICAL" if identical else f"differ ({diff} px)"
                print(f"  ◦ vs numba: {tag} | cpp n={cpp_n} valid={cpp_valid} | numba n={nb_n} valid={nb_valid}")
            print(f"  cpp Solver.label     ({nt:2} threads): {fmt(s)}")


def _is_valid_coloring(colored, original_mask):
    """Check that no adjacent (4-conn) pixels with non-bg labels share a color."""
    H, W = colored.shape
    # Compare to right and down neighbours.
    nz_right = (original_mask[:, :-1] != 0) & (original_mask[:, 1:] != 0)
    nz_down  = (original_mask[:-1, :] != 0) & (original_mask[1:, :] != 0)
    diff_label_right = original_mask[:, :-1] != original_mask[:, 1:]
    diff_label_down  = original_mask[:-1, :] != original_mask[1:, :]
    bad_right = (colored[:, :-1] == colored[:, 1:]) & nz_right & diff_label_right
    bad_down  = (colored[:-1, :] == colored[1:, :]) & nz_down  & diff_label_down
    return not (bad_right.any() or bad_down.any())


if __name__ == "__main__":
    main()
