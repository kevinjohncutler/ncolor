# ncolor cpp_proto — C++/threadpool drop-in for ncolor.label

Header-only C++ port of ncolor's hot kernels (`expand_labels`,
`_search_hashset_parallel`, `_color_graph_csr_legacy`, `_repair_coloring`,
`_apply_lut`) plus a `Solver` class that runs the full `ncolor.label(image,
expand=True)` pipeline in C++ with a single persistent ThreadPool.

## Output parity

`Solver.label` is **bit-identical** to `numba.ncolor.label` across all
tested sizes (287×377 through 2048×2048). Same colored output, same
`n_used`. The numba version's `_kempe_repair_csr` fallback is intentionally
omitted — it never triggered on any test input we threw at it; the
BFS+local-repair chain converges to the same coloring.

## Why this prototype exists

Two motivations:

1. **A persistent thread-pool model is structurally faster than numba's
   `@njit(parallel=True)` model for low-latency calls** — every prange
   region launches a new parallel team (even on a "warm" pool numba pays
   per-region dispatch + barrier). `Solver` keeps a single std::thread pool
   alive for its lifetime; per call we pay only enqueue + futures wait,
   ≈10 µs vs numba's hundreds of µs to ~1 ms per region. This shows up as
   a 2–6× speedup at small image sizes (≤512²) where per-call overhead
   dominates.
2. **Bit-identical drop-in for `ncolor.label`** with no JIT cold-start —
   useful for interactive viewers where the first call shouldn't take
   seconds while numba compiles.

A long-running diagnostic earlier in this repo's history reported
"700 ms" wall times for `ncolor.label` on a Threadripper PRO 3995WX and
attributed it to libgomp's per-parallel-region launch cost. **That turned
out to be wrong** — the actual cause was ~75 cores' worth of orphan
multiprocessing workers spinning on the host. With a clean machine,
numba's `omp` layer on the same threadripper completes the small-image
case in ≈1 ms (with `OMP_PROC_BIND=spread OMP_PLACES=cores`); the
prototype shaves ~10–50% on top of that for small/medium sizes and ties
at 2048².

## Status

- `connect.hpp` — `_search_hashset_parallel` with parallel pairwise tree-merge
- `expand.hpp` — `expand_labels` (envelope_pass + batch_transpose) with
  per-thread persistent scratch, divisionless pop comparison in phase-1,
  segmented phase-2 for clean autovec
- `color.hpp` — `build_csr_from_pairs`, `color_graph_csr_legacy`,
  `repair_coloring`, `has_conflict_csr`
- `binding.cpp` — pybind11 module exposing `ConnectEngine`, `ExpandEngine`,
  `Solver`
- `setup.py` — local build with `-O3 -march=native -ffp-contract=fast -funroll-loops`
- `bench.py` — comparison vs `numba.ncolor.label`, includes element-wise
  identity test against numba

Not ported (rarely needed in practice):

- `_kempe_repair_csr` — Kempe-chain swaps when `_repair_coloring` can't
  fix conflicts. None of the synthetic or real workloads tested needed it.
- `format_labels` — 1..N relabel of a non-contiguous label image. The
  Solver assumes the input is already normalized; if you need the full
  `ncolor.label(format_input=True)` semantics, normalize in Python first.

## Build

```bash
cd <ncolor>/cpp_proto
python setup.py build_ext --inplace
```

Requires `pybind11` and a C++17 compiler.

> **macOS + NAS gotcha**: importing a `.so` directly from `/Volumes/<NAS>/`
> hangs in `dlopen` on macOS (Gatekeeper code-validation deadlocks on SMB
> mounts). `bench.py` copies the in-place `.so` to `/tmp` as a fresh byte
> stream before importing — adapt the same trick if you import the module
> elsewhere from a NAS-resident source tree.

## Usage

```python
import numpy as np
import ncolor_cpp_proto

solver = ncolor_cpp_proto.Solver(n_threads=16)
colored, n_used = solver.label(label_image_int32, n_colors=4)
# colored is uint8, same shape as input. n_used is the number of colors used.
```

## Bench

```bash
python bench.py
# or with explicit numba threading:
NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=64 python bench.py
```

All benches: each implementation in its own best configuration; bit-identical
output verified element-wise.

### Threadripper PRO 3995WX (Linux, 64C/128T, idle, OMP_PROC_BIND=spread)

| shape       | numba   | cpp best         | speedup |
|-------------|--------:|-----------------:|--------:|
| 287×377     | 5.63 ms | 0.88 (T=8)       | **6.4×** |
| 512×512     | 4.79    | 1.78 (T=32)      | **2.7×** |
| 1024×1024   | 9.51    | 6.70 (T=32)      | **1.4×** |
| 2048×2048   | 27.32   | 25.82 (T=64)     | 1.06× (tied) |

### AMD Ryzen 9 7950X (Linux, 16C/32T, idle, default env)

| shape       | numba    | cpp best        | speedup |
|-------------|---------:|----------------:|--------:|
| 287×377     | 0.53 ms  | 0.39 (T=8)      | **1.4×** |
| 512×512     | 0.97     | 0.86 (T=16)     | 1.13×    |
| 1024×1024   | 3.40     | 3.72 (T=16)     | 0.91× (numba wins) |
| 2048×2048   | 15.02 (med) | 18.03 (T=32) | 0.83× (numba wins) |

### Intel i9-9900K (Linux, 8C/16T, idle, default env)

| shape       | numba   | cpp best        | speedup |
|-------------|--------:|----------------:|--------:|
| 287×377     | 0.62 ms | 0.50 (T=16)     | **1.2×** |
| 512×512     | 1.25    | 1.23 (T=16)     | tied     |
| 1024×1024   | 5.78    | 5.78 (T=16)     | tied     |
| 2048×2048   | 26.58   | 29.09 (T=8)     | 0.91× (numba wins) |

### Mac M-series (idle, workqueue — only layer numba builds on macOS)

| shape       | numba   | cpp best        | speedup |
|-------------|--------:|----------------:|--------:|
| 287×377     | 5.05 ms | 1.20 (T=8)      | **4.2×** |
| 512×512     | 5.51    | 2.51 (T=8)      | **2.2×** |
| 1024×1024   | 13.36   | 9.13 (T=16)     | **1.5×** |
| 2048×2048   | 32.95   | 32.76 (T=8)     | tied     |

### L1 chamfer fast path (`use_l1=True`)

`Solver.label(..., use_l1=True)` swaps the L2 parabolic Voronoi expand for
an L1 (Manhattan) chamfer (Rosenfeld–Pfaltz two-pass) and uses an
unpadded directly-on-row-major adjacency scan instead of the
padded-buffer roundtrip the L2 path needs. Bench at 2048², all values minimums:

| host | L2 default | L1 chamfer | speedup |
|---|---:|---:|---:|
| Mac M-series (T=16) | 19.49 ms | **7.33** | **2.7×** |
| Threadripper (T=16) | 23.56 | **12.82** | 1.8× |
| Threadripper (T=64) | 17.80 | 13.34 | 1.3× |

T=16 is the sweet spot for L1 on threadripper — the chamfer's row-band
partitioning + boundary-fixup serialises at the inter-band boundary, so
adding more threads means more boundaries to fix up serially. The L2
parabolic path keeps scaling further because each row's envelope build is
fully independent.

Trade-off: L1 boundaries differ from L2 at ~5% of pixels (mostly on
diagonals between adjacent regions). The resulting label adjacency graph
is essentially the same so the 4-coloring still works (verified — both
return valid 4-colorings with `n_used=5` on the 2048² synthetic test);
just don't expect bit-identity with `numba.ncolor.label` when `use_l1=True`.

### Pattern across all hosts

The Solver wins decisively at small images (where the per-call overhead
of numba's prange dispatch is unamortized) and ties or modestly loses at
2048² where both implementations are bound by the same algorithmic cost
of the parabolic-envelope build. **The shrinking-ratio-with-size pattern
is structural, not a regression** — once the actual algorithm work
exceeds the dispatch overhead, two well-tuned implementations of the same
algorithm necessarily converge.

The C++ Solver's primary value is therefore at small/medium image sizes
(interactive use, viewer pipelines) and on macOS (where numba can only
build `workqueue`). On healthy single-CCD Linux boxes (AMD/Intel) at
2048+, numba's @njit + omp is highly competitive and sometimes faster.

## Optimizations vs. numba's @njit

C++-side wins applied to `expand.hpp`:

1. **Divisionless pop comparison** in phase-1: `sv > z[top]` rewritten as
   `numer > z[top] * denom` (denom always positive), saves one FP divide
   per stack pop. Only the final break iteration computes `sv = numer/denom`.
2. **Pre-stored doubles** in stack scratch (`vd[k] = double(v[k])`,
   `vd_sq[k] = vd[k]*vd[k]`) so the data-dependent while loop avoids
   per-iteration int→double + extra multiply.
3. **Segmented phase-2 fill**: each parabola's domain is a contiguous range
   `[ceil(z[j]), ceil(z[j+1]))` with `lblstk[j]`, `g[j]`, `v[j]` lifted to
   loop invariants; lets the compiler emit clean NEON/AVX2 vector stores
   for the per-segment fill that the original interleaved while form blocked.
4. **Persistent threadpool**: workers live for the engine lifetime, so per
   call we pay only `enqueue + condition_variable::notify` (microseconds),
   not pthread_create. Critical on threadripper where every parallel
   region in numba would otherwise pay 14–43 ms of fan-out.
5. **Persistent scratch buffers** on `ExpandBuffers` (envelope stack +
   double-stack + transpose buffers): expand_labels gets called with the
   same shape repeatedly, so allocations amortize to zero.

C++-side wins applied to `connect.hpp`:

6. **Parallel pairwise tree merge** of per-thread hashtables. The numba
   version had this as an O(n_threads × ht_size) serial reduction at the
   end; the tree merge runs the reduction in log₂(n_threads) parallel rounds,
   eliminating the linear-in-thread-count term.

## Open work

- Port `_kempe_repair_csr` if any real workload turns out to need it
  (none have so far).
- Within-row parallelism for `envelope_pass` (chunk-and-merge) — would
  help short-wide images where row count is much less than the thread count.
- AVX-512/NEON 8×8 transpose intrinsics — `batch_transpose` is currently
  a tiled scalar loop; intrinsics would shave a couple ms at 4K+.
