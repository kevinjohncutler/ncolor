# ncolor architecture

Developer notes on the C++ engine, kernel layout, and runtime
machinery. End-users don't need any of this — `pip install ncolor`
gives you a precompiled wheel and the public API in
[README.md](README.md) is the whole story. This document is for
contributors and anyone debugging an editable / NAS-mounted install.

## C++ engine

The engine in `ncolor._backend` is the only backend, built from
`cpp/binding.cpp` into a single pybind11 extension
`ncolor._backend._impl`. It owns a persistent thread pool and runs

```
expand → find_pairs → color → soft post-pass → apply_lut
```

end-to-end under one `gil_scoped_release`.

`Solver.color_graph` enters that pipeline at the `color` stage: it takes
an edge list instead of an image, so expand / find_pairs / apply_lut are
skipped and the picker plus the soft post-pass run unchanged. That is
what lets the vector front end (`ncolor.geo.label`, adjacency from
Shapely rather than a pixel walk) reuse the engine as-is.

The C++ engine auto-calibrates its thread count once per machine
(~50–300 ms hidden under the user's first `import ncolor`) and caches
the result. Skip calibration with `NCOLOR_NO_CALIBRATE=1` (CI /
cross-compile builds).

The package holds two engine objects for the life of the process, a
`Solver` (label / connect / color_graph) and an `ExpandEngine`
(expand_labels / format_labels), created on first use in
`ncolor._engines`. Engines that resolve to the same thread count share
one `ForkJoinPool`. Engine calls must not overlap: the Python wrappers
take one process-wide lock, and each engine method also takes a C++
mutex inside its GIL-released region as a backstop for direct
`ncolor._backend` callers (the lock is taken only after the GIL is
dropped, so a waiter can never deadlock a thread that needs the GIL to
finish). Both engines keep the working set of the largest image they
have seen; `ncolor.release_buffers()` frees it.

The headers under `cpp/` have no Python dependency; only `binding.cpp`
does. `cpp/ncolor.hpp` includes all of them and
`cpp/example_standalone.cpp` runs the pipeline from plain C++; see
[cpp/README.md](cpp/README.md).

### Kernel files (under `cpp/`)

| file | role |
|---|---|
| `expand.hpp`, `expand_lp.hpp`, `chamfer.hpp` | ND Lp Voronoi expand (Saito-Toriwaki L1 + Felzenszwalb L2) |
| `expand_clean.hpp` | Antipodal-bridge test + despur cascade fused with expand (the default `"clean"` mode) |
| `connect.hpp`, `connect_with_face_count.hpp` | `find_pairs` adjacency scan (dual-emit hard + soft) |
| `cc_label.hpp` | Connected-components labeling (drop-in for `skimage.measure.label`) |
| `format_labels.hpp` | Compact non-sequential labels to `1..N`; range-checked parallel casts from bool, any integer width and float |
| `color.hpp` | BFS coloring + Welsh-Powell + repair |
| `picker.hpp` | The coloring picker: the per-color-count race of strategies that `label` and `color_graph` run |
| `bb_dsatur.hpp` | Iterative branch-and-bound exact DSATUR for the race |
| `tabucol.hpp`, `hea.hpp`, `kempe_sa.hpp`, `clique_lb.hpp` | Picker fallbacks: TabuCol, HEA, Kempe SA, clique lower bound |
| `soft_color.hpp` | Soft-edge local search (ILS + triangle weights) |
| `delete_spurs.hpp`, `delete_spurs_labels.hpp`, `fast_despur.hpp` | Skeleton spur / 1-voxel-thick bridge removal |
| `geometry.hpp`, `dispatch.hpp` | ND helpers + dtype dispatch |
| `threadpool.h` | Persistent fork-join pool with wait-on-address idle |

### Why the engine is structured this way

A few choices that aren't obvious from the code itself:

1. **Persistent thread pool.** Workers live for the engine's lifetime,
   so per call we pay only `enqueue + condition_variable::notify`
   (microseconds), not `pthread_create`. The earlier numba
   `@njit(parallel=True)` model paid 14–43 ms of fan-out per parallel
   region on high-thread-count x86 hosts; the C++ pool avoids that
   structurally.
2. **Kernel wait-on-address idle path.** After a short spin window,
   idle workers park via `__ulock_wait` (macOS) / `futex` (Linux) /
   `WaitOnAddress` (Windows). Replaces both the earlier `yield()`-based
   loop (which caused a `swtch_pri` storm consuming ~1700–1800% CPU
   across idle workers on macOS) and the intermediate `sleep_for(5ms)`
   fallback (whose ~10 ms macOS scheduler-tick rounding regressed small
   parallel jobs by 60–100×). Wake latency is now µs-class.
3. **Persistent scratch buffers** on `ExpandBuffers` (envelope stack,
   double-stack, transpose). `expand_labels` gets called with the same
   shape repeatedly across a session, so allocations amortize to zero.
4. **Divisionless pop comparison** in the Felzenszwalb envelope phase-1:
   `sv > z[top]` rewritten as `numer > z[top] * denom`, saving one FP
   divide per stack pop. Only the final break iteration computes
   `sv = numer/denom`.
5. **Segmented phase-2 fill** in the envelope: each parabola's domain
   is a contiguous range `[ceil(z[j]), ceil(z[j+1]))`. Lifting
   `lblstk[j]`, `g[j]`, `v[j]` to per-segment loop invariants lets the
   compiler emit clean NEON/AVX2 vector stores that the original
   interleaved while-form blocked.
6. **Parallel pairwise tree merge** of per-thread `find_pairs`
   hashtables. The reduction runs in `log₂(n_threads)` parallel rounds
   instead of the linear `O(n_threads × ht_size)` serial merge the
   numba version used.
7. **Dual-emit `find_pairs`** kernel
   (`find_pairs_dual_nd_unpadded`). Emits both base and delta (soft)
   pairs in a single cache-warm pixel walk, dropping the auto-soft
   build cost from a second full scan to ~the difference between the
   soft and hard offset counts.
8. **`bb_dsatur` is iterative.** Recursive backtracking blew the
   512 KB macOS worker-thread stack at `N ≥ 3000`; the heap-allocated
   state stack is safe for graphs of any size.
9. **Range-checked casts.** Every entry point works on int32 labels.
   Inputs that can hold larger values (int64, uint32, uint64, float)
   are checked while they are cast, one compare per element folded
   into a pass that is memory-bound anyway. A value that does not fit
   raises `OverflowError` from the engine; `label` and `format_labels`
   then compact the array with `numpy.unique` and retry. Before this,
   a label of 2^31 wrapped negative, the format pass took it for
   background, and the cell silently vanished.
10. **SIMD by target, not by hand-picked flag.** The envelope fill and
    the 4x4 transpose have NEON (arm64), AVX2 (8-wide, when the build
    targets it) and SSE (4-wide, any x86_64) variants. The SSE lane
    multiply is one instruction from SSE4.1 up and an SSE2 emulation
    below, so the plain-baseline and MSVC builds get the vector path
    too; the x86_64 wheels are built for x86-64-v2, the floor NumPy 2
    already assumes.
11. **No debug info in the wheels.** Python's own CFLAGS carry `-g`;
    the manylinux `.so` was 27 MB of which 25 MB were `.debug_*`
    sections. `-g0` plus `--strip-all` on Linux bring the wheel from
    6 MB to under 1 MB with no effect on the generated code.
12. **The bridge scan walks lines, not pixels.** The subspace the
    antipodal test runs over is always the trailing axes, so its last
    axis is the unit-stride axis and every chunk is a whole number of
    lines. Along a line only the innermost coordinate changes, so which
    neighbor offsets stay in bounds is fixed except at the two end
    pixels. The interior run therefore needs no bounds checks: its
    same-label face count is accumulated one offset at a time in a
    loop the compiler vectorizes, and neighbors are re-read only for
    pixels whose count is exactly two (cell boundaries). The end pixels
    and lines shorter than three go through the generic path. Nothing
    depends on the number of axes; validity comes from the offset
    vectors. Per-pixel bounds-check loops were what made this pass 3 to
    5 times the cost of the plain expand on Apple Silicon. The merged
    bridge queue is sorted before the peel-back so the result does not
    depend on which thread finished first.
13. **The soft adjacency scan skips cell interiors.** In the fused
    hard-plus-soft `find_pairs` the delta offsets are ordered by
    Chebyshev distance. When every distance-1 forward neighbor of a
    pixel carries its own label, the distance-2 offsets are not read:
    for any far pixel q with a different label, the neighbor m one
    step toward q has the pixel's label and sits within distance 1 of
    q, so the pair (m, q) is emitted from m or q, either as a hard pair
    (face offset) or as a distance-1 soft pair, and m or q is itself
    not interior. Hard pairs are removed from the soft set at the end,
    which makes the two routes equivalent. The argument needs the
    stepping neighbor to carry the pixel's label, which holds for a
    radius of 2 but not beyond (a chain through a third label), so the
    driver disables the skip for `soft_radius > 2`. For a cell interior
    this is 4 reads instead of 12 in 2D and 9 instead of 33 in 3D.

### Scaling pattern across image sizes

The C++ engine wins decisively at small / medium images (`≤ 1024²`)
where the per-call overhead of `@njit(parallel=True)`'s prange dispatch
dominates the actual algorithm work. At `2048²+` both implementations
are increasingly bound by the parabolic-envelope build itself and the
ratio narrows — this is structural, not a regression. Two well-tuned
implementations of the same algorithm converge once the algorithm work
dominates dispatch. The C++ engine's primary value is therefore at
small-to-medium sizes (interactive use, viewer pipelines) and on macOS
(where numba could only build the `workqueue` layer).

### Where the remaining time goes

Measured on an idle i9-9900K, where repeated runs agree to 0.1 ms, and
on an M5 Max. Figures below are the M5.

| case | total | expand | find_pairs | of which the soft kernel |
|---|---|---|---|---|
| 2D 2048² | 6.2 ms | 3.6 | 1.1 | 0.3 |
| 3D 256³ | 59 ms | 27 | 23.0 | 19.6 |

Two things dominate. Expand runs at 30–37 GB/s effective, near what
these machines give for a separable sweep over two int32 arrays, so it
is bandwidth-bound rather than badly written. The soft kernel is the
other: 85% of `find_pairs` in 3D and a third of total `label` time,
because the default `soft_conn=2, soft_radius=2` reads 33 offsets per
voxel against 3 for the hard graph.

A finer version of the interior skip was tried on the soft kernel and
reverted. Instead of skipping every distance-2 offset only when all
distance-1 neighbors match, each far offset can be skipped on its own
whenever the neighbor one step along it matches; the pair it would emit
is emitted from that neighbor instead. The rule is sound (checked
against exhaustive enumeration on 120 random images, and by diffing
full pair sets over 426 image and kernel combinations), but it needs a
runtime bound inside a loop the compiler currently unrolls with
constant offsets. Across four implementations it bought up to 12% of
3D `find_pairs` while costing 20–35% of the much smaller 2D one, so it
never became a clean win. Anyone resuming it should keep the offset
counts on both sides of the near/far split compile-time constants so
both loops stay fully unrolled.

## Calibration cache & NAS-mounted source

`ncolor` stores two things in the per-user cache directory resolved by
[platformdirs](https://pypi.org/project/platformdirs/) at
`platformdirs.user_cache_dir("ncolor")`:

1. `smt_threads.json` — the per-host SMT/HT-aware thread count chosen
   by `ncolor._backend._smt.calibrate()`. Keyed by `(hostname, CPU
   model)`. Read on every `Solver()` / `ConnectEngine()` /
   `ExpandEngine()` construction via `auto_threads()`.
2. `lib/<mtime_ns>_<size>/_impl.<so|pyd>` — only used when the package
   directory is on a network filesystem. The compiled extension is
   copied here and `dlopen`'d locally. See below.

Both paths resolve to the OS-native location:

| OS      | Path                                  |
|---------|---------------------------------------|
| Linux   | `~/.cache/ncolor/`                    |
| macOS   | `~/Library/Caches/ncolor/`            |
| Windows | `%LOCALAPPDATA%\ncolor\ncolor\Cache\` |

### Why the local-disk `.so` cache exists

If you `pip install ncolor` from a wheel (the normal path), the
compiled extension lives in `site-packages/`, which is on local disk
on every reasonable system. The loader in `ncolor._backend.__init__`
fast-paths to a direct `importlib` load and never touches the
platformdirs cache.

The cache only kicks in when the **package source itself lives on a
network filesystem** — i.e., a developer install (`pip install -e .`
or `setup.py build_ext --inplace`) where the source tree is
NAS-mounted (`smbfs` / `nfs` / `UNC`). Two OS-level bugs break direct
loading in that case:

- **macOS smbfs** — `dyld` calls `fcntl()` for code-signature
  validation during `dlopen`, and SMB hangs on those calls; `dlopen`
  blocks indefinitely in `JustInTimeLoader::withRegions`.
- **Windows UNC** — `LoadLibrary` raises *Access is denied* on certain
  server configurations (depends on the share's ACLs).

The loader detects this case (`smbfs` / `nfs` / `afpfs` on POSIX,
`UNC` anchor on Windows) and copies the `.so` / `.pyd` to the local
cache before `dlopen`'ing from there. Cache key is `(mtime_ns, size)`
so a rebuild gets a fresh local path — `dyld` retains stale
path-keyed state from prior failed loads at the same path, so reusing
the same path can still hang. On macOS the loader also strips
`com.apple.quarantine` from the copy.

Net effect: end-users with pip-installed wheels never see this code
path. Developers running an editable install from a NAS-mounted source
tree get transparent local-cache `.so` loading without doing anything.

To clear caches manually:

```bash
# macOS
rm -rf ~/Library/Caches/ncolor

# Linux
rm -rf ~/.cache/ncolor

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\ncolor"
```

The next `import ncolor` will recalibrate (~250 ms) and recopy the
`.so` if needed.
