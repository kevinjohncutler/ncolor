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

The C++ engine auto-calibrates its thread count once per machine
(~50–300 ms hidden under the user's first `import ncolor`) and caches
the result. Skip calibration with `NCOLOR_NO_CALIBRATE=1` (CI /
cross-compile builds).

### Kernel files (under `cpp/`)

| file | role |
|---|---|
| `expand.hpp`, `expand_lp.hpp`, `chamfer.hpp` | ND Lp Voronoi expand (Saito-Toriwaki L1 + Felzenszwalb L2) |
| `expand_clean.hpp` | Antipodal-bridge test + despur cascade fused with expand (the default `"clean"` mode) |
| `connect.hpp`, `connect_with_face_count.hpp` | `find_pairs` adjacency scan (dual-emit hard + soft) |
| `cc_label.hpp` | Connected-components labeling (drop-in for `skimage.measure.label`) |
| `format_labels.hpp` | Compact non-sequential labels to `1..N` |
| `color.hpp` | BFS coloring + Welsh-Powell + repair |
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
