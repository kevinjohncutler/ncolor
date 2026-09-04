# Changelog

All notable changes to ncolor are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [semantic versioning](https://semver.org/).

## [2.1.0] — unreleased

### Added

- **Vector-geometry support (`ncolor.geo.label` / `ncolor.geo.connect`).**
  Colors polygons directly, so data that starts out as vectors is never
  rasterized to get a coloring. Accepts a GeoPandas `GeoDataFrame` or
  `GeoSeries`, a GeoJSON file / string / `dict`, a list of Shapely
  geometries, or any object with a `__geo_interface__`, and returns a
  `uint8` color per feature in input order (0 for missing or empty
  geometries), or a `GeoDataFrame` with `return_frame=True`.
  Adjacency is read off the geometry with a Shapely `STRtree`:
  - `min_shared_length` (default `0.0`) requires a shared *border*
    rather than a point of contact, so corner-only touches do not
    constrain the coloring. That is the raster `conn=1` rule in vector
    form, and it is what keeps a planar partition 4-colorable. `None`
    counts point contacts and then 5 colors are likely.
  - `tolerance` treats features within a distance of each other as
    touching, for polygons left with hairline gaps by vectorization or
    reprojection. The analogue of `connect_radius`.
  - Pairs the filters reject are fed to the soft-constraint pass, so
    near-misses still differ in color where that is free.
  Requires Shapely 2.0+: `pip install "ncolor[geo]"`. GeoPandas is
  optional; GeoDataFrames are recognized by duck-typing.
  Resolves [#2](https://github.com/kevinjohncutler/ncolor/issues/2).
- **`ncolor.color_graph(edges, n_vertices=...)`.** The picker, decoupled
  from any image: colors an abstract graph from a 0-indexed `(M, 2)`
  edge list, with optional `soft_edges`. Backs `ncolor.geo.label`, and is the
  entry point for adjacency built by anything else (a mesh, a region
  adjacency graph from another library). Duplicate, reversed, self and
  out-of-range pairs are normalized away. Exposed on the C++ engine as
  `Solver.color_graph`.
- **Bool, float and uint64 inputs.** `label`, `connect`, `format_labels`,
  `expand_labels` and `connected_components` accept `bool` masks,
  `float32` / `float64` label arrays (as segmenters such as cellpose
  emit; values are truncated toward zero) and `uint64`, alongside the
  integer dtypes already supported. The cast to the engine's int32 runs
  in parallel inside the call; `expand_labels` no longer pays a
  single-threaded `numpy.astype` pass first.
- **`ncolor.label` threads by itself, and `ncolor.Engine` for doing it
  by hand.** Every call shared one thread pool, and only one call may be
  in flight per pool, so calls from several threads took turns:
  threading four images through the module-level functions ran no faster
  than doing them one after another (measured 1.03x). Now a lone call
  gets the full-width engine as before, while calls that overlap are
  handed narrower engines whose threads add up to about one machine.
  Four threads through plain `ncolor.label` run 1.3 to 2.2x faster than
  the same work in sequence on an 18-core machine, with nothing asked of
  the caller; when the parallel phase ends, calls go back to full width.

  Two things had to be true for that to pay off, and both were measured
  rather than assumed. Running the full-width engine alongside the
  narrow ones is the worst of both and was *slower* than serial (0.65x),
  because the pools spin against each other, so the wide one sits out
  while anything else is in flight. And an engine is bound to its thread
  rather than borrowed per call: handing one lock between four threads
  cost 2.3 ms a call, several times the work itself.

  Nothing is measured or configured for this, and that took some
  finding out. Overlapping calls are always given engines of their own.
  Whether that beats taking turns was swept over four machines and eight
  image sizes, and the better arrangement turns out to be a property of
  the image at least as much as of the machine: on an 8-core i9 it ran
  1.3x faster at 512 by 512, 0.8x at 1024, and 1.1x at 4096, and on a
  16-core Ryzen 2.4x at 512 and 0.9x at 2048. Any verdict measured once
  and cached is therefore fitted to whichever size was measured. An
  earlier version of this release did exactly that and cached the wrong
  answer for two of the four hosts, because it measured at 1024, which
  is near their worst case. Averaged over sizes, splitting pays on every
  machine tested: 1.07x on the i9, 1.3x on the Ryzen, 1.7x on the M5 Max
  and 1.9x on the Threadripper, with a worst single case of 0.76x.
  `NCOLOR_MAX_ENGINES=1` turns it off and restores taking turns.

  Engines are built only when calls actually overlap, so a
  single-threaded program still holds exactly one. At most
  `NCOLOR_MAX_ENGINES` (4 by default) are created, which bounds the
  worker threads and the memory alike, since each keeps the working set
  of the largest image it has seen (roughly 20 bytes per pixel).
  `ncolor.Engine` is the same thing under the caller's control, for
  sizing the threads by hand or running more workers than the limit.
- **`ncolor.release_buffers()`.** The engines keep the working set of the
  largest image processed so far (about 22 bytes per pixel for `label`)
  allocated between calls. After one whole-slide image that is
  gigabytes of resident memory; this gives it back. The thread pool is
  kept.
- **Wheels for CPython 3.15 and for Intel Macs** (`macosx_x86_64`), and
  the Windows ARM64 wheel is now documented.
- **`bench/bench_hosts.py` and `bench/run_host_bench.sh`.** A cross-host
  regression benchmark that records per-stage timings with host, CPU,
  thread count and compiler metadata, builds the tree on a remote
  machine and compares two tags. Every performance change below was
  checked with it on an Intel i9-9900K, a Ryzen 9 7950X, a Threadripper
  PRO 3995WX, an M1 Ultra, an M5 Max and a Windows 11 VM.

### Changed

- **The clean expand's bridge scan is 5 to 12x faster, and its output
  no longer depends on the thread count.** The antipodal-bridge scan
  that `expand_mode="clean"` runs after each axis sweep walked every
  pixel with a per-neighbor bounds-check loop and a mixed-radix
  coordinate counter: branchy scalar code that Apple Silicon in
  particular ran poorly (the clean expand cost 3 to 5 times the plain
  one on an M5 Max and an M1 Ultra, against 1.1 to 1.35 times on x86).
  The scan now walks the image one line at a time: which neighbor
  offsets stay in bounds is settled once per line, the interior of the
  line accumulates its same-label face count in a straight loop that
  vectorizes on NEON and SSE alike, and neighbors are re-read only for
  the few pixels whose count is two. Still one N-dimensional
  implementation driven by the offset tables, with no per-dimension
  special case, and bit-identical output on 400 test images. Because
  threads claim scan chunks in arrival order, the merged bridge queue
  used to vary run to run and the peel-back cascade with it (a few
  pixels per image); the queue is now sorted, so the parallel result
  equals the serial one. `label` at 2048 squared went from 12 to 6 ms
  on an M5 Max and from 38 to 10 ms on an M1 Ultra; the x86 hosts gain
  7 to 29% on `label`. The barrier-aware L2 sweep also gained the
  strided-slab path the plain sweep already had for 3D, and a header
  edit now triggers a rebuild (`setup.py` lists the headers as
  dependencies).
- **The soft-kernel adjacency scan skips cell interiors.** The fused
  hard-plus-soft `find_pairs` read every soft-radius offset for every
  pixel: 12 reads per pixel in 2D and 33 in 3D with the default
  `soft_conn=2, soft_radius=2`. A pixel whose distance-1 neighbors all
  carry its own label now skips the distance-2 offsets: any pair those
  could yield is also seen from the neighbor one step toward the far
  pixel, as a hard pair or as a distance-1 soft pair, so the pair set
  is unchanged (verified on 426 image and kernel combinations,
  wrap-around included). Soft pairs that are also hard pairs are now
  dropped from the soft set; they can never be violated and only
  distorted the soft weights (color counts and soft-violation counts on
  the reference images are unchanged). `find_pairs` at 2048 squared
  went from 2.1 to 0.9 ms and at 256 cubed from 33 to 21 ms on an M5
  Max. The skip is exact only up to soft radius 2 and is disabled
  automatically beyond that; `NCOLOR_NO_INTERIOR_SKIP=1` disables it
  for comparisons. The clean expand also keeps its per-pixel
  neighbor-count buffer between calls instead of allocating and zeroing
  it on every call, and its barrier-aware fill is a masked SIMD loop.
- **One thread pool for the whole package.** `Solver` and
  `ExpandEngine` instances that resolve to the same thread count share
  a single `ForkJoinPool`, and the format engine is the expand engine.
  A process that touched `label`, `expand_labels` and `format_labels`
  used to hold three pools (52 parked threads on an 18-core machine);
  it now holds one.
- **x86_64 wheels are built for x86-64-v2** (SSE4.2 / POPCNT, CPUs from
  2009 on, the same floor NumPy 2 assumes) and carry no debug info. The
  Linux wheel was 27 MB of which 25 MB were `.debug_*` sections; it is
  now under 1 MB. The SSE fill in the expand kernel previously compiled
  out of every x86 wheel (it was gated on SSE4.1, which the baseline
  build never defines and MSVC never defines at all); it is now on for
  every x86 build, with an SSE2 fallback for the lane multiply, an AVX2
  8-wide variant when the build targets it, and an SSE 4x4 in-register
  transpose matching the NEON one. Expand is 5 to 35% faster on the
  x86 hosts above; arm64 is unchanged.
- **Windows core count without `wmic`.** Microsoft removed the WMIC
  utility from Windows 11 24H2. The SMT calibration read the physical
  core count and CPU model through it and fell back to the logical
  count when it was missing, so recent Windows installs ran
  SMT-doubled, the case the calibration exists to avoid. The count now
  comes from `GetLogicalProcessorInformationEx` and the model from the
  registry.
- **`Solver(fraction)` rounding** is documented as round-half-up; the
  test that assumed Python's half-to-even rounding was wrong on hosts
  where the product lands on .5.

### Fixed

- **`label` returns the same coloring every time.** The picker races
  several searches per color count and keeps the lowest-numbered one
  that succeeds, but it abandoned every other search the moment any of
  them landed. That let the thread schedule pick the winner: a
  low-numbered search that would have won was cut off by a high-numbered
  one that happened to finish first, so the same image came back with
  different, equally valid, colorings from one call to the next. It
  showed up as flaky output on a busy machine and only above one thread.

  A search is now abandoned only once a *lower-numbered* one has
  succeeded, which by definition cannot change the answer. The early
  exit is kept, so the race still ends as soon as the lowest slot is
  settled rather than running all sixteen to their budgets.

  Measured against the same tree without the change, 416 paired
  measurements over four machines: 1.004x, which is no change. It is not
  free everywhere, though, and the exception is worth knowing. The
  answer is now the lowest-numbered search that succeeds, so the race
  cannot finish until every search below the winner has been decided.
  Where a low search is slow to *fail* while a higher one succeeds
  quickly, that costs: on one input in the benchmark corpus, `label`
  went from 3.0 to 3.6 ms, and the same case regressed on all four
  machines (0.77x to 0.84x). Instrumented, slot 3 wins it in 0.1 ms but
  slots 1 and 2 each take 0.8 ms to fail. Every other measurement,
  including the hard random graphs that make the picker work hardest,
  is unchanged.

  `ncolor.label` is now bit-identical across machines: the same twenty
  images gave byte-identical colorings, and the same number of colors,
  on an Apple M5 Max, an Intel i9-9900K, an AMD Ryzen 7950X and a
  Threadripper 3995WX, at each machine's own thread count (8, 16, 18 and
  64) and at a fixed four. One thread is the exception, and by design:
  it runs the picker's sequential path, a different algorithm rather
  than a different schedule.

- **One `label()` call could inherit the previous call's soft
  constraints.** The engine is a process-global singleton, and the soft
  pair list was cleared by the branches that build one rather than at
  the start of the call. A call using `weight_objective` or
  `min_contact` therefore reused the pairs of whatever image ran
  before; those are label ids of a different image, and the ones that
  happened to fall inside the new label range were applied as soft
  constraints. Out-of-range ids were already dropped, so this was never
  a memory error, but the coloring of the second image depended on the
  first. The list is now cleared before the branch runs.
- **Hard edges could be dropped when the soft kernel did not contain
  the hard one.** The fused scan enumerates one offset set at the soft
  connectivity and radius and marks the subset that is hard, which
  cannot represent a hard kernel with offsets outside the soft one
  (`conn=2, connect_radius=1` against `soft_conn=1, soft_radius=2`, and
  the mirror case). Those combinations now scan each kernel
  independently and subtract the hard pairs from the soft set; the
  fused path is used only when the soft kernel contains the hard one.
- **The fused scan now retries when a hash table fills.** The
  single-kernel scan already doubled its table and rescanned; the fused
  path did not, so a graph with more distinct adjacencies than the
  initial estimate could silently lose pairs. Each table's occupancy is
  reported separately and only the one that filled is doubled.
- **Out-of-bounds read in `wrap=True` adjacency scans** when an axis is
  shorter than the neighbor radius (a `connect_radius` or `soft_radius`
  of 2 on an image with a dimension of 1 or 2): the wrap-around was a
  single subtraction rather than a modulo, so the index could stay
  negative. Found by the pair-set harness; such inputs now wrap
  correctly.
- **Crash on concurrent `expand_labels` / `format_labels`.** 2.0.1
  serialized `label` and `connect`, but the expand and format engines
  were left unguarded and two threads calling them at once corrupted
  the shared pool and took the interpreter down with SIGSEGV. Every
  engine call now takes one process-wide lock in Python, and the C++
  engines hold a mutex of their own inside the GIL-released region as
  a backstop for callers of `ncolor._backend` directly. The
  parallelism inside a call is unaffected.
- **Labels at or above 2^31 silently vanished.** An `int64` or `uint32`
  (or `uint64` / float) array holding such a value was cast to int32
  with a plain `static_cast`; the wrapped value went negative, the
  format pass treated it as background, and whole cells disappeared
  from the output with no error. The cast is now range-checked at no
  extra cost; `label` and `format_labels` compact such inputs with
  `numpy.unique` and retry, and the identity-preserving operations
  (`connect`, `expand_labels`, `delete_spurs`) raise `OverflowError`
  with instructions instead.
- **Hang when the soft pass runs on a one-color palette.** If the hard
  adjacency graph has no edges at all but the soft graph does, the
  coloring is a single color and the soft search's restart loop draws
  "a color other than the current one" by rejection sampling, which
  never terminates when there is only one color to draw. Reachable from
  the shipped raster API: two cells touching only diagonally with
  `expand=False` have no `conn=1` adjacency, yet the auto-built
  `conn=2 / radius=2` soft kernel still emits their pair.
  `soft_local_search` now returns immediately when there are fewer than
  two colors to work with.

## [2.0.2] — 2026-06-19

### Fixed

- **3D Voronoi hang.** The `find_pairs` hashtable probe is bounded, so a
  pathological 3D input can no longer spin forever.

## [2.0.1] — 2026-06-07

### Fixed

- **Concurrent `label` / `connect` crashed.** The `Solver` is a
  process-wide singleton on one pool; calls from several threads are
  now serialized.

## [2.0.0] — 2026-06-02

The headline of 2.0 is a new default expand pipeline (`expand_mode="clean"`)
and an opt-out auto-soft constraint post-pass that together produce
cleaner 4-colorings on real microscopy data. A small number of kwargs
were removed; the major bump captures that surface change.

### Added

- **`expand_mode="clean"`** (now the default). ND Lp Voronoi sweep
  fused with an antipodal-bridge test and a despur peel-back cascade
  in one pass. Subsumes the older expand + despur chain; removes
  1-pixel bridges and ≤1-face stubs as graph barriers to prevent
  K_5-shaped convergence clusters that would otherwise block
  4-coloring.
- **Soft-constraint post-pass** for `ncolor.label()`. New kwargs
  `soft_conn`, `soft_radius`, `soft_extra_edges`:
  - `soft_conn=2, soft_radius=2` (new defaults) auto-build a soft
    kernel as the delta between the hard kernel `(conn,
    connect_radius)` and the richer `(soft_conn, soft_radius)`.
    The picker's 4-coloring is then refined by a local search
    (greedy + Kempe chains + iterated local-search restarts on
    small graphs) that minimizes the count of soft-kernel edges
    sharing a color, without breaking the hard graph.
  - `soft_extra_edges=(E,2) int32` lets callers pass an explicit
    pair list instead of using the auto-build.
  - `solver.get_last_n_soft_violations()` exposes the residual
    soft-violation count.
- **`clean_mask` kwarg** (default `False`): output preserves the
  input mask's foreground / background pattern exactly. The clean
  expand's internal barrier removal stays a graph-only step; the LUT
  is applied to a pre-clean snapshot of the foreground labels. Set
  `clean_mask=True` for the old behavior where barrier zeros surface
  in the output too (useful as a clean + label combined op).
- **`verbose=True`** writes a structured stage-breakdown summary
  (shape, n_used, residual soft violations, total ms, per-stage
  timings) to stderr after the call. 1.5.x's `verbose=True` printed
  a couple of ad-hoc progress lines to stdout — same intent, different
  content and stream.
- **Dual-emit `find_pairs`** kernel (`find_pairs_dual_nd_unpadded` in
  `cpp/connect.hpp`). Emits both base and delta pairs in a single
  pixel walk via a per-pixel routing inner loop. Drops the cost of
  the soft auto-build from a second full scan to ~the difference
  between the soft and hard offset counts.
- **`bb_dsatur` is iterative.** Recursive backtracking blew past the
  512 KB macOS worker-thread stack at N≈3000+; the new explicit
  heap-allocated state stack is safe for graphs of any size.
- **`expand_labels(mode="clean")`** in the Python `expand.py`
  wrapper for callers that want to expand without coloring.

### Changed

- **Dropped Python 3.10 support.** Python 3.10 reached EOL on
  2025-10-31; NumPy 2.3+ already requires 3.11+ and the rest of the
  scientific Python stack is following. The new floor is Python 3.11
  (`requires-python = ">=3.11"`). 1.5.3 wheels remain available on
  PyPI for users still on 3.10.
- **Default `conn=1` and `p=2`** (were `conn=2, p=1`). These match the
  ncolor 1.x defaults and the configuration the clean-expand + auto-soft
  stack was designed around. At the old `conn=2, p=1` the picker hit
  K_5 obstructions on dense microscopy data and fell back to n=5 (62 ms
  on the mm 2k² fixture); the new defaults reach n=4 in ~25 ms and
  achieve K_4=4 on the logo without per-call tuning.
- **Default `expand_mode` is now `"clean"`** (was the pre-2.0 numba
  default, plain Voronoi expand — now selectable as
  `expand_mode="standard"`).
- **Default `soft_conn=2, soft_radius=2`** turns on the auto-soft
  post-pass. Set both to `0` to disable.
- **Threadpool idle wait** now uses a kernel wait-on-address
  primitive (`__ulock_wait` on macOS, `futex` on Linux, `WaitOnAddress`
  on Windows) after a short spin window, replacing both the earlier
  `yield()`-based loop (which caused a `swtch_pri` storm consuming
  ~1700–1800% CPU across idle workers on macOS) and the intermediate
  `sleep_for(5ms)` fallback (whose ~10 ms macOS scheduler-tick
  rounding regressed small parallel jobs by 60–100×). Wake latency is
  now µs-class on every host.

### Removed

- **`offset` kwarg** — in 1.5.x this seeded the BFS picker's RNG
  (`seed = ... + offset`) and the per-attempt starting offset
  (`attempt_offset = offset + attempt`), letting callers ask for a
  different valid coloring of the same input. 2.0's picker races 16
  attempts per `cur_n` internally with offsets `local_depth + idx` and
  takes the lowest-index success, so a single user-controlled seed is
  no longer meaningful. No direct replacement; use `first_seen=True`
  or change `p` if you need a different valid coloring of the same
  graph.
- **`greedy` kwarg** — 1.5.x routed `greedy=True` to a pure-greedy
  coloring path with no repair step. 2.0's picker always does
  BFS + repair + tabucol + bb_dsatur race; there's no "skip the repair"
  shortcut. Drop the kwarg.
- **`experimental` kwarg** — 1.5.x routed `experimental=True` to an
  alternative `render_net_experimental` algorithm. That algorithm was
  not ported to 2.0 and would have been redundant with the
  multi-strategy race anyway. Drop the kwarg.
- **`despur_iters`** and **`despur_remove_thin`** — subsumed by the
  `"clean"` expand mode (now default), which includes its own fused
  barrier removal. The standalone despur pass was a no-op on top.
- **`expand_spur_free`** and **`spur_free_max_rounds`** — legacy
  aliases that pointed to the now-removed `expand_mode="spur_free"`.
- **`expand_mode="spur_free"`** entirely (and the standalone
  `ncolor._backend.expand_spur_free` binding). The BFS-dilation
  kernel built a strictly sparser contact graph than the new
  `"clean"` / `"standard"` modes by aggressively severing at "spur"
  pixels, which silently changed which adjacencies the picker saw —
  on `synthetic_800` the picker would happily return `n_used=3` with
  a very unbalanced [399, 233, 50] cell-count split, because the
  dropped edges made the problem 3-colorable in spur_free's graph.
  Outputs were correct relative to that graph but surprising to
  users who expected full Voronoi coverage.
  Passing `expand_mode="spur_free"` now raises
  `std::invalid_argument`. `cpp/expand_spur_free.hpp` is deleted.
- **`get_lut` public function.** A 5-line wrapper around
  `label(..., return_lut=True)` whose kwargs (and defaults) drifted out
  of sync with `label()`'s — by 2.0-pre it shipped a stale `conn=2`
  default with no `p` kwarg, silently producing different colorings
  than `label(m, return_lut=True)` on identical input. Drop it; use
  `label(..., return_lut=True)` directly. On the C++ pipeline the
  apply-LUT stage is 0.6 ms out of ~25 ms on mm 2k² (and faster than
  the equivalent numpy `lut[mask].astype(uint8)` anyway), so the
  numba-era "skip the apply step" rationale no longer applies.
- **`optimize` kwarg** (with its sole `"two_hop"` value) and the
  underlying `ncolor._optimize` module. The simulated-annealing
  Kempe-swap optimizer was pure Python and ran ~1300× slower than the
  rest of the pipeline (≈10 s on synth_800, scaling to minutes on
  mm-class data). It never had a C++ port; the module docstring
  flagged the gap. With auto-soft now doing global color-balance work
  in C++ at sub-ms-per-cell cost, the two_hop path was strictly
  dominated. `src/ncolor/_optimize.py` is deleted.
- **`format_labels(despur=True)` per-cell Python loop.** The slow
  bbox-cropped iteration over every cell — applying binary
  `delete_spurs` + `connected_components` + region-prop area
  filtering one label at a time — has been replaced by a single C++
  `delete_spurs_labels` pre-pass followed by the same
  `cc_label_per_label` component-splitting path the non-despur
  branch already uses. Behaviorally equivalent for the despur step
  (spurs and 1-voxel-thick interior bridges still removed) and
  faster on inputs with many cells. No public API change.
- **`balance` kwarg** — the Welsh-Powell visit-order path it gated was
  the picker's slot-0 warmup, and that warmup was disabled by default
  earlier in 2.0 dev. With the warmup off, `balance` was a silent no-op
  on every reference image (logo, synthetic_800, mm 2k² fov104) —
  byte-identical colorings at `balance=True` and `balance=False` —
  while auto-soft post-processing already produces strictly better
  color-balance (σ 7.30→2.87 on synth, σ 21.30→13.21 on mm). Slot 0 is
  now a regular BFS race entry like the rest, the warmup machinery
  (`NCOLOR_WARMUP_ENABLE`, `NCOLOR_WARMUP_DEBUG`) is gone, and the
  Welsh-Powell visit-order code path is no longer reachable.

### Fixed

- **`format_labels(clean=True)` min_area threshold is now symmetric.**
  The 1.5.x pipeline used `area <= min_area` for the primary component
  of each label and `area < min_area` for secondary (disjoint) parts.
  At exactly `min_area`, a primary was dropped but a secondary
  survived — same input geometry, opposite outcomes depending on which
  rank a component happened to land in. The threshold is now uniformly
  strict-less-than (`area < min_area`) across ranks: a component of
  exactly `min_area` pixels always survives, both as a primary and as a
  secondary. Matches the 1.5.x README documentation ("remove small
  labels (<9px)"), which was already the secondary-rank semantic.
  Boundary regression test added.
- **`de_table` user-supplied palette no longer segfaults.** The
  `(n+1)×(n+1)` palette override (used with `weight_objective != 0`)
  was being parsed via `py::array_t::ensure()` and `.request()`
  *inside* the GIL-released block, which is undefined behavior — any
  user-provided table reliably crashed the process. Parsing now
  happens before the GIL release, mirroring the pattern already used
  for `extra_edges` and `soft_extra_edges`. The built-in viridis
  default path was unaffected (it never touches Python objects).
- **Hash-table sizing for the soft find_pairs scratch buffer.** A
  hardcoded `1 << 20` HT was 8× larger than needed and dominated
  init / extract cost on small graphs. Now sized to
  `2 × n_fwd_delta × max_label`.
- **`kempe_force_flip` correctness** in the soft-search restart
  shake. A chain-size cap in the force-flip path was leaving
  dangling chain vertices and creating same-color hard edges
  (manifested as `sv_reported < sv_truth` mismatch on mm-class
  data). Force-flip now always completes the entire Kempe component;
  the cap remains only on the greedy `kempe_chain_pass`, which
  *skips* rather than partially applies.
- **`soft_local_search` public return value** is the unweighted
  violation *count*, not the triangle-weighted internal penalty.

### Performance

- **mm 2k² real microscopy (5128 cells, 14.6% fill):** ~25 ms at
  n=4 with auto-soft on — 8.6× faster than PyPI 1.5.3 (216 ms at
  n=5; the numba pipeline can't find a 4-coloring on this input).
  See README bench table and `bench/bench_vs_pypi.py`.
- **Logo K_4 cluster:** now reaches 4 distinct colors at n=4 via
  iterated local-search restarts + triangle-count edge weights in
  the soft search. Previous greedy single-Kempe was getting stuck
  at K_4=3 on the canonical 4-cell convergence cluster.
- **Right-sized soft HT + scratch reuse** save ~70 ms on mm.
- **Per-pixel dual-emit scan** saves ~14 ms on mm vs naïve two-pass.
- **Stuck-vertex worklist + 64-vertex Kempe chain cap** drop the
  soft post-pass from ~30 ms to ~5 ms on mm.
- **Stamp-based triangle counting** in `compute_triangle_weights` is
  4× faster than the nested-scan it replaced.

### Internal / tests

- Test suite: 330 passing, 3 skipped (count after the removal sweep
  that dropped `balance` / `two_hop` / `spur_free` parametrize
  matrices). Net suite is smaller than 2.0-pre but covers exactly
  the shipped surface.
- New tests added for the v2.0 surface: soft default behavior,
  explicit `soft_extra_edges`, `clean_mask` true/false semantics,
  soft-off fallback, `expand_labels(mode="clean")` p/metric coverage,
  `verbose=True` stage summary, `de_table` user-palette regression
  (segfault fix), removed-kwarg rejection tests.
- Added `.gitignore` entries for `_tmp_*` and `bench_outputs/` so
  per-session scratch and bench output directories stop polluting
  `git status`. Promoted five active benches to `bench/`.
- Legacy `test_files/test_ncolor.py` renamed to
  `_legacy_ncolor_tests.py` so pytest's default `test_*.py` glob
  doesn't auto-collect it from the fixtures directory.

### Migration notes from 1.5.x

The four real 1.5.x kwargs/functions removed in 2.0 are `offset`,
`greedy`, `experimental`, and the standalone `get_lut()` function. See
the **Removed** section above for the per-kwarg rationale. Calling any
of them now raises `TypeError: unexpected keyword argument`.

Two more kwargs (`despur_iters`, `despur_remove_thin`) and the
`expand_spur_free` / `spur_free_max_rounds` aliases were added during
2.0 development but never shipped to PyPI — 1.5.x users won't have
written code against them.

- The output of `ncolor.label(m)` is no longer bit-identical to 1.5.x
  for two reasons:
  1. Default `expand_mode` flipped from plain Voronoi to `"clean"`
     (Voronoi expand + bridge/spur barrier removal).
  2. Default soft-constraint post-pass is on. To recover bit-equivalent
     1.5.x behavior pass:

     ```python
     ncolor.label(m, expand_mode="standard",
                  soft_conn=0, soft_radius=0, clean_mask=True)
     ```

## [1.5.3] and earlier

See `git log` for the pre-2.0 history.
