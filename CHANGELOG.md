# Changelog

All notable changes to ncolor are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [semantic versioning](https://semver.org/).

## [2.0.0] — unreleased

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
