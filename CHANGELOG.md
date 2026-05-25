# Changelog

All notable changes to ncolor are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [semantic versioning](https://semver.org/).

## [2.0.0] — unreleased

The headline of 2.0 is a new default expand pipeline (`bridge_free`) and
an opt-out auto-soft constraint post-pass that together produce cleaner
4-colorings on real microscopy data. A small number of kwargs were
removed; the major bump captures that surface change.

### Added
- **`bridge_free` expand mode** (now the default `expand_mode`). ND Lp
  Voronoi sweep fused with an antipodal-bridge test and a despur
  peel-back cascade in one pass. Subsumes the older expand + despur
  chain; removes 1-pixel bridges and ≤1-face stubs as graph barriers
  to prevent K_5-shaped convergence clusters that would otherwise
  block 4-coloring.
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
  input mask's foreground / background pattern exactly. Bridge_free's
  internal barrier removal stays a graph-only step; the LUT is
  applied to a pre-bridge_free snapshot of the foreground labels.
  Set `clean_mask=True` for the old behavior where barrier zeros
  surface in the output too (useful as a clean + label combined op).
- **`verbose=True`** writes a one-line stage-breakdown summary
  (shape, n_used, residual soft violations, total ms, per-stage
  timings) to stderr after the call. Replaces what was previously a
  back-compat no-op slot.
- **Dual-emit `find_pairs`** kernel (`find_pairs_dual_nd_unpadded` in
  `cpp/connect.hpp`). Emits both base and delta pairs in a single
  pixel walk via a per-pixel routing inner loop. Drops the cost of
  the soft auto-build from a second full scan to ~the difference
  between the soft and hard offset counts.
- **`bb_dsatur` is iterative.** Recursive backtracking blew past the
  512 KB macOS worker-thread stack at N≈3000+; the new explicit
  heap-allocated state stack is safe for graphs of any size.
- **`expand_labels(mode="bridge_free")`** in the Python `expand.py`
  wrapper for callers that want to expand without coloring.

### Changed
- **Default `expand_mode` is now `"bridge_free"`** (was `"voronoi"`).
- **Default `soft_conn=2, soft_radius=2`** turns on the auto-soft
  post-pass. Set both to `0` to disable.
- **Threadpool idle wait on macOS** uses `sleep_for(5ms)` after the
  spin window, ported from edt. The earlier `yield()`-based loop
  caused a `swtch_pri` storm consuming ~1700-1800% CPU across idle
  worker threads. Linux and Windows keep the original yield path
  (yield is cheap there; a uniform sleep regressed Linux multi-T
  fork-join workloads 50-200%).

### Removed
- **`offset` kwarg** — was declared in `label()` and forwarded by
  `get_lut()` but never read anywhere. Dead since v1.0.
- **`despur_iters`** and **`despur_remove_thin`** — subsumed by
  `bridge_free`, which is now the default and includes its own
  fused barrier removal. The standalone despur pass was a no-op on
  top of bridge_free.
- **`expand_spur_free`** and **`spur_free_max_rounds`** — legacy
  aliases for `expand_mode="spur_free"`. Use `expand_mode` directly.

### Fixed
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
- **mm 2k² real microscopy (5128 cells, 14.6% fill):** auto-soft
  total ~26 ms at n=5 with sv=3 — ~11× faster than legacy numba
  ncolor 1.4.5 at ~280 ms with sv=104.
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
- Coverage 94% → 96% (`expand.py` 77% → 100%, `color.py` 94% → 97%).
- Six new tests: soft path default behavior, explicit
  `soft_extra_edges`, `clean_mask` true/false semantics, soft-off
  fallback, `expand_labels(mode="bridge_free")` p/metric coverage,
  `verbose=True` stage summary.
- Added `.gitignore` entries for `_tmp_*` and `bench_outputs/` so
  per-session scratch and bench output directories stop polluting
  `git status`. Promoted five active benches to `bench/`.

### Migration notes from 1.5.x

- A call like `ncolor.label(m, despur_iters=2)` now raises
  `TypeError: unexpected keyword argument 'despur_iters'`. Drop the
  kwarg — `expand_mode="bridge_free"` (default) already includes the
  cleanup.
- Same for `offset`, `expand_spur_free`, `spur_free_max_rounds`,
  `despur_remove_thin`.
- The output of `ncolor.label(m)` is no longer bit-identical to 1.5.x
  for two reasons:
    1. Default `expand_mode` flipped from `"voronoi"` to `"bridge_free"`.
    2. Default soft-constraint post-pass is on. To recover bit-equivalent
       1.5.x behavior pass:
       ```python
       ncolor.label(m, expand_mode="voronoi",
                    soft_conn=0, soft_radius=0, clean_mask=True)
       ```

## [1.5.3] and earlier

See `git log` for the pre-2.0 history.
