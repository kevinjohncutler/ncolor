"""Public ``ncolor.label`` / ``ncolor.connect`` API.

Thin wrappers over the C++ Solver in :mod:`ncolor._backend`.
"""
from __future__ import annotations

import numpy as np

from .format import format_labels


# Persistent thread pool; constructing per call costs ~5-10 ms.
_SOLVER = None


def _get_solver():
    global _SOLVER
    if _SOLVER is None:
        from ._backend import Solver
        _SOLVER = Solver()
    return _SOLVER


def label(lab, n=4, conn=1, max_depth=30, expand=True,
          return_n=False, return_lut=False, verbose=False,
          check_conflicts=False, return_conflicts=False, format_input=True,
          out=None, p=2, wrap=False, first_seen=False,
          weight_objective=0, de_table=None, weight_mode="min",
          extra_edges=None, connect_radius=1,
          min_contact=1, expand_mode="bridge_free",
          soft_extra_edges=None, soft_conn=2, soft_radius=2,
          clean_mask=False):
    """4-color graph coloring of a label image.

    Returns a uint8 image where every foreground pixel of ``lab`` has
    been assigned a color in ``1..n`` such that adjacent cells receive
    different colors. Background (``lab == 0``) stays 0.

    Default behavior (call with no kwargs beyond ``lab``):
        4-color, 4-connectivity (face-only), ``bridge_free`` Voronoi
        expand with L2 metric, auto-soft constraint post-pass at
        ``conn=2 r=2``, and output that preserves the input fg/bg
        pattern exactly. These defaults match ncolor 1.x and reliably
        4-color the reference logo / synth / mm fixtures.

    Pass ``out=`` (uint8 array, exact shape) to reuse an output buffer
    across calls.

    ``n`` is the *maximum* color budget. The picker tries ``cur_n``
    starting at the minimum needed (clique lower bound) and increments
    up to ``n``. If the graph is genuinely (n+1)-chromatic, the call
    returns with ``n_used == n+1`` and may not satisfy the coloring
    constraint — check ``return_n=True`` or ``check_conflicts=True``.

    ``conn`` is the connectivity for the *hard* adjacency graph:
        conn=1: face-only (4-conn in 2D, 6-conn in 3D)
        conn=2: face + edge (8-conn in 2D, 18-conn in 3D)
        conn=3: face + edge + corner (only meaningful in 3D+, 26-conn)

    ``connect_radius`` widens the neighbor offset window (Chebyshev
    distance) beyond r=1. ``r=2`` catches near-adjacent cells through
    1-pixel gaps. Combined with ``conn=2``, this picks up diagonal
    leak adjacencies. Larger r→ more edges → harder picker problem.

    ``expand_mode`` selects the Voronoi-expand kernel:
        "bridge_free" (default) — ND Lp Voronoi + antipodal-bridge
            test + despur peel-back cascade in one fused pass.
            Internally zeros bridge/stub pixels as graph barriers
            (these don't appear in the final output unless
            ``clean_mask=True``).
        "voronoi" — plain Lp Voronoi sweep without bridge/despur
            cleanup. Faster but graphs may contain K_5 obstructions
            at thin-cell convergences.

    ``p`` selects the Voronoi expand metric:
        p=1: Saito-Toriwaki separable sweep (Manhattan)
        p=2: Felzenszwalb parabolic envelope (Euclidean²; default)
    L2 places Voronoi boundaries that avoid K_5 obstructions on
    dense-cell data (e.g. microscopy segmentations); L1 is ~1.5× faster
    on the envelope step but can leave thin-cell convergences that
    force n=5. Both produce valid colorings.

    ``min_contact`` filters edges in the hard graph by their boundary
    pixel-pair count. When ``connect_radius>1``, r=2-only edges (1-pixel
    leaks across bg gaps) typically have ~2-pixel contact while
    legitimate r=1 face-adjacent pairs have ~44. Setting ``min_contact=4``
    drops the spurious leaks, restoring 4-colorability on data where
    they create Mycielski-like obstructions. Default 1 keeps everything.

    ``extra_edges`` adds *hard* adjacency edges between specific
    cell pairs beyond the geometric graph. ``(E, 2) int32`` array,
    1-indexed by cell label. Use case: forcing specific cells to be
    differently colored regardless of whether they touch geometrically.

    ``soft_conn`` / ``soft_radius`` enable an auto-built soft-constraint
    post-pass (default ``2`` / ``2``, on). After the hard 4-coloring,
    a local search recolors vertices (without breaking the hard graph)
    to minimize the number of edges in the soft ``(soft_conn,
    soft_radius)`` kernel whose endpoints share a color. The auto-build
    scans the image *once* via a fused dual-emit kernel that emits both
    hard and delta-soft pairs in one pixel walk. Set either to 0 to
    disable. Use case: catch edges that would create K_5 obstructions
    as hard constraints, but should still be discouraged from sharing
    a color when possible.

    ``soft_extra_edges`` is the explicit alternative to ``soft_conn`` /
    ``soft_radius``: pass a ``(E, 2)`` int32 array of 1-indexed cell pair
    IDs to use directly as soft constraints. When set, the auto-build
    path is bypassed.

    ``clean_mask``:
        False (default) — apply the color LUT to the *original*
            foreground labels, so cells in the input mask always
            retain their color in the output even when bridge_free
            zeroed them internally as graph barriers. Output fg/bg
            pattern exactly matches input.
        True — apply the LUT to the post-expand buffer instead, so
            bridge_free's barrier zeros surface in the output too.
            Useful as a "clean + label" combined operation when the
            barrier-removed mask is also wanted downstream.

    ``verbose=True`` prints a one-line stage-breakdown summary
    (shape, n_used, residual soft violations, total ms, per-stage
    timings) to stderr after the call. Useful for "why is this slow?"
    debugging.

    ``wrap=True`` treats the image as a torus: opposite edges are
    treated as adjacent. Useful when the cells of interest are crammed
    near the image borders (tight crops); the wrap-around adjacency
    adds constraint pressure on perimeter cells. For interior-clustered
    inputs it can over-constrain the graph, and for non-periodic data
    it bakes in a false adjacency. Runtime cost: ~5-15% extra in 2D,
    ~15-35% in 3D.

    ``first_seen=True`` numbers cells in raster-scan first-encounter
    order (matching ``fastremap.renumber`` from the pre-cpp pipeline)
    instead of the default ascending-source-ID order. Affects which
    cell is visited first by the BFS, so it shifts the coloring even
    though both orderings produce valid 4-colorings. Useful for
    bit-reproducing colorings from the legacy Python implementation.

    ``weight_objective`` enables boundary-weighted color selection.
    The BFS uses Σ_v w(u,v) × ΔE(c, color(v)) as a soft objective and
    picks colors that maximize or minimize this score.

      0 / "off"            (default) → ignore weights, behave as before
      +1 / "max" / "sharp"           → maximize Σ w × ΔE on heavy edges
      -1 / "min" / "soft"            → minimize Σ w × ΔE

    ``weight_mode`` selects the per-pair reducer that defines w. All
    reducers operate on (d_i + d_j) at boundary pixels, where d is the
    EDT distance from expand_labels. All have ~zero overhead — computed
    in the same parallel scan as find_pairs.

      "min"      (default) w = 1 / (1 + min(d_i+d_j))
                           closest physical approach. Corner-kiss and long
                           seam are indistinguishable (both have min=0).
      "mean"     w = 1 / (1 + mean(d_i+d_j))
                           average separation along the boundary. Penalizes
                           pairs whose Voronoi seam is mostly far from cells.
      "max"      w = 1 / (1 + max(d_i+d_j))
                           farthest point. Only pairs with everywhere-close
                           contact get high weight.
      "count"    w = boundary pixel-pair count
                           pure length, no distance info.
      "harmonic" w = Σ 1 / (1 + d_i + d_j)
                           combines length AND closeness in one number.
      "mean_inv" w = (Σ 1 / (1 + d_i + d_j)) / boundary_count
                           length-normalized harmonic. Removes the long-
                           boundary bias of plain harmonic so peripheral
                           cells (with much Voronoi-extended seam) aren't
                           penalized; per-pixel inverse-distance only.

    ``de_table`` lets you override the default viridis-ΔE palette table;
    pass an ``(n+1) × (n+1)`` float array.

    """
    lab_arr = np.asarray(lab)
    solver = _get_solver()

    # Normalize weight_objective: accept str ("max"/"min"/"off") or int (+1/-1/0).
    if isinstance(weight_objective, str):
        wobj = {"max": 1, "max_contrast": 1, "sharp": 1,
                "min": -1, "min_contrast": -1, "soft": -1,
                "off": 0, "none": 0,
                }.get(weight_objective.lower(), 0)
    else:
        wobj = int(weight_objective)

    de_arr = None
    if de_table is not None:
        de_arr = np.ascontiguousarray(de_table, dtype=np.float64)

    # weight_mode: which reducer over (d_i + d_j) the BFS uses as edge weight.
    _WEIGHT_MODE = {"off": 0, "min": 1, "max": 2, "mean": 3,
                    "count": 4, "harmonic": 5, "mean_inv": 6, "meaninv": 6}
    if isinstance(weight_mode, str):
        wmode_int = _WEIGHT_MODE.get(weight_mode.lower(), 1)
    else:
        wmode_int = int(weight_mode)

    extra_arr = None
    if extra_edges is not None:
        extra_arr = np.ascontiguousarray(extra_edges, dtype=np.int32)
        if extra_arr.ndim != 2 or extra_arr.shape[1] != 2:
            raise ValueError(
                f"extra_edges must be an (E, 2) int array of 1-indexed "
                f"cell-pair constraints; got shape {extra_arr.shape}")
    soft_extra_arr = None
    if soft_extra_edges is not None:
        soft_extra_arr = np.ascontiguousarray(soft_extra_edges, dtype=np.int32)
        if soft_extra_arr.ndim != 2 or soft_extra_arr.shape[1] != 2:
            raise ValueError(
                f"soft_extra_edges must be an (E, 2) int array of 1-indexed "
                f"cell-pair preferences; got shape {soft_extra_arr.shape}")

    out_array, n_used = solver.label(
        lab_arr,
        n_colors=int(n), max_depth=int(max_depth),
        conn=int(conn), p=int(p), format_input=bool(format_input),
        expand=bool(expand), out=out, wrap=bool(wrap),
        first_seen=bool(first_seen),
        weight_objective=wobj, de_table=de_arr,
        weight_mode=wmode_int, extra_edges=extra_arr,
        connect_radius=int(connect_radius),
        min_contact=int(min_contact),
        expand_mode=str(expand_mode),
        soft_extra_edges=soft_extra_arr,
        soft_conn=int(soft_conn),
        soft_radius=int(soft_radius),
        clean_mask=bool(clean_mask),
        capture_stages=bool(verbose))
    out = out_array

    if verbose:
        # Stage-level diagnostic summary. Activated by passing
        # verbose=True; reads the stage timings captured by the cpp
        # solver via capture_stages and prints a one-line shape +
        # stage breakdown to stderr.
        import sys as _sys
        stages = solver.get_last_stages()
        total = sum(ms for _, ms in stages) if stages else 0.0
        sv = solver.get_last_n_soft_violations()
        _shape = "x".join(str(s) for s in lab_arr.shape)
        _cells = int(lab_arr.max()) if lab_arr.size else 0
        head = (f"[ncolor.label] ({_shape}, {_cells} cells)  n_used={n_used}"
                f"  sv={int(sv)}  total={total:.1f} ms")
        breakdown = "  ".join(f"{name}={ms:.1f}" for name, ms in stages)
        _sys.stderr.write(head + "\n               " + breakdown + "\n")

    if return_lut or check_conflicts or return_conflicts:
        lut = solver.get_last_lut() if return_lut else None
        conflicts = solver.get_last_n_conflicts() \
            if (check_conflicts or return_conflicts) else 0
        if check_conflicts and conflicts:
            raise ValueError(
                f"Coloring conflict detected: {conflicts} adjacent pairs share a color.")
        if return_lut:
            if return_n and return_conflicts:
                return lut, int(np.max(lut)) if lut.size else 0, conflicts
            if return_n:
                return lut, int(np.max(lut)) if lut.size else 0
            if return_conflicts:
                return lut, conflicts
            return lut
        if return_n and return_conflicts:
            return out, int(n_used), conflicts
        if return_n:
            return out, int(n_used)
        if return_conflicts:
            return out, conflicts
        return out

    if return_n:
        return out, int(n_used)
    return out


def connect(img, conn=1):
    """Find adjacent label pairs in a label image.

    Returns an ``(M, 2)`` int array of unique (lo, hi) label pairs.
    """
    return _get_solver().connect(img, conn=int(conn))


def connected_components(mask, conn=2):
    """N-D connected-components labeling. Foreground = ``mask != 0``.

    Returns a tuple ``(labels, n_components)``: ``labels`` is an int32
    array of the same shape as ``mask`` with dense 1..N component IDs
    (0 = bg); ``n_components`` is the number of components found.

    ``conn`` selects the connectivity (1 = face-only, ndim = full
    diagonal; equivalent to scipy/skimage's ``connectivity`` argument).

    Drop-in replacement for ``skimage.measure.label`` for callers that
    only need the labeled array, without the scikit-image dep.
    """
    from ._backend import _impl as _b
    return _b.cc_label(mask, conn=int(conn))


def regionprops(labels, n_labels=0):
    """Region properties for a dense int32 1..N labeled image.

    Returns ``dict`` with vectorized numpy arrays:
        ``area`` (n_labels,):           int64 pixel counts
        ``bbox_min`` (n_labels, ndim):  int64 inclusive lower bounds
        ``bbox_max`` (n_labels, ndim):  int64 exclusive upper bounds
        ``centroid`` (n_labels, ndim):  float64 centroids
    Pass ``n_labels=0`` (default) to auto-detect from ``labels.max()``.

    Drop-in for the common subset of ``skimage.measure.regionprops``;
    returns vectorized arrays instead of per-region Python objects.
    """
    from ._backend import _impl as _b
    return _b.regionprops(labels, int(n_labels))


