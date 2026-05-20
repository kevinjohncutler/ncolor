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


def label(lab, n=4, conn=2, max_depth=30, offset=0, expand=True,
          return_n=False, return_lut=False, verbose=False,
          check_conflicts=False, return_conflicts=False, format_input=True,
          out=None, p=1, wrap=False, balance=True, first_seen=False,
          weight_objective=0, de_table=None, weight_mode="min",
          optimize=None, extra_edges=None, connect_radius=1, despur_iters=20,
          expand_spur_free=False, spur_free_max_rounds=1):
    """4-color graph coloring of a label image.

    Pass ``out=`` (uint8 array, exact shape) to reuse an output buffer
    across calls.

    ``p`` selects the Voronoi expand metric:
        p=1: Saito-Toriwaki separable sweep (Manhattan; default)
        p=2: Felzenszwalb parabolic envelope (Euclidean²)
    L1 is faster and produces a different (but equally valid) coloring
    at boundary tie-break regions; both satisfy the 4-coloring constraint.

    ``wrap=True`` treats the image as a torus: opposite edges are
    treated as adjacent. Useful when the cells of interest are crammed
    near the image borders (tight crops); the wrap-around adjacency
    adds constraint pressure on perimeter cells and tightens the color
    distribution. For interior-clustered inputs it can *worsen* balance
    by over-constraining the graph, and for non-periodic data it bakes
    in a false adjacency. Runtime cost: ~5-15% extra in 2D, ~15-35%
    in 3D.

    ``balance=True`` uses the Welsh-Powell heuristic: cells are
    visited in descending-degree order so the most-constrained cells
    get colored first. Spreads color usage more evenly than the
    default label-ID order at ~zero runtime cost. Recommended with
    p=1, where BFS would otherwise concentrate color 4 unevenly.

    ``first_seen=True`` numbers cells in raster-scan first-encounter
    order (matching ``fastremap.renumber`` from the pre-cpp pipeline)
    instead of the default ascending-source-ID order. Affects which
    cell is visited first by the BFS, so it shifts the coloring even
    though both orderings produce valid 4-colorings. Useful for
    bit-reproducing colorings from the legacy Python implementation.

    ``weight_objective`` enables boundary-weighted color selection.
    The BFS uses Σ_v w(u,v) × ΔE(c, color(v)) as a soft objective and
    picks colors that maximize or minimize this score.

      0 / "off" / "balance"  (default) → ignore weights, behave as before
      +1 / "max" / "sharp"             → maximize Σ w × ΔE on heavy edges
      -1 / "min" / "soft"              → minimize Σ w × ΔE

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
    pass an ``(n+1) × (n+1)`` float array. Requires ``balance=True``.

    ``optimize`` enables a post-greedy GLOBAL optimization of the color
    LUT. The greedy WP picker commits cell-by-cell and can leave local
    minima (e.g. a ring whose cells use only 3 of 4 colors even though
    a 4-color assignment exists). The optimizer sees the whole graph
    and finds a near-optimum under a label-equivariant loss — no
    colormap / ΔE information enters.

      None (default)   no optimization; pure greedy output
      "two_hop"        Simulated annealing minimizing same-color
                       2-hop pairs (cells at graph-distance 2). All
                       moves are Kempe swaps, so the result is always
                       a valid 4-coloring. Runtime ≈ tens of ms to a
                       few seconds for hundreds of cells; pure Python,
                       slows on very large graphs.

    ⚠ When to use ``optimize="two_hop"``: irregular inputs (real cell
    segmentations, blob fields). It improves "all 4 colors used
    uniformly" without hurting validity.

    ⚠ When NOT to use it: regular tilings (square grids, hex grids).
    The 2-hop objective penalizes the tight 2-period tile that those
    inputs naturally produce — same color repeats every 2 cells along
    each row, which IS the visually uniform tiling but registers as
    many 2-hop violations. ``optimize="two_hop"`` will trade that for
    a 4-cycle-per-row that uses all 4 colors but breaks the 2-period
    pattern. Stick with the default greedy for those inputs.

    Symmetric synthetic inputs (e.g. radial spokes) are an in-between
    case: the rasterisation breaks the geometric symmetry, so neither
    mode produces fully symmetric output. ``optimize="two_hop"`` at
    least guarantees all 4 colors appear wherever the graph allows.
    """
    del verbose  # accepted for back-compat; cpp pipeline doesn't trace stages

    lab_arr = np.asarray(lab)
    solver = _get_solver()

    # Normalize weight_objective: accept str ("max"/"min"/"off") or int (+1/-1/0).
    if isinstance(weight_objective, str):
        wobj = {"max": 1, "max_contrast": 1, "sharp": 1,
                "min": -1, "min_contrast": -1, "soft": -1,
                "off": 0, "balance": 0, "none": 0,
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

    out_array, n_used = solver.label(
        lab_arr,
        n_colors=int(n), max_depth=int(max_depth),
        conn=int(conn), p=int(p), format_input=bool(format_input),
        expand=bool(expand), out=out, wrap=bool(wrap),
        balance=bool(balance), first_seen=bool(first_seen),
        weight_objective=wobj, de_table=de_arr,
        weight_mode=wmode_int, extra_edges=extra_arr,
        connect_radius=int(connect_radius),
        despur_iters=int(despur_iters),
        expand_spur_free=bool(expand_spur_free),
        spur_free_max_rounds=int(spur_free_max_rounds))
    out = out_array

    if optimize is not None:
        opt_kind = str(optimize).lower()
        if opt_kind in ("two_hop", "twohop", "2hop", "2-hop"):
            from ._optimize import optimize_two_hop, build_adjacency_from_label
            # Recover the per-label LUT used by the greedy picker: the
            # value at any pixel with label L is color LUT[L]. The
            # input ``lab_arr`` may have non-contiguous labels (compacted
            # by format_input=True inside the solver); reconstruct the
            # LUT from ``solver.get_last_lut`` when available.
            lut = solver.get_last_lut()
            if lut is None or len(lut) == 0:
                return out  # nothing to optimize
            # Build the same adjacency graph the picker used.
            adj, N, _ = build_adjacency_from_label(lab_arr, p=int(p),
                                                    conn=int(conn))
            new_lut, _ = optimize_two_hop(list(lut), adj, N,
                                           n_colors=int(n_used) if n_used else int(n))
            # Apply optimized LUT back to the output image.
            lut_arr = np.array(new_lut, dtype=out.dtype)
            # ``out`` was produced via the same LUT indexing pipeline,
            # so we need to remap based on label positions. Rebuild
            # ``out`` from the (possibly relabeled) input.
            # The expanded image is the same one the picker used; we
            # already build it inside build_adjacency_from_label, but
            # we can simply remap ``lab_arr`` (or its expanded variant
            # when expand=True).
            if expand:
                from .expand import expand_labels
                expanded = expand_labels(lab_arr, p=int(p))
                out = lut_arr[expanded].astype(out.dtype)
            else:
                out = lut_arr[lab_arr].astype(out.dtype)
            # Mirror the cpp pipeline: zero out pixels that were bg in
            # the original input. Without this the optimize path emits
            # color at expanded-bg pixels, breaking parity with the
            # non-optimize path.
            out[lab_arr == 0] = 0
        else:
            raise ValueError(f"unknown optimize mode: {optimize!r}")

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


def get_lut(lab, n=4, conn=2, max_depth=30, offset=0, expand=True,
            return_n=False, verbose=False, check_conflicts=False,
            return_conflicts=False, format_input=True):
    """Return the label→color LUT used by :func:`label`."""
    return label(lab, n=n, conn=conn, max_depth=max_depth, offset=offset,
                 expand=expand, return_n=return_n, return_lut=True,
                 verbose=verbose, check_conflicts=check_conflicts,
                 return_conflicts=return_conflicts,
                 format_input=format_input)
