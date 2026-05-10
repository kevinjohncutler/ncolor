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
          out=None, p=1, wrap=False, balance=True):
    """4-color graph coloring of a label image.

    Pass ``out=`` (uint8 array, exact shape) to reuse an output buffer
    across calls.

    ``p`` selects the Voronoi expand metric:
        p=1: Saito-Toriwaki separable sweep (Manhattan; default)
        p=2: Felzenszwalb parabolic envelope (Euclidean²)
    L1 is faster and produces a different (but equally valid) coloring
    at boundary tie-break regions; both satisfy the 4-coloring constraint.

    ``wrap=True`` treats the image as a torus: opposite edges are
    treated as adjacent. Increases constraint pressure on perimeter
    cells and produces a more uniform colour distribution on
    tightly-cropped images. Negligible runtime cost.

    ``balance=True`` uses the Welsh-Powell heuristic: cells are
    visited in descending-degree order so the most-constrained cells
    get coloured first. Spreads colour usage more evenly than the
    default label-ID order at ~zero runtime cost. Recommended with
    p=1, where BFS would otherwise concentrate color 4 unevenly.
    """
    del verbose  # accepted for back-compat; cpp pipeline doesn't trace stages

    lab_arr = np.asarray(lab)
    solver = _get_solver()

    out_array, n_used = solver.label(
        lab_arr,
        n_colors=int(n), max_depth=int(max_depth),
        conn=int(conn), p=int(p), format_input=bool(format_input),
        expand=bool(expand), out=out, wrap=bool(wrap),
        balance=bool(balance))
    out = out_array

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
    """N-D connected-components labelling. Foreground = ``mask != 0``.

    Returns a tuple ``(labels, n_components)``: ``labels`` is an int32
    array of the same shape as ``mask`` with dense 1..N component IDs
    (0 = bg); ``n_components`` is the number of components found.

    ``conn`` selects the connectivity (1 = face-only, ndim = full
    diagonal; equivalent to scipy/skimage's ``connectivity`` argument).

    Drop-in replacement for ``skimage.measure.label`` for callers that
    only need the labelled array, without the scikit-image dep.
    """
    from ._backend import _impl as _b
    return _b.cc_label(mask, conn=int(conn))


def regionprops(labels, n_labels=0):
    """Region properties for a dense int32 1..N labelled image.

    Returns ``dict`` with vectorised numpy arrays:
        ``area`` (n_labels,):           int64 pixel counts
        ``bbox_min`` (n_labels, ndim):  int64 inclusive lower bounds
        ``bbox_max`` (n_labels, ndim):  int64 exclusive upper bounds
        ``centroid`` (n_labels, ndim):  float64 centroids
    Pass ``n_labels=0`` (default) to auto-detect from ``labels.max()``.

    Drop-in for the common subset of ``skimage.measure.regionprops``;
    returns vectorised arrays instead of per-region Python objects.
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
