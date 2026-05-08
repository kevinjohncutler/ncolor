"""Public ``ncolor.label`` / ``ncolor.connect`` API — thin wrappers over
the :mod:`ncolor._backend` C++ engine.

Less-common branches of the legacy numba ``label()`` (``return_lut``,
``check_conflicts``, ``return_conflicts``, ``expand=False``) fall through
to :mod:`ncolor._numba_legacy.color` because the C++ Solver doesn't yet
expose those codepaths. Day-to-day usage (``label(mask)`` with the
defaults) stays on the C++ fast path.
"""
from __future__ import annotations

import numpy as np

from .format import format_labels

# Re-export numba-side helpers that callers (tests, label_experimental_*,
# notebooks) reach for at this module path. The public wrappers below
# intentionally shadow ``label``/``connect``/``unique_nonzero``/``get_lut``
# with the C++-backed versions.
#
# The legacy module imports ``numba`` at top level. Since v1.6.0 numba is
# an optional ``[legacy]`` extra (the cpp pipeline doesn't need it), so
# this import may fail. We fall back to stub callables that raise an
# informative error — keeps ``import ncolor`` working without numba.
try:
    from ._numba_legacy.color import (  # noqa: F401
        is_sequential,
        _normalize_labels,
        neighbors,
        search,
        _PARALLEL_THRESHOLD,
    )
    _NUMBA_LEGACY_AVAILABLE = True
except ImportError:
    _NUMBA_LEGACY_AVAILABLE = False
    _PARALLEL_THRESHOLD = None
    def _missing_numba(name):
        def _raise(*a, **kw):
            raise ImportError(
                f"ncolor.color.{name} requires the legacy numba backend. "
                "Install with `pip install ncolor[legacy]`."
            )
        return _raise
    is_sequential     = _missing_numba("is_sequential")
    _normalize_labels = _missing_numba("_normalize_labels")
    neighbors         = _missing_numba("neighbors")
    search            = _missing_numba("search")


def _legacy_label(*args, **kwargs):
    if not _NUMBA_LEGACY_AVAILABLE:
        raise ImportError(
            "ncolor.label(..., verbose=True) routes to the legacy numba "
            "implementation, which requires the [legacy] extra. "
            "Install with `pip install ncolor[legacy]`."
        )
    from ._numba_legacy.color import label as _legacy
    return _legacy(*args, **kwargs)


# Module-level Solver singleton. The C++ Solver owns a persistent thread
# pool; constructing it per call adds ~5–10 ms of pool-spinup overhead
# that swamps small-image latencies. One Solver per process is plenty —
# label() / expand_labels() are not called concurrently from within a
# single ncolor consumer.
_SOLVER = None


def _get_solver():
    global _SOLVER
    if _SOLVER is None:
        from ._backend import Solver
        _SOLVER = Solver()  # auto-thread count from calibration cache
    return _SOLVER


def label(lab, n=4, conn=2, max_depth=30, offset=0, expand=True,
          return_n=False, return_lut=False, verbose=False,
          check_conflicts=False, return_conflicts=False, format_input=True,
          out=None, p=1, wrap=False, balance=True):
    """4-color graph coloring of a label image.

    Default path uses the C++ Solver. ``verbose`` still falls back to the
    numba reference (the cpp pipeline doesn't print stage info).

    Pass ``out=`` (uint8 array, exact shape) to reuse an output buffer
    across calls — useful for batch pipelines.

    ``p`` selects the Voronoi expand metric:
        p=1 — Saito-Toriwaki separable sweep (Manhattan, default)
        p=2 — Felzenszwalb parabolic envelope (Euclidean²)
    L1 is faster and produces a different (but equally valid) coloring
    at boundary tie-break regions; both satisfy the 4-coloring constraint.

    ``wrap=True`` treats the image as a torus: left/right and top/bottom
    edges are neighbours, so cells on opposite image edges become
    adjacent in the colouring graph. Increases constraint pressure on
    perimeter cells and produces a more uniform colour distribution on
    tightly-cropped images. Negligible runtime cost.

    ``balance=True`` uses the Welsh-Powell heuristic in the BFS
    coloring: cells are visited in descending-degree order so the
    most-constrained cells get coloured first. Spreads colour usage
    more evenly than the default label-ID order at ~zero runtime cost
    (one O(N) bucket sort). Recommended for visual uniformity,
    especially with p=1 where the BFS would otherwise concentrate
    color 4 unevenly.
    """
    if verbose:
        return _legacy_label(lab, n=n, conn=conn, max_depth=max_depth,
                             offset=offset, expand=expand, return_n=return_n,
                             return_lut=return_lut, verbose=verbose,
                             check_conflicts=check_conflicts,
                             return_conflicts=return_conflicts,
                             format_input=format_input)

    # Common path: cpp Solver handles format_labels (compact 1..N with
    # min-shift), expand (or skip when expand=False), connect, color,
    # apply_lut, and bg-masking under one GIL release. The wrapper just
    # dispatches; no per-call numpy scans.
    lab_arr = np.asarray(lab)
    solver = _get_solver()

    # Single cpp path for all (p, wrap) combinations: Solver.label's
    # internal expand step is natively toroidal for both L1 (chamfer
    # with extra wrap-aware sweeps) and L2 (envelope with ghost seeds).
    # No Python-level np.pad workaround.
    out_array, n_used = solver.label(
        lab_arr,
        n_colors=int(n), max_depth=int(max_depth),
        conn=int(conn), p=int(p), format_input=bool(format_input),
        expand=bool(expand), out=out, wrap=bool(wrap),
        balance=bool(balance))
    out = out_array

    # return_lut / check_conflicts / return_conflicts: read accessors from
    # the Solver (cheap — lut and conflict count were computed inside the
    # GIL-released label() call).
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


def unique_nonzero(labels):
    """Unique nonzero labels."""
    arr = np.asarray(labels)
    u = np.unique(arr)
    return u[u != 0]


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
        ``area`` (n_labels,)            — int64 pixel counts
        ``bbox_min`` (n_labels, ndim)   — int64 inclusive lower bounds
        ``bbox_max`` (n_labels, ndim)   — int64 exclusive upper bounds
        ``centroid`` (n_labels, ndim)   — float64 centroids
    Pass ``n_labels=0`` (default) to auto-detect from ``labels.max()``.

    Drop-in for the common subset of ``skimage.measure.regionprops`` —
    much faster (1.9–8× depending on shape) and returns vectorised
    arrays instead of per-region Python objects.
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
