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
from ._numba_legacy.color import (  # noqa: F401
    is_sequential,
    _normalize_labels,
    neighbors,
    search,
    _PARALLEL_THRESHOLD,
)


def _legacy_label(*args, **kwargs):
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
          out=None, p=2, wrap=False):
    """4-color graph coloring of a label image.

    Default path uses the C++ Solver. ``verbose`` still falls back to the
    numba reference (the cpp pipeline doesn't print stage info).

    Pass ``out=`` (uint8 array, exact shape) to reuse an output buffer
    across calls — useful for batch pipelines.

    ``p`` selects the Voronoi expand metric:
        p=2 — Felzenszwalb parabolic envelope (Euclidean², default)
        p=1 — Saito-Toriwaki separable sweep (Manhattan)
    L1 is ~2× faster than L2 and produces a different (but equally
    valid) coloring at boundary tie-break regions; both satisfy the
    4-coloring constraint.

    ``wrap=True`` treats the image as a torus: left/right and top/bottom
    edges are neighbours, so cells on opposite image edges become
    adjacent in the colouring graph. Increases constraint pressure on
    perimeter cells and produces a more uniform colour distribution on
    tightly-cropped images. Negligible runtime cost (only the boundary
    pixels — <1% of the image — pay the wrap-aware lookup).
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

    # wrap=True path:
    #   - p=1: Solver.label's internal expand step is now natively
    #     toroidal (chamfer with extra wrap-aware sweeps). Just pass
    #     wrap=True straight through and let cpp do everything in one
    #     released-GIL call.
    #   - p=2: native toroidal L2 envelope is not implemented yet, so
    #     for the expand=True case we fall back to a Python-level
    #     np.pad(mode='wrap') + standard expand + center-crop, then
    #     color with format_input=False, expand=False, wrap=True. ~9×
    #     overhead on the expand stage; TODO replace with native cpp.
    if wrap and expand and p == 2:
        from .format import format_labels as _format
        from .expand import expand_labels as _expand
        lab_fmt = _format(lab_arr) if format_input else lab_arr.astype(np.int32, copy=False)
        exp = _expand(lab_fmt, p=2, wrap=True)
        col_array, n_used = solver.label(
            exp, n_colors=int(n), max_depth=int(max_depth),
            conn=int(conn), p=2, format_input=False,
            expand=False, wrap=True)
        fg = (lab_fmt > 0)
        if out is not None:
            out[...] = col_array * fg
            out_array = out
        else:
            out_array = (col_array * fg).astype(np.uint8)
    else:
        # Native cpp path (covers wrap=False any p, and wrap=True p=1).
        out_array, n_used = solver.label(
            lab_arr,
            n_colors=int(n), max_depth=int(max_depth),
            conn=int(conn), p=int(p), format_input=bool(format_input),
            expand=bool(expand), out=out, wrap=bool(wrap))
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
    """Unique nonzero labels — wraps fastremap.unique."""
    # Pure-Python helper; identical behavior in legacy and new paths.
    from ._numba_legacy.color import unique_nonzero as _legacy
    return _legacy(labels)


def get_lut(lab, n=4, conn=2, max_depth=30, offset=0, expand=True,
            return_n=False, verbose=False, check_conflicts=False,
            return_conflicts=False, format_input=True):
    """Return the label→color LUT used by :func:`label`."""
    return label(lab, n=n, conn=conn, max_depth=max_depth, offset=offset,
                 expand=expand, return_n=return_n, return_lut=True,
                 verbose=verbose, check_conflicts=check_conflicts,
                 return_conflicts=return_conflicts,
                 format_input=format_input)
