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
          check_conflicts=False, return_conflicts=False, format_input=True):
    """4-color graph coloring of a label image.

    Default path uses the C++ Solver. Falls back to the numba reference
    when a flag is set that the C++ engine doesn't yet implement.
    """
    needs_legacy = (
        not expand
        or return_lut
        or check_conflicts
        or return_conflicts
        or verbose
    )
    if needs_legacy:
        return _legacy_label(lab, n=n, conn=conn, max_depth=max_depth,
                             offset=offset, expand=expand, return_n=return_n,
                             return_lut=return_lut, verbose=verbose,
                             check_conflicts=check_conflicts,
                             return_conflicts=return_conflicts,
                             format_input=format_input)

    # Common path: expand=True, no LUT/conflict introspection.
    # The C++ Solver handles format_labels (compact 1..N including the
    # min-shift case for min != 0), expand, connect, color, apply_lut,
    # and bg-masking — all under one GIL release. The wrapper just
    # dispatches; no per-call np.min/np.max scans needed.
    lab_arr = np.asarray(lab)
    out, n_used = _get_solver().label(
        lab_arr,
        n_colors=int(n), max_depth=int(max_depth),
        conn=int(conn), format_input=bool(format_input))

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
    return _legacy_label(lab, n=n, conn=conn, max_depth=max_depth,
                         offset=offset, expand=expand, return_n=return_n,
                         return_lut=True, verbose=verbose,
                         check_conflicts=check_conflicts,
                         return_conflicts=return_conflicts,
                         format_input=format_input)
