"""Public ``ncolor.expand_labels`` API — thin wrapper over the C++
:class:`ncolor._backend.ExpandEngine`. Falls back to the numba reference
when there's nothing to expand (empty array / no labels).
"""
from __future__ import annotations

import numpy as np


# Module-level singleton — the engine owns a persistent thread pool;
# constructing per call swamps small-image latencies.
_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from ._backend import ExpandEngine
        _ENGINE = ExpandEngine()  # auto-thread count from calibration cache
    return _ENGINE


def expand_labels(label_image, p: int = 2, *, metric: str | None = None):
    """Voronoi label expansion across background pixels under L_p metric.

    ``p=2`` uses the Felzenszwalb-Huttenlocher parabolic envelope (any
    ndim, default). ``p=1`` uses the Saito-Toriwaki separable sweep —
    Manhattan distance, ~5× faster on 2D, slightly different boundary
    placement at ties.

    Legacy ``metric='l1'``/``'l2'`` strings are accepted for backward
    compatibility and translated to ``p=1``/``p=2``.
    """
    if metric is not None:
        if metric == "l2":
            p = 2
        elif metric == "l1":
            p = 1
        else:
            raise ValueError(f"Unknown metric: {metric!r} (use 'l1' or 'l2')")
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p!r}")

    arr = np.asarray(label_image)
    if arr.size == 0 or int(arr.max()) == 0:
        # Nothing to expand — fall back to legacy (handles the empty case).
        from ._numba_legacy.expand import expand_labels as _legacy
        return _legacy(label_image, metric="l2" if p == 2 else "l1")

    return _get_engine().expand_labels(arr.astype(np.int32, copy=False), p=p)
