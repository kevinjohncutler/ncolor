"""Public ``ncolor.expand_labels`` API.

Thin wrapper over the C++ ExpandEngine in :mod:`ncolor._backend`.
"""
from __future__ import annotations

import numpy as np


# Persistent thread pool; constructing per call costs ~5-10 ms.
_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        from ._backend import ExpandEngine
        _ENGINE = ExpandEngine()
    return _ENGINE


def expand_labels(label_image, p: int = 2, *, metric: str | None = None,
                  wrap: bool = False, mode: str = "standard"):
    """Label expansion across background pixels.

    Two expansion algorithms are available, selected by ``mode``:

    * ``"standard"`` (default) — Voronoi expansion under the L_p metric.
      ``p=2`` uses the Felzenszwalb-Huttenlocher parabolic envelope
      (any ndim, default). ``p=1`` uses the Saito-Toriwaki separable
      sweep (Manhattan distance, ~5× faster on 2D, slightly different
      boundary placement at ties). ``metric='l1'``/``'l2'`` are legacy
      aliases for ``p=1``/``p=2``. ``wrap=True`` makes the expansion
      toroidal: opposite image edges are treated as adjacent, so a
      seed near one edge competes for territory with seeds near the
      opposite edge. ~1.1× cost for L1, ~1.4-1.6× for L2.
    * ``"clean"`` — Same separable Voronoi expansion as ``"standard"``
      (selectable via ``p`` / ``metric``: ``p=1`` Saito-Toriwaki L1,
      ``p=2`` Felzenszwalb L2), with an antipodal-only bridge test run
      on the final 2D-Voronoi labels: pixels with exactly two
      same-label neighbors arranged antipodally (N-S, E-W, NE-SW, or
      NW-SE) are marked bg-barriers. Prevents 1-pixel-wide bridges
      (face or corner) between Voronoi cells. L1 typically produces an
      order of magnitude more antipodal bridges than L2 — this is the
      metric where the test changes the output materially. 2D only for
      now — ND > 2 falls back to standard expand (no bridge
      prevention).
    """
    if mode not in ("standard", "clean"):
        raise ValueError(
            f"mode must be 'standard' or 'clean', got {mode!r}")
    if metric is not None:
        if metric == "l2":
            p = 2
        elif metric == "l1":
            p = 1
        else:
            raise ValueError(
                f"Unknown metric: {metric!r} (use 'l1' or 'l2')")
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p!r}")

    arr = np.asarray(label_image)
    if arr.size == 0 or int(arr.max()) == 0:
        # No seeds to expand from; return a fresh int32 copy so callers
        # always get a writable buffer of the canonical output dtype.
        return arr.astype(np.int32, copy=True)

    arr32 = arr.astype(np.int32, copy=False)
    if mode == "standard":
        return _get_engine().expand_labels(arr32, p=p, wrap=bool(wrap))
    return _get_engine().expand_labels_clean(arr32, p=int(p))
