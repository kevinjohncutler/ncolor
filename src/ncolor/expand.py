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
                  wrap: bool = False, mode: str = "voronoi",
                  max_rounds: int = 100,
                  connectivity_threshold: int = 1):
    """Label expansion across background pixels.

    Two expansion algorithms are available, selected by ``mode``:

    * ``"voronoi"`` (default) — Voronoi expansion under the L_p metric.
      ``p=2`` uses the Felzenszwalb-Huttenlocher parabolic envelope
      (any ndim, default). ``p=1`` uses the Saito-Toriwaki separable
      sweep (Manhattan distance, ~5× faster on 2D, slightly different
      boundary placement at ties). ``metric='l1'``/``'l2'`` are legacy
      aliases for ``p=1``/``p=2``. ``wrap=True`` makes the expansion
      toroidal: opposite image edges are treated as adjacent, so a
      seed near one edge competes for territory with seeds near the
      opposite edge. ~1.1× cost for L1, ~1.4-1.6× for L2.
    * ``"spur_free"`` — BFS dilation with a connectivity check. A bg
      pixel is claimed by a cell only if at least
      ``connectivity_threshold + 1`` of its face-neighbors share that
      label, so the claimed pixel won't be a spur. Pixels that never
      accumulate enough same-label neighbors stay bg, which naturally
      avoids the K_5-creating "starfish" convergence patterns Voronoi
      expand produces in densely-packed segmentations. Slower than
      Voronoi for small inputs (3-4× on bact / synth), comparable on
      large dense inputs where K_5 prevention matters. ``max_rounds``
      bounds the BFS (default 100; in practice 1-3 rounds suffice).
      ``p`` / ``metric`` / ``wrap`` are ignored in this mode.
    """
    if mode not in ("voronoi", "spur_free"):
        raise ValueError(
            f"mode must be 'voronoi' or 'spur_free', got {mode!r}")
    if mode == "voronoi":
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
    if mode == "voronoi":
        return _get_engine().expand_labels(arr32, p=p, wrap=bool(wrap))
    # spur_free
    from ._backend import _impl as _b
    out, _n_claimed = _b.expand_spur_free(
        arr32, max_rounds=int(max_rounds),
        connectivity_threshold=int(connectivity_threshold))
    return out
