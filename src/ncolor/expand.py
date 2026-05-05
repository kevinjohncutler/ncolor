"""Public ``ncolor.expand_labels`` API — thin wrapper over the C++
:class:`ncolor._backend.ExpandEngine`. Falls back to the numba reference
for ndims/metrics the C++ engine doesn't yet support.
"""
from __future__ import annotations

import numpy as np


def expand_labels(label_image, metric: str = "l2"):
    """Voronoi label expansion across background pixels.

    ``metric='l2'`` uses the Felzenszwalb-Huttenlocher parabolic envelope
    (any ndim). ``metric='l1'`` uses the Saito-Toriwaki separable chamfer
    (faster on 2D; ND is supported but only marginally faster than L2).
    """
    arr = np.asarray(label_image)
    if arr.size == 0 or int(arr.max()) == 0:
        # Nothing to expand — fall back to legacy (handles the empty case).
        from ._numba_legacy.expand import expand_labels as _legacy
        return _legacy(label_image, metric=metric)

    from ._backend import ExpandEngine
    eng = ExpandEngine()  # auto-thread count from calibration cache
    if metric == "l2":
        return eng.expand_labels(arr.astype(np.int32, copy=False))
    if metric == "l1":
        return eng.expand_labels_l1(arr.astype(np.int32, copy=False))
    raise ValueError(f"Unknown metric: {metric!r} (use 'l1' or 'l2')")
