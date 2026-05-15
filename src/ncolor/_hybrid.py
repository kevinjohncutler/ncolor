"""Hybrid 4-colouring: vector-chromatic seed + Kempe-SA refinement.

This composes two principled algorithms whose limitations are
complementary:

  - Vector-chromatic gradient descent (``_vector_chromatic``) integrates
    2-hop avoidance into a continuous objective. It produces a smooth
    near-optimal embedding, but the rounding step (snap to tetrahedron
    vertex) can land near boundaries between two vertices, leaving a
    coarse partition. On irregular graphs it doesn't reach the discrete
    optimum.

  - Kempe-SA on 2-hop loss (``_optimize.optimize_two_hop``) is a
    discrete search that PRESERVES validity at every step (Kempe-component
    swaps cannot create 1-hop conflicts). It can find better 4-colourings
    than continuous methods because the search space is discrete-Kempe-
    accessible. But from a random/greedy seed it spends iterations
    finding the right neighborhood; from a near-optimal seed it
    converges much faster.

The hybrid uses vector-chromatic to produce a high-quality seed (the
embedding accounts for 2-hop in its objective), then runs Kempe-SA to
polish to a local optimum. Validity is guaranteed throughout.

Note on palette
---------------
This algorithm produces LABELS (1..4). Display colour assignment is
SEPARATE: use ``ncolor._palette.relabel_for_display`` to map labels to
viridis (or any palette) for visual quality. The graph algorithm and
the palette assignment are decoupled.
"""
from __future__ import annotations

import time
from typing import Dict

import numpy as np

from ._vector_chromatic import vector_chromatic_4color, _two_hop_set
from ._optimize import optimize_two_hop


def hybrid_4color(adj: Dict[int, set], N: int,
                   alpha_2hop: float = 0.5,
                   vc_restarts: int = 3,
                   vc_iters: int = 600,
                   sa_iters: int = 30000,
                   verbose: bool = False):
    """Hybrid: vector-chromatic seed then Kempe-SA on 2-hop loss.

    Parameters
    ----------
    adj : dict[int, set[int]]
    N : int
    alpha_2hop : float
        Weight on the 2-hop hinge term in the vector-chromatic embedding.
    vc_restarts, vc_iters : int
        Vector-chromatic optimisation budget.
    sa_iters : int
        Kempe-SA refinement budget.

    Returns
    -------
    colors : list[int] of length N + 1
        Valid 4-colouring (no 1-hop conflicts), with low 2-hop pairs.
    """
    if verbose:
        t0 = time.time()
    seed = vector_chromatic_4color(adj, N, alpha_2hop=alpha_2hop,
                                     n_iters=vc_iters,
                                     n_restarts=vc_restarts)
    if verbose:
        from ._optimize import two_hop_loss as _l
        print(f"  vc seed: 2-hop loss = {_l(seed, adj, N)}  "
              f"({time.time()-t0:.2f}s)")

    refined, loss = optimize_two_hop(seed, adj, N, n_iters=sa_iters)
    if verbose:
        print(f"  +kempe-sa: 2-hop loss = {loss}  ({time.time()-t0:.2f}s)")
    return refined


def label_hybrid(lab, p: int = 1, conn: int = 2,
                  alpha_2hop: float = 0.5,
                  vc_restarts: int = 3, vc_iters: int = 600,
                  sa_iters: int = 30000):
    """Public entry: hybrid 4-colouring of a label image.

    Returns a uint8 coloured image (label ∈ {1..4} per cell). For
    display, pass the result through
    ``ncolor._palette.relabel_for_display`` to map labels to display
    colours.
    """
    from .expand import expand_labels
    from .color import connect

    lab_arr = np.asarray(lab)
    expanded = expand_labels(lab_arr, p=p)
    pairs = connect(expanded, conn=conn)
    N = int(lab_arr.max())
    adj: Dict[int, set] = {u: set() for u in range(1, N + 1)}
    for u, v in pairs:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))

    colors = hybrid_4color(adj, N, alpha_2hop=alpha_2hop,
                            vc_restarts=vc_restarts,
                            vc_iters=vc_iters,
                            sa_iters=sa_iters)
    lut = np.array(colors, dtype=np.uint8)
    return lut[expanded].astype(np.uint8)
