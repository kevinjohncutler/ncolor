"""Label-equivariant 4-coloring optimization.

The greedy WP picker in the core pipeline is locally optimal but commits
to color choices without seeing global structure. On graphs with rings
or cycles, this can leave local minima where some colors go unused
within a region even though a valid 4-coloring using all 4 exists.

This module provides a Simulated-Annealing optimizer that:

  * operates on the per-label color assignment (the "LUT" returned
    by ``ncolor.label(..., return_lut=True)``),
  * minimizes a label-equivariant loss (no colormap/ΔE info enters),
  * preserves 1-hop validity by construction (Kempe-swap moves only),
  * does not need symmetry detection — symmetric inputs yield
    symmetric-in-the-graph-quotient outputs naturally.

Loss
----
The default loss penalizes same-color cell pairs at graph distance
exactly 2 ("2-hop"). On a cycle of length n ≥ 4 with 4 colors, this
loss is minimized by colorings whose every 3-cell window uses 3
distinct colors, which forces all 4 colors to appear. Equivalent to
the L(2,1)-labeling soft constraint with weight 1.

For dense graphs (e.g. real cell-segmentation outputs), the achievable
minimum is generally > 0 — the graph topology doesn't admit zero
2-hop violations — but the optimizer still finds the lowest reachable
value, which empirically corresponds to "all 4 colors used everywhere
that they can be used".

Public surface
--------------
``optimize_two_hop(lut, adj, N, ...)`` — Kempe-SA on a 1-indexed LUT
  and adjacency dict. Returns ``(optimized_lut, final_loss)``.

``build_adjacency_from_label(lab, p, conn)`` — reproduces the same
  expand+connect graph that ``ncolor.label`` builds internally.

The Python implementation is fine for graphs of a few hundred cells
(seconds). For larger graphs the same algorithm can be ported to C++.
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np


def _two_hop_pairs(adj: Dict[int, set], N: int) -> List[Tuple[int, int]]:
    """All (u, v) pairs at graph-distance exactly 2 (excludes 1-hop and self)."""
    pairs = set()
    for u in range(1, N + 1):
        nbrs = adj[u]
        for v in nbrs:
            for w in adj[v]:
                if w == u:
                    continue
                if w in nbrs:
                    continue
                pairs.add((min(u, w), max(u, w)))
    return list(pairs)


def _two_hop_loss(lut, pairs) -> int:
    return sum(1 for (u, v) in pairs
               if lut[u] != 0 and lut[u] == lut[v])


def _kempe_component(lut, adj, seed, c1, c2):
    """BFS the bipartite c1/c2 subgraph from `seed`."""
    if lut[seed] not in (c1, c2):
        return set()
    comp = {seed}
    queue = deque([seed])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in comp and lut[v] in (c1, c2):
                comp.add(v)
                queue.append(v)
    return comp


def optimize_two_hop(lut_in, adj, N, n_colors=4,
                      n_iters=30000, T0=10.0, T_min=1e-3, alpha=0.9997,
                      rng_seed=None):
    """Simulated annealing on the 2-hop same-color count.

    Parameters
    ----------
    lut_in : sequence indexable from 1..N
        Initial LUT. ``lut_in[0]`` is ignored (background slot). Must
        already be a valid 4-coloring (no 1-hop conflicts) — caller
        typically passes the output of ``ncolor.label(..., return_lut=True)``.
    adj : dict {u: set of neighbors}
        1-hop adjacency, 1-indexed.
    N : int
        Number of labels (cells).
    n_colors : int
        Palette size. Default 4.
    n_iters : int
        SA iterations. 30 000 is comfortable for N ≤ 100.
    T0, T_min, alpha : floats
        Geometric cooling schedule. ``T_k = max(T_min, T0 · α^k)``.
    rng_seed : int, optional
        Seed for the move-sampling RNG (deterministic if set).

    Returns
    -------
    best_lut : list[int]
        LUT achieving the lowest 2-hop loss seen during the run.
    best_loss : int
        That loss value.

    Notes
    -----
    The only move is a Kempe ``(X, Y)``-swap on a connected ``{X, Y}``
    component, which is **guaranteed valid** by the four-color theorem
    argument: the component is maximal in colors ``{X, Y}``, so every
    boundary-crossing edge connects to a cell outside the component that
    is neither ``X`` nor ``Y``. Swapping ``X ↔ Y`` inside therefore
    introduces no 1-hop conflict at the boundary.
    """
    rng = random.Random(rng_seed)
    lut = list(lut_in)
    pairs = _two_hop_pairs(adj, N)
    if not pairs:
        return lut, 0
    cur_loss = _two_hop_loss(lut, pairs)
    best_lut = list(lut)
    best_loss = cur_loss
    T = T0
    palette = list(range(1, n_colors + 1))
    for _ in range(n_iters):
        u = rng.randint(1, N)
        cu = lut[u]
        if cu == 0:
            continue
        # Pick a Kempe partner color ≠ cu.
        Y = rng.choice([c for c in palette if c != cu])
        comp = _kempe_component(lut, adj, u, cu, Y)
        for w in comp:
            lut[w] = cu if lut[w] == Y else Y
        new_loss = _two_hop_loss(lut, pairs)
        delta = new_loss - cur_loss
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cur_loss = new_loss
            if cur_loss < best_loss:
                best_loss = cur_loss
                best_lut = list(lut)
        else:
            # Reject — undo the swap.
            for w in comp:
                lut[w] = cu if lut[w] == Y else Y
        T = max(T_min, T * alpha)
    return best_lut, best_loss


def build_adjacency_from_label(lab, p=1, conn=2):
    """Reconstruct the cell-adjacency graph that ``ncolor.label`` builds
    internally (expand + connect)."""
    from .expand import expand_labels
    from .color import connect as _connect

    expanded = expand_labels(lab, p=p)
    pairs = _connect(expanded, conn=conn)
    N = int(np.asarray(lab).max())
    adj: Dict[int, set] = {u: set() for u in range(1, N + 1)}
    for u, v in pairs:
        u, v = int(u), int(v)
        adj[u].add(v)
        adj[v].add(u)
    return adj, N, expanded


def two_hop_loss(lut, adj, N):
    """Compute the 2-hop same-color count for diagnostics."""
    pairs = _two_hop_pairs(adj, N)
    return _two_hop_loss(lut, pairs)
