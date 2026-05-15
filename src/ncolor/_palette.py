"""Palette assignment — decouple the graph 4-colouring from the display
colours.

After the graph algorithm produces a label image with labels in
{1, 2, 3, 4}, the labels are arbitrary up to permutation: any of the
4! = 24 permutations represents the same graph 4-colouring. This module
picks the permutation that maximises perceptual contrast on the
boundary edges that visually matter — heavy edges (long shared
boundary) should get the most-distinguishable colour pair.

The graph algorithm is now PURE — it uses no colormap information. The
palette assignment is a SEPARATE post-pass with its own objective.

Public surface
--------------
``optimal_palette_permutation(labels, adj, edge_weights, de_table)``
  — find the permutation σ ∈ S_4 that maximises
        Σ_{(u, v) ∈ E} w(u, v) · ΔE(σ(labels[u]), σ(labels[v]))

``apply_palette(label_image, perm)``
  — relabel an image according to the permutation.

``relabel_for_display(label_image, m, viridis_de=None, weight_mode="harmonic")``
  — convenience: build the adjacency + boundary weights from the input
    mask, find the optimal permutation under a viridis ΔE table (or
    user-provided), and return the relabelled image.
"""
from __future__ import annotations

from itertools import permutations
from typing import Dict, Tuple

import numpy as np


# Default viridis-4 ΔE table (sampled at evenly-spaced t values).
_VIRIDIS_DE4 = np.array([
    [0.0,    0.0,   0.0,   0.0,    0.0],
    [0.0,    0.0,  52.0,  104.74, 133.36],
    [0.0,   52.0,   0.0,   56.28, 100.98],
    [0.0,  104.74, 56.28,  0.0,   62.58],
    [0.0,  133.36, 100.98, 62.58,  0.0],
])


def optimal_palette_permutation(labels: np.ndarray,
                                  edges_with_weight: Dict[Tuple[int, int], float],
                                  de_table: np.ndarray = None,
                                  n_colors: int = 4) -> Tuple[int, ...]:
    """Find σ ∈ S_{n_colors} maximising the weighted ΔE score over edges.

    Parameters
    ----------
    labels : array-like of int
        ``labels[u]`` is the (1-indexed) label assigned to cell u
        by the graph algorithm. Cell index is 1-indexed; ``labels[0]``
        is unused.
    edges_with_weight : dict
        ``{(u, v): w(u, v)}`` for u < v.
    de_table : (n_colors+1, n_colors+1) array
        ``de_table[i, j]`` is the ΔE between colours i and j. Defaults
        to the viridis-4 table.
    n_colors : int
        Palette size.

    Returns
    -------
    perm : tuple of length n_colors
        ``perm[i-1]`` is the new label for any cell currently labelled i.
    """
    if de_table is None:
        de_table = _VIRIDIS_DE4
    best_score = -float("inf")
    best_perm = tuple(range(1, n_colors + 1))
    for perm in permutations(range(1, n_colors + 1)):
        score = 0.0
        for (u, v), w in edges_with_weight.items():
            lu, lv = labels[u], labels[v]
            if lu < 1 or lu > n_colors or lv < 1 or lv > n_colors:
                continue  # skip cells whose colour is outside the palette
            cu = perm[lu - 1]
            cv = perm[lv - 1]
            score += w * de_table[cu, cv]
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_perm


def apply_palette(label_image: np.ndarray, perm: Tuple[int, ...]) -> np.ndarray:
    """Relabel a uint8 image according to ``perm``. Background (0) is
    preserved."""
    n_colors = len(perm)
    lut = np.zeros(max(int(label_image.max()) + 1, n_colors + 1),
                    dtype=label_image.dtype)
    for i in range(1, n_colors + 1):
        lut[i] = perm[i - 1]
    return lut[label_image]


def relabel_for_display(label_image: np.ndarray, m: np.ndarray,
                         de_table: np.ndarray = None,
                         weight_mode: str = "boundary"):
    """Convenience wrapper that builds the adjacency + edge weights from
    an input mask and returns the optimally-permuted label image.

    weight_mode:
      "boundary"  — w(u, v) = shared 1-hop boundary length (pixel pairs)
                    between expanded labels u and v.
      "uniform"   — w(u, v) = 1 for all adjacencies.
    """
    from .expand import expand_labels
    from .color import connect

    expanded = expand_labels(m, p=1)
    pairs = connect(expanded, conn=2)
    N = int(m.max())

    # Build per-cell labels from the label_image (sample at cell pixel).
    labels = np.zeros(N + 1, dtype=int)
    for u in range(1, N + 1):
        ys, xs = np.where(m == u)
        if len(ys):
            labels[u] = int(label_image[ys[0], xs[0]])

    # Edge weights — vectorised boundary-pixel count.
    edges_w: Dict[Tuple[int, int], float] = {}
    if weight_mode == "boundary":
        H, W = expanded.shape
        packed_chunks = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                y_a = slice(max(0, dy), H + min(0, dy))
                x_a = slice(max(0, dx), W + min(0, dx))
                y_b = slice(max(0, -dy), H + min(0, -dy))
                x_b = slice(max(0, -dx), W + min(0, -dx))
                a = expanded[y_a, x_a].ravel()
                b = expanded[y_b, x_b].ravel()
                mask = (a != b) & (a > 0) & (b > 0)
                if not mask.any(): continue
                lo = np.minimum(a[mask], b[mask]).astype(np.int64)
                hi = np.maximum(a[mask], b[mask]).astype(np.int64)
                packed_chunks.append((hi << 32) | lo)
        if packed_chunks:
            all_packed = np.concatenate(packed_chunks)
            uniq, counts = np.unique(all_packed, return_counts=True)
            lo_arr = (uniq & 0xFFFFFFFF).astype(np.int64)
            hi_arr = (uniq >> 32).astype(np.int64)
            for u, v, w in zip(lo_arr, hi_arr, counts):
                edges_w[(int(u), int(v))] = float(w)
    else:  # uniform
        for u, v in pairs:
            u, v = int(u), int(v)
            if u > v: u, v = v, u
            edges_w[(u, v)] = 1.0

    perm = optimal_palette_permutation(labels, edges_w, de_table=de_table)
    return apply_palette(label_image, perm), perm
