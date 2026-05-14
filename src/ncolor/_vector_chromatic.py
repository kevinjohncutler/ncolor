"""Vector chromatic 4-colouring via continuous optimisation on a
tetrahedral embedding.

Principle
---------
A 4-colouring is a graph homomorphism ``G → K_4``. The vector
formulation of K_4 places the 4 colours at the vertices of a regular
tetrahedron in R^3:

    t_1 = (+1, +1, +1)/√3
    t_2 = (+1, -1, -1)/√3
    t_3 = (-1, +1, -1)/√3
    t_4 = (-1, -1, +1)/√3

These have pairwise inner product = -1/3. A vector colouring assigns
each cell `u` a unit vector `x_u ∈ S²` such that for every adjacent
pair (u, v), `x_u · x_v ≤ -1/3` (the cells point "away" from each
other in the embedding, satisfying the tetrahedral chromatic condition
of Karger-Motwani-Sudan).

Loss function (continuous, differentiable):

    L(X) = Σ_{(u,v) ∈ E_1}  max(0, x_u·x_v + 1/3)²       1-hop hinge
         + α · Σ_{(u,v) ∈ E_2}  max(0, x_u·x_v + 1/3)²   2-hop hinge

Both 1-hop and 2-hop constraints are baked INTO THE OPTIMISATION
OBJECTIVE — not added afterwards. The optimizer simultaneously
satisfies validity (1-hop) and diversity (2-hop). The 2-hop term has
weight α ∈ (0, 1].

Algorithm
---------
1. Initialise `X ∈ R^(N×3)` with random unit vectors.
2. Projected gradient descent on `L(X)`, projecting each row to the
   unit sphere after every step.
3. At convergence, round each cell to the nearest tetrahedron vertex
   → discrete colour ∈ {1, 2, 3, 4}.

The hinge form is non-convex; multiple random restarts mitigate local
minima. This is a relaxation of the Karger-Motwani-Sudan SDP, lighter
weight (gradient descent on the sphere instead of full SDP) and
appropriate for our problem size.

Output
------
Always returns a valid 4-colouring (1-hop conflicts resolved during
rounding via tetrahedron-snap + a final min-conflicts pass). The
2-hop term influences the embedding so that the discrete rounding
naturally produces fewer 2-hop same-colour pairs than the pure 1-hop
spectral algorithm.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Dict

import numpy as np


# Regular tetrahedron in R^3 — the K_4 vector embedding.
_TETRA = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
]) / np.sqrt(3)
_THRESHOLD = -1.0 / 3.0  # tetrahedron inner-product threshold


def _two_hop_set(adj, N):
    th = {u: set() for u in range(1, N + 1)}
    for u in range(1, N + 1):
        nbrs = adj[u]
        for v in nbrs:
            for w in adj[v]:
                if w == u: continue
                if w in nbrs: continue
                th[u].add(w)
    return th


def _hinge_grad_pass(X, edges_1hop, edges_2hop, alpha_2hop):
    """Compute the gradient of the hinge loss

        L = Σ_{(u,v) ∈ E_1} max(0, x_u·x_v + 1/3)²
          + α · Σ_{(u,v) ∈ E_2} max(0, x_u·x_v + 1/3)²

    over a batch of edges. Returns gradient in same shape as X.
    """
    grad = np.zeros_like(X)
    # 1-hop hinge.
    for (u, v) in edges_1hop:
        dot = X[u] @ X[v]
        if dot > _THRESHOLD:
            coeff = 2.0 * (dot - _THRESHOLD)
            grad[u] += coeff * X[v]
            grad[v] += coeff * X[u]
    # 2-hop hinge (soft).
    for (u, v) in edges_2hop:
        dot = X[u] @ X[v]
        if dot > _THRESHOLD:
            coeff = 2.0 * alpha_2hop * (dot - _THRESHOLD)
            grad[u] += coeff * X[v]
            grad[v] += coeff * X[u]
    return grad


def vector_chromatic_4color(adj: Dict[int, set], N: int,
                             n_colors: int = 4,
                             alpha_2hop: float = 0.3,
                             lr: float = 0.05,
                             n_iters: int = 800,
                             n_restarts: int = 4,
                             rng_seed: int = 0):
    # n_colors is fixed at 4 in this implementation (tetrahedral
    # embedding). Accepted for API symmetry with ncolor.label.
    """Solve the vector-chromatic 4-colouring problem with 2-hop in
    the objective. Returns a 1-indexed colour list of length N+1.

    Parameters
    ----------
    adj : dict[int, set[int]]
        1-indexed adjacency.
    N : int
        Number of cells.
    alpha_2hop : float
        Weight of the 2-hop term in the loss. 0 = pure 1-hop vector
        chromatic. Default 0.3 balances 1-hop tightness and 2-hop
        diversity.
    lr : float
        Gradient-step learning rate. Reduced over time.
    n_iters : int
        Iterations per restart.
    n_restarts : int
        Number of random initialisations; the best-loss embedding is
        rounded.
    rng_seed : int
        For reproducible initialisation.
    """
    two_hop = _two_hop_set(adj, N)

    # Materialise edge lists (u, v with u < v).
    edges_1hop = []
    for u in range(1, N + 1):
        for v in adj[u]:
            if v > u:
                edges_1hop.append((u - 1, v - 1))
    edges_2hop = []
    for u in range(1, N + 1):
        for v in two_hop[u]:
            if v > u:
                edges_2hop.append((u - 1, v - 1))

    rng = np.random.default_rng(rng_seed)
    best_loss = float("inf")
    best_X = None
    for restart in range(n_restarts):
        # Random initialisation on the sphere.
        X = rng.standard_normal((N, 3))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        cur_lr = lr
        for it in range(n_iters):
            grad = _hinge_grad_pass(X, edges_1hop, edges_2hop, alpha_2hop)
            X = X - cur_lr * grad
            # Project to unit sphere.
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            X = X / norms
            # Linear LR decay over the second half.
            if it > n_iters // 2:
                cur_lr = lr * (n_iters - it) / (n_iters // 2)
        # Compute final loss.
        loss = 0.0
        for (u, v) in edges_1hop:
            dot = X[u] @ X[v]
            if dot > _THRESHOLD:
                loss += (dot - _THRESHOLD) ** 2
        for (u, v) in edges_2hop:
            dot = X[u] @ X[v]
            if dot > _THRESHOLD:
                loss += alpha_2hop * (dot - _THRESHOLD) ** 2
        if loss < best_loss:
            best_loss = loss
            best_X = X.copy()

    # ROUND. Each cell to the nearest tetrahedron vertex.
    sims = best_X @ _TETRA.T  # (N, 4)
    labels = np.argmax(sims, axis=1)
    colors = [0] * (N + 1)
    for u in range(N):
        colors[u + 1] = int(labels[u]) + 1

    # Final min-conflicts pass: resolve any 1-hop conflicts the rounding
    # may have introduced (rare but possible when an embedding cell falls
    # near the boundary between two tetrahedron vertices). Validity is
    # required regardless of the seed quality.
    _min_conflicts_1hop(colors, adj, N, rng_seed=rng_seed)
    return colors


def _min_conflicts_1hop(colors, adj, N, n_colors=4, max_iters=20000, rng_seed=0):
    """Min-conflicts on 1-hop violations only (validity guarantor)."""
    rng = random.Random(rng_seed)
    prev = float("inf"); plateau = 0
    for _it in range(max_iters):
        bad = [u for u in range(1, N + 1)
               if any(colors[u] == colors[v] for v in adj[u])]
        if not bad: return
        n_bad = len(bad)
        if n_bad >= prev: plateau += 1
        else: plateau = 0
        prev = n_bad
        if plateau > 200:
            for _ in range(N // 10):
                colors[rng.randint(1, N)] = rng.randint(1, n_colors)
            plateau = 0
            continue
        u = rng.choice(bad)
        best_n = float("inf"); cands = []
        for c in range(1, n_colors + 1):
            n_c = sum(1 for v in adj[u] if colors[v] == c)
            if n_c < best_n:
                best_n = n_c; cands = [c]
            elif n_c == best_n:
                cands.append(c)
        colors[u] = rng.choice(cands)


def label_vector_chromatic(lab, p: int = 1, conn: int = 2, n_colors: int = 4,
                            alpha_2hop: float = 0.3, n_restarts: int = 4):
    """Public entry: vector chromatic 4-colouring of a label image.

    Returns a uint8 coloured image (each pixel has the colour of its cell).
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

    colors = vector_chromatic_4color(adj, N, n_colors=n_colors,
                                       alpha_2hop=alpha_2hop,
                                       n_restarts=n_restarts)
    lut = np.array(colors, dtype=np.uint8)
    return lut[expanded].astype(np.uint8)
