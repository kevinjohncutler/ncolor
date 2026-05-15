"""Geometry-aware 4-colouring — incorporates cell shape and orientation
into the optimisation objective.

Motivation
----------
On bacterial-cell layouts, the visual "stacking" failure mode isn't
captured by graph-distance 2-hop alone. The visually-bad case is:
two elongated cells with PARALLEL major axes whose displacement is
ORTHOGONAL to those axes (= bacteria stacked side-by-side like books
on a shelf) sharing a colour. Two cells far apart in graph distance can
still look "stacked" if their geometry aligns.

This module adds a GEOMETRIC stack score to the optimisation objective
of vector-chromatic gradient descent + Kempe-SA refinement.

Stack score
-----------
For each candidate pair (u, v) within a local radius:

    stack(u, v) = ecc_u · ecc_v
                · |axis_u · axis_v|         (parallel-axis term)
                · (1 − |disp̂ · axis_u|)    (displacement ⊥ to u's axis)
                · (1 − |disp̂ · axis_v|)    (displacement ⊥ to v's axis)
                / (1 + d_uv)                 (closeness term)

where ``ecc`` is shape eccentricity, ``axis`` is the major principal
axis, ``disp̂`` is the unit displacement vector between centroids, and
``d_uv`` is the centroid distance.

High stack score = both cells elongated + parallel + side-by-side + close.

Loss
----
The picker objective becomes (in both continuous and discrete forms):

    L(colours) = α_1 · #{1-hop conflicts}                       hard, β ≫ all others
               + α_2 · #{2-hop same-colour pairs}
               + α_s · Σ_{(u,v)} stack(u, v)² · 𝟙[same-colour]

Algorithm
---------
Same hybrid structure as before:
  1. Vector-chromatic gradient descent with the STACK term in the loss
     (continuous embedding aware of stack geometry).
  2. Kempe-SA on the discrete combined loss.

Both stages see the same objective. The stack term influences where the
continuous embedding places cells AND which Kempe swaps the SA accepts.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from ._vector_chromatic import _TETRA, _THRESHOLD, _two_hop_set
from ._optimize import _kempe_component


def shape_data(m: np.ndarray, N: int):
    """Per-cell geometry: (major_axis_uv, eccentricity, area, centroid_yx)."""
    out: Dict[int, Tuple[np.ndarray, float, int, np.ndarray]] = {}
    for c in range(1, N + 1):
        ys, xs = np.where(m == c)
        if len(ys) < 3:
            cy_cx = np.array([ys.mean() if len(ys) else 0,
                              xs.mean() if len(ys) else 0])
            out[c] = (np.array([1.0, 0.0]), 0.0, len(ys), cy_cx)
            continue
        cy, cx = ys.mean(), xs.mean()
        coords = np.column_stack([ys - cy, xs - cx])
        cov = np.cov(coords.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            out[c] = (np.array([1.0, 0.0]), 0.0, len(ys), np.array([cy, cx]))
            continue
        big = max(eigvals[-1], 1e-6); small = max(eigvals[0], 1e-6)
        ecc = float(np.sqrt(max(0.0, 1.0 - small / big)))
        out[c] = (eigvecs[:, -1], ecc, len(ys), np.array([cy, cx]))
    return out


def compute_iou_similarity(m: np.ndarray, N: int,
                            max_dist_factor: float = 5.0,
                            min_score: float = 0.05):
    """Per-pair visual similarity = IoU of the two cell masks after
    aligning their centroids, weighted by proximity.

    Captures (in a single number, principled rather than hand-crafted):
      - shape similarity (overlap when centred)
      - orientation similarity (rotated cells don't overlap)
      - size similarity (different-area cells have low IoU)
      - proximity (Gaussian falloff with centroid distance)

    Heavy compute: per pair O(area), but precomputed ONCE before the
    SA loop; subsequent loss evaluations are O(#pairs)."""
    shp = shape_data(m, N)
    if N < 2: return {}
    mean_radius = float(np.sqrt(np.mean([shp[c][2] for c in range(1, N + 1)]) / np.pi))
    max_dist = max_dist_factor * mean_radius
    centroids = np.array([shp[c][3] for c in range(1, N + 1)])

    # Precompute each cell's pixel coordinates as offsets-from-centroid.
    cells = {}
    for u in range(1, N + 1):
        ys, xs = np.where(m == u)
        if len(ys) < 3: continue
        cy, cx = ys.mean(), xs.mean()
        cells[u] = (np.column_stack([ys - cy, xs - cx]).astype(int),
                    int(len(ys)))

    pair_sim: Dict[Tuple[int, int], float] = {}
    for u in range(1, N + 1):
        if u not in cells: continue
        offsets_u, area_u = cells[u]
        set_u = set(map(tuple, offsets_u))
        for v in range(u + 1, N + 1):
            if v not in cells: continue
            d = float(np.linalg.norm(centroids[u - 1] - centroids[v - 1]))
            if d > max_dist: continue
            offsets_v, area_v = cells[v]
            set_v = set(map(tuple, offsets_v))
            inter = len(set_u & set_v)
            union = len(set_u | set_v)
            iou = inter / max(union, 1)
            # Combine IoU (shape similarity) with proximity falloff.
            score = iou * np.exp(-d / (2.0 * mean_radius))
            if score >= min_score:
                pair_sim[(u, v)] = float(score)
    return pair_sim


def stack_score(shp_u, shp_v, d_uv, scale: float = 30.0) -> float:
    """The visual stacking metric — fires for ANY arrangement of parallel
    elongated cells that are close. Captures both:
      • side-by-side stacks (books on a shelf)
      • end-to-end stacks (bacteria forming a line)

    The earlier formulation only caught side-by-side; this one is broader.
    """
    axis_u, ecc_u, _, c_u = shp_u
    axis_v, ecc_v, _, c_v = shp_v
    if ecc_u < 0.4 or ecc_v < 0.4: return 0.0
    a_uv = abs(float(np.dot(axis_u, axis_v)))  # parallel-axis term
    if a_uv < 0.7: return 0.0  # require near-parallel orientation
    # Stack score: eccentric, parallel, close. NO displacement-direction
    # term — both side-by-side and end-to-end count as stacks.
    return ecc_u * ecc_v * (a_uv ** 2) * np.exp(-d_uv / scale)


def compute_repulsion_pairs(m: np.ndarray, N: int, max_dist_factor: float = 4.0):
    """For every pair of cells within max_dist_factor·mean_radius (Euclidean
    centroid distance), compute a repulsion penalty (D_ideal − d)². Higher
    penalty for closer pairs. This is a GEOMETRIC, not graph-distance,
    spacing constraint — it penalises any close same-colour pair regardless
    of adjacency."""
    shp = shape_data(m, N)
    if N < 2: return {}, shp
    mean_radius = float(np.sqrt(np.mean([shp[c][2] for c in range(1, N + 1)]) / np.pi))
    D_ideal = max_dist_factor * mean_radius
    pairs: Dict[Tuple[int, int], float] = {}
    centroids = np.array([shp[c][3] for c in range(1, N + 1)])
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            d = float(np.linalg.norm(centroids[u - 1] - centroids[v - 1]))
            if d < D_ideal:
                pairs[(u, v)] = (D_ideal - d) ** 2 / (D_ideal ** 2)  # normalised
    return pairs, shp


def compute_stack_pairs(m: np.ndarray, N: int, max_dist_factor: float = 4.0,
                         score_threshold: float = 0.01):
    """For each pair of cells within `max_dist_factor × mean_radius`,
    compute the stack score. Return only pairs with score > threshold."""
    shp = shape_data(m, N)
    if N < 2: return {}, shp
    mean_radius = float(np.sqrt(np.mean([shp[c][2] for c in range(1, N + 1)]) / np.pi))
    max_dist = max_dist_factor * mean_radius

    # Pairwise nearest-pixel distance (EDT-based, robust).
    pair_scores: Dict[Tuple[int, int], float] = {}
    # Compute approximate centroid distances first (cheap).
    centroids = np.array([shp[c][3] for c in range(1, N + 1)])
    dy_mat = centroids[None, :, 0] - centroids[:, None, 0]
    dx_mat = centroids[None, :, 1] - centroids[:, None, 1]
    dist_mat = np.sqrt(dy_mat**2 + dx_mat**2)
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            d = dist_mat[u - 1, v - 1]
            if d > max_dist: continue
            s = stack_score(shp[u], shp[v], d)
            if s > score_threshold:
                pair_scores[(u, v)] = s
    return pair_scores, shp


# ------ continuous: vector-chromatic with stack term ------
def _hinge_grad_pass(X, pairs_1hop, pairs_2hop, pairs_stack,
                      alpha_2hop, beta_stack):
    """Gradient of the combined hinge loss."""
    grad = np.zeros_like(X)
    for (u, v) in pairs_1hop:
        dot = X[u] @ X[v]
        if dot > _THRESHOLD:
            coeff = 2.0 * (dot - _THRESHOLD)
            grad[u] += coeff * X[v]; grad[v] += coeff * X[u]
    for (u, v) in pairs_2hop:
        dot = X[u] @ X[v]
        if dot > _THRESHOLD:
            coeff = 2.0 * alpha_2hop * (dot - _THRESHOLD)
            grad[u] += coeff * X[v]; grad[v] += coeff * X[u]
    for (u, v), s in pairs_stack.items():
        # Stack term: weight by s² so high-stack pairs get strong penalty.
        dot = X[u - 1] @ X[v - 1]
        if dot > _THRESHOLD:
            coeff = 2.0 * beta_stack * (s * s) * (dot - _THRESHOLD)
            grad[u - 1] += coeff * X[v - 1]; grad[v - 1] += coeff * X[u - 1]
    return grad


def vc_with_stack(adj, N, m: np.ndarray,
                    alpha_2hop: float = 0.3,
                    beta_stack: float = 50.0,
                    lr: float = 0.05, n_iters: int = 600,
                    n_restarts: int = 3, rng_seed: int = 0):
    """Vector-chromatic gradient descent with 1-hop + 2-hop + stack terms."""
    two_hop = _two_hop_set(adj, N)
    edges_1hop = [(u - 1, v - 1) for u in range(1, N + 1)
                   for v in adj[u] if v > u]
    edges_2hop = [(u - 1, v - 1) for u in range(1, N + 1)
                   for v in two_hop[u] if v > u]
    stack_pairs, _ = compute_stack_pairs(m, N)
    if not stack_pairs:
        beta_stack = 0.0  # no stack pairs found

    rng = np.random.default_rng(rng_seed)
    best_loss = float("inf"); best_X = None
    for restart in range(n_restarts):
        X = rng.standard_normal((N, 3))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        cur_lr = lr
        for it in range(n_iters):
            grad = _hinge_grad_pass(X, edges_1hop, edges_2hop,
                                      stack_pairs, alpha_2hop, beta_stack)
            X = X - cur_lr * grad
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            X = X / norms
            if it > n_iters // 2:
                cur_lr = lr * (n_iters - it) / (n_iters // 2)
        # Loss at convergence.
        loss = 0.0
        for (u, v) in edges_1hop:
            dot = X[u] @ X[v]
            if dot > _THRESHOLD: loss += (dot - _THRESHOLD) ** 2
        for (u, v) in edges_2hop:
            dot = X[u] @ X[v]
            if dot > _THRESHOLD: loss += alpha_2hop * (dot - _THRESHOLD) ** 2
        for (u, v), s in stack_pairs.items():
            dot = X[u - 1] @ X[v - 1]
            if dot > _THRESHOLD:
                loss += beta_stack * (s * s) * (dot - _THRESHOLD) ** 2
        if loss < best_loss:
            best_loss = loss; best_X = X.copy()

    # Round to tetrahedron vertices.
    sims = best_X @ _TETRA.T
    labels = np.argmax(sims, axis=1)
    colors = [0] * (N + 1)
    for u in range(N): colors[u + 1] = int(labels[u]) + 1

    # Repair 1-hop residual.
    _min_conflicts_1hop(colors, adj, N, rng_seed=rng_seed)
    return colors, stack_pairs


def _min_conflicts_1hop(colors, adj, N, n_colors=4, max_iters=20000, rng_seed=0):
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


# ------ discrete: Kempe-SA on combined loss ------
def compute_combined_loss(colors, adj, two_hop,
                            iou_pairs,
                            alpha_2hop, gamma_iou):
    """L = β · n_1hop_conflicts            (hard, β=1000)
         + α₂ · n_2hop_same
         + γ_IoU · Σ iou(u,v) · 𝟙[same]   (visual similarity, principled)
    """
    n_1hop = sum(1 for u in range(1, len(colors)) for v in adj[u]
                 if v > u and colors[u] == colors[v])
    n_2hop = sum(1 for u in range(1, len(colors)) for v in two_hop[u]
                 if v > u and colors[u] == colors[v])
    iou_loss = sum(s for (u, v), s in iou_pairs.items()
                   if colors[u] == colors[v])
    return (1000.0 * n_1hop + alpha_2hop * n_2hop + gamma_iou * iou_loss)


def kempe_sa(colors, adj, N, m, alpha_2hop=1.0, gamma_iou=50.0,
              n_iters=30000, T0=2.0, T_min=0.001, alpha_cool=0.9998,
              rng_seed=0):
    """Kempe-SA on 1-hop (hard) + 2-hop + IoU-similarity loss."""
    rng = random.Random(rng_seed)
    two_hop = _two_hop_set(adj, N)
    iou_pairs = compute_iou_similarity(m, N)
    cur_loss = compute_combined_loss(colors, adj, two_hop, iou_pairs,
                                       alpha_2hop, gamma_iou)
    best = list(colors); best_loss = cur_loss
    T = T0
    for it in range(n_iters):
        if best_loss == 0: break
        u = rng.randint(1, N)
        cu = colors[u]
        if cu == 0: continue
        Y = rng.choice([c for c in range(1, 5) if c != cu])
        comp = _kempe_component(colors, adj, u, cu, Y)
        for w in comp: colors[w] = Y if colors[w] == cu else cu
        new_loss = compute_combined_loss(colors, adj, two_hop, iou_pairs,
                                           alpha_2hop, gamma_iou)
        delta = new_loss - cur_loss
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cur_loss = new_loss
            if cur_loss < best_loss:
                best_loss = cur_loss; best = list(colors)
        else:
            for w in comp: colors[w] = Y if colors[w] == cu else cu
        T = max(T_min, T * alpha_cool)
    for i in range(N + 1): colors[i] = best[i]
    return colors, best_loss


# ------ public ------
def label_geometric_hybrid(lab, p: int = 1, conn: int = 2,
                            alpha_2hop: float = 1.0,
                            gamma_iou: float = 50.0,
                            sa_iters: int = 30000):
    """Geometry-aware 4-colouring with IoU visual-similarity in the loss.

    The IoU similarity is a PRINCIPLED metric (not hand-crafted): for
    each pair of cells (u, v) within a local radius, compute the
    Intersection-over-Union of their masks after centroid alignment,
    weighted by proximity. Captures shape, orientation, size, and
    distance in a single number.

    Steps:
      1. Build adjacency (expand + connect).
      2. Greedy seed via ncolor.label (valid 4-coloring).
      3. Compute pairwise IoU similarities for cells within local radius.
      4. Kempe-SA on combined 1-hop (hard) + 2-hop + IoU-similarity loss.
      5. Return uint8 label image.

    Display colour assignment is separate — pass through
    ``ncolor._palette.relabel_for_display``.
    """
    from .expand import expand_labels
    from .color import connect
    from .color import label as _ncolor_label

    lab_arr = np.asarray(lab)
    expanded = expand_labels(lab_arr, p=p)
    pairs = connect(expanded, conn=conn)
    N = int(lab_arr.max())
    adj: Dict[int, set] = {u: set() for u in range(1, N + 1)}
    for u, v in pairs:
        adj[int(u)].add(int(v)); adj[int(v)].add(int(u))

    # Seed: greedy 4-coloring (already valid).
    nc_seed = _ncolor_label(lab_arr, expand=True, p=p, balance=True,
                              weight_objective=0)
    colors = [0] * (N + 1)
    for u in range(1, N + 1):
        ys, xs = np.where(lab_arr == u)
        if len(ys):
            colors[u] = int(nc_seed[ys[0], xs[0]])

    colors, _ = kempe_sa(colors, adj, N, lab_arr,
                          alpha_2hop=alpha_2hop, gamma_iou=gamma_iou,
                          n_iters=sa_iters)
    lut = np.array(colors, dtype=np.uint8)
    return lut[expanded].astype(np.uint8)
