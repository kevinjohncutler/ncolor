"""Scalable geometry-aware 4-colouring — O(N log N) pair finding + O(1)
approximate similarity + incremental Kempe-SA loss.

Same algorithm and visual goals as ``_geometric.py``, but optimised for
images with thousands of cells. The two big speedups:

  1. APPROXIMATE IoU.
     Exact IoU after centroid alignment is O(cell_area) per pair. We
     replace it with a closed-form shape-descriptor approximation:

       similarity(u, v) = axis_alignment(u, v)²
                        · area_match(u, v)
                        · proximity(u, v)

     where each factor is O(1) given per-cell precomputed PCA stats.
     For elongated cells with clean orientations (the bacterial case),
     this closely tracks exact IoU and is ~100× faster.

  2. INCREMENTAL LOSS in Kempe-SA.
     The full loss has O(#pairs) terms; recomputing every iteration is
     O(N²) per step. We maintain a running ``cur_loss`` and apply
     O(boundary_size · deg) updates per Kempe swap. For typical
     component sizes the per-iteration cost drops to ~100 ops
     regardless of N.

  + KDTree for pair finding so we skip O(N²) all-vs-all scanning;
    pairs within the local radius are enumerated in O(N log N + #pairs).
"""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import KDTree


def per_cell_geometry_cpp(m: np.ndarray):
    """C++ per-cell geometric features. Returns dict of arrays."""
    from ._backend import _impl as _backend_impl
    return _backend_impl.per_cell_geometry(np.ascontiguousarray(m))


def per_cell_geometry(m: np.ndarray, N: int):
    """Compute per-cell (centroid, major_axis_uv, eccentricity, area).

    Fully vectorised: one pass over all pixels using np.bincount for
    moments (sum, sum-x, sum-y, sum-xx, sum-yy, sum-xy), then per-cell
    O(1) covariance + 2×2 eigendecomposition.

    For N=10000 cells on a 4000×4000 image, runs in ~0.1s (was ~10s per
    cell × N = ~30 minutes with the np.where loop).
    """
    flat = m.ravel().astype(np.int64)
    Y, X = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    y_flat = Y.ravel().astype(np.float64)
    x_flat = X.ravel().astype(np.float64)

    # Moments via bincount, indexed by label.
    n = np.bincount(flat, minlength=N + 1)
    sy = np.bincount(flat, weights=y_flat, minlength=N + 1)
    sx = np.bincount(flat, weights=x_flat, minlength=N + 1)
    syy = np.bincount(flat, weights=y_flat * y_flat, minlength=N + 1)
    sxx = np.bincount(flat, weights=x_flat * x_flat, minlength=N + 1)
    sxy = np.bincount(flat, weights=y_flat * x_flat, minlength=N + 1)

    geom: Dict[int, dict] = {}
    for u in range(1, N + 1):
        area = int(n[u])
        if area == 0:
            geom[u] = dict(centroid=np.zeros(2), axis=np.array([1.0, 0.0]),
                           ecc=0.0, area=0)
            continue
        cy = sy[u] / area; cx = sx[u] / area
        if area < 3:
            geom[u] = dict(centroid=np.array([cy, cx]),
                           axis=np.array([1.0, 0.0]),
                           ecc=0.0, area=area)
            continue
        # Central second moments.
        myy = syy[u] / area - cy * cy
        mxx = sxx[u] / area - cx * cx
        mxy = sxy[u] / area - cy * cx
        # 2×2 covariance eigendecomp (closed form).
        tr = myy + mxx
        det = myy * mxx - mxy * mxy
        disc = max(tr * tr / 4 - det, 0.0)
        big = tr / 2 + np.sqrt(disc)
        small = tr / 2 - np.sqrt(disc)
        big = max(big, 1e-6); small = max(small, 1e-6)
        ecc = float(np.sqrt(max(0.0, 1.0 - small / big)))
        # Eigenvector for `big` eigenvalue.
        if abs(mxy) > 1e-12:
            ay = big - mxx; ax = mxy
        else:
            if myy >= mxx: ay, ax = 1.0, 0.0
            else:           ay, ax = 0.0, 1.0
        norm = np.hypot(ay, ax)
        if norm > 0: ay, ax = ay / norm, ax / norm
        geom[u] = dict(centroid=np.array([cy, cx]),
                       axis=np.array([ay, ax]),
                       ecc=float(ecc), area=area)
    return geom


def approximate_iou_similarity_cpp(m: np.ndarray,
                                     max_dist_factor: float = 5.0,
                                     min_score: float = 0.02):
    """C++ per-cell geometry + vectorised pair scoring."""
    geom = per_cell_geometry_cpp(m)
    N = int(geom["N"])
    if N < 2: return None, None, None, geom

    # Drop the 0-index (background slot).
    centroid_y = geom["centroid_y"][1:]
    centroid_x = geom["centroid_x"][1:]
    axis_y = geom["axis_y"][1:]
    axis_x = geom["axis_x"][1:]
    ecc = geom["ecc"][1:]
    area = geom["area"][1:].astype(np.float64)

    if area.size == 0 or area.sum() == 0:
        return None, None, None, geom
    mean_radius = float(np.sqrt(area.mean() / np.pi))
    max_dist = max_dist_factor * mean_radius
    sigma = 2.0 * mean_radius

    centroids = np.column_stack([centroid_y, centroid_x])
    tree = KDTree(centroids)
    pairs_idx = tree.query_pairs(r=max_dist, output_type="ndarray")
    if len(pairs_idx) == 0:
        return None, None, None, geom

    i = pairs_idx[:, 0]; j = pairs_idx[:, 1]
    axis_sim = np.abs(axis_y[i] * axis_y[j] + axis_x[i] * axis_x[j])
    area_ij = np.column_stack([area[i], area[j]])
    area_sim = area_ij.min(axis=1) / np.maximum(area_ij.max(axis=1), 1.0)
    dy = centroid_y[i] - centroid_y[j]
    dx = centroid_x[i] - centroid_x[j]
    d2 = dy * dy + dx * dx
    prox = np.exp(-d2 / (2.0 * sigma * sigma))
    ecc_term = np.where((ecc[i] < 0.3) | (ecc[j] < 0.3), 0.3, ecc[i] * ecc[j])
    scores = ecc_term * (axis_sim * axis_sim) * area_sim * prox
    keep = scores >= min_score
    # Return as 0-indexed arrays for direct C++ consumption.
    pair_u = i[keep].astype(np.int32)
    pair_v = j[keep].astype(np.int32)
    pair_w = scores[keep].astype(np.float64)
    return pair_u, pair_v, pair_w, geom


def approximate_iou_similarity(m: np.ndarray, N: int,
                                 max_dist_factor: float = 5.0,
                                 min_score: float = 0.02):
    """Approximate IoU via shape descriptors. FULLY VECTORISED.

    similarity(u, v) =  ecc_u · ecc_v · |axis_u · axis_v|²
                      · min(area_u, area_v)/max(area_u, area_v)
                      · exp(-d² / (2·σ²))

    For elongated cells this closely tracks exact centroid-aligned IoU.
    Computed in O(N log N + #pairs) via KDTree spatial queries; the per-
    pair scoring is a single batched numpy expression over the pair
    array (no Python loop).

    Returns (dict, geom) where dict maps (u, v) → score for pairs with
    score ≥ min_score.
    """
    geom = per_cell_geometry(m, N)
    if N < 2: return {}, geom

    # Pack per-cell arrays for vectorised pair scoring.
    centroid_y = np.array([geom[u]["centroid"][0] for u in range(1, N + 1)])
    centroid_x = np.array([geom[u]["centroid"][1] for u in range(1, N + 1)])
    axis_y = np.array([geom[u]["axis"][0] for u in range(1, N + 1)])
    axis_x = np.array([geom[u]["axis"][1] for u in range(1, N + 1)])
    ecc = np.array([geom[u]["ecc"] for u in range(1, N + 1)])
    area = np.array([geom[u]["area"] for u in range(1, N + 1)], dtype=np.float64)

    mean_radius = float(np.sqrt(area.mean() / np.pi))
    max_dist = max_dist_factor * mean_radius
    sigma = 2.0 * mean_radius

    centroids = np.column_stack([centroid_y, centroid_x])
    tree = KDTree(centroids)
    pairs_idx = tree.query_pairs(r=max_dist, output_type="ndarray")
    if len(pairs_idx) == 0:
        return {}, geom

    i = pairs_idx[:, 0]; j = pairs_idx[:, 1]

    # Vectorised score over the full pair array.
    axis_sim = np.abs(axis_y[i] * axis_y[j] + axis_x[i] * axis_x[j])
    area_ij = np.column_stack([area[i], area[j]])
    area_sim = area_ij.min(axis=1) / np.maximum(area_ij.max(axis=1), 1.0)
    dy = centroid_y[i] - centroid_y[j]
    dx = centroid_x[i] - centroid_x[j]
    d2 = dy * dy + dx * dx
    prox = np.exp(-d2 / (2.0 * sigma * sigma))
    # Round-cells (low ecc) get a baseline contribution so they aren't
    # ignored entirely.
    ecc_term = np.where((ecc[i] < 0.3) | (ecc[j] < 0.3), 0.3, ecc[i] * ecc[j])
    scores = ecc_term * (axis_sim * axis_sim) * area_sim * prox

    keep = scores >= min_score
    pair_sim: Dict[Tuple[int, int], float] = {}
    # 1-indexed cell labels.
    ii = (i[keep] + 1).astype(int)
    jj = (j[keep] + 1).astype(int)
    ss = scores[keep]
    for a, b, s in zip(ii, jj, ss):
        pair_sim[(int(a), int(b))] = float(s)
    return pair_sim, geom


# ----------------- Incremental Kempe-SA -----------------
def _kempe_comp(colors, adj, seed, c1, c2):
    if colors[seed] not in (c1, c2): return set()
    comp = {seed}; q = deque([seed])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in comp and colors[v] in (c1, c2):
                comp.add(v); q.append(v)
    return comp


def _two_hop_neighbors_list(adj, N):
    th = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        nbrs = adj[u]
        seen = set()
        for v in nbrs:
            for w in adj[v]:
                if w == u or w in nbrs: continue
                if w not in seen:
                    seen.add(w); th[u].append(w)
    return th


def _csr_from_neighbors(neighbors_list, N, weights_list=None):
    """Convert a 1-indexed adjacency-list-of-lists to a 0-indexed CSR
    (indptr, indices [, weights]). neighbors_list[u] for u in 1..N is
    the list of (neighbour_label) or (neighbour_label, weight) tuples."""
    indptr = np.zeros(N + 1, dtype=np.int32)
    for u in range(1, N + 1):
        indptr[u] = indptr[u - 1] + len(neighbors_list[u])
    total = indptr[N]
    indices = np.zeros(total, dtype=np.int32)
    if weights_list is not None:
        weights = np.zeros(total, dtype=np.float64)
    pos = 0
    for u in range(1, N + 1):
        for i, item in enumerate(neighbors_list[u]):
            if weights_list is not None:
                v, w = item
                indices[pos] = v - 1  # 0-indexed
                weights[pos] = w
            else:
                indices[pos] = item - 1
            pos += 1
    if weights_list is not None:
        return indptr, indices, weights
    return indptr, indices


def kempe_sa_native(colors, adj, two_hop, iou_pairs_by_cell, N,
                     alpha_2hop=1.0, gamma_iou=50.0,
                     n_iters=30000, T0=2.0, T_min=0.001, alpha_cool=0.9998,
                     rng_seed=0):
    """C++ implementation of Kempe-SA. ~20-50x faster than the Python loop."""
    from ._backend import _impl as _backend_impl
    _kempe_sa_cpp = _backend_impl.kempe_sa

    # Build CSR adjacency.
    adj_list = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in adj[u]:
            adj_list[u].append(v)
    adj_indptr, adj_indices = _csr_from_neighbors(adj_list, N)

    # 2-hop CSR (already in list form).
    th_list = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        for v in two_hop[u]:
            th_list[u].append(v)
    th_indptr, th_indices = _csr_from_neighbors(th_list, N)

    # IoU CSR with weights.
    iou_list = [[] for _ in range(N + 1)]
    for u in range(1, N + 1):
        for (v, s) in iou_pairs_by_cell[u]:
            iou_list[u].append((v, s))
    iou_indptr, iou_indices, iou_weights = _csr_from_neighbors(
        iou_list, N, weights_list=iou_list)

    # initial_colors: 0-indexed array of length N.
    initial = np.array(colors[1:N + 1], dtype=np.uint8)
    out_colors, best_loss = _kempe_sa_cpp(
        initial_colors=initial,
        adj_indptr=adj_indptr, adj_indices=adj_indices,
        twohop_indptr=th_indptr, twohop_indices=th_indices,
        iou_indptr=iou_indptr, iou_indices=iou_indices,
        iou_weights=iou_weights,
        n_colors=4, alpha_2hop=alpha_2hop, gamma_iou=gamma_iou,
        n_iters=n_iters, T0=T0, T_min=T_min, alpha_cool=alpha_cool,
        rng_seed=rng_seed,
    )
    # Back to 1-indexed list.
    new_colors = [0] * (N + 1)
    for i, c in enumerate(out_colors):
        new_colors[i + 1] = int(c)
    return new_colors, float(best_loss)


def kempe_sa_fast(colors, adj, two_hop, iou_pairs_by_cell, N,
                   alpha_2hop=1.0, gamma_iou=50.0,
                   n_iters=30000, T0=2.0, T_min=0.001, alpha_cool=0.9998,
                   rng_seed=0):
    """Kempe-SA with INCREMENTAL loss update.

    iou_pairs_by_cell[u] is a list of (v, score) for visual neighbours of u.
    Each Kempe swap touches O(boundary · avg_deg) pairs in the loss
    (only those crossing the swap component's boundary).
    """
    rng = random.Random(rng_seed)
    # Build per-cell neighbour-color counts? We compute losses directly.
    # Initial loss:
    n_2hop = sum(1 for u in range(1, N + 1) for v in two_hop[u]
                 if v > u and colors[u] == colors[v])
    iou_loss = 0.0
    # Avoid double-counting: iou_pairs_by_cell has both (u→v) and (v→u).
    iou_loss_double = 0.0
    for u in range(1, N + 1):
        for (v, s) in iou_pairs_by_cell[u]:
            if colors[u] == colors[v]:
                iou_loss_double += s
    iou_loss = iou_loss_double / 2.0
    cur_loss = alpha_2hop * n_2hop + gamma_iou * iou_loss
    best_colors = list(colors); best_loss = cur_loss
    T = T0

    for it in range(n_iters):
        if best_loss <= 0 and gamma_iou == 0 and alpha_2hop == 0: break
        u = rng.randint(1, N)
        cu = colors[u]
        if cu == 0: continue
        Y = rng.choice([c for c in range(1, 5) if c != cu])
        comp = _kempe_comp(colors, adj, u, cu, Y)

        # Incrementally compute delta loss for the swap.
        comp_set = comp  # set
        delta_2hop = 0
        delta_iou = 0.0
        for w in comp:
            cw_old = colors[w]
            cw_new = Y if cw_old == cu else cu
            # 2-hop neighbours crossing boundary.
            for nb in two_hop[w]:
                if nb in comp_set: continue
                cv = colors[nb]
                old_same = (cw_old == cv)
                new_same = (cw_new == cv)
                delta_2hop += int(new_same) - int(old_same)
            # IoU pairs crossing boundary.
            for (nb, s) in iou_pairs_by_cell[w]:
                if nb in comp_set: continue
                cv = colors[nb]
                old_same = (cw_old == cv)
                new_same = (cw_new == cv)
                delta_iou += (int(new_same) - int(old_same)) * s

        # Adjust 2-hop double-counting (pairs both in comp would cancel
        # — both endpoints flip same colour, so old_same == new_same).
        # No correction needed because we skip in_comp neighbours.

        delta = alpha_2hop * delta_2hop + gamma_iou * delta_iou
        new_loss = cur_loss + delta
        # Metropolis.
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
            cur_loss = new_loss
            for w in comp: colors[w] = Y if colors[w] == cu else cu
            if cur_loss < best_loss:
                best_loss = cur_loss; best_colors = list(colors)
        # Cool.
        T = max(T_min, T * alpha_cool)

    for i in range(N + 1): colors[i] = best_colors[i]
    return colors, best_loss


def label_geometric_fast(lab, p: int = 1, conn: int = 2,
                          alpha_2hop: float = 1.0,
                          gamma_iou: float = 50.0,
                          sa_iters: int = 30000,
                          verbose: bool = False):
    """Scalable geometry-aware 4-colouring.

    Same objective as ``label_geometric_hybrid`` but with fast pair
    enumeration (KDTree + approximate-IoU shape descriptors) and
    incremental Kempe-SA loss updates. Designed for N ∼ 10⁴ cells.
    """
    import time
    from .expand import expand_labels
    from .color import connect
    from .color import label as _ncolor_label

    t_start = time.time()
    lab_arr = np.asarray(lab)
    expanded = expand_labels(lab_arr, p=p)
    pairs = connect(expanded, conn=conn)
    N = int(lab_arr.max())
    adj: Dict[int, set] = {u: set() for u in range(1, N + 1)}
    for u, v in pairs:
        adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
    if verbose: print(f"  adjacency: N={N}, edges={sum(len(s) for s in adj.values())//2}, {time.time()-t_start:.2f}s")

    # Seed: greedy. Extract per-cell colour via vectorised np.unique.
    nc_seed = _ncolor_label(lab_arr, expand=True, p=p, balance=True,
                              weight_objective=0)
    flat_lab = lab_arr.ravel(); flat_nc = nc_seed.ravel()
    unique_labels, first_idx = np.unique(flat_lab, return_index=True)
    # Build 0-indexed initial_colors array directly (skip the 1-indexed
    # Python list and the second copy in kempe_sa_native).
    initial_colors = np.zeros(N, dtype=np.uint8)
    valid = (unique_labels > 0) & (unique_labels <= N)
    initial_colors[unique_labels[valid] - 1] = flat_nc[first_idx[valid]]
    if verbose: print(f"  greedy seed: {time.time()-t_start:.2f}s")

    # Build 1-hop CSR (0-indexed) from adj dict.
    adj_degrees = np.array([len(adj[u]) for u in range(1, N + 1)], dtype=np.int32)
    adj_indptr = np.zeros(N + 1, dtype=np.int32)
    adj_indptr[1:] = np.cumsum(adj_degrees)
    adj_indices = np.zeros(int(adj_indptr[N]), dtype=np.int32)
    pos = 0
    for u in range(1, N + 1):
        for v in adj[u]:
            adj_indices[pos] = v - 1; pos += 1

    # 2-hop CSR via C++.
    from ._backend import _impl as _backend_impl
    th_indptr, th_indices = _backend_impl.two_hop_csr(adj_indptr, adj_indices)
    if verbose: print(f"  2-hop CSR: {th_indices.size} edges, {time.time()-t_start:.2f}s")

    # Approximate IoU pairs via C++ geometry + vectorised scoring.
    pair_u, pair_v, pair_w, geom = approximate_iou_similarity_cpp(lab_arr)
    if pair_u is None:
        pair_u = np.zeros(0, dtype=np.int32)
        pair_v = np.zeros(0, dtype=np.int32)
        pair_w = np.zeros(0, dtype=np.float64)
    iou_indptr, iou_indices, iou_weights = _backend_impl.symmetric_pair_csr(
        pair_u, pair_v, pair_w, N)
    if verbose: print(f"  IoU pairs: {len(pair_u)}, {time.time()-t_start:.2f}s")

    # Kempe-SA — C++ implementation, called directly with CSR arrays.
    out_colors, best_loss = _backend_impl.kempe_sa(
        initial_colors=initial_colors,
        adj_indptr=adj_indptr, adj_indices=adj_indices,
        twohop_indptr=th_indptr, twohop_indices=th_indices,
        iou_indptr=iou_indptr, iou_indices=iou_indices,
        iou_weights=iou_weights,
        n_colors=4, alpha_2hop=alpha_2hop, gamma_iou=gamma_iou,
        n_iters=sa_iters)
    # Back to 1-indexed for LUT.
    colors = [0] * (N + 1)
    for i, c in enumerate(out_colors):
        colors[i + 1] = int(c)
    loss = float(best_loss)
    if verbose:
        print(f"  Kempe-SA: loss={loss:.2f}, {time.time()-t_start:.2f}s")

    lut = np.array(colors, dtype=np.uint8)
    return lut[expanded].astype(np.uint8)
