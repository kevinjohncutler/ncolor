#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements

import numpy as np
import fastremap
import heapq

# Import heavy dependencies but defer JIT compilation
from numba import njit
import scipy
from scipy.ndimage import distance_transform_edt
from .format_labels import format_labels

def is_sequential(labels):
    return np.all(np.diff(fastremap.unique(labels))==1)
    
def unique_nonzero(labels):
    """
    Get unique nonzero labels.
    """
    sub = labels[labels > 0].astype(int)
    if not sub.flags.writeable:
        sub = sub.copy()
    if sub.size:
        return fastremap.unique(sub)
    else:
        return np.array([], dtype=labels.dtype)
        

        
# def unique_nonzero(labels):
#     """
#     Get unique nonzero labels. 
#     """
#     if np.any(labels):
#         sub = labels[labels > 0]
#         if not sub.flags.writeable:
#             sub = sub.copy()
#         return fastremap.unique(sub)
#     else:
#         return np.array([0])


def label(lab, n=4, conn=2, max_depth=30, offset=0, expand=True, return_n=False, greedy=False, experimental=False, verbose=False, check_conflicts=False, return_conflicts=False):
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space
    # if not is_sequential(lab):
    #     lab = format_labels(lab)

    if verbose: print('verbose')
    # Use label_old as the default solver (legacy fast path).
    if not greedy and not experimental:
        from . import label_old as label_old_impl
        if expand:
            if isinstance(expand, str) and expand.lower() in ("fast", "wave", "approx", "wavefront"):
                lab_exp = expand_labels_wavefront(lab, conn=2)
            else:
                lab_exp = expand_labels(lab)
        else:
            lab_exp = lab
        colored_exp = label_old_impl.label(lab_exp, n=n, conn=conn, max_depth=max_depth, offset=offset)
        if colored_exp is None:
            raise ValueError("label_old failed to produce a valid coloring.")
        colored = colored_exp * (lab != 0) if expand else colored_exp
        conflicts = 0
        if check_conflicts or return_conflicts:
            lab_fmt = format_labels(lab_exp).astype(np.int32)
            pairs = connect(lab_fmt, conn)
            if pairs.size:
                lut = np.zeros(lab_fmt.max() + 1, dtype=colored_exp.dtype)
                lut[lab_fmt] = colored_exp
                conflicts = int(np.count_nonzero(lut[pairs[:, 0]] == lut[pairs[:, 1]]))
            if check_conflicts and conflicts:
                raise ValueError(f"Coloring conflict detected: {conflicts} adjacent pairs share a color.")
        if return_n and return_conflicts:
            return colored, int(colored.max()), conflicts
        if return_n:
            return colored, int(colored.max())
        if return_conflicts:
            return colored, conflicts
        return colored
    pad = 1
    unpad = tuple([slice(pad,-pad)]*lab.ndim)
    mask = lab!=0
    unique = unique_nonzero(lab)
    if verbose: print('number of masks', unique.size)
    
    if unique.size==1:
        ncl = mask.astype(int)
        nc = 1
        conflicts = 0
    else:
        # Expand only when explicitly requested
        if expand:
            if isinstance(expand, str) and expand.lower() in ("fast", "wave", "approx", "wavefront"):
                lab = expand_labels_wavefront(lab, conn=2)
            else:
                lab = expand_labels(lab)
        # lab = np.pad(format_labels(lab),pad)
        lab_padded = np.pad(lab, pad)
        if np.issubdtype(lab_padded.dtype, np.integer) and lab_padded.min() == 0 and is_sequential(lab_padded):
            lab = lab_padded.astype(np.int32, copy=False)
        else:
            lab = format_labels(lab_padded, background=0)
        lut = get_lut(lab,n,conn,max_depth,offset,greedy, experimental, verbose)

        conflicts = 0
        if check_conflicts or return_conflicts:
            pairs = connect(lab, conn)
            if pairs.size:
                a = pairs[:, 0]
                b = pairs[:, 1]
                conflicts = int(np.count_nonzero(lut[a] == lut[b]))
            if check_conflicts and conflicts:
                raise ValueError(f"Coloring conflict detected: {conflicts} adjacent pairs share a color.")

        ncl = lut[lab][unpad]*mask
        nc = np.max(lut)
    
    if return_n and return_conflicts:
        return ncl, nc, conflicts
    if return_n:
        return ncl, nc
    if return_conflicts:
        return ncl, conflicts
    return ncl

def get_lut(lab, n=4, conn=2, max_depth=30, offset=0, greedy=False, experimental=False, verbose=False):
    if np.issubdtype(lab.dtype, np.integer) and lab.min() == 0 and is_sequential(lab):
        lab = lab.astype(np.int32, copy=False)
    else:
        lab = format_labels(lab).astype(np.int32)

    pairs = connect(lab, conn)

    # Handle case where no masks touch one another.
    if pairs.size == 0:
        lut = np.ones(lab.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut

    if experimental:
        colors = render_net_experimental(pairs, n=n, rand=0, max_depth=max_depth, offset=offset, verbose=verbose)
    elif greedy:
        conmap = mapidx(pairs)
        colors = greedy_coloring(conmap)
    else:
        conmap = mapidx(pairs)
        colors = _render_net_legacy(conmap, n=n, rand=10, max_depth=max_depth, offset=offset)
        if colors is None:
            raise ValueError(f"Failed to color the labels with {n} colors. Try increasing n or max_depth.")
        colors = _legacy_color_with_repairs(pairs, colors, n, max_depth)
        if colors is None:
            raise ValueError(f"Failed to repair legacy coloring with {n} colors. Try increasing n or max_depth.")

    lut = np.ones(lab.max() + 1, dtype=np.uint8)
    for i in colors:
        lut[i] = colors[i]
    lut[0] = 0
    return lut

    
def neighbors(shape, conn=1, unique=True):
    dim = len(shape)
    block = scipy.ndimage.generate_binary_structure(dim, conn)
    block[tuple([1] * dim)] = 0
    idx = np.array(np.where(block > 0)).T  # shape: (K, dim) in {0,1,2}
    idx = idx - 1  # map to {-1,0,1}

    if unique:
        # Keep only offsets where the first nonzero component is positive
        keep = []
        for off in idx:
            sel = 0
            for v in off:
                if v != 0:
                    sel = 1 if v > 0 else 0
                    break
            keep.append(sel)
        idx = idx[np.array(keep, dtype=bool)]

    # Flattened offset for row-major array
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])



# @njit(fastmath=True, cache=True)
@njit(cache=True)
def search(img, nbs):
    line = img.ravel()
    total = len(line)
    D = len(nbs)

    # Worst case: every pixel connects to every neighbor
    rst = np.empty((total * D, 2), img.dtype)
    s = 0

    for i in range(total):
        vi = line[i]
        if vi == 0:
            continue
        for d in nbs:
            j = i + d
            # Guard out-of-bounds indices (NumPy negative wraps; disallow here)
            if j < 0 or j >= total:
                continue
            vj = line[j]
            if vj == 0:
                continue
            if vi == vj:
                continue
            a = vi
            b = vj
            if a > b:
                a, b = b, a
            rst[s, 0] = a
            rst[s, 1] = b
            s += 1

    return rst[:s]
                            

    
def connect(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = neighbors(buf.shape, conn, unique=True)
    rst = search(buf, nbs)
    if rst.size == 0:
        return rst

    a = rst[:, 0]; b = rst[:, 1]
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    # Sort by (lo, hi)
    order = np.lexsort((hi, lo))
    lo_s = lo[order]; hi_s = hi[order]

    # Unique consecutive pairs
    keep = np.empty(lo_s.size, dtype=bool)
    if keep.size:
        keep[0] = True
        keep[1:] = (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])

    m = int(keep.sum())
    out = np.empty((m, 2), dtype=lo.dtype)
    out[:, 0] = lo_s[keep]
    out[:, 1] = hi_s[keep]
    return out



def mapidx(idx):
    # Stack both directions
    idx_rev = idx[:, [1, 0]]
    idx_all = np.vstack((idx, idx_rev))

    # Sort by first col, then second (deterministic groups)
    order = np.lexsort((idx_all[:, 1], idx_all[:, 0]))
    idx_all_sorted = idx_all[order]

    i = idx_all_sorted[:, 0]
    j = idx_all_sorted[:, 1]

    # Run-length grouping via counts → no np.split object churn
    unique_i, counts = fastremap.unique(i, return_counts=True) # test np as well
    
    ends = np.cumsum(counts)
    starts = ends - counts

    # Views into j, no copies
    return {unique_i[k]: j[starts[k]:ends[k]] for k in range(unique_i.size)}


# Numba-accelerated CSR colorer
@njit(cache=True)
def _color_graph_csr(indptr, indices, n, rand, max_iter):
    N = indptr.size - 1
    colors = np.zeros(N, np.uint8)
    color_counts = np.zeros(n + 1, np.int32)  # global usage to balance distribution
    counter = np.zeros(N, np.int32)

    # queue as a (growable) ring buffer; start large enough to hold all edges
    qcap = max(indices.size + N, N * 8, 1)
    q = np.empty(qcap, np.int32)
    head = 0
    tail = N
    for i in range(N):
        q[i] = i

    # bitmask for colors 1..n: bits 1..n set → (1<<(n+1)) - 2
    fullmask = (1 << (n + 1)) - 2

    count = 0
    while head < tail and count < max_iter:
        u = q[head]
        head += 1
        count += 1
        counter[u] += 1

        # First pass: build presence mask, early-exit if all colors present
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        mask = 0
        all_present = False
        for k in range(row_beg, row_end):
            cv = colors[indices[k]]
            if cv != 0:
                mask |= (1 << cv)
                if mask == fullmask:
                    all_present = True
                    break

        if not all_present:
            # choose the missing color with lowest global usage to balance distribution
            csel = 0
            best_count = 2147483647
            for c in range(1, n + 1):
                if (mask & (1 << c)) == 0:
                    cnt = color_counts[c]
                    if cnt < best_count:
                        best_count = cnt
                        csel = c
            if csel == 0:
                csel = n  # fallback (should not happen)
        else:
            # Second pass only when needed: count frequencies
            # cnt[0] unused
            cnt = np.zeros(n + 1, np.int32)
            for k in range(row_beg, row_end):
                cv = colors[indices[k]]
                if cv != 0:
                    cnt[cv] += 1
            # pick least frequent color in 1..n
            csel = 1
            minv = cnt[1]
            for c in range(2, n + 1):
                if cnt[c] < minv:
                    minv = cnt[c]
                    csel = c
            if rand > 0 and counter[u] == rand:
                counter[u] = 0
                # lightweight LCG for pseudo-random choice in [1, n]
                seed = (1103515245 * (u + count) + 12345) & 0x7FFFFFFF
                csel = 1 + (seed % n)

        if colors[u] != csel:
            oldc = colors[u]
            colors[u] = csel
            if oldc != 0:
                color_counts[oldc] -= 1
            color_counts[csel] += 1
            # re-enqueue only neighbors that now conflict
            for k in range(row_beg, row_end):
                v = indices[k]
                if colors[v] == csel:
                    if tail >= qcap:
                        # grow queue to avoid silently dropping conflicts
                        newcap = max(qcap * 2, tail + 1)
                        q_new = np.empty(newcap, np.int32)
                        q_new[:tail] = q[:tail]
                        q = q_new
                        qcap = newcap
                    q[tail] = v
                    tail += 1

    return colors, (head < tail)


@njit(cache=True)
def _repair_coloring(indptr, indices, colors, n, max_passes=4):
    """
    Attempt a local repair of coloring conflicts by recoloring nodes to the
    smallest available color in 1..n not used by neighbors. Runs a few passes
    to converge. Returns (colors, conflict_exists).
    """
    N = indptr.size - 1
    passes = 0
    changed = True
    while changed and passes < max_passes:
        changed = False
        passes += 1
        for u in range(N):
            cu = colors[u]
            # Build neighbor color bitmask
            mask = 0
            row_beg = indptr[u]
            row_end = indptr[u + 1]
            for k in range(row_beg, row_end):
                cv = colors[indices[k]]
                if cv != 0:
                    mask |= (1 << cv)
            # If current color is zero or conflicting, pick smallest free color
            if cu == 0 or (mask & (1 << cu)) != 0:
                csel = 0
                for c in range(1, n + 1):
                    if (mask & (1 << c)) == 0:
                        csel = c
                        break
                if csel != 0 and csel != cu:
                    colors[u] = csel
                    changed = True

    # Final conflict check
    conflict = False
    for u in range(N):
        cu = colors[u]
        if cu == 0:
            continue
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        for k in range(row_beg, row_end):
            v = indices[k]
            if v == u:
                continue
            if colors[v] == cu:
                conflict = True
                break
        if conflict:
            break
    return colors, conflict


@njit(cache=True)
def _greedy_coloring_csr(indptr, indices, n_try):
    """
    Greedy coloring on CSR graph using degree-descending order.
    Tries to stay within 1..n_try colors; if not possible, may use more.
    Returns colors array (uint8) per node index.
    """
    N = indptr.size - 1
    colors = np.zeros(N, np.uint8)
    degrees = np.empty(N, np.int32)
    for u in range(N):
        degrees[u] = indptr[u + 1] - indptr[u]
    # argsort descending by degree
    order = np.argsort(-degrees)

    for t in range(N):
        u = int(order[t])
        # build neighbor color mask (use 64-bit to allow up to 63 colors)
        mask = np.uint64(0)
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        for k in range(row_beg, row_end):
            cv = colors[indices[k]]
            if cv != 0:
                mask |= (np.uint64(1) << np.uint64(cv))
        # choose smallest available color in 1..n_try
        csel = 0
        for c in range(1, n_try + 1):
            if (mask & (np.uint64(1) << np.uint64(c))) == 0:
                csel = c
                break
        if csel == 0:
            # allow using more colors if necessary
            for c in range(n_try + 1, 64):
                if (mask & (np.uint64(1) << np.uint64(c))) == 0:
                    csel = c
                    break
        colors[u] = np.uint8(csel)

    return colors


@njit(cache=True)
def _kempe_repair_csr(indptr, indices, colors, n, max_passes=2):
    """
    Attempt to resolve remaining conflicts by Kempe-chain swaps within colors 1..n.
    For each conflicting edge (u,v) with the same color c, try an alternate color a
    and flip the connected component of the subgraph induced by colors {c,a} that
    contains u. If v is not in that component, the swap resolves the conflict.
    Returns (colors, conflict_exists).
    """
    N = indptr.size - 1

    comp = np.empty(N, np.int32)
    q = np.empty(N, np.int32)
    visited = np.zeros(N, np.uint8)

    def has_conflict_local(u):
        cu = colors[u]
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        for k in range(row_beg, row_end):
            v = indices[k]
            if v == u:
                continue
            if colors[v] == cu and cu != 0:
                return v
        return -1

    passes = 0
    while passes < max_passes:
        passes += 1
        fixed_any = False
        for u in range(N):
            vconf = has_conflict_local(u)
            if vconf == -1:
                continue
            cu = colors[u]

            # Try alternate colors 1..n except cu
            for a in range(1, n + 1):
                if a == cu:
                    continue

                # BFS in {cu, a} starting from u
                for i in range(N):
                    visited[i] = 0
                head = 0
                tail = 0
                q[tail] = u
                tail += 1
                visited[u] = 1
                comp_size = 0
                reachable_conf = False

                while head < tail:
                    x = q[head]
                    head += 1
                    comp[comp_size] = x
                    comp_size += 1
                    rb = indptr[x]
                    re = indptr[x + 1]
                    for kk in range(rb, re):
                        y = indices[kk]
                        if visited[y] != 0:
                            continue
                        cy = colors[y]
                        if cy == cu or cy == a:
                            visited[y] = 1
                            q[tail] = y
                            tail += 1
                            if y == vconf:
                                reachable_conf = True
                                # Cannot resolve with this a
                                break
                    if reachable_conf:
                        break

                if reachable_conf:
                    continue

                # Swap colors in the component: cu <-> a
                for ii in range(comp_size):
                    node = comp[ii]
                    cnode = colors[node]
                    if cnode == cu:
                        colors[node] = a
                    elif cnode == a:
                        colors[node] = cu

                fixed_any = True
                break  # move to next node u

        if not fixed_any:
            break

    # Final conflict check
    conflict = False
    for u in range(N):
        cu = colors[u]
        if cu == 0:
            continue
        rb = indptr[u]
        re = indptr[u + 1]
        for kk in range(rb, re):
            v = indices[kk]
            if v == u:
                continue
            if colors[v] == cu:
                conflict = True
                break
        if conflict:
            break

    return colors, conflict


def _csr_color_with_repairs(indptr, indices, n, max_depth=5, verbose=False):
    """
    Try the CSR solver and apply deterministic repair steps when needed.
    """
    max_depth = max(1, max_depth)
    max_iter = max((indices.size + int(indptr.size)) * max_depth, 512)
    colors, unfinished = _color_graph_csr(indptr, indices, n, rand=0, max_iter=max_iter)
    needs_repair = unfinished or (colors == 0).any()
    if needs_repair:
        if verbose:
            print("CSR solver incomplete, running conflict repair.")
        colors, conflict = _repair_coloring(indptr, indices, colors, n, max_passes=max(4, max_depth))
        needs_repair = conflict or (colors == 0).any()
    if needs_repair:
        if verbose:
            print("CSR conflict repair left unresolved, attempting Kempe swaps.")
        colors, conflict = _kempe_repair_csr(indptr, indices, colors, n, max_passes=max(2, max_depth))
        needs_repair = conflict or (colors == 0).any()
    if needs_repair:
        return None
    return colors


def _legacy_color_with_repairs(pairs, colors, n, max_depth):
    nodes, indptr, indices = _build_csr_from_pairs(pairs)
    if nodes.size == 0:
        return {}
    color_arr = np.zeros(nodes.size, dtype=np.uint8)
    for i, node in enumerate(nodes):
        color_arr[i] = np.uint8(colors.get(int(node), 0))
    color_arr, conflict = _repair_coloring(indptr, indices, color_arr, n, max_passes=max(4, max_depth))
    needs_repair = conflict or (color_arr == 0).any()
    if needs_repair:
        color_arr, conflict = _kempe_repair_csr(indptr, indices, color_arr, n, max_passes=max(2, max_depth))
        needs_repair = conflict or (color_arr == 0).any()
    if needs_repair:
        return None
    return {int(nodes[i]): int(color_arr[i]) for i in range(nodes.size)}

def _build_csr_from_pairs(pairs_arr):
    pairs_sym = np.vstack((pairs_arr, pairs_arr[:, [1, 0]]))
    order = np.argsort(pairs_sym[:, 0], kind='mergesort')
    src = pairs_sym[order, 0]
    dst = pairs_sym[order, 1]
    unique_src, counts = fastremap.unique(src, return_counts=True)
    N = unique_src.size
    indptr = np.empty(N + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    id2idx = {int(v): i for i, v in enumerate(unique_src.tolist())}
    indices = fastremap.remap(dst, id2idx, preserve_missing_labels=False).astype(np.int32, copy=False)
    return unique_src, indptr, indices


def _csr_from_dict(conmap):
    nodes = np.fromiter(conmap.keys(), dtype=np.int64, count=len(conmap))
    degrees = np.fromiter((len(conmap[int(nid)]) for nid in nodes), dtype=np.int32, count=len(conmap))
    indptr = np.empty(len(conmap) + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(degrees, out=indptr[1:])
    if indptr[-1] == 0:
        return nodes, indptr, np.empty(0, dtype=np.int32)
    concat_neighbors = np.empty(indptr[-1], dtype=np.int64)
    pos = 0
    for nid in nodes:
        arr = conmap[int(nid)]
        ln = len(arr)
        concat_neighbors[pos:pos + ln] = arr
        pos += ln
    id2idx = {int(v): i for i, v in enumerate(nodes.tolist())}
    indices = fastremap.remap(concat_neighbors, id2idx, preserve_missing_labels=False).astype(np.int32, copy=False)
    return nodes, indptr, indices


@njit(cache=True)
def _degeneracy_order_jit(indptr, indices):
    N = indptr.size - 1
    if N == 0:
        return np.empty(0, dtype=np.int32)
    degrees = np.empty(N, dtype=np.int32)
    max_deg = 0
    for i in range(N):
        d = int(indptr[i + 1] - indptr[i])
        degrees[i] = d
        if d > max_deg:
            max_deg = d
    head = np.full(max_deg + 1, -1, dtype=np.int32)
    next_ptr = np.full(N, -1, dtype=np.int32)
    for u in range(N):
        d = degrees[u]
        next_ptr[u] = head[d]
        head[d] = u
    removed = np.zeros(N, dtype=np.uint8)
    order = np.empty(N, dtype=np.int32)
    current = 0
    count = 0
    while count < N:
        while current <= max_deg and head[current] == -1:
            current += 1
        if current > max_deg:
            break
        u = head[current]
        head[current] = next_ptr[u]
        if removed[u] or degrees[u] != current:
            continue
        removed[u] = 1
        order[count] = u
        count += 1
        for k in range(indptr[u], indptr[u + 1]):
            v = indices[k]
            if removed[v]:
                continue
            degrees[v] -= 1
            nd = degrees[v]
            next_ptr[v] = head[nd]
            head[nd] = v
            if nd < current:
                current = nd
    if count < N:
        out = np.empty(count, dtype=np.int32)
        for i in range(count):
            out[i] = order[i]
        return out
    return order


@njit(cache=True)
def _experimental_color_from_order_jit(indptr, indices, n):
    order = _degeneracy_order_jit(indptr, indices)
    N = indptr.size - 1
    colors = np.zeros(N, dtype=np.uint8)
    fullmask = (1 << (n + 1)) - 2
    K = order.size
    for idx in range(K - 1, -1, -1):
        u = order[idx]
        mask = 0
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        for k in range(row_beg, row_end):
            v = indices[k]
            cv = colors[v]
            if cv != 0:
                mask |= (1 << cv)
                if mask == fullmask:
                    break
        csel = 0
        for c in range(1, n + 1):
            if (mask & (1 << c)) == 0:
                csel = c
                break
        if csel == 0:
            return np.empty(0, dtype=np.uint8)
        colors[u] = np.uint8(csel)
    return colors


def _experimental_color_from_order(indptr, indices, n, verbose=False):
    colors = _experimental_color_from_order_jit(indptr, indices, n)
    if colors.size == 0:
        if verbose:
            print("Experimental degeneracy ordering could not fit within the color budget.")
        return None
    return colors


def render_net_experimental(conmap, n=4, rand=0, depth=0, max_depth=5, offset=0, verbose=False):
    if isinstance(conmap, dict):
        if verbose:
            print("experimental dict path")
        if len(conmap) == 0:
            return {}
        nodes, indptr, indices = _csr_from_dict(conmap)
        if indptr[-1] == 0:
            return {int(nid): 1 for nid in nodes}
    else:
        pairs = np.asarray(conmap, dtype=np.int64)
        if pairs.size == 0:
            return {}
        nodes, indptr, indices = _build_csr_from_pairs(pairs)

    N = nodes.size
    colors = _experimental_color_from_order(indptr, indices, n, verbose=verbose)
    if colors is None:
        if verbose:
            print("Experimental solver failed, falling back to CSR/DSATUR.")
        # Fall back to the stable baseline solver for full coverage.
        return render_net(conmap, n=n, rand=rand, depth=depth, max_depth=max_depth, offset=offset, verbose=verbose)

    return {int(nodes[i]): int(colors[i]) for i in range(N)}


def _render_net_legacy(conmap, n=4, rand=12, depth=0, max_depth=30, offset=0):
    node_count = len(conmap)
    thresh = max(10000, node_count * 20)
    if depth < max_depth:
        nodes = sorted(conmap.keys())
        rng = np.random.default_rng(depth + 1 + offset)
        rng.shuffle(nodes)
        colors = dict(zip(nodes, [0] * len(nodes)))
        counter = dict(zip(nodes, [0] * len(nodes)))
        count = 0
        while len(nodes) > 0 and count < thresh:
            count += 1
            k = nodes.pop(0)
            counter[k] += 1
            hist = [1e4] + [0] * n
            for p in conmap[k]:
                hist[colors[p]] += 1
            if min(hist) == 0:
                colors[k] = hist.index(min(hist))
                counter[k] = 0
                continue
            hist[colors[k]] = 1e4
            minc = hist.index(min(hist))
            if counter[k] == rand:
                counter[k] = 0
                minc = int(rng.integers(1, n + 1))
            colors[k] = minc
            for p in conmap[k]:
                if colors[p] == minc:
                    nodes.append(p)
        if count == thresh:
            return _render_net_legacy(conmap, n + 1, rand, depth + 1, max_depth, offset)
        return colors
    return None

def render_net(conmap, n=4, rand=0, depth=0, max_depth=5, offset=0, verbose=False):
    """
    Fast DSATUR-style coloring on CSR without post cleanup.
    """
    if isinstance(conmap, dict):
        if verbose:
            print("using dict")
        if len(conmap) == 0:
            return {}
        nodes, indptr, indices = _csr_from_dict(conmap)
        if indptr[-1] == 0:
            return {int(nid): 1 for nid in nodes}
    else:
        pairs = np.asarray(conmap, dtype=np.int64)
        if pairs.size == 0:
            return {}
        nodes, indptr, indices = _build_csr_from_pairs(pairs)

    N = nodes.size
    colors = np.zeros(N, np.uint8)
    sat_mask = np.zeros(N, np.uint64)
    degrees = indptr[1:] - indptr[:-1]

    heap = []
    for i in range(N):
        heapq.heappush(heap, (-0, -int(degrees[i]), i))
    failed = False
    while heap:
        _, _, u = heapq.heappop(heap)
        if colors[u] != 0:
            continue
        used = sat_mask[u]
        csel = 0
        for c in range(1, n + 1):
            if (used & (np.uint64(1) << np.uint64(c))) == 0:
                csel = c
                break
        if csel == 0:
            failed = True
            if verbose:
                print(f"DSATUR failed to assign a color using {n} colors, triggering CSR repair.")
            break
        colors[u] = np.uint8(csel)
        bit = np.uint64(1) << np.uint64(csel)
        rb = indptr[u]; re = indptr[u + 1]
        for k in range(rb, re):
            v = indices[k]
            if colors[v] == 0:
                sat_mask[v] |= bit
                sat = bin(sat_mask[v]).count("1")
                heapq.heappush(heap, (-sat, -int(degrees[v]), v))

    if failed or (colors == 0).any():
        if verbose:
            print("Switching to CSR solver with deterministic repairs.")
        csr_colors = _csr_color_with_repairs(indptr, indices, n, max_depth, verbose=verbose)
        if csr_colors is None:
            raise ValueError(f"Failed to color graph with {n} colors after deterministic repairs.")
        colors = csr_colors

    return {int(nodes[i]): int(colors[i]) for i in range(N)}
    
def greedy_coloring(conmap):
    # faster and uses fewer colors than render_net, much uglier 
    colors = {}
    for node in conmap:
        neighbor_colors = {colors.get(neigh) for neigh in conmap[node] if neigh in colors}
        for color in range(1, len(conmap) + 1):
            if color not in neighbor_colors:
                colors[node] = color
                break
    return colors
    

def expand_labels(label_image):
    """
    Sped-up version of the scikit-image function just by dropping the distance thresholding. 
    Here we expand the labels into every background pixel. Can be over 40% faster. 
    """
    nearest_label_coords = distance_transform_edt(label_image==0, 
                                                  return_distances=False, 
                                                  return_indices=True)
    return label_image[tuple(nearest_label_coords)]
    
    
@njit(cache=True)
def _expand_wavefront_core(line, shape_flat, nbs):
    total = line.size
    # Simple ring buffer queue
    q = np.empty(total, np.int64)
    head = 0
    tail = 0
    # Seed queue with all labeled pixels
    for i in range(total):
        if line[i] != 0:
            q[tail] = i
            tail += 1
    if head == tail:
        return  # nothing to do

    # Propagate labels outward to zero pixels
    while head < tail:
        i = q[head]
        head += 1
        vi = line[i]
        for d in nbs:
            j = i + d
            if j < 0 or j >= total:
                continue
            if line[j] == 0:
                line[j] = vi
                q[tail] = j
                tail += 1


def expand_labels_wavefront(label_image, conn=2):
    """
    Fast multi-source wavefront expansion of labels into background (approximate).
    Fills every 0 pixel with the nearest label via uniform-cost BFS using
    the chosen connectivity (2 for 8-neighborhood in 2D). This approximates
    Euclidean nearest-label from EDT at a fraction of the cost.
    """
    if not np.any(label_image == 0):
        return label_image
    buf = np.pad(label_image, 1, mode='constant')
    nbs = neighbors(buf.shape, conn=conn, unique=False)
    line = buf.ravel()
    _expand_wavefront_core(line, buf.shape, nbs)
    unpad = tuple([slice(1, -1)] * buf.ndim)
    return buf[unpad]


#################### experiments
def get_lut_expanded(lab, n=4, conn=2, max_depth=5, offset=0, greedy=False, verbose=False):
    """
    Compute a color LUT as if labels were expanded into background, without
    materializing the expanded array. Works by expanding ownership only across
    background and extracting adjacency from those visited background pixels,
    then coloring the resulting graph.

    Parameters
    ----------
    lab : ndarray
        Original label image (0 = background).
    n, conn, max_depth, offset, greedy : see get_lut.

    Returns
    -------
    lut : ndarray (uint8)
        Lookup table mapping label id -> color, with lut[0] = 0.
    """
    # Pad and format to standard form with background=0 (matches label() path)
    lab0 = format_labels(np.pad(lab, 1), background=0).astype(np.int32)

    # Early exit: no touching masks after expansion (single label or none)
    if lab0.max() <= 1:
        lut = np.ones(lab0.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut

    # Prepare a working buffer and expand labels into zeros in-place
    buf = lab0.copy()
    line = buf.ravel()
    # All directions for propagation
    nbs_all = neighbors(buf.shape, conn=conn, unique=False)
    _expand_wavefront_core(line, buf.shape, nbs_all)

    # Build adjacency by looking only at background pixels from the original image
    flat0 = lab0.ravel()
    zeros_idx = np.flatnonzero(flat0 == 0)
    if zeros_idx.size == 0:
        # No background: fallback to direct connect on labels
        pairs = connect(lab0, conn)
    else:
        # Half-space offsets to avoid double counting
        nbs_half = neighbors(buf.shape, conn=conn, unique=True)
        pairs_list = []
        for d in nbs_half:
            j = zeros_idx + d
            # in-bounds mask
            m = (j >= 0) & (j < line.size)
            if not np.any(m):
                continue
            i_sel = zeros_idx[m]
            j_sel = j[m]
            a = line[i_sel]
            b = line[j_sel]
            # edge when different nonzero owners meet
            keep = (a != 0) & (b != 0) & (a != b)
            if not np.any(keep):
                continue
            aa = a[keep]
            bb = b[keep]
            lo = np.minimum(aa, bb)
            hi = np.maximum(aa, bb)
            pairs_list.append(np.stack((lo, hi), axis=1))

        if pairs_list:
            pairs_bg = np.vstack(pairs_list).astype(np.int64, copy=False)
            # Union with original adjacency (labels that already touched)
            pairs_orig = connect(lab0, conn).astype(np.int64, copy=False)
            if pairs_orig.size:
                pairs_all = np.vstack((pairs_bg, pairs_orig))
            else:
                pairs_all = pairs_bg
        else:
            pairs_all = connect(lab0, conn).astype(np.int64, copy=False)

        # Deduplicate pairs (lo, hi)
        if pairs_all.size:
            order = np.lexsort((pairs_all[:, 1], pairs_all[:, 0]))
            p = pairs_all[order]
            keep = np.empty(p.shape[0], dtype=bool)
            keep[0] = True
            keep[1:] = (p[1:, 0] != p[:-1, 0]) | (p[1:, 1] != p[:-1, 1])
            pairs = p[keep]
        else:
            pairs = pairs_all

    # Color the graph via existing solver stack
    if pairs.size == 0:
        lut = np.ones(lab0.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut

    if greedy:
        colors = greedy_coloring(mapidx(pairs))
    else:
        colors = render_net(pairs, n=n, rand=10, max_depth=max_depth, offset=offset, verbose=verbose)

    lut = np.ones(lab0.max() + 1, dtype=np.uint8)
    for i in colors:
        lut[i] = colors[i]
    lut[0] = 0
    return lut


def get_lut_expanded_components(lab, n=4, conn=2, max_depth=5, offset=0, greedy=False):
    """
    Correct expanded adjacency via background ownership inside components.

    This computes, per background pixel, the nearest label owner (via a
    multi-source wavefront over zeros), then emits edges only across
    neighboring background pixels whose owners differ. This matches the
    adjacency you'd get by expand_labels + connect, without materializing the
    expanded array.
    """
    # Standard form: try to avoid format_labels when already sequential and bg==0
    if np.issubdtype(lab.dtype, np.integer) and np.min(lab) == 0:
        uniq = fastremap.unique(lab)
        if uniq.size and np.array_equal(uniq, np.arange(int(uniq[-1]) + 1)):
            pad_shape = tuple(int(s) + 2 for s in lab.shape)
            lab0 = np.zeros(pad_shape, dtype=np.int32)
            center = tuple(slice(1, s + 1) for s in lab.shape)
            lab0[center] = lab.astype(np.int32, copy=False)
        else:
            lab0 = format_labels(np.pad(lab, 1), background=0).astype(np.int32)
    else:
        lab0 = format_labels(np.pad(lab, 1), background=0).astype(np.int32)

    if lab0.max() <= 1:
        lut = np.ones(lab0.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut

    # Assign owners to background via ND wavefront (simple core)
    buf = lab0.copy()
    line = buf.ravel()
    nbs_all = neighbors(buf.shape, conn=conn, unique=False)
    _expand_wavefront_core(line, buf.shape, nbs_all)

    # Build edges from background-only neighbor comparisons (half-space) using safe slicing
    # Build edges from background-only neighbor comparisons (half-space) using safe slicing
    pairs_list = []
    block = scipy.ndimage.generate_binary_structure(lab0.ndim, conn)
    block[tuple([1] * lab0.ndim)] = 0
    offs = np.array(np.where(block > 0)).T - 1
    keep = []
    for off in offs:
        sel = 0
        for v in off:
            if v != 0:
                sel = 1 if v > 0 else 0
                break
        keep.append(sel)
    offs = offs[np.array(keep, dtype=bool)]

    for off in offs:
        src = []
        dst = []
        for ax, v in enumerate(off):
            if v == 0:
                src.append(slice(None))
                dst.append(slice(None))
            elif v > 0:
                src.append(slice(None, -v))
                dst.append(slice(v, None))
            else:
                vv = -v
                src.append(slice(vv, None))
                dst.append(slice(None, -vv))
        owner_src = buf[tuple(src)]
        owner_dst = buf[tuple(dst)]
        bg_src = (lab0[tuple(src)] == 0)
        bg_dst = (lab0[tuple(dst)] == 0)
        m = bg_src & bg_dst
        if not np.any(m):
            continue
        a = owner_src[m]
        b = owner_dst[m]
        keepm = (a != 0) & (b != 0) & (a != b)
        if not np.any(keepm):
            continue
        aa = a[keepm]
        bb = b[keepm]
        lo = np.minimum(aa, bb)
        hi = np.maximum(aa, bb)
        pairs_list.append(np.stack((lo, hi), axis=1))

    if pairs_list:
        pairs_bg = np.vstack(pairs_list).astype(np.int64, copy=False)
    else:
        pairs_bg = np.empty((0, 2), dtype=np.int64)

    # Union with original adjacency (labels that already touched)
    pairs_orig = connect(lab0, conn).astype(np.int64, copy=False)
    if pairs_orig.size and pairs_bg.size:
        pairs_all = np.vstack((pairs_bg, pairs_orig))
    elif pairs_orig.size:
        pairs_all = pairs_orig
    else:
        pairs_all = pairs_bg

    # Deduplicate edges (lo, hi)
    if pairs_all.size:
        order = np.lexsort((pairs_all[:, 1], pairs_all[:, 0]))
        p = pairs_all[order]
        keep = np.empty(p.shape[0], dtype=bool)
        keep[0] = True
        keep[1:] = (p[1:, 0] != p[:-1, 0]) | (p[1:, 1] != p[:-1, 1])
        pairs = p[keep]
    else:
        pairs = pairs_all

    # Color
    if pairs.size == 0:
        lut = np.ones(lab0.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut

    if greedy:
        colors = greedy_coloring(mapidx(pairs))
    else:
        colors = render_net(pairs, n=n, rand=10, max_depth=max_depth, offset=offset)

    lut = np.ones(lab0.max() + 1, dtype=np.uint8)
    for i in colors:
        lut[i] = colors[i]
    lut[0] = 0
    return lut
