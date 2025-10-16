#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements

import numpy as np
import fastremap
from collections import deque

# Import heavy dependencies but defer JIT compilation
from numba import njit
import scipy
from scipy.ndimage import distance_transform_edt
from .format_labels import format_labels

# Precomputed 2D neighbor deltas (int8) for fast wavefront expansion
DY4 = np.array([-1, 0, 0, 1], dtype=np.int8)
DX4 = np.array([0, -1, 1, 0], dtype=np.int8)
DY8 = np.array([-1, -1, -1, 0, 0, 1, 1, 1], dtype=np.int8)
DX8 = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int8)

# Simple workspace cache for 2D wavefront queues, keyed by (H, W)
_WAVE_WS_2D = {}

def _get_wavefront_ws_2d(shape):
    H, W = int(shape[0]), int(shape[1])
    key = (H, W)
    n = H * W
    ws = _WAVE_WS_2D.get(key)
    if ws is not None and ws[0].size >= n and ws[1].size >= n:
        return ws
    qy = np.empty(n, dtype=np.int32)
    qx = np.empty(n, dtype=np.int32)
    _WAVE_WS_2D[key] = (qy, qx)
    return _WAVE_WS_2D[key]


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


def label(lab, n=4, conn=2, max_depth=5, offset=0, expand=None, return_n=False, greedy=False, verbose=False):
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space
    # if not is_sequential(lab):
    #     lab = format_labels(lab)
    pad = 1
    unpad = tuple([slice(pad,-pad)]*lab.ndim)
    mask = lab!=0
    unique = unique_nonzero(lab)
    if verbose: print('number of masks', unique.size)
    
    if unique.size==1:
        ncl = mask.astype(int)
        nc = 1
    else:
        # by default, 2D images should be expanded, 3D should not
        # this allows expand to override either with True/False or a fast mode string
        if expand or (lab.squeeze().ndim==2 and expand is None):
            if isinstance(expand, str) and expand.lower() in ("fast", "wave", "approx", "wavefront"):
                lab = expand_labels_wavefront(lab, conn=2)
            else:
                lab = expand_labels(lab)
        # lab = np.pad(format_labels(lab),pad)
        lab = format_labels(np.pad(lab,pad),background=0) # is this necessary? 
        lut = get_lut(lab,n,conn,max_depth,offset,greedy)
        
        ncl = lut[lab][unpad]*mask
        nc = np.max(lut)
    
    if return_n: 
        return ncl, nc
    else:    
        return ncl

def get_lut(lab, n=4, conn=2, max_depth=5, offset=0, greedy=False):
    lab = format_labels(lab).astype(np.int32)

    pairs = connect(lab, conn)

    # Handle case where no masks touch one another.
    if pairs.size == 0:
        lut = np.ones(lab.max() + 1, dtype=np.uint8)
        lut[0] = 0
        return lut

    if greedy:
        # Greedy expects a dict-of-arrays; build only when needed
        conmap = mapidx(pairs)
        colors = greedy_coloring(conmap)
    else:
        # Fast path: pass edge list directly to render_net
        colors = render_net(pairs, n=n, rand=10, max_depth=max_depth, offset=offset)
        if colors is None:
            raise ValueError(f"Failed to color the labels with {n} colors. Try increasing n or max_depth.")

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

# Caches for ND helpers
_NBS_CACHE = {}
_ND_BUF_CACHE = {}
_ND_Q_CACHE = {}

def _get_neighbors_cached(shape, conn, unique):
    key = (tuple(shape), int(conn), bool(unique))
    nbs = _NBS_CACHE.get(key)
    if nbs is None:
        nbs = neighbors(shape, conn, unique)
        nbs = np.asarray(nbs, dtype=np.int64)
        _NBS_CACHE[key] = nbs
    return nbs

def _get_nd_workspace(shape, dtype):
    pad_shape = tuple(int(s) + 2 for s in shape)
    key_buf = (pad_shape, np.dtype(dtype).str)
    buf = _ND_BUF_CACHE.get(key_buf)
    if (buf is None) or (buf.shape != pad_shape) or (buf.dtype != dtype):
        buf = np.zeros(pad_shape, dtype=dtype)
        _ND_BUF_CACHE[key_buf] = buf
    total = int(np.prod(pad_shape))
    q = _ND_Q_CACHE.get(pad_shape)
    if (q is None) or (q.size < total):
        q = np.empty(total, dtype=np.int64)
        _ND_Q_CACHE[pad_shape] = q
    return buf, q

@njit(cache=True)
def _expand_wavefront_nd(line, nbs, q):
    total = line.size
    head = 0
    tail = 0
    # Seed only boundary-labeled indices (adjacent to zero)
    for i in range(total):
        vi = line[i]
        if vi == 0:
            continue
        for d in nbs:
            j = i + d
            if j < 0 or j >= total:
                continue
            if line[j] == 0:
                q[tail] = i
                tail += 1
                break
    if head == tail:
        return tail

    # Frontier BFS: process layer by layer
    front_start = 0
    front_end = tail
    while front_start < front_end:
        for pos in range(front_start, front_end):
            i = q[pos]
            vi = line[i]
            for d in nbs:
                j = i + d
                if j < 0 or j >= total:
                    continue
                if line[j] == 0:
                    line[j] = vi
                    q[tail] = j
                    tail += 1
        front_start = front_end
        front_end = tail
    return tail

@njit(cache=True)
def _edges_from_queue(line, q, qlen, bgmask, nbs_half):
    total = line.size
    cap = max(qlen * max(len(nbs_half), 1), 1)
    edges = np.empty((cap, 2), np.int64)
    e = 0
    for t in range(qlen):
        i = q[t]
        if not bgmask[i]:
            continue
        vi = line[i]
        if vi == 0:
            continue
        for d in nbs_half:
            j = i + d
            if j < 0 or j >= total:
                continue
            if not bgmask[j]:
                continue
            vj = line[j]
            if vj == 0 or vj == vi:
                continue
            a = vi
            b = vj
            if a > b:
                a, b = b, a
            if e < cap:
                edges[e, 0] = a
                edges[e, 1] = b
                e += 1
    return edges[:e]

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
    counter = np.zeros(N, np.int32)

    # queue as a simple ring buffer
    qcap = max(N * 4, 1)
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
            # choose the smallest missing color
            csel = 1
            while (mask & (1 << csel)) != 0 and csel <= n:
                csel += 1
            if csel > n:
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
            colors[u] = csel
            # re-enqueue only neighbors that now conflict
            for k in range(row_beg, row_end):
                v = indices[k]
                if colors[v] == csel:
                    if tail < qcap:
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


def render_net(conmap, n=4, rand=12, depth=0, max_depth=5, offset=0):
    # Accept either dict-of-arrays (adjacency) or ndarray of edges (M,2)
    if isinstance(conmap, dict):
        N = len(conmap)
        if N == 0:
            return {}

        nodes = np.fromiter(conmap.keys(), dtype=np.int64, count=N)
        degrees = np.fromiter((len(conmap[int(nid)]) for nid in nodes), dtype=np.int32, count=N)
        indptr = np.empty(N + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(degrees, out=indptr[1:])
        if indptr[-1] == 0:
            return {int(nid): 1 for nid in nodes}

        concat_neighbors = np.empty(indptr[-1], dtype=np.int64)
        pos = 0
        for nid in nodes:
            arr = conmap[int(nid)]
            ln = len(arr)
            concat_neighbors[pos:pos + ln] = arr
            pos += ln

        id2idx = {int(v): i for i, v in enumerate(nodes.tolist())}
        indices = fastremap.remap(concat_neighbors, id2idx, preserve_missing_labels=False).astype(np.int32, copy=False)

        max_iter = max(2 * N, 1)
        colors_arr, unfinished = _color_graph_csr(indptr, indices, int(n), int(rand), int(max_iter))
        # Validate coloring: no adjacent nodes share a color
        conflict = False
        for u in range(N):
            c = colors_arr[u]
            if c == 0:
                continue
            row_beg = indptr[u]
            row_end = indptr[u + 1]
            for k in range(row_beg, row_end):
                v = indices[k]
                if v == u:
                    continue
                if colors_arr[v] == c:
                    conflict = True
                    break
            if conflict:
                break

        # Try a quick local repair in CSR
        if conflict:
            colors_arr, conflict = _repair_coloring(indptr, indices, colors_arr, int(n), max_passes=4)
        # Try Kempe-chain style recoloring to keep colors ≤ n
        if conflict:
            colors_arr, conflict = _kempe_repair_csr(indptr, indices, colors_arr, int(n), max_passes=2)

        if unfinished and depth < max_depth:
            return render_net(conmap, n + 1, rand, depth + 1, max_depth, offset)

        if conflict:
            # As a last resort, use CSR greedy on the same graph
            colors_arr = _greedy_coloring_csr(indptr, indices, int(n))
            # No need to validate; greedy guarantees validity
            return {int(nodes[i]): int(colors_arr[i]) for i in range(N)}

        return {int(nodes[i]): int(colors_arr[i]) for i in range(N)}

    # ndarray edge-list path
    pairs = np.asarray(conmap, dtype=np.int64)
    if pairs.size == 0:
        return {}

    # Make adjacency symmetric without Python loops
    pairs_sym = np.vstack((pairs, pairs[:, [1, 0]]))

    # Group by source using a stable sort, then build CSR
    order = np.argsort(pairs_sym[:, 0], kind='mergesort')
    src = pairs_sym[order, 0]
    dst = pairs_sym[order, 1]

    unique_src, counts = fastremap.unique(src, return_counts=True) # about the same as np.unique

    N = unique_src.size
    indptr = np.empty(N + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    # Map node ids → [0..N-1] and remap dst
    id2idx = {int(v): i for i, v in enumerate(unique_src.tolist())}
    indices = fastremap.remap(dst, id2idx, preserve_missing_labels=False).astype(np.int32, copy=False)

    max_iter = max(2 * N, 1)
    colors_arr, unfinished = _color_graph_csr(indptr, indices, int(n), int(rand), int(max_iter))

    # Validate coloring: no adjacent nodes share a color
    conflict = False
    for u in range(N):
        c = colors_arr[u]
        if c == 0:
            continue
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        for k in range(row_beg, row_end):
            v = indices[k]
            if v == u:
                continue
            if colors_arr[v] == c:
                conflict = True
                break
        if conflict:
            break

    # Try a quick local repair in CSR
    if conflict:
        colors_arr, conflict = _repair_coloring(indptr, indices, colors_arr, int(n), max_passes=4)
    # Try Kempe-chain style recoloring to keep colors ≤ n
    if conflict:
        colors_arr, conflict = _kempe_repair_csr(indptr, indices, colors_arr, int(n), max_passes=2)

    if unfinished and depth < max_depth:
        return render_net(pairs, n + 1, rand, depth + 1, max_depth, offset)

    if conflict:
        # Fallback to CSR greedy coloring (should be rare)
        colors_arr = _greedy_coloring_csr(indptr, indices, int(n))
        return {int(unique_src[i]): int(colors_arr[i]) for i in range(N)}

    return {int(unique_src[i]): int(colors_arr[i]) for i in range(N)}
    
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


#################### get_lut_expanded here 
def get_lut_expanded(lab, n=4, conn=2, max_depth=5, offset=0, greedy=False):
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
        colors = render_net(pairs, n=n, rand=10, max_depth=max_depth, offset=offset)

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
