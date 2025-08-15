#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements 

import numpy as np
from numba import njit
import scipy
from .format_labels import format_labels
# from . import format_labels

import fastremap

# import edt # does not yet support return_indices
from scipy.ndimage import distance_transform_edt
from collections import deque


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
        # this allows expand to override either with True or False
        if expand or (lab.squeeze().ndim==2 and expand is None):
            lab = expand_labels(lab)
        # lab = np.pad(format_labels(lab),pad)
        lab = format_labels(np.pad(lab,pad),background=0)
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

    lo = rst[:, 0]
    hi = rst[:, 1]

    # Pack pairs into 64-bit keys and unique via fastremap (O(n))
    keys = (lo.astype(np.uint64) << np.uint64(32)) | hi.astype(np.uint64)
    uniq = np.unique(keys)

    m = uniq.size
    out = np.empty((m, 2), dtype=rst.dtype)
    out[:, 0] = (uniq >> np.uint64(32)).astype(rst.dtype)
    out[:, 1] = (uniq & np.uint64(0xFFFFFFFF)).astype(rst.dtype)
    return out
    
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


# maybe replace this with fastremap
# import fastremap
# def mapidx(idx):
#     dic = {}
#     # for i in np.unique(idx): dic[i] = []
#     for i in fastremap.unique(idx): dic[i] = [] # marginally faster 
#     for i,j in idx:
#         dic[i].append(j)
#         dic[j].append(i)
#     return dic
    

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
    unique_i, counts = fastremap.unique(i, return_counts=True)
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
        if unfinished and depth < max_depth:
            return render_net(conmap, n + 1, rand, depth + 1, max_depth, offset)
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

    unique_src, counts = fastremap.unique(src, return_counts=True)
    N = unique_src.size
    indptr = np.empty(N + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])

    # Map node ids → [0..N-1] and remap dst
    id2idx = {int(v): i for i, v in enumerate(unique_src.tolist())}
    indices = fastremap.remap(dst, id2idx, preserve_missing_labels=False).astype(np.int32, copy=False)

    max_iter = max(2 * N, 1)
    colors_arr, unfinished = _color_graph_csr(indptr, indices, int(n), int(rand), int(max_iter))
    if unfinished and depth < max_depth:
        return render_net(pairs, n + 1, rand, depth + 1, max_depth, offset)

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
    
    