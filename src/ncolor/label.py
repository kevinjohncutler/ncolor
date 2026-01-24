#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements

import numpy as np
import fastremap

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
        

def label(lab, n=4, conn=2, max_depth=30, offset=0, expand=True, return_n=False, return_lut=False, verbose=False, check_conflicts=False, return_conflicts=False, format_input=True):
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space
    # if not is_sequential(lab):
    #     lab = format_labels(lab)

    if verbose: print('verbose')
    if expand:
        if isinstance(expand, str) and expand.lower() in ("fast", "wave", "approx", "wavefront"):
            lab_exp = expand_labels_wavefront(lab, conn=2)
        else:
            lab_exp = expand_labels(lab)
    else:
        lab_exp = lab

    colored_expanded = _solver(
        lab_exp,
        n=n,
        conn=conn,
        max_depth=max_depth,
        offset=offset,
        format_input=format_input,
    )
    if colored_expanded is None:
        raise ValueError("Default solver failed to produce a valid coloring.")
    colored = colored_expanded * (lab != 0) if expand else colored_expanded

    conflicts = 0
    lut = None
    if check_conflicts or return_conflicts or return_lut:
        lab_fmt = format_labels(lab_exp).astype(np.int32) if format_input else lab_exp.astype(np.int32, copy=False)
        lut = np.zeros(lab_fmt.max() + 1, dtype=colored_expanded.dtype)
        lut[lab_fmt] = colored_expanded
        if check_conflicts or return_conflicts:
            pairs = connect(lab_fmt, conn)
            if pairs.size:
                conflicts = int(np.count_nonzero(lut[pairs[:, 0]] == lut[pairs[:, 1]]))
            if check_conflicts and conflicts:
                raise ValueError(f"Coloring conflict detected: {conflicts} adjacent pairs share a color.")
    if return_lut:
        if return_n and return_conflicts:
            return lut, int(np.max(lut)), conflicts
        if return_n:
            return lut, int(np.max(lut))
        if return_conflicts:
            return lut, conflicts
        return lut
    if return_n and return_conflicts:
        return colored, int(colored.max()), conflicts
    if return_n:
        return colored, int(colored.max())
    if return_conflicts:
        return colored, conflicts
    return colored


def _solver(lab, n=4, conn=2, max_depth=5, offset=0, format_input=True):
    lab = format_labels(lab).astype(np.int32) if format_input else lab.astype(np.int32, copy=False)
    idx = connect(lab, conn)
    max_label = int(lab.max())
    if idx.size == 0:
        lut = np.ones(max_label + 1, dtype=np.uint8)
        lut[0] = 0
        return lut[lab]
    nodes, indptr, indices = _build_csr_from_pairs(idx)
    if nodes.size == 0:
        lut = np.ones(max_label + 1, dtype=np.uint8)
        lut[0] = 0
        return lut[lab]

    attempts_per_n = 4
    cur_n = int(n)
    max_iter = max((indices.size + int(indptr.size)) * max_depth, 512)
    for _ in range(max_depth):
        for attempt in range(attempts_per_n):
            attempt_offset = offset + attempt
            colors, unfinished = _color_graph_csr_legacy(indptr, indices, cur_n, rand=10, offset=attempt_offset, max_iter=max_iter)
            needs_repair = unfinished or (colors == 0).any()
            if needs_repair:
                colors, conflict = _repair_coloring(indptr, indices, colors, cur_n, max_passes=max(4, max_depth))
                needs_repair = conflict or (colors == 0).any()
            if needs_repair:
                colors, conflict = _kempe_repair_csr(indptr, indices, colors, cur_n, max_passes=max(2, max_depth))
                needs_repair = conflict or (colors == 0).any()
            if not needs_repair:
                if _has_conflict_csr(indptr, indices, colors):
                    continue
                lut = np.ones(max_label + 1, dtype=np.uint8)
                lut[nodes] = colors
                lut[0] = 0
                return lut[lab]
        cur_n += 1
    return None


def get_lut(lab, n=4, conn=2, max_depth=30, offset=0, expand=True, return_n=False, verbose=False, check_conflicts=False, return_conflicts=False, format_input=True):
    """
    Return the label->color LUT used by the default solver.
    Mirrors the label() signature for convenience; non-applicable
    arguments are ignored in the current implementation.
    """
    return label(
        lab,
        n=n,
        conn=conn,
        max_depth=max_depth,
        offset=offset,
        expand=expand,
        return_n=return_n,
        return_lut=True,
        verbose=verbose,
        check_conflicts=check_conflicts,
        return_conflicts=return_conflicts,
        format_input=format_input,
    )


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




@njit(cache=True)
def _color_graph_csr_legacy(indptr, indices, n, rand, offset, max_iter):
    N = indptr.size - 1
    colors = np.zeros(N, np.uint8)
    counter = np.zeros(N, np.int32)

    qcap = max(indices.size + N, N * 8, 1)
    q = np.empty(qcap, np.int32)
    head = 0
    tail = N
    for i in range(N):
        q[i] = i

    fullmask = (1 << (n + 1)) - 2
    count = 0
    while head < tail and count < max_iter:
        u = q[head]
        head += 1
        count += 1
        counter[u] += 1

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
            csel = 0
            for c in range(1, n + 1):
                if (mask & (1 << c)) == 0:
                    csel = c
                    break
            counter[u] = 0
        else:
            cnt = np.zeros(n + 1, np.int32)
            for k in range(row_beg, row_end):
                cv = colors[indices[k]]
                if cv != 0:
                    cnt[cv] += 1
            csel = 1
            minv = cnt[1]
            cu = colors[u]
            if cu != 0:
                minv = 2147483647
            for c in range(1, n + 1):
                if c == cu:
                    continue
                if cnt[c] < minv:
                    minv = cnt[c]
                    csel = c

            if rand > 0 and counter[u] == rand:
                counter[u] = 0
                seed = (1103515245 * (u + count + offset) + 12345) & 0x7FFFFFFF
                csel = 1 + (seed % n)

        if colors[u] != csel:
            colors[u] = csel
            for k in range(row_beg, row_end):
                v = indices[k]
                if colors[v] == csel:
                    if tail >= qcap:
                        newcap = max(qcap * 2, tail + 1)
                        q_new = np.empty(newcap, np.int32)
                        q_new[:tail] = q[:tail]
                        q = q_new
                        qcap = newcap
                    q[tail] = v
                    tail += 1

    return colors, (head < tail)


@njit(cache=True)
def _has_conflict_csr(indptr, indices, colors):
    N = indptr.size - 1
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
                return True
    return False


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
