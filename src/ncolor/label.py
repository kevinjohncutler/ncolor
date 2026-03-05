#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements

import numpy as np
import fastremap
from functools import lru_cache

# Import heavy dependencies but defer JIT compilation
from numba import njit, prange
import numba
import scipy
import scipy.ndimage
import edt as _edt
_edt_has_expand = hasattr(_edt, 'expand_labels')
from .format_labels import format_labels

def _normalize_labels(labels):
    """Lightweight label normalization for ncolor's internal pipeline.

    Fast path: when labels already have background=0, skip fastremap.renumber
    and refit entirely — saving ~0.5ms for typical segmentations. The coloring
    algorithm only requires background=0 and a reasonable label range; phantom
    CSR nodes (degree-0 gaps) are harmless and get color 1 from lut init.

    Fallback: when min != 0 (e.g. dense-pixel inputs with no true background),
    delegate to format_labels which handles background assignment correctly and
    ensures the contiguous 1..N invariant that _build_csr_from_pairs relies on.
    """
    labels = labels.astype(np.int32)
    if int(labels.min()) == 0:
        # Fast path: skip renumber. Phantom CSR nodes (degree-0 gaps) are
        # harmless for correctness, but they inflate max_iter proportionally
        # to max_label. Guard: if max_label > n_pixels there are provably more
        # gaps than real labels, so renumber to keep the coloring loop bounded.
        if int(labels.max()) <= labels.size:
            return labels
    # Fall back to format_labels: handles min-shift + renumber correctly
    return format_labels(labels).astype(np.int32)


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

    # Normalize labels for the coloring pipeline.
    if format_input:
        if expand and int(lab.min()) == 0:
            # After expansion all pixels are labeled (values 1..N, no zeros).
            # No shift needed: connect() skips 0-pixels but there are none,
            # lut[0] is set to 0 but never accessed. Skipping the shift also
            # avoids incorrectly treating the minimum label as background.
            lab_fmt = lab_exp.astype(np.int32, copy=False)
        else:
            lab_fmt = _normalize_labels(lab_exp)
    else:
        lab_fmt = lab_exp.astype(np.int32, copy=False)

    colored_expanded = _solver(
        lab_fmt,
        n=n,
        conn=conn,
        max_depth=max_depth,
        offset=offset,
        format_input=False,  # already formatted above
    )
    if colored_expanded is None:
        raise ValueError("Default solver failed to produce a valid coloring.")
    colored = colored_expanded * (lab != 0) if expand else colored_expanded

    conflicts = 0
    lut = None
    if check_conflicts or return_conflicts or return_lut:
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


@njit(cache=True, parallel=True)
def _apply_lut(flat_lab, lut):
    """Parallel LUT gather: faster than NumPy fancy indexing for large arrays."""
    out = np.empty(flat_lab.size, dtype=lut.dtype)
    for i in prange(flat_lab.size):
        out[i] = lut[flat_lab[i]]
    return out


def _solver(lab, n=4, conn=2, max_depth=5, offset=0, format_input=True):
    lab = format_labels(lab).astype(np.int32) if format_input else lab.astype(np.int32, copy=False)
    idx = connect(lab, conn)
    max_label = int(lab.max())
    if idx.size == 0:
        lut = np.ones(max_label + 1, dtype=np.uint8)
        lut[0] = 0
        return _apply_lut(lab.ravel(), lut).reshape(lab.shape)
    nodes, indptr, indices = _build_csr_from_pairs(idx)
    if nodes.size == 0:
        lut = np.ones(max_label + 1, dtype=np.uint8)
        lut[0] = 0
        return _apply_lut(lab.ravel(), lut).reshape(lab.shape)

    attempts_per_n = 4
    cur_n = int(n)
    # One O(M+N) pass is enough for convergence; extra iterations only help
    # thrashing graphs (wrong n) burn more time — use early bailout instead.
    max_iter = max(indices.size + int(indptr.size), 512)
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
                return _apply_lut(lab.ravel(), lut).reshape(lab.shape)
            # BFS hit max_iter AND all repair failed: n colors insufficient for
            # this graph — more attempts with the same n won't help, bump n now.
            if unfinished and needs_repair:
                break
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


@lru_cache(maxsize=8)
def _neighbor_dirs(ndim, conn, unique):
    """Cached direction vectors for (ndim, conn) — shape-independent."""
    block = scipy.ndimage.generate_binary_structure(ndim, conn)
    block[tuple([1] * ndim)] = 0
    idx = np.array(np.where(block > 0)).T  # (K, ndim) in {-1,0,1}
    idx = idx - 1
    if unique:
        keep = []
        for off in idx:
            sel = 0
            for v in off:
                if v != 0:
                    sel = 1 if v > 0 else 0
                    break
            keep.append(sel)
        idx = idx[np.array(keep, dtype=bool)]
    return idx


def neighbors(shape, conn=1, unique=True):
    idx = _neighbor_dirs(len(shape), conn, unique)
    acc = np.cumprod((1,) + shape[::-1][:-1])
    return np.dot(idx, acc[::-1])



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
                            

    
@njit(cache=True)
def _search_hashset(line, total, nbs, ht_size):
    """
    Single-pass adjacent-pair search with inline deduplication via open-addressing
    hash table. Avoids the separate dedup step entirely for cases where the table
    fits in cache (n_labels small relative to image size).
    """
    EMPTY = np.uint64(0xFFFFFFFFFFFFFFFF)
    ht_mask = np.uint64(ht_size - 1)
    ht = np.empty(ht_size, dtype=np.uint64)
    for i in range(ht_size):
        ht[i] = EMPTY

    for i in range(total):
        vi = line[i]
        if vi == 0:
            continue
        for d in nbs:
            j = i + d
            if j < 0 or j >= total:
                continue
            vj = line[j]
            if vj == 0 or vj == vi:
                continue
            lo = np.uint64(vi) if vi < vj else np.uint64(vj)
            hi = np.uint64(vj) if vi < vj else np.uint64(vi)
            key = (lo << np.uint64(32)) | hi
            # Fibonacci hashing for uniform distribution
            h = (key * np.uint64(11400714819323198485)) & ht_mask
            while ht[int(h)] != EMPTY and ht[int(h)] != key:
                h = (h + np.uint64(1)) & ht_mask
            ht[int(h)] = key

    count = np.int64(0)
    for h in range(ht_size):
        if ht[h] != EMPTY:
            count += np.int64(1)

    out = np.empty((int(count), 2), dtype=np.int32)
    idx = np.int64(0)
    for h in range(ht_size):
        if ht[h] != EMPTY:
            key = ht[h]
            out[int(idx), 0] = np.int32(key >> np.uint64(32))
            out[int(idx), 1] = np.int32(key & np.uint64(0xFFFFFFFF))
            idx += np.int64(1)
    return out


@njit(parallel=True, cache=True)
def _search_hashset_parallel(line, total, nbs, ht_size, n_threads):
    """
    Parallel version of _search_hashset.  Each thread scans a contiguous strip
    of the (padded) line and writes into its own private hash table.  Because all
    neighbor offsets are positive (unique=True on a padded image), each undirected
    pair (lo,hi) is emitted by exactly the thread that owns the pixel with the
    smaller flat index — so there is no cross-thread duplication within a single
    region boundary.  Different regions that share the same label pair are
    deduplicated inside each thread's table.  A short serial merge step
    deduplicates any residual cross-thread collisions.
    """
    EMPTY = np.uint64(0xFFFFFFFFFFFFFFFF)
    ht_mask = np.uint64(ht_size - 1)

    # Per-thread hash tables (n_threads × ht_size); each thread inits its own
    # row in parallel so the memset is also distributed across cores.
    hts = np.empty((n_threads, ht_size), dtype=np.uint64)
    strip = (total + n_threads - 1) // n_threads

    for tid in prange(n_threads):
        for h in range(ht_size):
            hts[tid, h] = EMPTY
        start = tid * strip
        end = min(start + strip, total)
        for i in range(start, end):
            vi = line[i]
            if vi == 0:
                continue
            for d in nbs:
                j = i + d
                if j < 0 or j >= total:
                    continue
                vj = line[j]
                if vj == 0 or vj == vi:
                    continue
                lo = np.uint64(vi) if vi < vj else np.uint64(vj)
                hi = np.uint64(vj) if vi < vj else np.uint64(vi)
                key = (lo << np.uint64(32)) | hi
                h = (key * np.uint64(11400714819323198485)) & ht_mask
                while hts[tid, int(h)] != EMPTY and hts[tid, int(h)] != key:
                    h = (h + np.uint64(1)) & ht_mask
                hts[tid, int(h)] = key

    # Serial merge: insert all per-thread entries into a single table
    merge_ht = np.empty(ht_size, dtype=np.uint64)
    for h in range(ht_size):
        merge_ht[h] = EMPTY

    for t in range(n_threads):
        for h in range(ht_size):
            key = hts[t, h]
            if key == EMPTY:
                continue
            gh = (key * np.uint64(11400714819323198485)) & ht_mask
            while merge_ht[int(gh)] != EMPTY and merge_ht[int(gh)] != key:
                gh = (gh + np.uint64(1)) & ht_mask
            merge_ht[int(gh)] = key

    count = np.int64(0)
    for h in range(ht_size):
        if merge_ht[h] != EMPTY:
            count += np.int64(1)

    out = np.empty((int(count), 2), dtype=np.int32)
    idx = np.int64(0)
    for h in range(ht_size):
        if merge_ht[h] != EMPTY:
            key = merge_ht[h]
            out[int(idx), 0] = np.int32(key >> np.uint64(32))
            out[int(idx), 1] = np.int32(key & np.uint64(0xFFFFFFFF))
            idx += np.int64(1)
    return out


# Pixel count above which the parallel kernel is used (avoids thread-launch
# overhead for small images).
_PARALLEL_THRESHOLD = 100_000


def connect(img, conn=1):
    nbs = neighbors(img.shape, conn, True)
    n_labels = int(img.max())
    if n_labels > 0:
        ht_raw = len(nbs) * n_labels * 2
        ht_size = 1
        while ht_size < ht_raw:
            ht_size <<= 1
        if ht_size <= (1 << 20):
            buf = np.pad(img, 1, 'constant')
            nbs_pad = neighbors(buf.shape, conn, True)
            line = buf.ravel()
            n_threads = numba.get_num_threads()
            if len(line) > _PARALLEL_THRESHOLD and n_threads > 1:
                return _search_hashset_parallel(line, len(line), nbs_pad, ht_size, n_threads)
            return _search_hashset(line, len(line), nbs_pad, ht_size)

    # Fallback for very large label counts: pad then search+dedup.
    buf = np.pad(img, 1, 'constant')
    nbs_pad = neighbors(buf.shape, conn, unique=True)
    rst = search(buf, nbs_pad)
    if rst.size == 0:
        return rst
    keys = rst[:, 0].astype(np.uint64) << np.uint64(32) | rst[:, 1].astype(np.uint64)
    ukeys = np.unique(keys)
    out = np.empty((ukeys.size, 2), dtype=rst.dtype)
    out[:, 0] = (ukeys >> np.uint64(32)).astype(rst.dtype)
    out[:, 1] = (ukeys & np.uint64(0xFFFFFFFF)).astype(rst.dtype)
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
                # visited is clean on entry (reset after previous BFS)
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

                # Reset only the nodes we actually visited (O(visited) not O(N))
                for k in range(tail):
                    visited[q[k]] = 0

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


@njit(cache=True)
def _build_csr_jit(src, dst, N, M):
    """
    O(M + N) symmetric CSR construction from M directed pairs (src[i], dst[i]).
    Avoids the O(M log M) sort required by the vstack+argsort approach.
    Neighbour order within each row is arbitrary (not needed by the coloring kernels).
    """
    # 1. Count degree per node (each undirected edge contributes 1 to each endpoint)
    degree = np.zeros(N, np.int32)
    for i in range(M):
        degree[src[i]] += np.int32(1)
        degree[dst[i]] += np.int32(1)
    # 2. Prefix sum → indptr
    indptr = np.empty(N + 1, np.int32)
    indptr[0] = np.int32(0)
    for i in range(N):
        indptr[i + 1] = indptr[i] + degree[i]
    # 3. Fill indices with write pointers (one pointer per node, advanced as filled)
    write = indptr[:N].copy()
    indices = np.empty(2 * M, np.int32)
    for i in range(M):
        s = src[i]
        d = dst[i]
        indices[write[s]] = d
        write[s] += np.int32(1)
        indices[write[d]] = s
        write[d] += np.int32(1)
    return indptr, indices


def _build_csr_from_pairs(pairs_arr):
    # format_labels guarantees labels are contiguous 1..N, so the node index is
    # simply (label - 1). This replaces unique() + two O(M log N) searchsorted
    # calls with O(M) direct arithmetic — a ~25× speedup for large label counts.
    N = int(pairs_arr.max())
    M = len(pairs_arr)
    src_idx = (pairs_arr[:, 0] - 1).astype(np.int32)
    dst_idx = (pairs_arr[:, 1] - 1).astype(np.int32)
    all_nodes = np.arange(1, N + 1, dtype=pairs_arr.dtype)
    indptr, indices = _build_csr_jit(src_idx, dst_idx, N, M)
    return all_nodes, indptr, indices


def expand_labels(label_image):
    """
    Expand labels into every background pixel using nearest-label assignment.
    Uses edt.expand_labels when available (fast Cython); falls back to
    scipy.ndimage.distance_transform_edt on older edt versions.
    """
    if _edt_has_expand:
        return _edt.expand_labels(label_image, parallel=0).astype(label_image.dtype)
    coords = scipy.ndimage.distance_transform_edt(
        label_image == 0, return_distances=False, return_indices=True
    )
    return label_image[tuple(coords)]
    
    
@njit(cache=True)
def _expand_wavefront_core(line, nbs):
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
    _expand_wavefront_core(line, nbs)
    unpad = tuple([slice(1, -1)] * buf.ndim)
    return buf[unpad]
