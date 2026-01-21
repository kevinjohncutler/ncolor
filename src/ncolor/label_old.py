#4-color algorthm based on https://forum.image.sc/t/relabel-with-4-colors-like-map/33564 with extensions and improvements 

import heapq
import numpy as np
from numba import njit
from collections import deque
import scipy
from .format_labels import format_labels
from .label import connect as connect_csr
from .label import _build_csr_from_pairs
from .label import _repair_coloring
from .label import _kempe_repair_csr


def label(lab,n=4,conn=2,max_depth=5, offset=0):
    # needs to be in standard label form
    # needs to be in standard label form
    # but also needs to be in int32 data type to work properly; the formatting automatically
    # puts it into the smallest datatype to save space 
    lab = format_labels(lab).astype(np.int32)
    idx = connect_csr(lab, conn)
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


def _dsatur_legacy_color_csr(indptr, indices, n, bias, offset):
    N = indptr.size - 1
    if N == 0:
        return np.zeros(0, dtype=np.uint8)
    if n > 63:
        return None
    colors = np.zeros(N, np.uint8)
    sat_mask = np.zeros(N, np.uint64)
    sat_count = np.zeros(N, np.int32)
    degrees = indptr[1:] - indptr[:-1]
    usage = np.zeros(n + 1, np.int32)

    heap = []
    for i in range(N):
        heapq.heappush(heap, (0, -int(degrees[i]), int(bias[i]), i))

    while heap:
        neg_sat, neg_deg, _, u = heapq.heappop(heap)
        if colors[u] != 0:
            continue
        if -neg_sat != sat_count[u]:
            continue

        used = sat_mask[u]
        best = 0
        best_usage = 0
        best_tie = 0
        for c in range(1, n + 1):
            if (used & (np.uint64(1) << np.uint64(c))) != 0:
                continue
            uc = usage[c]
            tie = (c + offset) % n
            if best == 0 or uc < best_usage or (uc == best_usage and tie < best_tie):
                best = c
                best_usage = uc
                best_tie = tie
        if best == 0:
            return None

        colors[u] = np.uint8(best)
        usage[best] += 1
        bit = np.uint64(1) << np.uint64(best)
        rb = indptr[u]
        re = indptr[u + 1]
        for k in range(rb, re):
            v = indices[k]
            if colors[v] != 0:
                continue
            if (sat_mask[v] & bit) == 0:
                sat_mask[v] |= bit
                sat_count[v] += 1
                heapq.heappush(heap, (-sat_count[v], -int(degrees[v]), int(bias[v]), v))

    return colors


def _dsatur_relaxed_color_csr(indptr, indices, n, bias, offset):
    N = indptr.size - 1
    if N == 0:
        return np.zeros(0, dtype=np.uint8)
    if n > 63:
        return None
    colors = np.zeros(N, np.uint8)
    sat_mask = np.zeros(N, np.uint64)
    sat_count = np.zeros(N, np.int32)
    degrees = indptr[1:] - indptr[:-1]
    usage = np.zeros(n + 1, np.int32)

    heap = []
    for i in range(N):
        heapq.heappush(heap, (0, -int(degrees[i]), int(bias[i]), i))

    while heap:
        neg_sat, neg_deg, _, u = heapq.heappop(heap)
        if colors[u] != 0:
            continue
        if -neg_sat != sat_count[u]:
            continue

        used = sat_mask[u]
        best = 0
        best_usage = 0
        best_tie = 0
        for c in range(1, n + 1):
            if (used & (np.uint64(1) << np.uint64(c))) != 0:
                continue
            uc = usage[c]
            tie = (c + offset) % n
            if best == 0 or uc < best_usage or (uc == best_usage and tie < best_tie):
                best = c
                best_usage = uc
                best_tie = tie
        if best == 0:
            counts = np.zeros(n + 1, np.int32)
            rb = indptr[u]
            re = indptr[u + 1]
            for k in range(rb, re):
                v = indices[k]
                cv = colors[v]
                if cv != 0:
                    counts[cv] += 1
            best = 1
            best_conf = counts[1]
            best_usage = usage[1]
            best_tie = (1 + offset) % n
            for c in range(2, n + 1):
                conf = counts[c]
                uc = usage[c]
                tie = (c + offset) % n
                if conf < best_conf or (conf == best_conf and (uc < best_usage or (uc == best_usage and tie < best_tie))):
                    best = c
                    best_conf = conf
                    best_usage = uc
                    best_tie = tie

        colors[u] = np.uint8(best)
        usage[best] += 1
        bit = np.uint64(1) << np.uint64(best)
        rb = indptr[u]
        re = indptr[u + 1]
        for k in range(rb, re):
            v = indices[k]
            if colors[v] != 0:
                continue
            if (sat_mask[v] & bit) == 0:
                sat_mask[v] |= bit
                sat_count[v] += 1
                heapq.heappush(heap, (-sat_count[v], -int(degrees[v]), int(bias[v]), v))

    return colors




def neighbors(shape, conn=1):
    dim = len(shape)
    block = scipy.ndimage.generate_binary_structure(dim, conn)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])


@njit(cache=True, fastmath=True)
def search(img, nbs):
    s, line = 0, img.ravel()
    total = len(line)
    D = len(nbs)
    rst = np.zeros((total * D, 2), img.dtype)
    for i in range(total):
        vi = line[i]
        if vi == 0:
            continue
        for d in nbs:
            j = i + d
            if j < 0 or j >= total:
                continue
            vj = line[j]
            if vj == 0:
                continue
            if vi == vj:
                continue
            rst[s, 0] = vi
            rst[s, 1] = vj
            s += 1
    return rst[:s]
                            

def connect(img, conn=1):
    buf = np.pad(img, 1, 'constant')
    nbs = neighbors(buf.shape, conn)
    rst = search(buf, nbs)
    if len(rst)<2:
        return rst
    rst.sort(axis=1)
    key = (rst[:,0]<<16)
    key += rst[:,1]
    order = np.argsort(key)
    key[:] = key[order]
    diff = key[:-1]!=key[1:]
    idx = np.where(diff)[0]+1
    idx = np.hstack(([0], idx))
    return rst[order][idx]


def mapidx(idx):
    dic = {}
    for i in np.unique(idx): dic[i] = []
    for i,j in idx:
        dic[i].append(j)
        dic[j].append(i)
    return dic

@njit(cache=True)
def _legacy_style_color_csr(indptr, indices, n, rand_period, max_iter, order):
    N = indptr.size - 1
    colors = np.zeros(N, np.uint8)
    queue = np.empty(max(N * 4, 1), dtype=np.int32)
    head = 0
    tail = N
    for i in range(N):
        queue[i] = order[i]
    counter = np.zeros(N, np.int32)
    hist = np.zeros(n + 1, np.int32)
    seed = 123456789
    iter_count = 0

    while head < tail and iter_count < max_iter:
        u = queue[head]
        head += 1
        iter_count += 1
        counter[u] += 1
        hist[:] = 0
        rb = indptr[u]
        re = indptr[u + 1]
        for k in range(rb, re):
            v = indices[k]
            cv = colors[v]
            if cv != 0:
                hist[cv] += 1
        missing = 0
        for c in range(1, n + 1):
            if hist[c] == 0:
                missing = c
                break
        if missing != 0:
            new_color = missing
            counter[u] = 0
        else:
            best_color = colors[u] if colors[u] != 0 else 1
            best_score = 2147483647
            for c in range(1, n + 1):
                if hist[c] < best_score:
                    best_score = hist[c]
                    best_color = c
            new_color = best_color
            if rand_period > 0 and counter[u] >= rand_period:
                counter[u] = 0
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                new_color = 1 + (seed % n)
        if new_color != colors[u]:
            colors[u] = np.uint8(new_color)
            for k in range(rb, re):
                v = indices[k]
                if colors[v] == new_color:
                    if tail >= queue.size:
                        newsize = queue.size * 2
                        qnew = np.empty(newsize, dtype=np.int32)
                        qnew[:tail] = queue[:tail]
                        queue = qnew
                    queue[tail] = v
                    tail += 1

    return colors, (head < tail)

@njit(cache=True)
def _conflict_refine_csr(indptr, indices, colors, n, max_steps=2000):
    """
    Min-conflicts refinement for CSR coloring. Tries to resolve remaining
    conflicts without changing the solver family.
    """
    N = indptr.size - 1
    if N == 0:
        return colors, False
    conflicts = np.zeros(N, np.int32)
    color_counts = np.zeros(n + 1, np.int32)
    for u in range(N):
        cu = colors[u]
        if cu != 0:
            color_counts[cu] += 1
    total_conf = 0
    for u in range(N):
        cu = colors[u]
        if cu == 0:
            continue
        cnt = 0
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        for k in range(row_beg, row_end):
            v = indices[k]
            if colors[v] == cu:
                cnt += 1
        conflicts[u] = cnt
        total_conf += cnt
    if total_conf == 0:
        return colors, False

    seed = 1103515245 * (N + 1) + 12345
    for _ in range(max_steps):
        if total_conf == 0:
            break
        u = -1
        maxc = 0
        for i in range(N):
            if conflicts[i] > maxc:
                maxc = conflicts[i]
                u = i
        if u == -1 or maxc == 0:
            break
        row_beg = indptr[u]
        row_end = indptr[u + 1]
        counts = np.zeros(n + 1, np.int32)
        for k in range(row_beg, row_end):
            v = indices[k]
            cv = colors[v]
            if cv != 0:
                counts[cv] += 1

        best = 1
        best_conf = counts[1]
        best_usage = color_counts[1]
        for c in range(2, n + 1):
            conf = counts[c]
            usage = color_counts[c]
            if conf < best_conf or (conf == best_conf and usage < best_usage):
                best_conf = conf
                best_usage = usage
                best = c

        if best == colors[u]:
            seed = (1103515245 * seed + 12345) & 0x7FFFFFFF
            pick = seed % n
            idx = 0
            chosen = best
            for c in range(1, n + 1):
                if counts[c] == best_conf:
                    if idx == pick:
                        chosen = c
                        break
                    idx += 1
            best = chosen

        if best != colors[u]:
            old = colors[u]
            color_counts[old] -= 1
            colors[u] = best
            color_counts[best] += 1
            old_conf_u = conflicts[u]
            cnt = 0
            for k in range(row_beg, row_end):
                v = indices[k]
                if colors[v] == colors[u]:
                    cnt += 1
            conflicts[u] = cnt
            total_conf += conflicts[u] - old_conf_u
            for k in range(row_beg, row_end):
                v = indices[k]
                cu = colors[v]
                if cu == 0:
                    continue
                old_conf_v = conflicts[v]
                cntv = 0
                vb = indptr[v]
                ve = indptr[v + 1]
                for kk in range(vb, ve):
                    vv = indices[kk]
                    if colors[vv] == cu:
                        cntv += 1
                conflicts[v] = cntv
                total_conf += conflicts[v] - old_conf_v

    conflict = total_conf != 0
    return colors, conflict
def render_net(conmap, n=4, rand=12, depth=0, max_depth=5, offset=0):
    node_count = len(conmap)
    thresh = max(10000, node_count * 20)
    if depth < max_depth:
        nodes = list(conmap.keys())
        rng = np.random.default_rng(depth + 1 + offset)
        rng.shuffle(nodes)
        colors = dict(zip(nodes, [0] * len(nodes)))
        counter = dict(zip(nodes, [0] * len(nodes)))
        nodes = deque(nodes)
        count = 0
        while nodes and count < thresh:
            count += 1
            k = nodes.popleft()
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
            print(n, '-color algorthm failed,trying again with', n + 1, 'colors. Depth', depth)
            colors = render_net(conmap, n + 1, rand, depth + 1, max_depth, offset)
        return colors
    print('N-color algorthm exceeded max depth of', max_depth)
    return None
