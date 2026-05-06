/*
 * color.hpp — C++ port of ncolor's graph-coloring + CSR-build kernels.
 *
 * Pure scalar code (the numba originals are @njit, no parallel=True), so the
 * point is parity + lower call overhead, not parallelism. Together with
 * connect.hpp + expand.hpp, this lets ncolor.label run end-to-end without
 * touching the numba pipeline. The repair fallback (`_repair_coloring` /
 * `_kempe_repair_csr`) is intentionally omitted — for the 4-coloring of
 * planar adjacency graphs we hit in practice, the BFS converges on the first
 * attempt; if you need the full repair chain, fall back to the numba path.
 */

#ifndef NCOLOR_COLOR_HPP
#define NCOLOR_COLOR_HPP

#include <cstdint>
#include <cstring>
#include <vector>

namespace ncolor_cpp {

// Build symmetric CSR from M directed pairs (src[i], dst[i]). Output:
// indptr[N+1] and indices[2*M]. Neighbours within each row are unordered.
inline void build_csr_from_pairs(
        const int32_t* src, const int32_t* dst, int32_t N, int32_t M,
        std::vector<int32_t>& indptr, std::vector<int32_t>& indices) {
    indptr.assign(static_cast<size_t>(N) + 1, 0);
    std::vector<int32_t> degree(static_cast<size_t>(N), 0);
    for (int32_t i = 0; i < M; ++i) {
        degree[src[i]] += 1;
        degree[dst[i]] += 1;
    }
    indptr[0] = 0;
    for (int32_t i = 0; i < N; ++i) indptr[i + 1] = indptr[i] + degree[i];

    indices.assign(static_cast<size_t>(2) * M, 0);
    std::vector<int32_t> write(indptr.begin(), indptr.begin() + N);  // copy 1st N entries
    for (int32_t i = 0; i < M; ++i) {
        const int32_t s = src[i];
        const int32_t d = dst[i];
        indices[write[s]++] = d;
        indices[write[d]++] = s;
    }
}

// BFS-based legacy coloring. Returns true if all nodes were assigned a
// non-zero color before max_iter; false if the queue still had pending
// nodes (caller should retry with bigger n or repair).
//
// `rand`: every `rand`-th re-visit of a node, pick a deterministic
// pseudo-random color to break out of cycles. Match the numba LCG seed
// exactly so results are bit-identical.
inline bool color_graph_csr_legacy(
        const int32_t* indptr, const int32_t* indices, int32_t N,
        int32_t n_colors, int32_t rand_period, int32_t offset, int64_t max_iter,
        std::vector<uint8_t>& colors, bool welsh_powell = false) {
    colors.assign(static_cast<size_t>(N), 0);
    std::vector<int32_t> counter(static_cast<size_t>(N), 0);

    // Queue capacity heuristic mirrors the numba version; grows on demand.
    int64_t qcap = std::max<int64_t>({static_cast<int64_t>(indptr[N]) + N,
                                      static_cast<int64_t>(N) * 8, 1});
    std::vector<int32_t> q(static_cast<size_t>(qcap));
    int64_t head = 0;
    int64_t tail = N;
    if (welsh_powell) {
        // Visit nodes in DESCENDING degree order. High-degree nodes are
        // most constrained (most neighbours competing for distinct
        // colours), so colouring them first reduces forced retries and
        // tends to spread colour usage more evenly across the graph.
        // Bucket sort: O(N + max_degree). For our planar / near-planar
        // adjacency graphs, max_degree is small (typically ≤ 12).
        int32_t max_deg = 0;
        std::vector<int32_t> deg(static_cast<size_t>(N));
        for (int32_t i = 0; i < N; ++i) {
            deg[i] = indptr[i + 1] - indptr[i];
            if (deg[i] > max_deg) max_deg = deg[i];
        }
        std::vector<int32_t> bucket_head(static_cast<size_t>(max_deg) + 2, 0);
        for (int32_t i = 0; i < N; ++i) bucket_head[deg[i] + 1]++;
        // Convert to write-index per bucket, ordered high → low.
        std::vector<int32_t> wpos(static_cast<size_t>(max_deg) + 1, 0);
        // Compute starting position of each bucket (degree d) in the
        // descending-by-degree output. Bucket for degree max_deg starts
        // at 0; bucket for degree max_deg-1 starts after that; ...
        int32_t cum = 0;
        for (int32_t d = max_deg; d >= 0; --d) {
            wpos[d] = cum;
            cum += bucket_head[d + 1];
        }
        for (int32_t i = 0; i < N; ++i) {
            q[wpos[deg[i]]++] = i;
        }
    } else {
        for (int32_t i = 0; i < N; ++i) q[i] = i;
    }

    const uint32_t fullmask = (1u << (n_colors + 1)) - 2;
    int64_t count = 0;
    while (head < tail && count < max_iter) {
        const int32_t u = q[head++];
        ++count;
        ++counter[u];

        const int32_t row_beg = indptr[u];
        const int32_t row_end = indptr[u + 1];
        uint32_t mask = 0;
        bool all_present = false;
        for (int32_t k = row_beg; k < row_end; ++k) {
            const uint8_t cv = colors[indices[k]];
            if (cv != 0) {
                mask |= (1u << cv);
                if (mask == fullmask) { all_present = true; break; }
            }
        }

        uint8_t csel = 0;
        if (!all_present) {
            for (int32_t c = 1; c <= n_colors; ++c) {
                if ((mask & (1u << c)) == 0) { csel = static_cast<uint8_t>(c); break; }
            }
            counter[u] = 0;
        } else {
            // Tally colors among neighbours, pick least-used (excluding u's current).
            int32_t cnt[32] = {0};  // max 31 colors — comfortably above any usage
            for (int32_t k = row_beg; k < row_end; ++k) {
                const uint8_t cv = colors[indices[k]];
                if (cv != 0 && cv < 32) cnt[cv] += 1;
            }
            int32_t minv = cnt[1];
            csel = 1;
            const uint8_t cu = colors[u];
            if (cu != 0) minv = 2147483647;
            for (int32_t c = 1; c <= n_colors; ++c) {
                if (c == cu) continue;
                if (cnt[c] < minv) { minv = cnt[c]; csel = static_cast<uint8_t>(c); }
            }
            // Random restart after `rand_period` re-attempts. Use the exact LCG
            // seed scheme as the numba code so output matches bit-for-bit.
            if (rand_period > 0 && counter[u] == rand_period) {
                counter[u] = 0;
                const uint32_t seed = static_cast<uint32_t>(
                    1103515245u * (static_cast<uint32_t>(u) + static_cast<uint32_t>(count) + static_cast<uint32_t>(offset)) + 12345u
                ) & 0x7FFFFFFFu;
                csel = static_cast<uint8_t>(1 + (seed % static_cast<uint32_t>(n_colors)));
            }
        }

        if (colors[u] != csel) {
            colors[u] = csel;
            for (int32_t k = row_beg; k < row_end; ++k) {
                const int32_t v = indices[k];
                if (colors[v] == csel) {
                    if (tail >= qcap) {
                        const int64_t newcap = std::max<int64_t>(qcap * 2, tail + 1);
                        q.resize(static_cast<size_t>(newcap));
                        qcap = newcap;
                    }
                    q[tail++] = v;
                }
            }
        }
    }

    return head >= tail;  // true = finished all in queue
}

// Conflict check — true if any edge has same-colored endpoints.
inline bool has_conflict_csr(
        const int32_t* indptr, const int32_t* indices, int32_t N,
        const uint8_t* colors) {
    for (int32_t u = 0; u < N; ++u) {
        const uint8_t cu = colors[u];
        if (cu == 0) continue;
        const int32_t row_beg = indptr[u];
        const int32_t row_end = indptr[u + 1];
        for (int32_t k = row_beg; k < row_end; ++k) {
            const int32_t v = indices[k];
            if (v > u && colors[v] == cu) return true;
        }
    }
    return false;
}

// Local-greedy repair: a few passes of "if u has color 0 or matches a
// neighbour, recolor to the smallest free color in 1..n". Cheap and resolves
// most residual conflicts the BFS leaves behind. Returns true if no conflicts
// remain afterward. Mirrors numba's `_repair_coloring`.
inline bool repair_coloring(
        const int32_t* indptr, const int32_t* indices, int32_t N,
        int32_t n_colors, int max_passes,
        std::vector<uint8_t>& colors) {
    int passes = 0;
    bool changed = true;
    while (changed && passes < max_passes) {
        changed = false;
        ++passes;
        for (int32_t u = 0; u < N; ++u) {
            const uint8_t cu = colors[u];
            uint32_t mask = 0;
            const int32_t row_beg = indptr[u];
            const int32_t row_end = indptr[u + 1];
            for (int32_t k = row_beg; k < row_end; ++k) {
                const uint8_t cv = colors[indices[k]];
                if (cv != 0) mask |= (1u << cv);
            }
            if (cu == 0 || (mask & (1u << cu)) != 0) {
                uint8_t csel = 0;
                for (int32_t c = 1; c <= n_colors; ++c) {
                    if ((mask & (1u << c)) == 0) { csel = static_cast<uint8_t>(c); break; }
                }
                if (csel != 0 && csel != cu) {
                    colors[u] = csel;
                    changed = true;
                }
            }
        }
    }
    return !has_conflict_csr(indptr, indices, N, colors.data());
}

} // namespace ncolor_cpp

#endif // NCOLOR_COLOR_HPP
