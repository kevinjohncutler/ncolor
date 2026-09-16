// Graph helpers for two-hop neighbors and symmetric pair adjacency.
// Pure C++; no Python dependencies.

#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ncolor_cpp {

// 2-hop neighbors of a graph in CSR form.
// out[u] = { v : graph_distance(u, v) == 2 }. Both directions emitted
// (so the result is symmetric: u ∈ twohop[v] iff v ∈ twohop[u]).
// Uses an "epoch tag" per vertex (O(N) scratch, no per-cell set
// allocation) — visits cells at distance ≤ 2 in two passes through the
// CSR adjacency.
inline void compute_two_hop_csr(
    const int32_t* adj_indptr, const int32_t* adj_indices, int32_t N,
    std::vector<int32_t>& out_indptr,
    std::vector<int32_t>& out_indices)
{
    out_indptr.assign(static_cast<size_t>(N) + 1, 0);

    // Epoch-tag scratch: seen[v] == u + 1 means "marked during u's pass"
    // (offset by 1 so the zero-initialized state means "never seen").
    std::vector<int32_t> seen(static_cast<size_t>(N), 0);

    // First pass: count outgoing 2-hop edges per cell.
    for (int32_t u = 0; u < N; ++u) {
        const int32_t epoch = u + 1;
        const int32_t k_lo = adj_indptr[u], k_hi = adj_indptr[u + 1];
        // Mark 1-hop neighbors + self so they're skipped.
        seen[u] = epoch;
        for (int32_t k = k_lo; k < k_hi; ++k) seen[adj_indices[k]] = epoch;
        int32_t count = 0;
        for (int32_t k = k_lo; k < k_hi; ++k) {
            const int32_t v = adj_indices[k];
            const int32_t kk_lo = adj_indptr[v], kk_hi = adj_indptr[v + 1];
            for (int32_t kk = kk_lo; kk < kk_hi; ++kk) {
                const int32_t w = adj_indices[kk];
                if (seen[w] != epoch) {
                    seen[w] = epoch;
                    ++count;
                }
            }
        }
        out_indptr[u + 1] = count;
    }

    // Accumulate row ends without overwriting counts before they are read.
    int64_t total = 0;
    for (int32_t u = 0; u < N; ++u) {
        total += out_indptr[u + 1];
        if (total > std::numeric_limits<int32_t>::max())
            throw std::overflow_error("two-hop graph exceeds int32 edge capacity");
        out_indptr[u + 1] = static_cast<int32_t>(total);
    }

    out_indices.assign(static_cast<size_t>(total), 0);

    // Second pass: fill indices, re-using the epoch scratch.
    std::fill(seen.begin(), seen.end(), 0);
    std::vector<int32_t> write_pos = out_indptr;  // copy

    for (int32_t u = 0; u < N; ++u) {
        const int32_t epoch = u + 1;
        const int32_t k_lo = adj_indptr[u], k_hi = adj_indptr[u + 1];
        seen[u] = epoch;
        for (int32_t k = k_lo; k < k_hi; ++k) seen[adj_indices[k]] = epoch;
        for (int32_t k = k_lo; k < k_hi; ++k) {
            const int32_t v = adj_indices[k];
            const int32_t kk_lo = adj_indptr[v], kk_hi = adj_indptr[v + 1];
            for (int32_t kk = kk_lo; kk < kk_hi; ++kk) {
                const int32_t w = adj_indices[kk];
                if (seen[w] != epoch) {
                    seen[w] = epoch;
                    out_indices[write_pos[u]++] = w;
                }
            }
        }
    }
}


// Build a "symmetric pair CSR" from a list of unordered (u, v, w) triples.
// Each input pair (u, v, w) emits two CSR entries: u → (v, w) and v → (u, w).
// This is how the Kempe-SA kernel expects pair-weighted CSRs (both
// directions stored so boundary delta computation is symmetric).
inline void build_symmetric_pair_csr(
    const int32_t* pair_u, const int32_t* pair_v, const double* pair_w,
    int32_t n_pairs, int32_t N,
    std::vector<int32_t>& out_indptr,
    std::vector<int32_t>& out_indices,
    std::vector<double>&  out_weights)
{
    out_indptr.assign(static_cast<size_t>(N) + 1, 0);
    for (int32_t p = 0; p < n_pairs; ++p) {
        out_indptr[pair_u[p] + 1] += 1;
        out_indptr[pair_v[p] + 1] += 1;
    }
    for (int32_t u = 0; u < N; ++u) out_indptr[u + 1] += out_indptr[u];

    const int32_t total = out_indptr[N];
    out_indices.assign(static_cast<size_t>(total), 0);
    out_weights.assign(static_cast<size_t>(total), 0.0);

    std::vector<int32_t> write_pos = out_indptr;
    for (int32_t p = 0; p < n_pairs; ++p) {
        const int32_t u = pair_u[p], v = pair_v[p];
        const double  w = pair_w[p];
        out_indices[write_pos[u]] = v;
        out_weights[write_pos[u]++] = w;
        out_indices[write_pos[v]] = u;
        out_weights[write_pos[v]++] = w;
    }
}

} // namespace ncolor_cpp
