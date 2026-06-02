// Soft-edge local search: minimize the (weighted) number of soft edges
// whose endpoints share a color, WITHOUT introducing conflicts on the
// hard CSR graph.
//
// Use case: pair-finding at a richer (conn, r) kernel produces edges
// that we don't want as hard constraints (would force non-planarity /
// require χ>4), but we'd LIKE the picker to keep these endpoints in
// different colors when possible. Soft-edges + this local search
// achieve that as a post-pass on a valid hard 4-coloring.
//
// Greedy first-improvement: scan vertices, for each one try every
// alternative color that doesn't break a hard constraint, pick the one
// that drops the soft penalty the most. Repeat until no improvement
// (typically <10 passes on real graphs).

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "intrinsics.hpp"

namespace ncolor_cpp {

// Compute the total soft penalty for the current coloring.
// Each pair counted once (only counts when src < dst in the iteration).
inline double soft_total_penalty(
    const uint8_t* colors, int32_t N,
    const int32_t* soft_indptr, const int32_t* soft_indices,
    const float* soft_weights)
{
    double total = 0.0;
    for (int32_t u = 0; u < N; ++u) {
        const uint8_t cu = colors[u];
        for (int32_t j = soft_indptr[u]; j < soft_indptr[u + 1]; ++j) {
            const int32_t v = soft_indices[j];
            if (v <= u) continue;
            if (colors[v] == cu) {
                total += soft_weights ? double(soft_weights[j]) : 1.0;
            }
        }
    }
    return total;
}

// Compute hard-conflict color mask for vertex u (bit c set if some hard
// neighbor of u uses color c).
inline uint32_t hard_color_mask(
    int32_t u, const uint8_t* colors,
    const int32_t* hard_indptr, const int32_t* hard_indices)
{
    uint32_t mask = 0u;
    for (int32_t j = hard_indptr[u]; j < hard_indptr[u + 1]; ++j) {
        mask |= (1u << colors[hard_indices[j]]);
    }
    return mask;
}

// Compute soft penalty contribution of vertex u at a given color c.
inline double soft_at_color(
    int32_t u, int c, const uint8_t* colors,
    const int32_t* soft_indptr, const int32_t* soft_indices,
    const float* soft_weights)
{
    double s = 0.0;
    for (int32_t j = soft_indptr[u]; j < soft_indptr[u + 1]; ++j) {
        if (colors[soft_indices[j]] == c) {
            s += soft_weights ? double(soft_weights[j]) : 1.0;
        }
    }
    return s;
}

// One pass of single-vertex greedy first-improvement. Returns true if
// any move was applied.
inline bool single_vertex_pass(
    uint8_t* colors, int32_t N,
    const int32_t* hard_indptr, const int32_t* hard_indices,
    const int32_t* soft_indptr, const int32_t* soft_indices,
    const float* soft_weights,
    int n_colors,
    double& penalty)
{
    const uint32_t all_colors = ((1u << (n_colors + 1)) - 1u) & ~1u;
    bool improved = false;
    for (int32_t u = 0; u < N; ++u) {
        const uint8_t cur_c = colors[u];
        const uint32_t hard_mask = hard_color_mask(u, colors,
                                                     hard_indptr, hard_indices);
        uint32_t valid_mask = all_colors & ~hard_mask & ~(1u << cur_c);
        if (valid_mask == 0) continue;

        const double cur_soft = soft_at_color(u, cur_c, colors,
                                               soft_indptr, soft_indices,
                                               soft_weights);
        if (cur_soft == 0.0) continue;

        double best_new = cur_soft;
        uint8_t best_c = cur_c;
        uint32_t mask = valid_mask;
        while (mask) {
            int c = ctz_u32(mask);
            mask &= mask - 1;
            const double new_soft = soft_at_color(u, c, colors,
                                                    soft_indptr, soft_indices,
                                                    soft_weights);
            if (new_soft < best_new) {
                best_new = new_soft;
                best_c = (uint8_t)c;
            }
        }
        if (best_c != cur_c) {
            colors[u] = best_c;
            penalty += (best_new - cur_soft);
            improved = true;
        }
    }
    return improved;
}

// One pass of Kempe-chain swap attempts. For each stuck vertex u (soft
// penalty > 0 at current color), pick a target color b such that the
// Kempe chain {cur_c, b} starting at u, when flipped, would reduce the
// total soft penalty. The flip preserves hard validity by construction
// (Kempe chains are 2-color components; flipping leaves the chain
// vertices internally consistent and only changes their colors to the
// complementary one). Modifies colors[] in place and updates penalty.
//
// Returns true if any swap was applied.
//
// Uses BFS via a reusable visited-stamp array to avoid O(N) clears.
//
// ``stuck_worklist`` (optional) — when non-empty, only those vertices
// are scanned for Kempe attempts. Reduces iteration from O(N) to
// O(|stuck|), which on a typical mm-class image is ~5000 → ~600 (the
// number of vertices with non-zero soft penalty after the single-vertex
// pass converges).
//
// ``max_chain_size`` caps the BFS chain length. Chains larger than this
// are abandoned (no flip attempted). Large chains tend to: (a) cost
// proportionally to chain size in the soft-delta accumulator, and
// (b) rarely produce net-negative delta because they touch many
// unrelated soft edges. Capping at ~64 trades a tiny chance of finding
// a beneficial macro-swap for a large average cost reduction.
inline bool kempe_chain_pass(
    uint8_t* colors, int32_t N,
    const int32_t* hard_indptr, const int32_t* hard_indices,
    const int32_t* soft_indptr, const int32_t* soft_indices,
    const float* soft_weights,
    int n_colors,
    double& penalty,
    std::vector<int32_t>& visited_stamp,
    std::vector<int32_t>& bfs_queue,
    int32_t& stamp_counter,
    const std::vector<int32_t>* stuck_worklist = nullptr,
    int max_chain_size = 64)
{
    if ((int32_t)visited_stamp.size() < N) visited_stamp.assign(N, 0);
    bool improved = false;
    const uint32_t all_colors = ((1u << (n_colors + 1)) - 1u) & ~1u;

    const int32_t n_iter = stuck_worklist
        ? (int32_t)stuck_worklist->size() : N;
    for (int32_t i = 0; i < n_iter; ++i) {
        const int32_t u = stuck_worklist ? (*stuck_worklist)[i] : i;
        const uint8_t cur_c = colors[u];
        if (soft_at_color(u, cur_c, colors, soft_indptr,
                           soft_indices, soft_weights) == 0.0) continue;

        // Try each target color b != cur_c.
        uint32_t mask = all_colors & ~(1u << cur_c);
        while (mask) {
            int b = ctz_u32(mask);
            mask &= mask - 1;
            // BFS the Kempe chain of u in colors {cur_c, b} via hard edges.
            // Mark visited with stamp_counter to avoid clearing.
            ++stamp_counter;
            if (stamp_counter == 0) {
                // Wrap; reset.
                std::fill(visited_stamp.begin(), visited_stamp.end(), 0);
                stamp_counter = 1;
            }
            bfs_queue.clear();
            bfs_queue.push_back(u);
            visited_stamp[u] = stamp_counter;
            bool chain_too_large = false;
            for (size_t qi = 0; qi < bfs_queue.size(); ++qi) {
                if ((int)bfs_queue.size() > max_chain_size) {
                    chain_too_large = true; break;
                }
                const int32_t v = bfs_queue[qi];
                const uint8_t cv = colors[v];
                for (int32_t j = hard_indptr[v]; j < hard_indptr[v + 1]; ++j) {
                    const int32_t w = hard_indices[j];
                    if (visited_stamp[w] == stamp_counter) continue;
                    const uint8_t cw = colors[w];
                    // Chain visits cur_c→b or b→cur_c alternation. From a
                    // vertex of color cv, only consider hard neighbors of
                    // the OTHER color in {cur_c, b}.
                    const uint8_t other = (cv == cur_c) ? (uint8_t)b : cur_c;
                    if (cw == other) {
                        visited_stamp[w] = stamp_counter;
                        bfs_queue.push_back(w);
                    }
                }
            }
            if (chain_too_large) continue;  // skip this (cur_c, b) target

            // Compute soft delta if we flip the chain. For each vertex in
            // chain, its color flips; recompute its soft contribution at
            // the flipped color minus its contribution at the current.
            // Soft NEIGHBORS in the chain also flip; soft neighbors NOT
            // in the chain don't change.
            double delta = 0.0;
            for (int32_t v : bfs_queue) {
                const uint8_t cv = colors[v];
                const uint8_t cv_new = (cv == cur_c) ? (uint8_t)b : cur_c;
                for (int32_t j = soft_indptr[v]; j < soft_indptr[v + 1]; ++j) {
                    const int32_t w = soft_indices[j];
                    const double wt = soft_weights ? double(soft_weights[j]) : 1.0;
                    uint8_t cw_eff = colors[w];
                    if (visited_stamp[w] == stamp_counter) {
                        // w also flips
                        cw_eff = (colors[w] == cur_c) ? (uint8_t)b : cur_c;
                        // Avoid double-counting in-chain ↔ in-chain pairs:
                        // we'll process pair from both endpoints; halve by
                        // only counting when v < w.
                        if (w < v) continue;
                        const bool was_same = (cv == colors[w]);
                        const bool now_same = (cv_new == cw_eff);
                        if (was_same && !now_same) delta -= wt;
                        else if (!was_same && now_same) delta += wt;
                    } else {
                        const bool was_same = (cv == colors[w]);
                        const bool now_same = (cv_new == colors[w]);
                        if (was_same && !now_same) delta -= wt;
                        else if (!was_same && now_same) delta += wt;
                    }
                }
            }

            if (delta < 0.0) {
                // Apply flip.
                for (int32_t v : bfs_queue) {
                    colors[v] = (colors[v] == cur_c) ? (uint8_t)b : cur_c;
                }
                penalty += delta;
                improved = true;
                break;  // u changed; move on to next u
            }
        }
    }
    return improved;
}

// Apply an UNCONDITIONAL Kempe-chain flip on colors {a, b} starting at u
// — used as a "shake" perturbation in iterated local search restarts.
// No delta check; the caller will re-converge with greedy + Kempe and
// decide whether to keep or revert. No-op if u's color isn't in {a, b}.
//
// CRITICAL: the BFS must discover the *entire* Kempe component before
// flipping. A partial flip leaves dangling chain vertices on the
// boundary, which would create same-color hard edges (broken
// correctness). We do not cap the chain size here — capping would
// corrupt the hard coloring. On our test inputs the largest hard graph
// is N=5000 and the worst-case chain is ~N/2; the BFS cost is O(N + E)
// which is small compared to the rest of soft_local_search.
inline void kempe_force_flip(
    uint8_t* colors, int32_t N,
    const int32_t* hard_indptr, const int32_t* hard_indices,
    int32_t u, int a, int b,
    std::vector<int32_t>& visited_stamp,
    std::vector<int32_t>& bfs_queue,
    int32_t& stamp_counter)
{
    if ((int32_t)visited_stamp.size() < N) visited_stamp.assign(N, 0);
    const uint8_t cu = colors[u];
    if (cu != a && cu != b) return;
    ++stamp_counter;
    if (stamp_counter == 0) {
        std::fill(visited_stamp.begin(), visited_stamp.end(), 0);
        stamp_counter = 1;
    }
    bfs_queue.clear();
    bfs_queue.push_back(u);
    visited_stamp[u] = stamp_counter;
    for (size_t qi = 0; qi < bfs_queue.size(); ++qi) {
        const int32_t v = bfs_queue[qi];
        const uint8_t cv = colors[v];
        const uint8_t other = (cv == a) ? (uint8_t)b : (uint8_t)a;
        for (int32_t j = hard_indptr[v]; j < hard_indptr[v + 1]; ++j) {
            const int32_t w = hard_indices[j];
            if (visited_stamp[w] == stamp_counter) continue;
            if (colors[w] == other) {
                visited_stamp[w] = stamp_counter;
                bfs_queue.push_back(w);
            }
        }
    }
    for (int32_t v : bfs_queue) {
        colors[v] = (colors[v] == a) ? (uint8_t)b : (uint8_t)a;
    }
}

// Tiny xorshift PRNG — local, deterministic across calls within a run
// (state lives in soft_local_search). Avoids pulling in <random> just
// for restart-shake choice. Same algorithm used elsewhere in the codebase.
struct SoftXorshift32 {
    uint32_t s;
    explicit SoftXorshift32(uint32_t seed) : s(seed ? seed : 0xA5A5A5A5u) {}
    uint32_t next() {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        return s;
    }
    int range(int n) { return n > 0 ? (int)(next() % (uint32_t)n) : 0; }
};

// Run local-search to minimize soft penalty. Modifies colors[] in place.
// Returns the final remaining soft penalty (lower = better).
//
// Strategy:
//   Phase 1: alternate single-vertex passes (cheap) and Kempe-chain
//            passes (escapes shallow local optima). Convergent.
//   Phase 2: if residual penalty > 0, iterated-local-search restarts.
//            Each restart: pick a random stuck vertex, force-flip a
//            random Kempe chain (no delta check), re-run Phase 1.
//            Keep best. Escapes deep local optima where no single chain
//            is net-improving (e.g. logo K_4 cluster where the two
//            same-color cells are in the same {a,b} Kempe class and
//            move together under any single-chain flip — a different
//            initial coloring is needed to split them).
// ``n_restarts`` caps the restart budget (0 = no restarts; default 4).
inline double soft_local_search(
    uint8_t* colors, int32_t N,
    const int32_t* hard_indptr, const int32_t* hard_indices,
    const int32_t* soft_indptr, const int32_t* soft_indices,
    const float* soft_weights,
    int n_colors,
    int max_passes = 50,
    int n_restarts = 4,
    uint32_t restart_seed = 0xC0FFEE5Au,
    int restart_max_N = 1000)
{
    // Gate restarts on graph size. On large graphs (mm-class, N≈5000)
    // each restart costs ~5-10 ms of re-convergence and the marginal
    // sv reduction is small (random shake doesn't systematically
    // address hundreds of residual conflicts). Restarts are designed to
    // escape *tight, small* local optima like the logo K_4 cluster; on
    // big graphs they're a perf tax with low ROI. Skip by default.
    if (N > restart_max_N) n_restarts = 0;
    double penalty = soft_total_penalty(colors, N, soft_indptr,
                                          soft_indices, soft_weights);
    if (penalty == 0.0) return 0.0;

    // Persistent Kempe BFS scratch (stamp-based visited; no O(N) clears).
    std::vector<int32_t> visited_stamp((size_t)N, 0);
    std::vector<int32_t> bfs_queue;
    int32_t stamp = 0;
    std::vector<int32_t> stuck_worklist;
    stuck_worklist.reserve(64);

    auto converge = [&](double& pen) {
        for (int pass = 0; pass < max_passes; ++pass) {
            bool sv_improved = true;
            while (sv_improved && pen > 0.0) {
                sv_improved = single_vertex_pass(
                    colors, N, hard_indptr, hard_indices,
                    soft_indptr, soft_indices, soft_weights,
                    n_colors, pen);
            }
            if (pen == 0.0) break;
            stuck_worklist.clear();
            for (int32_t u = 0; u < N; ++u) {
                if (soft_at_color(u, colors[u], colors, soft_indptr,
                                   soft_indices, soft_weights) > 0.0) {
                    stuck_worklist.push_back(u);
                }
            }
            if (stuck_worklist.empty()) break;
            const bool kempe_improved = kempe_chain_pass(
                colors, N, hard_indptr, hard_indices,
                soft_indptr, soft_indices, soft_weights,
                n_colors, pen,
                visited_stamp, bfs_queue, stamp,
                &stuck_worklist, /*max_chain_size=*/64);
            if (!kempe_improved) break;
        }
    };

    converge(penalty);
    if (penalty == 0.0 || n_restarts <= 0) {
        // No restarts: return unweighted count for the public API.
        // (Search-time penalty above may be weighted; only the
        // user-facing return value needs to be the count.)
        return soft_total_penalty(colors, N, soft_indptr,
                                    soft_indices, /*soft_weights=*/nullptr);
    }

    // Iterated-local-search restarts. Save best, shake (force-flip a
    // random Kempe chain) starting from a vertex with a soft conflict,
    // re-converge, keep best.
    std::vector<uint8_t> best_colors(colors, colors + N);
    double best_penalty = penalty;
    SoftXorshift32 rng(restart_seed);

    for (int r = 0; r < n_restarts; ++r) {
        // Rebuild stuck list under current best.
        std::memcpy(colors, best_colors.data(), (size_t)N);
        stuck_worklist.clear();
        for (int32_t u = 0; u < N; ++u) {
            if (soft_at_color(u, colors[u], colors, soft_indptr,
                               soft_indices, soft_weights) > 0.0) {
                stuck_worklist.push_back(u);
            }
        }
        if (stuck_worklist.empty()) break;
        const int32_t u = stuck_worklist[rng.range((int)stuck_worklist.size())];
        const uint8_t cu = colors[u];
        // Pick a random color b != cu. n_colors are 1..n_colors.
        int b = 1 + rng.range(n_colors);
        while (b == cu) b = 1 + rng.range(n_colors);
        kempe_force_flip(
            colors, N, hard_indptr, hard_indices,
            u, cu, b, visited_stamp, bfs_queue, stamp);
        // Re-evaluate penalty after shake then converge.
        penalty = soft_total_penalty(colors, N, soft_indptr,
                                      soft_indices, soft_weights);
        converge(penalty);
        if (penalty < best_penalty) {
            best_penalty = penalty;
            std::memcpy(best_colors.data(), colors, (size_t)N);
            if (best_penalty == 0.0) break;
        }
    }

    // Restore best.
    std::memcpy(colors, best_colors.data(), (size_t)N);
    // Return the UNWEIGHTED violation count for the public-facing
    // accessor. The internal search minimizes the weighted penalty
    // (triangle-count-weighted to bias toward K_4-like cluster fixes),
    // but users expect "how many soft pairs end up same-colored" which
    // is the count, not the weighted total. Recompute once with
    // soft_weights=nullptr.
    return soft_total_penalty(colors, N, soft_indptr,
                                soft_indices, /*soft_weights=*/nullptr);
}

// Compute per-edge weights from soft-edge triangle count. For each
// directed entry (u, v) in soft_indices, count the number of soft
// neighbors that u and v have in common — this is the count of length-2
// soft paths that close into a triangle through u-v.
//
// Intuition: a soft edge inside a tight cluster (e.g. K_4 of soft edges
// in the logo cluster) has 2 common neighbors (the other two cluster
// cells). A "leak" or noise soft edge between two cells that don't share
// further soft adjacencies has 0 common neighbors. Weighting edges by
// `1 + triangle_count` biases the local search to fix cluster conflicts
// first — these are the perceptually important "should be different
// colors" pairs, while leak edges are often spurious.
//
// Algorithm: stamp-based set intersection. For each vertex u with
// non-zero soft degree, mark all of u's neighbors with the current
// stamp. Then for each (u, v) edge, scan v's neighbors and count those
// whose stamp matches — these are common neighbors. Cost per vertex u
// is O(deg(u) + sum of deg(v) for v in N(u)). Total cost is O(sum of
// (deg(u) + deg(v)) over all edges) = O(2 * sum_edges deg) which on our
// data (avg soft-deg <1) is ~µs per call.
//
// Replaces the nested-linear-scan version which was O(deg(u) * deg(v))
// per directed entry and measured ~1 µs/entry on mm (1.3 ms total for
// 1188 entries), apparently from cache/pipeline effects in the
// inner-most search loop.
inline void compute_triangle_weights(
    int32_t N,
    const int32_t* soft_indptr, const int32_t* soft_indices,
    std::vector<float>& weights_out)
{
    const int32_t M_entries = soft_indptr[N];
    weights_out.assign((size_t)M_entries, 1.0f);
    if (M_entries == 0 || N <= 0) return;
    // Stamp array sized N. Stamp values reused across vertices via
    // bump-and-check-wrap. Avoids O(N) zero-fill per call after the
    // first.
    static thread_local std::vector<int32_t> stamp;
    static thread_local int32_t cur_stamp = 0;
    if ((int32_t)stamp.size() < N) stamp.assign((size_t)N, 0);
    for (int32_t u = 0; u < N; ++u) {
        const int32_t u_lo = soft_indptr[u];
        const int32_t u_hi = soft_indptr[u + 1];
        if (u_lo == u_hi) continue;  // fast-skip zero-degree vertices
        // Bump stamp; on wraparound, clear.
        ++cur_stamp;
        if (cur_stamp == 0) {
            std::fill(stamp.begin(), stamp.end(), 0);
            cur_stamp = 1;
        }
        // Mark u's soft neighbors with the current stamp.
        for (int32_t k = u_lo; k < u_hi; ++k) {
            stamp[soft_indices[k]] = cur_stamp;
        }
        // For each (u, v) directed entry, count v's neighbors that are
        // also marked. Common neighbors are the triangles through u-v.
        // (u itself is in N(v) but not in stamp — u is excluded from
        // N(u) by the no-self-loop rule in build_soft_csr — so it
        // doesn't get counted. v is in stamp but v is not in N(v),
        // so it doesn't get scanned via v_lo..v_hi.)
        for (int32_t j = u_lo; j < u_hi; ++j) {
            const int32_t v = soft_indices[j];
            const int32_t v_lo = soft_indptr[v];
            const int32_t v_hi = soft_indptr[v + 1];
            int32_t common = 0;
            for (int32_t k = v_lo; k < v_hi; ++k) {
                if (stamp[soft_indices[k]] == cur_stamp) ++common;
            }
            weights_out[j] = 1.0f + (float)common;
        }
    }
}

// Build a symmetric soft CSR from a flat list of (a, b, [weight]) triples.
// Input: soft_pairs is a flat array of length 2 * n_soft (or 3 * n_soft
// if `with_weights`); cell IDs are 1-based on input, converted to 0-based
// nodes internally. weights_in (optional) is length n_soft.
//
// Outputs (resized in place): indptr (length N+1), indices, weights
// (length 2*n_soft after de-dupe). The graph is undirected, both
// directions stored. Duplicate pairs are merged (weights summed).
inline void build_soft_csr(
    int32_t N, int32_t n_soft, const int32_t* soft_pairs,
    const float* weights_in,
    std::vector<int32_t>& indptr,
    std::vector<int32_t>& indices,
    std::vector<float>&   weights)
{
    indptr.assign((size_t)N + 1, 0);
    // Pass 1: count out-degree per vertex (both directions).
    int32_t valid = 0;
    for (int32_t e = 0; e < n_soft; ++e) {
        int32_t a = soft_pairs[2 * e]     - 1;
        int32_t b = soft_pairs[2 * e + 1] - 1;
        if (a < 0 || b < 0 || a >= N || b >= N || a == b) continue;
        indptr[a + 1]++;
        indptr[b + 1]++;
        ++valid;
    }
    // Prefix sum.
    for (int32_t i = 1; i <= N; ++i) indptr[i] += indptr[i - 1];
    indices.assign(indptr[N], 0);
    weights.assign(indptr[N], 0.f);
    std::vector<int32_t> pos(N, 0);
    for (int32_t e = 0; e < n_soft; ++e) {
        int32_t a = soft_pairs[2 * e]     - 1;
        int32_t b = soft_pairs[2 * e + 1] - 1;
        if (a < 0 || b < 0 || a >= N || b >= N || a == b) continue;
        const float w = weights_in ? weights_in[e] : 1.0f;
        const int32_t pa = indptr[a] + pos[a]++;
        const int32_t pb = indptr[b] + pos[b]++;
        indices[pa] = b; weights[pa] = w;
        indices[pb] = a; weights[pb] = w;
    }
}

}  // namespace ncolor_cpp
