/*
 * chamfer_topk_nd.hpp — N-D L1 top-K Voronoi label expansion.
 *
 * Generalises chamfer_topk_l1_2d to arbitrary dimensions using the same
 * (A, B, C) tile decomposition as chamfer.hpp's chamfer_st_l1_nd:
 *   A = product of dims BEFORE the sweep axis (outer parallel domain)
 *   B = shape[axis]                            (serial sweep axis)
 *   C = product of dims AFTER the sweep axis   (inner contiguous)
 *
 * Per pixel state (SoA layout):
 *   buf_l[i * K + k]  → k-th nearest cell ID at pixel i (sorted ascending by dist)
 *   buf_d[i * K + k]  → matching L1 distance
 * Empty slots: label = 0, dist = INF.
 *
 * Sweep: per axis, forward + backward 1D pass. Each pass relaxes each
 * pixel against its axis-adjacent neighbour by propagating the neighbour's
 * entire K-slot state (each slot becomes a candidate with dist+1).
 *
 * Per-pixel SIMD K=4 specialisation is at the bottom (AVX2 + NEON paths),
 * dispatched automatically when K==4.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// NOTE on SIMD K=4: empirically (M1 NEON, Zen2 AVX2) scalar K=4 with OOO
// execution wins over hand-written SIMD because the relax operation has
// horizontal reductions (vmaxvq / movemask) and a sift-up that has to
// fall back to scalar anyway. Removed SIMD path entirely for K=4 to
// keep the code straightforward.

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

// -- Scalar top-K relax (works for any K) ---------------------------------
template <int K>
inline void topk_relax_scalar(int32_t* __restrict lbls, int32_t* __restrict dsts,
                                int32_t cand_l, int32_t cand_d)
{
    if (cand_d >= dsts[K - 1]) return;
    // Dedup: cand_l already present?
    for (int k = 0; k < K; ++k) {
        if (lbls[k] == cand_l) {
            if (cand_d < dsts[k]) {
                dsts[k] = cand_d;
                int j = k;
                while (j > 0 && dsts[j] < dsts[j - 1]) {
                    std::swap(dsts[j], dsts[j - 1]);
                    std::swap(lbls[j], lbls[j - 1]);
                    --j;
                }
            }
            return;
        }
    }
    // New label, replace worst, sift up.
    lbls[K - 1] = cand_l;
    dsts[K - 1] = cand_d;
    for (int j = K - 1; j > 0 && dsts[j] < dsts[j - 1]; --j) {
        std::swap(dsts[j], dsts[j - 1]);
        std::swap(lbls[j], lbls[j - 1]);
    }
}


// Dispatch wrapper. Currently always scalar — see SIMD note in the
// header comments.
template <int K>
inline void topk_relax(int32_t* __restrict lbls, int32_t* __restrict dsts,
                       int32_t cand_l, int32_t cand_d) {
    topk_relax_scalar<K>(lbls, dsts, cand_l, cand_d);
}


// -- Init a single pixel's K-slot state -----------------------------------
template <int K>
inline void topk_init_pixel(int32_t* __restrict lbls, int32_t* __restrict dsts,
                              int32_t seed_label)
{
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
    if (seed_label != 0) {
        lbls[0] = seed_label;
        dsts[0] = 0;
        for (int k = 1; k < K; ++k) { lbls[k] = 0; dsts[k] = INF; }
    } else {
        for (int k = 0; k < K; ++k) { lbls[k] = 0; dsts[k] = INF; }
    }
}

// -- Relax current pixel against upstream-neighbour's full K-tuple --------
template <int K>
inline void topk_relax_from_neighbour(
    int32_t* __restrict dst_lbls, int32_t* __restrict dst_dsts,
    const int32_t* __restrict src_lbls, const int32_t* __restrict src_dsts)
{
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
    for (int k = 0; k < K; ++k) {
        const int32_t src_l = src_lbls[k];
        if (src_l == 0) continue;
        const int32_t src_d = src_dsts[k];
        if (src_d >= INF / 2) continue;
        topk_relax<K>(dst_lbls, dst_dsts, src_l, src_d + 1);
    }
}


// -- N-D top-K sweep -------------------------------------------------------
// Per-pixel state in SoA buffers: buf_l[M*K] (labels) and buf_d[M*K] (dists).
// Single init pass + (forward + backward sweep) per axis.
template <int K>
inline void chamfer_topk_l1_nd_sweep(
    const int32_t* __restrict seed_image,
    const std::vector<int64_t>& shape,
    int32_t* __restrict buf_l, int32_t* __restrict buf_d,
    ForkJoinPool* pool, int n_threads)
{
    const int ndim = static_cast<int>(shape.size());
    int64_t M = 1;
    for (int64_t s : shape) M *= s;

    // NOTE: per-pixel init is fused into the FIRST axis sweep (the
    // innermost axis, processed first). This keeps init writes in L1
    // for the immediate read by the forward sweep, saving 0.5-1 ms on
    // the logo vs a separate pre-pass over M.

    // --- Per-axis forward + backward sweep ---
    // Process innermost axis first (stride 1, cache-friendly), then
    // outer axes. Matches chamfer.hpp's ordering for 1-NN.
    bool first_axis = true;
    for (int ax = ndim - 1; ax >= 0; --ax) {
        int64_t inner = 1, outer = 1;
        for (int d = ax + 1; d < ndim; ++d) inner *= shape[d];
        for (int d = 0; d < ax; ++d) outer *= shape[d];
        const int64_t len = shape[ax];

        // For each line along this axis: forward then backward sweep.
        // A line is identified by (a, c) with a < outer and c < inner.
        // Pixel index along the line at position b is: a*len*inner + b*inner + c.

        const bool init_in_sweep = first_axis;
        // Innermost axis fast-path (inner == 1): contiguous pixels, use
        // pointer arithmetic so the inner loop is as tight as the old
        // 2D-specific code.
        auto sweep_line_inner = [&](int64_t a, int64_t c) {
            int32_t* row_l = buf_l + (a * len + c) * K;
            int32_t* row_d = buf_d + (a * len + c) * K;
            const int32_t* row_seed = seed_image + a * len + c;
            if (init_in_sweep) {
                topk_init_pixel<K>(row_l, row_d, row_seed[0]);
                int32_t* prev_l = row_l;
                int32_t* prev_d = row_d;
                for (int64_t b = 1; b < len; ++b) {
                    int32_t* px_l = row_l + b * K;
                    int32_t* px_d = row_d + b * K;
                    topk_init_pixel<K>(px_l, px_d, row_seed[b]);
                    topk_relax_from_neighbour<K>(px_l, px_d, prev_l, prev_d);
                    prev_l = px_l; prev_d = px_d;
                }
            } else {
                for (int64_t b = 1; b < len; ++b) {
                    int32_t* px_l = row_l + b * K;
                    int32_t* px_d = row_d + b * K;
                    int32_t* prev_l = row_l + (b - 1) * K;
                    int32_t* prev_d = row_d + (b - 1) * K;
                    topk_relax_from_neighbour<K>(px_l, px_d, prev_l, prev_d);
                }
            }
            for (int64_t b = len - 2; b >= 0; --b) {
                int32_t* px_l = row_l + b * K;
                int32_t* px_d = row_d + b * K;
                int32_t* next_l = row_l + (b + 1) * K;
                int32_t* next_d = row_d + (b + 1) * K;
                topk_relax_from_neighbour<K>(px_l, px_d, next_l, next_d);
            }
        };
        // Non-innermost axis: index-based access with `inner` stride.
        auto sweep_line_outer = [&](int64_t a, int64_t c) {
            const int64_t base = a * len * inner + c;
            if (init_in_sweep) {
                topk_init_pixel<K>(buf_l + base * K, buf_d + base * K,
                                     seed_image[base]);
                for (int64_t b = 1; b < len; ++b) {
                    const int64_t dst = base + b * inner;
                    const int64_t src = base + (b - 1) * inner;
                    topk_init_pixel<K>(buf_l + dst * K, buf_d + dst * K,
                                         seed_image[dst]);
                    topk_relax_from_neighbour<K>(
                        buf_l + dst * K, buf_d + dst * K,
                        buf_l + src * K, buf_d + src * K);
                }
            } else {
                for (int64_t b = 1; b < len; ++b) {
                    const int64_t dst = base + b * inner;
                    const int64_t src = base + (b - 1) * inner;
                    topk_relax_from_neighbour<K>(
                        buf_l + dst * K, buf_d + dst * K,
                        buf_l + src * K, buf_d + src * K);
                }
            }
            for (int64_t b = len - 2; b >= 0; --b) {
                const int64_t dst = base + b * inner;
                const int64_t src = base + (b + 1) * inner;
                topk_relax_from_neighbour<K>(
                    buf_l + dst * K, buf_d + dst * K,
                    buf_l + src * K, buf_d + src * K);
            }
        };
        // Direct branch on inner == 1 to avoid std::function overhead.
        const bool use_inner_path = (inner == 1);
        auto run = [&](int64_t i_line) {
            int64_t a = i_line / inner;
            int64_t c = i_line % inner;
            if (use_inner_path) sweep_line_inner(a, c);
            else                sweep_line_outer(a, c);
        };

        first_axis = false;
        const int64_t n_lines = outer * inner;
        if (pool && n_threads > 1 && n_lines >= 16) {
            dispatch_parallel(*pool, static_cast<size_t>(n_lines),
                static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
                [&](size_t i0, size_t i1) {
                    for (size_t i = i0; i < i1; ++i) run((int64_t)i);
                });
        } else {
            for (int64_t i = 0; i < n_lines; ++i) run(i);
        }
    }
}


// -- Aggregation: pair counts from per-pixel top-K labels ----------------
template <int K>
inline void aggregate_topk_pairs_nd(
    const int32_t* __restrict labels,
    const int32_t* __restrict dists,
    int64_t M, int32_t N,
    int32_t max_margin,
    int64_t* __restrict pair_count)
{
    const int32_t stride = N + 1;
    for (int64_t i = 0; i < M; ++i) {
        const int32_t* L = labels + i * K;
        const int32_t* D = (dists != nullptr) ? dists + i * K : nullptr;
        const int32_t base = (D != nullptr) ? D[0] : 0;
        int32_t distinct[K];
        int nd = 0;
        for (int k = 0; k < K; ++k) {
            const int32_t l = L[k];
            if (l == 0) continue;
            if (D != nullptr && (D[k] - base) > max_margin) break;
            bool seen = false;
            for (int j = 0; j < nd; ++j) {
                if (distinct[j] == l) { seen = true; break; }
            }
            if (!seen) distinct[nd++] = l;
        }
        for (int p = 0; p < nd; ++p) {
            for (int q = p + 1; q < nd; ++q) {
                int32_t a = distinct[p], b = distinct[q];
                if (a > b) std::swap(a, b);
                pair_count[(int64_t)a * stride + b] += 1;
            }
        }
    }
}


// -- Fused: chamfer top-K + aggregation, no intermediate Python arrays --
template <int K>
inline void chamfer_topk_pair_counts_nd(
    const int32_t* __restrict seed_image,
    const std::vector<int64_t>& shape,
    int32_t N, int32_t max_margin,
    int64_t* __restrict pair_count,
    ForkJoinPool* pool, int n_threads)
{
    int64_t M = 1;
    for (int64_t s : shape) M *= s;
    std::vector<int32_t> buf_l((size_t)M * K);
    std::vector<int32_t> buf_d((size_t)M * K);
    chamfer_topk_l1_nd_sweep<K>(seed_image, shape, buf_l.data(), buf_d.data(),
                                  pool, n_threads);
    aggregate_topk_pairs_nd<K>(buf_l.data(), buf_d.data(),
                                  M, N, max_margin, pair_count);
}

// -- Backward-compatibility shims matching old chamfer_topk.hpp signatures
//    so binding.cpp's 2D entry points keep compiling. They build a 2-element
//    shape vector and call the ND kernel.
template <int K>
inline void chamfer_topk_l1_2d_sweep(
    const int32_t* seed_image, int32_t H, int32_t W,
    int32_t* buf_l, int32_t* buf_d,
    ForkJoinPool* pool, int n_threads)
{
    std::vector<int64_t> shape{(int64_t)H, (int64_t)W};
    chamfer_topk_l1_nd_sweep<K>(seed_image, shape, buf_l, buf_d, pool, n_threads);
}

template <int K>
inline void chamfer_topk_l1_2d(
    const int32_t* seed_image, int32_t H, int32_t W,
    int32_t* out_labels, int32_t* out_dists,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    std::vector<int64_t> shape{(int64_t)H, (int64_t)W};
    chamfer_topk_l1_nd_sweep<K>(seed_image, shape, out_labels, out_dists,
                                  pool, n_threads);
}

template <int K>
inline void chamfer_topk_pair_counts(
    const int32_t* seed_image, int32_t H, int32_t W,
    int32_t N, int32_t max_margin, int64_t* pair_count,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    std::vector<int64_t> shape{(int64_t)H, (int64_t)W};
    chamfer_topk_pair_counts_nd<K>(seed_image, shape, N, max_margin,
                                     pair_count, pool, n_threads);
}

template <int K>
inline void aggregate_topk_pairs(
    const int32_t* labels, const int32_t* dists,
    int64_t M, int32_t N, int32_t max_margin,
    int64_t* pair_count)
{
    aggregate_topk_pairs_nd<K>(labels, dists, M, N, max_margin, pair_count);
}

}  // namespace ncolor_cpp
