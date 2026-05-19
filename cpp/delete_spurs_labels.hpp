// Label-aware ND despur. Iteratively remove pixels whose count of
// face-adjacent same-label neighbours is ≤ `threshold`. Unlike the
// existing `delete_spurs.hpp` (which is binary — any nonzero pixel
// treated as foreground), this version respects label boundaries:
// a pixel of cell A whose only face-neighbour from cell A is one
// other pixel (the rest are other labels or bg) counts as a spur.
//
// Use case: after `expand_labels` fills the gaps between adjacent
// cells, two cells that originally touched only at a single point
// may now share a 1-pixel-wide "convergence pixel" that's only
// 1-same-label-neighbour-attached to its parent cell. Removing
// these widens the inter-cell gap, breaking K_5-creating contact
// patterns (5 cells meeting at a corner) so that 4-colouring
// becomes feasible at conn=1 r=1.
//
// Operates in place on the label buffer; removed pixels become 0
// (background).
//
// Parallelism: outer-axis-0 slabs are dispatched to the pool when
// available; threads scan their own slab independently and write to
// a shared `mark` buffer (no race — each pixel written by exactly
// one thread).

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "threadpool.h"

namespace ncolor_cpp {

namespace despur_detail {

template <typename T>
inline void count_and_mark_2d(
    const T* labels, uint8_t* mark, int64_t H, int64_t W,
    int threshold, int64_t y_lo, int64_t y_hi)
{
    for (int64_t y = y_lo; y < y_hi; ++y) {
        for (int64_t x = 0; x < W; ++x) {
            const int64_t i = y * W + x;
            const T lab = labels[i];
            if (lab == 0) continue;
            int same = 0;
            if (y > 0 && labels[i - W] == lab) ++same;
            if (y + 1 < H && labels[i + W] == lab) ++same;
            if (x > 0 && labels[i - 1] == lab) ++same;
            if (x + 1 < W && labels[i + 1] == lab) ++same;
            if (same <= threshold) mark[i] = 1;
        }
    }
}

// ND fallback: per-pixel coord recomputation. Slower than 2D fast path
// but works for any ndim.
template <typename T>
inline void count_and_mark_nd(
    const T* labels, uint8_t* mark,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    int threshold, int64_t i_lo, int64_t i_hi)
{
    const int ndim = (int)shape.size();
    std::vector<int64_t> coords(ndim, 0);
    // Initialise coords from i_lo (row-major).
    {
        int64_t rem = i_lo;
        for (int d = 0; d < ndim; ++d) {
            coords[d] = rem / strides[d];
            rem -= coords[d] * strides[d];
        }
    }
    for (int64_t i = i_lo; i < i_hi; ++i) {
        const T lab = labels[i];
        if (lab != 0) {
            int same = 0;
            for (int d = 0; d < ndim; ++d) {
                if (coords[d] > 0 && labels[i - strides[d]] == lab) ++same;
                if (coords[d] + 1 < shape[d] && labels[i + strides[d]] == lab) ++same;
            }
            if (same <= threshold) mark[i] = 1;
        }
        // Increment coords (row-major).
        int d = ndim - 1;
        ++coords[d];
        while (d > 0 && coords[d] >= shape[d]) {
            coords[d] = 0;
            --d;
            ++coords[d];
        }
    }
}

}  // namespace despur_detail

// Public ND entry point. Returns total pixels removed.
template <typename T>
inline int64_t delete_spurs_labels_nd_inplace(
    T* labels,
    const std::vector<int64_t>& shape,
    int threshold = 1,
    int max_iters = 20,
    ForkJoinPool* pool = nullptr,
    int n_threads = 1)
{
    const int ndim = (int)shape.size();
    if (ndim < 1) return 0;
    int64_t total = 1;
    for (auto s : shape) total *= s;
    if (total == 0) return 0;

    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

    int64_t total_removed = 0;
    std::vector<uint8_t> mark(total, 0);

    // 2D fast path detection.
    const bool fast_2d = (ndim == 2);
    const int64_t H = (ndim >= 1 ? shape[0] : 1);
    const int64_t W = (ndim >= 2 ? shape[1] : 1);

    for (int iter = 0; iter < max_iters; ++iter) {
        std::fill(mark.begin(), mark.end(), 0);

        if (fast_2d) {
            // Parallel by row chunks (work-stealing via atomic counter).
            const int nt = (pool && n_threads > 1) ? n_threads : 1;
            if (nt > 1) {
                std::atomic<int64_t> next_row{0};
                const int64_t chunk = std::max<int64_t>(1, H / (nt * 4));
                pool->parallel([&]() {
                    while (true) {
                        int64_t y_lo = next_row.fetch_add(chunk);
                        if (y_lo >= H) break;
                        int64_t y_hi = std::min(H, y_lo + chunk);
                        despur_detail::count_and_mark_2d<T>(
                            labels, mark.data(), H, W, threshold, y_lo, y_hi);
                    }
                });
            } else {
                despur_detail::count_and_mark_2d<T>(
                    labels, mark.data(), H, W, threshold, 0, H);
            }
        } else {
            // ND path: parallel by outer-axis-0 slabs.
            const int64_t outer = shape[0];
            const int64_t slab_size = strides[0];
            const int nt = (pool && n_threads > 1) ? n_threads : 1;
            if (nt > 1) {
                std::atomic<int64_t> next_slab{0};
                const int64_t chunk_slabs = std::max<int64_t>(1, outer / (nt * 4));
                pool->parallel([&]() {
                    while (true) {
                        int64_t s_lo = next_slab.fetch_add(chunk_slabs);
                        if (s_lo >= outer) break;
                        int64_t s_hi = std::min(outer, s_lo + chunk_slabs);
                        despur_detail::count_and_mark_nd<T>(
                            labels, mark.data(), shape, strides, threshold,
                            s_lo * slab_size, s_hi * slab_size);
                    }
                });
            } else {
                despur_detail::count_and_mark_nd<T>(
                    labels, mark.data(), shape, strides, threshold, 0, total);
            }
        }

        // Apply removals.
        int64_t n_removed = 0;
        for (int64_t i = 0; i < total; ++i) {
            if (mark[i]) { labels[i] = 0; ++n_removed; }
        }
        if (n_removed == 0) break;
        total_removed += n_removed;
    }
    return total_removed;
}

}  // namespace ncolor_cpp
