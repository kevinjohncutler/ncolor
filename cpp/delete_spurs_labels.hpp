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
#include <atomic>
#include <cstdint>
#include <mutex>
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

// BFS-style boundary-only despur. Same semantics as the full-scan
// variant below, but iterates only over the dynamic frontier of
// "pixels that might be spurs". Iter 0 is a full-image scan that
// finds all initial spurs and seeds the frontier for iter 1+; each
// subsequent iter processes only same-label neighbours of pixels
// removed in the previous iter. For typical microscopy seg's where
// only ~5% of pixels are at cell boundaries and despur cascades are
// shallow, this is 3-5× faster than full-scan iteration.
//
// Why iter 0 is still a full scan: the initial spur set IS the
// boundary set (every spur is a boundary pixel), and we have no
// cheaper way to identify boundary pixels than scanning everything
// once. After iter 0, we know which pixels could possibly be new
// spurs (only those adjacent to just-removed pixels) and skip the
// interior entirely.
template <typename T>
inline int64_t delete_spurs_labels_nd_bfs_inplace(
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

    auto coords_of = [&](int64_t i, std::vector<int64_t>& out) {
        for (int d = 0; d < ndim; ++d) {
            out[d] = i / strides[d];
            i -= out[d] * strides[d];
        }
    };

    // -- Iter 0: full-image scan, find all initial spurs, seed
    // frontier from their neighbours.
    std::vector<int64_t> spurs;     // pixels to remove this iter
    std::vector<int64_t> frontier;  // candidates for next iter
    spurs.reserve(total / 64);
    frontier.reserve(total / 32);

    // Iter 0 worker: scan a flat-index range, append spurs to a
    // thread-local list.
    auto worker = [&](int64_t i_lo, int64_t i_hi, std::vector<int64_t>& out) {
        std::vector<int64_t> c(ndim, 0);
        coords_of(i_lo, c);
        for (int64_t i = i_lo; i < i_hi; ++i) {
            const T lab = labels[i];
            if (lab != 0) {
                int same = 0;
                for (int d = 0; d < ndim; ++d) {
                    if (c[d] > 0 && labels[i - strides[d]] == lab) ++same;
                    if (c[d] + 1 < shape[d] && labels[i + strides[d]] == lab) ++same;
                }
                if (same <= threshold) out.push_back(i);
            }
            int d = ndim - 1;
            ++c[d];
            while (d > 0 && c[d] >= shape[d]) {
                c[d] = 0;
                --d;
                ++c[d];
            }
        }
    };

    const int nt = (pool && n_threads > 1) ? n_threads : 1;
    if (nt > 1 && total >= 1024 && ndim >= 1) {
        const int64_t outer = shape[0];
        const int64_t slab_size = strides[0];
        std::atomic<int64_t> next_slab{0};
        const int64_t chunk = std::max<int64_t>(1, outer / (nt * 4));
        std::mutex spurs_mtx;
        pool->parallel([&]() {
            std::vector<int64_t> local;
            local.reserve(1024);
            while (true) {
                int64_t s_lo = next_slab.fetch_add(chunk);
                if (s_lo >= outer) break;
                int64_t s_hi = std::min(outer, s_lo + chunk);
                worker(s_lo * slab_size, s_hi * slab_size, local);
            }
            if (!local.empty()) {
                std::lock_guard<std::mutex> lk(spurs_mtx);
                spurs.insert(spurs.end(), local.begin(), local.end());
            }
        });
    } else {
        worker(0, total, spurs);
    }

    int64_t total_removed = (int64_t)spurs.size();
    if (spurs.empty()) return 0;

    // Apply iter 0 removals + build frontier.
    std::vector<int64_t> coords_tmp(ndim, 0);
    for (int64_t i : spurs) {
        labels[i] = 0;
    }
    // Build frontier: same-label neighbours of removed pixels. The
    // labels are now 0 for removed pixels; this means "same-label
    // before removal" must be derived from spurs list. Instead, we
    // simply add every nonzero neighbour of every removed pixel —
    // these may or may not have lost a same-label neighbour, but
    // they're the only candidates that could newly become spurs.
    std::vector<uint8_t> in_frontier(total, 0);
    for (int64_t i : spurs) {
        coords_of(i, coords_tmp);
        for (int d = 0; d < ndim; ++d) {
            if (coords_tmp[d] > 0) {
                int64_t j = i - strides[d];
                if (labels[j] != 0 && !in_frontier[j]) {
                    in_frontier[j] = 1;
                    frontier.push_back(j);
                }
            }
            if (coords_tmp[d] + 1 < shape[d]) {
                int64_t j = i + strides[d];
                if (labels[j] != 0 && !in_frontier[j]) {
                    in_frontier[j] = 1;
                    frontier.push_back(j);
                }
            }
        }
    }

    // -- Iter 1+: BFS on frontier.
    for (int iter = 1; iter < max_iters; ++iter) {
        spurs.clear();
        for (int64_t i : frontier) {
            in_frontier[i] = 0;  // reset for next iter's dedupe
            const T lab = labels[i];
            if (lab == 0) continue;
            coords_of(i, coords_tmp);
            int same = 0;
            for (int d = 0; d < ndim; ++d) {
                if (coords_tmp[d] > 0 && labels[i - strides[d]] == lab) ++same;
                if (coords_tmp[d] + 1 < shape[d] && labels[i + strides[d]] == lab) ++same;
            }
            if (same <= threshold) spurs.push_back(i);
        }
        if (spurs.empty()) break;

        // Apply removals; build next frontier.
        std::vector<int64_t> next_frontier;
        next_frontier.reserve(spurs.size() * 4);
        for (int64_t i : spurs) labels[i] = 0;
        for (int64_t i : spurs) {
            coords_of(i, coords_tmp);
            for (int d = 0; d < ndim; ++d) {
                if (coords_tmp[d] > 0) {
                    int64_t j = i - strides[d];
                    if (labels[j] != 0 && !in_frontier[j]) {
                        in_frontier[j] = 1;
                        next_frontier.push_back(j);
                    }
                }
                if (coords_tmp[d] + 1 < shape[d]) {
                    int64_t j = i + strides[d];
                    if (labels[j] != 0 && !in_frontier[j]) {
                        in_frontier[j] = 1;
                        next_frontier.push_back(j);
                    }
                }
            }
        }
        total_removed += (int64_t)spurs.size();
        std::swap(frontier, next_frontier);
        if (frontier.empty()) break;
    }
    return total_removed;
}

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
