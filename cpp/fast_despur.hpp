// Fast despur via pre-computed face-count.
//
// The classic ``delete_spurs_labels`` does its iter-0 work as a full-
// image scan that, for each pixel, counts same-label face-neighbours
// and marks spurs. On a 2000×2000 MM image that scan alone costs
// ~4 ms — the dominant term of the whole despur (subsequent iters
// are queue-based and cheap).
//
// This file separates that scan from the despur logic:
//   1. ``compute_face_count_nd``: tight, parallel, branchless full-image
//      scan that writes the per-pixel same-label face-neighbour count
//      into a uint8 array. Targets ~1 ms on MM.
//   2. ``despur_via_face_count_nd``: drives spur removal from a pre-
//      computed face_count array. Initial spurs (count ≤ threshold) are
//      reverted; the queue-based peel-back decrements neighbours' counts
//      and re-checks until quiescent. No full-image rescans.
//
// The face_count array is also useful downstream (it tells us which
// pixels border another cell), so producing it once and consuming it
// twice is a deliberate design.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

#include "threadpool.h"

namespace ncolor_cpp {

// Branchless face-count for an ND label image. ``count[i]`` is set to
// the number of same-label face-neighbours pixel i has (0 if labels[i]
// is bg). Each axis is checked in order; same-label face-pairs simply
// produce coincident increments on each pixel.
//
// 2D fast path: hand-written; ND fallback uses strides.
template <typename T>
inline void compute_face_count_nd(
    const T* labels, uint8_t* count,
    const std::vector<int64_t>& shape,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    const int ndim = (int)shape.size();
    if (ndim == 0) return;
    int64_t total = 1;
    for (auto s : shape) total *= s;
    if (total == 0) return;

    const int nt = (pool && n_threads > 1) ? n_threads : 1;

    if (ndim == 2) {
        const int64_t H = shape[0], W = shape[1];
        auto kernel = [&](int64_t y_lo, int64_t y_hi) {
            for (int64_t y = y_lo; y < y_hi; ++y) {
                const bool yL = (y > 0);
                const bool yH = (y + 1 < H);
                for (int64_t x = 0; x < W; ++x) {
                    const int64_t i = y * W + x;
                    const T lab = labels[i];
                    // Branchless: the bg check (lab != 0) and edge
                    // checks (yL/yH/xL/xH) are AND'd so an OOB or bg
                    // pixel always contributes 0.
                    const int hN = yL          && labels[i - W] == lab;
                    const int hS = yH          && labels[i + W] == lab;
                    const int hW = (x > 0)     && labels[i - 1] == lab;
                    const int hE = (x + 1 < W) && labels[i + 1] == lab;
                    const int c  = (lab != 0) ? (hN + hS + hW + hE) : 0;
                    count[i] = (uint8_t)c;
                }
            }
        };
        if (nt > 1 && H >= (int64_t)(nt * 4)) {
            std::atomic<int64_t> next_row{0};
            const int64_t chunk = std::max<int64_t>(1, H / (nt * 4));
            pool->parallel([&]() {
                while (true) {
                    int64_t y_lo = next_row.fetch_add(chunk);
                    if (y_lo >= H) break;
                    int64_t y_hi = std::min(H, y_lo + chunk);
                    kernel(y_lo, y_hi);
                }
            });
        } else {
            kernel(0, H);
        }
        return;
    }

    // ND fallback (any ndim ≥ 1). Strides for the 2*ndim face offsets.
    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

    auto kernel_nd = [&](int64_t i_lo, int64_t i_hi) {
        // Compute coords incrementally.
        int64_t c[8] = {0};
        int64_t rem = i_lo;
        for (int d = 0; d < ndim; ++d) {
            c[d] = rem / strides[d];
            rem -= c[d] * strides[d];
        }
        for (int64_t i = i_lo; i < i_hi; ++i) {
            const T lab = labels[i];
            int cnt = 0;
            if (lab != 0) {
                for (int d = 0; d < ndim; ++d) {
                    if (c[d] > 0          && labels[i - strides[d]] == lab) ++cnt;
                    if (c[d] + 1 < shape[d] && labels[i + strides[d]] == lab) ++cnt;
                }
            }
            count[i] = (uint8_t)cnt;
            // Increment coords.
            int d = ndim - 1;
            ++c[d];
            while (d > 0 && c[d] >= shape[d]) {
                c[d] = 0; --d; ++c[d];
            }
        }
    };

    if (nt > 1 && total >= 1024) {
        const int64_t outer = shape[0];
        const int64_t slab_size = strides[0];
        std::atomic<int64_t> next_slab{0};
        const int64_t chunk_slabs = std::max<int64_t>(1, outer / (nt * 4));
        pool->parallel([&]() {
            while (true) {
                int64_t s_lo = next_slab.fetch_add(chunk_slabs);
                if (s_lo >= outer) break;
                int64_t s_hi = std::min(outer, s_lo + chunk_slabs);
                kernel_nd(s_lo * slab_size, s_hi * slab_size);
            }
        });
    } else {
        kernel_nd(0, total);
    }
}


// Despur via a pre-computed face_count array. Returns the number of
// pixels removed (zeroed in labels).
//
//   • Initial pass: any pixel with face_count[i] ≤ threshold AND
//     labels[i] != 0 is a spur — revert (set to 0) and enqueue.
//   • Queue peel-back: for each reverted pixel, walk its face-neighbours.
//     If a neighbour's label matches what we just reverted, decrement
//     its face_count. If the neighbour's face_count now ≤ threshold,
//     it becomes a new spur — enqueue it.
//   • No full-image rescans.
template <typename T>
inline int64_t despur_via_face_count_nd(
    T* labels, uint8_t* face_count,
    const std::vector<int64_t>& shape,
    int threshold = 1,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    const int ndim = (int)shape.size();
    if (ndim == 0) return 0;
    int64_t total = 1;
    for (auto s : shape) total *= s;
    if (total == 0) return 0;

    const int nt = (pool && n_threads > 1) ? n_threads : 1;
    const int8_t th = (int8_t)threshold;

    // Build initial spur queue. Each entry stores (flat_idx, label_at_time_of_revert)
    // so that during peel-back we can check whether a neighbour's
    // face_count was contributed-to by THIS particular reverted pixel
    // (and decrement only then). Without the stored label we'd decrement
    // any labelled neighbour and over-revert by ~10× on real data.
    using QEnt = std::pair<int64_t, T>;
    std::vector<QEnt> queue;
    if (nt > 1 && total >= 1024) {
        const int64_t outer = shape[0];
        const int64_t slab_size = total / outer;
        std::vector<std::vector<QEnt>> per_thread(nt);
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_slab{0};
        const int64_t chunk_slabs = std::max<int64_t>(1, outer / (nt * 4));
        pool->parallel([&]() {
            int my_tid = tid_counter.fetch_add(1);
            if (my_tid >= nt) return;
            auto& local = per_thread[my_tid];
            while (true) {
                int64_t s_lo = next_slab.fetch_add(chunk_slabs);
                if (s_lo >= outer) break;
                int64_t s_hi = std::min(outer, s_lo + chunk_slabs);
                int64_t i_lo = s_lo * slab_size;
                int64_t i_hi = s_hi * slab_size;
                for (int64_t i = i_lo; i < i_hi; ++i) {
                    const T lab = labels[i];
                    if (lab != 0 && (int8_t)face_count[i] <= th) {
                        local.emplace_back(i, lab);
                    }
                }
            }
        });
        size_t sz = 0;
        for (auto& v : per_thread) sz += v.size();
        queue.reserve(sz);
        for (auto& v : per_thread) queue.insert(queue.end(), v.begin(), v.end());
    } else {
        for (int64_t i = 0; i < total; ++i) {
            const T lab = labels[i];
            if (lab != 0 && (int8_t)face_count[i] <= th) queue.emplace_back(i, lab);
        }
    }

    if (queue.empty()) return 0;

    // Face-offset table for ND traversal.
    std::vector<int64_t> strides(ndim);
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

    // For 2D, hot path:
    const int64_t H = (ndim >= 1 ? shape[0] : 1);
    const int64_t W = (ndim >= 2 ? shape[1] : 1);

    // First revert all initially-marked spurs.
    int64_t removed = 0;
    for (auto& [i, _lab] : queue) {
        if (labels[i] != 0) {
            labels[i] = 0;
            ++removed;
        }
    }

    // Queue peel-back. For each reverted pixel i (with stored label
    // ``old_lab``), decrement face_count only on those face-neighbours
    // j whose current label equals old_lab — i.e. only those that
    // actually had a same-label contribution from us. This is what
    // keeps the decrements correct.
    size_t head = 0;
    if (ndim == 2) {
        while (head < queue.size()) {
            const int64_t i = queue[head].first;
            const T old_lab  = queue[head].second;
            ++head;
            const int64_t y = i / W;
            const int64_t x = i - y * W;
            auto poke = [&](int64_t j) {
                if (labels[j] != old_lab) return;  // not same-label contribution
                uint8_t fc = face_count[j];
                if (fc > 0) {
                    fc = (uint8_t)(fc - 1);
                    face_count[j] = fc;
                }
                if ((int8_t)fc <= th) {
                    labels[j] = 0;
                    ++removed;
                    queue.emplace_back(j, old_lab);
                }
            };
            if (y > 0)     poke(i - W);
            if (y + 1 < H) poke(i + W);
            if (x > 0)     poke(i - 1);
            if (x + 1 < W) poke(i + 1);
        }
    } else {
        std::vector<int64_t> c(ndim, 0);
        while (head < queue.size()) {
            const int64_t i = queue[head].first;
            const T old_lab  = queue[head].second;
            ++head;
            int64_t rem = i;
            for (int d = 0; d < ndim; ++d) {
                c[d] = rem / strides[d];
                rem -= c[d] * strides[d];
            }
            for (int d = 0; d < ndim; ++d) {
                if (c[d] > 0) {
                    const int64_t j = i - strides[d];
                    if (labels[j] == old_lab) {
                        uint8_t fc = face_count[j];
                        if (fc > 0) { fc = (uint8_t)(fc - 1); face_count[j] = fc; }
                        if ((int8_t)fc <= th) {
                            labels[j] = 0; ++removed; queue.emplace_back(j, old_lab);
                        }
                    }
                }
                if (c[d] + 1 < shape[d]) {
                    const int64_t j = i + strides[d];
                    if (labels[j] == old_lab) {
                        uint8_t fc = face_count[j];
                        if (fc > 0) { fc = (uint8_t)(fc - 1); face_count[j] = fc; }
                        if ((int8_t)fc <= th) {
                            labels[j] = 0; ++removed; queue.emplace_back(j, old_lab);
                        }
                    }
                }
            }
        }
    }
    (void)pool;
    (void)n_threads;
    return removed;
}

}  // namespace ncolor_cpp
