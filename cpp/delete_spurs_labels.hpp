// Label-aware ND despur. Iteratively remove pixels whose count of
// face-adjacent same-label neighbors is ≤ `threshold`. Unlike the
// existing `delete_spurs.hpp` (which is binary — any nonzero pixel
// treated as foreground), this version respects label boundaries:
// a pixel of cell A whose only face-neighbor from cell A is one
// other pixel (the rest are other labels or bg) counts as a spur.
//
// Use case: after `expand_labels` fills the gaps between adjacent
// cells, two cells that originally touched only at a single point
// may now share a 1-pixel-wide "convergence pixel" that's only
// 1-same-label-neighbor-attached to its parent cell. Removing
// these widens the inter-cell gap, breaking K_5-creating contact
// patterns (5 cells meeting at a corner) so that 4-coloring
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
#include <vector>

#include "threadpool.h"

namespace ncolor_cpp {


namespace despur_detail {

struct NDOffset {
    std::vector<int8_t> dc;   // length ndim, values in {-1, 0, +1}
    int64_t flat_offset;      // sum dc[d] * strides[d]
    bool is_face;             // exactly one nonzero coord
    int opposite_idx;         // index of -dc in the same table
};

// A pixel is marked for removal when it's a "spur" (≤ threshold
// same-label face-neighbors) OR optionally a "thin-line interior":
// exactly 2 same-label 8-connectivity neighbors AND those two are
// on opposite sides of the pixel (N-S, W-E, NW-SE, or NE-SW). With
// remove_thin = true a 1-px-wide straight line gets wiped wholesale
// in a single pass instead of needing N/2 iterations to peel from
// both ends. The thin check is 2D-only here; the ND helper below
// supports the axis-aligned subset (no diagonals) for 3D+.
inline std::vector<NDOffset> build_nd_offsets(
    int ndim, const std::vector<int64_t>& strides)
{
    std::vector<NDOffset> out;
    std::vector<int8_t> dc(ndim, -1);
    while (true) {
        int n_nz = 0;
        bool any_nz = false;
        int64_t flat = 0;
        for (int d = 0; d < ndim; ++d) {
            if (dc[d] != 0) { ++n_nz; any_nz = true; }
            flat += (int64_t)dc[d] * strides[d];
        }
        if (any_nz) {
            NDOffset o;
            o.dc.assign(dc.begin(), dc.end());
            o.flat_offset = flat;
            o.is_face = (n_nz == 1);
            o.opposite_idx = -1;
            out.push_back(std::move(o));
        }
        // increment dc as a base-3 odometer over {-1,0,+1}
        int d = ndim - 1;
        while (d >= 0) {
            ++dc[d];
            if (dc[d] <= 1) break;
            dc[d] = -1;
            --d;
        }
        if (d < 0) break;
    }
    // Fill opposite_idx by linear scan.
    for (size_t k = 0; k < out.size(); ++k) {
        for (size_t j = 0; j < out.size(); ++j) {
            bool opp = true;
            for (int d = 0; d < ndim; ++d) {
                if (out[k].dc[d] != -out[j].dc[d]) { opp = false; break; }
            }
            if (opp) { out[k].opposite_idx = (int)j; break; }
        }
    }
    return out;
}

// The one marking kernel, for every dimension. With remove_thin = true
// this catches any 1-voxel-wide straight segment, whether axis-aligned
// or diagonal: a pixel is marked iff its same-label 8-connectivity
// neighbours count exactly two AND they sit at opposite offsets
// (their displacement vectors negate). A 2D specialization used to sit
// beside it with this as its parity reference; it was the only
// dimension-specific path in the package and is gone.
template <typename T>
inline void count_and_mark_nd(
    const T* labels, uint8_t* mark,
    const std::vector<int64_t>& shape,
    const std::vector<int64_t>& strides,
    const std::vector<NDOffset>& offsets,
    int threshold, int64_t i_lo, int64_t i_hi,
    bool remove_thin = false)
{
    const int ndim = (int)shape.size();
    std::vector<int64_t> coords(ndim, 0);
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
            int face_count = 0;
            int total_count = 0;
            int first_idx = -1, second_idx = -1;
            bool over_two = false;
            for (size_t k = 0; k < offsets.size(); ++k) {
                const auto& o = offsets[k];
                bool in_bounds = true;
                for (int d = 0; d < ndim; ++d) {
                    if (o.dc[d] == 0) continue;
                    int64_t nc = coords[d] + o.dc[d];
                    if (nc < 0 || nc >= shape[d]) {
                        in_bounds = false; break;
                    }
                }
                if (!in_bounds) continue;
                if (labels[i + o.flat_offset] != lab) continue;
                ++total_count;
                if (o.is_face) ++face_count;
                if (first_idx < 0) first_idx = (int)k;
                else if (second_idx < 0) second_idx = (int)k;
                else { over_two = true; }
            }
            if (face_count <= threshold) {
                mark[i] = 1;
            } else if (remove_thin && !over_two
                       && total_count == 2 && second_idx >= 0
                       && offsets[first_idx].opposite_idx == second_idx) {
                mark[i] = 1;
            }
        }
        // increment coords (row-major)
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

// Iter 0 is a parallel mark-bitmap scan over the full image; once we
// know which pixels were zeroed, iter 1+ only re-examines their
// face-neighbors (the small set of pixels that *could* have become
// new spurs by losing a same-label connection). Iter-0 cost matches
// a pure full-scan; iter 1+ cost stays flat as max_iters grows since
// the frontier shrinks fast.
//
// Bench on macOS M1 Ultra (May 2026), 2000² microscopy seg:
//   max_iters=1:  3.78ms  (matches a plain full-scan)
//   max_iters=2:  5.44ms  (~10% faster than full-scan)
//   max_iters=20: 5.99ms  (7.7× faster than full-scan = 46ms)
template <typename T>
inline int64_t delete_spurs_labels_nd_inplace(
    T* labels,
    const std::vector<int64_t>& shape,
    int threshold = 1,
    int max_iters = 20,
    ForkJoinPool* pool = nullptr,
    int n_threads = 1,
    bool remove_thin = false)
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

    const int nt = (pool && n_threads > 1) ? n_threads : 1;

    // ND offsets table (used by count_and_mark_nd). Built once per call.
    std::vector<despur_detail::NDOffset> nd_offsets =
        despur_detail::build_nd_offsets(ndim, strides);

    // -- Iter 0: mark spurs in a shared bitmap via parallel slab
    // dispatch, then zero them out. ``remove_thin`` also kills
    // 1-voxel-wide straight interior pixels in the same pass.
    std::vector<uint8_t> mark(total, 0);
    {
        const int64_t outer = shape[0];
        const int64_t slab_size = strides[0];
        if (nt > 1) {
            std::atomic<int64_t> next_slab{0};
            const int64_t chunk_slabs = std::max<int64_t>(1, outer / (nt * 4));
            pool->parallel([&]() {
                while (true) {
                    int64_t s_lo = next_slab.fetch_add(chunk_slabs);
                    if (s_lo >= outer) break;
                    int64_t s_hi = std::min(outer, s_lo + chunk_slabs);
                    despur_detail::count_and_mark_nd<T>(
                        labels, mark.data(), shape, strides, nd_offsets,
                        threshold,
                        s_lo * slab_size, s_hi * slab_size, remove_thin);
                }
            });
        } else {
            despur_detail::count_and_mark_nd<T>(
                labels, mark.data(), shape, strides, nd_offsets,
                threshold, 0, total, remove_thin);
        }
    }

    // Apply iter 0 removals. Sequential is fine — pure linear write.
    int64_t total_removed = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (mark[i]) { labels[i] = 0; ++total_removed; }
    }
    if (total_removed == 0) return 0;
    if (max_iters <= 1) return total_removed;

    // -- Collect spurs list from the mark bitmap (parallel scan over
    // mark[], one pass). Each thread emits indices where mark==1 into
    // a local vector; we concat at the end. Spurs are the seeds for
    // the iter-1 frontier.
    std::vector<std::vector<int64_t>> per_thread_spurs(nt);
    if (nt > 1) {
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_slab{0};
        const int64_t outer = shape[0];
        const int64_t slab_size = strides[0];
        const int64_t chunk_slabs = std::max<int64_t>(1, outer / (nt * 4));
        pool->parallel([&]() {
            const int tid = tid_counter.fetch_add(1);
            if (tid >= nt) return;
            auto& local = per_thread_spurs[tid];
            local.reserve(64);
            while (true) {
                int64_t s_lo = next_slab.fetch_add(chunk_slabs);
                if (s_lo >= outer) break;
                int64_t s_hi = std::min(outer, s_lo + chunk_slabs);
                int64_t i_lo = s_lo * slab_size;
                int64_t i_hi = s_hi * slab_size;
                for (int64_t i = i_lo; i < i_hi; ++i) {
                    if (mark[i]) local.push_back(i);
                }
            }
        });
    } else {
        per_thread_spurs[0].reserve(64);
        for (int64_t i = 0; i < total; ++i) {
            if (mark[i]) per_thread_spurs[0].push_back(i);
        }
    }
    std::vector<int64_t> spurs;
    {
        size_t n = 0;
        for (auto& v : per_thread_spurs) n += v.size();
        spurs.reserve(n);
        for (auto& v : per_thread_spurs)
            spurs.insert(spurs.end(), v.begin(), v.end());
    }

    // -- Build iter-1 frontier: face-neighbors of every spur whose
    // label is still non-zero. We walk the (small) spurs list in
    // parallel and emit candidates to per-thread vectors; the final
    // concatenated list is sort+unique'd to dedup.
    std::vector<std::vector<int64_t>> per_thread_nbrs(nt);
    if (nt > 1 && spurs.size() >= 256) {
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_chunk{0};
        const int64_t chunk = std::max<int64_t>(
            64, (int64_t)spurs.size() / (nt * 4));
        pool->parallel([&]() {
            const int tid = tid_counter.fetch_add(1);
            if (tid >= nt) return;
            auto& local = per_thread_nbrs[tid];
            local.reserve(spurs.size() * 4 / nt + 64);
            std::vector<int64_t> c(ndim, 0);
            while (true) {
                int64_t lo = next_chunk.fetch_add(chunk);
                if (lo >= (int64_t)spurs.size()) break;
                int64_t hi = std::min((int64_t)spurs.size(), lo + chunk);
                for (int64_t k = lo; k < hi; ++k) {
                    int64_t i = spurs[k];
                    coords_of(i, c);
                    for (int d = 0; d < ndim; ++d) {
                        if (c[d] > 0) {
                            int64_t j = i - strides[d];
                            if (labels[j] != 0) local.push_back(j);
                        }
                        if (c[d] + 1 < shape[d]) {
                            int64_t j = i + strides[d];
                            if (labels[j] != 0) local.push_back(j);
                        }
                    }
                }
            }
        });
    } else {
        per_thread_nbrs[0].reserve(spurs.size() * 4);
        std::vector<int64_t> c(ndim, 0);
        for (int64_t i : spurs) {
            coords_of(i, c);
            for (int d = 0; d < ndim; ++d) {
                if (c[d] > 0) {
                    int64_t j = i - strides[d];
                    if (labels[j] != 0) per_thread_nbrs[0].push_back(j);
                }
                if (c[d] + 1 < shape[d]) {
                    int64_t j = i + strides[d];
                    if (labels[j] != 0) per_thread_nbrs[0].push_back(j);
                }
            }
        }
    }
    std::vector<int64_t> frontier;
    {
        size_t n = 0;
        for (auto& v : per_thread_nbrs) n += v.size();
        frontier.reserve(n);
        for (auto& v : per_thread_nbrs)
            frontier.insert(frontier.end(), v.begin(), v.end());
    }
    // Dedup (a pixel can be a face-neighbor of multiple spurs).
    std::sort(frontier.begin(), frontier.end());
    frontier.erase(std::unique(frontier.begin(), frontier.end()),
                    frontier.end());

    if (frontier.empty()) return total_removed;

    // -- Iter 1+: BFS on the shrinking frontier. Reuse mark as the
    // "in-frontier" set across iters: a pixel is in this iter's
    // frontier iff mark[i] == 1. Reset mark[i] = 0 once processed.
    // (Old mark==1 entries from iter 0 are about to be overwritten —
    // first reset them, then mark the new frontier.)
    std::fill(mark.begin(), mark.end(), 0);
    for (int64_t i : frontier) mark[i] = 1;

    std::vector<int64_t> coords_tmp(ndim, 0);
    std::vector<int64_t> next_frontier;
    std::vector<int64_t> just_removed;
    for (int iter = 1; iter < max_iters; ++iter) {
        just_removed.clear();
        for (int64_t i : frontier) {
            mark[i] = 0;
            const T lab = labels[i];
            if (lab == 0) continue;
            coords_of(i, coords_tmp);
            int same = 0;
            for (int d = 0; d < ndim; ++d) {
                if (coords_tmp[d] > 0 && labels[i - strides[d]] == lab) ++same;
                if (coords_tmp[d] + 1 < shape[d]
                    && labels[i + strides[d]] == lab) ++same;
            }
            if (same <= threshold) just_removed.push_back(i);
        }
        if (just_removed.empty()) break;

        for (int64_t i : just_removed) labels[i] = 0;
        next_frontier.clear();
        for (int64_t i : just_removed) {
            coords_of(i, coords_tmp);
            for (int d = 0; d < ndim; ++d) {
                if (coords_tmp[d] > 0) {
                    int64_t j = i - strides[d];
                    if (labels[j] != 0 && !mark[j]) {
                        mark[j] = 1;
                        next_frontier.push_back(j);
                    }
                }
                if (coords_tmp[d] + 1 < shape[d]) {
                    int64_t j = i + strides[d];
                    if (labels[j] != 0 && !mark[j]) {
                        mark[j] = 1;
                        next_frontier.push_back(j);
                    }
                }
            }
        }
        total_removed += (int64_t)just_removed.size();
        std::swap(frontier, next_frontier);
        if (frontier.empty()) break;
    }
    return total_removed;
}

}  // namespace ncolor_cpp
