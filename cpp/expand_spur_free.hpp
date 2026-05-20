// Spur-free expand: BFS dilation with connectivity check.
//
// Replaces the Voronoi-style expand_labels + post-hoc despur sequence
// with a single growth algorithm that REFUSES to claim bg pixels that
// would create spurs (≤1 same-label face-neighbour after assignment).
//
// Each round: each frontier bg pixel asks "which label do my face-
// neighbours mostly agree on?" — if at least `connectivity_threshold+1`
// face-neighbours share a single non-bg label L, the pixel is claimed
// for L. Otherwise it stays bg. Iterate until no more claims happen.
//
// The K_5 "starfish" failure mode (5 cells meeting at a corner under
// conn=1 r=2) is naturally avoided: the convergence pixel has 4 face-
// neighbours from 4 different cells, so no single label appears
// twice — the pixel never gets claimed, the cells never form K_5
// contact, and the resulting adjacency graph is 4-colourable.
//
// Differences vs the standard Voronoi expand:
//   * Some bg pixels remain bg (those where no label dominates the
//     neighbourhood) — visible as thin gaps between tightly-packed
//     cells. This is the algorithm's intent.
//   * For cells separated by wide enough gaps, identical to Voronoi.
//   * Connectivity-aware: cells only grow into pixels they have a
//     "claim" on (≥2 face-neighbours already).
//
// Complexity: O(N) initial frontier scan + O(total claimed pixels)
// across all rounds (each pixel handled once when claimed, and once
// per round it's at the frontier without being claimed).

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include "threadpool.h"

namespace ncolor_cpp {

namespace expand_sf_detail {

// Find the label among `nbrs[0..n-1]` (skipping 0) that appears the
// most times. Returns (best_label, best_count). For ties, returns the
// smaller label (deterministic).
template <typename T>
inline std::pair<T, int> dominant_label(const T* nbrs, int n) {
    T best_lab = 0;
    int best_cnt = 0;
    for (int a = 0; a < n; ++a) {
        const T la = nbrs[a];
        if (la == 0) continue;
        // Skip if we already scored this label.
        bool seen = false;
        for (int b = 0; b < a; ++b) {
            if (nbrs[b] == la) { seen = true; break; }
        }
        if (seen) continue;
        int cnt = 1;
        for (int b = a + 1; b < n; ++b) {
            if (nbrs[b] == la) ++cnt;
        }
        if (cnt > best_cnt || (cnt == best_cnt && la < best_lab)) {
            best_cnt = cnt;
            best_lab = la;
        }
    }
    return {best_lab, best_cnt};
}

}  // namespace expand_sf_detail

// 2D fast-path implementation. Returns number of bg pixels claimed.
//
// Round 0 is fused with the initial-frontier scan: we do a single
// full-image pass that BOTH finds the candidate pixels (bg with ≥1
// labelled face-neighbour) AND computes their round-0 claim in one
// go. For the common `max_rounds=1` case this is the only pass
// needed — no separate frontier-building step. For larger
// max_rounds, the BFS loop continues on round-0's claim-neighbours
// as before.
template <typename T>
inline int64_t expand_spur_free_2d_inplace(
    T* labels, int64_t H, int64_t W,
    int max_rounds = 100,
    int connectivity_threshold = 1,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    const int64_t total = H * W;
    if (total == 0) return 0;

    const int nt = (pool && n_threads > 1) ? n_threads : 1;

    // Round 0 = fused initial-scan + claim computation. Single pass
    // over the full image; for each bg pixel, count same-label face-
    // neighbours and emit a claim if the count exceeds the threshold.
    // No separate "is this pixel a frontier candidate?" pass needed.
    std::vector<std::pair<int64_t, T>> claims;
    auto scan_chunk_2d = [&](int64_t y_lo, int64_t y_hi,
                              std::vector<std::pair<int64_t, T>>& out) {
        for (int64_t y = y_lo; y < y_hi; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                const int64_t i = y * W + x;
                if (labels[i] != 0) continue;
                T nbrs[4];
                int n = 0;
                if (y > 0)     nbrs[n++] = labels[i - W];
                if (y + 1 < H) nbrs[n++] = labels[i + W];
                if (x > 0)     nbrs[n++] = labels[i - 1];
                if (x + 1 < W) nbrs[n++] = labels[i + 1];
                auto [lab, cnt] = expand_sf_detail::dominant_label(nbrs, n);
                if (cnt > connectivity_threshold) {
                    out.emplace_back(i, lab);
                }
            }
        }
    };

    if (nt > 1 && total >= 1024) {
        std::vector<std::vector<std::pair<int64_t, T>>> per_thread(nt);
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_row{0};
        const int64_t chunk = std::max<int64_t>(1, H / (nt * 4));
        pool->parallel([&]() {
            int my_tid = tid_counter.fetch_add(1);
            if (my_tid >= nt) return;
            auto& local = per_thread[my_tid];
            while (true) {
                int64_t y_lo = next_row.fetch_add(chunk);
                if (y_lo >= H) break;
                int64_t y_hi = std::min(H, y_lo + chunk);
                scan_chunk_2d(y_lo, y_hi, local);
            }
        });
        size_t sz = 0;
        for (auto& v : per_thread) sz += v.size();
        claims.reserve(sz);
        for (auto& v : per_thread) claims.insert(claims.end(), v.begin(), v.end());
    } else {
        scan_chunk_2d(0, H, claims);
    }

    if (claims.empty()) return 0;

    // Apply round-0 claims.
    if (nt > 1 && claims.size() >= 2048) {
        std::atomic<size_t> idx{0};
        const size_t cs = 1024;
        pool->parallel([&]() {
            while (true) {
                size_t lo = idx.fetch_add(cs);
                if (lo >= claims.size()) break;
                size_t hi = std::min(claims.size(), lo + cs);
                for (size_t k = lo; k < hi; ++k) {
                    labels[claims[k].first] = claims[k].second;
                }
            }
        });
    } else {
        for (auto& [i, lab] : claims) labels[i] = lab;
    }

    int64_t total_claimed = (int64_t)claims.size();

    // For max_rounds == 1 (the default) we're done — no further
    // frontier work needed.
    if (max_rounds <= 1) return total_claimed;

    // Build next_frontier from round-0 claims' bg neighbours.
    std::vector<int64_t> frontier;
    if (nt > 1 && claims.size() >= 1024) {
        std::vector<std::vector<int64_t>> per_thread_next(nt);
        std::atomic<int> tid_c{0};
        std::atomic<size_t> idx_c{0};
        const size_t chunk_c = std::max<size_t>(64, claims.size() / (nt * 8));
        pool->parallel([&]() {
            int my_tid = tid_c.fetch_add(1);
            if (my_tid >= nt) return;
            auto& local = per_thread_next[my_tid];
            while (true) {
                size_t lo = idx_c.fetch_add(chunk_c);
                if (lo >= claims.size()) break;
                size_t hi = std::min(claims.size(), lo + chunk_c);
                for (size_t k = lo; k < hi; ++k) {
                    const int64_t i = claims[k].first;
                    const int64_t y = i / W;
                    const int64_t x = i - y * W;
                    if (y > 0     && labels[i - W] == 0) local.push_back(i - W);
                    if (y + 1 < H && labels[i + W] == 0) local.push_back(i + W);
                    if (x > 0     && labels[i - 1] == 0) local.push_back(i - 1);
                    if (x + 1 < W && labels[i + 1] == 0) local.push_back(i + 1);
                }
            }
        });
        size_t sz = 0;
        for (auto& v : per_thread_next) sz += v.size();
        frontier.reserve(sz);
        for (auto& v : per_thread_next) frontier.insert(frontier.end(), v.begin(), v.end());
    } else {
        for (auto& [i, lab] : claims) {
            (void)lab;
            const int64_t y = i / W;
            const int64_t x = i - y * W;
            if (y > 0     && labels[i - W] == 0) frontier.push_back(i - W);
            if (y + 1 < H && labels[i + W] == 0) frontier.push_back(i + W);
            if (x > 0     && labels[i - 1] == 0) frontier.push_back(i - 1);
            if (x + 1 < W && labels[i + 1] == 0) frontier.push_back(i + 1);
        }
    }
    std::sort(frontier.begin(), frontier.end());
    frontier.erase(std::unique(frontier.begin(), frontier.end()), frontier.end());

    // 2. Iterative growth (parallelised within each round).
    std::vector<int64_t> next_frontier;
    for (int round = 1; round < max_rounds; ++round) {
        if (frontier.empty()) break;

        claims.clear();

        if (nt > 1 && frontier.size() >= 1024) {
            std::vector<std::vector<std::pair<int64_t, T>>> per_thread(nt);
            std::atomic<int> tid_counter{0};
            std::atomic<size_t> next_chunk{0};
            const size_t chunk = std::max<size_t>(256, frontier.size() / (nt * 8));
            pool->parallel([&]() {
                int my_tid = tid_counter.fetch_add(1);
                if (my_tid >= nt) return;
                auto& local = per_thread[my_tid];
                while (true) {
                    size_t lo = next_chunk.fetch_add(chunk);
                    if (lo >= frontier.size()) break;
                    size_t hi = std::min(frontier.size(), lo + chunk);
                    for (size_t k = lo; k < hi; ++k) {
                        const int64_t i = frontier[k];
                        if (labels[i] != 0) continue;
                        const int64_t y = i / W;
                        const int64_t x = i - y * W;
                        T nbrs[4];
                        int n = 0;
                        if (y > 0)     nbrs[n++] = labels[i - W];
                        if (y + 1 < H) nbrs[n++] = labels[i + W];
                        if (x > 0)     nbrs[n++] = labels[i - 1];
                        if (x + 1 < W) nbrs[n++] = labels[i + 1];
                        auto [lab, cnt] = expand_sf_detail::dominant_label(nbrs, n);
                        if (cnt > connectivity_threshold) {
                            local.emplace_back(i, lab);
                        }
                    }
                }
            });
            for (auto& v : per_thread) claims.insert(claims.end(), v.begin(), v.end());
        } else {
            claims.reserve(frontier.size());
            for (int64_t i : frontier) {
                if (labels[i] != 0) continue;
                const int64_t y = i / W;
                const int64_t x = i - y * W;
                T nbrs[4];
                int n = 0;
                if (y > 0)     nbrs[n++] = labels[i - W];
                if (y + 1 < H) nbrs[n++] = labels[i + W];
                if (x > 0)     nbrs[n++] = labels[i - 1];
                if (x + 1 < W) nbrs[n++] = labels[i + 1];
                auto [lab, cnt] = expand_sf_detail::dominant_label(nbrs, n);
                if (cnt > connectivity_threshold) {
                    claims.emplace_back(i, lab);
                }
            }
        }

        if (claims.empty()) {
            break;
        }

        // Apply claims (each pixel written once, no race).
        if (nt > 1 && claims.size() >= 2048) {
            std::atomic<size_t> idx{0};
            const size_t cs = 1024;
            pool->parallel([&]() {
                while (true) {
                    size_t lo = idx.fetch_add(cs);
                    if (lo >= claims.size()) break;
                    size_t hi = std::min(claims.size(), lo + cs);
                    for (size_t k = lo; k < hi; ++k) {
                        labels[claims[k].first] = claims[k].second;
                    }
                }
            });
        } else {
            for (auto& [i, lab] : claims) labels[i] = lab;
        }

        // next_frontier = { bg face-neighbours of pixels just claimed }.
        // An old-frontier pixel that wasn't claimed and has no newly-
        // claimed neighbour can't become claimable later, so it's safe
        // to drop. Halves the work and removes the dedup-by-bitmap
        // requirement (duplicates are fine; we sort+unique at the end).
        next_frontier.clear();
        if (nt > 1 && claims.size() >= 1024) {
            std::vector<std::vector<int64_t>> per_thread_next(nt);
            std::atomic<int> tid_c{0};
            std::atomic<size_t> idx_c{0};
            const size_t chunk_c = std::max<size_t>(64, claims.size() / (nt * 8));
            pool->parallel([&]() {
                int my_tid = tid_c.fetch_add(1);
                if (my_tid >= nt) return;
                auto& local = per_thread_next[my_tid];
                while (true) {
                    size_t lo = idx_c.fetch_add(chunk_c);
                    if (lo >= claims.size()) break;
                    size_t hi = std::min(claims.size(), lo + chunk_c);
                    for (size_t k = lo; k < hi; ++k) {
                        const int64_t i = claims[k].first;
                        const int64_t y = i / W;
                        const int64_t x = i - y * W;
                        if (y > 0     && labels[i - W] == 0) local.push_back(i - W);
                        if (y + 1 < H && labels[i + W] == 0) local.push_back(i + W);
                        if (x > 0     && labels[i - 1] == 0) local.push_back(i - 1);
                        if (x + 1 < W && labels[i + 1] == 0) local.push_back(i + 1);
                    }
                }
            });
            size_t total_size = 0;
            for (auto& v : per_thread_next) total_size += v.size();
            next_frontier.reserve(total_size);
            for (auto& v : per_thread_next) next_frontier.insert(next_frontier.end(), v.begin(), v.end());
        } else {
            for (auto& [i, lab] : claims) {
                (void)lab;
                const int64_t y = i / W;
                const int64_t x = i - y * W;
                if (y > 0     && labels[i - W] == 0) next_frontier.push_back(i - W);
                if (y + 1 < H && labels[i + W] == 0) next_frontier.push_back(i + W);
                if (x > 0     && labels[i - 1] == 0) next_frontier.push_back(i - 1);
                if (x + 1 < W && labels[i + 1] == 0) next_frontier.push_back(i + 1);
            }
        }
        // Dedup: sort + unique. O(N log N) — fast for typical
        // frontier sizes of 10-100K.
        std::sort(next_frontier.begin(), next_frontier.end());
        next_frontier.erase(
            std::unique(next_frontier.begin(), next_frontier.end()),
            next_frontier.end());

        total_claimed += (int64_t)claims.size();
        std::swap(frontier, next_frontier);
    }
    return total_claimed;
}

// ND fallback. Same algorithm, generic over ndim using coord-tracked
// iteration. The 2D fast path above should be preferred when ndim==2.
template <typename T>
inline int64_t expand_spur_free_nd_inplace(
    T* labels, const std::vector<int64_t>& shape,
    int max_rounds = 100,
    int connectivity_threshold = 1,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    const int ndim = (int)shape.size();
    if (ndim == 0) return 0;
    if (ndim == 2) {
        return expand_spur_free_2d_inplace<T>(
            labels, shape[0], shape[1], max_rounds,
            connectivity_threshold, pool, n_threads);
    }
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

    // Initial frontier
    std::vector<int64_t> frontier;
    std::vector<uint8_t> in_frontier(total, 0);
    std::vector<int64_t> c(ndim, 0);
    for (int64_t i = 0; i < total; ++i) {
        if (labels[i] == 0) {
            bool has_lab = false;
            for (int d = 0; d < ndim && !has_lab; ++d) {
                if (c[d] > 0 && labels[i - strides[d]] != 0) has_lab = true;
                if (!has_lab && c[d] + 1 < shape[d] && labels[i + strides[d]] != 0) has_lab = true;
            }
            if (has_lab) {
                frontier.push_back(i);
                in_frontier[i] = 1;
            }
        }
        // increment coords
        int d = ndim - 1;
        ++c[d];
        while (d > 0 && c[d] >= shape[d]) {
            c[d] = 0; --d; ++c[d];
        }
    }

    int64_t total_claimed = 0;
    std::vector<std::pair<int64_t, T>> claims;
    std::vector<int64_t> next_frontier;
    std::vector<int64_t> coords_tmp(ndim, 0);

    for (int round = 0; round < max_rounds; ++round) {
        if (frontier.empty()) break;
        claims.clear();
        claims.reserve(frontier.size());
        for (int64_t i : frontier) {
            if (labels[i] != 0) { in_frontier[i] = 0; continue; }
            coords_of(i, coords_tmp);
            T nbrs[16];
            int n = 0;
            for (int d = 0; d < ndim && n < 16; ++d) {
                if (coords_tmp[d] > 0)               nbrs[n++] = labels[i - strides[d]];
                if (coords_tmp[d] + 1 < shape[d])    nbrs[n++] = labels[i + strides[d]];
            }
            auto [lab, cnt] = expand_sf_detail::dominant_label(nbrs, n);
            if (cnt > connectivity_threshold) claims.emplace_back(i, lab);
        }
        if (claims.empty()) {
            for (int64_t i : frontier) in_frontier[i] = 0;
            break;
        }
        for (auto& [i, lab] : claims) labels[i] = lab;
        next_frontier.clear();
        for (int64_t i : frontier) in_frontier[i] = 0;
        for (auto& [i, lab] : claims) {
            (void)lab;
            coords_of(i, coords_tmp);
            for (int d = 0; d < ndim; ++d) {
                if (coords_tmp[d] > 0) {
                    int64_t j = i - strides[d];
                    if (labels[j] == 0 && !in_frontier[j]) {
                        in_frontier[j] = 1; next_frontier.push_back(j);
                    }
                }
                if (coords_tmp[d] + 1 < shape[d]) {
                    int64_t j = i + strides[d];
                    if (labels[j] == 0 && !in_frontier[j]) {
                        in_frontier[j] = 1; next_frontier.push_back(j);
                    }
                }
            }
        }
        for (int64_t i : frontier) {
            if (labels[i] != 0) continue;
            coords_of(i, coords_tmp);
            bool has_lab = false;
            for (int d = 0; d < ndim && !has_lab; ++d) {
                if (coords_tmp[d] > 0 && labels[i - strides[d]] != 0) has_lab = true;
                if (!has_lab && coords_tmp[d] + 1 < shape[d] && labels[i + strides[d]] != 0) has_lab = true;
            }
            if (has_lab && !in_frontier[i]) {
                in_frontier[i] = 1;
                next_frontier.push_back(i);
            }
        }
        total_claimed += (int64_t)claims.size();
        std::swap(frontier, next_frontier);
    }
    return total_claimed;
}

}  // namespace ncolor_cpp
