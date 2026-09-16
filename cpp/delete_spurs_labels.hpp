// Label-aware spur removal in any dimension. Each synchronous round uses
// the same face-count and optional antipodal thin-line predicates.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "threadpool.h"

namespace ncolor_cpp {
namespace despur_detail {

inline void coordinates(int64_t i, const std::vector<int64_t>& strides,
                        std::vector<int64_t>& coords) {
    for (size_t d = 0; d < strides.size(); ++d) {
        coords[d] = i / strides[d];
        i -= coords[d] * strides[d];
    }
}

// Enumerate only in-bounds neighbors, with rank-sized scratch. Returning
// true from visit stops immediately; no exponential offset table is built.
template <typename F>
inline void visit_neighbors(const std::vector<int64_t>& shape,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& coords,
                            std::vector<int64_t>& delta, F&& visit) {
    const int rank = static_cast<int>(shape.size());
    int64_t offset = 0;
    for (int d = 0; d < rank; ++d) {
        delta[d] = coords[d] > 0 ? -1 : 0;
        offset += delta[d] * strides[d];
    }
    for (;;) {
        if (offset != 0 && visit(offset, delta)) return;
        int d = rank - 1;
        for (; d >= 0; --d) {
            const int64_t hi = coords[d] + 1 < shape[d] ? 1 : 0;
            if (delta[d] < hi) { ++delta[d]; offset += strides[d]; break; }
            const int64_t lo = coords[d] > 0 ? -1 : 0;
            offset += (lo - delta[d]) * strides[d];
            delta[d] = lo;
        }
        if (d < 0) return;
    }
}

template <typename T>
inline bool should_remove(const T* labels, int64_t i,
                          const std::vector<int64_t>& shape,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& coords,
                          std::vector<int64_t>& delta,
                          std::vector<int64_t>& first,
                          int threshold, bool remove_thin) {
    const T label = labels[i];
    if (label == 0) return false;
    int faces = 0, first_axis = -1;
    bool opposite_faces = false;
    for (int d = 0; d < static_cast<int>(shape.size()); ++d) {
        for (int sign : {-1, 1}) {
            if (coords[d] + sign < 0 || coords[d] + sign >= shape[d]) continue;
            if (labels[i + sign * strides[d]] != label) continue;
            if (faces == 0) first_axis = d;
            else if (faces == 1) opposite_faces = first_axis == d;
            ++faces;
            if (faces > threshold && (!remove_thin || faces > 2)) return false;
        }
    }
    if (faces <= threshold) return true;
    if (!remove_thin || faces == 1 || (faces == 2 && !opposite_faces)) return false;

    int neighbors = 0;
    bool opposite = false;
    visit_neighbors(shape, strides, coords, delta,
        [&](int64_t offset, const std::vector<int64_t>& displacement) {
            if (labels[i + offset] != label) return false;
            if (++neighbors == 1) first = displacement;
            else if (neighbors == 2) {
                opposite = true;
                for (size_t d = 0; d < shape.size(); ++d)
                    if (first[d] != -displacement[d]) { opposite = false; break; }
                if (!opposite) return true;
            }
            return neighbors > 2;
        });
    return neighbors == 2 && opposite;
}

} // namespace despur_detail

template <typename T>
inline int64_t delete_spurs_labels_nd_inplace(
        T* labels, const std::vector<int64_t>& original_shape,
        int threshold = 1, int max_iters = 20,
        ForkJoinPool* pool = nullptr, int n_threads = 1,
        bool remove_thin = false) {
    int64_t total = 1;
    std::vector<int64_t> shape;
    for (int64_t n : original_shape) {
        if (n == 0) return 0;
        if (n < 0 || total > std::numeric_limits<int64_t>::max() / n)
            throw std::overflow_error("invalid spur image shape");
        total *= n;
        if (n > 1) shape.push_back(n);
    }
    if (original_shape.empty() || max_iters == 0) return 0;
    if (shape.empty()) shape.push_back(1);
    const int rank = static_cast<int>(shape.size());
    std::vector<int64_t> strides(rank, 1);
    for (int d = rank - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];
    const int nt = pool && n_threads > 1 && total >= 8192 ? n_threads : 1;
    std::vector<uint8_t> mark(total, 0);
    std::vector<int64_t> frontier, removed;
    bool full_scan = true;
    int64_t total_removed = 0;
    std::vector<int64_t> coords(rank), delta(rank), first(rank);
    for (int round = 0; max_iters < 0 || round < max_iters; ++round) {
        if (full_scan) {
            auto scan = [&](int64_t lo, int64_t hi) {
                std::vector<int64_t> c(rank), step(rank), first_neighbor(rank);
                despur_detail::coordinates(lo, strides, c);
                for (int64_t i = lo; i < hi; ++i) {
                    mark[i] = despur_detail::should_remove(labels, i, shape, strides,
                        c, step, first_neighbor, threshold, remove_thin);
                    for (int d = rank - 1; d >= 0; --d) {
                        if (++c[d] < shape[d]) break;
                        c[d] = 0;
                    }
                }
            };
            if (nt == 1) scan(0, total);
            else {
                std::atomic<int64_t> next{0};
                const int64_t block = std::max<int64_t>(4096, total / (nt * 4));
                pool->parallel([&]() {
                    for (;;) {
                        const int64_t lo = next.fetch_add(block);
                        if (lo >= total) break;
                        scan(lo, std::min(total, lo + block));
                    }
                });
            }
        } else {
            for (int64_t i : frontier) {
                despur_detail::coordinates(i, strides, coords);
                mark[i] = despur_detail::should_remove(labels, i, shape, strides,
                    coords, delta, first, threshold, remove_thin);
            }
        }
        removed.clear();
        auto apply = [&](int64_t i) {
            if (mark[i]) { labels[i] = 0; mark[i] = 0; removed.push_back(i); }
        };
        if (full_scan) for (int64_t i = 0; i < total; ++i) apply(i);
        else for (int64_t i : frontier) apply(i);
        if (removed.empty()) break;
        total_removed += static_cast<int64_t>(removed.size());
        if (total_removed == total || (max_iters >= 0 && round + 1 == max_iters)) break;

        // A dense removal wave is cheaper to rescan than to enumerate all
        // affected neighborhoods. Sparse waves retain the frontier shortcut.
        int64_t neighborhood = 2 * rank;
        if (remove_thin) {
            neighborhood = 1;
            for (int64_t n : shape) {
                if (neighborhood > total / std::min<int64_t>(n, 3)) { neighborhood = total; break; }
                neighborhood *= std::min<int64_t>(n, 3);
            }
        }
        full_scan = static_cast<int64_t>(removed.size()) > total / std::max<int64_t>(1, neighborhood);
        frontier.clear();
        if (full_scan) continue;
        auto enqueue = [&](int64_t i) {
            if (labels[i] != 0 && !mark[i]) { mark[i] = 1; frontier.push_back(i); }
        };
        for (int64_t i : removed) {
            despur_detail::coordinates(i, strides, coords);
            if (remove_thin) {
                despur_detail::visit_neighbors(shape, strides, coords, delta,
                    [&](int64_t offset, const auto&) { enqueue(i + offset); return false; });
            } else {
                for (int d = 0; d < rank; ++d) {
                    if (coords[d] > 0) enqueue(i - strides[d]);
                    if (coords[d] + 1 < shape[d]) enqueue(i + strides[d]);
                }
            }
        }
        if (!full_scan && frontier.empty()) break;
    }
    return total_removed;
}
} // namespace ncolor_cpp
