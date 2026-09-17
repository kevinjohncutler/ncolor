// N-D skeleton / boundary cleanup, pure C++ — no pybind dependency.
//
// Two steps over a row-major N-D label / mask buffer:
//
//   1. Fill enclosed background components up to hole_threshold
//      via face-connected component labeling without padding.
//   2. Iteratively prune pixels whose fg-neighbor count is in
//      [1, threshold). The connectivity used for the neighbor count is
//      controlled by ``conn_kind``: 1 → cardinal (face only, 2·ndim
//      neighbors), ndim → full diagonal (3^ndim − 1 neighbors).
//      Isolated pixels (count == 0) are always preserved.
//
// Templated on input dtype T: any pixel where ``input[i] != T{0}`` is
// foreground. The output is a separate ``bool`` buffer of the same
// shape. Callers own both the input and output allocations; raw
// row-major contiguous element layout is assumed.
//
// C++ usage:
//
//   #include "delete_spurs.hpp"
//   std::vector<uint8_t> mask(W * H);   // row-major
//   std::vector<bool>    out(W * H);
//   ncolor_cpp::delete_spurs_nd<uint8_t>(
//       mask.data(), out.data(), {H, W},
//       /*hole_threshold=*/5, /*conn_kind=*/1, /*threshold=*/-1, /*max_iter=*/-1);
#pragma once

#include <cstdint>
#include <cstdlib>
#include <atomic>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <stdexcept>
#include <vector>

#include "cc_label.hpp"

namespace ncolor_cpp {

// ``input``  — row-major N-D buffer of any integer dtype; non-zero = fg.
// ``output`` — row-major N-D bool buffer of the same shape; caller-owned.
// ``shape``  — extent of each axis.
//
// ``conn_kind`` 1 = cardinal (external-spur rule, more
//               aggressive, fewer iterations to converge); ndim = full
//               diagonal (preserves 1-voxel-wide skeleton interiors).
// ``threshold`` — a pixel is pruned when its fg-neighbor count is in
//                 [1, threshold). Use -1 to default to ndim.
// ``max_iter``  — caps the pruning loop. -1 runs to convergence.
//
// ``pool`` / ``n_threads`` optionally spread the work. Both phases give
// the same result either way: hole filling is a reduction, and a pruning
// round counts every candidate against the state left by the previous
// round, so removals within a round never depend on their order.
template <typename T, bool Threaded>
inline void delete_spurs_impl(const T* input, bool* output,
                              const std::vector<int64_t>& shape,
                              int hole_threshold, int conn_kind,
                              int threshold, int max_iter,
                              ForkJoinPool* pool, int n_threads) {
    const int ndim = static_cast<int>(shape.size());
    validate_neighborhood_ndim(ndim);
    if (ndim < 2) {
        throw std::invalid_argument("delete_spurs_nd requires shape.size() >= 2");
    }
    if (conn_kind < 1) conn_kind = 1;
    if (conn_kind > ndim) conn_kind = ndim;
    if (threshold < 1) threshold = ndim;

    // Traverse only active axes, but keep the original rank's threshold.
    // A singleton axis exposes every background voxel to the exterior.
    std::vector<int64_t> active_shape;
    int64_t total = 1;
    for (int64_t n : shape) {
        if (n == 0) return;
        if (n < 0 || total > std::numeric_limits<int64_t>::max() / n)
            throw std::overflow_error("shape exceeds int64 capacity");
        total *= n;
        if (n > 1) active_shape.push_back(n);
    }
    const int rank = static_cast<int>(active_shape.size());
    std::vector<int64_t> strides(rank, 1);
    for (int d = rank - 2; d >= 0; --d)
        strides[d] = strides[d + 1] * active_shape[d + 1];

    // Work spreaders. Without a pool both run the body once, unsplit.
    const size_t workers = Threaded ? static_cast<size_t>(std::max(1, n_threads)) : 1;
    auto over_elements = [&](int64_t count, auto&& body) {
        if constexpr (Threaded) {
            dispatch_parallel(*pool, static_cast<size_t>(count),
                workers * DISPATCH_CHUNKS_PER_THREAD, body);
        } else {
            body(size_t{0}, static_cast<size_t>(count));
        }
    };
    auto over_parts = [&](size_t parts, auto&& body) {
        if constexpr (Threaded) {
            if (parts > 1) {
                dispatch_parallel(*pool, parts, parts, [&](size_t lo, size_t hi) {
                    for (size_t part = lo; part < hi; ++part) body(part);
                });
                return;
            }
        }
        for (size_t part = 0; part < parts; ++part) body(part);
    };
    // Coordinates of a flat index, for the start of a part.
    auto coords_of = [&](int64_t flat, std::vector<int64_t>& coords) {
        for (int d = 0; d < rank; ++d) coords[d] = (flat / strides[d]) % active_shape[d];
    };
    // Concatenate per-part buffers in part order, so the result never
    // depends on which worker finished first.
    auto join = [](const std::vector<std::vector<int64_t>>& parts,
                   std::vector<int64_t>& out) {
        size_t count = 0;
        for (const auto& part : parts) count += part.size();
        out.clear();
        out.reserve(count);
        for (const auto& part : parts) out.insert(out.end(), part.begin(), part.end());
    };

    std::vector<uint8_t> skel(static_cast<size_t>(total));
    over_elements(total, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) skel[i] = input[i] != T{0};
    });

    // Face-connected background components touching an image boundary
    // belong to the exterior, regardless of the requested hole size.
    if (hole_threshold > 0 && rank == ndim) {
        std::vector<uint8_t> inv(static_cast<size_t>(total));
        std::vector<int32_t> components(static_cast<size_t>(total));
        over_elements(total, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) inv[i] = !skel[i];
        });
        int32_t n;
        if constexpr (Threaded) {
            n = cc_label_parallel_nd<uint8_t>(inv.data(), components.data(),
                                              active_shape, 1, *pool, n_threads);
        } else {
            n = cc_label_nd<uint8_t>(inv.data(), components.data(), active_shape, 1);
        }
        std::vector<int64_t> areas(static_cast<size_t>(n) + 1, 0);
        std::vector<uint8_t> exterior(static_cast<size_t>(n) + 1, 0);
        // Each part counts into its own tables, bounded by a scratch
        // budget so a heavily fragmented background stays in memory.
        constexpr size_t HOLE_SCRATCH_BYTES = 64u << 20;
        const size_t width = (static_cast<size_t>(n) + 1) * 9;
        const size_t parts = std::max<size_t>(1, std::min<size_t>(
            Threaded ? workers : 1, width ? HOLE_SCRATCH_BYTES / width : 1));
        std::vector<std::vector<int64_t>> part_areas(parts);
        std::vector<std::vector<uint8_t>> part_exterior(parts);
        over_parts(parts, [&](size_t part) {
            auto& count = part_areas[part];
            auto& touches = part_exterior[part];
            count.assign(static_cast<size_t>(n) + 1, 0);
            touches.assign(static_cast<size_t>(n) + 1, 0);
            const int64_t begin = static_cast<int64_t>(total * part / parts);
            const int64_t end = static_cast<int64_t>(total * (part + 1) / parts);
            std::vector<int64_t> coords(rank);
            coords_of(begin, coords);
            for (int64_t i = begin; i < end; ++i) {
                const int32_t c = components[i];
                if (c) {
                    ++count[c];
                    for (int d = 0; d < rank; ++d) {
                        if (coords[d] == 0 || coords[d] + 1 == active_shape[d]) {
                            touches[c] = 1;
                            break;
                        }
                    }
                }
                for (int d = rank - 1; d >= 0; --d) {
                    if (++coords[d] < active_shape[d]) break;
                    coords[d] = 0;
                }
            }
        });
        for (size_t part = 0; part < parts; ++part) {
            for (size_t c = 1; c <= static_cast<size_t>(n); ++c) {
                areas[c] += part_areas[part][c];
                exterior[c] |= part_exterior[part][c];
            }
        }
        over_elements(total, [&](size_t begin, size_t end) {
            for (size_t i = begin; i < end; ++i) {
                const int32_t c = components[i];
                if (c && !exterior[c] && areas[c] <= hole_threshold) skel[i] = 1;
            }
        });
    }

    if (threshold > 1 && max_iter != 0 && rank > 0) {
        struct Neighbor {
            std::vector<int8_t> delta;
            int64_t flat;
        };
        std::vector<Neighbor> neighbors;
        detail::for_each_forward_neighbor(active_shape, std::min(conn_kind, rank), 1,
            [&](const std::vector<int8_t>& dc, int, int) {
                int64_t flat = 0;
                std::vector<int8_t> opposite(rank);
                for (int d = 0; d < rank; ++d) {
                    flat += dc[d] * strides[d];
                    opposite[d] = -dc[d];
                }
                neighbors.push_back({dc, flat});
                neighbors.push_back({std::move(opposite), -flat});
            });
        // Each worker owns its coordinate buffer, so this stays reentrant.
        auto visit_neighbors = [&](int64_t i, std::vector<int64_t>& coords, auto&& visit) {
            bool interior = true;
            for (int d = 0; d < rank; ++d) {
                coords[d] = (i / strides[d]) % active_shape[d];
                interior = interior && coords[d] > 0 && coords[d] + 1 < active_shape[d];
            }
            if (interior) {
                for (const auto& nb : neighbors)
                    if (!visit(i + nb.flat)) break;
                return;
            }
            for (const auto& nb : neighbors) {
                bool valid = true;
                for (int d = 0; d < rank; ++d) {
                    const int64_t x = coords[d] + nb.delta[d];
                    if (x < 0 || x >= active_shape[d]) { valid = false; break; }
                }
                if (valid && !visit(i + nb.flat)) break;
            }
        };
        // Claiming a pixel for the next round must happen once even when
        // several removed neighbors reach it at the same moment.
        using Queued = std::conditional_t<Threaded, std::atomic<uint8_t>, uint8_t>;
        std::vector<Queued> queued(static_cast<size_t>(total));
        for (auto& flag : queued) {
            if constexpr (Threaded) flag.store(0, std::memory_order_relaxed);
            else flag = 0;
        }
        auto claim = [](Queued& flag) {
            if constexpr (Threaded) return flag.exchange(1, std::memory_order_relaxed) == 0;
            else { if (flag) return false; flag = 1; return true; }
        };
        auto release = [](Queued& flag) {
            if constexpr (Threaded) flag.store(0, std::memory_order_relaxed);
            else flag = 0;
        };

        std::vector<int64_t> candidates, removed;
        const size_t parts = Threaded ? workers * DISPATCH_CHUNKS_PER_THREAD : 1;
        std::vector<std::vector<int64_t>> buckets(parts);
        over_parts(parts, [&](size_t part) {
            auto& bucket = buckets[part];
            bucket.clear();
            const int64_t begin = static_cast<int64_t>(total * part / parts);
            const int64_t end = static_cast<int64_t>(total * (part + 1) / parts);
            for (int64_t i = begin; i < end; ++i)
                if (skel[i]) bucket.push_back(i);
        });
        join(buckets, candidates);
        int iter = 0;
        while (!candidates.empty() && (max_iter < 0 || iter < max_iter)) {
            const size_t active = std::min(parts, std::max<size_t>(1, candidates.size() / 4096));
            over_parts(active, [&](size_t part) {
                auto& bucket = buckets[part];
                bucket.clear();
                const size_t begin = candidates.size() * part / active;
                const size_t end = candidates.size() * (part + 1) / active;
                std::vector<int64_t> coords(rank);
                for (size_t at = begin; at < end; ++at) {
                    const int64_t i = candidates[at];
                    release(queued[i]);
                    int count = 0;
                    visit_neighbors(i, coords, [&](int64_t j) {
                        count += skel[j];
                        return count < threshold;
                    });
                    if (count > 0 && count < threshold) bucket.push_back(i);
                }
            });
            for (size_t part = active; part < parts; ++part) buckets[part].clear();
            join(buckets, removed);
            if (removed.empty()) break;
            ++iter;
            // Synchronous removal: all counts saw the same image state.
            over_elements(static_cast<int64_t>(removed.size()), [&](size_t begin, size_t end) {
                for (size_t at = begin; at < end; ++at) skel[removed[at]] = 0;
            });
            const size_t spread = std::min(parts, std::max<size_t>(1, removed.size() / 4096));
            over_parts(spread, [&](size_t part) {
                auto& bucket = buckets[part];
                bucket.clear();
                const size_t begin = removed.size() * part / spread;
                const size_t end = removed.size() * (part + 1) / spread;
                std::vector<int64_t> coords(rank);
                for (size_t at = begin; at < end; ++at) {
                    visit_neighbors(removed[at], coords, [&](int64_t j) {
                        if (skel[j] && claim(queued[j])) bucket.push_back(j);
                        return true;
                    });
                }
            });
            for (size_t part = spread; part < parts; ++part) buckets[part].clear();
            join(buckets, candidates);
        }
    }
    over_elements(total, [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) output[i] = skel[i] != 0;
    });
}

// Serial entry point, kept for callers without a pool.
template <typename T>
inline void delete_spurs_nd(const T* input, bool* output,
                            const std::vector<int64_t>& shape,
                            int hole_threshold, int conn_kind,
                            int threshold, int max_iter) {
    delete_spurs_impl<T, false>(input, output, shape, hole_threshold,
                                conn_kind, threshold, max_iter, nullptr, 1);
}

// Pool entry point. Small images stay serial: splitting them costs more
// than the scan itself.
template <typename T>
inline void delete_spurs_nd(const T* input, bool* output,
                            const std::vector<int64_t>& shape,
                            int hole_threshold, int conn_kind,
                            int threshold, int max_iter,
                            ForkJoinPool& pool, int n_threads) {
    int64_t total = 1;
    for (int64_t n : shape) total *= n;
    constexpr int64_t DELETE_SPURS_SERIAL_THRESHOLD = 65536;
    if (n_threads <= 1 || total < DELETE_SPURS_SERIAL_THRESHOLD) {
        delete_spurs_nd<T>(input, output, shape, hole_threshold,
                           conn_kind, threshold, max_iter);
        return;
    }
    delete_spurs_impl<T, true>(input, output, shape, hole_threshold,
                               conn_kind, threshold, max_iter, &pool, n_threads);
}

}  // namespace ncolor_cpp
