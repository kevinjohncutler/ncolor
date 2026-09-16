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
template <typename T>
inline void delete_spurs_nd(const T* input, bool* output,
                            const std::vector<int64_t>& shape,
                            int hole_threshold, int conn_kind,
                            int threshold, int max_iter) {
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
    std::vector<uint8_t> skel(static_cast<size_t>(total));
    for (int64_t i = 0; i < total; ++i) skel[i] = input[i] != T{0};

    // Face-connected background components touching an image boundary
    // belong to the exterior, regardless of the requested hole size.
    if (hole_threshold > 0 && rank == ndim) {
        std::vector<uint8_t> inv(static_cast<size_t>(total));
        std::vector<int32_t> components(static_cast<size_t>(total));
        for (int64_t i = 0; i < total; ++i) inv[i] = !skel[i];
        const int32_t n = cc_label_nd<uint8_t>(
            inv.data(), components.data(), active_shape, 1);
        std::vector<int64_t> areas(static_cast<size_t>(n) + 1, 0);
        std::vector<uint8_t> exterior(static_cast<size_t>(n) + 1, 0);
        for (int64_t i = 0; i < total; ++i) {
            const int32_t c = components[i];
            if (!c) continue;
            ++areas[c];
            for (int d = 0; d < rank; ++d) {
                const int64_t x = (i / strides[d]) % active_shape[d];
                if (x == 0 || x + 1 == active_shape[d]) {
                    exterior[c] = 1;
                    break;
                }
            }
        }
        for (int64_t i = 0; i < total; ++i) {
            const int32_t c = components[i];
            if (c && !exterior[c] && areas[c] <= hole_threshold) skel[i] = 1;
        }
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
        std::vector<int64_t> coords(rank);
        auto visit_neighbors = [&](int64_t i, auto&& visit) {
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
        std::vector<int64_t> candidates, removed, next;
        std::vector<uint8_t> queued(static_cast<size_t>(total), 0);
        for (int64_t i = 0; i < total; ++i)
            if (skel[i]) candidates.push_back(i);
        int iter = 0;
        while (!candidates.empty() && (max_iter < 0 || iter < max_iter)) {
            removed.clear();
            for (int64_t i : candidates) {
                queued[i] = 0;
                int count = 0;
                visit_neighbors(i, [&](int64_t j) {
                    count += skel[j];
                    return count < threshold;
                });
                if (count > 0 && count < threshold) removed.push_back(i);
            }
            if (removed.empty()) break;
            ++iter;
            // Synchronous removal: all counts saw the same image state.
            for (int64_t i : removed) skel[i] = 0;
            next.clear();
            for (int64_t i : removed) {
                visit_neighbors(i, [&](int64_t j) {
                    if (skel[j] && !queued[j]) {
                        queued[j] = 1;
                        next.push_back(j);
                    }
                    return true;
                });
            }
            candidates.swap(next);
        }
    }
    for (int64_t i = 0; i < total; ++i) output[i] = skel[i] != 0;

}

}  // namespace ncolor_cpp
