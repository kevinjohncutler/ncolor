// N-D skeleton / boundary cleanup, pure C++ — no pybind dependency.
//
// Two steps over a row-major N-D label / mask buffer:
//
//   1. Pad-by-1, fill bg components with pixel count ≤ hole_threshold
//      via face-connected CCL.
//   2. Iteratively prune pixels whose fg-neighbour count is in
//      [1, threshold). The connectivity used for the neighbour count is
//      controlled by ``conn_kind``: 1 → cardinal (face only, 2·ndim
//      neighbours), ndim → full diagonal (3^ndim − 1 neighbours).
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

namespace delete_spurs_detail {

// Strided offsets to N-D neighbours under the given connectivity.
// kind=1 is face-only (2·ndim offsets); kind=ndim is full diagonal
// (3^ndim - 1 offsets). The centre cell (offset 0) is excluded.
inline std::vector<int64_t>
make_neighbour_offsets(const std::vector<int64_t>& strides, int ndim, int kind) {
    std::vector<int64_t> offsets;
    std::vector<int> coord(ndim, -1);
    while (true) {
        int manhattan = 0;
        bool nonzero = false;
        for (int d = 0; d < ndim; ++d) {
            manhattan += std::abs(coord[d]);
            if (coord[d] != 0) nonzero = true;
        }
        if (nonzero && manhattan <= kind) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) {
                off += static_cast<int64_t>(coord[d]) * strides[d];
            }
            offsets.push_back(off);
        }
        // Odometer increment over base-3 digits in [-1, 0, 1].
        int d = ndim - 1;
        while (d >= 0) {
            ++coord[d];
            if (coord[d] <= 1) break;
            coord[d] = -1;
            --d;
        }
        if (d < 0) break;
    }
    return offsets;
}

}  // namespace delete_spurs_detail


// ``input``  — row-major N-D buffer of any integer dtype; non-zero = fg.
// ``output`` — row-major N-D bool buffer of the same shape; caller-owned.
// ``shape``  — extent of each axis.
//
// ``conn_kind`` 1 = cardinal (omnipose-style external-spur rule, more
//               aggressive, fewer iterations to converge); ndim = full
//               diagonal (preserves 1-voxel-wide skeleton interiors).
// ``threshold`` — a pixel is pruned when its fg-neighbour count is in
//                 [1, threshold). Use -1 to default to ndim.
// ``max_iter``  — caps the pruning loop. -1 runs to convergence.
template <typename T>
inline void delete_spurs_nd(const T* input, bool* output,
                            const std::vector<int64_t>& shape,
                            int hole_threshold, int conn_kind,
                            int threshold, int max_iter) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("delete_spurs_nd requires shape.size() >= 2");
    }
    if (conn_kind < 1) conn_kind = 1;
    if (conn_kind > ndim) conn_kind = ndim;
    if (threshold < 1) threshold = ndim;

    // Row-major input strides (in element units).
    std::vector<int64_t> in_strides(ndim);
    in_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        in_strides[d] = in_strides[d + 1] * shape[d + 1];
    }

    // Padded geometry: each axis grows by 2.
    std::vector<int64_t> padded_shape(ndim);
    int64_t padded_total = 1;
    for (int d = 0; d < ndim; ++d) {
        padded_shape[d] = shape[d] + 2;
        padded_total *= padded_shape[d];
    }
    std::vector<int64_t> pstrides(ndim);
    pstrides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        pstrides[d] = pstrides[d + 1] * padded_shape[d + 1];
    }

    std::vector<uint8_t> skel(static_cast<size_t>(padded_total), 0u);

    // Copy input into the interior of skel (offset by +1 in each axis).
    // N-D walk via an odometer; "fg" is just ``input[i] != T{0}``.
    {
        std::vector<int64_t> idx(ndim, 0);
        while (true) {
            int64_t in_off = 0;
            int64_t skel_off = 0;
            for (int d = 0; d < ndim; ++d) {
                in_off   += idx[d] * in_strides[d];
                skel_off += (idx[d] + 1) * pstrides[d];
            }
            if (input[in_off] != T{0}) {
                skel[static_cast<size_t>(skel_off)] = 1u;
            }
            int d = ndim - 1;
            while (d >= 0) {
                ++idx[d];
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }
    }

    // Step 1: remove_small_holes (face-connected bg components ≤ threshold).
    // The pad-by-1 step above is what makes this safe: the outer
    // background wraps the entire image, so it always shows up as one
    // huge component well over any sane hole_threshold. Only truly
    // interior holes can fall below the threshold and get filled.
    if (hole_threshold > 0) {
        std::vector<uint8_t> inv(static_cast<size_t>(padded_total));
        for (int64_t i = 0; i < padded_total; ++i) inv[i] = skel[i] ? 0u : 1u;
        std::vector<int32_t> bg_lbl(static_cast<size_t>(padded_total));
        const int32_t n_bg = cc_label_nd<uint8_t>(
            inv.data(), bg_lbl.data(), padded_shape, /*conn=*/1);
        if (n_bg > 0) {
            std::vector<int64_t> areas(static_cast<size_t>(n_bg) + 1, 0);
            for (int64_t i = 0; i < padded_total; ++i) {
                const int32_t c = bg_lbl[i];
                if (c > 0) ++areas[c];
            }
            const int64_t fill_thresh = static_cast<int64_t>(hole_threshold);
            for (int64_t i = 0; i < padded_total; ++i) {
                const int32_t c = bg_lbl[i];
                if (c > 0 && areas[c] <= fill_thresh) skel[i] = 1u;
            }
        }
    }

    // Step 2: iterative endpoint pruning, candidate-list driven.
    // Each iteration only re-checks pixels whose neighbour count could
    // have changed (the previous iteration's removals + their fg
    // neighbours). Total work is O(pixels_removed · n_neighbours)
    // instead of O(iterations · image_pixels). Endpoints are collected
    // before any are removed so the parallel-removal semantics of the
    // naive algorithm are preserved.
    {
        const std::vector<int64_t> offsets =
            delete_spurs_detail::make_neighbour_offsets(pstrides, ndim, conn_kind);
        const int n_nbs = static_cast<int>(offsets.size());

        std::vector<int64_t> candidates;
        std::vector<int64_t> ep;
        std::vector<int64_t> next_candidates;
        candidates.reserve(static_cast<size_t>(padded_total) / 16 + 16);

        // Seed: one full sweep to collect every fg pixel.
        for (int64_t i = 0; i < padded_total; ++i) {
            if (skel[i]) candidates.push_back(i);
        }

        int iter_count = 0;
        while (!candidates.empty()) {
            if (max_iter >= 0 && iter_count >= max_iter) break;
            ep.clear();
            for (int64_t i : candidates) {
                if (!skel[i]) continue;  // already removed this round
                int count = 0;
                for (int k = 0; k < n_nbs; ++k) {
                    if (skel[static_cast<size_t>(i + offsets[k])]) ++count;
                }
                if (count > 0 && count < threshold) ep.push_back(i);
            }
            if (ep.empty()) break;
            ++iter_count;

            // Bulk subtract so the read pass above saw a consistent skel.
            for (int64_t i : ep) skel[i] = 0u;

            // Duplicates in next_candidates are harmless: the skel check
            // at the top of the next iteration filters re-entries.
            next_candidates.clear();
            next_candidates.reserve(ep.size() * static_cast<size_t>(n_nbs));
            for (int64_t i : ep) {
                for (int k = 0; k < n_nbs; ++k) {
                    const int64_t nb = i + offsets[k];
                    if (skel[static_cast<size_t>(nb)]) next_candidates.push_back(nb);
                }
            }
            candidates.swap(next_candidates);
        }
    }

    // Unpad: copy interior of skel back to ``output``.
    {
        std::vector<int64_t> idx(ndim, 0);
        int64_t out_pos = 0;
        while (true) {
            int64_t skel_off = 0;
            for (int d = 0; d < ndim; ++d) skel_off += (idx[d] + 1) * pstrides[d];
            output[out_pos++] = skel[static_cast<size_t>(skel_off)] != 0u;
            int d = ndim - 1;
            while (d >= 0) {
                ++idx[d];
                if (idx[d] < shape[d]) break;
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }
    }
}

}  // namespace ncolor_cpp
