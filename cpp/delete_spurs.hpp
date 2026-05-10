// N-D skeleton cleanup: pad-by-1, fill bg holes ≤ hole_threshold via
// face-connected CCL, then iteratively strip pixels with exactly one
// foreground neighbour until convergence. 2D uses 8-connectivity for
// the endpoint check; ndim≥3 uses face-only.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cc_label.hpp"

namespace py = pybind11;

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

// Dtype-agnostic nonzero test via the numpy buffer protocol's
// itemsize. Accepts bool / uint8 / int32 / etc. without templating.
inline bool nonzero_at(const char* base, py::ssize_t byte_off, py::ssize_t itemsize) {
    const char* p = base + byte_off;
    for (py::ssize_t b = 0; b < itemsize; ++b) {
        if (p[b] != 0) return true;
    }
    return false;
}

}  // namespace delete_spurs_detail


// Returns a fresh boolean array of the input shape. Components with
// pixel count ≤ ``hole_threshold`` are filled (face-connected).
inline py::array_t<bool>
delete_spurs_nd(py::array mask, int hole_threshold) {
    py::buffer_info info = mask.request();
    const int ndim = info.ndim;
    if (ndim < 2) {
        throw std::invalid_argument("delete_spurs requires an array of ndim >= 2");
    }

    // Padded geometry: each axis grows by 2 (one cell on each side).
    std::vector<int64_t> padded_shape(ndim);
    int64_t padded_total = 1;
    for (int d = 0; d < ndim; ++d) {
        padded_shape[d] = info.shape[d] + 2;
        padded_total *= padded_shape[d];
    }
    // Row-major strides over the padded buffer (in element units, since
    // the padded buffer is uint8 with itemsize=1).
    std::vector<int64_t> pstrides(ndim);
    pstrides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        pstrides[d] = pstrides[d + 1] * padded_shape[d + 1];
    }

    std::vector<uint8_t> skel(static_cast<size_t>(padded_total), 0u);

    // Copy ``mask`` into the interior of skel (offset by +1 in each
    // axis). N-D walk via an odometer; works for any input dtype since
    // we only test "nonzero".
    {
        const char* base = static_cast<const char*>(info.ptr);
        const py::ssize_t itemsize = info.itemsize;
        std::vector<int64_t> idx(ndim, 0);
        while (true) {
            py::ssize_t input_byte_off = 0;
            int64_t skel_off = 0;
            for (int d = 0; d < ndim; ++d) {
                input_byte_off += static_cast<py::ssize_t>(idx[d]) * info.strides[d];
                skel_off += (idx[d] + 1) * pstrides[d];
            }
            if (delete_spurs_detail::nonzero_at(base, input_byte_off, itemsize)) {
                skel[static_cast<size_t>(skel_off)] = 1u;
            }
            int d = ndim - 1;
            while (d >= 0) {
                ++idx[d];
                if (idx[d] < info.shape[d]) break;
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }
    }

    // Step 1: remove_small_holes (face-connected bg components ≤ threshold).
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
            const int64_t threshold = static_cast<int64_t>(hole_threshold);
            for (int64_t i = 0; i < padded_total; ++i) {
                const int32_t c = bg_lbl[i];
                if (c > 0 && areas[c] <= threshold) skel[i] = 1u;
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
        // Match endpoints_nd: 2D uses full (8-conn), ndim≥3 uses face-only.
        const int neigh_kind = (ndim == 2) ? ndim : 1;
        const std::vector<int64_t> offsets =
            delete_spurs_detail::make_neighbour_offsets(pstrides, ndim, neigh_kind);
        const int n_nbs = static_cast<int>(offsets.size());

        std::vector<int64_t> candidates;
        std::vector<int64_t> ep;
        std::vector<int64_t> next_candidates;
        candidates.reserve(static_cast<size_t>(padded_total) / 16 + 16);

        // Seed: one full sweep to collect every fg pixel.
        for (int64_t i = 0; i < padded_total; ++i) {
            if (skel[i]) candidates.push_back(i);
        }

        while (!candidates.empty()) {
            ep.clear();
            // Mark endpoints among the candidate set.
            for (int64_t i : candidates) {
                if (!skel[i]) continue;  // already removed this round
                int count = 0;
                for (int k = 0; k < n_nbs; ++k) {
                    if (skel[static_cast<size_t>(i + offsets[k])]) ++count;
                }
                if (count == 1) ep.push_back(i);
            }
            if (ep.empty()) break;

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

    // Unpad: copy interior of skel back to a fresh bool array of the
    // original shape (row-major contiguous).
    py::array_t<bool> result(info.shape);
    bool* out = static_cast<bool*>(result.mutable_data());
    {
        std::vector<int64_t> idx(ndim, 0);
        py::ssize_t out_pos = 0;
        while (true) {
            int64_t skel_off = 0;
            for (int d = 0; d < ndim; ++d) skel_off += (idx[d] + 1) * pstrides[d];
            out[out_pos++] = skel[static_cast<size_t>(skel_off)] != 0u;
            int d = ndim - 1;
            while (d >= 0) {
                ++idx[d];
                if (idx[d] < info.shape[d]) break;
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }
    }
    return result;
}

}  // namespace ncolor_cpp
