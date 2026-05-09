// delete_spurs — N-D skeleton cleanup (matches the original Python
// delete_spurs in src/ncolor/format.py). Three steps:
//
//   1. Pad input by 1 with constant 0 (so the outer background wraps
//      into a single huge component at step 2 and edge-adjacent
//      endpoints have well-defined neighbours at step 3).
//   2. ``remove_small_holes`` — invert, label face-connected bg
//      components via ``cc_label_nd``, fill components whose pixel
//      count is ≤ ``hole_threshold`` back into the foreground.
//   3. Iterative endpoint pruning — for each fg pixel, count fg
//      neighbours; flip pixels with exactly one neighbour to bg;
//      repeat until a sweep produces no change. Connectivity here
//      matches scipy.ndimage.generate_binary_structure(ndim, k):
//      k=ndim (full diagonal) for 2D, k=1 (face) for ndim≥3.
//
// All steps work on a flat uint8 buffer keyed by row-major strides
// computed from the padded shape — no dimension-specific code paths.
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

// Build the set of byte-offsets within an N-D buffer of given strides
// that correspond to neighbours under the requested connectivity. The
// offsets do not include 0 (the centre pixel itself).
//
//   kind=1     → face only            (2·ndim offsets)
//   kind=2     → face + edge          (depends on ndim; 8 for 2D, 18 for 3D)
//   kind=ndim  → face + edge + corner (3^ndim − 1 offsets)
//
// Walks every cell of the {-1,0,1}^ndim cube, keeps cells whose L1
// distance from origin lies in [1, kind], and turns each kept coord
// into a strided byte-offset into the buffer.
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

// Read the input buffer at byte-offset ``byte_off``, returning whether
// any byte of the underlying value is nonzero. The caller passes
// ``itemsize`` from the numpy buffer protocol — this lets us accept
// bool / uint8 / int32 / etc. from Python without dtype-specific paths.
inline bool nonzero_at(const char* base, py::ssize_t byte_off, py::ssize_t itemsize) {
    const char* p = base + byte_off;
    for (py::ssize_t b = 0; b < itemsize; ++b) {
        if (p[b] != 0) return true;
    }
    return false;
}

}  // namespace delete_spurs_detail


// Public entry: returns a fresh boolean array of the input shape.
//
// ``hole_threshold`` matches skimage.morphology.remove_small_holes'
// ``area_threshold`` semantics: components with pixel count ≤ threshold
// are filled. Default mirrors the original Python implementation.
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

    // Step 2: iterative endpoint pruning.
    //
    // The naive algorithm scans the full padded buffer twice per
    // iteration (mark + subtract). For a 1024² image with stroke
    // lengths ~256 the iteration count is on that order, so the cost
    // is ~iterations × image_pixels. On dense skeletons this is fine,
    // but for sparse foregrounds it bills work proportional to the
    // empty background.
    //
    // Optimised version: maintain a *candidate set* of pixel offsets
    // that need checking on each iteration. Initial set = every fg
    // pixel (one full pass). On subsequent iterations the candidate
    // set is the union of *neighbours of pixels removed last
    // iteration* — only those can have had their neighbour count
    // change. Equivalent to the parallel-removal semantics of the
    // original algorithm because all endpoints in iteration k are
    // identified before any are removed (we collect into ``ep``
    // first, then subtract in bulk). Total work scales with
    // (pixels removed × neighbours_per_pixel) instead of
    // (iterations × image_pixels).
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

            // Subtract — done in bulk so all reads above saw the same skel state.
            for (int64_t i : ep) skel[i] = 0u;

            // Next candidates: fg neighbours of just-removed pixels. Duplicates
            // are harmless (the skel check above filters re-entries on the next
            // iteration), so we skip an explicit visited set.
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
