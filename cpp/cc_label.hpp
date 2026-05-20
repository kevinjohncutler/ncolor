/*
 * cc_label.hpp — N-D connected-components labeling + minimal regionprops.
 *
 * Two-pass union-find (Wu et al. 2009-style) over an N-D foreground mask.
 * Backward-neighbor set is computed via the same odometer machinery as
 * connect.hpp, so connectivity (conn ∈ [1, ndim]) generalizes cleanly to
 * any ndim. Single-threaded for the prototype; parallelization via
 * strip-merge is straightforward if benchmarks demand it.
 *
 * Public entry points:
 *   - ncolor_cpp::cc_label_nd<T>(...) → int32_t (n_components)
 *   - ncolor_cpp::regionprops_nd(...) → fills area / bbox / centroid arrays
 */

#ifndef NCOLOR_CC_LABEL_HPP
#define NCOLOR_CC_LABEL_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "connect.hpp"  // detail::build_forward_neighbors

namespace ncolor_cpp {

// Optional per-stage timing for cc_label_nd. Set the pointer non-null to
// receive {fg_mask_ms, pass1_ms, pass2_ms}. Otherwise the kernel skips
// the std::chrono calls entirely.
struct CCStageTimes {
    double fg_mask_ms = 0.0;
    double pass1_ms = 0.0;
    double pass2_ms = 0.0;
};

namespace cc_detail {

// Union-Find with path-halving and union-by-rank.
struct UnionFind {
    std::vector<int32_t> parent;
    std::vector<int32_t> rank_;
    void reserve(size_t n) { parent.reserve(n); rank_.reserve(n); }
    int32_t make_set() {
        const int32_t id = static_cast<int32_t>(parent.size());
        parent.push_back(id);
        rank_.push_back(0);
        return id;
    }
    int32_t find(int32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    }
    void unite(int32_t a, int32_t b) {
        int32_t ra = find(a), rb = find(b);
        if (ra == rb) return;
        if (rank_[ra] < rank_[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rank_[ra] == rank_[rb]) ++rank_[ra];
    }
};

}  // namespace cc_detail

// Inner-axis fast pass-1 scan: walks an interior row whose outer coords
// are guaranteed all-interior, so every backward neighbor at offset
// ``-nb[k]`` is in-bounds (except at the inner axis's two endpoints,
// which the caller handles separately). Templated on ``N_NBS`` so the
// per-pixel inner loop unrolls and the offset constants live in
// registers, not in nb_flat[k] reads.
template <int N_NBS>
static inline void cc_pass1_interior_inner(
        const uint8_t* fg, int32_t* lab,
        int64_t x_start, int64_t x_end,
        const int64_t* nb_flat, cc_detail::UnionFind& uf) {
    int64_t nb[N_NBS];
    for (int i = 0; i < N_NBS; ++i) nb[i] = nb_flat[i];
    for (int64_t x = x_start; x < x_end; ++x) {
        if (!fg[x]) continue;
        int32_t best = 0;
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC unroll 16
#endif
        for (int k = 0; k < N_NBS; ++k) {
            const int32_t l = lab[x - nb[k]];
            if (l == 0) continue;
            if (best == 0) best = l;
            else if (best != l) uf.unite(best, l);
        }
        lab[x] = (best == 0) ? uf.make_set() : best;
    }
}

// Runtime-N_NBS fallback (used for connectivities outside the small
// dispatch table — typically only exotic >3-D inputs).
static inline void cc_pass1_interior_inner_runtime(
        const uint8_t* fg, int32_t* lab,
        int64_t x_start, int64_t x_end,
        int n_nbs, const int64_t* nb_flat,
        cc_detail::UnionFind& uf) {
    for (int64_t x = x_start; x < x_end; ++x) {
        if (!fg[x]) continue;
        int32_t best = 0;
        for (int k = 0; k < n_nbs; ++k) {
            const int32_t l = lab[x - nb_flat[k]];
            if (l == 0) continue;
            if (best == 0) best = l;
            else if (best != l) uf.unite(best, l);
        }
        lab[x] = (best == 0) ? uf.make_set() : best;
    }
}

// Connected-components labeling. Foreground = (input != 0). Output is
// int32 labels: 0 = bg, 1..N = component IDs, dense and sequential.
// Returns N (number of components).
//
// Implementation: classic two-pass union-find. Pass 1 raster-scans with
// an outer odometer over coords[0..ndim-2] × inner axis (ndim-1). When
// the outer coords are all-interior the inner axis runs as a tight
// templated unrolled loop touching only the pre-computed
// backward-neighbor offsets (no per-axis bounds check). The endpoints
// of every inner row, and any row whose outer coords land on a
// boundary axis, take the per-pixel boundary-mask path.
template <typename T>
inline int32_t cc_label_nd(const T* input, int32_t* output,
                           const std::vector<int64_t>& shape, int conn,
                           CCStageTimes* times = nullptr) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 1) return 0;
    if (conn < 1) conn = 1;
    if (conn > ndim) conn = ndim;
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    if (total == 0) return 0;

    using clk = std::chrono::steady_clock;
    auto now_ms = [](clk::time_point a, clk::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    auto t_start = times ? clk::now() : clk::time_point{};

    // Forward-neighbor set (lex-first +1). Backward offsets = negate.
    std::vector<int64_t> strides, nb_fwd;
    std::vector<int8_t> nb_dc_fwd;
    detail::build_forward_neighbors(shape, conn, strides, nb_fwd, nb_dc_fwd);
    const int n_nbs = static_cast<int>(nb_fwd.size());

    cc_detail::UnionFind uf;
    uf.reserve(static_cast<size_t>(total) / 16 + 16);
    (void)uf.make_set();  // id 0 reserved for bg sentinel

    // Build a uint8 foreground mask once. The pass-1 inner loop reads it
    // at every pixel; reading a uint8 is faster than templated-T compare
    // and gives the compiler a tighter loop body.
    std::vector<uint8_t> fg(total);
    for (int64_t i = 0; i < total; ++i) fg[i] = (input[i] != T{0}) ? 1u : 0u;

    std::fill_n(output, total, int32_t{0});

    auto t_after_fg = times ? clk::now() : clk::time_point{};

    const int inner = ndim - 1;
    const int64_t W = shape[inner];

    // Per-pixel boundary-aware step (used at inner-row endpoints and on
    // rows whose outer coords are boundary).
    auto step_pixel_checked = [&](const int64_t* coords, int64_t flat) {
        if (!fg[flat]) return;
        int32_t best = 0;
        for (int k = 0; k < n_nbs; ++k) {
            const int8_t* dc = nb_dc_fwd.data() + k * ndim;
            bool valid = true;
            for (int d = 0; d < ndim; ++d) {
                if (dc[d] == 0) continue;
                const int64_t nc = coords[d] - dc[d];
                if (nc < 0 || nc >= shape[d]) { valid = false; break; }
            }
            if (!valid) continue;
            const int32_t l = output[flat - nb_fwd[k]];
            if (l == 0) continue;
            if (best == 0) best = l;
            else if (best != l) uf.unite(best, l);
        }
        output[flat] = (best == 0) ? uf.make_set() : best;
    };

    // Outer odometer over coords[0..ndim-2]. Boundary mask says whether
    // any outer axis is at its first or last index.
    constexpr int MAX_NDIM = 16;
    int64_t coords[MAX_NDIM] = {0};
    uint32_t outer_bnd = 0;
    for (int d = 0; d < inner; ++d) {
        if (coords[d] == 0 || coords[d] >= shape[d] - 1) outer_bnd |= (1u << d);
    }
    int64_t row_base = 0;  // flat offset to (coords[0..ndim-2], inner=0)
    const int64_t outer_total = (inner == 0) ? 1 : (total / W);

    auto inner_dispatch = [&](int64_t x_start, int64_t x_end) {
        switch (n_nbs) {
            case 2:  cc_pass1_interior_inner<2 >(fg.data() + row_base,
                                                  output + row_base,
                                                  x_start, x_end,
                                                  nb_fwd.data(), uf); break;
            case 3:  cc_pass1_interior_inner<3 >(fg.data() + row_base,
                                                  output + row_base,
                                                  x_start, x_end,
                                                  nb_fwd.data(), uf); break;
            case 4:  cc_pass1_interior_inner<4 >(fg.data() + row_base,
                                                  output + row_base,
                                                  x_start, x_end,
                                                  nb_fwd.data(), uf); break;
            case 9:  cc_pass1_interior_inner<9 >(fg.data() + row_base,
                                                  output + row_base,
                                                  x_start, x_end,
                                                  nb_fwd.data(), uf); break;
            case 13: cc_pass1_interior_inner<13>(fg.data() + row_base,
                                                  output + row_base,
                                                  x_start, x_end,
                                                  nb_fwd.data(), uf); break;
            default: cc_pass1_interior_inner_runtime(fg.data() + row_base,
                                                     output + row_base,
                                                     x_start, x_end,
                                                     n_nbs, nb_fwd.data(), uf);
                     break;
        }
    };

    for (int64_t outer_idx = 0; outer_idx < outer_total; ++outer_idx) {
        if (outer_bnd != 0 || W < 3) {
            // Outer coords on a boundary OR inner axis < 3: per-pixel checked.
            for (int64_t x = 0; x < W; ++x) {
                coords[inner] = x;
                step_pixel_checked(coords, row_base + x);
            }
        } else {
            // Outer all-interior: inner-row endpoints take the checked
            // path, the (1, W-1) interval takes the unrolled fast path.
            coords[inner] = 0;
            step_pixel_checked(coords, row_base);
            inner_dispatch(/*x_start=*/1, /*x_end=*/W - 1);
            coords[inner] = W - 1;
            step_pixel_checked(coords, row_base + (W - 1));
        }
        // Advance the outer odometer (axes [0 .. ndim-2]).
        if (inner == 0) break;
        int d = inner - 1;
        ++coords[d];
        row_base += strides[d];
        while (coords[d] >= shape[d] && d > 0) {
            row_base -= coords[d] * strides[d];
            coords[d] = 0;
            outer_bnd |= (1u << d);
            --d;
            ++coords[d];
            row_base += strides[d];
        }
        if (coords[d] >= shape[d]) break;
        const bool is_bnd = (coords[d] == 0 || coords[d] >= shape[d] - 1);
        if (is_bnd) outer_bnd |= (1u << d);
        else outer_bnd &= ~(1u << d);
    }

    auto t_after_pass1 = times ? clk::now() : clk::time_point{};

    // Flatten the union-find: rewrite every entry to its root so pass 2
    // can do a direct uf.parent[prov] lookup instead of recursive find().
    // O(K log* K) for K provisional labels, but K ≪ total in typical
    // inputs and we only do it once. The savings inside the per-pixel
    // pass 2 loop dwarf this cost.
    for (size_t i = 1; i < uf.parent.size(); ++i) {
        uf.parent[i] = uf.find(static_cast<int32_t>(i));
    }

    // Pass 2: resolve provisional labels to dense 1..N IDs. Run-length
    // coalescing (cache the most-recent prov→final translation) skips
    // the lookup when consecutive pixels share the same provisional
    // label — common because pass 1 propagates labels along rows.
    std::vector<int32_t> remap(uf.parent.size(), 0);
    int32_t next_label = 0;
    int32_t prev_prov = -1, prev_final = 0;
    for (int64_t flat = 0; flat < total; ++flat) {
        const int32_t prov = output[flat];
        if (prov == 0) { prev_prov = -1; continue; }
        if (prov == prev_prov) {
            output[flat] = prev_final;
            continue;
        }
        const int32_t root = uf.parent[prov];
        int32_t final_lab = remap[root];
        if (final_lab == 0) {
            final_lab = ++next_label;
            remap[root] = final_lab;
        }
        output[flat] = final_lab;
        prev_prov = prov;
        prev_final = final_lab;
    }

    if (times) {
        auto t_end = clk::now();
        times->fg_mask_ms = now_ms(t_start, t_after_fg);
        times->pass1_ms   = now_ms(t_after_fg, t_after_pass1);
        times->pass2_ms   = now_ms(t_after_pass1, t_end);
    }
    return next_label;
}


// Label-aware connected components. Like cc_label_nd, but neighbors
// are only unioned when they share the same nonzero input value, so
// each output component lies entirely within one source label.
// ``output`` holds dense 1..N component IDs (0 = bg);
// ``source_labels_out`` is sized to N with the source value of each
// component so callers can group components by source without
// rescanning the image.
template <typename T>
inline int32_t cc_label_per_label_nd(const T* input, int32_t* output,
                                      const std::vector<int64_t>& shape,
                                      int conn,
                                      std::vector<T>& source_labels_out) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 1) { source_labels_out.clear(); return 0; }
    if (conn < 1) conn = 1;
    if (conn > ndim) conn = ndim;
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    if (total == 0) { source_labels_out.clear(); return 0; }

    std::vector<int64_t> strides, nb_fwd;
    std::vector<int8_t> nb_dc_fwd;
    detail::build_forward_neighbors(shape, conn, strides, nb_fwd, nb_dc_fwd);
    const int n_nbs = static_cast<int>(nb_fwd.size());

    cc_detail::UnionFind uf;
    uf.reserve(static_cast<size_t>(total) / 16 + 16);
    (void)uf.make_set();

    std::fill_n(output, total, int32_t{0});

    const int inner = ndim - 1;
    const int64_t W = shape[inner];

    // No inner-row fast path: the per-label union check has to read
    // input[] at every neighbor anyway, so the unrolled fg-only
    // variant in cc_label_nd doesn't help here.
    auto step_pixel_checked = [&](const int64_t* coords, int64_t flat) {
        const T cur = input[flat];
        if (cur == T{0}) return;  // bg
        int32_t best = 0;
        for (int k = 0; k < n_nbs; ++k) {
            const int8_t* dc = nb_dc_fwd.data() + k * ndim;
            bool valid = true;
            for (int d = 0; d < ndim; ++d) {
                if (dc[d] == 0) continue;
                const int64_t nc = coords[d] - dc[d];
                if (nc < 0 || nc >= shape[d]) { valid = false; break; }
            }
            if (!valid) continue;
            const int64_t nb_off = flat - nb_fwd[k];
            if (input[nb_off] != cur) continue;  // different label → no union
            const int32_t l = output[nb_off];
            if (l == 0) continue;
            if (best == 0) best = l;
            else if (best != l) uf.unite(best, l);
        }
        output[flat] = (best == 0) ? uf.make_set() : best;
    };

    constexpr int MAX_NDIM = 16;
    int64_t coords[MAX_NDIM] = {0};
    int64_t row_base = 0;
    const int64_t outer_total = (inner == 0) ? 1 : (total / W);

    for (int64_t outer_idx = 0; outer_idx < outer_total; ++outer_idx) {
        for (int64_t x = 0; x < W; ++x) {
            coords[inner] = x;
            step_pixel_checked(coords, row_base + x);
        }
        if (inner == 0) break;
        int d = inner - 1;
        ++coords[d];
        row_base += strides[d];
        while (coords[d] >= shape[d] && d > 0) {
            row_base -= coords[d] * strides[d];
            coords[d] = 0;
            --d;
            ++coords[d];
            row_base += strides[d];
        }
        if (coords[d] >= shape[d]) break;
    }

    // Flatten UF (path-halve every entry to its root) so pass 2 can
    // read directly from parent[].
    for (size_t i = 1; i < uf.parent.size(); ++i) {
        uf.parent[i] = uf.find(static_cast<int32_t>(i));
    }

    // Pass 2: provisional → final 1..N relabel, with run-length
    // coalescing on consecutive same-prov pixels (matches cc_label_nd).
    std::vector<int32_t> remap(uf.parent.size(), 0);
    int32_t next_label = 0;
    int32_t prev_prov = -1, prev_final = 0;
    for (int64_t flat = 0; flat < total; ++flat) {
        const int32_t prov = output[flat];
        if (prov == 0) { prev_prov = -1; continue; }
        if (prov == prev_prov) {
            output[flat] = prev_final;
            continue;
        }
        const int32_t root = uf.parent[prov];
        int32_t final_lab = remap[root];
        if (final_lab == 0) {
            final_lab = ++next_label;
            remap[root] = final_lab;
        }
        output[flat] = final_lab;
        prev_prov = prov;
        prev_final = final_lab;
    }

    // Source-label table: input value of each component, recorded on
    // first sight. Kept as a separate scan so pass 2 stays tight.
    // ``left`` lets us bail out as soon as every component has been
    // seen — for typical compact outputs that stops well before the
    // end of the image.
    source_labels_out.assign(static_cast<size_t>(next_label), T{0});
    if (next_label > 0) {
        std::vector<uint8_t> seen(static_cast<size_t>(next_label) + 1, 0u);
        int32_t left = next_label;
        for (int64_t flat = 0; flat < total && left > 0; ++flat) {
            const int32_t lab = output[flat];
            if (lab > 0 && !seen[lab]) {
                source_labels_out[lab - 1] = input[flat];
                seen[lab] = 1u;
                --left;
            }
        }
    }

    return next_label;
}


// Region properties for a labeled image (output of cc_label_nd or any
// dense 1..N labeling). Fills the four output arrays:
//   areas[i]      = pixel count of component (i + 1)
//   bbox_min[i*ndim + d], bbox_max[i*ndim + d]
//                 = inclusive min / exclusive max of axis d for cmp (i+1)
//   centroids_sum[i*ndim + d]
//                 = sum of axis-d coordinates for component (i+1)
//                   (caller divides by area to get the centroid)
//
// All arrays must be sized for n_labels components by the caller. Fills
// in a single raster pass — cache-friendly, no per-component data
// structures (no map / no list-of-pixels).
inline void regionprops_nd(const int32_t* labels, int32_t n_labels,
                           const std::vector<int64_t>& shape,
                           int64_t* areas,
                           int64_t* bbox_min, int64_t* bbox_max,
                           double* centroids_sum) {
    const int ndim = static_cast<int>(shape.size());
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    // Init bbox to opposite extremes; area/centroid to 0.
    for (int32_t i = 0; i < n_labels; ++i) {
        areas[i] = 0;
        for (int d = 0; d < ndim; ++d) {
            bbox_min[i * ndim + d] = std::numeric_limits<int64_t>::max();
            bbox_max[i * ndim + d] = std::numeric_limits<int64_t>::min();
            centroids_sum[i * ndim + d] = 0.0;
        }
    }
    std::vector<int64_t> coords(ndim, 0);
    for (int64_t flat = 0; flat < total; ++flat) {
        const int32_t lab = labels[flat];
        if (lab > 0 && lab <= n_labels) {
            const int32_t i = lab - 1;
            areas[i] += 1;
            for (int d = 0; d < ndim; ++d) {
                const int64_t c = coords[d];
                if (c < bbox_min[i * ndim + d]) bbox_min[i * ndim + d] = c;
                if (c >= bbox_max[i * ndim + d]) bbox_max[i * ndim + d] = c + 1;
                centroids_sum[i * ndim + d] += static_cast<double>(c);
            }
        }
        // Odometer advance.
        for (int d = ndim - 1; d >= 0; --d) {
            if (++coords[d] < shape[d]) break;
            coords[d] = 0;
        }
    }
    // Components with area=0 (none in input) get bbox cleared to 0.
    for (int32_t i = 0; i < n_labels; ++i) {
        if (areas[i] == 0) {
            for (int d = 0; d < ndim; ++d) {
                bbox_min[i * ndim + d] = 0;
                bbox_max[i * ndim + d] = 0;
            }
        }
    }
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_CC_LABEL_HPP
