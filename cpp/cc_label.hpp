/*
 * cc_label.hpp — N-D connected-components labeling + minimal regionprops.
 *
 * Two-pass union-find (Wu et al. 2009-style) over an N-D foreground mask.
 * Backward-neighbor set is computed via the same odometer machinery as
 * connect.hpp, so connectivity (conn ∈ [1, ndim]) generalizes cleanly to
 * any ndim. Serial scans are also used within parallel slabs; cross-slab
 * boundary unions preserve the serial component numbering.
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
    validate_neighborhood_ndim(ndim);
    if (shape.size() > 1 && std::find(shape.begin(), shape.end(), 1) != shape.end()) {
        std::vector<int64_t> active;
        for (int64_t n : shape) if (n != 1) active.push_back(n);
        if (active.empty()) active.push_back(1);
        return cc_label_nd(input, output, active, conn, times);
    }
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
    constexpr int MAX_NDIM = FIND_PAIRS_MAX_NDIM;
    int64_t coords[MAX_NDIM];
    std::fill_n(coords, ndim, int64_t{0});
    uint64_t outer_bnd = 0;
    for (int d = 0; d < inner; ++d) {
        if (coords[d] == 0 || coords[d] >= shape[d] - 1) outer_bnd |= (uint64_t{1} << d);
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
            outer_bnd |= (uint64_t{1} << d);
            --d;
            ++coords[d];
            row_base += strides[d];
        }
        if (coords[d] >= shape[d]) break;
        const bool is_bnd = (coords[d] == 0 || coords[d] >= shape[d] - 1);
        if (is_bnd) outer_bnd |= (uint64_t{1} << d);
        else outer_bnd &= ~(uint64_t{1} << d);
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
    validate_neighborhood_ndim(ndim);
    if (shape.size() > 1 && std::find(shape.begin(), shape.end(), 1) != shape.end()) {
        std::vector<int64_t> active;
        for (int64_t n : shape) if (n != 1) active.push_back(n);
        if (active.empty()) active.push_back(1);
        return cc_label_per_label_nd(input, output, active, conn, source_labels_out);
    }
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

    constexpr int MAX_NDIM = FIND_PAIRS_MAX_NDIM;
    int64_t coords[MAX_NDIM];
    std::fill_n(coords, ndim, int64_t{0});
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


// Label contiguous slabs independently, merge only their shared boundaries,
// then remap local component IDs in parallel. Slab-local IDs follow raster
// order, so increasing (slab, local ID) gives the same final numbering as
// the serial scan regardless of worker scheduling.
template <typename T, bool PerLabel = false>
inline int32_t cc_label_parallel_nd(
        const T* input, int32_t* output, const std::vector<int64_t>& input_shape,
        int conn, ForkJoinPool& pool, int n_threads,
        std::vector<T>* sources = nullptr) {
    if constexpr (PerLabel) {
        if (!sources) throw std::invalid_argument("per-label components require a source table");
    }
    validate_neighborhood_ndim(static_cast<int>(input_shape.size()));
    std::vector<int64_t> shape;
    int64_t total = 1;
    for (int64_t extent : input_shape) {
        total *= extent;
        if (extent != 1) shape.push_back(extent);
    }
    if (shape.empty()) shape.push_back(1);
    conn = std::max(1, std::min(conn, static_cast<int>(shape.size())));
    auto serial = [&](const T* src, int32_t* dst,
                      const std::vector<int64_t>& dims, std::vector<T>& values) {
        if constexpr (PerLabel) return cc_label_per_label_nd(src, dst, dims, conn, values);
        else return cc_label_nd(src, dst, dims, conn);
    };
    const int target = static_cast<int>(std::min<int64_t>(
        std::max(1, n_threads), total / 131072));
    if (total < 262144 || shape.size() < 2 || target < 2) {
        std::vector<T> unused;
        return serial(input, output, shape, sources ? *sources : unused);
    }
    // Partition contiguous subvolumes within thin leading dimensions.
    // This uses all workers without transposing the input or changing its
    // raster order. Prefix coordinates belong to separate subvolumes.
    int axis = 0;
    int64_t prefixes = 1;
    while (axis + 1 < static_cast<int>(shape.size()) &&
           prefixes * shape[axis] < target) {
        prefixes *= shape[axis++];
    }
    // Extra cuts help face-connected, thin volumes only when each slab
    // still contains many rows. Dense diagonal seams cost more to merge
    // than the additional workers save; retain their original partition.
    if (axis != 1 || conn != 1 || shape[axis] < 32 || target / prefixes < 2) {
        axis = 0;
        prefixes = 1;
    }
    const int parts = static_cast<int>(std::min<int64_t>(shape[axis],
        std::max<int64_t>(1, target / prefixes)));
    const int slabs = static_cast<int>(prefixes) * parts;
    const int64_t plane = total / (prefixes * shape[axis]);
    std::vector<int64_t> starts(slabs + 1);
    std::vector<int32_t> counts(slabs), bases(slabs + 1, 0);
    std::vector<std::vector<T>> local_sources(slabs);
    for (int s = 0; s < slabs; ++s)
        starts[s] = ((s / parts) * shape[axis] +
            shape[axis] * (s % parts) / parts) * plane;
    starts[slabs] = total;
    dispatch_parallel(pool, slabs, slabs, [&](size_t lo, size_t hi) {
        for (size_t s = lo; s < hi; ++s) {
            std::vector<int64_t> dims(shape.begin() + axis, shape.end());
            dims[0] = (starts[s + 1] - starts[s]) / plane;
            counts[s] = serial(input + starts[s], output + starts[s],
                               dims, local_sources[s]);
        }
    });
    int64_t provisional = 0;
    for (int s = 0; s < slabs; ++s) {
        provisional += counts[s];
        if (provisional >= std::numeric_limits<int32_t>::max())
            throw std::overflow_error("component count exceeds int32 capacity");
        bases[s + 1] = static_cast<int32_t>(provisional);
    }
    cc_detail::UnionFind uf;

    std::vector<int64_t> strides, flat;
    std::vector<int8_t> offsets;
    detail::build_forward_neighbors(shape, conn, strides, flat, offsets);
    const int ndim = static_cast<int>(shape.size());
    if (axis == 0) {
        std::vector<int> crossing;
        for (size_t k = 0; k < flat.size(); ++k)
            if (offsets[k * ndim] == 1) crossing.push_back(static_cast<int>(k));
        // Clip outer coordinates once per row. Only the two row endpoints
        // need an inner-axis bounds check; interior pixels reuse this list.
        std::vector<int64_t> point(ndim, 0);
        std::vector<int> row_crossing;
        row_crossing.reserve(crossing.size());
        const int64_t width = shape.back();
        for (int s = 1; s < slabs; ++s) {
            std::fill(point.begin(), point.end(), 0);
            const int64_t first = starts[s] - plane;
            int32_t previous_a = 0, previous_b = 0;
            for (int64_t row = 0; row < plane; row += width) {
                row_crossing.clear();
                for (int k : crossing) {
                    bool valid = true;
                    for (int d = 1; d < ndim - 1; ++d) {
                        const int64_t q = point[d] + offsets[k * ndim + d];
                        if (q < 0 || q >= shape[d]) { valid = false; break; }
                    }
                    if (valid) row_crossing.push_back(k);
                }
                auto merge_pixel = [&](int64_t x, bool boundary) {
                    const int64_t at = first + row + x;
                    const int32_t a = output[at];
                    if (!a) return;
                    for (int k : row_crossing) {
                        if (boundary) {
                            const int64_t q = x + offsets[k * ndim + ndim - 1];
                            if (q < 0 || q >= width) continue;
                        }
                        const int64_t neighbor = at + flat[k];
                        const int32_t b = output[neighbor];
                        if (!b) continue;
                        if constexpr (PerLabel) if (input[at] != input[neighbor]) continue;
                        // Repeated local component pairs need only one union.
                        if (a == previous_a && b == previous_b) continue;
                        previous_a = a;
                        previous_b = b;
                        if (uf.parent.empty()) {
                            uf.parent.resize(static_cast<size_t>(provisional) + 1);
                            uf.rank_.assign(uf.parent.size(), 0);
                            for (size_t id = 0; id < uf.parent.size(); ++id)
                                uf.parent[id] = static_cast<int32_t>(id);
                        }
                        uf.unite(bases[s - 1] + a, bases[s] + b);
                    }
                };
                merge_pixel(0, true);
                for (int64_t x = 1; x < width - 1; ++x) merge_pixel(x, false);
                merge_pixel(width - 1, true);
                for (int d = ndim - 2; d > 0; --d) {
                    if (++point[d] < shape[d]) break;
                    point[d] = 0;
                }
            }
        }
    } else {
        auto merge = [&](int64_t at, int64_t neighbor, int sa, int sb) {
            const int32_t a = output[at], b = output[neighbor];
            if (!a || !b) return;
            if constexpr (PerLabel) if (input[at] != input[neighbor]) return;
            if (uf.parent.empty()) {
                uf.parent.resize(static_cast<size_t>(provisional) + 1);
                uf.rank_.assign(uf.parent.size(), 0);
                for (size_t id = 0; id < uf.parent.size(); ++id)
                    uf.parent[id] = static_cast<int32_t>(id);
            }
            uf.unite(bases[sa] + a, bases[sb] + b);
        };
        // This adaptive path uses face connectivity. Across a prefix plane,
        // both endpoints have the same inner partition; across an inner cut,
        // they belong to consecutive partitions in the same prefix plane.
        for (int prefix = 1; prefix < static_cast<int>(prefixes); ++prefix) {
            for (int part = 0; part < parts; ++part) {
                const int lower = (prefix - 1) * parts + part;
                for (int64_t at = starts[lower]; at < starts[lower + 1]; ++at)
                    merge(at, at + shape[1] * plane, lower, lower + parts);
            }
        }
        for (int prefix = 0; prefix < static_cast<int>(prefixes); ++prefix) {
            for (int part = 1; part < parts; ++part) {
                const int upper = prefix * parts + part;
                for (int64_t i = 0; i < plane; ++i)
                    merge(starts[upper] - plane + i, starts[upper] + i, upper - 1, upper);
            }
        }
    }
    // Disconnected slabs need only an ID offset. Avoid a global union-find
    // for fragmented masks whose components never cross a slab boundary.
    if (uf.parent.empty()) {
        if constexpr (PerLabel) {
            sources->clear();
            for (const auto& local : local_sources)
                sources->insert(sources->end(), local.begin(), local.end());
        }
        dispatch_parallel(pool, slabs, slabs, [&](size_t lo, size_t hi) {
            for (size_t s = lo; s < hi; ++s)
                for (int64_t i = starts[s]; i < starts[s + 1]; ++i)
                    if (output[i]) output[i] += bases[s];
        });
        return static_cast<int32_t>(provisional);
    }
    // Reuse union-find storage for the root-to-final and local-to-final
    // tables. No extra full component-sized remapping arrays are needed.
    for (size_t i = 1; i < uf.parent.size(); ++i)
        uf.parent[i] = uf.find(static_cast<int32_t>(i));
    std::fill(uf.rank_.begin(), uf.rank_.end(), 0);
    auto& root_ids = uf.rank_;
    auto& remap = uf.parent;
    int32_t count = 0;
    if constexpr (PerLabel) sources->clear();
    for (int s = 0; s < slabs; ++s) {
        for (int32_t local = 1; local <= counts[s]; ++local) {
            const int32_t id = bases[s] + local;
            const int32_t root = remap[id];
            if (!root_ids[root]) {
                root_ids[root] = ++count;
                if constexpr (PerLabel) sources->push_back(local_sources[s][local - 1]);
            }
            remap[id] = root_ids[root];
        }
    }
    dispatch_parallel(pool, slabs, slabs, [&](size_t lo, size_t hi) {
        for (size_t s = lo; s < hi; ++s) {
            const int32_t* table = remap.data() + bases[s];
            for (int64_t i = starts[s]; i < starts[s + 1]; ++i)
                if (output[i]) output[i] = table[output[i]];
        }
    });
    return count;
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
