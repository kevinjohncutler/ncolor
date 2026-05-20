/*
 * connect.hpp — adjacency-pair search for label images.
 *
 * Each worker scans a contiguous strip of the label image and records every
 * (lo, hi) label adjacency in a private linear-probing hashtable keyed by
 * (lo<<32 | hi). The per-thread tables are merged via log2(n_threads)
 * pairwise rounds, and the survivor is walked once to extract unique pairs.
 *
 * Public entry: ``find_pairs_nd_unpadded<T>(...)``.
 */

#ifndef NCOLOR_CONNECT_HPP
#define NCOLOR_CONNECT_HPP

#include <cstdint>
#include <vector>
#include <utility>
#include <atomic>

#if defined(_MSC_VER) && !defined(__clang__)
#  include <intrin.h>
#endif

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

// Portable count-trailing-zeros for non-zero uint32_t. clang/gcc use the
// builtin; MSVC has _BitScanForward (since VS2005). Caller must ensure
// ``m != 0`` — the result for 0 is undefined.
static inline int ctz32(uint32_t m) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long idx;
    _BitScanForward(&idx, m);
    return static_cast<int>(idx);
#else
    return __builtin_ctz(m);
#endif
}

// ``ForkJoinPool`` is declared at file scope in threadpool.h (vendored from
// edt). Bring it into our namespace so callers don't have to mix qualifiers.
using ::ForkJoinPool;

constexpr uint64_t HT_EMPTY = 0xFFFFFFFFFFFFFFFFull;
// Knuth's golden-ratio multiplicative hash (matches ncolor's @njit constant).
constexpr uint64_t HT_HASH_MUL = 11400714819323198485ull;

// Per-pair reducer for the boundary-weighted coloring path. Picks what
// statistic of (d_i + d_j) over the shared-boundary pixels to track.
// "Off" disables weighting and matches the default unweighted find_pairs.
enum class ReduceMode : int {
    Off       = 0,
    Min       = 1,   // min(d) — closest physical approach
    Max       = 2,   // max(d) — farthest contact point
    Mean      = 3,   // sum(d) + count → mean = sum/count
    Count     = 4,   // boundary length only (ignores distance)
    Harmonic  = 5,   // sum(1 / (1 + d)) — length AND closeness combined
    MeanInv   = 6,   // sum(1 / (1 + d)) / count — length-normalized harmonic
};
constexpr bool mode_uses_primary(ReduceMode m) {
    return m == ReduceMode::Min || m == ReduceMode::Max ||
           m == ReduceMode::Mean || m == ReduceMode::Harmonic ||
           m == ReduceMode::MeanInv;
}
constexpr bool mode_uses_count(ReduceMode m) {
    return m == ReduceMode::Mean || m == ReduceMode::Count ||
           m == ReduceMode::MeanInv;
}

// Realistic upper bound for ndim. Used to size stack-allocated coord
// arrays in the scan kernels. Limits exotic >16D inputs; the public
// dispatcher rejects ndim above this.
constexpr int FIND_PAIRS_MAX_NDIM = 16;

// Insert ``key`` into a power-of-two-sized linear-probing hashtable.
// ``ht_mask`` must be ``ht_size - 1``.  Idempotent (silently ignores duplicates).
inline void ht_insert(uint64_t* ht, uint64_t ht_mask, uint64_t key) {
    uint64_t h = (key * HT_HASH_MUL) & ht_mask;
    while (ht[h] != HT_EMPTY && ht[h] != key) {
        h = (h + 1) & ht_mask;
    }
    ht[h] = key;
}

// Merge all entries from ``src`` into ``dst``.  Both tables are size ht_size.
inline void ht_merge(const uint64_t* src, uint64_t* dst, uint64_t ht_size) {
    const uint64_t ht_mask = ht_size - 1;
    for (uint64_t h = 0; h < ht_size; ++h) {
        const uint64_t key = src[h];
        if (key == HT_EMPTY) continue;
        ht_insert(dst, ht_mask, key);
    }
}

// Templated variants that maintain optional parallel reducer arrays
// (``primary`` double-valued, ``counts`` int32) alongside the dedup
// table. The Mode template parameter selects which reducer to compute;
// branches are eliminated at compile time so the only cost is the
// updates actually needed for that mode.
//
// Per slot storage layouts (only fields used by the mode are touched):
//   Min:      primary holds min(d) seen so far. counts unused.
//   Max:      primary holds max(d). counts unused.
//   Mean:     primary holds sum(d). counts holds the pair-pixel count.
//   Count:    counts holds the count. primary unused.
//   Harmonic: primary holds sum(1 / (1 + d)). counts unused.
template <ReduceMode Mode>
inline void ht_insert_acc(uint64_t* ht, double* primary, int32_t* counts,
                          uint64_t ht_mask, uint64_t key, int32_t cost) {
    uint64_t h = (key * HT_HASH_MUL) & ht_mask;
    while (ht[h] != HT_EMPTY && ht[h] != key) {
        h = (h + 1) & ht_mask;
    }
    const bool is_new = (ht[h] == HT_EMPTY);
    if (is_new) ht[h] = key;

    if constexpr (Mode == ReduceMode::Min) {
        if (is_new || static_cast<double>(cost) < primary[h])
            primary[h] = static_cast<double>(cost);
    } else if constexpr (Mode == ReduceMode::Max) {
        if (is_new || static_cast<double>(cost) > primary[h])
            primary[h] = static_cast<double>(cost);
    } else if constexpr (Mode == ReduceMode::Mean) {
        if (is_new) { primary[h] = static_cast<double>(cost); counts[h] = 1; }
        else        { primary[h] += static_cast<double>(cost); counts[h] += 1; }
    } else if constexpr (Mode == ReduceMode::Count) {
        if (is_new) counts[h] = 1; else counts[h] += 1;
    } else if constexpr (Mode == ReduceMode::Harmonic) {
        const double contrib = 1.0 / (1.0 + static_cast<double>(cost));
        if (is_new) primary[h] = contrib; else primary[h] += contrib;
    } else if constexpr (Mode == ReduceMode::MeanInv) {
        const double contrib = 1.0 / (1.0 + static_cast<double>(cost));
        if (is_new) { primary[h] = contrib; counts[h] = 1; }
        else        { primary[h] += contrib; counts[h] += 1; }
    }
    // ReduceMode::Off: nothing else to do; key already inserted above.
}

template <ReduceMode Mode>
inline void ht_merge_acc(const uint64_t* src_ht,
                         const double* src_primary, const int32_t* src_counts,
                         uint64_t* dst_ht,
                         double* dst_primary, int32_t* dst_counts,
                         uint64_t ht_size) {
    const uint64_t ht_mask = ht_size - 1;
    for (uint64_t h = 0; h < ht_size; ++h) {
        const uint64_t key = src_ht[h];
        if (key == HT_EMPTY) continue;
        uint64_t dh = (key * HT_HASH_MUL) & ht_mask;
        while (dst_ht[dh] != HT_EMPTY && dst_ht[dh] != key) {
            dh = (dh + 1) & ht_mask;
        }
        const bool is_new = (dst_ht[dh] == HT_EMPTY);
        if (is_new) dst_ht[dh] = key;
        if constexpr (Mode == ReduceMode::Min) {
            if (is_new || src_primary[h] < dst_primary[dh]) dst_primary[dh] = src_primary[h];
        } else if constexpr (Mode == ReduceMode::Max) {
            if (is_new || src_primary[h] > dst_primary[dh]) dst_primary[dh] = src_primary[h];
        } else if constexpr (Mode == ReduceMode::Mean ||
                             Mode == ReduceMode::MeanInv) {
            if (is_new) { dst_primary[dh] = src_primary[h]; dst_counts[dh] = src_counts[h]; }
            else        { dst_primary[dh] += src_primary[h]; dst_counts[dh] += src_counts[h]; }
        } else if constexpr (Mode == ReduceMode::Count) {
            if (is_new) dst_counts[dh] = src_counts[h]; else dst_counts[dh] += src_counts[h];
        } else if constexpr (Mode == ReduceMode::Harmonic) {
            if (is_new) dst_primary[dh] = src_primary[h]; else dst_primary[dh] += src_primary[h];
        }
    }
}

// Backward-compat aliases (the old Min-only API surface).
inline void ht_insert_min(uint64_t* ht, int32_t* mins, uint64_t ht_mask,
                          uint64_t key, int32_t cost) {
    uint64_t h = (key * HT_HASH_MUL) & ht_mask;
    while (ht[h] != HT_EMPTY && ht[h] != key) {
        h = (h + 1) & ht_mask;
    }
    if (ht[h] == HT_EMPTY) {
        ht[h] = key;
        mins[h] = cost;
    } else if (cost < mins[h]) {
        mins[h] = cost;
    }
}

// =============================================================================
// ND unpadded find_pairs.
//
// One source-level entry point (`find_pairs_nd_unpadded`) handles any
// (ndim ≥ 2, conn ∈ [1, ndim]) combination via a single ND odometer
// kernel. Interior pixels (bnd_mask == 0) take the fast path that just
// adds pre-computed nb_flat[k] offsets; only edge pixels pay the per-axis
// bounds check.
//
// Forward-neighbor set: enumerate dc ∈ {-1,0,1}^ndim with
//   - 1 ≤ #(nonzero coords) ≤ conn  (Chebyshev radius / connectivity strength)
//   - lex-first nonzero coord is +1  (ensures neighbor has strictly greater
//     flat index in row-major layout, so each undirected adjacency is emitted
//     exactly once)
// Counts: 2D conn=1 → 2; 2D conn=2 → 4; 3D conn=1 → 3; conn=2 → 9; conn=3 → 13.

namespace detail {

// Count the forward neighbors produced by the (ndim, conn) connectivity.
// Caller uses this to size the per-thread hashtables in Solver. The full
// (dc, flat_offset) tuples come from build_forward_neighbors below; this
// is a cheap wrapper that just iterates the same odometer.
inline int64_t count_forward_neighbors(int ndim, int conn, int radius = 1) {
    if (ndim < 1 || radius < 1) return 0;
    int64_t n_fwd = 0;
    std::vector<int8_t> dc(ndim, (int8_t)-radius);
    while (true) {
        int n_nz = 0, first_nz = -1;
        for (int d = 0; d < ndim; ++d) if (dc[d] != 0) {
            if (first_nz < 0) first_nz = d;
            ++n_nz;
        }
        // Forward-only: first nonzero coord is positive. Generalizes the
        // radius=1 condition `dc[first_nz] == 1` so the predicate works
        // for any radius (positive offset in row-major order ⇒ greater
        // flat index ⇒ each undirected pair emitted at most once).
        if (n_nz > 0 && n_nz <= conn && first_nz >= 0 && dc[first_nz] > 0) ++n_fwd;
        int d = ndim - 1;
        while (d >= 0) {
            ++dc[d];
            if (dc[d] <= radius) break;
            dc[d] = (int8_t)-radius;
            --d;
        }
        if (d < 0) break;
    }
    return n_fwd;
}

// Generate the forward-neighbor set's (dc[0..ndim-1], flat_offset) tuples.
// Used by find_pairs_unpadded_impl.
inline void build_forward_neighbors(
        const std::vector<int64_t>& shape, int conn,
        std::vector<int64_t>& strides_out,
        std::vector<int64_t>& nb_flat_out,
        std::vector<int8_t>& nb_dc_out,
        int radius = 1) {
    const int ndim = static_cast<int>(shape.size());
    strides_out.assign(ndim, 1);
    for (int d = ndim - 2; d >= 0; --d) strides_out[d] = strides_out[d + 1] * shape[d + 1];
    nb_flat_out.clear();
    nb_dc_out.clear();
    if (radius < 1) return;
    // Sort offsets by Chebyshev distance ascending so r=1 1-NN neighbors
    // are emitted before r=2 gap-bridges. WP greedy commits early picks
    // firmly; we want physically-adjacent edges driving those, with the
    // wider-radius edges filling in afterwards.
    std::vector<std::tuple<int, int64_t, std::vector<int8_t>>> cands;
    std::vector<int8_t> dc(ndim, (int8_t)-radius);
    while (true) {
        int n_nz = 0, first_nz = -1, cheb = 0;
        for (int d = 0; d < ndim; ++d) {
            const int a = dc[d] < 0 ? -dc[d] : dc[d];
            if (a > cheb) cheb = a;
            if (dc[d] != 0) {
                if (first_nz < 0) first_nz = d;
                ++n_nz;
            }
        }
        // Forward-only: first nonzero coord positive. See count_forward_neighbors.
        if (n_nz > 0 && n_nz <= conn && first_nz >= 0 && dc[first_nz] > 0) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += static_cast<int64_t>(dc[d]) * strides_out[d];
            cands.emplace_back(cheb, off, dc);
        }
        int d = ndim - 1;
        while (d >= 0) {
            ++dc[d];
            if (dc[d] <= radius) break;
            dc[d] = (int8_t)-radius;
            --d;
        }
        if (d < 0) break;
    }
    // Stable sort by Chebyshev distance ascending (ties keep odometer order).
    std::stable_sort(cands.begin(), cands.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    for (auto& c : cands) {
        nb_flat_out.push_back(std::get<1>(c));
        for (int8_t v : std::get<2>(c)) nb_dc_out.push_back(v);
    }
}

}  // namespace detail

// Inner-axis fast scan: walk the open interval (x_start, x_end) of a
// row, emit forward-neighbor pairs to ``ht`` using pre-computed flat
// offsets. Templated on ``N_NBS`` so the per-pixel inner loop unrolls;
// the offsets are hoisted into local int64s so the compiler keeps them
// in registers across the x sweep.
template <typename T, int N_NBS, ReduceMode Mode = ReduceMode::Off>
static inline void scan_inner_axis_fast(
        const T* row, int64_t x_start, int64_t x_end,
        const int64_t* nb_flat, uint64_t* ht, uint64_t ht_mask,
        const int32_t* dist_row = nullptr,
        double* primary = nullptr, int32_t* counts = nullptr) {
    int64_t nb[N_NBS];
    for (int i = 0; i < N_NBS; ++i) nb[i] = nb_flat[i];
    for (int64_t x = x_start; x < x_end; ++x) {
        const T vi = row[x];
        if (vi == 0) continue;
        const T* p = row + x;
        int32_t di = 0;
        if constexpr (Mode != ReduceMode::Off) di = dist_row[x];
        // Compile-time-bounded; clang/gcc fully unroll the loop. MSVC
        // doesn't have a portable unroll pragma — its loop unroller
        // handles N_NBS ≤ 16 fine without a hint.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC unroll 16
#endif
        for (int k = 0; k < N_NBS; ++k) {
            const T vj = p[nb[k]];
            if (vj == 0 || vj == vi) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            const uint64_t key = (static_cast<uint64_t>(lo) << 32) |
                                 static_cast<uint64_t>(hi);
            if constexpr (Mode == ReduceMode::Off) {
                ht_insert(ht, ht_mask, key);
            } else {
                const int32_t dj = dist_row[x + nb[k]];
                ht_insert_acc<Mode>(ht, primary, counts, ht_mask, key, di + dj);
            }
        }
    }
}

// Runtime-N_NBS fallback for cases that don't hit the dispatch table
// (e.g. ndim ≥ 5 with custom conn). Identical body, just no unroll.
template <typename T, ReduceMode Mode = ReduceMode::Off>
static inline void scan_inner_axis_fast_runtime(
        const T* row, int64_t x_start, int64_t x_end,
        int n_nbs, const int64_t* nb_flat, uint64_t* ht, uint64_t ht_mask,
        const int32_t* dist_row = nullptr,
        double* primary = nullptr, int32_t* counts = nullptr) {
    for (int64_t x = x_start; x < x_end; ++x) {
        const T vi = row[x];
        if (vi == 0) continue;
        const T* p = row + x;
        int32_t di = 0;
        if constexpr (Mode != ReduceMode::Off) di = dist_row[x];
        for (int k = 0; k < n_nbs; ++k) {
            const T vj = p[nb_flat[k]];
            if (vj == 0 || vj == vi) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            const uint64_t key = (static_cast<uint64_t>(lo) << 32) |
                                 static_cast<uint64_t>(hi);
            if constexpr (Mode == ReduceMode::Off) {
                ht_insert(ht, ht_mask, key);
            } else {
                const int32_t dj = dist_row[x + nb_flat[k]];
                ht_insert_acc<Mode>(ht, primary, counts, ht_mask, key, di + dj);
            }
        }
    }
}

// Dispatch on the actual forward-neighbor counts produced by
// (ndim, conn): 2D conn=1 → 2; 2D conn=2 → 4; 3D conn=1 → 3; conn=2 → 9;
// conn=3 → 13. Other counts (5D+ or non-default conn) take the runtime
// fallback.
template <typename T, ReduceMode Mode = ReduceMode::Off>
static inline void scan_inner_axis_dispatch(
        const T* row, int64_t x_start, int64_t x_end,
        int n_nbs, const int64_t* nb_flat, uint64_t* ht, uint64_t ht_mask,
        const int32_t* dist_row = nullptr,
        double* primary = nullptr, int32_t* counts = nullptr) {
    switch (n_nbs) {
        case 2:  scan_inner_axis_fast<T, 2,  Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        case 3:  scan_inner_axis_fast<T, 3,  Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        case 4:  scan_inner_axis_fast<T, 4,  Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        // n_nbs=6: 2D conn=1 radius=2 (cross-shape gap-bridging).
        case 6:  scan_inner_axis_fast<T, 6,  Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        case 9:  scan_inner_axis_fast<T, 9,  Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        // n_nbs=12: 2D conn=2 radius=2 (square-shape gap-bridging) —
        // the default `connect_radius=2` augmented-graph path. Without
        // this case the inner-axis kernel fell into the runtime-N_NBS
        // fallback (no unroll), which dominated find_pairs cost.
        case 12: scan_inner_axis_fast<T, 12, Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        case 13: scan_inner_axis_fast<T, 13, Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        default: scan_inner_axis_fast_runtime<T, Mode>(row, x_start, x_end, n_nbs, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
    }
}

// Internal scan kernel: walks one strip of axis-0, emits forward-neighbor
// pairs into a single hashtable. Generic ND odometer — interior pixels use
// the pre-computed flat offsets in nb_flat; boundary pixels rebuild offsets
// per-axis (with optional wrap). `Wrap` is templated so the boundary path
// has no runtime cost when it's off.
template <typename T, bool Wrap = false, ReduceMode Mode = ReduceMode::Off>
inline void scan_band_unpadded(
        const T* lbl, const std::vector<int64_t>& shape,
        const int64_t* strides, const int64_t* nb_flat,
        const int8_t* nb_dc, int n_nbs,
        int64_t outer_start, int64_t outer_end,
        uint64_t* ht, uint64_t ht_mask,
        const int32_t* dist = nullptr,
        double* primary = nullptr, int32_t* counts = nullptr,
        int radius = 1) {
    const int ndim = static_cast<int>(shape.size());
    auto emit_pair = [ht_mask, primary, counts](uint64_t* h, T vi, T vj,
                                                  int32_t di, int32_t dj) {
        (void)primary; (void)counts;  // unused when Mode == Off
        if (vj == 0 || vj == vi) return;
        const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
        const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
        const uint64_t key = (lo << 32) | hi;
        if constexpr (Mode == ReduceMode::Off) {
            (void)di; (void)dj;
            ht_insert(h, ht_mask, key);
        } else {
            ht_insert_acc<Mode>(h, primary, counts, ht_mask, key, di + dj);
        }
    };
    // Boundary scan kernel.
    //   Wrap=false (default): out-of-bounds neighbors are skipped — matches
    //     the legacy padded-buffer behavior (no edges across the image edge).
    //   Wrap=true: out-of-bounds neighbors wrap to the opposite edge of the
    //     same axis (toroidal topology). For each OOB axis we recompute the
    //     wrapped coord and rebuild the flat offset directly, since the
    //     pre-computed nb_flat[k] assumed no wrap.
    auto scan_pixel_checked = [&](const int64_t* coords, uint32_t bnd_mask, int64_t flat) {
        const T vi = lbl[flat];
        if (vi == 0) return;
        int32_t di = 0;
        if constexpr (Mode != ReduceMode::Off) di = dist[flat];
        for (int k = 0; k < n_nbs; ++k) {
            const int8_t* dc = nb_dc + k * ndim;
            if constexpr (Wrap) {
                int64_t neigh_flat = 0;
                for (int d = 0; d < ndim; ++d) {
                    int64_t nc = coords[d] + dc[d];
                    if (nc < 0) nc += shape[d];
                    else if (nc >= shape[d]) nc -= shape[d];
                    neigh_flat += nc * strides[d];
                }
                int32_t dj = 0;
                if constexpr (Mode != ReduceMode::Off) dj = dist[neigh_flat];
                emit_pair(ht, vi, lbl[neigh_flat], di, dj);
            } else {
                bool valid = true;
                uint32_t m = bnd_mask;
                while (m) {
                    const int d = ctz32(m);
                    m &= m - 1;
                    const int64_t nc = coords[d] + dc[d];
                    if (nc < 0 || nc >= shape[d]) { valid = false; break; }
                }
                if (valid) {
                    int32_t dj = 0;
                    if constexpr (Mode != ReduceMode::Off) dj = dist[flat + nb_flat[k]];
                    emit_pair(ht, vi, lbl[flat + nb_flat[k]], di, dj);
                }
            }
        }
    };

    // Iterate over (outer coords) × (inner axis). The outer loop is an
    // odometer over axes [0 .. ndim-2]; for each outer state, the inner
    // axis (ndim-1) is walked as a tight contiguous run. When the outer
    // coords are all interior (outer_bnd == 0) and the inner axis is wide
    // enough (W ≥ 3) we get the same fast path the 2D/3D specializations
    // had: split inner axis into [0], [1, W-1), [W-1] and the middle slice
    // touches only the pre-computed nb_flat[k] offsets — no per-pixel
    // coord arithmetic, no boundary mask updates.
    const int inner = ndim - 1;
    const int64_t W = shape[inner];           // inner-axis length
    const uint32_t inner_bit = 1u << inner;
    int64_t coords[FIND_PAIRS_MAX_NDIM] = {0};
    coords[0] = outer_start;
    uint32_t outer_bnd = 0;                   // bnd mask for coords[0..ndim-2]
    for (int d = 0; d < inner; ++d) {
        if (coords[d] < radius || coords[d] >= shape[d] - radius) outer_bnd |= (1u << d);
    }
    // Compute base flat offset for (coords[0..ndim-2], inner=0).
    int64_t row_base = 0;
    for (int d = 0; d < inner; ++d) row_base += coords[d] * strides[d];
    const int64_t end_outer = outer_end;
    while (coords[0] < end_outer) {
        if (outer_bnd != 0 || W < 2 * radius + 1) {
            // Full per-pixel boundary checks across the entire inner axis.
            for (int64_t x = 0; x < W; ++x) {
                const uint32_t bnd = outer_bnd |
                    ((x < radius || x >= W - radius) ? inner_bit : 0u);
                coords[inner] = x;
                scan_pixel_checked(coords, bnd, row_base + x);
            }
        } else {
            // Outer coords all interior, W ≥ 2*radius+1: fast path on
            // the open interval (radius..W-radius) — pre-computed
            // flat offsets, no per-pixel coord arithmetic.
            for (int64_t x = 0; x < radius; ++x) {
                coords[inner] = x;
                scan_pixel_checked(coords, inner_bit, row_base + x);
            }
            if constexpr (Mode != ReduceMode::Off) {
                scan_inner_axis_dispatch<T, Mode>(
                    lbl + row_base, radius, W - radius, n_nbs, nb_flat, ht, ht_mask,
                    dist + row_base, primary, counts);
            } else {
                scan_inner_axis_dispatch<T>(
                    lbl + row_base, radius, W - radius, n_nbs, nb_flat, ht, ht_mask);
            }
            for (int64_t x = W - radius; x < W; ++x) {
                coords[inner] = x;
                scan_pixel_checked(coords, inner_bit, row_base + x);
            }
        }
        // Advance the outer odometer (axes [0 .. ndim-2]).
        if (ndim == 1) break;
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
        const bool is_bnd = (coords[d] < radius || coords[d] >= shape[d] - radius);
        if (is_bnd) outer_bnd |= (1u << d);
        else outer_bnd &= ~(1u << d);
    }
}

// Internal driver — generates neighbors, allocates per-thread HTs,
// parallel-scans dim-0 strips, merges. The public dispatcher below
// routes here. ``Wrap`` is templated so the wrap branch in
// scan_band_unpadded is compile-time-elided when not needed.
// ``Mode`` selects an optional per-pair reducer (min/mean/max/count/
// harmonic of d_i+d_j); when not Off, the per-thread HTs are paired
// with primary (double) and counts (int32) arrays. Outputs the
// reducer values via ``out_primary`` and ``out_counts`` (parallel to
// the returned pair list).
template <typename T, bool Wrap = false, ReduceMode Mode = ReduceMode::Off>
inline std::vector<std::pair<int32_t, int32_t>>
find_pairs_unpadded_impl(const T* lbl, const std::vector<int64_t>& shape,
                         int conn, uint64_t ht_size, int n_threads,
                         ForkJoinPool& pool,
                         const int32_t* dist = nullptr,
                         std::vector<double>* out_primary = nullptr,
                         std::vector<int32_t>* out_counts = nullptr,
                         int radius = 1,
                         std::vector<uint64_t>* ht_scratch = nullptr,
                         std::vector<double>*  primary_scratch = nullptr,
                         std::vector<int32_t>* counts_scratch = nullptr) {
    if (n_threads < 1) n_threads = 1;
    std::vector<int64_t> strides;
    std::vector<int64_t> nb_flat;
    std::vector<int8_t> nb_dc;
    detail::build_forward_neighbors(shape, conn, strides, nb_flat, nb_dc, radius);
    const int n_nbs = static_cast<int>(nb_flat.size());
    const uint64_t ht_mask = ht_size - 1;

    // Allocate (or reuse caller-provided scratch) for the per-thread
    // hashtables. delete[]/new[] of tens of MB on every call is a
    // measurable cost at high thread counts (~5 ms at 64 threads for
    // ht_size=65536), so callers running many find_pairs back-to-back
    // can pass persistent vectors to amortise it. The scan kernel
    // re-fills each thread's slice with HT_EMPTY before use, so
    // stale data between calls is safe.
    const size_t ht_total = static_cast<size_t>(n_threads) * ht_size;
    std::vector<uint64_t> ht_local;
    uint64_t* hts_ptr;
    if (ht_scratch) {
        if (ht_scratch->size() < ht_total) ht_scratch->resize(ht_total);
        hts_ptr = ht_scratch->data();
    } else {
        ht_local.resize(ht_total);
        hts_ptr = ht_local.data();
    }
    double*  primary_ptr = nullptr;
    int32_t* counts_ptr  = nullptr;
    std::vector<double>  primary_local;
    std::vector<int32_t> counts_local;
    if constexpr (mode_uses_primary(Mode)) {
        if (primary_scratch) {
            if (primary_scratch->size() < ht_total) primary_scratch->resize(ht_total);
            primary_ptr = primary_scratch->data();
        } else {
            primary_local.resize(ht_total);
            primary_ptr = primary_local.data();
        }
    }
    if constexpr (mode_uses_count(Mode)) {
        if (counts_scratch) {
            if (counts_scratch->size() < ht_total) counts_scratch->resize(ht_total);
            counts_ptr = counts_scratch->data();
        } else {
            counts_local.resize(ht_total);
            counts_ptr = counts_local.data();
        }
    }
    auto thread_primary = [&](int t) -> double* {
        return primary_ptr ? primary_ptr + static_cast<size_t>(t) * ht_size : nullptr;
    };
    auto thread_counts = [&](int t) -> int32_t* {
        return counts_ptr ? counts_ptr + static_cast<size_t>(t) * ht_size : nullptr;
    };

    if (n_threads == 1 || shape[0] < 2) {
        std::fill_n(hts_ptr, ht_size, HT_EMPTY);
        scan_band_unpadded<T, Wrap, Mode>(
            lbl, shape, strides.data(), nb_flat.data(),
            nb_dc.data(), n_nbs, 0, shape[0],
            hts_ptr, ht_mask, dist,
            thread_primary(0), thread_counts(0), radius);
    } else {
        // Phase 1: per-worker scan + first-touch HT (NUCA-local).
        std::atomic<int> next{0};
        const int64_t per = (shape[0] + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                uint64_t* ht = hts_ptr + static_cast<size_t>(t) * ht_size;
                std::fill_n(ht, ht_size, HT_EMPTY);
                // primary/counts only valid where ht[h] != HT_EMPTY; no init needed.
                const int64_t z0 = static_cast<int64_t>(t) * per;
                const int64_t z1 = std::min(z0 + per, shape[0]);
                if (z0 < z1) {
                    scan_band_unpadded<T, Wrap, Mode>(
                        lbl, shape, strides.data(), nb_flat.data(),
                        nb_dc.data(), n_nbs, z0, z1, ht, ht_mask,
                        dist, thread_primary(t), thread_counts(t), radius);
                }
            }
        });
        // Phase 2: pairwise tree merge.
        int stride = 1;
        while (stride < n_threads) {
            const int n_pairs = (n_threads + 2 * stride - 1) / (2 * stride);
            std::atomic<int> nx{0};
            pool.parallel([&]() {
                int p;
                while ((p = nx.fetch_add(1, std::memory_order_relaxed)) < n_pairs) {
                    const int dst = p * 2 * stride;
                    const int src = dst + stride;
                    if (src >= n_threads) continue;
                    uint64_t* dst_ht = hts_ptr + static_cast<size_t>(dst) * ht_size;
                    const uint64_t* src_ht = hts_ptr + static_cast<size_t>(src) * ht_size;
                    if constexpr (Mode == ReduceMode::Off) {
                        ht_merge(src_ht, dst_ht, ht_size);
                    } else {
                        ht_merge_acc<Mode>(
                            src_ht, thread_primary(src), thread_counts(src),
                            dst_ht, thread_primary(dst), thread_counts(dst),
                            ht_size);
                    }
                }
            });
            stride *= 2;
        }
    }
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(64);
    const uint64_t* root = hts_ptr;
    const double*  root_p = primary_ptr;
    const int32_t* root_c = counts_ptr;
    if constexpr (Mode != ReduceMode::Off) {
        if (out_primary) out_primary->clear();
        if (out_counts)  out_counts->clear();
    }
    for (uint64_t h = 0; h < ht_size; ++h) {
        const uint64_t key = root[h];
        if (key == HT_EMPTY) continue;
        out.emplace_back(static_cast<int32_t>(key >> 32),
                         static_cast<int32_t>(key & 0xFFFFFFFFull));
        if constexpr (mode_uses_primary(Mode)) {
            if (out_primary) out_primary->push_back(root_p[h]);
        }
        if constexpr (mode_uses_count(Mode)) {
            if (out_counts) out_counts->push_back(root_c[h]);
        }
    }
    return out;
}

// Public entry point — dispatches on ``wrap`` only (the kernel itself is
// fully ND). Returns ``{}`` for ndim ∉ [2, 16] or invalid conn.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_nd_unpadded(const T* lbl, const std::vector<int64_t>& shape,
                       int conn, uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool, bool wrap = false,
                       int radius = 1,
                       std::vector<uint64_t>* ht_scratch = nullptr) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 2 || ndim > FIND_PAIRS_MAX_NDIM || conn < 1 || conn > ndim) return {};
    if (radius < 1) radius = 1;
    return wrap
        ? find_pairs_unpadded_impl<T, true >(lbl, shape, conn, ht_size, n_threads, pool,
                                              nullptr, nullptr, nullptr, radius, ht_scratch)
        : find_pairs_unpadded_impl<T, false>(lbl, shape, conn, ht_size, n_threads, pool,
                                              nullptr, nullptr, nullptr, radius, ht_scratch);
}

// Weighted variant: returns adjacency pairs AND per-pair reducer
// values, computed in the SAME parallel scan as find_pairs (no extra
// traversal). ``Mode`` picks which reducer:
//   Min/Max:  primary[i] = min/max(d_i+d_j) over the pair's boundary.
//   Mean:     primary[i] = sum, counts[i] = N → mean = sum/N.
//   Count:    counts[i] = boundary pixel-pair count. primary unused.
//   Harmonic: primary[i] = Σ 1/(1+d_i+d_j) over the pair's boundary.
template <typename T, ReduceMode Mode>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_weighted_nd_unpadded(const T* lbl, const int32_t* dist,
                                const std::vector<int64_t>& shape,
                                int conn, uint64_t ht_size, int n_threads,
                                ForkJoinPool& pool, bool wrap,
                                std::vector<double>& out_primary,
                                std::vector<int32_t>& out_counts,
                                std::vector<uint64_t>* ht_scratch = nullptr,
                                std::vector<double>*  primary_scratch = nullptr,
                                std::vector<int32_t>* counts_scratch = nullptr) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 2 || ndim > FIND_PAIRS_MAX_NDIM || conn < 1 || conn > ndim) {
        out_primary.clear();
        out_counts.clear();
        return {};
    }
    return wrap
        ? find_pairs_unpadded_impl<T, true,  Mode>(
              lbl, shape, conn, ht_size, n_threads, pool, dist,
              &out_primary, &out_counts, /*radius=*/1,
              ht_scratch, primary_scratch, counts_scratch)
        : find_pairs_unpadded_impl<T, false, Mode>(
              lbl, shape, conn, ht_size, n_threads, pool, dist,
              &out_primary, &out_counts, /*radius=*/1,
              ht_scratch, primary_scratch, counts_scratch);
}


} // namespace ncolor_cpp

#endif // NCOLOR_CONNECT_HPP
