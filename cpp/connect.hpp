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
#include <cstdlib>
#include <vector>
#include <utility>
#include <atomic>

#if defined(_MSC_VER) && !defined(__clang__)
#  include <intrin.h>
#endif

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "intrinsics.hpp"
#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

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

// Boundary masks use one bit per axis, covering NumPy's maximum rank.
constexpr int FIND_PAIRS_MAX_NDIM = 64;
inline void validate_neighborhood_ndim(int ndim) {
    if (ndim < 1 || ndim > FIND_PAIRS_MAX_NDIM)
        throw std::invalid_argument("neighborhood kernels require 1..64 dimensions");
}
inline void validate_neighbor_radius(int radius) {
    if (radius > 127)
        throw std::invalid_argument("neighbor radius must be <= 127");
}

// Bounded linear probe. Returns the slot index holding ``key`` (an
// existing match or the first EMPTY slot to claim), or ``ht_mask + 1``
// when the table is COMPLETELY full and ``key`` is absent.
//
// The bound (at most ht_size = ht_mask + 1 steps) is what makes the
// hashtable robust: the distinct-edge count is not bounded a priori by
// the table sizing heuristic (a dense ND Voronoi cell can be adjacent to
// far more neighbors than 2*n_fwd — see find_pairs_ in binding.cpp), so a
// table can legitimately fill to 100%. An unbounded `while` probe then
// spins forever on the next insert. Returning the "no slot" sentinel lets
// callers drop the key instead; find_pairs_ detects the full table
// (occupancy == ht_size) and retries with a doubled table, so the dropped
// edge is recovered. The common case (a few probes into a sub-50%-full
// table) costs one extra comparison per probe.
inline uint64_t ht_probe(const uint64_t* ht, uint64_t ht_mask, uint64_t key) {
    uint64_t h = (key * HT_HASH_MUL) & ht_mask;
    for (uint64_t p = 0; p <= ht_mask; ++p) {
        if (ht[h] == HT_EMPTY || ht[h] == key) return h;
        h = (h + 1) & ht_mask;
    }
    return ht_mask + 1;   // table full + key absent ⇒ caller drops & retries
}

// Insert ``key`` into a power-of-two-sized linear-probing hashtable.
// ``ht_mask`` must be ``ht_size - 1``.  Idempotent (silently ignores
// duplicates). Drops the key if the table is completely full (recovered
// by the find_pairs_ retry-on-full path).
inline void ht_insert(uint64_t* ht, uint64_t ht_mask, uint64_t key) {
    const uint64_t h = ht_probe(ht, ht_mask, key);
    if (h <= ht_mask) ht[h] = key;
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
    const uint64_t h = ht_probe(ht, ht_mask, key);
    if (h > ht_mask) return;   // table full ⇒ drop (recovered on retry)
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
        const uint64_t dh = ht_probe(dst_ht, ht_mask, key);
        if (dh > ht_mask) continue;   // dst full ⇒ drop (recovered on retry)
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
    const uint64_t h = ht_probe(ht, ht_mask, key);
    if (h > ht_mask) return;   // table full ⇒ drop (recovered on retry)
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

// Sum C(ndim,k) * (2*radius)^k / 2, k=1..conn. Build the
// weighted binomial terms with Pascal's recurrence to avoid intermediate
// overflow. Saturation is sufficient for hash-table sizing heuristics.
inline int64_t count_forward_neighbors(int ndim, int conn, int radius = 1) {
    if (ndim < 1 || radius < 1 || conn < 1) return 0;
    validate_neighborhood_ndim(ndim);
    validate_neighbor_radius(radius);
    conn = std::min(conn, ndim);
    // static: a constexpr local read inside a capture-less lambda is not
    // an odr-use, and clang and gcc accept it, but MSVC refuses to
    // compile it (C3493) and then cannot call the lambdas at all. A
    // static local needs no capture anywhere.
    static constexpr int64_t cap = std::numeric_limits<int64_t>::max() / 4;
    auto add = [](int64_t a, int64_t b) { return a > cap - b ? cap : a + b; };
    auto mul = [](int64_t a, int64_t b) { return a > cap / b ? cap : a * b; };
    std::vector<int64_t> terms(conn + 1, 0);
    terms[0] = 1;
    for (int d = 0; d < ndim; ++d)
        for (int k = std::min(conn, d + 1); k >= 1; --k)
            terms[k] = add(terms[k], mul(terms[k - 1], 2 * radius));
    int64_t count = 0;
    for (int k = 1; k <= conn; ++k) count = add(count, terms[k] / 2);
    return count;
}

inline int64_t count_forward_neighbors(const std::vector<int64_t>& shape,
                                      int conn, int radius = 1) {
    validate_neighborhood_ndim(static_cast<int>(shape.size()));
    const int active = static_cast<int>(std::count_if(shape.begin(), shape.end(),
        [](int64_t n) { return n > 1; }));
    return count_forward_neighbors(active, conn, radius);
}

// Visit only qualifying forward offsets, in the original odometer order.
// Zero-length/singleton axes cannot have a distinct neighbor. Pruning
// once conn nonzero coordinates are chosen makes face-only enumeration
// polynomial in rank rather than walking the entire (2r+1)^ndim cube.
template <typename Emit>
inline void for_each_forward_neighbor(const std::vector<int64_t>& shape,
                                     int conn, int radius, Emit emit) {
    const int ndim = static_cast<int>(shape.size());
    validate_neighborhood_ndim(ndim);
    validate_neighbor_radius(radius);
    if (radius < 1 || conn < 1) return;
    std::vector<int8_t> dc(ndim, 0);
    auto visit = [&](auto&& self, int d, int used, int cheb) -> void {
        if (d == ndim) {
            if (used) emit(dc, used, cheb);
            return;
        }
        if (shape[d] <= 1 || used == conn) {
            dc[d] = 0;
            self(self, d + 1, used, cheb);
            return;
        }
        // Before the first nonzero axis, only zero or positive offsets
        // can belong to the forward half of the neighborhood.
        for (int v = used ? -radius : 0; v <= radius; ++v) {
            dc[d] = static_cast<int8_t>(v);
            self(self, d + 1, used + (v != 0), std::max(cheb, std::abs(v)));
        }
    };
    visit(visit, 0, 0, 0);
}

// Generate the forward-neighbor set's (dc[0..ndim-1], flat_offset) tuples.
// Used by find_pairs_unpadded_impl. For the dual base+soft scan (the
// soft_extra_edges auto-builder), see build_forward_neighbors_dual.
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
    for_each_forward_neighbor(shape, conn, radius,
        [&](const std::vector<int8_t>& dc, int, int cheb) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += static_cast<int64_t>(dc[d]) * strides_out[d];
            cands.emplace_back(cheb, off, dc);
        });
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

// Dual-emit per-pixel inner-axis scan. ``vi`` is loaded ONCE per pixel,
// then BOTH the base offsets (0..N_BASE-1) and the delta offsets
// (N_BASE..N_BASE+N_DELTA-1) are checked in the same iteration. Loads of
// pixel data stay in registers/L1 for both checks. Templated on both
// counts so each loop body unrolls separately. Used by the soft
// auto-builder's fused dual scan; Mode=Off only.
//
// Interior skip. The delta offsets are ordered by Chebyshev distance,
// so the first ``n_delta_near`` of them are the distance-1 ones. When
// every distance-1 forward neighbor (base and delta) carries the
// pixel's own label, the distance-2 delta offsets are not read: any
// pair they could emit is also emitted from the stepping neighbor one
// step toward the far pixel, either as a hard pair or as a distance-1
// soft pair (the driver removes hard pairs from the soft set, which
// makes the two routes equivalent). Exact for soft radius <= 2; the
// driver disables the skip (n_delta_near == n_delta) beyond that. For
// a cell interior this replaces 12 reads by 4 in 2D and 33 by 9 in 3D.
template <typename T, int N_BASE, int N_DELTA>
static inline void scan_inner_axis_dual_fast(
        const T* row, int64_t x_start, int64_t x_end,
        const int64_t* nb_flat, int n_delta_near,
        uint64_t* ht_base, uint64_t base_mask,
        uint64_t* ht_soft, uint64_t soft_mask) {
    int64_t nb_b[N_BASE];
    int64_t nb_s[N_DELTA];
    for (int i = 0; i < N_BASE;  ++i) nb_b[i] = nb_flat[i];
    for (int i = 0; i < N_DELTA; ++i) nb_s[i] = nb_flat[N_BASE + i];
    for (int64_t x = x_start; x < x_end; ++x) {
        const T vi = row[x];
        if (vi == 0) continue;
        const T* p = row + x;
        bool all_same = true;
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC unroll 16
#endif
        for (int k = 0; k < N_BASE; ++k) {
            const T vj = p[nb_b[k]];
            if (vj == vi) continue;
            all_same = false;
            if (vj == 0) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            ht_insert(ht_base, base_mask,
                       (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC unroll 16
#endif
        for (int k = 0; k < N_DELTA; ++k) {
            if (k == n_delta_near && all_same) break;
            const T vj = p[nb_s[k]];
            if (vj == vi) continue;
            all_same = false;
            if (vj == 0) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            ht_insert(ht_soft, soft_mask,
                       (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
    }
}

// Runtime-N variant for (N_BASE, N_DELTA) combinations not in the
// dispatch table.
template <typename T>
static inline void scan_inner_axis_dual_runtime(
        const T* row, int64_t x_start, int64_t x_end,
        int n_base, int n_delta, const int64_t* nb_flat, int n_delta_near,
        uint64_t* ht_base, uint64_t base_mask,
        uint64_t* ht_soft, uint64_t soft_mask) {
    for (int64_t x = x_start; x < x_end; ++x) {
        const T vi = row[x];
        if (vi == 0) continue;
        const T* p = row + x;
        bool all_same = true;
        for (int k = 0; k < n_base; ++k) {
            const T vj = p[nb_flat[k]];
            if (vj == vi) continue;
            all_same = false;
            if (vj == 0) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            ht_insert(ht_base, base_mask,
                       (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
        for (int k = 0; k < n_delta; ++k) {
            if (k == n_delta_near && all_same) break;
            const T vj = p[nb_flat[n_base + k]];
            if (vj == vi) continue;
            all_same = false;
            if (vj == 0) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            ht_insert(ht_soft, soft_mask,
                       (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
    }
}

// Dispatch table for common (N_BASE, N_DELTA) pairings.
template <typename T>
static inline void scan_inner_axis_dual_dispatch(
        const T* row, int64_t x_start, int64_t x_end,
        int n_base, int n_delta, const int64_t* nb_flat, int n_delta_near,
        uint64_t* ht_base, uint64_t base_mask,
        uint64_t* ht_soft, uint64_t soft_mask) {
    // 2D conn=1 r=1 base (N_BASE=2) is the dominant case; delta sizes
    // 2/10 cover conn=1 r=2 or conn=2 r=1, and the default conn=2 r=2.
    // N_BASE=4, N_DELTA=8 covers a 2D conn=2 r=1 base widened to r=2.
    // In 3D a conn=1 r=1 base has 3 offsets; the common soft kernels add
    // 3 offsets (conn=1 r=2), 6 (conn=2 r=1), or 27 (conn=2 r=2).
    // The last count is 27, not 30: 30 is the size of the whole soft
    // kernel, including the 3 offsets already partitioned into the base.
    if (n_base == 2 && n_delta == 2)
        scan_inner_axis_dual_fast<T, 2, 2>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 2 && n_delta == 8)
        scan_inner_axis_dual_fast<T, 2, 8>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 2 && n_delta == 10)
        scan_inner_axis_dual_fast<T, 2, 10>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 4 && n_delta == 8)
        scan_inner_axis_dual_fast<T, 4, 8>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 3 && n_delta == 3)
        scan_inner_axis_dual_fast<T, 3, 3>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 3 && n_delta == 6)
        scan_inner_axis_dual_fast<T, 3, 6>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 3 && n_delta == 27)
        scan_inner_axis_dual_fast<T, 3, 27>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else if (n_base == 3 && n_delta == 30)
        scan_inner_axis_dual_fast<T, 3, 30>(row, x_start, x_end, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
    else
        scan_inner_axis_dual_runtime<T>(row, x_start, x_end, n_base, n_delta, nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
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
        // n_nbs=8 / n_nbs=10: 2D auto-soft delta scans. When the soft
        // kernel is one step richer than the base kernel, the delta
        // offset set typically has 8 (single-step extension) or 10
        // (conn=1 r=1 → conn=2 r=2) forward neighbors. Without these
        // cases the delta scan fell into the runtime fallback and ran
        // slower than the full-kernel scan it was meant to replace.
        case 8:  scan_inner_axis_fast<T, 8,  Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        case 10: scan_inner_axis_fast<T, 10, Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        // n_nbs=12: 2D conn=2 radius=2 (square-shape gap-bridging) —
        // the default `connect_radius=2` augmented-graph path. Without
        // this case the inner-axis kernel fell into the runtime-N_NBS
        // fallback (no unroll), which dominated find_pairs cost.
        case 12: scan_inner_axis_fast<T, 12, Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        case 13: scan_inner_axis_fast<T, 13, Mode>(row, x_start, x_end, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
        default: scan_inner_axis_fast_runtime<T, Mode>(row, x_start, x_end, n_nbs, nb_flat, ht, ht_mask, dist_row, primary, counts); break;
    }
}

// Internal scan kernel: walks a contiguous range of inner-axis lines and
// emits forward-neighbor pairs into a single hashtable. Generic ND odometer — interior pixels use
// the pre-computed flat offsets in nb_flat; boundary pixels rebuild offsets
// per-axis (with optional wrap). `Wrap` is templated so the boundary path
// has no runtime cost when it's off.
template <typename T, bool Wrap = false, ReduceMode Mode = ReduceMode::Off>
inline void scan_band_unpadded(
        const T* lbl, const std::vector<int64_t>& shape,
        const int64_t* strides, const int64_t* nb_flat,
        const int8_t* nb_dc, int n_nbs,
        int64_t line_start, int64_t line_end,
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
    auto scan_pixel_checked = [&](const int64_t* coords, uint64_t bnd_mask, int64_t flat) {
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
                    // A radius larger than the axis needs a real modulo,
                    // not a single wrap-around.
                    nc %= shape[d];
                    if (nc < 0) nc += shape[d];
                    neigh_flat += nc * strides[d];
                }
                int32_t dj = 0;
                if constexpr (Mode != ReduceMode::Off) dj = dist[neigh_flat];
                emit_pair(ht, vi, lbl[neigh_flat], di, dj);
            } else {
                bool valid = true;
                uint64_t m = bnd_mask;
                while (m) {
                    const int d = ctz_u64(m);
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
    const uint64_t inner_bit = uint64_t{1} << inner;
    // Every active outer coordinate is decoded below; the inner one
    // is assigned by the pixel loop. Do not clear unused rank capacity.
    int64_t coords[FIND_PAIRS_MAX_NDIM];
    int64_t q = line_start;
    for (int d = inner - 1; d >= 0; --d) {
        coords[d] = q % shape[d];
        q /= shape[d];
    }
    for (int64_t line = line_start; line < line_end; ++line) {
        uint64_t outer_bnd = 0;               // bnd mask for coords[0..ndim-2]
        for (int d = 0; d < inner; ++d) {
            if (coords[d] < radius || coords[d] >= shape[d] - radius)
                outer_bnd |= (uint64_t{1} << d);
        }
        // C-contiguous layout: each flattened outer coordinate owns one
        // complete W-element row.
        const int64_t row_base = line * W;
        if (outer_bnd != 0 || W < 2 * radius + 1) {
            // Full per-pixel boundary checks across the entire inner axis.
            for (int64_t x = 0; x < W; ++x) {
                const uint64_t bnd = outer_bnd |
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
        int d = inner - 1;
        while (d >= 0 && ++coords[d] >= shape[d]) {
            coords[d] = 0;
            --d;
        }
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

    int64_t n_lines = 1;
    for (size_t d = 0; d + 1 < shape.size(); ++d) n_lines *= shape[d];
    if (n_threads == 1 || n_lines < 2) {
        std::fill_n(hts_ptr, ht_size, HT_EMPTY);
        scan_band_unpadded<T, Wrap, Mode>(
            lbl, shape, strides.data(), nb_flat.data(),
            nb_dc.data(), n_nbs, 0, n_lines,
            hts_ptr, ht_mask, dist,
            thread_primary(0), thread_counts(0), radius);
    } else {
        // Phase 1: per-worker scan + first-touch HT (NUCA-local).
        std::atomic<int> next{0};
        const int64_t per = (n_lines + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                uint64_t* ht = hts_ptr + static_cast<size_t>(t) * ht_size;
                std::fill_n(ht, ht_size, HT_EMPTY);
                // primary/counts only valid where ht[h] != HT_EMPTY; no init needed.
                const int64_t line0 = static_cast<int64_t>(t) * per;
                const int64_t line1 = std::min(line0 + per, n_lines);
                if (line0 < line1) {
                    scan_band_unpadded<T, Wrap, Mode>(
                        lbl, shape, strides.data(), nb_flat.data(),
                        nb_dc.data(), n_nbs, line0, line1, ht, ht_mask,
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
// fully ND). Unsupported ranks raise instead of silently losing edges.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_nd_unpadded(const T* lbl, const std::vector<int64_t>& shape,
                       int conn, uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool, bool wrap = false,
                       int radius = 1,
                       std::vector<uint64_t>* ht_scratch = nullptr) {
    const int ndim = static_cast<int>(shape.size());
    validate_neighborhood_ndim(ndim);
    if (conn < 1 || conn > ndim) return {};
    if (radius < 1) radius = 1;
    return wrap
        ? find_pairs_unpadded_impl<T, true >(lbl, shape, conn, ht_size, n_threads, pool,
                                              nullptr, nullptr, nullptr, radius, ht_scratch)
        : find_pairs_unpadded_impl<T, false>(lbl, shape, conn, ht_size, n_threads, pool,
                                              nullptr, nullptr, nullptr, radius, ht_scratch);
}

// =============================================================================
// Dual-emit variant for the soft_extra_edges auto-builder.
//
// Walks the image ONCE and emits two pair lists in a single pass:
//   - base pairs from forward neighbors at (base_conn, base_radius)
//   - delta pairs from forward neighbors at (soft_conn, soft_radius)
//     EXCLUDING the base offsets — i.e. the soft-only set.
//
// Replaces the older "two separate find_pairs scans + set_difference"
// path used by the soft_extra_edges feature. Saves the second pixel walk
// and the second HT init/merge/extract pass.
//
// Implementation: build_forward_neighbors_dual builds one combined offset
// list at (soft_conn, soft_radius). build_forward_neighbors guarantees
// stable_sort by Chebyshev distance — but offsets in the base kernel can
// interleave with delta offsets (e.g. axial r=1 in base, axial r=2 in
// delta, diagonals r=1 also in delta). So we *partition* the combined
// list explicitly: base offsets first (n_base count), delta after, then
// run two inner-axis dispatch calls per row interval — first for base,
// second for delta. Pixel data stays in L1 between the two calls (rows
// are at most ~8 KB on our test images).

namespace detail {

// Build combined forward-neighbor list at (soft_conn, soft_radius),
// partitioned: base offsets (in (base_conn, base_radius)) first, then
// the delta offsets. Returns n_base via out param.
inline void build_forward_neighbors_dual(
        const std::vector<int64_t>& shape,
        int base_conn, int base_radius,
        int soft_conn, int soft_radius,
        std::vector<int64_t>& strides_out,
        std::vector<int64_t>& nb_flat_out,
        std::vector<int8_t>& nb_dc_out,
        int& n_base_out, int* n_delta_near_out = nullptr) {
    const int ndim = static_cast<int>(shape.size());
    if (n_delta_near_out) *n_delta_near_out = 0;
    strides_out.assign(ndim, 1);
    for (int d = ndim - 2; d >= 0; --d) strides_out[d] = strides_out[d + 1] * shape[d + 1];
    nb_flat_out.clear();
    nb_dc_out.clear();
    n_base_out = 0;
    if (soft_radius < 1) return;
    // Group 0 = base, group 1 = delta. cands[k] = (group, cheb, off, dc).
    // Sort by (group, cheb) so all base offsets precede all delta offsets,
    // and within each group, lower Chebyshev distance comes first.
    std::vector<std::tuple<int, int, int64_t, std::vector<int8_t>>> cands;
    for_each_forward_neighbor(shape, soft_conn, soft_radius,
        [&](const std::vector<int8_t>& dc, int n_nz, int cheb) {
            const bool in_base = (base_conn > 0 && base_radius >= 1 &&
                                   n_nz <= base_conn && cheb <= base_radius);
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += static_cast<int64_t>(dc[d]) * strides_out[d];
            cands.emplace_back(in_base ? 0 : 1, cheb, off, dc);
        });
    std::stable_sort(cands.begin(), cands.end(),
        [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
            return std::get<1>(a) < std::get<1>(b);
        });
    for (auto& c : cands) {
        if (std::get<0>(c) == 0) ++n_base_out;
        else if (std::get<1>(c) == 1 && n_delta_near_out) ++*n_delta_near_out;
        nb_flat_out.push_back(std::get<2>(c));
        for (int8_t v : std::get<3>(c)) nb_dc_out.push_back(v);
    }
}

}  // namespace detail

// Band scan emitting to two hashtables. Offsets 0..n_base-1 emit to
// ht_base, n_base..n_nbs-1 emit to ht_soft. Mode=Off only; weighted
// reducers are not supported on the dual path (not needed for soft).
template <typename T, bool Wrap = false>
inline void scan_band_unpadded_dual(
        const T* lbl, const std::vector<int64_t>& shape,
        const int64_t* strides, const int64_t* nb_flat,
        const int8_t* nb_dc, int n_base, int n_nbs, int n_delta_near,
        int64_t line_start, int64_t line_end,
        uint64_t* ht_base, uint64_t* ht_soft,
        uint64_t base_mask, uint64_t soft_mask,
        int radius) {
    const int ndim = static_cast<int>(shape.size());
    const int n_delta = n_nbs - n_base;
    auto emit = [](uint64_t* h, uint64_t mask, T vi, T vj) {
        if (vj == 0 || vj == vi) return;
        const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
        const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
        ht_insert(h, mask, (lo << 32) | hi);
    };
    // Same interior skip as the fast inner loop (see
    // scan_inner_axis_dual_fast); an out-of-bounds near neighbor counts
    // as "different" so the far offsets are still checked.
    const int n_near = n_base + n_delta_near;
    auto scan_pixel_checked = [&](const int64_t* coords, uint64_t bnd_mask, int64_t flat) {
        const T vi = lbl[flat];
        if (vi == 0) return;
        bool all_same = true;
        for (int k = 0; k < n_nbs; ++k) {
            if (k == n_near && all_same) break;
            const int8_t* dc = nb_dc + k * ndim;
            const bool is_base = (k < n_base);
            uint64_t* h = is_base ? ht_base : ht_soft;
            const uint64_t mask = is_base ? base_mask : soft_mask;
            if constexpr (Wrap) {
                int64_t neigh_flat = 0;
                for (int d = 0; d < ndim; ++d) {
                    int64_t nc = coords[d] + dc[d];
                    // A radius larger than the axis needs a real modulo,
                    // not a single wrap-around.
                    nc %= shape[d];
                    if (nc < 0) nc += shape[d];
                    neigh_flat += nc * strides[d];
                }
                const T vj = lbl[neigh_flat];
                if (vj != vi) all_same = false;
                emit(h, mask, vi, vj);
            } else {
                bool valid = true;
                uint64_t m = bnd_mask;
                while (m) {
                    const int d = ctz_u64(m);
                    m &= m - 1;
                    const int64_t nc = coords[d] + dc[d];
                    if (nc < 0 || nc >= shape[d]) { valid = false; break; }
                }
                if (valid) {
                    const T vj = lbl[flat + nb_flat[k]];
                    if (vj != vi) all_same = false;
                    emit(h, mask, vi, vj);
                } else {
                    all_same = false;
                }
            }
        }
    };
    const int inner = ndim - 1;
    const int64_t W = shape[inner];
    const uint64_t inner_bit = uint64_t{1} << inner;
    // Every active outer coordinate is decoded below; the inner one
    // is assigned by the pixel loop. Do not clear unused rank capacity.
    int64_t coords[FIND_PAIRS_MAX_NDIM];
    int64_t q = line_start;
    for (int d = inner - 1; d >= 0; --d) {
        coords[d] = q % shape[d];
        q /= shape[d];
    }
    for (int64_t line = line_start; line < line_end; ++line) {
        uint64_t outer_bnd = 0;
        for (int d = 0; d < inner; ++d) {
            if (coords[d] < radius || coords[d] >= shape[d] - radius)
                outer_bnd |= (uint64_t{1} << d);
        }
        const int64_t row_base = line * W;
        if (outer_bnd != 0 || W < 2 * radius + 1) {
            for (int64_t x = 0; x < W; ++x) {
                const uint64_t bnd = outer_bnd |
                    ((x < radius || x >= W - radius) ? inner_bit : 0u);
                coords[inner] = x;
                scan_pixel_checked(coords, bnd, row_base + x);
            }
        } else {
            for (int64_t x = 0; x < radius; ++x) {
                coords[inner] = x;
                scan_pixel_checked(coords, inner_bit, row_base + x);
            }
            // Fast inner-axis interval: single per-pixel walk that
            // checks base offsets AND delta offsets in one loop body
            // (vi loaded once, both inner sub-loops fully unrolled).
            // Saves one full pixel walk vs two separate dispatch calls.
            if (n_base > 0 && n_delta > 0) {
                scan_inner_axis_dual_dispatch<T>(
                    lbl + row_base, radius, W - radius, n_base, n_delta,
                    nb_flat, n_delta_near, ht_base, base_mask, ht_soft, soft_mask);
            } else if (n_base > 0) {
                scan_inner_axis_dispatch<T>(
                    lbl + row_base, radius, W - radius, n_base,
                    nb_flat, ht_base, base_mask);
            } else if (n_delta > 0) {
                scan_inner_axis_dispatch<T>(
                    lbl + row_base, radius, W - radius, n_delta,
                    nb_flat, ht_soft, soft_mask);
            }
            for (int64_t x = W - radius; x < W; ++x) {
                coords[inner] = x;
                scan_pixel_checked(coords, inner_bit, row_base + x);
            }
        }
        int d = inner - 1;
        while (d >= 0 && ++coords[d] >= shape[d]) {
            coords[d] = 0;
            --d;
        }
    }
}

// Driver: same parallel structure as find_pairs_unpadded_impl but with
// two hashtables per thread. Mode=Off only.
template <typename T, bool Wrap = false>
inline int find_pairs_dual_unpadded_impl(
        const T* lbl, const std::vector<int64_t>& shape,
        int base_conn, int base_radius, int soft_conn, int soft_radius,
        uint64_t base_ht_size, uint64_t soft_ht_size,
        int n_threads, ForkJoinPool& pool,
        std::vector<std::pair<int32_t, int32_t>>& out_base,
        std::vector<std::pair<int32_t, int32_t>>& out_soft,
        std::vector<uint64_t>* base_ht_scratch,
        std::vector<uint64_t>* soft_ht_scratch) {
    out_base.clear();
    out_soft.clear();
    if (n_threads < 1) n_threads = 1;
    std::vector<int64_t> strides;
    std::vector<int64_t> nb_flat;
    std::vector<int8_t> nb_dc;
    int n_base = 0;
    int n_delta_near = 0;
    detail::build_forward_neighbors_dual(
        shape, base_conn, base_radius, soft_conn, soft_radius,
        strides, nb_flat, nb_dc, n_base, &n_delta_near);
    const int n_nbs = static_cast<int>(nb_flat.size());
    if (n_nbs == 0) return 0;
    // The interior skip (scan_inner_axis_dual_fast) is exact only up to
    // soft radius 2: beyond that a far pair's witness chain can pass
    // through a third label. NCOLOR_NO_INTERIOR_SKIP=1 disables it for
    // A/B checks.
    static const bool no_skip = std::getenv("NCOLOR_NO_INTERIOR_SKIP") != nullptr;
    if (no_skip || soft_radius > 2) n_delta_near = n_nbs - n_base;
    const int radius = std::max(1, soft_radius);
    const uint64_t base_mask = base_ht_size - 1;
    const uint64_t soft_mask = soft_ht_size - 1;

    const size_t base_total = (size_t)n_threads * base_ht_size;
    const size_t soft_total = (size_t)n_threads * soft_ht_size;
    std::vector<uint64_t> base_local, soft_local;
    uint64_t* base_hts;
    uint64_t* soft_hts;
    if (base_ht_scratch) {
        if (base_ht_scratch->size() < base_total) base_ht_scratch->resize(base_total);
        base_hts = base_ht_scratch->data();
    } else { base_local.resize(base_total); base_hts = base_local.data(); }
    if (soft_ht_scratch) {
        if (soft_ht_scratch->size() < soft_total) soft_ht_scratch->resize(soft_total);
        soft_hts = soft_ht_scratch->data();
    } else { soft_local.resize(soft_total); soft_hts = soft_local.data(); }

    int64_t n_lines = 1;
    for (size_t d = 0; d + 1 < shape.size(); ++d) n_lines *= shape[d];
    if (n_threads == 1 || n_lines < 2) {
        std::fill_n(base_hts, base_ht_size, HT_EMPTY);
        std::fill_n(soft_hts, soft_ht_size, HT_EMPTY);
        scan_band_unpadded_dual<T, Wrap>(
            lbl, shape, strides.data(), nb_flat.data(),
            nb_dc.data(), n_base, n_nbs, n_delta_near, 0, n_lines,
            base_hts, soft_hts, base_mask, soft_mask, radius);
    } else {
        std::atomic<int> next{0};
        const int64_t per = (n_lines + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                uint64_t* hb = base_hts + (size_t)t * base_ht_size;
                uint64_t* hs = soft_hts + (size_t)t * soft_ht_size;
                std::fill_n(hb, base_ht_size, HT_EMPTY);
                std::fill_n(hs, soft_ht_size, HT_EMPTY);
                const int64_t line0 = (int64_t)t * per;
                const int64_t line1 = std::min(line0 + per, n_lines);
                if (line0 < line1) {
                    scan_band_unpadded_dual<T, Wrap>(
                        lbl, shape, strides.data(), nb_flat.data(),
                        nb_dc.data(), n_base, n_nbs, n_delta_near, line0, line1,
                        hb, hs, base_mask, soft_mask, radius);
                }
            }
        });
        // Tree-merge each HT family separately.
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
                    ht_merge(base_hts + (size_t)src * base_ht_size,
                              base_hts + (size_t)dst * base_ht_size, base_ht_size);
                    ht_merge(soft_hts + (size_t)src * soft_ht_size,
                              soft_hts + (size_t)dst * soft_ht_size, soft_ht_size);
                }
            });
            stride *= 2;
        }
    }
    out_base.reserve(64);
    for (uint64_t h = 0; h < base_ht_size; ++h) {
        const uint64_t key = base_hts[h];
        if (key == HT_EMPTY) continue;
        out_base.emplace_back((int32_t)(key >> 32),
                               (int32_t)(key & 0xFFFFFFFFull));
    }
    out_soft.reserve(64);
    uint64_t soft_occupancy = 0;
    for (uint64_t h = 0; h < soft_ht_size; ++h) {
        const uint64_t key = soft_hts[h];
        if (key == HT_EMPTY) continue;
        ++soft_occupancy;
        // A soft pair that is also a hard pair can never be violated; it
        // only distorts the soft weights. Dropping it also makes the soft
        // set independent of the interior skip above, which may or may
        // not have seen such a pair through a far offset.
        const uint64_t hb = ht_probe(base_hts, base_mask, key);
        if (hb <= base_mask && base_hts[hb] == key) continue;
        out_soft.emplace_back((int32_t)(key >> 32),
                               (int32_t)(key & 0xFFFFFFFFull));
    }
    // A completely full private table can drop later unseen keys. If any
    // private table filled, its complete key set also fills the fixed-size
    // root during the union merge, so root occupancy detects both scan-time
    // and merge-time overflow. Report each family independently; the caller
    // doubles only the table(s) that filled and reruns the fused scan.
    int full_mask = 0;
    if (out_base.size() == base_ht_size) full_mask |= 1;
    if (soft_occupancy == soft_ht_size) full_mask |= 2;
    return full_mask;
}

// Public entry — wraps Wrap dispatch.
template <typename T>
inline int find_pairs_dual_nd_unpadded(
        const T* lbl, const std::vector<int64_t>& shape,
        int base_conn, int base_radius, int soft_conn, int soft_radius,
        uint64_t base_ht_size, uint64_t soft_ht_size,
        int n_threads, ForkJoinPool& pool, bool wrap,
        std::vector<std::pair<int32_t, int32_t>>& out_base,
        std::vector<std::pair<int32_t, int32_t>>& out_soft,
        std::vector<uint64_t>* base_ht_scratch = nullptr,
        std::vector<uint64_t>* soft_ht_scratch = nullptr) {
    out_base.clear();
    out_soft.clear();
    const int ndim = static_cast<int>(shape.size());
    validate_neighborhood_ndim(ndim);
    if (base_conn < 1 || base_conn > ndim) return 0;
    if (soft_conn < 1 || soft_conn > ndim) return 0;
    if (wrap) {
        return find_pairs_dual_unpadded_impl<T, true>(
            lbl, shape, base_conn, base_radius, soft_conn, soft_radius,
            base_ht_size, soft_ht_size, n_threads, pool,
            out_base, out_soft, base_ht_scratch, soft_ht_scratch);
    } else {
        return find_pairs_dual_unpadded_impl<T, false>(
            lbl, shape, base_conn, base_radius, soft_conn, soft_radius,
            base_ht_size, soft_ht_size, n_threads, pool,
            out_base, out_soft, base_ht_scratch, soft_ht_scratch);
    }
}

// =============================================================================
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
                                int radius = 1,
                                std::vector<uint64_t>* ht_scratch = nullptr,
                                std::vector<double>*  primary_scratch = nullptr,
                                std::vector<int32_t>* counts_scratch = nullptr) {
    const int ndim = static_cast<int>(shape.size());
    validate_neighborhood_ndim(ndim);
    if (conn < 1 || conn > ndim) {
        out_primary.clear();
        out_counts.clear();
        return {};
    }
    if (radius < 1) radius = 1;
    return wrap
        ? find_pairs_unpadded_impl<T, true,  Mode>(
              lbl, shape, conn, ht_size, n_threads, pool, dist,
              &out_primary, &out_counts, radius,
              ht_scratch, primary_scratch, counts_scratch)
        : find_pairs_unpadded_impl<T, false, Mode>(
              lbl, shape, conn, ht_size, n_threads, pool, dist,
              &out_primary, &out_counts, radius,
              ht_scratch, primary_scratch, counts_scratch);
}


} // namespace ncolor_cpp

#endif // NCOLOR_CONNECT_HPP
