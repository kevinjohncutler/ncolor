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

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

// ``ForkJoinPool`` is declared at file scope in threadpool.h (vendored from
// edt). Bring it into our namespace so callers don't have to mix qualifiers.
using ::ForkJoinPool;

constexpr uint64_t HT_EMPTY = 0xFFFFFFFFFFFFFFFFull;
// Knuth's golden-ratio multiplicative hash (matches ncolor's @njit constant).
constexpr uint64_t HT_HASH_MUL = 11400714819323198485ull;

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

// =============================================================================
// ND unpadded find_pairs.
//
// One source-level entry point (`find_pairs_nd_unpadded`) handles any
// (ndim ≥ 2, conn ∈ [1, ndim]) combination via a single ND odometer
// kernel. Interior pixels (bnd_mask == 0) take the fast path that just
// adds pre-computed nb_flat[k] offsets; only edge pixels pay the per-axis
// bounds check.
//
// Forward-neighbour set: enumerate dc ∈ {-1,0,1}^ndim with
//   - 1 ≤ #(nonzero coords) ≤ conn  (Chebyshev radius / connectivity strength)
//   - lex-first nonzero coord is +1  (ensures neighbour has strictly greater
//     flat index in row-major layout, so each undirected adjacency is emitted
//     exactly once)
// Counts: 2D conn=1 → 2; 2D conn=2 → 4; 3D conn=1 → 3; conn=2 → 9; conn=3 → 13.

namespace detail {

// Count the forward neighbours produced by the (ndim, conn) connectivity.
// Caller uses this to size the per-thread hashtables in Solver. The full
// (dc, flat_offset) tuples come from build_forward_neighbours below; this
// is a cheap wrapper that just iterates the same odometer.
inline int64_t count_forward_neighbours(int ndim, int conn) {
    if (ndim < 1) return 0;
    int64_t n_fwd = 0;
    std::vector<int8_t> dc(ndim, -1);
    while (true) {
        int n_nz = 0, first_nz = -1;
        for (int d = 0; d < ndim; ++d) if (dc[d] != 0) {
            if (first_nz < 0) first_nz = d;
            ++n_nz;
        }
        if (n_nz > 0 && n_nz <= conn && first_nz >= 0 && dc[first_nz] == 1) ++n_fwd;
        int d = ndim - 1;
        while (d >= 0) {
            ++dc[d];
            if (dc[d] <= 1) break;
            dc[d] = -1;
            --d;
        }
        if (d < 0) break;
    }
    return n_fwd;
}

// Generate the forward-neighbour set's (dc[0..ndim-1], flat_offset) tuples.
// Used by find_pairs_unpadded_impl.
inline void build_forward_neighbours(
        const std::vector<int64_t>& shape, int conn,
        std::vector<int64_t>& strides_out,
        std::vector<int64_t>& nb_flat_out,
        std::vector<int8_t>& nb_dc_out) {
    const int ndim = static_cast<int>(shape.size());
    strides_out.assign(ndim, 1);
    for (int d = ndim - 2; d >= 0; --d) strides_out[d] = strides_out[d + 1] * shape[d + 1];
    nb_flat_out.clear();
    nb_dc_out.clear();
    std::vector<int8_t> dc(ndim, -1);
    while (true) {
        int n_nz = 0, first_nz = -1;
        for (int d = 0; d < ndim; ++d) if (dc[d] != 0) {
            if (first_nz < 0) first_nz = d;
            ++n_nz;
        }
        if (n_nz > 0 && n_nz <= conn && first_nz >= 0 && dc[first_nz] == 1) {
            int64_t off = 0;
            for (int d = 0; d < ndim; ++d) off += static_cast<int64_t>(dc[d]) * strides_out[d];
            nb_flat_out.push_back(off);
            for (int d = 0; d < ndim; ++d) nb_dc_out.push_back(dc[d]);
        }
        int d = ndim - 1;
        while (d >= 0) {
            ++dc[d];
            if (dc[d] <= 1) break;
            dc[d] = -1;
            --d;
        }
        if (d < 0) break;
    }
}

}  // namespace detail

// Inner-axis fast scan: walk the open interval (x_start, x_end) of a
// row, emit forward-neighbour pairs to ``ht`` using pre-computed flat
// offsets. Templated on ``N_NBS`` so the per-pixel inner loop unrolls;
// the offsets are hoisted into local int64s so the compiler keeps them
// in registers across the x sweep.
template <typename T, int N_NBS>
static inline void scan_inner_axis_fast(
        const T* row, int64_t x_start, int64_t x_end,
        const int64_t* nb_flat, uint64_t* ht, uint64_t ht_mask) {
    int64_t nb[N_NBS];
    for (int i = 0; i < N_NBS; ++i) nb[i] = nb_flat[i];
    for (int64_t x = x_start; x < x_end; ++x) {
        const T vi = row[x];
        if (vi == 0) continue;
        const T* p = row + x;
        // Compile-time-bounded; clang fully unrolls.
        #pragma GCC unroll 16
        for (int k = 0; k < N_NBS; ++k) {
            const T vj = p[nb[k]];
            if (vj == 0 || vj == vi) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            ht_insert(ht, ht_mask,
                (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
    }
}

// Runtime-N_NBS fallback for cases that don't hit the dispatch table
// (e.g. ndim ≥ 5 with custom conn). Identical body, just no unroll.
template <typename T>
static inline void scan_inner_axis_fast_runtime(
        const T* row, int64_t x_start, int64_t x_end,
        int n_nbs, const int64_t* nb_flat, uint64_t* ht, uint64_t ht_mask) {
    for (int64_t x = x_start; x < x_end; ++x) {
        const T vi = row[x];
        if (vi == 0) continue;
        const T* p = row + x;
        for (int k = 0; k < n_nbs; ++k) {
            const T vj = p[nb_flat[k]];
            if (vj == 0 || vj == vi) continue;
            const T lo = vi < vj ? vi : vj;
            const T hi = vi < vj ? vj : vi;
            ht_insert(ht, ht_mask,
                (static_cast<uint64_t>(lo) << 32) | static_cast<uint64_t>(hi));
        }
    }
}

// Dispatch on the actual forward-neighbour counts produced by
// (ndim, conn): 2D conn=1 → 2; 2D conn=2 → 4; 3D conn=1 → 3; conn=2 → 9;
// conn=3 → 13. Other counts (5D+ or non-default conn) take the runtime
// fallback.
template <typename T>
static inline void scan_inner_axis_dispatch(
        const T* row, int64_t x_start, int64_t x_end,
        int n_nbs, const int64_t* nb_flat, uint64_t* ht, uint64_t ht_mask) {
    switch (n_nbs) {
        case 2:  scan_inner_axis_fast<T, 2 >(row, x_start, x_end, nb_flat, ht, ht_mask); break;
        case 3:  scan_inner_axis_fast<T, 3 >(row, x_start, x_end, nb_flat, ht, ht_mask); break;
        case 4:  scan_inner_axis_fast<T, 4 >(row, x_start, x_end, nb_flat, ht, ht_mask); break;
        case 9:  scan_inner_axis_fast<T, 9 >(row, x_start, x_end, nb_flat, ht, ht_mask); break;
        case 13: scan_inner_axis_fast<T, 13>(row, x_start, x_end, nb_flat, ht, ht_mask); break;
        default: scan_inner_axis_fast_runtime<T>(row, x_start, x_end, n_nbs, nb_flat, ht, ht_mask); break;
    }
}

// Internal scan kernel: walks one strip of axis-0, emits forward-neighbour
// pairs into a single hashtable. Generic ND odometer — interior pixels use
// the pre-computed flat offsets in nb_flat; boundary pixels rebuild offsets
// per-axis (with optional wrap). `Wrap` is templated so the boundary path
// has no runtime cost when it's off.
template <typename T, bool Wrap = false>
inline void scan_band_unpadded(
        const T* lbl, const std::vector<int64_t>& shape,
        const int64_t* strides, const int64_t* nb_flat,
        const int8_t* nb_dc, int n_nbs,
        int64_t outer_start, int64_t outer_end,
        uint64_t* ht, uint64_t ht_mask) {
    const int ndim = static_cast<int>(shape.size());
    auto emit_pair = [ht_mask](uint64_t* h, T vi, T vj) {
        if (vj == 0 || vj == vi) return;
        const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
        const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
        ht_insert(h, ht_mask, (lo << 32) | hi);
    };
    // Boundary scan kernel.
    //   Wrap=false (default): out-of-bounds neighbours are skipped — matches
    //     the legacy padded-buffer behaviour (no edges across the image edge).
    //   Wrap=true: out-of-bounds neighbours wrap to the opposite edge of the
    //     same axis (toroidal topology). For each OOB axis we recompute the
    //     wrapped coord and rebuild the flat offset directly, since the
    //     pre-computed nb_flat[k] assumed no wrap.
    auto scan_pixel_checked = [&](const int64_t* coords, uint32_t bnd_mask, int64_t flat) {
        const T vi = lbl[flat];
        if (vi == 0) return;
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
                emit_pair(ht, vi, lbl[neigh_flat]);
            } else {
                bool valid = true;
                uint32_t m = bnd_mask;
                while (m) {
                    const int d = __builtin_ctz(m);
                    m &= m - 1;
                    const int64_t nc = coords[d] + dc[d];
                    if (nc < 0 || nc >= shape[d]) { valid = false; break; }
                }
                if (valid) emit_pair(ht, vi, lbl[flat + nb_flat[k]]);
            }
        }
    };

    // Iterate over (outer coords) × (inner axis). The outer loop is an
    // odometer over axes [0 .. ndim-2]; for each outer state, the inner
    // axis (ndim-1) is walked as a tight contiguous run. When the outer
    // coords are all interior (outer_bnd == 0) and the inner axis is wide
    // enough (W ≥ 3) we get the same fast path the 2D/3D specialisations
    // had: split inner axis into [0], [1, W-1), [W-1] and the middle slice
    // touches only the pre-computed nb_flat[k] offsets — no per-pixel
    // coord arithmetic, no boundary mask updates.
    constexpr int MAX_NDIM = 16;  // realistic upper bound; trips fast in practice
    const int inner = ndim - 1;
    const int64_t W = shape[inner];           // inner-axis length
    const uint32_t inner_bit = 1u << inner;
    int64_t coords[MAX_NDIM] = {0};
    coords[0] = outer_start;
    uint32_t outer_bnd = 0;                   // bnd mask for coords[0..ndim-2]
    for (int d = 0; d < inner; ++d) {
        if (coords[d] == 0 || coords[d] >= shape[d] - 1) outer_bnd |= (1u << d);
    }
    // Compute base flat offset for (coords[0..ndim-2], inner=0).
    int64_t row_base = 0;
    for (int d = 0; d < inner; ++d) row_base += coords[d] * strides[d];
    const int64_t end_outer = outer_end;
    while (coords[0] < end_outer) {
        if (outer_bnd != 0 || W < 3) {
            // Full per-pixel boundary checks across the entire inner axis.
            for (int64_t x = 0; x < W; ++x) {
                const uint32_t bnd = outer_bnd |
                    ((x == 0 || x == W - 1) ? inner_bit : 0u);
                coords[inner] = x;
                scan_pixel_checked(coords, bnd, row_base + x);
            }
        } else {
            // Outer coords all interior, W ≥ 3: fast path on the
            // open interval (0, W-1) — fully unrolled at the n_nbs the
            // (ndim, conn) connectivity actually produces.
            coords[inner] = 0;
            scan_pixel_checked(coords, inner_bit, row_base);
            scan_inner_axis_dispatch<T>(
                lbl + row_base, 1, W - 1, n_nbs, nb_flat, ht, ht_mask);
            coords[inner] = W - 1;
            scan_pixel_checked(coords, inner_bit, row_base + (W - 1));
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
        const bool is_bnd = (coords[d] == 0 || coords[d] >= shape[d] - 1);
        if (is_bnd) outer_bnd |= (1u << d);
        else outer_bnd &= ~(1u << d);
    }
}

// Internal driver — generates neighbours, allocates per-thread HTs,
// parallel-scans dim-0 strips, merges. The public dispatcher below
// routes here. ``Wrap`` is templated so the wrap branch in
// scan_band_unpadded is compile-time-elided when not needed.
template <typename T, bool Wrap = false>
inline std::vector<std::pair<int32_t, int32_t>>
find_pairs_unpadded_impl(const T* lbl, const std::vector<int64_t>& shape,
                         int conn, uint64_t ht_size, int n_threads,
                         ForkJoinPool& pool) {
    if (n_threads < 1) n_threads = 1;
    std::vector<int64_t> strides;
    std::vector<int64_t> nb_flat;
    std::vector<int8_t> nb_dc;
    detail::build_forward_neighbours(shape, conn, strides, nb_flat, nb_dc);
    const int n_nbs = static_cast<int>(nb_flat.size());
    const uint64_t ht_mask = ht_size - 1;

    std::unique_ptr<uint64_t[]> hts(new uint64_t[static_cast<size_t>(n_threads) * ht_size]);
    if (n_threads == 1 || shape[0] < 2) {
        std::fill_n(hts.get(), ht_size, HT_EMPTY);
        scan_band_unpadded<T, Wrap>(lbl, shape, strides.data(), nb_flat.data(),
                                    nb_dc.data(), n_nbs, 0, shape[0],
                                    hts.get(), ht_mask);
    } else {
        // Phase 1: per-worker scan + first-touch HT (NUCA-local).
        std::atomic<int> next{0};
        const int64_t per = (shape[0] + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                uint64_t* ht = hts.get() + static_cast<size_t>(t) * ht_size;
                std::fill_n(ht, ht_size, HT_EMPTY);
                const int64_t z0 = static_cast<int64_t>(t) * per;
                const int64_t z1 = std::min(z0 + per, shape[0]);
                if (z0 < z1) {
                    scan_band_unpadded<T, Wrap>(lbl, shape, strides.data(), nb_flat.data(),
                                                nb_dc.data(), n_nbs, z0, z1,
                                                ht, ht_mask);
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
                    uint64_t* dst_ht = hts.get() + static_cast<size_t>(dst) * ht_size;
                    const uint64_t* src_ht = hts.get() + static_cast<size_t>(src) * ht_size;
                    ht_merge(src_ht, dst_ht, ht_size);
                }
            });
            stride *= 2;
        }
    }
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(64);
    const uint64_t* root = hts.get();
    for (uint64_t h = 0; h < ht_size; ++h) {
        const uint64_t key = root[h];
        if (key == HT_EMPTY) continue;
        out.emplace_back(static_cast<int32_t>(key >> 32),
                         static_cast<int32_t>(key & 0xFFFFFFFFull));
    }
    return out;
}

// Public entry point — dispatches on ``wrap`` only (the kernel itself is
// fully ND). Returns ``{}`` for ndim ∉ [2, 16] or invalid conn.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_nd_unpadded(const T* lbl, const std::vector<int64_t>& shape,
                       int conn, uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool, bool wrap = false) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 2 || ndim > 16 || conn < 1 || conn > ndim) return {};
    return wrap
        ? find_pairs_unpadded_impl<T, true >(lbl, shape, conn, ht_size, n_threads, pool)
        : find_pairs_unpadded_impl<T, false>(lbl, shape, conn, ht_size, n_threads, pool);
}

} // namespace ncolor_cpp

#endif // NCOLOR_CONNECT_HPP
