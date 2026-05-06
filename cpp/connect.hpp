/*
 * connect.hpp — header-only C++ port of ncolor's `_search_hashset_parallel`.
 *
 * Drop-in replacement for the @njit kernel.  The Python/numba version pays
 * 14–43 ms per @njit(parallel=True) region launch on multi-CCX hosts (e.g.
 * Threadripper PRO 3995WX); ncolor.label issues 7+ such regions, swamping the
 * actual work for typical interactive image sizes.  This C++ implementation
 * avoids that by using a single persistent ThreadPool (one std::thread per
 * worker, lives across calls) plus a parallel pairwise tree-reduce of the
 * per-thread hashtables.
 *
 * Algorithm matches ncolor.color._search_hashset_parallel exactly:
 *   1. Each worker scans a contiguous strip of the padded label image and
 *      records every (lo, hi) label adjacency in a private linear-probing
 *      hashtable keyed by (lo<<32 | hi).
 *   2. The per-thread tables are merged via log2(n_threads) pairwise rounds.
 *   3. The surviving table (workers[0]) is walked once to extract the unique
 *      pairs as a (M, 2) int32 array.
 *
 * Public entry point: search_hashset_parallel<T>(line, total, nbs, ht_size,
 * pool) -> std::vector<std::pair<int32_t,int32_t>>.  T is the label dtype
 * (int32 typical).
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

// Per-thread scan: walk a strip of the padded label line and emit every
// distinct (lo, hi) adjacency into ``ht``.  ``nbs`` are the flat-index
// neighbor offsets for the connectivity (always positive on a padded image).
template <typename T>
void scan_strip(const T* line, int64_t start, int64_t end, int64_t total,
                const int64_t* nbs, int n_nbs,
                uint64_t* ht, uint64_t ht_mask) {
    for (int64_t i = start; i < end; ++i) {
        const T vi = line[i];
        if (vi == 0) continue;
        for (int n = 0; n < n_nbs; ++n) {
            const int64_t j = i + nbs[n];
            if (j < 0 || j >= total) continue;
            const T vj = line[j];
            if (vj == 0 || vj == vi) continue;
            const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
            const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
            const uint64_t key = (lo << 32) | hi;
            ht_insert(ht, ht_mask, key);
        }
    }
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

// Public entry point: scan + parallel tree-merge, returns the unique pairs.
// ``pool`` is reused across calls; constructing it once and passing it in
// amortizes the std::thread creation cost across many ncolor.label calls.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
search_hashset_parallel(const T* line, int64_t total,
                        const int64_t* nbs, int n_nbs,
                        uint64_t ht_size, int n_threads,
                        ForkJoinPool& pool) {
    const uint64_t ht_mask = ht_size - 1;

    // NUCA-correct allocation (see find_pairs_2d_unpadded for rationale):
    // workers first-touch their own hashtable slice with HT_EMPTY so pages
    // map to the worker's L2 cluster rather than the main thread's.
    std::unique_ptr<uint64_t[]> hts(new uint64_t[static_cast<size_t>(n_threads) * ht_size]);

    // Phase 1: each worker first-touches its own hashtable + scans its strip.
    {
        std::atomic<int> next{0};
        const int64_t strip = (total + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                uint64_t* ht = hts.get() + static_cast<size_t>(t) * ht_size;
                std::fill_n(ht, ht_size, HT_EMPTY);
                const int64_t s = static_cast<int64_t>(t) * strip;
                const int64_t e = std::min(s + strip, total);
                scan_strip<T>(line, s, e, total, nbs, n_nbs, ht, ht_mask);
            }
        });
    }

    // Phase 2: pairwise tree merge — log2(n_threads) parallel rounds, each
    // round folds the second half of every active group into the first half.
    int stride = 1;
    while (stride < n_threads) {
        const int n_pairs = (n_threads + 2 * stride - 1) / (2 * stride);
        std::atomic<int> next{0};
        pool.parallel([&]() {
            int p;
            while ((p = next.fetch_add(1, std::memory_order_relaxed)) < n_pairs) {
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

    // Phase 3: extract unique pairs from workers[0]'s table.
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

// =============================================================================
// Unified ND unpadded find_pairs.
//
// Replaces the previous 2D and 3D specialised variants. One source-level
// entry point (`find_pairs_nd_unpadded`) handles any (ndim ≥ 2,
// conn ∈ [1, ndim]) combination — same algorithm as the legacy padded
// `search_hashset_parallel` but skips the (s_0+2)*…  pad buffer +
// pad_nd_into pass.
//
// Implementation is template-specialised on NDIM at compile time so the
// nested boundary/interior loop structure unrolls cleanly. NDIM=2 and
// NDIM=3 cover the production cases (2D imaging + 3D volumetric) at full
// hand-coded performance; NDIM≥4 falls through to the runtime-odometer
// path inside the same template, keeping the source unified.
//
// Forward-neighbour set: enumerate dc ∈ {-1,0,1}^NDIM with
//   - 1 ≤ #(nonzero coords) ≤ conn  (Chebyshev radius / connectivity strength)
//   - lex-first nonzero coord is +1  (ensures neighbour has strictly greater
//     flat index in row-major layout, so each undirected adjacency is emitted
//     exactly once)
// Counts: 2D conn=1 → 2; 2D conn=2 → 4; 3D conn=1 → 3; conn=2 → 9; conn=3 → 13.

namespace detail {

// Generate the forward-neighbour set's (dc[0..ndim-1], flat_offset) tuples.
// Used by all NDIM specialisations of find_pairs_unpadded_impl.
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

// Internal scan kernel: walks one strip of axis-0, emits forward-neighbour
// pairs into a single hashtable. Templated on NDIM so 2D and 3D get hand-
// coded nested loops with explicit boundary/interior splits; NDIM>=4 falls
// through to the generic odometer path.
template <typename T, int NDIM>
inline void scan_band_unpadded(
        const T* lbl, const std::vector<int64_t>& shape,
        const int64_t* strides, const int64_t* nb_flat,
        const int8_t* nb_dc, int n_nbs,
        int64_t outer_start, int64_t outer_end,
        uint64_t* ht, uint64_t ht_mask) {
    auto emit_pair = [ht_mask](uint64_t* h, T vi, T vj) {
        if (vj == 0 || vj == vi) return;
        const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
        const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
        ht_insert(h, ht_mask, (lo << 32) | hi);
    };
    auto scan_pixel_checked = [&](const int64_t* coords, uint32_t bnd_mask, int64_t flat) {
        const T vi = lbl[flat];
        if (vi == 0) return;
        for (int k = 0; k < n_nbs; ++k) {
            const int8_t* dc = nb_dc + k * NDIM;
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
    };

    if constexpr (NDIM == 2) {
        const int64_t H = shape[0], W = shape[1];
        for (int64_t y = outer_start; y < outer_end; ++y) {
            const bool y_bnd = (y == 0) || (y == H - 1);
            const T* row = lbl + y * W;
            if (y_bnd || W < 3) {
                for (int64_t x = 0; x < W; ++x) {
                    const bool x_bnd = (x == 0) || (x == W - 1);
                    const uint32_t mask = (y_bnd ? 1u : 0u) | (x_bnd ? 2u : 0u);
                    int64_t coords[2] = {y, x};
                    scan_pixel_checked(coords, mask, y * W + x);
                }
                continue;
            }
            // y interior: x=0 edge, x in [1, W-2] interior, x=W-1 edge.
            int64_t coords[2] = {y, 0};
            scan_pixel_checked(coords, /*bnd=*/2u, y * W + 0);
            for (int64_t x = 1; x < W - 1; ++x) {
                const T vi = row[x];
                if (vi == 0) continue;
                const int64_t base = y * W + x;
                for (int k = 0; k < n_nbs; ++k) emit_pair(ht, vi, lbl[base + nb_flat[k]]);
            }
            coords[1] = W - 1;
            scan_pixel_checked(coords, /*bnd=*/2u, y * W + (W - 1));
        }
    } else if constexpr (NDIM == 3) {
        const int64_t D = shape[0], H = shape[1], W = shape[2];
        for (int64_t z = outer_start; z < outer_end; ++z) {
            const bool z_bnd = (z == 0) || (z == D - 1);
            if (z_bnd || H < 3 || W < 3) {
                for (int64_t y = 0; y < H; ++y)
                    for (int64_t x = 0; x < W; ++x) {
                        const uint32_t mask =
                            (z_bnd ? 1u : 0u) |
                            ((y == 0 || y == H - 1) ? 2u : 0u) |
                            ((x == 0 || x == W - 1) ? 4u : 0u);
                        int64_t coords[3] = {z, y, x};
                        scan_pixel_checked(coords, mask, (z * H + y) * W + x);
                    }
                continue;
            }
            // z interior: handle y=0, y interior, y=H-1
            for (int64_t y = 0; y < H; ++y) {
                const bool y_bnd = (y == 0) || (y == H - 1);
                if (y_bnd) {
                    for (int64_t x = 0; x < W; ++x) {
                        const uint32_t mask = 2u | ((x == 0 || x == W - 1) ? 4u : 0u);
                        int64_t coords[3] = {z, y, x};
                        scan_pixel_checked(coords, mask, (z * H + y) * W + x);
                    }
                    continue;
                }
                // z, y both interior: x=0 edge, x in [1, W-2] interior, x=W-1 edge.
                {
                    int64_t coords[3] = {z, y, 0};
                    scan_pixel_checked(coords, /*bnd=*/4u, (z * H + y) * W + 0);
                }
                {
                    const T* row = lbl + (z * H + y) * W;
                    const int64_t row_base = (z * H + y) * W;
                    for (int64_t x = 1; x < W - 1; ++x) {
                        const T vi = row[x];
                        if (vi == 0) continue;
                        for (int k = 0; k < n_nbs; ++k)
                            emit_pair(ht, vi, lbl[row_base + x + nb_flat[k]]);
                    }
                }
                {
                    int64_t coords[3] = {z, y, W - 1};
                    scan_pixel_checked(coords, /*bnd=*/4u, (z * H + y) * W + (W - 1));
                }
            }
        }
    } else {
        // NDIM >= 4: generic odometer fallback. Same correctness, ~10-20%
        // slower than the hand-coded NDIM=2/3 paths from the per-iteration
        // boundary-mask maintenance.
        const int ndim = NDIM;
        std::vector<int64_t> coords(ndim, 0);
        coords[0] = outer_start;
        uint32_t bnd_mask = 0;
        for (int d = 0; d < ndim; ++d) {
            if (coords[d] == 0 || coords[d] >= shape[d] - 1) bnd_mask |= (1u << d);
        }
        int64_t flat = outer_start * strides[0];
        const int64_t end_flat = outer_end * strides[0];
        while (flat < end_flat) {
            const T vi = lbl[flat];
            if (vi != 0) {
                if (bnd_mask == 0) {
                    for (int k = 0; k < n_nbs; ++k) emit_pair(ht, vi, lbl[flat + nb_flat[k]]);
                } else {
                    scan_pixel_checked(coords.data(), bnd_mask, flat);
                }
            }
            ++flat;
            int d = ndim - 1;
            ++coords[d];
            while (coords[d] >= shape[d] && d > 0) {
                coords[d] = 0;
                bnd_mask |= (1u << d);
                --d;
                ++coords[d];
            }
            if (coords[d] >= shape[d]) break;
            const bool is_bnd = (coords[d] == 0 || coords[d] >= shape[d] - 1);
            if (is_bnd) bnd_mask |= (1u << d);
            else bnd_mask &= ~(1u << d);
        }
    }
}

// Internal: per-NDIM driver — generates neighbours, allocates HTs, parallel-
// scans dim-0 strips, merges. The public dispatcher below routes here.
template <typename T, int NDIM>
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
        scan_band_unpadded<T, NDIM>(lbl, shape, strides.data(), nb_flat.data(),
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
                    scan_band_unpadded<T, NDIM>(lbl, shape, strides.data(), nb_flat.data(),
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

// Public entry point: dispatches on shape.size() to the compile-time-
// specialised NDIM=2/3 implementations or (rare) NDIM=4 generic path.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_nd_unpadded(const T* lbl, const std::vector<int64_t>& shape,
                       int conn, uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim < 2 || conn < 1 || conn > ndim) return {};
    switch (ndim) {
        case 2: return find_pairs_unpadded_impl<T, 2>(lbl, shape, conn, ht_size, n_threads, pool);
        case 3: return find_pairs_unpadded_impl<T, 3>(lbl, shape, conn, ht_size, n_threads, pool);
        case 4: return find_pairs_unpadded_impl<T, 4>(lbl, shape, conn, ht_size, n_threads, pool);
        default: return {};  // ndim ≥ 5: caller should fall back to padded path
    }
}

// Convenience wrapper: 2D specialisation. Kept for callers that have raw
// (H, W) without building a shape vector.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_2d_unpadded(const T* lbl, int64_t H, int64_t W,
                       uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool) {
    return find_pairs_nd_unpadded<T>(lbl, std::vector<int64_t>{H, W},
                                     /*conn=*/2, ht_size, n_threads, pool);
}


// Single-threaded variant (no pool, no merge) for the small-image case.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
search_hashset_serial(const T* line, int64_t total,
                      const int64_t* nbs, int n_nbs,
                      uint64_t ht_size) {
    const uint64_t ht_mask = ht_size - 1;
    std::vector<uint64_t> ht(ht_size, HT_EMPTY);
    scan_strip<T>(line, 0, total, total, nbs, n_nbs, ht.data(), ht_mask);

    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(64);
    for (uint64_t h = 0; h < ht_size; ++h) {
        const uint64_t key = ht[h];
        if (key == HT_EMPTY) continue;
        out.emplace_back(static_cast<int32_t>(key >> 32),
                         static_cast<int32_t>(key & 0xFFFFFFFFull));
    }
    return out;
}

} // namespace ncolor_cpp

#endif // NCOLOR_CONNECT_HPP
