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

// Find label adjacency pairs by scanning a 2D row-major image directly,
// skipping the padded-buffer construction that the original
// search_hashset_parallel needs. Each thread takes a strip of rows; for each
// pixel, compares to its right and down neighbours (lower-flat-index pair
// emission rule from the original kernel still holds because neighbours are
// strictly forward in row-major order). Per-thread hashtable + parallel
// pairwise tree merge as before.
//
// `lbl` is a row-major (H, W) int32 buffer of fully-labeled pixels (every
// nonzero entry is a label; entries == 0 are skipped). This is the typical
// post-expand_labels output. Use this instead of search_hashset_parallel
// to save the ~1ms padding step at 2048².
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_2d_unpadded(const T* lbl, int64_t H, int64_t W,
                       uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool) {
    const uint64_t ht_mask = ht_size - 1;
    if (n_threads < 1) n_threads = 1;

    // NUCA-correct allocation: skip std::vector's value-init (which would
    // first-touch every page from the main thread, mapping all hashtables
    // into a single L2 cluster on multi-die hosts like M1 Ultra). Instead
    // allocate uninitialised, then have each worker fill its own slice
    // with HT_EMPTY — first-touch from the worker pins the pages into
    // that worker's L2/CCD-local memory. Eliminates ~5 ms cross-cluster
    // latency on M1 Ultra (16 P-cores in 4 L2 clusters of 4); negligible
    // effect on monolithic-L3 hosts (AMD 7950X, Intel desktop).
    std::unique_ptr<uint64_t[]> hts(new uint64_t[static_cast<size_t>(n_threads) * ht_size]);

    // Phase 1: parallel scan over row-bands. Per-pixel: check the 4 forward
    // neighbours (right, down-left, down, down-right) — this matches numba
    // ncolor's conn=2 (8-connectivity, the default for `ncolor.label`). With
    // unique=True the neighbour set is just the forward half.
    auto emit_pair = [ht_mask](uint64_t* ht, T vi, T vj) {
        if (vj == 0 || vj == vi) return;
        const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
        const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
        ht_insert(ht, ht_mask, (lo << 32) | hi);
    };
    auto scan_band = [&](int64_t y0, int64_t y1, uint64_t* ht) {
        for (int64_t y = y0; y < y1; ++y) {
            const T* row  = lbl + y * W;
            const T* rowd = (y + 1 < H) ? lbl + (y + 1) * W : nullptr;
            for (int64_t x = 0; x < W; ++x) {
                const T vi = row[x];
                if (vi == 0) continue;
                if (x + 1 < W) emit_pair(ht, vi, row[x + 1]);  // right
                if (rowd) {
                    if (x > 0)       emit_pair(ht, vi, rowd[x - 1]);  // down-left
                                      emit_pair(ht, vi, rowd[x]);      // down
                    if (x + 1 < W)   emit_pair(ht, vi, rowd[x + 1]);  // down-right
                }
            }
        }
    };

    if (n_threads == 1 || H < 4) {
        std::fill_n(hts.get(), ht_size, HT_EMPTY);
        scan_band(0, H, hts.get());
    } else {
        // Phase 1: each worker first-touches its own hashtable slice with
        // HT_EMPTY (NUCA-local) before scanning. Combining init+scan in one
        // dispatch avoids an extra pool.parallel barrier roundtrip.
        {
            std::atomic<int> next{0};
            const int64_t per = (H + n_threads - 1) / n_threads;
            pool.parallel([&]() {
                int t;
                while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                    uint64_t* ht = hts.get() + static_cast<size_t>(t) * ht_size;
                    std::fill_n(ht, ht_size, HT_EMPTY);
                    const int64_t y0 = static_cast<int64_t>(t) * per;
                    const int64_t y1 = std::min(y0 + per, H);
                    if (y0 < y1) scan_band(y0, y1, ht);
                }
            });
        }

        // Phase 2: pairwise tree merge.
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
    }

    // Extract.
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

// 3D unpadded variant (analogous to find_pairs_2d_unpadded but for 3D
// inputs). Skips the (D+2)(H+2)(W+2) padded buffer and the pad_nd_into
// pass — saves ~4 ms at 256³ uint16 — by doing per-pixel bounds checks
// against (D, H, W) on each forward-neighbour lookup.
//
// Forward-neighbour set encodes ``ncolor.color.connect(conn)``:
//   conn=1 (face):           3 nbs   ( +z, +y, +x )
//   conn=2 (face + edge):    9 nbs
//   conn=3 (face + corner): 13 nbs
// Each chosen offset has a strictly forward flat-index, so emitting
// (vi, neighbour) only once per i suffices to count each pair once.
template <typename T>
std::vector<std::pair<int32_t, int32_t>>
find_pairs_3d_unpadded(const T* lbl, int64_t D, int64_t H, int64_t W,
                       int conn, uint64_t ht_size, int n_threads,
                       ForkJoinPool& pool) {
    if (conn < 1 || conn > 3) {
        return {};  // caller is expected to pass conn ∈ {1,2,3}
    }
    const uint64_t ht_mask = ht_size - 1;
    if (n_threads < 1) n_threads = 1;

    // Pre-tabulate forward neighbour offsets (dz, dy, dx).
    struct Nb { int dz, dy, dx; };
    Nb nbs_buf[13];
    int nb_n = 0;
    auto add = [&](int dz, int dy, int dx) { nbs_buf[nb_n++] = {dz, dy, dx}; };
    add(1, 0, 0); add(0, 1, 0); add(0, 0, 1);                          // conn>=1
    if (conn >= 2) {
        add(1, 1, 0); add(1, -1, 0);
        add(1, 0, 1); add(1, 0, -1);
        add(0, 1, 1); add(0, 1, -1);
    }
    if (conn >= 3) {
        add(1, 1, 1); add(1, 1, -1);
        add(1, -1, 1); add(1, -1, -1);
    }

    std::unique_ptr<uint64_t[]> hts(new uint64_t[static_cast<size_t>(n_threads) * ht_size]);

    auto emit_pair = [ht_mask](uint64_t* ht, T vi, T vj) {
        if (vj == 0 || vj == vi) return;
        const uint64_t lo = static_cast<uint64_t>(vi < vj ? vi : vj);
        const uint64_t hi = static_cast<uint64_t>(vi < vj ? vj : vi);
        ht_insert(ht, ht_mask, (lo << 32) | hi);
    };
    // Pre-compute flat offsets for the interior path (no per-pixel
    // multi-dim reconstruction needed).
    int64_t flat_off[13];
    for (int k = 0; k < nb_n; ++k) {
        flat_off[k] = (int64_t)nbs_buf[k].dz * H * W
                    + (int64_t)nbs_buf[k].dy * W
                    + (int64_t)nbs_buf[k].dx;
    }
    // Bounds-checked path (for edge pixels).
    auto scan_pixel_checked = [&](int64_t z, int64_t y, int64_t x, uint64_t* ht) {
        const T vi = lbl[(z * H + y) * W + x];
        if (vi == 0) return;
        for (int k = 0; k < nb_n; ++k) {
            const int64_t nz = z + nbs_buf[k].dz;
            if (nz < 0 || nz >= D) continue;
            const int64_t ny = y + nbs_buf[k].dy;
            if (ny < 0 || ny >= H) continue;
            const int64_t nx = x + nbs_buf[k].dx;
            if (nx < 0 || nx >= W) continue;
            emit_pair(ht, vi, lbl[(nz * H + ny) * W + nx]);
        }
    };
    // Z-band scan: split into interior z (0 < z < D-1) and edge z (0 / D-1).
    // For each interior y (0 < y < H-1), inner x split into edge cols
    // (x=0, x=W-1) and interior cols (1 ≤ x ≤ W-2). The interior path
    // uses a flat-offset loop with no bounds checks — that's where the
    // bulk of work goes for D, H, W ≥ 3.
    auto scan_band = [&](int64_t z0, int64_t z1, uint64_t* ht) {
        for (int64_t z = z0; z < z1; ++z) {
            const bool z_edge = (z == 0) || (z == D - 1);
            if (z_edge || H < 3 || W < 3) {
                for (int64_t y = 0; y < H; ++y)
                    for (int64_t x = 0; x < W; ++x)
                        scan_pixel_checked(z, y, x, ht);
                continue;
            }
            // y = 0 row (edge)
            for (int64_t x = 0; x < W; ++x) scan_pixel_checked(z, 0, x, ht);
            // interior y rows
            for (int64_t y = 1; y < H - 1; ++y) {
                // x = 0 col (edge)
                scan_pixel_checked(z, y, 0, ht);
                // interior cols: no bounds checks needed (every neighbour
                // is in-range because z, y, x are interior and dz/dy/dx
                // are all in {-1, 0, 1}).
                const T* row = lbl + (z * H + y) * W;
                for (int64_t x = 1; x < W - 1; ++x) {
                    const T vi = row[x];
                    if (vi == 0) continue;
                    const int64_t base = (z * H + y) * W + x;
                    for (int k = 0; k < nb_n; ++k) {
                        emit_pair(ht, vi, lbl[base + flat_off[k]]);
                    }
                }
                // x = W-1 col (edge)
                scan_pixel_checked(z, y, W - 1, ht);
            }
            // y = H-1 row (edge)
            for (int64_t x = 0; x < W; ++x) scan_pixel_checked(z, H - 1, x, ht);
        }
    };

    if (n_threads == 1 || D < 2) {
        std::fill_n(hts.get(), ht_size, HT_EMPTY);
        scan_band(0, D, hts.get());
    } else {
        // Phase 1: parallel scan over z-bands (each worker owns its HT).
        {
            std::atomic<int> next{0};
            const int64_t per = (D + n_threads - 1) / n_threads;
            pool.parallel([&]() {
                int t;
                while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                    uint64_t* ht = hts.get() + static_cast<size_t>(t) * ht_size;
                    std::fill_n(ht, ht_size, HT_EMPTY);
                    const int64_t z0 = static_cast<int64_t>(t) * per;
                    const int64_t z1 = std::min(z0 + per, D);
                    if (z0 < z1) scan_band(z0, z1, ht);
                }
            });
        }
        // Phase 2: pairwise tree merge.
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
