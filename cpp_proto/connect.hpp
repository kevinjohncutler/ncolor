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

    // Per-thread hashtables, allocated as one contiguous block for locality.
    std::vector<uint64_t> hts(static_cast<size_t>(n_threads) * ht_size, HT_EMPTY);

    // Phase 1: parallel scan via atomic work-stealing. Strip count = n_threads
    // (one strip per thread is correct here since the per-thread hashtables
    // are pre-allocated and indexed by strip id).
    {
        std::atomic<int> next{0};
        const int64_t strip = (total + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                const int64_t s = static_cast<int64_t>(t) * strip;
                const int64_t e = std::min(s + strip, total);
                uint64_t* ht = hts.data() + static_cast<size_t>(t) * ht_size;
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
                uint64_t* dst_ht = hts.data() + static_cast<size_t>(dst) * ht_size;
                const uint64_t* src_ht = hts.data() + static_cast<size_t>(src) * ht_size;
                ht_merge(src_ht, dst_ht, ht_size);
            }
        });
        stride *= 2;
    }

    // Phase 3: extract unique pairs from workers[0]'s table.
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(64);
    const uint64_t* root = hts.data();
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

    std::vector<uint64_t> hts(static_cast<size_t>(n_threads) * ht_size, HT_EMPTY);

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
        scan_band(0, H, hts.data());
    } else {
        // Phase 1: scan over row-bands via atomic work-stealing.
        {
            std::atomic<int> next{0};
            const int64_t per = (H + n_threads - 1) / n_threads;
            pool.parallel([&]() {
                int t;
                while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                    const int64_t y0 = static_cast<int64_t>(t) * per;
                    const int64_t y1 = std::min(y0 + per, H);
                    if (y0 >= y1) continue;
                    uint64_t* ht = hts.data() + static_cast<size_t>(t) * ht_size;
                    scan_band(y0, y1, ht);
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
                    uint64_t* dst_ht = hts.data() + static_cast<size_t>(dst) * ht_size;
                    const uint64_t* src_ht = hts.data() + static_cast<size_t>(src) * ht_size;
                    ht_merge(src_ht, dst_ht, ht_size);
                }
            });
            stride *= 2;
        }
    }

    // Extract.
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(64);
    const uint64_t* root = hts.data();
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
