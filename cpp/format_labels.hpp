// In-place label compaction: rewrite an int32 label array so the
// nonzero labels are sequential 1..N (with 0 still meaning background).
// Mirrors the common path of ``ncolor.format_labels`` but stays in C++
// so the GIL can stay released for the full Solver pipeline.
//
// Precondition: bg == 0. Inputs whose minimum label is nonzero (i.e.
// the legacy "treat min as bg and shift" case) are handled by the
// Python wrapper falling back to the numpy/numba ``format_labels``
// before calling Solver.label(format_input=True). Documented at the
// call site.
//
// Algorithm:
//   1. Parallel reduce → max_lbl.
//   2. Parallel scatter → present[1..max_lbl] (uint8_t — single-byte
//      writes of 1, race is benign).
//   3. Serial scan over [1..max_lbl] → remap[l] = next_lbl++ if present.
//   4. If ``next_lbl == max_lbl`` (input was already compact 1..max),
//      skip the apply pass — fast path for omnipose's typical input.
//   5. Parallel scatter → lbl[i] = remap[lbl[i]].
// Returns the number of distinct nonzero labels after compaction.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

inline int32_t format_labels_inplace(int32_t* lbl, int64_t total,
                                     ForkJoinPool& pool, int n_threads) {
    if (total <= 0) return 0;

    // 1. Parallel max reduce.
    int32_t max_lbl = 0;
    if (n_threads <= 1 || total < 8192) {
        for (int64_t i = 0; i < total; ++i) {
            if (lbl[i] > max_lbl) max_lbl = lbl[i];
        }
    } else {
        const size_t n_chunks =
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD;
        const size_t total_sz = static_cast<size_t>(total);
        const size_t actual_chunks = std::min(n_chunks, total_sz);
        const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
        std::vector<int32_t> partials(actual_chunks, 0);
        std::atomic<size_t> next{0};
        pool.parallel([&]() {
            size_t idx;
            while ((idx = next.fetch_add(1, std::memory_order_relaxed))
                   < actual_chunks) {
                const size_t i0 = idx * chunk_sz;
                const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                int32_t m = 0;
                for (size_t i = i0; i < i1; ++i) {
                    if (lbl[i] > m) m = lbl[i];
                }
                partials[idx] = m;
            }
        });
        for (size_t i = 0; i < actual_chunks; ++i) {
            if (partials[i] > max_lbl) max_lbl = partials[i];
        }
    }
    if (max_lbl <= 0) return 0;

    // 2. Mark present labels. Concurrent writes of value 1 to the same
    // byte are a benign race in practice; std::memory_order_relaxed
    // atomics aren't strictly needed (every write is value-stable).
    std::vector<uint8_t> present(static_cast<size_t>(max_lbl) + 1, 0);
    if (n_threads <= 1 || total < 8192) {
        for (int64_t i = 0; i < total; ++i) {
            const int32_t v = lbl[i];
            if (v > 0) present[static_cast<size_t>(v)] = 1;
        }
    } else {
        uint8_t* present_ptr = present.data();
        dispatch_parallel(pool, static_cast<size_t>(total),
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
            [lbl, present_ptr](size_t i0, size_t i1) {
                for (size_t i = i0; i < i1; ++i) {
                    const int32_t v = lbl[i];
                    if (v > 0) present_ptr[static_cast<size_t>(v)] = 1;
                }
            });
    }

    // 3. Build sequential remap.
    std::vector<int32_t> remap(static_cast<size_t>(max_lbl) + 1, 0);
    int32_t next_lbl = 0;
    for (int32_t l = 1; l <= max_lbl; ++l) {
        if (present[static_cast<size_t>(l)]) remap[static_cast<size_t>(l)] = ++next_lbl;
    }

    // 4. Fast path: input was already 1..max, no apply needed.
    if (next_lbl == max_lbl) return next_lbl;

    // 5. Apply remap.
    if (n_threads <= 1 || total < 8192) {
        for (int64_t i = 0; i < total; ++i) {
            lbl[i] = remap[static_cast<size_t>(lbl[i])];
        }
    } else {
        const int32_t* remap_ptr = remap.data();
        dispatch_parallel(pool, static_cast<size_t>(total),
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
            [lbl, remap_ptr](size_t i0, size_t i1) {
                for (size_t i = i0; i < i1; ++i) {
                    lbl[i] = remap_ptr[static_cast<size_t>(lbl[i])];
                }
            });
    }
    return next_lbl;
}

// Apply background mask: out_ptr[i] = 0 wherever input_mask[i] == 0.
// Used as the final stage of Solver.label when ``mask_bg=True`` to
// match numba's ``colored * (lab != 0)`` semantics.
template <typename OutT>
inline void apply_bg_mask(const int32_t* input_mask, OutT* out_ptr,
                          int64_t total,
                          ForkJoinPool& pool, int n_threads) {
    if (n_threads <= 1 || total < 8192) {
        for (int64_t i = 0; i < total; ++i) {
            if (input_mask[i] == 0) out_ptr[i] = 0;
        }
        return;
    }
    dispatch_parallel(pool, static_cast<size_t>(total),
        static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
        [input_mask, out_ptr](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                if (input_mask[i] == 0) out_ptr[i] = 0;
            }
        });
}

}  // namespace ncolor_cpp
