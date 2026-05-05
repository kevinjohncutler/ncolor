// In-place label compaction: rewrite an int32 label array so the
// nonzero labels are sequential 1..N (with 0 still meaning background).
// Drop-in for the perf-critical path of ``ncolor.format_labels`` —
// stays in C++ so the GIL can be released for the full Solver
// pipeline.
//
// Background semantics (mirrors the legacy ``format_labels`` shift):
//   - If min(lbl) == 0: 0 is bg as-is; just compact nonzero → 1..N.
//   - If min(lbl) != 0: treat min as bg, shift everything by -min so
//     the new min is 0, then compact. Handles inputs like {3, 5, 7}
//     (min=3 is bg → output {0, 1, 2}) or {-1, 0, 5} (min=-1 is bg
//     → shift by +1 → {0, 1, 6} → compact to {0, 1, 2}).
//
// Algorithm (when min == 0; otherwise add one parallel shift pass first):
//   1. Parallel reduce → (min_lbl, max_lbl).
//   2. If min_lbl != 0: parallel apply lbl[i] -= min_lbl. min_lbl=0 now.
//   3. Parallel scatter → present[1..max_lbl] (uint8_t — single-byte
//      writes of value 1, race is benign).
//   4. Serial scan over [1..max_lbl] → remap[l] = next_lbl++ if present.
//   5. If ``next_lbl == max_lbl`` (input was already compact 1..max),
//      skip the apply pass — fast path for omnipose's typical input.
//   6. Parallel scatter → lbl[i] = remap[lbl[i]].
// Returns the number of distinct nonzero labels after compaction.

#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

// First-seen-numbering variant: mirrors fastremap.renumber's output
// exactly. Build pass is inherently serial — assigns a new sequential
// label the first time each source value is encountered in input scan
// order. Apply pass parallelizes. Used to bench the cost of bit-equality
// with fastremap; production code uses ``format_labels_inplace`` (the
// ascending-source variant) which is faster.
inline int32_t format_labels_inplace_first_seen(
        int32_t* lbl, int64_t total,
        ForkJoinPool& pool, int n_threads) {
    if (total <= 0) return 0;

    // 1. Min/max reduce (parallel) — same as the ascending variant.
    constexpr int32_t INT32_MIN_VAL = std::numeric_limits<int32_t>::min();
    constexpr int32_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max();
    int32_t min_lbl = INT32_MAX_VAL, max_lbl = INT32_MIN_VAL;
    if (n_threads <= 1 || total < 500000) {
        for (int64_t i = 0; i < total; ++i) {
            const int32_t v = lbl[i];
            if (v < min_lbl) min_lbl = v;
            if (v > max_lbl) max_lbl = v;
        }
    } else {
        const size_t n_chunks =
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD;
        const size_t total_sz = static_cast<size_t>(total);
        const size_t actual_chunks = std::min(n_chunks, total_sz);
        const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
        std::vector<int32_t> mins(actual_chunks, INT32_MAX_VAL);
        std::vector<int32_t> maxs(actual_chunks, INT32_MIN_VAL);
        std::atomic<size_t> next{0};
        pool.parallel([&]() {
            size_t idx;
            while ((idx = next.fetch_add(1, std::memory_order_relaxed))
                   < actual_chunks) {
                const size_t i0 = idx * chunk_sz;
                const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                int32_t mn = INT32_MAX_VAL, mx = INT32_MIN_VAL;
                for (size_t i = i0; i < i1; ++i) {
                    const int32_t v = lbl[i];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                mins[idx] = mn; maxs[idx] = mx;
            }
        });
        for (size_t i = 0; i < actual_chunks; ++i) {
            if (mins[i] < min_lbl) min_lbl = mins[i];
            if (maxs[i] > max_lbl) max_lbl = maxs[i];
        }
    }
    if (max_lbl <= min_lbl) return 0;

    // 2. Min-shift if needed (parallel).
    if (min_lbl != 0) {
        const int32_t shift = -min_lbl;
        if (n_threads <= 1 || total < 500000) {
            for (int64_t i = 0; i < total; ++i) lbl[i] += shift;
        } else {
            dispatch_parallel(pool, static_cast<size_t>(total),
                static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
                [lbl, shift](size_t i0, size_t i1) {
                    for (size_t i = i0; i < i1; ++i) lbl[i] += shift;
                });
        }
        max_lbl += shift;
    }
    if (max_lbl <= 0) return 0;

    // 3. SERIAL build: dense table[l] = remapped_label. The first time
    // we encounter a label, assign next_lbl++ and remember. This is
    // the part that's inherently sequential — fastremap's bit-equality
    // requires this exact scan order.
    std::vector<int32_t> table(static_cast<size_t>(max_lbl) + 1, 0);
    int32_t next_lbl = 0;
    for (int64_t i = 0; i < total; ++i) {
        const int32_t l = lbl[i];
        if (l > 0 && table[static_cast<size_t>(l)] == 0) {
            table[static_cast<size_t>(l)] = ++next_lbl;
        }
    }
    if (next_lbl == 0) return 0;

    // 4. Parallel apply.
    if (n_threads <= 1 || total < 500000) {
        for (int64_t i = 0; i < total; ++i) {
            lbl[i] = table[static_cast<size_t>(lbl[i])];
        }
    } else {
        const int32_t* table_ptr = table.data();
        dispatch_parallel(pool, static_cast<size_t>(total),
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
            [lbl, table_ptr](size_t i0, size_t i1) {
                for (size_t i = i0; i < i1; ++i) {
                    lbl[i] = table_ptr[static_cast<size_t>(lbl[i])];
                }
            });
    }
    return next_lbl;
}

inline int32_t format_labels_inplace(int32_t* lbl, int64_t total,
                                     ForkJoinPool& pool, int n_threads) {
    if (total <= 0) return 0;

    // 1. Parallel min/max reduce in one pass.
    constexpr int32_t INT32_MIN_VAL = std::numeric_limits<int32_t>::min();
    constexpr int32_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max();
    int32_t min_lbl = INT32_MAX_VAL;
    int32_t max_lbl = INT32_MIN_VAL;
    if (n_threads <= 1 || total < 500000) {
        for (int64_t i = 0; i < total; ++i) {
            const int32_t v = lbl[i];
            if (v < min_lbl) min_lbl = v;
            if (v > max_lbl) max_lbl = v;
        }
    } else {
        const size_t n_chunks =
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD;
        const size_t total_sz = static_cast<size_t>(total);
        const size_t actual_chunks = std::min(n_chunks, total_sz);
        const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
        std::vector<int32_t> mins(actual_chunks, INT32_MAX_VAL);
        std::vector<int32_t> maxs(actual_chunks, INT32_MIN_VAL);
        std::atomic<size_t> next{0};
        pool.parallel([&]() {
            size_t idx;
            while ((idx = next.fetch_add(1, std::memory_order_relaxed))
                   < actual_chunks) {
                const size_t i0 = idx * chunk_sz;
                const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                int32_t mn = INT32_MAX_VAL, mx = INT32_MIN_VAL;
                for (size_t i = i0; i < i1; ++i) {
                    const int32_t v = lbl[i];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
                mins[idx] = mn;
                maxs[idx] = mx;
            }
        });
        for (size_t i = 0; i < actual_chunks; ++i) {
            if (mins[i] < min_lbl) min_lbl = mins[i];
            if (maxs[i] > max_lbl) max_lbl = maxs[i];
        }
    }
    if (max_lbl <= min_lbl) return 0;  // empty or constant array.

    // 2. If min != 0, shift every label by -min so the new min is 0.
    // Treats ``min`` as the background marker (matches the legacy
    // ``format_labels`` "shift to 0" semantics).
    if (min_lbl != 0) {
        const int32_t shift = -min_lbl;
        if (n_threads <= 1 || total < 500000) {
            for (int64_t i = 0; i < total; ++i) lbl[i] += shift;
        } else {
            dispatch_parallel(pool, static_cast<size_t>(total),
                static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
                [lbl, shift](size_t i0, size_t i1) {
                    for (size_t i = i0; i < i1; ++i) lbl[i] += shift;
                });
        }
        max_lbl += shift;  // min_lbl is now 0.
    }
    if (max_lbl <= 0) return 0;

    // 2. Mark present labels. Concurrent writes of value 1 to the same
    // byte are a benign race in practice; std::memory_order_relaxed
    // atomics aren't strictly needed (every write is value-stable).
    std::vector<uint8_t> present(static_cast<size_t>(max_lbl) + 1, 0);
    if (n_threads <= 1 || total < 500000) {
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
    if (n_threads <= 1 || total < 500000) {
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

// Cast a typed input array to int32 AND capture the bg pattern (lbl == 0)
// to a uint8 mask in one parallel pass. This is what Solver.label uses
// for multi-dtype input: fuses the dtype conversion (which numpy.astype
// would have done single-threaded outside the GIL release) with the
// bg-mask capture (so apply_lut can do the post-expand zero-out without
// keeping a typed pointer to the original input around).
template <typename InT>
inline void cast_with_bg(const InT* src, int32_t* dst, uint8_t* bg_mask,
                         int64_t total,
                         ForkJoinPool& pool, int n_threads) {
    if (n_threads <= 1 || total < 500000) {
        for (int64_t i = 0; i < total; ++i) {
            const InT v = src[i];
            dst[i] = static_cast<int32_t>(v);
            bg_mask[i] = (v == InT{0}) ? uint8_t{1} : uint8_t{0};
        }
        return;
    }
    dispatch_parallel(pool, static_cast<size_t>(total),
        static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
        [src, dst, bg_mask](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                const InT v = src[i];
                dst[i] = static_cast<int32_t>(v);
                bg_mask[i] = (v == InT{0}) ? uint8_t{1} : uint8_t{0};
            }
        });
}

// Cast a typed input array (uint8/uint16/uint32/int8/int16/int32/int64)
// to int32 in parallel. For int32 → int32 the templated path becomes a
// straight copy; we provide an explicit specialization that uses memcpy
// (with a same-pointer guard, so callers can no-op when src == dst).
//
// Used by Solver.label so the dtype conversion fuses naturally with the
// downstream format_labels_inplace pass — same memory bandwidth either
// way, but we avoid the single-threaded numpy.astype the Python wrapper
// would otherwise have done before calling in.
template <typename InT>
inline void cast_to_int32(const InT* src, int32_t* dst, int64_t total,
                          ForkJoinPool& pool, int n_threads) {
    if (n_threads <= 1 || total < 500000) {
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = static_cast<int32_t>(src[i]);
        }
        return;
    }
    dispatch_parallel(pool, static_cast<size_t>(total),
        static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
        [src, dst](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                dst[i] = static_cast<int32_t>(src[i]);
            }
        });
}

// int32 → int32 specialization: memcpy when src != dst, no-op otherwise.
template <>
inline void cast_to_int32<int32_t>(const int32_t* src, int32_t* dst,
                                   int64_t total,
                                   ForkJoinPool& /*pool*/, int /*n_threads*/) {
    if (src != dst) {
        std::memcpy(dst, src, static_cast<size_t>(total) * sizeof(int32_t));
    }
}

// Apply background mask: out_ptr[i] = 0 wherever input_mask[i] == 0.
// Used as the final stage of Solver.label when ``mask_bg=True`` to
// match numba's ``colored * (lab != 0)`` semantics.
template <typename OutT>
inline void apply_bg_mask(const int32_t* input_mask, OutT* out_ptr,
                          int64_t total,
                          ForkJoinPool& pool, int n_threads) {
    if (n_threads <= 1 || total < 500000) {
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
