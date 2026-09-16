// In-place label compaction: rewrite an int32 label array so the
// nonzero labels are sequential 1..N (with 0 still meaning background).
// Drop-in for the perf-critical path of ``ncolor.format_labels`` —
// stays in C++ so the GIL can be released for the full Solver
// pipeline.
//
// Background semantics:
//   - Zero remains background when present.
//   - A negative minimum is the background; other values are shifted up.
//   - All-positive input has no background, including a constant image.
//
// Reduce the label range first. Sparse or very wide source IDs are compacted
// through sorted unique values, so table sizes scale with the image. Dense
// inputs use a presence table and a compact remapping table. The first-seen
// variant assigns codes in input scan order after the same normalization.
// Returns the number of distinct foreground labels. The optional
// background_changed flag reports normalization of a negative background.

#ifndef NCOLOR_FORMAT_LABELS_HPP
#define NCOLOR_FORMAT_LABELS_HPP

#include <algorithm>
#include <stdexcept>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

namespace detail {

// Threshold below which we run serial (parallel overhead exceeds work).
constexpr int64_t FORMAT_LABELS_SERIAL_THRESHOLD = 500000;

// Shared prologue for the two format_labels_inplace variants:
//   1. Parallel (min, max) reduce over the array.
//   2. If min < 0: shift every element by -min so bg moves to 0. Skipped
//      for min ≥ 0 — the input either already has bg at 0, or has no bg
//      (every pixel labeled — typical for already-expanded label maps);
//      shifting would absorb the smallest cell into bg.
// Returns max_lbl after the optional shift; -1 if the array is empty,
// or entirely background (caller should return 0).
inline int32_t format_labels_minmax_and_shift(
        int32_t* lbl, int64_t total, ForkJoinPool& pool, int n_threads,
        bool* background_changed = nullptr) {
    constexpr int32_t INT32_MIN_VAL = std::numeric_limits<int32_t>::min();
    constexpr int32_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max();
    int32_t min_lbl = INT32_MAX_VAL, max_lbl = INT32_MIN_VAL;
    if (n_threads <= 1 || total < FORMAT_LABELS_SERIAL_THRESHOLD) {
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
    if (background_changed) *background_changed = min_lbl < 0;
    if (max_lbl == min_lbl) {
        const int32_t value = min_lbl > 0 ? 1 : 0;
        std::fill(lbl, lbl + total, value);
        return value ? 1 : -1;
    }
    const int64_t span = static_cast<int64_t>(max_lbl) - std::min(min_lbl, 0);
    // Dense tables must scale with the image, not arbitrary source IDs.
    // This also handles shifts whose result would exceed signed int32.
    if (span > INT32_MAX_VAL || span > std::max<int64_t>(4096, total * 4)) {
        std::vector<int32_t> unique(lbl, lbl + total);
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        const int offset = min_lbl <= 0 ? 0 : 1;
        const int64_t count = static_cast<int64_t>(unique.size()) - 1 + offset;
        if (count > INT32_MAX_VAL)
            throw std::overflow_error("too many distinct labels for int32");
        auto apply = [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i)
                lbl[i] = static_cast<int32_t>(
                    std::lower_bound(unique.begin(), unique.end(), lbl[i]) - unique.begin()) + offset;
        };
        if (n_threads <= 1 || total < FORMAT_LABELS_SERIAL_THRESHOLD) apply(0, total);
        else dispatch_parallel(pool, static_cast<size_t>(total),
            static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD, apply);
        return static_cast<int32_t>(count);
    }
    if (min_lbl < 0) {
        const int64_t shift = -static_cast<int64_t>(min_lbl);
        if (n_threads <= 1 || total < FORMAT_LABELS_SERIAL_THRESHOLD) {
            for (int64_t i = 0; i < total; ++i) lbl[i] = static_cast<int32_t>(lbl[i] + shift);
        } else {
            dispatch_parallel(pool, static_cast<size_t>(total),
                static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
                [lbl, shift](size_t i0, size_t i1) {
                    for (size_t i = i0; i < i1; ++i) lbl[i] = static_cast<int32_t>(lbl[i] + shift);
                });
        }
        max_lbl = static_cast<int32_t>(max_lbl + shift);
    }
    if (max_lbl <= 0) return -1;
    return max_lbl;
}

}  // namespace detail

// First-seen-numbering variant: assigns new labels in input scan order,
// matching fastremap.renumber bit-for-bit. The build pass is inherently
// serial (we only learn a label is new on first encounter); ~2× slower
// than ascending-source. Available as an opt-in via
// `ncolor.format_labels(arr, first_seen=True)` when the caller relies on
// the historical fastremap output ordering.
inline int32_t format_labels_inplace_first_seen(
        int32_t* lbl, int64_t total,
        ForkJoinPool& pool, int n_threads, bool* background_changed = nullptr) {
    if (background_changed) *background_changed = false;
    if (total <= 0) return 0;
    const int32_t max_lbl =
        detail::format_labels_minmax_and_shift(lbl, total, pool, n_threads, background_changed);
    if (max_lbl < 0) return 0;
    // Serial build: dense table[l] = remapped_label, assigned on first
    // encounter in input scan order.
    std::vector<int32_t> table(static_cast<size_t>(max_lbl) + 1, 0);
    int32_t next_lbl = 0;
    for (int64_t i = 0; i < total; ++i) {
        const int32_t l = lbl[i];
        if (l > 0 && table[static_cast<size_t>(l)] == 0) {
            table[static_cast<size_t>(l)] = ++next_lbl;
        }
    }
    if (next_lbl == 0) return 0;
    if (n_threads <= 1 || total < detail::FORMAT_LABELS_SERIAL_THRESHOLD) {
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
                                     ForkJoinPool& pool, int n_threads,
                                     bool* background_changed = nullptr) {
    if (background_changed) *background_changed = false;
    if (total <= 0) return 0;
    const int32_t max_lbl =
        detail::format_labels_minmax_and_shift(lbl, total, pool, n_threads, background_changed);
    if (max_lbl < 0) return 0;

    // Relaxed atomic flags make overlapping label writes well-defined.
    // Skip stores once marked to avoid repeatedly invalidating shared cache lines.
    std::vector<std::atomic<uint8_t>> present(static_cast<size_t>(max_lbl) + 1);
    for (auto& flag : present) std::atomic_init(&flag, uint8_t{0});
    auto mark = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const int32_t v = lbl[i];
            if (v > 0 && !present[v].load(std::memory_order_relaxed))
                present[v].store(1, std::memory_order_relaxed);
        }
    };
    if (n_threads <= 1 || total < detail::FORMAT_LABELS_SERIAL_THRESHOLD) {
        if (max_lbl < 4096) {
            // The serial small-domain case needs neither atomic reads nor a
            // conditional store for every pixel, including background pixels.
            std::vector<uint8_t> local(static_cast<size_t>(max_lbl) + 1, 0);
            for (int64_t i = 0; i < total; ++i) local[lbl[i]] = 1;
            for (int32_t v = 1; v <= max_lbl; ++v)
                present[v].store(local[v], std::memory_order_relaxed);
        } else mark(0, total);
    } else if (max_lbl < 4096) {
        // Private byte flags avoid shared-cache-line traffic and the
        // read-modify-write dependency of packed bitsets. Bound their total
        // scratch to 1 MiB even when the caller owns a large worker pool.
        const size_t width = static_cast<size_t>(max_lbl) + 1;
        const size_t chunks = std::min(static_cast<size_t>(n_threads) *
            DISPATCH_CHUNKS_PER_THREAD, size_t{1048576} / width);
        std::vector<uint8_t> local(chunks * width, 0);
        dispatch_parallel(pool, chunks, chunks, [&](size_t lo, size_t hi) {
            for (size_t c = lo; c < hi; ++c) {
                uint8_t* flags = local.data() + c * width;
                const size_t begin = static_cast<size_t>(total) * c / chunks;
                const size_t end = static_cast<size_t>(total) * (c + 1) / chunks;
                for (size_t i = begin; i < end; ++i) flags[lbl[i]] = 1;
            }
        });
        for (size_t v = 1; v < width; ++v) {
            uint8_t found = 0;
            for (size_t c = 0; c < chunks; ++c) found |= local[c * width + v];
            present[v].store(found, std::memory_order_relaxed);
        }
    } else dispatch_parallel(pool, static_cast<size_t>(total),
        static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD, mark);

    // 3. Build sequential remap.
    std::vector<int32_t> remap(static_cast<size_t>(max_lbl) + 1, 0);
    int32_t next_lbl = 0;
    for (int64_t l = 1; l <= max_lbl; ++l) {
        if (present[static_cast<size_t>(l)].load(std::memory_order_relaxed)) remap[static_cast<size_t>(l)] = ++next_lbl;
    }

    // 4. Fast path: input was already 1..max, no apply needed.
    if (next_lbl == max_lbl) return next_lbl;

    // 5. Apply remap.
    if (n_threads <= 1 || total < detail::FORMAT_LABELS_SERIAL_THRESHOLD) {
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

// Narrow integer dtypes provide an exact domain bound before reading pixels.
// Build presence from the source, then cast and remap together. This avoids
// an int32 copy, range scan, and another scan of the larger converted image.
template <typename T>
inline int32_t format_byte_labels(const T* input, int32_t* output, int64_t total,
                                  ForkJoinPool& pool, int n_threads) {
    static_assert(std::is_integral<T>::value && sizeof(T) == 1,
                  "format_byte_labels requires a one-byte integer dtype");
    constexpr int minimum = static_cast<int>(std::numeric_limits<T>::min());
    constexpr size_t width = static_cast<size_t>(static_cast<int>(std::numeric_limits<T>::max()) - minimum + 1);
    const size_t chunks = (n_threads > 1 && total >= detail::FORMAT_LABELS_SERIAL_THRESHOLD)
        ? std::min(static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
                   size_t{1048576} / width) : 1;
    std::vector<uint8_t> present(chunks * width, 0);
    auto collect = [&](size_t lo, size_t hi) {
        for (size_t c = lo; c < hi; ++c) {
            const size_t size = static_cast<size_t>(total);
            const size_t begin = size / chunks * c + std::min(size % chunks, c);
            const size_t end = size / chunks * (c + 1) + std::min(size % chunks, c + 1);
            uint8_t* local = present.data() + c * width;
            for (size_t i = begin; i < end; ++i)
                local[static_cast<int>(input[i]) - minimum] = 1;
        }
    };
    if (chunks == 1) collect(0, 1);
    else dispatch_parallel(pool, chunks, chunks, collect);
    std::vector<int32_t> mapping(width, 0);
    int32_t count = 0;
    bool first = true;
    for (size_t v = 0; v < width; ++v) {
        uint8_t found = 0;
        for (size_t c = 0; c < chunks; ++c) found |= present[c * width + v];
        if (found) {
            if (!(first && static_cast<int>(v) + minimum <= 0)) mapping[v] = ++count;
            first = false;
        }
    }
    auto apply = [&](size_t lo, size_t hi) {
        for (size_t i = lo; i < hi; ++i) {
            const int32_t value = mapping[static_cast<int>(input[i]) - minimum];
            output[i] = value;
        }
    };
    if (chunks == 1) apply(0, static_cast<size_t>(total));
    else dispatch_parallel(pool, static_cast<size_t>(total), chunks, apply);
    return count;
}

// ---- Range-checked casts to int32 -----------------------------------------
//
// Every engine entry point works on int32 labels internally. Inputs that
// can hold values outside the int32 range (int64, uint32, uint64, float,
// double) are checked while they are cast: a plain ``static_cast`` would
// wrap a label like 2^31 + 5 to a negative number, which the format pass
// then treats as background, so whole cells silently vanish from the
// output. The check is one compare per element folded into a pass that
// is memory-bound anyway; for the narrow types it compiles away.
//
// The cast functions return ``true`` when every value fit. On ``false``
// the destination contents are unspecified and the caller raises; the
// Python wrappers catch that and compact the labels with ``np.unique``
// before retrying, so callers never see it for the label()/format
// paths. Floats are truncated toward zero; NaN and infinities count as
// out of range.

namespace detail {

template <typename InT>
constexpr bool cast_needs_range_check() {
    return std::is_floating_point<InT>::value ||
           (sizeof(InT) > 4) ||
           (sizeof(InT) == 4 && !std::is_signed<InT>::value);
}

template <typename InT>
inline bool fits_int32(InT v) {
    if constexpr (std::is_floating_point<InT>::value) {
        // Compare in double so the float32 rounding of INT32_MAX
        // (2147483648.0f) cannot admit a value that overflows the cast.
        // NaN fails both comparisons.
        const double d = static_cast<double>(v);
        return d >= -2147483648.0 && d < 2147483648.0;
    } else if constexpr (std::is_signed<InT>::value) {
        return v >= static_cast<InT>(std::numeric_limits<int32_t>::min()) &&
               v <= static_cast<InT>(std::numeric_limits<int32_t>::max());
    } else {
        return v <= static_cast<InT>(std::numeric_limits<int32_t>::max());
    }
}

// One element: the cast plus, for the wide types, the range test. The
// float branch writes 0 for an out-of-range value instead of performing
// an undefined float-to-int conversion; the result is discarded anyway.
template <typename InT>
inline int32_t checked_cast_int32(InT v, bool& ok) {
    if constexpr (cast_needs_range_check<InT>()) {
        const bool fits = fits_int32(v);
        ok &= fits;
        if constexpr (std::is_floating_point<InT>::value) {
            return fits ? static_cast<int32_t>(v) : int32_t{0};
        } else {
            return static_cast<int32_t>(v);
        }
    } else {
        return static_cast<int32_t>(v);
    }
}

}  // namespace detail

// Cast a typed input array to int32 AND capture the bg pattern (lbl == 0)
// to a uint8 mask in one parallel pass. This is what Solver.label uses
// for multi-dtype input: fuses the dtype conversion (which numpy.astype
// would have done single-threaded outside the GIL release) with the
// bg-mask capture (so apply_lut can do the post-expand zero-out without
// keeping a typed pointer to the original input around).
//
// Returns false if any value was outside the int32 range (see above).
template <typename InT>
inline bool cast_with_bg(const InT* src, int32_t* dst, uint8_t* bg_mask,
                         int64_t total,
                         ForkJoinPool& pool, int n_threads) {
    if (n_threads <= 1 || total < 500000) {
        bool ok = true;
        for (int64_t i = 0; i < total; ++i) {
            const InT v = src[i];
            dst[i] = detail::checked_cast_int32<InT>(v, ok);
            bg_mask[i] = (dst[i] == 0) ? uint8_t{1} : uint8_t{0};
        }
        return ok;
    }
    // A single shared flag, written only on failure, so the common path
    // never touches a contended cache line.
    std::atomic<bool> any_bad{false};
    dispatch_parallel(pool, static_cast<size_t>(total),
        static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
        [src, dst, bg_mask, &any_bad](size_t i0, size_t i1) {
            bool ok = true;
            for (size_t i = i0; i < i1; ++i) {
                const InT v = src[i];
                dst[i] = detail::checked_cast_int32<InT>(v, ok);
                bg_mask[i] = (dst[i] == 0) ? uint8_t{1} : uint8_t{0};
            }
            if (!ok) any_bad.store(true, std::memory_order_relaxed);
        });
    return !any_bad.load(std::memory_order_relaxed);
}

// Cast a typed input array to int32 in parallel. For int32 → int32 the
// templated path becomes a straight copy; we provide an explicit
// specialization that uses memcpy (with a same-pointer guard, so callers
// can no-op when src == dst).
//
// Used by ExpandEngine.format_labels so the dtype conversion fuses
// naturally with the downstream format_labels_inplace pass — same memory
// bandwidth either way, but we avoid the single-threaded numpy.astype the
// Python wrapper would otherwise have done before calling in.
//
// Returns false if any value was outside the int32 range (see above).
template <typename InT>
inline bool cast_to_int32(const InT* src, int32_t* dst, int64_t total,
                          ForkJoinPool& pool, int n_threads) {
    if (n_threads <= 1 || total < 500000) {
        bool ok = true;
        for (int64_t i = 0; i < total; ++i) {
            dst[i] = detail::checked_cast_int32<InT>(src[i], ok);
        }
        return ok;
    }
    std::atomic<bool> any_bad{false};
    dispatch_parallel(pool, static_cast<size_t>(total),
        static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
        [src, dst, &any_bad](size_t i0, size_t i1) {
            bool ok = true;
            for (size_t i = i0; i < i1; ++i) {
                dst[i] = detail::checked_cast_int32<InT>(src[i], ok);
            }
            if (!ok) any_bad.store(true, std::memory_order_relaxed);
        });
    return !any_bad.load(std::memory_order_relaxed);
}

// int32 → int32 specialization: memcpy when src != dst, no-op otherwise.
template <>
inline bool cast_to_int32<int32_t>(const int32_t* src, int32_t* dst,
                                   int64_t total,
                                   ForkJoinPool& /*pool*/, int /*n_threads*/) {
    if (src != dst) {
        std::memcpy(dst, src, static_cast<size_t>(total) * sizeof(int32_t));
    }
    return true;
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_FORMAT_LABELS_HPP
