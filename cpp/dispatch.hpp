/*
 * dispatch.hpp — atomic work-stealing dispatch + per-pass thread cap.
 *
 * Ported from edt/src/edt.hpp's ``dispatch_parallel`` and
 * ``compute_threads`` helpers. Replaces our previous "enqueue N futures
 * up-front, wait on each" pattern with edt's much faster pattern: workers
 * spin on a sense-reversing barrier (zero mutex / condvar on hot path),
 * grab chunks via ``atomic<size_t>::fetch_add``, and load-balance
 * automatically.
 *
 * Why this matters: my old static partitioning + per-chunk std::future
 * pattern was paying ~100-1000 us of pthread/futex overhead per call to
 * `pool.enqueue` × n_threads chunks. edt's pool has no per-chunk overhead;
 * the only cost is two barrier crossings (≈10 us total even at 64 threads).
 */

#ifndef NCOLOR_DISPATCH_HPP
#define NCOLOR_DISPATCH_HPP

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "threadpool.h"

namespace ncolor_cpp {

// More chunks than threads → atomic work-stealing has slack to balance load.
// edt uses ND_CHUNKS_PER_THREAD = 4 in their hot path; same default here.
constexpr size_t DISPATCH_CHUNKS_PER_THREAD = 4;

// Distribute [0, total) into up to max_chunks chunks across the pool's
// threads. Workers atomically claim chunks; main thread participates.
// Calls work(begin, end) once per claimed chunk. Blocks until all chunks
// complete. ``threads <= 1`` or ``total == 0`` falls back to direct call.
template <typename F>
inline void dispatch_parallel(ForkJoinPool& pool, size_t total,
                              size_t max_chunks, F work) {
    if (max_chunks == 0 || total == 0) {
        work(size_t{0}, total);
        return;
    }
    const size_t n_chunks = std::min(max_chunks, total);
    const size_t chunk_sz = (total + n_chunks - 1) / n_chunks;
    std::atomic<size_t> next{0};
    pool.parallel([&]() {
        size_t idx;
        while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < n_chunks) {
            const size_t begin = idx * chunk_sz;
            const size_t end = std::min(total, begin + chunk_sz);
            work(begin, end);
        }
    });
}

// Per-pass thread cap (verbatim from edt/src/edt.hpp::compute_threads).
// For small passes the per-chunk overhead exceeds the work, so capping at
// 4-12 threads is a clean win. For large passes we use the caller's
// requested thread count.
inline size_t compute_threads(size_t desired, size_t total_lines, size_t axis_len) {
    if (desired <= 1 || total_lines <= 1) return 1;
    size_t threads = std::min<size_t>(desired, total_lines);
    const size_t total_work = axis_len * total_lines;
    if (total_work <= 60000) {
        threads = std::min<size_t>(threads, 4);
    } else if (total_work <= 120000) {
        threads = std::min<size_t>(threads, 8);
    } else if (total_work <= 400000) {
        threads = std::min<size_t>(threads, 12);
    }
    return std::max<size_t>(1, threads);
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_DISPATCH_HPP
