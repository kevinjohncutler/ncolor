// Fused connect-graph + face-count scan.
//
// Goal: combine the two full-image scans that delete_spurs_labels and
// find_pairs each do into one cache-warm pass. Each pixel does:
//   * 4 face-conn neighbour checks  → ``face_count[i]`` (self-only writes,
//     so no atomics across thread boundaries).
//   * 2 forward diagonals (conn=2)  → pair emission to a per-thread
//     hashtable for the connect graph.
//   * Forward face neighbours (E, S) that emit pairs piggyback on the
//     same checks; backward face neighbours (W, N) update face_count
//     only.
//
// The per-thread hashtable layout matches the existing find_pairs
// (one hashtable per thread, merged at the end). This way the picker
// receives the same pair list it would have from a separate find_pairs
// call. Bit-identical results verified against find_pairs on real
// inputs.
//
// Scope: 2D, conn=2, no wrap, no weighting — the common case used by
// Solver::label. ND / wrap / weighted callers continue to use the
// original find_pairs path.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include "connect.hpp"   // for HT_EMPTY, ht_insert, FIND_PAIRS_MAX_NDIM
#include "threadpool.h"

namespace ncolor_cpp {

template <typename T>
inline std::vector<std::pair<int32_t, int32_t>>
find_pairs_with_face_count_2d(
    const T* labels, int64_t H, int64_t W,
    int conn,                       // 1 (face only) or 2 (face + diag)
    uint8_t* face_count,            // output, size H*W, pre-zeroed by caller
    uint64_t ht_size, int n_threads,
    ForkJoinPool& pool,
    std::vector<uint64_t>* ht_scratch = nullptr)
{
    if (n_threads < 1) n_threads = 1;
    const uint64_t ht_mask = ht_size - 1;
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

    // Per-thread scan over a row band.
    auto scan_band = [&](int t, int64_t y0, int64_t y1) {
        uint64_t* ht = hts_ptr + (size_t)t * ht_size;
        std::fill_n(ht, ht_size, HT_EMPTY);
        for (int64_t y = y0; y < y1; ++y) {
            const bool yL = (y > 0);
            const bool yH = (y + 1 < H);
            for (int64_t x = 0; x < W; ++x) {
                const int64_t i = y * W + x;
                const T vi = labels[i];
                if (vi == 0) {
                    face_count[i] = 0;
                    continue;
                }
                int fc = 0;
                // Helper: classify a neighbour. If same-label & non-bg,
                // bump self-only face_count. If different-label & non-
                // bg AND forward (i.e. this pixel is the smaller of the
                // pair), insert into hashtable. We pass an explicit
                // ``forward`` flag for clarity.
                auto check_face = [&](T vj, bool forward) {
                    if (vj == 0) return;
                    if (vj == vi) { ++fc; return; }
                    if (!forward) return;
                    const uint64_t lo = vi < vj ? (uint64_t)vi : (uint64_t)vj;
                    const uint64_t hi = vi < vj ? (uint64_t)vj : (uint64_t)vi;
                    ht_insert(ht, ht_mask, (lo << 32) | hi);
                };
                // 4 face-conn neighbours.
                if (yL)        check_face(labels[i - W], /*forward=*/false);  // N
                if (yH)        check_face(labels[i + W], /*forward=*/true);   // S
                if (x > 0)     check_face(labels[i - 1], /*forward=*/false);  // W
                if (x + 1 < W) check_face(labels[i + 1], /*forward=*/true);   // E
                face_count[i] = (uint8_t)fc;
                if (conn >= 2) {
                    // Forward diagonals only (SE, SW). Backward diagonals
                    // are emitted by their own pixels' forward checks.
                    auto check_diag = [&](T vj) {
                        if (vj == 0 || vj == vi) return;
                        const uint64_t lo = vi < vj ? (uint64_t)vi : (uint64_t)vj;
                        const uint64_t hi = vi < vj ? (uint64_t)vj : (uint64_t)vi;
                        ht_insert(ht, ht_mask, (lo << 32) | hi);
                    };
                    if (yH && x > 0)     check_diag(labels[i + W - 1]);  // SW
                    if (yH && x + 1 < W) check_diag(labels[i + W + 1]);  // SE
                }
            }
        }
    };

    // Always initialise EVERY per-thread HT slice with HT_EMPTY,
    // even slices belonging to threads that get no row-range to scan
    // (happens when H < n_threads). The merge step iterates all
    // n_threads slices, so any uninitialised slot would leak stale
    // data from fp_ht_buf_ (or from another solver call) into the
    // output pair list.
    for (int t = 0; t < n_threads; ++t) {
        std::fill_n(hts_ptr + (size_t)t * ht_size, ht_size, HT_EMPTY);
    }
    if (n_threads == 1 || H < 2) {
        scan_band(0, 0, H);
    } else {
        std::atomic<int> next{0};
        const int64_t per = (H + n_threads - 1) / n_threads;
        pool.parallel([&]() {
            int t;
            while ((t = next.fetch_add(1, std::memory_order_relaxed)) < n_threads) {
                int64_t y0 = (int64_t)t * per;
                int64_t y1 = std::min(y0 + per, H);
                if (y0 < y1) scan_band(t, y0, y1);
            }
        });
    }

    // Merge per-thread hashtables into a single output pair list, just
    // like find_pairs's serial union. Each thread's HT slice is dense
    // (slot == HT_EMPTY when free); collect every non-empty slot,
    // dedupe by inserting into a master HT, then materialise.
    std::vector<uint64_t> master_ht(ht_size, HT_EMPTY);
    for (int t = 0; t < n_threads; ++t) {
        const uint64_t* ht = hts_ptr + (size_t)t * ht_size;
        for (uint64_t s = 0; s < ht_size; ++s) {
            const uint64_t key = ht[s];
            if (key != HT_EMPTY) ht_insert(master_ht.data(), ht_mask, key);
        }
    }
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(128);
    for (uint64_t s = 0; s < ht_size; ++s) {
        const uint64_t key = master_ht[s];
        if (key == HT_EMPTY) continue;
        const int32_t lo = (int32_t)(key >> 32);
        const int32_t hi = (int32_t)(key & 0xFFFFFFFFu);
        out.emplace_back(lo, hi);
    }
    return out;
}

}  // namespace ncolor_cpp
