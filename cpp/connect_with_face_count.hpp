// Fused connect-graph + face-count scan, with per-pair contact counts.
//
// Single cache-warm pass that produces:
//   • ``face_count[i]`` — per-pixel same-label face-neighbour count
//     (consumed by ``despur_via_face_count_with_pair_decrement_*``).
//   • A hashtable of (lo, hi) cell-pair keys plus a parallel
//     ``pair_counts`` array storing the number of pixel-level contacts
//     each pair has. The count is summed across all forward edges
//     (face + diagonal under conn=2) between the two cells.
//
// Why per-pair counts: the fused pair list is computed BEFORE despur,
// so pairs whose only contact pixels are spurs would become "ghost"
// edges after despur reverts them. The despur peel-back uses the
// pair_counts array to decrement counts when reverting a spur (one
// decrement per (this_label, neighbour_label) contact lost). After
// peel-back finishes, the output pair list is filtered to count > 0,
// so the picker only sees real, post-despur adjacencies.
//
// Scope: 2D, conn=2 (face + diagonal), no wrap, no weighting.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

#include "connect.hpp"   // HT_EMPTY, ht_insert
#include "threadpool.h"

namespace ncolor_cpp {

namespace fp_fc_detail {

// Bit-mixed slot for a given key (same Fibonacci mixer as ht_insert).
inline uint64_t key_slot(uint64_t key, uint64_t ht_mask) {
    return (key * 11400714819323198485ULL) & ht_mask;
}

// Lookup-or-insert: finds the slot for ``key`` in (ht_keys, ht_cnts)
// via linear probing. If a free slot is hit, the key is written and
// the count is initialised to 1. Otherwise the existing slot's count
// is incremented. Returns the slot index (for callers that need it).
inline uint64_t ht_insert_count(
    uint64_t* ht_keys, uint32_t* ht_cnts,
    uint64_t ht_mask, uint64_t key)
{
    uint64_t slot = key_slot(key, ht_mask);
    while (true) {
        const uint64_t cur = ht_keys[slot];
        if (cur == HT_EMPTY) {
            ht_keys[slot] = key;
            ht_cnts[slot] = 1;
            return slot;
        }
        if (cur == key) {
            ++ht_cnts[slot];
            return slot;
        }
        slot = (slot + 1) & ht_mask;
    }
}

// Lookup (no insert): returns the slot if found, or UINT64_MAX if
// the key isn't present. Used by despur peel-back to decrement
// pair counts when a spur pixel reverts.
inline uint64_t ht_lookup(const uint64_t* ht_keys, uint64_t ht_mask, uint64_t key) {
    uint64_t slot = key_slot(key, ht_mask);
    while (true) {
        const uint64_t cur = ht_keys[slot];
        if (cur == HT_EMPTY) return UINT64_MAX;  // not present
        if (cur == key) return slot;
        slot = (slot + 1) & ht_mask;
    }
}

}  // namespace fp_fc_detail


// Returns the merged hashtable as (ht_keys, ht_cnts) — both sized
// ``ht_size``. Pair list is NOT materialised here; the caller is
// expected to run despur (decrementing counts on revert) and then
// scan the table for surviving (count > 0) pairs.
//
// face_count must be pre-zeroed (size H*W).
template <typename T>
inline void find_pairs_with_face_count_2d_v2(
    const T* labels, int64_t H, int64_t W,
    int conn,                         // 1 (face) or 2 (face + diag)
    uint8_t* face_count,              // OUT: H*W same-label face-neighbour count
    std::vector<uint64_t>& ht_keys,   // OUT: ht_size, HT_EMPTY for free slots
    std::vector<uint32_t>& ht_cnts,   // OUT: ht_size, per-pair contact counts
    uint64_t ht_size, int n_threads,
    ForkJoinPool& pool,
    std::vector<uint64_t>* per_thread_ht_buf = nullptr)
{
    if (n_threads < 1) n_threads = 1;
    const uint64_t ht_mask = ht_size - 1;

    // Per-thread keys + counts (interleaved isn't easier; use two
    // parallel scratch arrays). Same fp_ht_buf_ pattern as find_pairs.
    const size_t ht_total = static_cast<size_t>(n_threads) * ht_size;
    std::vector<uint64_t> ht_keys_local;
    std::vector<uint32_t> ht_cnts_local;
    uint64_t* tk_ptr;
    uint32_t* tc_ptr;
    if (per_thread_ht_buf) {
        if (per_thread_ht_buf->size() < ht_total) per_thread_ht_buf->resize(ht_total);
        tk_ptr = per_thread_ht_buf->data();
    } else {
        ht_keys_local.resize(ht_total);
        tk_ptr = ht_keys_local.data();
    }
    ht_cnts_local.resize(ht_total);
    tc_ptr = ht_cnts_local.data();

    // Initialise EVERY per-thread slice (see find_pairs_with_face_count_2d
    // for the bug history that motivates the up-front fill).
    for (int t = 0; t < n_threads; ++t) {
        std::fill_n(tk_ptr + (size_t)t * ht_size, ht_size, HT_EMPTY);
        std::fill_n(tc_ptr + (size_t)t * ht_size, ht_size, 0u);
    }

    auto scan_band = [&](int t, int64_t y0, int64_t y1) {
        uint64_t* ht_k = tk_ptr + (size_t)t * ht_size;
        uint32_t* ht_c = tc_ptr + (size_t)t * ht_size;
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
                auto check_face = [&](T vj, bool forward) {
                    if (vj == 0) return;
                    if (vj == vi) { ++fc; return; }
                    if (!forward) return;
                    const uint64_t lo = vi < vj ? (uint64_t)vi : (uint64_t)vj;
                    const uint64_t hi = vi < vj ? (uint64_t)vj : (uint64_t)vi;
                    fp_fc_detail::ht_insert_count(ht_k, ht_c, ht_mask, (lo << 32) | hi);
                };
                if (yL)        check_face(labels[i - W], /*forward=*/false);  // N
                if (yH)        check_face(labels[i + W], /*forward=*/true);   // S
                if (x > 0)     check_face(labels[i - 1], /*forward=*/false);  // W
                if (x + 1 < W) check_face(labels[i + 1], /*forward=*/true);   // E
                face_count[i] = (uint8_t)fc;
                if (conn >= 2) {
                    auto check_diag = [&](T vj) {
                        if (vj == 0 || vj == vi) return;
                        const uint64_t lo = vi < vj ? (uint64_t)vi : (uint64_t)vj;
                        const uint64_t hi = vi < vj ? (uint64_t)vj : (uint64_t)vi;
                        fp_fc_detail::ht_insert_count(ht_k, ht_c, ht_mask, (lo << 32) | hi);
                    };
                    if (yH && x > 0)     check_diag(labels[i + W - 1]);  // SW
                    if (yH && x + 1 < W) check_diag(labels[i + W + 1]);  // SE
                }
            }
        }
    };

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

    // Merge per-thread tables. master_keys + master_cnts.
    ht_keys.assign(ht_size, HT_EMPTY);
    ht_cnts.assign(ht_size, 0u);
    for (int t = 0; t < n_threads; ++t) {
        const uint64_t* ht_k = tk_ptr + (size_t)t * ht_size;
        const uint32_t* ht_c = tc_ptr + (size_t)t * ht_size;
        for (uint64_t s = 0; s < ht_size; ++s) {
            const uint64_t key = ht_k[s];
            if (key == HT_EMPTY) continue;
            const uint64_t mslot =
                fp_fc_detail::ht_insert_count(ht_keys.data(), ht_cnts.data(),
                                                ht_mask, key);
            // ht_insert_count already set count to 1 if first time, or
            // incremented to ++existing. We want master count = sum of
            // per-thread counts, so adjust: replace with (current - 1 +
            // per-thread count).
            ht_cnts[mslot] += ht_c[s] - 1;
        }
    }
}


// Despur peel-back that ALSO decrements per-pair contact counts.
// When a spur pixel is reverted, every labelled face-AND-diagonal
// neighbour represents a (this_label, neighbour_label) contact that
// the pair just lost; we look it up and decrement. Pairs whose count
// drops to 0 become ghost edges that the caller will filter out when
// scanning ht_keys / ht_cnts for surviving pairs.
//
// face_count is updated in the same queue-based way as the standard
// despur_via_face_count_nd (decrement face-neighbours' counts on
// revert, re-enqueue if their count drops below threshold).
template <typename T>
inline int64_t despur_via_face_count_with_pair_decrement_2d(
    T* labels, uint8_t* face_count,
    int64_t H, int64_t W,
    uint64_t* ht_keys, uint32_t* ht_cnts, uint64_t ht_mask,
    int conn,                        // 1 or 2 — controls which neighbour offsets contribute
    int threshold = 1)
{
    const int64_t total = H * W;
    const int8_t th = (int8_t)threshold;

    // Queue initial spurs (still labelled — zeroing happens *during*
    // peel-back so that pair-count decrements see each adjacency with
    // both endpoints still set, including adjacencies where both
    // endpoints are themselves spurs).
    using QEnt = std::pair<int64_t, T>;
    std::vector<QEnt> queue;
    for (int64_t i = 0; i < total; ++i) {
        const T lab = labels[i];
        if (lab != 0 && (int8_t)face_count[i] <= th) queue.emplace_back(i, lab);
    }
    if (queue.empty()) return 0;

    auto decrement_pair = [&](T a, T b) {
        if (a == b || a == 0 || b == 0) return;
        const uint64_t lo = a < b ? (uint64_t)a : (uint64_t)b;
        const uint64_t hi = a < b ? (uint64_t)b : (uint64_t)a;
        const uint64_t key = (lo << 32) | hi;
        const uint64_t slot = fp_fc_detail::ht_lookup(ht_keys, ht_mask, key);
        if (slot == UINT64_MAX) return;
        if (ht_cnts[slot] > 0) --ht_cnts[slot];
    };

    // Peel-back. For each spur entry: while neighbours are still
    // labelled, walk all 8 (face + diag) and decrement pair counts
    // for different-label ones / decrement face_count for same-label
    // ones. THEN zero the label. Later cascade spurs see this pixel
    // as already-bg and skip it (correct — the lost adjacency between
    // it and the cascade pixel was decremented when this pixel was
    // processed).
    int64_t removed = 0;
    size_t head = 0;
    while (head < queue.size()) {
        const int64_t i = queue[head].first;
        const T old_lab  = queue[head].second;
        ++head;
        if (labels[i] == 0) continue;  // already reverted via cascade
        const int64_t y = i / W;
        const int64_t x = i - y * W;

        auto handle_face = [&](int64_t j) {
            const T jl = labels[j];
            if (jl == 0) return;
            if (jl == old_lab) {
                // Same-label face contact lost — decrement face_count[j]
                // and cascade if the neighbour becomes a spur.
                uint8_t fc = face_count[j];
                if (fc > 0) { fc = (uint8_t)(fc - 1); face_count[j] = fc; }
                if ((int8_t)fc <= th) {
                    queue.emplace_back(j, jl);
                    // ``labels[j]`` stays set until j's own loop iteration
                    // — that's how the next pixel-pair decrement against
                    // j's still-existing label works.
                }
            } else {
                decrement_pair(old_lab, jl);
            }
        };
        if (y > 0)     handle_face(i - W);
        if (y + 1 < H) handle_face(i + W);
        if (x > 0)     handle_face(i - 1);
        if (x + 1 < W) handle_face(i + 1);
        if (conn >= 2) {
            auto handle_diag = [&](int64_t j) {
                const T jl = labels[j];
                if (jl == 0 || jl == old_lab) return;
                decrement_pair(old_lab, jl);
            };
            if (y > 0     && x > 0)     handle_diag(i - W - 1);
            if (y > 0     && x + 1 < W) handle_diag(i - W + 1);
            if (y + 1 < H && x > 0)     handle_diag(i + W - 1);
            if (y + 1 < H && x + 1 < W) handle_diag(i + W + 1);
        }
        // Now safe to zero this pixel; subsequent peel-back iterations
        // (including cascades enqueued just above) see labels[i] == 0
        // and skip the redundant decrement.
        labels[i] = 0;
        ++removed;
    }
    return removed;
}


// Materialise the pair list from the merged HT, keeping only pairs
// with count > 0 (i.e. those that survived ghost-edge pruning).
inline std::vector<std::pair<int32_t, int32_t>>
collect_surviving_pairs(const std::vector<uint64_t>& ht_keys,
                         const std::vector<uint32_t>& ht_cnts)
{
    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(128);
    const size_t n = ht_keys.size();
    for (size_t s = 0; s < n; ++s) {
        const uint64_t key = ht_keys[s];
        if (key == HT_EMPTY) continue;
        if (ht_cnts[s] == 0) continue;  // ghost — all contacts pruned
        const int32_t lo = (int32_t)(key >> 32);
        const int32_t hi = (int32_t)(key & 0xFFFFFFFFu);
        out.emplace_back(lo, hi);
    }
    return out;
}


// Backwards-compat thin wrapper: the v1 API (returns pairs directly).
// Used by the pybind11 testing entry; not by Solver::label.
template <typename T>
inline std::vector<std::pair<int32_t, int32_t>>
find_pairs_with_face_count_2d(
    const T* labels, int64_t H, int64_t W,
    int conn,
    uint8_t* face_count,
    uint64_t ht_size, int n_threads,
    ForkJoinPool& pool,
    std::vector<uint64_t>* ht_scratch = nullptr)
{
    std::vector<uint64_t> ht_keys;
    std::vector<uint32_t> ht_cnts;
    find_pairs_with_face_count_2d_v2(
        labels, H, W, conn, face_count,
        ht_keys, ht_cnts, ht_size, n_threads, pool, ht_scratch);
    return collect_surviving_pairs(ht_keys, ht_cnts);
}

}  // namespace ncolor_cpp
