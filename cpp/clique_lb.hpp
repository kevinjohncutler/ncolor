// Clique-number lower bound via Bron-Kerbosch with pivoting and a
// wall-clock deadline.
//
// χ(G) ≥ ω(G) (chromatic ≥ clique number). When ω is high we can skip
// directly to cur_n = max(user_n_colors, ω) instead of bumping by 1
// at each depth iteration. For 2D conn=2 segmentations ω is usually
// 4-5; for 3D ω can be 5-8+ depending on packing.
//
// Partial searches are still useful: any clique found is a valid
// lower bound on ω. The deadline lets us early-terminate without
// completing the search. Returns the LARGEST clique discovered so far.
//
// Bit-packed adjacency for O(N²/64) memory; gate by N to avoid blowing
// memory on huge graphs.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "intrinsics.hpp"

#include "timing.hpp"

namespace ncolor_cpp {

// Bit-set helpers over a vector<uint64_t> view of length words.
namespace bk_detail {

inline int popcount_bits(const uint64_t* a, int words) {
    int c = 0;
    for (int i = 0; i < words; ++i) c += popcount_u64(a[i]);
    return c;
}

inline void bit_set(uint64_t* a, int i) {
    a[i >> 6] |= (1ULL << (i & 63));
}

inline void bit_clear(uint64_t* a, int i) {
    a[i >> 6] &= ~(1ULL << (i & 63));
}

// dst = a AND b
inline void bit_and(uint64_t* dst, const uint64_t* a, const uint64_t* b, int words) {
    for (int i = 0; i < words; ++i) dst[i] = a[i] & b[i];
}

// dst = a AND NOT b
inline void bit_andn(uint64_t* dst, const uint64_t* a, const uint64_t* b, int words) {
    for (int i = 0; i < words; ++i) dst[i] = a[i] & ~b[i];
}

// Iterate set bits of `a` via ctz_u64 on each word.
// Calls fn(int bit) for each set bit; fn may return true to abort.
template <typename F>
inline void for_each_bit(const uint64_t* a, int words, F&& fn) {
    for (int wi = 0; wi < words; ++wi) {
        uint64_t w = a[wi];
        while (w) {
            const int b = ctz_u64(w);
            w &= w - 1;
            if (fn(wi * 64 + b)) return;
        }
    }
}

}  // namespace bk_detail

// Bron-Kerbosch state holder. Reused across recursive calls via a
// pre-allocated scratch buffer of (depth × words_per_row) uint64s.
struct BKState {
    int words;                            // words_per_row = (N + 63) / 64
    const uint64_t* adj;                  // N rows × words uint64
    int best_clique = 0;                  // largest clique found so far
    int64_t deadline_ns = 0;              // 0 = no wall cap
    bool deadline_hit = false;
    int target = 0;                       // if best_clique >= target, stop

    // Scratch buffers: depth × words. Avoid reallocation in recursion.
    std::vector<uint64_t> scratch;        // size = N * words (max depth N)
    uint64_t visited_nodes = 0;

    bool past_deadline() {
        if (deadline_ns == 0) return false;
        const auto now = steady_time_ns();
        return now > deadline_ns;
    }

    void bk(int R_size, uint64_t* P, int depth) {
        if (deadline_hit) return;
        if (best_clique >= target && target > 0) return;
        // Check the first entry, then every 256 search nodes, even in
        // shallow searches that never reach depth 256.
        if ((visited_nodes++ & 0xff) == 0 && past_deadline()) {
            deadline_hit = true;
            return;
        }

        // Update best with current R if P is empty.
        const int Pcount = bk_detail::popcount_bits(P, words);
        if (Pcount == 0) {
            if (R_size > best_clique) {
                best_clique = R_size;
            }
            return;
        }
        // Prune: if R_size + Pcount <= best_clique, can't extend.
        if (R_size + Pcount <= best_clique) return;

        // Pivot: first vertex of P. Only candidate partitioning is needed
        // for a size bound, so no excluded-vertex set is maintained. The "most
        // neighbors in P" pivot would be a tighter bound but the loop
        // to find it costs more than it saves on our sparse graphs.
        int pivot = -1;
        bk_detail::for_each_bit(P, words, [&](int v) {
            pivot = v;
            return true;
        });

        // candidates = P \ N(pivot). Recurse on each candidate v:
        //   bk(R union {v}, P intersect N(v))
        // Keep the candidate bitset in scratch at this depth.
        uint64_t* cand = &scratch[(size_t)depth * words];
        if (pivot >= 0) {
            bk_detail::bit_andn(cand, P, adj + (size_t)pivot * words, words);
        } else {
            std::copy(P, P + words, cand);
        }

        // For each v in cand, recurse.
        bk_detail::for_each_bit(cand, words, [&](int v) -> bool {
            if (deadline_hit) return true;
            // P_new = P ∩ adj(v)
            std::vector<uint64_t> P_new(words);
            bk_detail::bit_and(P_new.data(), P, adj + (size_t)v * words, words);
            bk(R_size + 1, P_new.data(), depth + 1);
            // Remove v from P for siblings.
            bk_detail::bit_clear(P, v);
            return false;
        });
    }
};

// Compute a lower bound on the clique number ω(G).
//
// Returns the size of the largest clique found within deadline_ns.
// If `target > 0`, abort early once a clique of size `target` is found
// (useful when you only need to know "is ω >= k").
//
// `max_N`: skip if N exceeds this (returns 1, a trivial lower bound).
// The adjacency bit-matrix is N²/64 bits; 3 MB at N=5128, 50 MB at
// N=20000.
inline int clique_lower_bound(
    int32_t N,
    const int32_t* indptr,
    const int32_t* indices,
    int target = 0,
    int64_t deadline_ns = 0,
    int32_t max_N = 20000)
{
    if (N < 1) return 0;
    if (N == 1) return 1;
    if (N > max_N) return 1;  // trivial bound; skip to avoid memory blow-up

    const int words = (N + 63) / 64;
    std::vector<uint64_t> adj((size_t)N * (size_t)words, 0);
    for (int32_t u = 0; u < N; ++u) {
        uint64_t* row = adj.data() + (size_t)u * (size_t)words;
        const int32_t end = indptr[u + 1];
        for (int32_t k = indptr[u]; k < end; ++k) {
            const int32_t v = indices[k];
            if (v >= 0 && v < N) bk_detail::bit_set(row, v);
        }
    }

    BKState s;
    s.words = words;
    s.adj = adj.data();
    s.deadline_ns = deadline_ns;
    s.target = target;
    s.best_clique = 1;  // singleton vertices are trivially clique-1
    s.scratch.assign((size_t)N * (size_t)words, 0);

    // Descending-degree vertex order: find big cliques early so the
    // size prune kicks in sooner.
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return (indptr[a + 1] - indptr[a]) > (indptr[b + 1] - indptr[b]);
    });

    std::vector<uint64_t> P_root(words, 0);
    for (int i = 0; i < N; ++i) bk_detail::bit_set(P_root.data(), i);

    // Process each vertex as a singleton {v} with restricted candidates.
    for (int v : order) {
        if (s.deadline_hit) break;
        if (s.target > 0 && s.best_clique >= s.target) break;

        // P_v = P intersect adj(v)
        std::vector<uint64_t> P_v(words);
        bk_detail::bit_and(P_v.data(), P_root.data(), adj.data() + (size_t)v * words, words);

        s.bk(1, P_v.data(), 1);

        // Remove v from P for subsequent iterations.
        bk_detail::bit_clear(P_root.data(), v);
    }

    return s.best_clique;
}

}  // namespace ncolor_cpp
