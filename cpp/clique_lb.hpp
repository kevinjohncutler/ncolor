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
// memory on huge graphs. Reuses the same bit-matrix machinery as
// k5_detect, but generalises to arbitrary clique size.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <vector>

namespace ncolor_cpp {

// Bit-set helpers over a vector<uint64_t> view of length words.
namespace bk_detail {

inline int popcount_bits(const uint64_t* a, int words) {
    int c = 0;
    for (int i = 0; i < words; ++i) c += __builtin_popcountll(a[i]);
    return c;
}

inline bool bit_get(const uint64_t* a, int i) {
    return (a[i >> 6] >> (i & 63)) & 1ULL;
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

// Iterate set bits of `a` via __builtin_ctzll on each word.
// Calls fn(int bit) for each set bit; fn may return true to abort.
template <typename F>
inline void for_each_bit(const uint64_t* a, int words, F&& fn) {
    for (int wi = 0; wi < words; ++wi) {
        uint64_t w = a[wi];
        while (w) {
            const int b = __builtin_ctzll(w);
            w &= w - 1;
            if (fn(wi * 64 + b)) return;
        }
    }
}

}  // namespace bk_detail

// Bron-Kerbosch state holder. Reused across recursive calls via a
// pre-allocated scratch buffer of (depth × words_per_row) uint64s.
struct BKState {
    int N;
    int words;                            // words_per_row = (N + 63) / 64
    const uint64_t* adj;                  // N rows × words uint64
    int best_clique = 0;                  // largest clique found so far
    std::vector<int> best_clique_members; // vertices forming `best_clique`
    std::vector<int> current_R;           // running clique (mutated by bk)
    int64_t deadline_ns = 0;              // 0 = no wall cap
    bool deadline_hit = false;
    int target = 0;                       // if best_clique >= target, stop

    // Scratch buffers: depth × words. Avoid reallocation in recursion.
    std::vector<uint64_t> scratch;        // size = N * words (max depth N)
    int max_depth_used = 0;

    bool past_deadline() {
        if (deadline_ns == 0) return false;
        const auto now = std::chrono::steady_clock::now()
                          .time_since_epoch().count();
        return now > deadline_ns;
    }

    void bk(int R_size, uint64_t* P, uint64_t* X, int depth) {
        if (deadline_hit) return;
        if (best_clique >= target && target > 0) return;
        if (depth > max_depth_used) max_depth_used = depth;

        // Update best with current R if P is empty.
        const int Pcount = bk_detail::popcount_bits(P, words);
        if (Pcount == 0) {
            if (R_size > best_clique) {
                best_clique = R_size;
                best_clique_members.assign(current_R.begin(),
                                            current_R.begin() + R_size);
            }
            return;
        }
        // Prune: if R_size + Pcount <= best_clique, can't extend.
        if (R_size + Pcount <= best_clique) return;

        // Periodic deadline check (every recursion entry).
        if ((depth & 0xff) == 0 && past_deadline()) {
            deadline_hit = true;
            return;
        }

        // Pivot: choose u in P ∪ X with most neighbours in P.
        // (Skip X since we only need to PARTITION the recursive call
        // set, and a P-pivot is sufficient.) For simplicity use the
        // first vertex of P as pivot — degrades worst-case but keeps
        // the loop tight.
        int pivot = -1;
        bk_detail::for_each_bit(P, words, [&](int v) {
            pivot = v;
            return true;
        });

        // candidates = P \ N(pivot). Recurse on each candidate v:
        //   bk(R ∪ {v}, P ∩ N(v), X ∩ N(v))
        // Pre-allocate work bitsets for P_new and X_new at this depth.
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
            std::vector<uint64_t> X_new(words);
            bk_detail::bit_and(P_new.data(), P, adj + (size_t)v * words, words);
            bk_detail::bit_and(X_new.data(), X, adj + (size_t)v * words, words);
            current_R.push_back(v);
            bk(R_size + 1, P_new.data(), X_new.data(), depth + 1);
            current_R.pop_back();
            // Move v from P to X for siblings.
            bk_detail::bit_clear(P, v);
            bk_detail::bit_set(X, v);
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
// Build bit-packed adjacency from CSR. Returns words_per_row.
inline int build_bit_adj(
    int32_t N, const int32_t* indptr, const int32_t* indices,
    std::vector<uint64_t>& adj_out)
{
    const int words = (N + 63) / 64;
    adj_out.assign((size_t)N * (size_t)words, 0);
    for (int32_t u = 0; u < N; ++u) {
        uint64_t* row = adj_out.data() + (size_t)u * (size_t)words;
        const int32_t end = indptr[u + 1];
        for (int32_t k = indptr[u]; k < end; ++k) {
            const int32_t v = indices[k];
            if (v >= 0 && v < N) bk_detail::bit_set(row, v);
        }
    }
    return words;
}

// Run Bron-Kerbosch on the bit-packed adjacency stored in `adj`.
// Fills `s.best_clique` and `s.best_clique_members`. Uses `s.scratch`
// as preallocated working space. Caller is responsible for setting
// `s.N`, `s.words`, `s.adj`, `s.target`, `s.deadline_ns` and resizing
// `s.scratch` to N×words uint64s.
inline void run_bk(BKState& s) {
    const int N = s.N;
    const int words = s.words;
    const uint64_t* adj = s.adj;

    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        // Higher degree = more likely to be in a big clique.
        // Compare popcount of adjacency row bits.
        int da = 0, db = 0;
        const uint64_t* ra = adj + (size_t)a * (size_t)words;
        const uint64_t* rb = adj + (size_t)b * (size_t)words;
        for (int w = 0; w < words; ++w) { da += __builtin_popcountll(ra[w]); db += __builtin_popcountll(rb[w]); }
        return da > db;
    });

    std::vector<uint64_t> P_root(words, 0);
    std::vector<uint64_t> X_root(words, 0);
    for (int i = 0; i < N; ++i) bk_detail::bit_set(P_root.data(), i);

    s.best_clique = 1;
    s.best_clique_members.assign(1, 0);
    s.current_R.clear();

    for (int v : order) {
        if (s.deadline_hit) break;
        if (s.target > 0 && s.best_clique >= s.target) break;

        std::vector<uint64_t> P_v(words);
        std::vector<uint64_t> X_v(words);
        bk_detail::bit_and(P_v.data(), P_root.data(), adj + (size_t)v * words, words);
        bk_detail::bit_and(X_v.data(), X_root.data(), adj + (size_t)v * words, words);

        s.current_R.push_back(v);
        s.bk(1, P_v.data(), X_v.data(), 1);
        s.current_R.pop_back();

        bk_detail::bit_clear(P_root.data(), v);
        bk_detail::bit_set(X_root.data(), v);
    }
}

// Find ONE clique of size ≥ `target` in `adj_bits` (a working copy
// that the caller may have edited to remove edges). Returns true if
// found; on success `out_members` is filled with the clique vertices.
inline bool find_one_clique_at_least(
    int32_t N, std::vector<uint64_t>& adj_bits, int words,
    int target, int64_t deadline_ns,
    std::vector<int>& out_members)
{
    BKState s;
    s.N = N;
    s.words = words;
    s.adj = adj_bits.data();
    s.target = target;
    s.deadline_ns = deadline_ns;
    s.scratch.assign((size_t)N * (size_t)words, 0);
    run_bk(s);
    if (s.best_clique >= target) {
        out_members = s.best_clique_members;
        return true;
    }
    return false;
}

// Iteratively prune r=2-only edges to break K_{target_omega} cliques
// in the graph until ω < target_omega or no breakable edge remains.
//
// `r1_lookup(a, b)` should return true iff (a, b) is an r=1 edge.
// `out_pruned` receives the removed (lo, hi) edges (0-indexed).
//
// Returns the final ω lower bound (post-pruning). If equal to
// `target_omega - 1` or less, the graph is now provably χ ≥ target_omega
// breakable — i.e., the K_target_omega obstruction is gone.
template <typename R1Lookup>
inline int prune_breakable_cliques(
    int32_t N, const int32_t* indptr, const int32_t* indices,
    R1Lookup&& r1_lookup,
    int target_omega,
    int max_iters,
    int64_t deadline_ns,
    std::vector<std::pair<int32_t, int32_t>>& out_pruned,
    int32_t max_N = 20000)
{
    if (N < target_omega || N > max_N) return 0;
    std::vector<uint64_t> adj;
    const int words = build_bit_adj(N, indptr, indices, adj);

    out_pruned.clear();
    int last_omega = target_omega;  // assume we'll find at least one
    for (int it = 0; it < max_iters; ++it) {
        std::vector<int> clique;
        const bool found = find_one_clique_at_least(
            N, adj, words, target_omega, deadline_ns, clique);
        if (!found) {
            last_omega = target_omega - 1;  // ω is below target now
            break;
        }
        // Pick an r=2-only edge in the clique (strategy d: first found).
        int ra = -1, rb = -1;
        for (size_t i = 0; i < clique.size() && ra < 0; ++i) {
            for (size_t j = i + 1; j < clique.size(); ++j) {
                const int a = clique[i], b = clique[j];
                const int32_t lo = std::min(a, b);
                const int32_t hi = std::max(a, b);
                if (!r1_lookup(lo, hi)) {
                    ra = lo; rb = hi;
                    break;
                }
            }
        }
        if (ra < 0) {
            // Clique made of all r=1 edges — can't break by pruning.
            last_omega = target_omega;
            break;
        }
        // Remove edge in both rows of adj.
        bk_detail::bit_clear(adj.data() + (size_t)ra * words, rb);
        bk_detail::bit_clear(adj.data() + (size_t)rb * words, ra);
        out_pruned.emplace_back(ra, rb);
    }
    return last_omega;
}

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
    s.N = N;
    s.words = words;
    s.adj = adj.data();
    s.deadline_ns = deadline_ns;
    s.target = target;
    s.best_clique = 1;  // singleton vertices are trivially clique-1
    s.scratch.assign((size_t)N * (size_t)words, 0);

    // Degeneracy ordering would be ideal but adds complexity; for
    // sparse cell graphs the naive vertex order with pivoting is fast
    // enough. Process vertices in DESCENDING degree to find big
    // cliques early (helps the prune kick in sooner).
    std::vector<int> order(N);
    for (int i = 0; i < N; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return (indptr[a + 1] - indptr[a]) > (indptr[b + 1] - indptr[b]);
    });

    std::vector<uint64_t> P_root(words, 0);
    std::vector<uint64_t> X_root(words, 0);
    for (int i = 0; i < N; ++i) bk_detail::bit_set(P_root.data(), i);

    // Process each vertex as a singleton {v} with restricted P, X.
    for (int v : order) {
        if (s.deadline_hit) break;
        if (s.target > 0 && s.best_clique >= s.target) break;

        // P_v = P ∩ adj(v); X_v = X ∩ adj(v)
        std::vector<uint64_t> P_v(words);
        std::vector<uint64_t> X_v(words);
        bk_detail::bit_and(P_v.data(), P_root.data(), adj.data() + (size_t)v * words, words);
        bk_detail::bit_and(X_v.data(), X_root.data(), adj.data() + (size_t)v * words, words);

        s.bk(1, P_v.data(), X_v.data(), 1);

        // Remove v from P, add to X for subsequent iterations.
        bk_detail::bit_clear(P_root.data(), v);
        bk_detail::bit_set(X_root.data(), v);
    }

    return s.best_clique;
}

}  // namespace ncolor_cpp
