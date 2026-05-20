// Branch-and-bound k-coloring using DSatur vertex ordering.
//
// Brélaz (1979) DSatur: at each step color the uncolored vertex with
// maximum saturation degree (number of distinct colors among its
// already-colored neighbors). Adds backtracking on top: when a
// vertex has all k colors forbidden, undo recent choices and try a
// different color.
//
// Forward checking via per-vertex forbidden-color bitmask (uint8, K≤4):
// before recursing we update neighbors' masks; on backtrack we undo
// only the bits we actually added (tracked per-recursion).
//
// For K_5-free graphs of a few hundred to a few thousand vertices that
// ARE k-colorable, the search tree stays small and the algorithm is
// effectively linear with low constants. Empirically faster than SAT
// on the kind of "near-planar" cell-adjacency graphs we hit.
//
// Returns true if a k-coloring exists; on success `colors[]` is filled
// (1-indexed color values).

#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <cstdint>
#include <climits>
#include <algorithm>

namespace ncolor_cpp {

struct BBDSatur {
    int32_t N;
    int32_t k;
    const int32_t* indptr;
    const int32_t* indices;
    std::vector<uint8_t> colors;     // [N] 0=uncolored, else 1..k
    std::vector<uint8_t> forbidden;  // [N] bitmask of forbidden colors (bit c)
    std::vector<int32_t> sat_deg;    // [N] cached saturation (popcount of forbidden)
    int64_t node_count = 0;
    int64_t node_budget = 0;         // 0 = unbounded
    int64_t deadline_ns = 0;         // 0 = no wall-clock cap
    const std::atomic<bool>* cancel = nullptr;  // sibling-cancel flag

    // Pick the uncolored vertex with the highest saturation degree.
    // Ties: higher actual degree first. Linear scan; we run on small N
    // (≤ few thousand) so a heap would be overkill.
    int32_t pick_next() const {
        int32_t best = -1;
        int32_t best_sat = -1;
        int32_t best_deg = -1;
        for (int32_t u = 0; u < N; ++u) {
            if (colors[u] != 0) continue;
            const int32_t s = sat_deg[u];
            if (s > best_sat ||
                (s == best_sat && (indptr[u + 1] - indptr[u]) > best_deg)) {
                best = u;
                best_sat = s;
                best_deg = indptr[u + 1] - indptr[u];
            }
        }
        return best;
    }

    // Apply color c to u; propagate to neighbors' forbidden masks.
    // Returns list of (vertex, was_already_forbidden) so backtrack can
    // undo only the bits this call actually set.
    void apply(int32_t u, uint8_t c, std::vector<int32_t>& touched) {
        colors[u] = c;
        const uint8_t bit = (uint8_t)(1u << c);
        const int32_t kend = indptr[u + 1];
        for (int32_t k_ = indptr[u]; k_ < kend; ++k_) {
            const int32_t v = indices[k_];
            if (colors[v] != 0) continue;
            if (!(forbidden[v] & bit)) {
                forbidden[v] |= bit;
                ++sat_deg[v];
                touched.push_back(v);
            }
        }
    }

    void unapply(int32_t u, uint8_t c, const std::vector<int32_t>& touched) {
        colors[u] = 0;
        const uint8_t bit = (uint8_t)(1u << c);
        for (int32_t v : touched) {
            forbidden[v] &= (uint8_t)~bit;
            --sat_deg[v];
        }
    }

    // Recursive backtracking. Returns true on success.
    bool recurse() {
        // Sibling-cancel: check every recurse() so a winner found by
        // another worker terminates the entire search tree quickly.
        // Checking only every N nodes is WRONG: when cancel returns
        // false from a deep child, the parent's `for c=1..k` loop
        // moves on to the next color and re-descends — without an
        // entry-point check, unwinding the recursion stack would
        // take O(depth × k) extra mod-N intervals (~10 ms for our
        // graphs). Relaxed-load on x86 is essentially free with
        // branch prediction once the flag flips.
        // Wall-clock deadline: same logic — per-node cost is O(N)
        // (pick_next is a linear scan), so even checking every 1024
        // nodes is too rare when N is in the thousands (a fixed node
        // budget at large N translates to seconds). Check every
        // recurse() when a deadline is set. Steady-clock cost is
        // ~50-100 ns per call vs ~5 µs per recurse at N=5000 —
        // negligible overhead in the regime where deadlines matter.
        if (cancel && cancel->load(std::memory_order_relaxed)) return false;
        if (deadline_ns > 0 &&
            std::chrono::steady_clock::now()
                .time_since_epoch().count() > deadline_ns) return false;
        ++node_count;
        if (node_budget > 0 && node_count > node_budget) return false;
        const int32_t u = pick_next();
        if (u < 0) return true;  // all colored

        const uint8_t forb = forbidden[u];
        // Try colors in 1..k order.
        std::vector<int32_t> touched;
        touched.reserve(32);
        for (int32_t c = 1; c <= k; ++c) {
            if (forb & (1u << c)) continue;
            touched.clear();
            apply(u, (uint8_t)c, touched);
            if (recurse()) return true;
            unapply(u, (uint8_t)c, touched);
        }
        return false;
    }
};

// Solve k-coloring via B&B + DSatur. `colors` is output (1..k on
// success). Returns true iff a k-coloring exists.
//
// node_budget: cap on recursion node count (0 = unbounded). Useful as
// a safety net; for K_5-free graphs of our size the tree is small in
// practice (1-2× N nodes typical).
inline bool bb_dsatur(
    const int32_t* indptr, const int32_t* indices, int32_t N, int32_t k,
    std::vector<uint8_t>& colors,
    int64_t node_budget = 0,
    const std::atomic<bool>* cancel = nullptr,
    int64_t deadline_ns = 0)
{
    if (N <= 0) { colors.clear(); return true; }
    if (k <= 0 || k > 7) return false;  // bitmask is uint8, supports up to k=7

    BBDSatur s;
    s.N = N; s.k = k;
    s.indptr = indptr; s.indices = indices;
    s.colors.assign((size_t)N, 0);
    s.forbidden.assign((size_t)N, 0);
    s.sat_deg.assign((size_t)N, 0);
    s.node_budget = node_budget;
    s.deadline_ns = deadline_ns;
    s.cancel = cancel;
    if (s.recurse()) {
        colors = std::move(s.colors);
        return true;
    }
    return false;
}

}  // namespace ncolor_cpp
