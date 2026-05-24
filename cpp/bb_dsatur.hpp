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

    // Iterative backtracking. Recursion depth = N, which blows past
    // worker-thread stack limits (512KB on macOS) once N > ~3000;
    // using an explicit heap-allocated stack instead keeps the
    // algorithm safe for graphs of any size. Behaviour matches the
    // original recursive search node-for-node.
    bool recurse() {
        // Per-level state. ``next_c`` is the next color to try
        // (1..k+1); the color currently applied to ``u`` (if any) is
        // ``next_c - 1`` whenever it is in [1, k].
        struct Frame {
            int32_t u;
            uint8_t next_c;
            std::vector<int32_t> touched;
        };
        std::vector<Frame> stack;
        stack.reserve(64);

        // Per-node bookkeeping. Returns:
        //   PUSHED   → frame pushed for a freshly-picked u, keep searching
        //   COMPLETE → all vertices coloured; return true with colors[]
        //              intact (do NOT unwind — that would undo the answer)
        //   LIMIT    → cancel/deadline/budget hit; unwind and return false
        enum DResult : int { PUSHED = 0, COMPLETE = 1, LIMIT = 2 };
        auto descend = [&]() -> DResult {
            // Sibling-cancel: check on every descent so a winner from
            // a sibling worker terminates the entire search tree
            // promptly. Relaxed-load is essentially free with branch
            // prediction once the flag flips.
            if (cancel && cancel->load(std::memory_order_relaxed)) return LIMIT;
            // Wall-clock deadline: per-node cost is O(N) (pick_next is
            // a linear scan), so a fixed node budget can translate to
            // seconds on large graphs. Check every descent when a
            // deadline is set — steady_clock cost (~50-100 ns) is
            // negligible vs the ~5 µs per descent at N=5000.
            if (deadline_ns > 0 &&
                std::chrono::steady_clock::now()
                    .time_since_epoch().count() > deadline_ns) return LIMIT;
            ++node_count;
            if (node_budget > 0 && node_count > node_budget) return LIMIT;
            const int32_t u = pick_next();
            if (u < 0) return COMPLETE;
            Frame f;
            f.u = u;
            f.next_c = 1;
            f.touched.reserve(32);
            stack.push_back(std::move(f));
            return PUSHED;
        };

        // Undo every still-applied color (only used when a limit fires).
        auto unwind_all = [&]() {
            while (!stack.empty()) {
                Frame& cur = stack.back();
                const uint8_t applied = (uint8_t)(cur.next_c - 1);
                if (applied >= 1 && applied <= (uint8_t)k) {
                    unapply(cur.u, applied, cur.touched);
                }
                stack.pop_back();
            }
        };

        DResult r = descend();
        if (r == COMPLETE) return true;
        if (r == LIMIT)    { unwind_all(); return false; }

        while (!stack.empty()) {
            Frame& f = stack.back();
            // Try the next allowed color on this frame.
            bool applied_one = false;
            while (f.next_c <= (uint8_t)k) {
                const uint8_t c = f.next_c++;
                if (forbidden[f.u] & (1u << c)) continue;
                f.touched.clear();
                apply(f.u, c, f.touched);
                applied_one = true;
                break;
            }
            if (applied_one) {
                // "Recursive call" → push a new frame for the next vertex.
                DResult rr = descend();
                if (rr == COMPLETE) return true;
                if (rr == LIMIT)    { unwind_all(); return false; }
                continue;  // PUSHED: keep going
            }
            // Exhausted all colors for f.u → backtrack.
            stack.pop_back();
            if (stack.empty()) return false;
            Frame& parent = stack.back();
            const uint8_t parent_c = (uint8_t)(parent.next_c - 1);
            unapply(parent.u, parent_c, parent.touched);
            // parent.next_c is already past parent_c, so the outer
            // loop will try parent_c + 1 on the next iteration.
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
