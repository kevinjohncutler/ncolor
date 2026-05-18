// TabuCol: tabu search for k-coloring.
//
// Used as a fallback in the labeler when the local-greedy repair fails
// to clean up a conflicted k-coloring. The greedy + Welsh-Powell + BFS
// attempts handle ~all real-world inputs; tabu search rescues the rare
// hard cases where a k-coloring exists (verified by SAT) but every
// vertex-ordering greedy hits the same local minimum.
//
// Classical TabuCol (Hertz & de Werra, 1987):
//   - State: a colouring with possible conflicts.
//   - Move: change one conflicting vertex's colour.
//   - Move choice: best Δconflicts, ties broken at random.
//   - Tabu: forbid (u, c_old) for some tenure after u→c_new; aspiration
//     allows tabu moves that strictly reduce conflicts.
//   - Stop: zero conflicts (success) or max_iters reached (failure).

#pragma once

#include <atomic>
#include <vector>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <chrono>

namespace ncolor_cpp {

// Returns true iff the function found a k-colouring with zero
// conflicts. `colors` is read+written in place. Inputs are CSR
// adjacency; `colors[v]` must be in 1..k for every v in 1..N on entry
// (call sites pass the conflicted greedy output, which already
// satisfies this).
// `deadline_ns`: if > 0, abort early when steady_clock::now() exceeds
// this absolute timestamp (caller computes start + budget). 0 = no
// deadline (only max_iters governs). Checked every 256 iters.
// `cancel`: optional atomic bool; when set to true by another thread
// (e.g. a sibling parallel worker that already found a winner), this
// call aborts early and returns false. Checked every 256 iters
// alongside the deadline. Used to bound parallel-race latency by the
// FIRST success, not the slowest in-flight worker's full budget.
inline bool tabucol(
    const int32_t* indptr, const int32_t* indices, int32_t N,
    int32_t k, int max_iters,
    std::vector<uint8_t>& colors,
    uint64_t seed = 0,
    int64_t deadline_ns = 0,
    const std::atomic<bool>* cancel = nullptr)
{
    if (k <= 1) return false;

    auto cidx = [&](int32_t u, int32_t c) -> size_t {
        return (size_t)u * (size_t)(k + 1) + (size_t)c;
    };

    // conf[u, c] = number of u's neighbours coloured c. Vertices and
    // CSR indices are 0-based (matching the rest of the ncolor C++
    // pipeline).
    std::vector<int32_t> conf((size_t)N * (size_t)(k + 1), 0);
    for (int32_t u = 0; u < N; ++u) {
        for (int32_t j = indptr[u]; j < indptr[u + 1]; ++j) {
            int32_t v = indices[j];
            ++conf[cidx(u, colors[v])];
        }
    }
    int32_t total_conf = 0;
    for (int32_t u = 0; u < N; ++u) {
        total_conf += conf[cidx(u, colors[u])];
    }
    total_conf /= 2;
    if (total_conf == 0) return true;

    // tabu[u, c] = first iter at which u→c is permitted again.
    std::vector<int32_t> tabu((size_t)N * (size_t)(k + 1), 0);

    uint64_t rs = seed ? seed : 0x9e3779b97f4a7c15ULL;
    auto next32 = [&]() -> uint32_t {
        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
        return (uint32_t)(rs >> 32);
    };

    // Best-move scan buffer (vertices × colours).
    std::vector<std::pair<int32_t, int32_t>> best_moves;
    best_moves.reserve(64);

    static const bool dbg_traj = std::getenv("NCOLOR_TABU_TRAJ") != nullptr;
    int best_total = total_conf;
    int iters_since_best = 0;
    if (dbg_traj) {
        std::fprintf(stderr, "[tabu seed=%llu] start total_conf=%d\n",
                      (unsigned long long)seed, total_conf);
    }
    for (int it = 0; it < max_iters; ++it) {
        if (total_conf == 0) {
            if (dbg_traj) std::fprintf(stderr,
                "[tabu seed=%llu] SOLVED at it=%d\n",
                (unsigned long long)seed, it);
            return true;
        }
        if ((it & 0xff) == 0) {
            if (cancel && cancel->load(std::memory_order_relaxed)) {
                if (dbg_traj) std::fprintf(stderr,
                    "[tabu seed=%llu] CANCELLED it=%d final=%d best=%d\n",
                    (unsigned long long)seed, it, total_conf, best_total);
                return false;
            }
            if (deadline_ns > 0) {
                const auto now = std::chrono::steady_clock::now()
                                  .time_since_epoch().count();
                if (now > deadline_ns) {
                    if (dbg_traj) std::fprintf(stderr,
                        "[tabu seed=%llu] DEADLINE it=%d final=%d best=%d\n",
                        (unsigned long long)seed, it, total_conf, best_total);
                    return false;
                }
            }
        }
        if (dbg_traj && (it % 500 == 0)) {
            std::fprintf(stderr,
                "[tabu seed=%llu] it=%d total_conf=%d best=%d\n",
                (unsigned long long)seed, it, total_conf, best_total);
        }

        int best_delta = INT_MAX;
        best_moves.clear();

        for (int32_t u = 0; u < N; ++u) {
            const uint8_t cu = colors[u];
            const int32_t cur = conf[cidx(u, cu)];
            if (cur == 0) continue;  // u not in conflict
            for (int32_t c = 1; c <= k; ++c) {
                if (c == cu) continue;
                const int32_t nx = conf[cidx(u, c)];
                const int32_t delta = nx - cur;
                // Aspiration: skip tabu moves UNLESS they strictly
                // reduce conflicts (delta < 0). Matches the classical
                // TabuCol aspiration criterion.
                const bool is_tabu = tabu[cidx(u, c)] > it;
                if (is_tabu && delta >= 0) continue;
                if (delta < best_delta) {
                    best_delta = delta;
                    best_moves.clear();
                    best_moves.emplace_back(u, c);
                } else if (delta == best_delta) {
                    best_moves.emplace_back(u, c);
                }
            }
        }
        if (best_moves.empty()) {
            if (dbg_traj) std::fprintf(stderr,
                "[tabu seed=%llu] EMPTY_MOVES it=%d total_conf=%d best=%d "
                "iters_since_best=%d\n",
                (unsigned long long)seed, it, total_conf, best_total,
                iters_since_best);
            return false;
        }

        auto [u, c_new] = best_moves[next32() % best_moves.size()];
        const uint8_t c_old = colors[u];
        colors[u] = (uint8_t)c_new;
        for (int32_t j = indptr[u]; j < indptr[u + 1]; ++j) {
            const int32_t v = indices[j];
            --conf[cidx(v, c_old)];
            ++conf[cidx(v, c_new)];
        }
        total_conf += best_delta;
        if (total_conf < best_total) {
            best_total = total_conf;
            iters_since_best = 0;
        } else {
            ++iters_since_best;
        }
        // Tabu the reverse move (u → c_old).
        tabu[cidx(u, c_old)] = it + 10 + (int)(next32() % (total_conf + 1));
    }
    if (dbg_traj) std::fprintf(stderr,
        "[tabu seed=%llu] MAXITER total_conf=%d best=%d "
        "iters_since_best=%d\n",
        (unsigned long long)seed, total_conf, best_total, iters_since_best);
    return total_conf == 0;
}

}  // namespace ncolor_cpp
