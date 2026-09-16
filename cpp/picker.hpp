/*
 * picker.hpp — the coloring picker: the k-coloring search that
 * ncolor.label runs on the cell-adjacency graph and ncolor.color_graph
 * runs on an arbitrary edge list.
 *
 * Given a graph in CSR form (indptr / indices) plus its edge list
 * (src_idx / dst_idx), fills ``colors`` with a proper coloring using as
 * few colors as it can find within the budgets below, escalating the
 * color count when the graph needs more. Strategy, per color count:
 * a race of BFS / greedy attempts (parallel on the ForkJoinPool when the
 * graph is big enough), local repair, TabuCol, an exact branch-and-bound
 * DSATUR on small graphs, and a hybrid evolutionary fallback, gated by
 * a clique lower bound so hopeless color counts are skipped.
 *
 * This header has no Python dependency: together with the kernels it
 * includes it is the whole engine minus the numpy glue, so a C, Rust or
 * command-line front end can reuse it as is. ``binding.cpp`` forwards
 * ``Solver::solve_coloring_`` here.
 */

#ifndef NCOLOR_PICKER_HPP
#define NCOLOR_PICKER_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "bb_dsatur.hpp"
#include "clique_lb.hpp"
#include "color.hpp"
#include "dispatch.hpp"
#include "hea.hpp"
#include "tabucol.hpp"
#include "threadpool.h"

#include "timing.hpp"

namespace ncolor_cpp {

// Renumber the colors actually present to a dense 1..k, preserving their
// relative order, and return k.
//
// The picker's working palette can end up with holes: it searches with a
// fixed number of colors and a coloring may simply not need one of them
// (the weighted objective picks widely separated palette entries and
// leaves the rest unused), and the soft post-pass can vacate a color
// too. Reporting the largest index as the color count would then
// overstate it, and callers that normalize an image by the count would
// be wrong about it as well, so the two are kept in step: the values are
// made dense and the count is the number of them.
inline int densify_colors(std::vector<uint8_t>& colors, int32_t N) {
    if (N <= 0 || colors.empty()) return 0;
    const int32_t n = std::min<int32_t>(N, (int32_t)colors.size());
    uint8_t remap[256] = {0};
    for (int32_t i = 0; i < n; ++i) remap[colors[i]] = 1;
    int k = 0;
    // Index 0 is background and stays 0; 1..255 are colors.
    for (int c = 1; c < 256; ++c) {
        if (remap[c]) remap[c] = (uint8_t)(++k);
    }
    if (k == 0) return 0;
    bool already_dense = true;
    for (int c = 1; c <= k; ++c) {
        if (remap[c] != c) { already_dense = false; break; }
    }
    if (!already_dense) {
        for (int32_t i = 0; i < n; ++i) colors[i] = remap[colors[i]];
    }
    return k;
}

// Per-attempt scratch for the parallel race (one colors vector per
// racing attempt). Keep one per engine and reuse it across calls.
struct PickerScratch {
    std::vector<std::vector<uint8_t>> per_attempt_colors_;
    std::vector<int> per_attempt_ok_;
};

// Coloring loop: race ``attempts_per_n`` BFS-with-random-offset
// attempts in parallel at each cur_n; bb_dsatur takes the last slot
// as an exact-coloring race entry. The weighted-objective path
// (weight_obj != 0) substitutes weighted-WP for the BFS body in
// slots 0..N-2; otherwise WP is unused.
//
// Slot layout at each cur_n:
//   default          slots: [BFS+0, BFS+1, ..., BFS+(N-2), bb_dsatur]
//   weight_obj != 0  slots: [weighted-WP+0, ..., +(N-2), pure WP]
//
// Lowest-index successful attempt wins. Big graphs (N+M ≥ 500) race
// the attempts in parallel on the pool; small graphs run them
// serially.
//
// Side effects: writes the winning coloring into ``colors_`` and the
// adjacency-conflict count into ``last_n_conflicts_``. Returns
// ``n_used`` = max color value in the winning coloring.
//
// Parameters keep the names the code was written against inside
// Solver: ``indptr_`` / ``indices_`` are the CSR of the hard graph,
// ``src_idx_`` / ``dst_idx_`` its M edges, ``colors_`` receives one
// color (1..n_used) per vertex, ``last_n_conflicts_`` the number of
// edges left monochromatic (0 for a proper coloring). ``pool_`` may be
// a single-thread pool; ``n_threads_`` is its participant count.
inline int pick_coloring(int32_t N, int32_t M, int n_colors,
                         int max_depth, int rand_period,
                         int color_mode, int ndim, bool wrap,
                         const double* edge_weights,
                         const double* de_table,
                         int weight_obj,
                         const std::vector<int32_t>& indptr_,
                         const std::vector<int32_t>& indices_,
                         const std::vector<int32_t>& src_idx_,
                         const std::vector<int32_t>& dst_idx_,
                         std::vector<uint8_t>& colors_,
                         int& last_n_conflicts_,
                         PickerScratch& scratch,
                         ForkJoinPool* pool_, int n_threads_) {
    validate_coloring_budget(n_colors, max_depth);
    auto& per_attempt_colors_ = scratch.per_attempt_colors_;
    auto& per_attempt_ok_ = scratch.per_attempt_ok_;
    constexpr int attempts_per_n = 16;
    const int64_t max_iter = std::max<int64_t>(
        static_cast<int64_t>(indices_.size()) +
        static_cast<int64_t>(indptr_.size()), 512);
    // color_mode: -1 = auto (threshold-based), 0 = forced serial,
    // 1 = forced parallel. Auto threshold tuned on M2 / 20-thread
    // ForkJoinPool: below ~500 edges the BFS finishes in <100 µs and
    // dispatch overhead eats the win.
    bool color_parallel;
    if (color_mode == 0) color_parallel = false;
    else if (color_mode == 1) color_parallel = (n_threads_ > 1);
    else color_parallel = (n_threads_ > 1) &&
        (static_cast<int64_t>(N) + M >= 500);
    if (color_parallel) {
        per_attempt_colors_.resize(attempts_per_n);
        per_attempt_ok_.assign(attempts_per_n, 0);
    }

    int cur_n = n_colors;
    bool ok = false;
    static const bool dbg_solve = std::getenv("NCOLOR_SOLVE_DEBUG") != nullptr;
    // ω(G) lower bound: χ(G) ≥ ω(G). If a clique larger than the
    // user's target k exists, the graph requires ≥ ω colors and
    // we'd otherwise burn ~200 ms per (race+tabu-restart+
    // bb_dsatur+HEA) round each time we increment cur_n on the
    // way up to ω. Bron-Kerbosch with a tight deadline (10 ms)
    // returns a valid lower bound even on partial searches —
    // worst case: no time saved when ω ≤ n_colors (the common
    // case). Skipped for N > 20000 (memory) and for the weighted
    // path (perceptual objective is orthogonal to clique
    // structure).
    const bool wobj_active_for_clique = weight_obj != 0 && edge_weights != nullptr;
    // Clique-lower-bound: detect K_{k+1} (or larger) in the graph
    // to skip doomed cur_n=k attempts. For typical cell-adjacency
    // graphs ω = target (no K_5), so CLB returns "no adjustment"
    // — pure overhead. But on dense or higher-connectivity inputs
    // (conn=2, connect_radius=2) K_5 is common and CLB saves the
    // ~200 ms the picker would otherwise burn at cur_n=n_colors.
    //
    // Two regimes:
    //   • N ≤ 1500: 2 ms deadline, classic behavior. CLB rarely
    //     fires but is cheap when it does.
    //   • N > 1500: 5 ms deadline, max_N up to 8000. Bron-Kerbosch
    //     on dense graphs >8k vertices has a memory/time profile
    //     that loses to slot-race failure detection. Below 8k,
    //     the early-out at target+1 finds K_5 in <2 ms on real
    //     cell-adjacency graphs (verified mm 2k² L1 r=2: ~1 ms
    //     to detect K_5, vs the 200 ms the picker otherwise burns
    //     on n=4 attempts before bumping).
    static constexpr int32_t CLB_TIGHT_N = 1500;
    static constexpr int32_t CLB_MAX_N   = 8000;
    if (!wobj_active_for_clique && N >= 5 && N <= CLB_MAX_N) {
        const auto clb_t0 = std::chrono::steady_clock::now();
        const int64_t budget_ns = (N <= CLB_TIGHT_N)
            ? (2LL * 1000LL * 1000LL)
            : (5LL * 1000LL * 1000LL);
        const int64_t clb_deadline_ns =
            steady_time_ns(clb_t0) + budget_ns;
        const int omega = ncolor_cpp::clique_lower_bound(
            N, indptr_.data(), indices_.data(),
            /*target=*/n_colors + 1, clb_deadline_ns);
        if (dbg_solve) {
            const double clb_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - clb_t0).count();
            std::fprintf(stderr,
                "[clique-lb] N=%d ω≥%d (target=%d) %.1fms\n",
                N, omega, n_colors + 1, clb_ms);
        }
        if (omega > cur_n) cur_n = std::min(omega, 255);  // χ ≥ ω, so skip doomed cur_n values
    }
    for (int depth = 0; depth < max_depth && !ok; ++depth) {
        const auto depth_t0 = dbg_solve ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        // A palette supplied for the original target has no entries for
        // escalated colors. Preserve its original block and zero-pad new
        // colors so every attempt uses the correct row stride.
        std::vector<double> expanded_palette;
        const double* attempt_palette = de_table;
        if (de_table && cur_n != n_colors) {
            expanded_palette.assign(static_cast<size_t>(cur_n + 1) * (cur_n + 1), 0.0);
            for (int c = 0; c <= n_colors; ++c)
                std::copy_n(de_table + c * (n_colors + 1), n_colors + 1,
                            expanded_palette.data() + c * (cur_n + 1));
            attempt_palette = expanded_palette.data();
        }
        if (color_parallel) {
            const int local_cur_n = cur_n;
            const int local_depth = depth;
            const int32_t* ip = indptr_.data();
            const int32_t* ix = indices_.data();

            // Lowest slot index that has succeeded so far, or
            // attempts_per_n while none has.
            //
            // The winner is the lowest successful index, so a slot may
            // be abandoned once a *lower-numbered* one has won: its
            // result could not have been used. Abandoning on any
            // success, which is what this did, let thread scheduling
            // decide the answer, because a low slot that would have
            // won got cut off by a high one that happened to finish
            // first. That is what made ncolor.label return different
            // (valid) colorings for the same image from one call to
            // the next, and only above one thread.
            //
            // The early exit itself is kept: every slot above the
            // current best still stops immediately, both by not being
            // started and by having its in-flight tabucol cancelled,
            // so the race still ends as soon as slot 0 is settled
            // rather than running all 16 to their full budget.
            // A slot can succeed two ways: from greedy plus repair
            // alone, which takes about a tenth of a millisecond, or by
            // going on to tabucol, which takes closer to a
            // millisecond. Both are tracked, and a cheap success
            // outranks a dear one however the indices fall.
            //
            // That ranking is what makes the race cheap to settle. The
            // winner is the lowest-numbered success, so every slot
            // below the winner has to be decided before the race can
            // end, and deciding a slot the dear way is ten times the
            // work. Ranking cheap first means that once any slot has
            // come back cheaply, no slot's tabucol can affect the
            // answer any more, so all of them stop at once. One image
            // in the corpus was paying 0.6 ms for exactly this: slot 3
            // colored it cheaply while slots 1 and 2 each spent 0.8 ms
            // failing the dear way, for a result that could not have
            // been used.
            //
            // Still one pass. Sorting the two kinds into separate
            // passes was tried and is worse: greedy and tabucol stop
            // overlapping across slots, so the race pays two barriers
            // instead of one, and the hard graphs that need tabucol
            // everywhere lost 25%.
            constexpr int OK_NONE = 0, OK_CHEAP = 1, OK_TABU = 2;
            std::atomic<int> best_cheap{attempts_per_n};
            std::atomic<int> best_tabu{attempts_per_n};
            std::vector<std::atomic<bool>> slot_cancel(attempts_per_n);
            for (auto& f : slot_cancel) {
                f.store(false, std::memory_order_relaxed);
            }
            auto claim_min = [](std::atomic<int>& best, int idx) {
                int prev = best.load(std::memory_order_relaxed);
                while (idx < prev
                       && !best.compare_exchange_weak(
                              prev, idx, std::memory_order_relaxed)) {
                }
            };
            auto claim_success = [&](int idx, int kind) {
                if (kind == OK_CHEAP) {
                    claim_min(best_cheap, idx);
                    // No tabucol result can win now, at any index, so
                    // every in-flight one may stop. Only tabucol and
                    // the exact search watch this flag; the greedy pass
                    // does not, and a lower slot may still come back
                    // cheaply and take the race.
                    for (auto& f : slot_cancel) {
                        f.store(true, std::memory_order_relaxed);
                    }
                } else {
                    claim_min(best_tabu, idx);
                    for (int j = idx + 1; j < attempts_per_n; ++j) {
                        slot_cancel[j].store(true, std::memory_order_relaxed);
                    }
                }
            };

            // Each worker runs one independently seeded coloring attempt.
            auto run_one_attempt = [&, local_cur_n, local_depth, ip, ix](
                    int idx, const std::atomic<bool>* cancel,
                    int64_t race_deadline_ns) -> int {
                auto& cv = per_attempt_colors_[idx];
                const int attempt_offset = local_depth + idx;
                // Special slot: branch-and-bound exact coloring
                // racing the BFS+tabucol slots. For K_5-free
                // planar-ish cell-adjacency graphs at our scale
                // (N ≤ few thousand), B&B+DSatur typically finds
                // a k-coloring in <1 ms — often faster than
                // BFS+tabucol convergence. When the search tree
                // blows up on adversarial vertex orderings, the
                // node budget + cancel-flag short-circuit kick in
                // and one of the BFS slots wins instead. Only
                // fires at the user's target k (no value running
                // exact search at cur_n > n_colors). Disabled
                // when the user opted into a perceptual weight
                // objective, since bb_dsatur doesn't honour
                // weights. Slot picked: the LAST one (free slot —
                // slots 0..N-2 are BFS, slot N-1 is bb_dsatur).
                // Cancel propagation: bb_dsatur checks the cancel
                // flag every 1024 nodes.
                const bool wobj_active = weight_obj != 0 && edge_weights != nullptr;
                const bool bb_slot = !wobj_active
                                     && (idx == attempts_per_n - 1)
                                     && local_cur_n == n_colors;
                if (bb_slot) {
                    const int64_t node_budget = std::max<int64_t>(
                        20000, (int64_t)N * 30);
                    const bool bb_ok = ncolor_cpp::bb_dsatur(
                        ip, ix, N, local_cur_n, cv, node_budget,
                        cancel, race_deadline_ns);
                    // The exact search is a dear win like tabucol.
                    per_attempt_ok_[idx] = bb_ok ? OK_TABU : OK_NONE;
                    return bb_ok ? OK_TABU : OK_NONE;
                }
                // Slot 0 = user-preferred algorithm; remaining
                // slots = alternate algorithm with different
                // random offsets (algorithm-switching fallback).
                // For the weighted opt-in, slots 0-(attempts-2)
                // use weighted-WP with different offsets; the
                // LAST slot is a pure WP fallback. If all
                // weighted attempts fail at n_colors, the WP
                // attempt can still produce a clean 4-coloring
                // before we bump cur_n. With wobj off (default),
                // all slots run BFS; WP is never selected here.
                const bool wp = wobj_active;
                const bool weighted_attempt = wobj_active &&
                                              (idx < attempts_per_n - 1);
                const double* w_ptr = weighted_attempt ? edge_weights : nullptr;
                const double* de_ptr = weighted_attempt ? attempt_palette : nullptr;
                const int w_obj_local = weighted_attempt ? weight_obj : 0;
                const bool finished = ncolor_cpp::color_graph_csr_legacy(
                    ip, ix, N, local_cur_n, rand_period,
                    attempt_offset, max_iter, cv, wp,
                    w_ptr, de_ptr, w_obj_local);
                const bool conflict = !finished ||
                    ncolor_cpp::has_conflict_csr(ip, ix, N, cv.data());
                bool a_ok = !conflict || ncolor_cpp::repair_coloring(
                    ip, ix, N, local_cur_n, std::max(4, max_depth), cv);
                // Per-attempt TabuCol fallback when greedy+repair
                // fail at the user's target k. Each attempt runs
                // its own small-budget tabucol seeded uniquely,
                // turning the parallel block into a true K-way
                // race of (greedy+repair+tabucol) jobs. Only
                // fires for the user's target k. The `cancel`
                // pointer lets the tabucol loop short-circuit
                // when a sibling worker has already won.
                if (a_ok) {
                    const bool clean_wp_ok =
                        !(wp && !weighted_attempt) || !conflict;
                    per_attempt_ok_[idx] = clean_wp_ok ? OK_CHEAP : OK_NONE;
                    return clean_wp_ok ? OK_CHEAP : OK_NONE;
                }
                // Nothing tabucol produces can win once some slot has
                // come back cheaply, or once a lower slot has already
                // won the dear way, so in either case do not run it.
                const bool tabu_could_matter =
                    best_cheap.load(std::memory_order_relaxed) == attempts_per_n
                    && idx < best_tabu.load(std::memory_order_relaxed);
                if (!a_ok && local_cur_n == n_colors
                        && tabu_could_matter) {
                    for (int32_t u = 0; u < N; ++u) {
                        if (cv[u] < 1 || cv[u] > local_cur_n) {
                            cv[u] = (uint8_t)(1 + (u % local_cur_n));
                        }
                    }
                    const int per_attempt_tabu_iters = std::min(
                        5000, std::max(500, N * 5));
                    const uint64_t tabu_seed =
                        (uint64_t)(attempt_offset + 1)
                            * 0x9e3779b97f4a7c15ULL
                        ^ (uint64_t)(local_depth + 1)
                            * 0x517cc1b727220a95ULL;
                    if (ncolor_cpp::tabucol(
                            ip, ix, N, local_cur_n,
                            per_attempt_tabu_iters, cv, tabu_seed,
                            race_deadline_ns, cancel)) {
                        a_ok = true;
                    }
                }
                // For the weighted path the user has opted in to
                // a perceptual objective and accepts repair. With
                // wobj off there's no clean-WP gate (wp is false).
                const bool clean_wp_required = wp && !weighted_attempt;
                const bool slot_ok = a_ok && (!clean_wp_required || !conflict);
                per_attempt_ok_[idx] = slot_ok ? OK_TABU : OK_NONE;
                return slot_ok ? OK_TABU : OK_NONE;
            };

            // All attempts go through the parallel race. The earlier
            // WP-first warmup path was retired when the ``balance``
            // kwarg was dropped — slot 0 is now a regular BFS+offset
            // slot raced against the rest.
            {
                std::atomic<int> next{0};
                static const bool dbg_slots = std::getenv("NCOLOR_SLOT_DEBUG") != nullptr;
                std::vector<double> slot_ms(dbg_slots ? attempts_per_n : 0, -1.0);
                std::vector<int> slot_done(dbg_slots ? attempts_per_n : 0, 0);
                const auto race_t0 = std::chrono::steady_clock::now();
                // Race wall-clock deadline shared across all slots:
                // bounds time wasted on infeasible-at-cur_n graphs.
                // Without this, the 16 per-attempt tabucols all run
                // their full iter budget (up to several seconds on
                // N≥few-thousand) trying to escape a non-k-colorable
                // graph. 50 ms is plenty for any feasible case at
                // our scale — successes typically converge in <5 ms.
                // For large N (> 1500) on graphs that are
                // genuinely (k+1)-chromatic (Mycielski-like, e.g.
                // mm r=2 after despur), no race slot will find a
                // k-coloring. Cap race wall budget more tightly
                // to bound the wasted time on the failure path.
                // Successes on real cell-adjacency graphs converge
                // in <5ms anyway; the 50ms slack was tuned for
                // adversarial despur-perturbation cases on small N
                // that benefit from longer per-slot tabucol.
                const int64_t race_budget_ns = (N > 1500)
                    ? (15LL * 1000LL * 1000LL)
                    : (50LL * 1000LL * 1000LL);
                const int64_t race_deadline_ns =
                    steady_time_ns(race_t0)
                    + race_budget_ns;
                pool_->parallel([&]() {
                    int idx;
                    while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < attempts_per_n) {
                        // Indices only grow for this worker, so once
                        // one is above the best it took, so is the
                        // rest of its share.
                        // A cheap win below this index settles the
                        // race whatever this slot would have done.
                        if (idx > best_cheap.load(std::memory_order_relaxed)) break;
                        const auto t0 = dbg_slots ? std::chrono::steady_clock::now()
                            : std::chrono::steady_clock::time_point{};
                        const int kind = run_one_attempt(
                            idx, &slot_cancel[idx],
                            race_deadline_ns);
                        if (dbg_slots) {
                            const auto t1 = std::chrono::steady_clock::now();
                            slot_ms[idx] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                            slot_done[idx] = kind != OK_NONE ? 1 : 2;
                        }
                        if (kind != OK_NONE) {
                            claim_success(idx, kind);
                        }
                    }
                });
                if (dbg_solve) {
                    const double race_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - race_t0).count();
                    std::fprintf(stderr, "  [race] %.1fms\n", race_ms);
                }
                if (dbg_slots) {
                    const auto race_t1 = std::chrono::steady_clock::now();
                    const double race_ms = std::chrono::duration<double, std::milli>(race_t1 - race_t0).count();
                    std::fprintf(stderr, "[race] total=%.3fms\n", race_ms);
                    for (int a = 0; a < attempts_per_n; ++a) {
                        const char* st = slot_done[a] == 1 ? "OK"
                                       : slot_done[a] == 2 ? "FAIL"
                                       : "skip";
                        if (slot_ms[a] >= 0) {
                            std::fprintf(stderr, "  slot[%d] %.3fms %s\n", a, slot_ms[a], st);
                        }
                    }
                }
                // The lowest-numbered cheap success wins; failing
                // that, the lowest-numbered dear one.
                int winner = -1;
                for (int a = 0; a < attempts_per_n && winner < 0; ++a) {
                    if (per_attempt_ok_[a] == OK_CHEAP) winner = a;
                }
                for (int a = 0; a < attempts_per_n && winner < 0; ++a) {
                    if (per_attempt_ok_[a] == OK_TABU) winner = a;
                }
                if (winner >= 0) {
                    colors_.swap(per_attempt_colors_[winner]);
                    ok = true;
                }
            }
        } else {
            for (int attempt = 0; attempt < attempts_per_n && !ok; ++attempt) {
                // When weight_obj != 0: slots 0..(attempts-2) run
                // weighted-WP (different offsets); LAST slot is a
                // pure WP fallback so a 4-colorable graph doesn't
                // get bumped to 5 colors when broad-support
                // reducers (count, harmonic) over-constrain the BFS.
                // With wobj off, all slots run plain BFS.
                const bool wobj_active = weight_obj != 0 && edge_weights != nullptr;
                const bool wp = wobj_active;
                const bool weighted_attempt = wobj_active &&
                                              (attempt < attempts_per_n - 1);
                const double* w_ptr = weighted_attempt ? edge_weights : nullptr;
                const double* de_ptr = weighted_attempt ? attempt_palette : nullptr;
                const int w_obj_local = weighted_attempt ? weight_obj : 0;
                const bool finished = ncolor_cpp::color_graph_csr_legacy(
                    indptr_.data(), indices_.data(), N,
                    cur_n, rand_period, depth + attempt, max_iter,
                    colors_, wp, w_ptr, de_ptr, w_obj_local);
                const bool conflict = !finished || ncolor_cpp::has_conflict_csr(
                    indptr_.data(), indices_.data(), N, colors_.data());
                bool a_ok = !conflict || ncolor_cpp::repair_coloring(
                    indptr_.data(), indices_.data(), N,
                    cur_n, std::max(4, max_depth), colors_);
                // For the weighted path the user has opted into a
                // perceptual objective and accepts repair as part of
                // the deal — otherwise the WP-weighted result is
                // silently dropped in favor of a non-WP, non-weighted
                // fallback (which defeats the point). With wobj off
                // wp is false and the gate doesn't fire.
                const bool clean_wp_required = wp && !weighted_attempt;
                if (a_ok && (!clean_wp_required || !conflict)) ok = true;
            }
        }
        if (!ok && cur_n == n_colors) {
            const auto tabu_restart_t0 = std::chrono::steady_clock::now();
            // TabuCol fallback at the user's target k. Some graphs
            // are k-colorable (verified by SAT) but every
            // vertex-ordering greedy hits the same local minimum
            // (e.g. dense corner-touching cells under conn=2 with
            // K4 substructures). Tabu search rescues these by
            // allowing temporary conflict increases to escape.
            // Cost is bounded to one tabucol call per ncolor.label
            // — fast graphs never hit this branch.
            if (color_parallel) {
                // Parallel path doesn't auto-update colors_ on
                // failure; pull in the attempt with fewest
                // conflicts (one of them must be sized N since
                // color_graph_csr_legacy was called on each).
                int best_idx = -1, best_conf = INT_MAX;
                for (int a = 0; a < attempts_per_n; ++a) {
                    if ((int)per_attempt_colors_[a].size() < N) continue;
                    int c = 0;
                    for (int32_t i = 0; i < M; ++i) {
                        if (per_attempt_colors_[a][src_idx_[i]] ==
                            per_attempt_colors_[a][dst_idx_[i]]) ++c;
                    }
                    if (c < best_conf) { best_conf = c; best_idx = a; }
                }
                if (best_idx < 0) goto skip_tabucol;
                colors_.swap(per_attempt_colors_[best_idx]);
            }
            // Sanity-check colors_ before handing to TabuCol; bail
            // out if any vertex has color 0 or > cur_n (would
            // corrupt the conf[] accumulator). Vertices are
            // 0-indexed (matches the rest of the C++ pipeline).
            if ((int)colors_.size() < N) goto skip_tabucol;
            for (int32_t u = 0; u < N; ++u) {
                if (colors_[u] < 1 || colors_[u] > cur_n) goto skip_tabucol;
            }
            {
                // Tabu-search restart loop. First restart uses the
                // conflicted-greedy coloring as a starting point
                // (often within a few moves of valid). Subsequent
                // restarts use a fresh uniform-random coloring so
                // they sample different basins.
                //
                // Total time is capped by a shared wall-clock budget
                // (default 200 ms) so dense graphs that don't
                // 4-color quickly fall through to cur_n bump
                // instead of burning seconds. Each restart still has
                // its own iter cap as a secondary bound.
                const int per_seed_iters = std::min(
                    50000, std::max(2000, N * 30));
                // 50 ms wall budget. Tabucol with restart from the
                // race's best partial coloring rescues feasible-
                // but-hard graphs that the race + bb_dsatur + HEA
                // chain can otherwise miss. Two real cases this
                // catches on MM r=1:
                //   (a) despur_iters=30 (no remove_thin): the
                //       cascade removes 16 1-px bridges, leaving
                //       a graph that's a strict subgraph of the
                //       iter-20 graph (which 4-colors fine) but
                //       that the race's BFS-order heuristic gets
                //       stuck on. χ is unchanged but the search
                //       trajectory shifts.
                //   (b) despur_iters=2 with remove_thin=True: same
                //       failure mode, ~400 scattered single-pixel
                //       removals at corner junctions perturb the
                //       graph just enough to break the race.
                // Without this budget both cases bump cur_n to 5
                // even though χ ≤ 4. 50 ms is enough for tabucol
                // to climb out of the local minimum on both.
                // Skipped when the race already found a valid
                // coloring (the !ok guard above).
                // Tighter budget for large N — see race_budget_ns
                // rationale above. Tabucol can't escape Mycielski-
                // like local minima within ANY reasonable budget,
                // so a shorter cap just bounds the inevitable wait.
                const int64_t budget_ns = (N > 1500)
                    ? (15LL * 1000LL * 1000LL)
                    : (50LL * 1000LL * 1000LL);
                const int64_t deadline_ns =
                    steady_time_ns() + budget_ns;
                std::vector<uint8_t> saved = colors_;
                uint64_t base_seed =
                    (uint64_t)(depth + 1) * 0x9e3779b97f4a7c15ULL;
                for (int s = 0; s < 24 && !ok; ++s) {
                    if (steady_time_ns() > deadline_ns) break;
                    uint64_t rs = base_seed + (uint64_t)s * 0xdeadbeefcafebabeULL;
                    auto next32 = [&]() -> uint32_t {
                        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
                        return (uint32_t)(rs >> 32);
                    };
                    if (s == 0) {
                        colors_ = saved;
                    } else {
                        for (int32_t u = 0; u < N; ++u) {
                            colors_[u] = (uint8_t)(1 + (next32() % (uint32_t)cur_n));
                        }
                    }
                    if (ncolor_cpp::tabucol(
                            indptr_.data(), indices_.data(), N,
                            cur_n, per_seed_iters, colors_, rs,
                            deadline_ns)) {
                        ok = true;
                    }
                }
            }
            skip_tabucol: ;
            if (dbg_solve) {
                const double tabu_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - tabu_restart_t0).count();
                std::fprintf(stderr, "  [tabu-restart] %.1fms ok=%d\n", tabu_ms, (int)ok);
            }
            // bb_dsatur + HEA fallback chain. Wall-clock deadlines
            // bound both: bb_dsatur's per-node cost is O(N) (pick_
            // next is a linear scan), so a fixed node budget
            // translates to multi-second wall time on graphs with
            // N in the thousands. A graph that is genuinely NOT
            // k-colorable would otherwise burn the entire
            // bb_dsatur node budget AND HEA's generation budget
            // proving infeasibility before cur_n bumps to k+1.
            // The 50 ms-each deadlines cap the wasted time per
            // cur_n bump at ~100 ms; the feasible-but-hard cases
            // (e.g. the 2 adversarial shuffles in our stress test
            // where HEA solves in 25-37 ms) still finish well
            // within budget. Fires only when every cheaper path
            // has failed → zero cost on the common path.
            // For graphs that defeat the race + tabu-restart, the
            // bb_dsatur + HEA chain is a desperation play. When a
            // graph is genuinely (k+1)-chromatic (e.g. Mycielski-
            // like structure: no K_{k+1} subgraph but χ = k+1
            // anyway), neither bb_dsatur nor HEA can find a
            // k-coloring — they just exhaust their budgets proving
            // it. For N > 1500 those budgets dominate the failure-
            // path cost (~200ms on mm 2k² r=2). Trim them for
            // large N to bound the wasted time. Small N keeps the
            // generous budget because (a) it amortizes to little
            // wall time anyway, (b) HEA has a real win-rate there
            // on the despur-perturbation cases the original budget
            // was tuned for.
            const int64_t bb_budget_ns = (N > 1500)
                ? (10LL * 1000LL * 1000LL)   // 10 ms for large N
                : (50LL * 1000LL * 1000LL);  // 50 ms otherwise
            const int64_t hea_budget_ns = (N > 1500)
                ? (20LL * 1000LL * 1000LL)   // 20 ms for large N
                : (100LL * 1000LL * 1000LL); // 100 ms otherwise
            if (!ok) {
                const int64_t node_budget = std::max<int64_t>(
                    200000, (int64_t)N * 100);
                std::vector<uint8_t> bb_colors;
                static const bool dbg_bb = std::getenv("NCOLOR_BB_DEBUG") != nullptr;
                const auto bb_t0 = std::chrono::steady_clock::now();
                const int64_t bb_deadline_ns =
                    steady_time_ns(bb_t0) + bb_budget_ns;
                const bool bb_ok = ncolor_cpp::bb_dsatur(
                        indptr_.data(), indices_.data(),
                        N, cur_n, bb_colors, node_budget,
                        /*cancel=*/nullptr, bb_deadline_ns);
                if (dbg_bb) {
                    const auto bb_t1 = std::chrono::steady_clock::now();
                    const double bb_ms = std::chrono::duration<double, std::milli>(
                        bb_t1 - bb_t0).count();
                    std::fprintf(stderr,
                        "[bb_dsatur] N=%d cur_n=%d budget=%lld ok=%d %.2fms\n",
                        N, cur_n, (long long)node_budget, (int)bb_ok, bb_ms);
                }
                if (bb_ok) {
                    colors_ = std::move(bb_colors);
                    ok = true;
                }
            }
            if (!ok) {
                std::vector<uint8_t> hea_colors;
                const uint64_t hea_seed =
                    ((uint64_t)N * 0x9e3779b97f4a7c15ULL)
                    ^ ((uint64_t)(depth + 1) * 0xc6a4a7935bd1e995ULL);
                static const bool dbg_hea = std::getenv("NCOLOR_BB_DEBUG") != nullptr;
                const auto hea_t0 = std::chrono::steady_clock::now();
                const int64_t hea_deadline_ns =
                    steady_time_ns(hea_t0) + hea_budget_ns;
                const bool hea_ok = ncolor_cpp::hea(
                        indptr_.data(), indices_.data(),
                        N, cur_n, hea_colors,
                        /*max_generations=*/80,
                        /*pop_size=*/8,
                        /*init_tabu_iters=*/500,
                        /*gen_tabu_iters=*/2000,
                        hea_seed, hea_deadline_ns);
                if (dbg_hea) {
                    const auto hea_t1 = std::chrono::steady_clock::now();
                    const double hea_ms = std::chrono::duration<double, std::milli>(
                        hea_t1 - hea_t0).count();
                    std::fprintf(stderr,
                        "[hea] N=%d cur_n=%d ok=%d %.2fms\n",
                        N, cur_n, (int)hea_ok, hea_ms);
                }
                if (hea_ok) {
                    colors_ = std::move(hea_colors);
                    ok = true;
                }
            }
        }
        if (dbg_solve) {
            const double depth_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - depth_t0).count();
            std::fprintf(stderr,
                "[solve] depth=%d cur_n=%d ok=%d total=%.1fms\n",
                depth, cur_n, (int)ok, depth_ms);
        }
        if (!ok) {
            if (cur_n == 255) break;
            ++cur_n;
            // ndim-aware floor on the FIRST failure only. Planar
            // (ndim=2) inputs hit ≤ 4 colors by the 4-color theorem,
            // so the floor is a no-op there. For ndim ≥ 3 there's no
            // such bound; empirically dense-blob inputs hit ~3·ndim − 2
            // colors (more with wrap), so jumping directly to that
            // floor skips 2-5 doomed sequential attempts on the
            // fallback path. Triggered only after depth==0 fails, so
            // user-supplied n_colors and planar workloads remain
            // bit-identical to the pre-patch behavior.
            if (depth == 0 && ndim >= 3) {
                const int floor_n = static_cast<int>(std::min<int64_t>(
                    255, 3LL * ndim - 2 + (wrap ? 1 : 0)));
                if (cur_n < floor_n) cur_n = floor_n;
            }
        }
    }

    // Post-success class-merge decoloring: if the picker had to
    // bump cur_n past the user's target k (n_colors), try the
    // CHEAP recovery — find two color classes (a, b) that have no
    // mutual edges and merge them. O(M + cur_n²) per merge.
    //
    // This is a no-op when the (k+1)-coloring has all classes
    // pairwise adjacent (e.g. mm 2k² p=2 clean expand, where the
    // 5-coloring is "tight"); in those cases χ is still k but
    // recovering it from a fresh coloring would require Kempe-chain
    // recoloring or a stronger heuristic. We don't attempt that
    // here — leaving cur_n at the picker's discovered value rather
    // than burning more wall-clock time on a recovery that's not
    // reliable for graphs where tabucol got stuck in the first
    // place. Documented limitation; the picker's race +
    // tabu-restart already runs ~100ms of effort at cur_n=n_colors.
    if (ok && cur_n > n_colors && (int32_t)colors_.size() >= N) {
        const int32_t* ip = indptr_.data();
        const int32_t* ix = indices_.data();
        while (cur_n > n_colors) {
            const int dim = cur_n + 1;
            std::vector<uint8_t> cadj((size_t)dim * dim, 0);
            for (int32_t u = 0; u < N; ++u) {
                const int cu = colors_[u];
                if (cu < 1) continue;
                for (int32_t j = ip[u]; j < ip[u + 1]; ++j) {
                    const int32_t v = ix[j];
                    const int cv = colors_[v];
                    if (cv < 1 || cv == cu) continue;
                    cadj[(size_t)cu * dim + cv] = 1;
                    cadj[(size_t)cv * dim + cu] = 1;
                }
            }
            int merge_a = -1, merge_b = -1;
            for (int a = 1; a <= cur_n && merge_a < 0; ++a) {
                for (int b = a + 1; b <= cur_n; ++b) {
                    if (!cadj[(size_t)a * dim + b]) {
                        merge_a = a; merge_b = b; break;
                    }
                }
            }
            if (merge_a < 0) break;  // No mergeable pair; bail out.
            for (int32_t u = 0; u < N; ++u) {
                if (colors_[u] == merge_b) colors_[u] = (uint8_t)merge_a;
                else if (colors_[u] > merge_b) colors_[u]--;
            }
            --cur_n;
            if (dbg_solve) {
                std::fprintf(stderr,
                    "[decolor merge] class %d -> %d, cur_n=%d\n",
                    merge_b, merge_a, cur_n);
            }
        }
    }

    const int n_used = densify_colors(colors_, N);
    // O(M) tally of adjacent same-color pairs so callers that
    // request return_conflicts don't pay another scan over labels.
    last_n_conflicts_ = 0;
    for (int32_t i = 0; i < M; ++i) {
        if (colors_[src_idx_[i]] == colors_[dst_idx_[i]]) ++last_n_conflicts_;
    }
    return n_used;
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_PICKER_HPP
