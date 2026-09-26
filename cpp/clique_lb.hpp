// Clique-number lower bound for sparse graphs, with a wall-clock deadline.
//
// χ(G) ≥ ω(G) (chromatic ≥ clique number), so the picker can start at
// max(n_colors, ω) instead of failing its way up one count at a time.
// For 2D conn=2 segmentations ω is usually 4-5; for cells packed like
// tissue in 3D it is about 6.
//
// Every clique is found at its lowest-numbered vertex, among that
// vertex's higher-numbered neighbors, and a cell graph has only a few
// of those per vertex (about 7 for cells packed like tissue in 3D). So
// the search runs on one small neighborhood at a time, as bit masks,
// and never builds anything of size N². The earlier version kept an
// N x N bit matrix (50 MB at 20000 vertices), which limited it to
// graphs of at most 8000 vertices, fewer than the cells in a 256^3
// volume, and those graphs paid for the full failed attempts at every
// count below ω.
//
// Partial searches are still useful: any clique found is a valid lower
// bound. Returns the largest clique found before the deadline or, when
// ``target`` > 0, as soon as one of at least that size is found. Only
// cliques larger than ``floor`` are searched for, since a caller that
// will try ``floor`` colors anyway learns nothing from smaller ones; on
// a graph whose clique number is at most ``floor``, most neighborhoods
// are then too small to need a search at all.

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "intrinsics.hpp"

#include "timing.hpp"

namespace ncolor_cpp {

namespace clique_detail {

inline int popcount_words(const uint64_t* a, int words) {
    int c = 0;
    for (int i = 0; i < words; ++i) c += popcount_u64(a[i]);
    return c;
}

// Branch and bound for the largest clique of one small graph, stored as
// ``d`` rows of ``words`` 64-bit masks. Candidate sets live in a stack
// of rows, one per depth, so the recursion allocates nothing.
struct LocalSearch {
    int words = 1;
    const uint64_t* adj = nullptr;  // d rows x words
    int best = 1;
    int floor = 0;   // cliques of this size or smaller are not sought
    int target = 0;
    int64_t deadline_ns = 0;
    bool deadline_hit = false;
    uint64_t nodes = 0;
    std::vector<uint64_t> cand;     // candidates at each depth
    std::vector<uint64_t> todo;     // branch set at each depth

    bool stop() {
        if (deadline_hit || (target > 0 && best >= target)) return true;
        // Poll the clock on the first node and every 256 after.
        if ((nodes++ & 0xff) == 0 && deadline_ns != 0 &&
            steady_time_ns() > deadline_ns) {
            deadline_hit = true;
            return true;
        }
        return false;
    }

    // Extend a clique of ``r`` vertices by the candidates at ``depth``.
    void expand(int r, int depth) {
        if (stop()) return;
        uint64_t* P = &cand[static_cast<size_t>(depth) * words];
        const int count = popcount_words(P, words);
        if (count == 0) {
            best = std::max(best, r);
            return;
        }
        if (r + count <= std::max(best, floor)) return;
        // Pivot on the candidate with the most candidate neighbors, so
        // only vertices outside its neighborhood need a branch.
        int pivot = -1, pivot_deg = -1;
        for (int w = 0; w < words; ++w) {
            for (uint64_t m = P[w]; m; m &= m - 1) {
                const int u = w * 64 + ctz_u64(m);
                const uint64_t* row = adj + static_cast<size_t>(u) * words;
                int deg = 0;
                for (int k = 0; k < words; ++k) deg += popcount_u64(P[k] & row[k]);
                if (deg > pivot_deg) { pivot_deg = deg; pivot = u; }
            }
        }
        uint64_t* T = &todo[static_cast<size_t>(depth) * words];
        const uint64_t* prow = adj + static_cast<size_t>(pivot) * words;
        for (int k = 0; k < words; ++k) T[k] = P[k] & ~prow[k];
        uint64_t* next = &cand[static_cast<size_t>(depth + 1) * words];
        for (int w = 0; w < words; ++w) {
            while (T[w]) {
                const int b = ctz_u64(T[w]);
                T[w] &= T[w] - 1;
                const int v = w * 64 + b;
                const uint64_t* vrow = adj + static_cast<size_t>(v) * words;
                for (int k = 0; k < words; ++k) next[k] = P[k] & vrow[k];
                expand(r + 1, depth + 1);
                if (deadline_hit || (target > 0 && best >= target)) return;
                P[w] &= ~(uint64_t{1} << b);
                if (r + popcount_words(P, words) <= std::max(best, floor)) return;
            }
        }
    }
};

}  // namespace clique_detail

// Compute a lower bound on the clique number ω(G) of an undirected graph
// in CSR form. Exact unless the deadline passes first. Entries out of
// range, self loops and repeated entries are ignored.
inline int clique_lower_bound(
    int32_t N,
    const int32_t* indptr,
    const int32_t* indices,
    int target = 0,
    int64_t deadline_ns = 0,
    int floor = 0)
{
    if (N < 1) return 0;
    if (N == 1) return 1;
    if (deadline_ns != 0 && steady_time_ns() > deadline_ns) return 1;

    clique_detail::LocalSearch s;
    s.target = target;
    s.floor = floor;
    s.deadline_ns = deadline_ns;
    std::vector<int32_t> slot(N, -1);   // local index of a higher neighbor
    std::vector<int32_t> higher;
    std::vector<uint64_t> adj;
    for (int32_t v = 0; v < N; ++v) {
        if (s.deadline_hit || (target > 0 && s.best >= target)) break;
        // A neighborhood too small to hold a clique larger than the one
        // already known (or the floor) needs no search. Counting first,
        // repeats included, keeps that check free of any bookkeeping.
        const int threshold = std::max(s.best, floor);
        int32_t upper = 0;
        for (int32_t k = indptr[v]; k < indptr[v + 1]; ++k) upper += indices[k] > v;
        if (upper + 1 <= threshold) continue;
        higher.clear();
        for (int32_t k = indptr[v]; k < indptr[v + 1]; ++k) {
            const int32_t u = indices[k];
            if (u <= v || u >= N || slot[u] >= 0) continue;
            slot[u] = static_cast<int32_t>(higher.size());
            higher.push_back(u);
        }
        const int d = static_cast<int>(higher.size());
        if (d + 1 > threshold) {
            const int words = (d + 63) / 64;
            adj.assign(static_cast<size_t>(d) * words, 0);
            for (int a = 0; a < d; ++a) {
                const int32_t u = higher[a];
                uint64_t* row = adj.data() + static_cast<size_t>(a) * words;
                for (int32_t k = indptr[u]; k < indptr[u + 1]; ++k) {
                    const int32_t w = indices[k];
                    if (w < 0 || w >= N || w == u) continue;
                    const int32_t b = slot[w];
                    if (b >= 0) row[b >> 6] |= uint64_t{1} << (b & 63);
                }
            }
            s.words = words;
            s.adj = adj.data();
            s.cand.assign(static_cast<size_t>(d + 2) * words, 0);
            s.todo.assign(static_cast<size_t>(d + 2) * words, 0);
            for (int b = 0; b < d; ++b) s.cand[b >> 6] |= uint64_t{1} << (b & 63);
            s.expand(1, 0);
        }
        for (int32_t u : higher) slot[u] = -1;
    }
    return s.best;
}

}  // namespace ncolor_cpp
