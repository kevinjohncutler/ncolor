// HEA: Hybrid Evolutionary Algorithm for k-coloring.
// Galinier & Hao (1999), "Hybrid Evolutionary Algorithms for Graph
// Coloring", J. Combinatorial Optimization 3(4).
//
// A small population of candidate colorings is improved over generations
// via:
//   1. GPX (Greedy Partition Crossover): combines two parents into a
//      child by greedily transferring the largest color classes.
//   2. TabuCol local search: short bursts of tabu-search refinement
//      applied to each new child.
//   3. Replacement: child replaces a randomly-chosen population member.
//
// Solves k-coloring instances where vanilla TabuCol gets stuck in a
// local minimum — population diversity + crossover provides the
// escape mechanism vanilla TabuCol lacks.

#pragma once

#include <vector>
#include <cstdint>
#include <climits>
#include <algorithm>
#include "tabucol.hpp"

namespace ncolor_cpp {

inline int count_conflicts_csr(const int32_t* indptr, const int32_t* indices,
                                int32_t N, const uint8_t* colors) {
    int c = 0;
    for (int32_t u = 0; u < N; ++u) {
        const uint8_t cu = colors[u];
        for (int32_t k = indptr[u]; k < indptr[u + 1]; ++k) {
            const int32_t v = indices[k];
            if (v > u && colors[v] == cu) ++c;
        }
    }
    return c;
}

// Greedy Partition Crossover. For each color c in 1..k, alternately
// pick from parent p1 / p2 the color class with the most vertices,
// assign those vertices to c in the child, and remove them from both
// parents (mark as "assigned"). Any vertices left over at the end are
// assigned colors randomly.
inline void gpx_crossover(
    const std::vector<uint8_t>& p1,
    const std::vector<uint8_t>& p2,
    int32_t N, int32_t k,
    std::vector<uint8_t>& child,
    uint64_t& rng_state)
{
    auto next32 = [&]() -> uint32_t {
        rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
        return (uint32_t)(rng_state >> 32);
    };

    // Build color-class membership for both parents (1-indexed colors).
    // class_p[c] = set of vertices in p with color c.
    std::vector<std::vector<int32_t>> class_p1((size_t)(k + 1));
    std::vector<std::vector<int32_t>> class_p2((size_t)(k + 1));
    for (int32_t v = 0; v < N; ++v) {
        if (p1[v] >= 1 && p1[v] <= k) class_p1[p1[v]].push_back(v);
        if (p2[v] >= 1 && p2[v] <= k) class_p2[p2[v]].push_back(v);
    }

    child.assign((size_t)N, 0);
    std::vector<uint8_t> assigned((size_t)N, 0);

    // For c = 1..k, alternately pick from p1 / p2.
    for (int32_t c = 1; c <= k; ++c) {
        auto& classes = (c % 2 == 1) ? class_p1 : class_p2;
        // Find class with most remaining (unassigned) vertices.
        int best = -1; int best_size = -1;
        for (int32_t ci = 1; ci <= k; ++ci) {
            int sz = 0;
            for (int32_t v : classes[ci]) if (!assigned[v]) ++sz;
            if (sz > best_size) { best_size = sz; best = ci; }
        }
        if (best < 0 || best_size == 0) continue;
        // Assign these vertices to c.
        for (int32_t v : classes[best]) {
            if (!assigned[v]) {
                child[v] = (uint8_t)c;
                assigned[v] = 1;
            }
        }
        // No need to physically clear classes; the `assigned[]` mask
        // in the size computation above filters out already-consumed
        // vertices on subsequent iterations. (Clearing `class_p2[best]`
        // here would mistakenly drop p2's color-`best` vertices —
        // which are NOT the same set as the ones we just consumed
        // from p1's color-`best`.)
    }

    // Remaining vertices: random color.
    for (int32_t v = 0; v < N; ++v) {
        if (!assigned[v]) {
            child[v] = (uint8_t)(1 + (next32() % (uint32_t)k));
        }
    }
}

// Run HEA on a CSR graph. `colors` is initialised in-place (if it
// already contains a valid k-coloring, the function returns
// immediately with that). Returns true if a 0-conflict k-coloring was
// found.
inline bool hea(
    const int32_t* indptr, const int32_t* indices, int32_t N,
    int32_t k, std::vector<uint8_t>& colors,
    int max_generations = 50,
    int pop_size = 5,
    int init_tabu_iters = 200,
    int gen_tabu_iters = 1000,
    uint64_t seed = 0)
{
    if (k <= 0 || N <= 0) return false;

    uint64_t rs = seed ? seed : 0x9e3779b97f4a7c15ULL;
    auto next32 = [&]() -> uint32_t {
        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
        return (uint32_t)(rs >> 32);
    };

    // Quick check on input: if already 0-conflict, done.
    if ((int)colors.size() == N) {
        bool valid = true;
        for (int32_t u = 0; u < N && valid; ++u) {
            if (colors[u] < 1 || colors[u] > k) { valid = false; break; }
        }
        if (valid && count_conflicts_csr(indptr, indices, N, colors.data()) == 0)
            return true;
    }

    // Build initial population: random colorings + brief TabuCol.
    std::vector<std::vector<uint8_t>> pop((size_t)pop_size,
                                          std::vector<uint8_t>((size_t)N, 1));
    std::vector<int> pop_conf((size_t)pop_size, 0);
    int best_idx = 0; int best_conf = INT_MAX;
    for (int p = 0; p < pop_size; ++p) {
        for (int32_t v = 0; v < N; ++v) {
            pop[p][v] = (uint8_t)(1 + (next32() % (uint32_t)k));
        }
        ncolor_cpp::tabucol(indptr, indices, N, k, init_tabu_iters,
                             pop[p], rs + (uint64_t)p * 0xc6a4a7935bd1e995ULL);
        pop_conf[p] = count_conflicts_csr(indptr, indices, N, pop[p].data());
        if (pop_conf[p] < best_conf) {
            best_conf = pop_conf[p];
            best_idx = p;
        }
        if (pop_conf[p] == 0) {
            colors = pop[p];
            return true;
        }
    }

    // Main loop: generate offspring via GPX, refine, replace worst.
    std::vector<uint8_t> child;
    child.reserve((size_t)N);
    for (int gen = 0; gen < max_generations; ++gen) {
        // Select two distinct parents at random.
        int i1 = (int)(next32() % (uint32_t)pop_size);
        int i2 = (int)(next32() % (uint32_t)(pop_size - 1));
        if (i2 >= i1) ++i2;

        // Crossover.
        gpx_crossover(pop[i1], pop[i2], N, k, child, rs);

        // Refine.
        if (ncolor_cpp::tabucol(indptr, indices, N, k, gen_tabu_iters,
                                 child, rs + (uint64_t)gen * 0xbf58476d1ce4e5b9ULL)) {
            colors = child;
            return true;
        }
        int child_conf = count_conflicts_csr(indptr, indices, N, child.data());

        // Replace the WORST population member (or a random one if tied
        // with the child). This keeps diversity better than always
        // replacing the same slot.
        int worst = 0; int worst_conf = pop_conf[0];
        for (int p = 1; p < pop_size; ++p) {
            if (pop_conf[p] > worst_conf) { worst_conf = pop_conf[p]; worst = p; }
        }
        if (child_conf <= worst_conf) {
            pop[worst] = child;
            pop_conf[worst] = child_conf;
            if (child_conf < best_conf) { best_conf = child_conf; best_idx = worst; }
            if (child_conf == 0) {
                colors = pop[worst];
                return true;
            }
        }
    }

    // Out of generations; return the best we found (probably with
    // residual conflicts).
    colors = pop[best_idx];
    return best_conf == 0;
}

}  // namespace ncolor_cpp
