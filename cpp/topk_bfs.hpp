/*
 * topk_bfs.hpp — Alternative top-K kernel using BFS on the 1-NN
 * Voronoi map (l1) instead of propagating K candidates through axis
 * sweeps.
 *
 * Pipeline assumed:
 *   1. Compute l1 (1-NN Voronoi expansion) via standard chamfer.
 *      That's ~0.3 ms on the logo and is reused if already needed
 *      downstream (e.g., to apply the colouring LUT).
 *   2. Call topk_pairs_bfs<K>(l1, ..., pair_count): for each pixel,
 *      walk Manhattan rings outward, recording each distinct
 *      cell-id encountered, with early exit once K cells are found
 *      OR after `empty_rings_exit` consecutive rings with no new
 *      cell. For each pixel, emit one increment per unordered pair
 *      of distinct cells in the K-tuple (margin-filtered by ring
 *      distance vs. the closest ring containing a non-self cell).
 *
 * Why this can beat propagate-K chamfer:
 *   - Interior pixels of large cells exit after a few rings (no new
 *     cell appears), instead of paying K candidates × 4 sweeps.
 *   - Only the "interesting" pixels near multi-cell boundaries pay
 *     the full BFS cost.
 *   - No insertion-sort branchiness; only a tiny K-element dedup
 *     check.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ncolor_cpp {

template <int K>
inline void topk_pairs_bfs(
    const int32_t* __restrict l1,  // [H*W] 1-NN cell IDs (0 = bg)
    int32_t H, int32_t W,
    int32_t N,                      // max cell id
    int32_t r_max,                  // max BFS ring radius
    int32_t max_margin,             // margin filter on ring distance
    int32_t empty_rings_exit,       // exit after this many empty rings
    int64_t* __restrict pair_count) // [(N+1)*(N+1)] zero-initialised
{
    const int32_t stride = N + 1;
    for (int32_t y0 = 0; y0 < H; ++y0) {
        for (int32_t x0 = 0; x0 < W; ++x0) {
            const int32_t c0 = l1[(int64_t)y0 * W + x0];
            if (c0 == 0) continue;
            // Fast-path: skip pixels whose 3x3 neighbourhood is all the
            // same cell. Interior pixels (~80% of typical inputs) take
            // this path with just 8 lookups instead of the 24+ ring
            // scan. Pixels classified "interior" by 3x3 but with other
            // cells further out would only contribute pairs at large
            // ring distance — margin-filtered out anyway.
            {
                bool any_other = false;
                const int32_t y_lo = y0 > 0 ? y0 - 1 : 0;
                const int32_t y_hi = y0 + 1 < H ? y0 + 1 : H - 1;
                const int32_t x_lo = x0 > 0 ? x0 - 1 : 0;
                const int32_t x_hi = x0 + 1 < W ? x0 + 1 : W - 1;
                for (int32_t y = y_lo; y <= y_hi && !any_other; ++y) {
                    for (int32_t x = x_lo; x <= x_hi && !any_other; ++x) {
                        const int32_t l = l1[(int64_t)y * W + x];
                        if (l != 0 && l != c0) any_other = true;
                    }
                }
                if (!any_other) continue;
            }

            // Slot 0: the pixel's own 1-NN cell at distance 0.
            int32_t seen[K];
            int32_t dists[K];
            seen[0] = c0;
            dists[0] = 0;
            int n = 1;
            int first_other_r = -1;
            int empty_streak = 0;

            for (int32_t r = 1; r <= r_max && n < K; ++r) {
                const int n_before = n;
                // Manhattan ring r: pixels with |dy| + |dx| = r.
                for (int32_t dy = -r; dy <= r; ++dy) {
                    const int32_t y = y0 + dy;
                    if (y < 0 || y >= H) continue;
                    const int32_t adx = r - (dy < 0 ? -dy : dy);
                    // Two pixels per dy (±adx), except adx == 0 (single).
                    const int32_t xs[2] = { x0 - adx, x0 + adx };
                    const int n_x = (adx == 0) ? 1 : 2;
                    for (int xi = 0; xi < n_x; ++xi) {
                        const int32_t x = xs[xi];
                        if (x < 0 || x >= W) continue;
                        const int32_t l = l1[(int64_t)y * W + x];
                        if (l == 0 || l == c0) continue;
                        // Margin check: stop accepting once we're past
                        // first-other-cell-distance + max_margin.
                        if (first_other_r >= 0 && r > first_other_r + max_margin)
                            goto done;
                        // Dedup against seen.
                        bool dup = false;
                        for (int k = 1; k < n; ++k) {
                            if (seen[k] == l) { dup = true; break; }
                        }
                        if (dup) continue;
                        seen[n] = l;
                        dists[n] = r;
                        if (first_other_r < 0) first_other_r = r;
                        ++n;
                        if (n == K) goto done;
                    }
                }
                if (n == n_before) {
                    if (++empty_streak >= empty_rings_exit) break;
                } else {
                    empty_streak = 0;
                }
            }
            done:
            // Emit pair counts among the K-tuple (margin-filtered).
            // (We already enforced the margin during expansion above.)
            for (int p = 0; p < n; ++p) {
                for (int q = p + 1; q < n; ++q) {
                    int32_t a = seen[p], b = seen[q];
                    if (a > b) std::swap(a, b);
                    pair_count[(int64_t)a * stride + b] += 1;
                }
            }
        }
    }
}

}  // namespace ncolor_cpp
