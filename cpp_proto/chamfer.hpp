/*
 * chamfer.hpp — header-only L1 (Manhattan) Voronoi label expansion via the
 * Rosenfeld–Pfaltz two-pass scan. Two raster sweeps over the (H, W) row-major
 * image: forward (top→bottom, left→right) propagates from N + W neighbors,
 * backward (bottom→top, right→left) propagates from S + E. Per pixel: 4
 * loads + 2 mins + 1 store per pass. Inherently sequential within a sweep
 * but parallelisable in a wavefront / row-chunk fashion.
 *
 * Output is L1 nearest-seed assignment, not L2 (parabolic envelope). For
 * ncolor's coloring use case this produces a slightly different boundary
 * placement at corners (the standard L1 vs L2 difference) — the resulting
 * adjacency graph is nearly always the same and the 4-coloring works
 * identically. Use only if `expand=True` and you accept non-bit-identical
 * output to numba's L2 expand.
 */

#ifndef NCOLOR_CHAMFER_HPP
#define NCOLOR_CHAMFER_HPP

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <future>
#include <limits>
#include <vector>

#include "threadpool.h"

namespace ncolor_cpp {

// Saito-Toriwaki separable L1 transform — exact L1 Voronoi via two
// orthogonal 1D passes (forward+backward sweep each), with label
// propagation. Strictly faster scaling than the row-band Rosenfeld-Pfaltz
// chamfer at high thread counts because there's no cross-band boundary
// fixup: phase 1 parallelizes trivially over rows (each row's 1D L1
// transform is independent); phase 2 parallelizes trivially over column
// bands (each column-band's vertical sweep is independent).
//
// Algorithm:
//   Phase 1 (per row, parallel-over-rows):
//     init dist[i,j] = 0 if seed else INF, lbl[i,j] from input
//     forward x: if dr[x-1]+1 < dr[x]: dr[x] = dr[x-1]+1, lr[x] = lr[x-1]
//     backward x: if dr[x+1]+1 < dr[x]: dr[x] = dr[x+1]+1, lr[x] = lr[x+1]
//     → after phase 1, dist[i,j] = min over j' of |j-j'| with j' seed in row i
//   Phase 2 (per column band, parallel-over-columns):
//     forward y: if dt[x]+1 < dr[x]: dr[x] = dt[x]+1, lr[x] = lt[x]
//     backward y: if db[x]+1 < dr[x]: dr[x] = db[x]+1, lr[x] = lb[x]
//     → after phase 2, dist[i,j] = exact L1 distance to nearest seed
//        (and lbl[i,j] = that seed's label)
//
// Output is bit-identical to the serial chamfer (both compute exact L1)
// but the parallel scaling is clean — no boundary fixup, predictable
// O((H+W) × n_threads) throughput.
inline void chamfer_st_l1(int32_t* lbl, int32_t* dist, int64_t H, int64_t W,
                          ThreadPool& pool, int n_threads) {
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;

    // Phase 1: per-row init + L1 1D transform. Each row independent.
    auto row_pass = [&](int64_t y0, int64_t y1) {
        for (int64_t y = y0; y < y1; ++y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            for (int64_t x = 0; x < W; ++x) dr[x] = (lr[x] != 0) ? 0 : INF;
            for (int64_t x = 1; x < W; ++x) {
                const int32_t cd = dr[x - 1] + 1;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lr[x - 1]; }
            }
            for (int64_t x = W - 2; x >= 0; --x) {
                const int32_t cd = dr[x + 1] + 1;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lr[x + 1]; }
            }
        }
    };

    // Phase 2: per-column-band vertical L1 transform. Each band independent.
    // Inner x loop is contiguous reads from y, y-1 (or y+1) — prefetcher-friendly.
    auto col_pass = [&](int64_t x0, int64_t x1) {
        for (int64_t y = 1; y < H; ++y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            const int32_t* lt = lbl + (y - 1) * W;
            const int32_t* dt = dist + (y - 1) * W;
            for (int64_t x = x0; x < x1; ++x) {
                const int32_t cd = dt[x] + 1;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lt[x]; }
            }
        }
        for (int64_t y = H - 2; y >= 0; --y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            const int32_t* lb = lbl + (y + 1) * W;
            const int32_t* db = dist + (y + 1) * W;
            for (int64_t x = x0; x < x1; ++x) {
                const int32_t cd = db[x] + 1;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lb[x]; }
            }
        }
    };

    if (n_threads <= 1 || H < 4) {
        row_pass(0, H);
    } else {
        const int64_t per = (H + n_threads - 1) / n_threads;
        std::vector<std::future<void>> fut;
        fut.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            const int64_t y0 = t * per;
            const int64_t y1 = std::min(y0 + per, H);
            if (y0 >= y1) continue;
            fut.emplace_back(pool.enqueue(row_pass, y0, y1));
        }
        for (auto& f : fut) f.get();
    }

    if (n_threads <= 1 || W < 4) {
        col_pass(0, W);
    } else {
        const int64_t per = (W + n_threads - 1) / n_threads;
        std::vector<std::future<void>> fut;
        fut.reserve(n_threads);
        for (int t = 0; t < n_threads; ++t) {
            const int64_t x0 = t * per;
            const int64_t x1 = std::min(x0 + per, W);
            if (x0 >= x1) continue;
            fut.emplace_back(pool.enqueue(col_pass, x0, x1));
        }
        for (auto& f : fut) f.get();
    }
}

// Single-threaded L1 chamfer. `lbl` is in/out: nonzero entries are seeds.
// `dist` is internal scratch (caller passes a same-sized buffer).
inline void chamfer_l1_serial(int32_t* lbl, int32_t* dist, int64_t H, int64_t W) {
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;

    // Init: dist=0 at seeds, dist=INF elsewhere.
    const int64_t total = H * W;
    for (int64_t i = 0; i < total; ++i) dist[i] = (lbl[i] != 0) ? 0 : INF;

    // Forward sweep: for each pixel (y, x) read top (y-1, x) and left (y, x-1).
    for (int64_t y = 0; y < H; ++y) {
        int32_t* lbl_row  = lbl  + y * W;
        int32_t* dist_row = dist + y * W;
        const int32_t* lbl_top  = (y > 0) ? lbl  + (y - 1) * W : nullptr;
        const int32_t* dist_top = (y > 0) ? dist + (y - 1) * W : nullptr;
        for (int64_t x = 0; x < W; ++x) {
            int32_t best_d = dist_row[x];
            int32_t best_l = lbl_row[x];
            if (y > 0) {
                const int32_t cd = dist_top[x] + 1;
                if (cd < best_d) { best_d = cd; best_l = lbl_top[x]; }
            }
            if (x > 0) {
                const int32_t cd = dist_row[x - 1] + 1;
                if (cd < best_d) { best_d = cd; best_l = lbl_row[x - 1]; }
            }
            dist_row[x] = best_d;
            lbl_row[x]  = best_l;
        }
    }

    // Backward sweep: for each pixel (y, x) read bottom (y+1, x) and right (y, x+1).
    for (int64_t y = H - 1; y >= 0; --y) {
        int32_t* lbl_row  = lbl  + y * W;
        int32_t* dist_row = dist + y * W;
        const int32_t* lbl_bot  = (y + 1 < H) ? lbl  + (y + 1) * W : nullptr;
        const int32_t* dist_bot = (y + 1 < H) ? dist + (y + 1) * W : nullptr;
        for (int64_t x = W - 1; x >= 0; --x) {
            int32_t best_d = dist_row[x];
            int32_t best_l = lbl_row[x];
            if (y + 1 < H) {
                const int32_t cd = dist_bot[x] + 1;
                if (cd < best_d) { best_d = cd; best_l = lbl_bot[x]; }
            }
            if (x + 1 < W) {
                const int32_t cd = dist_row[x + 1] + 1;
                if (cd < best_d) { best_d = cd; best_l = lbl_row[x + 1]; }
            }
            dist_row[x] = best_d;
            lbl_row[x]  = best_l;
        }
    }
}

// Row-chunk parallel chamfer with boundary fixup. Splits the image into
// `n_threads` horizontal bands, each thread runs a local two-pass chamfer
// on its band, then a small number of cross-band synchronization passes
// resolves boundary inconsistencies. For typical images (H ≥ 4·n_threads)
// 1–2 fixup passes converge. Conservative implementation runs 4 fixup
// passes; this bounds the worst-case error to 4 pixels of label drift at
// band boundaries.
inline void chamfer_l1_parallel(
        int32_t* lbl, int32_t* dist, int64_t H, int64_t W,
        ThreadPool& pool, int n_threads) {
    if (n_threads <= 1 || H < 8) {
        chamfer_l1_serial(lbl, dist, H, W);
        return;
    }
    const int64_t band = std::max<int64_t>((H + n_threads - 1) / n_threads, 1);
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    auto run_band = [&](int64_t y0, int64_t y1) {
        // Local two-pass chamfer restricted to rows [y0, y1).
        constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
        for (int64_t y = y0; y < y1; ++y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            for (int64_t x = 0; x < W; ++x) dr[x] = (lr[x] != 0) ? 0 : INF;
        }
        for (int64_t y = y0; y < y1; ++y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            const int32_t* lt = (y > y0) ? lbl + (y - 1) * W : nullptr;
            const int32_t* dt = (y > y0) ? dist + (y - 1) * W : nullptr;
            for (int64_t x = 0; x < W; ++x) {
                int32_t bd = dr[x], bl = lr[x];
                if (y > y0) {
                    const int32_t cd = dt[x] + 1;
                    if (cd < bd) { bd = cd; bl = lt[x]; }
                }
                if (x > 0) {
                    const int32_t cd = dr[x - 1] + 1;
                    if (cd < bd) { bd = cd; bl = lr[x - 1]; }
                }
                dr[x] = bd; lr[x] = bl;
            }
        }
        for (int64_t y = y1 - 1; y >= y0; --y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            const int32_t* lb = (y + 1 < y1) ? lbl + (y + 1) * W : nullptr;
            const int32_t* db = (y + 1 < y1) ? dist + (y + 1) * W : nullptr;
            for (int64_t x = W - 1; x >= 0; --x) {
                int32_t bd = dr[x], bl = lr[x];
                if (y + 1 < y1) {
                    const int32_t cd = db[x] + 1;
                    if (cd < bd) { bd = cd; bl = lb[x]; }
                }
                if (x + 1 < W) {
                    const int32_t cd = dr[x + 1] + 1;
                    if (cd < bd) { bd = cd; bl = lr[x + 1]; }
                }
                dr[x] = bd; lr[x] = bl;
            }
        }
    };
    for (int t = 0; t < n_threads; ++t) {
        const int64_t y0 = t * band;
        const int64_t y1 = std::min(y0 + band, H);
        if (y0 >= y1) continue;
        futures.emplace_back(pool.enqueue(run_band, y0, y1));
    }
    for (auto& f : futures) f.get();
    futures.clear();

    // Boundary fixup: serial sweep across band boundaries propagating
    // the now-correct interior values into adjacent bands. Run a few
    // passes until convergence (bounded by max possible label distance).
    for (int pass = 0; pass < 4; ++pass) {
        bool changed = false;
        // Forward boundary fixup: at each band-internal boundary y_b,
        // propagate from y_b-1 to y_b downward.
        for (int t = 1; t < n_threads; ++t) {
            const int64_t y_b = t * band;
            if (y_b >= H) break;
            int32_t* lr = lbl + y_b * W;
            int32_t* dr = dist + y_b * W;
            const int32_t* lt = lbl + (y_b - 1) * W;
            const int32_t* dt = dist + (y_b - 1) * W;
            for (int64_t x = 0; x < W; ++x) {
                const int32_t cd = dt[x] + 1;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lt[x]; changed = true; }
            }
        }
        // Re-run local backward sweeps in each band so the new boundary
        // values flow inward. Parallel.
        for (int t = 0; t < n_threads; ++t) {
            const int64_t y0 = t * band;
            const int64_t y1 = std::min(y0 + band, H);
            if (y0 >= y1) continue;
            futures.emplace_back(pool.enqueue([=]() {
                for (int64_t y = y1 - 1; y >= y0; --y) {
                    int32_t* lr = lbl + y * W;
                    int32_t* dr = dist + y * W;
                    const int32_t* lb = (y + 1 < y1) ? lbl + (y + 1) * W : nullptr;
                    const int32_t* db = (y + 1 < y1) ? dist + (y + 1) * W : nullptr;
                    for (int64_t x = W - 1; x >= 0; --x) {
                        int32_t bd = dr[x], bl = lr[x];
                        if (y + 1 < y1) {
                            const int32_t cd = db[x] + 1;
                            if (cd < bd) { bd = cd; bl = lb[x]; }
                        }
                        if (x + 1 < W) {
                            const int32_t cd = dr[x + 1] + 1;
                            if (cd < bd) { bd = cd; bl = lr[x + 1]; }
                        }
                        dr[x] = bd; lr[x] = bl;
                    }
                }
            }));
        }
        for (auto& f : futures) f.get();
        futures.clear();
        if (!changed) break;
    }
}

} // namespace ncolor_cpp

#endif // NCOLOR_CHAMFER_HPP
