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
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "dispatch.hpp"
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
                          ForkJoinPool& pool, int n_threads) {
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;

    // Phase 1: per-row init + L1 1D transform. Init fused with forward sweep
    // — saves one full pass over the row's data (~16MB at 2048² → ~0.3 ms).
    auto row_pass = [&](int64_t y0, int64_t y1) {
        for (int64_t y = y0; y < y1; ++y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            // Fused init + forward. `prev_d` carries the previous-cell's
            // post-forward distance; for x=0 there's no left neighbour so
            // prev_d starts at INF (saturating with +1 stays >> any seed).
            int32_t prev_d = INF;
            int32_t prev_l = 0;
            for (int64_t x = 0; x < W; ++x) {
                const int32_t init_l = lr[x];
                const int32_t init_d = (init_l != 0) ? 0 : INF;
                const int32_t cd = prev_d + 1;
                int32_t out_d, out_l;
                if (cd < init_d) { out_d = cd; out_l = prev_l; }
                else             { out_d = init_d; out_l = init_l; }
                dr[x] = out_d;
                lr[x] = out_l;
                prev_d = out_d;
                prev_l = out_l;
            }
            // Backward sweep — same as before.
            for (int64_t x = W - 2; x >= 0; --x) {
                const int32_t cd = dr[x + 1] + 1;
                if (cd < dr[x]) { dr[x] = cd; lr[x] = lr[x + 1]; }
            }
        }
    };

    // Phase 2: per-column-band vertical L1 transform. Each band independent.
    // Inner x loop is contiguous reads from y, y-1 (or y+1) — prefetcher-friendly.
    // The branchful form is fastest with gcc/clang (it vectorises with
    // masked stores and skips work on no-update rows). MSVC's auto-vectoriser
    // doesn't kick in for either branchful or branchless int32 conditional
    // updates — switch the Windows toolchain to clang-cl for vectorised
    // builds; the macro is then a single source of truth across compilers.
#define NCOLOR_L1_UPDATE(D, L, NEWD, NEWL)              \
    do { if ((NEWD) < (D)) { (D) = (NEWD); (L) = (NEWL); } } while (0)

    auto col_pass = [&](int64_t x0, int64_t x1) {
        for (int64_t y = 1; y < H; ++y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            const int32_t* lt = lbl + (y - 1) * W;
            const int32_t* dt = dist + (y - 1) * W;
            for (int64_t x = x0; x < x1; ++x) {
                NCOLOR_L1_UPDATE(dr[x], lr[x], dt[x] + 1, lt[x]);
            }
        }
        for (int64_t y = H - 2; y >= 0; --y) {
            int32_t* lr = lbl + y * W;
            int32_t* dr = dist + y * W;
            const int32_t* lb = lbl + (y + 1) * W;
            const int32_t* db = dist + (y + 1) * W;
            for (int64_t x = x0; x < x1; ++x) {
                NCOLOR_L1_UPDATE(dr[x], lr[x], db[x] + 1, lb[x]);
            }
        }
    };

    // Phase 1: parallel over rows. Use atomic work-stealing — much better
    // load balancing than static partitioning when bands have unequal seeds.
    const size_t row_threads = compute_threads(n_threads, H, W);
    dispatch_parallel(pool, static_cast<size_t>(H),
                      row_threads * DISPATCH_CHUNKS_PER_THREAD,
                      [&](size_t y0, size_t y1) { row_pass(static_cast<int64_t>(y0), static_cast<int64_t>(y1)); });

    // Phase 2: parallel over column bands. Each band's vertical sweep is
    // independent; atomic work-stealing balances cache-warm threads across
    // chunks of bands automatically.
    const size_t col_threads = compute_threads(n_threads, W, H);
    dispatch_parallel(pool, static_cast<size_t>(W),
                      col_threads * DISPATCH_CHUNKS_PER_THREAD,
                      [&](size_t x0, size_t x1) { col_pass(static_cast<int64_t>(x0), static_cast<int64_t>(x1)); });
}


// L1 chamfer pass over a contiguous (B, C) slab. Sweep axis is B (rows),
// inner loop is contiguous over the C-band [c0, c1) so the compiler can
// auto-vectorise. The serial RAW dep along the swept axis is unchanged
// (carries through `dprev[c]`), but we now process a full column-band
// per-row, so each row's work is data-parallel across C.
//
//   `init=true`  → row 0 initialises dist from labels, rows 1..B-1 fuse
//                  init with forward propagate. Used for the first axis.
//   `init=false` → rows 1..B-1 propagate forward only (subsequent axes).
//
// Mirrors numba's `_expand_l1_axis_propagate`. For non-innermost axes
// this replaces the strided per-line approach (which had no inner SIMD
// and walked memory with stride = next-axis-product). Cache-friendly
// because consecutive c's land in the same cache lines.
inline void chamfer_l1_slab_pass(int32_t* __restrict lbl,
                                 int32_t* __restrict dist,
                                 int64_t B, int64_t C,
                                 int64_t c0, int64_t c1,
                                 bool init) {
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
    // Update macro is defined above (chamfer_st_l1's col_pass): branchful for
    // gcc/clang, branchless for MSVC. See note there for why.
    if (init) {
        // Row 0: pure init from labels (no prev row exists).
        for (int64_t c = c0; c < c1; ++c) {
            dist[c] = (lbl[c] != 0) ? 0 : INF;
        }
        // Rows 1..B-1: fused init + forward propagate. dr[c] is uninitialised
        // here so it must always be written; only the lr[c] write is
        // conditional on the candidate beating init.
        for (int64_t b = 1; b < B; ++b) {
            const int32_t* lprev = lbl  + (b - 1) * C;
            const int32_t* dprev = dist + (b - 1) * C;
            int32_t* lr = lbl  + b * C;
            int32_t* dr = dist + b * C;
            for (int64_t c = c0; c < c1; ++c) {
                const int32_t init_l = lr[c];
                const int32_t init_d = (init_l != 0) ? 0 : INF;
                const int32_t cd     = dprev[c] + 1;
                if (cd < init_d) { dr[c] = cd;     lr[c] = lprev[c]; }
                else             { dr[c] = init_d; /* lr[c] kept */  }
            }
        }
    } else {
        // Forward propagate, no init (dist already filled by prior axis).
        for (int64_t b = 1; b < B; ++b) {
            const int32_t* lprev = lbl  + (b - 1) * C;
            const int32_t* dprev = dist + (b - 1) * C;
            int32_t* lr = lbl  + b * C;
            int32_t* dr = dist + b * C;
            for (int64_t c = c0; c < c1; ++c) {
                NCOLOR_L1_UPDATE(dr[c], lr[c], dprev[c] + 1, lprev[c]);
            }
        }
    }
    // Backward (always).
    for (int64_t b = B - 2; b >= 0; --b) {
        const int32_t* lnext = lbl  + (b + 1) * C;
        const int32_t* dnext = dist + (b + 1) * C;
        int32_t* lr = lbl  + b * C;
        int32_t* dr = dist + b * C;
        for (int64_t c = c0; c < c1; ++c) {
            NCOLOR_L1_UPDATE(dr[c], lr[c], dnext[c] + 1, lnext[c]);
        }
    }
}

// N-D Saito-Toriwaki separable L1 distance transform with label
// propagation. For each axis ax we view the array as (A, B, C) where
//   A = product of axes BEFORE ax    (outer parallel domain)
//   B = shape[ax]                    (serial sweep axis)
//   C = product of axes AFTER ax     (inner contiguous, auto-vectorised)
//
// This is the same reshape trick used by the dedicated 2D ``chamfer_st_l1``
// kernel and by numba's ``_expand_l1_axis_propagate``. The previous
// implementation processed one strided line at a time, which for axis 0
// of a large 3D volume meant every memory access stepped by the full
// next-axis product (e.g. 256K ints = 1 MB on 512³). That was a cache
// disaster and serial within each line. The (A, B, C) slab pattern keeps
// the inner loop contiguous on every axis, which is roughly 4-5× faster
// for 3D ≥ 256³ on the hosts we benched.
//
// When A is small (axis 0 in 3D has A=1), C is band-split so the
// thread pool still fills out. ``MIN_BAND_W`` keeps each band wide enough
// that the SIMD loop has real work to do.
inline void chamfer_st_l1_nd(int32_t* lbl, int32_t* dist,
                             const std::vector<int64_t>& shape,
                             ForkJoinPool& pool, int n_threads) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim == 0) return;

    // Each band processes ≥ this many contiguous ints per row. 256 ints =
    // 1 KB — a couple of vector registers' worth, plenty to amortise loop
    // overhead while still letting many threads share C when A is small.
    constexpr int64_t MIN_BAND_W = 256;

    for (int ax = ndim - 1; ax >= 0; --ax) {
        const bool first = (ax == ndim - 1);
        int64_t A = 1, C = 1;
        for (int d = 0; d < ax; ++d)        A *= shape[d];
        for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
        const int64_t B = shape[ax];

        // Pick band count so A * n_bands ≈ threads * DISPATCH_CHUNKS_PER_THREAD.
        // Each band must be ≥ MIN_BAND_W ints (when C allows it).
        const int64_t target_chunks =
            static_cast<int64_t>(n_threads) * static_cast<int64_t>(DISPATCH_CHUNKS_PER_THREAD);
        int64_t n_bands = std::max<int64_t>(1, (target_chunks + A - 1) / A);
        int64_t band_w = (C + n_bands - 1) / n_bands;
        if (band_w < MIN_BAND_W && C > MIN_BAND_W) {
            band_w = MIN_BAND_W;
            n_bands = (C + band_w - 1) / band_w;
        }
        n_bands = std::min<int64_t>(n_bands, std::max<int64_t>(1, C));
        band_w  = (C + n_bands - 1) / n_bands;

        const int64_t total_chunks = A * n_bands;
        const size_t threads_for = compute_threads(
            static_cast<size_t>(n_threads),
            static_cast<size_t>(total_chunks),
            static_cast<size_t>(B));

        dispatch_parallel(pool, static_cast<size_t>(total_chunks),
                          threads_for * DISPATCH_CHUNKS_PER_THREAD,
                          [&](size_t i0, size_t i1) {
            for (size_t i = i0; i < i1; ++i) {
                const int64_t a   = static_cast<int64_t>(i) / n_bands;
                const int64_t bnd = static_cast<int64_t>(i) % n_bands;
                const int64_t c0  = bnd * band_w;
                const int64_t c1  = std::min(C, c0 + band_w);
                if (c0 >= c1) continue;
                const int64_t off = a * B * C;
                chamfer_l1_slab_pass(lbl + off, dist + off, B, C, c0, c1, first);
            }
        });
    }
}

} // namespace ncolor_cpp

#endif // NCOLOR_CHAMFER_HPP
