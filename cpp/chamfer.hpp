/*
 * chamfer.hpp — N-D L1 (Manhattan) Voronoi label expansion via the
 * Saito-Toriwaki separable transform. For each axis we do one
 * forward + backward 1D sweep with label propagation; the innermost
 * axis fuses init with the forward sweep into a single register-carry
 * pass over each row, the rest use a slab pass with C-band split for
 * SIMD-friendly auto-vectorisation.
 *
 * Output is L1 nearest-seed assignment (slightly different boundary
 * placement at corner ties vs L2; the 4-colouring graph is the same in
 * practice).
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

// Update macro used by both the 1D row sweep and the (B, C) slab pass:
// branchful on gcc/clang (auto-vectorises with masked stores; skips work
// on no-update lanes), branchless wins on MSVC. Switch the Windows
// toolchain to clang-cl for vectorised builds; the macro is then a
// single source of truth across compilers.
#define NCOLOR_L1_UPDATE(D, L, NEWD, NEWL)              \
    do { if ((NEWD) < (D)) { (D) = (NEWD); (L) = (NEWL); } } while (0)

// Per-row 1D L1 sweep with init: fuses dist init (INF / 0) with the
// forward sweep so we touch each cell once carrying ``prev_d``/``prev_l``
// in registers. Then a backward sweep, then the optional toroidal
// forward+backward pair. Used for the innermost axis of any ND input.
inline void chamfer_l1_row_init(int32_t* __restrict lr, int32_t* __restrict dr,
                                int64_t W, bool wrap) {
    constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
    int32_t prev_d = INF, prev_l = 0;
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
    auto relax_x = [&](int64_t x_dst, int64_t x_src) {
        const int32_t cd = dr[x_src] + 1;
        if (cd < dr[x_dst]) { dr[x_dst] = cd; lr[x_dst] = lr[x_src]; }
    };
    for (int64_t x = W - 2; x >= 0; --x) relax_x(x, x + 1);
    if (wrap && W > 1) {
        relax_x(0, W - 1);                                       // wrap-forward seed
        for (int64_t x = 1;     x < W; ++x)  relax_x(x, x - 1);  // forward sweep
        relax_x(W - 1, 0);                                       // wrap-backward seed
        for (int64_t x = W - 2; x >= 0; --x) relax_x(x, x + 1);  // backward sweep
    }
}

// L1 chamfer pass over a contiguous (B, C) slab. Sweep axis is B (rows),
// inner loop is contiguous over the C-band [c0, c1) so the compiler can
// auto-vectorise. The serial RAW dep along the swept axis is unchanged
// (carries through `dprev[c]`), but we now process a full column-band
// per-row, so each row's work is data-parallel across C.
//
// dist is already filled by the prior axis's pass; we just propagate
// the running min along axis B with both sweeps. Cache-friendly because
// consecutive c's land in the same cache lines.
inline void chamfer_l1_slab_pass(int32_t* __restrict lbl,
                                 int32_t* __restrict dist,
                                 int64_t B, int64_t C,
                                 int64_t c0, int64_t c1,
                                 bool wrap = false) {
    // Per-cell relax: dist[b_dst][c] vs (dist[b_src][c] + 1), keep min.
    auto relax_axis = [&](int64_t b_dst, int64_t b_src) {
        int32_t* lr = lbl  + b_dst * C;
        int32_t* dr = dist + b_dst * C;
        const int32_t* lo = lbl  + b_src * C;
        const int32_t* dn = dist + b_src * C;
        for (int64_t c = c0; c < c1; ++c) {
            NCOLOR_L1_UPDATE(dr[c], lr[c], dn[c] + 1, lo[c]);
        }
    };
    auto forward_sweep  = [&]() { for (int64_t b = 1;     b < B; ++b)   relax_axis(b, b - 1); };
    auto backward_sweep = [&]() { for (int64_t b = B - 2; b >= 0; --b)  relax_axis(b, b + 1); };

    forward_sweep();
    backward_sweep();

    // Toroidal extension: one extra pair of forward+backward sweeps where
    // the leading edge inherits from the wrapped opposite end. ~2× cost.
    if (wrap && B > 1) {
        relax_axis(0,     B - 1);
        forward_sweep();
        relax_axis(B - 1, 0);
        backward_sweep();
    }
}

// N-D Saito-Toriwaki separable L1 distance transform with label
// propagation. For each axis ax we view the array as (A, B, C) where
//   A = product of axes BEFORE ax    (outer parallel domain)
//   B = shape[ax]                    (serial sweep axis)
//   C = product of axes AFTER ax     (inner contiguous, auto-vectorised)
//
// Innermost axis (ax = ndim-1, C = 1): each "row" is a contiguous run of
// length B, parallel-over-rows; per-row uses the fused
// init+forward+backward sweep with prev_d/prev_l carried in registers
// (chamfer_l1_row_init). This is the perf-critical pass on dense inputs
// because every cell is touched.
//
// Other axes (C > 1): slab pass with C-band split so the inner loop is
// contiguous and vectorises. Each band must be ≥ MIN_BAND_W ints; the
// remaining parallelism comes from A. Mirrors numba's
// _expand_l1_axis_propagate.
inline void chamfer_st_l1_nd(int32_t* lbl, int32_t* dist,
                             const std::vector<int64_t>& shape,
                             ForkJoinPool& pool, int n_threads,
                             bool wrap = false) {
    const int ndim = static_cast<int>(shape.size());
    if (ndim == 0) return;

    // Each band processes ≥ this many contiguous ints per row. 256 ints =
    // 1 KB — a couple of vector registers' worth, plenty to amortise loop
    // overhead while still letting many threads share C when A is small.
    constexpr int64_t MIN_BAND_W = 256;

    for (int ax = ndim - 1; ax >= 0; --ax) {
        int64_t A = 1, C = 1;
        for (int d = 0; d < ax; ++d)        A *= shape[d];
        for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
        const int64_t B = shape[ax];

        if (ax == ndim - 1) {
            // Innermost axis (C == 1): per-row 1D sweep, fully fused
            // init + forward + backward (+ optional wrap). The slab path's
            // C-band split would degenerate to a serial scalar loop here.
            const size_t row_threads = compute_threads(
                static_cast<size_t>(n_threads),
                static_cast<size_t>(A),
                static_cast<size_t>(B));
            dispatch_parallel(pool, static_cast<size_t>(A),
                              row_threads * DISPATCH_CHUNKS_PER_THREAD,
                              [&](size_t a0, size_t a1) {
                for (size_t a = a0; a < a1; ++a) {
                    const int64_t off = static_cast<int64_t>(a) * B;
                    chamfer_l1_row_init(lbl + off, dist + off, B, wrap);
                }
            });
            continue;
        }

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
                chamfer_l1_slab_pass(lbl + off, dist + off, B, C, c0, c1,
                                     wrap);
            }
        });
    }
}

} // namespace ncolor_cpp

#endif // NCOLOR_CHAMFER_HPP
