/*
 * expand.hpp — header-only C++ port of ncolor.expand.expand_labels.
 *
 * Felzenszwalb–Huttenlocher (2012) parabolic-envelope distance transform,
 * separable over axes. For an N-dim label image:
 *   for ax in reversed(range(ndim)):
 *     1. transpose so ax is innermost (skip if already last)
 *     2. parabolic envelope pass over the innermost axis
 *     3. transpose back (skip if first iteration)
 *
 * Like connect.hpp, this version uses a persistent ThreadPool — work is
 * distributed across rows (envelope) and tile pairs (transpose), so for a
 * (H, W) image we get H + W + tile-count tasks per call vs the numba
 * version's 5+ @njit(parallel=True) regions × per-region launch cost.
 */

#ifndef NCOLOR_EXPAND_HPP
#define NCOLOR_EXPAND_HPP

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#elif defined(__SSE2__)
#  include <emmintrin.h>
#  if defined(__SSE4_1__)
#    include <smmintrin.h>
#  endif
#endif
#include <vector>

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

using ::ForkJoinPool;

// Strided indexer: lbl[i * stride], dist[i * stride]. With stride=1 this
// is the contiguous row case; with stride=W it operates on a column of an
// (H, W) image without transposing first. Per-iteration cache lines are
// touched once each but for a 2048-tall column the working set fits in L2.
struct StridedView {
    int32_t* base_lbl;
    int32_t* base_dist;
    int64_t stride;
    inline int32_t& lbl(int64_t i) { return base_lbl[i * stride]; }
    inline int32_t& dist(int64_t i) { return base_dist[i * stride]; }
    inline int32_t lbl_r(int64_t i) const { return base_lbl[i * stride]; }
    inline int32_t dist_r(int64_t i) const { return base_dist[i * stride]; }
};

// Parabolic-envelope pass on one line of length N.
//
// `lbl[i]` = 0 means "no seed at i"; nonzero is the label currently at i.
// `dist[i]` is the squared distance to the nearest seed (or unused if lbl=0).
// On exit, every cell is filled with the nearest-seed label and the squared
// distance to that seed (Euclidean over the 1D axis).
//
// Scratch buffers `v`, `lblstk`, `g`, `z` must each have at least N+1 entries
// and are reused across lines by the caller for cache locality. `vd` (double
// version of v) and `vd_sq` (v[k]*v[k] as double) are pre-stored at push
// time so the inner while loop avoids per-iteration int→double conversion
// and the `ft*ft` multiply — both are pure win on -O3 because LLVM can't
// reliably hoist the cast/multiply out of the data-dependent while body.
//
// `stride` lets us skip transposes: for a row-major (H, W) image, calling
// this on a column with `lbl=base+col, dist=...+col, N=H, stride=W` walks
// the column in place. Cost: each access loads a separate cache line (we
// pay full L2 miss latency for each i), but for typical column lengths
// (≤4K) the per-column working set fits in L2 and end-to-end this beats
// the transpose+contiguous variant by a ~2× margin (transpose is dominated
// by the strided write half anyway).
// SIMD fill helper: writes lbl[i_start..i_end) = lbl_j and
// dist[i_start..i_end) = g_j + (i - v_j)². Vectorized for ARM64 NEON
// (Apple Silicon — 4×int32 per iteration) and x86_64 SSE2/AVX2 (4×int32
// per iteration via _mm_mullo_epi32). Scalar tail handles the remainder.
//
// Hand-rolled because clang -O3 -march=native consistently fails to
// vectorize the int32 ``di*di`` multiply + paired stores even with
// __restrict qualifiers.
static inline void envelope_fill_simd(
        int32_t* __restrict lbl, int32_t* __restrict dist,
        int64_t i_start, int64_t i_end,
        int32_t lbl_j, int32_t g_j, int32_t v_j) {
    int64_t i = i_start;
#if defined(__aarch64__) || defined(__ARM_NEON)
    const int32x4_t v_lbl = vdupq_n_s32(lbl_j);
    const int32x4_t v_g   = vdupq_n_s32(g_j);
    const int32x4_t v_vj  = vdupq_n_s32(v_j);
    const int32x4_t v_inc = {0, 1, 2, 3};
    const int32x4_t v_four = vdupq_n_s32(4);
    int32x4_t v_i = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i_start)), v_inc);
    for (; i + 4 <= i_end; i += 4) {
        int32x4_t v_di = vsubq_s32(v_i, v_vj);
        int32x4_t v_di_sq = vmulq_s32(v_di, v_di);
        int32x4_t v_dist = vaddq_s32(v_di_sq, v_g);
        vst1q_s32(lbl + i, v_lbl);
        vst1q_s32(dist + i, v_dist);
        v_i = vaddq_s32(v_i, v_four);
    }
#elif defined(__SSE2__)
    const __m128i v_lbl = _mm_set1_epi32(lbl_j);
    const __m128i v_g   = _mm_set1_epi32(g_j);
    const __m128i v_vj  = _mm_set1_epi32(v_j);
    const __m128i v_four = _mm_set1_epi32(4);
    __m128i v_i = _mm_add_epi32(_mm_set1_epi32(static_cast<int32_t>(i_start)),
                                _mm_set_epi32(3, 2, 1, 0));
    for (; i + 4 <= i_end; i += 4) {
        __m128i v_di = _mm_sub_epi32(v_i, v_vj);
        // _mm_mullo_epi32 needs SSE4.1; fall back to scalar tail otherwise.
    #if defined(__SSE4_1__)
        __m128i v_di_sq = _mm_mullo_epi32(v_di, v_di);
    #else
        // SSE2-only: spill to scalar
        break;
    #endif
        __m128i v_dist = _mm_add_epi32(v_di_sq, v_g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(lbl + i), v_lbl);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dist + i), v_dist);
        v_i = _mm_add_epi32(v_i, v_four);
    }
#endif
    for (; i < i_end; ++i) {
        const int32_t di = static_cast<int32_t>(i) - v_j;
        lbl[i] = lbl_j;
        dist[i] = g_j + di * di;
    }
}

// Contiguous (stride=1) variant — separate symbol so the compiler can
// vectorize the phase-2 fill loop without proving stride=1 every call.
// ``__restrict`` lets it assume lbl and dist don't alias.
inline void envelope_pass_row_contig(
        int32_t* __restrict lbl, int32_t* __restrict dist, int64_t N,
        int32_t* __restrict v, int32_t* __restrict lblstk,
        int32_t* __restrict g, double* __restrict z,
        double* __restrict vd, double* __restrict vd_sq) {
    int32_t k = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (lbl[i] == 0) continue;
        const int32_t gi = dist[i];
        const double fi = static_cast<double>(i);
        const double gf = static_cast<double>(gi);
        const double fi_sq_plus_gf = fi * fi + gf;
        double new_z = -1e18;
        while (k > 0) {
            const int32_t top = k - 1;
            const double ft = vd[top];
            const double ft_sq = vd_sq[top];
            const double g_top = static_cast<double>(g[top]);
            const double numer = fi_sq_plus_gf - g_top - ft_sq;
            const double denom = 2.0 * (fi - ft);
            if (numer > z[top] * denom) {
                new_z = numer / denom;
                break;
            }
            k -= 1;
        }
        z[k] = new_z;
        v[k] = static_cast<int32_t>(i);
        vd[k] = fi;
        vd_sq[k] = fi * fi;
        lblstk[k] = lbl[i];
        g[k] = gi;
        k += 1;
    }
    if (k == 0) return;
    int64_t i_start = 0;
    for (int32_t j = 0; j < k; ++j) {
        int64_t i_end;
        if (j + 1 == k) {
            i_end = N;
        } else {
            const double zj1 = z[j + 1];
            if (zj1 <= static_cast<double>(i_start)) continue;
            i_end = (zj1 >= static_cast<double>(N)) ? N : static_cast<int64_t>(std::ceil(zj1));
            if (i_end > N) i_end = N;
        }
        if (i_end <= i_start) continue;
        const int32_t lbl_j = lblstk[j];
        const int32_t g_j = g[j];
        const int32_t v_j = v[j];
        envelope_fill_simd(lbl, dist, i_start, i_end, lbl_j, g_j, v_j);
        i_start = i_end;
    }
}

inline void envelope_pass_row(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    if (stride == 1) {
        envelope_pass_row_contig(lbl, dist, N, v, lblstk, g, z, vd, vd_sq);
        return;
    }
    int32_t k = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (lbl[i * stride] == 0) continue;
        const int32_t gi = dist[i * stride];
        const double fi = static_cast<double>(i);
        const double gf = static_cast<double>(gi);
        const double fi_sq_plus_gf = fi * fi + gf;
        double new_z = -1e18;
        while (k > 0) {
            const int32_t top = k - 1;
            const double ft = vd[top];
            const double ft_sq = vd_sq[top];
            const double g_top = static_cast<double>(g[top]);
            const double numer = fi_sq_plus_gf - g_top - ft_sq;
            const double denom = 2.0 * (fi - ft);
            if (numer > z[top] * denom) {
                new_z = numer / denom;
                break;
            }
            k -= 1;
        }
        z[k] = new_z;
        v[k] = static_cast<int32_t>(i);
        vd[k] = fi;
        vd_sq[k] = fi * fi;
        lblstk[k] = lbl[i * stride];
        g[k] = gi;
        k += 1;
    }
    if (k == 0) return;
    int64_t i_start = 0;
    for (int32_t j = 0; j < k; ++j) {
        int64_t i_end;
        if (j + 1 == k) {
            i_end = N;
        } else {
            const double zj1 = z[j + 1];
            if (zj1 <= static_cast<double>(i_start)) {
                continue;
            }
            i_end = (zj1 >= static_cast<double>(N)) ? N : static_cast<int64_t>(std::ceil(zj1));
            if (i_end > N) i_end = N;
        }
        if (i_end <= i_start) {
            continue;
        }
        const int32_t lbl_j = lblstk[j];
        const int32_t g_j = g[j];
        const int32_t v_j = v[j];
        for (int64_t i = i_start; i < i_end; ++i) {
            const int32_t di = static_cast<int32_t>(i) - v_j;
            lbl[i * stride] = lbl_j;
            dist[i * stride] = g_j + di * di;
        }
        i_start = i_end;
    }
}

// =============================================================================
// Per-worker scratch for the unified Lp envelope pass and the legacy L2
// kernels. Held by ExpandBuffers so allocations persist across calls.
struct EnvelopeScratch {
    std::vector<int32_t> v, lblstk, g;
    std::vector<double> z, vd, vd_sq;
    void resize(size_t cap) {
        if (v.size() < cap) {
            v.resize(cap);
            lblstk.resize(cap);
            g.resize(cap);
            z.resize(cap);
            vd.resize(cap);
            vd_sq.resize(cap);
        }
    }
};


// =============================================================================

// Pass-0 fast path (sparse first-axis input, all seeds have dist=0).
//
// On the first axis, every nonzero pixel is a "seed" with dist=0. The
// parabolic envelope of zero-height parabolas at positions s_k reduces to
// the simple midpoint Voronoi: pixel i belongs to seed s_k where
// ``(s_k + s_{k+1})/2 <= i < (s_{k+1} + s_{k+2})/2``. No FP division, no
// stack management — just collect seeds, then walk midpoints. This is
// the optimization edt uses to skip the full envelope build on the first
// axis. For sparse inputs (typical ncolor case where most pixels are bg),
// the seed list per row is tiny and the per-row work is O(N + n_seeds).
//
// In-place: reads `lbl[i]` (nonzero = seed), writes `lbl[i]` (nearest
// seed's label) and `dist[i]` (squared distance). Initial dist values
// are ignored (they get overwritten).
inline void envelope_pass0_row(
        int32_t* lbl, int32_t* dist, int64_t N,
        int32_t* seeds, int32_t* lbl_save) {
    // Collect seed positions; copy original labels (we overwrite lbl as we go).
    int32_t n_seeds = 0;
    for (int64_t i = 0; i < N; ++i) {
        if (lbl[i] != 0) {
            seeds[n_seeds] = static_cast<int32_t>(i);
            lbl_save[n_seeds] = lbl[i];
            ++n_seeds;
        }
    }
    if (n_seeds == 0) {
        // All-zero row: no seeds means the algorithm propagates "no label"
        // to phase-2 of the next axis. Fill with 0 / large dist.
        std::memset(lbl, 0, N * sizeof(int32_t));
        for (int64_t i = 0; i < N; ++i) dist[i] = INT32_MAX / 4;
        return;
    }
    int32_t k = 0;
    for (int64_t i = 0; i < N; ++i) {
        // Advance k while position i is at-or-past the midpoint to seed k+1.
        // Numba's parabolic envelope uses ``i >= z[k+1]`` (non-strict), so at
        // the exact midpoint between two seeds the *later* seed wins. We match
        // bit-for-bit with the integer form ``2*i >= s_k + s_{k+1}``.
        while (k + 1 < n_seeds && 2 * static_cast<int32_t>(i) >= seeds[k] + seeds[k + 1]) {
            ++k;
        }
        const int32_t di = static_cast<int32_t>(i) - seeds[k];
        dist[i] = di * di;
        lbl[i] = lbl_save[k];
    }
}

// Pass-0 over a (n_slices, N) buffer in parallel. ``input_was_sparse``
// must be true: every nonzero entry of `lbl` is a seed (dist=0). Used for
// the first axis of expand_labels where the input is the original label
// image (mostly background).
inline void envelope_pass0(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_slices, int64_t N,
        ForkJoinPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch) {
    if (n_threads < 1) n_threads = 1;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(n_slices),
        static_cast<size_t>(N)));
    if (static_cast<int>(scratch.size()) < eff_threads) scratch.resize(eff_threads);
    const size_t cap = static_cast<size_t>(N) + 1;
    for (int t = 0; t < eff_threads; ++t) scratch[t].resize(cap);

    if (eff_threads == 1 || n_slices < 2) {
        for (int64_t s = 0; s < n_slices; ++s) {
            envelope_pass0_row(h_lbl + s * N, h_dist + s * N, N,
                               scratch[0].v.data(), scratch[0].lblstk.data());
        }
        return;
    }

    // Atomic work-stealing: split slices into chunks, each thread claims a
    // tid once (so it can reuse its scratch[tid]) then loops claiming chunks.
    const int n_chunks = static_cast<int>(std::min<int64_t>(
        n_slices, static_cast<int64_t>(eff_threads) * DISPATCH_CHUNKS_PER_THREAD));
    const int64_t chunk_sz = (n_slices + n_chunks - 1) / n_chunks;
    std::atomic<int> tid_next{0};
    std::atomic<int> chunk_next{0};
    pool.parallel([&]() {
        const int my_tid = tid_next.fetch_add(1, std::memory_order_relaxed);
        if (my_tid >= eff_threads) return;
        int32_t* sp = scratch[my_tid].v.data();
        int32_t* lp = scratch[my_tid].lblstk.data();
        int idx;
        while ((idx = chunk_next.fetch_add(1, std::memory_order_relaxed)) < n_chunks) {
            const int64_t s0 = static_cast<int64_t>(idx) * chunk_sz;
            const int64_t s1 = std::min(s0 + chunk_sz, n_slices);
            for (int64_t s = s0; s < s1; ++s) {
                envelope_pass0_row(h_lbl + s * N, h_dist + s * N, N, sp, lp);
            }
        }
    });
}

// Pass over (n_slices, N) row-major arrays in parallel.
inline void envelope_pass(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_slices, int64_t N,
        ForkJoinPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch) {
    if (n_threads < 1) n_threads = 1;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(n_slices),
        static_cast<size_t>(N)));
    if (static_cast<int>(scratch.size()) < eff_threads) scratch.resize(eff_threads);
    const size_t cap = static_cast<size_t>(N) + 1;
    for (int t = 0; t < eff_threads; ++t) scratch[t].resize(cap);

    if (eff_threads == 1 || n_slices < 2) {
        auto& sc = scratch[0];
        for (int64_t s = 0; s < n_slices; ++s) {
            envelope_pass_row(h_lbl + s * N, h_dist + s * N, N, /*stride=*/1,
                              sc.v.data(), sc.lblstk.data(), sc.g.data(),
                              sc.z.data(), sc.vd.data(), sc.vd_sq.data());
        }
        return;
    }

    const int n_chunks = static_cast<int>(std::min<int64_t>(
        n_slices, static_cast<int64_t>(eff_threads) * DISPATCH_CHUNKS_PER_THREAD));
    const int64_t chunk_sz = (n_slices + n_chunks - 1) / n_chunks;
    std::atomic<int> tid_next{0};
    std::atomic<int> chunk_next{0};
    pool.parallel([&]() {
        const int my_tid = tid_next.fetch_add(1, std::memory_order_relaxed);
        if (my_tid >= eff_threads) return;
        auto& sc = scratch[my_tid];
        int32_t* vp = sc.v.data();
        int32_t* lp = sc.lblstk.data();
        int32_t* gp = sc.g.data();
        double* zp = sc.z.data();
        double* vdp = sc.vd.data();
        double* vdsqp = sc.vd_sq.data();
        int idx;
        while ((idx = chunk_next.fetch_add(1, std::memory_order_relaxed)) < n_chunks) {
            const int64_t s0 = static_cast<int64_t>(idx) * chunk_sz;
            const int64_t s1 = std::min(s0 + chunk_sz, n_slices);
            for (int64_t s = s0; s < s1; ++s) {
                envelope_pass_row(h_lbl + s * N, h_dist + s * N, N, /*stride=*/1,
                                  vp, lp, gp, zp, vdp, vdsqp);
            }
        }
    });
}

// Strided variant: process one logical "line" per call where pixels are
// stride-spaced in memory. Use case: a column of an (H, W) row-major image
// with stride=W, base=col_index. Avoids the transpose round-trip entirely.
inline void envelope_pass_strided(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_lines, int64_t N, int64_t stride,
        ForkJoinPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch) {
    if (n_threads < 1) n_threads = 1;
    if (static_cast<int>(scratch.size()) < n_threads) scratch.resize(n_threads);
    const size_t cap = static_cast<size_t>(N) + 1;
    for (int t = 0; t < n_threads; ++t) scratch[t].resize(cap);

    if (n_threads == 1 || n_lines < 2) {
        auto& sc = scratch[0];
        for (int64_t s = 0; s < n_lines; ++s) {
            envelope_pass_row(h_lbl + s, h_dist + s, N, stride,
                              sc.v.data(), sc.lblstk.data(), sc.g.data(),
                              sc.z.data(), sc.vd.data(), sc.vd_sq.data());
        }
        return;
    }

    const int n_chunks = static_cast<int>(std::min<int64_t>(
        n_lines, static_cast<int64_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD));
    const int64_t chunk_sz = (n_lines + n_chunks - 1) / n_chunks;
    std::atomic<int> tid_next{0};
    std::atomic<int> chunk_next{0};
    pool.parallel([&]() {
        const int my_tid = tid_next.fetch_add(1, std::memory_order_relaxed);
        if (my_tid >= n_threads) return;
        auto& sc = scratch[my_tid];
        int32_t* vp = sc.v.data();
        int32_t* lp = sc.lblstk.data();
        int32_t* gp = sc.g.data();
        double* zp = sc.z.data();
        double* vdp = sc.vd.data();
        double* vdsqp = sc.vd_sq.data();
        int idx;
        while ((idx = chunk_next.fetch_add(1, std::memory_order_relaxed)) < n_chunks) {
            const int64_t s0 = static_cast<int64_t>(idx) * chunk_sz;
            const int64_t s1 = std::min(s0 + chunk_sz, n_lines);
            for (int64_t s = s0; s < s1; ++s) {
                envelope_pass_row(h_lbl + s, h_dist + s, N, stride,
                                  vp, lp, gp, zp, vdp, vdsqp);
            }
        }
    });
}

// Blocked batched transpose: src(A,B,C) → dst(A,C,B), for two arrays
// in lockstep (label + dist). Tile size 64 matches edt::TRANSPOSE_BLOCK
// (also matches the numba version's Bi=64). Uses atomic work-stealing
// dispatch over tile triples (a, rb, cb) — load balances naturally even
// when total_tiles is not a clean multiple of n_threads.
template <typename T>
void batch_transpose(
        const T* src_a, const T* src_b,
        T* dst_a, T* dst_b,
        int64_t A, int64_t B, int64_t C,
        ForkJoinPool& pool, int n_threads) {
    constexpr int Bi = 64;
    const int64_t n_b = (B + Bi - 1) / Bi;
    const int64_t n_c = (C + Bi - 1) / Bi;
    const int64_t bpp = n_b * n_c;
    const size_t total_tiles = static_cast<size_t>(A * bpp);

    if (n_threads < 1) n_threads = 1;
    auto tile_work = [=](size_t begin, size_t end) {
        for (size_t i = begin; i < end; ++i) {
            const int64_t a   = static_cast<int64_t>(i) / bpp;
            const int64_t blk = static_cast<int64_t>(i) % bpp;
            const int64_t b0  = (blk / n_c) * Bi;
            const int64_t c0  = (blk % n_c) * Bi;
            const int64_t b1  = std::min<int64_t>(b0 + Bi, B);
            const int64_t c1  = std::min<int64_t>(c0 + Bi, C);
            const int64_t plane  = a * B * C;
            const int64_t tplane = a * C * B;
            for (int64_t b = b0; b < b1; ++b) {
                for (int64_t c = c0; c < c1; ++c) {
                    dst_a[tplane + c * B + b] = src_a[plane + b * C + c];
                    dst_b[tplane + c * B + b] = src_b[plane + b * C + c];
                }
            }
        }
    };

    if (n_threads == 1 || total_tiles < 4) {
        tile_work(0, total_tiles);
        return;
    }

    dispatch_parallel(pool, total_tiles,
                      static_cast<size_t>(n_threads) * DISPATCH_CHUNKS_PER_THREAD,
                      tile_work);
}

// =============================================================================

// Holds buffers for repeated calls. Keep one per Python ExpandEngine instance.
class ExpandBuffers {
public:
    void resize(int64_t total) {
        if (total > capacity_) {
            h_lbl_.resize(total);
            h_dist_.resize(total);
            t_lbl_.resize(total);
            t_dist_.resize(total);
            capacity_ = total;
        }
        size_ = total;
    }
    int32_t* lbl()   { return h_lbl_.data(); }
    int32_t* dist()  { return h_dist_.data(); }
    int32_t* lbl_T()  { return t_lbl_.data(); }
    int32_t* dist_T() { return t_dist_.data(); }
    int64_t size() const { return size_; }
    // Per-worker envelope scratch (resized lazily).
    std::vector<EnvelopeScratch>& scratch() { return scratch_; }
private:
    std::vector<int32_t> h_lbl_, h_dist_, t_lbl_, t_dist_;
    int64_t capacity_ = 0;
    int64_t size_ = 0;
    std::vector<EnvelopeScratch> scratch_;
};


// =============================================================================

// Run expand_labels on a row-major label image of arbitrary ndim.
// `shape` is the image shape; total = product of shape entries; the output
// is written into `bufs.lbl()` which is also the working scratch.
inline void expand_labels_inplace(
        const int32_t* input, ExpandBuffers& bufs,
        const std::vector<int64_t>& shape,
        ForkJoinPool& pool, int n_threads) {
    const int ndim = static_cast<int>(shape.size());
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    bufs.resize(total);
    int32_t* h_lbl = bufs.lbl();
    int32_t* h_dist = bufs.dist();
    int32_t* t_lbl = bufs.lbl_T();
    int32_t* t_dist = bufs.dist_T();

    if (input != h_lbl) {
        std::memcpy(h_lbl, input, total * sizeof(int32_t));
    }
    // dist init unnecessary — pass0 overwrites every entry.

    for (int ax = ndim - 1; ax >= 0; --ax) {
        const int64_t n = shape[ax];
        if (ax == ndim - 1) {
            // First axis: midpoint fast path (edt's _expand_pass0).
            const int64_t n_slices = total / n;
            envelope_pass0(h_lbl, h_dist, n_slices, n, pool, n_threads, bufs.scratch());
        } else {
            // Subsequent axes: tested both stride-aware and transpose-then-
            // contiguous. For 2D images with axis-0 length ≤ ~512 the strided
            // variant wins (column fits in L1); for ≥ 1024 the cache-line
            // utilization at column stride busts L2 (1/16 of every line is
            // useful) and the transpose roundtrip wins by a 2× margin.
            // Transpose is the universal default; strided is available via
            // ``envelope_pass_strided`` for callers who know their geometry.
            int64_t A = 1;
            for (int d = 0; d < ax; ++d) A *= shape[d];
            int64_t C = 1;
            for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
            const int64_t B = n;
            batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist, A, B, C, pool, n_threads);
            envelope_pass(t_lbl, t_dist, A * C, B, pool, n_threads, bufs.scratch());
            batch_transpose<int32_t>(t_lbl, t_dist, h_lbl, h_dist, A, C, B, pool, n_threads);
        }
    }
}

// Profile breakdown helper — times each stage. Same algorithm as
// expand_labels_inplace, but emits per-stage durations via callback.
template <typename Cb>
inline void expand_labels_inplace_timed(
        const int32_t* input, ExpandBuffers& bufs,
        const std::vector<int64_t>& shape,
        ForkJoinPool& pool, int n_threads,
        Cb&& report_ms) {
    const int ndim = static_cast<int>(shape.size());
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    bufs.resize(total);
    int32_t* h_lbl = bufs.lbl();
    int32_t* h_dist = bufs.dist();
    int32_t* t_lbl = bufs.lbl_T();
    int32_t* t_dist = bufs.dist_T();

    auto t0 = std::chrono::steady_clock::now();
    auto stage = [&](const char* name) {
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        report_ms(name, ms);
        t0 = t1;
    };

    if (input != h_lbl) {
        std::memcpy(h_lbl, input, total * sizeof(int32_t));
    }
    stage("memcpy_in");

    for (int ax = ndim - 1; ax >= 0; --ax) {
        const int64_t n = shape[ax];
        if (ax == ndim - 1) {
            const int64_t n_slices = total / n;
            envelope_pass0(h_lbl, h_dist, n_slices, n, pool, n_threads, bufs.scratch());
            stage("pass0_axis1");
        } else {
            int64_t A = 1;
            for (int d = 0; d < ax; ++d) A *= shape[d];
            int64_t C = 1;
            for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
            const int64_t B = n;
            batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist, A, B, C, pool, n_threads);
            stage("transpose_fwd");
            envelope_pass(t_lbl, t_dist, A * C, B, pool, n_threads, bufs.scratch());
            stage("parabolic_axis0");
            batch_transpose<int32_t>(t_lbl, t_dist, h_lbl, h_dist, A, C, B, pool, n_threads);
            stage("transpose_back");
        }
    }
}

} // namespace ncolor_cpp

#endif // NCOLOR_EXPAND_HPP
