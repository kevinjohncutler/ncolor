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

// Contiguous (stride=1) variant. Templated on ``Wrap`` so the compiler
// emits two specialised codepaths and the wrap branches cost nothing
// at runtime when Wrap=false.
//
// Wrap=true: the envelope considers ghost seeds at v - N and v + N
// (toroidal Voronoi). Phase 1 iterates i ∈ [-N, 2N), reading lbl/dist
// from (i mod N). Phase 2 still fills only [0, N); segments entirely
// in [-N, 0) are skipped naturally by the existing i_end <= i_start
// guard, and the last segment is clamped to N. ~2-3× the standard
// Phase 1 cost (one extra full-axis pass per ghost side); Phase 2
// unchanged.
template <bool Wrap>
inline void envelope_pass_row_contig_impl(
        int32_t* __restrict lbl, int32_t* __restrict dist, int64_t N,
        int32_t* __restrict v, int32_t* __restrict lblstk,
        int32_t* __restrict g, double* __restrict z,
        double* __restrict vd, double* __restrict vd_sq) {
    int32_t k = 0;
    // Inline push-onto-envelope helper (lambda captures all the stack arrays).
    auto push_seed = [&](int64_t i, int32_t lbl_val, int32_t gi) {
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
        lblstk[k] = lbl_val;
        g[k] = gi;
        k += 1;
    };

    if constexpr (Wrap) {
        // Pass 1a: ghost seeds at v - N (i ∈ [-N, 0), source from i + N).
        for (int64_t i = -N; i < 0; ++i) {
            const int64_t src = i + N;
            if (lbl[src] == 0) continue;
            push_seed(i, lbl[src], dist[src]);
        }
    }
    // Pass 1b: real seeds (i ∈ [0, N)).
    for (int64_t i = 0; i < N; ++i) {
        if (lbl[i] == 0) continue;
        push_seed(i, lbl[i], dist[i]);
    }
    if constexpr (Wrap) {
        // Pass 1c: ghost seeds at v + N (i ∈ [N, 2N), source from i - N).
        for (int64_t i = N; i < 2 * N; ++i) {
            const int64_t src = i - N;
            if (lbl[src] == 0) continue;
            push_seed(i, lbl[src], dist[src]);
        }
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

inline void envelope_pass_row_contig(
        int32_t* __restrict lbl, int32_t* __restrict dist, int64_t N,
        int32_t* __restrict v, int32_t* __restrict lblstk,
        int32_t* __restrict g, double* __restrict z,
        double* __restrict vd, double* __restrict vd_sq) {
    envelope_pass_row_contig_impl<false>(lbl, dist, N, v, lblstk, g, z, vd, vd_sq);
}

inline void envelope_pass_row_contig_wrap(
        int32_t* __restrict lbl, int32_t* __restrict dist, int64_t N,
        int32_t* __restrict v, int32_t* __restrict lblstk,
        int32_t* __restrict g, double* __restrict z,
        double* __restrict vd, double* __restrict vd_sq) {
    envelope_pass_row_contig_impl<true>(lbl, dist, N, v, lblstk, g, z, vd, vd_sq);
}

// Strided variant. Same algorithm as envelope_pass_row_contig but indexes
// lbl/dist with i*stride. Templated on Wrap so the wrap branches cost
// nothing at runtime when Wrap=false. See contig version for the wrap-
// algorithm rationale.
template <bool Wrap>
inline void envelope_pass_row_strided_impl(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    int32_t k = 0;
    auto push_seed = [&](int64_t i, int32_t lbl_val, int32_t gi) {
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
        lblstk[k] = lbl_val;
        g[k] = gi;
        k += 1;
    };

    if constexpr (Wrap) {
        for (int64_t i = -N; i < 0; ++i) {
            const int64_t src = (i + N) * stride;
            if (lbl[src] == 0) continue;
            push_seed(i, lbl[src], dist[src]);
        }
    }
    for (int64_t i = 0; i < N; ++i) {
        if (lbl[i * stride] == 0) continue;
        push_seed(i, lbl[i * stride], dist[i * stride]);
    }
    if constexpr (Wrap) {
        for (int64_t i = N; i < 2 * N; ++i) {
            const int64_t src = (i - N) * stride;
            if (lbl[src] == 0) continue;
            push_seed(i, lbl[src], dist[src]);
        }
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
        for (int64_t i = i_start; i < i_end; ++i) {
            const int32_t di = static_cast<int32_t>(i) - v_j;
            lbl[i * stride] = lbl_j;
            dist[i * stride] = g_j + di * di;
        }
        i_start = i_end;
    }
}

inline void envelope_pass_row(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    if (stride == 1) {
        envelope_pass_row_contig_impl<false>(lbl, dist, N, v, lblstk, g, z, vd, vd_sq);
    } else {
        envelope_pass_row_strided_impl<false>(lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
    }
}

inline void envelope_pass_row_wrap(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    if (stride == 1) {
        envelope_pass_row_contig_impl<true>(lbl, dist, N, v, lblstk, g, z, vd, vd_sq);
    } else {
        envelope_pass_row_strided_impl<true>(lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
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
    // Per-segment fill: for each consecutive seed pair (k, k+1), the midpoint
    // ceil((s_k + s_{k+1}) / 2) is the first index that snaps to seed k+1
    // (non-strict ``2*i >= s_k + s_{k+1}`` matches the integer envelope).
    // Within each segment seeds[k] and lbl_save[k] are constant, so we can
    // vectorise the (di*di + 0) write via envelope_fill_simd.
    int64_t i_start = 0;
    for (int32_t k = 0; k < n_seeds; ++k) {
        int64_t i_end;
        if (k + 1 == n_seeds) {
            i_end = N;
        } else {
            const int32_t mid_sum = seeds[k] + seeds[k + 1];
            // i_end = smallest i with 2*i >= mid_sum  =  ceil(mid_sum / 2).
            const int64_t mid_ceil = (static_cast<int64_t>(mid_sum) + 1) >> 1;
            i_end = mid_ceil > N ? N : mid_ceil;
        }
        if (i_end <= i_start) continue;
        envelope_fill_simd(lbl, dist, i_start, i_end,
                           lbl_save[k], /*g_j=*/0, seeds[k]);
        i_start = i_end;
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
        std::vector<EnvelopeScratch>& scratch, bool wrap = false) {
    if (n_threads < 1) n_threads = 1;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(n_slices),
        static_cast<size_t>(N)));
    if (static_cast<int>(scratch.size()) < eff_threads) scratch.resize(eff_threads);
    // Wrap pushes seeds with v in [-N, 2N), so envelope can hold up to 3N+1
    // dominant parabolas in pathological inputs.
    const size_t cap = static_cast<size_t>(wrap ? 3 * N : N) + 1;
    for (int t = 0; t < eff_threads; ++t) scratch[t].resize(cap);

    auto run_row = [&](int32_t* l, int32_t* d, int32_t* vp, int32_t* lp,
                       int32_t* gp, double* zp, double* vdp, double* vdsqp) {
        if (wrap) envelope_pass_row_wrap(l, d, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp);
        else      envelope_pass_row     (l, d, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp);
    };

    if (eff_threads == 1 || n_slices < 2) {
        auto& sc = scratch[0];
        for (int64_t s = 0; s < n_slices; ++s) {
            run_row(h_lbl + s * N, h_dist + s * N,
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
                run_row(h_lbl + s * N, h_dist + s * N, vp, lp, gp, zp, vdp, vdsqp);
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

// ABC strided variant: sweep axis B in an (A, B, C)-laid-out array.
// Each line k = (a, c) starts at base = a*B*C + c, length B, stride C.
// Avoids the 4-pass transpose+contiguous+transpose round-trip on axes
// where the column working set fits in cache. For 3D 256³ axis 1
// (B*C=65K elements ≈ 256 KiB) this halves expand time vs transpose.
//
// Bigger strides (e.g. 3D axis 0 where stride=H*W spans the whole image)
// are still cache-unfriendly; the caller is responsible for choosing
// strided vs transpose. See expand_labels_inplace for the threshold.
inline void envelope_pass_strided_abc(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t A, int64_t B, int64_t C,
        ForkJoinPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch, bool wrap = false) {
    if (n_threads < 1) n_threads = 1;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(A * C),
        static_cast<size_t>(B)));
    if (static_cast<int>(scratch.size()) < eff_threads) scratch.resize(eff_threads);
    // Wrap may push up to 3B seeds (ghost copies); see envelope_pass.
    const size_t cap = static_cast<size_t>(wrap ? 3 * B : B) + 1;
    for (int t = 0; t < eff_threads; ++t) scratch[t].resize(cap);

    auto run_row = [&](int32_t* l, int32_t* d, int32_t* vp, int32_t* lp,
                       int32_t* gp, double* zp, double* vdp, double* vdsqp) {
        if (wrap) envelope_pass_row_wrap(l, d, B, /*stride=*/C, vp, lp, gp, zp, vdp, vdsqp);
        else      envelope_pass_row     (l, d, B, /*stride=*/C, vp, lp, gp, zp, vdp, vdsqp);
    };

    const int64_t n_lines = A * C;
    if (eff_threads == 1 || n_lines < 2) {
        auto& sc = scratch[0];
        for (int64_t k = 0; k < n_lines; ++k) {
            const int64_t a = k / C;
            const int64_t c = k % C;
            const int64_t base = a * B * C + c;
            run_row(h_lbl + base, h_dist + base,
                    sc.v.data(), sc.lblstk.data(), sc.g.data(),
                    sc.z.data(), sc.vd.data(), sc.vd_sq.data());
        }
        return;
    }

    const int n_chunks = static_cast<int>(std::min<int64_t>(
        n_lines, static_cast<int64_t>(eff_threads) * DISPATCH_CHUNKS_PER_THREAD));
    const int64_t chunk_sz = (n_lines + n_chunks - 1) / n_chunks;
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
            const int64_t k0 = static_cast<int64_t>(idx) * chunk_sz;
            const int64_t k1 = std::min(k0 + chunk_sz, n_lines);
            for (int64_t k = k0; k < k1; ++k) {
                const int64_t a = k / C;
                const int64_t c = k % C;
                const int64_t base = a * B * C + c;
                run_row(h_lbl + base, h_dist + base, vp, lp, gp, zp, vdp, vdsqp);
            }
        }
    });
}

// 4x4 in-register transpose for 32-bit elements. src is 4 rows of 4 ints
// at stride sb; dst is 4 rows of 4 ints at stride db. Stage 1 does a
// pairwise 32-bit interleave; stage 2 swaps the 64-bit halves to finish.
#if defined(__aarch64__) || defined(__ARM_NEON)
template <typename T>
static inline void transpose_4x4_4byte(
        const T* __restrict__ src, int64_t sb,
        T* __restrict__ dst, int64_t db) {
    static_assert(sizeof(T) == 4, "transpose_4x4_4byte requires 4-byte T");
    uint32x4_t r0 = vld1q_u32(reinterpret_cast<const uint32_t*>(src + 0 * sb));
    uint32x4_t r1 = vld1q_u32(reinterpret_cast<const uint32_t*>(src + 1 * sb));
    uint32x4_t r2 = vld1q_u32(reinterpret_cast<const uint32_t*>(src + 2 * sb));
    uint32x4_t r3 = vld1q_u32(reinterpret_cast<const uint32_t*>(src + 3 * sb));
    uint32x4_t t0 = vtrn1q_u32(r0, r1);
    uint32x4_t t1 = vtrn2q_u32(r0, r1);
    uint32x4_t t2 = vtrn1q_u32(r2, r3);
    uint32x4_t t3 = vtrn2q_u32(r2, r3);
    uint32x4_t o0 = vreinterpretq_u32_u64(vtrn1q_u64(
        vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    uint32x4_t o2 = vreinterpretq_u32_u64(vtrn2q_u64(
        vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
    uint32x4_t o1 = vreinterpretq_u32_u64(vtrn1q_u64(
        vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    uint32x4_t o3 = vreinterpretq_u32_u64(vtrn2q_u64(
        vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
    vst1q_u32(reinterpret_cast<uint32_t*>(dst + 0 * db), o0);
    vst1q_u32(reinterpret_cast<uint32_t*>(dst + 1 * db), o1);
    vst1q_u32(reinterpret_cast<uint32_t*>(dst + 2 * db), o2);
    vst1q_u32(reinterpret_cast<uint32_t*>(dst + 3 * db), o3);
}
#endif

// Blocked batched transpose: src(A,B,C) → dst(A,C,B), for two arrays
// in lockstep (label + dist). Tile size 64 matches edt::TRANSPOSE_BLOCK
// (also matches the numba version's Bi=64). Uses atomic work-stealing
// dispatch over tile triples (a, rb, cb) — load balances naturally even
// when total_tiles is not a clean multiple of n_threads.
//
// Inside each tile:
//   - The two streams (label + dist) are transposed in separate inner
//     loops. Lockstep alternation forces the store buffer to drain to
//     two different destination cache lines per iteration; splitting
//     keeps each pass focused on one cache-line stream and lets the
//     compiler schedule the loads/stores independently per stream.
//   - On ARM64 with sizeof(T)==4 we transpose in 4×4 NEON sub-tiles,
//     which is roughly 2× faster on the 2D 4096² L2 expand benchmark
//     than scalar with the same blocking.
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
#if defined(__aarch64__) || defined(__ARM_NEON)
            if constexpr (sizeof(T) == 4) {
                const int64_t b1m = b0 + ((b1 - b0) & ~3);
                const int64_t c1m = c0 + ((c1 - c0) & ~3);
                // Transpose the two streams in separate passes.
                const T* base_sa = src_a + plane;
                const T* base_sb = src_b + plane;
                T*       base_da = dst_a + tplane;
                T*       base_db = dst_b + tplane;
                for (int pass = 0; pass < 2; ++pass) {
                    const T* base_s = (pass == 0) ? base_sa : base_sb;
                    T*       base_d = (pass == 0) ? base_da : base_db;
                    for (int64_t b = b0; b < b1m; b += 4) {
                        for (int64_t c = c0; c < c1m; c += 4) {
                            transpose_4x4_4byte<T>(
                                base_s + b * C + c, C,
                                base_d + c * B + b, B);
                        }
                        // c-edge fragment (only when C is not multiple of 4).
                        for (int64_t c = c1m; c < c1; ++c) {
                            for (int64_t bb = b; bb < b + 4; ++bb) {
                                base_d[c * B + bb] = base_s[bb * C + c];
                            }
                        }
                    }
                    // b-edge fragment.
                    for (int64_t b = b1m; b < b1; ++b) {
                        for (int64_t c = c0; c < c1; ++c) {
                            base_d[c * B + b] = base_s[b * C + c];
                        }
                    }
                }
                continue;
            }
#endif
            for (int64_t b = b0; b < b1; ++b) {
                const T* sa = src_a + plane + b * C;
                for (int64_t c = c0; c < c1; ++c) {
                    dst_a[tplane + c * B + b] = sa[c];
                }
            }
            for (int64_t b = b0; b < b1; ++b) {
                const T* sb = src_b + plane + b * C;
                for (int64_t c = c0; c < c1; ++c) {
                    dst_b[tplane + c * B + b] = sb[c];
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
        ForkJoinPool& pool, int n_threads, bool wrap = false) {
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
    // dist init unnecessary — pass0 overwrites every entry. (For wrap mode
    // we initialise dist explicitly below since the first axis routes to
    // envelope_pass instead of pass0.)

    for (int ax = ndim - 1; ax >= 0; --ax) {
        const int64_t n = shape[ax];
        if (ax == ndim - 1) {
            const int64_t n_slices = total / n;
            if (wrap) {
                // Wrap pass for the innermost axis: the midpoint-based
                // pass0 fast path doesn't generalise cleanly to torus
                // tie-break, so route to envelope_pass with wrap=true.
                // Init dist: 0 at seeds, INF elsewhere (envelope_pass
                // expects dist already populated).
                constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
                for (int64_t i = 0; i < total; ++i) {
                    h_dist[i] = (h_lbl[i] != 0) ? 0 : INF;
                }
                envelope_pass(h_lbl, h_dist, n_slices, n,
                              pool, n_threads, bufs.scratch(), /*wrap=*/true);
            } else {
                envelope_pass0(h_lbl, h_dist, n_slices, n, pool, n_threads, bufs.scratch());
            }
            continue;
        }
        int64_t A = 1;
        for (int d = 0; d < ax; ++d) A *= shape[d];
        int64_t C = 1;
        for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
        const int64_t B = n;

        // Pick strided slab sweep vs transpose+contig+transpose. Strided
        // wins only when there's an outer A dimension providing implicit
        // cache-blocking across slabs AND each slab (B*C elements) fits
        // in L2. For 3D 256³ axis 1 (A=256, B*C=64K) the per-slab
        // working set stays in cache and we skip 2× full-array transpose
        // bandwidth. For 2D and 3D outermost axes (A=1) there is no
        // slab structure: the entire image is one sweep, and strided
        // access through the whole array busts the cache and loses
        // vectorization, so transpose+contig wins by ~25%.
        //
        // Threshold: A >= 2 (have a slab axis) AND B*C <= 4M ints (~16
        // MiB ≤ M2 shared L2). Tuned on M2 / AMD Ryzen / Threadripper.
        constexpr int64_t STRIDED_SLAB_LIMIT = 4 * 1024 * 1024;
        const bool use_strided = (A >= 2) && (B * C <= STRIDED_SLAB_LIMIT);
        if (use_strided) {
            envelope_pass_strided_abc(h_lbl, h_dist, A, B, C,
                                      pool, n_threads, bufs.scratch(), wrap);
        } else {
            batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist, A, B, C, pool, n_threads);
            envelope_pass(t_lbl, t_dist, A * C, B, pool, n_threads, bufs.scratch(), wrap);
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
