/*
 * expand.hpp — Voronoi label expansion under L2 (squared Euclidean).
 *
 * Felzenszwalb–Huttenlocher (2012) parabolic-envelope distance transform,
 * separable over axes. For an N-dim label image:
 *   for ax in reversed(range(ndim)):
 *     1. transpose so ax is innermost (skip if already last)
 *     2. parabolic envelope pass over the innermost axis
 *     3. transpose back (skip if first iteration)
 *
 * Work is distributed across the persistent ForkJoinPool — rows for the
 * envelope passes, tile pairs for the transposes.
 */

#ifndef NCOLOR_EXPAND_HPP
#define NCOLOR_EXPAND_HPP

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

// SIMD selection. NEON is unconditional on arm64. On x86 the 4-wide SSE
// path needs only SSE2, which every x86_64 compiler has on by default
// (GCC/clang define __SSE2__; MSVC defines _M_X64 and never __SSE2__),
// so the wheel builds get it too. AVX2 adds an 8-wide fill when the
// build targets it (-march=native / x86-64-v3, or MSVC /arch:AVX2).
#if defined(__aarch64__) || defined(__ARM_NEON)
#  include <arm_neon.h>
#  define NCOLOR_SIMD_NEON 1
#elif defined(__SSE2__) || defined(_M_X64) || \
      (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#  include <immintrin.h>
#  define NCOLOR_SIMD_X86 1
#endif
#include <vector>

#include "dispatch.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

using ::ForkJoinPool;

// Strided slab sweeps beat transpose + contiguous + transpose only while
// one slab (B*C elements) fits in L2: 4M ints (16 MiB). Shared by the
// plain and the barrier-aware axis drivers.
constexpr int64_t STRIDED_SLAB_LIMIT = 4 * 1024 * 1024;

#if defined(NCOLOR_SIMD_X86)
// 32-bit lane-wise multiply. One instruction from SSE4.1 up; on a plain
// SSE2 target it is two 32x32->64 multiplies on the even and odd lanes
// whose low halves are re-interleaved (two's complement makes the low 32
// bits identical for signed inputs).
static inline __m128i mullo_epi32_sse(__m128i a, __m128i b) {
#if defined(__SSE4_1__) || defined(__AVX__)
    return _mm_mullo_epi32(a, b);
#else
    const __m128i even = _mm_mul_epu32(a, b);
    const __m128i odd  = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(even, _MM_SHUFFLE(0, 0, 2, 0)),
                              _mm_shuffle_epi32(odd,  _MM_SHUFFLE(0, 0, 2, 0)));
#endif
}
#endif

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
// (4×int32 per iteration), x86 AVX2 (8×int32) and x86 SSE2/SSE4.1
// (4×int32). Scalar tail handles the remainder.
//
// Hand-rolled because clang -O3 -march=native consistently fails to
// vectorize the int32 ``di*di`` multiply + paired stores even with
// __restrict qualifiers.
static inline void envelope_fill_simd(
        int32_t* __restrict lbl, int32_t* __restrict dist,
        int64_t i_start, int64_t i_end,
        int32_t lbl_j, int32_t g_j, int32_t v_j) {
    int64_t i = i_start;
#if defined(NCOLOR_SIMD_NEON)
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
#elif defined(NCOLOR_SIMD_X86)
#  if defined(__AVX2__)
    {
        const __m256i w_lbl  = _mm256_set1_epi32(lbl_j);
        const __m256i w_g    = _mm256_set1_epi32(g_j);
        const __m256i w_vj   = _mm256_set1_epi32(v_j);
        const __m256i w_step = _mm256_set1_epi32(8);
        __m256i w_i = _mm256_add_epi32(_mm256_set1_epi32(static_cast<int32_t>(i)),
                                       _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        for (; i + 8 <= i_end; i += 8) {
            const __m256i w_di   = _mm256_sub_epi32(w_i, w_vj);
            const __m256i w_dist = _mm256_add_epi32(_mm256_mullo_epi32(w_di, w_di), w_g);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(lbl + i), w_lbl);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dist + i), w_dist);
            w_i = _mm256_add_epi32(w_i, w_step);
        }
    }
#  endif
    {
        const __m128i v_lbl = _mm_set1_epi32(lbl_j);
        const __m128i v_g   = _mm_set1_epi32(g_j);
        const __m128i v_vj  = _mm_set1_epi32(v_j);
        const __m128i v_four = _mm_set1_epi32(4);
        __m128i v_i = _mm_add_epi32(_mm_set1_epi32(static_cast<int32_t>(i)),
                                    _mm_set_epi32(3, 2, 1, 0));
        for (; i + 4 <= i_end; i += 4) {
            const __m128i v_di   = _mm_sub_epi32(v_i, v_vj);
            const __m128i v_dist = _mm_add_epi32(mullo_epi32_sse(v_di, v_di), v_g);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(lbl + i), v_lbl);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dist + i), v_dist);
            v_i = _mm_add_epi32(v_i, v_four);
        }
    }
#endif
    for (; i < i_end; ++i) {
        const int32_t di = static_cast<int32_t>(i) - v_j;
        lbl[i] = lbl_j;
        dist[i] = g_j + di * di;
    }
}

// Barrier-aware variant of envelope_fill_simd for the clean expand:
// pixels whose dist equals ``barrier`` are left untouched (both arrays),
// everything else is filled exactly as above. Vector lanes are selected
// with a compare mask, so the barrier check costs no branch.
static inline void envelope_fill_barrier_simd(
        int32_t* __restrict lbl, int32_t* __restrict dist,
        int64_t i_start, int64_t i_end,
        int32_t lbl_j, int32_t g_j, int32_t v_j, int32_t barrier) {
    int64_t i = i_start;
#if defined(NCOLOR_SIMD_NEON)
    const int32x4_t v_lbl = vdupq_n_s32(lbl_j);
    const int32x4_t v_g   = vdupq_n_s32(g_j);
    const int32x4_t v_vj  = vdupq_n_s32(v_j);
    const int32x4_t v_bar = vdupq_n_s32(barrier);
    const int32x4_t v_inc = {0, 1, 2, 3};
    const int32x4_t v_four = vdupq_n_s32(4);
    int32x4_t v_i = vaddq_s32(vdupq_n_s32(static_cast<int32_t>(i_start)), v_inc);
    for (; i + 4 <= i_end; i += 4) {
        const int32x4_t d_old = vld1q_s32(dist + i);
        const uint32x4_t keep = vceqq_s32(d_old, v_bar);
        const int32x4_t v_di = vsubq_s32(v_i, v_vj);
        const int32x4_t d_new = vaddq_s32(vmulq_s32(v_di, v_di), v_g);
        const int32x4_t l_old = vld1q_s32(lbl + i);
        vst1q_s32(lbl + i, vbslq_s32(keep, l_old, v_lbl));
        vst1q_s32(dist + i, vbslq_s32(keep, d_old, d_new));
        v_i = vaddq_s32(v_i, v_four);
    }
#elif defined(NCOLOR_SIMD_X86)
    {
        const __m128i v_lbl = _mm_set1_epi32(lbl_j);
        const __m128i v_g   = _mm_set1_epi32(g_j);
        const __m128i v_vj  = _mm_set1_epi32(v_j);
        const __m128i v_bar = _mm_set1_epi32(barrier);
        const __m128i v_four = _mm_set1_epi32(4);
        __m128i v_i = _mm_add_epi32(_mm_set1_epi32(static_cast<int32_t>(i)),
                                    _mm_set_epi32(3, 2, 1, 0));
        for (; i + 4 <= i_end; i += 4) {
            const __m128i d_old = _mm_loadu_si128(reinterpret_cast<const __m128i*>(dist + i));
            const __m128i keep  = _mm_cmpeq_epi32(d_old, v_bar);
            const __m128i v_di  = _mm_sub_epi32(v_i, v_vj);
            const __m128i d_new = _mm_add_epi32(mullo_epi32_sse(v_di, v_di), v_g);
            const __m128i l_old = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lbl + i));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(lbl + i),
                             _mm_or_si128(_mm_and_si128(keep, l_old), _mm_andnot_si128(keep, v_lbl)));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dist + i),
                             _mm_or_si128(_mm_and_si128(keep, d_old), _mm_andnot_si128(keep, d_new)));
            v_i = _mm_add_epi32(v_i, v_four);
        }
    }
#endif
    for (; i < i_end; ++i) {
        if (dist[i] == barrier) continue;
        const int32_t di = static_cast<int32_t>(i) - v_j;
        lbl[i] = lbl_j;
        dist[i] = g_j + di * di;
    }
}

// Templated on (Wrap, Contig) so each (wrap × stride==1) pair gets its own
// inlined specialization with all branches resolved at compile time:
//   - Wrap=false : Phase 1 = real seeds only (one [0,N) sweep).
//   - Wrap=true  : ghost seeds at v±N → Phase 1 sweeps [-N, 2N), reading
//                  lbl/dist from (i mod N). Phase 2 still fills only
//                  [0, N); segments entirely in [-N, 0) are skipped by
//                  the i_end ≤ i_start guard, and the last segment is
//                  clamped to N. ~2-3× the standard Phase 1 cost.
//   - Contig=true  : direct lbl[i] / dist[i] access; Phase 2 uses the
//                    SIMD fill helper.
//   - Contig=false : strided lbl[i*stride] / dist[i*stride]; scalar fill.
template <bool Wrap, bool Contig, bool Barrier = false>
inline void envelope_pass_row_impl(
        int32_t* __restrict lbl, int32_t* __restrict dist,
        int64_t N, int64_t stride,
        int32_t* __restrict v, int32_t* __restrict lblstk,
        int32_t* __restrict g, double* __restrict z,
        double* __restrict vd, double* __restrict vd_sq) {
    int32_t k = 0;
    // Inline push-onto-envelope helper.
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

    auto load_lbl  = [&](int64_t idx) -> int32_t {
        if constexpr (Contig) return lbl[idx]; else return lbl[idx * stride];
    };
    auto load_dist = [&](int64_t idx) -> int32_t {
        if constexpr (Contig) return dist[idx]; else return dist[idx * stride];
    };

    if constexpr (Wrap) {
        // Pass 1a: ghost seeds at v - N (i ∈ [-N, 0), source v = i + N).
        // A ghost-left v - N is everywhere ≥ its real counterpart v over
        // [0, N) when v ≤ N/2 (crossover at i = v - N/2 ≤ 0), so it can
        // never win the envelope and we skip the push. Equivalently, only
        // ghost positions with i > -N/2 contribute. Smallest integer i
        // satisfying i > -N/2 strictly is -((N - 1) / 2).
        const int64_t pass1a_start = -((N - 1) / 2);
        for (int64_t i = pass1a_start; i < 0; ++i) {
            const int32_t lv = load_lbl(i + N);
            if (lv == 0) continue;
            push_seed(i, lv, load_dist(i + N));
        }
    }
    // Pass 1b: real seeds (i ∈ [0, N)).
    for (int64_t i = 0; i < N; ++i) {
        const int32_t lv = load_lbl(i);
        if (lv == 0) continue;
        push_seed(i, lv, load_dist(i));
    }
    if constexpr (Wrap) {
        // Pass 1c: ghost seeds at v + N (i ∈ [N, 2N), source v = i - N).
        // Symmetric to 1a: ghost-right is dominated by its real when
        // v ≥ N/2. Stop at i = N + ceil(N/2) exclusive ((N + 1)/2 in
        // integer arithmetic).
        const int64_t pass1c_end = N + (N + 1) / 2;
        for (int64_t i = N; i < pass1c_end; ++i) {
            const int32_t lv = load_lbl(i - N);
            if (lv == 0) continue;
            push_seed(i, lv, load_dist(i - N));
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
        if constexpr (Contig) {
            if constexpr (Barrier) {
                envelope_fill_barrier_simd(lbl, dist, i_start, i_end,
                    lbl_j, g_j, v_j, INT32_MIN);
            } else {
                envelope_fill_simd(lbl, dist, i_start, i_end, lbl_j, g_j, v_j);
            }
        } else {
            for (int64_t i = i_start; i < i_end; ++i) {
                if constexpr (Barrier) {
                    if (dist[i * stride] == INT32_MIN) continue;
                }
                const int32_t di = static_cast<int32_t>(i) - v_j;
                lbl[i * stride] = lbl_j;
                dist[i * stride] = g_j + di * di;
            }
        }
        i_start = i_end;
    }
}

inline void envelope_pass_row(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    if (stride == 1) envelope_pass_row_impl<false, true >(lbl, dist, N, 1,      v, lblstk, g, z, vd, vd_sq);
    else             envelope_pass_row_impl<false, false>(lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
}

inline void envelope_pass_row_wrap(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    if (stride == 1) envelope_pass_row_impl<true, true >(lbl, dist, N, 1,      v, lblstk, g, z, vd, vd_sq);
    else             envelope_pass_row_impl<true, false>(lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
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
    // vectorize the (di*di + 0) write via envelope_fill_simd.
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

    dispatch_parallel_with_scratch(pool, eff_threads,
        static_cast<size_t>(n_slices),
        static_cast<size_t>(eff_threads) * DISPATCH_CHUNKS_PER_THREAD,
        scratch,
        [&](EnvelopeScratch& sc, size_t s0, size_t s1) {
            int32_t* sp = sc.v.data();
            int32_t* lp = sc.lblstk.data();
            for (size_t s = s0; s < s1; ++s) {
                envelope_pass0_row(h_lbl + s * N, h_dist + s * N, N, sp, lp);
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

    dispatch_parallel_with_scratch(pool, eff_threads,
        static_cast<size_t>(n_slices),
        static_cast<size_t>(eff_threads) * DISPATCH_CHUNKS_PER_THREAD,
        scratch,
        [&](EnvelopeScratch& sc, size_t s0, size_t s1) {
            int32_t* vp = sc.v.data(); int32_t* lp = sc.lblstk.data();
            int32_t* gp = sc.g.data();
            double* zp = sc.z.data();  double* vdp = sc.vd.data();
            double* vdsqp = sc.vd_sq.data();
            for (size_t s = s0; s < s1; ++s) {
                int32_t* l = h_lbl + s * N;
                int32_t* d = h_dist + s * N;
                if (wrap) envelope_pass_row_wrap(l, d, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp);
                else      envelope_pass_row     (l, d, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp);
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
    const int64_t n_lines = A * C;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(n_lines),
        static_cast<size_t>(B)));
    if (static_cast<int>(scratch.size()) < eff_threads) scratch.resize(eff_threads);
    // Wrap may push up to 3B seeds (ghost copies); see envelope_pass.
    const size_t cap = static_cast<size_t>(wrap ? 3 * B : B) + 1;
    for (int t = 0; t < eff_threads; ++t) scratch[t].resize(cap);

    dispatch_parallel_with_scratch(pool, eff_threads,
        static_cast<size_t>(n_lines),
        static_cast<size_t>(eff_threads) * DISPATCH_CHUNKS_PER_THREAD,
        scratch,
        [&](EnvelopeScratch& sc, size_t k0, size_t k1) {
            int32_t* vp = sc.v.data(); int32_t* lp = sc.lblstk.data();
            int32_t* gp = sc.g.data();
            double* zp = sc.z.data();  double* vdp = sc.vd.data();
            double* vdsqp = sc.vd_sq.data();
            for (size_t k = k0; k < k1; ++k) {
                const int64_t a = static_cast<int64_t>(k) / C;
                const int64_t c = static_cast<int64_t>(k) % C;
                const int64_t base = a * B * C + c;
                int32_t* l = h_lbl + base;
                int32_t* d = h_dist + base;
                if (wrap) envelope_pass_row_wrap(l, d, B, /*stride=*/C, vp, lp, gp, zp, vdp, vdsqp);
                else      envelope_pass_row     (l, d, B, /*stride=*/C, vp, lp, gp, zp, vdp, vdsqp);
            }
        });
}

// 4x4 in-register transpose for 32-bit elements. src is 4 rows of 4 ints
// at stride sb; dst is 4 rows of 4 ints at stride db. Stage 1 does a
// pairwise 32-bit interleave; stage 2 swaps the 64-bit halves to finish.
#if defined(NCOLOR_SIMD_X86)
// SSE2: the classic 4x4 float transpose applied to the integer bit
// patterns. Shuffles move lanes verbatim, so the ints are unchanged.
template <typename T>
static inline void transpose_4x4_4byte(
        const T* __restrict src, int64_t sb,
        T* __restrict dst, int64_t db) {
    static_assert(sizeof(T) == 4, "transpose_4x4_4byte requires 4-byte T");
    __m128 r0 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 0 * sb)));
    __m128 r1 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 1 * sb)));
    __m128 r2 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * sb)));
    __m128 r3 = _mm_castsi128_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3 * sb)));
    _MM_TRANSPOSE4_PS(r0, r1, r2, r3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 0 * db), _mm_castps_si128(r0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 1 * db), _mm_castps_si128(r1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 2 * db), _mm_castps_si128(r2));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 3 * db), _mm_castps_si128(r3));
}
#elif defined(NCOLOR_SIMD_NEON)
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
//   - With sizeof(T)==4 we transpose in 4×4 in-register sub-tiles (NEON
//     on arm64, SSE2 on x86), which is roughly 2× faster on the 2D 4096²
//     L2 expand benchmark than scalar with the same blocking.
template <typename T>
void batch_transpose(
        const T* src_a, const T* src_b,
        T* dst_a, T* dst_b,
        int64_t A, int64_t B, int64_t C,
        ForkJoinPool& pool, int n_threads, bool copy_distances = true) {
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
#if defined(NCOLOR_SIMD_NEON) || defined(NCOLOR_SIMD_X86)
            if constexpr (sizeof(T) == 4) {
                const int64_t b1m = b0 + ((b1 - b0) & ~3);
                const int64_t c1m = c0 + ((c1 - c0) & ~3);
                // Transpose the two streams in separate passes.
                const T* base_sa = src_a + plane;
                const T* base_sb = src_b + plane;
                T*       base_da = dst_a + tplane;
                T*       base_db = dst_b + tplane;
                for (int pass = 0; pass < (copy_distances ? 2 : 1); ++pass) {
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
            if (copy_distances) for (int64_t b = b0; b < b1; ++b) {
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
    void resize(int64_t total, bool working_labels = true) {
        if (working_labels && h_lbl_.size() < static_cast<size_t>(total))
            h_lbl_.resize(total);
        if (h_dist_.size() < static_cast<size_t>(total)) h_dist_.resize(total);
        size_ = total;
        wide_ = false;
    }
    int32_t* lbl()   { return h_lbl_.data(); }
    int32_t* dist()  { return h_dist_.data(); }
    int32_t* lbl_T() {
        if (t_lbl_.size() < static_cast<size_t>(size_)) t_lbl_.resize(size_);
        return t_lbl_.data();
    }
    int32_t* dist_T() {
        if (t_dist_.size() < static_cast<size_t>(size_)) t_dist_.resize(size_);
        return t_dist_.data();
    }
    void use_wide_distance() { wide_dist_.resize(size_); wide_ = true; }
    bool wide_distance() const { return wide_; }
    int64_t* dist64() { return wide_dist_.data(); }
    double distance_at(int64_t i) const {
        return wide_ ? static_cast<double>(wide_dist_[i])
                     : static_cast<double>(h_dist_[i]);
    }
    int64_t size() const { return size_; }
    // Per-worker envelope scratch (resized lazily).
    std::vector<EnvelopeScratch>& scratch() { return scratch_; }
    // Give back persistent image and worker scratch. Transpose and wide
    // distance buffers are allocated only when their kernels need them.
    // The next call simply reallocates.
    void release() {
        std::vector<int32_t>().swap(h_lbl_);
        std::vector<int32_t>().swap(h_dist_);
        std::vector<int32_t>().swap(t_lbl_);
        std::vector<int32_t>().swap(t_dist_);
        std::vector<EnvelopeScratch>().swap(scratch_);
        std::vector<uint8_t>().swap(nbr_);
        std::vector<int64_t>().swap(wide_dist_);
        wide_ = false;
        size_ = 0;
    }
    // Per-pixel neighbor-count scratch for the clean expand's bridge
    // check (one byte per pixel, no zeroing needed between calls).
    std::vector<uint8_t>& nbr_scratch() { return nbr_; }
private:
    std::vector<int32_t> h_lbl_, h_dist_, t_lbl_, t_dist_;
    std::vector<int64_t> wide_dist_;
    bool wide_ = false;
    int64_t size_ = 0;
    std::vector<EnvelopeScratch> scratch_;
    std::vector<uint8_t> nbr_;
};


// =============================================================================

// Choose distance storage from geometry, not pixel count. Reserve headroom
// for squared ghost positions and envelope intersections in the wide path.
inline bool l2_needs_wide_distance(const std::vector<int64_t>& shape) {
    constexpr int64_t limit = INT64_MAX / 16;
    int64_t bound = 0;
    for (int64_t n : shape) {
        if (n <= 1) continue;
        const int64_t span = n - 1;
        if (span > limit / span || span * span > limit - bound)
            throw std::overflow_error("squared image diameter exceeds distance capacity");
        bound += span * span;
    }
    return bound > INT32_MAX;
}

// Wide envelope uses exact integer breakpoints: the first lattice position
// at which the later seed wins. This also preserves the narrow path's tie
// rule without floating-point cancellation on long axes.
struct WideEnvelopeScratch {
    std::vector<int64_t> v, g, start;
    std::vector<int32_t> label;
    explicit WideEnvelopeScratch(size_t n) : v(n), g(n), start(n), label(n) {}
};

inline void envelope_row_wide(int32_t* lbl, int64_t* dist,
                              int64_t n, int64_t stride,
                              WideEnvelopeScratch& sc, bool wrap) {
    int64_t k = 0;
    const int64_t begin = wrap ? -n : 0;
    const int64_t end = wrap ? 2 * n : n;
    for (int64_t i = begin; i < end; ++i) {
        const int64_t src = ((i + n) % n) * stride;
        if (!lbl[src]) continue;
        int64_t start = INT64_MIN;
        while (k) {
            const int64_t j = k - 1;
            const int64_t num = dist[src] - sc.g[j] + i * i - sc.v[j] * sc.v[j];
            const int64_t den = 2 * (i - sc.v[j]);
            start = num / den + (num % den > 0);
            if (start > sc.start[j]) break;
            --k;
        }
        if (!k) start = INT64_MIN;
        sc.v[k] = i;
        sc.g[k] = dist[src];
        sc.label[k] = lbl[src];
        sc.start[k++] = start;
    }
    for (int64_t j = 0; j < k; ++j) {
        const int64_t lo = std::max<int64_t>(0, sc.start[j]);
        const int64_t hi = j + 1 == k ? n : std::min(n, sc.start[j + 1]);
        for (int64_t i = lo; i < hi; ++i) {
            const int64_t off = i * stride;
            if (dist[off] == INT32_MIN) continue;  // sticky clean barrier
            const int64_t delta = i - sc.v[j];
            lbl[off] = sc.label[j];
            dist[off] = sc.g[j] + delta * delta;
        }
    }
}

inline void l2_sweep_axis_wide(int32_t* lbl, int64_t* dist,
                              const std::vector<int64_t>& shape, int ax,
                              ForkJoinPool& pool, int n_threads, bool wrap) {
    int64_t a = 1, c = 1;
    for (int d = 0; d < ax; ++d) a *= shape[d];
    for (size_t d = ax + 1; d < shape.size(); ++d) c *= shape[d];
    const int64_t n = shape[ax];
    if (ax == static_cast<int>(shape.size()) - 1) {
        for (int64_t i = 0; i < a * n * c; ++i)
            dist[i] = lbl[i] ? 0 : INT64_MAX / 16;
    }
    if (n <= 1) return;
    const size_t workers = compute_threads(std::max(1, n_threads), a * c, n);
    std::vector<WideEnvelopeScratch> scratch;
    for (size_t t = 0; t < workers; ++t)
        scratch.emplace_back(static_cast<size_t>(wrap ? 3 * n : n));
    dispatch_parallel_with_scratch(pool, workers, a * c,
        workers * DISPATCH_CHUNKS_PER_THREAD, scratch,
        [&](WideEnvelopeScratch& sc, size_t lo, size_t hi) {
            for (size_t line = lo; line < hi; ++line) {
                const int64_t off = (line / c) * n * c + line % c;
                envelope_row_wide(lbl + off, dist + off, n, c, sc, wrap);
            }
        });
}

// Run expand_labels on a row-major label image of arbitrary ndim.
// `shape` is the image shape; total = product of shape entries; the output
// is written into `bufs.lbl()` which is also the working scratch.
// Internal contact consumers may request final_shape: narrow transforms with
// two or more active axes then leave labels/distances in lbl_T()/dist_T()
// and return the cyclically rotated active shape. Wide or 1D transforms
// retain the original storage. Public expansion callers omit this request.
inline void expand_labels_inplace(
        const int32_t* input, ExpandBuffers& bufs,
        const std::vector<int64_t>& shape,
        ForkJoinPool& pool, int n_threads, bool wrap = false,
        bool keep_distances = true,
        std::vector<int64_t>* final_shape = nullptr) {
    // Singleton axes contribute no distance. Removing them preserves axis
    // order and lets the first active axis use the sparse-seed fast pass.
    if (shape.empty()) {
        expand_labels_inplace(input, bufs, {1}, pool, n_threads, wrap, keep_distances, final_shape);
        return;
    }
    if (shape.size() > 1 && std::find(shape.begin(), shape.end(), 1) != shape.end()) {
        std::vector<int64_t> active;
        for (int64_t n : shape) if (n != 1) active.push_back(n);
        if (active.empty()) active.push_back(1);
        expand_labels_inplace(input, bufs, active, pool, n_threads, wrap, keep_distances, final_shape);
        return;
    }
    if (final_shape) *final_shape = shape;
    const int ndim = static_cast<int>(shape.size());
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    bufs.resize(total);
    int32_t* h_lbl = bufs.lbl();
    int32_t* h_dist = bufs.dist();

    if (input != h_lbl) {
        std::memcpy(h_lbl, input, total * sizeof(int32_t));
    }
    if (total == 0) return;
    if (l2_needs_wide_distance(shape)) {
        bufs.use_wide_distance();
        for (int ax = ndim - 1; ax >= 0; --ax)
            l2_sweep_axis_wide(h_lbl, bufs.dist64(), shape, ax, pool, n_threads, wrap);
        return;
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
                // pass0 fast path doesn't generalize cleanly to torus
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
        if (n == 1) continue;
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
        // MiB ≤ M1 Ultra shared L2). Tuned on M1 Ultra / AMD Ryzen / Threadripper.
        const bool use_strided = (A >= 2) && (B * C <= STRIDED_SLAB_LIMIT);
        if (use_strided) {
            envelope_pass_strided_abc(h_lbl, h_dist, A, B, C,
                                      pool, n_threads, bufs.scratch(), wrap);
        } else {
            int32_t* t_lbl = bufs.lbl_T();
            int32_t* t_dist = bufs.dist_T();
            batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist, A, B, C, pool, n_threads);
            envelope_pass(t_lbl, t_dist, A * C, B, pool, n_threads, bufs.scratch(), wrap);
            if (final_shape && ax == 0) {
                // Contacts can consume this layout directly. Public expansion
                // callers omit final_shape and receive the original layout.
                std::rotate(final_shape->begin(), final_shape->begin() + 1, final_shape->end());
            } else {
                batch_transpose<int32_t>(t_lbl, t_dist, h_lbl, h_dist, A, C, B, pool, n_threads,
                                          keep_distances || ax != 0);
            }
        }
    }
}

} // namespace ncolor_cpp

#endif // NCOLOR_EXPAND_HPP
