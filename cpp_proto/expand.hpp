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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <vector>

#include "threadpool.h"

namespace ncolor_cpp {

using ::ThreadPool;

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
inline void envelope_pass_row(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
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

// Per-worker scratch for envelope_pass. Held by ExpandBuffers so allocations
// persist across calls (the numba version pays the same cost per call inside
// each @njit prange iteration).
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
        ThreadPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch) {
    if (n_threads < 1) n_threads = 1;
    if (static_cast<int>(scratch.size()) < n_threads) scratch.resize(n_threads);
    const size_t cap = static_cast<size_t>(N) + 1;
    for (int t = 0; t < n_threads; ++t) scratch[t].resize(cap);

    if (n_threads == 1 || n_slices < 2) {
        for (int64_t s = 0; s < n_slices; ++s) {
            envelope_pass0_row(h_lbl + s * N, h_dist + s * N, N,
                               scratch[0].v.data(), scratch[0].lblstk.data());
        }
        return;
    }

    const int64_t per_thread = (n_slices + n_threads - 1) / n_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int64_t s0 = t * per_thread;
        const int64_t s1 = std::min(s0 + per_thread, n_slices);
        if (s0 >= s1) continue;
        int32_t* sp = scratch[t].v.data();
        int32_t* lp = scratch[t].lblstk.data();
        futures.emplace_back(pool.enqueue([h_lbl, h_dist, N, s0, s1, sp, lp]() {
            for (int64_t s = s0; s < s1; ++s) {
                envelope_pass0_row(h_lbl + s * N, h_dist + s * N, N, sp, lp);
            }
        }));
    }
    for (auto& f : futures) f.get();
}

// Pass over (n_slices, N) row-major arrays in parallel.
inline void envelope_pass(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_slices, int64_t N,
        ThreadPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch) {
    if (n_threads < 1) n_threads = 1;
    if (static_cast<int>(scratch.size()) < n_threads) scratch.resize(n_threads);
    const size_t cap = static_cast<size_t>(N) + 1;
    for (int t = 0; t < n_threads; ++t) scratch[t].resize(cap);

    if (n_threads == 1 || n_slices < 2) {
        auto& sc = scratch[0];
        for (int64_t s = 0; s < n_slices; ++s) {
            envelope_pass_row(h_lbl + s * N, h_dist + s * N, N, /*stride=*/1,
                              sc.v.data(), sc.lblstk.data(), sc.g.data(),
                              sc.z.data(), sc.vd.data(), sc.vd_sq.data());
        }
        return;
    }

    const int64_t per_thread = (n_slices + n_threads - 1) / n_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int64_t s0 = t * per_thread;
        const int64_t s1 = std::min(s0 + per_thread, n_slices);
        if (s0 >= s1) continue;
        auto& sc = scratch[t];
        int32_t* vp = sc.v.data();
        int32_t* lp = sc.lblstk.data();
        int32_t* gp = sc.g.data();
        double* zp = sc.z.data();
        double* vdp = sc.vd.data();
        double* vdsqp = sc.vd_sq.data();
        futures.emplace_back(pool.enqueue([h_lbl, h_dist, N, s0, s1, vp, lp, gp, zp, vdp, vdsqp]() {
            for (int64_t s = s0; s < s1; ++s) {
                envelope_pass_row(h_lbl + s * N, h_dist + s * N, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp);
            }
        }));
    }
    for (auto& f : futures) f.get();
}

// Strided variant: process one logical "line" per call where pixels are
// stride-spaced in memory. Use case: a column of an (H, W) row-major image
// with stride=W, base=col_index. Avoids the transpose round-trip entirely.
inline void envelope_pass_strided(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_lines, int64_t N, int64_t stride,
        ThreadPool& pool, int n_threads,
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

    const int64_t per_thread = (n_lines + n_threads - 1) / n_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int64_t s0 = t * per_thread;
        const int64_t s1 = std::min(s0 + per_thread, n_lines);
        if (s0 >= s1) continue;
        auto& sc = scratch[t];
        int32_t* vp = sc.v.data();
        int32_t* lp = sc.lblstk.data();
        int32_t* gp = sc.g.data();
        double* zp = sc.z.data();
        double* vdp = sc.vd.data();
        double* vdsqp = sc.vd_sq.data();
        futures.emplace_back(pool.enqueue([h_lbl, h_dist, N, stride, s0, s1, vp, lp, gp, zp, vdp, vdsqp]() {
            for (int64_t s = s0; s < s1; ++s) {
                envelope_pass_row(h_lbl + s, h_dist + s, N, stride, vp, lp, gp, zp, vdp, vdsqp);
            }
        }));
    }
    for (auto& f : futures) f.get();
}

// Blocked batched transpose: src(A,B,C) → dst(A,C,B), for two arrays
// in lockstep (label + dist). Tile size matches the numba version (Bi=64).
template <typename T>
void batch_transpose(
        const T* src_a, const T* src_b,
        T* dst_a, T* dst_b,
        int64_t A, int64_t B, int64_t C,
        ThreadPool& pool, int n_threads) {
    constexpr int Bi = 64;
    const int64_t n_b = (B + Bi - 1) / Bi;
    const int64_t n_c = (C + Bi - 1) / Bi;
    const int64_t total_tiles = A * n_b * n_c;

    if (n_threads < 1) n_threads = 1;
    if (n_threads == 1 || total_tiles < 4) {
        for (int64_t i = 0; i < total_tiles; ++i) {
            const int64_t a = i / (n_b * n_c);
            const int64_t bc = i % (n_b * n_c);
            const int64_t b0 = (bc / n_c) * Bi;
            const int64_t c0 = (bc % n_c) * Bi;
            for (int64_t b = b0; b < std::min<int64_t>(b0 + Bi, B); ++b) {
                for (int64_t c = c0; c < std::min<int64_t>(c0 + Bi, C); ++c) {
                    dst_a[a * (C * B) + c * B + b] = src_a[a * (B * C) + b * C + c];
                    dst_b[a * (C * B) + c * B + b] = src_b[a * (B * C) + b * C + c];
                }
            }
        }
        return;
    }

    const int64_t per_thread = (total_tiles + n_threads - 1) / n_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);
    for (int t = 0; t < n_threads; ++t) {
        const int64_t i0 = t * per_thread;
        const int64_t i1 = std::min(i0 + per_thread, total_tiles);
        if (i0 >= i1) continue;
        futures.emplace_back(pool.enqueue([src_a, src_b, dst_a, dst_b, A, B, C, n_b, n_c, i0, i1]() {
            for (int64_t i = i0; i < i1; ++i) {
                const int64_t a = i / (n_b * n_c);
                const int64_t bc = i % (n_b * n_c);
                const int64_t b0 = (bc / n_c) * Bi;
                const int64_t c0 = (bc % n_c) * Bi;
                for (int64_t b = b0; b < std::min<int64_t>(b0 + Bi, B); ++b) {
                    for (int64_t c = c0; c < std::min<int64_t>(c0 + Bi, C); ++c) {
                        dst_a[a * (C * B) + c * B + b] = src_a[a * (B * C) + b * C + c];
                        dst_b[a * (C * B) + c * B + b] = src_b[a * (B * C) + b * C + c];
                    }
                }
            }
        }));
    }
    for (auto& f : futures) f.get();
}

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

// Run expand_labels on a row-major label image of arbitrary ndim.
// `shape` is the image shape; total = product of shape entries; the output
// is written into `bufs.lbl()` which is also the working scratch.
inline void expand_labels_inplace(
        const int32_t* input, ExpandBuffers& bufs,
        const std::vector<int64_t>& shape,
        ThreadPool& pool, int n_threads) {
    const int ndim = static_cast<int>(shape.size());
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    bufs.resize(total);
    int32_t* h_lbl = bufs.lbl();
    int32_t* h_dist = bufs.dist();
    int32_t* t_lbl = bufs.lbl_T();
    int32_t* t_dist = bufs.dist_T();

    std::memcpy(h_lbl, input, total * sizeof(int32_t));
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
        ThreadPool& pool, int n_threads,
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

    std::memcpy(h_lbl, input, total * sizeof(int32_t));
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
