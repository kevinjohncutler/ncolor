// N-D Lp Voronoi expansion with sticky bridge barriers. After each
// swept subspace of at least two active axes, classify thin bridges and
// stubs, then peel back their face-neighbor tails. Later sweeps preserve
// removed pixels as background. Both sweeps and cleanup can be periodic.

#ifndef NCOLOR_EXPAND_CLEAN_HPP
#define NCOLOR_EXPAND_CLEAN_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "chamfer.hpp"
#include "dispatch.hpp"
#include "expand.hpp"
#include "fast_despur.hpp"
#include "threadpool.h"

namespace ncolor_cpp {

// Sentinel value in dist[] indicating a barrier pixel (refused bridge).
// INT32_MIN is impossible for a normal squared L2 distance (always >= 0).
constexpr int32_t BRIDGE_BARRIER_DIST = INT32_MIN;

// ND helper: compute the 3^k - 1 displacement offsets in the subspace
// defined by `subset_axes` (size k), and the antipodal-partner index
// for each offset.
struct SubspaceAntipodalTable {
    std::vector<std::vector<int>> coord_offsets;  // n_disps × k; coord per subspace axis
    std::vector<int64_t> flat_disps;              // n_disps; flat-index displacement
    std::vector<int> pair_idx;                    // n_disps; partner index in arrays
    std::vector<uint8_t> is_face;                 // n_disps; 1 iff displacement is a face (exactly one nonzero coord)
};

inline SubspaceAntipodalTable build_subspace_antipodal_table(
    const std::vector<int64_t>& strides,
    const std::vector<int>& subset_axes)
{
    const int k = (int)subset_axes.size();
    // Total number of base-3 vectors of length k, excluding the all-zero one.
    int n_disps_total = 1;
    for (int i = 0; i < k; ++i) {
        if (n_disps_total > INT_MAX / 3)
            throw std::overflow_error("clean neighborhood exceeds int32 offset capacity");
        n_disps_total *= 3;
    }
    n_disps_total -= 1;
    if (k > 0 && n_disps_total > INT_MAX / k)
        throw std::overflow_error("clean neighborhood exceeds int32 offset capacity");

    SubspaceAntipodalTable t;
    t.coord_offsets.reserve(n_disps_total);
    t.flat_disps.reserve(n_disps_total);

    // Enumerate base-3 vectors (digits 0,1,2 mapped to -1,0,+1).
    std::vector<int> v(k, 0);
    for (int rep = 0; rep <= n_disps_total; ++rep) {
        // Skip the all-zero vector
        bool all_zero = true;
        for (int d = 0; d < k; ++d) if (v[d] != 1) { all_zero = false; break; }
        if (!all_zero) {
            std::vector<int> co(k);
            int64_t fd = 0;
            for (int d = 0; d < k; ++d) {
                co[d] = v[d] - 1;  // map 0,1,2 -> -1,0,+1
                fd += (int64_t)co[d] * strides[subset_axes[d]];
            }
            t.coord_offsets.push_back(std::move(co));
            t.flat_disps.push_back(fd);
        }
        // Increment base-3 counter
        for (int d = 0; d < k; ++d) {
            if (v[d] < 2) { ++v[d]; break; }
            v[d] = 0;
        }
    }

    // Negating a base-3 vector reverses its enumeration index, also
    // after removing the central zero vector.
    const int n = (int)t.flat_disps.size();
    t.pair_idx.resize(n);
    for (int a = 0; a < n; ++a) t.pair_idx[a] = n - 1 - a;

    // is_face[d] = 1 iff displacement d has exactly one nonzero coord
    // (a cardinal-direction face neighbor within the subspace).
    t.is_face.assign(n, 0);
    for (int d = 0; d < n; ++d) {
        int n_nonzero = 0;
        for (int j = 0; j < k; ++j) {
            if (t.coord_offsets[d][j] != 0) ++n_nonzero;
        }
        t.is_face[d] = (n_nonzero == 1) ? 1 : 0;
    }
    return t;
}


// ND subspace bridge check + stub peel-back. Fused scan:
//
//   Phase 1 (parallel): compute face_count[i] (count of same-label
//     SUBSPACE FACE neighbors — the 2*k cardinal directions; matches
//     despur convention) AND track antipodal status using the full
//     subspace neighborhood. A pixel is bad if either
//       (a) face_count <= 1   — stub or isolated. Catches single-pixel
//           face-stubs (face=1) and corner-only-connected pixels
//           (face=0). Matches despur threshold=1 semantics. Peel-back
//           decrement propagates through face neighbors only, so K_4
//           diagonal-adjacency pixels aren't collateral-damaged by
//           cascading face-stub removal.
//       (b) total_count == 2 in antipodal arrangement — 1-wide bridge
//           interior (faces OR corners), 1-pixel-thin string.
//
//   Phase 2 (serial queue): peel back. For each removed pixel, decrement
//     face_count of same-label SUBSPACE FACE neighbors (2*k of them).
//     If any drops to ≤1, mark and enqueue. Cascades face-stubs in
//     O(removed_pixels × 2k) work without rescanning. Equivalent to
//     despur_via_face_count_nd's cascade behavior.
//
// This subsumes a separate despur post-pass: bridge endpoints (which
// become face_count=1 after the bridge interior is removed) get picked
// up by the peel-back automatically, and any cascading thin tails get
// peeled away too. One fused scan + cascade replaces a chain of
// (bridge_check, compute_face_count, despur) calls.
template <typename Distance>
inline int64_t bridge_check_subspace_nd(
    int32_t* labels, Distance* dist,
    const std::vector<int64_t>& shape,
    const std::vector<int>& subset_axes,
    ForkJoinPool* pool = nullptr, int n_threads = 1,
    std::vector<uint8_t>* nbr_scratch = nullptr, bool wrap = false)
{
    const int N = (int)shape.size();
    const int k = (int)subset_axes.size();
    if (k < 2) return 0;   // 1D antipodal test false-positives, skip.

    // Strides (row-major, C order).
    std::vector<int64_t> strides(N);
    strides[N - 1] = 1;
    for (int d = N - 2; d >= 0; --d) strides[d] = strides[d + 1] * shape[d + 1];

    auto table = build_subspace_antipodal_table(strides, subset_axes);
    const int n_disps = (int)table.flat_disps.size();
    const int64_t* flat_disps = table.flat_disps.data();
    const int* pair_idx = table.pair_idx.data();
    const uint8_t* is_face_arr = table.is_face.data();

    // Precompute face vs corner displacement indices. The scan loop
    // iterates faces first (early-exit on face > 2 fires after 3 reads
    // for interior pixels of uniform cells), then corners for the
    // antipodal-2 bridge total. Peel-back also iterates face indices
    // only.
    std::vector<int> face_d_idx;
    std::vector<int> corner_d_idx;
    std::vector<int64_t> face_flat_disps;
    std::vector<std::vector<int>> face_coord_offsets;
    for (int d = 0; d < n_disps; ++d) {
        if (is_face_arr[d]) {
            face_d_idx.push_back(d);
            face_flat_disps.push_back(flat_disps[d]);
            face_coord_offsets.push_back(table.coord_offsets[d]);
        } else {
            corner_d_idx.push_back(d);
        }
    }
    const int n_face_disps   = (int)face_d_idx.size();
    const int n_corner_disps = (int)corner_d_idx.size();

    int64_t total = 1;
    for (auto s : shape) total *= s;
    if (total == 0) return 0;

    const int nt = (pool && n_threads > 1) ? n_threads : 1;

    std::vector<int64_t> shape_sub(k);
    for (int d = 0; d < k; ++d) shape_sub[d] = shape[subset_axes[d]];

    // Flatten coord_offsets[d][j] into a single n_disps × k contiguous
    // array for the boundary inner loop and peel-back bounds checks.
    std::vector<int> co_flat(n_disps * k);
    for (int d = 0; d < n_disps; ++d) {
        for (int j = 0; j < k; ++j) co_flat[d * k + j] = table.coord_offsets[d][j];
    }

    // Phase 1: single-pass scan with saturation, one line at a time.
    //
    // The subspace is always the trailing axes, so its last axis is the
    // innermost (unit-stride) axis of the image and every chunk is a
    // whole number of lines along it. Along a line only the innermost
    // coordinate changes, so which displacements stay in bounds is fixed
    // for the whole line except at its two end pixels. The line therefore
    // splits into an interior run, where no pixel needs a bounds check,
    // and at most two end pixels handled by the generic path. Over the
    // interior run the same-label face count is accumulated one
    // displacement at a time in a straight loop that vectorizes; the
    // per-pixel decision then re-reads neighbors only for pixels with
    // face count <= 2 (cell boundaries), the rare case. Nothing here
    // depends on k or N: validity comes from the offset vectors.
    //
    //   • face >= 3   -> SATURATED (lazy exact recount in peel-back)
    //   • face <= 1   -> stub, queued
    //   • face == 2   -> SATURATED if any same-label corner neighbor,
    //                    else queued iff the two face matches are
    //                    antipodal
    constexpr uint8_t SATURATED = 255;
    constexpr uint8_t QUEUED = 254;
    // One byte per pixel. Every labeled pixel is written by the scan
    // before the peel-back reads it, and background pixels are never
    // read, so the buffer needs no zeroing and can persist across calls
    // (a fresh 16 MB allocation plus memset per call at 256 cubed).
    std::vector<uint8_t> nbr_local;
    std::vector<uint8_t>& nbr_vec = nbr_scratch ? *nbr_scratch : nbr_local;
    if (nbr_vec.size() < (size_t)total) nbr_vec.resize((size_t)total);
    uint8_t* const nbr_count = nbr_vec.data();

    using QEnt = std::pair<int64_t, int32_t>;
    std::vector<QEnt> queue;

    const int64_t W = shape[N - 1];            // innermost axis == subset axis k-1
    const int64_t n_lines = total / W;
    constexpr int64_t RUN_BLOCK = 1024;        // keeps cnt + neighbor lines in L1

    // Generic classification of one pixel given the displacements valid
    // for it (indices into flat_disps; faces then corners). Same
    // decision as the interior path, with the original early exits.
    auto classify = [&](int64_t i, int32_t A,
                        const int* fd, int nf, const int* cd, int nc,
                        const int64_t* disps, std::vector<QEnt>& out) {
        int face = 0, match_a = -1, match_b = -1;
        for (int ff = 0; ff < nf; ++ff) {
            const int d = fd[ff];
            if (labels[i + disps[d]] == A) {
                if (face == 0) match_a = d;
                else if (face == 1) match_b = d;
                if (++face > 2) { nbr_count[(size_t)i] = SATURATED; return; }
            }
        }
        if (face == 2) {
            for (int cc = 0; cc < nc; ++cc) {
                if (labels[i + disps[cd[cc]]] == A) {
                    nbr_count[(size_t)i] = SATURATED;
                    return;
                }
            }
            nbr_count[(size_t)i] = 2;
            if (pair_idx[match_a] == match_b) out.emplace_back(i, A);
            return;
        }
        nbr_count[(size_t)i] = (uint8_t)face;
        out.emplace_back(i, A);
    };

    auto scan_lines = [&](int64_t line_lo, int64_t line_hi,
                          std::vector<QEnt>& out, uint8_t* __restrict cnt) {
        // Coordinates of the current line along axes 0..N-2 (mixed radix,
        // axis N-2 fastest), advanced incrementally per line.
        std::vector<int64_t> lc(N - 1, 0);
        {
            int64_t r = line_lo;
            for (int d = N - 2; d >= 0; --d) { lc[d] = r % shape[d]; r /= shape[d]; }
        }
        std::vector<int> fd_line, cd_line, fd_end, cd_end;
        std::vector<int64_t> periodic_disps(wrap ? n_disps : 0);
        std::vector<int64_t> end_disps(wrap ? n_disps : 0);
        fd_line.reserve(n_face_disps); fd_end.reserve(n_face_disps);
        cd_line.reserve(n_corner_disps); cd_end.reserve(n_corner_disps);

        for (int64_t line = line_lo; line < line_hi; ++line) {
            const int64_t base = line * W;
            const int64_t* line_disps = flat_disps;
            if (wrap) {
                for (int d = 0; d < n_disps; ++d) {
                    int64_t off = flat_disps[d];
                    for (int j = 0; j < k - 1; ++j) {
                        const int ax = subset_axes[j];
                        const int64_t c = lc[ax] + co_flat[d * k + j];
                        if (c < 0) off += shape[ax] * strides[ax];
                        else if (c >= shape[ax]) off -= shape[ax] * strides[ax];
                    }
                    periodic_disps[d] = off;
                }
                line_disps = periodic_disps.data();
            }

            // Displacements whose non-innermost offsets stay inside the
            // subspace on this line.
            auto line_valid = [&](int d) {
                if (wrap) return true;
                const int* co = co_flat.data() + d * k;
                for (int j = 0; j < k - 1; ++j) {
                    const int64_t c = lc[subset_axes[j]];
                    if ((co[j] < 0 && c == 0) ||
                        (co[j] > 0 && c == shape_sub[j] - 1)) return false;
                }
                return true;
            };
            fd_line.clear(); cd_line.clear();
            for (int ff = 0; ff < n_face_disps; ++ff) {
                if (line_valid(face_d_idx[ff])) fd_line.push_back(face_d_idx[ff]);
            }
            for (int cc = 0; cc < n_corner_disps; ++cc) {
                if (line_valid(corner_d_idx[cc])) cd_line.push_back(corner_d_idx[cc]);
            }

            // An end pixel: the innermost offset must stay in range too.
            auto end_pixel = [&](int64_t x) {
                const int64_t i = base + x;
                const int32_t A = labels[i];
                if (A == 0) return;
                auto in_range = [&](int d) {
                    if (wrap) return true;
                    const int o = co_flat[d * k + (k - 1)];
                    return !((o < 0 && x == 0) || (o > 0 && x == W - 1));
                };
                fd_end.clear(); cd_end.clear();
                for (int d : fd_line) if (in_range(d)) fd_end.push_back(d);
                for (int d : cd_line) if (in_range(d)) cd_end.push_back(d);
                const int64_t* disps = line_disps;
                if (wrap) {
                    for (int d = 0; d < n_disps; ++d) {
                        const int64_t nx = x + co_flat[d * k + k - 1];
                        end_disps[d] = line_disps[d] +
                            (nx < 0 ? W : (nx >= W ? -W : 0));
                    }
                    disps = end_disps.data();
                }
                classify(i, A, fd_end.data(), (int)fd_end.size(),
                         cd_end.data(), (int)cd_end.size(), disps, out);
            };

            if (W <= 2) {
                for (int64_t x = 0; x < W; ++x) end_pixel(x);
            } else {
                end_pixel(0);
                for (int64_t x0 = 1; x0 < W - 1; x0 += RUN_BLOCK) {
                    const int64_t x1 = std::min<int64_t>(W - 1, x0 + RUN_BLOCK);
                    const int64_t n = x1 - x0;
                    const int32_t* cur = labels + base + x0;
                    std::memset(cnt, 0, (size_t)n);
                    // Face count, one displacement per pass: a compare
                    // and a byte add per pixel, no branches.
                    for (int d : fd_line) {
                        const int32_t* nb = cur + line_disps[d];
                        for (int64_t x = 0; x < n; ++x) {
                            cnt[x] += (uint8_t)(nb[x] == cur[x]);
                        }
                    }
                    for (int64_t x = 0; x < n; ++x) {
                        const int32_t A = cur[x];
                        if (A == 0) continue;
                        const int64_t i = base + x0 + x;
                        const uint8_t c = cnt[x];
                        if (c >= 3) { nbr_count[(size_t)i] = SATURATED; continue; }
                        if (c <= 1) {
                            nbr_count[(size_t)i] = c;
                            out.emplace_back(i, A);
                            continue;
                        }
                        // c == 2: a corner match saturates; otherwise the
                        // two face matches decide by antipodality.
                        bool sat = false;
                        for (int d : cd_line) {
                            if (labels[i + line_disps[d]] == A) { sat = true; break; }
                        }
                        if (sat) { nbr_count[(size_t)i] = SATURATED; continue; }
                        nbr_count[(size_t)i] = 2;
                        int a = -1, b = -1;
                        for (int d : fd_line) {
                            if (labels[i + line_disps[d]] == A) {
                                if (a < 0) a = d; else { b = d; break; }
                            }
                        }
                        if (pair_idx[a] == b) out.emplace_back(i, A);
                    }
                }
                end_pixel(W - 1);
            }

            for (int d = N - 2; d >= 0; --d) {
                if (++lc[d] < shape[d]) break;
                lc[d] = 0;
            }
        }
    };

    // Phase-level timing gated on NCOLOR_BRIDGE_PROFILE env var. When
    // the env var is unset, BFDEBUG is false and the chrono::now() calls
    // are still executed (~50 ns each, negligible); the fprintf below
    // is the only branch with non-trivial cost.
    static const bool BFDEBUG = std::getenv("NCOLOR_BRIDGE_PROFILE") != nullptr;
    auto t_p1_start = std::chrono::steady_clock::now();

    const int64_t cnt_len = std::min<int64_t>(W, RUN_BLOCK);
    if (nt > 1 && total >= 1024) {
        std::vector<std::vector<QEnt>> per_thread(nt);
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_line{0};
        // Fine chunks so cores of unequal speed (the performance /
        // efficiency mix on Apple Silicon) balance instead of the slowest
        // one holding the barrier; the fetch_add per chunk is negligible.
        const int64_t chunk = std::max<int64_t>(1, n_lines / (nt * 16));
        pool->parallel([&]() {
            const int my_tid = tid_counter.fetch_add(1);
            if (my_tid >= nt) return;
            std::vector<uint8_t> cnt((size_t)cnt_len);
            auto& local = per_thread[my_tid];
            for (;;) {
                const int64_t lo = next_line.fetch_add(chunk);
                if (lo >= n_lines) break;
                scan_lines(lo, std::min(n_lines, lo + chunk), local, cnt.data());
            }
        });
        size_t sz = 0;
        for (auto& v : per_thread) sz += v.size();
        queue.reserve(sz);
        for (auto& v : per_thread) {
            queue.insert(queue.end(), v.begin(), v.end());
        }
        // Threads claim chunks in whatever order they wake, so the merged
        // queue order varies run to run. Keep its traversal deterministic
        // across thread counts; the queue is small.
        std::sort(queue.begin(), queue.end());
    } else {
        std::vector<uint8_t> cnt((size_t)cnt_len);
        scan_lines(0, n_lines, queue, cnt.data());
    }

    auto t_p1_end = std::chrono::steady_clock::now();
    const size_t initial_queue_size = queue.size();
    if (BFDEBUG) {
        auto ms = std::chrono::duration<double, std::milli>(t_p1_end - t_p1_start).count();
        std::fprintf(stderr, "[bridge_check] Phase 1 (scan): %.3f ms, queue=%zu, total_px=%lld\n",
                      ms, initial_queue_size, (long long)total);
    }

    if (queue.empty()) return 0;

    // Mark the queue, but remove each pixel only when it is popped.
    // A lazy recount then includes every not-yet-processed neighbor.
    // Clearing the whole queue up front would make recounts subtract
    // future removals twice (once here, again when their queue entry pops).
    int64_t removed = 0;
    for (auto& [i, _lab] : queue) nbr_count[(size_t)i] = QUEUED;

    // Phase 2: queue-based peel-back. Decrement face_count of each
    // same-label subspace FACE neighbor of a removed pixel (2*k face
    // dirs, not the full 3^k-1 — keeps cascade local along face
    // adjacency, matching despur semantics). Saturated entries get
    // lazy-recounted (face count, exact) on first decrement.
    std::vector<int64_t> coords_all(N);
    std::vector<int> coords_sub(k);
    std::vector<int64_t> jcoords_all(N);
    std::vector<int> jcoords_sub(k);

    auto face_neighbor = [&](int64_t i, const std::vector<int>& c, int f) {
        int64_t off = face_flat_disps[f];
        for (int j = 0; j < k; ++j) {
            const int64_t nc = static_cast<int64_t>(c[j]) + face_coord_offsets[f][j];
            if (nc < 0 || nc >= shape_sub[j]) {
                if (!wrap) return int64_t{-1};
                off += (nc < 0 ? shape_sub[j] : -shape_sub[j]) * strides[subset_axes[j]];
            }
        }
        return i + off;
    };

    auto recount_face_exact = [&](int64_t j_idx, int32_t lab) -> int {
        int64_t r = j_idx;
        for (int d = 0; d < N; ++d) {
            jcoords_all[d] = r / strides[d];
            r -= jcoords_all[d] * strides[d];
        }
        for (int d = 0; d < k; ++d) jcoords_sub[d] = (int)jcoords_all[subset_axes[d]];
        int cnt = 0;
        for (int f = 0; f < n_face_disps; ++f) {
            const int64_t nb = face_neighbor(j_idx, jcoords_sub, f);
            if (nb >= 0 && labels[nb] == lab) ++cnt;
        }
        return cnt;
    };

    size_t head = 0;
    while (head < queue.size()) {
        const int64_t i = queue[head].first;
        const int32_t old_lab = queue[head].second;
        ++head;
        labels[i] = 0;
        dist[i] = BRIDGE_BARRIER_DIST;
        ++removed;

        int64_t rem = i;
        for (int d = 0; d < N; ++d) {
            coords_all[d] = rem / strides[d];
            rem -= coords_all[d] * strides[d];
        }
        for (int d = 0; d < k; ++d) coords_sub[d] = (int)coords_all[subset_axes[d]];

        for (int f = 0; f < n_face_disps; ++f) {
            const int64_t j_idx = face_neighbor(i, coords_sub, f);
            if (j_idx < 0) continue;
            if (labels[j_idx] != old_lab) continue;
            uint8_t fc = nbr_count[(size_t)j_idx];
            if (fc == QUEUED) continue;
            if (fc == SATURATED) {
                // First touch of a saturated entry: recount gives the
                // current face_count (already reflects i's removal,
                // since labels[i] was set to 0 before this loop).
                // Don't decrement again — that'd double-count.
                fc = (uint8_t)recount_face_exact(j_idx, old_lab);
            } else if (fc > 0) {
                // Stored value is exact face_count of j prior to this
                // removal. Subtract 1 for i's removal.
                fc = (uint8_t)(fc - 1);
            }
            nbr_count[(size_t)j_idx] = fc;
            if (fc <= 1) {
                nbr_count[(size_t)j_idx] = QUEUED;
                queue.emplace_back(j_idx, old_lab);
            }
        }
    }

    if (BFDEBUG) {
        auto t_end = std::chrono::steady_clock::now();
        auto ms_p2 = std::chrono::duration<double, std::milli>(
            t_end - t_p1_end).count();
        std::fprintf(stderr, "[bridge_check] Phase 2 (peel-back): %.3f ms, "
                              "cascaded_to=%zu, total_removed=%lld\n",
                      ms_p2, queue.size(), (long long)removed);
    }

    return removed;
}


// Use the same parabolic envelope for periodic and bounded expansion;
// only the fill skips sticky barriers. Ghost seeds never include barriers
// because those pixels have label zero.
inline void envelope_pass_row_barrier(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq, bool wrap = false) {
    if (wrap) {
        if (stride == 1) envelope_pass_row_impl<true, true, true>(
            lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
        else envelope_pass_row_impl<true, false, true>(
            lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
    } else {
        if (stride == 1) envelope_pass_row_impl<false, true, true>(
            lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
        else envelope_pass_row_impl<false, false, true>(
            lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
    }
}

// Parallel driver. Identical structure to envelope_pass in expand.hpp but
// calls the barrier-aware row kernel.
inline void envelope_pass_barrier(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_slices, int64_t N,
        ForkJoinPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch, bool wrap = false) {
    if (n_threads < 1) n_threads = 1;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(n_slices),
        static_cast<size_t>(N)));
    if (static_cast<int>(scratch.size()) < eff_threads) {
        scratch.resize(eff_threads);
    }
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
                envelope_pass_row_barrier(
                    l, d, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp, wrap);
            }
        });
}


// Barrier-aware single-axis L1 chamfer slab pass. Identical to
// chamfer_l1_slab_pass in chamfer.hpp but skips writing to barrier
// pixels (`dist[i] == BRIDGE_BARRIER_DIST`) and refuses to propagate
// from barrier pixels (so barriers behave like infinite-distance
// blockers from the perspective of downstream relax).
inline void chamfer_l1_slab_pass_barrier(int32_t* __restrict lbl,
                                          int32_t* __restrict dist,
                                          int64_t B, int64_t C,
                                          int64_t c0, int64_t c1,
                                          bool wrap = false) {
    auto relax_axis = [&](int64_t b_dst, int64_t b_src) {
        int32_t* lr = lbl  + b_dst * C;
        int32_t* dr = dist + b_dst * C;
        const int32_t* lo = lbl  + b_src * C;
        const int32_t* dn = dist + b_src * C;
        for (int64_t c = c0; c < c1; ++c) {
            if (dr[c] == BRIDGE_BARRIER_DIST) continue;  // dest is barrier
            if (dn[c] == BRIDGE_BARRIER_DIST) continue;  // source is barrier
            const int32_t cd = dn[c] + 1;
            if (cd < dr[c]) { dr[c] = cd; lr[c] = lo[c]; }
        }
    };
    auto forward_sweep  = [&]() { for (int64_t b = 1;     b < B; ++b)   relax_axis(b, b - 1); };
    auto backward_sweep = [&]() { for (int64_t b = B - 2; b >= 0; --b)  relax_axis(b, b + 1); };

    forward_sweep();
    backward_sweep();

    if (wrap && B > 1) {
        relax_axis(0,     B - 1);
        forward_sweep();
        relax_axis(B - 1, 0);
        backward_sweep();
    }
}


// Per-axis L1 sweep driver, barrier-aware. Mirrors the structure of
// chamfer_st_l1_nd's outer loop but exposes the axis loop here so we
// can interleave bridge_check between axes.
inline void chamfer_st_l1_axis(int32_t* lbl, int32_t* dist,
                                const std::vector<int64_t>& shape,
                                int ax,
                                ForkJoinPool& pool, int n_threads,
                                bool barriers_present, bool wrap = false,
                                   bool keep_distances = true)
{
    const int ndim = (int)shape.size();
    constexpr int64_t MIN_BAND_W = 256;

    int64_t A = 1, C = 1;
    for (int d = 0; d < ax; ++d)        A *= shape[d];
    for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
    const int64_t B = shape[ax];

    if (ax == ndim - 1) {
        // Innermost: per-row 1D sweep. The original chamfer_l1_row_init
        // fuses the dist-init pass with the forward sweep — since this
        // is the FIRST axis processed, there are no barriers to respect.
        // Barrier-aware variant is unnecessary here.
        const size_t row_threads = compute_threads(
            (size_t)n_threads, (size_t)A, (size_t)B);
        dispatch_parallel(pool, (size_t)A,
                          row_threads * DISPATCH_CHUNKS_PER_THREAD,
                          [&](size_t a0, size_t a1) {
            for (size_t a = a0; a < a1; ++a) {
                const int64_t off = (int64_t)a * B;
                chamfer_l1_row_init(lbl + off, dist + off, B, wrap);
            }
        });
        return;
    }

    // Non-innermost axis: slab pass. Use barrier-aware variant if any
    // barriers exist (set by a prior bridge_check), else standard.
    const int64_t target_chunks =
        (int64_t)n_threads * (int64_t)DISPATCH_CHUNKS_PER_THREAD;
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
        (size_t)n_threads, (size_t)total_chunks, (size_t)B);

    dispatch_parallel(pool, (size_t)total_chunks,
                      threads_for * DISPATCH_CHUNKS_PER_THREAD,
                      [&](size_t i0, size_t i1) {
        for (size_t i = i0; i < i1; ++i) {
            const int64_t a   = (int64_t)i / n_bands;
            const int64_t bnd = (int64_t)i % n_bands;
            const int64_t c0  = bnd * band_w;
            const int64_t c1  = std::min(C, c0 + band_w);
            if (c0 >= c1) continue;
            const int64_t off = a * B * C;
            if (barriers_present) {
                chamfer_l1_slab_pass_barrier(lbl + off, dist + off,
                                              B, C, c0, c1, wrap);
            } else {
                chamfer_l1_slab_pass(lbl + off, dist + off,
                                      B, C, c0, c1, wrap);
            }
        }
    });
}


// Strided barrier-aware sweep of axis B in an (A, B, C) layout: the
// barrier-aware row kernel walking each line in place with stride C, so
// the two full-array transposes are skipped. Same selection rule as the
// plain expand (see STRIDED_SLAB_LIMIT).
inline void envelope_pass_strided_abc_barrier(
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
                envelope_pass_row_barrier(h_lbl + base, h_dist + base, B,
                                          /*stride=*/C, vp, lp, gp, zp, vdp, vdsqp, wrap);
            }
        });
}


// Per-axis L2 sweep driver, barrier-aware. Mirrors the inner loop body
// of expand_labels_inplace for one axis, with the expand_clean_detail
// barrier-aware envelope_pass when needed.
inline void l2_sweep_axis_barrier(int32_t* h_lbl, int32_t* h_dist,
                                   ExpandBuffers& bufs,
                                   const std::vector<int64_t>& shape,
                                   int ax,
                                   ForkJoinPool& pool, int n_threads,
                                   std::vector<EnvelopeScratch>& scratch,
                                   bool barriers_present, bool wrap = false,
                                   bool keep_distances = true, bool retain_layout = false)
{
    const int ndim = (int)shape.size();
    const int64_t n = shape[ax];

    if (ax == ndim - 1) {
        const int64_t total = [&](){ int64_t t = 1; for (auto s : shape) t *= s; return t; }();
        const int64_t n_slices = total / n;
        // No barriers possible yet (first axis); pass0 is safe.
        if (wrap) {
            constexpr int32_t INF = std::numeric_limits<int32_t>::max() / 4;
            for (int64_t i = 0; i < total; ++i) h_dist[i] = h_lbl[i] ? 0 : INF;
            envelope_pass(h_lbl, h_dist, n_slices, n, pool, n_threads, scratch, true);
        } else {
            envelope_pass0(h_lbl, h_dist, n_slices, n, pool, n_threads, scratch);
        }
        return;
    }
    int64_t A = 1;
    for (int d = 0; d < ax; ++d) A *= shape[d];
    int64_t C = 1;
    for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
    const int64_t B = n;

    // Same strided-versus-transpose rule as the plain expand: strided
    // needs an outer slab axis (A >= 2, never true in 2D) and a slab
    // that fits in cache. Before any barrier exists the plain kernels
    // apply unchanged.
    const bool use_strided = (A >= 2) && (B * C <= STRIDED_SLAB_LIMIT);
    if (use_strided) {
        if (barriers_present) {
            envelope_pass_strided_abc_barrier(h_lbl, h_dist, A, B, C,
                                              pool, n_threads, scratch, wrap);
        } else {
            envelope_pass_strided_abc(h_lbl, h_dist, A, B, C,
                                      pool, n_threads, scratch, wrap);
        }
        return;
    }
    int32_t* t_lbl = bufs.lbl_T();
    int32_t* t_dist = bufs.dist_T();
    batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist, A, B, C, pool, n_threads);
    if (barriers_present) {
        envelope_pass_barrier(t_lbl, t_dist, A * C, B, pool, n_threads, scratch, wrap);
    } else {
        envelope_pass(t_lbl, t_dist, A * C, B, pool, n_threads, scratch, wrap);
    }
    if (!retain_layout) batch_transpose<int32_t>(t_lbl, t_dist, h_lbl, h_dist, A, C, B, pool, n_threads,
                             keep_distances || ax != 0);
}


// Unified ND bridge-free expansion. Loops axes from innermost (ndim-1)
// down to outermost (0); after each axis k < ndim-1 (skipping innermost
// where the antipodal test would false-positive on 1D-stripe state),
// runs the subspace antipodal bridge_check with axes {k, k+1, ..., ndim-1}
// and writes barrier sentinels for refused pixels. Subsequent axis
// sweeps respect barriers (skip writes, refuse to propagate from them).
//
// p: 1 = L1 (Saito-Toriwaki), 2 = L2 (Felzenszwalb). Default 2.
// keep_distances=false skips the last distance transpose where possible.
// Final cleanup reads labels and only writes barrier marks into distances;
// it does not need the final distances when there is no subsequent sweep.
// final_shape requests the same retained-layout contract as expand.hpp for
// narrow L2 transforms. Final cleanup runs in that layout before returning.
inline void expand_labels_clean_nd_inplace(
    const int32_t* input, ExpandBuffers& bufs,
    const std::vector<int64_t>& input_shape,
    ForkJoinPool& pool, int n_threads, int p = 2, bool wrap = false,
    bool keep_distances = true, std::vector<int64_t>* final_shape = nullptr)
{
    std::vector<int64_t> shape;
    for (int64_t n : input_shape) if (n != 1) shape.push_back(n);
    if (shape.empty()) shape.push_back(1);
    if (final_shape) *final_shape = shape;
    const int ndim = (int)shape.size();
    int64_t total = 1;
    for (auto s : shape) total *= s;
    bufs.resize(total);
    if (total == 0) return;
    int32_t* h_lbl  = bufs.lbl();
    int32_t* h_dist = bufs.dist();
    const bool wide = p == 2 && l2_needs_wide_distance(shape);
    if (wide) bufs.use_wide_distance();

    if (input != h_lbl) {
        std::memcpy(h_lbl, input, total * sizeof(int32_t));
    }

    // For L1, init the dist buffer (chamfer_l1_row_init does its own
    // init on the innermost axis, so we don't need to here; subsequent
    // axes inherit dist from prior sweeps). For L2 ditto (pass0 init).
    // BUT if the innermost has been replaced with bridge-aware, dist
    // is still uninitialised — handled by the pass0 / chamfer_l1_row_init
    // first call below.

    bool barriers_present = false;

    for (int ax = ndim - 1; ax >= 0; --ax) {
        if (p == 1) {
            chamfer_st_l1_axis(h_lbl, h_dist, shape, ax,
                                pool, n_threads, barriers_present, wrap);
        } else if (wide) {
            l2_sweep_axis_wide(h_lbl, bufs.dist64(), shape, ax, pool, n_threads, wrap);
        } else {
            l2_sweep_axis_barrier(h_lbl, h_dist, bufs, shape, ax,
                                   pool, n_threads, bufs.scratch(),
                                   barriers_present, wrap, keep_distances,
                                   final_shape && ax == 0 && ndim > 1);
            if (final_shape && ax == 0 && ndim > 1) {
                h_lbl = bufs.lbl_T();
                h_dist = bufs.dist_T();
                std::rotate(shape.begin(), shape.begin() + 1, shape.end());
                *final_shape = shape;
            }
        }
        // After this axis, the swept subspace is {ax, ax+1, ..., ndim-1}.
        // Skip the innermost (subspace size 1 false-positives); for any
        // larger subspace the single ND bridge_check_subspace_nd handles
        // 2D, 3D, and higher uniformly. Internal queue-based peel-back
        // cascades stubs in one call — no outer iteration needed.
        const int subspace_size = ndim - ax;
        if (subspace_size >= 2) {
            std::vector<int> subset_axes(subspace_size);
            for (int j = 0; j < subspace_size; ++j) subset_axes[j] = ax + j;
            const int64_t n_new = wide
                ? bridge_check_subspace_nd(h_lbl, bufs.dist64(), shape, subset_axes,
                    &pool, n_threads, &bufs.nbr_scratch(), wrap)
                : bridge_check_subspace_nd(h_lbl, h_dist, shape, subset_axes,
                    &pool, n_threads, &bufs.nbr_scratch(), wrap);
            if (n_new > 0) barriers_present = true;
        }
    }
}


// ND entry. Per-axis bridge prevention with sticky barriers.
//
// Algorithm:
//   for ax = ndim-1 down to 0:
//     run axis-ax sweep (barrier-aware if barriers exist)
//     if (ndim - ax) >= 2:
//       run bridge_check_subspace_nd on axes {ax, ..., ndim-1}
//       (the swept subspace, which is geometrically complete at this
//       point — antipodal bridges in this subspace are real)
//
// Skipping the innermost axis (subspace size 1) avoids 1D-stripe
// false-positives. Subsequent axes respect the barriers.
//
// p: integer Lp norm. 1 = L1 (Saito-Toriwaki), 2 = L2 (Felzenszwalb).
inline void expand_labels_clean_inplace(
    const int32_t* input, ExpandBuffers& bufs,
    const std::vector<int64_t>& shape,
    ForkJoinPool& pool, int n_threads, int p, bool wrap = false,
    bool keep_distances = true, std::vector<int64_t>* final_shape = nullptr)
{
    expand_labels_clean_nd_inplace(
        input, bufs, shape, pool, n_threads, p, wrap, keep_distances, final_shape);
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_EXPAND_CLEAN_HPP
