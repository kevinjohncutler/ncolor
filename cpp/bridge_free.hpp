// Bridge-free Voronoi expansion.
//
// Adds an antipodal-bridge test to the L2 separable expand_labels: after
// each EDT axis sweep, pixels that pass the local antipodal-only test are
// marked as "barriers" (lbl=0, dist=BRIDGE_BARRIER_DIST) and the next
// axis sweep refuses to refill them. A 1-pixel-wide string of pixels
// connected by opposite faces or corners (the bridge topology) is what
// the test catches: exactly 2 same-label neighbors in one of the four
// antipodal pairings {N,S}, {E,W}, {NE,SW}, {NW,SE} in 2D.
//
// Why not just despur after expand_labels:
//   The standard despur uses face_count <= threshold, which misses
//   2-wide bridges (face_count = 2-3 per pixel). The antipodal test
//   specifically identifies the bridge topology — pixels in a thin
//   string between two regions. AND by marking them as sticky barriers
//   between axes, the next axis can't refill them.
//
// Why "after each axis" rather than "during":
//   The bridge test needs an 8-neighborhood (2D) — most of those reads
//   are cross-scanline. Doing the test concurrently with the EDT sweep
//   requires either double-buffering or careful read/write ordering of
//   the in-place labels array. The "after each axis" version does one
//   extra parallel scan per axis with all reads happening after the axis
//   has fully committed. The cost is small relative to EDT itself.

#ifndef NCOLOR_BRIDGE_FREE_HPP
#define NCOLOR_BRIDGE_FREE_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace bridge_free_detail {

// Antipodal-only same-label test for 2D 8-neighborhood. Returns true iff
// labels[y*W + x] is non-zero AND has EXACTLY 2 same-label neighbors
// forming one of the antipodal pairs {N,S}, {E,W}, {NE,SW}, {NW,SE}.
//
// Image-boundary neighbors count as "not same label" — a 1-wide strip
// along the image edge is still considered a bridge (the missing side
// is treated as different territory).
template <typename T>
inline bool is_antipodal_bridge_2d(
    const T* __restrict labels,
    int64_t y, int64_t x,
    int64_t H, int64_t W)
{
    const T A = labels[y * W + x];
    if (A == 0) return false;

    auto same = [&](int64_t dy, int64_t dx) -> int {
        const int64_t ny = y + dy;
        const int64_t nx = x + dx;
        if (ny < 0 || ny >= H || nx < 0 || nx >= W) return 0;
        return labels[ny * W + nx] == A ? 1 : 0;
    };

    const int N  = same(-1,  0), S  = same( 1,  0);
    const int E  = same( 0,  1), Ww = same( 0, -1);
    const int NE = same(-1,  1), SW = same( 1, -1);
    const int NW = same(-1, -1), SE = same( 1,  1);

    const int total = N + S + E + Ww + NE + SW + NW + SE;
    if (total != 2) return false;

    // Antipodal pair patterns. With total==2 we just check that one of
    // the pairs is on and all other six neighbors are off.
    if (N  && S  && !(E || Ww || NE || SW || NW || SE)) return true;
    if (E  && Ww && !(N || S  || NE || SW || NW || SE)) return true;
    if (NE && SW && !(N || S  || E  || Ww || NW || SE)) return true;
    if (NW && SE && !(N || S  || E  || Ww || NE || SW)) return true;
    return false;
}

// Subspace antipodal test. Given an ND label image and an ordered list
// of `k = subset_axes.size()` axes (the swept subspace), check whether
// the pixel at flat index `i` is a bridge in this subspace: exactly 2
// same-label neighbors, both in one of the (3^k - 1)/2 antipodal pairs.
//
// `flat_disps` lists the 3^k - 1 flat-index displacements (in stride
// space) into the subspace; `pair_a_pair_b` gives the (a, b) indices
// (into flat_disps) of each antipodal pair, so pair (a, b) means
// flat_disps[a] and flat_disps[b] are antipodes. `oob_mask[d]` is true
// iff the dth displacement would land out of bounds at this pixel
// (caller pre-checks).
//
// Returns true iff antipodal-only same-label condition holds.
template <typename T>
inline bool is_antipodal_bridge_subspace(
    const T* __restrict labels,
    int64_t i,
    const int64_t* flat_disps,
    int n_disps,
    const int* pair_idx,    // length n_disps; pair_idx[a] = b for partner; -1 if no partner (impossible for full subspace)
    const uint8_t* in_bounds)  // length n_disps; nonzero iff displacement d is valid for pixel i
{
    const T A = labels[i];
    if (A == 0) return false;
    // Count same-label and find which two displacements (if exactly 2) match.
    int match_a = -1, match_b = -1;
    int count = 0;
    for (int d = 0; d < n_disps; ++d) {
        if (!in_bounds[d]) continue;
        if (labels[i + flat_disps[d]] == A) {
            if (count == 0) match_a = d;
            else if (count == 1) match_b = d;
            else { return false; }   // > 2 same-label
            ++count;
        }
    }
    if (count != 2) return false;
    // Are match_a and match_b antipodal? i.e. pair_idx[match_a] == match_b.
    return pair_idx[match_a] == match_b;
}

}  // namespace bridge_free_detail


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
    for (int i = 0; i < k; ++i) n_disps_total *= 3;
    n_disps_total -= 1;

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

    // Build pair_idx: pair_idx[a] = b such that coord_offsets[b] == -coord_offsets[a].
    const int n = (int)t.flat_disps.size();
    t.pair_idx.assign(n, -1);
    for (int a = 0; a < n; ++a) {
        if (t.pair_idx[a] != -1) continue;
        for (int b = a + 1; b < n; ++b) {
            if (t.pair_idx[b] != -1) continue;
            bool is_antipode = true;
            for (int d = 0; d < k; ++d) {
                if (t.coord_offsets[a][d] != -t.coord_offsets[b][d]) {
                    is_antipode = false; break;
                }
            }
            if (is_antipode) {
                t.pair_idx[a] = b;
                t.pair_idx[b] = a;
                break;
            }
        }
    }

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
inline int64_t bridge_check_subspace_nd(
    int32_t* labels, int32_t* dist,
    const std::vector<int64_t>& shape,
    const std::vector<int>& subset_axes,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
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

    // Phase 1: single-pass scan with saturation.
    //   • count same-label subspace neighbors, early-exit at count > 2
    //   • for count <= 2 track match_a/match_b (antipodal check input)
    //   • store SATURATED for count > 2 (lazy recount in peel-back);
    //     exact count otherwise
    //   • bad pixels (stub or antipodal-2) appended to per-thread queue
    //
    // Interior pixels go through a tight nested-loop iterator with no
    // per-pixel coord update or bounds check. Boundary pixels use the
    // generic scan with bounds checks (small fraction).
    constexpr uint8_t SATURATED = 255;
    std::vector<uint8_t> nbr_count((size_t)total, 0);

    using QEnt = std::pair<int64_t, int32_t>;
    std::vector<QEnt> queue;

    auto scan_kernel = [&](int64_t i_lo, int64_t i_hi,
                            std::vector<QEnt>& out) {
        std::vector<int64_t> coords_all(N);
        std::vector<int> coords_sub(k, 0);

        int64_t rem = i_lo;
        for (int d = 0; d < N; ++d) {
            coords_all[d] = rem / strides[d];
            rem -= coords_all[d] * strides[d];
        }
        for (int d = 0; d < k; ++d) coords_sub[d] = (int)coords_all[subset_axes[d]];

        for (int64_t i = i_lo; i < i_hi; ++i) {
            const int32_t A = labels[i];
            if (A != 0) {
                bool interior = true;
                for (int j = 0; j < k; ++j) {
                    const int c = coords_sub[j];
                    if (c == 0 || c == (int)shape_sub[j] - 1) {
                        interior = false; break;
                    }
                }

                int face = 0, total_cnt = 0;
                int match_a = -1, match_b = -1;
                bool face_saturated = false;

                // Faces first — early-exit on face > 2 fires after 3
                // reads for interior pixels of uniform regions. Then
                // corners/edges for the antipodal-2 bridge total.
                if (interior) {
                    for (int ff = 0; ff < n_face_disps; ++ff) {
                        const int d = face_d_idx[ff];
                        if (labels[i + flat_disps[d]] == A) {
                            if (total_cnt == 0) match_a = d;
                            else if (total_cnt == 1) match_b = d;
                            ++total_cnt; ++face;
                            if (face > 2) { face_saturated = true; break; }
                        }
                    }
                    // Skip corners entirely when face <= 1 — the pixel
                    // is a stub regardless of corner contribution, so
                    // we already know the queue decision and total_cnt
                    // isn't needed for the antipodal check. This is the
                    // common case for stub/sliver pixels and trims the
                    // corner loop overhead. When face == 2, we still
                    // need corners to verify the antipodal-2 case (and
                    // can early-exit once total_cnt > 2).
                    if (!face_saturated && face >= 2) {
                        for (int cc = 0; cc < n_corner_disps; ++cc) {
                            const int d = corner_d_idx[cc];
                            if (labels[i + flat_disps[d]] == A) {
                                if (total_cnt == 0) match_a = d;
                                else if (total_cnt == 1) match_b = d;
                                ++total_cnt;
                                if (total_cnt > 2) { face_saturated = true; break; }
                            }
                        }
                    }
                } else {
                    for (int ff = 0; ff < n_face_disps; ++ff) {
                        const int d = face_d_idx[ff];
                        const int* co_d = co_flat.data() + d * k;
                        bool ok = true;
                        for (int j = 0; j < k; ++j) {
                            const int c = coords_sub[j] + co_d[j];
                            if (c < 0 || c >= (int)shape_sub[j]) { ok = false; break; }
                        }
                        if (!ok) continue;
                        if (labels[i + flat_disps[d]] == A) {
                            if (total_cnt == 0) match_a = d;
                            else if (total_cnt == 1) match_b = d;
                            ++total_cnt; ++face;
                            if (face > 2) { face_saturated = true; break; }
                        }
                    }
                    if (!face_saturated && face >= 2) {
                        for (int cc = 0; cc < n_corner_disps; ++cc) {
                            const int d = corner_d_idx[cc];
                            const int* co_d = co_flat.data() + d * k;
                            bool ok = true;
                            for (int j = 0; j < k; ++j) {
                                const int c = coords_sub[j] + co_d[j];
                                if (c < 0 || c >= (int)shape_sub[j]) { ok = false; break; }
                            }
                            if (!ok) continue;
                            if (labels[i + flat_disps[d]] == A) {
                                if (total_cnt == 0) match_a = d;
                                else if (total_cnt == 1) match_b = d;
                                ++total_cnt;
                                if (total_cnt > 2) { face_saturated = true; break; }
                            }
                        }
                    }
                }

                if (face_saturated) {
                    nbr_count[(size_t)i] = SATURATED;
                } else {
                    nbr_count[(size_t)i] = (uint8_t)face;
                    if (face <= 1
                            || (total_cnt == 2
                                && pair_idx[match_a] == match_b)) {
                        out.emplace_back(i, A);
                    }
                }
            }
            int d = N - 1;
            ++coords_all[d];
            while (d > 0 && coords_all[d] >= shape[d]) {
                coords_all[d] = 0; --d; ++coords_all[d];
            }
            for (int j = 0; j < k; ++j) coords_sub[j] = (int)coords_all[subset_axes[j]];
        }
    };

    // Phase-level timing gated on NCOLOR_BRIDGE_PROFILE env var. When
    // the env var is unset, BFDEBUG is false and the chrono::now() calls
    // are still executed (~50 ns each, negligible); the fprintf below
    // is the only branch with non-trivial cost. Useful to keep so a
    // future investigation can re-enable with no code change.
    static const bool BFDEBUG = std::getenv("NCOLOR_BRIDGE_PROFILE") != nullptr;
    auto t_p1_start = std::chrono::steady_clock::now();

    if (nt > 1 && total >= 1024) {
        const int64_t outer = shape[0];
        const int64_t slab = total / outer;
        std::vector<std::vector<QEnt>> per_thread(nt);
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_slab{0};
        const int64_t chunk_slabs = std::max<int64_t>(1, outer / (nt * 4));
        pool->parallel([&]() {
            int my_tid = tid_counter.fetch_add(1);
            if (my_tid >= nt) return;
            auto& local = per_thread[my_tid];
            while (true) {
                int64_t s_lo = next_slab.fetch_add(chunk_slabs);
                if (s_lo >= outer) break;
                int64_t s_hi = std::min(outer, s_lo + chunk_slabs);
                scan_kernel(s_lo * slab, s_hi * slab, local);
            }
        });
        size_t sz = 0;
        for (auto& v : per_thread) sz += v.size();
        queue.reserve(sz);
        for (auto& v : per_thread) {
            queue.insert(queue.end(), v.begin(), v.end());
        }
    } else {
        scan_kernel(0, total, queue);
    }

    auto t_p1_end = std::chrono::steady_clock::now();
    const size_t initial_queue_size = queue.size();
    if (BFDEBUG) {
        auto ms = std::chrono::duration<double, std::milli>(t_p1_end - t_p1_start).count();
        std::fprintf(stderr, "[bridge_check] Phase 1 (scan): %.3f ms, queue=%zu, total_px=%lld\n",
                      ms, initial_queue_size, (long long)total);
    }

    if (queue.empty()) return 0;

    // Apply initial removals.
    int64_t removed = 0;
    for (auto& [i, _lab] : queue) {
        if (labels[i] != 0) {
            labels[i] = 0;
            dist[i] = BRIDGE_BARRIER_DIST;
            ++removed;
        }
    }

    // Phase 2: queue-based peel-back. Decrement face_count of each
    // same-label subspace FACE neighbor of a removed pixel (2*k face
    // dirs, not the full 3^k-1 — keeps cascade local along face
    // adjacency, matching despur semantics). Saturated entries get
    // lazy-recounted (face count, exact) on first decrement.
    std::vector<int64_t> coords_all(N);
    std::vector<int> coords_sub(k);
    std::vector<int64_t> jcoords_all(N);
    std::vector<int> jcoords_sub(k);

    auto recount_face_exact = [&](int64_t j_idx, int32_t lab) -> int {
        int64_t r = j_idx;
        for (int d = 0; d < N; ++d) {
            jcoords_all[d] = r / strides[d];
            r -= jcoords_all[d] * strides[d];
        }
        for (int d = 0; d < k; ++d) jcoords_sub[d] = (int)jcoords_all[subset_axes[d]];
        int cnt = 0;
        for (int f = 0; f < n_face_disps; ++f) {
            bool ok = true;
            for (int jx = 0; jx < k; ++jx) {
                const int c = jcoords_sub[jx] + face_coord_offsets[f][jx];
                if (c < 0 || c >= (int)shape_sub[jx]) { ok = false; break; }
            }
            if (ok && labels[j_idx + face_flat_disps[f]] == lab) ++cnt;
        }
        return cnt;
    };

    size_t head = 0;
    while (head < queue.size()) {
        const int64_t i = queue[head].first;
        const int32_t old_lab = queue[head].second;
        ++head;

        int64_t rem = i;
        for (int d = 0; d < N; ++d) {
            coords_all[d] = rem / strides[d];
            rem -= coords_all[d] * strides[d];
        }
        for (int d = 0; d < k; ++d) coords_sub[d] = (int)coords_all[subset_axes[d]];

        for (int f = 0; f < n_face_disps; ++f) {
            bool ok = true;
            for (int j = 0; j < k; ++j) {
                const int c = coords_sub[j] + face_coord_offsets[f][j];
                if (c < 0 || c >= (int)shape_sub[j]) { ok = false; break; }
            }
            if (!ok) continue;
            const int64_t j_idx = i + face_flat_disps[f];
            if (labels[j_idx] != old_lab) continue;
            uint8_t fc = nbr_count[(size_t)j_idx];
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
                labels[j_idx] = 0;
                dist[j_idx] = BRIDGE_BARRIER_DIST;
                queue.emplace_back(j_idx, old_lab);
                ++removed;
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


// Bridge detection scan on a 2D label image. Two-phase: read-only scan
// collects bridge indices, then a commit phase writes barrier sentinels.
// The two-phase split prevents a bridge-detection race where an early
// pixel's barrier write would suppress a later pixel's detection.
//
// Returns the number of pixels marked as barriers.
template <typename T>
inline int64_t bridge_check_2d(
    T* labels, int32_t* dist,
    int64_t H, int64_t W,
    ForkJoinPool* pool = nullptr, int n_threads = 1)
{
    const int64_t total = H * W;
    if (total == 0) return 0;

    const int nt = (pool && n_threads > 1) ? n_threads : 1;

    std::vector<int64_t> bridges;

    auto scan_chunk = [&](int64_t y_lo, int64_t y_hi,
                           std::vector<int64_t>& out) {
        for (int64_t y = y_lo; y < y_hi; ++y) {
            for (int64_t x = 0; x < W; ++x) {
                if (bridge_free_detail::is_antipodal_bridge_2d(
                        labels, y, x, H, W)) {
                    out.push_back(y * W + x);
                }
            }
        }
    };

    if (nt > 1 && H >= (int64_t)(nt * 4)) {
        std::vector<std::vector<int64_t>> per_thread(nt);
        std::atomic<int> tid_counter{0};
        std::atomic<int64_t> next_row{0};
        const int64_t chunk = std::max<int64_t>(1, H / (nt * 4));
        pool->parallel([&]() {
            int my_tid = tid_counter.fetch_add(1);
            if (my_tid >= nt) return;
            auto& local = per_thread[my_tid];
            while (true) {
                int64_t y_lo = next_row.fetch_add(chunk);
                if (y_lo >= H) break;
                int64_t y_hi = std::min(H, y_lo + chunk);
                scan_chunk(y_lo, y_hi, local);
            }
        });
        size_t sz = 0;
        for (auto& v : per_thread) sz += v.size();
        bridges.reserve(sz);
        for (auto& v : per_thread) {
            bridges.insert(bridges.end(), v.begin(), v.end());
        }
    } else {
        scan_chunk(0, H, bridges);
    }

    for (int64_t i : bridges) {
        labels[i] = 0;
        dist[i] = BRIDGE_BARRIER_DIST;
    }
    return (int64_t)bridges.size();
}


// Barrier-respecting envelope_pass_row. Identical to envelope_pass_row_impl
// (in expand.hpp), but the Phase 2 segment fill reads dist[i] before
// writing and skips pixels where dist[i] == BRIDGE_BARRIER_DIST. Scalar
// fill only — SIMD masked stores aren't worth the complexity for this
// (rare) path; non-bridge-free expand still uses the SIMD fill via the
// original envelope_pass_row.
//
// Phase 1 already handles barriers implicitly: a barrier pixel has
// lbl=0 so it doesn't get pushed as a seed.
template <bool Contig>
inline void envelope_pass_row_barrier_impl(
        int32_t* __restrict lbl, int32_t* __restrict dist,
        int64_t N, int64_t stride,
        int32_t* __restrict v, int32_t* __restrict lblstk,
        int32_t* __restrict g, double* __restrict z,
        double* __restrict vd, double* __restrict vd_sq) {
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

    auto load_lbl = [&](int64_t idx) -> int32_t {
        if constexpr (Contig) return lbl[idx]; else return lbl[idx * stride];
    };
    auto load_dist = [&](int64_t idx) -> int32_t {
        if constexpr (Contig) return dist[idx]; else return dist[idx * stride];
    };

    for (int64_t i = 0; i < N; ++i) {
        const int32_t lv = load_lbl(i);
        if (lv == 0) continue;
        push_seed(i, lv, load_dist(i));
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
            i_end = (zj1 >= static_cast<double>(N))
                ? N : static_cast<int64_t>(std::ceil(zj1));
            if (i_end > N) i_end = N;
        }
        if (i_end <= i_start) continue;
        const int32_t lbl_j = lblstk[j];
        const int32_t g_j = g[j];
        const int32_t v_j = v[j];
        for (int64_t i = i_start; i < i_end; ++i) {
            const int64_t idx = Contig ? i : i * stride;
            if (dist[idx] == BRIDGE_BARRIER_DIST) continue;
            const int32_t di = static_cast<int32_t>(i) - v_j;
            lbl[idx] = lbl_j;
            dist[idx] = g_j + di * di;
        }
        i_start = i_end;
    }
}

inline void envelope_pass_row_barrier(
        int32_t* lbl, int32_t* dist, int64_t N, int64_t stride,
        int32_t* v, int32_t* lblstk, int32_t* g, double* z,
        double* vd, double* vd_sq) {
    if (stride == 1) {
        envelope_pass_row_barrier_impl<true>(
            lbl, dist, N, 1, v, lblstk, g, z, vd, vd_sq);
    } else {
        envelope_pass_row_barrier_impl<false>(
            lbl, dist, N, stride, v, lblstk, g, z, vd, vd_sq);
    }
}

// Parallel driver. Identical structure to envelope_pass in expand.hpp but
// calls the barrier-aware row kernel.
inline void envelope_pass_barrier(
        int32_t* h_lbl, int32_t* h_dist,
        int64_t n_slices, int64_t N,
        ForkJoinPool& pool, int n_threads,
        std::vector<EnvelopeScratch>& scratch) {
    if (n_threads < 1) n_threads = 1;
    const int eff_threads = static_cast<int>(compute_threads(
        static_cast<size_t>(n_threads),
        static_cast<size_t>(n_slices),
        static_cast<size_t>(N)));
    if (static_cast<int>(scratch.size()) < eff_threads) {
        scratch.resize(eff_threads);
    }
    const size_t cap = static_cast<size_t>(N) + 1;
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
                    l, d, N, /*stride=*/1, vp, lp, gp, zp, vdp, vdsqp);
            }
        });
}


// Top-level 2D entry. Replicates expand_labels_inplace for 2D but
// inserts a bridge_check after the final axis. Output written to
// bufs.lbl().
//
// Sweep order (matches expand_labels_inplace): axis 1 (innermost) first
// via pass0, then axis 0 (outermost) via transpose + envelope_pass_barrier
// + transpose-back. After the final axis, bridge_check_2d marks barriers
// and zeroes labels on bridge pixels.
//
// For 2D, there's only one meaningful bridge-check point: after the
// final axis, when labels are full 2D-Voronoi. Applying the test after
// axis 1 (1D-row-Voronoi state) would flag whole stripes spuriously,
// because every interior pixel in a wide horizontal segment has
// {E,W} same-label by construction. The barrier-aware envelope_pass is
// still used for axis 0 — it's a no-op when no barriers exist, but
// keeps the code path uniform with the ND case (where intermediate
// barriers do exist).
inline void expand_labels_bridge_free_2d_inplace(
    const int32_t* input, ExpandBuffers& bufs,
    int64_t H, int64_t W,
    ForkJoinPool& pool, int n_threads)
{
    const int64_t total = H * W;
    bufs.resize(total);
    int32_t* h_lbl  = bufs.lbl();
    int32_t* h_dist = bufs.dist();
    int32_t* t_lbl  = bufs.lbl_T();
    int32_t* t_dist = bufs.dist_T();

    if (input != h_lbl) {
        std::memcpy(h_lbl, input, total * sizeof(int32_t));
    }

    // Axis 1 (innermost): standard pass0. No barriers exist yet so we
    // can use the fast path; it writes dist for every pixel along the
    // row, so no leftover BARRIER sentinels survive into axis 0.
    {
        const int64_t n_slices = H;
        const int64_t Nlen = W;
        envelope_pass0(h_lbl, h_dist, n_slices, Nlen,
                       pool, n_threads, bufs.scratch());
    }

    // Axis 0 (outermost): transpose to make it contiguous, then
    // barrier-aware envelope_pass, then transpose back. For 2D no
    // barriers have been marked yet, so this behaves like the standard
    // sweep; the path is kept barrier-aware so the same machinery
    // generalizes to ND (where intermediate axes would have barriers).
    const int64_t A = 1, B = H, C = W;
    batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist,
                              A, B, C, pool, n_threads);
    envelope_pass_barrier(t_lbl, t_dist, A * C, B,
                          pool, n_threads, bufs.scratch());
    batch_transpose<int32_t>(t_lbl, t_dist, h_lbl, h_dist,
                              A, C, B, pool, n_threads);

    // Final bridge check: scan the 2D-Voronoi result and mark antipodal
    // bridges as barriers. Caller reads result from bufs.lbl(); barrier
    // pixels are labels[i] = 0 with dist[i] = BRIDGE_BARRIER_DIST.
    bridge_check_2d(h_lbl, h_dist, H, W, &pool, n_threads);
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
                                bool barriers_present)
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
                chamfer_l1_row_init(lbl + off, dist + off, B, /*wrap=*/false);
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
                                              B, C, c0, c1, /*wrap=*/false);
            } else {
                chamfer_l1_slab_pass(lbl + off, dist + off,
                                      B, C, c0, c1, /*wrap=*/false);
            }
        }
    });
}


// Per-axis L2 sweep driver, barrier-aware. Mirrors the inner loop body
// of expand_labels_inplace for one axis, with the bridge_free_detail
// barrier-aware envelope_pass when needed.
inline void l2_sweep_axis_barrier(int32_t* h_lbl, int32_t* h_dist,
                                   int32_t* t_lbl, int32_t* t_dist,
                                   const std::vector<int64_t>& shape,
                                   int ax,
                                   ForkJoinPool& pool, int n_threads,
                                   std::vector<EnvelopeScratch>& scratch,
                                   bool barriers_present)
{
    const int ndim = (int)shape.size();
    const int64_t n = shape[ax];

    if (ax == ndim - 1) {
        const int64_t total = [&](){ int64_t t = 1; for (auto s : shape) t *= s; return t; }();
        const int64_t n_slices = total / n;
        // No barriers possible yet (first axis); pass0 is safe.
        envelope_pass0(h_lbl, h_dist, n_slices, n, pool, n_threads, scratch);
        return;
    }
    int64_t A = 1;
    for (int d = 0; d < ax; ++d) A *= shape[d];
    int64_t C = 1;
    for (int d = ax + 1; d < ndim; ++d) C *= shape[d];
    const int64_t B = n;

    // For ND > 2: prefer strided slab sweep when it's available; for the
    // bridge-free path we route everything through transpose for now
    // (barrier-aware strided not yet implemented). For ND = 2, strided
    // doesn't apply (A=1) so we use transpose.
    batch_transpose<int32_t>(h_lbl, h_dist, t_lbl, t_dist, A, B, C, pool, n_threads);
    if (barriers_present) {
        envelope_pass_barrier(t_lbl, t_dist, A * C, B, pool, n_threads, scratch);
    } else {
        envelope_pass(t_lbl, t_dist, A * C, B, pool, n_threads, scratch, /*wrap=*/false);
    }
    batch_transpose<int32_t>(t_lbl, t_dist, h_lbl, h_dist, A, C, B, pool, n_threads);
}


// Unified ND bridge-free expansion. Loops axes from innermost (ndim-1)
// down to outermost (0); after each axis k < ndim-1 (skipping innermost
// where the antipodal test would false-positive on 1D-stripe state),
// runs the subspace antipodal bridge_check with axes {k, k+1, ..., ndim-1}
// and writes barrier sentinels for refused pixels. Subsequent axis
// sweeps respect barriers (skip writes, refuse to propagate from them).
//
// p: 1 = L1 (Saito-Toriwaki), 2 = L2 (Felzenszwalb). Default 2.
inline void expand_labels_bridge_free_nd_inplace(
    const int32_t* input, ExpandBuffers& bufs,
    const std::vector<int64_t>& shape,
    ForkJoinPool& pool, int n_threads, int p = 2)
{
    const int ndim = (int)shape.size();
    int64_t total = 1;
    for (auto s : shape) total *= s;
    bufs.resize(total);
    int32_t* h_lbl  = bufs.lbl();
    int32_t* h_dist = bufs.dist();
    int32_t* t_lbl  = bufs.lbl_T();
    int32_t* t_dist = bufs.dist_T();

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
                                pool, n_threads, barriers_present);
        } else {
            l2_sweep_axis_barrier(h_lbl, h_dist, t_lbl, t_dist, shape, ax,
                                   pool, n_threads, bufs.scratch(),
                                   barriers_present);
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
            int64_t n_new = bridge_check_subspace_nd(
                h_lbl, h_dist, shape, subset_axes, &pool, n_threads);
            if (n_new > 0) barriers_present = true;
        }
    }
}


// L1 (Manhattan) variant. Runs the standard Saito-Toriwaki sweep
// (chamfer_st_l1_nd) to full expansion, then applies the antipodal
// bridge check on the final 2D-Voronoi labels. For 2D no intermediate
// per-axis check is meaningful (axis-1 produces 1D-row-Voronoi which
// false-positives the antipodal test), so a post-pass is equivalent
// to per-axis with sticky barriers.
//
// L1 produces materially more antipodal bridges than L2 (Manhattan
// boundaries are axis-aligned and create 1-wide diagonal strings where
// two cells meet at a corner). This is the metric where the bridge
// test actually changes the output visibly.
inline void expand_labels_bridge_free_l1_2d_inplace(
    const int32_t* input, ExpandBuffers& bufs,
    int64_t H, int64_t W,
    ForkJoinPool& pool, int n_threads)
{
    const int64_t total = H * W;
    bufs.resize(total);
    int32_t* h_lbl  = bufs.lbl();
    int32_t* h_dist = bufs.dist();

    if (input != h_lbl) {
        std::memcpy(h_lbl, input, total * sizeof(int32_t));
    }

    std::vector<int64_t> shape = {H, W};
    chamfer_st_l1_nd(h_lbl, h_dist, shape, pool, n_threads, /*wrap=*/false);
    bridge_check_2d(h_lbl, h_dist, H, W, &pool, n_threads);
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
inline void expand_labels_bridge_free_inplace(
    const int32_t* input, ExpandBuffers& bufs,
    const std::vector<int64_t>& shape,
    ForkJoinPool& pool, int n_threads, int p)
{
    expand_labels_bridge_free_nd_inplace(
        input, bufs, shape, pool, n_threads, p);
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_BRIDGE_FREE_HPP
