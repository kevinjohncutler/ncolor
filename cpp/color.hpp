/*
 * color.hpp — graph-coloring + CSR-build kernels.
 *
 * Pure scalar code; together with connect.hpp + expand.hpp this lets
 * Solver run end-to-end. The BFS coloring converges on the first attempt
 * for the planar adjacency graphs we hit in practice; ``repair_coloring``
 * provides the Kempe-chain fallback for the rare cases where it doesn't
 * (typically L2 + wrap on tightly-cropped images).
 */

#ifndef NCOLOR_COLOR_HPP
#define NCOLOR_COLOR_HPP

#include <cstdint>
#include <cstring>
#include <vector>

namespace ncolor_cpp {

// Build symmetric CSR from M directed pairs (src[i], dst[i]). Output:
// indptr[N+1] and indices[2*M]. Neighbors within each row are unordered.
inline void build_csr_from_pairs(
        const int32_t* src, const int32_t* dst, int32_t N, int32_t M,
        std::vector<int32_t>& indptr, std::vector<int32_t>& indices) {
    indptr.assign(static_cast<size_t>(N) + 1, 0);
    std::vector<int32_t> degree(static_cast<size_t>(N), 0);
    for (int32_t i = 0; i < M; ++i) {
        degree[src[i]] += 1;
        degree[dst[i]] += 1;
    }
    indptr[0] = 0;
    for (int32_t i = 0; i < N; ++i) indptr[i + 1] = indptr[i] + degree[i];

    indices.assign(static_cast<size_t>(2) * M, 0);
    std::vector<int32_t> write(indptr.begin(), indptr.begin() + N);  // copy 1st N entries
    for (int32_t i = 0; i < M; ++i) {
        const int32_t s = src[i];
        const int32_t d = dst[i];
        indices[write[s]++] = d;
        indices[write[d]++] = s;
    }
}

// Weighted variant: also produces a parallel weights[] array of length 2*M
// (symmetric: weight(s, d) = weight(d, s) = w[i]). Used by the
// boundary-weighted coloring path; the un-weighted variant above stays
// bit-identical for the default path. Weights are real-valued to encode
// EDT-distance-based contact strength.
inline void build_csr_from_pairs_weighted(
        const int32_t* src, const int32_t* dst, const double* w,
        int32_t N, int32_t M,
        std::vector<int32_t>& indptr, std::vector<int32_t>& indices,
        std::vector<double>& weights) {
    indptr.assign(static_cast<size_t>(N) + 1, 0);
    std::vector<int32_t> degree(static_cast<size_t>(N), 0);
    for (int32_t i = 0; i < M; ++i) {
        degree[src[i]] += 1;
        degree[dst[i]] += 1;
    }
    indptr[0] = 0;
    for (int32_t i = 0; i < N; ++i) indptr[i + 1] = indptr[i] + degree[i];
    indices.assign(static_cast<size_t>(2) * M, 0);
    weights.assign(static_cast<size_t>(2) * M, 0.0);
    std::vector<int32_t> write(indptr.begin(), indptr.begin() + N);
    for (int32_t i = 0; i < M; ++i) {
        const int32_t s = src[i];
        const int32_t d = dst[i];
        const double wi = w[i];
        indices[write[s]] = d; weights[write[s]] = wi; ++write[s];
        indices[write[d]] = s; weights[write[d]] = wi; ++write[d];
    }
}

// BFS-based legacy coloring. Returns true if all nodes were assigned a
// non-zero color before max_iter; false if the queue still had pending
// nodes (caller should retry with bigger n or repair).
//
// `rand`: every `rand`-th re-visit of a node, pick a deterministic
// pseudo-random color to break out of cycles. Match the numba LCG seed
// exactly so results are bit-identical.
// `weights`: optional parallel array to `indices` with per-edge boundary
//            weight (e.g. shared-pixel-pair count). When non-null AND
//            de_table non-null AND welsh_powell, the WP color-pick uses a
//            weighted-ΔE objective instead of the balanced-greedy rule.
// `de_table`: optional (n_colors+1) × (n_colors+1) row-major palette
//             distance matrix. de_table[c1*(n_colors+1) + c2] is the
//             ΔE between color c1 and c2 (any units; viridis-LAB by
//             default). Indices 0..n_colors valid; row/col 0 (bg) ignored.
// `weight_obj`: 0 = balance (current default), +1 = maximize weighted ΔE
//               (sharp contrast on heavy edges), -1 = minimize weighted ΔE
//               (soft contrast). Ignored when weights or de_table is null.
inline bool color_graph_csr_legacy(
        const int32_t* indptr, const int32_t* indices, int32_t N,
        int32_t n_colors, int32_t rand_period, int32_t offset, int64_t max_iter,
        std::vector<uint8_t>& colors, bool welsh_powell = false,
        const double* weights = nullptr,
        const double* de_table = nullptr,
        int weight_obj = 0) {
    const bool use_weighted = welsh_powell && weights != nullptr &&
                              de_table != nullptr && weight_obj != 0;
    const int32_t de_stride = n_colors + 1;
    // Soft balance pull: only active on LOW-DEGREE nodes (degree ≤ 2),
    // i.e. chains and leaves. On those the local score has just 1–2
    // neighbors of information and greedy max-ΔE otherwise locks onto
    // the pairwise max-contrast pair forever (1D chain 2-cycles).
    // Higher-degree nodes (grid interior with deg 8, generic 2D cells
    // with deg 3+) already have enough multi-edge consensus for the
    // natural max-contrast tiling to win — balance is left off so it
    // can't disturb the periodic pattern. Pull is scaled by local edge
    // weight, so it stays calibrated to the local score magnitude
    // regardless of weight_mode (harmonic / mean_inv / count / …).
    // (An earlier anisotropy-based gate `(top2 weight)/(total) > 0.6`
    // — meant to catch chains embedded in denser graphs like spoke
    // patterns — was tried and reverted. It improved per-spoke color
    // count on the 8-spoke synthetic but destroyed the mirror symmetry
    // across the x-axis that the degree-gate output naturally has,
    // which is the more important structural property to preserve.)
    double balance_coeff = 0.0;
    if (use_weighted) {
        double max_de = 0.0;
        for (int32_t c1 = 1; c1 <= n_colors; ++c1) {
            for (int32_t c2 = 1; c2 <= n_colors; ++c2) {
                if (de_table[c1 * de_stride + c2] > max_de)
                    max_de = de_table[c1 * de_stride + c2];
            }
        }
        balance_coeff = 0.4 * max_de;
    }
    colors.assign(static_cast<size_t>(N), 0);
    std::vector<int32_t> counter(static_cast<size_t>(N), 0);
    // Global slot population — balanced greedy picks the least-populated
    // valid color instead of the lowest-numbered one. Maintained
    // incrementally on every recolor.
    std::vector<int32_t> color_counts(static_cast<size_t>(n_colors) + 1, 0);

    // Queue capacity heuristic mirrors the numba version; grows on demand.
    int64_t qcap = std::max<int64_t>({static_cast<int64_t>(indptr[N]) + N,
                                      static_cast<int64_t>(N) * 8, 1});
    std::vector<int32_t> q(static_cast<size_t>(qcap));
    int64_t head = 0;
    int64_t tail = N;
    if (welsh_powell) {
        // Visit nodes in DESCENDING degree order. High-degree nodes are
        // most constrained (most neighbors competing for distinct
        // colors), so coloring them first reduces forced retries and
        // tends to spread color usage more evenly across the graph.
        // Bucket sort: O(N + max_degree). For our planar / near-planar
        // adjacency graphs, max_degree is small (typically ≤ 12).
        int32_t max_deg = 0;
        std::vector<int32_t> deg(static_cast<size_t>(N));
        for (int32_t i = 0; i < N; ++i) {
            deg[i] = indptr[i + 1] - indptr[i];
            if (deg[i] > max_deg) max_deg = deg[i];
        }
        std::vector<int32_t> bucket_head(static_cast<size_t>(max_deg) + 2, 0);
        for (int32_t i = 0; i < N; ++i) bucket_head[deg[i] + 1]++;
        // Convert to write-index per bucket, ordered high → low.
        std::vector<int32_t> wpos(static_cast<size_t>(max_deg) + 1, 0);
        // Compute starting position of each bucket (degree d) in the
        // descending-by-degree output. Bucket for degree max_deg starts
        // at 0; bucket for degree max_deg-1 starts after that; ...
        int32_t cum = 0;
        for (int32_t d = max_deg; d >= 0; --d) {
            wpos[d] = cum;
            cum += bucket_head[d + 1];
        }
        for (int32_t i = 0; i < N; ++i) {
            q[wpos[deg[i]]++] = i;
        }
        // Shuffle WITHIN each degree bucket — same-degree vertices were
        // in natural ID order, which makes parallel attempts with
        // different `offset` values produce highly correlated visit
        // orders (degree-ties broken identically). Shuffling
        // diversifies attempts genuinely: different `offset` ⇒
        // different bucket permutations ⇒ different greedy outcome on
        // adversarial graphs (heuristic 4-coloring can be order-fragile).
        // Fisher-Yates per bucket, seeded from `offset + 1` so even
        // offset=0 produces a non-identity shuffle (the natural-order
        // result is what slot 0 used to give; diversifying it too
        // increases the parallel-attempts' independent success
        // probability).
        {
            int32_t bucket_start = 0;
            const uint64_t seed_base = ((uint64_t)offset + 1)
                * 0x9e3779b97f4a7c15ULL;
            for (int32_t d = max_deg; d >= 0; --d) {
                const int32_t bucket_size = bucket_head[d + 1];
                if (bucket_size > 1) {
                    uint64_t rs = seed_base
                        ^ ((uint64_t)d * 0xc6a4a7935bd1e995ULL);
                    for (int32_t i = bucket_size - 1; i > 0; --i) {
                        rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
                        const int32_t j = (int32_t)((rs >> 32) % (uint32_t)(i + 1));
                        std::swap(q[bucket_start + i], q[bucket_start + j]);
                    }
                }
                bucket_start += bucket_size;
            }
        }
    } else {
        // BFS path: diversify the natural-ID-order queue too. Each
        // attempt's `offset` gives a different starting permutation.
        for (int32_t i = 0; i < N; ++i) q[i] = i;
        uint64_t rs = ((uint64_t)offset + 1) * 0x9e3779b97f4a7c15ULL;
        for (int32_t i = N - 1; i > 0; --i) {
            rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
            const int32_t j = (int32_t)((rs >> 32) % (uint32_t)(i + 1));
            std::swap(q[i], q[j]);
        }
    }

    const uint32_t fullmask = (1u << (n_colors + 1)) - 2;
    int64_t count = 0;
    while (head < tail && count < max_iter) {
        const int32_t u = q[head++];
        ++count;
        ++counter[u];

        const int32_t row_beg = indptr[u];
        const int32_t row_end = indptr[u + 1];
        uint32_t mask = 0;
        bool all_present = false;
        for (int32_t k = row_beg; k < row_end; ++k) {
            const uint8_t cv = colors[indices[k]];
            if (cv != 0) {
                mask |= (1u << cv);
                if (mask == fullmask) { all_present = true; break; }
            }
        }

        uint8_t csel = 0;
        if (!all_present) {
            if (use_weighted) {
                // Score = Σ_v weight_v·ΔE(c, color_v)  − pull·count[c]
                // (sign flipped for weight_obj < 0). pull is non-zero
                // ONLY on chain/leaf nodes (degree ≤ 2) — picks where
                // greedy max-contrast otherwise locks onto a 2-cycle.
                // pull = balance_coeff · w_local; the local-weight
                // factor keeps balance calibrated to the local score
                // gap regardless of weight_mode.
                const int32_t deg = row_end - row_beg;
                int32_t colored_count = 0;
                double w_local = 0.0;
                for (int32_t k = row_beg; k < row_end; ++k) {
                    if (colors[indices[k]] != 0) {
                        ++colored_count;
                        if (colored_count == 1) w_local = weights[k];
                        if (colored_count >= 2) break;
                    }
                }
                const double pull = (deg <= 2 && colored_count == 1)
                    ? balance_coeff * w_local : 0.0;
                double best_score = (weight_obj > 0) ? -1e30 : 1e30;
                for (int32_t c = 1; c <= n_colors; ++c) {
                    if ((mask & (1u << c)) != 0) continue;
                    double score = 0.0;
                    for (int32_t k = row_beg; k < row_end; ++k) {
                        const uint8_t cv = colors[indices[k]];
                        if (cv == 0) continue;
                        score += weights[k] * de_table[c * de_stride + cv];
                    }
                    const double penalty = pull
                        * static_cast<double>(color_counts[c]);
                    score = (weight_obj > 0) ? (score - penalty)
                                              : (score + penalty);
                    bool better = (csel == 0)
                        || ((weight_obj > 0) ? (score > best_score)
                                              : (score < best_score));
                    if (better) {
                        best_score = score;
                        csel = static_cast<uint8_t>(c);
                    }
                }
            } else if (welsh_powell) {
                // Balanced greedy: among valid (non-conflicting) colors,
                // pick the one with the lowest global usage. Ties resolved
                // by lowest color index for determinism. Paired with the
                // WP descending-degree visit order — together they balance
                // both the *assignment moment* (rare slots get priority)
                // and the *graph order* (high-degree nodes first), yielding
                // a near-uniform 4-color distribution.
                int32_t best = 2147483647;
                for (int32_t c = 1; c <= n_colors; ++c) {
                    if ((mask & (1u << c)) != 0) continue;
                    if (color_counts[c] < best) {
                        best = color_counts[c];
                        csel = static_cast<uint8_t>(c);
                    }
                }
            } else {
                // Lowest-numbered valid color (bit-identical with numba).
                for (int32_t c = 1; c <= n_colors; ++c) {
                    if ((mask & (1u << c)) == 0) { csel = static_cast<uint8_t>(c); break; }
                }
            }
            counter[u] = 0;
        } else {
            // Tally colors among neighbors, pick least-used (excluding u's current).
            int32_t cnt[32] = {0};  // max 31 colors — comfortably above any usage
            for (int32_t k = row_beg; k < row_end; ++k) {
                const uint8_t cv = colors[indices[k]];
                if (cv != 0 && cv < 32) cnt[cv] += 1;
            }
            int32_t minv = cnt[1];
            csel = 1;
            const uint8_t cu = colors[u];
            if (cu != 0) minv = 2147483647;
            for (int32_t c = 1; c <= n_colors; ++c) {
                if (c == cu) continue;
                if (cnt[c] < minv) { minv = cnt[c]; csel = static_cast<uint8_t>(c); }
            }
            // Random restart after `rand_period` re-attempts. Use the exact LCG
            // seed scheme as the numba code so output matches bit-for-bit.
            if (rand_period > 0 && counter[u] == rand_period) {
                counter[u] = 0;
                const uint32_t seed = static_cast<uint32_t>(
                    1103515245u * (static_cast<uint32_t>(u) + static_cast<uint32_t>(count) + static_cast<uint32_t>(offset)) + 12345u
                ) & 0x7FFFFFFFu;
                csel = static_cast<uint8_t>(1 + (seed % static_cast<uint32_t>(n_colors)));
            }
        }

        if (colors[u] != csel) {
            if (welsh_powell) {
                const uint8_t old_c = colors[u];
                if (old_c != 0) --color_counts[old_c];
                if (csel != 0) ++color_counts[csel];
            }
            colors[u] = csel;
            for (int32_t k = row_beg; k < row_end; ++k) {
                const int32_t v = indices[k];
                if (colors[v] == csel) {
                    if (tail >= qcap) {
                        const int64_t newcap = std::max<int64_t>(qcap * 2, tail + 1);
                        q.resize(static_cast<size_t>(newcap));
                        qcap = newcap;
                    }
                    q[tail++] = v;
                }
            }
        }
    }

    return head >= tail;  // true = finished all in queue
}

// Conflict check — true if any edge has same-colored endpoints.
inline bool has_conflict_csr(
        const int32_t* indptr, const int32_t* indices, int32_t N,
        const uint8_t* colors) {
    for (int32_t u = 0; u < N; ++u) {
        const uint8_t cu = colors[u];
        if (cu == 0) continue;
        const int32_t row_beg = indptr[u];
        const int32_t row_end = indptr[u + 1];
        for (int32_t k = row_beg; k < row_end; ++k) {
            const int32_t v = indices[k];
            if (v > u && colors[v] == cu) return true;
        }
    }
    return false;
}

// Local-greedy repair: a few passes of "if u has color 0 or matches a
// neighbor, recolor to the smallest free color in 1..n". Cheap and resolves
// most residual conflicts the BFS leaves behind. Returns true if no conflicts
// remain afterward. Mirrors numba's `_repair_coloring`.
inline bool repair_coloring(
        const int32_t* indptr, const int32_t* indices, int32_t N,
        int32_t n_colors, int max_passes,
        std::vector<uint8_t>& colors) {
    int passes = 0;
    bool changed = true;
    while (changed && passes < max_passes) {
        changed = false;
        ++passes;
        for (int32_t u = 0; u < N; ++u) {
            const uint8_t cu = colors[u];
            uint32_t mask = 0;
            const int32_t row_beg = indptr[u];
            const int32_t row_end = indptr[u + 1];
            for (int32_t k = row_beg; k < row_end; ++k) {
                const uint8_t cv = colors[indices[k]];
                if (cv != 0) mask |= (1u << cv);
            }
            if (cu == 0 || (mask & (1u << cu)) != 0) {
                uint8_t csel = 0;
                for (int32_t c = 1; c <= n_colors; ++c) {
                    if ((mask & (1u << c)) == 0) { csel = static_cast<uint8_t>(c); break; }
                }
                if (csel != 0 && csel != cu) {
                    colors[u] = csel;
                    changed = true;
                }
            }
        }
    }
    return !has_conflict_csr(indptr, indices, N, colors.data());
}

} // namespace ncolor_cpp

#endif // NCOLOR_COLOR_HPP
