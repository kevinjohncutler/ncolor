// Lp distance metrics for the unified separable distance transform in
// expand.hpp. We support integer P in {1, 2} for now (the values ncolor
// actually uses); arbitrary float-p is left to dedicated DT packages
// (e.g. seung-lab/edt).
//
// The single envelope-of-functions algorithm in expand.hpp processes one
// 1D line per axis and replaces each cell's (label, accumulated p-power
// distance) with the closest seed under metric Lp. Per-axis the algorithm
// only needs three operations from the metric:
//
//   - pow_p(d):                   contribution of a 1D offset d to the
//                                 accumulated distance.
//   - intersect(v_q, g_q, v_r, g_r):
//                                 the i value at which the lower
//                                 envelope switches from f_q to f_r,
//                                 where f_j(i) = g_j + |i - v_j|^P.
//                                 (Stack-based envelope build pops while
//                                 this intersection is ≤ the prior
//                                 intersection on the stack.)
//   - dist_at(d, g):              g + |d|^P, used to fill the output
//                                 line once the envelope is built.
//
// For separable Lp DTs the per-axis pass replaces dist[i] with
//   min over j of g[j] + |i - v[j]|^P
// where g[j] is the dist accumulated from prior axes at position j. For
// p=2 this gives squared Euclidean (sum of squared per-axis offsets); for
// p=1 it gives Manhattan (sum of absolute per-axis offsets). Both fall
// out of the same envelope-of-functions algorithm — only the per-pair
// intersection formula and the contribution function change.
//
// Note: the intersection is returned as a double (not int) because the
// stack-build comparison `s > z[top]` is exact only when intersections
// can be represented exactly. For p=2 with integer (v, g) the
// intersection is rational; for p=1 it's half-integer at worst. Doubles
// give ample precision for both.

#pragma once

#include <cstdint>

namespace ncolor_cpp {

template <int P>
struct LpMetric;

// L2 (squared Euclidean): ``dist`` accumulates Σ (i - v_j)^2 across axes;
// final answer is √dist (we don't sqrt — caller wants Voronoi labels, not
// distances). Two parabolas g_q + (i - v_q)^2 and g_r + (i - v_r)^2
// (v_q < v_r) intersect at i = ((g_r - g_q) + (v_r^2 - v_q^2)) / (2(v_r - v_q)).
template <>
struct LpMetric<2> {
    static constexpr int p = 2;
    using AccumT = int32_t;

    static inline AccumT pow_p(int32_t d) {
        AccumT x = d;
        return x * x;
    }

    static inline AccumT dist_at(int32_t d, AccumT g) {
        AccumT x = d;
        return g + x * x;
    }

    static inline double intersect(int32_t v_q, AccumT g_q,
                                   int32_t v_r, AccumT g_r) {
        const double v_q_d = static_cast<double>(v_q);
        const double v_r_d = static_cast<double>(v_r);
        const double dv = v_r_d - v_q_d;
        const double numer = static_cast<double>(g_r - g_q)
                             + (v_r_d * v_r_d - v_q_d * v_q_d);
        return numer / (2.0 * dv);
    }
};

// L1 (Manhattan): ``dist`` accumulates Σ |i - v_j| across axes. Two V-shapes
// g_q + |i - v_q| and g_r + |i - v_r| (v_q < v_r) cross between v_q and
// v_r at i = ((g_r - g_q) + v_r + v_q) / 2. Outside [v_q, v_r] one V
// strictly dominates the other (no crossing), so the envelope only needs
// the in-between intersection.
template <>
struct LpMetric<1> {
    static constexpr int p = 1;
    using AccumT = int32_t;

    static inline AccumT pow_p(int32_t d) {
        return d < 0 ? -static_cast<AccumT>(d) : static_cast<AccumT>(d);
    }

    static inline AccumT dist_at(int32_t d, AccumT g) {
        return g + (d < 0 ? -static_cast<AccumT>(d) : static_cast<AccumT>(d));
    }

    static inline double intersect(int32_t v_q, AccumT g_q,
                                   int32_t v_r, AccumT g_r) {
        return (static_cast<double>(g_r - g_q)
                + static_cast<double>(v_r) + static_cast<double>(v_q)) * 0.5;
    }
};

}  // namespace ncolor_cpp
