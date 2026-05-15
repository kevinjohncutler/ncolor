// Per-cell geometric features + graph 2-hop CSR construction.
//
// Both routines are simple single-pass kernels that were the dominant
// Python-side bottlenecks in _geometric_fast.py after the Kempe-SA loop
// moved to C++. Pure C++; no pybind/numpy dependencies in this header.

#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ncolor_cpp {

// Per-cell geometric features computed from a 2-D label image in a
// single pass over the pixels.
struct CellGeom {
    double cy;       // centroid y
    double cx;       // centroid x
    double axis_y;   // major axis unit vector (y component)
    double axis_x;   // major axis unit vector (x component)
    double ecc;      // eccentricity (in [0, 1))
    int32_t area;
};

template <typename L>
inline void per_cell_geometry(
    const L* labels, int64_t H, int64_t W, int32_t N,
    std::vector<CellGeom>& out,
    // Optional: extract a per-label LUT from a second label image
    // (e.g. an existing 4-colouring) in the same pixel pass. Saves a
    // separate np.unique on the input label image, which scales as
    // O(M log N) and dominated geometric_fast's runtime at high N.
    const uint8_t* second_lookup = nullptr,
    uint8_t* second_lut_out = nullptr)
{
    // Accumulate moments per label in one image pass.
    const size_t Nplus = static_cast<size_t>(N) + 1;
    std::vector<int64_t> n(Nplus, 0);
    std::vector<double>  sy(Nplus, 0.0), sx(Nplus, 0.0);
    std::vector<double>  syy(Nplus, 0.0), sxx(Nplus, 0.0), sxy(Nplus, 0.0);
    if (second_lut_out) std::fill(second_lut_out, second_lut_out + Nplus, 0);
    const int64_t HW = H * W;
    for (int64_t i = 0; i < HW; ++i) {
        const int64_t lab = static_cast<int64_t>(labels[i]);
        if (lab > 0 && lab <= N) {
            const double y = static_cast<double>(i / W);
            const double x = static_cast<double>(i % W);
            n[lab]   += 1;
            sy[lab]  += y;
            sx[lab]  += x;
            syy[lab] += y * y;
            sxx[lab] += x * x;
            sxy[lab] += y * x;
            // First-pixel LUT (only writes once per label since 0 = unset
            // and we only write when the slot is still 0).
            if (second_lookup && second_lut_out && second_lut_out[lab] == 0) {
                second_lut_out[lab] = second_lookup[i];
            }
        }
    }
    out.assign(Nplus, CellGeom{0.0, 0.0, 1.0, 0.0, 0.0, 0});
    for (int32_t u = 1; u <= N; ++u) {
        const int64_t area = n[u];
        if (area == 0) continue;
        const double cy = sy[u] / static_cast<double>(area);
        const double cx = sx[u] / static_cast<double>(area);
        out[u].cy = cy; out[u].cx = cx; out[u].area = static_cast<int32_t>(area);
        if (area < 3) {
            out[u].axis_y = 1.0; out[u].axis_x = 0.0; out[u].ecc = 0.0;
            continue;
        }
        const double myy = syy[u] / static_cast<double>(area) - cy * cy;
        const double mxx = sxx[u] / static_cast<double>(area) - cx * cx;
        const double mxy = sxy[u] / static_cast<double>(area) - cy * cx;
        // 2x2 covariance eigendecomp (closed form).
        const double tr  = myy + mxx;
        const double det = myy * mxx - mxy * mxy;
        const double disc = std::max(tr * tr / 4.0 - det, 0.0);
        const double sq = std::sqrt(disc);
        double big   = tr / 2.0 + sq;
        double small = tr / 2.0 - sq;
        big   = std::max(big, 1e-6);
        small = std::max(small, 1e-6);
        out[u].ecc = std::sqrt(std::max(0.0, 1.0 - small / big));
        // Eigenvector for the larger eigenvalue.
        double ay, ax;
        if (std::abs(mxy) > 1e-12) { ay = big - mxx; ax = mxy; }
        else if (myy >= mxx)       { ay = 1.0; ax = 0.0; }
        else                        { ay = 0.0; ax = 1.0; }
        const double norm = std::hypot(ay, ax);
        if (norm > 0.0) { ay /= norm; ax /= norm; }
        out[u].axis_y = ay; out[u].axis_x = ax;
    }
}


// 2-hop neighbours of a graph in CSR form.
// out[u] = { v : graph_distance(u, v) == 2 }. Both directions emitted
// (so the result is symmetric: u ∈ twohop[v] iff v ∈ twohop[u]).
// Uses an "epoch tag" per vertex (O(N) scratch, no per-cell set
// allocation) — visits cells at distance ≤ 2 in two passes through the
// CSR adjacency.
inline void compute_two_hop_csr(
    const int32_t* adj_indptr, const int32_t* adj_indices, int32_t N,
    std::vector<int32_t>& out_indptr,
    std::vector<int32_t>& out_indices)
{
    out_indptr.assign(static_cast<size_t>(N) + 1, 0);

    // Epoch-tag scratch: seen[v] == u + 1 means "marked during u's pass"
    // (offset by 1 so the zero-initialised state means "never seen").
    std::vector<int32_t> seen(static_cast<size_t>(N), 0);

    // First pass: count outgoing 2-hop edges per cell.
    for (int32_t u = 0; u < N; ++u) {
        const int32_t epoch = u + 1;
        const int32_t k_lo = adj_indptr[u], k_hi = adj_indptr[u + 1];
        // Mark 1-hop neighbours + self so they're skipped.
        seen[u] = epoch;
        for (int32_t k = k_lo; k < k_hi; ++k) seen[adj_indices[k]] = epoch;
        int32_t count = 0;
        for (int32_t k = k_lo; k < k_hi; ++k) {
            const int32_t v = adj_indices[k];
            const int32_t kk_lo = adj_indptr[v], kk_hi = adj_indptr[v + 1];
            for (int32_t kk = kk_lo; kk < kk_hi; ++kk) {
                const int32_t w = adj_indices[kk];
                if (seen[w] != epoch) {
                    seen[w] = epoch;
                    ++count;
                }
            }
        }
        out_indptr[u + 1] = count;
    }

    // Prefix sum to indptr.
    int32_t total = 0;
    for (int32_t u = 0; u < N; ++u) {
        const int32_t c = out_indptr[u + 1];
        out_indptr[u + 1] = total;
        total += c;
    }
    // Now out_indptr[u+1] is the START of u's row, and total is the
    // grand total. Shift so out_indptr[u] is the start of u's row.
    for (int32_t u = N; u > 0; --u) out_indptr[u] = out_indptr[u - 1];
    out_indptr[0] = 0;
    out_indptr[N] = total;

    out_indices.assign(static_cast<size_t>(total), 0);

    // Second pass: fill indices, re-using the epoch scratch.
    std::fill(seen.begin(), seen.end(), 0);
    std::vector<int32_t> write_pos = out_indptr;  // copy

    for (int32_t u = 0; u < N; ++u) {
        const int32_t epoch = u + 1;
        const int32_t k_lo = adj_indptr[u], k_hi = adj_indptr[u + 1];
        seen[u] = epoch;
        for (int32_t k = k_lo; k < k_hi; ++k) seen[adj_indices[k]] = epoch;
        for (int32_t k = k_lo; k < k_hi; ++k) {
            const int32_t v = adj_indices[k];
            const int32_t kk_lo = adj_indptr[v], kk_hi = adj_indptr[v + 1];
            for (int32_t kk = kk_lo; kk < kk_hi; ++kk) {
                const int32_t w = adj_indices[kk];
                if (seen[w] != epoch) {
                    seen[w] = epoch;
                    out_indices[write_pos[u]++] = w;
                }
            }
        }
    }
}


// Build a "symmetric pair CSR" from a list of unordered (u, v, w) triples.
// Each input pair (u, v, w) emits two CSR entries: u → (v, w) and v → (u, w).
// This is how the Kempe-SA kernel expects pair-weighted CSRs (both
// directions stored so boundary delta computation is symmetric).
inline void build_symmetric_pair_csr(
    const int32_t* pair_u, const int32_t* pair_v, const double* pair_w,
    int32_t n_pairs, int32_t N,
    std::vector<int32_t>& out_indptr,
    std::vector<int32_t>& out_indices,
    std::vector<double>&  out_weights)
{
    out_indptr.assign(static_cast<size_t>(N) + 1, 0);
    for (int32_t p = 0; p < n_pairs; ++p) {
        out_indptr[pair_u[p] + 1] += 1;
        out_indptr[pair_v[p] + 1] += 1;
    }
    for (int32_t u = 0; u < N; ++u) out_indptr[u + 1] += out_indptr[u];

    const int32_t total = out_indptr[N];
    out_indices.assign(static_cast<size_t>(total), 0);
    out_weights.assign(static_cast<size_t>(total), 0.0);

    std::vector<int32_t> write_pos = out_indptr;
    for (int32_t p = 0; p < n_pairs; ++p) {
        const int32_t u = pair_u[p], v = pair_v[p];
        const double  w = pair_w[p];
        out_indices[write_pos[u]] = v;
        out_weights[write_pos[u]++] = w;
        out_indices[write_pos[v]] = u;
        out_weights[write_pos[v]++] = w;
    }
}

} // namespace ncolor_cpp
