// Standalone Kempe-component simulated annealing for 4-coloring.
//
// Loss = α₂ · n_same_2hop  +  γ · Σ w(u, v) · 𝟙[color(u) = color(v)]
//
// Moves: random Kempe (X, Y)-component swap on the 1-hop adjacency.
// Kempe swaps preserve 1-hop validity by construction; combined with
// Metropolis acceptance on the loss, this drives valid colorings
// toward low 2-hop and low similarity-weighted same-color pairs.
//
// All inputs are CSR arrays (indptr, indices) with 0-indexed cell IDs.
// The similarity-pair CSR (iou_*) has float64 weights per edge.
//
// Per-iteration cost: O(component_size + boundary). For typical bacterial
// cell graphs (avg degree ~6-8, avg component ~5-10), each iter is a few
// hundred ops. Designed to run thousands of iterations per millisecond.
//
// Reusable beyond the geometric variant: pass empty iou_* arrays to get
// pure 2-hop SA, or extend by adding more pair-loss terms in the same
// CSR pattern.

#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace ncolor_cpp {

struct KempeSAParams {
    double alpha_2hop  = 1.0;
    double gamma_iou   = 50.0;
    int    n_iters     = 30000;
    // Stop early once no improvement to best_loss is seen for this many
    // consecutive iterations. The SA loss plateaus quickly in practice
    // (typically by 500-1000 iters), so n_iters acts as a safety cap
    // and patience is what actually terminates most calls. Set to <=0
    // to disable early-stop.
    int    patience    = 1000;
    double T0          = 2.0;
    double T_min       = 0.001;
    double alpha_cool  = 0.9998;
    uint64_t rng_seed  = 0;
    int    n_colors    = 4;
};

// `colors` is in-place: input = seed coloring (1-indexed values, but
// addressed by 0-indexed cell), output = best coloring found.
// All CSR pair arrays store both directions (u→v and v→u) so the
// boundary delta computation in this function is symmetric.
//
// Returns the best loss value attained.
inline double kempe_sa(
    int32_t N,
    const int32_t* adj_indptr,    const int32_t* adj_indices,
    const int32_t* twohop_indptr, const int32_t* twohop_indices,
    const int32_t* iou_indptr,    const int32_t* iou_indices,
    const double*  iou_weights,
    std::vector<uint8_t>& colors,
    const KempeSAParams& params)
{
    // --- Initial loss (per-pair, counted once via v > u filter). ---
    int32_t n_2hop = 0;
    double  iou_loss = 0.0;
    for (int32_t u = 0; u < N; ++u) {
        const uint8_t cu = colors[u];
        if (cu == 0) continue;
        for (int32_t k = twohop_indptr[u]; k < twohop_indptr[u + 1]; ++k) {
            const int32_t v = twohop_indices[k];
            if (v > u && colors[v] == cu) ++n_2hop;
        }
        for (int32_t k = iou_indptr[u]; k < iou_indptr[u + 1]; ++k) {
            const int32_t v = iou_indices[k];
            if (v > u && colors[v] == cu) iou_loss += iou_weights[k];
        }
    }
    double cur_loss  = params.alpha_2hop * static_cast<double>(n_2hop)
                     + params.gamma_iou  * iou_loss;
    double best_loss = cur_loss;
    std::vector<uint8_t> best_colors = colors;

    // --- RNG, scratch buffers. ---
    std::mt19937_64 rng(params.rng_seed);
    std::uniform_real_distribution<double> rand01(0.0, 1.0);

    std::vector<int32_t> comp;     comp.reserve(static_cast<size_t>(N));
    std::vector<int32_t> queue;    queue.reserve(static_cast<size_t>(N));
    std::vector<uint8_t> in_comp(static_cast<size_t>(N), 0);

    double T = params.T0;
    int last_improvement_it = 0;

    // --- Main loop. ---
    for (int it = 0; it < params.n_iters; ++it) {
        if (params.patience > 0 && (it - last_improvement_it) > params.patience) {
            break;
        }
        // Random cell.
        const int32_t u = static_cast<int32_t>(rand01(rng) * N);
        const uint8_t cu = colors[u];
        if (cu == 0) {
            T = std::max(params.T_min, T * params.alpha_cool);
            continue;
        }
        // Random partner color.
        uint8_t Y;
        do {
            Y = static_cast<uint8_t>(1 + static_cast<int>(rand01(rng) * params.n_colors));
        } while (Y == cu);

        // BFS Kempe (cu, Y) component containing u.
        comp.clear(); queue.clear();
        comp.push_back(u);
        in_comp[u] = 1;
        queue.push_back(u);
        while (!queue.empty()) {
            const int32_t w = queue.back();
            queue.pop_back();
            const int32_t kend = adj_indptr[w + 1];
            for (int32_t k = adj_indptr[w]; k < kend; ++k) {
                const int32_t v = adj_indices[k];
                if (!in_comp[v] && (colors[v] == cu || colors[v] == Y)) {
                    in_comp[v] = 1;
                    comp.push_back(v);
                    queue.push_back(v);
                }
            }
        }

        // --- Delta loss across component boundary. ---
        int32_t delta_2hop = 0;
        double  delta_iou  = 0.0;
        for (size_t i = 0; i < comp.size(); ++i) {
            const int32_t w = comp[i];
            const uint8_t cw_old = colors[w];
            const uint8_t cw_new = (cw_old == cu) ? Y : cu;
            // 2-hop boundary.
            {
                const int32_t kend = twohop_indptr[w + 1];
                for (int32_t k = twohop_indptr[w]; k < kend; ++k) {
                    const int32_t nb = twohop_indices[k];
                    if (in_comp[nb]) continue;
                    const uint8_t cv = colors[nb];
                    const int old_same = (cw_old == cv) ? 1 : 0;
                    const int new_same = (cw_new == cv) ? 1 : 0;
                    delta_2hop += (new_same - old_same);
                }
            }
            // IoU boundary.
            {
                const int32_t kend = iou_indptr[w + 1];
                for (int32_t k = iou_indptr[w]; k < kend; ++k) {
                    const int32_t nb = iou_indices[k];
                    if (in_comp[nb]) continue;
                    const uint8_t cv = colors[nb];
                    const int old_same = (cw_old == cv) ? 1 : 0;
                    const int new_same = (cw_new == cv) ? 1 : 0;
                    delta_iou += iou_weights[k] * static_cast<double>(new_same - old_same);
                }
            }
        }
        const double delta = params.alpha_2hop * static_cast<double>(delta_2hop)
                           + params.gamma_iou  * delta_iou;

        // --- Metropolis accept/reject. ---
        bool accept = (delta < 0.0);
        if (!accept) {
            const double prob = std::exp(-delta / std::max(T, 1e-9));
            accept = (rand01(rng) < prob);
        }

        if (accept) {
            for (int32_t w : comp) {
                colors[w] = (colors[w] == cu) ? Y : cu;
            }
            cur_loss += delta;
            if (cur_loss < best_loss) {
                best_loss = cur_loss;
                best_colors = colors;
                last_improvement_it = it;
            }
        }

        // Clear in_comp scratch (only the cells we touched).
        for (int32_t w : comp) in_comp[w] = 0;

        T = std::max(params.T_min, T * params.alpha_cool);
    }

    colors = std::move(best_colors);
    return best_loss;
}

} // namespace ncolor_cpp
