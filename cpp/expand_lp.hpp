// Unified Lp Voronoi label expansion — single entry point, one code path
// from the caller's perspective, compile-time dispatch on the integer
// norm p.
//
// Why each metric owns its full algorithm (not just a per-axis kernel
// behind a shared ND loop):
//
//   L1 is mathematically unique among integer Lp norms. The L1 distance
//   admits the chain decomposition  |i - v_j| = |i - i'| + |i' - v_j|
//   (when i' lies between i and v_j), which collapses the per-axis 1D
//   pass into a forward + backward running-min sweep (Saito-Toriwaki).
//   For p ≥ 2 the same decomposition doesn't hold for |i - v_j|^p, so
//   the per-axis pass requires building the lower envelope of the
//   per-seed distance functions explicitly (Felzenszwalb-Huttenlocher).
//   Sharing only the per-axis kernel imposes a structure that costs
//   the L1 sweep its register-carry win on 2D (~2×). Sharing at the
//   metric level gets us the speed and keeps the user-visible API
//   identical.
//
// What IS shared: the entry point ``expand_labels_lp<P>``, the buffer
// manager ``ExpandBuffers`` (used for scratch by both metrics), the
// ``(input, output)`` contract, and the ``out`` ownership. Adding L_3,
// L_4, L_∞ later means a new ``LpExpand<P>::expand`` specialization
// with its own pipeline.

#ifndef NCOLOR_EXPAND_LP_HPP
#define NCOLOR_EXPAND_LP_HPP

#include "expand.hpp"
#include "chamfer.hpp"

namespace ncolor_cpp {

// Each LpExpand specialization writes its result directly into ``output``
// (caller-allocated). ``bufs`` is scratch only — the metric uses its
// internal buffers as appropriate. This keeps the L1 path single-memcpy
// (input→output, then in-place sweep) while the L2 path naturally uses
// its multi-buffer transpose pipeline.

template <int P>
struct LpExpand;

// L2 (Euclidean²): Felzenszwalb separable parabolic envelope. First axis
// uses the pass-0 midpoint Voronoi fast path (sparse seeds, all g=0).
// Subsequent axes do transpose+envelope+transpose so the inner envelope
// loop is contiguous (cache-friendly above ~512 column lengths).
template <>
struct LpExpand<2> {
    static void expand(const int32_t* input, int32_t* output,
                       ExpandBuffers& bufs,
                       const std::vector<int64_t>& shape,
                       ForkJoinPool& pool, int n_threads, bool wrap = false) {
        // wrap=true routes the inner envelope sweeps through their
        // ghost-seed (toroidal) variants — see envelope_pass_row_impl
        // <Wrap=true> in expand.hpp. The innermost-axis pass0 fast path
        // is skipped in wrap mode (its midpoint trick doesn't generalise
        // cleanly to torus tie-break); expand_labels_inplace falls back
        // to envelope_pass(wrap=true) there. ~2-3× the standard cost.
        expand_labels_inplace(input, bufs, shape, pool, n_threads, wrap);
        if (output != bufs.lbl()) {
            std::memcpy(output, bufs.lbl(), bufs.size() * sizeof(int32_t));
        }
    }
};

// L1 (Manhattan): Saito-Toriwaki separable sweep. Operates in-place on
// the output buffer for the labels (with ``dist`` as scratch) — one
// input→output memcpy, then the ND driver runs the per-row fused
// init+forward+backward sweep on the innermost axis and (B, C)-banded
// slab passes on the rest.
template <>
struct LpExpand<1> {
    static void expand(const int32_t* input, int32_t* output,
                       ExpandBuffers& bufs,
                       const std::vector<int64_t>& shape,
                       ForkJoinPool& pool, int n_threads, bool wrap = false) {
        int64_t total = 1;
        for (int64_t d : shape) total *= d;
        bufs.resize(total);  // for the dist scratch
        if (input != output) {
            std::memcpy(output, input, total * sizeof(int32_t));
        }
        chamfer_st_l1_nd(output, bufs.dist(), shape, pool, n_threads, wrap);
    }
};

// =============================================================================
// Single user-facing entry: dispatch on p at compile time.
template <int P>
inline void expand_labels_lp(const int32_t* input, int32_t* output,
                             ExpandBuffers& bufs,
                             const std::vector<int64_t>& shape,
                             ForkJoinPool& pool, int n_threads, bool wrap = false) {
    LpExpand<P>::expand(input, output, bufs, shape, pool, n_threads, wrap);
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_EXPAND_LP_HPP
