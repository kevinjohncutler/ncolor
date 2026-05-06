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

#pragma once

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
        // L2 toroidal expansion is not yet implemented natively — the
        // public ncolor.expand_labels(wrap=True) routes through a
        // np.pad(mode='wrap') prelude when p=2 (~9× cost on the expand
        // step). TODO: native L2 wrap via ghost-seed envelopes.
        (void)wrap;
        expand_labels_inplace(input, bufs, shape, pool, n_threads);
        if (output != bufs.lbl()) {
            std::memcpy(output, bufs.lbl(), bufs.size() * sizeof(int32_t));
        }
    }
};

// L1 (Manhattan): Saito-Toriwaki separable sweep. Operates in-place on
// a single buffer for the labels (with ``dist`` as scratch) — single
// input→output memcpy, then the sweep runs on ``output`` directly. The
// 2D fast path uses the dedicated kernel that fuses init + forward +
// backward sweep with register-carried state across cells (faster than
// the generic ND slab dispatch); ndim ≥ 3 falls through to the generic
// (A, B, C)-banded slab driver.
template <>
struct LpExpand<1> {
    static void expand(const int32_t* input, int32_t* output,
                       ExpandBuffers& bufs,
                       const std::vector<int64_t>& shape,
                       ForkJoinPool& pool, int n_threads, bool wrap = false) {
        const int ndim = static_cast<int>(shape.size());
        int64_t total = 1;
        for (int64_t d : shape) total *= d;
        bufs.resize(total);  // for the dist scratch
        if (input != output) {
            std::memcpy(output, input, total * sizeof(int32_t));
        }
        if (ndim == 2) {
            chamfer_st_l1(output, bufs.dist(),
                          shape[0], shape[1], pool, n_threads, wrap);
        } else {
            chamfer_st_l1_nd(output, bufs.dist(),
                             shape, pool, n_threads, wrap);
        }
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
