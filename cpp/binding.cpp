/*
 * Pybind11 binding for ncolor C++. Exposes two classes — each wraps a
 * persistent ForkJoinPool, so callers construct once and reuse:
 *   - ``ExpandEngine``  : Voronoi label expansion (``expand_labels``)
 *   - ``Solver``        : end-to-end ncolor.label pipeline (``label`` /
 *                        ``connect``)
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "cc_label.hpp"
#include "chamfer.hpp"
#include "color.hpp"
#include "delete_spurs.hpp"
#include "delete_spurs_labels.hpp"
#include "expand_spur_free.hpp"
#include "expand_lp.hpp"
#include "format_labels.hpp"
#include "connect.hpp"
#include "geometry.hpp"
#include "kempe_sa.hpp"
#include "tabucol.hpp"
#include "hea.hpp"
#include "bb_dsatur.hpp"
#include "clique_lb.hpp"
#include "dispatch.hpp"
#include "expand.hpp"

namespace py = pybind11;

// Resolve a user-supplied n_threads value into a concrete positive int.
//
//   -1 / 0 / negative          →  _smt.auto_threads()  (cached calibration)
//   0 < x < 1                  →  round(x × os.cpu_count()), min 1
//   1                          →  1 (serial)
//   x ≥ 1                      →  round(x), exact
//
// Note: `1` is always interpreted as one thread, never as 100% of cores —
// the fractional-ratio interpretation only applies for values strictly
// between 0 and 1.
//
// Accepts double for flexibility (Python int/float both convert), with -1.0
// as the default sentinel for "auto". Python ``None`` would also be natural
// but ``py::object`` constructors break import on macOS arm64 with
// pybind11 3.0.4, so we use double here and let users pass -1 for auto.
static int resolve_threads(double v) {
    if (v <= 0.0) {
        return py::module_::import("ncolor._backend._smt").attr("auto_threads")().cast<int>();
    }
    if (v < 1.0) {
        const long ncpu = py::module_::import("os").attr("cpu_count")().cast<long>();
        const long n = static_cast<long>(v * static_cast<double>(ncpu) + 0.5);
        return static_cast<int>(std::max<long>(1, n));
    }
    return static_cast<int>(std::max<long>(1, static_cast<long>(v + 0.5)));
}

// Dispatch on a numpy buffer's dtype, calling `f<T>()` with the matched
// integer type. `f` is a generic lambda that takes a tag pointer:
//
//   dispatch_int_dtype(fmt, itemsize, "Solver.label", [&](auto* tag) {
//       using T = std::remove_pointer_t<decltype(tag)>;
//       ncolor_cpp::cast_with_bg<T>(static_cast<const T*>(src), ...);
//   });
//
// Resolves dtype by (itemsize, signedness) so the same code handles macOS
// `l` (int64) and pybind11 `q` (long long) — relying on format-code matching
// alone breaks across platforms. Throws on unsupported dtype with `api_name`
// in the error message.
template <typename Func>
static inline void dispatch_int_dtype(const std::string& fmt, py::ssize_t itemsize,
                                      const char* api_name, Func&& f) {
    bool is_signed = false, is_unsigned = false;
    if (!fmt.empty()) {
        const char c = fmt[0];
        if (c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' || c == 'n')
            is_signed = true;
        else if (c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' || c == 'N')
            is_unsigned = true;
    }
    if (is_signed) {
        switch (itemsize) {
            case 1: f(static_cast<int8_t*>(nullptr));  return;
            case 2: f(static_cast<int16_t*>(nullptr)); return;
            case 4: f(static_cast<int32_t*>(nullptr)); return;
            case 8: f(static_cast<int64_t*>(nullptr)); return;
        }
    } else if (is_unsigned) {
        switch (itemsize) {
            case 1: f(static_cast<uint8_t*>(nullptr));  return;
            case 2: f(static_cast<uint16_t*>(nullptr)); return;
            case 4: f(static_cast<uint32_t*>(nullptr)); return;
        }
    }
    throw std::invalid_argument(std::string(api_name) +
        ": unsupported dtype '" + fmt + "' (need uint8/uint16/uint32/"
        "int8/int16/int32/int64)");
}

// Pack a vector of (lo, hi) adjacency pairs into a fresh (M, 2) int32 array.
static inline py::array_t<int32_t> pairs_to_array(
        const std::vector<std::pair<int32_t, int32_t>>& pairs) {
    const py::ssize_t m = static_cast<py::ssize_t>(pairs.size());
    py::array_t<int32_t> out({m, py::ssize_t{2}});
    int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);
    for (py::ssize_t i = 0; i < m; ++i) {
        out_ptr[i * 2 + 0] = pairs[i].first;
        out_ptr[i * 2 + 1] = pairs[i].second;
    }
    return out;
}

// Persistent-pool wrapper for expand_labels + parallel LUT apply.
// One ExpandEngine per ncolor.label "pipeline" — the pool + buffers persist
// across calls so the only per-call cost is task enqueue.
class ExpandEngine {
public:
    explicit ExpandEngine(double n_threads)
        : n_threads_(resolve_threads(n_threads)),
          pool_(std::make_unique<ncolor_cpp::ForkJoinPool>(n_threads_ <= 1 ? 1 : n_threads_)) {}

    int n_threads() const { return n_threads_; }

    // Voronoi label expansion under L_p metric. ``p=1`` (Manhattan) uses the
    // Saito-Toriwaki separable sweep; ``p=2`` (Euclidean²) uses the
    // Felzenszwalb-Huttenlocher parabolic envelope. Same ND driver,
    // dispatched at compile time on p — see ``expand_lp.hpp``. Default is
    // p=2 (matches numba's ``expand_labels(metric='l2')``).
    py::array_t<int32_t> expand_labels(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
            int p = 2, bool wrap = false) {
        const auto buf = labels.request();
        std::vector<int64_t> shape(buf.ndim);
        for (int i = 0; i < buf.ndim; ++i) shape[i] = buf.shape[i];

        const int32_t* input = static_cast<const int32_t*>(buf.ptr);
        py::array_t<int32_t> out(buf.shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);

        {
            py::gil_scoped_release release;
            if (p == 2) {
                ncolor_cpp::expand_labels_lp<2>(input, out_ptr, bufs_, shape, *pool_, n_threads_, wrap);
            } else if (p == 1) {
                ncolor_cpp::expand_labels_lp<1>(input, out_ptr, bufs_, shape, *pool_, n_threads_, wrap);
            } else {
                throw std::invalid_argument("expand_labels: p must be 1 or 2");
            }
        }
        return out;
    }

    // Voronoi label expansion that ALSO returns the distance field.
    // ``p=2`` returns Euclidean (not squared); ``p=1`` returns L1. The
    // underlying Felzenszwalb / Saito-Toriwaki sweep computes the
    // distance internally as scratch — exposing it costs one extra
    // ``shape``-sized buffer + parallel sqrt.
    std::pair<py::array_t<int32_t>, py::array_t<double>> expand_labels_with_dist(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
            int p = 2, bool wrap = false) {
        const auto buf = labels.request();
        std::vector<int64_t> shape(buf.ndim);
        int64_t total = 1;
        for (int i = 0; i < buf.ndim; ++i) {
            shape[i] = buf.shape[i];
            total *= buf.shape[i];
        }

        const int32_t* input = static_cast<const int32_t*>(buf.ptr);
        py::array_t<int32_t> out_lbl(buf.shape);
        py::array_t<double>  out_dist(buf.shape);
        int32_t* out_lbl_ptr  = static_cast<int32_t*>(out_lbl.request().ptr);
        double*  out_dist_ptr = static_cast<double*>(out_dist.request().ptr);

        {
            py::gil_scoped_release release;
            if (p == 2) {
                ncolor_cpp::expand_labels_lp<2>(input, out_lbl_ptr, bufs_, shape, *pool_, n_threads_, wrap);
                const int32_t* d = bufs_.dist();
                for (int64_t i = 0; i < total; ++i) {
                    out_dist_ptr[i] = std::sqrt(static_cast<double>(d[i]));
                }
            } else if (p == 1) {
                ncolor_cpp::expand_labels_lp<1>(input, out_lbl_ptr, bufs_, shape, *pool_, n_threads_, wrap);
                const int32_t* d = bufs_.dist();
                for (int64_t i = 0; i < total; ++i) {
                    out_dist_ptr[i] = static_cast<double>(d[i]);
                }
            } else {
                throw std::invalid_argument("expand_labels_with_dist: p must be 1 or 2");
            }
        }
        return {std::move(out_lbl), std::move(out_dist)};
    }

    // Per-class minimum distance fields. ``class_of`` is a length-(N+1)
    // array; ``class_of[u]`` is the class of cell ``u`` in 1..K (0 means
    // cell ``u`` is excluded). For each c in 1..K, run a single-source
    // L_p expansion with the mask {pixels whose label has class c} as
    // seeds, and stack the resulting distance fields.
    //
    // Output shape: ``(K, *labels.shape)`` float64. ``out[c-1, ...]`` is
    // the distance from each pixel to the nearest pixel of any cell with
    // ``class_of[label] == c``. ``p=2`` returns Euclidean (sqrt-applied);
    // ``p=1`` returns L1. Pixels in classes with no seeds get +inf.
    py::array_t<double> per_class_min_edt(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> class_of,
            int n_classes, int p = 2, bool wrap = false) {
        const auto lbuf = labels.request();
        const auto cbuf = class_of.request();
        std::vector<int64_t> shape(lbuf.ndim);
        int64_t total = 1;
        for (int i = 0; i < lbuf.ndim; ++i) {
            shape[i] = lbuf.shape[i];
            total *= lbuf.shape[i];
        }
        if (cbuf.ndim != 1) throw std::invalid_argument(
            "per_class_min_edt: class_of must be 1-D (length N+1)");
        if (n_classes < 1) throw std::invalid_argument(
            "per_class_min_edt: n_classes must be >= 1");

        const int32_t* lbl_in = static_cast<const int32_t*>(lbuf.ptr);
        const int32_t* class_of_ptr = static_cast<const int32_t*>(cbuf.ptr);
        const int64_t n_class_of = static_cast<int64_t>(cbuf.shape[0]);

        std::vector<py::ssize_t> out_shape;
        out_shape.reserve(static_cast<size_t>(lbuf.ndim) + 1);
        out_shape.push_back(n_classes);
        for (int i = 0; i < lbuf.ndim; ++i) out_shape.push_back(lbuf.shape[i]);
        py::array_t<double> out(out_shape);
        double* out_ptr = static_cast<double*>(out.request().ptr);

        // Scratch: per-class masked label image (rewritten each pass).
        std::vector<int32_t> masked(static_cast<size_t>(total));
        std::vector<int32_t> labels_out_scratch(static_cast<size_t>(total));

        {
            py::gil_scoped_release release;
            // Initialise output to +infinity. If a class has no seeds the
            // expansion writes labels=0 and dist=INT_MAX/4; we replace
            // those with +inf for safe min-aggregation downstream.
            const double INF = std::numeric_limits<double>::infinity();
            for (int64_t i = 0; i < static_cast<int64_t>(n_classes) * total; ++i)
                out_ptr[i] = INF;

            for (int c = 1; c <= n_classes; ++c) {
                // Build mask: pixels whose label maps to class c stay; others zero.
                bool any_seed = false;
                for (int64_t i = 0; i < total; ++i) {
                    const int32_t u = lbl_in[i];
                    if (u > 0 && u < n_class_of && class_of_ptr[u] == c) {
                        masked[static_cast<size_t>(i)] = u;
                        any_seed = true;
                    } else {
                        masked[static_cast<size_t>(i)] = 0;
                    }
                }
                if (!any_seed) continue;  // out stays +inf for this class

                if (p == 2) {
                    ncolor_cpp::expand_labels_lp<2>(masked.data(),
                        labels_out_scratch.data(), bufs_, shape, *pool_, n_threads_, wrap);
                    const int32_t* d = bufs_.dist();
                    double* out_c = out_ptr + static_cast<int64_t>(c - 1) * total;
                    for (int64_t i = 0; i < total; ++i) {
                        out_c[i] = std::sqrt(static_cast<double>(d[i]));
                    }
                } else if (p == 1) {
                    ncolor_cpp::expand_labels_lp<1>(masked.data(),
                        labels_out_scratch.data(), bufs_, shape, *pool_, n_threads_, wrap);
                    const int32_t* d = bufs_.dist();
                    double* out_c = out_ptr + static_cast<int64_t>(c - 1) * total;
                    for (int64_t i = 0; i < total; ++i) {
                        out_c[i] = static_cast<double>(d[i]);
                    }
                } else {
                    throw std::invalid_argument("per_class_min_edt: p must be 1 or 2");
                }
            }
        }
        return out;
    }

    // Pairwise N×N minimum-distance matrix. ``D[a, b]`` is the minimum
    // distance from any pixel of cell ``a+1`` to any pixel of cell
    // ``b+1`` (0-indexed). For each source cell u in 1..N we run a
    // single-source expansion seeded only at u's pixels, then scan the
    // image once to fill row u of D from the dist field. ``D[u, u] = 0``.
    //
    // Output shape (N, N) float64. ``p=2`` returns Euclidean, ``p=1`` L1.
    py::array_t<double> pairwise_nearest_distance(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
            int n_labels, int p = 2, bool wrap = false) {
        const auto lbuf = labels.request();
        std::vector<int64_t> shape(lbuf.ndim);
        int64_t total = 1;
        for (int i = 0; i < lbuf.ndim; ++i) {
            shape[i] = lbuf.shape[i];
            total *= lbuf.shape[i];
        }
        if (n_labels < 1) throw std::invalid_argument(
            "pairwise_nearest_distance: n_labels must be >= 1");

        const int32_t* lbl_in = static_cast<const int32_t*>(lbuf.ptr);
        py::array_t<double> out({n_labels, n_labels});
        double* D = static_cast<double*>(out.request().ptr);

        std::vector<int32_t> masked(static_cast<size_t>(total));
        std::vector<int32_t> labels_out_scratch(static_cast<size_t>(total));

        {
            py::gil_scoped_release release;
            const double INF = std::numeric_limits<double>::infinity();
            for (int64_t i = 0; i < static_cast<int64_t>(n_labels) * n_labels; ++i)
                D[i] = INF;
            for (int i = 0; i < n_labels; ++i) D[i * n_labels + i] = 0.0;

            for (int u = 1; u <= n_labels; ++u) {
                // Mask: keep only cell u's pixels as seeds.
                bool any_seed = false;
                for (int64_t i = 0; i < total; ++i) {
                    if (lbl_in[i] == u) { masked[i] = u; any_seed = true; }
                    else masked[i] = 0;
                }
                if (!any_seed) continue;

                if (p == 2) {
                    ncolor_cpp::expand_labels_lp<2>(masked.data(),
                        labels_out_scratch.data(), bufs_, shape, *pool_, n_threads_, wrap);
                } else if (p == 1) {
                    ncolor_cpp::expand_labels_lp<1>(masked.data(),
                        labels_out_scratch.data(), bufs_, shape, *pool_, n_threads_, wrap);
                } else {
                    throw std::invalid_argument("pairwise_nearest_distance: p must be 1 or 2");
                }
                const int32_t* d = bufs_.dist();
                double* D_row = D + static_cast<int64_t>(u - 1) * n_labels;

                // Single pass: for each pixel, update D[u-1, label-1] = min(..., dist)
                // for label != u. For p=2 we sqrt; for p=1 we cast.
                if (p == 2) {
                    for (int64_t i = 0; i < total; ++i) {
                        const int32_t v = lbl_in[i];
                        if (v <= 0 || v == u || v > n_labels) continue;
                        const double dd = std::sqrt(static_cast<double>(d[i]));
                        if (dd < D_row[v - 1]) D_row[v - 1] = dd;
                    }
                } else {
                    for (int64_t i = 0; i < total; ++i) {
                        const int32_t v = lbl_in[i];
                        if (v <= 0 || v == u || v > n_labels) continue;
                        const double dd = static_cast<double>(d[i]);
                        if (dd < D_row[v - 1]) D_row[v - 1] = dd;
                    }
                }
            }
        }
        return out;
    }

    // In-place label compaction: rewrite nonzero labels to 1..N (with
    // bg=0). Min-shift semantics match the legacy ``format_labels``:
    // if min(labels) != 0, the min is treated as bg and everything is
    // shifted before compacting.
    //
    // Accepts any of the supported integer dtypes (uint8/uint16/uint32,
    // int8/int16/int32, int64) — the int32 cast happens inside the
    // released-GIL block in parallel via cast_to_int32, so the public
    // Python wrapper avoids a single-threaded numpy.astype + .copy()
    // round-trip outside the GIL release (~5 ms saved at 256³ uint16).
    //
    // Default (first_seen=false) uses ascending-source numbering: the
    // new label assigned to source-label k is its rank among present
    // labels — i.e. for source labels {3, 7, 12} the remap is
    // {3→1, 7→2, 12→3}. Parallel build, faster.
    //
    // first_seen=true uses input-order numbering, matching
    // fastremap.renumber bit-for-bit. Available for callers that depend
    // on the historical fastremap output ordering. Build pass is serial
    // (we only learn a label is new on first encounter), ~2× slower.
    //
    // Returns (formatted_array, n_labels).
    std::pair<py::array_t<int32_t>, int> format_labels(
            py::array labels, bool first_seen = false) {
        if (!(labels.flags() & py::array::c_style)) {
            labels = py::array::ensure(labels, py::array::c_style);
        }
        const auto buf = labels.request();
        const int64_t total = buf.size;
        const void* src_ptr = buf.ptr;

        // Allocate int32 output array. pybind11 value-inits (memset 0) from
        // the main thread, but cast_to_int32 immediately rewrites every byte
        // from the worker that owns each chunk, and modern allocators don't
        // zero freshly-mapped pages anyway — net cost is negligible.
        std::vector<py::ssize_t> out_shape(buf.ndim);
        for (py::ssize_t d = 0; d < buf.ndim; ++d) out_shape[d] = buf.shape[d];
        py::array_t<int32_t> out(out_shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);

        int n_labels;
        {
            py::gil_scoped_release release;
            // Cast to int32 in parallel inside the released-GIL block.
            dispatch_int_dtype(buf.format, buf.itemsize,
                "ExpandEngine.format_labels", [&](auto* tag) {
                    using T = std::remove_pointer_t<decltype(tag)>;
                    ncolor_cpp::cast_to_int32<T>(
                        static_cast<const T*>(src_ptr), out_ptr, total,
                        *pool_, n_threads_);
                });
            n_labels = first_seen
                ? ncolor_cpp::format_labels_inplace_first_seen(
                    out_ptr, total, *pool_, n_threads_)
                : ncolor_cpp::format_labels_inplace(
                    out_ptr, total, *pool_, n_threads_);
        }
        return {std::move(out), n_labels};
    }

private:
    int n_threads_;
    std::unique_ptr<ncolor_cpp::ForkJoinPool> pool_;
    ncolor_cpp::ExpandBuffers bufs_;
};

// Helpers shared by Solver — connect-style preprocessing replicated in C++ so
// we don't bounce back to Python between phases.
static inline int64_t ipow2_ge(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

// Minimum hashtable capacity for the find_pairs scan. Below this the
// power-of-two rounding gives degenerate sizes that hurt insert
// throughput on tiny graphs; the cost of overshooting is just a few
// hundred bytes per worker.
static constexpr int64_t MIN_HT_SIZE = 16;

// Solver: end-to-end ncolor.label equivalent in C++. Owns a thread pool +
// the scratch buffers for cast / format_labels / expand / connect / CSR
// build / coloring / apply_lut.
class Solver {
public:
    explicit Solver(double n_threads)
        : n_threads_(resolve_threads(n_threads)),
          pool_(std::make_unique<ncolor_cpp::ForkJoinPool>(n_threads_ <= 1 ? 1 : n_threads_)) {}

    int n_threads() const { return n_threads_; }

    // Per-stage timing breakdown of the most recent label() call. Empty
    // unless capture_stages=true was passed.
    std::vector<std::pair<std::string, double>> get_last_stages() const { return last_stages_; }

    // Adjacency pairs for a label image. Takes the image directly (any of
    // the supported integer dtypes) and returns an (M, 2) int32 array of
    // unique (lo, hi) pairs of adjacent labels under connectivity ``conn``
    // (1..ndim). Routes through the unified ND unpadded scan kernel.
    py::array_t<int32_t> connect(py::array mask, int conn = 1, bool wrap = false) {
        if (!(mask.flags() & py::array::c_style)) {
            mask = py::array::ensure(mask, py::array::c_style);
        }
        const auto buf = mask.request();
        const int ndim = static_cast<int>(buf.ndim);
        if (ndim < 2) throw std::invalid_argument(
            "Solver.connect expects a label image with ndim >= 2");
        if (conn < 1 || conn > ndim) throw std::invalid_argument(
            "Solver.connect: conn must be in [1, ndim]");

        std::vector<int64_t> shape(ndim);
        int64_t total = 1;
        for (int d = 0; d < ndim; ++d) {
            shape[d] = static_cast<int64_t>(buf.shape[d]);
            total *= shape[d];
        }
        // Unified ND unpadded find_pairs handles all (ndim, conn) cases.
        const void* src_ptr = buf.ptr;

        std::vector<std::pair<int32_t, int32_t>> pairs;
        {
            py::gil_scoped_release release;
            // Cast to int32 in expand_bufs_.lbl(); the bg mask is unused
            // here (Solver.connect never applies a LUT) but cast_with_bg
            // is the parallel cast we already use elsewhere — bg writes
            // are cheap and let us share the kernel.
            expand_bufs_.resize(total);
            int32_t* labels = expand_bufs_.lbl();
            bg_mask_.resize(static_cast<size_t>(total));
            uint8_t* bg = bg_mask_.data();
            dispatch_int_dtype(buf.format, buf.itemsize, "Solver.connect",
                [&](auto* tag) {
                    using T = std::remove_pointer_t<decltype(tag)>;
                    ncolor_cpp::cast_with_bg<T>(
                        static_cast<const T*>(src_ptr), labels, bg, total,
                        *pool_, n_threads_);
                });

            const int32_t max_label = parallel_max_label_(labels, total);
            pairs = find_pairs_(labels, shape, conn, wrap, max_label);
        }

        return pairs_to_array(pairs);
    }

    std::pair<py::array_t<uint8_t>, int> label(
            py::array mask,
            int n_colors = 4, int max_depth = 30, int rand_period = 10,
            int conn = 2, int p = 2, bool capture_stages = false,
            bool format_input = true, bool expand = true,
            py::object out_arg = py::none(),
            int color_mode = -1, bool wrap = false, bool balance = false,
            bool first_seen = false,
            int weight_objective = 0,
            py::object de_table_obj = py::none(),
            int weight_mode = 1 /* ReduceMode::Min */,
            py::object extra_edges_obj = py::none(),
            int connect_radius = 1,
            int despur_iters = 2,
            bool expand_spur_free = false,
            int spur_free_max_rounds = 1) {
        // color_mode: -1 = auto (default; threshold-based), 0 = force serial,
        // 1 = force parallel. Used by benchmarks to A/B test the parallel
        // coloring path without rebuilding the extension.
        // Require C-contiguous; pybind11 doesn't enforce that for the
        // untyped py::array, so check explicitly. Common dtypes accepted
        // (uint8/uint16/uint32, int8/int16/int32/int64) and fused with
        // the format_labels pass inside the GIL-released block.
        if (!(mask.flags() & py::array::c_style)) {
            mask = py::array::ensure(mask, py::array::c_style);
        }
        const auto buf = mask.request();
        const int ndim = static_cast<int>(buf.ndim);
        if (ndim < 2) throw std::invalid_argument(
            "Solver.label expects a label image with ndim >= 2");
        if (conn < 1 || conn > ndim) throw std::invalid_argument(
            "Solver.label: conn must be in [1, ndim]");
        if (p != 1 && p != 2) throw std::invalid_argument(
            "Solver.label: p must be 1 or 2");

        std::vector<int64_t> shape(ndim);
        int64_t total = 1;
        for (int d = 0; d < ndim; ++d) {
            shape[d] = static_cast<int64_t>(buf.shape[d]);
            total *= shape[d];
        }
        const void* src_ptr = buf.ptr;
        py::array_t<uint8_t> out = prepare_out_buffer_(out_arg, buf, ndim);
        uint8_t* out_ptr = static_cast<uint8_t*>(out.request().ptr);

        int n_used = 0;
        last_stages_.clear();
        // Reset per-call accessor state so get_last_lut() / get_last_n_conflicts()
        // always reflect the current call (and never silently report data from
        // the previous call when this one short-circuits).
        last_n_conflicts_ = 0;
        lut_.assign(1, 0);  // {bg=0}; overwritten if pipeline runs to completion
        std::chrono::steady_clock::time_point t_start, t_now;
        if (capture_stages) t_start = std::chrono::steady_clock::now();
        auto stage = [&](const char* name) {
            if (!capture_stages) return;
            t_now = std::chrono::steady_clock::now();
            last_stages_.emplace_back(name,
                std::chrono::duration<double, std::milli>(t_now - t_start).count());
            t_start = t_now;
        };
        // Parse `extra_edges` HERE — before the GIL release. We need
        // the GIL to touch any Python object, including the
        // py::array_t::ensure() conversion. After this block we hold
        // raw pointers + ownership-keeping array_t for the duration of
        // the GIL-released compute.
        int32_t n_extra = 0;
        const int32_t* extra_ptr = nullptr;
        py::array_t<int32_t> extra_arr_holder;
        if (!extra_edges_obj.is_none()) {
            extra_arr_holder = py::array_t<int32_t,
                py::array::c_style | py::array::forcecast>::ensure(extra_edges_obj);
            if (extra_arr_holder) {
                const auto eb = extra_arr_holder.request();
                if (eb.ndim == 2 && eb.shape[1] == 2) {
                    n_extra = static_cast<int32_t>(eb.shape[0]);
                    extra_ptr = static_cast<const int32_t*>(eb.ptr);
                }
            }
        }

        bool early_exit_empty = false;
        {
            py::gil_scoped_release release;

            // 0a. Cast input dtype → int32 (in expand_bufs_.lbl()) AND
            // capture the bg pattern (input == 0) into bg_mask_, all in
            // one parallel pass. For int32 input the cast is still a
            // straight copy (with bg-mask write); the alternative
            // pattern of "skip the copy when input is int32" was a tiny
            // saving but cost us the multi-dtype generality.
            //
            // Note: I tried fusing the min/max reduce from format_labels
            // into this pass to "save a redundant read." It regressed:
            // the data is still cache-hot from this pass when format
            // does its own reduce, so we save no memory traffic, while
            // the per-element min/max compares slow this kernel down.
            expand_bufs_.resize(total);
            bg_mask_.resize(static_cast<size_t>(total));
            int32_t* expanded = expand_bufs_.lbl();
            uint8_t* bg = bg_mask_.data();
            dispatch_int_dtype(buf.format, buf.itemsize, "Solver.label",
                [&](auto* tag) {
                    using T = std::remove_pointer_t<decltype(tag)>;
                    ncolor_cpp::cast_with_bg<T>(
                        static_cast<const T*>(src_ptr), expanded, bg, total,
                        *pool_, n_threads_);
                });
            stage("cast");

            // 0b. Optional format_labels: compact nonzero labels to 1..N
            // in place inside expand_bufs_.lbl(). When format_input=False
            // the caller is asserting labels are already 1..N.
            const int32_t* expand_input = expanded;
            if (format_input) {
                const int n_labels = first_seen
                    ? ncolor_cpp::format_labels_inplace_first_seen(
                        expanded, total, *pool_, n_threads_)
                    : ncolor_cpp::format_labels_inplace(
                        expanded, total, *pool_, n_threads_);
                stage("format");
                // Empty / all-bg input: output is all zeros, no
                // expansion / coloring needed.
                if (n_labels == 0) {
                    std::memset(out_ptr, 0,
                                static_cast<size_t>(total) * sizeof(uint8_t));
                    early_exit_empty = true;
                }
            }
        if (!early_exit_empty) {

            // 1. Expand labels (Voronoi). With ``expand=False`` we skip
            // this step entirely and let find_pairs / build_csr / color
            // operate directly on the (possibly bg-heavy) cast+formatted
            // buffer. find_pairs already skips lbl==0 cells, and the bg
            // pattern is preserved through to apply_lut via bg_mask_, so
            // the only difference is that the colored output retains the
            // original bg pattern instead of the Voronoi-expanded one.
            //
            // When expand=True the ND driver dispatches on p at compile
            // time: p=1 → Saito-Toriwaki sweep; p=2 → Felzenszwalb
            // envelope. Result lands in expand_bufs_.lbl(); the
            // ``input == output`` self-copy guards inside LpExpand
            // variants make this a no-op when expand_input == expanded.
            if (expand) {
                if (expand_spur_free) {
                    // Spur-free expand: BFS dilation with connectivity
                    // check. Naturally avoids K_5-creating "starfish"
                    // convergence patterns by refusing to claim bg
                    // pixels that would have ≤1 same-label face-
                    // neighbour. Equivalent to (expand_labels +
                    // delete_spurs_labels) at the result level, but
                    // computed in a single pass that bounds growth
                    // at the connectivity check rather than fixing
                    // up afterwards. Slower than Voronoi expand for
                    // small/medium inputs (3-4× on bact/synth) but
                    // same speed as expand+despur on large dense
                    // inputs where the K_5 prevention matters most.
                    // max_rounds=spur_free_max_rounds bounds the BFS;
                    // 3 is empirically enough to prevent all K_5
                    // formations on real microscopy data.
                    if (expand_input != expanded) {
                        std::memcpy(expanded, expand_input,
                                    (size_t)total * sizeof(int32_t));
                    }
                    ncolor_cpp::expand_spur_free_nd_inplace<int32_t>(
                        expanded, shape, spur_free_max_rounds,
                        /*connectivity_threshold=*/1,
                        pool_.get(), n_threads_);
                } else if (p == 2) {
                    ncolor_cpp::expand_labels_lp<2>(expand_input, expanded, expand_bufs_, shape, *pool_, n_threads_, wrap);
                } else {
                    ncolor_cpp::expand_labels_lp<1>(expand_input, expanded, expand_bufs_, shape, *pool_, n_threads_, wrap);
                }
            }
            // (Suppress unused-var warning when expand=false — expanded
            // already equals expand_input == expand_bufs_.lbl().)
            (void)expand_input;
            stage("expand");

            // 1b. Optional label-aware despur. After expand_labels
            // fills the gaps between adjacent cells, two cells that
            // originally touched only at a single point may now share
            // a 1-pixel-wide "convergence pixel" — one cell narrows
            // to a single-same-label-neighbour pixel inside another's
            // territory. These convergence points create K_5
            // obstructions in densely-packed segmentations (5 cells
            // meeting at a corner) that prevent 4-colouring at conn=1
            // r=1. Iterative despur strips them: pixels with ≤
            // threshold same-label face-neighbours become bg. 2 iters
            // is empirically enough to break K_5 cascades on real
            // microscopy data; 20 = full convergence (more aggressive,
            // ~5× slower).
            // ``lut_lbl_ptr`` is the buffer apply_color_lut_ reads at the
            // end: by default that's the post-expand ``expanded`` buffer.
            // When despur runs we want to keep colouring the spur pixels
            // with their parent cell's colour rather than turning them
            // into bg, so we save the pre-despur labels into ``lut_lbl_``
            // and use that for the final LUT application. Despur still
            // modifies ``expanded`` in place so find_pairs / coloring
            // operate on the despurred graph.
            const int32_t* lut_lbl_ptr = expanded;
            if (despur_iters > 0 && expand) {
                lut_lbl_.assign(expanded, expanded + total);
                lut_lbl_ptr = lut_lbl_.data();
                ncolor_cpp::delete_spurs_labels_nd_inplace<int32_t>(
                    expanded, shape, /*threshold=*/1,
                    despur_iters, pool_.get(), n_threads_);
                stage("despur");
            }

            // 2. Find adjacency pairs. Parallel max-reduce first
            // (was a single-threaded 1.2 ms loop at 2048²).
            const int32_t max_label = parallel_max_label_(expanded, total);
            stage("max_scan");
            const int wobj = weight_objective;
            const int wmode = weight_mode;  // 0=Min (default), see binding kwargs
            std::vector<std::pair<int32_t, int32_t>> pairs;
            std::vector<double> pair_primary;
            std::vector<int32_t> pair_counts;
            if (wobj != 0) {
                // Fused weighted find_pairs: same parallel scan computes
                // a per-pair reducer over (d_i + d_j) at boundary pixels.
                // The reducer (min/max/mean/count/harmonic) is picked by
                // weight_mode; templated dispatch eliminates dead branches.
                using ncolor_cpp::ReduceMode;
                switch (static_cast<ReduceMode>(wmode)) {
                    case ReduceMode::Max:
                        pairs = find_pairs_weighted_<ReduceMode::Max>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            max_label, pair_primary, pair_counts); break;
                    case ReduceMode::Mean:
                        pairs = find_pairs_weighted_<ReduceMode::Mean>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            max_label, pair_primary, pair_counts); break;
                    case ReduceMode::Count:
                        pairs = find_pairs_weighted_<ReduceMode::Count>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            max_label, pair_primary, pair_counts); break;
                    case ReduceMode::Harmonic:
                        pairs = find_pairs_weighted_<ReduceMode::Harmonic>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            max_label, pair_primary, pair_counts); break;
                    case ReduceMode::MeanInv:
                        pairs = find_pairs_weighted_<ReduceMode::MeanInv>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            max_label, pair_primary, pair_counts); break;
                    case ReduceMode::Min:
                    default:
                        pairs = find_pairs_weighted_<ReduceMode::Min>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            max_label, pair_primary, pair_counts); break;
                }
            } else {
                // Unified pair-find: `connect_radius` widens the
                // neighbour offset window (Chebyshev distance) for
                // each pixel. radius=1 is the standard 8-connectivity
                // path; radius>1 catches near-adjacent cells
                // separated by a thin gap of another cell's
                // territory. Same parallel + offset-precomputed
                // kernel regardless of radius.
                pairs = find_pairs_(expanded, shape, conn, wrap,
                                     max_label, connect_radius);
            }
            // Canonical sort: orders pairs by (src, dst). Without this,
            // pairs come from HT iteration (slot order = hash-mixed,
            // dependent on offset enumeration order). Same EDGE SET
            // produces different CSR neighbour-iteration order, which
            // TabuCol's heuristic can be sensitive to. Sorting gives
            // determinism + matches the legacy weight-find_pairs path
            // that emitted in canonical order via sort+unique.
            std::sort(pairs.begin(), pairs.end());
            stage("find_pairs");
            static bool dbg_pairs = std::getenv("NCOLOR_DEBUG_PAIRS") != nullptr;
            if (dbg_pairs) {
                std::fprintf(stderr, "[ncolor] find_pairs: %zu pairs "
                              "(conn=%d, radius=%d, ndim=%d)\n",
                              pairs.size(), conn, connect_radius,
                              (int)shape.size());
            }

            // 3. Build CSR (labels are 1..max_label after expand → node = label-1).
            const int32_t N = max_label;
            int32_t M = static_cast<int32_t>(pairs.size());

            // `extra_edges` parsed above (pre-GIL-release) into n_extra
            // and extra_ptr. Splice the extras after the connect() pairs.
            src_idx_.resize(static_cast<size_t>(M) + n_extra);
            dst_idx_.resize(static_cast<size_t>(M) + n_extra);
            for (int32_t i = 0; i < M; ++i) {
                src_idx_[i] = pairs[i].first - 1;
                dst_idx_[i] = pairs[i].second - 1;
            }
            for (int32_t e = 0; e < n_extra; ++e) {
                int32_t a = extra_ptr[2 * e]     - 1;
                int32_t b = extra_ptr[2 * e + 1] - 1;
                if (a < 0 || b < 0 || a >= N || b >= N || a == b) {
                    // Skip invalid entries.
                    continue;
                }
                src_idx_[M] = a;
                dst_idx_[M] = b;
                ++M;
            }
            src_idx_.resize(M);
            dst_idx_.resize(M);
            // Boundary-weighted opt-in path: convert per-pair reducer
            // values (collected during find_pairs) to weights per
            // ``weight_mode`` and build a CSR with parallel weights[].
            //   Min/Max:  w = 1 / (1 + primary)             (inverse-distance)
            //   Mean:     w = 1 / (1 + primary / counts)    (inverse-mean)
            //   Count:    w = counts                        (boundary length)
            //   Harmonic: w = primary                       (Σ 1/(1+d))
            std::vector<double> pair_w;
            const double* edge_weights_ptr = nullptr;
            if (wobj != 0 && M > 0) {
                using ncolor_cpp::ReduceMode;
                pair_w.resize(static_cast<size_t>(M));
                const auto mode = static_cast<ReduceMode>(wmode);
                for (int32_t i = 0; i < M; ++i) {
                    if (mode == ReduceMode::Mean) {
                        const double mean = pair_counts[i] > 0
                            ? pair_primary[i] / static_cast<double>(pair_counts[i])
                            : 0.0;
                        pair_w[i] = 1.0 / (1.0 + mean);
                    } else if (mode == ReduceMode::Count) {
                        pair_w[i] = static_cast<double>(pair_counts[i]);
                    } else if (mode == ReduceMode::Harmonic) {
                        pair_w[i] = pair_primary[i];
                    } else if (mode == ReduceMode::MeanInv) {
                        // Length-normalized harmonic: mean of 1/(1+d) over
                        // boundary pixels. Removes the "long-boundary bias"
                        // of plain Harmonic, so peripheral cells with much
                        // Voronoi-extended boundary aren't penalised.
                        pair_w[i] = pair_counts[i] > 0
                            ? pair_primary[i] / static_cast<double>(pair_counts[i])
                            : 0.0;
                    } else {  // Min or Max
                        pair_w[i] = 1.0 / (1.0 + pair_primary[i]);
                    }
                }
                ncolor_cpp::build_csr_from_pairs_weighted(
                    src_idx_.data(), dst_idx_.data(), pair_w.data(),
                    N, M, indptr_, indices_, edge_weights_);
                edge_weights_ptr = edge_weights_.data();
            } else {
                ncolor_cpp::build_csr_from_pairs(src_idx_.data(), dst_idx_.data(),
                                                 N, M, indptr_, indices_);
            }
            stage("build_csr");

            // Resolve de_table override or fall back to the viridis default.
            std::vector<double> de_table_vec;
            const double* de_ptr = nullptr;
            if (wobj != 0) {
                if (!de_table_obj.is_none()) {
                    auto arr = py::array_t<double,
                        py::array::c_style | py::array::forcecast>::ensure(de_table_obj);
                    if (arr) {
                        const auto buf = arr.request();
                        if (buf.ndim == 2 &&
                            buf.shape[0] == n_colors + 1 &&
                            buf.shape[1] == n_colors + 1) {
                            const size_t total_de = static_cast<size_t>(n_colors + 1) * (n_colors + 1);
                            de_table_vec.assign(static_cast<const double*>(buf.ptr),
                                                static_cast<const double*>(buf.ptr) + total_de);
                            de_ptr = de_table_vec.data();
                        }
                    }
                }
                if (de_ptr == nullptr) {
                    de_table_vec.assign(static_cast<size_t>(n_colors + 1) * (n_colors + 1), 0.0);
                    const double viridis_de4[5][5] = {
                        {0.0,   0.0,   0.0,   0.0,   0.0},
                        {0.0,   0.0,  52.0, 104.74, 133.36},
                        {0.0,  52.0,   0.0,  56.28, 100.98},
                        {0.0, 104.74, 56.28,  0.0,  62.58},
                        {0.0, 133.36,100.98, 62.58,  0.0},
                    };
                    const int K = std::min(n_colors + 1, 5);
                    for (int i = 0; i < K; ++i)
                        for (int j = 0; j < K; ++j)
                            de_table_vec[i * (n_colors + 1) + j] = viridis_de4[i][j];
                    // For n_colors > 4 (3D fallback), entries beyond 4 stay 0:
                    // those colours then contribute no contrast preference,
                    // which is acceptable — the weight_obj only matters when
                    // a perceptual palette is provided explicitly.
                    de_ptr = de_table_vec.data();
                }
            }

            // 4. Coloring: BFS + repair fallback × attempts_per_n random
            // offsets, retrying with cur_n+1 if all attempts fail. See
            // ``solve_coloring_`` for full algorithm + parallel-attempt
            // dispatch.
            n_used = solve_coloring_(N, M, n_colors, max_depth, rand_period,
                                     balance, color_mode,
                                     static_cast<int>(shape.size()), wrap,
                                     edge_weights_ptr, de_ptr, wobj);
            stage("color");

            // 5. Build LUT (expanded[i] is in 1..N, so lut size = N+1) and
            // apply it to the expanded-label buffer. When despur ran,
            // ``lut_lbl_ptr`` points at the pre-despur copy so spur pixels
            // get their parent cell's colour (they only need to be hidden
            // from the adjacency graph, not from the final image).
            lut_.assign(static_cast<size_t>(N) + 1, 0);
            for (int32_t i = 0; i < N; ++i) lut_[i + 1] = colors_[i];
            apply_color_lut_(lut_lbl_ptr, out_ptr, total);
            stage("apply_lut");
        }  // close: if (!early_exit_empty)
        }  // close: gil_scoped_release scope
        return {std::move(out), n_used};
    }

    // Accessors for the most recent label() call. Used by the public
    // ncolor.label wrapper to satisfy return_lut / check_conflicts /
    // return_conflicts without re-running connect()/coloring.
    py::array_t<uint8_t> get_last_lut() const {
        py::array_t<uint8_t> arr(static_cast<py::ssize_t>(lut_.size()));
        std::memcpy(arr.request().ptr, lut_.data(),
                    lut_.size() * sizeof(uint8_t));
        return arr;
    }
    int get_last_n_conflicts() const { return last_n_conflicts_; }

private:
    // Validate or allocate the uint8 output buffer. Returns the (possibly
    // caller-supplied) py::array_t to write into. Throws if a supplied
    // buffer is the wrong dtype / shape / not C-contiguous.
    py::array_t<uint8_t> prepare_out_buffer_(
            py::object out_arg, const py::buffer_info& src_buf, int ndim) {
        if (out_arg.is_none()) {
            std::vector<py::ssize_t> out_shape(ndim);
            for (int d = 0; d < ndim; ++d) out_shape[d] = src_buf.shape[d];
            return py::array_t<uint8_t>(out_shape);
        }
        // Caller-supplied buffer: must be uint8, C-contiguous, exact shape.
        // Reusing an output buffer across calls saves the per-call alloc
        // (16 MiB at 4096²); useful for batch pipelines. Strict dtype check
        // (not pybind11's auto-cast) so the caller's buffer is actually the
        // one written — a silent copy would defeat the purpose of out=.
        const py::array out_view = py::cast<py::array>(out_arg);
        if (out_view.dtype().kind() != 'u' || out_view.dtype().itemsize() != 1) {
            throw std::invalid_argument("Solver.label: out buffer must be uint8");
        }
        py::array_t<uint8_t> out = py::cast<py::array_t<uint8_t>>(out_arg);
        const auto out_buf = out.request();
        if (out_buf.ndim != ndim) {
            throw std::invalid_argument(
                "Solver.label: out buffer ndim does not match input");
        }
        for (int d = 0; d < ndim; ++d) {
            if (out_buf.shape[d] != src_buf.shape[d]) {
                throw std::invalid_argument(
                    "Solver.label: out buffer shape does not match input");
            }
        }
        if (!(out.flags() & py::array::c_style)) {
            throw std::invalid_argument(
                "Solver.label: out buffer must be C-contiguous");
        }
        return out;
    }

    // Parallel max-reduce over the (already-expanded) int32 label buffer.
    // Below the threshold runs serially (dispatch overhead exceeds work).
    int32_t parallel_max_label_(const int32_t* lbl, int64_t total) {
        int32_t max_label = 0;
        if (n_threads_ <= 1 || total < 8192) {
            for (int64_t i = 0; i < total; ++i) {
                if (lbl[i] > max_label) max_label = lbl[i];
            }
            return max_label;
        }
        const size_t total_sz = static_cast<size_t>(total);
        const size_t n_chunks = static_cast<size_t>(n_threads_) *
                                ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD;
        const size_t actual_chunks = std::min(n_chunks, total_sz);
        const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
        partials_.assign(actual_chunks, 0);
        std::atomic<size_t> next{0};
        int32_t* partials_ptr = partials_.data();
        pool_->parallel([&, partials_ptr]() {
            size_t idx;
            while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < actual_chunks) {
                const size_t i0 = idx * chunk_sz;
                const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                int32_t m = 0;
                for (size_t i = i0; i < i1; ++i) if (lbl[i] > m) m = lbl[i];
                partials_ptr[idx] = m;
            }
        });
        for (size_t i = 0; i < actual_chunks; ++i) {
            if (partials_[i] > max_label) max_label = partials_[i];
        }
        return max_label;
    }

    // Find adjacency pairs in an int32 label image. Sizes the hashtable
    // from the (ndim, conn) connectivity's forward-neighbour count and
    // the maximum label value, then dispatches to the ND scan kernel.
    // Returns {} for an empty input (max_label == 0).
    std::vector<std::pair<int32_t, int32_t>> find_pairs_(
            const int32_t* labels, const std::vector<int64_t>& shape,
            int conn, bool wrap, int32_t max_label, int radius = 1) {
        if (max_label == 0) return {};
        const int ndim = static_cast<int>(shape.size());
        const int64_t n_fwd = ncolor_cpp::detail::count_forward_neighbours(
            ndim, conn, radius);
        const int64_t ht_raw = 2 * n_fwd * static_cast<int64_t>(max_label);
        const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, MIN_HT_SIZE));
        return ncolor_cpp::find_pairs_nd_unpadded<int32_t>(
            labels, shape, conn,
            static_cast<uint64_t>(ht_size), n_threads_, *pool_, wrap, radius,
            &fp_ht_buf_);
    }

    // Weighted variant: same parallel scan also computes a per-pair
    // reducer over the boundary using the EDT distance map. ``Mode``
    // picks the reducer (min/max/mean/count/harmonic of d_i+d_j).
    // Out arrays ``primary``/``counts`` are parallel to the returned
    // pair list; the caller picks the right one per mode.
    template <ncolor_cpp::ReduceMode Mode>
    std::vector<std::pair<int32_t, int32_t>> find_pairs_weighted_(
            const int32_t* labels, const int32_t* dist,
            const std::vector<int64_t>& shape,
            int conn, bool wrap, int32_t max_label,
            std::vector<double>& primary,
            std::vector<int32_t>& counts) {
        primary.clear(); counts.clear();
        if (max_label == 0) return {};
        const int ndim = static_cast<int>(shape.size());
        const int64_t n_fwd = ncolor_cpp::detail::count_forward_neighbours(ndim, conn);
        const int64_t ht_raw = 2 * n_fwd * static_cast<int64_t>(max_label);
        const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, MIN_HT_SIZE));
        return ncolor_cpp::find_pairs_weighted_nd_unpadded<int32_t, Mode>(
            labels, dist, shape, conn,
            static_cast<uint64_t>(ht_size), n_threads_, *pool_, wrap,
            primary, counts,
            &fp_ht_buf_, &fp_primary_buf_, &fp_counts_buf_);
    }

    // Coloring loop: try the user-preferred algorithm first, switch to
    // the alternate algorithm before bumping cur_n. Welsh-Powell and
    // BFS-with-random-restarts have nearly disjoint failure modes on
    // planar adjacency graphs — running both at each cur_n nearly
    // eliminates the cur_n→cur_n+1 bump that was costing ~10% of L2
    // colorings on hard inputs.
    //
    // Slot assignment within attempts_per_n at each cur_n:
    //   balance=True   slots: [WP,  BFS, BFS, BFS]
    //                  WP first (uniform colour distribution); on failure
    //                  three BFS variants with offsets {1, 2, 3} as
    //                  fallback before bumping cur_n.
    //   balance=False  slots: [BFS, BFS, BFS, WP]
    //                  BFS first (user-preferred ordering, deterministic
    //                  ID-order visit); on failure WP with offset 3 as
    //                  last-ditch alternative before bumping cur_n.
    //
    // Lowest-index successful attempt wins, so the user-preferred algo
    // is preserved whenever it succeeds. Big graphs (N+M ≥ 500) race
    // the 4 attempts in parallel on the pool; small graphs run them
    // serially (dispatch overhead would exceed BFS work).
    //
    // ``WP must be CLEAN to count`` rule is retained: a repaired WP
    // coloring loses its uniform-distribution advantage, so we'd rather
    // fall through to a clean BFS than accept a repaired WP. Under
    // balance=False this means a conflicted-WP fallback doesn't save us
    // from bumping cur_n — but in practice WP needing repair on the
    // graphs that already failed three BFS attempts is vanishingly
    // rare.
    //
    // Side effects: writes the winning coloring into ``colors_`` and the
    // adjacency-conflict count into ``last_n_conflicts_``. Returns
    // ``n_used`` = max colour value in the winning coloring.
    int solve_coloring_(int32_t N, int32_t M, int n_colors,
                        int max_depth, int rand_period, bool balance,
                        int color_mode, int ndim, bool wrap,
                        const double* edge_weights = nullptr,
                        const double* de_table = nullptr,
                        int weight_obj = 0) {
        constexpr int attempts_per_n = 16;
        const int64_t max_iter = std::max<int64_t>(
            static_cast<int64_t>(indices_.size()) +
            static_cast<int64_t>(indptr_.size()), 512);
        // color_mode: -1 = auto (threshold-based), 0 = forced serial,
        // 1 = forced parallel. Auto threshold tuned on M2 / 20-thread
        // ForkJoinPool: below ~500 edges the BFS finishes in <100 µs and
        // dispatch overhead eats the win.
        bool color_parallel;
        if (color_mode == 0) color_parallel = false;
        else if (color_mode == 1) color_parallel = (n_threads_ > 1);
        else color_parallel = (n_threads_ > 1) &&
            (static_cast<int64_t>(N) + M >= 500);
        if (color_parallel) {
            per_attempt_colors_.resize(attempts_per_n);
            per_attempt_ok_.assign(attempts_per_n, 0);
        }

        int cur_n = n_colors;
        bool ok = false;
        static const bool dbg_solve = std::getenv("NCOLOR_SOLVE_DEBUG") != nullptr;
        // ω(G) lower bound: χ(G) ≥ ω(G). If a clique larger than the
        // user's target k exists, the graph requires ≥ ω colours and
        // we'd otherwise burn ~200 ms per (race+tabu-restart+
        // bb_dsatur+HEA) round each time we increment cur_n on the
        // way up to ω. Bron-Kerbosch with a tight deadline (10 ms)
        // returns a valid lower bound even on partial searches —
        // worst case: no time saved when ω ≤ n_colors (the common
        // case). Skipped for N > 20000 (memory) and for the weighted
        // path (perceptual objective is orthogonal to clique
        // structure).
        const bool wobj_active_for_clique = weight_obj != 0 && edge_weights != nullptr;
        if (!wobj_active_for_clique && N >= 5) {
            const auto clb_t0 = std::chrono::steady_clock::now();
            const int64_t clb_deadline_ns =
                clb_t0.time_since_epoch().count() + 10LL * 1000LL * 1000LL;
            const int omega = ncolor_cpp::clique_lower_bound(
                N, indptr_.data(), indices_.data(),
                /*target=*/n_colors + 1, clb_deadline_ns);
            if (dbg_solve) {
                const double clb_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - clb_t0).count();
                std::fprintf(stderr,
                    "[clique-lb] N=%d ω≥%d (target=%d) %.1fms\n",
                    N, omega, n_colors + 1, clb_ms);
            }
            if (omega > cur_n) cur_n = omega;
        }
        for (int depth = 0; depth < max_depth && !ok; ++depth) {
            const auto depth_t0 = std::chrono::steady_clock::now();
            if (color_parallel) {
                const int local_cur_n = cur_n;
                const int local_depth = depth;
                const int32_t* ip = indptr_.data();
                const int32_t* ix = indices_.data();

                // Shared early-exit flag: as soon as any parallel
                // attempt finds a 0-conflict colouring, other workers
                // (a) skip subsequent slots and (b) abort any
                // in-flight per-attempt tabucol. Without (b), race
                // latency = slowest worker's full tabucol budget; with
                // it, race latency = first success.
                std::atomic<bool> winner_found{false};

                // Run-one-attempt body factored out so we can call it
                // once sequentially as a warmup before paying pool-
                // dispatch overhead, and dispatch the rest in parallel
                // only on warmup failure.
                auto run_one_attempt = [&, local_cur_n, local_depth, ip, ix](
                        int idx, const std::atomic<bool>* cancel,
                        bool allow_tabucol, int64_t race_deadline_ns) -> bool {
                    auto& cv = per_attempt_colors_[idx];
                    const int attempt_offset = local_depth + idx;
                    // Special slot: branch-and-bound exact coloring
                    // racing the BFS+tabucol slots. For K_5-free
                    // planar-ish cell-adjacency graphs at our scale
                    // (N ≤ few thousand), B&B+DSatur typically finds
                    // a k-coloring in <1 ms — often faster than
                    // BFS+tabucol convergence. When the search tree
                    // blows up on adversarial vertex orderings, the
                    // node budget + cancel-flag short-circuit kick in
                    // and one of the BFS slots wins instead. Only
                    // fires at the user's target k (no value running
                    // exact search at cur_n > n_colors). Disabled
                    // when the user opted into a perceptual weight
                    // objective, since bb_dsatur doesn't honour
                    // weights. Slot picked: the LAST one for
                    // balance=True (free slot — slot 0 is the WP
                    // warmup, slots 1..N-2 are BFS); not used for
                    // balance=False (LAST slot is the WP fallback
                    // there). Cancel propagation: bb_dsatur checks
                    // the cancel flag every 1024 nodes.
                    const bool wobj_active = weight_obj != 0 && edge_weights != nullptr;
                    const bool bb_slot = balance && !wobj_active
                                         && (idx == attempts_per_n - 1)
                                         && local_cur_n == n_colors;
                    if (bb_slot) {
                        const int64_t node_budget = std::max<int64_t>(
                            20000, (int64_t)N * 30);
                        const bool bb_ok = ncolor_cpp::bb_dsatur(
                            ip, ix, N, local_cur_n, cv, node_budget,
                            cancel, race_deadline_ns);
                        per_attempt_ok_[idx] = bb_ok ? 1 : 0;
                        return bb_ok;
                    }
                    // Slot 0 = user-preferred algorithm; remaining
                    // slots = alternate algorithm with different
                    // random offsets (algorithm-switching fallback).
                    // For the weighted opt-in, slots 0-(attempts-2)
                    // use weighted-WP with different offsets; the
                    // LAST slot is a pure-balance WP fallback. If
                    // all weighted attempts fail at n_colors, the
                    // balance attempt can still produce a clean
                    // 4-coloring before we bump cur_n.
                    const bool wp = wobj_active
                        ? true
                        : (balance ? (idx == 0)
                                   : (idx == attempts_per_n - 1));
                    const bool weighted_attempt = wobj_active &&
                                                  (idx < attempts_per_n - 1);
                    const double* w_ptr = weighted_attempt ? edge_weights : nullptr;
                    const double* de_ptr = weighted_attempt ? de_table : nullptr;
                    const int w_obj_local = weighted_attempt ? weight_obj : 0;
                    const bool finished = ncolor_cpp::color_graph_csr_legacy(
                        ip, ix, N, local_cur_n, rand_period,
                        attempt_offset, max_iter, cv, wp,
                        w_ptr, de_ptr, w_obj_local);
                    const bool conflict = !finished ||
                        ncolor_cpp::has_conflict_csr(ip, ix, N, cv.data());
                    bool a_ok = !conflict || ncolor_cpp::repair_coloring(
                        ip, ix, N, local_cur_n, std::max(4, max_depth), cv);
                    // Per-attempt TabuCol fallback when greedy+repair
                    // fail at the user's target k. Each attempt runs
                    // its own small-budget tabucol seeded uniquely,
                    // turning the parallel block into a true K-way
                    // race of (greedy+repair+tabucol) jobs. Only
                    // fires for the user's target k. The `cancel`
                    // pointer lets the tabucol loop short-circuit
                    // when a sibling worker has already won.
                    // Disabled for the warmup attempt (allow_tabucol
                    // false): the warmup is meant to be a sub-ms
                    // fast-path; if WP+repair fails we'd rather fall
                    // through to the parallel race (which has
                    // sibling-cancellable tabucol) than burn up to
                    // 5 ms of un-cancellable budget here.
                    if (!a_ok && local_cur_n == n_colors && allow_tabucol) {
                        for (int32_t u = 0; u < N; ++u) {
                            if (cv[u] < 1 || cv[u] > local_cur_n) {
                                cv[u] = (uint8_t)(1 + (u % local_cur_n));
                            }
                        }
                        const int per_attempt_tabu_iters = std::min(
                            5000, std::max(500, N * 5));
                        const uint64_t tabu_seed =
                            (uint64_t)(attempt_offset + 1)
                                * 0x9e3779b97f4a7c15ULL
                            ^ (uint64_t)(local_depth + 1)
                                * 0x517cc1b727220a95ULL;
                        if (ncolor_cpp::tabucol(
                                ip, ix, N, local_cur_n,
                                per_attempt_tabu_iters, cv, tabu_seed,
                                race_deadline_ns, cancel)) {
                            a_ok = true;
                        }
                    }
                    // WP must be clean for the balanced path — repair
                    // would shadow its uniform-distribution promise.
                    // For the weighted path the user has opted in to
                    // a perceptual objective and accepts repair.
                    const bool clean_wp_required = wp && !weighted_attempt;
                    const bool slot_ok = a_ok && (!clean_wp_required || !conflict);
                    per_attempt_ok_[idx] = slot_ok ? 1 : 0;
                    return slot_ok;
                };

                // Conditional fan-out: try slot 0 sequentially first.
                // The vast majority of inputs (including all bact-logo
                // shuffles at radius=1) succeed on the first attempt;
                // skipping the pool dispatch in that case recovers
                // ~0.5 ms of median latency that the 16-way race
                // unconditionally pays. Only fan out when warmup fails.
                static const bool dbg_warmup = std::getenv("NCOLOR_WARMUP_DEBUG") != nullptr;
                const auto warmup_t0 = std::chrono::steady_clock::now();
                const bool warmup_ok = run_one_attempt(
                    0, nullptr, /*allow_tabucol=*/false, /*race_deadline_ns=*/0);
                if (dbg_solve) {
                    const double warmup_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - warmup_t0).count();
                    std::fprintf(stderr, "  [warmup] %.1fms ok=%d\n", warmup_ms, (int)warmup_ok);
                }
                if (dbg_warmup) {
                    std::fprintf(stderr,
                        "[warmup] N=%d M=%d cur_n=%d depth=%d ok=%d\n",
                        N, M, local_cur_n, local_depth, (int)warmup_ok);
                }
                if (warmup_ok) {
                    colors_.swap(per_attempt_colors_[0]);
                    ok = true;
                } else {
                    std::atomic<int> next{1};
                    static const bool dbg_slots = std::getenv("NCOLOR_SLOT_DEBUG") != nullptr;
                    std::vector<double> slot_ms(attempts_per_n, -1.0);
                    std::vector<int> slot_done(attempts_per_n, 0);
                    const auto race_t0 = std::chrono::steady_clock::now();
                    // Race wall-clock deadline shared across all slots:
                    // bounds time wasted on infeasible-at-cur_n graphs.
                    // Without this, the 16 per-attempt tabucols all run
                    // their full iter budget (up to several seconds on
                    // N≥few-thousand) trying to escape a non-k-colourable
                    // graph. 50 ms is plenty for any feasible case at
                    // our scale — successes typically converge in <5 ms.
                    const int64_t race_deadline_ns =
                        race_t0.time_since_epoch().count()
                        + 50LL * 1000LL * 1000LL;
                    pool_->parallel([&]() {
                        int idx;
                        while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < attempts_per_n) {
                            if (winner_found.load(std::memory_order_relaxed)) break;
                            const auto t0 = std::chrono::steady_clock::now();
                            const bool ok_slot = run_one_attempt(
                                idx, &winner_found, /*allow_tabucol=*/true,
                                race_deadline_ns);
                            if (dbg_slots) {
                                const auto t1 = std::chrono::steady_clock::now();
                                slot_ms[idx] = std::chrono::duration<double, std::milli>(t1 - t0).count();
                                slot_done[idx] = ok_slot ? 1 : 2;
                            }
                            if (ok_slot) {
                                winner_found.store(true, std::memory_order_relaxed);
                            }
                        }
                    });
                    if (dbg_solve) {
                        const double race_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - race_t0).count();
                        std::fprintf(stderr, "  [race] %.1fms\n", race_ms);
                    }
                    if (dbg_slots) {
                        const auto race_t1 = std::chrono::steady_clock::now();
                        const double race_ms = std::chrono::duration<double, std::milli>(race_t1 - race_t0).count();
                        std::fprintf(stderr, "[race] total=%.3fms\n", race_ms);
                        for (int a = 1; a < attempts_per_n; ++a) {
                            const char* st = slot_done[a] == 1 ? "OK"
                                           : slot_done[a] == 2 ? "FAIL"
                                           : "skip";
                            if (slot_ms[a] >= 0) {
                                std::fprintf(stderr, "  slot[%d] %.3fms %s\n", a, slot_ms[a], st);
                            }
                        }
                    }
                    // Lowest-index successful attempt wins
                    // (deterministic preference for the first random
                    // offset). Slot 0 already failed so it's skipped
                    // by the per_attempt_ok_[a] check.
                    for (int a = 0; a < attempts_per_n; ++a) {
                        if (per_attempt_ok_[a]) {
                            colors_.swap(per_attempt_colors_[a]);
                            ok = true;
                            break;
                        }
                    }
                }
            } else {
                for (int attempt = 0; attempt < attempts_per_n && !ok; ++attempt) {
                    // When weight_obj != 0: slots 0..(attempts-2) run
                    // weighted-WP (different offsets); LAST slot is a
                    // pure-balance WP fallback so a 4-colourable graph
                    // doesn't get bumped to 5 colours when broad-support
                    // reducers (count, harmonic) over-constrain the BFS.
                    const bool wobj_active = weight_obj != 0 && edge_weights != nullptr;
                    const bool wp = wobj_active
                        ? true
                        : (balance ? (attempt == 0)
                                   : (attempt == attempts_per_n - 1));
                    const bool weighted_attempt = wobj_active &&
                                                  (attempt < attempts_per_n - 1);
                    const double* w_ptr = weighted_attempt ? edge_weights : nullptr;
                    const double* de_ptr = weighted_attempt ? de_table : nullptr;
                    const int w_obj_local = weighted_attempt ? weight_obj : 0;
                    const bool finished = ncolor_cpp::color_graph_csr_legacy(
                        indptr_.data(), indices_.data(), N,
                        cur_n, rand_period, depth + attempt, max_iter,
                        colors_, wp, w_ptr, de_ptr, w_obj_local);
                    const bool conflict = !finished || ncolor_cpp::has_conflict_csr(
                        indptr_.data(), indices_.data(), N, colors_.data());
                    bool a_ok = !conflict || ncolor_cpp::repair_coloring(
                        indptr_.data(), indices_.data(), N,
                        cur_n, std::max(4, max_depth), colors_);
                    // "WP must be clean" enforces the uniform-distribution
                    // promise for the balanced path. For the weighted path
                    // the user has opted into a perceptual objective and
                    // accepts repair as part of the deal — otherwise the
                    // WP-weighted result is silently dropped in favour of a
                    // non-WP, non-weighted fallback (which defeats the point).
                    const bool clean_wp_required = wp && !weighted_attempt;
                    if (a_ok && (!clean_wp_required || !conflict)) ok = true;
                }
            }
            if (!ok && cur_n == n_colors) {
                const auto tabu_restart_t0 = std::chrono::steady_clock::now();
                // TabuCol fallback at the user's target k. Some graphs
                // are k-colourable (verified by SAT) but every
                // vertex-ordering greedy hits the same local minimum
                // (e.g. dense corner-touching cells under conn=2 with
                // K4 substructures). Tabu search rescues these by
                // allowing temporary conflict increases to escape.
                // Cost is bounded to one tabucol call per ncolor.label
                // — fast graphs never hit this branch.
                if (color_parallel) {
                    // Parallel path doesn't auto-update colors_ on
                    // failure; pull in the attempt with fewest
                    // conflicts (one of them must be sized N since
                    // color_graph_csr_legacy was called on each).
                    int best_idx = -1, best_conf = INT_MAX;
                    for (int a = 0; a < attempts_per_n; ++a) {
                        if ((int)per_attempt_colors_[a].size() < N) continue;
                        int c = 0;
                        for (int32_t i = 0; i < M; ++i) {
                            if (per_attempt_colors_[a][src_idx_[i]] ==
                                per_attempt_colors_[a][dst_idx_[i]]) ++c;
                        }
                        if (c < best_conf) { best_conf = c; best_idx = a; }
                    }
                    if (best_idx < 0) goto skip_tabucol;
                    colors_.swap(per_attempt_colors_[best_idx]);
                }
                // Sanity-check colors_ before handing to TabuCol; bail
                // out if any vertex has color 0 or > cur_n (would
                // corrupt the conf[] accumulator). Vertices are
                // 0-indexed (matches the rest of the C++ pipeline).
                if ((int)colors_.size() < N) goto skip_tabucol;
                for (int32_t u = 0; u < N; ++u) {
                    if (colors_[u] < 1 || colors_[u] > cur_n) goto skip_tabucol;
                }
                {
                    // Tabu-search restart loop. First restart uses the
                    // conflicted-greedy colouring as a starting point
                    // (often within a few moves of valid). Subsequent
                    // restarts use a fresh uniform-random colouring so
                    // they sample different basins.
                    //
                    // Total time is capped by a shared wall-clock budget
                    // (default 200 ms) so dense graphs that don't
                    // 4-colour quickly fall through to cur_n bump
                    // instead of burning seconds. Each restart still has
                    // its own iter cap as a secondary bound.
                    const int per_seed_iters = std::min(
                        50000, std::max(2000, N * 30));
                    // 0 ms wall budget — bb_dsatur + HEA fallbacks
                    // cover the same role at lower cost (HEA's
                    // built-in init-population calls tabucol per
                    // member, replicating the restart-loop effect
                    // with crossover diversity on top). Skipping this
                    // loop saves 50 ms on infeasible-at-k graphs. If
                    // a regression on feasible-but-hard cases is
                    // observed in stress testing, bump back to ~10 ms
                    // for a cheap safety net.
                    const int64_t budget_ns = 0LL;
                    const int64_t deadline_ns =
                        std::chrono::steady_clock::now()
                            .time_since_epoch().count() + budget_ns;
                    std::vector<uint8_t> saved = colors_;
                    uint64_t base_seed =
                        (uint64_t)(depth + 1) * 0x9e3779b97f4a7c15ULL;
                    for (int s = 0; s < 24 && !ok; ++s) {
                        if (std::chrono::steady_clock::now()
                                .time_since_epoch().count() > deadline_ns) break;
                        uint64_t rs = base_seed + (uint64_t)s * 0xdeadbeefcafebabeULL;
                        auto next32 = [&]() -> uint32_t {
                            rs = rs * 6364136223846793005ULL + 1442695040888963407ULL;
                            return (uint32_t)(rs >> 32);
                        };
                        if (s == 0) {
                            colors_ = saved;
                        } else {
                            for (int32_t u = 0; u < N; ++u) {
                                colors_[u] = (uint8_t)(1 + (next32() % (uint32_t)cur_n));
                            }
                        }
                        if (ncolor_cpp::tabucol(
                                indptr_.data(), indices_.data(), N,
                                cur_n, per_seed_iters, colors_, rs,
                                deadline_ns)) {
                            ok = true;
                        }
                    }
                }
                skip_tabucol: ;
                if (dbg_solve) {
                    const double tabu_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tabu_restart_t0).count();
                    std::fprintf(stderr, "  [tabu-restart] %.1fms ok=%d\n", tabu_ms, (int)ok);
                }
                // bb_dsatur + HEA fallback chain. Wall-clock deadlines
                // bound both: bb_dsatur's per-node cost is O(N) (pick_
                // next is a linear scan), so a fixed node budget
                // translates to multi-second wall time on graphs with
                // N in the thousands. A graph that is genuinely NOT
                // k-colourable would otherwise burn the entire
                // bb_dsatur node budget AND HEA's generation budget
                // proving infeasibility before cur_n bumps to k+1.
                // The 50 ms-each deadlines cap the wasted time per
                // cur_n bump at ~100 ms; the feasible-but-hard cases
                // (e.g. the 2 adversarial shuffles in our stress test
                // where HEA solves in 25-37 ms) still finish well
                // within budget. Fires only when every cheaper path
                // has failed → zero cost on the common path.
                const int64_t bb_budget_ns =
                    50LL * 1000LL * 1000LL;  // 50 ms
                // HEA gets 100 ms — empirically the trickiest
                // adversarial shuffles (which bb_dsatur exhausts its
                // budget on) need 25-90 ms of HEA to converge; 50 ms
                // dropped a previously-passing shuffle.
                const int64_t hea_budget_ns =
                    100LL * 1000LL * 1000LL;
                if (!ok) {
                    const int64_t node_budget = std::max<int64_t>(
                        200000, (int64_t)N * 100);
                    std::vector<uint8_t> bb_colors;
                    static const bool dbg_bb = std::getenv("NCOLOR_BB_DEBUG") != nullptr;
                    const auto bb_t0 = std::chrono::steady_clock::now();
                    const int64_t bb_deadline_ns =
                        bb_t0.time_since_epoch().count() + bb_budget_ns;
                    const bool bb_ok = ncolor_cpp::bb_dsatur(
                            indptr_.data(), indices_.data(),
                            N, cur_n, bb_colors, node_budget,
                            /*cancel=*/nullptr, bb_deadline_ns);
                    if (dbg_bb) {
                        const auto bb_t1 = std::chrono::steady_clock::now();
                        const double bb_ms = std::chrono::duration<double, std::milli>(
                            bb_t1 - bb_t0).count();
                        std::fprintf(stderr,
                            "[bb_dsatur] N=%d cur_n=%d budget=%lld ok=%d %.2fms\n",
                            N, cur_n, (long long)node_budget, (int)bb_ok, bb_ms);
                    }
                    if (bb_ok) {
                        colors_ = std::move(bb_colors);
                        ok = true;
                    }
                }
                if (!ok) {
                    std::vector<uint8_t> hea_colors;
                    const uint64_t hea_seed =
                        ((uint64_t)N * 0x9e3779b97f4a7c15ULL)
                        ^ ((uint64_t)(depth + 1) * 0xc6a4a7935bd1e995ULL);
                    static const bool dbg_hea = std::getenv("NCOLOR_BB_DEBUG") != nullptr;
                    const auto hea_t0 = std::chrono::steady_clock::now();
                    const int64_t hea_deadline_ns =
                        hea_t0.time_since_epoch().count() + hea_budget_ns;
                    const bool hea_ok = ncolor_cpp::hea(
                            indptr_.data(), indices_.data(),
                            N, cur_n, hea_colors,
                            /*max_generations=*/80,
                            /*pop_size=*/8,
                            /*init_tabu_iters=*/500,
                            /*gen_tabu_iters=*/2000,
                            hea_seed, hea_deadline_ns);
                    if (dbg_hea) {
                        const auto hea_t1 = std::chrono::steady_clock::now();
                        const double hea_ms = std::chrono::duration<double, std::milli>(
                            hea_t1 - hea_t0).count();
                        std::fprintf(stderr,
                            "[hea] N=%d cur_n=%d ok=%d %.2fms\n",
                            N, cur_n, (int)hea_ok, hea_ms);
                    }
                    if (hea_ok) {
                        colors_ = std::move(hea_colors);
                        ok = true;
                    }
                }
            }
            if (dbg_solve) {
                const double depth_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - depth_t0).count();
                std::fprintf(stderr,
                    "[solve] depth=%d cur_n=%d ok=%d total=%.1fms\n",
                    depth, cur_n, (int)ok, depth_ms);
            }
            if (!ok) {
                ++cur_n;
                // ndim-aware floor on the FIRST failure only. Planar
                // (ndim=2) inputs hit ≤ 4 colours by the 4-colour theorem,
                // so the floor is a no-op there. For ndim ≥ 3 there's no
                // such bound; empirically dense-blob inputs hit ~3·ndim − 2
                // colours (more with wrap), so jumping directly to that
                // floor skips 2-5 doomed sequential attempts on the
                // fallback path. Triggered only after depth==0 fails, so
                // user-supplied n_colors and planar workloads remain
                // bit-identical to the pre-patch behaviour.
                if (depth == 0 && ndim >= 3) {
                    const int floor_n = 3 * ndim - 2 + (wrap ? 1 : 0);
                    if (cur_n < floor_n) cur_n = floor_n;
                }
            }
        }

        int n_used = 0;
        for (uint8_t c : colors_) if (c > n_used) n_used = c;
        // Cheap O(M) tally of adjacent same-colour pairs. Matches the
        // legacy ``conflicts = sum(lut[pairs[:,0]] == lut[pairs[:,1]])``
        // since src_idx_/dst_idx_ are exactly the pairs from connect()
        // (shifted to 0-based) and colors_[k] == lut_[k+1].
        last_n_conflicts_ = 0;
        for (int32_t i = 0; i < M; ++i) {
            if (colors_[src_idx_[i]] == colors_[dst_idx_[i]]) ++last_n_conflicts_;
        }
        return n_used;
    }

    // Apply the colour LUT to ``expanded[i]``: bg pixels (bg_mask_[i]==1)
    // get colour 0; foreground pixels get ``lut_[expanded[i]]``. Parallel
    // when total ≥ 8192. The bg pattern was captured by ``cast_with_bg``
    // at the start of label(); using a uint8 mask here keeps the inner
    // loop typeless wrt the original input dtype.
    void apply_color_lut_(const int32_t* expanded, uint8_t* out_ptr,
                          int64_t total) {
        const int nt = std::max(1, n_threads_);
        const uint8_t* bg_p = bg_mask_.data();
        const uint8_t* lp = lut_.data();
        if (nt == 1 || total < 8192) {
            for (int64_t i = 0; i < total; ++i) {
                out_ptr[i] = bg_p[i] ? 0 : lp[expanded[i]];
            }
            return;
        }
        ncolor_cpp::dispatch_parallel(*pool_, static_cast<size_t>(total),
            static_cast<size_t>(nt) * ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD,
            [bg_p, expanded, lp, out_ptr](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    out_ptr[i] = bg_p[i] ? 0 : lp[expanded[i]];
                }
            });
    }

    int n_threads_;
    std::unique_ptr<ncolor_cpp::ForkJoinPool> pool_;
    ncolor_cpp::ExpandBuffers expand_bufs_;
    std::vector<uint8_t> bg_mask_;     // captured from cast, used by apply_lut
    std::vector<int32_t> partials_;     // max-reduce partials, reused across calls
    std::vector<int32_t> src_idx_, dst_idx_;
    std::vector<int32_t> indptr_, indices_;
    // Optional parallel-to-indices_ edge weights used by the boundary-
    // weighted coloring path. Empty/unused for the default coloring.
    // Real-valued to encode EDT-distance-based contact strength.
    std::vector<double> edge_weights_;
    std::vector<uint8_t> colors_;
    std::vector<uint8_t> lut_;
    // Pre-despur copy of the expanded-label buffer, used by apply_lut so
    // spur pixels (zeroed out of the despurred working buffer) still get
    // their parent cell's colour in the final image. Only populated when
    // ``despur_iters > 0``; otherwise apply_lut reads ``expanded`` directly.
    std::vector<int32_t> lut_lbl_;
    // Persistent per-thread hashtable buffer for find_pairs (n_threads_ *
    // ht_size entries). Reused across calls so we don't pay malloc/free
    // for ~tens of MB on every label() invocation. find_pairs itself
    // re-initialises per-thread slots to HT_EMPTY at the start of each
    // scan, so leaving stale data here between calls is safe.
    std::vector<uint64_t> fp_ht_buf_;
    // Companion scratch for the boundary-weighted find_pairs path only;
    // empty / unused when weight_objective==0 (the default).
    std::vector<double>   fp_primary_buf_;
    std::vector<int32_t>  fp_counts_buf_;
    int last_n_conflicts_ = 0;
    std::vector<std::pair<std::string, double>> last_stages_;
    // Per-attempt scratch for parallel coloring (one colors vector per
    // racing attempt). Reused across calls.
    std::vector<std::vector<uint8_t>> per_attempt_colors_;
    std::vector<int> per_attempt_ok_;
};

PYBIND11_MODULE(_impl, m) {
    m.doc() = "ncolor C++ engine: connect / expand / color pipeline + "
              "ForkJoinPool. Public Python API in ncolor.color, ncolor.expand "
              "wraps the engines exposed here.";
    py::class_<ExpandEngine>(m, "ExpandEngine",
        "Persistent threadpool wrapper for expand_labels + format_labels.\n"
        "One engine per pipeline; the pool and intermediate buffers are\n"
        "reused across calls.")
        .def(py::init<double>(), py::arg("n_threads") = -1.0)
        .def_property_readonly("n_threads", &ExpandEngine::n_threads)
        .def("expand_labels", &ExpandEngine::expand_labels,
             py::arg("labels"), py::arg("p") = 2, py::arg("wrap") = false,
             "Voronoi label expansion under L_p metric. p=1 (Manhattan,\n"
             "Saito-Toriwaki sweep) or p=2 (Euclidean², Felzenszwalb\n"
             "envelope). Same ND driver, dispatched at compile time on p.\n"
             "Default p=2 matches numba's expand_labels(metric='l2').\n"
             "wrap=True makes the expansion toroidal: cells whose Voronoi\n"
             "territories cross the image edge wrap to the opposite side.\n"
             "Both metrics implement this natively in the cpp envelope/\n"
             "chamfer kernels (no Python-level padding): L1 ~1.1× std,\n"
             "L2 ~1.4-1.6× std. Verified bit-equal to a np.pad reference\n"
             "on standard inputs.")
        .def("expand_labels_with_dist", &ExpandEngine::expand_labels_with_dist,
             py::arg("labels"), py::arg("p") = 2, py::arg("wrap") = false,
             "Same as expand_labels but also returns the distance field.\n"
             "Returns (labels: int32, dist: float64), both shape = input shape.\n"
             "p=2 returns Euclidean (sqrt of squared); p=1 returns L1.")
        .def("per_class_min_edt", &ExpandEngine::per_class_min_edt,
             py::arg("labels"), py::arg("class_of"), py::arg("n_classes"),
             py::arg("p") = 2, py::arg("wrap") = false,
             "Per-class minimum distance fields.\n"
             "class_of is a length-(max_label+1) array; class_of[u] in 1..K is\n"
             "the class of label u (0 = exclude). Returns (K, *labels.shape)\n"
             "float64 with out[c-1, ...] = distance from each pixel to the\n"
             "nearest pixel of any label whose class is c. Pixels for classes\n"
             "with no seeds get +inf. Cost: K expand passes.")
        .def("pairwise_nearest_distance", &ExpandEngine::pairwise_nearest_distance,
             py::arg("labels"), py::arg("n_labels"),
             py::arg("p") = 2, py::arg("wrap") = false,
             "Pairwise N×N matrix D[a, b] = min distance from any pixel of\n"
             "label a+1 to any pixel of label b+1. Diagonal is 0; missing\n"
             "labels yield +inf rows/columns. Cost: N expand passes.")
        .def("format_labels", &ExpandEngine::format_labels,
             py::arg("labels"), py::arg("first_seen") = false,
             "Compact nonzero labels to 1..N. If min(labels) != 0 the\n"
             "min is treated as background and everything is shifted\n"
             "before compaction.\n"
             "Default (first_seen=False) uses ascending-source numbering\n"
             "(parallel build, faster); the new label is the source's\n"
             "rank among present values. first_seen=True uses input-order\n"
             "numbering matching fastremap.renumber bit-for-bit (serial\n"
             "build, ~2× slower) — opt in when bit-equality matters.\n"
             "Accepts uint8/uint16/uint32, int8/int16/int32, int64 input;\n"
             "cast to int32 happens in parallel inside the released-GIL\n"
             "block. Returns (formatted_array, n_labels).");

    py::class_<Solver>(m, "Solver",
        "End-to-end ncolor.label() equivalent. Wraps a single ThreadPool\n"
        "and re-uses all intermediate buffers, so the per-call cost is just\n"
        "task enqueue + the actual work. Returns (colored_image_uint8,\n"
        "n_colors_used).\n"
        "\n"
        "Supports any ndim ≥ 2. conn ∈ [1, ndim] with\n"
        "scipy.ndimage.generate_binary_structure semantics (e.g. 2D conn=2\n"
        "is 8-connectivity; 3D conn=3 is 26-connectivity).\n"
        "\n"
        "n_threads conventions:\n"
        "  -1 (default), 0, negative  → auto (use cached calibration)\n"
        "  0 < x < 1                  → fraction × os.cpu_count() (e.g. 0.5)\n"
        "  1                          → serial\n"
        "  N >= 1                     → exact thread count")
        .def(py::init<double>(), py::arg("n_threads") = -1.0)
        .def_property_readonly("n_threads", &Solver::n_threads)
        .def("label", &Solver::label,
             py::arg("mask"), py::arg("n_colors") = 4,
             py::arg("max_depth") = 30, py::arg("rand_period") = 10,
             py::arg("conn") = 2,
             py::arg("p") = 2, py::arg("capture_stages") = false,
             py::arg("format_input") = true, py::arg("expand") = true,
             py::arg("out") = py::none(), py::arg("color_mode") = -1,
             py::arg("wrap") = false, py::arg("balance") = false,
             py::arg("first_seen") = false,
             py::arg("weight_objective") = 0,
             py::arg("de_table") = py::none(),
             py::arg("weight_mode") = 1,
             py::arg("extra_edges") = py::none(),
             py::arg("connect_radius") = 1,
             py::arg("despur_iters") = 2,
             py::arg("expand_spur_free") = false,
             py::arg("spur_free_max_rounds") = 1,
             "Run [format_labels →] [expand →] connect → CSR → color → apply LUT.\n"
             "Any ndim ≥ 2; conn ∈ [1, ndim].\n"
             "p selects the expand metric: p=1 (Saito-Toriwaki sweep,\n"
             "Manhattan, default) or p=2 (Felzenszwalb parabolic envelope,\n"
             "Euclidean²) — different boundary placement at ties.\n"
             "format_input=True (default) compacts non-sequential nonzero\n"
             "labels to 1..N in-place inside the released-GIL section.\n"
             "Precondition: bg=0 in the input. Pass format_input=False if\n"
             "labels are already 1..N (saves ~one full-image pass).\n"
             "Background masking (output=0 wherever input=0) is always\n"
             "applied alongside the LUT in the final stage.\n"
             "out: optional preallocated uint8 array of the same shape as\n"
             "mask. If supplied, results are written there and returned\n"
             "instead of allocating a new array — useful for batch\n"
             "pipelines that reuse the same output buffer across calls.\n"
             "wrap=True treats the image as a torus (left/right edges are\n"
             "neighbours, top/bottom edges are neighbours), adding wrap-\n"
             "around adjacencies between cells whose Voronoi territories\n"
             "land on opposite image edges. Useful for tile-equivalent or\n"
             "periodic-imaging assumptions; balances colour frequencies on\n"
             "tightly-cropped microcolony images at ~zero runtime cost.\n"
             "balance=True visits cells in descending-degree order during\n"
             "the BFS coloring (Welsh-Powell heuristic). High-degree (most\n"
             "constrained) cells are coloured first, which spreads colour\n"
             "usage more evenly across the graph. ~zero runtime cost\n"
             "(O(N) bucket sort). Recommended for visual uniformity.")
        .def("connect", &Solver::connect,
             py::arg("mask"), py::arg("conn") = 1, py::arg("wrap") = false,
             "Adjacency pairs for a label image. Returns an (M, 2) int32\n"
             "array of unique (lo, hi) label pairs that share a boundary\n"
             "under connectivity ``conn``. Mirrors ncolor.connect()'s\n"
             "signature; runs the cpp connect kernel directly.\n"
             "wrap=True treats the image as a torus (opposite edges are\n"
             "adjacent), adding wrap-around pairs between cells on the\n"
             "image perimeter.")
        .def("get_last_stages", &Solver::get_last_stages,
             "Per-stage timing breakdown from the most recent label() call\n"
             "made with capture_stages=True.")
        .def("get_last_lut", &Solver::get_last_lut,
             "label→color LUT from the most recent label() call. uint8\n"
             "array of length (max_label + 1). lut[0] = 0 (bg); lut[k] is\n"
             "the color assigned to formatted-label k for k = 1..max_label.")
        .def("get_last_n_conflicts", &Solver::get_last_n_conflicts,
             "Number of adjacent same-color pairs in the most recent\n"
             "label() output. 0 means the coloring is valid; nonzero\n"
             "means the solver bailed out without finding a clean coloring.");

    m.def("cc_label",
          [](py::array mask, int conn) -> std::pair<py::array_t<int32_t>, int32_t> {
              if (!(mask.flags() & py::array::c_style)) {
                  mask = py::array::ensure(mask, py::array::c_style);
              }
              const auto buf = mask.request();
              const int ndim = static_cast<int>(buf.ndim);
              if (ndim < 1) throw std::invalid_argument("cc_label: input must be ≥ 1-D");
              if (conn < 1 || conn > ndim) throw std::invalid_argument(
                  "cc_label: conn must be in [1, ndim]");
              std::vector<int64_t> shape(ndim);
              std::vector<py::ssize_t> out_shape(ndim);
              for (int d = 0; d < ndim; ++d) {
                  shape[d]     = static_cast<int64_t>(buf.shape[d]);
                  out_shape[d] = static_cast<py::ssize_t>(buf.shape[d]);
              }
              py::array_t<int32_t> out(out_shape);
              int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);
              const void* src_ptr = buf.ptr;
              int32_t n_labels = 0;
              {
                  py::gil_scoped_release release;
                  dispatch_int_dtype(buf.format, buf.itemsize, "cc_label",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          n_labels = ncolor_cpp::cc_label_nd<T>(
                              static_cast<const T*>(src_ptr), out_ptr, shape, conn);
                      });
              }
              return {std::move(out), n_labels};
          },
          py::arg("mask"), py::arg("conn") = 2,
          "N-D connected-components labelling. Returns (labels, n_components).\n"
          "Foreground = (mask != 0). conn = 1 (face only) up to ndim\n"
          "(full diagonal). Compatible with skimage.measure.label output\n"
          "format (int32, dense 1..N labels, 0 = bg).");

    m.def("regionprops",
          [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
             int n_labels_arg) -> py::dict {
              const auto buf = labels.request();
              const int ndim = static_cast<int>(buf.ndim);
              std::vector<int64_t> shape(ndim);
              for (int d = 0; d < ndim; ++d) shape[d] = static_cast<int64_t>(buf.shape[d]);
              const int32_t* lab_ptr = static_cast<const int32_t*>(buf.ptr);
              int64_t total = 1;
              for (int64_t d : shape) total *= d;
              // Auto-detect n_labels if caller passed 0.
              int32_t n_labels = n_labels_arg;
              if (n_labels <= 0) {
                  for (int64_t i = 0; i < total; ++i) {
                      if (lab_ptr[i] > n_labels) n_labels = lab_ptr[i];
                  }
              }
              py::array_t<int64_t> areas({static_cast<py::ssize_t>(n_labels)});
              py::array_t<int64_t> bbox_min({static_cast<py::ssize_t>(n_labels),
                                             static_cast<py::ssize_t>(ndim)});
              py::array_t<int64_t> bbox_max({static_cast<py::ssize_t>(n_labels),
                                             static_cast<py::ssize_t>(ndim)});
              py::array_t<double>  centroid({static_cast<py::ssize_t>(n_labels),
                                             static_cast<py::ssize_t>(ndim)});
              // Grab raw pointers BEFORE releasing the GIL — buffer_info()
              // calls into Python's buffer protocol.
              int64_t* areas_ptr    = static_cast<int64_t*>(areas.request().ptr);
              int64_t* bbox_min_ptr = static_cast<int64_t*>(bbox_min.request().ptr);
              int64_t* bbox_max_ptr = static_cast<int64_t*>(bbox_max.request().ptr);
              double*  cent_ptr     = static_cast<double*>(centroid.request().ptr);
              {
                  py::gil_scoped_release release;
                  ncolor_cpp::regionprops_nd(
                      lab_ptr, n_labels, shape,
                      areas_ptr, bbox_min_ptr, bbox_max_ptr, cent_ptr);
                  // centroid /= area
                  for (int32_t i = 0; i < n_labels; ++i) {
                      const double a = static_cast<double>(areas_ptr[i]);
                      if (a > 0.0) {
                          for (int d = 0; d < ndim; ++d) cent_ptr[i * ndim + d] /= a;
                      }
                  }
              }
              py::dict out;
              out["area"]     = areas;
              out["bbox_min"] = bbox_min;
              out["bbox_max"] = bbox_max;
              out["centroid"] = centroid;
              return out;
          },
          py::arg("labels"), py::arg("n_labels") = 0,
          "Region properties of a dense int32 1..N labelled image.\n"
          "Returns dict with keys 'area' (n_labels,), 'bbox_min'/'bbox_max'\n"
          "(n_labels, ndim), 'centroid' (n_labels, ndim). Pass n_labels=0\n"
          "to auto-detect from labels.max(). One raster pass; no per-component\n"
          "Python objects.");

    m.def("cc_label_per_label",
          [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> input,
             int conn) {
              const auto buf = input.request();
              const int ndim = static_cast<int>(buf.ndim);
              if (ndim < 1) throw std::invalid_argument("cc_label_per_label: input must be ≥ 1-D");
              if (conn < 1 || conn > ndim) throw std::invalid_argument(
                  "cc_label_per_label: conn must be in [1, ndim]");
              std::vector<int64_t> shape(ndim);
              for (int d = 0; d < ndim; ++d) shape[d] = static_cast<int64_t>(buf.shape[d]);

              std::vector<py::ssize_t> py_shape(ndim);
              for (int d = 0; d < ndim; ++d) py_shape[d] = static_cast<py::ssize_t>(buf.shape[d]);
              py::array_t<int32_t> output(py_shape);

              const int32_t* in_ptr  = static_cast<const int32_t*>(buf.ptr);
              int32_t*       out_ptr = static_cast<int32_t*>(output.request().ptr);

              std::vector<int32_t> source_labels;
              int32_t n;
              {
                  py::gil_scoped_release release;
                  n = ncolor_cpp::cc_label_per_label_nd<int32_t>(
                      in_ptr, out_ptr, shape, conn, source_labels);
              }
              py::array_t<int32_t> sl_arr({static_cast<py::ssize_t>(n)});
              if (n > 0) {
                  std::memcpy(sl_arr.mutable_data(),
                              source_labels.data(),
                              static_cast<size_t>(n) * sizeof(int32_t));
              }
              return py::make_tuple(output, n, sl_arr);
          },
          py::arg("input"), py::arg("conn") = 2,
          "Per-label connected components: pixels merge into one component\n"
          "only when they share the same nonzero input value. Returns\n"
          "(labels, n_components, source_labels) where source_labels[i] is\n"
          "the input value of the (i+1)-th component.");

    m.def("delete_spurs",
          [](py::array mask, int hole_threshold, int conn_kind,
             int threshold, int max_iter) {
              if (!(mask.flags() & py::array::c_style)) {
                  mask = py::array::ensure(mask, py::array::c_style);
              }
              const auto buf = mask.request();
              const int ndim = static_cast<int>(buf.ndim);
              if (ndim < 2) throw std::invalid_argument(
                  "delete_spurs requires an array of ndim >= 2");

              std::vector<int64_t> shape(ndim);
              std::vector<py::ssize_t> out_shape(ndim);
              for (int d = 0; d < ndim; ++d) {
                  shape[d]     = static_cast<int64_t>(buf.shape[d]);
                  out_shape[d] = static_cast<py::ssize_t>(buf.shape[d]);
              }
              py::array_t<bool> out(out_shape);
              bool* out_ptr = static_cast<bool*>(out.request().ptr);
              const void* src_ptr = buf.ptr;

              // numpy ``bool`` has format '?' and itemsize 1 — share the
              // uint8 codepath since the memory layout is identical.
              std::string fmt = buf.format;
              if (fmt == "?") fmt = "B";

              {
                  py::gil_scoped_release release;
                  dispatch_int_dtype(fmt, buf.itemsize, "delete_spurs",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          ncolor_cpp::delete_spurs_nd<T>(
                              static_cast<const T*>(src_ptr),
                              out_ptr, shape, hole_threshold,
                              conn_kind, threshold, max_iter);
                      });
              }
              return out;
          },
          py::arg("mask"), py::arg("hole_threshold") = 5,
          py::arg("conn_kind") = 1, py::arg("threshold") = -1,
          py::arg("max_iter") = -1,
          "N-D skeleton/boundary cleanup: fill bg holes ≤ hole_threshold\n"
          "pixels (face-connected), then iteratively strip pixels whose\n"
          "fg-neighbour count under the chosen connectivity is below\n"
          "``threshold`` (default ndim). ``conn_kind`` = 1 → cardinal\n"
          "(face only, omnipose-style external-spur rule, fewer iters);\n"
          "ndim → full diagonal (preserves 1-voxel skeletons). Isolated\n"
          "pixels (count == 0) are always preserved. ``max_iter`` < 0\n"
          "runs to convergence.");

    m.def("two_hop_csr",
          [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> adj_indptr,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> adj_indices)
             -> std::pair<py::array_t<int32_t>, py::array_t<int32_t>>
          {
              const auto ai = adj_indptr.request();
              const auto ax = adj_indices.request();
              if (ai.size < 1) throw std::invalid_argument(
                  "two_hop_csr: empty adj_indptr");
              const int32_t N = static_cast<int32_t>(ai.size - 1);
              std::vector<int32_t> out_indptr, out_indices;
              {
                  py::gil_scoped_release release;
                  ncolor_cpp::compute_two_hop_csr(
                      static_cast<const int32_t*>(ai.ptr),
                      static_cast<const int32_t*>(ax.ptr),
                      N, out_indptr, out_indices);
              }
              py::array_t<int32_t> indptr({static_cast<py::ssize_t>(N + 1)});
              py::array_t<int32_t> indices({static_cast<py::ssize_t>(out_indices.size())});
              std::memcpy(indptr.request().ptr, out_indptr.data(),
                          (N + 1) * sizeof(int32_t));
              std::memcpy(indices.request().ptr, out_indices.data(),
                          out_indices.size() * sizeof(int32_t));
              return {std::move(indptr), std::move(indices)};
          },
          py::arg("adj_indptr"), py::arg("adj_indices"),
          "Build 2-hop neighbour CSR from a 1-hop adjacency CSR.\n"
          "Both directions emitted (symmetric output). O(N · avg_deg²) time.");

    m.def("symmetric_pair_csr",
          [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> pair_u,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> pair_v,
             py::array_t<double,  py::array::c_style | py::array::forcecast> pair_w,
             int32_t N)
             -> std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<double>>
          {
              const auto pu = pair_u.request();
              const auto pv = pair_v.request();
              const auto pw = pair_w.request();
              if (pu.size != pv.size || pu.size != pw.size)
                  throw std::invalid_argument("symmetric_pair_csr: u/v/w size mismatch");
              const int32_t n_pairs = static_cast<int32_t>(pu.size);
              std::vector<int32_t> indptr; std::vector<int32_t> indices;
              std::vector<double>  weights;
              {
                  py::gil_scoped_release release;
                  ncolor_cpp::build_symmetric_pair_csr(
                      static_cast<const int32_t*>(pu.ptr),
                      static_cast<const int32_t*>(pv.ptr),
                      static_cast<const double*>(pw.ptr),
                      n_pairs, N, indptr, indices, weights);
              }
              py::array_t<int32_t> a({static_cast<py::ssize_t>(N + 1)});
              py::array_t<int32_t> b({static_cast<py::ssize_t>(indices.size())});
              py::array_t<double>  c({static_cast<py::ssize_t>(weights.size())});
              std::memcpy(a.request().ptr, indptr.data(), (N + 1) * sizeof(int32_t));
              std::memcpy(b.request().ptr, indices.data(), indices.size() * sizeof(int32_t));
              std::memcpy(c.request().ptr, weights.data(), weights.size() * sizeof(double));
              return {std::move(a), std::move(b), std::move(c)};
          },
          py::arg("pair_u"), py::arg("pair_v"), py::arg("pair_w"), py::arg("N"),
          "Build a symmetric pair-weighted CSR from (u, v, w) triples.\n"
          "Each input pair emits two CSR entries (u→v and v→u). Returns\n"
          "(indptr, indices, weights).");

    m.def("kempe_sa",
          [](py::array_t<uint8_t, py::array::c_style | py::array::forcecast> initial_colors,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> adj_indptr,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> adj_indices,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> twohop_indptr,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> twohop_indices,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> iou_indptr,
             py::array_t<int32_t, py::array::c_style | py::array::forcecast> iou_indices,
             py::array_t<double,  py::array::c_style | py::array::forcecast> iou_weights,
             int n_colors, double alpha_2hop, double gamma_iou,
             int n_iters, int patience,
             double T0, double T_min, double alpha_cool,
             uint64_t rng_seed) -> std::pair<py::array_t<uint8_t>, double>
          {
              const auto ic_buf = initial_colors.request();
              const auto ai_buf = adj_indptr.request();
              const auto ax_buf = adj_indices.request();
              const auto ti_buf = twohop_indptr.request();
              const auto tx_buf = twohop_indices.request();
              const auto ii_buf = iou_indptr.request();
              const auto ix_buf = iou_indices.request();
              const auto iw_buf = iou_weights.request();
              if (ai_buf.size < 1) throw std::invalid_argument(
                  "kempe_sa: adj_indptr empty");
              const int32_t N = static_cast<int32_t>(ai_buf.size - 1);
              if (ic_buf.size != N) throw std::invalid_argument(
                  "kempe_sa: initial_colors length must equal N");
              if (ti_buf.size != N + 1) throw std::invalid_argument(
                  "kempe_sa: twohop_indptr length must be N+1");
              if (ii_buf.size != N + 1) throw std::invalid_argument(
                  "kempe_sa: iou_indptr length must be N+1");
              if (ix_buf.size != iw_buf.size) throw std::invalid_argument(
                  "kempe_sa: iou_indices and iou_weights must have same length");

              std::vector<uint8_t> colors(static_cast<size_t>(N));
              std::memcpy(colors.data(), ic_buf.ptr,
                          static_cast<size_t>(N) * sizeof(uint8_t));

              ncolor_cpp::KempeSAParams params;
              params.n_colors   = n_colors;
              params.alpha_2hop = alpha_2hop;
              params.gamma_iou  = gamma_iou;
              params.n_iters    = n_iters;
              params.patience   = patience;
              params.T0         = T0;
              params.T_min      = T_min;
              params.alpha_cool = alpha_cool;
              params.rng_seed   = rng_seed;

              double best_loss = 0.0;
              {
                  py::gil_scoped_release release;
                  best_loss = ncolor_cpp::kempe_sa(
                      N,
                      static_cast<const int32_t*>(ai_buf.ptr),
                      static_cast<const int32_t*>(ax_buf.ptr),
                      static_cast<const int32_t*>(ti_buf.ptr),
                      static_cast<const int32_t*>(tx_buf.ptr),
                      static_cast<const int32_t*>(ii_buf.ptr),
                      static_cast<const int32_t*>(ix_buf.ptr),
                      static_cast<const double*>(iw_buf.ptr),
                      colors, params);
              }

              py::array_t<uint8_t> out({static_cast<py::ssize_t>(N)});
              std::memcpy(out.request().ptr, colors.data(),
                          static_cast<size_t>(N) * sizeof(uint8_t));
              return {std::move(out), best_loss};
          },
          py::arg("initial_colors"),
          py::arg("adj_indptr"), py::arg("adj_indices"),
          py::arg("twohop_indptr"), py::arg("twohop_indices"),
          py::arg("iou_indptr"), py::arg("iou_indices"), py::arg("iou_weights"),
          py::arg("n_colors") = 4,
          py::arg("alpha_2hop") = 1.0,
          py::arg("gamma_iou") = 50.0,
          py::arg("n_iters") = 30000,
          py::arg("patience") = 1000,
          py::arg("T0") = 2.0,
          py::arg("T_min") = 0.001,
          py::arg("alpha_cool") = 0.9998,
          py::arg("rng_seed") = 0,
          "Kempe-component simulated annealing for 4-coloring.\n"
          "Loss = alpha_2hop · #2-hop_same + gamma_iou · sum(w · 1[same])\n"
          "All CSR pair arrays must store both directions (u→v and v→u).\n"
          "Returns (final_colors, best_loss).");
}
