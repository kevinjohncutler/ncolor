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

#include <algorithm>   // std::sort / std::unique in color_graph
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#include "cc_label.hpp"
#include "chamfer.hpp"
#include "color.hpp"
#include "delete_spurs.hpp"
#include "delete_spurs_labels.hpp"
#include "fast_despur.hpp"
#include "connect_with_face_count.hpp"
#include "expand_lp.hpp"
#include "expand_clean.hpp"
#include "soft_color.hpp"
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
#include "picker.hpp"

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
// alone breaks across platforms. numpy ``bool`` (format '?', one byte) is
// routed to the uint8 kernel: the memory layout is identical and every
// kernel only asks whether a value is zero. Throws on unsupported dtype
// with `api_name` in the error message.
//
// ``allow_float`` admits float32 / float64 as well. Only the entry points
// that immediately cast to int32 (label, connect, format_labels, cc_label)
// take floats: segmenters such as cellpose hand back label maps in float
// arrays, and for those the range-checked cast truncates toward zero.
template <typename Func>
static inline void dispatch_int_dtype(const std::string& fmt, py::ssize_t itemsize,
                                      const char* api_name, Func&& f,
                                      bool allow_float = false) {
    bool is_signed = false, is_unsigned = false, is_float = false;
    if (!fmt.empty()) {
        const char c = fmt[0];
        if (c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' || c == 'n')
            is_signed = true;
        else if (c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' ||
                 c == 'N' || c == '?')
            is_unsigned = true;
        else if (c == 'f' || c == 'd')
            is_float = true;
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
            case 8: f(static_cast<uint64_t*>(nullptr)); return;
        }
    } else if (is_float && allow_float) {
        switch (itemsize) {
            case 4: f(static_cast<float*>(nullptr));  return;
            case 8: f(static_cast<double*>(nullptr)); return;
        }
    }
    throw std::invalid_argument(std::string(api_name) +
        ": unsupported dtype '" + fmt + "' (need bool, uint8/16/32/64, "
        "int8/16/32/64" + (allow_float ? ", float32/64)" : ")"));
}

// The cast-based entry points: everything dispatch_int_dtype takes, plus
// float32 / float64.
template <typename Func>
static inline void dispatch_cast_dtype(const std::string& fmt, py::ssize_t itemsize,
                                       const char* api_name, Func&& f) {
    dispatch_int_dtype(fmt, itemsize, api_name, std::forward<Func>(f),
                       /*allow_float=*/true);
}

// Raised (as Python OverflowError) when a wide-dtype input holds a value
// the int32 engine cannot represent. ncolor's Python wrappers catch this
// for label() / format_labels() and retry on a compacted copy.
[[noreturn]] static void throw_label_overflow(const char* api_name) {
    throw std::overflow_error(std::string(api_name) +
        ": label values outside the int32 range (or NaN / inf). Compact the "
        "labels first, e.g. with ncolor.format_labels, which handles this "
        "automatically.");
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

// ---- Thread pools -------------------------------------------------------
//
// A pool and the mutex that serializes calls on it are one object. Only
// one ``parallel()`` may be in flight per pool, so every engine method
// takes its pool's mutex for the duration of the call: engines that
// share a pool take turns, and engines holding private pools run at the
// same time. The mutex is taken only after the GIL is released, so a
// thread waiting here can never block one that needs the GIL to finish.
struct PoolSlot {
    explicit PoolSlot(int n) : pool(static_cast<size_t>(n <= 1 ? 1 : n)) {}
    ncolor_cpp::ForkJoinPool pool;
    std::mutex mu;
};

// Pools are shared by (thread count, group). Group 0 is the process-wide
// default: the Solver and the ExpandEngine the module functions use both
// land on it, so the package holds one pool rather than one each. A
// caller wanting to work on several images at once takes a fresh group
// per worker (``ncolor.Engine``); the two engines within one group still
// share, since they are never in a call at the same time. The registry
// holds weak references, so a pool dies with the last engine using it
// (the SMT calibration builds and discards engines with two different
// counts back to back).
static std::shared_ptr<PoolSlot> resolve_pool(int n_threads, int pool_group) {
    const int n = n_threads <= 1 ? 1 : n_threads;
    static std::mutex registry_mutex;
    static std::map<std::pair<int, int>, std::weak_ptr<PoolSlot>> registry;
    std::lock_guard<std::mutex> lk(registry_mutex);
    auto& slot = registry[{n, pool_group}];
    if (auto live = slot.lock()) return live;
    auto fresh = std::make_shared<PoolSlot>(n);
    slot = fresh;
    return fresh;
}

// Persistent-pool wrapper for expand_labels + parallel LUT apply.
// One ExpandEngine per ncolor.label "pipeline" — the pool + buffers persist
// across calls so the only per-call cost is task enqueue.
class ExpandEngine {
public:
    explicit ExpandEngine(double n_threads, int pool_group = 0)
        : n_threads_(resolve_threads(n_threads)),
          pool_(resolve_pool(n_threads_, pool_group)) {}

    int n_threads() const { return n_threads_; }

    // Free the persistent scratch buffers (see ExpandBuffers::release).
    void release() {
        py::gil_scoped_release gil;
        std::lock_guard<std::mutex> engine_lock(pool_->mu);
        bufs_.release();
    }

    // Voronoi label expansion under L_p metric. ``p=1`` (Manhattan) uses the
    // Saito-Toriwaki separable sweep; ``p=2`` (Euclidean²) uses the
    // Felzenszwalb-Huttenlocher parabolic envelope. Same ND driver,
    // dispatched at compile time on p — see ``expand_lp.hpp``. Default is
    // p=2 (matches numba's ``expand_labels(metric='l2')``).
    //
    // Takes any supported label dtype (see dispatch_cast_dtype): the cast
    // to int32 is range-checked and runs in parallel, straight into the
    // output array, which the kernels then expand in place. Raises
    // OverflowError for values outside int32 rather than renumbering,
    // because expand keeps label identities.
    py::array_t<int32_t> expand_labels(py::array labels, int p = 2, bool wrap = false) {
        if (p != 1 && p != 2) {
            throw std::invalid_argument("expand_labels: p must be 1 or 2");
        }
        if (!(labels.flags() & py::array::c_style)) {
            labels = py::array::ensure(labels, py::array::c_style);
        }
        const auto buf = labels.request();
        std::vector<int64_t> shape(buf.ndim);
        int64_t total = 1;
        for (int i = 0; i < buf.ndim; ++i) {
            shape[i] = buf.shape[i];
            total *= buf.shape[i];
        }
        const void* src_ptr = buf.ptr;
        py::array_t<int32_t> out(buf.shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);

        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
            cast_into_(buf, src_ptr, out_ptr, total, "ExpandEngine.expand_labels");
            if (p == 2) {
                ncolor_cpp::expand_labels_lp<2>(out_ptr, out_ptr, bufs_, shape, pool_->pool, n_threads_, wrap);
            } else {
                ncolor_cpp::expand_labels_lp<1>(out_ptr, out_ptr, bufs_, shape, pool_->pool, n_threads_, wrap);
            }
        }
        return out;
    }

    // Bridge-free Voronoi label expansion. After each EDT axis sweep,
    // pixels that form an antipodal-only bridge (exactly two same-label
    // neighbors arranged opposite each other: N-S, E-W, NE-SW, or NW-SE
    // in 2D) are marked as barriers (lbl=0) so the next axis sweep
    // can't refill them. Currently 2D L2 only — falls back to standard
    // L2 expand for ND > 2 until 3D antipodal generalization lands.
    py::array_t<int32_t> expand_labels_clean(py::array labels, int p = 2) {
        if (p != 1 && p != 2) {
            throw std::invalid_argument(
                "expand_labels_clean: p must be 1 or 2");
        }
        if (!(labels.flags() & py::array::c_style)) {
            labels = py::array::ensure(labels, py::array::c_style);
        }
        const auto buf = labels.request();
        std::vector<int64_t> shape(buf.ndim);
        int64_t total = 1;
        for (int i = 0; i < buf.ndim; ++i) {
            shape[i] = buf.shape[i];
            total *= buf.shape[i];
        }
        const void* src_ptr = buf.ptr;
        py::array_t<int32_t> out(buf.shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);

        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
            cast_into_(buf, src_ptr, out_ptr, total, "ExpandEngine.expand_labels_clean");
            ncolor_cpp::expand_labels_clean_inplace(
                out_ptr, bufs_, shape, pool_->pool, n_threads_, p);
            std::memcpy(out_ptr, bufs_.lbl(),
                        bufs_.size() * sizeof(int32_t));
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
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
            if (p == 2) {
                ncolor_cpp::expand_labels_lp<2>(input, out_lbl_ptr, bufs_, shape, pool_->pool, n_threads_, wrap);
                const int32_t* d = bufs_.dist();
                for (int64_t i = 0; i < total; ++i) {
                    out_dist_ptr[i] = std::sqrt(static_cast<double>(d[i]));
                }
            } else if (p == 1) {
                ncolor_cpp::expand_labels_lp<1>(input, out_lbl_ptr, bufs_, shape, pool_->pool, n_threads_, wrap);
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
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
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
                        labels_out_scratch.data(), bufs_, shape, pool_->pool, n_threads_, wrap);
                    const int32_t* d = bufs_.dist();
                    double* out_c = out_ptr + static_cast<int64_t>(c - 1) * total;
                    for (int64_t i = 0; i < total; ++i) {
                        out_c[i] = std::sqrt(static_cast<double>(d[i]));
                    }
                } else if (p == 1) {
                    ncolor_cpp::expand_labels_lp<1>(masked.data(),
                        labels_out_scratch.data(), bufs_, shape, pool_->pool, n_threads_, wrap);
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
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
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
                        labels_out_scratch.data(), bufs_, shape, pool_->pool, n_threads_, wrap);
                } else if (p == 1) {
                    ncolor_cpp::expand_labels_lp<1>(masked.data(),
                        labels_out_scratch.data(), bufs_, shape, pool_->pool, n_threads_, wrap);
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
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
            // Cast to int32 in parallel inside the released-GIL block.
            bool fits = true;
            dispatch_cast_dtype(buf.format, buf.itemsize,
                "ExpandEngine.format_labels", [&](auto* tag) {
                    using T = std::remove_pointer_t<decltype(tag)>;
                    fits = ncolor_cpp::cast_to_int32<T>(
                        static_cast<const T*>(src_ptr), out_ptr, total,
                        pool_->pool, n_threads_);
                });
            if (!fits) throw_label_overflow("ExpandEngine.format_labels");
            n_labels = first_seen
                ? ncolor_cpp::format_labels_inplace_first_seen(
                    out_ptr, total, pool_->pool, n_threads_)
                : ncolor_cpp::format_labels_inplace(
                    out_ptr, total, pool_->pool, n_threads_);
        }
        return {std::move(out), n_labels};
    }

private:
    // Range-checked parallel cast of a caller's buffer into ``dst``.
    // Must be called with the GIL released and the engine lock held.
    void cast_into_(const py::buffer_info& buf, const void* src_ptr,
                    int32_t* dst, int64_t total, const char* api_name) {
        bool fits = true;
        dispatch_cast_dtype(buf.format, buf.itemsize, api_name,
            [&](auto* tag) {
                using T = std::remove_pointer_t<decltype(tag)>;
                fits = ncolor_cpp::cast_to_int32<T>(
                    static_cast<const T*>(src_ptr), dst, total,
                    pool_->pool, n_threads_);
            });
        if (!fits) throw_label_overflow(api_name);
    }

    int n_threads_;
    std::shared_ptr<PoolSlot> pool_;
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
    explicit Solver(double n_threads, int pool_group = 0)
        : n_threads_(resolve_threads(n_threads)),
          pool_(resolve_pool(n_threads_, pool_group)) {}

    int n_threads() const { return n_threads_; }

    // Free every persistent scratch buffer. The engine keeps the working
    // set of the largest image it has processed (roughly 22 bytes per
    // pixel) alive between calls so repeated same-shape calls never
    // allocate; after a single whole-slide image that is gigabytes. The
    // next call reallocates as needed. Accessor state (last LUT, stage
    // timings, conflict count) is reset too.
    void release() {
        py::gil_scoped_release gil;
        std::lock_guard<std::mutex> engine_lock(pool_->mu);
        expand_bufs_.release();
        drop_(bg_mask_); drop_(partials_);
        drop_(src_idx_); drop_(dst_idx_);
        drop_(indptr_); drop_(indices_); drop_(edge_weights_);
        drop_(soft_indptr_); drop_(soft_indices_); drop_(soft_weights_);
        drop_(colors_); drop_(lut_); drop_(lut_lbl_); drop_(orig_labels_);
        drop_(despur_face_count_);
        drop_(fp_ht_buf_); drop_(fp_primary_buf_); drop_(fp_counts_buf_);
        drop_(fp_soft_ht_buf_); drop_(fused_soft_pairs_);
        drop_(last_stages_);
        drop_(picker_scratch_.per_attempt_colors_);
        drop_(picker_scratch_.per_attempt_ok_);
        last_n_conflicts_ = 0;
        n_soft_violations_last_ = 0.0;
        lut_.assign(1, 0);
    }

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
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
            // Cast to int32 in expand_bufs_.lbl(); the bg mask is unused
            // here (Solver.connect never applies a LUT) but cast_with_bg
            // is the parallel cast we already use elsewhere — bg writes
            // are cheap and let us share the kernel.
            expand_bufs_.resize(total);
            int32_t* labels = expand_bufs_.lbl();
            bg_mask_.resize(static_cast<size_t>(total));
            uint8_t* bg = bg_mask_.data();
            bool fits = true;
            dispatch_cast_dtype(buf.format, buf.itemsize, "Solver.connect",
                [&](auto* tag) {
                    using T = std::remove_pointer_t<decltype(tag)>;
                    fits = ncolor_cpp::cast_with_bg<T>(
                        static_cast<const T*>(src_ptr), labels, bg, total,
                        pool_->pool, n_threads_);
                });
            if (!fits) throw_label_overflow("Solver.connect");

            const int32_t max_label = parallel_max_label_(labels, total);
            const int32_t n_labels = distinct_labels_(labels, total, max_label);
            pairs = find_pairs_(labels, shape, conn, wrap, n_labels);
        }

        return pairs_to_array(pairs);
    }

    std::pair<py::array_t<uint8_t>, int> label(
            py::array mask,
            int n_colors = 4, int max_depth = 30, int rand_period = 10,
            int conn = 1, int p = 2, bool capture_stages = false,
            bool format_input = true, bool expand = true,
            py::object out_arg = py::none(),
            int color_mode = -1, bool wrap = false,
            bool first_seen = false,
            int weight_objective = 0,
            py::object de_table_obj = py::none(),
            int weight_mode = 1 /* ReduceMode::Min */,
            py::object extra_edges_obj = py::none(),
            int connect_radius = 1,
            // despur_iters / despur_remove_thin default to 0/false:
            // the "clean" expand_mode (default) already subsumes the
            // bridge + stub removal, so running an additional despur pass
            // on top is a no-op-but-still-O(N) pass. The Python public
            // ``ncolor.label()`` API dropped these kwargs in 2.0; they
            // remain on the C++ binding only for the niche callers who
            // use expand_mode="standard" + want explicit despur.
            int despur_iters = 0,
            bool despur_remove_thin = false,
            int min_contact = 1,
            std::string expand_mode = "clean",
            py::object soft_extra_edges_obj = py::none(),
            int soft_conn = 2,
            int soft_radius = 2,
            bool clean_mask = false) {
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
        // Same parse for soft_extra_edges (Nx2 int32 pairs). Soft edges
        // are NOT added to the hard CSR; they go to a separate post-solve
        // local-search pass that minimises the count of soft edges whose
        // endpoints share a color.
        int32_t n_soft = 0;
        const int32_t* soft_ptr = nullptr;
        py::array_t<int32_t> soft_arr_holder;
        if (!soft_extra_edges_obj.is_none()) {
            soft_arr_holder = py::array_t<int32_t,
                py::array::c_style | py::array::forcecast>::ensure(soft_extra_edges_obj);
            if (soft_arr_holder) {
                const auto sb = soft_arr_holder.request();
                if (sb.ndim == 2 && sb.shape[1] == 2) {
                    n_soft = static_cast<int32_t>(sb.shape[0]);
                    soft_ptr = static_cast<const int32_t*>(sb.ptr);
                }
            }
        }
        // Same pre-release parse for de_table (user-supplied (n+1)x(n+1)
        // perceptual-distance palette override). Calling py::array_t::ensure
        // inside the GIL-released block was the cause of a hard segfault
        // when users passed a custom palette; copy here, use the data
        // pointer below.
        const double* user_de_ptr = nullptr;
        int32_t user_de_dim = 0;
        py::array_t<double> de_arr_holder;
        if (!de_table_obj.is_none()) {
            de_arr_holder = py::array_t<double,
                py::array::c_style | py::array::forcecast>::ensure(de_table_obj);
            if (de_arr_holder) {
                const auto db = de_arr_holder.request();
                if (db.ndim == 2 && db.shape[0] == db.shape[1]) {
                    user_de_dim = static_cast<int32_t>(db.shape[0]);
                    user_de_ptr = static_cast<const double*>(db.ptr);
                }
            }
        }

        bool early_exit_empty = false;
        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> engine_lock(pool_->mu);

            // 0a. Cast input dtype → int32 (in expand_bufs_.lbl()) AND
            // capture the bg pattern (input == 0) into bg_mask_, all in
            // one parallel pass. For int32 input the cast is still a
            // straight copy (with bg-mask write); the alternative
            // pattern of "skip the copy when input is int32" was a tiny
            // saving but cost us the multi-dtype generality.
            //
            // Tried fusing the min/max reduce from format_labels into
            // this pass — regressed. The data is still cache-hot when
            // format does its own reduce, so no memory traffic is saved,
            // and the per-element compares slow this kernel down.
            expand_bufs_.resize(total);
            bg_mask_.resize(static_cast<size_t>(total));
            int32_t* expanded = expand_bufs_.lbl();
            uint8_t* bg = bg_mask_.data();
            bool fits = true;
            dispatch_cast_dtype(buf.format, buf.itemsize, "Solver.label",
                [&](auto* tag) {
                    using T = std::remove_pointer_t<decltype(tag)>;
                    fits = ncolor_cpp::cast_with_bg<T>(
                        static_cast<const T*>(src_ptr), expanded, bg, total,
                        pool_->pool, n_threads_);
                });
            if (!fits) throw_label_overflow("Solver.label");
            stage("cast");

            // 0b. Optional format_labels: compact nonzero labels to 1..N
            // in place inside expand_bufs_.lbl(). When format_input=False
            // the caller is asserting labels are already 1..N.
            const int32_t* expand_input = expanded;
            if (format_input) {
                const int n_labels = first_seen
                    ? ncolor_cpp::format_labels_inplace_first_seen(
                        expanded, total, pool_->pool, n_threads_)
                    : ncolor_cpp::format_labels_inplace(
                        expanded, total, pool_->pool, n_threads_);
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
            // Resolve expand strategy. Default is "clean": ND Lp Voronoi
            // expand fused with an antipodal-bridge test and a despur
            // peel-back cascade in one pass (subsumes despur, so
            // despur_iters is unused on that path).
            std::string em = expand_mode;
            if (em.empty()) em = "clean";
            // When `clean_mask=false` (default) and the "clean" expand
            // mode is in use, the output LUT should be applied to the
            // ORIGINAL foreground labels (before the clean pass zeroes
            // bridge/stub pixels as graph barriers). The picker's
            // coloring is still built from the cleaned graph; we only
            // redirect the final pixel-by-pixel LUT lookup. Snapshot the
            // post-format / pre-expand labels here so apply_color_lut_
            // can use them.
            const bool need_orig_snapshot = !clean_mask
                && expand && em == "clean";
            if (need_orig_snapshot) {
                orig_labels_.assign(expand_input, expand_input + total);
            }
            if (expand) {
                if (em == "clean") {
                    // Voronoi expand + antipodal-bridge test + despur
                    // cascade (ND, Lp). Single pass does the EDT sweep +
                    // antipodal bridge removal + despur cascade; result
                    // lands in expand_bufs_.lbl() which `expanded`
                    // already points to.
                    ncolor_cpp::expand_labels_clean_inplace(
                        expand_input, expand_bufs_, shape,
                        pool_->pool, n_threads_, p);
                } else if (em == "standard") {
                    if (p == 2) {
                        ncolor_cpp::expand_labels_lp<2>(expand_input, expanded, expand_bufs_, shape, pool_->pool, n_threads_, wrap);
                    } else {
                        ncolor_cpp::expand_labels_lp<1>(expand_input, expanded, expand_bufs_, shape, pool_->pool, n_threads_, wrap);
                    }
                } else {
                    throw std::invalid_argument(
                        "expand_mode must be 'clean' or 'standard'; "
                        "got '" + em + "'");
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
            // to a single-same-label-neighbor pixel inside another's
            // territory. These convergence points create K_5
            // obstructions in densely-packed segmentations (5 cells
            // meeting at a corner) that prevent 4-coloring at conn=1
            // r=1. Iterative despur strips them: pixels with ≤
            // threshold same-label face-neighbors become bg. Iter 1
            // is empirically enough to break the K_5 cascade and
            // reach n=4 on dense microscopy data; iter 2 (default) is
            // chosen with margin. Running to convergence is NOT
            // monotonically better — on dense data past ~iter 20 the
            // cascade can eliminate small cells, allowing previously-
            // buffered cells to touch and form a new K_5 that pushes
            // n_used back up to 5. Stay well below that cliff.
            //
            // ``despur_remove_thin`` ALSO catches 1-voxel-thick straight
            // interior pixels (the 1462 thin bridges L1 Voronoi creates
            // on MM-class data). Off by default in the labeling pipeline
            // because removing those edges triggers the same picker
            // heuristic failure as the iter-30 cliff, just earlier (at
            // iter 2 instead of iter 30). The principled fix is the
            // default ``expand_mode="clean"``, which never creates
            // the thin bridges in the first place.
            // ``lut_lbl_ptr`` is the buffer apply_color_lut_ reads at the
            // end: by default that's the post-expand ``expanded`` buffer.
            // When despur runs we want to keep coloring the spur pixels
            // with their parent cell's color rather than turning them
            // into bg, so we save the pre-despur labels into ``lut_lbl_``
            // and use that for the final LUT application. Despur still
            // modifies ``expanded`` in place so find_pairs / coloring
            // operate on the despurred graph.
            const int32_t* lut_lbl_ptr = expanded;
            // Note on fusion: ``find_pairs_with_face_count_2d_v2`` +
            // ``despur_via_face_count_with_pair_decrement_2d`` (see
            // ``connect_with_face_count.hpp``) provide a correct fused
            // path that emits pairs + face_count in one scan and
            // prunes ghost edges via per-pair contact counts. Verified
            // to produce a bit-identical pair list to the safe
            // separate-pass path on MM (14769 pairs both ways).
            //
            // BUT: the HT_lookup-driven decrements during the despur
            // peel-back (~56 k lookups on MM with random access in a
            // 512 KB hashtable) cost ~5 ms — more than the ~1.5 ms
            // saved by skipping a standalone find_pairs pass. Net
            // regression. Keeping the algorithm in-tree but not wired
            // here; can be turned on by callers that want it.
            if (despur_iters > 0 && expand) {
                lut_lbl_.assign(expanded, expanded + total);
                lut_lbl_ptr = lut_lbl_.data();
                if (despur_remove_thin) {
                    ncolor_cpp::delete_spurs_labels_nd_inplace<int32_t>(
                        expanded, shape, /*threshold=*/1,
                        despur_iters, &pool_->pool, n_threads_,
                        despur_remove_thin);
                } else {
                    despur_face_count_.assign((size_t)total, 0);
                    ncolor_cpp::compute_face_count_nd<int32_t>(
                        expanded, despur_face_count_.data(), shape,
                        &pool_->pool, n_threads_);
                    ncolor_cpp::despur_via_face_count_nd<int32_t>(
                        expanded, despur_face_count_.data(), shape,
                        /*threshold=*/1, &pool_->pool, n_threads_);
                }
                stage("despur");
            }

            // 2. Find adjacency pairs. Parallel max-reduce first
            // (was a single-threaded 1.2 ms loop at 2048²).
            const int32_t max_label = parallel_max_label_(expanded, total);
            const int32_t n_labels = distinct_labels_(expanded, total, max_label);
            stage("max_scan");
            const int wobj = weight_objective;
            const int wmode = weight_mode;  // 0=Min (default), see binding kwargs
            std::vector<std::pair<int32_t, int32_t>> pairs;
            std::vector<double> pair_primary;
            std::vector<int32_t> pair_counts;
            // Start every call with an empty soft list. Only some branches
            // below produce one, and the Solver is a process-global
            // singleton: without this, a call taking the weighted or
            // min_contact branch would inherit the previous call's soft
            // pairs. Those are label ids of a different image, so the ones
            // that happen to fall inside the new label range get applied
            // as soft constraints and can push the color count up.
            fused_soft_pairs_.clear();
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
                            n_labels, pair_primary, pair_counts); break;
                    case ReduceMode::Mean:
                        pairs = find_pairs_weighted_<ReduceMode::Mean>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            n_labels, pair_primary, pair_counts); break;
                    case ReduceMode::Count:
                        pairs = find_pairs_weighted_<ReduceMode::Count>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            n_labels, pair_primary, pair_counts); break;
                    case ReduceMode::Harmonic:
                        pairs = find_pairs_weighted_<ReduceMode::Harmonic>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            n_labels, pair_primary, pair_counts); break;
                    case ReduceMode::MeanInv:
                        pairs = find_pairs_weighted_<ReduceMode::MeanInv>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            n_labels, pair_primary, pair_counts); break;
                    case ReduceMode::Min:
                    default:
                        pairs = find_pairs_weighted_<ReduceMode::Min>(
                            expanded, expand_bufs_.dist(), shape, conn, wrap,
                            n_labels, pair_primary, pair_counts); break;
                }
            } else if (min_contact > 1 && connect_radius > 1) {
                // Contact-filtered pair-find: tracks per-pair pixel-
                // contact count via ReduceMode::Count and drops pairs
                // whose count is below `min_contact`. At r=2 the
                // wider neighbor window picks up cells that share
                // only a 1-pixel "leak" through a bg gap (median
                // r=2-only contact is 2 on mm-class data, vs 44 for
                // legitimate r=1 face-adjacent pairs). Filtering
                // these by contact count removes the spurious
                // Mycielski-like obstruction that pushes χ from 4 to
                // 5. Only fires when r > 1 and the user asks for it.
                using ncolor_cpp::ReduceMode;
                std::vector<double> primary_unused;
                pair_counts.clear();
                pairs = find_pairs_weighted_<ReduceMode::Count>(
                    expanded, /*dist=*/nullptr, shape, conn, wrap,
                    n_labels, primary_unused, pair_counts,
                    connect_radius);
                // Filter pairs by count.
                int kept = 0;
                for (size_t i = 0; i < pairs.size(); ++i) {
                    if (pair_counts[i] >= min_contact) {
                        pairs[kept] = pairs[i];
                        ++kept;
                    }
                }
                pairs.resize(kept);
            } else if (soft_conn > 0 && soft_radius > 0 &&
                        soft_conn >= conn &&
                        soft_radius >= connect_radius &&
                        (soft_conn > conn || soft_radius > connect_radius) &&
                        max_label > 0) {
                // Fused base+soft pair-find: one pixel walk emits both
                // base pairs (at conn, connect_radius) and the delta
                // pairs at (soft_conn, soft_radius) excluding base.
                // Replaces the separate post-pass scan for the soft
                // auto-builder; saves the second pixel walk + HT init +
                // merge on the soft side. Falls through to the single-
                // emit path when no soft kernel is requested or when it
                // would be a subset of the base.
                const int ndim_local = (int)shape.size();
                const int64_t n_fwd_base = ncolor_cpp::detail::
                    count_forward_neighbors(ndim_local, conn, connect_radius);
                const int64_t n_fwd_soft = ncolor_cpp::detail::
                    count_forward_neighbors(ndim_local, soft_conn, soft_radius);
                const int64_t n_fwd_delta = std::max<int64_t>(1, n_fwd_soft - n_fwd_base);
                const int64_t base_ht_raw =
                    2 * n_fwd_base * (int64_t)n_labels;
                const int64_t soft_ht_raw =
                    2 * n_fwd_delta * (int64_t)n_labels;
                uint64_t base_ht_size = (uint64_t)ipow2_ge(
                    std::max<int64_t>(base_ht_raw, MIN_HT_SIZE));
                uint64_t soft_ht_size = (uint64_t)ipow2_ge(
                    std::max<int64_t>(soft_ht_raw, MIN_HT_SIZE));
                for (;;) {
                    const int full =
                        ncolor_cpp::find_pairs_dual_nd_unpadded<int32_t>(
                            expanded, shape, conn, connect_radius,
                            soft_conn, soft_radius,
                            base_ht_size, soft_ht_size,
                            n_threads_, pool_->pool, wrap,
                            pairs, fused_soft_pairs_,
                            /*base_ht_scratch=*/&fp_ht_buf_,
                            /*soft_ht_scratch=*/&fp_soft_ht_buf_);
                    if (full == 0) break;
                    bool can_retry = true;
                    if ((full & 1) != 0) {
                        if (base_ht_size >= HT_SIZE_CAP) can_retry = false;
                        else base_ht_size <<= 1;
                    }
                    if ((full & 2) != 0) {
                        if (soft_ht_size >= HT_SIZE_CAP) can_retry = false;
                        else soft_ht_size <<= 1;
                    }
                    if (!can_retry)
                        throw std::overflow_error(
                            "Solver.label: adjacency table exceeded safety cap");
                }
            } else if (soft_conn > 0 && soft_radius > 0 &&
                       (soft_conn > conn || soft_radius > connect_radius) &&
                       max_label > 0) {
                // The hard and soft kernels are incomparable: one has the
                // richer connectivity while the other has the larger radius.
                // The dual builder enumerates a containing soft kernel, so it
                // cannot represent this union. Scan each kernel independently
                // and remove hard pairs from the soft preference set.
                pairs = find_pairs_(expanded, shape, conn, wrap,
                                    max_label, connect_radius);
                auto soft_all = find_pairs_(expanded, shape, soft_conn, wrap,
                                            max_label, soft_radius);
                std::sort(pairs.begin(), pairs.end());
                std::sort(soft_all.begin(), soft_all.end());
                fused_soft_pairs_.clear();
                std::set_difference(
                    soft_all.begin(), soft_all.end(),
                    pairs.begin(), pairs.end(),
                    std::back_inserter(fused_soft_pairs_));
            } else {
                // Unified pair-find: `connect_radius` widens the
                // neighbor offset window (Chebyshev distance) for
                // each pixel. radius=1 is the standard 8-connectivity
                // path; radius>1 catches near-adjacent cells
                // separated by a thin gap of another cell's
                // territory. Same parallel + offset-precomputed
                // kernel regardless of radius.
                pairs = find_pairs_(expanded, shape, conn, wrap,
                                     max_label, connect_radius);
                fused_soft_pairs_.clear();
            }
            // Canonical sort: orders pairs by (src, dst). Without this,
            // pairs come from HT iteration (slot order = hash-mixed,
            // dependent on offset enumeration order). Same EDGE SET
            // produces different CSR neighbor-iteration order, which
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
                        // Voronoi-extended boundary aren't penalized.
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
            // user_de_ptr / user_de_dim were parsed pre-GIL-release above.
            std::vector<double> de_table_vec;
            const double* de_ptr = nullptr;
            if (wobj != 0) {
                if (user_de_ptr != nullptr && user_de_dim == n_colors + 1) {
                    const size_t total_de = static_cast<size_t>(n_colors + 1) * (n_colors + 1);
                    de_table_vec.assign(user_de_ptr, user_de_ptr + total_de);
                    de_ptr = de_table_vec.data();
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
                    // those colors then contribute no contrast preference,
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
                                     color_mode,
                                     static_cast<int>(shape.size()), wrap,
                                     edge_weights_ptr, de_ptr, wobj);
            stage("color");

            // 4b. Optional soft-edge local search. Operates on the valid
            // hard coloring just found, recoloring vertices (without
            // breaking hard adjacency) to reduce the count of soft edges
            // whose endpoints share a color. See cpp/soft_color.hpp.
            n_soft_violations_last_ = 0.0;
            // Soft edges can come from either an explicit (E, 2) array
            // (soft_ptr/n_soft, parsed above) OR from auto-build using
            // a richer (soft_conn, soft_radius) kernel. The latter is
            // cheaper since it stays in C++ and reuses find_pairs.
            std::vector<int32_t> auto_soft_pairs_flat;
            int32_t n_soft_eff = n_soft;
            const int32_t* soft_ptr_eff = soft_ptr;
            if (n_soft == 0 && soft_conn > 0 && soft_radius > 0
                && N > 0 && n_used > 0) {
                // Delta pairs were already computed in the fused base+soft
                // find_pairs call (stage 2). Re-pack them as a flat int32
                // array for build_soft_csr. The fused-scan path is empty
                // when (soft_conn ≤ conn AND soft_radius ≤ connect_radius);
                // we get an empty list and the soft search no-ops cleanly.
                auto_soft_pairs_flat.reserve(2 * fused_soft_pairs_.size());
                for (auto& p : fused_soft_pairs_) {
                    auto_soft_pairs_flat.push_back(p.first);
                    auto_soft_pairs_flat.push_back(p.second);
                }
                n_soft_eff = static_cast<int32_t>(
                    auto_soft_pairs_flat.size() / 2);
                soft_ptr_eff = auto_soft_pairs_flat.data();
            }
            if (n_soft_eff > 0 && N > 0 && n_used > 0) {
                ncolor_cpp::build_soft_csr(
                    N, n_soft_eff, soft_ptr_eff, /*weights_in=*/nullptr,
                    soft_indptr_, soft_indices_, soft_weights_);
                stage("soft_csr");
                // Triangle-count weights bias the search toward fixing
                // edges inside dense soft sub-cliques (K_4-soft clusters
                // like the logo cluster) ahead of isolated leak/noise
                // edges. Replaces uniform weights. See compute_triangle
                // _weights for the rationale. Cost: O(sum of soft-deg²)
                // ≈ tiny on our test images (avg soft-deg < 1).
                ncolor_cpp::compute_triangle_weights(
                    N, soft_indptr_.data(), soft_indices_.data(),
                    soft_weights_);
                stage("soft_weights");
                n_soft_violations_last_ = ncolor_cpp::soft_local_search(
                    colors_.data(), N,
                    indptr_.data(), indices_.data(),
                    soft_indptr_.data(), soft_indices_.data(),
                    soft_weights_.data(),
                    n_used);
                stage("soft_search");
                // The soft search may have vacated a color; keep the
                // reported count and the pixel values in step.
                n_used = ncolor_cpp::densify_colors(colors_, N);
            }

            // 5. Build LUT (expanded[i] is in 1..N, so lut size = N+1) and
            // apply it to the fg-label buffer. Default (clean_mask=false)
            // uses the pre-expand snapshot ``orig_labels_`` so original-fg
            // pixels keep their cell's color even if the clean expand zeroed
            // them as barriers for graph cleanup. With clean_mask=true,
            // the post-expand buffer (lut_lbl_ptr) is used and the clean expand
            // barriers surface as 0 in the output.
            lut_.assign(static_cast<size_t>(N) + 1, 0);
            for (int32_t i = 0; i < N; ++i) lut_[i + 1] = colors_[i];
            const int32_t* lut_src =
                (need_orig_snapshot ? orig_labels_.data() : lut_lbl_ptr);
            apply_color_lut_(lut_src, out_ptr, total);
            stage("apply_lut");
        }  // close: if (!early_exit_empty)
        }  // close: gil_scoped_release scope
        return {std::move(out), n_used};
    }

    // Color an arbitrary graph from an edge list, running the same
    // picker ``label`` runs on the pixel-adjacency graph, decoupled from
    // any image. Backs the vector-geometry front end
    // (``ncolor.geo.label`` / ``ncolor.color_graph``), where adjacency
    // comes from polygon topology instead of a pixel walk.
    //
    // ``edges`` is an (M, 2) integer array of **0-indexed** vertex pairs;
    // ``soft_edges`` (optional, same shape) feeds the soft_local_search
    // post-pass: edges that should differ in color when possible but do
    // not constrain the hard coloring. Both lists are normalized to
    // (lo, hi), sorted and de-duplicated, so a symmetric edge list (both
    // (a, b) and (b, a)) is accepted without double-counting. Self-loops
    // and out-of-range endpoints are dropped rather than raising, so a
    // geometry front end can pass its raw candidate pairs.
    //
    // Returns (colors uint8[n_vertices] with values in 1..n_used,
    // n_used). Isolated vertices get color 1. get_last_n_conflicts() /
    // get_last_lut() / get_last_n_soft_violations() are updated exactly
    // as they are by label().
    std::pair<py::array_t<uint8_t>, int> color_graph(
            py::object edges_obj,
            int n_vertices,
            int n_colors = 4,
            int max_depth = 30,
            int rand_period = 10,
            py::object soft_edges_obj = py::none(),
            int color_mode = -1,
            bool capture_stages = false) {
        if (n_vertices < 0) throw std::invalid_argument(
            "Solver.color_graph: n_vertices must be >= 0");
        if (n_colors < 1) throw std::invalid_argument(
            "Solver.color_graph: n_colors must be >= 1");
        const int32_t N = static_cast<int32_t>(n_vertices);

        // Parse both edge arrays HERE, while we still hold the GIL; the
        // compute below runs with it released and may only touch the raw
        // pointers kept alive by the holders.
        auto parse_edges = [](py::object obj, const char* name,
                              py::array_t<int32_t>& holder,
                              const int32_t*& ptr) -> int32_t {
            if (obj.is_none()) return 0;
            holder = py::array_t<int32_t,
                py::array::c_style | py::array::forcecast>::ensure(obj);
            if (!holder) throw std::invalid_argument(
                std::string("Solver.color_graph: ") + name +
                " must be an (M, 2) integer array");
            const auto eb = holder.request();
            if (eb.ndim != 2 || eb.shape[1] != 2) throw std::invalid_argument(
                std::string("Solver.color_graph: ") + name +
                " must have shape (M, 2)");
            ptr = static_cast<const int32_t*>(eb.ptr);
            return static_cast<int32_t>(eb.shape[0]);
        };
        py::array_t<int32_t> edges_holder, soft_holder;
        const int32_t* edge_ptr = nullptr;
        const int32_t* soft_ptr = nullptr;
        const int32_t n_edges_in = parse_edges(edges_obj, "edges",
                                               edges_holder, edge_ptr);
        const int32_t n_soft_in = parse_edges(soft_edges_obj, "soft_edges",
                                              soft_holder, soft_ptr);

        py::array_t<uint8_t> out(static_cast<py::ssize_t>(N));
        uint8_t* out_ptr = static_cast<uint8_t*>(out.request().ptr);

        int n_used = 0;
        last_stages_.clear();
        last_n_conflicts_ = 0;
        n_soft_violations_last_ = 0.0;
        lut_.assign(1, 0);
        std::chrono::steady_clock::time_point t_start, t_now;
        if (capture_stages) t_start = std::chrono::steady_clock::now();
        auto stage = [&](const char* name) {
            if (!capture_stages) return;
            t_now = std::chrono::steady_clock::now();
            last_stages_.emplace_back(name,
                std::chrono::duration<double, std::milli>(t_now - t_start).count());
            t_start = t_now;
        };

        // Drop invalid entries, orient each pair (lo, hi), then sort +
        // unique so duplicates and reversed duplicates collapse.
        auto clean_pairs = [N](const int32_t* src, int32_t count,
                               std::vector<std::pair<int32_t, int32_t>>& dst) {
            dst.clear();
            dst.reserve(static_cast<size_t>(count));
            for (int32_t i = 0; i < count; ++i) {
                int32_t a = src[2 * i], b = src[2 * i + 1];
                if (a < 0 || b < 0 || a >= N || b >= N || a == b) continue;
                dst.emplace_back(std::min(a, b), std::max(a, b));
            }
            std::sort(dst.begin(), dst.end());
            dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
        };

        {
            py::gil_scoped_release release;
            std::lock_guard<std::mutex> engine_lock(pool_->mu);
            if (N > 0) {
                std::vector<std::pair<int32_t, int32_t>> uniq;
                clean_pairs(edge_ptr, n_edges_in, uniq);
                const int32_t M = static_cast<int32_t>(uniq.size());
                src_idx_.resize(static_cast<size_t>(M));
                dst_idx_.resize(static_cast<size_t>(M));
                for (int32_t i = 0; i < M; ++i) {
                    src_idx_[i] = uniq[i].first;
                    dst_idx_[i] = uniq[i].second;
                }
                stage("edges");

                ncolor_cpp::build_csr_from_pairs(src_idx_.data(), dst_idx_.data(),
                                                 N, M, indptr_, indices_);
                stage("build_csr");

                // ndim=2 / wrap=false only steer the picker's internal
                // heuristics (attempt budgets); an abstract graph has no
                // embedding, so the 2D setting is the right neutral value.
                n_used = solve_coloring_(N, M, n_colors, max_depth, rand_period,
                                         color_mode, /*ndim=*/2, /*wrap=*/false);
                stage("color");

                if (n_soft_in > 0 && n_used > 0) {
                    clean_pairs(soft_ptr, n_soft_in, uniq);
                    // build_soft_csr indexes from 1 (it consumes label IDs
                    // in label()); shift our 0-indexed vertices to match.
                    std::vector<int32_t> soft_flat;
                    soft_flat.reserve(2 * uniq.size());
                    for (auto& pr : uniq) {
                        soft_flat.push_back(pr.first + 1);
                        soft_flat.push_back(pr.second + 1);
                    }
                    const int32_t n_soft =
                        static_cast<int32_t>(soft_flat.size() / 2);
                    if (n_soft > 0) {
                        ncolor_cpp::build_soft_csr(
                            N, n_soft, soft_flat.data(), /*weights_in=*/nullptr,
                            soft_indptr_, soft_indices_, soft_weights_);
                        ncolor_cpp::compute_triangle_weights(
                            N, soft_indptr_.data(), soft_indices_.data(),
                            soft_weights_);
                        n_soft_violations_last_ = ncolor_cpp::soft_local_search(
                            colors_.data(), N,
                            indptr_.data(), indices_.data(),
                            soft_indptr_.data(), soft_indices_.data(),
                            soft_weights_.data(),
                            n_used);
                        n_used = ncolor_cpp::densify_colors(colors_, N);
                    }
                    stage("soft_search");
                }

                std::memcpy(out_ptr, colors_.data(),
                            static_cast<size_t>(N) * sizeof(uint8_t));
                lut_.assign(static_cast<size_t>(N) + 1, 0);
                for (int32_t i = 0; i < N; ++i) lut_[i + 1] = colors_[i];
            }
        }
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
    double get_last_n_soft_violations() const { return n_soft_violations_last_; }
    // Soft (delta-kernel) pairs from the most recent label() call: an
    // (M, 2) int32 array of 1-indexed (lo, hi) label ids, hard pairs
    // excluded. Empty when the call built no auto-soft kernel.
    py::array_t<int32_t> get_last_soft_pairs() const {
        return pairs_to_array(fused_soft_pairs_);
    }

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
    // How many distinct nonzero labels an image holds, for sizing the
    // adjacency hashtables. They were sized from the largest label
    // value, which is the same thing for compacted labels and wildly
    // different for sparse ids: an image whose 500 cells were numbered
    // up to a million got a table for a million, 9.4 ms of connect()
    // against 0.6 ms for the same cells numbered 1..500. Distinct count
    // is what the table actually has to hold. The pass is a byte per
    // label value, set in parallel (every writer stores 1, so the race
    // is benign) and counted once; the array is kept between calls
    // like the other scratch. Pairs are still emitted with the labels
    // as given; nothing is renumbered.
    int32_t distinct_labels_(const int32_t* lbl, int64_t total,
                             int32_t max_label) {
        if (max_label <= 0) return 0;
        seen_.assign(static_cast<size_t>(max_label) + 1, 0);
        uint8_t* seen = seen_.data();
        const size_t total_sz = static_cast<size_t>(total);
        if (n_threads_ <= 1 || total < 8192) {
            for (size_t i = 0; i < total_sz; ++i) seen[lbl[i]] = 1;
        } else {
            const size_t n_chunks = static_cast<size_t>(n_threads_) *
                                    ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD;
            const size_t actual_chunks = std::min(n_chunks, total_sz);
            const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
            std::atomic<size_t> next{0};
            pool_->pool.parallel([&]() {
                size_t idx;
                while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < actual_chunks) {
                    const size_t i0 = idx * chunk_sz;
                    const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                    for (size_t i = i0; i < i1; ++i) seen[lbl[i]] = 1;
                }
            });
        }
        int64_t n = 0;
        for (size_t v = 1; v <= static_cast<size_t>(max_label); ++v) n += seen[v];
        return static_cast<int32_t>(n);
    }

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
        pool_->pool.parallel([&, partials_ptr]() {
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
    // from the (ndim, conn) connectivity's forward-neighbor count and
    // the maximum label value, then dispatches to the ND scan kernel.
    // Returns {} for an empty input (max_label == 0).
    std::vector<std::pair<int32_t, int32_t>> find_pairs_(
            const int32_t* labels, const std::vector<int64_t>& shape,
            int conn, bool wrap, int32_t max_label, int radius = 1) {
        if (max_label == 0) return {};
        const int ndim = static_cast<int>(shape.size());
        const int64_t n_fwd = ncolor_cpp::detail::count_forward_neighbors(
            ndim, conn, radius);
        uint64_t ht_size = initial_ht_size_(ndim, n_fwd, max_label);
        // Retry-on-full: the number of DISTINCT adjacency edges is not
        // bounded by n_fwd*max_label — a dense ND Voronoi cell can be
        // face-adjacent to many more neighbors than 2*n_fwd (3D Poisson-
        // Voronoi mean degree ≈ 15.5 vs 2*n_fwd=6 at conn=1). If the
        // estimate is low the open-addressing table fills to 100%; the
        // bounded probe (ht_probe) then drops edges rather than spinning.
        // out.size() == occupancy, so out.size() == ht_size means the
        // table filled and an edge may have been dropped — double and
        // retry. Terminates: distinct edges ≤ max_label², finite.
        for (;;) {
            auto out = ncolor_cpp::find_pairs_nd_unpadded<int32_t>(
                labels, shape, conn,
                ht_size, n_threads_, pool_->pool, wrap, radius,
                &fp_ht_buf_);
            if (out.size() < ht_size || ht_size >= HT_SIZE_CAP) return out;
            ht_size <<= 1;
        }
    }

    // Seed hashtable size for find_pairs. 2*n_fwd*max_label is correct for
    // planar (2D) adjacency graphs (avg degree < 6 ⇒ 2*n_fwd=4 has
    // headroom), but undersizes non-planar ND graphs. Give ndim≥3 enough
    // headroom (degree ~32) to size correctly on the first try; the
    // retry-on-full loop guarantees correctness if even this is low.
    static uint64_t initial_ht_size_(int ndim, int64_t n_fwd,
                                     int32_t n_labels) {
        int64_t deg = 2 * n_fwd;
        if (ndim >= 3) deg = std::max<int64_t>(deg, 32);
        const int64_t ht_raw = deg * static_cast<int64_t>(n_labels);
        return static_cast<uint64_t>(
            ipow2_ge(std::max<int64_t>(ht_raw, MIN_HT_SIZE)));
    }
    // Absolute safety cap on table growth (2^31 slots). Never reached for
    // realistic inputs — distinct edges ≤ max_label² forces termination
    // far sooner — but bounds the retry loop against a pathological case.
    static constexpr uint64_t HT_SIZE_CAP = (uint64_t{1} << 31);

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
            std::vector<int32_t>& counts,
            int radius = 1) {
        primary.clear(); counts.clear();
        if (max_label == 0) return {};
        const int ndim = static_cast<int>(shape.size());
        const int64_t n_fwd = ncolor_cpp::detail::count_forward_neighbors(
            ndim, conn, radius);
        uint64_t ht_size = initial_ht_size_(ndim, n_fwd, max_label);
        // Retry-on-full (see find_pairs_): the weighted reducer arrays are
        // re-cleared and refilled on each call, so a retry is self-consistent.
        for (;;) {
            auto out = ncolor_cpp::find_pairs_weighted_nd_unpadded<int32_t, Mode>(
                labels, dist, shape, conn,
                ht_size, n_threads_, pool_->pool, wrap,
                primary, counts, radius,
                &fp_ht_buf_, &fp_primary_buf_, &fp_counts_buf_);
            if (out.size() < ht_size || ht_size >= HT_SIZE_CAP) return out;
            ht_size <<= 1;
        }
    }

    // The picker itself lives in picker.hpp (no Python dependency); this
    // forwards the Solver's CSR, edge list, color buffer and scratch.
    int solve_coloring_(int32_t N, int32_t M, int n_colors,
                        int max_depth, int rand_period,
                        int color_mode, int ndim, bool wrap,
                        const double* edge_weights = nullptr,
                        const double* de_table = nullptr,
                        int weight_obj = 0) {
        return ncolor_cpp::pick_coloring(
            N, M, n_colors, max_depth, rand_period, color_mode, ndim, wrap,
            edge_weights, de_table, weight_obj,
            indptr_, indices_, src_idx_, dst_idx_,
            colors_, last_n_conflicts_, picker_scratch_,
            &pool_->pool, n_threads_);
    }

    // Apply the color LUT to ``expanded[i]``: bg pixels (bg_mask_[i]==1)
    // get color 0; foreground pixels get ``lut_[expanded[i]]``. Parallel
    // when total ≥ 8192. The bg pattern was captured by ``cast_with_bg``
    // at the start of label(); using a uint8 mask here keeps the inner
    // loop typeless wrt the original input dtype.
    // Apply LUT to fg pixels. bg pixels (bg_mask_[i] == 1) ALWAYS get
    // 0 in the output — `expand` is an internal graph-building thing,
    // not a visual fill. ``src`` provides the label for each fg pixel:
    //   - default (clean_mask=false): a pre-clean-expand snapshot of the
    //     fg labels, so original-fg pixels never lose their cell color
    //     to a barrier zero from the clean expand.
    //   - clean_mask=true OR no snapshot available: the post-expand
    //     buffer (lut_lbl_ptr), which surfaces the clean expand's barriers
    //     as 0 in the output too.
    void apply_color_lut_(const int32_t* src, uint8_t* out_ptr,
                           int64_t total) {
        const int nt = std::max(1, n_threads_);
        const uint8_t* bg_p = bg_mask_.data();
        const uint8_t* lp = lut_.data();
        if (nt == 1 || total < 8192) {
            for (int64_t i = 0; i < total; ++i) {
                out_ptr[i] = bg_p[i] ? 0 : lp[src[i]];
            }
            return;
        }
        ncolor_cpp::dispatch_parallel(pool_->pool, static_cast<size_t>(total),
            static_cast<size_t>(nt) * ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD,
            [bg_p, src, lp, out_ptr](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    out_ptr[i] = bg_p[i] ? 0 : lp[src[i]];
                }
            });
    }

    template <typename V>
    static void drop_(V& v) { V().swap(v); }

    int n_threads_;
    std::shared_ptr<PoolSlot> pool_;
    ncolor_cpp::ExpandBuffers expand_bufs_;
    std::vector<uint8_t> bg_mask_;     // captured from cast, used by apply_lut
    std::vector<int32_t> partials_;     // max-reduce partials, reused across calls
    std::vector<uint8_t> seen_;         // distinct-label scratch, reused across calls
    std::vector<int32_t> src_idx_, dst_idx_;
    std::vector<int32_t> indptr_, indices_;
    // Optional parallel-to-indices_ edge weights used by the boundary-
    // weighted coloring path. Empty/unused for the default coloring.
    // Real-valued to encode EDT-distance-based contact strength.
    std::vector<double> edge_weights_;
    // Soft-edge CSR + weights, populated only when soft_extra_edges is
    // passed to label(). Used by ncolor_cpp::soft_local_search.
    std::vector<int32_t> soft_indptr_, soft_indices_;
    std::vector<float>   soft_weights_;
    double n_soft_violations_last_ = 0.0;  // exposed via accessor
    std::vector<uint8_t> colors_;
    std::vector<uint8_t> lut_;
    // Pre-despur copy of the expanded-label buffer, used by apply_lut so
    // spur pixels (zeroed out of the despurred working buffer) still get
    // their parent cell's color in the final image. Only populated when
    // ``despur_iters > 0``; otherwise apply_lut reads ``expanded`` directly.
    std::vector<int32_t> lut_lbl_;
    // Snapshot of the post-format, pre-expand labels (just the original
    // foreground pixels with their 1..N IDs; bg is 0). Used by
    // apply_color_lut_ when clean_mask=false so original-fg pixels
    // ALWAYS get their cell's color even when the clean expand zeroed them
    // as bridges/stubs for graph cleanup. Only populated when
    // clean_mask=false AND expand_mode="clean" AND expand=true.
    std::vector<int32_t> orig_labels_;
    // Per-pixel same-label face-neighbour count, reused by fast_despur
    // (compute_face_count_nd + despur_via_face_count_nd). Only allocated
    // when fast_despur runs.
    std::vector<uint8_t> despur_face_count_;
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
    // Second per-thread HT buffer for the dual base+soft find_pairs path
    // (auto-build of soft_extra_edges). Empty / unused when soft_conn or
    // soft_radius is 0.
    std::vector<uint64_t> fp_soft_ht_buf_;
    // Soft (delta-only) pair list captured by the fused base+soft scan in
    // label(). Consumed by the soft_local_search post-pass without a
    // second pixel walk. Cleared on every non-soft label() call.
    std::vector<std::pair<int32_t, int32_t>> fused_soft_pairs_;
    int last_n_conflicts_ = 0;
    std::vector<std::pair<std::string, double>> last_stages_;
    // Per-attempt scratch for the picker's parallel race. Reused across calls.
    ncolor_cpp::PickerScratch picker_scratch_;
};

PYBIND11_MODULE(_impl, m) {
    m.doc() = "ncolor C++ engine: connect / expand / color pipeline + "
              "ForkJoinPool. Public Python API in ncolor.color, ncolor.expand "
              "wraps the engines exposed here.";
    py::class_<ExpandEngine>(m, "ExpandEngine",
        "Persistent threadpool wrapper for expand_labels + format_labels.\n"
        "One engine per pipeline; the pool and intermediate buffers are\n"
        "reused across calls.")
        .def(py::init<double, int>(), py::arg("n_threads") = -1.0,
             py::arg("pool_group") = 0)
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
        .def("expand_labels_clean", &ExpandEngine::expand_labels_clean,
             py::arg("labels"), py::arg("p") = 2,
             "Bridge-free Voronoi label expansion (2D only for now).\n"
             "Identical to expand_labels except an antipodal-only bridge\n"
             "test runs on the final 2D-Voronoi labels: pixels with\n"
             "exactly two same-label neighbors arranged antipodally\n"
             "(N-S, E-W, NE-SW, or NW-SE) are marked bg. Prevents 1-\n"
             "pixel-wide bridges (face or corner) from connecting cells.\n"
             "p=1 (L1, Saito-Toriwaki) tends to produce many more such\n"
             "bridges than p=2 (L2, Felzenszwalb) — L1 is the metric\n"
             "where the test actually changes the output materially.\n"
             "ND > 2 falls back to standard expand_labels (no bridge\n"
             "prevention) until 3D antipodal generalization lands.")
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
             "Accepts bool, uint8/16/32/64, int8/16/32/64 and float32/64\n"
             "input; the cast to int32 happens in parallel inside the\n"
             "released-GIL block and raises OverflowError if a value does\n"
             "not fit. Returns (formatted_array, n_labels).")
        .def("release", &ExpandEngine::release,
             "Free the persistent scratch buffers; the next call reallocates.");

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
        .def(py::init<double, int>(), py::arg("n_threads") = -1.0,
             py::arg("pool_group") = 0)
        .def_property_readonly("n_threads", &Solver::n_threads)
        .def("label", &Solver::label,
             py::arg("mask"), py::arg("n_colors") = 4,
             py::arg("max_depth") = 30, py::arg("rand_period") = 10,
             py::arg("conn") = 1,
             py::arg("p") = 2, py::arg("capture_stages") = false,
             py::arg("format_input") = true, py::arg("expand") = true,
             py::arg("out") = py::none(), py::arg("color_mode") = -1,
             py::arg("wrap") = false,
             py::arg("first_seen") = false,
             py::arg("weight_objective") = 0,
             py::arg("de_table") = py::none(),
             py::arg("weight_mode") = 1,
             py::arg("extra_edges") = py::none(),
             py::arg("connect_radius") = 1,
             py::arg("despur_iters") = 0,
             py::arg("despur_remove_thin") = false,
             py::arg("min_contact") = 1,
             py::arg("expand_mode") = "clean",
             py::arg("soft_extra_edges") = py::none(),
             py::arg("soft_conn") = 2,
             py::arg("soft_radius") = 2,
             py::arg("clean_mask") = false,
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
             "neighbors, top/bottom edges are neighbors), adding wrap-\n"
             "around adjacencies between cells whose Voronoi territories\n"
             "land on opposite image edges. Useful for tile-equivalent or\n"
             "periodic-imaging assumptions; balances color frequencies on\n"
             "tightly-cropped microcolony images at ~zero runtime cost.")
        .def("connect", &Solver::connect,
             py::arg("mask"), py::arg("conn") = 1, py::arg("wrap") = false,
             "Adjacency pairs for a label image. Returns an (M, 2) int32\n"
             "array of unique (lo, hi) label pairs that share a boundary\n"
             "under connectivity ``conn``. Mirrors ncolor.connect()'s\n"
             "signature; runs the cpp connect kernel directly.\n"
             "wrap=True treats the image as a torus (opposite edges are\n"
             "adjacent), adding wrap-around pairs between cells on the\n"
             "image perimeter.")
        .def("color_graph", &Solver::color_graph,
             py::arg("edges"), py::arg("n_vertices"),
             py::arg("n_colors") = 4, py::arg("max_depth") = 30,
             py::arg("rand_period") = 10,
             py::arg("soft_edges") = py::none(),
             py::arg("color_mode") = -1,
             py::arg("capture_stages") = false,
             "Color an abstract graph given its edge list, using the same\n"
             "picker label() runs on the pixel-adjacency graph.\n\n"
             "``edges`` is an (M, 2) integer array of 0-indexed vertex\n"
             "pairs; duplicate / reversed / self / out-of-range entries are\n"
             "normalized away. ``soft_edges`` (same shape) feeds the\n"
             "soft-constraint local search. Returns (colors uint8[n_vertices]\n"
             "in 1..n_used, n_used).")
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
             "means the solver bailed out without finding a clean coloring.")
        .def("get_last_n_soft_violations", &Solver::get_last_n_soft_violations,
             "Total weight (or count if unit weights) of soft_extra_edges\n"
             "whose endpoints share a color after the post-solve local\n"
             "search. 0 means all soft preferences satisfied. Only nonzero\n"
             "when soft_extra_edges was passed to label().")
        .def("get_last_soft_pairs", &Solver::get_last_soft_pairs,
             "Soft (delta-kernel) pairs from the most recent label() call as\n"
             "an (M, 2) int32 array of 1-indexed (lo, hi) label ids, hard\n"
             "pairs excluded. Empty when no auto-soft kernel was built.")
        .def("release", &Solver::release,
             "Free every persistent scratch buffer (the working set of the\n"
             "largest image processed so far); the next call reallocates.");

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
                  dispatch_cast_dtype(buf.format, buf.itemsize, "cc_label",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          n_labels = ncolor_cpp::cc_label_nd<T>(
                              static_cast<const T*>(src_ptr), out_ptr, shape, conn);
                      });
              }
              return {std::move(out), n_labels};
          },
          py::arg("mask"), py::arg("conn") = 2,
          "N-D connected-components labeling. Returns (labels, n_components).\n"
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
          "Region properties of a dense int32 1..N labeled image.\n"
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
          "fg-neighbor count under the chosen connectivity is below\n"
          "``threshold`` (default ndim). ``conn_kind`` = 1 → cardinal\n"
          "(face only, omnipose-style external-spur rule, fewer iters);\n"
          "ndim → full diagonal (preserves 1-voxel skeletons). Isolated\n"
          "pixels (count == 0) are always preserved. ``max_iter`` < 0\n"
          "runs to convergence.");

    m.def("delete_spurs_labels",
          [](py::array labels_in, int threshold, int max_iters, int n_threads,
             bool remove_thin) {
              if (!(labels_in.flags() & py::array::c_style)) {
                  labels_in = py::array::ensure(labels_in, py::array::c_style);
              }
              const auto buf = labels_in.request();
              const int ndim = static_cast<int>(buf.ndim);
              if (ndim < 1) throw std::invalid_argument(
                  "delete_spurs_labels requires ndim >= 1");
              std::vector<int64_t> shape(ndim);
              std::vector<py::ssize_t> out_shape(ndim);
              for (int d = 0; d < ndim; ++d) {
                  shape[d]     = static_cast<int64_t>(buf.shape[d]);
                  out_shape[d] = static_cast<py::ssize_t>(buf.shape[d]);
              }
              py::array out(labels_in.dtype(), out_shape);
              std::memcpy(out.request().ptr, buf.ptr,
                          (size_t)buf.size * (size_t)buf.itemsize);
              int64_t n_removed = 0;
              const int nt = n_threads > 0 ? n_threads :
                  (int)std::thread::hardware_concurrency();
              {
                  py::gil_scoped_release release;
                  std::unique_ptr<ncolor_cpp::ForkJoinPool> pool;
                  if (nt > 1) {
                      pool = std::make_unique<ncolor_cpp::ForkJoinPool>(nt);
                  }
                  dispatch_int_dtype(buf.format, buf.itemsize, "delete_spurs_labels",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          n_removed = ncolor_cpp::delete_spurs_labels_nd_inplace<T>(
                              static_cast<T*>(out.mutable_data()),
                              shape, threshold, max_iters,
                              pool.get(), nt, remove_thin);
                      });
              }
              return std::make_pair(std::move(out), n_removed);
          },
          py::arg("labels"), py::arg("threshold") = 1, py::arg("max_iters") = 20,
          py::arg("n_threads") = 0, py::arg("remove_thin") = false,
          "Label-aware despur. Iteratively zeros pixels whose count of\n"
          "face-adjacent SAME-label neighbors is ≤ threshold. Returns\n"
          "(cleaned_labels, n_removed). threshold=1 removes pixels with\n"
          "only 1 same-label neighbor AND isolated pixels (count 0).\n"
          "Stops when no further removals or after max_iters.\n\n"
          "``remove_thin=True`` also zeros 1-voxel-thick straight\n"
          "interior pixels in the SAME pass (a pixel with exactly two\n"
          "same-label 8-connectivity neighbors that sit at opposite\n"
          "offsets — axis-aligned in 3D+, axis-aligned and diagonal\n"
          "in 2D). Useful for cleaning up 1-px bridges left by L1\n"
          "Voronoi expand without paying the iter-by-iter end-peeling\n"
          "cost.");

    // Fast despur built on a pre-computed face-count array. Avoids the
    // iter-0 full-image scan that dominates ``delete_spurs_labels``.
    m.def("fast_despur",
          [](py::array labels_in, int threshold, int n_threads) {
              if (!(labels_in.flags() & py::array::c_style)) {
                  labels_in = py::array::ensure(labels_in, py::array::c_style);
              }
              const auto buf = labels_in.request();
              const int ndim = static_cast<int>(buf.ndim);
              std::vector<int64_t> shape(ndim);
              std::vector<py::ssize_t> out_shape(ndim);
              for (int d = 0; d < ndim; ++d) {
                  shape[d]     = static_cast<int64_t>(buf.shape[d]);
                  out_shape[d] = static_cast<py::ssize_t>(buf.shape[d]);
              }
              py::array out(labels_in.dtype(), out_shape);
              std::memcpy(out.request().ptr, buf.ptr,
                          (size_t)buf.size * (size_t)buf.itemsize);
              int64_t n_removed = 0;
              const int nt = n_threads > 0 ? n_threads :
                  (int)std::thread::hardware_concurrency();
              const int64_t total = (int64_t)buf.size;
              std::vector<uint8_t> face_count((size_t)total);
              {
                  py::gil_scoped_release release;
                  std::unique_ptr<ncolor_cpp::ForkJoinPool> pool;
                  if (nt > 1) pool = std::make_unique<ncolor_cpp::ForkJoinPool>(nt);
                  dispatch_int_dtype(buf.format, buf.itemsize, "fast_despur",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          T* lbl = static_cast<T*>(out.mutable_data());
                          ncolor_cpp::compute_face_count_nd<T>(
                              lbl, face_count.data(), shape, pool.get(), nt);
                          n_removed = ncolor_cpp::despur_via_face_count_nd<T>(
                              lbl, face_count.data(), shape,
                              threshold, pool.get(), nt);
                      });
              }
              return std::make_pair(std::move(out), n_removed);
          },
          py::arg("labels"), py::arg("threshold") = 1, py::arg("n_threads") = 0,
          "Fast despur: precompute per-pixel same-label face-neighbour\n"
          "count once (parallel branchless scan), then peel back spurs\n"
          "via a queue that decrements neighbours' counts on revert.\n"
          "No full-image rescans after iter 0.");

    // Fused connect+face_count scan: returns (pairs, face_count) from
    // one cache-warm pass. 2D only for now.
    m.def("find_pairs_with_face_count_2d",
          [](py::array labels_in, int conn, int n_threads, uint64_t ht_size) {
              if (!(labels_in.flags() & py::array::c_style)) {
                  labels_in = py::array::ensure(labels_in, py::array::c_style);
              }
              const auto buf = labels_in.request();
              if (buf.ndim != 2) throw std::invalid_argument("2D only");
              const int64_t H = buf.shape[0];
              const int64_t W = buf.shape[1];
              const int nt = n_threads > 0 ? n_threads :
                  (int)std::thread::hardware_concurrency();
              std::vector<py::ssize_t> fc_shape = {(py::ssize_t)H, (py::ssize_t)W};
              py::array_t<uint8_t> face_count(fc_shape);
              std::memset(face_count.mutable_data(), 0, (size_t)(H * W));
              std::vector<std::pair<int32_t, int32_t>> pairs;
              {
                  py::gil_scoped_release release;
                  ncolor_cpp::ForkJoinPool pool(nt);
                  dispatch_int_dtype(buf.format, buf.itemsize, "find_pairs_with_face_count_2d",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          pairs = ncolor_cpp::find_pairs_with_face_count_2d<T>(
                              static_cast<const T*>(buf.ptr), H, W, conn,
                              face_count.mutable_data(), ht_size, nt, pool);
                      });
              }
              py::array_t<int32_t> pair_arr(
                  {(py::ssize_t)pairs.size(), (py::ssize_t)2});
              auto pa = pair_arr.mutable_unchecked<2>();
              for (size_t k = 0; k < pairs.size(); ++k) {
                  pa(k, 0) = pairs[k].first;
                  pa(k, 1) = pairs[k].second;
              }
              return std::make_tuple(std::move(pair_arr), std::move(face_count));
          },
          py::arg("labels"), py::arg("conn") = 2, py::arg("n_threads") = 0,
          py::arg("ht_size") = (uint64_t)65536,
          "Fused 2D connect+face_count scan. Returns (pairs[K,2],\n"
          "face_count[H,W]) in a single cache-warm pass — used to\n"
          "replace separate find_pairs + compute_face_count calls.");

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
          "Build 2-hop neighbor CSR from a 1-hop adjacency CSR.\n"
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
