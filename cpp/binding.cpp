/*
 * Pybind11 binding for ncolor C++. Exposes three classes — each wraps a
 * persistent ForkJoinPool, so callers construct once and reuse:
 *   - ``ConnectEngine``  : adjacency-pair search (``find_pairs``)
 *   - ``ExpandEngine``   : Voronoi label expansion (``expand_labels``)
 *   - ``Solver``         : end-to-end ncolor.label pipeline
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "chamfer.hpp"
#include "color.hpp"
#include "expand_lp.hpp"
#include "format_labels.hpp"
#include "connect.hpp"
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

class ConnectEngine {
public:
    explicit ConnectEngine(double n_threads)
        : n_threads_(resolve_threads(n_threads)),
          pool_(std::make_unique<ncolor_cpp::ForkJoinPool>(n_threads_ <= 1 ? 1 : n_threads_)) {}

    int n_threads() const { return n_threads_; }

    // Match ncolor.color.connect's signature: takes the padded label line as
    // a 1D int32 array, neighbor offsets as int64, and a power-of-two hash
    // table size. Returns an (M, 2) int32 numpy array of unique (lo, hi) pairs.
    py::array_t<int32_t> find_pairs(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> line,
            py::array_t<int64_t, py::array::c_style | py::array::forcecast> nbs,
            uint64_t ht_size,
            int parallel_threshold = 100000) {
        const auto line_buf = line.request();
        const auto nbs_buf = nbs.request();
        if (line_buf.ndim != 1) throw std::invalid_argument("line must be 1D");
        if (nbs_buf.ndim != 1) throw std::invalid_argument("nbs must be 1D");
        if ((ht_size & (ht_size - 1)) != 0)
            throw std::invalid_argument("ht_size must be a power of two");

        const int32_t* line_ptr = static_cast<const int32_t*>(line_buf.ptr);
        const int64_t* nbs_ptr = static_cast<const int64_t*>(nbs_buf.ptr);
        const int64_t total = line_buf.shape[0];
        const int n_nbs = static_cast<int>(nbs_buf.shape[0]);

        std::vector<std::pair<int32_t, int32_t>> pairs;
        if (total > parallel_threshold && n_threads_ > 1) {
            py::gil_scoped_release release;
            pairs = ncolor_cpp::search_hashset_parallel<int32_t>(
                line_ptr, total, nbs_ptr, n_nbs, ht_size, n_threads_, *pool_);
        } else {
            py::gil_scoped_release release;
            pairs = ncolor_cpp::search_hashset_serial<int32_t>(
                line_ptr, total, nbs_ptr, n_nbs, ht_size);
        }

        // Pack into (M, 2) int32 ndarray.
        const py::ssize_t m = static_cast<py::ssize_t>(pairs.size());
        py::array_t<int32_t> out({m, py::ssize_t{2}});
        auto out_buf = out.request();
        int32_t* out_ptr = static_cast<int32_t*>(out_buf.ptr);
        for (py::ssize_t i = 0; i < m; ++i) {
            out_ptr[i * 2 + 0] = pairs[i].first;
            out_ptr[i * 2 + 1] = pairs[i].second;
        }
        return out;
    }

private:
    int n_threads_;
    std::unique_ptr<ncolor_cpp::ForkJoinPool> pool_;
};

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
        // Detect dtype by (itemsize, signedness) — same approach as
        // Solver.label (handles macOS 'l' vs pybind11 'q' for int64).
        const std::string fmt = buf.format;
        const ssize_t itemsize = buf.itemsize;
        bool is_signed = false, is_unsigned = false;
        if (!fmt.empty()) {
            const char c = fmt[0];
            if (c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' || c == 'n')
                is_signed = true;
            else if (c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' || c == 'N')
                is_unsigned = true;
        }
        if (!(is_signed || is_unsigned)) {
            throw std::invalid_argument(
                "ExpandEngine.format_labels: unsupported dtype '" + fmt +
                "' (need uint8/uint16/uint32/int8/int16/int32/int64)");
        }
        const bool is_uint8  = is_unsigned && itemsize == 1;
        const bool is_uint16 = is_unsigned && itemsize == 2;
        const bool is_uint32 = is_unsigned && itemsize == 4;
        const bool is_int8   = is_signed   && itemsize == 1;
        const bool is_int16  = is_signed   && itemsize == 2;
        const bool is_int32  = is_signed   && itemsize == 4;
        const bool is_int64  = is_signed   && itemsize == 8;
        if (!(is_int32 || is_uint8 || is_uint16 || is_uint32
              || is_int8 || is_int16 || is_int64)) {
            throw std::invalid_argument(
                "ExpandEngine.format_labels: unsupported dtype (itemsize=" +
                std::to_string(itemsize) + ")");
        }
        const void* src_ptr = buf.ptr;

        // Allocate int32 output array. pybind11 will value-init (memset 0)
        // from the main thread — that pre-touches the pages, but we want
        // first-touch from worker threads in cast_to_int32 for NUCA-locality.
        // Allocate via shape vector (same effect; pybind value-inits as
        // before, but the cast immediately rewrites every byte from the
        // worker that owns each chunk, and modern allocators don't zero
        // freshly-mapped pages anyway).
        std::vector<py::ssize_t> out_shape(buf.ndim);
        for (ssize_t d = 0; d < buf.ndim; ++d) out_shape[d] = buf.shape[d];
        py::array_t<int32_t> out(out_shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);

        int n_labels;
        {
            py::gil_scoped_release release;
            // Cast to int32 in parallel inside the released-GIL block.
            if (is_int32) {
                ncolor_cpp::cast_to_int32<int32_t>(
                    static_cast<const int32_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            } else if (is_uint8) {
                ncolor_cpp::cast_to_int32<uint8_t>(
                    static_cast<const uint8_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            } else if (is_uint16) {
                ncolor_cpp::cast_to_int32<uint16_t>(
                    static_cast<const uint16_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            } else if (is_uint32) {
                ncolor_cpp::cast_to_int32<uint32_t>(
                    static_cast<const uint32_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            } else if (is_int8) {
                ncolor_cpp::cast_to_int32<int8_t>(
                    static_cast<const int8_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            } else if (is_int16) {
                ncolor_cpp::cast_to_int32<int16_t>(
                    static_cast<const int16_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            } else /* is_int64 */ {
                ncolor_cpp::cast_to_int32<int64_t>(
                    static_cast<const int64_t*>(src_ptr), out_ptr, total, *pool_, n_threads_);
            }
            n_labels = first_seen
                ? ncolor_cpp::format_labels_inplace_first_seen(
                    out_ptr, total, *pool_, n_threads_)
                : ncolor_cpp::format_labels_inplace(
                    out_ptr, total, *pool_, n_threads_);
        }
        return {std::move(out), n_labels};
    }

    // Same as expand_labels but returns a (output, list[(stage_name, ms)])
    // tuple — used by callers that want to attribute time to expand /
    // find_pairs / build_csr / color / apply_lut. p=2 only for now.
    std::pair<py::array_t<int32_t>, std::vector<std::pair<std::string, double>>>
    expand_labels_timed(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels) {
        const auto buf = labels.request();
        std::vector<int64_t> shape(buf.ndim);
        for (int i = 0; i < buf.ndim; ++i) shape[i] = buf.shape[i];
        const int32_t* input = static_cast<const int32_t*>(buf.ptr);
        py::array_t<int32_t> out(buf.shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);
        std::vector<std::pair<std::string, double>> stages;
        {
            py::gil_scoped_release release;
            ncolor_cpp::expand_labels_inplace_timed(input, bufs_, shape, *pool_, n_threads_,
                [&](const char* name, double ms) { stages.emplace_back(name, ms); });
            std::memcpy(out_ptr, bufs_.lbl(), bufs_.size() * sizeof(int32_t));
        }
        return {std::move(out), std::move(stages)};
    }

    // Parallel scatter: out[i] = lut[flat_lab[i]] for every i. Used by
    // ncolor's coloring step to apply the per-label color LUT to the image.
    template <typename LutT>
    py::array_t<LutT> apply_lut_impl(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> flat_lab,
            py::array_t<LutT, py::array::c_style | py::array::forcecast> lut) {
        const auto lab_buf = flat_lab.request();
        const auto lut_buf = lut.request();
        const int64_t total = lab_buf.size;
        const int32_t* lab_ptr = static_cast<const int32_t*>(lab_buf.ptr);
        const LutT* lut_ptr = static_cast<const LutT*>(lut_buf.ptr);
        py::array_t<LutT> out(lab_buf.shape);
        LutT* out_ptr = static_cast<LutT*>(out.request().ptr);

        py::gil_scoped_release release;
        const int n_threads = n_threads_;
        if (n_threads <= 1 || total < 4096) {
            for (int64_t i = 0; i < total; ++i) {
                out_ptr[i] = lut_ptr[lab_ptr[i]];
            }
            return out;
        }
        ncolor_cpp::dispatch_parallel(*pool_, static_cast<size_t>(total),
            static_cast<size_t>(n_threads) * ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD,
            [lab_ptr, lut_ptr, out_ptr](size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    out_ptr[i] = lut_ptr[lab_ptr[i]];
                }
            });
        return out;
    }

    py::array apply_lut(py::array flat_lab, py::array lut) {
        // Dispatch on lut dtype (uint8 for ncolor coloring; int32 generally).
        const auto kind = lut.dtype().kind();
        const auto itemsize = lut.dtype().itemsize();
        if (kind == 'u' && itemsize == 1) {
            return apply_lut_impl<uint8_t>(flat_lab, lut);
        }
        if (kind == 'i' && itemsize == 4) {
            return apply_lut_impl<int32_t>(flat_lab, lut);
        }
        throw std::invalid_argument("apply_lut: lut dtype must be uint8 or int32");
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

// Solver: end-to-end ncolor.label equivalent in C++. Wraps a ConnectEngine
// and ExpandEngine plus its own scratch for CSR build + coloring.
class Solver {
public:
    explicit Solver(double n_threads)
        : n_threads_(resolve_threads(n_threads)),
          pool_(std::make_unique<ncolor_cpp::ForkJoinPool>(n_threads_ <= 1 ? 1 : n_threads_)) {}

    int n_threads() const { return n_threads_; }

    // ncolor.label(mask, n=4, expand=True, return_n=False) for 2D conn=1.
    // Returns (colored_image, n_colors_used).
    // The coloring step is single-attempt with 4 colors — for the rare graph
    // where 4 colors aren't enough or BFS fails to converge, the caller
    // should fall back to ncolor.label (which has the full repair chain).
    //
    // ``p`` selects the expand metric: p=2 (Felzenszwalb parabolic
    // envelope, default — bit-identical to numba's L2 expand) or p=1
    // (Saito-Toriwaki sweep — Manhattan distance, ~5× faster at 2048²).
    // Boundary placement under p=1 differs from p=2 at ~5% of pixels;
    // the adjacency graph is nearly always isomorphic — the 4-coloring
    // still works either way.
    // Per-stage timing breakdown (filled by `label` when capture_stages=true).
    std::vector<std::pair<std::string, double>> last_stages_;
    std::vector<std::pair<std::string, double>> get_last_stages() const { return last_stages_; }

    // Adjacency pairs for a label image, mirroring ncolor.color.connect.
    // Takes the image directly (any of the supported integer dtypes) and
    // returns an (M, 2) int32 array of unique (lo, hi) pairs of adjacent
    // labels under connectivity ``conn`` (1..ndim). 2D conn=2 takes the
    // unpadded fast path; everything else pads and uses the generic
    // hashset parallel search.
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

        // (itemsize, signedness) dispatch — see the parallel block in
        // Solver.label for why this beats format-code matching.
        const std::string fmt = buf.format;
        const ssize_t itemsize = buf.itemsize;
        bool is_signed = false, is_unsigned = false;
        if (!fmt.empty()) {
            const char c = fmt[0];
            if (c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' || c == 'n') is_signed = true;
            else if (c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' || c == 'N') is_unsigned = true;
        }
        const bool is_uint8  = is_unsigned && itemsize == 1;
        const bool is_uint16 = is_unsigned && itemsize == 2;
        const bool is_uint32 = is_unsigned && itemsize == 4;
        const bool is_int8   = is_signed   && itemsize == 1;
        const bool is_int16  = is_signed   && itemsize == 2;
        const bool is_int32  = is_signed   && itemsize == 4;
        const bool is_int64  = is_signed   && itemsize == 8;
        if (!(is_int32 || is_uint8 || is_uint16 || is_uint32
              || is_int8 || is_int16 || is_int64)) {
            throw std::invalid_argument(
                "Solver.connect: unsupported dtype '" + fmt + "'");
        }
        const void* src_ptr = buf.ptr;

        std::vector<std::pair<int32_t, int32_t>> pairs;
        {
            py::gil_scoped_release release;
            // Cast to int32 in expand_bufs_.lbl(); discard bg pattern
            // (connect doesn't need it).
            expand_bufs_.resize(total);
            int32_t* labels = expand_bufs_.lbl();
            bg_mask_.resize(static_cast<size_t>(total));
            uint8_t* bg = bg_mask_.data();
            if (is_int32) ncolor_cpp::cast_with_bg<int32_t>(
                static_cast<const int32_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);
            else if (is_uint8) ncolor_cpp::cast_with_bg<uint8_t>(
                static_cast<const uint8_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);
            else if (is_uint16) ncolor_cpp::cast_with_bg<uint16_t>(
                static_cast<const uint16_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);
            else if (is_uint32) ncolor_cpp::cast_with_bg<uint32_t>(
                static_cast<const uint32_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);
            else if (is_int8) ncolor_cpp::cast_with_bg<int8_t>(
                static_cast<const int8_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);
            else if (is_int16) ncolor_cpp::cast_with_bg<int16_t>(
                static_cast<const int16_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);
            else /* is_int64 */ ncolor_cpp::cast_with_bg<int64_t>(
                static_cast<const int64_t*>(src_ptr), labels, bg, total, *pool_, n_threads_);

            // Find max for hashtable sizing.
            int32_t max_label = 0;
            for (int64_t i = 0; i < total; ++i) {
                if (labels[i] > max_label) max_label = labels[i];
            }
            if (max_label == 0) {
                // Empty input — no pairs.
            } else {
                // Unified ND unpadded: count forward neighbours for sizing.
                int64_t n_fwd = 0;
                std::vector<int8_t> dc(ndim, -1);
                while (true) {
                    int n_nz = 0, first_nz = -1;
                    for (int d = 0; d < ndim; ++d) if (dc[d] != 0) {
                        if (first_nz < 0) first_nz = d;
                        ++n_nz;
                    }
                    if (n_nz > 0 && n_nz <= conn && first_nz >= 0 && dc[first_nz] == 1) ++n_fwd;
                    int d = ndim - 1;
                    while (d >= 0) {
                        ++dc[d];
                        if (dc[d] <= 1) break;
                        dc[d] = -1;
                        --d;
                    }
                    if (d < 0) break;
                }
                const int64_t ht_raw = 2 * n_fwd * static_cast<int64_t>(max_label);
                const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, 16));
                pairs = ncolor_cpp::find_pairs_nd_unpadded<int32_t>(
                    labels, shape, conn,
                    static_cast<uint64_t>(ht_size), n_threads_, *pool_, wrap);
            }
        }

        // Pack as (M, 2) int32 ndarray.
        const py::ssize_t m = static_cast<py::ssize_t>(pairs.size());
        py::array_t<int32_t> out({m, py::ssize_t{2}});
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);
        for (py::ssize_t i = 0; i < m; ++i) {
            out_ptr[i * 2 + 0] = pairs[i].first;
            out_ptr[i * 2 + 1] = pairs[i].second;
        }
        return out;
    }

    std::pair<py::array_t<uint8_t>, int> label(
            py::array mask,
            int n_colors = 4, int max_depth = 30, int rand_period = 10,
            int conn = 2, int p = 2, bool capture_stages = false,
            bool format_input = true, bool expand = true,
            py::object out_arg = py::none(),
            int color_mode = -1, bool wrap = false, bool balance = false) {
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
        // 2D fast path uses the unpadded ConnectEngine optimisation; ND or
        // non-default conn falls back to the generic padded path.
        // Unified ND unpadded find_pairs handles all (ndim, conn) cases.

        // Validate + classify dtype by (itemsize, signedness). Doing it
        // by buf.format string runs into a platform quirk: numpy's int64
        // is 'l' (long) on macOS / Linux LP64 while pybind11's
        // format_descriptor<int64_t>::format() is 'q' (long long).
        // (itemsize, signedness) is the underlying truth that's portable.
        const std::string fmt = buf.format;
        const ssize_t itemsize = buf.itemsize;
        bool is_signed = false, is_unsigned = false;
        if (!fmt.empty()) {
            const char c = fmt[0];
            // pybind11 / numpy struct format codes.
            if (c == 'b' || c == 'h' || c == 'i' || c == 'l' || c == 'q' || c == 'n')
                is_signed = true;
            else if (c == 'B' || c == 'H' || c == 'I' || c == 'L' || c == 'Q' || c == 'N')
                is_unsigned = true;
        }
        if (!(is_signed || is_unsigned)) {
            throw std::invalid_argument(
                "Solver.label: unsupported dtype '" + fmt +
                "' (need one of: uint8, uint16, uint32, int8, int16, int32, int64)");
        }
        const bool is_uint8  = is_unsigned && itemsize == 1;
        const bool is_uint16 = is_unsigned && itemsize == 2;
        const bool is_uint32 = is_unsigned && itemsize == 4;
        const bool is_int8   = is_signed   && itemsize == 1;
        const bool is_int16  = is_signed   && itemsize == 2;
        const bool is_int32  = is_signed   && itemsize == 4;
        const bool is_int64  = is_signed   && itemsize == 8;
        if (!(is_int32 || is_uint8 || is_uint16 || is_uint32
              || is_int8 || is_int16 || is_int64)) {
            throw std::invalid_argument(
                "Solver.label: unsupported dtype (itemsize=" +
                std::to_string(itemsize) + ", " +
                (is_signed ? "signed" : "unsigned") + ")");
        }
        const void* src_ptr = buf.ptr;

        std::vector<py::ssize_t> out_shape(ndim);
        for (int d = 0; d < ndim; ++d) out_shape[d] = static_cast<py::ssize_t>(shape[d]);
        py::array_t<uint8_t> out;
        if (out_arg.is_none()) {
            out = py::array_t<uint8_t>(out_shape);
        } else {
            // Caller-supplied buffer: must be uint8, C-contiguous, exact-shape.
            // Reusing an output buffer across calls saves the per-call alloc
            // (16 MiB at 4096²); useful for batch pipelines. We need strict
            // dtype check (not pybind11's auto-cast) so that the caller's
            // buffer is actually the one written — a silent copy would defeat
            // the purpose of passing out=.
            const py::array out_view = py::cast<py::array>(out_arg);
            if (out_view.dtype().kind() != 'u' || out_view.dtype().itemsize() != 1) {
                throw std::invalid_argument(
                    "Solver.label: out buffer must be uint8");
            }
            out = py::cast<py::array_t<uint8_t>>(out_arg);
            const auto out_buf = out.request();
            if (out_buf.ndim != ndim) throw std::invalid_argument(
                "Solver.label: out buffer ndim does not match input");
            for (int d = 0; d < ndim; ++d) {
                if (out_buf.shape[d] != buf.shape[d]) throw std::invalid_argument(
                    "Solver.label: out buffer shape does not match input");
            }
            if (!(out.flags() & py::array::c_style)) throw std::invalid_argument(
                "Solver.label: out buffer must be C-contiguous");
        }
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
            if (is_int32) {
                ncolor_cpp::cast_with_bg<int32_t>(
                    static_cast<const int32_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            } else if (is_uint8) {
                ncolor_cpp::cast_with_bg<uint8_t>(
                    static_cast<const uint8_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            } else if (is_uint16) {
                ncolor_cpp::cast_with_bg<uint16_t>(
                    static_cast<const uint16_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            } else if (is_uint32) {
                ncolor_cpp::cast_with_bg<uint32_t>(
                    static_cast<const uint32_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            } else if (is_int8) {
                ncolor_cpp::cast_with_bg<int8_t>(
                    static_cast<const int8_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            } else if (is_int16) {
                ncolor_cpp::cast_with_bg<int16_t>(
                    static_cast<const int16_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            } else /* is_int64 */ {
                ncolor_cpp::cast_with_bg<int64_t>(
                    static_cast<const int64_t*>(src_ptr),
                    expanded, bg, total, *pool_, n_threads_);
            }
            stage("cast");

            // 0b. Optional format_labels: compact nonzero labels to 1..N
            // in place inside expand_bufs_.lbl(). When format_input=False
            // the caller is asserting labels are already 1..N.
            const int32_t* expand_input = expanded;
            if (format_input) {
                const int n_labels = ncolor_cpp::format_labels_inplace(
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
                if (p == 2) {
                    ncolor_cpp::expand_labels_lp<2>(expand_input, expanded, expand_bufs_, shape, *pool_, n_threads_, wrap);
                } else {
                    ncolor_cpp::expand_labels_lp<1>(expand_input, expanded, expand_bufs_, shape, *pool_, n_threads_, wrap);
                }
            }
            // (Suppress unused-var warning when expand=false — expanded
            // already equals expand_input == expand_bufs_.lbl().)
            (void)expand_input;
            stage("expand");

            // 2. Find adjacency pairs.
            // Parallel max-reduce — was a single-threaded 1.2 ms loop at 2048².
            int32_t max_label = 0;
            const int64_t hw = total;
            if (n_threads_ <= 1 || hw < 8192) {
                for (int64_t i = 0; i < hw; ++i) {
                    if (expanded[i] > max_label) max_label = expanded[i];
                }
            } else {
                // Per-chunk partial maxes; final reduce on the main thread.
                const size_t n_chunks = static_cast<size_t>(n_threads_) *
                                        ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD;
                partials_.assign(n_chunks, 0);
                std::atomic<size_t> next{0};
                const size_t total_sz = static_cast<size_t>(hw);
                const size_t actual_chunks = std::min(n_chunks, total_sz);
                const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
                const int32_t* ep = expanded;
                int32_t* partials_ptr = partials_.data();
                pool_->parallel([&, partials_ptr]() {
                    size_t idx;
                    while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < actual_chunks) {
                        const size_t i0 = idx * chunk_sz;
                        const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                        int32_t m = 0;
                        for (size_t i = i0; i < i1; ++i) if (ep[i] > m) m = ep[i];
                        partials_ptr[idx] = m;
                    }
                });
                for (size_t i = 0; i < actual_chunks; ++i) {
                    if (partials_[i] > max_label) max_label = partials_[i];
                }
            }
            stage("max_scan");
            std::vector<std::pair<int32_t, int32_t>> pairs;
            {
                // Unified ND unpadded find_pairs: one path for any
                // (ndim ≥ 2, conn ∈ [1, ndim]) combination. Forward-neighbour
                // count for sizing ht: enumerate dc ∈ {-1,0,1}^ndim with
                // 1 ≤ #nz ≤ conn and lex-first nonzero = +1.
                int64_t n_fwd = 0;
                {
                    const int n = ndim;
                    // Cheap closed-form: sum over k=1..conn of  C(n, k) * 2^{k-1} / 2 …
                    // simpler to just enumerate the same way the kernel does.
                    std::vector<int8_t> dc(n, -1);
                    while (true) {
                        int n_nz = 0, first_nz = -1;
                        for (int d = 0; d < n; ++d) if (dc[d] != 0) {
                            if (first_nz < 0) first_nz = d;
                            ++n_nz;
                        }
                        if (n_nz > 0 && n_nz <= conn && first_nz >= 0 && dc[first_nz] == 1) ++n_fwd;
                        int d = n - 1;
                        while (d >= 0) {
                            ++dc[d];
                            if (dc[d] <= 1) break;
                            dc[d] = -1;
                            --d;
                        }
                        if (d < 0) break;
                    }
                }
                const int64_t ht_raw = 2 * n_fwd * static_cast<int64_t>(max_label);
                const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, 16));
                pairs = ncolor_cpp::find_pairs_nd_unpadded<int32_t>(
                    expanded, shape, conn,
                    static_cast<uint64_t>(ht_size), n_threads_, *pool_, wrap);
            }
            stage("find_pairs");

            // 3. Build CSR (labels are 1..max_label after expand → node = label-1).
            const int32_t N = max_label;
            const int32_t M = static_cast<int32_t>(pairs.size());
            src_idx_.resize(M);
            dst_idx_.resize(M);
            for (int32_t i = 0; i < M; ++i) {
                src_idx_[i] = pairs[i].first - 1;
                dst_idx_[i] = pairs[i].second - 1;
            }
            ncolor_cpp::build_csr_from_pairs(src_idx_.data(), dst_idx_.data(), N, M,
                                             indptr_, indices_);
            stage("build_csr");

            // 4. Color: BFS attempt → if conflicts/unfinished, run local
            // repair pass; only bump n_colors if both fail. Mirrors the
            // numba `_solver` retry chain (sans Kempe-chain swaps, which
            // are rarely triggered for the planar adjacency graphs ncolor
            // sees in practice).
            //
            // The ``attempts_per_n`` random-offset attempts at each cur_n
            // are independent (they only differ in the LCG seed used for
            // restarts), so we race them in parallel on the pool. Wall-clock
            // is now max(attempt_time) per cur_n iteration instead of sum.
            // For small graphs the dispatch overhead dominates the BFS, so
            // we fall back to the serial path below a threshold.
            const int64_t max_iter = std::max<int64_t>(static_cast<int64_t>(indices_.size()) +
                                                      static_cast<int64_t>(indptr_.size()), 512);
            int cur_n = n_colors;
            const int attempts_per_n = 4;
            bool ok = false;
            // Threshold tuned on M2 / 20-thread ForkJoinPool: below ~500
            // edges the BFS finishes in <100 µs and dispatch eats the win.
            // color_mode override: 0 = forced serial, 1 = forced parallel.
            bool color_parallel;
            if (color_mode == 0) color_parallel = false;
            else if (color_mode == 1) color_parallel = (n_threads_ > 1);
            else color_parallel = (n_threads_ > 1) &&
                (static_cast<int64_t>(N) + M >= 500);
            if (color_parallel) {
                per_attempt_colors_.resize(attempts_per_n);
                per_attempt_ok_.assign(attempts_per_n, 0);
            }
            for (int depth = 0; depth < max_depth && !ok; ++depth) {
                if (color_parallel) {
                    // Run the attempts_per_n random-offset attempts in parallel.
                    // Each attempt has its own colors buffer (the BFS internally
                    // also allocates its own counter / queue, so attempts are
                    // fully independent — no cross-thread state).
                    std::atomic<int> next{0};
                    const int A = attempts_per_n;
                    const int local_cur_n = cur_n;
                    const int local_depth = depth;
                    const int32_t* ip = indptr_.data();
                    const int32_t* ix = indices_.data();
                    pool_->parallel([&, local_cur_n, local_depth, ip, ix]() {
                        int idx;
                        while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < A) {
                            auto& cv = per_attempt_colors_[idx];
                            const int attempt_offset = local_depth + idx;
                            // Welsh-Powell ordering on the first
                            // attempts_per_n - 1 attempts (each with its
                            // own random offset for the BFS restart
                            // logic), with the last attempt reserved as
                            // a label-ID fallback. WP can occasionally
                            // paint itself into a corner (e.g. L2 + wrap
                            // pushes some cells to a 5th colour); the
                            // fallback recovers the 4-colouring without
                            // bumping cur_n. Lowest-index successful
                            // attempt wins, so a working WP is always
                            // preferred over the label-ID fallback.
                            const bool wp = balance && (idx < A - 1);
                            const bool finished = ncolor_cpp::color_graph_csr_legacy(
                                ip, ix, N,
                                local_cur_n, rand_period, attempt_offset, max_iter, cv, wp);
                            const bool conflict = !finished ||
                                ncolor_cpp::has_conflict_csr(ip, ix, N, cv.data());
                            bool a_ok;
                            if (conflict) {
                                a_ok = ncolor_cpp::repair_coloring(
                                    ip, ix, N,
                                    local_cur_n, std::max(4, max_depth), cv);
                            } else {
                                a_ok = true;
                            }
                            // For Welsh-Powell attempts (all but the last
                            // when balance=true), require a CLEAN result
                            // — repair tends to merge colours unevenly,
                            // so a repaired WP coloring is usually less
                            // balanced than the label-ID fallback. The
                            // last attempt (label-ID safety net) accepts
                            // any successful result.
                            const bool require_clean = wp;
                            per_attempt_ok_[idx] = (a_ok && (!require_clean || !conflict)) ? 1 : 0;
                        }
                    });
                    // Take the lowest-index successful attempt
                    // (preserves deterministic preference for the first
                    // random offset when multiple attempts succeed).
                    for (int a = 0; a < attempts_per_n; ++a) {
                        if (per_attempt_ok_[a]) {
                            colors_.swap(per_attempt_colors_[a]);
                            ok = true;
                            break;
                        }
                    }
                } else {
                    for (int attempt = 0; attempt < attempts_per_n && !ok; ++attempt) {
                        // See parallel branch — Welsh-Powell on the
                        // first attempts_per_n - 1 attempts; last is
                        // label-ID fallback. WP attempts must be clean
                        // (no repair) to count, so a repaired WP doesn't
                        // pre-empt the label-ID safety net.
                        const bool wp = balance && (attempt < attempts_per_n - 1);
                        const bool finished = ncolor_cpp::color_graph_csr_legacy(
                            indptr_.data(), indices_.data(), N,
                            cur_n, rand_period, depth + attempt, max_iter, colors_, wp);
                        bool conflict = !finished || ncolor_cpp::has_conflict_csr(
                            indptr_.data(), indices_.data(), N, colors_.data());
                        bool a_ok;
                        if (conflict) {
                            a_ok = ncolor_cpp::repair_coloring(
                                indptr_.data(), indices_.data(), N,
                                cur_n, std::max(4, max_depth), colors_);
                        } else {
                            a_ok = true;
                        }
                        // Same gate as parallel: WP only counts when clean.
                        if (a_ok && (!wp || !conflict)) ok = true;
                    }
                }
                if (!ok) ++cur_n;  // bump n only when all attempts at cur_n failed
            }
            n_used = 0;
            for (uint8_t c : colors_) if (c > n_used) n_used = c;

            // Cheap O(M) tally of adjacent same-color pairs. Matches the
            // legacy ``conflicts = sum(lut[pairs[:,0]] == lut[pairs[:,1]])``
            // since src_idx_/dst_idx_ are exactly the pairs from connect()
            // (shifted to 0-based) and colors_[k] == lut_[k+1].
            last_n_conflicts_ = 0;
            for (int32_t i = 0; i < M; ++i) {
                if (colors_[src_idx_[i]] == colors_[dst_idx_[i]]) ++last_n_conflicts_;
            }

            stage("color");
            // 5. Build LUT and apply (expanded[i] is in 1..N, so lut size = N+1).
            lut_.assign(static_cast<size_t>(N) + 1, 0);
            for (int32_t i = 0; i < N; ++i) lut_[i + 1] = colors_[i];

            // Apply LUT in parallel — bg pixels get color 0, foreground
            // pixels get lut[expanded[i]]. The bg pattern was captured
            // by ``cast_with_bg`` at the start (bg_mask_[i]==1 wherever
            // the original input was 0); using a uint8 mask here keeps
            // the inner loop typeless wrt the original dtype.
            {
                const int nt = std::max(1, n_threads_);
                const uint8_t* bg_p = bg_mask_.data();
                const int32_t* ep = expanded;
                const uint8_t* lp = lut_.data();
                uint8_t* op = out_ptr;
                if (nt == 1 || total < 8192) {
                    for (int64_t i = 0; i < total; ++i) {
                        op[i] = bg_p[i] ? 0 : lp[ep[i]];
                    }
                } else {
                    ncolor_cpp::dispatch_parallel(*pool_, static_cast<size_t>(total),
                        static_cast<size_t>(nt) * ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD,
                        [bg_p, ep, lp, op](size_t begin, size_t end) {
                            for (size_t i = begin; i < end; ++i) {
                                op[i] = bg_p[i] ? 0 : lp[ep[i]];
                            }
                        });
                }
            }
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
    int n_threads_;
    std::unique_ptr<ncolor_cpp::ForkJoinPool> pool_;
    ncolor_cpp::ExpandBuffers expand_bufs_;
    std::vector<uint8_t> bg_mask_;     // captured from cast, used by apply_lut
    std::vector<int32_t> partials_;     // max-reduce partials, reused across calls
    std::vector<int32_t> src_idx_, dst_idx_;
    std::vector<int32_t> indptr_, indices_;
    std::vector<uint8_t> colors_;
    std::vector<uint8_t> lut_;
    int last_n_conflicts_ = 0;
    // Per-attempt scratch for parallel coloring (one colors vector per
    // racing attempt). Reused across calls.
    std::vector<std::vector<uint8_t>> per_attempt_colors_;
    std::vector<int> per_attempt_ok_;
};

PYBIND11_MODULE(_impl, m) {
    m.doc() = "ncolor C++ engine: connect / expand / color pipeline + "
              "ForkJoinPool. Public Python API in ncolor.color, ncolor.expand "
              "wraps the engines exposed here.";
    py::class_<ConnectEngine>(m, "ConnectEngine",
        "Persistent threadpool wrapper for the search-hashset kernel.\n"
        "Construct once with the desired worker count; the underlying\n"
        "std::thread workers live for the engine's lifetime, so call cost\n"
        "is just task enqueue + dispatch (sub-millisecond on a warm pool).")
        .def(py::init<double>(), py::arg("n_threads") = -1.0)
        .def_property_readonly("n_threads", &ConnectEngine::n_threads)
        .def("find_pairs", &ConnectEngine::find_pairs,
             py::arg("line"), py::arg("nbs"), py::arg("ht_size"),
             py::arg("parallel_threshold") = 100000,
             "Find unique label-adjacency pairs on a 1D padded label line.");

    py::class_<ExpandEngine>(m, "ExpandEngine",
        "Persistent threadpool wrapper for expand_labels + apply_lut.\n"
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
        .def("expand_labels_timed", &ExpandEngine::expand_labels_timed, py::arg("labels"),
             "expand_labels(p=2) + per-stage (name, ms) breakdown.")
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
             "block. Returns (formatted_array, n_labels).")
        .def("apply_lut", &ExpandEngine::apply_lut,
             py::arg("flat_lab"), py::arg("lut"),
             "Parallel scatter: out[i] = lut[flat_lab[i]]. Lut must be uint8 or int32.");

    py::class_<Solver>(m, "Solver",
        "End-to-end ncolor.label() equivalent for 2D conn=1 inputs.\n"
        "Wraps a single ThreadPool and re-uses all intermediate buffers,\n"
        "so the per-call cost is just task enqueue + the actual work.\n"
        "Returns (colored_image_uint8, n_colors_used).\n"
        "\n"
        "Supports 2D and 3D inputs. ``Solver.label(mask, conn=2)`` mirrors\n"
        "ncolor.label's 2D default (8-connectivity); for 3D pass conn∈{1,2,3}.\n"
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
             "Run [format_labels →] [expand →] connect → CSR → color → apply_lut.\n"
             "Supports 2D and 3D inputs (any ndim ≥ 2 actually).\n"
             "conn: 2D ∈ {1, 2}, 3D ∈ {1, 2, 3}. Matches\n"
             "scipy.ndimage.generate_binary_structure semantics. The 2D conn=2\n"
             "case takes a fast path that skips padding (~1 ms saved at 2048²).\n"
             "p selects the expand metric: p=2 (Felzenszwalb parabolic\n"
             "envelope, default) or p=1 (Saito-Toriwaki sweep, ~5× faster\n"
             "with slightly different boundary placement at ties).\n"
             "format_input=True (default) compacts non-sequential nonzero\n"
             "labels to 1..N in-place inside the released-GIL section.\n"
             "Precondition: bg=0 in the input. Pass format_input=False if\n"
             "labels are already 1..N (saves ~one full-image pass).\n"
             "Background masking (output=0 wherever input=0) is always\n"
             "applied in the apply_lut step.\n"
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
}
