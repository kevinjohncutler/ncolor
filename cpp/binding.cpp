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
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "cc_label.hpp"
#include "chamfer.hpp"
#include "color.hpp"
#include "delete_spurs.hpp"
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

            // 2. Find adjacency pairs. Parallel max-reduce first
            // (was a single-threaded 1.2 ms loop at 2048²).
            const int32_t max_label = parallel_max_label_(expanded, total);
            stage("max_scan");
            std::vector<std::pair<int32_t, int32_t>> pairs =
                find_pairs_(expanded, shape, conn, wrap, max_label);
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

            // 4. Coloring: BFS + repair fallback × attempts_per_n random
            // offsets, retrying with cur_n+1 if all attempts fail. See
            // ``solve_coloring_`` for full algorithm + parallel-attempt
            // dispatch.
            n_used = solve_coloring_(N, M, n_colors, max_depth, rand_period,
                                     balance, color_mode);
            stage("color");

            // 5. Build LUT (expanded[i] is in 1..N, so lut size = N+1) and
            // apply it to the expanded-label buffer.
            lut_.assign(static_cast<size_t>(N) + 1, 0);
            for (int32_t i = 0; i < N; ++i) lut_[i + 1] = colors_[i];
            apply_color_lut_(expanded, out_ptr, total);
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
            int conn, bool wrap, int32_t max_label) {
        if (max_label == 0) return {};
        const int ndim = static_cast<int>(shape.size());
        const int64_t n_fwd = ncolor_cpp::detail::count_forward_neighbours(ndim, conn);
        const int64_t ht_raw = 2 * n_fwd * static_cast<int64_t>(max_label);
        const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, MIN_HT_SIZE));
        return ncolor_cpp::find_pairs_nd_unpadded<int32_t>(
            labels, shape, conn,
            static_cast<uint64_t>(ht_size), n_threads_, *pool_, wrap);
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
                        int color_mode) {
        constexpr int attempts_per_n = 4;
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
        for (int depth = 0; depth < max_depth && !ok; ++depth) {
            if (color_parallel) {
                std::atomic<int> next{0};
                const int local_cur_n = cur_n;
                const int local_depth = depth;
                const int32_t* ip = indptr_.data();
                const int32_t* ix = indices_.data();
                pool_->parallel([&, local_cur_n, local_depth, ip, ix]() {
                    int idx;
                    while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < attempts_per_n) {
                        auto& cv = per_attempt_colors_[idx];
                        const int attempt_offset = local_depth + idx;
                        // Slot 0 = user-preferred algorithm; remaining
                        // slots = alternate algorithm with different
                        // random offsets (algorithm-switching fallback).
                        const bool wp = balance ? (idx == 0)
                                                : (idx == attempts_per_n - 1);
                        const bool finished = ncolor_cpp::color_graph_csr_legacy(
                            ip, ix, N, local_cur_n, rand_period,
                            attempt_offset, max_iter, cv, wp);
                        const bool conflict = !finished ||
                            ncolor_cpp::has_conflict_csr(ip, ix, N, cv.data());
                        bool a_ok = !conflict || ncolor_cpp::repair_coloring(
                            ip, ix, N, local_cur_n, std::max(4, max_depth), cv);
                        // WP must be clean — repair pre-empting a WP attempt
                        // would shadow the cleaner label-ID safety net.
                        per_attempt_ok_[idx] = (a_ok && (!wp || !conflict)) ? 1 : 0;
                    }
                });
                // Lowest-index successful attempt wins (deterministic
                // preference for the first random offset).
                for (int a = 0; a < attempts_per_n; ++a) {
                    if (per_attempt_ok_[a]) {
                        colors_.swap(per_attempt_colors_[a]);
                        ok = true;
                        break;
                    }
                }
            } else {
                for (int attempt = 0; attempt < attempts_per_n && !ok; ++attempt) {
                    const bool wp = balance ? (attempt == 0)
                                            : (attempt == attempts_per_n - 1);
                    const bool finished = ncolor_cpp::color_graph_csr_legacy(
                        indptr_.data(), indices_.data(), N,
                        cur_n, rand_period, depth + attempt, max_iter,
                        colors_, wp);
                    const bool conflict = !finished || ncolor_cpp::has_conflict_csr(
                        indptr_.data(), indices_.data(), N, colors_.data());
                    bool a_ok = !conflict || ncolor_cpp::repair_coloring(
                        indptr_.data(), indices_.data(), N,
                        cur_n, std::max(4, max_depth), colors_);
                    if (a_ok && (!wp || !conflict)) ok = true;
                }
            }
            if (!ok) ++cur_n;
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
    std::vector<uint8_t> colors_;
    std::vector<uint8_t> lut_;
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

    // Standalone N-D connected-components labelling. Single-threaded
    // two-pass union-find. Drop-in for skimage.measure.label on integer
    // (or boolean) input.
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

    // Timed variant: returns (labels, n, {fg_mask_ms, pass1_ms, pass2_ms})
    // for diagnostic profiling.
    m.def("cc_label_timed",
          [](py::array mask, int conn) {
              if (!(mask.flags() & py::array::c_style)) {
                  mask = py::array::ensure(mask, py::array::c_style);
              }
              const auto buf = mask.request();
              const int ndim = static_cast<int>(buf.ndim);
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
              ncolor_cpp::CCStageTimes times;
              {
                  py::gil_scoped_release release;
                  dispatch_int_dtype(buf.format, buf.itemsize, "cc_label_timed",
                      [&](auto* tag) {
                          using T = std::remove_pointer_t<decltype(tag)>;
                          n_labels = ncolor_cpp::cc_label_nd<T>(
                              static_cast<const T*>(src_ptr), out_ptr,
                              shape, conn, &times);
                      });
              }
              py::dict t;
              t["fg_mask_ms"] = times.fg_mask_ms;
              t["pass1_ms"]   = times.pass1_ms;
              t["pass2_ms"]   = times.pass2_ms;
              return py::make_tuple(out, n_labels, t);
          },
          py::arg("mask"), py::arg("conn") = 2,
          "Diagnostic: same as cc_label but returns (labels, n, stage_times).");

    // Region properties for a dense 1..N labelled image. Returns a dict
    // of int64/double arrays — minimal-overhead, vectorised consumption
    // pattern (a list of namedtuples like skimage.measure.regionprops
    // returns is much heavier).
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
          [](py::array mask, int hole_threshold) {
              return ncolor_cpp::delete_spurs_nd(mask, hole_threshold);
          },
          py::arg("mask"), py::arg("hole_threshold") = 5,
          "N-D skeleton cleanup: fill bg holes ≤ hole_threshold pixels\n"
          "(face-connected), then iteratively strip endpoints (fg pixels\n"
          "with exactly one fg neighbour) until convergence. Endpoint\n"
          "connectivity is 8-conn for 2D, face-only for ndim≥3.");
}
