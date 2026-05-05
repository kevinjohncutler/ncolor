/*
 * Pybind11 binding for the ncolor C++ prototype.
 *
 * Exposes a single class ``ConnectEngine`` that wraps a persistent ThreadPool.
 * Construct once, call ``find_pairs`` many times — that's the whole win
 * over the numba version, which pays per-call thread-pool startup on every
 * @njit(parallel=True) region launch.
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
            int p = 2) {
        const auto buf = labels.request();
        std::vector<int64_t> shape(buf.ndim);
        for (int i = 0; i < buf.ndim; ++i) shape[i] = buf.shape[i];

        const int32_t* input = static_cast<const int32_t*>(buf.ptr);
        py::array_t<int32_t> out(buf.shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);

        {
            py::gil_scoped_release release;
            if (p == 2) {
                ncolor_cpp::expand_labels_lp<2>(input, out_ptr, bufs_, shape, *pool_, n_threads_);
            } else if (p == 1) {
                ncolor_cpp::expand_labels_lp<1>(input, out_ptr, bufs_, shape, *pool_, n_threads_);
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
    // ``first_seen=False`` (default) uses ascending-source numbering
    // (fast: parallel mark + serial scan over [1..max] + parallel
    // apply). ``first_seen=True`` uses input-order numbering (matches
    // fastremap.renumber bit-for-bit; build pass is serial through
    // the input array — slower).
    //
    // Returns (formatted_array, n_labels).
    std::pair<py::array_t<int32_t>, int> format_labels(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> labels,
            bool first_seen = false) {
        const auto buf = labels.request();
        const int64_t total = buf.size;
        py::array_t<int32_t> out(buf.shape);
        int32_t* out_ptr = static_cast<int32_t*>(out.request().ptr);
        std::memcpy(out_ptr, buf.ptr, total * sizeof(int32_t));
        int n_labels;
        {
            py::gil_scoped_release release;
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

// Compute the connectivity neighbour flat-index offsets for a padded image.
// Matches ncolor.color.neighbors(buf.shape, conn, unique=True) where buf is
// the input padded by 1 zero in each dim. ``conn`` is the max number of
// nonzero components in the offset (scipy.ndimage.generate_binary_structure
// semantics): 2D conn=1 → 4-conn (face only), 2D conn=2 → 8-conn (face +
// diagonals); 3D conn=1 → 6-conn, conn=2 → 18-conn, conn=3 → 26-conn.
//
// "unique forward half" means we keep only directions whose first nonzero
// component is positive — each undirected pair is then emitted exactly
// once across both directions during the scan.
static inline std::vector<int64_t> neighbors_nd_padded(
        const std::vector<int64_t>& shape, int conn) {
    const int ndim = static_cast<int>(shape.size());
    std::vector<int64_t> ps(ndim);  // padded strides (row-major)
    {
        int64_t s = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            ps[d] = s;
            s *= (shape[d] + 2);
        }
    }
    int total = 1;
    for (int d = 0; d < ndim; ++d) total *= 3;
    std::vector<int64_t> nbs;
    nbs.reserve(static_cast<size_t>((total - 1) / 2));
    for (int idx = 0; idx < total; ++idx) {
        int v = idx;
        int n_nonzero = 0, first_nz = -1;
        std::vector<int> off(ndim);
        for (int d = 0; d < ndim; ++d) {
            off[d] = (v % 3) - 1;
            v /= 3;
            if (off[d] != 0) {
                ++n_nonzero;
                if (first_nz < 0) first_nz = d;
            }
        }
        if (n_nonzero == 0 || n_nonzero > conn) continue;
        if (off[first_nz] < 0) continue;  // forward half only
        int64_t flat = 0;
        for (int d = 0; d < ndim; ++d) flat += off[d] * ps[d];
        nbs.push_back(flat);
    }
    return nbs;
}

// Pad an N-D row-major buffer by 1 zero in each dimension. ``dst`` must
// have shape (s[0]+2, s[1]+2, ..., s[ndim-1]+2) and total ``total_padded``
// elements; this fills it with ``pad_value`` then copies ``src`` into the
// interior. Iterates over "innermost rows" so the inner copy is one
// memcpy per (shape[ndim-1])-length row.
template <typename T>
inline void pad_nd_into(const T* src, T* dst,
                        const std::vector<int64_t>& shape, T pad_value) {
    const int ndim = static_cast<int>(shape.size());
    int64_t total_padded = 1;
    for (int d = 0; d < ndim; ++d) total_padded *= (shape[d] + 2);
    std::fill_n(dst, total_padded, pad_value);

    std::vector<int64_t> ps(ndim), ss(ndim);  // padded / source strides
    {
        int64_t p = 1, s = 1;
        for (int d = ndim - 1; d >= 0; --d) {
            ps[d] = p; p *= (shape[d] + 2);
            ss[d] = s; s *= shape[d];
        }
    }
    int64_t base_off = 0;
    for (int d = 0; d < ndim; ++d) base_off += ps[d];

    const int64_t row_len = shape[ndim - 1];
    int64_t n_rows = 1;
    for (int d = 0; d < ndim - 1; ++d) n_rows *= shape[d];

    std::vector<int64_t> coord(std::max(0, ndim - 1), 0);
    for (int64_t r = 0; r < n_rows; ++r) {
        int64_t src_start = 0, dst_start = base_off;
        for (int d = 0; d < ndim - 1; ++d) {
            src_start += coord[d] * ss[d];
            dst_start += coord[d] * ps[d];
        }
        std::memcpy(dst + dst_start, src + src_start, row_len * sizeof(T));
        for (int d = ndim - 2; d >= 0; --d) {
            ++coord[d];
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
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

    std::pair<py::array_t<uint8_t>, int> label(
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> mask,
            int n_colors = 4, int max_depth = 30, int rand_period = 10,
            int conn = 2, int p = 2, bool capture_stages = false,
            bool format_input = true) {
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
        const bool fast_2d = (ndim == 2 && conn == 2);
        const int64_t H = shape[0];
        const int64_t W = (ndim >= 2) ? shape[1] : 1;

        const int32_t* mask_ptr = static_cast<const int32_t*>(buf.ptr);

        std::vector<py::ssize_t> out_shape(ndim);
        for (int d = 0; d < ndim; ++d) out_shape[d] = static_cast<py::ssize_t>(shape[d]);
        py::array_t<uint8_t> out(out_shape);
        uint8_t* out_ptr = static_cast<uint8_t*>(out.request().ptr);

        int n_used = 0;
        last_stages_.clear();
        std::chrono::steady_clock::time_point t_start, t_now;
        if (capture_stages) t_start = std::chrono::steady_clock::now();
        auto stage = [&](const char* name) {
            if (!capture_stages) return;
            t_now = std::chrono::steady_clock::now();
            last_stages_.emplace_back(name,
                std::chrono::duration<double, std::milli>(t_now - t_start).count());
            t_start = t_now;
        };
        {
            py::gil_scoped_release release;

            // 0. Optional format_labels: compact nonzero labels to 1..N
            // in place inside expand_bufs_.lbl(). Subsequent stages see
            // a normalized buffer; the original mask_ptr is preserved
            // for the bg-mask step at the end.
            expand_bufs_.resize(total);
            int32_t* expanded = expand_bufs_.lbl();
            const int32_t* expand_input;
            if (format_input) {
                std::memcpy(expanded, mask_ptr, total * sizeof(int32_t));
                ncolor_cpp::format_labels_inplace(expanded, total, *pool_, n_threads_);
                expand_input = expanded;  // already in expand_bufs_.lbl()
                stage("format");
            } else {
                expand_input = mask_ptr;
            }

            // 1. Expand labels (Voronoi) — every pixel ends up labeled.
            // Single ND driver dispatches on p at compile time:
            // p=1 → Saito-Toriwaki sweep; p=2 → Felzenszwalb envelope.
            // Result lands in expand_bufs_.lbl(); the L2 ``output ==
            // bufs.lbl()`` self-copy is elided inside LpExpand<2>::expand,
            // and L1's in-place sweep writes directly there. When
            // format_input=true, expand_input == expanded already, and
            // the in-place memcpys inside both LpExpand variants
            // detect input==output and skip.
            if (p == 2) {
                ncolor_cpp::expand_labels_lp<2>(expand_input, expanded, expand_bufs_, shape, *pool_, n_threads_);
            } else {
                ncolor_cpp::expand_labels_lp<1>(expand_input, expanded, expand_bufs_, shape, *pool_, n_threads_);
            }
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
                std::vector<int32_t> partials(n_chunks, 0);
                std::atomic<size_t> next{0};
                const size_t total_sz = static_cast<size_t>(hw);
                const size_t actual_chunks = std::min(n_chunks, total_sz);
                const size_t chunk_sz = (total_sz + actual_chunks - 1) / actual_chunks;
                const int32_t* ep = expanded;
                pool_->parallel([&]() {
                    size_t idx;
                    while ((idx = next.fetch_add(1, std::memory_order_relaxed)) < actual_chunks) {
                        const size_t i0 = idx * chunk_sz;
                        const size_t i1 = std::min(i0 + chunk_sz, total_sz);
                        int32_t m = 0;
                        for (size_t i = i0; i < i1; ++i) if (ep[i] > m) m = ep[i];
                        partials[idx] = m;
                    }
                });
                for (size_t i = 0; i < actual_chunks; ++i) {
                    if (partials[i] > max_label) max_label = partials[i];
                }
            }
            stage("max_scan");
            std::vector<std::pair<int32_t, int32_t>> pairs;
            if (fast_2d) {
                // 2D conn=2 fast path: skip padding (saves ~1 ms at 2048²).
                // For 4 forward neighbours per pixel (right, down-left, down,
                // down-right), ht_raw = 2 contribs/pair × 4 = 8N; round up.
                const int64_t ht_raw = 4 * static_cast<int64_t>(max_label);
                const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, 16));
                pairs = ncolor_cpp::find_pairs_2d_unpadded<int32_t>(
                    expanded, H, W, static_cast<uint64_t>(ht_size),
                    n_threads_, *pool_);
            } else {
                // ND or non-default conn: pad + generic search_hashset_parallel.
                std::vector<int64_t> nbs = neighbors_nd_padded(shape, conn);
                int64_t total_padded = 1;
                for (int64_t s : shape) total_padded *= (s + 2);
                std::unique_ptr<int32_t[]> padded(new int32_t[total_padded]);
                pad_nd_into<int32_t>(expanded, padded.get(), shape, /*pad_value=*/0);
                const int64_t ht_raw = 2 * static_cast<int64_t>(nbs.size())
                                        * static_cast<int64_t>(max_label);
                const int64_t ht_size = ipow2_ge(std::max<int64_t>(ht_raw, 16));
                pairs = ncolor_cpp::search_hashset_parallel<int32_t>(
                    padded.get(), total_padded, nbs.data(),
                    static_cast<int>(nbs.size()),
                    static_cast<uint64_t>(ht_size), n_threads_, *pool_);
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
            const int64_t max_iter = std::max<int64_t>(static_cast<int64_t>(indices_.size()) +
                                                      static_cast<int64_t>(indptr_.size()), 512);
            int cur_n = n_colors;
            const int attempts_per_n = 4;
            bool ok = false;
            for (int depth = 0; depth < max_depth && !ok; ++depth) {
                for (int attempt = 0; attempt < attempts_per_n && !ok; ++attempt) {
                    const bool finished = ncolor_cpp::color_graph_csr_legacy(
                        indptr_.data(), indices_.data(), N,
                        cur_n, rand_period, depth + attempt, max_iter, colors_);
                    bool conflict = !finished || ncolor_cpp::has_conflict_csr(
                        indptr_.data(), indices_.data(), N, colors_.data());
                    if (conflict) {
                        // Local greedy repair — fixes most leftovers cheaply.
                        ok = ncolor_cpp::repair_coloring(
                            indptr_.data(), indices_.data(), N,
                            cur_n, std::max(4, max_depth), colors_);
                    } else {
                        ok = true;
                    }
                }
                if (!ok) ++cur_n;  // bump n only when all attempts at cur_n failed
            }
            n_used = 0;
            for (uint8_t c : colors_) if (c > n_used) n_used = c;

            stage("color");
            // 5. Build LUT and apply (expanded[i] is in 1..N, so lut size = N+1).
            lut_.assign(static_cast<size_t>(N) + 1, 0);
            for (int32_t i = 0; i < N; ++i) lut_[i + 1] = colors_[i];

            // Apply LUT in parallel — bg pixels (0 in the original input) get
            // color 0, foreground pixels get lut[expanded[i]]. The single-
            // threaded version was costing ~5–10 ms at 2048² (4M loops with
            // a branch each); pool-dispatched chunked write drops it to <1 ms.
            {
                const int nt = std::max(1, n_threads_);
                if (nt == 1 || total < 8192) {
                    for (int64_t i = 0; i < total; ++i) {
                        out_ptr[i] = (mask_ptr[i] != 0) ? lut_[expanded[i]] : 0;
                    }
                } else {
                    const int32_t* mp = mask_ptr;
                    const int32_t* ep = expanded;
                    const uint8_t* lp = lut_.data();
                    uint8_t* op = out_ptr;
                    ncolor_cpp::dispatch_parallel(*pool_, static_cast<size_t>(total),
                        static_cast<size_t>(nt) * ncolor_cpp::DISPATCH_CHUNKS_PER_THREAD,
                        [mp, ep, lp, op](size_t begin, size_t end) {
                            for (size_t i = begin; i < end; ++i) {
                                op[i] = (mp[i] != 0) ? lp[ep[i]] : 0;
                            }
                        });
                }
            }
            stage("apply_lut");
        }
        return {std::move(out), n_used};
    }

private:
    int n_threads_;
    std::unique_ptr<ncolor_cpp::ForkJoinPool> pool_;
    ncolor_cpp::ExpandBuffers expand_bufs_;
    std::vector<int32_t> padded_;
    std::vector<int32_t> src_idx_, dst_idx_;
    std::vector<int32_t> indptr_, indices_;
    std::vector<uint8_t> colors_;
    std::vector<uint8_t> lut_;
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
             py::arg("labels"), py::arg("p") = 2,
             "Voronoi label expansion under L_p metric. p=1 (Manhattan,\n"
             "Saito-Toriwaki sweep) or p=2 (Euclidean², Felzenszwalb\n"
             "envelope). Same ND driver, dispatched at compile time on p.\n"
             "Default p=2 matches numba's expand_labels(metric='l2').")
        .def("expand_labels_timed", &ExpandEngine::expand_labels_timed, py::arg("labels"),
             "expand_labels(p=2) + per-stage (name, ms) breakdown.")
        .def("format_labels", &ExpandEngine::format_labels,
             py::arg("labels"), py::arg("first_seen") = false,
             "Compact nonzero labels to 1..N. If min(labels) != 0 the\n"
             "min is treated as background and everything is shifted\n"
             "before compaction (matches legacy fastremap-based\n"
             "format_labels). first_seen=True matches fastremap.renumber\n"
             "bit-for-bit (input-order numbering, serial build pass);\n"
             "default first_seen=False uses ascending-source numbering\n"
             "(parallel build, faster). Returns (formatted_array, n_labels).")
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
             py::arg("format_input") = true,
             "Run [format_labels →] expand → connect → CSR → color → apply_lut.\n"
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
             "applied in the apply_lut step.")
        .def("get_last_stages", &Solver::get_last_stages,
             "Per-stage timing breakdown from the most recent label() call\n"
             "made with capture_stages=True.");
}
