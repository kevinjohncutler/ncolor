// Native contracts: allocation, exact wide envelopes, and weighted reducers.
#include <cassert>
#include <cstdlib>
#include <map>
#include <memory>
#include <new>
#include <random>
#include "expand_lp.hpp"
#include "expand_clean.hpp"
#include "connect.hpp"
#include "delete_spurs.hpp"
#include "delete_spurs_labels.hpp"
#include "fast_despur.hpp"
#include "clique_lb.hpp"
#include "color.hpp"
#include "soft_color.hpp"
#include "format_labels.hpp"
#include "picker.hpp"
#include "cc_label.hpp"

static bool track_allocations = false;
static size_t allocated_bytes = 0;
static size_t tracked_size = 0, full_volume_allocations = 0;
void* operator new(size_t n) {
    if (track_allocations) {
        allocated_bytes += n;
        if (n == tracked_size) ++full_volume_allocations;
    }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }

using namespace ncolor_cpp;

static void check_buffers() {
    constexpr int64_t n = 1024 * 1024;
    allocated_bytes = 0;
    ExpandBuffers b;
    track_allocations = true;
    b.resize(n);
    track_allocations = false;
    assert(allocated_bytes == 2 * n * sizeof(int32_t));
    b.lbl_T()[n - 1] = 7;
    b.dist_T()[n - 1] = 9;
    b.use_wide_distance();
    b.dist64()[0] = int64_t{1} << 40;
    assert(b.distance_at(0) == static_cast<double>(int64_t{1} << 40));
    b.resize(3);
    assert(!b.wide_distance());
    b.dist()[0] = 42;
    assert(b.distance_at(0) == 42);
    b.lbl_T()[2] = 1;
    b.resize(n + 13);
    b.lbl_T()[n + 12] = 3;
    b.dist_T()[n + 12] = 4;
    b.release();
    assert(b.size() == 0);
    allocated_bytes = 0;
    track_allocations = true;
    b.resize(n);
    track_allocations = false;
    assert(allocated_bytes == 2 * n * sizeof(int32_t));
    assert(!l2_needs_wide_distance({46341, 2}));
    assert(l2_needs_wide_distance({46342, 2}));
    assert(l2_needs_wide_distance({40000, 40000}));
    bool overflow = false;
    try { l2_needs_wide_distance({INT64_MAX, 2}); }
    catch (const std::overflow_error&) { overflow = true; }
    assert(overflow);
}

static void check_l1_allocations() {
    ForkJoinPool pool(1);
    const std::vector<int64_t> shape{64, 64, 64};
    constexpr size_t n = 64 * 64 * 64;
    std::vector<int32_t> image(n, 1), output(n);
    for (bool clean : {false, true}) {
        ExpandBuffers b;
        tracked_size = n * sizeof(int32_t);
        full_volume_allocations = 0;
        track_allocations = true;
        if (clean) expand_labels_clean_inplace(image.data(), b, shape, pool, 1, 1);
        else expand_labels_lp<1>(image.data(), output.data(), b, shape, pool, 1);
        track_allocations = false;
        // Standard expansion owns labels in the caller output and needs
        // only distance scratch. Clean expansion also needs working labels.
        assert(full_volume_allocations == (clean ? 2 : 1));
        const int32_t* actual = clean ? b.lbl() : output.data();
        assert(std::equal(image.begin(), image.end(), actual));
    }
}

static void check_wide_clean(ForkJoinPool& pool) {
    std::mt19937 rng(49);
    for (bool wrap : {false, true}) {
        for (const auto& shape : std::vector<std::vector<int64_t>>{
                {7, 9}, {4, 5, 6}, {3, 4, 3, 5}}) {
            int64_t total = 1;
            for (auto n : shape) total *= n;
            for (int trial = 0; trial < 12; ++trial) {
                std::vector<int32_t> image(total);
                for (auto& v : image) v = rng() % 5 == 0 ? 1 + rng() % 4 : 0;
                ExpandBuffers narrow;
                expand_labels_clean_inplace(image.data(), narrow, shape, pool, 2, 2, wrap);
                auto wide_labels = image;
                std::vector<int64_t> distances(total);
                for (int ax = static_cast<int>(shape.size()) - 1; ax >= 0; --ax) {
                    l2_sweep_axis_wide(wide_labels.data(), distances.data(), shape,
                                       ax, pool, 2, wrap);
                    if (shape.size() - ax >= 2) {
                        std::vector<int> axes;
                        for (int d = ax; d < static_cast<int>(shape.size()); ++d) axes.push_back(d);
                        bridge_check_subspace_nd(wide_labels.data(), distances.data(),
                                                  shape, axes, &pool, 2, nullptr, wrap);
                    }
                }
                for (int64_t i = 0; i < total; ++i) {
                    assert(wide_labels[i] == narrow.lbl()[i]);
                    if (wide_labels[i]) assert(distances[i] == narrow.dist()[i]);
                }
            }
        }
    }
}

template <ReduceMode Mode>
static void check_weights(ForkJoinPool& pool) {
    const std::vector<int64_t> shape{4, 5};
    std::vector<int32_t> image(20);
    std::vector<int64_t> dist(20);
    for (int i = 0; i < 20; ++i) {
        image[i] = i % 5;
        dist[i] = int64_t{i + 1} * 1000000000;
    }
    for (bool wrap : {false, true}) for (int radius : {1, 2}) {
        std::map<std::pair<int32_t, int32_t>, std::vector<double>> expected;
        for (int y = 0; y < 4; ++y) for (int x = 0; x < 5; ++x) {
            for (int dy = 0; dy <= radius; ++dy) for (int dx = -radius; dx <= radius; ++dx) {
                if (dy == 0 && dx <= 0) continue;
                int yy = y + dy, xx = x + dx;
                if (wrap) { yy %= 4; xx = (xx + 5) % 5; }
                else if (yy >= 4 || xx < 0 || xx >= 5) continue;
                int i = y * 5 + x, j = yy * 5 + xx;
                if (!image[i] || !image[j] || image[i] == image[j]) continue;
                auto pair = std::minmax(image[i], image[j]);
                expected[pair].push_back(static_cast<double>(dist[i]) + dist[j]);
            }
        }
        std::vector<double> primary;
        std::vector<int32_t> counts;
        auto pairs = find_pairs_weighted_nd_unpadded<int32_t, Mode>(
            image.data(), dist.data(), shape, 2, 128, 2, pool, wrap,
            primary, counts, radius);
        assert(pairs.size() == expected.size());
        assert(counts.size() == pairs.size());
        for (size_t i = 0; i < pairs.size(); ++i) {
            auto values = expected.at(pairs[i]);
            assert(counts[i] == static_cast<int32_t>(values.size()));
            if constexpr (Mode != ReduceMode::Count) {
                double value = 0;
                if constexpr (Mode == ReduceMode::Min) value = *std::min_element(values.begin(), values.end());
                else if constexpr (Mode == ReduceMode::Max) value = *std::max_element(values.begin(), values.end());
                else for (double v : values) {
                    if constexpr (Mode == ReduceMode::Mean) value += v;
                    else value += 1.0 / (1.0 + v);
                }
                assert(std::abs(primary[i] - value) <= std::abs(value) * 1e-12);
            }
        }
    }
}

static void check_binary_boundaries() {
    const std::vector<int64_t> shape{5, 5};
    std::vector<uint8_t> image(25, 1);
    image[0] = image[1] = image[12] = 0;
    std::unique_ptr<bool[]> output(new bool[25]);
    for (int conn : {1, 2}) for (int rounds : {0, 1, 5}) {
        delete_spurs_nd(image.data(), output.get(), shape, 10000, conn, 2, rounds);
        assert(!output[0] && !output[1] && output[12]);
    }
    for (int rank : {16, 32, 64}) {
        const std::vector<int64_t> singleton_shape(rank, 1);
        delete_spurs_nd(image.data(), output.get(), singleton_shape, 10000, rank, rank, 5);
        assert(!output[0]);
    }
}

static void check_clique_bounds() {
    // Exhaust every simple graph on five vertices against subset enumeration.
    // Includes a triangle plus isolates, which previously overread obsolete
    // clique-member tracking because the root vertex was never recorded.
    constexpr int n = 5;
    for (unsigned graph = 0; graph < (1u << 10); ++graph) {
        bool adjacent[n][n] = {};
        unsigned bit = 0;
        for (int u = 0; u < n; ++u) for (int v = u + 1; v < n; ++v)
            adjacent[u][v] = adjacent[v][u] = (graph >> bit++) & 1;
        std::vector<int32_t> indptr{0}, indices;
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) if (adjacent[u][v]) indices.push_back(v);
            indptr.push_back(static_cast<int32_t>(indices.size()));
        }
        int exact = 0;
        for (unsigned subset = 1; subset < (1u << n); ++subset) {
            bool clique = true;
            int count = 0;
            for (int u = 0; u < n; ++u) if ((subset >> u) & 1) {
                ++count;
                for (int v = u + 1; v < n; ++v)
                    if (((subset >> v) & 1) && !adjacent[u][v]) clique = false;
            }
            if (clique) exact = std::max(exact, count);
        }
        assert(clique_lower_bound(n, indptr.data(), indices.data()) == exact);
        const int bounded = clique_lower_bound(n, indptr.data(), indices.data(), 3);
        assert(bounded <= exact && bounded >= std::min(3, exact));
        // An already expired budget must stop even a shallow search.
        assert(clique_lower_bound(n, indptr.data(), indices.data(), 0, 1) == 1);
    }
    // Cross a 64-bit adjacency word boundary with known clique numbers.
    for (int family = 0; family < 3; ++family) {
        constexpr int count = 65;
        std::vector<int32_t> indptr{0}, indices;
        for (int u = 0; u < count; ++u) {
            for (int v = 0; v < count; ++v) {
                const bool edge = family == 0 ? u != v :
                    family == 1 ? (u < 32) != (v < 32) :
                    (u + 1) % count == v || (v + 1) % count == u;
                if (edge) indices.push_back(v);
            }
            indptr.push_back(static_cast<int32_t>(indices.size()));
        }
        assert(clique_lower_bound(count, indptr.data(), indices.data()) ==
               (family == 0 ? count : 2));
    }
    BKState state;
    state.deadline_ns = 1;
    state.visited_nodes = 256;
    // Budget polling depends on visited nodes, not recursion depth.
    state.bk(1, nullptr, 1);
    assert(state.deadline_hit);
}

static void check_large_palettes() {
    for (int k : {31, 32, 64, 255}) {
        // An impossible clique exercises the all-colors-present counter
        // path as well as initialization, unlike an easy large palette.
        const int n = k + 1;
        std::vector<int32_t> ip{0}, ix;
        for (int u = 0; u < n; ++u) {
            for (int v = 0; v < n; ++v) if (u != v) ix.push_back(v);
            ip.push_back(static_cast<int32_t>(ix.size()));
        }
        std::vector<uint8_t> colors;
        color_graph_csr_legacy(ip.data(), ix.data(), n, k, 10, 0, n * 3, colors);
        assert(colors.size() == static_cast<size_t>(n));
        for (auto c : colors) assert(c >= 1 && c <= k);
        assert(!repair_coloring(ip.data(), ix.data(), n, k, 2, colors));

        // The central vertex's only legal improving move uses color k.
        std::vector<int32_t> src, dst;
        for (int v = 1; v < k - 1; ++v) { src.push_back(0); dst.push_back(v); }
        build_csr_from_pairs(src.data(), dst.data(), k, k - 2, ip, ix);
        colors.assign(k, static_cast<uint8_t>(k - 1));
        for (int v = 1; v < k - 1; ++v) colors[v] = static_cast<uint8_t>(v);
        std::vector<int32_t> sp, sx;
        const int32_t a = 0, b = k - 1;
        build_csr_from_pairs(&a, &b, k, 1, sp, sx);
        double penalty = 1;
        assert(single_vertex_pass(colors.data(), k, ip.data(), ix.data(),
            sp.data(), sx.data(), nullptr, k, penalty));
        assert(colors[0] == k && penalty == 0);
        assert(!has_conflict_csr(ip.data(), ix.data(), k, colors.data()));
    }

    // Escalating a weighted palette must not read past the supplied table.
    const int32_t src[] = {0, 0, 1}, dst[] = {1, 2, 2};
    const double pair_weights[] = {1, 1, 1};
    std::vector<int32_t> ip, ix;
    std::vector<double> weights;
    build_csr_from_pairs_weighted(src, dst, pair_weights, 3, 3, ip, ix, weights);
    const double palette[] = {0, 0, 0, 0, 0, 1, 0, 1, 0};
    ForkJoinPool pool(2);
    for (int mode : {0, 1}) {
        std::vector<uint8_t> colors;
        PickerScratch scratch;
        int conflicts = -1;
        const int used = pick_coloring(3, 3, 2, 2, 10, mode, 11, false,
            weights.data(), palette, 1, ip, ix,
            std::vector<int32_t>(src, src + 3), std::vector<int32_t>(dst, dst + 3),
            colors, conflicts, scratch, &pool, 2);
        assert(used == 3 && conflicts == 0);
    }
}

static void check_sparse_formatting() {
    ForkJoinPool pool(1);
    for (bool first_seen : {false, true}) {
        std::vector<int32_t> labels{INT32_MIN, INT32_MAX, 0, INT32_MAX};
        const int count = first_seen
            ? format_labels_inplace_first_seen(labels.data(), labels.size(), pool, 1)
            : format_labels_inplace(labels.data(), labels.size(), pool, 1);
        assert(count == 2);
        const std::vector<int32_t> expected = first_seen
            ? std::vector<int32_t>{0, 1, 2, 1} : std::vector<int32_t>{0, 2, 1, 2};
        assert(labels == expected);
    }
}

static void check_fast_spurs() {
    ForkJoinPool pool(4);
    std::mt19937 rng(723);
    for (const auto& shape : {std::vector<int64_t>{9}, {7, 8}, {4, 5, 6},
                              {1, 96, 128}, std::vector<int64_t>(40, 1)}) {
        int64_t total = 1;
        for (auto n : shape) total *= n;
        std::vector<int32_t> input(total);
        for (auto& v : input) v = rng() % 5 == 0 ? 0 : 1;
        for (int threads : {1, 4}) for (int threshold : {-1, 0, 1, 3, 128}) {
            for (int rounds : {0, 1, 2, 5, -1}) {
                auto actual = input, expected = input;
                std::vector<uint8_t> counts(total);
                compute_face_count_nd(actual.data(), counts.data(), shape, &pool, threads);
                const auto removed = despur_via_face_count_nd(actual.data(), counts.data(),
                    shape, threshold, &pool, threads, rounds);
                const auto expected_removed = delete_spurs_labels_nd_inplace(expected.data(),
                    shape, threshold, rounds, &pool, threads, false);
                assert(actual == expected);
                assert(removed == expected_removed);
            }
        }
    }
}

static void check_thin_spur_allocations() {
    const std::vector<int64_t> shape(12, 2);
    std::vector<int32_t> image(4096, 1);
    allocated_bytes = 0;
    track_allocations = true;
    const auto removed = delete_spurs_labels_nd_inplace(image.data(), shape,
        1, 3, nullptr, 1, true);
    track_allocations = false;
    assert(removed == 0);
    // Scratch scales with pixels and rank, without a 3^rank offset table.
    assert(allocated_bytes < image.size() * 8);
}

static void check_soft_endpoint_bounds() {
    const int32_t edges[] = {INT32_MIN, 2, 1, INT32_MAX, 1, 2};
    std::vector<int32_t> indptr, indices;
    std::vector<float> weights;
    build_soft_csr(2, 3, edges, nullptr, indptr, indices, weights);
    assert((indptr == std::vector<int32_t>{0, 1, 2}));
    assert((indices == std::vector<int32_t>{1, 0}));
}

static void check_deadline_units() {
    using clock = std::chrono::steady_clock;
    const auto at = clock::time_point(std::chrono::seconds(7));
    assert(steady_time_ns(at) == 7000000000LL);
    const int32_t ip[] = {0, 2, 4, 6}, ix[] = {1, 2, 0, 2, 0, 1};
    std::vector<uint8_t> colors(3, 1);
    // A past deadline stops infeasible searches before their work budget.
    assert(!tabucol(ip, ix, 3, 2, 100000, colors, 1, 1));
    assert(!bb_dsatur(ip, ix, 3, 2, colors, 100000, nullptr, 1));
}

static void check_feature_transfers() {
    ForkJoinPool pool(4);
    const std::vector<int64_t> shape{13, 17, 19};
    const size_t size = 13 * 17 * 19;
    std::vector<int32_t> input(size, 0);
    for (size_t i = 7; i < size; i += 97) input[i] = static_cast<int32_t>(i + 1);
    for (bool clean : {false, true}) for (bool wrap : {false, true}) {
        ExpandBuffers reference, labels_only;
        if (clean) {
            expand_labels_clean_inplace(input.data(), reference, shape, pool, 4, 2, wrap, true);
            expand_labels_clean_inplace(input.data(), labels_only, shape, pool, 4, 2, wrap, false);
        } else {
            expand_labels_inplace(input.data(), reference, shape, pool, 4, wrap, true);
            expand_labels_inplace(input.data(), labels_only, shape, pool, 4, wrap, false);
        }
        assert(std::equal(reference.lbl(), reference.lbl() + size, labels_only.lbl()));
    }
    std::vector<int32_t> label{1, 2, 3, 4, 5, 6}, distance(6, 10);
    std::vector<int32_t> out(6), unused(6, -99);
    batch_transpose(label.data(), distance.data(), out.data(), unused.data(),
                    1, 2, 3, pool, 4, false);
    assert((out == std::vector<int32_t>{1, 4, 2, 5, 3, 6}));
    assert(std::all_of(unused.begin(), unused.end(), [](int32_t v) { return v == -99; }));
}

static void check_parallel_components() {
    ForkJoinPool pool(4);
    const std::vector<int64_t> shape{517, 521};
    const size_t size = 517 * 521;
    std::vector<int32_t> input(size), serial(size), parallel(size), expected_sources, sources;
    std::mt19937 rng(9);
    for (auto& value : input) value = rng() % 5;
    for (int conn : {1, 2}) {
        const auto expected = cc_label_nd(input.data(), serial.data(), shape, conn);
        const auto actual = cc_label_parallel_nd(input.data(), parallel.data(), shape, conn, pool, 4);
        assert(expected == actual && serial == parallel);
        const auto expected_per_label = cc_label_per_label_nd(
            input.data(), serial.data(), shape, conn, expected_sources);
        const auto actual_per_label = cc_label_parallel_nd<int32_t, true>(
            input.data(), parallel.data(), shape, conn, pool, 4, &sources);
        assert(expected_per_label == actual_per_label && serial == parallel && expected_sources == sources);
    }
}

static void check_dispatch_ranges() {
    ForkJoinPool pool(4);
    std::vector<int> scratch(4);
    for (size_t items : {0, 1, 2, 5, 10, 17, 31, 65}) {
        for (size_t chunks : {0, 1, 4, 8, 16, 64}) {
            for (bool with_scratch : {false, true}) {
                std::vector<std::atomic<int>> visits(items);
                for (auto& v : visits) std::atomic_init(&v, 0);
                auto check = [&](size_t begin, size_t end) {
                    assert(begin <= end && end <= items);
                    for (size_t i = begin; i < end; ++i) ++visits[i];
                };
                if (with_scratch)
                    dispatch_parallel_with_scratch(pool, 4, items, chunks, scratch,
                        [&](int&, size_t begin, size_t end) { check(begin, end); });
                else dispatch_parallel(pool, items, chunks, check);
                for (auto& v : visits) assert(v == 1);
            }
        }
    }
    // The partition arithmetic must also work near size_t's limit,
    // without allocating or iterating over the logical item range.
    const size_t limit = std::numeric_limits<size_t>::max();
    std::atomic<size_t> covered{0};
    dispatch_parallel(pool, limit, 4, [&](size_t begin, size_t end) {
        assert(begin < end);
        covered.fetch_add(end - begin);
    });
    assert(covered == limit);
}

static void check_pool_exception_recovery() {
    for (int threads : {1, 4}) {
        ForkJoinPool pool(threads);
        for (int failure_mode = 0; failure_mode < 3; ++failure_mode) {
            std::atomic<int> calls{0};
            bool caught = false;
            try {
                pool.parallel([&]() {
                    const int call = calls.fetch_add(1);
                    if (failure_mode == 2 && call == 0) throw std::bad_alloc();
                    if (failure_mode == 0 || call == 0)
                        throw std::runtime_error("injected worker failure");
                });
            } catch (const std::runtime_error&) { caught = failure_mode != 2; }
              catch (const std::bad_alloc&) { caught = failure_mode == 2; }
            assert(caught && calls == threads);
            calls = 0;
            pool.parallel([&]() { ++calls; });
            assert(calls == threads);
        }
    }
}

int main() {
    check_feature_transfers();
    check_parallel_components();
    check_dispatch_ranges();
    check_pool_exception_recovery();
    check_deadline_units();
    check_fast_spurs();
    check_thin_spur_allocations();
    check_soft_endpoint_bounds();
    check_large_palettes();
    check_sparse_formatting();
    check_clique_bounds();
    check_buffers();
    check_binary_boundaries();
    check_l1_allocations();
    ForkJoinPool pool(2);
    check_wide_clean(pool);
    check_weights<ReduceMode::Min>(pool);
    check_weights<ReduceMode::Max>(pool);
    check_weights<ReduceMode::Mean>(pool);
    check_weights<ReduceMode::Count>(pool);
    check_weights<ReduceMode::Harmonic>(pool);
    check_weights<ReduceMode::MeanInv>(pool);
}
