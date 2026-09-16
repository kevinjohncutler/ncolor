// Compare the complete feature-transform/contact phase with a retained layout.
// Cleanup and contacts use the final transposed field. The public label image
// is reconstructed outside timing to verify exact labels and barrier placement.
#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include "expand_clean.hpp"
#include "connect.hpp"

using namespace ncolor_cpp;
using Pairs = std::vector<std::pair<int32_t, int32_t>>;

static std::vector<int64_t> retained_expand(const std::vector<int32_t>& input,
        ExpandBuffers& buffers, const std::vector<int64_t>& shape,
        ForkJoinPool& pool, bool clean, bool wrap) {
    std::vector<int64_t> layout;
    if (clean) expand_labels_clean_inplace(input.data(), buffers, shape, pool, 4, 2,
                                           wrap, false, &layout);
    else expand_labels_inplace(input.data(), buffers, shape, pool, 4, wrap, false, &layout);
    return layout;
}

int main() {
    ForkJoinPool pool(4);
    std::mt19937 rng(744);
    std::cout << "shape,clean,wrap,baseline_ms,retained_ms\n";
    for (const auto& shape : std::vector<std::vector<int64_t>>{
            {2, 3}, {63, 65}, {257, 263}, {512, 512}, {2048, 2048},
            {2, 513, 517}, {17, 19, 23}, {96, 96, 96}, {3, 5, 7, 11}}) {
        const int64_t size = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
        for (bool clean : {false, true}) for (bool wrap : {false, true}) {
            std::vector<int32_t> input(size), restored(size), unused(size);
            int n = 0;
            for (auto& v : input) if (rng() % 1024 == 0) v = ++n;
            input[0] = ++n;
            ExpandBuffers normal, retained;
            uint64_t hsize = 16;
            while (hsize < static_cast<uint64_t>(n) * 32) hsize *= 2;
            std::vector<uint64_t> hard_a, hard_b, soft_a, soft_b;
            Pairs ha, hb, sa, sb;
            std::vector<double> ta, tb;
            std::vector<int64_t> rotated;
            for (int rep = 0; rep < 17; ++rep) {
                auto baseline = [&] {
                    if (clean) expand_labels_clean_inplace(input.data(), normal, shape, pool, 4, 2, wrap, false);
                    else expand_labels_inplace(input.data(), normal, shape, pool, 4, wrap, false);
                    assert(!find_pairs_dual_nd_unpadded(normal.lbl(), shape, 1, 1, 2, 2,
                        hsize, hsize, 4, pool, wrap, ha, sa, &hard_a, &soft_a));
                };
                auto candidate = [&] {
                    rotated = retained_expand(input, retained, shape, pool, clean, wrap);
                    assert(!find_pairs_dual_nd_unpadded(retained.lbl_T(), rotated, 1, 1, 2, 2,
                        hsize, hsize, 4, pool, wrap, hb, sb, &hard_b, &soft_b));
                };
                auto measure = [&](auto call, auto& times) {
                    const auto start = std::chrono::steady_clock::now();
                    call();
                    const auto end = std::chrono::steady_clock::now();
                    if (rep >= 2) times.push_back(std::chrono::duration<double, std::milli>(end-start).count());
                };
                if (rep % 2) { measure(candidate, tb); measure(baseline, ta); }
                else { measure(baseline, ta); measure(candidate, tb); }
                batch_transpose(retained.lbl_T(), retained.dist_T(), restored.data(), unused.data(),
                                1, size / shape[0], shape[0], pool, 4, false);
                assert(std::equal(restored.begin(), restored.end(), normal.lbl()));
                std::sort(ha.begin(), ha.end()); std::sort(hb.begin(), hb.end());
                std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
                assert(ha == hb && sa == sb);
            }
            std::sort(ta.begin(), ta.end()); std::sort(tb.begin(), tb.end());
            for (size_t i = 0; i < shape.size(); ++i) std::cout << (i ? "x" : "") << shape[i];
            std::cout << ',' << clean << ',' << wrap << ',' << ta[7] << ',' << tb[7] << '\n';
        }
    }
}
