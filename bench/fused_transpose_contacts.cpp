// Prototype: transpose into a tile with a halo, emit contacts while hot,
// and publish the same final label image. Uses the shipped contact kernel.
#include <array>
#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include "expand_clean.hpp"
#include "connect.hpp"

using namespace ncolor_cpp;
using Pairs = std::vector<std::pair<int32_t, int32_t>>;

struct Fusion {
    static constexpr int tile_size = 64, radius = 2, pitch = tile_size + 2 * radius;
    int threads;
    uint64_t base_size, soft_size;
    std::vector<uint64_t> base, soft;
    std::vector<int64_t> offsets;
    int n_base = 0, n_near = 0;
    Fusion(int nt, uint64_t hb, uint64_t hs) : threads(nt), base_size(hb), soft_size(hs),
        base(nt * hb), soft(nt * hs) {
        std::vector<int64_t> strides;
        std::vector<int8_t> deltas;
        detail::build_forward_neighbors_dual({pitch, pitch}, 1, 1, 2, 2,
            strides, offsets, deltas, n_base, &n_near);
    }
    void run(const int32_t* transposed, int32_t* output, int64_t height, int64_t width,
             ForkJoinPool& pool, bool wrap, Pairs& hard, Pairs& preferences) {
        const int64_t nx = (width + tile_size - 1) / tile_size;
        const int64_t ny = (height + tile_size - 1) / tile_size;
        std::atomic<int> worker{0};
        std::atomic<int64_t> next{0};
        pool.parallel([&] {
            const int id = worker.fetch_add(1);
            uint64_t* hb = base.data() + id * base_size;
            uint64_t* hs = soft.data() + id * soft_size;
            std::fill_n(hb, base_size, HT_EMPTY);
            std::fill_n(hs, soft_size, HT_EMPTY);
            std::array<int32_t, pitch * pitch> tile;
            int64_t t;
            while ((t = next.fetch_add(1)) < nx * ny) {
                const int64_t y0 = (t / nx) * tile_size;
                const int64_t x0 = (t % nx) * tile_size;
                for (int x = 0; x < pitch; x += 4) {
                    for (int y = 0; y < pitch; y += 4) {
                        const int64_t gx = x0 + x - radius, gy = y0 + y - radius;
#if defined(NCOLOR_SIMD_NEON) || defined(NCOLOR_SIMD_X86)
                        if (gx >= 0 && gy >= 0 && gx + 4 <= width && gy + 4 <= height) {
                            transpose_4x4_4byte(transposed + gx * height + gy, height,
                                               tile.data() + y * pitch + x, pitch);
                            continue;
                        }
#endif
                        for (int dx = 0; dx < 4; ++dx) for (int dy = 0; dy < 4; ++dy) {
                            int64_t xx = gx + dx, yy = gy + dy;
                            if (wrap) {
                                xx = (xx % width + width) % width;
                                yy = (yy % height + height) % height;
                            }
                            tile[(y + dy) * pitch + x + dx] =
                                xx >= 0 && yy >= 0 && xx < width && yy < height
                                ? transposed[xx * height + yy] : 0;
                        }
                    }
                }
                const int64_t h = std::min<int64_t>(tile_size, height - y0);
                const int64_t w = std::min<int64_t>(tile_size, width - x0);
                for (int64_t y = 0; y < h; ++y) {
                    const int32_t* row = tile.data() + (y + radius) * pitch;
                    std::copy_n(row + radius, w, output + (y0 + y) * width + x0);
                    scan_inner_axis_dual_dispatch(row, radius, radius + w, n_base,
                        static_cast<int>(offsets.size()) - n_base, offsets.data(), n_near,
                        hb, base_size - 1, hs, soft_size - 1);
                }
            }
        });
        for (int stride = 1; stride < threads; stride *= 2) {
            const int count = (threads + 2 * stride - 1) / (2 * stride);
            dispatch_parallel(pool, count, count, [&](size_t lo, size_t hi) {
                for (size_t p = lo; p < hi; ++p) {
                    const int dst = static_cast<int>(p) * 2 * stride, src = dst + stride;
                    if (src >= threads) continue;
                    ht_merge(base.data() + src * base_size, base.data() + dst * base_size, base_size);
                    ht_merge(soft.data() + src * soft_size, soft.data() + dst * soft_size, soft_size);
                }
            });
        }
        auto extract = [](const std::vector<uint64_t>& table, uint64_t size, Pairs& pairs) {
            pairs.clear();
            for (uint64_t i = 0; i < size; ++i) if (table[i] != HT_EMPTY)
                pairs.emplace_back(static_cast<int32_t>(table[i] >> 32), static_cast<int32_t>(table[i]));
            assert(pairs.size() < size);
        };
        extract(base, base_size, hard);
        extract(soft, soft_size, preferences);
        preferences.erase(std::remove_if(preferences.begin(), preferences.end(), [&](const auto& pair) {
            const uint64_t key = (static_cast<uint64_t>(pair.first) << 32) | static_cast<uint32_t>(pair.second);
            const uint64_t slot = ht_probe(base.data(), base_size - 1, key);
            return slot < base_size && base[slot] == key;
        }), preferences.end());
    }
};

int main() {
    ForkJoinPool pool(4);
    std::cout << "height,width,clean,wrap,jitter,separate_ms,fused_ms\n";
    for (const auto& shape : {std::vector<int64_t>{1, 1}, {2, 3}, {63, 65}, {65, 67},
                              {257, 263}, {512, 512}, {2048, 2048}}) {
        const int64_t height = shape[0], width = shape[1], total = height * width;
        for (bool clean : {false, true}) for (bool wrap : {false, true})
        for (bool jitter : {false, true}) {
            std::vector<int32_t> seeds(total), transposed(total), dummy(total), expected(total), actual(total);
            int label = 0;
            std::mt19937 rng(493);
            for (int64_t y = 8; y < height; y += 32)
                for (int64_t x = 8; x < width; x += 32) {
                    const int64_t yy = jitter ? std::min(height - 1, y + rng() % 19) : y;
                    const int64_t xx = jitter ? std::min(width - 1, x + rng() % 19) : x;
                    seeds[yy * width + xx] = ++label;
                }
            if (height < 8 || width < 8) {
                for (int64_t i = 0; i < total; ++i) seeds[i] = static_cast<int32_t>(i % 3);
                label = 2;
            }
            ExpandBuffers expansion;
            if (clean) expand_labels_clean_inplace(seeds.data(), expansion, shape, pool, 4, 2, wrap);
            else expand_labels_inplace(seeds.data(), expansion, shape, pool, 4, wrap);
            batch_transpose(expansion.lbl(), dummy.data(), transposed.data(), dummy.data(),
                            1, height, width, pool, 4, false);
            uint64_t base_size = 16, soft_size = 16;
            while (base_size < static_cast<uint64_t>(label * 8)) base_size *= 2;
            while (soft_size < static_cast<uint64_t>(label * 16)) soft_size *= 2;
            Fusion fusion(4, base_size, soft_size);
            std::vector<uint64_t> base_scratch, soft_scratch;
            Pairs hard, soft, fused_hard, fused_soft;
            std::vector<double> separate_times, fused_times;
            for (int rep = 0; rep < 27; ++rep) {
                auto separate = [&] {
                    batch_transpose(transposed.data(), dummy.data(), expected.data(), dummy.data(),
                                    1, width, height, pool, 4, false);
                    const int full = find_pairs_dual_nd_unpadded(expected.data(), shape, 1, 1, 2, 2,
                        base_size, soft_size, 4, pool, wrap, hard, soft, &base_scratch, &soft_scratch);
                    assert(!full);
                };
                auto fused = [&] { fusion.run(transposed.data(), actual.data(), height, width,
                                               pool, wrap, fused_hard, fused_soft); };
                auto measure = [&](auto call, std::vector<double>& times) {
                    const auto begin = std::chrono::steady_clock::now(); call();
                    const auto end = std::chrono::steady_clock::now();
                    if (rep >= 2) times.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
                };
                if (rep % 2) { measure(fused, fused_times); measure(separate, separate_times); }
                else { measure(separate, separate_times); measure(fused, fused_times); }
                assert(actual == expected);
                std::sort(hard.begin(), hard.end()); std::sort(soft.begin(), soft.end());
                std::sort(fused_hard.begin(), fused_hard.end()); std::sort(fused_soft.begin(), fused_soft.end());
                if (hard != fused_hard || soft != fused_soft) {
                    std::cerr << "Mismatch " << height << "," << width << "," << clean << "," << wrap
                              << " hard " << hard.size() << "," << fused_hard.size()
                              << " soft " << soft.size() << "," << fused_soft.size() << "\n";
                    Pairs missing, extra;
                    std::set_difference(soft.begin(), soft.end(), fused_soft.begin(), fused_soft.end(), std::back_inserter(missing));
                    std::set_difference(fused_soft.begin(), fused_soft.end(), soft.begin(), soft.end(), std::back_inserter(extra));
                    for (auto edge : missing) std::cerr << "missing " << edge.first << "," << edge.second << "\n";
                    for (auto edge : extra) std::cerr << "extra " << edge.first << "," << edge.second << "\n";
                    return 1;
                }
            }
            std::sort(separate_times.begin(), separate_times.end());
            std::sort(fused_times.begin(), fused_times.end());
            std::cout << height << ',' << width << ',' << clean << ',' << wrap << ',' << jitter << ','
                      << separate_times[12] << ',' << fused_times[12] << '\n';
        }
    }
}
