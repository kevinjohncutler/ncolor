// The ncolor pipeline from C++ alone: no Python, no numpy.
//
//   c++ -std=c++17 -O3 -pthread -march=native -I. example_standalone.cpp -o example_standalone
//   ./example_standalone
//
// Builds a synthetic label image, runs connected components, Voronoi
// expansion, the adjacency scan, and the coloring picker, then reports
// how many colors were needed and checks the result. The same headers
// are what ncolor._backend is compiled from; binding.cpp only adds the
// numpy glue.
#include <cstdint>
#include <cstdio>
#include <vector>

#include "ncolor.hpp"

int main() {
    const int64_t H = 256, W = 256;
    const int n_threads = 4;
    ForkJoinPool pool(n_threads);

    // 1. A binary mask of blobs, labeled by connected components.
    std::vector<uint8_t> mask(H * W, 0);
    for (int64_t y = 8; y < H - 8; y += 24) {
        for (int64_t x = 8; x < W - 8; x += 24) {
            for (int64_t dy = 0; dy < 12; ++dy) {
                for (int64_t dx = 0; dx < 12; ++dx) {
                    mask[(y + dy) * W + (x + dx)] = 1;
                }
            }
        }
    }
    std::vector<int32_t> labels(H * W, 0);
    const std::vector<int64_t> shape{H, W};
    const int32_t n_cells = ncolor_cpp::cc_label_nd<uint8_t>(
        mask.data(), labels.data(), shape, /*conn=*/1);

    // 2. Voronoi-expand the cells so near neighbors become adjacent.
    ncolor_cpp::ExpandBuffers bufs;
    ncolor_cpp::expand_labels_lp<2>(labels.data(), labels.data(), bufs, shape,
                                    pool, n_threads, /*wrap=*/false);

    // 3. Adjacency pairs (face connectivity) and the CSR the picker wants.
    std::vector<uint64_t> ht_buf;
    const uint64_t ht_size = 1u << 16;
    auto pairs = ncolor_cpp::find_pairs_nd_unpadded<int32_t>(
        labels.data(), shape, /*conn=*/1, ht_size, n_threads, pool,
        /*wrap=*/false, /*radius=*/1, &ht_buf);
    const int32_t M = static_cast<int32_t>(pairs.size());
    std::vector<int32_t> src(M), dst(M);
    for (int32_t i = 0; i < M; ++i) {
        // Cell ids are 1-based; the picker's vertices are 0-based.
        src[i] = pairs[i].first - 1;
        dst[i] = pairs[i].second - 1;
    }
    std::vector<int32_t> indptr, indices;
    ncolor_cpp::build_csr_from_pairs(src.data(), dst.data(), n_cells, M,
                                     indptr, indices);

    // 4. Color it.
    std::vector<uint8_t> colors;
    int n_conflicts = 0;
    ncolor_cpp::PickerScratch scratch;
    const int n_used = ncolor_cpp::pick_coloring(
        n_cells, M, /*n_colors=*/4, /*max_depth=*/30, /*rand_period=*/10,
        /*color_mode=*/-1, /*ndim=*/2, /*wrap=*/false,
        /*edge_weights=*/nullptr, /*de_table=*/nullptr, /*weight_obj=*/0,
        indptr, indices, src, dst, colors, n_conflicts, scratch,
        &pool, n_threads);

    std::printf("%d cells, %d adjacencies, colored with %d colors, %d conflicts\n",
                n_cells, M, n_used, n_conflicts);
    return (n_conflicts == 0 && n_used >= 1 && n_used <= 4) ? 0 : 1;
}
