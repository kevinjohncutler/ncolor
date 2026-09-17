// Direct versus pooled dispatch, including an unchanged multiple-chunk control.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include "dispatch.hpp"

int main() {
    using clock = std::chrono::steady_clock;
    std::cout << "threads,items,chunks,median_us\n";
    for (int threads : {1, 4, 16}) {
        ForkJoinPool pool(threads);
        for (size_t size : {size_t{1}, size_t{8192}}) {
            std::vector<uint32_t> data(size, 0);
            for (size_t chunks : {size_t{1}, size_t{16}}) {
                std::vector<double> times;
                for (int rep = 0; rep < 25; ++rep) {
                    auto start = clock::now();
                    for (int i = 0; i < 100; ++i)
                        ncolor_cpp::dispatch_parallel(pool, size, chunks,
                            [&](size_t begin, size_t end) {
                                for (size_t j = begin; j < end; ++j) data[j] += 1;
                            });
                    if (rep >= 4)
                        times.push_back(std::chrono::duration<double, std::micro>(clock::now()-start).count()/100);
                }
                const uint32_t expected = chunks == 1 ? 2500 : 5000;
                for (auto value : data) if (value != expected) return 1;
                std::sort(times.begin(), times.end());
                std::cout << threads << ',' << size << ',' << chunks << ',' << times[times.size()/2] << '\n';
            }
        }
    }
}
