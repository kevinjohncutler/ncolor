// Compare shared atomic flags, private byte flags, and private packed bits.
// Build: c++ -std=c++17 -O3 -pthread -I cpp bench/presence_experiment.cpp -o /tmp/ncolor-presence
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>
#include "dispatch.hpp"

int main() {
    ForkJoinPool pool(4);
    const size_t total = 2048 * 2048, chunks = 16;
    for (int domain : {32, 4096, 65536}) {
        std::vector<int32_t> labels(total);
        uint32_t state = 17;
        for (auto& value : labels) {
            state = state * 1664525u + 1013904223u;
            value = state % domain;
        }
        for (int method = 0; method < 3; ++method) {
            std::vector<double> times;
            for (int repetition = 0; repetition < 25; ++repetition) {
                const auto start = std::chrono::steady_clock::now();
                int count = 0;
                if (method == 0) {
                    std::vector<std::atomic<uint8_t>> flags(domain);
                    for (auto& flag : flags) std::atomic_init(&flag, uint8_t{0});
                    ncolor_cpp::dispatch_parallel(pool, total, chunks, [&](size_t a, size_t b) {
                        for (size_t i = a; i < b; ++i)
                            if (!flags[labels[i]].load(std::memory_order_relaxed))
                                flags[labels[i]].store(1, std::memory_order_relaxed);
                    });
                    for (auto& flag : flags) count += flag.load(std::memory_order_relaxed);
                } else if (method == 1) {
                    std::vector<uint8_t> flags(chunks * domain, 0);
                    ncolor_cpp::dispatch_parallel(pool, chunks, chunks, [&](size_t a, size_t b) {
                        for (size_t c = a; c < b; ++c) {
                            uint8_t* local = flags.data() + c * domain;
                            for (size_t i = total * c / chunks; i < total * (c + 1) / chunks; ++i)
                                local[labels[i]] = 1;
                        }
                    });
                    for (int value = 0; value < domain; ++value) {
                        bool any = false;
                        for (size_t c = 0; c < chunks; ++c) any |= flags[c * domain + value];
                        count += any;
                    }
                } else {
                    const size_t words = (domain + 63) / 64;
                    std::vector<uint64_t> flags(chunks * words, 0);
                    ncolor_cpp::dispatch_parallel(pool, chunks, chunks, [&](size_t a, size_t b) {
                        for (size_t c = a; c < b; ++c) {
                            uint64_t* local = flags.data() + c * words;
                            for (size_t i = total * c / chunks; i < total * (c + 1) / chunks; ++i)
                                local[labels[i] / 64] |= uint64_t{1} << (labels[i] % 64);
                        }
                    });
                    for (size_t word = 0; word < words; ++word) {
                        uint64_t bits = 0;
                        for (size_t c = 0; c < chunks; ++c) bits |= flags[c * words + word];
                        count += __builtin_popcountll(bits);
                    }
                }
                if (count != domain) return 1;
                const auto end = std::chrono::steady_clock::now();
                if (repetition > 4)
                    times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
            }
            std::sort(times.begin(), times.end());
            std::cout << domain << "," << method << "," << times[times.size() / 2] << "\n";
        }
    }
}
