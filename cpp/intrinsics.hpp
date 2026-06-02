// Portable bit-twiddling intrinsic wrappers.
//
// Provides inline shims for the GCC/Clang `__builtin_ctz` /
// `__builtin_ctzll` / `__builtin_popcountll` calls used throughout the
// engine, with MSVC equivalents (`_BitScanForward`,
// `_BitScanForward64`, `__popcnt64`). C++20's `<bit>` would give us
// `std::countr_zero` / `std::popcount` portably; we target C++17, so we
// wrap manually.

#ifndef NCOLOR_INTRINSICS_HPP
#define NCOLOR_INTRINSICS_HPP

#include <cstdint>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

namespace ncolor_cpp {

// Count trailing zeros of a non-zero 32-bit value.
// Undefined for x == 0 (matches __builtin_ctz semantics).
static inline int ctz_u32(uint32_t x) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanForward(&index, x);
    return static_cast<int>(index);
#else
    return __builtin_ctz(x);
#endif
}

// Count trailing zeros of a non-zero 64-bit value.
static inline int ctz_u64(uint64_t x) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long index;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
#else
    return __builtin_ctzll(x);
#endif
}

// Population count of a 64-bit value.
static inline int popcount_u64(uint64_t x) {
#if defined(_MSC_VER) && !defined(__clang__)
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}

}  // namespace ncolor_cpp

#endif  // NCOLOR_INTRINSICS_HPP
