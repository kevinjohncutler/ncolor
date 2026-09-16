// Shared units for absolute search deadlines, independent of clock period.
#pragma once

#include <chrono>
#include <cstdint>

namespace ncolor_cpp {
inline int64_t steady_time_ns(
        std::chrono::steady_clock::time_point at = std::chrono::steady_clock::now()) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        at.time_since_epoch()).count();
}
} // namespace ncolor_cpp
