/*
Fork-Join Thread Pool for parallel dispatch.

Sense-reversing centralized barrier with platform-specific idle fallback:
  * **Spin** (FORKJOIN_SPIN_PAUSES pauses) catches tight back-to-back
    fork-join cycles within microseconds. Cross-platform.
  * After the spin window expires:
      - On **Linux/Windows**: interspersed spin + ``sched_yield``
        (matches pre-patch behavior). ``sched_yield`` is cheap here, so
        an idle pool burns low single-digit %% CPU and a warm pool
        wakes in ~us.
      - On **macOS**: ``std::this_thread::sleep_for(FORKJOIN_SLEEP_US)``
        in a loop. macOS ``yield()`` was causing a ``swtch_pri`` storm
        (~1700-1800 %% CPU across 19 idle workers); sleep avoids that.

Why ifdef-scoped rather than uniform: the original macOS storm is a
macOS-scheduler-specific behavior. A uniform sleep fallback regresses
Linux fork-join workloads 50-200 %% at multi-T on a 64-core x86 host
(measured during the edt port, 2026-05-23) because ``sleep_for`` is too
coarse for tight loops. A uniform condvar park (alternative also tested)
costs ~25 %% on nd_profile p=16. Pre-patch ``yield()`` was already
perf-correct on Linux; only the macOS path needed the fix.

Original ThreadPool: Copyright (c) 2012 Jakob Progsch, Václav Zeman (zlib license).
Rewritten by William Silversmith and Kevin Cutler, 2025-2026.
*/

#ifndef NCOLOR_THREADPOOL_H
#define NCOLOR_THREADPOOL_H

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

// Cross-platform spin-pause hint
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #include <immintrin.h>
  #define FORKJOIN_PAUSE() _mm_pause()
#elif defined(__aarch64__) || defined(_M_ARM64)
  #ifdef _MSC_VER
    #include <intrin.h>
    #define FORKJOIN_PAUSE() __yield()
  #else
    #define FORKJOIN_PAUSE() __asm__ __volatile__("yield")
  #endif
#else
  #define FORKJOIN_PAUSE() ((void)0)
#endif

// How many CPU pauses to spin before falling back to the platform
// idle path. ~10 ns/pause on Apple Silicon, ~30 ns on x86_64 --
// long enough to cover tight back-to-back fork-join cycles, short
// enough that an idle pool quickly hands off to the OS scheduler.
#ifndef FORKJOIN_SPIN_PAUSES
#define FORKJOIN_SPIN_PAUSES 8192
#endif

#if defined(__APPLE__)
// macOS-only: sleep granularity once the spin window expires. 5 ms
// keeps idle 19-worker pools at < 1 %% total CPU. nanosleep below
// ~500 us degrades into a kernel busy-wait on macOS, so 5 ms is the
// practical floor.
#ifndef FORKJOIN_SLEEP_US
#define FORKJOIN_SLEEP_US 5000
#endif
#else
// Linux/Windows: yield is cheap, so we use the pre-patch interspersed
// spin + yield loop (unbounded, exits as soon as sense flips).
// FORKJOIN_INNER_SPIN sets pauses-per-yield in that loop.
#ifndef FORKJOIN_INNER_SPIN
#define FORKJOIN_INNER_SPIN 1024
#endif
#endif

class ForkJoinPool {
public:
    explicit ForkJoinPool(size_t num_threads)
        : num_participants_(num_threads > 0 ? num_threads : 1),
          num_workers_(num_participants_ - 1),
          alive_(true),
          bar_count_(0),
          bar_sense_(0)
    {
        workers_.reserve(num_workers_);
        for (size_t i = 0; i < num_workers_; ++i) {
            workers_.emplace_back(&ForkJoinPool::worker_main_, this);
        }
    }

    // Execute fn on all workers + calling thread, block until all complete.
    // fn must be safe to call from multiple threads concurrently.
    template <typename F>
    void parallel(F&& fn) {
        if (num_workers_ == 0) {
            fn();
            return;
        }
        work_fn_ = std::forward<F>(fn);
        barrier_wait_();   // release workers (they're waiting at start barrier)
        work_fn_();        // main thread participates
        barrier_wait_();   // wait for all workers to finish
    }

    ~ForkJoinPool() {
        alive_.store(false, std::memory_order_relaxed);
        // Release workers from their start-barrier wait so they can see alive_==false
        barrier_wait_();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    // Non-copyable, non-movable
    ForkJoinPool(const ForkJoinPool&) = delete;
    ForkJoinPool& operator=(const ForkJoinPool&) = delete;

private:
    void worker_main_() {
        for (;;) {
            barrier_wait_();   // wait for work to be posted
            if (!alive_.load(std::memory_order_relaxed)) return;
            work_fn_();        // execute work
            barrier_wait_();   // signal completion
        }
    }

    // Sense-reversing centralized barrier with platform-scoped idle path.
    // Spin window is cross-platform; the post-spin idle behavior differs
    // because Linux/macOS scheduler costs for ``yield`` differ by ~100x.
    void barrier_wait_() {
        const int local_sense = 1 - bar_sense_.load(std::memory_order_relaxed);
        const size_t arrived = bar_count_.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (arrived == num_participants_) {
            // Last to arrive: reset count and flip sense to release everyone.
            bar_count_.store(0, std::memory_order_relaxed);
            bar_sense_.store(local_sense, std::memory_order_release);
            return;
        }

        // Phase 1: spin. Covers tight back-to-back fork-join cycles.
        for (int i = 0; i < FORKJOIN_SPIN_PAUSES; ++i) {
            if (bar_sense_.load(std::memory_order_acquire) == local_sense) {
                return;
            }
            FORKJOIN_PAUSE();
        }

#if defined(__APPLE__)
        // macOS: sleep. ``yield`` here caused a ``swtch_pri`` storm
        // (~1700-1800 %% CPU across 19 idle workers); sleep avoids it.
        // Wake-up after a true idle period costs up to one
        // FORKJOIN_SLEEP_US interval (~5 ms); the spin window above
        // covers warm/tight back-to-back cycles before any sleep happens.
        while (bar_sense_.load(std::memory_order_acquire) != local_sense) {
            std::this_thread::sleep_for(std::chrono::microseconds(FORKJOIN_SLEEP_US));
        }
#else
        // Linux/Windows: interspersed spin + yield (matches pre-patch
        // behavior). ``sched_yield`` is cheap on Linux (~1 us syscall,
        // no swtch_pri equivalent), so unbounded yielding is fine --
        // idle pools tick at low CPU and a warm pool wakes in ~us.
        // Sleep-based or cv-based fallbacks both regress real workloads
        // (sleep: 50-200 %% on nd_profile p=8/16/32; cv: 20-35 %% same)
        // because their fixed per-wait overhead is large relative to
        // small ``parallel()`` calls.
        while (bar_sense_.load(std::memory_order_acquire) != local_sense) {
            for (int i = 0; i < FORKJOIN_INNER_SPIN; ++i) {
                if (bar_sense_.load(std::memory_order_acquire) == local_sense) {
                    return;
                }
                FORKJOIN_PAUSE();
            }
            std::this_thread::yield();
        }
#endif
    }

    const size_t num_participants_;  // workers + 1 (main thread)
    const size_t num_workers_;
    std::atomic<bool> alive_;

    // Barrier state
    std::atomic<size_t> bar_count_;
    std::atomic<int> bar_sense_;

    // Current work function (set by parallel(), read by workers)
    std::function<void()> work_fn_;

    std::vector<std::thread> workers_;
};

#endif // NCOLOR_THREADPOOL_H
