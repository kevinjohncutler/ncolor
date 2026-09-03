/*
Optional CPU affinity for ForkJoinPool workers.  EXPERIMENTAL / opt-in.

Off unless the environment variable NCOLOR_PIN_THREADS is set to a
non-empty value other than "0".  When enabled (Linux only for now) each
persistent worker pins itself once, at startup, to a single logical CPU
of a *distinct physical core*.  The win is cache/scheduling locality:
the worker's L1/L2/CCX-L3 stays warm and the scheduler can't migrate it
mid-pass.  This is NOT NUMA placement -- that needs first-touch on
multi-node hardware and is deliberately out of scope.

Pinning only helps on many-core parts; on small boxes rigid placement
loses to the OS scheduler (measured +26% regression at 8 cores).  So even
when enabled it is suppressed below NCOLOR_PIN_MIN_CORES physical cores
(default 32) -- see pin_min_cores().  Net: enabling it can never regress a
machine the floor doesn't clear.

Why one-per-physical-core rather than the naive `cpu = index + 1`:
the naive map is SMT-blind.  On a box that enumerates SMT siblings
adjacently (cpu0,cpu1 == core0; cpu2,cpu3 == core1; ...) it double-books
one physical core while idling another, which regressed a test machine
to 0.72x.  ncolor runs at most `physical` threads (auto_threads() caps
there), so the correct map is simply "one logical CPU per physical
core," read here from sysfs.

Reversibility: this header is self-contained and referenced from exactly
one place (ForkJoinPool::worker_main_ in threadpool.h).  To delete the
feature, remove the `#include "affinity.hpp"` and the single pin_worker()
call there; nothing else depends on it.  With the env var unset it is
already a no-op (one getenv per process).

Single-active-pool assumption: the worker->core map is per-pool (it keys
on a pool-local worker index), so two ForkJoinPools alive *at the same
time* with pinning enabled would both pin onto the same cores and
oversubscribe.  ncolor's normal usage is one engine, and the calibration
path builds its pools sequentially (each destroyed before the next), so
this doesn't arise in practice -- but if a caller holds two engines
concurrently, leave NCOLOR_PIN_THREADS unset.

Non-Linux: pin_worker() is an intentional no-op for now.  macOS has no
hard thread affinity on Apple Silicon (only a QoS P-core hint that
measured as a wash), and Windows needs GetLogicalProcessorInformationEx
to do the same SMT-aware mapping.  Both are deferred until measured to
help on real hardware.
*/
#ifndef NCOLOR_AFFINITY_HPP
#define NCOLOR_AFFINITY_HPP

#include <cstdlib>   // std::getenv

#if defined(__linux__)
#include <sched.h>   // sched_setaffinity, cpu_set_t, CPU_SET
#include <thread>    // std::thread::hardware_concurrency (scan bound)
#include <fstream>
#include <string>
#include <vector>
#include <set>
#endif

namespace ncolor { namespace affinity {

// True iff NCOLOR_PIN_THREADS is set to a non-empty value other than "0".
// Evaluated exactly once (function-local static, shared across TUs).
inline bool enabled() {
    static const bool on = [] {
        const char* e = std::getenv("NCOLOR_PIN_THREADS");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

#if defined(__linux__)

// Minimum physical-core count below which pinning is suppressed even when
// NCOLOR_PIN_THREADS is set.  Rigid pinning needs scheduler slack to pay off:
// measured a clean +26% regression at 1024^2 on an 8-core i9-9900K (the
// floating main thread loses a placement lottery against the pinned workers
// on a core-starved box), versus a ~5-11% win on a 64-core part.  We have
// positive evidence only at 64C and a regression at 8C with nothing measured
// between, so the floor is deliberately conservative.  Override with
// NCOLOR_PIN_MIN_CORES (e.g. to A/B a 16/32-core part).  Evaluated once.
inline unsigned pin_min_cores() {
    static const unsigned m = [] {
        const char* e = std::getenv("NCOLOR_PIN_MIN_CORES");
        if (e && e[0]) { long v = std::atol(e); if (v > 0) return static_cast<unsigned>(v); }
        return 32u;
    }();
    return m;
}

// One representative logical CPU per physical core, ascending.  Built once
// from /sys/.../topology/thread_siblings_list (the representative is the
// smallest CPU id in each sibling group).  Empty if topology is unreadable
// (containers, exotic kernels) -- callers then skip pinning.
inline const std::vector<int>& physical_core_cpus() {
    static const std::vector<int> reps = [] {
        unsigned hc = std::thread::hardware_concurrency();
        // +8 tolerates a few sparse / offline ids above the logical count.
        const int bound = hc > 0 ? static_cast<int>(hc) + 8 : 1024;
        std::set<int> seen;            // dedup by group representative
        std::vector<int> out;
        for (int cpu = 0; cpu < bound; ++cpu) {
            std::string path = "/sys/devices/system/cpu/cpu" +
                std::to_string(cpu) + "/topology/thread_siblings_list";
            std::ifstream f(path);
            if (!f.is_open()) continue;   // offline/absent id -- skip, keep scanning
            std::string list;
            std::getline(f, list);
            // Smallest integer in the list (handles "0,4" and "0-1" forms).
            int rep = cpu, val = 0;
            bool have = false;
            for (char c : list) {
                if (c >= '0' && c <= '9') { val = (have ? val * 10 : 0) + (c - '0'); have = true; }
                else { if (have && val < rep) rep = val; have = false; }
            }
            if (have && val < rep) rep = val;
            if (seen.insert(rep).second) out.push_back(rep);
        }
        return out;
    }();
    return reps;
}

// Pin the calling worker to a distinct physical core, spread evenly across
// the whole core list so an under-subscribed pool (workers + 1 < physical)
// still touches every L3/CCX domain instead of clustering on the low-numbered
// cores -- clustering starves half the caches/memory controllers on
// multi-CCX parts and measured as a net loss at T < physical.  worker_index
// is 0-based; participant 0 (the unpinned, GIL-owning main thread) takes the
// first slot, workers take the rest, so reps[0]'s core stays free for main.
// At full subscription (participants == physical) this reduces exactly to the
// consecutive map cpu = reps[index + 1].  Modulo keeps off-label
// oversubscription (threads > physical) in range.
inline void pin_worker(unsigned worker_index, unsigned num_workers) {
    if (!enabled()) return;
    const std::vector<int>& reps = physical_core_cpus();
    if (reps.empty()) return;                       // unknown topology -> no-op
    const unsigned P = static_cast<unsigned>(reps.size());
    if (P < pin_min_cores()) return;                // small box: pinning regresses -> no-op
    const unsigned participants = num_workers + 1;  // workers + main thread
    unsigned slot;
    if (participants <= P) {
        // Even spread: round((worker_index + 1) * P / participants), which
        // lands in [1, P-1] (never 0 -> main keeps reps[0]) and is strictly
        // increasing in worker_index, so cores stay distinct.
        slot = ((worker_index + 1) * P + participants / 2) / participants;
        if (slot >= P) slot = P - 1;
    } else {
        slot = (worker_index + 1) % P;              // oversubscribed: wrap
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(reps[slot], &set);
    sched_setaffinity(0, sizeof(set), &set);
}

#else  // non-Linux: deferred, no-op

inline void pin_worker(unsigned /*worker_index*/, unsigned /*num_workers*/) {}

#endif

}}  // namespace ncolor::affinity

#endif  // NCOLOR_AFFINITY_HPP
