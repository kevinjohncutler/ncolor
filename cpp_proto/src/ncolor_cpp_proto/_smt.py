"""SMT/HT thread-count auto-calibration for ``ncolor_cpp_proto.Solver``.

Whether SMT (AMD) / HT (Intel) helps for our expand kernel depends on the
specific chip's memory subsystem and cache topology. Empirically:

    * AMD Ryzen 9 7950X (Zen4, 16C/32T):  T=32 wins by ~25%
    * AMD TR PRO 3995WX  (Zen2, 64C/128T): T=64 wins by ~50% (T=128 hurts)
    * Intel i9-9900K     (Coffee, 8C/16T): T=16 wins by ~5%
    * Intel i7-7820HQ    (Skylake, 4C/8T): T=8 wins by ~10%

There's no static heuristic from /proc/cpuinfo that picks the right answer
across all of these. Memory-bandwidth probes (STREAM, pointer-chase) also
fail to predict because the deciding factor is workload-specific stall
behavior. The robust answer: time the actual workload at T=physical and
T=logical once per machine, cache the result.

Calibration runs at 1024² (the smallest size where SMT vs no-SMT separates
cleanly across all hosts we tested) and takes ~200 ms. Result is cached at
``~/.cache/ncolor_cpp_proto/smt_threads.json`` keyed by (hostname, CPU model).

Public API:
    auto_threads()         — return cached optimal, fall back to physical
    calibrate(force=False) — run calibration, write cache, return optimal
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

CACHE_PATH = Path.home() / ".cache" / "ncolor_cpp_proto" / "smt_threads.json"
CALIBRATION_SIZE = 1024


def _cpu_model() -> str:
    """Best-effort CPU model string. Used as part of the cache key."""
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
            ).strip()
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "Name", "/value"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for line in out.splitlines():
                if line.startswith("Name="):
                    return line.split("=", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def _physical_cores() -> int:
    """Physical (not logical) core count."""
    if sys.platform == "linux":
        try:
            cores: set[tuple[str, str]] = set()
            phys_id = core_id = None
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("physical id"):
                        phys_id = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core_id = line.split(":", 1)[1].strip()
                    elif line.strip() == "":
                        if phys_id is not None and core_id is not None:
                            cores.add((phys_id, core_id))
                        phys_id = core_id = None
            if cores:
                return len(cores)
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            return int(subprocess.check_output(
                ["sysctl", "-n", "hw.physicalcpu"], text=True).strip())
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            out = subprocess.check_output(
                ["wmic", "cpu", "get", "NumberOfCores", "/value"],
                text=True, stderr=subprocess.DEVNULL,
            )
            n = sum(int(line.split("=", 1)[1].strip())
                    for line in out.splitlines()
                    if line.startswith("NumberOfCores="))
            if n:
                return n
        except Exception:
            pass
    return os.cpu_count() or 1


def _cache_key() -> str:
    return f"{socket.gethostname()}|{_cpu_model()}"


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def _make_calibration_mask(H: int, seed: int = 0):
    import numpy as np
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, H), dtype=np.int32)
    n = max(20, H * H // 8000)
    for i in range(1, n + 1):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, H - 20)
        r = int(rng.integers(8, max(16, H // 30)))
        yy, xx = np.ogrid[:H, :H]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return mask


def _time_solver(sv, mask, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        sv.label(mask)
    best = float("inf")
    for _ in range(iters):
        t = time.perf_counter()
        sv.label(mask)
        dt = time.perf_counter() - t
        if dt < best:
            best = dt
    return best


def calibrate(force: bool = False, verbose: bool = False) -> int:
    """Run a calibration at ``CALIBRATION_SIZE`` and write the result to cache.

    Compares T=physical vs T=logical timings for ``Solver.label`` on a 1024²
    synthetic mask. Picks logical only if it beats physical by ≥2%; ties go to
    physical (safer choice — never catastrophically wrong).

    If ``force=False`` and a cache entry already exists for this machine,
    returns the cached value without re-running.

    Pool lifetimes are sequential: the physical-thread pool is destroyed
    before the logical-thread pool is created, so spin-waiting workers from
    the unused pool don't contaminate the measurement.

    Returns the picked thread count.
    """
    import ncolor_cpp_proto as nc

    phys = _physical_cores()
    log = os.cpu_count() or phys
    key = _cache_key()
    cache = _load_cache()

    if not force and key in cache:
        return int(cache[key])

    if log <= phys:
        # No SMT/HT — only one choice.
        cache[key] = phys
        _save_cache(cache)
        return phys

    mask = _make_calibration_mask(CALIBRATION_SIZE)

    sv = nc.Solver(phys)
    t_phys = _time_solver(sv, mask)
    del sv

    sv = nc.Solver(log)
    t_log = _time_solver(sv, mask)
    del sv

    chosen = log if t_log < t_phys * 0.98 else phys

    if verbose:
        print(f"[ncolor_cpp_proto] calibrated SMT on {key}: "
              f"T={phys}: {t_phys*1000:.2f} ms, T={log}: {t_log*1000:.2f} ms "
              f"→ T={chosen}")

    cache[key] = chosen
    _save_cache(cache)
    return chosen


def auto_threads() -> int:
    """Return the optimal thread count for ``Solver``, using the cache.

    If no cache entry exists for this machine, returns ``_physical_cores()``
    as a safe default. Call :func:`calibrate` once to populate the cache.
    """
    cache = _load_cache()
    key = _cache_key()
    if key in cache:
        return int(cache[key])
    return _physical_cores()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Calibrate SMT thread count for ncolor_cpp_proto.Solver.")
    p.add_argument("--force", action="store_true", help="Re-run even if cached.")
    p.add_argument("--show", action="store_true", help="Print current cache and exit.")
    args = p.parse_args()
    if args.show:
        print(f"cache: {CACHE_PATH}")
        print(json.dumps(_load_cache(), indent=2))
        sys.exit(0)
    n = calibrate(force=args.force, verbose=True)
    print(f"auto_threads() = {n}")
