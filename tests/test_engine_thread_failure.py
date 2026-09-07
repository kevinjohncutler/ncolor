"""An engine whose threads cannot be created raises; it does not abort.

Measured before the fix: asking for a fifth engine under a memory cap,
or for any engine with more threads than the system will give, ended
the Python process with ``terminate called without an active exception``
and a core dump. The pool's constructor threw partway through creating
its workers, and the workers already running were destroyed joinable.

The subprocess is the assertion: a crash is a signal, not an exception,
so it has to be observed from outside. On Linux the subprocess caps its
own address space so thread creation fails after a handful of threads
rather than after the hundred thousand the kernel would otherwise allow
(that took 95 s on an 8-core box). The cap is relative to what the
process already holds when it is applied: a fixed cap that fit an
8-thread machine left a 32-thread one, whose wide engine alone reserves
256 MB of stacks, unable to start the test's own threads. macOS refuses
early on its own and does not enforce the cap; nor does Windows.
"""
import subprocess
import sys
import textwrap

# Defines cap_here(): on Linux, limit the address space to what this
# process holds right now plus headroom for a few more threads and small
# allocations. Called at the point in each script where everything the
# test needs already exists, so only the engine under test runs out.
_CAP = textwrap.dedent(
    """
    import os, sys
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    def cap_here(headroom_mb=160):
        if sys.platform != "linux":
            return
        import resource
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    vm_kb = int(line.split()[1])
                    break
        cap = vm_kb * 1024 + headroom_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    """
)

_ASK_FOR_TOO_MANY = _CAP + textwrap.dedent(
    """
    import ncolor
    cap_here()
    try:
        ncolor.Engine(n_threads=10_000_000)
    except (MemoryError, RuntimeError) as e:
        print("raised", type(e).__name__)
        sys.exit(0)
    print("created")          # no machine has these threads; treat as failure
    sys.exit(2)
    """
)


def test_impossible_thread_count_raises_instead_of_aborting():
    proc = subprocess.run([sys.executable, "-c", _ASK_FOR_TOO_MANY],
                          capture_output=True, timeout=120)
    assert proc.returncode == 0, (
        f"returncode={proc.returncode} (negative or 134 => the process died)\n"
        f"stdout: {proc.stdout.decode(errors='replace')[-800:]}\n"
        f"stderr: {proc.stderr.decode(errors='replace')[-800:]}")
    assert b"raised" in proc.stdout


def test_module_functions_keep_working_after_a_failed_engine():
    """The pool falls back to taking turns rather than failing calls."""
    code = _CAP + textwrap.dedent(
        """
        import threading
        import numpy as np
        import ncolor
        from ncolor import _engines
        a = np.zeros((96, 96), np.int32); a[4:40, 4:40] = 1; a[50:90, 50:90] = 2
        ncolor.label(a)                 # the full-width engine now exists
        # Start the test's own threads before the cap, parked on a gate,
        # so their stacks are already counted; then make the narrow
        # engines impossible to build and let the threads go.
        go = threading.Event()
        errs = []
        def work():
            go.wait()
            try:
                for _ in range(3):
                    out, conflicts = ncolor.label(a, return_conflicts=True)
                    if conflicts:
                        errs.append("conflicts")
            except BaseException as e:
                errs.append(repr(e))
        ts = [threading.Thread(target=work) for _ in range(4)]
        for t in ts: t.start()
        cap_here()
        _engines._max_engines_cached = 4
        _engines._narrow_threads = lambda: 10_000_000
        go.set()
        for t in ts: t.join()
        print("errors:", errs, "cannot_grow:", _engines._cannot_grow)
        raise SystemExit(1 if errs or not _engines._cannot_grow else 0)
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    assert proc.returncode == 0, (
        f"returncode={proc.returncode}\n"
        f"stdout: {proc.stdout.decode(errors='replace')[-800:]}\n"
        f"stderr: {proc.stderr.decode(errors='replace')[-800:]}")
