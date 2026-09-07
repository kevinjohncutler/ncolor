"""An engine whose threads cannot be created raises; it does not abort.

Measured before the fix: asking for a fifth engine under a memory cap,
or for any engine with more threads than the system will give, ended
the Python process with ``terminate called without an active exception``
and a core dump. The pool's constructor threw partway through creating
its workers, and the workers already running were destroyed joinable.

The subprocess is the assertion: a crash is a signal, not an exception,
so it has to be observed from outside. On Linux the subprocess caps its
own address space first, so thread creation fails after a few threads
rather than after the hundred thousand the kernel would otherwise allow
(that took 95 s on an 8-core box); macOS refuses early on its own, and
neither it nor Windows enforces the cap, so there the count does the
work by itself.
"""
import subprocess
import sys
import textwrap

# Runs first in every subprocess: a cap that Linux enforces, with
# OpenBLAS pinned so numpy still imports under it.
_CAP = textwrap.dedent(
    """
    import os, sys
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if sys.platform == "linux":
        import resource
        cap = 1_200_000_000
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    """
)

_ASK_FOR_TOO_MANY = _CAP + textwrap.dedent(
    """
    import ncolor
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
        ncolor.label(a)
        # Make the narrow engines impossible to build, then ask for them.
        _engines._max_engines_cached = 4
        _engines._narrow_threads = lambda: 10_000_000
        errs = []
        def work():
            try:
                for _ in range(3):
                    out, conflicts = ncolor.label(a, return_conflicts=True)
                    if conflicts:
                        errs.append("conflicts")
            except BaseException as e:
                errs.append(repr(e))
        ts = [threading.Thread(target=work) for _ in range(4)]
        for t in ts: t.start()
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
