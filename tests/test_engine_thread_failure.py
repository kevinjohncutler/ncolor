"""An engine whose threads cannot be created raises; it does not abort.

Measured before the fix: asking for a fifth engine under a memory cap,
or for any engine with more threads than the system will give, ended
the Python process with ``terminate called without an active exception``
and a core dump. The pool's constructor threw partway through creating
its workers, and the workers already running were destroyed joinable.

The subprocess is the assertion: a crash is a signal, not an exception,
so it has to be observed from outside.

Two ways to make thread creation fail. ``NCOLOR_TEST_FAIL_THREAD_AT=N``
makes the pool's Nth worker fail to start, which exercises the same
partial-construction path as a real failure, instantly, on every
platform; that is what the tests below use. Exhausting the operating
system for real is kept as one extra Linux-only check, because Linux
can be made to run out in milliseconds under an address-space cap.
Windows cannot: it refuses new threads only once kernel resources are
gone, which took 97 s on an 8-vCPU VM and more than 400 s on the
2-vCPU CI runner.
"""
import os
import subprocess
import sys
import textwrap

import pytest

_PRELUDE = textwrap.dedent(
    """
    import os, sys
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    """
)


def _run(code, env_extra=None, timeout=120):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", _PRELUDE + textwrap.dedent(code)],
                          capture_output=True, timeout=timeout, env=env)


def _explain(proc):
    return (f"returncode={proc.returncode} (negative or 134 => the process died)\n"
            f"stdout: {proc.stdout.decode(errors='replace')[-800:]}\n"
            f"stderr: {proc.stderr.decode(errors='replace')[-800:]}")


_ASK_AND_REPORT = """
    import ncolor
    try:
        ncolor.Engine(n_threads=8)
    except (MemoryError, RuntimeError) as e:
        print("raised", type(e).__name__)
        sys.exit(0)
    print("created")
    sys.exit(2)
    """


@pytest.mark.parametrize("fail_at", [0, 1, 3])
def test_thread_failure_raises_instead_of_aborting(fail_at):
    """Worker 0, 1 or 3 of eight fails to start: the workers already
    running must be released and joined, and the caller gets an
    exception."""
    proc = _run(_ASK_AND_REPORT, {"NCOLOR_TEST_FAIL_THREAD_AT": str(fail_at)})
    assert proc.returncode == 0, _explain(proc)
    assert b"raised" in proc.stdout


def test_module_functions_keep_working_after_a_failed_engine():
    """The pool falls back to taking turns rather than failing calls.

    The injection index is the full-width engine's thread count, which
    its pool never reaches (it creates workers 0 to count-2), and the
    narrow engines are made wider than that so each of them does. Set
    at process start rather than mid-process: on Windows, os.environ
    assignment updates the Win32 environment block but not the C
    runtime's copy that std::getenv reads, so a variable set after
    import is invisible to the extension there.

    The narrow engine is requested by calling the pool's bind step
    directly. Relying on four quick calls to overlap so that the pool
    would ask for one on its own passed by timing luck and failed
    inside the full suite on Windows.
    """
    from ncolor._backend import _smt
    wide = _smt.auto_threads()
    code = """
        import threading
        import numpy as np
        import ncolor
        from ncolor import _engines
        a = np.zeros((96, 96), np.int32); a[4:40, 4:40] = 1; a[50:90, 50:90] = 2
        ncolor.label(a)                 # the full-width engine now exists
        _engines._max_engines_cached = 4
        _engines._narrow_threads = lambda: %d
        # The first bind is a lone caller; the second is the first
        # concurrent one and asks for a narrow engine, which must fail
        # and be remembered, not raise.
        _engines._bind(); _engines._bind()
        if not _engines._cannot_grow:
            print("narrow engine was not attempted, or did not fail")
            raise SystemExit(3)
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
        raise SystemExit(1 if errs else 0)
        """ % (wide + 8)
    proc = _run(code, {"NCOLOR_TEST_FAIL_THREAD_AT": str(wide)})
    assert proc.returncode == 0, _explain(proc)


@pytest.mark.skipif(sys.platform != "linux",
                    reason="only Linux can be made to run out of threads quickly")
def test_real_exhaustion_raises_instead_of_aborting():
    """The genuine operating-system failure, under an address-space cap
    relative to what the process already holds."""
    code = """
        import resource
        import ncolor
        with open("/proc/self/status") as f:
            vm_kb = next(int(l.split()[1]) for l in f if l.startswith("VmSize:"))
        cap = vm_kb * 1024 + 160 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        try:
            ncolor.Engine(n_threads=10_000_000)
        except (MemoryError, RuntimeError) as e:
            print("raised", type(e).__name__)
            sys.exit(0)
        print("created")
        sys.exit(2)
        """
    proc = _run(code)
    assert proc.returncode == 0, _explain(proc)
    assert b"raised" in proc.stdout
