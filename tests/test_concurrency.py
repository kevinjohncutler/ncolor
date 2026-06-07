"""Thread-safety regression for the public solve API.

The ``Solver`` is a process-global singleton backed by a single C++
``ForkJoinPool``. Before the serialization lock in ``ncolor.color``, two Python
threads entering ``label`` / ``connect`` at once corrupted that shared pool and
hard-crashed the process (``SIGSEGV`` in ``bridge_check_subspace_nd``, the
``expand=True`` path). A single call from any thread was always fine — only
cross-thread *overlap* crashed, so this guards against that overlap.

A segfault kills the whole interpreter (it would take pytest down with it), so
the concurrent stress runs in a SUBPROCESS and the test asserts the child exits
cleanly. A regression therefore shows up as a normal test failure (non-zero
return code), not a crashed test session.
"""
import subprocess
import sys
import textwrap

import numpy as np

import ncolor


# Concurrent stress: N threads each repeatedly color their own label image with
# ``expand=True`` (the path that crashed). Exits 1 on any worker error, else 0.
# A pre-fix interpreter SIGSEGVs here → child return code is negative (signal).
_STRESS = textwrap.dedent(
    """
    import sys, threading
    import numpy as np
    import ncolor

    def make(seed, n=1024, k=800):
        rng = np.random.default_rng(seed)
        a = np.zeros((n, n), np.int32)
        ys = rng.integers(0, n, k); xs = rng.integers(0, n, k)
        for i, (y, x) in enumerate(zip(ys, xs), 1):
            a[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = i
        return a

    THREADS, REPS = 4, 6
    segs = [make(t) for t in range(THREADS)]
    errs = []

    def work(seg):
        try:
            for _ in range(REPS):
                out = np.asarray(ncolor.label(seg, expand=True))
                assert out.shape == seg.shape
                # also exercise connect() (shares the singleton / pool)
                ncolor.connect(seg, conn=1)
        except BaseException as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(s,)) for s in segs]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    sys.exit(1 if errs else 0)
    """
)


def test_concurrent_solve_does_not_crash():
    """``label``/``connect`` must survive concurrent calls from many threads."""
    proc = subprocess.run(
        [sys.executable, "-c", _STRESS],
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        "concurrent ncolor solve crashed or errored "
        f"(returncode={proc.returncode}; negative => killed by signal, "
        "e.g. -11 = SIGSEGV from the unguarded shared ForkJoinPool).\n"
        f"stderr tail:\n{proc.stderr.decode(errors='replace')[-3000:]}"
    )


def test_concurrent_label_matches_serial():
    """Serialized concurrent calls must still produce valid, stable colorings:
    the per-thread results match a plain serial call on the same input."""
    import threading

    arr = np.zeros((12, 14), dtype=np.int32)
    arr[1:11, 1:5] = 1
    arr[1:11, 5:9] = 2
    arr[1:11, 9:13] = 3

    serial = np.asarray(ncolor.label(arr, expand=True))
    results = [None] * 8

    def work(i):
        results[i] = np.asarray(ncolor.label(arr, expand=True))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    for r in results:
        assert r is not None
        assert np.array_equal(r, serial)
