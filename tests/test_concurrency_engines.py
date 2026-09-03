"""Thread-safety of every engine-backed entry point, not just ``label``.

``test_concurrency.py`` covers ``label`` / ``connect``. The expand and
format engines were left unlocked when that fix landed, and concurrent
``expand_labels`` / ``format_labels`` calls crashed the interpreter the
same way. As there, the stress runs in a subprocess so a crash is a
return code rather than a dead test session.

Outputs are checked for validity, not equality with a serial run: the
picker races time-budgeted searches, so two serial ``label`` calls on a
large enough input can legitimately return different valid colorings.
"""
import subprocess
import sys
import textwrap

import numpy as np

import ncolor

_STRESS = textwrap.dedent(
    """
    import sys, threading
    import numpy as np
    import ncolor

    def make(seed, n=640, k=400):
        rng = np.random.default_rng(seed)
        a = np.zeros((n, n), np.int32)
        for i, (y, x) in enumerate(zip(rng.integers(0, n, k), rng.integers(0, n, k)), 1):
            a[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = i
        return a

    THREADS, REPS = 4, 6
    segs = [make(t) for t in range(THREADS)]
    ref_expand = [np.asarray(ncolor.expand_labels(s)) for s in segs]
    ref_format = [np.asarray(ncolor.format_labels(s * 3)) for s in segs]
    errs = []

    def work(i):
        try:
            for _ in range(REPS):
                if not np.array_equal(ncolor.expand_labels(segs[i]), ref_expand[i]):
                    errs.append(("expand_labels", i))
                if not np.array_equal(ncolor.format_labels(segs[i] * 3), ref_format[i]):
                    errs.append(("format_labels", i))
                out, conflicts = ncolor.label(segs[i], return_conflicts=True)
                if conflicts or (out != 0).sum() != (segs[i] != 0).sum():
                    errs.append(("label", i))
                ncolor.connect(segs[i])
                ncolor.connected_components(segs[i] > 0)
                ncolor.color_graph([[0, 1], [1, 2]], 3)
        except BaseException as e:  # noqa: BLE001
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(THREADS)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("errors:", errs[:5])
    sys.exit(1 if errs else 0)
    """
)


def test_concurrent_engine_calls_neither_crash_nor_corrupt():
    proc = subprocess.run([sys.executable, "-c", _STRESS],
                          capture_output=True, timeout=300)
    assert proc.returncode == 0, (
        f"returncode={proc.returncode} (negative => killed by a signal)\n"
        f"stdout: {proc.stdout.decode(errors='replace')[-1500:]}\n"
        f"stderr: {proc.stderr.decode(errors='replace')[-1500:]}"
    )


def test_engines_share_one_thread_pool():
    """Both engines resolve to the same thread count and so share a pool;
    touching every entry point must not spawn a second set of workers."""
    from ncolor import _engines
    base = np.zeros((32, 32), np.int32)
    base[4:12, 4:12] = 1
    base[20:28, 20:28] = 2
    ncolor.label(base)
    ncolor.expand_labels(base)
    ncolor.format_labels(base)
    assert _engines._SOLVER.n_threads == _engines._EXPAND.n_threads
