"""``Engine`` instances run concurrently; the default engine serializes.

Only one call may be in flight per thread pool, so the module-level
functions, which share one, take turns. That is right for one image at a
time and it is what kept concurrent calls from corrupting the pool, but
it means threading several images through the shared engine buys
nothing. An ``Engine`` holds its own pool and buffers, so several of them
work at once.

The timing check here is deliberately loose: it asks only that per-engine
threading beats shared-engine threading, which is a structural property
(one lock versus several), not a tuned speedup.
"""
import threading
import time

import numpy as np
import pytest

import ncolor


def _image(seed, n=384, k=250):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), np.int32)
    for i, (y, x) in enumerate(zip(rng.integers(0, n, k), rng.integers(0, n, k)), 1):
        a[max(0, y - 6):y + 6, max(0, x - 6):x + 6] = i
    return a


def test_engine_matches_the_module_level_functions():
    """The deterministic operations agree exactly.

    ``label`` is left out: the picker races several attempts under a time
    budget and takes the first that lands, so two engines with different
    thread counts can return different, equally valid colorings. What it
    owes is checked in the next test.
    """
    eng = ncolor.Engine(n_threads=2)
    for seed in range(3):
        m = _image(seed)
        assert np.array_equal(eng.expand_labels(m), ncolor.expand_labels(m))
        assert np.array_equal(eng.format_labels(m * 3), ncolor.format_labels(m * 3))
        assert np.array_equal(eng.connect(m), ncolor.connect(m))


def test_engine_colorings_are_valid():
    """Same foreground, no conflicts, colors dense from 1."""
    eng = ncolor.Engine(n_threads=2)
    for seed in range(3):
        m = _image(seed)
        out, n, conflicts = eng.label(m, return_n=True, return_conflicts=True)
        assert conflicts == 0
        assert ((out != 0) == (np.asarray(ncolor.label(m)) != 0)).all()
        present = sorted(set(np.asarray(out).ravel().tolist()) - {0})
        assert present == list(range(1, n + 1))
    colors, n = eng.color_graph([[0, 1], [1, 2]], 3, return_n=True)
    assert colors[0] != colors[1] and colors[1] != colors[2]


def test_engine_reports_its_thread_count():
    eng = ncolor.Engine(n_threads=3)
    assert eng.n_threads == 3
    assert "3" in repr(eng)
    assert ncolor.Engine(n_threads=1).n_threads == 1


def test_engines_are_independent():
    """One engine's buffers and last-call state do not touch another's."""
    a, b = ncolor.Engine(n_threads=2), ncolor.Engine(n_threads=2)
    big, small = _image(0, n=384), _image(1, n=64, k=20)
    ref_small = b.label(small)
    a.label(big)
    assert np.array_equal(b.label(small), ref_small)
    a.release_buffers()                       # must not disturb b
    assert np.array_equal(b.label(small), ref_small)


def test_many_threads_on_one_engine_are_serialized_not_corrupted():
    """Sharing an Engine across threads is allowed; calls take turns."""
    eng = ncolor.Engine(n_threads=2)
    images = [_image(s) for s in range(4)]
    refs = [eng.label(m) for m in images]
    errors = []

    def work(i):
        try:
            for _ in range(4):
                if not np.array_equal(eng.label(images[i]), refs[i]):
                    errors.append(i)
        except BaseException as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


def _elapsed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


@pytest.mark.parametrize("n_threads", [4])
def test_separate_engines_overlap_where_a_shared_one_cannot(n_threads):
    images = [_image(s) for s in range(n_threads)]
    reps = 3
    engines = [ncolor.Engine(n_threads=2) for _ in range(n_threads)]

    for m in images:                       # warm both paths
        ncolor.label(m)
    for e, m in zip(engines, images):
        e.label(m)

    def run(call):
        ts = [threading.Thread(target=call, args=(i,)) for i in range(n_threads)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    shared = _elapsed(lambda: run(
        lambda i: [ncolor.label(images[i]) for _ in range(reps)]))
    private = _elapsed(lambda: run(
        lambda i: [engines[i].label(images[i]) for _ in range(reps)]))

    # Structural, not tuned: separate pools overlap, one pool cannot.
    assert private < shared, (
        f"per-engine threading ({private*1e3:.1f} ms) should beat the shared "
        f"engine ({shared*1e3:.1f} ms)")
