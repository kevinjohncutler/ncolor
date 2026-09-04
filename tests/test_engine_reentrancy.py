"""Concurrency: automatic for ``ncolor.label``, explicit with ``Engine``.

Only one call may be in flight per thread pool. A lone caller gets the
full-width engine; callers that overlap are given narrower engines whose
threads add up to about one machine, so ``ncolor.label`` threads without
the caller arranging anything. ``Engine`` is the same thing by hand.

The timing checks are deliberately loose: they ask only that threading
beats doing the images one after another, which is structural, not a
tuned speedup. They use enough work per call to stay clear of noise.
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
def test_explicit_engines_overlap(n_threads):
    """An Engine per thread beats doing the images one after another.

    Skipped where the machine says splitting does not pay: there one call
    already uses every core, and the same is true by hand as
    automatically.
    """
    from ncolor import _engines
    if not _engines._auto_split():
        pytest.skip("machine too small for concurrent calls to pay off")
    images = [_image(s, n=512) for s in range(n_threads)]
    reps = 3
    engines = [ncolor.Engine(n_threads=_engines._narrow_threads())
               for _ in range(n_threads)]
    for e, m in zip(engines, images):       # warm the engines
        e.label(m)

    def sequential():
        for e, m in zip(engines, images):
            for _ in range(reps):
                e.label(m)

    def overlapped():
        ts = [threading.Thread(
            target=lambda e=e, m=m: [e.label(m) for _ in range(reps)])
            for e, m in zip(engines, images)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    sequential(); overlapped()              # warm both arrangements
    one_at_a_time = _elapsed(sequential)
    together = _elapsed(overlapped)
    assert together < one_at_a_time, (
        f"engines run together ({together*1e3:.1f} ms) should beat one at a "
        f"time ({one_at_a_time*1e3:.1f} ms)")


# ------------------------------------------------- the automatic path


def test_plain_label_threads_without_an_engine():
    """``ncolor.label`` from several threads beats doing them in turn.

    Only where splitting is on. Below the machine-size threshold a single
    call already uses every core, so calls take turns as they always
    have and there is nothing to beat.
    """
    from ncolor import _engines
    if not _engines._auto_split():           # also settles the one-time
        pytest.skip("machine too small for concurrent calls to pay off")
    images = [_image(s, n=512) for s in range(4)]
    reps = 4

    def serial():
        for m in images:
            for _ in range(reps):
                ncolor.label(m)

    def threaded():
        ts = [threading.Thread(
            target=lambda m=m: [ncolor.label(m) for _ in range(reps)])
            for m in images]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

    serial()
    threaded()                      # build the per-thread engines first
    t_serial = _elapsed(serial)
    t_threaded = _elapsed(threaded)
    assert t_threaded < t_serial, (
        f"threaded {t_threaded*1e3:.1f} ms should beat serial "
        f"{t_serial*1e3:.1f} ms")


def test_a_lone_caller_gets_the_full_width_engine():
    """Before any concurrency, and again once it is over."""
    from ncolor import _engines
    m = _image(0, n=128)
    ncolor.label(m)
    assert _engines._last_used() is _engines._all[0]

    def work():
        for _ in range(2):
            ncolor.label(m)

    ts = [threading.Thread(target=work) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    # Nothing else running now, so the wide engine comes back.
    ncolor.label(m)
    assert _engines._last_used() is _engines._all[0]


def test_engine_count_is_bounded():
    """However many threads call, the pool stops growing."""
    from ncolor import _engines
    if not _engines._auto_split():
        pytest.skip("no splitting on this machine, so one engine only")
    m = _image(1, n=128)

    def work():
        for _ in range(2):
            ncolor.label(m)

    ts = [threading.Thread(target=work) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    # One full-width engine plus at most NCOLOR_MAX_ENGINES narrow ones.
    assert len(_engines._all) <= 1 + _engines._max_engines()
    assert all(e.n_threads >= 1 for e in _engines._all)
