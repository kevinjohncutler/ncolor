"""The same image gets the same coloring, whatever the thread count.

``label`` races several searches and takes the lowest-numbered one that
succeeds. It used to abandon every other search the moment any of them
landed, which let thread scheduling pick the winner: a low-numbered
attempt that would have won got cut off by a high-numbered one that
happened to finish first, so the same image came back with different
(equally valid) colorings from one call to the next, and only above one
thread. A search is now abandoned only once a lower-numbered one has
won, which cannot change the answer.

One thread is excluded on purpose: with a single thread the picker runs
its sequential path (``color_mode``), which is a different algorithm,
not a different schedule.
"""
import threading

import numpy as np
import pytest

import ncolor


def _image(seed, n=256, k=140):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), np.int32)
    for i, (y, x) in enumerate(zip(rng.integers(0, n, k), rng.integers(0, n, k)), 1):
        a[max(0, y - 7):y + 7, max(0, x - 7):x + 7] = i
    return a


def _disks(seed, n=256, k=200):
    """Round, touching cells: the shape that varied most before the fix."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n), np.int32)
    yy, xx = np.ogrid[:n, :n]
    for i, (cy, cx, r) in enumerate(zip(rng.integers(0, n, k), rng.integers(0, n, k),
                                        rng.integers(4, 12, k)), 1):
        a[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return a


IMAGES = [_image(0), _image(1), _disks(2), _disks(3)]


@pytest.mark.parametrize("n_threads", [2, 4])
def test_same_coloring_every_call(n_threads):
    eng = ncolor.Engine(n_threads=n_threads)
    for m in IMAGES:
        first = np.asarray(eng.label(m))
        for _ in range(6):
            assert np.array_equal(np.asarray(eng.label(m)), first)


def test_same_coloring_at_every_thread_count():
    """Two engines of different widths agree exactly."""
    a, b = ncolor.Engine(n_threads=2), ncolor.Engine(n_threads=4)
    for m in IMAGES:
        assert np.array_equal(np.asarray(a.label(m)), np.asarray(b.label(m)))


def test_same_coloring_while_the_machine_is_busy():
    """Load changes how the searches interleave, not what comes out."""
    eng = ncolor.Engine(n_threads=4)
    refs = [np.asarray(eng.label(m)) for m in IMAGES]
    stop = threading.Event()
    mismatches = []

    def churn():
        other = ncolor.Engine(n_threads=2)
        while not stop.is_set():
            other.label(IMAGES[0])

    busy = [threading.Thread(target=churn, daemon=True) for _ in range(3)]
    for t in busy:
        t.start()
    try:
        for _ in range(4):
            for m, ref in zip(IMAGES, refs):
                if not np.array_equal(np.asarray(eng.label(m)), ref):
                    mismatches.append(int(m.max()))
    finally:
        stop.set()
        for t in busy:
            t.join(timeout=10)
    assert not mismatches
