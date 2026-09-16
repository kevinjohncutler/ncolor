"""Engines, and the pool the module-level functions draw from.

The C++ engines (``Solver`` for label / connect / color_graph,
``ExpandEngine`` for expand_labels / format_labels) each keep persistent
scratch buffers and run on a thread pool. Only one call may be in flight
per pool, so a call holds its engine for its duration.

A call made while nothing else is running gets the full-width engine,
which shares its pool the way the package always has: one image at a
time is unchanged. Calls that overlap instead share a set of narrower
engines whose threads add up to about one machine, so several images
color at once without the pools fighting over the cores. Running one
wide engine alongside the narrow ones is the worst of both, and measured
slower than doing the images one after another, so the wide one sits out
whenever anything else is in flight.

Overlapping callers always get their own engines. Whether that beats
taking turns was measured across four machines and eight image sizes,
and it varies far more with the image than with the machine: on an
8-core i9 the same arrangement ran 1.3x faster at 512 by 512, 0.8x at
1024, and 1.1x at 4096. No fact about the machine predicts that, so
there is nothing worth measuring once and caching. Averaged over sizes
it pays on every machine tested, from 1.07x on that i9 to 1.9x on a
64-core Threadripper, and the worst single case is 0.76x. Set
``NCOLOR_MAX_ENGINES=1`` to turn it off and have overlapping calls take
turns on one pool, as they did before there was more than one.

The narrow engines are built only when calls actually overlap, so a
single-threaded program holds exactly one engine. There are at most
``NCOLOR_MAX_ENGINES`` of them (4 by default); past that callers wait,
as they used to. That bounds the worker threads and the memory alike,
since each engine keeps the scratch of the largest image it has seen.

:class:`Engine` is the same thing by hand, for callers who would rather
hold one per worker than share this pool.
"""
from __future__ import annotations

import os
import threading

# Engines the module-level functions may create. Past this a call waits
# for a free one rather than adding more; each engine also costs the
# scratch of the largest image it sees (about 20 bytes per pixel), so
# this bounds memory as much as it bounds threads.
_max_engines_cached = None


def _max_engines():
    global _max_engines_cached
    if _max_engines_cached is None:
        try:
            n = int(os.environ.get("NCOLOR_MAX_ENGINES", "4"))
        except ValueError:
            n = 4
        _max_engines_cached = max(1, n)
    return _max_engines_cached


_LOCK = threading.Lock()
_all = []            # every engine created, in creation order
_narrow = []         # the engines overlapping calls share
_bound = 0           # engines handed to threads so far
_cannot_grow = False # a narrow engine could not be built; stop trying
_in_flight = 0       # calls running right now
_concurrent = False  # has more than one thread ever called?
_next_group = 1      # pool group 0 is the process-wide default
_tls = threading.local()

# Kept for callers that reach for the lock directly.
LOCK = _LOCK


def _new_group():
    """A pool group nobody else uses."""
    global _next_group
    g = _next_group
    _next_group += 1
    return g


def _narrow_threads():
    """Thread count for the engines that overlapping calls share.

    They divide one machine between them, so all of them running at once
    comes to about the core count rather than a multiple of it. Measured
    on an 18-core machine, four threads coloring 1024 by 1024 images:
    sharing a machine this way ran 2.0x faster than doing them one after
    another, while adding the full-width engine to the mix (18 + 3 x 4
    threads on 18 cores) was *slower* than serial, because the pools
    spin against each other.
    """
    from ._backend import _smt
    return max(1, _smt.auto_threads() // _max_engines())


def _splits():
    """Whether overlapping calls get engines of their own at all.

    One engine means they take turns on the full-width pool, which is
    what ncolor did before it had more than one.
    """
    return _max_engines() > 1


def _primary():
    """The full-width engine, created on first use."""
    with _LOCK:
        if not _all:
            _all.append(Engine(_pool_group=0))
        return _all[0]


def solver():
    """The process-wide ``Solver`` (label / connect / color_graph)."""
    return _primary()._solver


def expand_engine():
    """The process-wide ``ExpandEngine`` (expand_labels / format_labels)."""
    return _primary()._expand


def _bind():
    """Give the calling thread an engine, and remember it.

    Taken once per thread, and again for the thread holding the
    full-width engine when a second thread turns up. Everything after
    that is a thread-local read, because taking a lock per call cost more
    than the call: four threads handing one lock back and forth measured
    2.3 ms of overhead per call, several times the work itself.
    """
    global _concurrent, _bound, _cannot_grow
    primary = _primary()
    with _LOCK:
        _bound += 1
        if _bound > 1:
            _concurrent = True
        concurrent = _concurrent
    if not concurrent or not _splits():
        # Calls take turns on the one engine, as they always have.
        _tls.engine = primary
        return primary
    with _LOCK:
        # Overlapping callers share the narrow engines: one machine's
        # worth of threads between them, however many threads call.
        if len(_narrow) < _max_engines() and not _cannot_grow:
            try:
                eng = Engine(n_threads=_narrow_threads())
            except (MemoryError, RuntimeError):
                # The machine would not give us another engine (out of
                # address space, or out of threads). Stop asking, and
                # let this caller take turns on whatever exists: the
                # full-width engine if nothing narrow was built yet.
                # Slower, not broken.
                _cannot_grow = True
                eng = _narrow[(_bound - 1) % len(_narrow)] if _narrow else primary
            else:
                _narrow.append(eng)
                _all.append(eng)
        else:
            # More callers than engines: they share, and take turns.
            # Round-robin on the number handed out, so threads that
            # arrive later spread over the engines instead of piling
            # onto one (thread ids get recycled, so they cannot be the
            # thing that distributes them). If no narrow engine could
            # ever be built, everyone takes turns on the wide one.
            eng = _narrow[(_bound - 1) % len(_narrow)] if _narrow else primary
        _tls.engine = eng
        return eng


def _borrow():
    """The engine this thread should use. No lock on the steady path.

    Nothing else running means the full-width engine, whichever thread
    asks, so a program that finishes its parallel phase goes back to full
    speed. While calls overlap, everyone uses the narrow engines: a wide
    pool running alongside them fights for the same cores and measured
    slower than doing the images one after another.
    """
    if (_in_flight == 0 or not _splits()) and _all:
        eng = _all[0]
        _tls.last = eng
        return eng
    eng = getattr(_tls, "engine", None)
    if eng is None or eng is (_all[0] if _all else None):
        eng = _bind()
    _tls.last = eng
    return eng


def _last_used():
    """The engine that ran this thread's most recent call."""
    return getattr(_tls, "last", None) or _primary()


class _Borrowed:
    """Yields the engine a call runs on."""

    __slots__ = ("engine",)

    def __init__(self, engine=None):
        self.engine = engine

    def __enter__(self):
        global _in_flight
        if self.engine is None:
            self.engine = _borrow()
        # Plain increments: a lock here cost more than the call itself,
        # and a miscount only means one call picks the other engine.
        _in_flight += 1
        try:
            self.engine._call_lock.acquire()
        except BaseException:
            _in_flight -= 1
            raise
        return self.engine

    def __exit__(self, *exc):
        global _in_flight
        _in_flight -= 1
        self.engine._call_lock.release()
        return False


def _use(engine):
    """Context manager yielding the engine a call should run on.

    ``engine`` is an explicit :class:`Engine` to use as-is, or None to
    borrow one from the pool.
    """
    return _Borrowed(engine)


def release_buffers():
    """Free the scratch memory the engines keep between calls.

    ncolor keeps the working set of the largest image it has processed
    (about 20 bytes per pixel, so 327 MB after one 4096 by 4096 image)
    allocated between calls, so repeated calls on same-sized images never
    pay for allocation. This gives it back for every engine the module
    functions have created; the next call simply reallocates. Thread
    pools are kept, so there is no start-up cost afterwards.
    :class:`Engine` has a method of the same name for its own buffers.

    On macOS the process's resident size may not fall afterwards, and
    that is not this function failing: the allocator there returns
    freed pages lazily, only once the system wants them, and the same
    is true of deleting a large NumPy array. The memory is free and the
    next allocation reuses it. Measured on a 4096 by 4096 image with
    five engines: on Linux the resident size fell from 2056 MB to
    287 MB; on macOS it stayed at 2068 MB while a fresh engine's next
    call cost no new memory at all.
    """
    with _LOCK:
        engines = list(_all)
    for eng in engines:
        eng.release_buffers()


class Engine:
    """An engine of one's own, with its own pool and scratch buffers.

    The module-level functions already spread concurrent calls over a
    small pool of these, so most callers never need one. Hold one per
    worker when the work is long-lived and you would rather size the
    threads yourself than share the pool, or when more than
    ``NCOLOR_MAX_ENGINES`` workers should run at once.

        engines = [ncolor.Engine(n_threads=4) for _ in range(4)]
        with ThreadPoolExecutor(4) as pool:
            out = list(pool.map(lambda a: a[0].label(a[1]),
                                zip(engines, images)))

    Two things to size against each other. Threads: engines do not know
    about one another, so ``n_engines * n_threads`` should be about the
    core count. Memory: each engine keeps the working set of the largest
    image *it* has seen, roughly 20 bytes per pixel, so four engines on
    4096 by 4096 images hold about 1.3 GB between them.
    :meth:`release_buffers` gives one engine's share back.

    An ``Engine`` is safe to call from several threads; those calls take
    turns.

    Parameters
    ----------
    n_threads : int or float, optional
        Threads for this engine's pool. ``-1`` (default) uses the
        calibrated count for the machine, which is usually every core and
        so too many if several engines are to run at once. A fraction
        between 0 and 1 is that share of the core count.
    """

    __slots__ = ("_solver", "_expand", "_n_threads", "_call_lock")

    def __init__(self, n_threads=-1, *, _pool_group=None):
        from ._backend import ExpandEngine, Solver
        # One pool per engine, shared by its two halves: they are never
        # in a call at the same time, so they need only one between them.
        self._call_lock = threading.RLock()
        group = _new_group() if _pool_group is None else _pool_group
        self._solver = Solver(n_threads, pool_group=group)
        self._expand = ExpandEngine(n_threads, pool_group=group)
        self._n_threads = self._solver.n_threads

    @property
    def n_threads(self):
        """Threads in this engine's pool."""
        return self._n_threads

    def label(self, lab, **kwargs):
        """As :func:`ncolor.label`, on this engine."""
        from .color import label as _label
        return _label(lab, _engine=self, **kwargs)

    def connect(self, img, conn=1):
        """As :func:`ncolor.connect`, on this engine."""
        from .color import connect as _connect
        return _connect(img, conn=conn, _engine=self)

    def color_graph(self, edges, n_vertices=None, **kwargs):
        """As :func:`ncolor.color_graph`, on this engine."""
        from .color import color_graph as _color_graph
        return _color_graph(edges, n_vertices, _engine=self, **kwargs)

    def expand_labels(self, label_image, **kwargs):
        """As :func:`ncolor.expand_labels`, on this engine."""
        from .expand import expand_labels as _expand_labels
        return _expand_labels(label_image, _engine=self, **kwargs)

    def prepare_labels(self, lab, **kwargs):
        """As :func:`ncolor.prepare_labels`, using this engine's workers."""
        from .prepared import prepare_labels
        return prepare_labels(lab, _engine=self, **kwargs)

    def connected_components(self, mask, conn=None):
        """As :func:`ncolor.connected_components`, using this engine's pool."""
        from .color import connected_components
        return connected_components(mask, conn=conn, _engine=self)

    def format_labels(self, labels, **kwargs):
        """As :func:`ncolor.format_labels`, on this engine."""
        from .format import format_labels as _format_labels
        return _format_labels(labels, _engine=self, **kwargs)

    def delete_spurs(self, arr, **kwargs):
        """As :func:`ncolor.delete_spurs`, using this engine's pool."""
        from .format import delete_spurs as _delete_spurs
        return _delete_spurs(arr, _engine=self, **kwargs)

    def release_buffers(self):
        """Free this engine's scratch buffers; the pool is kept."""
        with self._call_lock:
            self._solver.release()
            self._expand.release()

    def __repr__(self):
        return f"<ncolor.Engine n_threads={self._n_threads}>"
