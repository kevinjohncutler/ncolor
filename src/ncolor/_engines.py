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
def _max_engines():
    try:
        n = int(os.environ.get("NCOLOR_MAX_ENGINES", "4"))
    except ValueError:
        n = 4
    return max(1, n)


_LOCK = threading.Lock()
_all = []            # every engine created, in creation order
_narrow = []         # the engines overlapping calls share
_bound = 0           # engines handed to threads so far
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


# Whether spreading overlapping calls across narrower engines is worth
# it depends on the machine, and not in a way that reads off the core
# count: it comes down to whether one call already uses the whole machine
# well enough that splitting only adds per-call overhead and multiplies
# the working set. Measured with four threads against the same work in
# sequence, it was 1.3-1.9x on an 18-core M5 Max and 1.3x on a 64-core
# Threadripper, but 0.9x on a 16-core Ryzen and 0.8x on an 8-core i9.
#
# So it is measured rather than guessed, the same way the thread count
# itself is (see ncolor._backend._smt): the first time calls actually
# overlap, time the two arrangements against each other on a synthetic
# mask and keep the answer in the calibration cache, keyed by host and
# CPU. About a quarter of a second, once per machine, and only for a
# program that threads at all. NCOLOR_AUTO_THREADS=0/1 skips the
# measurement and forces the answer.
# Splitting has to be a clear win, not a marginal one. The verdict is
# cached for the life of the machine, so where the two arrangements are
# close the safe answer is the one that changes nothing, and a machine
# that is genuinely on the fence will measure either way from run to
# run: a 16-core Ryzen flipped at both a 5% and a 10% margin. A fifth
# faster is past the noise, and a machine that clears it (an 18-core M5
# Max at 1.9x, a 64-core Threadripper at 1.3x) clears it every time.
_SPLIT_MARGIN = 0.80
_SPLIT_REPS = 2          # calls per engine per round, as real use would
_SPLIT_ROUNDS = 3
_split_cached = None
_split_lock = threading.Lock()


def _probe_split(mask=None):
    """Time k concurrent narrow calls against k sequential wide ones.

    Returns True if splitting was faster. Measures the arrangement it is
    deciding between rather than a proxy for it: concurrent calls also
    multiply the working set, which timing one call at two widths would
    miss.
    """
    import statistics
    import time
    from ._backend import _smt
    k = _max_engines()
    if k < 2:
        return False
    if mask is None:
        mask = _smt._make_calibration_mask(1024)

    wide = _primary()
    narrow = [Engine(n_threads=_narrow_threads()) for _ in range(k)]
    try:
        for _ in range(2):                       # warm both arrangements
            wide.label(mask)
        for e in narrow:
            e.label(mask)

        def run_narrow():
            ts = [threading.Thread(
                target=lambda e=e: [e.label(mask) for _ in range(_SPLIT_REPS)])
                for e in narrow]
            for th in ts:
                th.start()
            for th in ts:
                th.join()

        run_narrow()                             # warm the concurrent path too
        # Medians, not minima: the concurrent arrangement runs four
        # threads and so has the longer tail, and taking the best of each
        # would flatter it. Rounds alternate so drift affects both.
        serial, split = [], []
        for _ in range(_SPLIT_ROUNDS):
            t0 = time.perf_counter()
            for _ in range(k * _SPLIT_REPS):
                wide.label(mask)
            serial.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            run_narrow()
            split.append(time.perf_counter() - t0)
        return statistics.median(split) < statistics.median(serial) * _SPLIT_MARGIN
    finally:
        for e in narrow:
            e.release_buffers()


def _auto_split():
    """Whether overlapping calls should be spread over narrow engines.

    Worked out once per process: it reads a file, and this is on the path
    of every call.
    """
    global _split_cached
    if _split_cached is not None:
        return _split_cached
    # One thread measures while the others wait for its answer.
    with _split_lock:
        if _split_cached is not None:
            return _split_cached
        forced = os.environ.get("NCOLOR_AUTO_THREADS")
        if forced is not None:
            _split_cached = forced not in ("0", "", "false", "False")
            return _split_cached
        from ._backend import _smt
        key = _smt._cache_key() + "|split"
        cache = _smt._load_cache()
        if key in cache:
            _split_cached = bool(cache[key])
            return _split_cached
        try:
            verdict = _probe_split()
            cache = _smt._load_cache()           # re-read; another process may have written
            cache[key] = bool(verdict)
            _smt._save_cache(cache)
            _split_cached = verdict
        except Exception:                        # noqa: BLE001
            # A machine we could not measure keeps the old behavior.
            _split_cached = False
        return _split_cached


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
    global _concurrent, _bound
    primary = _primary()
    with _LOCK:
        _bound += 1
        if _bound > 1:
            _concurrent = True
        concurrent = _concurrent
    # Only once a second thread has actually turned up is it worth asking
    # whether splitting pays, because the asking costs a measurement. A
    # program that never threads never pays for it. Asked outside the
    # lock: the measurement itself colors images, which needs engines.
    if not concurrent or not _auto_split():
        # Calls take turns on the one engine, as they always have.
        _tls.engine = primary
        return primary
    with _LOCK:
        # Overlapping callers share the narrow engines: one machine's
        # worth of threads between them, however many threads call.
        if len(_narrow) < _max_engines():
            eng = Engine(n_threads=_narrow_threads())
            _narrow.append(eng)
            _all.append(eng)
        else:
            # More callers than engines: they share, and take turns.
            # Round-robin on the number handed out, so threads that
            # arrive later spread over the engines instead of piling
            # onto one (thread ids get recycled, so they cannot be the
            # thing that distributes them).
            eng = _narrow[(_bound - 1) % len(_narrow)]
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
    if (_in_flight == 0 or not _auto_split()) and _all:
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


def release_buffers():
    """Free the scratch memory the engines keep between calls.

    ncolor keeps the working set of the largest image it has processed
    (about 20 bytes per pixel, so 327 MB after one 4096 by 4096 image)
    allocated between calls, so repeated calls on same-sized images never
    pay for allocation. This gives it back for every engine the module
    functions have created; the next call simply reallocates. Thread
    pools are kept, so there is no start-up cost afterwards.
    :class:`Engine` has a method of the same name for its own buffers.
    """
    with _LOCK:
        engines = list(_all)
    for eng in engines:
        eng.release_buffers()


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
        return self.engine

    def __exit__(self, *exc):
        global _in_flight
        _in_flight -= 1
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

    __slots__ = ("_solver", "_expand", "_n_threads")

    def __init__(self, n_threads=-1, *, _pool_group=None):
        from ._backend import ExpandEngine, Solver
        # One pool per engine, shared by its two halves: they are never
        # in a call at the same time, so they need only one between them.
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

    def format_labels(self, labels, **kwargs):
        """As :func:`ncolor.format_labels`, on this engine."""
        from .format import format_labels as _format_labels
        return _format_labels(labels, _engine=self, **kwargs)

    def release_buffers(self):
        """Free this engine's scratch buffers; the pool is kept."""
        self._solver.release()
        self._expand.release()

    def __repr__(self):
        return f"<ncolor.Engine n_threads={self._n_threads}>"
