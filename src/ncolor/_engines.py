"""Engine objects, the process-wide default, and the lock that guards it.

The C++ engines (``Solver`` for label / connect / color_graph,
``ExpandEngine`` for expand_labels / format_labels) each keep persistent
scratch buffers and run on a thread pool. Only one call may be in flight
per pool, so each engine takes its pool's mutex for the duration of a
call.

The module-level functions share one default pair of engines, and so one
pool: calls from several threads take turns, which is what the pool
requires and what a single-image-at-a-time caller wants. Processing
several images at once instead wants an :class:`Engine` per thread; each
holds a pool and buffers of its own and so runs alongside the others.

The engines are created on first use, so a bare ``import ncolor`` starts
no threads.
"""
from __future__ import annotations

import threading

# Guards creation of the default engines. The C++ side serializes the
# calls themselves, so this is only for the check-then-set below.
LOCK = threading.RLock()
_SOLVER = None
_EXPAND = None


def solver():
    """The process-wide ``Solver`` (label / connect / color_graph)."""
    global _SOLVER
    if _SOLVER is None:
        with LOCK:
            if _SOLVER is None:
                from ._backend import Solver
                _SOLVER = Solver()
    return _SOLVER


def expand_engine():
    """The process-wide ``ExpandEngine`` (expand_labels / format_labels)."""
    global _EXPAND
    if _EXPAND is None:
        with LOCK:
            if _EXPAND is None:
                from ._backend import ExpandEngine
                _EXPAND = ExpandEngine()
    return _EXPAND


def release_buffers():
    """Free the scratch memory the default engines keep between calls.

    ncolor keeps the working set of the largest image it has processed
    (about 20 bytes per pixel, so 327 MB after one 4096 by 4096 image)
    allocated between calls, so repeated calls on same-sized images never
    pay for allocation. Call this to give it back; the next call simply
    reallocates. The thread pool is kept, so there is no start-up cost
    afterwards. :class:`Engine` has a method of the same name for its own
    buffers.
    """
    with LOCK:
        for engine in (_SOLVER, _EXPAND):
            if engine is not None:
                engine.release()


class Engine:
    """An independent engine, for coloring several images at once.

    The module-level functions all share one thread pool, so calls from
    different threads queue behind each other. That is the right
    arrangement for one image at a time, where a single call already uses
    every core. To work on several images concurrently, give each thread
    an ``Engine``: it holds its own pool and its own scratch buffers, so
    it runs alongside the others.

        import concurrent.futures as cf, ncolor

        def run(image, engine=None):
            return (engine or ncolor.Engine(n_threads=4)).label(image)

        engines = [ncolor.Engine(n_threads=4) for _ in range(4)]
        with cf.ThreadPoolExecutor(4) as pool:
            out = list(pool.map(lambda a: a[0].label(a[1]),
                                zip(engines, images)))

    Two things to size against each other. Threads: engines do not know
    about one another, so ``n_engines * n_threads`` should be about the
    core count, not a multiple of it. Memory: each engine keeps the
    working set of the largest image *it* has seen, roughly 20 bytes per
    pixel, so four engines on 4096 by 4096 images hold about 1.3 GB
    between them. :meth:`release_buffers` gives one engine's share back.

    An ``Engine`` is safe to call from several threads; those calls take
    turns, exactly as the module-level functions do.

    Parameters
    ----------
    n_threads : int or float, optional
        Threads for this engine's pool. ``-1`` (default) uses the
        calibrated count for the machine, which is usually every core and
        so too many if several engines are to run at once. A fraction
        between 0 and 1 is that share of the core count.
    """

    __slots__ = ("_solver", "_expand", "_n_threads")

    def __init__(self, n_threads=-1):
        from ._backend import ExpandEngine, Solver
        self._solver = Solver(n_threads, private_pool=True)
        self._expand = ExpandEngine(n_threads, private_pool=True)
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


# ---- internals used by the wrappers ---------------------------------------

def _solver_for(engine):
    """The Solver to use: an Engine's own, or the process-wide default."""
    return solver() if engine is None else engine._solver


def _expand_for(engine):
    """The ExpandEngine to use: an Engine's own, or the default."""
    return expand_engine() if engine is None else engine._expand
