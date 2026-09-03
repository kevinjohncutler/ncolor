"""Process-wide engine singletons and the lock that serializes them.

The C++ engines (``Solver`` for label / connect / color_graph,
``ExpandEngine`` for expand_labels / format_labels) each keep persistent
scratch buffers and run on one shared thread pool. Neither may be entered
from two Python threads at once: a concurrent call used to corrupt the
pool and crash the interpreter. Every public entry point therefore takes
``LOCK`` around its engine call. The lock only prevents *overlap* between
calls; the parallelism inside a call is unaffected. It is reentrant so a
wrapper that calls another wrapper on the same thread cannot deadlock.

The engines are created on first use, so a bare ``import ncolor`` starts
no threads.
"""
from __future__ import annotations

import threading

LOCK = threading.RLock()
_SOLVER = None
_EXPAND = None


def solver():
    """The process-wide ``Solver`` (label / connect / color_graph)."""
    global _SOLVER
    if _SOLVER is None:
        with LOCK:                          # guard the check-then-set race
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
    """Free the scratch memory the engines keep between calls.

    ncolor keeps the working set of the largest image it has processed
    (about 22 bytes per pixel for :func:`ncolor.label`, 16 for
    :func:`ncolor.expand_labels`) allocated between calls, so repeated
    calls on same-sized images never pay for allocation. After a single
    very large image that is gigabytes of resident memory. Call this to
    give it back; the next call simply reallocates. The thread pool is
    kept, so there is no start-up cost afterwards.
    """
    with LOCK:
        for engine in (_SOLVER, _EXPAND):
            if engine is not None:
                engine.release()
