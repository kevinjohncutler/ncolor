"""One call must not inherit state from the previous one.

The engine is a process-global singleton that reuses its buffers across
calls. Every accessor and every internal list therefore has to be reset
at the start of a call, not only by the branch that happens to fill it.

The soft-pair list was the case that got this wrong: it was cleared by
the branches that build one, so a call taking the weighted or
min-contact branch reused the previous call's pairs. Those are label ids
of a different image; the ones falling inside the new label range were
applied as soft constraints, and on a two-cell image that pushed the
color count from 2 to 4.
"""
import numpy as np
import pytest

import ncolor
from ncolor.color import _get_solver


def _many_cells(side=200, step=8, size=6):
    m = np.zeros((side, side), np.int32)
    k = 1
    for y in range(0, side, step):
        for x in range(0, side, step):
            m[y:y + size, x:x + size] = k
            k += 1
    return m


def _two_cells():
    m = np.zeros((12, 12), np.int32)
    m[1:5, 1:5] = 1
    m[7:11, 7:11] = 2
    return m


def _alone(fn):
    """Result of ``fn`` in a solver that has seen nothing else."""
    ncolor.release_buffers()
    return fn()


# Kwargs that route label() through each branch of the find_pairs chain.
_BRANCHES = {
    "weighted": dict(weight_objective=1),
    "min_contact": dict(min_contact=4, connect_radius=2),
    "default_dual": {},
    "incomparable": dict(conn=2, connect_radius=1, soft_conn=1, soft_radius=2),
    "no_soft": dict(soft_conn=0, soft_radius=0),
}


@pytest.mark.parametrize("name", list(_BRANCHES))
def test_result_does_not_depend_on_the_previous_call(name):
    kwargs = _BRANCHES[name]
    small = _two_cells()

    clean, clean_n = _alone(lambda: ncolor.label(small, return_n=True, **kwargs))

    ncolor.label(_many_cells())          # leaves 600+ cells of state behind
    after, after_n = ncolor.label(small, return_n=True, **kwargs)

    assert after_n == clean_n, f"{name}: color count changed after a prior call"
    assert np.array_equal(after, clean), f"{name}: output changed after a prior call"


@pytest.mark.parametrize("name", list(_BRANCHES))
def test_soft_pairs_never_reference_absent_labels(name):
    ncolor.label(_many_cells())
    small = _two_cells()
    ncolor.label(small, **_BRANCHES[name])
    soft = _get_solver().get_last_soft_pairs()
    if len(soft):
        assert int(soft.max()) <= int(small.max()), "stale label ids from a prior call"


def test_soft_violation_count_is_not_inherited():
    """The reported soft-violation count belongs to the current call."""
    small = _two_cells()

    ncolor.release_buffers()
    ncolor.label(small, weight_objective=1)
    clean = _get_solver().get_last_n_soft_violations()

    ncolor.label(_many_cells())
    ncolor.label(small, weight_objective=1)
    assert _get_solver().get_last_n_soft_violations() == clean
