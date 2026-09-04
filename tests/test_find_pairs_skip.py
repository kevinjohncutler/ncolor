"""The soft-kernel interior skip must not change the pair sets.

The fused hard-plus-soft adjacency scan skips the distance-2 soft
offsets for pixels whose distance-1 neighbors all share their label.
``NCOLOR_NO_INTERIOR_SKIP=1`` turns the skip off; it is read once per
process, so the reference run happens in a subprocess. Hard pairs come
from ``connect``, soft pairs from the engine's ``get_last_soft_pairs``.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import ncolor
from ncolor.color import _get_solver

_CASES = [
    # (shape, seed, wrap)
    ((64, 80), 0, False),
    ((64, 80), 1, True),
    ((7, 200), 2, False),
    ((2, 9), 3, True),        # axis shorter than the radius: wrap must modulo
    ((1, 7), 4, True),
    ((24, 20, 28), 5, False),
    ((16, 16, 16), 6, True),
]


def _image(shape, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros(shape, np.int32)
    n = max(3, int(np.prod(shape)) // 250)
    for lab in range(1, n + 1):
        c = [rng.integers(0, s) for s in shape]
        sl = tuple(slice(ci, min(s, ci + 2 + rng.integers(0, 6))) for ci, s in zip(c, shape))
        m[sl] = lab
    return m


def _pairs(shape, seed, wrap):
    m = _image(shape, seed)
    ncolor.label(m, expand=False, wrap=wrap)
    soft = _get_solver().get_last_soft_pairs()
    hard = ncolor.connect(m)
    return sorted(map(tuple, hard.tolist())), sorted(map(tuple, soft.tolist()))


_REFERENCE = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, sys.argv[1])
    from test_find_pairs_skip import _CASES, _pairs
    print(json.dumps([_pairs(*c) for c in _CASES]))
    """
)


@pytest.fixture(scope="module")
def reference():
    env = dict(os.environ, NCOLOR_NO_INTERIOR_SKIP="1", NCOLOR_NO_CALIBRATE="1")
    proc = subprocess.run([sys.executable, "-c", _REFERENCE, os.path.dirname(__file__)],
                          capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


@pytest.mark.parametrize("idx", range(len(_CASES)))
def test_pair_sets_match_the_unskipped_scan(reference, idx):
    shape, seed, wrap = _CASES[idx]
    hard, soft = _pairs(shape, seed, wrap)
    ref_hard, ref_soft = reference[idx]
    assert hard == [tuple(p) for p in ref_hard]
    assert soft == [tuple(p) for p in ref_soft]
    assert all(1 <= a < b for a, b in soft), "soft pairs are (lo, hi) label ids"


def test_soft_pairs_exclude_hard_pairs():
    m = _image((64, 80), 7)
    ncolor.label(m, expand=False)
    soft = {tuple(p) for p in _get_solver().get_last_soft_pairs().tolist()}
    hard = {tuple(p) for p in ncolor.connect(m).tolist()}
    assert soft and hard
    assert not (soft & hard)


@pytest.mark.parametrize(
    "hard_conn,hard_radius,soft_conn,soft_radius,delta",
    [
        (2, 1, 1, 2, (1, 1)),  # hard diagonal absent from axial soft kernel
        (1, 2, 2, 1, (0, 2)),  # hard radius-2 offset absent from r=1 soft kernel
    ],
)
def test_incomparable_hard_and_soft_kernels_keep_hard_edges(
        hard_conn, hard_radius, soft_conn, soft_radius, delta):
    """A fused scan is valid only when the soft kernel contains the hard one."""
    m = np.zeros((7, 7), np.int32)
    p = (3, 2)
    q = (p[0] + delta[0], p[1] + delta[1])
    m[p] = 1
    m[q] = 2

    lut = ncolor.label(
        m, n=2, expand=False, format_input=False, return_lut=True,
        conn=hard_conn, connect_radius=hard_radius,
        soft_conn=soft_conn, soft_radius=soft_radius,
    )
    assert lut[1] != lut[2]


def test_dual_scan_retries_when_base_hash_table_fills():
    """The fused path must not silently drop hard pairs from a full table."""
    n_labels = 16
    ii, jj = np.triu_indices(n_labels, k=1)
    edges = np.column_stack((ii + 1, jj + 1)).astype(np.int32)

    # Every label pair appears as an isolated horizontal contact. There are
    # 120 hard edges, more than the dual path's initial 64-slot base table.
    # Blank rows prevent contacts between neighboring encoded pairs.
    m = np.zeros((2 * len(edges), 3), np.int32)
    m[::2, :2] = edges

    assert len(ncolor.connect(m, conn=1)) == len(edges)
    lut, n_used, conflicts = ncolor.label(
        m, n=n_labels, expand=False, format_input=False,
        return_lut=True, return_n=True, return_conflicts=True,
    )
    assert conflicts == 0
    assert n_used == n_labels
    assert np.all(lut[edges[:, 0]] != lut[edges[:, 1]])
