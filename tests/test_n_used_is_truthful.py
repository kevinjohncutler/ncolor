"""``n_used`` is the number of colors used, and the values are dense.

It used to be the largest color index. Those agree only when the
coloring happens to use every index: the weighted objective picks
widely separated palette entries and leaves the rest unused, so a
two-cell image colored 1 and 4 reported four colors. The soft
post-pass can vacate a color too.

Both halves of the contract are checked here: the reported count is the
number of distinct colors present, and those colors are exactly
``1..n_used`` so a caller can normalize an image by the count.
"""
import numpy as np
import pytest

import ncolor


def _two_cells():
    m = np.zeros((12, 12), np.int32)
    m[1:5, 1:5] = 1
    m[7:11, 7:11] = 2
    return m


def _grid(side=64, step=8, size=6):
    m = np.zeros((side, side), np.int32)
    k = 1
    for y in range(0, side, step):
        for x in range(0, side, step):
            m[y:y + size, x:x + size] = k
            k += 1
    return m


def _blobs_3d(d=48):
    rng = np.random.default_rng(0)
    m = np.zeros((d, d, d), np.int32)
    for lab in range(1, 40):
        c = rng.integers(4, d - 4, size=3)
        m[c[0] - 3:c[0] + 3, c[1] - 3:c[1] + 3, c[2] - 3:c[2] + 3] = lab
    return m


_KWARGS = [
    {},
    dict(weight_objective=1),
    dict(weight_objective=-1),
    dict(weight_objective=1, weight_mode="harmonic"),
    dict(soft_conn=0, soft_radius=0),
    dict(min_contact=4, connect_radius=2),
    dict(n=5),
    dict(n=8),
    dict(expand=False),
]


def _images():
    return [("two_cells", _two_cells()), ("grid", _grid()), ("blobs_3d", _blobs_3d())]


@pytest.mark.parametrize("kwargs", _KWARGS, ids=lambda k: str(sorted(k)) or "default")
@pytest.mark.parametrize("name,image", _images(), ids=lambda v: v if isinstance(v, str) else "")
def test_reported_count_matches_the_colors_present(name, image, kwargs):
    out, n = ncolor.label(image, return_n=True, **kwargs)
    present = sorted(set(np.asarray(out).ravel().tolist()) - {0})
    assert n == len(present), f"reported {n}, image has {len(present)} colors"
    assert present == list(range(1, n + 1)), f"colors are not dense: {present}"


def test_the_case_that_used_to_overreport():
    """Two cells, maximum-contrast objective: two colors, not four."""
    out, n = ncolor.label(_two_cells(), weight_objective=1, return_n=True)
    assert n == 2
    assert sorted(set(np.asarray(out).ravel().tolist())) == [0, 1, 2]


def test_lut_agrees_with_the_reported_count():
    for kwargs in ({}, dict(weight_objective=1)):
        lut, n = ncolor.label(_grid(), return_lut=True, return_n=True, **kwargs)
        used = sorted(set(np.asarray(lut).ravel().tolist()) - {0})
        assert used == list(range(1, n + 1))


def test_color_graph_reports_the_colors_it_used():
    # Two disjoint edges: two colors suffice, and a soft edge between the
    # components must not inflate the count.
    colors, n = ncolor.color_graph([[0, 1], [2, 3]], 4,
                                   soft_edges=[[1, 2]], return_n=True)
    present = sorted(set(colors.tolist()))
    assert n == len(present)
    assert present == list(range(1, n + 1))
