"""Tests for ``ncolor.color_graph``: coloring an abstract edge list."""
import numpy as np
import pytest

import ncolor


def _is_proper(colors, edges):
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    if not len(edges):
        return True
    return bool((colors[edges[:, 0]] != colors[edges[:, 1]]).all())


def test_path_graph():
    edges = [[0, 1], [1, 2], [2, 3]]
    colors, n = ncolor.color_graph(edges, n_vertices=4, return_n=True)
    assert colors.dtype == np.uint8
    assert colors.shape == (4,)
    assert 2 <= n <= 4                    # greedy: proper, not minimal
    assert _is_proper(colors, edges)


def test_cycles():
    for k in (6, 7, 11):
        edges = [[i, (i + 1) % k] for i in range(k)]
        colors, n = ncolor.color_graph(edges, k, return_n=True)
        assert _is_proper(colors, edges)
        assert n <= 4


def test_complete_graphs_need_k_colors():
    for k in (2, 3, 4):
        edges = [[i, j] for i in range(k) for j in range(i + 1, k)]
        colors, n = ncolor.color_graph(edges, k, return_n=True)
        assert n == k
        assert len(set(colors.tolist())) == k
        assert _is_proper(colors, edges)


@pytest.mark.parametrize("k", [5, 6, 8])
def test_graph_over_budget_escalates_instead_of_conflicting(k):
    """A graph that needs more than n gets more, not a broken coloring."""
    edges = [[i, j] for i in range(k) for j in range(i + 1, k)]
    colors, n, conflicts = ncolor.color_graph(
        edges, k, n=4, return_n=True, return_conflicts=True)
    assert n == k
    assert conflicts == 0
    assert _is_proper(colors, edges)


def test_max_depth_caps_the_escalation():
    """Exhausting the escalation budget is the one path to conflicts."""
    odd_cycle = [[i, (i + 1) % 5] for i in range(5)]
    _, n, conflicts = ncolor.color_graph(
        odd_cycle, 5, n=2, max_depth=1, return_n=True, return_conflicts=True)
    assert n == 2 and conflicts > 0
    _, n, conflicts = ncolor.color_graph(
        odd_cycle, 5, n=2, return_n=True, return_conflicts=True)
    assert n == 3 and conflicts == 0


def test_isolated_vertices_get_color_one():
    colors = ncolor.color_graph([[0, 1]], n_vertices=5)
    assert colors[2] == colors[3] == colors[4] == 1
    assert colors[0] != colors[1]


def test_n_vertices_inferred_from_edges():
    colors = ncolor.color_graph([[0, 1], [1, 7]])
    assert colors.shape == (8,)


def test_duplicate_reversed_and_self_edges_are_normalized():
    """A symmetric edge list with self-loops colors like the simple one."""
    messy = [[0, 1], [1, 0], [0, 1], [2, 2], [1, 2], [2, 1]]
    colors, conflicts = ncolor.color_graph(
        messy, 3, return_conflicts=True)
    assert conflicts == 0
    assert _is_proper(colors, [[0, 1], [1, 2]])


def test_out_of_range_edges_are_dropped():
    colors = ncolor.color_graph([[0, 1], [1, 99], [-3, 0]], n_vertices=3)
    assert colors.shape == (3,)
    assert colors[0] != colors[1]


def test_empty_graph():
    assert ncolor.color_graph(np.zeros((0, 2), np.int32), 0).shape == (0,)
    assert ncolor.color_graph(None, 3).tolist() == [1, 1, 1]


def test_check_conflicts_raises():
    odd_cycle = [[i, (i + 1) % 5] for i in range(5)]
    with pytest.raises(ValueError, match="conflict"):
        ncolor.color_graph(odd_cycle, 5, n=2, max_depth=1, check_conflicts=True)


def test_soft_edges_are_respected_when_free():
    """Two components: the soft edge should still separate the colors."""
    hard = [[0, 1], [2, 3]]
    soft = [[1, 2]]
    colors = ncolor.color_graph(hard, 4, soft_edges=soft)
    assert _is_proper(colors, hard)
    assert colors[1] != colors[2]


def test_soft_edges_never_break_the_hard_coloring():
    edges = [[i, j] for i in range(4) for j in range(i + 1, 4)]
    soft = [[0, 1], [0, 2], [1, 3]]
    colors, conflicts = ncolor.color_graph(
        edges, 4, soft_edges=soft, return_conflicts=True)
    assert conflicts == 0
    assert _is_proper(colors, edges)


def test_bad_edge_shape_raises():
    with pytest.raises(ValueError, match=r"\(M, 2\)"):
        ncolor.color_graph(np.zeros((4, 3), np.int32), 4)
    with pytest.raises(ValueError, match=r"\(E, 2\)"):
        ncolor.color_graph([[0, 1]], 2, soft_edges=np.zeros((2, 5), np.int32))


def test_negative_n_vertices_raises():
    with pytest.raises(ValueError, match="n_vertices"):
        ncolor.color_graph([[0, 1]], n_vertices=-1)


def test_grid_graph_matches_raster_label():
    """The same adjacency, reached two ways, needs the same color count."""
    side = 12
    img = np.arange(1, side * side + 1, dtype=np.int32).reshape(side, side)
    pairs = ncolor.connect(img, conn=1)          # 1-indexed label pairs
    colors, n = ncolor.color_graph(pairs - 1, side * side, return_n=True)
    assert n <= 4
    assert _is_proper(colors, pairs - 1)


@pytest.mark.parametrize("bad", [
    np.array([[0, 1], [1, 2 ** 31 + 5]], dtype=np.int64),
    np.array([[0, 2 ** 32 + 1]], dtype=np.uint64),
    np.array([[0, -(2 ** 31) - 5]], dtype=np.int64),
])
def test_vertex_ids_outside_int32_raise_instead_of_wrapping(bad):
    """A wrapped id would silently land on a different, valid vertex."""
    with pytest.raises(OverflowError, match="int32"):
        ncolor.color_graph(bad, n_vertices=3)


def test_soft_edges_outside_int32_raise_too():
    with pytest.raises(OverflowError, match="soft_edges"):
        ncolor.color_graph([[0, 1]], 2,
                           soft_edges=np.array([[0, 2 ** 31]], dtype=np.int64))


def test_int64_edges_within_int32_are_accepted():
    colors = ncolor.color_graph(np.array([[0, 1], [1, 2]], dtype=np.int64), 3)
    assert colors[0] != colors[1] and colors[1] != colors[2]
