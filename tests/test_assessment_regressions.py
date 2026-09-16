"""Regression tests for palette limits, graph contracts, and retained buffers."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys

import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


def test_invalid_coloring_budgets_in_child():
    code = r'''
import numpy as np
import ncolor
from ncolor._backend import _impl
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass
solver = _impl.Solver(1)
for empty in (False, True):
    image = np.zeros((0, 2), np.int32) if empty else np.array([[1, 2]], np.int32)
    edges = np.empty((0, 2), np.int32) if empty else np.array([[0, 1]], np.int32)
    for n, depth in [(0, 1), (-1, 1), (256, 1), (4, 0), (4, -1)]:
        for call in (
            lambda: ncolor.label(image, n=n, max_depth=depth),
            lambda: ncolor.color_graph(edges, n_vertices=2, n=n, max_depth=depth),
            lambda: solver.label(image, n_colors=n, max_depth=depth),
            lambda: solver.color_graph(edges, 2, n_colors=n, max_depth=depth),
        ):
            try:
                call()
            except ValueError:
                pass
            else:
                raise AssertionError((n, depth))
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('n', [31, 32, 64, 255])
@pytest.mark.parametrize('threads', [1, 4])
def test_large_palette_complete_graph(n, threads):
    edges = np.column_stack(np.triu_indices(n, 1)).astype(np.int32)
    engine = ncolor.Engine(n_threads=threads)
    colors, used, conflicts = engine.color_graph(
        edges, n_vertices=n, n=n, return_n=True, return_conflicts=True)
    assert used == n and conflicts == 0
    np.testing.assert_array_equal(np.sort(colors), np.arange(1, n + 1))


@pytest.mark.parametrize('weighted', [False, True])
def test_large_palette_image_with_soft_edges(weighted):
    # Hard complete graph on 32 labels, plus two unconstrained labels.
    # Soft edges exercise the post-pass at a palette size above 31.
    image = np.arange(1, 35, dtype=np.int32).reshape(2, 17)
    edges = np.column_stack(np.triu_indices(32, 1)).astype(np.int32) + 1
    engine = ncolor.Engine(n_threads=2)
    out, used, conflicts = engine.label(
        image, n=32, extra_edges=edges, expand=False,
        weight_objective=int(weighted), soft_conn=0, soft_radius=0,
        soft_extra_edges=np.array([[33, 34]], np.int32),
        return_n=True, return_conflicts=True)
    assert conflicts == 0 and used >= 32
    assert np.unique(out.ravel()[:32]).size == 32
    assert np.all(out > 0)


@pytest.mark.parametrize('value', [256, 512, -256, 2**40, 0.5, -0.5])
def test_binary_cleanup_preserves_nonzero_values(value):
    image = np.full((5, 5), value)
    image[0, 0] = 0
    actual = ncolor.delete_spurs(image, kind='binary', hole_threshold=0, max_iter=0)
    np.testing.assert_array_equal(actual, image != 0)


def _csr(rows):
    lengths = [len(row) for row in rows]
    return (np.array([0] + list(np.cumsum(lengths)), np.int32),
            np.array([v for row in rows for v in row], np.int32))


@pytest.mark.parametrize('n', [0, 1, 3, 15])
def test_twohop_matches_reference(n):
    rng = np.random.default_rng(44)
    graph = rng.random((n, n)) < 0.3
    graph |= graph.T
    np.fill_diagonal(graph, False)
    rows = [set(np.flatnonzero(row)) for row in graph]
    ip, ix = _impl.two_hop_csr(*_csr([sorted(row) for row in rows]))
    assert len(ip) == n + 1 and ip[-1] == len(ix)
    for u in range(n):
        expected = set().union(*(rows[v] for v in rows[u])) - rows[u] - {u}
        assert set(ix[ip[u]:ip[u + 1]]) == expected
        assert ip[u + 1] - ip[u] == len(expected)


@pytest.mark.parametrize('ip,ix', [
    ([], []), ([1], []), ([0, -1], []), ([0, 2], [0]),
    ([0, 2, 1], [0]), ([0, 1], [-1]), ([0, 1], [1]),
    ([[0, 0]], []), ([0, 1], [[0]]), ([0, 1], [2**32]),
])
def test_twohop_rejects_malformed_graphs(ip, ix):
    with pytest.raises((ValueError, OverflowError)):
        _impl.two_hop_csr(np.asarray(ip, dtype=np.int64), np.asarray(ix, dtype=np.int64))


@pytest.mark.parametrize('u,v,w,n', [
    ([0], [1], [1], 1), ([-1], [0], [1], 1), ([], [], [], -1),
    ([0], [0], [], 1), ([0], [0], [[1]], 1),
    ([2**32], [0], [1], 1), ([0], [0], [np.nan], 1),
])
def test_pair_graph_rejects_invalid_inputs(u, v, w, n):
    with pytest.raises((ValueError, OverflowError)):
        _impl.symmetric_pair_csr(np.array(u, np.int64), np.array(v, np.int64),
                                 np.array(w, float), n)


def test_pair_graph_preserves_rows_and_weights():
    ip, ix, weights = _impl.symmetric_pair_csr(
        np.array([0, 2]), np.array([2, 1]), np.array([1.5, 7.0]), 4)
    np.testing.assert_array_equal(ip, [0, 1, 2, 4, 4])
    np.testing.assert_array_equal(ix, [2, 2, 0, 1])
    np.testing.assert_array_equal(weights, [1.5, 7.0, 1.5, 7.0])


def _kempe_args(n=2):
    ip, ix = _csr([[] for _ in range(n)])
    return dict(initial_colors=np.ones(n, np.int32),
                adj_indptr=ip, adj_indices=ix, twohop_indptr=ip, twohop_indices=ix,
                iou_indptr=ip, iou_indices=ix, iou_weights=np.array([], float))


@pytest.mark.parametrize('field,value', [
    ('adj_indptr', [0, 1, 1]), ('adj_indices', [99]),
    ('twohop_indptr', [0, 0]), ('iou_indptr', [0, -1, 0]),
    ('initial_colors', [1, 256]), ('initial_colors', [[1, 1]]),
    ('iou_weights', [[1.0]]), ('n_colors', 256),
])
def test_kempe_rejects_malformed_inputs(field, value):
    args = _kempe_args()
    args[field] = value if field == 'n_colors' else np.asarray(value)
    with pytest.raises((ValueError, OverflowError)):
        _impl.kempe_sa(**args)


@pytest.mark.parametrize('n', [0, 2])
def test_kempe_single_color_terminates(n):
    out, loss = _impl.kempe_sa(**_kempe_args(n), n_colors=1)
    np.testing.assert_array_equal(out, np.ones(n))
    assert loss == 0


@pytest.mark.parametrize('first_seen', [False, True])
@pytest.mark.parametrize('background', [None, 0])
def test_verbose_does_not_change_numbering(first_seen, background, capsys):
    image = np.array([[0, 9, 3]], np.int32)
    expected = [[0, 1, 2]] if first_seen else [[0, 2, 1]]
    for verbose in [False, True]:
        actual = ncolor.format_labels(image, background=background,
                                      first_seen=first_seen, verbose=verbose)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('background', [-2, 9, 2**40])
@pytest.mark.parametrize('clean', [False, True])
def test_explicit_background_preserves_other_labels(background, clean):
    image = np.array([[background, 0, 0], [-3, -3, background]], np.int64)
    out = ncolor.format_labels(image, background=background, clean=clean, min_area=2)
    assert np.all(out[image == background] == 0)
    assert np.all(out[image == 0] > 0) and np.all(out[image == -3] > 0)
    assert out[0, 1] != out[1, 0]


@pytest.mark.parametrize('clean', [False, True])
def test_ignore_markers_survive_formatting(clean):
    image = np.array([[0, 7, 7, 1], [0, 1, 1, 9]], np.int32)
    out = ncolor.format_labels(image, ignore=True, clean=clean, min_area=2)
    assert np.all(out[image == 0] == 0)
    assert np.all(out[image == 1] == 1)
    assert np.all(out[image == 7] >= 2)
    assert out[image == 9][0] == (1 if clean else 3)


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('wrap', [False, True])
def test_standard_expansion_singleton_axes_and_engine_reuse(p, wrap):
    rng = np.random.default_rng(54)
    engine = ncolor.Engine(n_threads=2)
    for base_shape in [(7, 9), (3, 4, 5), (1, 1)]:
        image = rng.integers(0, 5, base_shape, dtype=np.int32)
        image[rng.random(base_shape) < 0.8] = 0
        expected, dist = engine._expand.expand_labels_with_dist(image, p=p, wrap=wrap)
        for axis in range(len(base_shape) + 1):
            expanded = np.expand_dims(image, axis)
            actual, actual_dist = engine._expand.expand_labels_with_dist(expanded, p=p, wrap=wrap)
            np.testing.assert_array_equal(actual.reshape(base_shape), expected)
            np.testing.assert_array_equal(actual_dist.reshape(base_shape), dist)
    # Grow in distance-only mode, then request working labels at the same size.
    image = np.zeros((64, 65), np.int32)
    image[0, 0], image[-1, -1] = 1, 2
    engine.expand_labels(image, p=1, mode='standard')
    np.testing.assert_array_equal(engine.expand_labels(image, p=2, mode='standard'),
                                  ncolor.expand_labels(image, p=2, mode='standard'))


def test_spur_pool_reuse_and_concurrent_calls():
    rng = np.random.default_rng(81)
    image = rng.integers(0, 3, (128, 129), dtype=np.int32)
    reference = _impl.delete_spurs_labels(image, n_threads=1, remove_thin=True)
    engine = ncolor.Engine(n_threads=4)
    for call in [lambda: engine.delete_spurs(image, kind='labels', remove_thin=True),
                 lambda: _impl.delete_spurs_labels(image, n_threads=4, remove_thin=True)]:
        with ThreadPoolExecutor(3) as workers:
            outputs = list(workers.map(lambda _: call(), range(6)))
        for out, removed in outputs:
            np.testing.assert_array_equal(out, reference[0])
            assert removed == reference[1]
    # The clean formatter shares the supplied engine's pool too.
    np.testing.assert_array_equal(engine.format_labels(image, clean=True, despur=True),
                                  ncolor.Engine(n_threads=1).format_labels(image, clean=True, despur=True))


@pytest.mark.parametrize('first_seen', [False, True])
def test_sparse_extreme_labels(first_seen):
    image = np.array([[-2**31, 2**31 - 1, 0, 2**31 - 1]], np.int32)
    expected = [[0, 1, 2, 1]] if first_seen else [[0, 2, 1, 2]]
    np.testing.assert_array_equal(ncolor.format_labels(image, first_seen=first_seen), expected)
    dense_positive = np.full((3, 3), 2**31 - 1, np.int32)
    np.testing.assert_array_equal(ncolor.format_labels(dense_positive), np.ones((3, 3)))


@pytest.mark.parametrize('threads', [1, 4])
def test_weighted_palette_escalation(threads):
    image = np.array([[1, 2, 3]], np.int32)
    table = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    out, used, conflicts = ncolor.Engine(n_threads=threads).label(
        image, n=2, max_depth=2, expand=False, extra_edges=np.array([[1, 3]]),
        weight_objective=1, de_table=table, soft_conn=0, soft_radius=0,
        return_n=True, return_conflicts=True)
    assert used == 3 and conflicts == 0 and np.unique(out).size == 3


def test_graph_helpers_accept_empty_sequences():
    ip, ix = _impl.two_hop_csr([0], [])
    np.testing.assert_array_equal(ip, [0])
    assert ix.size == 0
    ip, ix, weights = _impl.symmetric_pair_csr([], [], [], 0)
    np.testing.assert_array_equal(ip, [0])
    assert ix.size == weights.size == 0
