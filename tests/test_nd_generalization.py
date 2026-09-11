"""Rank, degenerate-axis and periodic-boundary regression coverage."""
from itertools import product

import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


_MAX_NDIM = 64 if np.lib.NumpyVersion(np.__version__) >= '2.0.0' else 32


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('wrap', [False, True])
def test_clean_ignores_singleton_axes(p, wrap):
    image = np.zeros((8, 9), np.int32)
    image[1:4, 1:4] = 1
    image[4:7, 5:8] = 2
    expected = ncolor.expand_labels(image, p=p, mode='clean', wrap=wrap)
    for shape in [(1, 8, 9), (8, 1, 9), (8, 9, 1),
                  (1,) * 15 + (8, 1, 9), (8, 9) + (1,) * 62]:
        if len(shape) > _MAX_NDIM:
            continue
        actual = ncolor.expand_labels(image.reshape(shape), p=p, mode='clean', wrap=wrap)
        assert actual.shape == shape
        np.testing.assert_array_equal(actual.reshape(image.shape), expected)


@pytest.mark.parametrize('rank', [17, 32, 64])
@pytest.mark.parametrize('active_axis', [0, -1])
def test_high_rank_neighborhoods(rank, active_axis):
    if rank > _MAX_NDIM:
        pytest.skip('this NumPy version cannot construct arrays with this rank')
    shape = [1] * rank
    shape[active_axis] = 3
    labels = np.array([4, 9, 12], np.int32).reshape(shape)
    for conn in (1, rank):
        np.testing.assert_array_equal(np.sort(ncolor.connect(labels, conn=conn), axis=0),
                                      [[4, 9], [9, 12]])
        mask = np.array([1, 0, 1], np.uint8).reshape(shape)
        cc, n = ncolor.connected_components(mask, conn=conn)
        assert n == 2
        np.testing.assert_array_equal(cc.ravel(), [1, 0, 2])
        cc, n, sources = _impl.cc_label_per_label(labels, conn=conn)
        assert n == 3
        np.testing.assert_array_equal(cc.ravel(), [1, 2, 3])
        np.testing.assert_array_equal(sources, [4, 9, 12])
    out, n, conflicts = ncolor.label(labels, expand=False, return_n=True,
                                     return_conflicts=True)
    assert conflicts == 0
    assert n == len(np.unique(out))
    assert out.ravel()[0] != out.ravel()[1] != out.ravel()[2]


def _pairs_reference(labels, conn, radius=1, wrap=False):
    pairs = set()
    offsets = [o for o in product(range(-radius, radius + 1), repeat=labels.ndim)
               if 0 < np.count_nonzero(o) <= conn]
    for pos in np.ndindex(labels.shape):
        a = int(labels[pos])
        if not a:
            continue
        for offset in offsets:
            q = tuple(x + d for x, d in zip(pos, offset))
            if wrap:
                q = tuple(x % n for x, n in zip(q, labels.shape))
            elif any(x < 0 or x >= n for x, n in zip(q, labels.shape)):
                continue
            b = int(labels[q])
            if b and b != a:
                pairs.add(tuple(sorted((a, b))))
    return pairs


@pytest.mark.parametrize('ndim', [2, 3, 4, 5])
@pytest.mark.parametrize('wrap', [False, True])
def test_connect_matches_exhaustive_offsets(ndim, wrap):
    rng = np.random.default_rng(91 + ndim)
    labels = rng.integers(0, 8, size=(3,) * ndim, dtype=np.int32)
    engine = ncolor.Engine(n_threads=1)
    for conn in sorted({1, 2, ndim}):
        actual = engine._solver.connect(labels, conn=conn, wrap=wrap)
        assert set(map(tuple, actual)) == _pairs_reference(labels, conn, wrap=wrap)


def _despur_mark_reference(labels, threshold, remove_thin):
    offsets = [o for o in product((-1, 0, 1), repeat=labels.ndim) if any(o)]
    out = labels.copy()
    for pos in np.ndindex(labels.shape):
        if not labels[pos]:
            continue
        same = []
        for offset in offsets:
            q = tuple(x + d for x, d in zip(pos, offset))
            if all(0 <= x < n for x, n in zip(q, labels.shape)) and labels[q] == labels[pos]:
                same.append(offset)
        faces = sum(np.count_nonzero(o) == 1 for o in same)
        thin = len(same) == 2 and all(a == -b for a, b in zip(*same))
        if faces <= threshold or (remove_thin and thin):
            out[pos] = 0
    return out


@pytest.mark.parametrize('ndim', [2, 3, 4, 5])
@pytest.mark.parametrize('remove_thin', [False, True])
def test_despur_matches_full_stencil_reference(ndim, remove_thin):
    rng = np.random.default_rng(ndim)
    labels = rng.integers(0, 3, size=(3,) * ndim, dtype=np.int32)
    expected = _despur_mark_reference(labels, 1, remove_thin)
    actual, removed = ncolor.delete_spurs(labels, kind='labels', max_iters=1,
                                         remove_thin=remove_thin)
    np.testing.assert_array_equal(actual, expected)
    assert removed == np.count_nonzero(labels != expected)


@pytest.mark.parametrize('remove_thin', [False, True])
def test_despur_many_singleton_axes(remove_thin):
    labels = np.ones((5, 6), np.int32)
    expected, count = ncolor.delete_spurs(labels, kind='labels', remove_thin=remove_thin)
    actual, removed = ncolor.delete_spurs(labels.reshape((1,) * (_MAX_NDIM - 2) + labels.shape),
                                         kind='labels', remove_thin=remove_thin)
    np.testing.assert_array_equal(actual.reshape(labels.shape), expected)
    assert removed == count


@pytest.mark.parametrize('format_input', [False, True])
def test_cleaned_away_cells_keep_valid_colors(format_input):
    engine = ncolor.Engine(n_threads=1)
    # No background to expand into, and every label is a cleanup stub.
    image = np.array([[1, 2], [3, 4]], np.int32)
    assert not np.any(ncolor.expand_labels(image, mode='clean'))
    for _ in range(2):
        out, n = engine.label(image, return_n=True, format_input=format_input)
        assert np.all(out > 0)
        assert n == len(np.unique(out))
        lut, lut_n = engine.label(image, return_lut=True, return_n=True,
                                  format_input=format_input)
        assert lut_n == n
        # Public LUT is a mapping from source labels to colors.
        for label in range(1, 5):
            assert np.all(out[image == label] == lut[label])
        engine.label(np.ones((12, 12), np.int32))


@pytest.mark.parametrize('p', [1, 2])
def test_clean_wrap_changes_voronoi_assignment(p):
    image = np.zeros((12, 15), np.int32)
    image[4:8, 1:4] = 1
    image[4:8, 7:10] = 2
    ordinary = ncolor.expand_labels(image, mode='clean', p=p)
    periodic = ncolor.expand_labels(image, mode='clean', p=p, wrap=True)
    assert ordinary[5, -1] == 2
    assert periodic[5, -1] == 1


@pytest.mark.parametrize('p', [1, 2])
def test_periodic_cleanup_preserves_seam_connected_block(p):
    # One 3x3 block straddling both periodic seams. Clipped cleanup
    # removes its isolated corner and edge stubs; periodic cleanup keeps it.
    image = np.arange(1, 50, dtype=np.int32).reshape(7, 7)
    image[np.ix_([6, 0, 1], [6, 0, 1])] = 50
    clean = ncolor.expand_labels(image, mode='clean', p=p, wrap=True)
    np.testing.assert_array_equal(clean == 50, image == 50)
    bounded = ncolor.expand_labels(image, mode='clean', p=p)
    assert bounded[6, 6] == 0


@pytest.mark.parametrize('p', [1, 2])
def test_label_clean_wrap_matches_explicit_expansion(p):
    image = np.zeros((12, 15), np.int32)
    image[4:8, 1:4] = 1
    image[4:8, 7:10] = 2
    engine = ncolor.Engine(n_threads=1)
    clean = engine.expand_labels(image, mode='clean', p=p, wrap=True)
    lut = engine.label(image, p=p, wrap=True, return_lut=True,
                       soft_conn=0, soft_radius=0)
    pairs = engine._solver.connect(clean, conn=1, wrap=True)
    for a, b in pairs:
        assert lut[a] != lut[b]
    # Exposing the cleaned mask exercises the Solver's expand dispatch.
    out = engine.label(image, p=p, wrap=True, clean_mask=True,
                       soft_conn=0, soft_radius=0)
    for a in (1, 2):
        # label() restores the original background mask by API contract.
        assert np.all(out[(clean == a) & (image != 0)] == lut[a])


def test_neighbor_radius_outside_storage_range_raises():
    image = np.array([[1, 2], [3, 4]], np.int32)
    for kw in [dict(connect_radius=128), dict(soft_radius=128)]:
        with pytest.raises(ValueError, match='radius'):
            ncolor.label(image, expand=False, **kw)


@pytest.mark.parametrize('shape', [(4, 5), (3, 4, 5), (1, 3, 4), (3, 3, 3, 3)])
@pytest.mark.parametrize('wrap', [False, True])
def test_wide_soft_neighborhood_matches_exhaustive_offsets(shape, wrap):
    rng = np.random.default_rng(193)
    labels = rng.integers(0, 6, size=shape, dtype=np.int32)
    engine = ncolor.Engine(n_threads=1)
    engine.label(labels, expand=False, format_input=False, wrap=wrap,
                 soft_conn=2, soft_radius=2)
    hard = _pairs_reference(labels, conn=1, radius=1, wrap=wrap)
    soft = _pairs_reference(labels, conn=2, radius=2, wrap=wrap) - hard
    actual = set(map(tuple, engine._solver.get_last_soft_pairs()))
    assert actual == soft


@pytest.mark.parametrize('p', [1, 2])
def test_clean_wrap_with_single_seed_fills_all_axes(p):
    # No barriers should form with one Voronoi cell. This checks that
    # periodic initialization and propagation leave no stale distances.
    engine = ncolor.Engine(n_threads=2)
    labels = np.zeros((4, 5, 6, 7), np.int32)
    labels[0, -1, 0, -1] = 12
    for _ in range(2):
        out = engine.expand_labels(labels, mode='clean', wrap=True, p=p)
        assert np.all(out == 12)
        engine.expand_labels(np.zeros_like(labels), mode='clean', wrap=True, p=p)
