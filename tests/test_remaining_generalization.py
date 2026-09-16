"""Regression coverage for option composition, numeric range, and cleanup."""
import subprocess
import sys

import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


MODES = ['min', 'max', 'mean', 'count', 'harmonic', 'mean_inv']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('shape', [(2, 3), (1, 2, 3), (2, 3, 1)])
def test_weighted_radius_and_contact_filter(mode, shape):
    image = np.array([[1, 0, 2], [1, 0, 2]], np.int32).reshape(shape)
    options = dict(expand=False, conn=1, connect_radius=2,
                   weight_objective=1, weight_mode=mode,
                   soft_conn=0, soft_radius=0, return_n=True)
    out, n = ncolor.label(image, min_contact=2, **options)
    assert n == 2
    assert out[image == 1][0] != out[image == 2][0]
    out, n = ncolor.label(image, min_contact=3, **options)
    assert n == 1
    assert np.all(out[image > 0] == 1)


def test_weighted_extra_edges_do_not_crash():
    # A native memory error must fail this test, not kill the pytest process.
    code = r"""
import numpy as np
import ncolor
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
except ImportError:
    pass
image = np.array([[1, 0, 2], [1, 0, 2]], np.int32)
for mode in ('min', 'max', 'mean', 'count', 'harmonic', 'mean_inv'):
    for radius in (1, 2):
        out, n = ncolor.label(image, expand=False, weight_objective=1,
            weight_mode=mode, connect_radius=radius, return_n=True,
            extra_edges=np.array([[0, 1], [1, 1], [1, 2], [2, 1], [2, 9]], np.int32),
            soft_conn=0, soft_radius=0)
        assert n == 2
        assert out[0, 0] != out[0, 2]
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                            text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('mode', MODES)
def test_weighted_without_expansion_is_independent_of_previous_call(mode):
    image = np.array([[1, 0, 2], [3, 0, 4]], np.int32)
    options = dict(expand=False, weight_objective=1, weight_mode=mode,
                   soft_conn=0, soft_radius=0)
    engine = ncolor.Engine(n_threads=1)
    before = engine.label(image, **options)
    large = np.zeros((2, 50001), np.int32)
    large[:, 0] = 1
    large[:, -1] = 2
    engine.label(large, expand_mode='standard', weight_objective=1,
                 weight_mode=mode)
    np.testing.assert_array_equal(engine.label(image, **options), before)


@pytest.mark.parametrize('wrap', [False, True])
@pytest.mark.parametrize('shape', [(2, 50001), (50001, 2), (1, 2, 50001),
                                   (2, 50001, 1), (50001,)])
def test_long_axis_expansion_and_distances(shape, wrap):
    image = np.zeros(shape, np.int32)
    image.flat[0], image.flat[-1] = 1, 2
    coords = np.indices(shape, dtype=np.int64)
    costs = []
    for seed in [np.zeros(len(shape), np.int64), np.array(shape) - 1]:
        delta = abs(coords - seed.reshape((-1,) + (1,) * len(shape)))
        if wrap:
            delta = np.minimum(delta, np.array(shape).reshape((-1,) + (1,) * len(shape)) - delta)
        costs.append(np.sum(delta * delta, axis=0))
    engine = ncolor.Engine(n_threads=2)
    out, dist = engine._expand.expand_labels_with_dist(image, p=2, wrap=wrap)
    selected_cost = np.where(out == 1, costs[0], costs[1])
    np.testing.assert_array_equal(selected_cost, np.minimum(*costs))
    np.testing.assert_allclose(dist, np.sqrt(np.minimum(*costs)), rtol=1e-14)
    np.testing.assert_array_equal(out[image > 0], image[image > 0])
    # Reuse after a wide call must return to the narrow and L1 paths.
    small = np.array([[1, 0, 0, 2]], np.int32)
    for p in (1, 2):
        _, d = engine._expand.expand_labels_with_dist(small, p=p)
        np.testing.assert_array_equal(d, [[0, 1, 1, 0]])


@pytest.mark.parametrize('wrap', [False, True])
def test_wide_distance_consumers(wrap):
    image = np.zeros((2, 50001), np.int32)
    image[0, 0], image[1, -1] = 1, 2
    expected = np.sqrt(2 if wrap else 50000**2 + 1)
    engine = ncolor.Engine(n_threads=2)
    matrix = engine._expand.pairwise_nearest_distance(image, 2, p=2, wrap=wrap)
    np.testing.assert_allclose(matrix, [[0, expected], [expected, 0]])
    classes = engine._expand.per_class_min_edt(
        image, np.array([0, 1, 2], np.int32), 3, p=2, wrap=wrap)
    assert classes[0, 1, -1] == pytest.approx(expected)
    assert classes[1, 0, 0] == pytest.approx(expected)
    assert np.isinf(classes[2]).all()


@pytest.mark.parametrize('wrap', [False, True])
def test_clean_long_axis_preserves_solid_regions(wrap):
    image = np.zeros((2, 3, 50001), np.int32)
    image[..., 0], image[..., -1] = 1, 2
    expected = ncolor.expand_labels(image, mode='standard', wrap=wrap)
    np.testing.assert_array_equal(
        ncolor.expand_labels(image, mode='clean', wrap=wrap), expected)


@pytest.mark.parametrize('value', [2**31 - 1, 2**32 - 1, 2**63 - 1])
@pytest.mark.parametrize('shape', [(4, 7), (1, 4, 7), (4, 7, 1)])
def test_clean_format_large_source_ids(value, shape):
    image = np.zeros((4, 7), np.int64)
    image[:2, :2] = image[2:, 5:] = value
    image[0, -1] = 7  # dropped separately
    expected = np.zeros_like(image, dtype=np.uint8)
    expected[:2, :2], expected[2:, 5:] = 1, 2
    out = ncolor.format_labels(image.reshape(shape), clean=True, min_area=4)
    np.testing.assert_array_equal(out.reshape(image.shape), expected)
    assert out.dtype == np.uint8


@pytest.mark.parametrize('shape', [(0, 7), (1, 0, 3)])
def test_clean_format_empty(shape):
    out = ncolor.format_labels(np.zeros(shape, np.int32), clean=True)
    assert out.shape == shape and out.size == 0


@pytest.mark.parametrize('rank', [2, 6, 16, 32])
@pytest.mark.parametrize('mode', ['cardinal', 'total'])
def test_binary_cleanup_high_rank(rank, mode):
    image = np.array([0, 1, 1, 1, 0], np.uint8).reshape((1,) * (rank - 1) + (5,))
    out = ncolor.delete_spurs(image, kind='binary', mode=mode,
                              hole_threshold=100, max_iter=1)
    # Keep the original rank's threshold after skipping singleton axes.
    expected = [0, 0, 1, 0, 0] if rank == 2 else [0, 0, 0, 0, 0]
    np.testing.assert_array_equal(out.ravel(), expected)


@pytest.mark.parametrize('mode', ['cardinal', 'total'])
def test_binary_holes_exclude_exterior_even_with_large_threshold(mode):
    image = np.ones((5, 5), np.uint8)
    image[0, :2] = 0
    image[2, 2] = 0
    expected = image.astype(bool)
    expected[2, 2] = True
    out = ncolor.delete_spurs(image, kind='binary', mode=mode,
                              hole_threshold=10000, max_iter=0)
    np.testing.assert_array_equal(out, expected)
    # A singleton axis exposes the apparent planar hole to the exterior.
    out = ncolor.delete_spurs(image[None], kind='binary', mode=mode,
                              hole_threshold=10000, max_iter=0)
    np.testing.assert_array_equal(out[0], image.astype(bool))


@pytest.mark.parametrize('remove_thin', [False, True])
@pytest.mark.parametrize('max_iters', [0, 1, 5])
def test_parallel_despur_with_singleton_axes(remove_thin, max_iters):
    rng = np.random.default_rng(57)
    image = rng.integers(0, 4, (129, 133), dtype=np.int32)
    expected, count = _impl.delete_spurs_labels(
        image, n_threads=1, max_iters=max_iters, remove_thin=remove_thin)
    if max_iters == 0:
        np.testing.assert_array_equal(expected, image)
        assert count == 0
    for shape in [(1, 129, 133), (129, 1, 133), (129, 133, 1)]:
        out, removed = _impl.delete_spurs_labels(
            image.reshape(shape), n_threads=4, max_iters=max_iters,
            remove_thin=remove_thin)
        np.testing.assert_array_equal(out.reshape(image.shape), expected)
        assert removed == count


@pytest.mark.parametrize('mode', MODES)
def test_weighted_auto_soft_edges_still_apply(mode):
    image = np.array([[1, 2, 0, 0, 3, 0, 4]] * 2, np.int32)
    out = ncolor.label(image, expand=False, conn=1, weight_objective=1,
                        weight_mode=mode, soft_conn=1, soft_radius=2)
    assert out[0, 0] != out[0, 1]  # hard edge
    assert out[0, 4] != out[0, 6]  # soft edge across the gap


@pytest.mark.parametrize('shape', [(1, 5, 7), (5, 1, 7), (5, 7, 1),
                                   (3, 4, 5), (2, 3, 2, 4)])
@pytest.mark.parametrize('mode', ['cardinal', 'total'])
def test_unpadded_binary_matches_padded_reference(shape, mode):
    pytest.importorskip('scipy')
    from test_delete_spurs import _ref_delete_spurs

    rng = np.random.default_rng(73)
    for threshold in (None, 2, 4):
        image = rng.random(shape) > 0.4
        options = dict(mode=mode, threshold=threshold, hole_threshold=3, max_iter=5)
        expected = _ref_delete_spurs(image, **options)
        np.testing.assert_array_equal(
            ncolor.delete_spurs(image, kind='binary', **options), expected)
