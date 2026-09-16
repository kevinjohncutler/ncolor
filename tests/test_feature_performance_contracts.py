"""Optimized feature and component paths preserve exact label assignments."""
import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


@pytest.mark.parametrize('shape', [(513, 517), (17, 129, 131), (8, 17, 33, 65),
                                  (1, 513, 1, 517), (2, 512, 512)])
@pytest.mark.parametrize('conn', [1, 2])
@pytest.mark.parametrize('threads', [2, 4])
def test_parallel_components_match_serial_scan_order(shape, conn, threads):
    rng = np.random.default_rng(20260916)
    image = rng.integers(0, 5, shape, dtype=np.int32)
    engine = ncolor.Engine(n_threads=threads)
    expected, count = _impl.cc_label(image, conn=conn)
    actual, found = engine.connected_components(image, conn=conn)
    assert count == found
    np.testing.assert_array_equal(actual, expected)
    expected, count, values = _impl.cc_label_per_label(image, conn=conn)
    actual, found, sources = engine._expand.components_per_label(image, conn=conn)
    assert count == found
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(sources, values)


@pytest.mark.parametrize('conn', [1, 2, 3, 4])
def test_parallel_diagonal_seams_and_source_ids(conn):
    shape = (8, 17, 33, 65)
    image = np.zeros(shape, np.int32)
    # Same-label diagonals cross slab seams, while a different label must
    # stay a separate component even when face-adjacent.
    for axis0 in range(shape[0]):
        image[axis0, 5 + axis0 % 2, 8 + axis0 % 2, 12 + axis0 % 2] = -7
        image[axis0, 5 + axis0 % 2, 8 + axis0 % 2, 13 + axis0 % 2] = 91
    engine = ncolor.Engine(n_threads=4)
    expected = _impl.cc_label_per_label(image, conn=conn)
    actual = engine._expand.components_per_label(image, conn=conn)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('value', [0, 1])
def test_parallel_components_uniform_and_strided(value):
    engine = ncolor.Engine(n_threads=4)
    image = np.full((1026, 517), value, np.uint8)[::2, ::-1]
    out, count = engine.connected_components(image)
    assert count == value
    np.testing.assert_array_equal(out, image)


@pytest.mark.parametrize('shape', [(13, 17), (7, 9, 11), (1, 13, 1, 17), (2, 47000)])
@pytest.mark.parametrize('wrap', [False, True])
def test_labels_only_feature_transform_matches_distance_returning_path(shape, wrap):
    engine = ncolor.Engine(n_threads=4)
    image = np.zeros(shape, np.int32)
    rng = np.random.default_rng(515)
    image.flat[rng.choice(image.size, 9, replace=False)] = np.arange(1, 10)
    expected, distances = engine._expand.expand_labels_with_dist(image, p=2, wrap=wrap)
    actual = engine.expand_labels(image, p=2, wrap=wrap)
    np.testing.assert_array_equal(actual, expected)
    # Reusing scratch after a labels-only call must still produce complete
    # distances on the next distance-returning call.
    repeated, current = engine._expand.expand_labels_with_dist(image, p=2, wrap=wrap)
    np.testing.assert_array_equal(repeated, expected)
    np.testing.assert_array_equal(current, distances)


@pytest.mark.parametrize('mode', ['standard', 'clean'])
@pytest.mark.parametrize('wrap', [False, True])
def test_weighted_calls_after_unweighted_feature_transform(mode, wrap):
    image = np.zeros((97, 101), np.int32)
    image[5:25, 9:35] = 1
    image[50:70, 25:45] = 2
    image[25:45, 70:90] = 3
    engine = ncolor.Engine(n_threads=4)
    options = dict(expand_mode=mode, wrap=wrap, weight_objective=1)
    expected = engine.label(image, **options)
    engine.label(image, expand_mode=mode, wrap=wrap)
    np.testing.assert_array_equal(engine.label(image, **options), expected)


@pytest.mark.parametrize('maximum', [31, 4095, 4096, 65535])
def test_private_presence_matches_serial_at_dispatch_boundaries(maximum):
    rng = np.random.default_rng(811)
    image = rng.integers(0, maximum // 2 + 1, (1025, 1027), dtype=np.int32) * 2
    image.flat[0] = maximum
    expected = ncolor.Engine(n_threads=1).format_labels(image)
    engine = ncolor.Engine(n_threads=4)
    for _ in range(3):
        np.testing.assert_array_equal(engine.format_labels(image), expected)


def test_parallel_disconnected_slabs_need_only_offsets():
    shape = (517, 521)
    image = np.indices(shape).sum(axis=0).astype(np.uint8) % 2
    engine = ncolor.Engine(n_threads=4)
    for call, reference in [
        (engine.connected_components, _impl.cc_label),
        (engine._expand.components_per_label, _impl.cc_label_per_label),
    ]:
        actual, expected = call(image, conn=1), reference(image, conn=1)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)
