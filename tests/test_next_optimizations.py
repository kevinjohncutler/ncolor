"""Contracts for adaptive component partitions and prepared coloring."""
import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


@pytest.mark.parametrize('shape', [(2, 513, 517), (2, 2, 257, 263),
                                   (2, 3, 129, 1024), (3, 513, 517)])
@pytest.mark.parametrize('conn', [1, 2, 3])
def test_thin_component_partitions_preserve_numbering(shape, conn):
    rng = np.random.default_rng(411)
    image = rng.integers(-1, 4, shape, dtype=np.int32)
    engine = ncolor.Engine(n_threads=4)
    for actual, reference in [
        (engine.connected_components, _impl.cc_label),
        (engine._expand.components_per_label, _impl.cc_label_per_label),
    ]:
        expected = reference(image, conn=conn)
        result = actual(image, conn=conn)
        for got, want in zip(result, expected):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('conn', [1, 2, 3, 4])
def test_thin_partition_diagonals_cross_both_cut_directions(conn):
    image = np.zeros((2, 2, 513, 517), np.int32)
    image[0, 1, 257, 258] = image[1, 0, 256, 257] = -7
    image[0, 0, 257, 258] = image[1, 1, 256, 257] = 9
    engine = ncolor.Engine(n_threads=4)
    expected = _impl.cc_label_per_label(image, conn=conn)
    actual = engine._expand.components_per_label(image, conn=conn)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('shape', [(31, 37), (5, 7, 9), (1, 1, 37), (0, 4)])
@pytest.mark.parametrize('mode', ['standard', 'clean'])
@pytest.mark.parametrize('wrap', [False, True])
def test_prepared_matches_label(shape, mode, wrap):
    rng = np.random.default_rng(303)
    image = rng.integers(0, 7, shape, dtype=np.int32)
    engine = ncolor.Engine(n_threads=2)
    options = dict(expand_mode=mode, wrap=wrap)
    prepared = engine.prepare_labels(image, **options)
    assert prepared.shape == shape
    for n in [3, 5]:
        expected = engine.label(image, n=n, return_n=True, return_conflicts=True, **options)
        actual = prepared.color(n=n, return_n=True, return_conflicts=True, engine=engine)
        for got, want in zip(actual, expected):
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('weight_mode', ['min', 'max', 'mean', 'count', 'harmonic', 'mean_inv'])
@pytest.mark.parametrize('clean_mask', [False, True])
def test_prepared_weighted_palette_and_constraints(weight_mode, clean_mask):
    image = np.zeros((43, 47), np.int32)
    image[3::8, 4::9] = np.arange(1, image[3::8, 4::9].size + 1).reshape(image[3::8, 4::9].shape)
    engine = ncolor.Engine(n_threads=2)
    options = dict(weight_objective=-1, weight_mode=weight_mode, clean_mask=clean_mask,
                   min_contact=2, extra_edges=[[1, 3]], soft_extra_edges=[[2, 5]])
    prepared = engine.prepare_labels(image, **options)
    for n in [3, 4, 7]:
        palette = np.abs(np.arange(n + 1)[:, None] - np.arange(n + 1)[None, :]).astype(float)
        expected = engine.label(image, n=n, de_table=palette, return_lut=True, **options)
        actual = prepared.color(n=n, de_table=palette, return_lut=True, engine=engine)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('image', [np.zeros((3, 4), np.int32), np.full((3, 4), -9),
                                    np.full((3, 4), 13),
                                    np.array([[0, 2**40], [2**41, 2**40]], dtype=np.uint64),
                                    np.array([[-2.1, 0.1], [5.9, 5.2]])])
def test_prepared_owns_input_and_survives_engine_reuse(image):
    engine = ncolor.Engine(n_threads=2)
    expected = engine.label(image)
    prepared = engine.prepare_labels(image)
    assert prepared.nbytes >= image.size
    image[:] = 0
    engine.label(np.ones((71, 73), np.int32))
    engine.release_buffers()
    out = np.empty(expected.shape, np.uint8)
    assert prepared.color(out=out, engine=engine) is out
    np.testing.assert_array_equal(out, expected)
    from concurrent.futures import ThreadPoolExecutor
    def run(_):
        return prepared.color(engine=ncolor.Engine(n_threads=1))
    with ThreadPoolExecutor(2) as pool:
        for result in pool.map(run, range(4)):
            np.testing.assert_array_equal(result, expected)


def test_prepared_validates_output_and_palette():
    prepared = ncolor.prepare_labels([[0, 1], [2, 2]])
    for out in [np.empty((2, 2), np.int32), np.empty((3, 2), np.uint8),
                np.empty((2, 4), np.uint8)[:, ::2]]:
        with pytest.raises(ValueError):
            prepared.color(out=out)
    readonly = np.empty((2, 2), np.uint8)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        prepared.color(out=readonly)
    for palette in [np.zeros((2, 2)), np.full((5, 5), np.nan)]:
        with pytest.raises(ValueError):
            prepared.color(de_table=palette)
    with pytest.raises(ValueError):
        prepared.color(n=256)


@pytest.mark.parametrize('dtype', [np.bool_, np.uint8, np.int8, np.uint16, np.int16])
@pytest.mark.parametrize('size', [499999, 500000, 500001])
def test_byte_presence_dispatch_matches_int32(dtype, size):
    rng = np.random.default_rng(831)
    low = 0 if dtype == np.bool_ or np.iinfo(dtype).min == 0 else -21
    image = rng.integers(low, 2 if dtype == np.bool_ else 37, (1, size), dtype=np.int32).astype(dtype)
    engine = ncolor.Engine(n_threads=4)
    expected = engine.format_labels(image.astype(np.int32))
    actual = engine.format_labels(image)
    np.testing.assert_array_equal(actual, expected)
    # All positive and negative constants preserve the background contract.
    for value in ([0, 1] if low == 0 else [-5, 0, 7]):
        image[:] = value
        np.testing.assert_array_equal(engine.format_labels(image),
                                      engine.format_labels(image.astype(np.int32)))


@pytest.mark.parametrize('dtype', [np.uint8, np.int8, np.uint16, np.int16])
def test_narrow_input_pipeline_preserves_background(dtype):
    image = np.zeros((513, 517), dtype=dtype)
    image[3:20, 4:30] = 13
    image[45:60, 21:45] = 27
    if np.iinfo(dtype).min < 0:
        image[0, 0] = -3
    engine = ncolor.Engine(n_threads=4)
    for mode in ['standard', 'clean']:
        expected = engine.label(image.astype(np.int32), expand_mode=mode)
        np.testing.assert_array_equal(engine.label(image, expand_mode=mode), expected)
        prepared = engine.prepare_labels(image, expand_mode=mode)
        np.testing.assert_array_equal(prepared.color(engine=engine), expected)


@pytest.mark.parametrize('options', [
    dict(expand=False, first_seen=True),
    dict(format_input=False, expand_mode='standard', p=1),
    dict(conn=2, connect_radius=1, soft_conn=1, soft_radius=3),
    dict(weight_objective=1, min_contact=3, wrap=True),
    dict(clean_mask=True, first_seen=True, p=1),
])
def test_prepared_reuses_exact_topology(options):
    rng = np.random.default_rng(741)
    image = rng.integers(0, 9, (5, 7, 9), dtype=np.int32)
    engine = ncolor.Engine(n_threads=2)
    edges = np.array([[1, 3], [2, 6]], np.int32)
    prepared = engine.prepare_labels(image, extra_edges=edges, **options)
    expected = engine.label(image, extra_edges=edges, return_lut=True, **options)
    edges[:] = 0
    engine.release_buffers()
    np.testing.assert_array_equal(prepared.color(return_lut=True, engine=engine), expected)


def test_prepared_weighted_isolates_after_weighted_graph():
    engine = ncolor.Engine(n_threads=2)
    engine.label([[1, 2], [3, 4]], weight_objective=1)
    image = np.array([[1, 0, 2]], np.int32)
    options = dict(expand=False, weight_objective=1, soft_extra_edges=np.empty((0, 2), np.int32))
    prepared = engine.prepare_labels(image, **options)
    np.testing.assert_array_equal(prepared.color(engine=engine), engine.label(image, **options))


@pytest.mark.parametrize('count', [1, 255, 256, 65535, 65536])
def test_prepared_compact_render_boundaries(count):
    image = np.arange(count + 1, dtype=np.int32).reshape(1, -1)
    engine = ncolor.Engine(n_threads=2)
    options = dict(expand=False, soft_conn=0, soft_radius=0)
    prepared = engine.prepare_labels(image, **options)
    expected, lut = engine.label(image, **options), engine.label(image, return_lut=True, **options)
    np.testing.assert_array_equal(prepared.color(engine=engine), expected)
    np.testing.assert_array_equal(prepared.color(return_lut=True, engine=engine), lut)
    out = np.full(image.shape, 255, dtype=np.uint8)
    np.testing.assert_array_equal(prepared.color(return_lut=True, out=out, engine=engine), lut)
    np.testing.assert_array_equal(out, expected)
    # Account for all array payload: shape, labels, row offsets, adjacency,
    # and source/destination edge lists of this simple chain.
    width = 1 if count <= 255 else 2 if count <= 65535 else 4
    assert prepared.nbytes == 16 + image.size * width + (count + 1) * 4 + (count - 1) * 16


def test_prepared_lookup_skips_native_raster_and_still_validates_out():
    prepared = ncolor.prepare_labels(np.ones((513, 517), np.int32))
    data = prepared._PreparedLabels__data
    solver = _impl.Solver(1)
    image, count = solver.color_prepared(data, render=False)
    assert image.shape == (0,) and count == 1
    with pytest.raises(ValueError):
        prepared.color(return_lut=True, out=np.zeros((1,), np.uint8))


@pytest.mark.parametrize('shape', [(513, 517), (1, 513, 1, 517), (2, 257, 263)])
@pytest.mark.parametrize('mode', ['standard', 'clean'])
@pytest.mark.parametrize('wrap', [False, True])
def test_retained_layout_matches_original_pipeline(shape, mode, wrap):
    rng = np.random.default_rng(524)
    image = np.zeros(shape, np.int32)
    selected = rng.choice(image.size, 101, replace=False)
    image.flat[selected] = np.arange(1, 102)
    engine = ncolor.Engine(n_threads=4)
    solver = engine._solver
    for options in [dict(), dict(conn=2, connect_radius=2, min_contact=3),
                    dict(conn=len(shape), soft_conn=1, soft_radius=3),
                    dict(clean_mask=True), dict(weight_objective=1),
                    dict(extra_edges=np.array([[1, 4]], np.int32),
                         soft_extra_edges=np.array([[3, 6]], np.int32))]:
        common = dict(p=2, expand_mode=mode, wrap=wrap, **options)
        expected, count = solver.label(image, _retain_layout=False, **common)
        lut = solver.get_last_lut().copy()
        soft = solver.get_last_soft_pairs().copy()
        actual, actual_count = solver.label(image, **common)
        np.testing.assert_array_equal(actual, expected)
        assert actual_count == count
        np.testing.assert_array_equal(solver.get_last_lut(), lut)
        np.testing.assert_array_equal(solver.get_last_soft_pairs(), soft)
        data = _impl.PreparedRaster()
        solver.label(image, _prepared=data, **common)
        prepared, prepared_count = solver.color_prepared(data)
        np.testing.assert_array_equal(prepared, expected)
        assert prepared_count == count


@pytest.mark.parametrize('dtype', [np.int8, np.int16, np.int32, np.int64, np.float64])
@pytest.mark.parametrize('shape', [(3, 4), (513, 1024)])
def test_unformatted_negative_labels_raise_before_expansion(dtype, shape):
    image = np.ones(shape, dtype=dtype)
    image.flat[-1] = -3
    engine = ncolor.Engine(n_threads=4)
    for call in (engine.label, engine.prepare_labels):
        with pytest.raises(ValueError, match='nonnegative'):
            call(image, format_input=False)
    # A failed call must leave the engine reusable, including the direct
    # path for fractions that legitimately truncate to background.
    image.flat[-1] = 0
    np.testing.assert_array_equal(engine.label(image, format_input=False), engine.label(image))


def test_unformatted_float_truncation_to_background():
    image = np.array([[-0.9, 1.2], [2.8, 0.9]])
    expected = ncolor.label(image.astype(np.int32), format_input=False)
    np.testing.assert_array_equal(ncolor.label(image, format_input=False), expected)
    np.testing.assert_array_equal(ncolor.prepare_labels(image, format_input=False).color(), expected)
