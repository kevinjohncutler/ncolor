"""Cross-interface and dimensional contracts found by iterative review."""
from collections import UserDict

import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


@pytest.mark.parametrize('shape', [(17,), (7, 8), (4, 5, 3)])
@pytest.mark.parametrize('conn', [1, 2, 3])
@pytest.mark.parametrize('per_label', [False, True])
def test_component_numbering_ignores_singleton_axes(shape, conn, per_label):
    rng = np.random.default_rng(204)
    image = rng.integers(0, 4, shape, dtype=np.int32)
    call = _impl.cc_label_per_label if per_label else _impl.cc_label
    expected = call(image, conn=min(conn, image.ndim))
    for padded in [(1,) * 9 + shape, shape + (1,) * 9,
                   sum(((n, 1) for n in shape), ())]:
        got = call(image.reshape(padded), conn=min(conn, image.ndim))
        assert got[1] == expected[1]
        np.testing.assert_array_equal(got[0].reshape(shape), expected[0])
        if per_label:
            np.testing.assert_array_equal(got[2], expected[2])


@pytest.mark.parametrize('shape', [(1,) * 20, (1, 0, 1), (0,)])
def test_components_singletons_and_empty(shape):
    image = np.ones(shape, np.int32)
    for call in (_impl.cc_label, _impl.cc_label_per_label):
        out = call(image, conn=1)
        np.testing.assert_array_equal(out[0], image)
        assert out[1] == bool(image.size)


def test_one_dimensional_components_default():
    labels, count = ncolor.connected_components([1, 0, -2, 3, 0])
    np.testing.assert_array_equal(labels, [1, 0, 2, 2, 0])
    assert count == 2
    with pytest.raises(ValueError, match='conn'):
        ncolor.connected_components(np.array([1]), conn=2)


@pytest.mark.parametrize('dtype,value', [(np.int64, 2**32 + 1),
                                         (np.uint64, 2**63), (np.float64, np.inf)])
def test_components_do_not_merge_unrepresentable_source_labels(dtype, value):
    with pytest.raises(OverflowError):
        _impl.cc_label_per_label(np.array([[1, value]], dtype), conn=1)


@pytest.mark.parametrize('followup', ['empty', 'zero', 'graph'])
def test_soft_pair_metadata_reset_on_every_solver_path(followup):
    solver = _impl.Solver(1)
    solver.label(np.array([[1, 0, 2]], np.int32), expand=False,
                 soft_conn=1, soft_radius=2)
    assert solver.get_last_soft_pairs().tolist() == [[1, 2]]
    if followup == 'graph':
        solver.color_graph(np.empty((0, 2), np.int32), 0)
    else:
        solver.label(np.zeros((0, 3) if followup == 'empty' else (2, 3), np.int32))
    assert solver.get_last_soft_pairs().shape == (0, 2)
    assert solver.get_last_n_soft_violations() == 0


def test_geo_null_features_and_geometry_collections():
    shapely = pytest.importorskip('shapely', minversion='2.0')
    from ncolor import geo
    blank = {'type': 'Feature', 'geometry': None, 'properties': {}}

    class Interface:
        __geo_interface__ = blank

        def __len__(self):
            return 1

    for data in ([blank], [UserDict(blank)], [Interface()], Interface(),
                 {'type': 'FeatureCollection', 'features': [blank]}):
        colors, count = geo.label(data, return_n=True)
        np.testing.assert_array_equal(colors, [0])
        assert count == 0
    collection = shapely.GeometryCollection([shapely.Point(0, 0), shapely.Point(1, 1)])
    assert geo.label(collection).tolist() == [1]
    assert geo.label([None, shapely.Polygon()], return_n=True)[1] == 0


@pytest.mark.parametrize('argument', ['tolerance', 'min_shared_length'])
@pytest.mark.parametrize('value', [-1, np.nan, np.inf])
def test_geo_distance_options_validated_even_without_features(argument, value):
    pytest.importorskip('shapely', minversion='2.0')
    from ncolor import geo
    with pytest.raises(ValueError, match=argument):
        geo.connect([], **{argument: value})


def test_tolerance_buffers_each_participating_geometry_once(monkeypatch):
    shapely = pytest.importorskip('shapely', minversion='2.0')
    from ncolor import geo
    geoms = np.array([shapely.box(x * 1.01, y * 1.01, x * 1.01 + 1, y * 1.01 + 1)
                      for x in range(5) for y in range(5)], dtype=object)
    candidates = geo.connect(geoms, tolerance=.02, min_shared_length=None)
    a, b = geoms[candidates[:, 0]], geoms[candidates[:, 1]]
    lengths = shapely.length(shapely.intersection(shapely.buffer(a, .01),
                                                  shapely.buffer(b, .01))) / 2
    expected = candidates[lengths > .2]
    native_buffer = shapely.buffer
    calls = []

    def counted_buffer(geometry, *args, **kwargs):
        calls.append(len(geometry))
        return native_buffer(geometry, *args, **kwargs)

    monkeypatch.setattr(shapely, 'buffer', counted_buffer)
    got = geo.connect(geoms, tolerance=.02, min_shared_length=.2)
    np.testing.assert_array_equal(got, expected)
    assert calls == [len(geoms)]


@pytest.mark.parametrize('depth', [2, 3])
@pytest.mark.parametrize('wrap', [False, True])
def test_coloring_fallback_ignores_singleton_axes(depth, wrap):
    image = np.array([[1, 2], [3, 4]], np.int32)
    options = dict(n=2, conn=2, max_depth=depth, expand=False,
                   wrap=wrap, soft_conn=0, soft_radius=0,
                   return_n=True, return_conflicts=True)
    engine = ncolor.Engine(n_threads=2)
    expected = engine.label(image, **options)
    for shape in [(1,) * 8 + image.shape, image.shape + (1,) * 8]:
        got = engine.label(image.reshape(shape), **options)
        np.testing.assert_array_equal(got[0].reshape(image.shape), expected[0])
        assert got[1:] == expected[1:]


@pytest.mark.parametrize('palette', [np.ones(5), np.ones((4, 4)),
                                      np.ones((5, 6)), np.full((5, 5), np.nan),
                                      np.full((5, 5), np.inf)])
@pytest.mark.parametrize('empty', [False, True])
def test_palette_validation_precedes_all_solver_paths(palette, empty):
    image = np.zeros((0, 2) if empty else (2, 2), np.int32)
    with pytest.raises(ValueError, match='de_table'):
        ncolor.label(image, weight_objective=1, de_table=palette)


def test_repeated_engine_lifetimes_keep_live_pool_usable():
    live = ncolor.Engine(n_threads=2)
    image = np.array([[1, 2], [3, 4]], np.int32)
    expected = live.label(image)
    for _ in range(30):
        transient = ncolor.Engine(n_threads=1)
        np.testing.assert_array_equal(transient.format_labels(image), image)
        del transient
    np.testing.assert_array_equal(live.label(image), expected)


def test_random_graph_and_raster_conflict_counts_match_edges():
    rng = np.random.default_rng(521)
    engine = ncolor.Engine(n_threads=2)
    for _ in range(60):
        n = int(rng.integers(1, 22))
        pairs = np.argwhere(np.triu(rng.random((n, n)) < rng.uniform(.05, .6), 1)).astype(np.int32)
        colors, used, conflicts = engine.color_graph(pairs, n_vertices=n, n=4,
            max_depth=1, return_n=True, return_conflicts=True)
        assert conflicts == np.count_nonzero(colors[pairs[:, 0]] == colors[pairs[:, 1]])
        assert used == len(np.unique(colors)) and colors.min() > 0
        shape = (int(rng.integers(2, 5)),) * int(rng.integers(2, 5))
        image = ncolor.format_labels(rng.integers(0, 7, shape, dtype=np.int32))
        conn = int(rng.integers(1, len(shape) + 1))
        wrap = bool(rng.integers(2))
        lut, _, conflicts = engine.label(image, expand=False, conn=conn, wrap=wrap,
            soft_conn=0, soft_radius=0, return_lut=True, return_n=True, return_conflicts=True)
        pairs = engine._solver.connect(image, conn=conn, wrap=wrap)
        assert conflicts == np.count_nonzero(lut[pairs[:, 0]] == lut[pairs[:, 1]])


@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.uint8, np.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_connect_accepts_readonly_inputs_without_mutation(dtype, strided):
    source = np.array([[1, 1, 2, 2], [3, 0, 2, 4], [3, 4, 4, 4]], dtype=dtype)
    image = source[:, ::-1] if strided else source
    image.flags.writeable = False
    before = source.copy()
    got = ncolor.connect(image)
    assert {tuple(row) for row in got} == {(1, 2), (1, 3), (2, 4), (3, 4)}
    np.testing.assert_array_equal(source, before)
