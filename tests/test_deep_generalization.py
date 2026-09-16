"""Contracts for repeated cleanup, metadata ownership, and checked conversions."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from itertools import product
import subprocess
import sys
import threading

import numpy as np
import pytest

import ncolor
from ncolor._backend import _impl


OPTIONS = dict(expand=False, soft_conn=0, soft_radius=0)


@pytest.mark.parametrize('mapped', [False, True])
def test_readonly_output_rejected(tmp_path, mapped):
    if mapped:
        # A future invalid write must fail the child, without terminating pytest.
        path = tmp_path / 'output.bin'
        path.write_bytes(b'abcd')
        code = '''
import sys, numpy as np, ncolor
out = np.memmap(sys.argv[1], mode='r', dtype=np.uint8, shape=(2, 2))
try:
    ncolor.label(np.ones((2, 2), np.int32), out=out)
except ValueError as error:
    assert 'writ' in str(error)
else:
    raise AssertionError('read-only output accepted')
'''
        result = subprocess.run([sys.executable, '-c', code, str(path)],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert path.read_bytes() == b'abcd'
    else:
        out = np.full((2, 2), 99, np.uint8)
        out.flags.writeable = False
        with pytest.raises(ValueError, match='writ'):
            ncolor.label(np.ones((2, 2), np.int32), out=out)
        assert np.all(out == 99)


def test_shared_engine_keeps_metadata_with_its_call():
    engine = ncolor.Engine(n_threads=1)
    a = np.array([[1, 2], [1, 2]], np.int32)
    b = np.array([[1, 2, 3], [1, 2, 3]], np.int32)
    options = dict(OPTIONS, return_lut=True, return_n=True, return_conflicts=True)
    expected = [engine.label(x, **options) for x in (a, b)]
    native = engine._solver
    first_done, second_started, allow_return = (threading.Event() for _ in range(3))

    class ScheduledSolver:
        def label(self, image, **kwargs):
            result = native.label(image, **kwargs)
            if image.shape == a.shape:
                first_done.set()
                assert allow_return.wait(5)
            return result

        def __getattr__(self, name):
            return getattr(native, name)

    engine._solver = ScheduledSolver()

    def second_call():
        second_started.set()
        return engine.label(b, **options)

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(engine.label, a, **options)
        try:
            assert first_done.wait(5)
            second = pool.submit(second_call)
            assert second_started.wait(5)
            # The second transaction cannot finish before the first reads metadata.
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
        finally:
            allow_return.set()
        actual = [first.result(timeout=5), second.result(timeout=5)]
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(got[0], want[0])
        assert got[1:] == want[1:]


def test_native_metadata_getters_during_shared_calls():
    solver = _impl.Solver(2)

    def work(index):
        for _ in range(15):
            solver.color_graph(np.array([[0, 1]], np.int32), index + 2)
            lut = solver.get_last_lut()
            assert lut.ndim == 1 and lut[0] == 0
            assert solver.get_last_n_conflicts() == 0
            assert solver.get_last_n_soft_violations() == 0
            assert solver.get_last_soft_pairs().shape[1] == 2
            assert isinstance(solver.get_last_stages(), list)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(work, range(3)))


@pytest.mark.parametrize('method', ['expand_labels_with_dist', 'per_class_min_edt',
                                    'pairwise_nearest_distance', 'regionprops'])
@pytest.mark.parametrize('dtype,value', [(np.int64, 2**32 + 1),
                                         (np.uint64, 2**63 + 1),
                                         (np.float64, np.inf)])
def test_distance_and_properties_reject_narrowing(method, dtype, value):
    image = np.array([[1, 0, value]], dtype=dtype)
    engine = _impl.ExpandEngine(1)
    args = {'expand_labels_with_dist': (), 'per_class_min_edt': (np.array([0, 1]), 1),
            'pairwise_nearest_distance': (1,), 'regionprops': (1,)}
    owner = _impl if method == 'regionprops' else engine
    with pytest.raises(OverflowError):
        getattr(owner, method)(image, *args[method])


@pytest.mark.parametrize('p', [1, 2])
@pytest.mark.parametrize('wrap', [False, True])
def test_scalar_distances_after_scratch_reuse(p, wrap):
    engine = _impl.ExpandEngine(1)
    engine.expand_labels_with_dist(np.array([0, 0, 0, 0, 1], np.int32), p=p)
    for value in (7, 0):
        out, distance = engine.expand_labels_with_dist(np.array(value, np.int32), p=p, wrap=wrap)
        ref_out, ref_dist = engine.expand_labels_with_dist(np.array([value], np.int32), p=p, wrap=wrap)
        assert out.shape == distance.shape == ()
        assert out == ref_out[0]
        assert distance == ref_dist[0]
    classes = engine.per_class_min_edt(np.array(1), np.array([0, 1]), 1, p=p)
    np.testing.assert_array_equal(classes, [0])


@pytest.mark.parametrize('method', ['expand_labels_with_dist', 'per_class_min_edt',
                                    'pairwise_nearest_distance'])
def test_distance_metric_checked_without_seeds(method):
    engine = _impl.ExpandEngine(1)
    args = {'expand_labels_with_dist': (), 'per_class_min_edt': (np.array([0, 1]), 1),
            'pairwise_nearest_distance': (1,)}
    with pytest.raises(ValueError, match='p must'):
        getattr(engine, method)(np.zeros((0, 2), np.int32), *args[method], p=3)


def test_class_indices_reject_narrowing():
    with pytest.raises(OverflowError):
        _impl.ExpandEngine(1).per_class_min_edt(np.array([[1]]),
                                              np.array([0, 2**32 + 1]), 1)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('name', ['extra_edges', 'soft_extra_edges'])
@pytest.mark.parametrize('edges,error', [([[2**32 + 1, 2]], OverflowError),
                                         ([[1.9, 2.9]], ValueError),
                                         ([[np.nan, 2]], ValueError),
                                         ([1, 2], ValueError),
                                         ([[1, 2, 3]], ValueError)])
def test_raster_edges_checked(native, name, edges, error):
    call = _impl.Solver(1).label if native else ncolor.label
    with pytest.raises(error):
        call(np.array([[1, 0, 2]], np.int32), **OPTIONS, **{name: np.array(edges)})


@pytest.mark.parametrize('name', ['extra_edges', 'soft_extra_edges'])
def test_invalid_one_based_endpoints_are_dropped(name):
    image = np.array([[1, 0, 2]], np.int32)
    edges = np.array([[-2**31, 2], [0, 2], [1, 99]], np.int32)
    np.testing.assert_array_equal(ncolor.label(image, **OPTIONS, **{name: edges}),
                                  ncolor.label(image, **OPTIONS))


def spur_reference(image, threshold, thin, rounds):
    out = image.copy()
    offsets = list(product((-1, 0, 1), repeat=out.ndim))
    for _ in range(rounds):
        remove = []
        for c in np.ndindex(out.shape):
            if out[c] == 0:
                continue
            neighbors = []
            for delta in offsets:
                if not any(delta):
                    continue
                q = tuple(x + dx for x, dx in zip(c, delta))
                if all(0 <= x < n for x, n in zip(q, out.shape)) and out[q] == out[c]:
                    neighbors.append(delta)
            faces = sum(sum(abs(v) for v in delta) == 1 for delta in neighbors)
            antipodal = (len(neighbors) == 2 and
                         all(a == -b for a, b in zip(*neighbors)))
            if faces <= threshold or (thin and antipodal):
                remove.append(c)
        for c in remove:
            out[c] = 0
        if not remove:
            break
    return out


@pytest.mark.parametrize('shape', [(7,), (5, 6), (4, 4, 4), (2, 3, 2, 3)])
@pytest.mark.parametrize('thin', [False, True])
@pytest.mark.parametrize('threshold', [-1, 0, 1, 3])
def test_cleanup_matches_repeated_neighborhood_rule(shape, thin, threshold):
    rng = np.random.default_rng(912)
    for _ in range(5):
        image = rng.choice([0, 1, 2], shape, p=[.2, .7, .1]).astype(np.int32)
        for rounds in (0, 1, 3, -1):
            expected = spur_reference(image, threshold, thin, image.size if rounds < 0 else rounds)
            got, count = _impl.delete_spurs_labels(image, threshold=threshold,
                max_iters=rounds, remove_thin=thin, n_threads=1)
            np.testing.assert_array_equal(got, expected)
            assert count == np.count_nonzero(image) - np.count_nonzero(expected)


def test_parallel_singleton_cleanup_matches_repeated_passes():
    rng = np.random.default_rng(62)
    image = (rng.random((1, 96, 128)) < .78).astype(np.int32)
    for thin in (False, True):
        expected = image.copy()
        for _ in range(4):
            expected, _ = _impl.delete_spurs_labels(expected, max_iters=1,
                remove_thin=thin, n_threads=1)
        got, _ = _impl.delete_spurs_labels(image, max_iters=4,
            remove_thin=thin, n_threads=4)
        np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('first_seen', [False, True])
@pytest.mark.parametrize('mode', ['standard', 'clean'])
@pytest.mark.parametrize('image', [np.array([[.5, 1.0], [.5, 2.0]]),
                                   np.array([[-1, -1, 2], [-1, 0, 2]])])
@pytest.mark.parametrize('expand', [False, True])
def test_coloring_background_follows_normalization(image, mode, expand, first_seen):
    formatted = ncolor.format_labels(image, first_seen=first_seen)
    got = ncolor.label(image, expand_mode=mode, first_seen=first_seen, **dict(OPTIONS, expand=expand))
    expected = ncolor.label(formatted, expand_mode=mode, **dict(OPTIONS, expand=expand))
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(got == 0, formatted == 0)


@pytest.mark.parametrize('dtype', [np.int32, np.uint64, np.float64])
def test_explicit_zero_background_avoids_unique_sort(monkeypatch, dtype):
    image = np.array([[0, 8, 9], [9, 0, 8]], dtype=dtype)
    expected = ncolor.format_labels(image)

    def unexpected_sort(*args, **kwargs):
        raise AssertionError('equivalent zero background should use native compaction')

    monkeypatch.setattr(np, 'unique', unexpected_sort)
    np.testing.assert_array_equal(ncolor.format_labels(image, background=0), expected)


@pytest.mark.parametrize('rounds', [1, 3])
def test_fast_spur_binding_above_eight_dimensions(rounds):
    image = np.array([[1, 0, 0, 0, 2], [1, 0, 0, 2, 2]], np.int32)
    options = dict(despur_iters=rounds, despur_remove_thin=False,
                   expand_mode='standard', soft_conn=0, soft_radius=0)
    solver = _impl.Solver(2)
    expected, count = solver.label(image, **options)
    got, actual_count = solver.label(image.reshape((1,) * 8 + image.shape), **options)
    np.testing.assert_array_equal(got.reshape(image.shape), expected)
    assert actual_count == count
