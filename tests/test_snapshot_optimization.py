"""Compact snapshots must preserve pixels across representation boundaries."""
import numpy as np
import pytest

import ncolor


@pytest.mark.parametrize('shape', [(0, 4), (1, 128), (512, 512), (1, 32, 16, 17)])
@pytest.mark.parametrize('format_input', [False, True])
@pytest.mark.parametrize('mode', ['standard', 'clean'])
def test_empty_snapshot_has_no_pixel_storage(shape, format_input, mode):
    engine = ncolor.Engine(n_threads=4)
    image = np.zeros(shape, np.int32)
    snapshot = engine.prepare_labels(image, format_input=format_input, expand_mode=mode)
    assert snapshot.n_labels == 0
    assert snapshot.nbytes < 256
    engine.label(np.ones((17, 19), np.int32))
    engine.release_buffers()
    for value in (255, 13):
        out = np.full(shape, value, np.uint8)
        actual, count = snapshot.color(out=out, engine=engine, return_n=True)
        assert actual is out
        assert count == 0
        np.testing.assert_array_equal(actual, image)
        out.fill(value)
        lut = snapshot.color(out=out, return_lut=True, engine=engine)
        np.testing.assert_array_equal(lut, [0])
        assert not out.any()


@pytest.mark.parametrize('shape', [(511, 513), (512, 512), (1, 513, 517)])
@pytest.mark.parametrize('delta', [-1, 0, 1])
@pytest.mark.parametrize('clustered', [False, True])
def test_snapshot_density_boundary_and_partial_blocks(shape, delta, clustered):
    image = np.zeros(shape, np.int32)
    count = image.size // 32 + delta
    if clustered:
        indices = np.arange(image.size-count, image.size)
    else:
        indices = np.random.default_rng(754).choice(image.size, count, replace=False)
    image.flat[indices] = 1 + np.arange(count) % 7
    engine = ncolor.Engine(n_threads=4)
    options = dict(expand=False, format_input=False)
    expected = engine.label(image, n=16, **options)
    snapshot = engine.prepare_labels(image, **options)
    if image.size >= 262144 and delta <= 0:
        assert snapshot.nbytes < image.size // 2
    else:
        assert snapshot.nbytes >= image.size
    np.testing.assert_array_equal(snapshot.color(n=16, engine=engine), expected)
    out = np.full(shape, 255, np.uint8)
    snapshot.color(n=16, engine=engine, out=out)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize('mode', ['standard', 'clean'])
@pytest.mark.parametrize('wrap', [False, True])
@pytest.mark.parametrize('shape', [(64, 128), (65, 128), (16, 16, 32), (1, 64, 128)])
def test_small_threaded_envelopes_preserve_labels(shape, mode, wrap):
    rng = np.random.default_rng(729)
    serial = ncolor.Engine(n_threads=1)
    parallel = ncolor.Engine(n_threads=4)
    for fraction in (.005, .2):
        image = np.where(rng.random(shape) < fraction,
                         rng.integers(1, 9, shape), 0).astype(np.int32)
        expected = serial.expand_labels(image, p=2, mode=mode, wrap=wrap)
        actual = parallel.expand_labels(image, p=2, mode=mode, wrap=wrap)
        np.testing.assert_array_equal(actual, expected)
