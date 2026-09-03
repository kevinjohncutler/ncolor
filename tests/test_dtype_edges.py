"""Input dtypes at the edges: bool, float, uint64, and values outside int32.

The engine works on int32 internally. Before these tests existed, a
label value at or above 2^31 in an int64 / uint32 array was cast with a
plain ``static_cast``, wrapped negative, and the format pass treated it
as background: whole cells vanished with no error. Bool and float
arrays, which segmenters such as cellpose hand back, were rejected
outright.
"""
import numpy as np
import pytest

import ncolor


def _blocks():
    m = np.zeros((40, 40), dtype=np.int32)
    m[2:12, 2:12] = 1
    m[2:12, 14:24] = 2
    m[14:24, 2:12] = 3
    m[30:38, 30:38] = 4
    return m


def _fg(arr):
    return int((np.asarray(arr) != 0).sum())


# ------------------------------------------------------- accepted dtypes


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint64, np.int64])
def test_label_matches_int32_reference(dtype):
    base = _blocks()
    ref = ncolor.label(base)
    assert np.array_equal(ncolor.label(base.astype(dtype)), ref)


def test_label_accepts_bool_mask():
    out, n = ncolor.label(_blocks() > 0, return_n=True)
    assert out.dtype == np.uint8
    assert n >= 1
    assert _fg(out) == _fg(_blocks())


@pytest.mark.parametrize("dtype", [bool, np.float32, np.uint64])
def test_connected_components_accepts(dtype):
    labels, n = ncolor.connected_components(_blocks().astype(dtype))
    assert n == 4
    assert _fg(labels) == _fg(_blocks())


@pytest.mark.parametrize("dtype", [bool, np.float32, np.float64, np.uint64, np.int64])
def test_expand_and_format_accept(dtype):
    base = _blocks()
    assert np.array_equal(ncolor.expand_labels(base.astype(dtype)),
                          ncolor.expand_labels(base.astype(bool) if dtype is bool else base))
    out = ncolor.format_labels(base.astype(dtype))
    assert out.dtype == np.int32
    assert _fg(out) == _fg(base)


def test_connect_accepts_bool_and_float():
    base = _blocks()
    base[2:12, 12:14] = 5                         # make 1 and 5 touch
    assert ncolor.connect(base.astype(np.float64)).tolist() == ncolor.connect(base).tolist()
    assert ncolor.connect(base > 0).shape[1] == 2


def test_float_labels_are_truncated_toward_zero():
    base = _blocks().astype(np.float64)
    base[base > 0] += 0.75                        # 1.75, 2.75, ...
    assert np.array_equal(ncolor.format_labels(base), ncolor.format_labels(_blocks()))


def test_nan_in_float_labels_is_an_error():
    arr = np.where(_blocks() > 0, _blocks(), np.nan).astype(np.float32)
    with pytest.raises(ValueError, match="NaN"):
        ncolor.label(arr)


# --------------------------------------------- values outside int32 range


def _wide_variants():
    base = _blocks()
    fg = base > 0
    return {
        "int64 just past int32": (base.astype(np.int64) + 2 ** 31) * fg,
        "int64 1e11": base.astype(np.int64) * 10 ** 11,
        "uint32 top bit": (base.astype(np.uint32) + 2 ** 31) * fg.astype(np.uint32),
        "uint64 2^40": base.astype(np.uint64) * 2 ** 40,
        "float64 1e12": base.astype(np.float64) * 1e12,
    }


@pytest.mark.parametrize("name", list(_wide_variants()))
def test_label_keeps_every_cell_for_wide_values(name):
    arr = _wide_variants()[name]
    ref = ncolor.label(_blocks())
    out, n = ncolor.label(arr, return_n=True)
    assert _fg(out) == _fg(arr), "cells vanished into background"
    assert np.array_equal(out, ref)


@pytest.mark.parametrize("name", list(_wide_variants()))
def test_format_labels_compacts_wide_values(name):
    arr = _wide_variants()[name]
    out = ncolor.format_labels(arr)
    assert _fg(out) == _fg(arr)
    assert sorted(np.unique(out).tolist()) == [0, 1, 2, 3, 4]
    # The clean=True path takes the generic route; same answer expected.
    assert np.array_equal(ncolor.format_labels(arr, clean=True),
                          ncolor.format_labels(_blocks(), clean=True))


def test_wide_negative_background_follows_min_shift_rule():
    """min < 0 is background, exactly as for narrow inputs."""
    base = _blocks()
    wide = np.where(base > 0, base.astype(np.int64) * 10 ** 11, -1)
    assert np.array_equal(ncolor.label(wide), ncolor.label(np.where(base > 0, base, -1)))


@pytest.mark.parametrize("fn", [ncolor.connect, ncolor.expand_labels, ncolor.delete_spurs])
def test_identity_preserving_ops_refuse_wide_values(fn):
    """These keep label IDs, so they cannot renumber; they must say so."""
    with pytest.raises(OverflowError, match="int32"):
        fn(_blocks().astype(np.int64) * 10 ** 11)


def test_raw_engine_raises_overflow_error():
    from ncolor._backend import Solver
    with pytest.raises(OverflowError, match="int32"):
        Solver(1).label(_blocks().astype(np.int64) * 10 ** 11)


# ------------------------------------------------------- release_buffers


def test_release_buffers_is_safe_before_and_after_use():
    ncolor.release_buffers()                      # engines may not exist yet
    base = _blocks()
    ref = ncolor.label(base)
    ncolor.expand_labels(base)
    ncolor.format_labels(base)
    ncolor.release_buffers()
    assert np.array_equal(ncolor.label(base), ref)
    assert ncolor.expand_labels(base).max() == 4
