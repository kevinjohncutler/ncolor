"""Multi-dtype tests for Solver.label / Solver.connect / ExpandEngine.

Verifies the dtype-dispatch path added in commit 25ba195: the cpp Solver
casts uint8/uint16/uint32/int8/int16/int64 inputs to int32 in the
released-GIL block via ``cast_with_bg<InT>`` and produces the same
coloring as the int32 reference.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ncolor
from ncolor._backend import Solver, ExpandEngine


SUPPORTED_DTYPES = [
    np.uint8, np.uint16, np.uint32,
    np.int8, np.int16, np.int32, np.int64,
]


def _circles_2d(H, W, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=dtype)
    info = np.iinfo(dtype)
    pad_h = max(8, H // 6)
    pad_w = max(8, W // 6)
    r_lo = max(2, H // 32)
    r_hi = max(r_lo + 1, H // 12)
    for i in range(1, n + 1):
        if i > info.max:
            break
        cy = rng.integers(pad_h, H - pad_h)
        cx = rng.integers(pad_w, W - pad_w)
        r = int(rng.integers(r_lo, r_hi))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def _balls_3d(D, H, W, n, dtype, seed=0):
    rng = np.random.default_rng(seed)
    m = np.zeros((D, H, W), dtype=dtype)
    info = np.iinfo(dtype)
    pad_d = max(2, D // 6)
    pad_h = max(2, H // 6)
    pad_w = max(2, W // 6)
    r_lo = 2
    r_hi = max(r_lo + 1, D // 6)
    for i in range(1, n + 1):
        if i > info.max:
            break
        cz = rng.integers(pad_d, D - pad_d)
        cy = rng.integers(pad_h, H - pad_h)
        cx = rng.integers(pad_w, W - pad_w)
        r = int(rng.integers(r_lo, r_hi))
        zz, yy, xx = np.ogrid[:D, :H, :W]
        m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


def _is_valid_coloring_2d(colored, mask):
    """Every adjacent pair of distinct nonzero labels in `mask` must have
    different colors in `colored`."""
    nz_right = (mask[:, :-1] != 0) & (mask[:, 1:] != 0)
    nz_down = (mask[:-1, :] != 0) & (mask[1:, :] != 0)
    diff_right = mask[:, :-1] != mask[:, 1:]
    diff_down = mask[:-1, :] != mask[1:, :]
    bad_right = (colored[:, :-1] == colored[:, 1:]) & nz_right & diff_right
    bad_down = (colored[:-1, :] == colored[1:, :]) & nz_down & diff_down
    return not (bad_right.any() or bad_down.any())


@pytest.fixture(scope="module")
def solver():
    return Solver(-1)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_label_2d_supports_dtype(solver, dtype):
    m = _circles_2d(128, 128, 12, dtype)
    out, n_used = solver.label(m, conn=2)
    assert out.shape == m.shape
    assert out.dtype == np.uint8
    assert 0 < n_used <= 8
    assert _is_valid_coloring_2d(out, m.astype(np.int32))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_label_3d_supports_dtype(solver, dtype):
    m = _balls_3d(32, 32, 32, 8, dtype)
    out, n_used = solver.label(m, conn=2)
    assert out.shape == m.shape
    assert out.dtype == np.uint8
    assert 0 < n_used <= 8


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_label_matches_int32_reference(solver, dtype):
    """Output coloring should be a valid coloring regardless of input
    dtype. (Specific color values can differ since the random seed varies
    between attempts; we check semantic correctness instead.)"""
    m = _circles_2d(96, 96, 10, dtype)
    out_dtype, _ = solver.label(m, conn=2)
    out_int32, _ = solver.label(m.astype(np.int32), conn=2)
    # Both must be valid colorings of the same underlying mask.
    m_int = m.astype(np.int32)
    assert _is_valid_coloring_2d(out_dtype, m_int)
    assert _is_valid_coloring_2d(out_int32, m_int)
    # Bg pattern must match exactly across dtypes.
    assert np.array_equal((out_dtype == 0), (m == 0))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_connect_supports_dtype(solver, dtype):
    m = _circles_2d(64, 64, 6, dtype)
    pairs = solver.connect(m, conn=2)
    assert pairs.dtype == np.int32
    assert pairs.shape[1] == 2
    # All returned pairs should be (lo, hi) with lo < hi and both nonzero.
    if pairs.size > 0:
        assert (pairs[:, 0] < pairs[:, 1]).all()
        assert (pairs > 0).all()


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_format_labels_supports_dtype(dtype):
    """ExpandEngine.format_labels accepts int32 today; verify that the
    legacy path (numpy-side astype before calling) still works."""
    eng = ExpandEngine(-1)
    m = _circles_2d(32, 32, 5, dtype).astype(np.int32, copy=False)
    out, n = eng.format_labels(m)
    assert n > 0
    assert out.dtype == np.int32
    # Compact 1..N (with bg=0).
    uniq = set(np.unique(out).tolist())
    assert uniq == set(range(n + 1))


def test_label_empty_input(solver):
    """All-zero input produces all-zero output (handled in cpp)."""
    m = np.zeros((64, 64), dtype=np.int32)
    out, n_used = solver.label(m, conn=2)
    assert out.shape == m.shape
    assert (out == 0).all()
    assert n_used == 0


def test_label_uint16_round_trip(solver):
    """Bigger 2D uint16 input — most common in microscopy. Just ensure
    the pipeline doesn't crash and produces a valid coloring."""
    m = _circles_2d(512, 512, 50, np.uint16)
    out, n_used = solver.label(m, conn=2)
    assert _is_valid_coloring_2d(out, m.astype(np.int32))
    assert 0 < n_used <= 8


def test_label_out_buffer_reuse(solver):
    """out= passes a preallocated buffer; identity should be preserved
    (no silent copy) and the result should match a fresh-alloc call."""
    m = _circles_2d(128, 128, 12, np.int32)
    out_alloc, n_alloc = solver.label(m, conn=2)
    buf = np.empty_like(out_alloc)
    out_buf, n_buf = solver.label(m, conn=2, out=buf)
    assert out_buf is buf, "out= should return the supplied buffer"
    assert n_buf == n_alloc
    assert np.array_equal(out_buf, out_alloc)


def test_label_out_buffer_validates(solver):
    """out= rejects wrong shape / dtype / non-contiguous so that callers
    can't accidentally feed a buffer that won't be written."""
    m = _circles_2d(64, 64, 4, np.int32)
    with pytest.raises(ValueError):
        solver.label(m, out=np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(ValueError):
        solver.label(m, out=np.zeros((64, 64), dtype=np.int32))
    with pytest.raises(ValueError):
        solver.label(m, out=np.zeros((64, 64), dtype=np.float32))
    non_contig = np.zeros((64, 128), dtype=np.uint8)[:, ::2]
    with pytest.raises(ValueError):
        solver.label(m, out=non_contig)


def test_label_expand_false_keeps_bg(solver):
    """expand=False: bg pixels stay 0 (no Voronoi fill) and only
    fg pixels get a color."""
    m = _circles_2d(64, 64, 6, np.int32)
    out_with_expand, _ = solver.label(m, conn=2, expand=True)
    out_no_expand, _ = solver.label(m, conn=2, expand=False)
    # bg mask matches input bg mask in both cases (we always do
    # bg-masking based on the original input).
    assert np.array_equal((out_with_expand == 0), (m == 0))
    assert np.array_equal((out_no_expand == 0), (m == 0))
    # When expand=True every fg pixel ends up with one of the
    # expanded labels' color; same when expand=False, except the
    # graph is built over the un-expanded mask. Both must be valid.
    assert _is_valid_coloring_2d(out_with_expand, m)
    assert _is_valid_coloring_2d(out_no_expand, m)
