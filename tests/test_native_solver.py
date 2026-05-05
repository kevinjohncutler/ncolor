"""Smoke tests for ncolor._backend (the C++ engine) — focused on the
n_threads resolution conventions and end-to-end labeling correctness."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncolor import _backend as nc
from ncolor._backend import _smt


# --- n_threads resolution conventions ---------------------------------------

def test_default_resolves_to_auto():
    """``Solver()`` (no arg) → auto_threads()."""
    expected = _smt.auto_threads()
    assert nc.Solver().n_threads == expected


@pytest.mark.parametrize("sentinel", [-1, 0, -2, -10, -1.0, 0.0])
def test_zero_or_negative_resolves_to_auto(sentinel):
    """``-1``, ``0``, and any negative (int or float) → auto."""
    expected = _smt.auto_threads()
    assert nc.Solver(sentinel).n_threads == expected


def test_one_is_serial_not_full():
    """``Solver(1)`` is always 1 thread — never 100% of cores."""
    assert nc.Solver(1).n_threads == 1


def test_explicit_int_passes_through():
    assert nc.Solver(8).n_threads == 8


def test_float_at_or_above_one_rounds():
    assert nc.Solver(8.0).n_threads == 8
    assert nc.Solver(8.4).n_threads == 8
    assert nc.Solver(8.6).n_threads == 9


def test_fraction_scales_to_cpu_count():
    """``0 < x < 1`` → round(x × os.cpu_count())."""
    ncpu = os.cpu_count() or 1
    assert nc.Solver(0.5).n_threads == max(1, round(0.5 * ncpu))
    assert nc.Solver(0.25).n_threads == max(1, round(0.25 * ncpu))
    assert nc.Solver(0.75).n_threads == max(1, round(0.75 * ncpu))


def test_tiny_fraction_rounds_up_to_1():
    """A fraction so small the product rounds to 0 still gets at least 1."""
    assert nc.Solver(1e-9).n_threads >= 1


def test_string_raises():
    """Non-numeric input is rejected by pybind11's double conversion."""
    with pytest.raises(TypeError):
        nc.Solver("not-a-number")


def test_engines_share_convention():
    """``ConnectEngine`` and ``ExpandEngine`` use the same resolver."""
    expected = _smt.auto_threads()
    assert nc.ConnectEngine().n_threads == expected
    assert nc.ExpandEngine().n_threads == expected
    assert nc.ConnectEngine(0.5).n_threads == nc.Solver(0.5).n_threads


# --- _smt helper -------------------------------------------------------------

def test_auto_threads_returns_positive_int():
    n = _smt.auto_threads()
    assert isinstance(n, int) and n >= 1


def test_physical_cores_le_logical():
    phys = _smt._physical_cores()
    log = os.cpu_count() or phys
    assert 1 <= phys <= log


def test_calibrate_returns_int_in_range():
    """``calibrate(force=False)`` returns a value within [physical, logical]."""
    n = _smt.calibrate(force=False)
    phys = _smt._physical_cores()
    log = os.cpu_count() or phys
    assert phys <= n <= log


# --- end-to-end labeling correctness ----------------------------------------

def _make_mask(H=128, n_circles=12, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((H, H), dtype=np.int32)
    for i in range(1, n_circles + 1):
        cy, cx = rng.integers(15, H - 15), rng.integers(15, H - 15)
        r = int(rng.integers(5, max(8, H // 16)))
        yy, xx = np.ogrid[:H, :H]
        mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return mask


def test_label_returns_uint8_and_count():
    mask = _make_mask(H=128, n_circles=8)
    sv = nc.Solver()
    out, n_used = sv.label(mask)
    assert out.dtype == np.uint8
    assert out.shape == mask.shape
    assert 1 <= n_used <= 4


def test_label_assigns_color_per_label():
    """Every input label maps to a non-zero output color; background stays 0."""
    mask = _make_mask(H=128, n_circles=10)
    sv = nc.Solver()
    out, _ = sv.label(mask)
    # Background pixels (where mask==0) should be 0 in output.
    assert (out[mask == 0] == 0).all()
    # Foreground pixels should be > 0.
    assert (out[mask > 0] > 0).all()


def test_label_invariant_across_thread_counts():
    """Same input → same color count regardless of thread count.

    Exact color assignment per-region depends on graph traversal order so
    isn't bit-stable across thread counts, but the number of colors used is
    a property of the graph and should match.
    """
    mask = _make_mask(H=192, n_circles=20, seed=42)
    counts = []
    for nt in [1, 2, 4, _smt.auto_threads()]:
        out, n_used = nc.Solver(nt).label(mask)
        counts.append(n_used)
        # All colors used must be in [1, n_used]
        assert set(np.unique(out[mask > 0]).tolist()) <= set(range(1, n_used + 1))
    assert len(set(counts)) == 1, f"color count varies by thread count: {counts}"


# --- ND support: 2D conn∈{1,2}, 3D conn∈{1,2,3} ---------------------------

def _make_3d_mask(D=32, n_blobs=20, seed=0):
    rng = np.random.default_rng(seed)
    mask = np.zeros((D, D, D), dtype=np.int32)
    for i in range(1, n_blobs + 1):
        cz = rng.integers(4, D - 4)
        cy = rng.integers(4, D - 4)
        cx = rng.integers(4, D - 4)
        r = int(rng.integers(2, max(3, D // 6)))
        zz, yy, xx = np.ogrid[:D, :D, :D]
        mask[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return mask


def _coloring_is_valid(mask, colored, conn):
    """Every pair of touching regions in mask must have different colors."""
    import scipy.ndimage as snd
    struct = snd.generate_binary_structure(mask.ndim, conn)
    labels = np.unique(mask)
    labels = labels[labels > 0]
    color_of = {}
    for l in labels:
        coords = np.nonzero(mask == l)
        color_of[int(l)] = int(colored[tuple(c[0] for c in coords)])
    for l in labels:
        m = (mask == l)
        d = snd.binary_dilation(m, structure=struct) & ~m
        for t in np.unique(mask[d]):
            if t > 0 and t != l and color_of[int(t)] == color_of[int(l)]:
                return False, (int(l), int(t), color_of[int(l)])
    return True, None


@pytest.mark.parametrize("conn", [1, 2])
def test_label_2d_conn(conn):
    mask = _make_mask(H=128, n_circles=20)
    out, n_used = nc.Solver().label(mask, conn=conn)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
    assert (out[mask == 0] == 0).all()
    assert (out[mask > 0] > 0).all()
    valid, err = _coloring_is_valid(mask, out, conn)
    assert valid, f"adjacent regions share color: {err}"


@pytest.mark.parametrize("conn", [1, 2, 3])
def test_label_3d_conn(conn):
    mask = _make_3d_mask(D=32, n_blobs=15, seed=conn)
    out, n_used = nc.Solver().label(mask, conn=conn)
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
    assert (out[mask == 0] == 0).all()
    assert (out[mask > 0] > 0).all()
    valid, err = _coloring_is_valid(mask, out, conn)
    assert valid, f"adjacent regions share color: {err}"


def test_label_rejects_1d():
    with pytest.raises(ValueError):
        nc.Solver().label(np.zeros(64, dtype=np.int32))


def test_label_rejects_invalid_conn():
    mask = _make_mask(H=64, n_circles=5)
    with pytest.raises(ValueError):
        nc.Solver().label(mask, conn=3)  # 2D → conn must be 1 or 2
    with pytest.raises(ValueError):
        nc.Solver().label(mask, conn=0)
