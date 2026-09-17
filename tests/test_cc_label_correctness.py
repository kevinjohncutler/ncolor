"""Equivalence tests for ncolor.connected_components and ncolor.regionprops.

The parity-vs-skimage cases carry a ``needs_skimage`` mark and
auto-skip when scikit-image isn't installed. The two pure edge-case
tests (empty / single component) only exercise the cpp side and run
on every install.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import ncolor


needs_skimage = pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None,
    reason="parity-vs-skimage comparison needs scikit-image",
)


def _make_2d(size, n, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((size, size), dtype=np.uint16)
    yy, xx = np.ogrid[:size, :size]
    rmax = max(4, size // 12)
    for i in range(1, n + 1):
        cy, cx = rng.integers(0, size), rng.integers(0, size)
        r = int(rng.integers(max(3, rmax // 2), rmax))
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = i
    return m


def _make_3d(D, n, seed):
    rng = np.random.default_rng(seed)
    m = np.zeros((D, D, D), dtype=np.uint16)
    zz, yy, xx = np.ogrid[:D, :D, :D]
    rmax = max(2, D // 8)
    for i in range(1, n + 1):
        cz = rng.integers(0, D); cy = rng.integers(0, D); cx = rng.integers(0, D)
        r = int(rng.integers(2, rmax))
        m[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = i
    return m


@needs_skimage
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("conn", [1, 2])
@pytest.mark.parametrize("size,n", [(128, 30), (256, 50), (512, 100)])
def test_cc_label_equivalence_2d(seed, conn, size, n):
    """cpp cc_label and skimage.measure.label induce the same partition."""
    from skimage import measure as skimage_measure
    arr = _make_2d(size, n, seed)
    sk = skimage_measure.label(arr > 0, connectivity=conn).astype(np.int32)
    cp, n_components = ncolor.connected_components(arr, conn=conn)
    assert sk.shape == cp.shape
    assert np.array_equal(sk > 0, cp > 0), "fg pattern differs"
    assert int(sk.max()) == int(cp.max()) == n_components
    # Every cpp component → unique skimage component (and vice versa).
    for cp_lbl in range(1, n_components + 1):
        coords = np.argwhere(cp == cp_lbl)
        sk_lbls = {int(sk[tuple(c)]) for c in coords}
        assert len(sk_lbls) == 1, (
            f"cp label {cp_lbl} maps to multiple sk labels {sk_lbls}"
        )


@needs_skimage
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("conn", [1, 2])
@pytest.mark.parametrize("D,n", [(32, 15), (64, 30), (96, 60)])
def test_cc_label_equivalence_3d(seed, conn, D, n):
    from skimage import measure as skimage_measure
    arr = _make_3d(D, n, seed)
    sk = skimage_measure.label(arr > 0, connectivity=conn).astype(np.int32)
    cp, n_components = ncolor.connected_components(arr, conn=conn)
    assert np.array_equal(sk > 0, cp > 0)
    assert int(sk.max()) == int(cp.max()) == n_components
    for cp_lbl in range(1, n_components + 1):
        coords = np.argwhere(cp == cp_lbl)
        sk_lbls = {int(sk[tuple(c)]) for c in coords}
        assert len(sk_lbls) == 1


@needs_skimage
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("conn", [1, 2])
@pytest.mark.parametrize("size,n", [(128, 30), (256, 50)])
def test_regionprops_equivalence_2d(seed, conn, size, n):
    from skimage import measure as skimage_measure
    arr = _make_2d(size, n, seed)
    sk_lab = skimage_measure.label(arr > 0, connectivity=conn).astype(np.int32)
    sk_props = skimage_measure.regionprops(sk_lab)
    n_lab = len(sk_props)
    cp = ncolor.regionprops(sk_lab, n_lab)

    assert np.array_equal(
        cp["area"],
        np.array([p.area for p in sk_props], dtype=np.int64),
    )
    sk_bbox_min = np.array([p.bbox[:arr.ndim] for p in sk_props], dtype=np.int64)
    sk_bbox_max = np.array([p.bbox[arr.ndim:] for p in sk_props], dtype=np.int64)
    assert np.array_equal(cp["bbox_min"], sk_bbox_min)
    assert np.array_equal(cp["bbox_max"], sk_bbox_max)
    sk_centroid = np.array([p.centroid for p in sk_props], dtype=np.float64)
    np.testing.assert_allclose(cp["centroid"], sk_centroid, atol=1e-9, rtol=0)


@needs_skimage
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("D,n", [(32, 15), (64, 30)])
def test_regionprops_equivalence_3d(seed, D, n):
    from skimage import measure as skimage_measure
    arr = _make_3d(D, n, seed)
    sk_lab = skimage_measure.label(arr > 0, connectivity=2).astype(np.int32)
    sk_props = skimage_measure.regionprops(sk_lab)
    n_lab = len(sk_props)
    cp = ncolor.regionprops(sk_lab, n_lab)

    assert np.array_equal(
        cp["area"],
        np.array([p.area for p in sk_props], dtype=np.int64),
    )
    sk_centroid = np.array([p.centroid for p in sk_props], dtype=np.float64)
    np.testing.assert_allclose(cp["centroid"], sk_centroid, atol=1e-9, rtol=0)


@pytest.mark.parametrize("shape", [(64, 64), (300, 301), (2, 200, 201), (7, 8, 9, 10)])
def test_regionprops_is_independent_of_worker_count(shape):
    """Every worker count reproduces the serial scan exactly.

    Ranges are scanned into private accumulators and merged. Areas and
    bounding boxes are integers, and the centroid sums add integer
    coordinates, which stay exact in a double, so no worker count can
    change the result.
    """
    from ncolor._backend import _impl

    rng = np.random.default_rng(abs(hash(shape)) % 2**32)
    labels = np.zeros(shape, np.int32)
    flat = labels.reshape(-1)
    chosen = rng.choice(flat.size, size=flat.size // 2, replace=False)
    flat[chosen] = rng.integers(1, max(4, flat.size // 500), chosen.size)

    serial = _impl.regionprops(labels, 0)
    for n_threads in (1, 2, 3, 4, 8):
        got = ncolor.Engine(n_threads=n_threads).regionprops(labels)
        for key, expected in serial.items():
            assert np.array_equal(got[key], expected), (shape, n_threads, key)


def test_regionprops_labels_absent_from_the_image():
    """A label with no pixels keeps a zero area and a zeroed bounding box."""
    labels = np.array([[0, 1, 0], [0, 0, 0], [3, 0, 3]], np.int32)
    props = ncolor.Engine(n_threads=4).regionprops(labels, n_labels=5)
    assert props["area"].tolist() == [1, 0, 2, 0, 0]
    for absent in (1, 3, 4):
        assert props["bbox_min"][absent].tolist() == [0, 0]
        assert props["bbox_max"][absent].tolist() == [0, 0]


def test_cc_label_empty_input():
    """All-zero mask → no components."""
    m = np.zeros((64, 64), dtype=np.uint16)
    labels, n = ncolor.connected_components(m, conn=2)
    assert n == 0
    assert int(labels.sum()) == 0


def test_cc_label_single_component():
    """Solid block → one component."""
    m = np.ones((64, 64), dtype=np.uint16)
    labels, n = ncolor.connected_components(m, conn=2)
    assert n == 1
    assert int(labels.min()) == 1 and int(labels.max()) == 1
