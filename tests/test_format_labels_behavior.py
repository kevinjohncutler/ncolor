import importlib.util

import numpy as np
import pytest

from ncolor.format import format_labels

# label_experimental_wip is local dev scratch (untracked, not shipped). When
# the user has it on disk the parity-vs-reference tests below run; on a clean
# checkout (e.g. CI building from actions/checkout) they skip.
try:
    from ncolor.label_experimental_wip import _format_labels_experimental
    HAS_EXPERIMENTAL = True
except ImportError:
    _format_labels_experimental = None
    HAS_EXPERIMENTAL = False

needs_experimental = pytest.mark.skipif(
    not HAS_EXPERIMENTAL,
    reason="ncolor.label_experimental_wip is dev scratch, not shipped in the package",
)

# Parity-vs-skimage reference checks: only the parity tests below
# need scikit-image + fastremap to construct the reference output.
# format_labels(clean=True) itself runs cpp-only and works without them.
needs_skimage_ref = pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None
    or importlib.util.find_spec("fastremap") is None,
    reason="parity-vs-skimage reference needs scikit-image + fastremap",
)


def test_format_labels_handles_negative_background():
    arr = np.array(
        [
            [-1, -1, 5],
            [-1, 2, -1],
            [3, -1, -1],
        ],
        dtype=np.int32,
    )
    out = format_labels(arr.copy(), background=None)
    assert out.min() == 0
    assert 0 in out
    nonzero = np.unique(out[out > 0])
    assert np.array_equal(nonzero, np.arange(1, nonzero.size + 1))


def test_format_labels_clean_removes_single_pixel_regions():
    arr = np.array(
        [
            [0, 1, 0],
            [0, 0, 2],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    out = format_labels(arr.copy(), clean=True, min_area=2, background=0)
    # Single-pixel label (value 1) should be removed
    assert out[0, 1] == 0
    # Only one remaining label, remapped to 1, occupying four pixels
    assert set(np.unique(out)) <= {0, 1}
    assert np.count_nonzero(out == 1) == 4


@needs_experimental
def test_experimental_formatter_matches_reference_random_inputs():
    """Both formatters compact labels to 1..N with bg=0 and the same
    bg-mask. They differ only in the permutation of new labels (the
    production cpp path uses ascending-source numbering; the experimental
    path uses fastremap's input-order numbering), so we check semantic
    equivalence rather than bit-equality."""
    rng = np.random.default_rng(42)
    for _ in range(25):
        arr = rng.integers(-5, 15, size=(10, 12), dtype=np.int32)
        ref = format_labels(arr.copy(), background=None)
        fast = _format_labels_experimental(arr.copy(), background=None)
        # Same bg pattern (zero where the legacy min-shift maps to bg).
        assert np.array_equal(ref == 0, fast == 0)
        # Both compact to {0, 1, ..., N} with the same N.
        ref_unique = set(np.unique(ref).tolist())
        fast_unique = set(np.unique(fast).tolist())
        assert ref_unique == fast_unique
        n = max(ref_unique)
        assert ref_unique == set(range(n + 1))
        # Each source label maps to exactly one target label in each
        # output (i.e. both are valid bijective compactions of the same
        # underlying partition).
        for src in np.unique(arr):
            ref_vals = np.unique(ref[arr == src])
            fast_vals = np.unique(fast[arr == src])
            assert ref_vals.size == 1 and fast_vals.size == 1


@needs_experimental
def test_format_labels_first_seen_matches_fastremap_bit_for_bit():
    """When opted-in, first_seen=True should produce bit-identical output
    to fastremap.renumber (which is what _format_labels_experimental
    relies on at the end of its pipeline). This guards the legacy
    compatibility path against regressions."""
    rng = np.random.default_rng(7)
    for _ in range(25):
        arr = rng.integers(-5, 15, size=(10, 12), dtype=np.int32)
        prod = format_labels(arr.copy(), background=None, first_seen=True)
        legacy = _format_labels_experimental(arr.copy(), background=None)
        assert np.array_equal(prod, legacy)


def _ref_clean_with_skimage(arr, min_area, background=0):
    """Reference clean=True implementation using the original
    skimage+fastremap pipeline. Used to lock the cpp rewrite to
    bit-identical output."""
    from skimage import measure
    import fastremap

    labels = arr.astype(np.int32, copy=True)
    if background is None:
        m = int(np.min(labels))
        background = m if m < 0 else 0
    if background != 0:
        labels -= background
        background = 0
    labels = labels.astype(np.uint32)

    inds = np.unique(labels)
    for j in inds[inds > background]:
        mask = labels == j
        lbl = measure.label(mask)
        regions = measure.regionprops(lbl)
        regions.sort(key=lambda x: x.area, reverse=True)
        if len(regions) == 0:
            continue
        if len(regions) > 1:
            for rg in regions[1:]:
                if rg.area < min_area:
                    labels[tuple(rg.coords.T)] = background
                else:
                    labels[tuple(rg.coords.T)] = np.max(labels) + 1
        rg0 = regions[0]
        if rg0.area <= min_area:
            labels[tuple(rg0.coords.T)] = background

    fastremap.renumber(labels, in_place=True)
    return fastremap.refit(labels)


def _make_clean_test_inputs(rng, n_cases=15):
    """Generate label images with disjoint components, single pixels,
    and varied geometry to exercise the clean=True logic."""
    cases = []
    for _ in range(n_cases):
        H = int(rng.integers(20, 64))
        W = int(rng.integers(20, 64))
        arr = np.zeros((H, W), dtype=np.uint16)
        n_labels = int(rng.integers(3, 12))
        yy, xx = np.ogrid[:H, :W]
        for k in range(1, n_labels + 1):
            n_blobs = int(rng.integers(1, 4))  # may be disjoint
            for _ in range(n_blobs):
                cy, cx = rng.integers(0, H), rng.integers(0, W)
                r = int(rng.integers(1, 6))
                arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = k
        # Sprinkle a few stray single pixels of various labels to test
        # the small-region removal branch.
        for _ in range(int(rng.integers(0, 5))):
            y, x = rng.integers(0, H), rng.integers(0, W)
            arr[y, x] = int(rng.integers(1, n_labels + 1))
        cases.append(arr)
    return cases


@needs_skimage_ref
@pytest.mark.parametrize("min_area", [3, 9, 25])
def test_clean_path_matches_skimage_reference(min_area):
    """format_labels(clean=True) is bit-identical to the skimage+fastremap
    reference across a varied corpus of 2D disjoint-region label images."""
    rng = np.random.default_rng(min_area * 17)
    for arr in _make_clean_test_inputs(rng, n_cases=20):
        ours = format_labels(arr.copy(), clean=True, min_area=min_area, background=0)
        ref = _ref_clean_with_skimage(arr.copy(), min_area=min_area, background=0)
        assert ours.dtype == ref.dtype, f"dtype mismatch: {ours.dtype} vs {ref.dtype}"
        assert np.array_equal(ours, ref), (
            f"clean output diverged from skimage ref (min_area={min_area})"
        )


def _make_clean_3d_inputs(rng, n_cases=8, D=24):
    """3D label volumes with disjoint blobs and stray voxels."""
    cases = []
    for _ in range(n_cases):
        arr = np.zeros((D, D, D), dtype=np.uint16)
        zz, yy, xx = np.ogrid[:D, :D, :D]
        n_labels = int(rng.integers(3, 9))
        for k in range(1, n_labels + 1):
            n_blobs = int(rng.integers(1, 3))
            for _ in range(n_blobs):
                cz, cy, cx = rng.integers(0, D, 3)
                r = int(rng.integers(1, 4))
                arr[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = k
        # Single-voxel strays for the small-region branch.
        for _ in range(int(rng.integers(0, 5))):
            z, y, x = rng.integers(0, D, 3)
            arr[z, y, x] = int(rng.integers(1, n_labels + 1))
        cases.append(arr)
    return cases


@needs_skimage_ref
@pytest.mark.parametrize("min_area", [2, 5, 15])
def test_clean_path_matches_skimage_reference_3d(min_area):
    """3D parity vs the skimage reference."""
    rng = np.random.default_rng(min_area * 23 + 1)
    for arr in _make_clean_3d_inputs(rng, n_cases=8):
        ours = format_labels(arr.copy(), clean=True, min_area=min_area, background=0)
        ref = _ref_clean_with_skimage(arr.copy(), min_area=min_area, background=0)
        assert ours.dtype == ref.dtype, f"dtype mismatch: {ours.dtype} vs {ref.dtype}"
        assert np.array_equal(ours, ref), (
            f"3D clean output diverged from skimage ref (min_area={min_area})"
        )


def test_clean_path_4d_smoke():
    """4D smoke: skimage.measure.regionprops caps at 3D, so no parity
    cross-check; just verify the cpp path runs and applies the rules."""
    arr = np.zeros((6, 6, 6, 6), dtype=np.uint16)
    arr[1:4, 1:4, 1:4, 1:4] = 1
    arr[5, 5, 5, 5] = 2  # isolated single-voxel speckle, should drop
    out = format_labels(arr, clean=True, min_area=2, background=0)
    assert out.shape == arr.shape
    assert int(out.max()) == 1
    assert out[5, 5, 5, 5] == 0
    assert (out[1:4, 1:4, 1:4, 1:4] == 1).all()


@needs_experimental
@needs_skimage_ref
def test_experimental_formatter_falls_back_when_clean_requested():
    arr = np.array([[0, 1], [0, 0]], dtype=np.int32)
    ref = format_labels(arr.copy(), clean=True, min_area=2, background=0)
    fast = _format_labels_experimental(arr.copy(), clean=True, min_area=2, background=0)
    assert np.array_equal(ref, fast)
