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

# format_labels(clean=True) routes through scikit-image; auto-skip when
# the [clean] extra isn't installed (default install is cpp-only).
needs_clean_extra = pytest.mark.skipif(
    importlib.util.find_spec("skimage") is None,
    reason="format_labels(clean=True) needs ncolor[clean] (scikit-image)",
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


@needs_clean_extra
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


@needs_experimental
@needs_clean_extra
def test_experimental_formatter_falls_back_when_clean_requested():
    arr = np.array([[0, 1], [0, 0]], dtype=np.int32)
    ref = format_labels(arr.copy(), clean=True, min_area=2, background=0)
    fast = _format_labels_experimental(arr.copy(), clean=True, min_area=2, background=0)
    assert np.array_equal(ref, fast)
