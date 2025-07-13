# test_files/test_ncolor.py
import os
from pathlib import Path
import itertools
from itertools import product

import skimage.io
import ncolor
from ncolor.format_labels import format_labels
from ncolor.label import is_sequential
import pytest
import numpy as np
import tempfile


# -------- helpers ----------------------------------------------------------

def load_masks():
    masks_dir = Path(__file__).parent
    return skimage.io.imread(masks_dir / "example.png")

# -------- synthetic mask generators --------------------------------------

def generate_mask(kind):
    """
    Return a masks array according to the requested kind.
    Shapes are small so tests stay fast.
    """
    if kind == "example":
        return load_masks()
    
    if kind == "empty":
        return np.zeros((5, 5), dtype=np.int32)

    if kind == "single":
        m = np.zeros((5, 5), dtype=np.int32)
        m[2:4, 2:4] = 1
        return m

    if kind == "two_disjoint":
        m = np.zeros((5, 5), dtype=np.int32)
        m[1, 1] = 1
        m[3, 3] = 2
        return m

    if kind == "memmap":
        # create a temporary on‑disk array that is *read‑only*
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".dat")
        tf.close()
        # write mask
        mm_w = np.memmap(tf.name, dtype=np.int32, mode="w+", shape=(5, 5))
        mm_w[:] = 0
        mm_w[2:4, 2:4] = 1
        mm_w.flush()
        # reopen read‑only
        mm_r = np.memmap(tf.name, dtype=np.int32, mode="r", shape=(5, 5))
        mm_r.setflags(write=False)
        return mm_r

    if kind == "disjoint_same":
        # One label (1) in three separate regions:
        #   • large component (4 × 4)  – area 16  (> min_area)
        #   • medium component (3 × 4) – area 12  (> min_area)
        #   • tiny component (single pixel) – area 1 (≤ min_area)
        m = np.zeros((10, 10), dtype=np.int32)
        m[0:4, 0:4] = 1      # large
        m[6:9, 0:4] = 1      # medium
        m[5, 5] = 1          # tiny
        return m
    raise ValueError(f"Unknown mask kind: {kind}")


# -------- parametrised test ------------------------------------------------

@pytest.mark.parametrize("mask_kind", ["example", "empty", "single", "two_disjoint", "memmap", "disjoint_same"])
@pytest.mark.parametrize("expand, greedy, return_n", list(itertools.product([0, 1], [0, 1], [0, 1])))
def test_ncolor_all_variants(mask_kind, expand, greedy, return_n):
    masks = generate_mask(mask_kind)
    # call ncolor.label; when return_n is True it returns (masks, n_colors)
    result = ncolor.label(masks, expand=expand, greedy=greedy, return_n=return_n)
    if return_n:
        ncolor_masks, nc = result
    else:
        ncolor_masks = result

    # basic sanity checks
    assert ncolor_masks.shape == masks.shape
    assert ncolor_masks.dtype.kind in "iu"           # integer type
    
    
# may not be needed, used in ncolor.label 
# def test_unique_nonzero():
#     masks = generate_mask("example")   # simple single‑object case
#     result = ncolor.unique_nonzero(masks)
#     assert (result == np.array([1], dtype=np.int32)).all()
    

# -------- tests for format_labels ----------------------------------------

@pytest.mark.parametrize("mask_kind", ["example", "empty", "single", "two_disjoint", "memmap", "disjoint_same"])
@pytest.mark.parametrize("clean", [False, True])
@pytest.mark.parametrize("ignore", [False, True])
@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("background", [None, 0])
@pytest.mark.parametrize("despur", [False, True])
def test_format_labels_variants(mask_kind, clean, ignore, verbose, background, despur):
    """
    Exercise format_labels across mask types and flag combinations.

    * clean toggles area/cleanup logic.
    * ignore toggles the special-background behaviour.
    """
    masks = generate_mask(mask_kind)
    formatted = format_labels(
        masks,
        clean=clean,
        ignore=ignore,
        verbose=verbose,
        background=background,
        despur=despur,
    )

    # shape and type preserved
    assert formatted.shape == masks.shape
    assert formatted.dtype.kind in "iu"

    # background is 0 after formatting (unless ignore=True in which case 1 is background)
    if ignore:
        assert np.min(formatted) in (0, 1)
    else:
        assert np.min(formatted) == 0

    # labels should be sequential (0,1,2,...) with no gaps
    uniq = np.sort(np.unique(formatted))
    # allow 0 or 1 to be background based on ignore flag
    expected_start = 0
    assert (uniq == np.arange(expected_start, expected_start + uniq.size)).all()

@pytest.mark.parametrize("mask_kind", ["example", "empty", "single", "two_disjoint", "memmap", "disjoint_same"])
def test_is_sequential(mask_kind):
    """
    Test the is_sequential function with various inputs.
    """
    masks = generate_mask(mask_kind)
    
    # Test with a single element array
    assert is_sequential(np.array([0]))
    
def run_tests():
    # Run all tests in this file
    pytest.main([str(Path(__file__).resolve())])
    
if __name__ == '__main__':
    run_tests()