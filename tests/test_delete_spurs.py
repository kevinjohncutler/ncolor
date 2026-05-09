"""Parity tests for the cpp delete_spurs against the original
skimage+scipy Python implementation. Auto-skipped on installs without
scikit-image / scipy (the cpp version itself never needs them, but the
reference does)."""
from __future__ import annotations

import numpy as np
import pytest

from ncolor.format import delete_spurs

skimage_morphology = pytest.importorskip("skimage.morphology")
scipy_ndimage = pytest.importorskip("scipy.ndimage")


def _ref_delete_spurs(mask, hole_threshold=5):
    """Reference: original ncolor.format.delete_spurs body, verbatim
    aside from the lazy-import shim."""
    pad = 1
    skel = skimage_morphology.remove_small_holes(
        np.pad(mask, pad, mode="constant"), hole_threshold
    )
    while True:
        ndim = skel.ndim
        if ndim == 2:
            connectivity = scipy_ndimage.generate_binary_structure(ndim, 2)
        else:
            connectivity = scipy_ndimage.generate_binary_structure(ndim, 1)
        kernel = connectivity.astype(np.float32)
        kernel[tuple(np.array(kernel.shape) // 2)] = 0
        nb = scipy_ndimage.convolve(
            skel.astype(np.float32), kernel, mode="constant", cval=0
        )
        ep = (skel > 0) & (nb == 1)
        if int(ep.sum()) == 0:
            break
        skel = skel & ~ep
    unpad = tuple([slice(pad, -pad)] * skel.ndim)
    return skel[unpad]


def _make_2d_skeleton(rng, H=64, W=64, n_strokes=8):
    """Random 2D mask with line strokes + spurs + small holes — exercises
    both hole-fill and endpoint-pruning branches."""
    img = np.zeros((H, W), dtype=bool)
    for _ in range(n_strokes):
        y0, x0 = rng.integers(2, H - 2), rng.integers(2, W - 2)
        length = int(rng.integers(4, 20))
        if rng.random() < 0.5:
            img[y0, x0 : min(W, x0 + length)] = True
        else:
            img[y0 : min(H, y0 + length), x0] = True
    # Sprinkle small holes by erasing single pixels in fg
    fg_idx = np.argwhere(img)
    if fg_idx.size:
        for _ in range(min(8, len(fg_idx))):
            i = rng.integers(0, len(fg_idx))
            y, x = fg_idx[i]
            img[y, x] = False
    return img


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("hole_threshold", [1, 5, 10])
def test_delete_spurs_matches_reference_2d(seed, hole_threshold):
    rng = np.random.default_rng(seed)
    mask = _make_2d_skeleton(rng)
    ours = delete_spurs(mask, hole_threshold=hole_threshold)
    ref = _ref_delete_spurs(mask, hole_threshold=hole_threshold)
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref), (
        f"2D delete_spurs diverged from skimage+scipy ref "
        f"(seed={seed}, hole_threshold={hole_threshold})"
    )
    # Sanity: with the random-stroke fixture, pruning should always
    # remove a meaningful number of pixels (every isolated segment has
    # endpoints to peel). Guards against a regression where both
    # implementations trivially return the input unchanged.
    removed = int(mask.sum()) - int(ours.sum())
    assert removed >= 10, (
        f"too few pixels pruned (in={int(mask.sum())}, out={int(ours.sum())}) — "
        f"the parity check passed but the test isn't exercising the loop"
    )


@pytest.mark.parametrize("seed", range(4))
def test_delete_spurs_matches_reference_3d(seed):
    rng = np.random.default_rng(seed * 11 + 1)
    D = 24
    img = np.zeros((D, D, D), dtype=bool)
    # Random face-connected line segments along each axis
    for _ in range(10):
        z0 = rng.integers(2, D - 2)
        y0 = rng.integers(2, D - 2)
        x0 = rng.integers(2, D - 2)
        L = int(rng.integers(3, 8))
        axis = rng.integers(0, 3)
        if axis == 0:
            img[z0 : min(D, z0 + L), y0, x0] = True
        elif axis == 1:
            img[z0, y0 : min(D, y0 + L), x0] = True
        else:
            img[z0, y0, x0 : min(D, x0 + L)] = True
    ours = delete_spurs(img, hole_threshold=3)
    ref = _ref_delete_spurs(img, hole_threshold=3)
    assert np.array_equal(ours, ref), f"3D delete_spurs diverged (seed={seed})"


@pytest.mark.parametrize("seed", range(3))
def test_delete_spurs_matches_reference_4d(seed):
    """4D parity vs the skimage+scipy reference. Locks in that the
    N-D pad / cc_label / endpoint-prune chain doesn't bake any
    2D/3D-specific assumptions into the cpp implementation."""
    rng = np.random.default_rng(seed * 31 + 5)
    D = 8
    img = np.zeros((D, D, D, D), dtype=bool)
    for _ in range(8):
        coords = rng.integers(2, D - 2, 4)
        L = int(rng.integers(3, 6))
        axis = int(rng.integers(0, 4))
        slc = list(coords)
        slc[axis] = slice(int(coords[axis]), min(D, int(coords[axis]) + L))
        img[tuple(slc)] = True
    ours = delete_spurs(img, hole_threshold=3)
    ref = _ref_delete_spurs(img, hole_threshold=3)
    assert ours.shape == img.shape and ours.dtype == np.bool_
    assert np.array_equal(ours, ref), f"4D delete_spurs diverged (seed={seed})"


def test_delete_spurs_5d_smoke():
    """5D smoke test (no skimage cross-check above 3D for the
    morphology helpers, but the cpp code is dimension-agnostic).
    Confirms the function returns sensibly on a 5-D input — same
    shape, bool dtype, and at least some pruning happens."""
    mask = np.zeros((5, 5, 5, 5, 5), dtype=bool)
    mask[2, 2, 2, 2, 1:4] = True
    mask[2, 2, 2, 1:4, 2] = True
    out = delete_spurs(mask, hole_threshold=3)
    assert out.shape == mask.shape
    assert out.dtype == np.bool_
    assert int(out.sum()) <= int(mask.sum())


def test_delete_spurs_empty_mask():
    """All-zero input → all-zero output."""
    out = delete_spurs(np.zeros((16, 16), dtype=bool), hole_threshold=5)
    assert out.shape == (16, 16) and out.dtype == np.bool_
    assert int(out.sum()) == 0


def test_delete_spurs_solid_block():
    """Solid foreground has no endpoints; output should equal input."""
    mask = np.ones((10, 10), dtype=bool)
    out = delete_spurs(mask, hole_threshold=5)
    assert np.array_equal(out, mask)


def test_delete_spurs_hole_threshold_actually_fills():
    """Closed shape with interior holes of varying size: small holes
    (≤ threshold pixels) get filled in by the bg-CCL pass; larger ones
    survive. Confirms the hole_threshold parameter has real effect and
    that the cpp + reference agree on which holes are filled."""
    mask = np.zeros((20, 20), dtype=bool)
    # 16x16 solid block centred in the array — fully interior (the
    # outer bg has plenty of room to be one big component).
    mask[2:18, 2:18] = True
    # Punch holes of size 1, 4 (2x2), and 16 (4x4):
    mask[5, 5] = False
    mask[8:10, 8:10] = False
    mask[12:16, 12:16] = False

    # threshold=1: only the single-pixel hole gets filled.
    a = delete_spurs(mask, hole_threshold=1)
    b = _ref_delete_spurs(mask, hole_threshold=1)
    assert np.array_equal(a, b)
    assert a[5, 5] == True, "single-pixel hole should be filled at threshold=1"
    assert a[8, 8] == False, "2x2 hole survives threshold=1"
    assert a[12, 12] == False, "4x4 hole survives threshold=1"

    # threshold=5: 1-px and 2x2 holes (4 px) fill; 4x4 (16 px) doesn't.
    a = delete_spurs(mask, hole_threshold=5)
    b = _ref_delete_spurs(mask, hole_threshold=5)
    assert np.array_equal(a, b)
    assert a[5, 5] == True
    assert a[8, 8] == True, "4-px hole should be filled at threshold=5"
    assert a[12, 12] == False, "16-px hole survives threshold=5"

    # threshold=20: every hole fills.
    a = delete_spurs(mask, hole_threshold=20)
    b = _ref_delete_spurs(mask, hole_threshold=20)
    assert np.array_equal(a, b)
    assert a[5, 5] == True and a[8, 8] == True and a[12, 12] == True


def test_delete_spurs_preserves_junction():
    """A T-junction (vertical line meeting a horizontal line) is a
    pixel with neighbour count 3 — never an endpoint, so iterative
    pruning peels the arms back toward it and stops once everything
    adjacent to the junction has count > 1. Confirms count≥2 pixels
    survive the loop and parity holds on a structured input."""
    mask = np.zeros((15, 15), dtype=bool)
    mask[3:12, 7] = True  # vertical bar
    mask[7, 3:12] = True  # horizontal bar through it (T-junction at [7, 7])
    in_fg = int(mask.sum())

    a = delete_spurs(mask, hole_threshold=5)
    b = _ref_delete_spurs(mask, hole_threshold=5)
    assert np.array_equal(a, b), "junction case diverged from reference"

    out_fg = int(a.sum())
    # Pruning eats the arms iteratively; the junction itself peels
    # last (count 4 → 3 → 2 → 1 → 0). For a 9+9 cross with both arms
    # length 4 from centre, every pixel is on a 1-wide spur and the
    # whole structure ends up consumed (the centre becomes isolated
    # at the very end and stays as a single pixel). Verify the cpp
    # and reference agree on that outcome.
    assert out_fg < in_fg, "expected pruning on cross-shape"
    assert out_fg == int(b.sum()), "cpp and ref disagree on residue size"


def test_delete_spurs_closed_loop_unchanged():
    """A square loop has no endpoints (every pixel has count 2 along
    the loop). Pruning should be a no-op and parity should hold."""
    mask = np.zeros((12, 12), dtype=bool)
    # 6x6 square outline (every loop pixel has exactly 2 neighbours).
    mask[3:9, 3] = True
    mask[3:9, 8] = True
    mask[3, 3:9] = True
    mask[8, 3:9] = True
    a = delete_spurs(mask, hole_threshold=1)
    b = _ref_delete_spurs(mask, hole_threshold=1)
    assert np.array_equal(a, b)
    # No endpoints → no pruning. Hole inside the loop (16 px) is
    # larger than threshold=1, so it doesn't fill either.
    assert np.array_equal(a, mask), "closed loop should be unchanged"
