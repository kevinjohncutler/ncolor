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
