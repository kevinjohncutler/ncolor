"""Parity tests for delete_spurs against a skimage + scipy reference.
Auto-skipped on installs without scikit-image or scipy."""
from __future__ import annotations

import numpy as np
import pytest

from ncolor.format import delete_spurs

skimage_morphology = pytest.importorskip("skimage.morphology")
scipy_ndimage = pytest.importorskip("scipy.ndimage")


def _ref_delete_spurs(mask, hole_threshold=5, *, mode="cardinal",
                      threshold=None, max_iter=-1):
    """skimage + scipy reference for delete_spurs.

    Mirrors the cpp implementation: hole-fill via remove_small_holes,
    then iterative pruning where a pixel is a spur when its
    fg-neighbour count is in [1, threshold). ``mode`` selects the
    connectivity ('cardinal' or 'total'); default ``threshold`` is
    ``ndim``.
    """
    pad = 1
    skel = skimage_morphology.remove_small_holes(
        np.pad(mask, pad, mode="constant"), hole_threshold
    )
    ndim = skel.ndim
    conn_arg = 1 if mode == "cardinal" else ndim
    thr = ndim if threshold is None else int(threshold)
    iters = 0
    while True:
        if max_iter >= 0 and iters >= max_iter:
            break
        connectivity = scipy_ndimage.generate_binary_structure(ndim, conn_arg)
        kernel = connectivity.astype(np.float32)
        kernel[tuple(np.array(kernel.shape) // 2)] = 0
        nb = scipy_ndimage.convolve(
            skel.astype(np.float32), kernel, mode="constant", cval=0
        )
        ep = (skel > 0) & (nb > 0) & (nb < thr)
        if int(ep.sum()) == 0:
            break
        skel = skel & ~ep
        iters += 1
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
    # Guard against both implementations trivially returning the input.
    removed = int(mask.sum()) - int(ours.sum())
    assert removed >= 10, (
        f"too few pixels pruned (in={int(mask.sum())}, out={int(ours.sum())})"
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
    """4D parity vs the skimage+scipy reference."""
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
    """5D smoke: same shape and bool dtype out, at least some pruning."""
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
    """hole_threshold actually selects which interior holes get filled."""
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:18, 2:18] = True
    mask[5, 5] = False           # 1-px hole
    mask[8:10, 8:10] = False     # 4-px hole
    mask[12:16, 12:16] = False   # 16-px hole

    a = delete_spurs(mask, hole_threshold=1)
    b = _ref_delete_spurs(mask, hole_threshold=1)
    assert np.array_equal(a, b)
    assert a[5, 5] == True
    assert a[8, 8] == False
    assert a[12, 12] == False

    a = delete_spurs(mask, hole_threshold=5)
    b = _ref_delete_spurs(mask, hole_threshold=5)
    assert np.array_equal(a, b)
    assert a[5, 5] == True
    assert a[8, 8] == True
    assert a[12, 12] == False

    a = delete_spurs(mask, hole_threshold=20)
    b = _ref_delete_spurs(mask, hole_threshold=20)
    assert np.array_equal(a, b)
    assert a[5, 5] == True and a[8, 8] == True and a[12, 12] == True


def test_delete_spurs_preserves_junction():
    """A cross of two 1-wide bars eats down to the junction pixel."""
    mask = np.zeros((15, 15), dtype=bool)
    mask[3:12, 7] = True
    mask[7, 3:12] = True
    in_fg = int(mask.sum())

    a = delete_spurs(mask, hole_threshold=5)
    b = _ref_delete_spurs(mask, hole_threshold=5)
    assert np.array_equal(a, b)

    out_fg = int(a.sum())
    assert out_fg < in_fg
    assert out_fg == int(b.sum())


def test_delete_spurs_total_mode_removes_diagonal_spurs_3d():
    """Under ``mode='total'``, single-voxel spurs touching a 3D ball at
    a face / edge / vertex are all pruned (each has total=1 fg
    neighbour). Locks in the full-diagonal rule's symmetric treatment
    of contact types."""
    grids = np.indices((16, 16, 16))
    cz, cy, cx = 8, 8, 8
    base = ((grids[0] - cz) ** 2 + (grids[1] - cy) ** 2 + (grids[2] - cx) ** 2) <= 16

    spurs = {
        "face":   (8, 8, 13),    # +1 along one axis
        "edge":   (8, 9, 13),    # +1 along two axes
        "vertex": (9, 9, 13),    # +1 along three axes
    }
    for kind, p in spurs.items():
        assert not base[p], f"{kind} spur must start outside the ball"
        m = base.copy()
        m[p] = True
        out = delete_spurs(m, hole_threshold=0, mode="total")
        assert not out[p], f"{kind}-touching spur was not pruned under total mode"


def test_delete_spurs_cardinal_mode_keeps_diagonal_only_spurs():
    """Cardinal mode keys on face neighbours, so a voxel attached only
    via a diagonal contact has cardinal count == 0 — treated as
    isolated and preserved. (Use ``mode='total'`` if you want diagonal
    contacts to count as connections.)"""
    grids = np.indices((16, 16, 16))
    cz, cy, cx = 8, 8, 8
    base = ((grids[0] - cz) ** 2 + (grids[1] - cy) ** 2 + (grids[2] - cx) ** 2) <= 16

    # Vertex-attached spur: only touches the ball via a vertex (3-axis
    # diagonal). face count = 0.
    p = (9, 9, 13)
    assert not base[p]
    m = base.copy()
    m[p] = True
    out = delete_spurs(m, hole_threshold=0, mode="cardinal")
    assert out[p], "cardinal mode should preserve diagonal-only attachments"
    # And a face-attached spur on the same ball IS pruned, even in cardinal mode.
    p_face = (8, 8, 13)
    m2 = base.copy()
    m2[p_face] = True
    out2 = delete_spurs(m2, hole_threshold=0, mode="cardinal")
    assert not out2[p_face], "cardinal mode should prune face-attached spurs"


def test_delete_spurs_cardinal_catches_flat_row_protrusion():
    """The motivating case for cardinal mode: a single pixel sticking up
    out of a flat row has face=1 + 2 diagonals (total=3). Under total
    mode it has 3 neighbours, not an endpoint. Under cardinal mode its
    face count is 1 < ndim=2, so it gets pruned along with the row's
    own endpoints."""
    H, W = 7, 21
    m = np.zeros((H, W), dtype=bool)
    m[4, 2:19] = True       # flat row
    sticky = (3, 10)
    m[sticky] = True
    out_total = delete_spurs(m, hole_threshold=0, mode="total")
    out_card  = delete_spurs(m, hole_threshold=0, mode="cardinal")
    assert out_total[sticky], "total mode preserves the sticky pixel"
    assert not out_card[sticky], "cardinal mode prunes the sticky pixel"


def test_delete_spurs_threshold_controls_aggressiveness():
    """Raising ``threshold`` peels more iterations / more pixels.
    Lowering it (≤1) effectively disables endpoint pruning."""
    rng = np.random.default_rng(0)
    mask = _make_2d_skeleton(rng)
    out_default = delete_spurs(mask, hole_threshold=0)
    out_high    = delete_spurs(mask, hole_threshold=0, threshold=4)
    out_off     = delete_spurs(mask, hole_threshold=0, threshold=1)
    assert int(out_high.sum()) <= int(out_default.sum())
    # threshold=1 means "spur iff count in [1, 1)" — empty set, no pruning.
    assert int(out_off.sum()) == int(mask.sum())


def test_delete_spurs_max_iter_caps_loop():
    """``max_iter=N`` runs at most N pruning passes. A long 2D row
    needs many iterations to fully peel — capping should leave the
    middle of the row intact."""
    m = np.zeros((5, 41), dtype=bool)
    m[2, 0:41] = True   # row of 41 pixels, both ends are endpoints
    out_unbounded = delete_spurs(m, hole_threshold=0, mode="cardinal", max_iter=-1)
    out_1step     = delete_spurs(m, hole_threshold=0, mode="cardinal", max_iter=1)
    out_5step     = delete_spurs(m, hole_threshold=0, mode="cardinal", max_iter=5)
    # Unbounded peels until only the (now-isolated) middle pixel remains.
    assert int(out_unbounded.sum()) == 1
    # One step peels both endpoints → row of 39.
    assert int(out_1step.sum()) == 39
    # Five steps peel 5 from each end → row of 31.
    assert int(out_5step.sum()) == 31


def test_delete_spurs_closed_loop_unchanged():
    """A square loop has no endpoints; pruning is a no-op."""
    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3] = True
    mask[3:9, 8] = True
    mask[3, 3:9] = True
    mask[8, 3:9] = True
    a = delete_spurs(mask, hole_threshold=1)
    b = _ref_delete_spurs(mask, hole_threshold=1)
    assert np.array_equal(a, b)
    assert np.array_equal(a, mask)
