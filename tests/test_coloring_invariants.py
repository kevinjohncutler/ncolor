"""Invariants every ncolor.label output must satisfy, for any input/p/balance.

This test exists because we once had a debate about whether two visually-
adjacent yellow cells in a viridis-rendered figure were a real coloring
violation or a viridis-perception artifact (it was the latter — the BFS
heuristic had bumped n_colors from 4 to 5, and viridis(vmax=4) was
clipping color 5 to the same yellow shade as color 4). Either way, the
ground-truth invariant — *no two cells whose pixels are 8-connected
across the unexpanded mask share a color* — is something a test should
enforce, not something we eyeball.
"""
from __future__ import annotations

import numpy as np
import pytest

import ncolor


def _adj_same_color_pairs(label_image: np.ndarray, colored: np.ndarray) -> int:
    """Brute-force pixel-pair check (8-conn). Returns the count of adjacent
    pixel pairs in DIFFERENT cells that share a nonzero color. A valid
    4-/5-/whatever-coloring of a label image must have this == 0."""
    H, W = label_image.shape
    bad = 0
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if dx >= 0:
            a_lab = label_image[: H - dy, : W - dx]
            b_lab = label_image[dy:, dx:]
            a_col = colored[: H - dy, : W - dx]
            b_col = colored[dy:, dx:]
        else:
            adx = -dx
            a_lab = label_image[: H - dy, adx:]
            b_lab = label_image[dy:, : W - adx]
            a_col = colored[: H - dy, adx:]
            b_col = colored[dy:, : W - adx]
        diff_cell = (a_lab != b_lab) & (a_lab > 0) & (b_lab > 0)
        same_color = (a_col == b_col) & (a_col > 0)
        bad += int((diff_cell & same_color).sum())
    return bad


def _make_packed_circles(size: int, n: int, seed: int) -> np.ndarray:
    """Densely-packed non-overlapping disks — same generator as the
    examples/ benches but with a per-test seed."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), dtype=np.int32)
    yy, xx = np.ogrid[:size, :size]
    radius_max = max(4, size // 12)
    label_id, attempts = 1, 0
    while label_id <= n and attempts < n * 10:
        attempts += 1
        r = int(rng.integers(max(3, radius_max // 2), radius_max))
        cy = int(rng.integers(-r, size + r))
        cx = int(rng.integers(-r, size + r))
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        if np.any(labels[m] != 0):
            continue
        labels[m] = label_id
        label_id += 1
    return labels


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("balance", [True, False])
@pytest.mark.parametrize("seed", range(8))
def test_no_adjacent_same_color(p, balance, seed):
    """For every (p, balance, image-seed), ncolor.label must produce a
    coloring with zero adjacent same-color cell pairs in the unexpanded
    input mask. This is the ground-truth correctness invariant."""
    img = _make_packed_circles(256, 60, seed)
    out = ncolor.label(img, p=p, balance=balance, expand=True)
    bad = _adj_same_color_pairs(img, out)
    assert bad == 0, (
        f"ncolor.label(p={p}, balance={balance}, seed={seed}) produced "
        f"{bad} adjacent same-color pixel-pairs — coloring is invalid."
    )


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("balance", [True, False])
def test_no_adjacent_same_color_under_label_permutation(p, balance):
    """Permuting the input label IDs must preserve the no-conflict
    invariant. The graph topology is identical, only node IDs change —
    the algorithm should still find a valid coloring."""
    base = _make_packed_circles(256, 60, seed=42)
    n_cells = int(base.max())
    for perm_seed in range(8):
        rng = np.random.default_rng(perm_seed)
        perm = rng.permutation(n_cells) + 1
        lut = np.zeros(n_cells + 1, dtype=base.dtype)
        lut[1:] = perm
        permuted = lut[base]
        out = ncolor.label(permuted, p=p, balance=balance, expand=True)
        bad = _adj_same_color_pairs(permuted, out)
        assert bad == 0, (
            f"ncolor.label(p={p}, balance={balance}) on permuted labels "
            f"(perm_seed={perm_seed}) produced {bad} adjacent same-color "
            f"pixel-pairs."
        )
