"""Coverage for ncolor public-API combinatorics: return_n / return_lut /
return_conflicts flag combos, get_lut, and 2D / expand toggles.

Companion to tests/test_ncolor_3d.py (3D smoke) and test_files/test_ncolor.py
(parametrized mask-kind coverage)."""

import numpy as np
import pytest

import ncolor


def _two_circle_2d():
    """Two disjoint blobs in a small 2D image."""
    arr = np.zeros((16, 32), dtype=np.int32)
    arr[2:6, 2:6] = 1
    arr[10:14, 24:28] = 2
    return arr


def test_connect_returns_adjacency_pairs():
    """ncolor.connect is part of the public top-level API. Returns the
    unique (lo, hi) label-adjacency pairs as an (M, 2) int array."""
    arr = np.array([[0, 1, 1, 2],
                    [0, 1, 2, 2],
                    [3, 3, 0, 0]], dtype=np.int32)
    pairs = ncolor.connect(arr, conn=2)
    # Expect: (1,2), (1,3), (2,3) all touching under 8-conn.
    pair_set = {tuple(p) for p in pairs.tolist()}
    assert pair_set == {(1, 2), (1, 3), (2, 3)}


def _adjacent_strips():
    """Touching strips with a zero border → forces non-trivial coloring (≥2 colors)
    while preserving label 1 (format_labels shifts min(label) to 0, so we need
    an explicit zero background)."""
    arr = np.zeros((12, 14), dtype=np.int32)
    arr[1:11, 1:5] = 1
    arr[1:11, 5:9] = 2
    arr[1:11, 9:13] = 3
    return arr


def test_label_return_n():
    arr = _adjacent_strips()
    out, n = ncolor.label(arr, expand=False, return_n=True)
    assert out.shape == arr.shape
    assert n == int(out.max())
    assert n >= 2  # adjacent strips → ≥2 colors


def test_label_return_lut():
    arr = _adjacent_strips()
    lut = ncolor.label(arr, expand=False, return_lut=True)
    # LUT length is max_label + 1, lut[0] == 0 (background)
    assert lut.shape == (int(arr.max()) + 1,)
    assert lut[0] == 0
    assert (lut[1:] > 0).all()


def test_label_return_lut_with_n():
    arr = _adjacent_strips()
    lut, n = ncolor.label(arr, expand=False, return_lut=True, return_n=True)
    assert lut.shape == (int(arr.max()) + 1,)
    assert n == int(lut.max())


def test_label_return_lut_with_conflicts():
    arr = _adjacent_strips()
    lut, conflicts = ncolor.label(
        arr, expand=False, return_lut=True, return_conflicts=True
    )
    assert lut.shape == (int(arr.max()) + 1,)
    assert conflicts == 0


def test_label_return_lut_with_n_and_conflicts():
    arr = _adjacent_strips()
    lut, n, conflicts = ncolor.label(
        arr,
        expand=False,
        return_lut=True,
        return_n=True,
        return_conflicts=True,
    )
    assert n == int(lut.max())
    assert conflicts == 0


def test_label_return_n_with_conflicts():
    arr = _adjacent_strips()
    out, n, conflicts = ncolor.label(
        arr, expand=False, return_n=True, return_conflicts=True
    )
    assert conflicts == 0
    assert n == int(out.max())


def test_label_return_conflicts_only():
    arr = _adjacent_strips()
    out, conflicts = ncolor.label(arr, expand=False, return_conflicts=True)
    assert conflicts == 0
    assert out.shape == arr.shape


def test_get_lut_matches_label_lut():
    arr = _adjacent_strips()
    lut_a = ncolor.get_lut(arr, expand=False)
    lut_b = ncolor.label(arr, expand=False, return_lut=True)
    np.testing.assert_array_equal(lut_a, lut_b)


def test_label_single_component_uses_one_color():
    """Single connected label → empty adjacency → trivial 1-color path."""
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[2:6, 2:6] = 1
    out = ncolor.label(arr, expand=False)
    # Background remains 0; foreground gets exactly one color
    assert (out[arr == 0] == 0).all()
    assert (out[arr != 0] == 1).all()


def test_label_disjoint_components_share_color_or_not():
    """Disjoint components: coloring is valid (no shared edge means free choice)."""
    arr = _two_circle_2d()
    out = ncolor.label(arr, expand=False)
    # Both blobs colored, no conflicts (no edge between them)
    assert (out[arr == 0] == 0).all()
    assert (out[arr != 0] > 0).all()


def test_label_format_input_false_passthrough():
    """format_input=False skips relabel; caller is asserting input is already 1..N."""
    arr = _adjacent_strips().astype(np.int32)  # already 1..N
    out = ncolor.label(arr, expand=False, format_input=False)
    assert out.shape == arr.shape
    assert out.max() >= 2


def test_label_3d_no_conflicts():
    """Small 3D label volume — covers the 3D 26-conn path in connect()."""
    arr = np.zeros((6, 6, 6), dtype=np.int32)
    arr[0:3, 0:3, 0:3] = 1
    arr[0:3, 0:3, 3:6] = 2  # touches label 1 along axis 2
    arr[3:6, 3:6, 0:3] = 3
    out, conflicts = ncolor.label(arr, expand=False, return_conflicts=True)
    assert conflicts == 0
    assert out.shape == arr.shape


@pytest.mark.parametrize("expand", [False, True])
def test_label_check_conflicts_passes_on_valid(expand):
    arr = _adjacent_strips()
    # Should not raise
    ncolor.label(arr, expand=expand, check_conflicts=True)


def test_init_unknown_attribute_raises():
    with pytest.raises(AttributeError, match="no attribute 'does_not_exist'"):
        ncolor.does_not_exist


# ----------------------------------------------------------------------
# wrap=True (toroidal expand + toroidal find_pairs)
# ----------------------------------------------------------------------


def _circles_2d_dense(seed=0, H=128, W=128, n=20):
    """Random circles to populate a colony — useful for testing wrap effects."""
    rng = np.random.default_rng(seed)
    m = np.zeros((H, W), dtype=np.int32)
    for i in range(1, n + 1):
        cy, cx = rng.integers(20, H - 20), rng.integers(20, W - 20)
        r = int(rng.integers(4, 10))
        yy, xx = np.ogrid[:H, :W]
        m[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = i
    return m


@pytest.mark.parametrize("p", [1, 2])
def test_label_wrap_returns_valid_coloring(p):
    """wrap=True must still produce a valid 4-coloring under both metrics."""
    m = _circles_2d_dense()
    out, conflicts = ncolor.label(m, p=p, wrap=True, return_conflicts=True)
    assert conflicts == 0
    assert out.shape == m.shape
    assert out.dtype == np.uint8
    # bg pattern preserved.
    assert np.array_equal(out == 0, m == 0)


def test_label_wrap_output_differs_from_non_wrap():
    """wrap=True should generally produce a different coloring than the
    non-wrap path (because the adjacency graph gains wrap-around edges).
    Asserts there's *some* difference rather than a specific permutation."""
    m = _circles_2d_dense(seed=42)
    no_wrap = ncolor.label(m, p=1, wrap=False)
    yes_wrap = ncolor.label(m, p=1, wrap=True)
    # Both valid, both compact 0..n_colors.
    assert no_wrap.max() <= 4 and yes_wrap.max() <= 4
    # Foreground bg patterns identical (only the colors change).
    assert np.array_equal(no_wrap == 0, yes_wrap == 0)
    # At least *some* fg pixels should differ — otherwise wrap is a no-op
    # and we may have wired it incorrectly.
    fg = m > 0
    assert (no_wrap[fg] != yes_wrap[fg]).any(), \
        "wrap=True produced an identical coloring — toroidal path may be no-op"


@pytest.mark.parametrize("p", [1, 2])
def test_expand_labels_wrap_changes_extent(p):
    """expand_labels(wrap=True) should fully label every pixel (same as the
    non-wrap version on a single connected colony) and differ from the
    non-wrap version near the image borders."""
    m = _circles_2d_dense()
    e_std = ncolor.expand_labels(m, p=p, wrap=False)
    e_wrap = ncolor.expand_labels(m, p=p, wrap=True)
    assert e_std.shape == m.shape == e_wrap.shape
    # Both fully fill (no zeros after expansion).
    assert (e_std > 0).all()
    assert (e_wrap > 0).all()
    # Near image edges the wrap expansion should differ from the standard
    # version (because cells from opposite edges can win those pixels).
    border = 8
    edge_diff = (e_std[:border] != e_wrap[:border]).sum() + \
                (e_std[-border:] != e_wrap[-border:]).sum() + \
                (e_std[:, :border] != e_wrap[:, :border]).sum() + \
                (e_std[:, -border:] != e_wrap[:, -border:]).sum()
    assert edge_diff > 0, \
        "wrap=True gave identical Voronoi as non-wrap near image borders"


# ----------------------------------------------------------------------
# Smoke coverage for the kwargs that the live pipeline exposes but
# weren't otherwise exercised. Each test asserts the same minimum:
# the kwarg path runs, returns a valid 4-coloring (no conflicts), and
# the output shape/dtype/bg pattern survive.
# ----------------------------------------------------------------------


def _assert_valid_coloring(out, mask, expected_n_max=8):
    assert out.shape == mask.shape
    assert out.dtype == np.uint8
    assert np.array_equal(out == 0, mask == 0)


@pytest.mark.parametrize("r", [1, 2])
def test_label_connect_radius(r):
    """connect_radius widens the neighbor search window. r=2 catches
    near-adjacent cells through 1-pixel gaps; both should 4-color the
    dense circles input."""
    m = _circles_2d_dense()
    out, conflicts = ncolor.label(m, conn=1, connect_radius=r,
                                  return_conflicts=True)
    assert conflicts == 0
    _assert_valid_coloring(out, m)


@pytest.mark.parametrize("despur", [1, 2, 5])
def test_label_despur_iters(despur):
    """Iterative label-aware despur should not break the coloring."""
    m = _circles_2d_dense()
    out, conflicts = ncolor.label(m, despur_iters=despur,
                                  return_conflicts=True)
    assert conflicts == 0
    _assert_valid_coloring(out, m)


@pytest.mark.parametrize("rounds", [1, 3])
def test_label_expand_spur_free(rounds):
    """expand_spur_free=True is the BFS dilation path that avoids
    creating K_5-shaped convergence pixels. Should still 4-color."""
    m = _circles_2d_dense()
    out, n_used = ncolor.label(m, expand_spur_free=True,
                               spur_free_max_rounds=rounds, return_n=True)
    _assert_valid_coloring(out, m)
    assert 1 <= int(n_used) <= 4


def test_label_extra_edges_constrains_pair():
    """extra_edges adds (1-indexed) cell-pair constraints on top of the
    geometric adjacency. The constrained pair must end up with different
    colors even though the two blobs are otherwise disjoint."""
    m = _two_circle_2d()  # disjoint blobs with labels 1 and 2
    extra = np.array([[1, 2]], dtype=np.int32)
    out, conflicts = ncolor.label(m, extra_edges=extra, return_conflicts=True)
    assert conflicts == 0
    assert out[m == 1][0] != out[m == 2][0]
    _assert_valid_coloring(out, m)


@pytest.mark.parametrize("mode", ["min", "max", "mean", "count",
                                   "harmonic", "mean_inv"])
def test_label_weight_objective_modes(mode):
    """Boundary-weighted picker: weight_objective != 0 routes through
    find_pairs_weighted, which computes a per-pair (d_i+d_j) reducer
    selected by weight_mode. All reducers must still 4-color."""
    m = _circles_2d_dense()
    out, conflicts = ncolor.label(m, weight_objective=1, weight_mode=mode,
                                  return_conflicts=True)
    assert conflicts == 0
    _assert_valid_coloring(out, m)


def test_label_optimize_two_hop():
    """optimize='two_hop' post-processes the BFS coloring via a 2-hop
    SA optimizer. Exercises ncolor._optimize.optimize_two_hop."""
    m = _circles_2d_dense()
    out, conflicts = ncolor.label(m, optimize="two_hop",
                                  return_conflicts=True)
    assert conflicts == 0
    _assert_valid_coloring(out, m)


# ----------------------------------------------------------------------
# delete_spurs dispatcher: binary vs label-aware
# ----------------------------------------------------------------------


def test_delete_spurs_binary_mode_default():
    """Mask with values in {0,1} → binary mode auto-detected. Returns
    a single mask (the cleaned bg pattern), not a (cleaned, n) tuple."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:7, 1:7] = 1
    mask[4, 7] = 1  # spur sticking off the side
    out = ncolor.delete_spurs(mask)
    assert out.shape == mask.shape
    # The spur is gone in the cleaned mask.
    assert out[4, 7] == 0


def test_delete_spurs_label_mode_autodetected():
    """Multi-label image → label-aware mode auto-detected. Returns the
    (cleaned_labels, n_removed) tuple from the cpp binding."""
    arr = np.zeros((12, 12), dtype=np.int32)
    arr[2:5, 2:5] = 1
    arr[7:10, 7:10] = 2
    arr[6, 6] = 1   # isolated label-1 pixel inside label-2's reach
    cleaned, n_removed = ncolor.delete_spurs(arr)
    assert cleaned.shape == arr.shape
    assert n_removed >= 1
    # The isolated label-1 pixel got zeroed; the two main blobs survive.
    assert cleaned[6, 6] == 0
    assert (cleaned[2:5, 2:5] == 1).all()
    assert (cleaned[7:10, 7:10] == 2).all()


def test_delete_spurs_force_kind_labels_on_binary_input():
    """kind='labels' on a 0/1 mask forces the label-aware path. The
    cpp binding returns (out, n_removed)."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    cleaned, n_removed = ncolor.delete_spurs(mask, kind="labels")
    assert cleaned.shape == mask.shape
    # Tuple shape — label-aware path was taken.
    assert isinstance(n_removed, int)


def test_delete_spurs_rejects_unknown_kind():
    arr = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="kind must be"):
        ncolor.delete_spurs(arr, kind="laser")


# ----------------------------------------------------------------------
# expand_labels mode kwarg: voronoi vs spur_free
# ----------------------------------------------------------------------


def test_expand_labels_spur_free_mode():
    """mode='spur_free' runs the BFS dilation path. Output is a label
    image of the same shape with seeds preserved. Growth requires a
    bg pixel to have ≥ threshold+1 = 2 face-neighbors of the same
    label, which happens naturally in concave geometry / dense packings
    but NOT along a single flat seed boundary (each bg pixel sees only
    one seed face there). _circles_2d_dense gives the right geometry."""
    m = _circles_2d_dense()
    out = ncolor.expand_labels(m, mode="spur_free", max_rounds=3)
    assert out.shape == m.shape
    # Original seeds preserved.
    fg = m > 0
    assert np.array_equal(out[fg], m[fg])
    # Some bg got claimed (BFS dilated into concavities).
    assert (out[m == 0] != 0).any()


def test_expand_labels_rejects_unknown_mode():
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[2:4, 2:4] = 1
    with pytest.raises(ValueError, match="mode must be"):
        ncolor.expand_labels(arr, mode="chamfer")


def test_expand_labels_voronoi_default_unchanged():
    """The default mode='voronoi' must behave exactly like the pre-mode
    expand_labels (regression check on the existing API)."""
    arr = np.zeros((10, 10), dtype=np.int32)
    arr[2:4, 2:4] = 1
    arr[6:8, 6:8] = 2
    out_default = ncolor.expand_labels(arr)
    out_explicit = ncolor.expand_labels(arr, mode="voronoi")
    assert np.array_equal(out_default, out_explicit)
