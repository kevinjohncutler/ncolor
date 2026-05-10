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
    """wrap=True should generally produce a different colouring than the
    non-wrap path (because the adjacency graph gains wrap-around edges).
    Asserts there's *some* difference rather than a specific permutation."""
    m = _circles_2d_dense(seed=42)
    no_wrap = ncolor.label(m, p=1, wrap=False)
    yes_wrap = ncolor.label(m, p=1, wrap=True)
    # Both valid, both compact 0..n_colors.
    assert no_wrap.max() <= 4 and yes_wrap.max() <= 4
    # Foreground bg patterns identical (only the colours change).
    assert np.array_equal(no_wrap == 0, yes_wrap == 0)
    # At least *some* fg pixels should differ — otherwise wrap is a no-op
    # and we may have wired it incorrectly.
    fg = m > 0
    assert (no_wrap[fg] != yes_wrap[fg]).any(), \
        "wrap=True produced an identical colouring — toroidal path may be no-op"


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
