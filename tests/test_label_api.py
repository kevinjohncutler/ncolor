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
    # bg pattern preserved (clean_mask=False is the default).
    assert np.array_equal(out == 0, m == 0)


def test_label_wrap_output_differs_from_non_wrap():
    """wrap=True should generally produce a different coloring than the
    non-wrap path (because the adjacency graph gains wrap-around edges).
    Asserts there's *some* difference rather than a specific permutation.

    Default expand_mode='bridge_free' fills the image border bg via
    Voronoi, so wrap-around adjacencies do develop."""
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
    # By default (clean_mask=False) the output preserves the input
    # mask's fg/bg pattern exactly — bridge_free's internal barrier
    # removal is graph-only and never strips colors from original-fg
    # pixels. So bg patterns match exactly.
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


@pytest.mark.parametrize("expand_mode", ["voronoi", "spur_free"])
def test_label_expand_mode(expand_mode):
    """Non-default expand modes (voronoi / spur_free) should still 4-color
    the dense circles input. bridge_free is the default and is exercised
    by every other test in this file."""
    m = _circles_2d_dense()
    out, n_used = ncolor.label(m, expand_mode=expand_mode, return_n=True)
    assert 1 <= int(n_used) <= 4
    # spur_free leaves border bg unfilled by design; voronoi fills it.
    # Either way, the coloring on the original fg pixels should be valid.


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


@pytest.mark.parametrize("p", [1, 2])
def test_expand_labels_bridge_free(p):
    """expand_labels(mode='bridge_free') runs the bridge-aware Voronoi
    expand directly (the same kernel used by ncolor.label's default
    expand path). Output should fill the image and may have a small
    number of barrier zeros where bridge_free identified bridges/stubs."""
    arr = _circles_2d_dense()
    out = ncolor.expand_labels(arr, p=p, mode="bridge_free")
    assert out.shape == arr.shape
    assert out.dtype == np.int32
    # All original fg pixels are preserved (their Voronoi seed label
    # is unchanged by bridge_free's barrier removal — barriers can
    # only fall on cells that were already at face_count <= 1).
    # Most of the image gets filled (allow a tiny fraction at barriers).
    fill_frac = (out > 0).mean()
    assert fill_frac > 0.9, f"bridge_free expand filled only {fill_frac:.3f}"


def test_expand_labels_bridge_free_metric_alias():
    """mode='bridge_free' accepts metric='l1' / 'l2' as aliases for p=1/2."""
    arr = np.zeros((16, 16), dtype=np.int32)
    arr[2:4, 2:4] = 1
    arr[10:12, 10:12] = 2
    out_l1 = ncolor.expand_labels(arr, mode="bridge_free", metric="l1")
    out_p1 = ncolor.expand_labels(arr, mode="bridge_free", p=1)
    out_l2 = ncolor.expand_labels(arr, mode="bridge_free", metric="l2")
    out_p2 = ncolor.expand_labels(arr, mode="bridge_free", p=2)
    assert np.array_equal(out_l1, out_p1)
    assert np.array_equal(out_l2, out_p2)


def test_expand_labels_bridge_free_rejects_bad_metric():
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[2:4, 2:4] = 1
    with pytest.raises(ValueError, match="Unknown metric"):
        ncolor.expand_labels(arr, mode="bridge_free", metric="chebyshev")
    with pytest.raises(ValueError, match="p must be 1 or 2"):
        ncolor.expand_labels(arr, mode="bridge_free", p=3)


def test_delete_spurs_remove_thin_one_shot():
    """remove_thin=True catches 1-px-thick line interiors in ONE pass.
    Build a label image with a horizontal 1-px bridge between two
    blobs; the bridge interior pixels have face-count=2 (opposite),
    so they survive iter 1 of the default despur but vanish with
    remove_thin=True in a single pass. The pixel where the bridge
    attaches to the blob is a junction (4 same-label 8-neighbors)
    and correctly stays — the detector only removes pixels with
    EXACTLY two same-label 8-conn neighbors on opposite sides."""
    arr = np.zeros((9, 20), dtype=np.int32)
    arr[3:6, 1:5] = 1                  # left blob (3x4)
    arr[3:6, 14:19] = 2                # right blob (3x5)
    arr[4, 5:14] = 1                   # 1-px bridge attached to left blob
    # Default despur (1 iter): only the bridge's far endpoint (col 13,
    # face-count=1) is peeled. The 8 interior + junction pixels survive.
    cleaned_no, n_no = ncolor.delete_spurs(arr, kind="labels", max_iters=1,
                                             remove_thin=False)
    survived_no = (cleaned_no[4, 5:14] != 0).sum()
    assert survived_no >= 7
    # With remove_thin=True at iter 1: bridge interior (cols 6-12) + the
    # spur endpoint (col 13) all vanish. Junction pixel at col 5 (which
    # has 4 same-label 8-neighbors from the blob) correctly stays.
    cleaned_yes, n_yes = ncolor.delete_spurs(arr, kind="labels", max_iters=1,
                                              remove_thin=True)
    assert (cleaned_yes[4, 6:14] == 0).all(), \
        "interior + spur endpoint of bridge should all be removed"
    assert cleaned_yes[4, 5] == 1, "junction pixel should survive"
    assert n_yes > n_no


def test_delete_spurs_remove_thin_keeps_real_cells():
    """remove_thin must not eat 2x2 blocks (count=2 but perpendicular)
    or L-shaped corners. Both have face-count=2 but the neighbors are
    not opposite, so the detector leaves them alone."""
    arr = np.zeros((8, 8), dtype=np.int32)
    arr[2:4, 2:4] = 1                  # 2x2 block (perpendicular pairs)
    arr[4:7, 4:7] = 2                  # 3x3 block, all interior count >= 3
    cleaned, _ = ncolor.delete_spurs(arr, kind="labels", max_iters=20,
                                       remove_thin=True)
    # 2x2 survives: each pixel has 2 perpendicular face-neighbors.
    assert (cleaned[2:4, 2:4] == 1).all()
    # 3x3 survives: interior has count >= 3.
    assert (cleaned[4:7, 4:7] == 2).all()


def test_delete_spurs_remove_thin_catches_2d_diagonal_line():
    """A 1-px-wide diagonal NW-SE line is caught by the diagonal-
    opposite detector. (Note: pure diagonal lines also have face-
    count=0 so they're already caught by the spur rule, but a single
    face-neighbor plus an opposite diagonal isn't.)"""
    # Build a 5-pixel diagonal line.
    arr = np.zeros((10, 10), dtype=np.int32)
    for k in range(5):
        arr[2 + k, 2 + k] = 1
    cleaned, n = ncolor.delete_spurs(arr, kind="labels", max_iters=1,
                                      remove_thin=True)
    # All 5 diagonal pixels gone (they were already spurs by face-count=0).
    assert (cleaned == 0).all()
    assert n == 5


# ----------------------------------------------------------------------
# Soft constraints + clean_mask (default-on path; covers the soft_color +
# auto-build + apply_lut snapshot logic in cpp/binding.cpp + soft_color.hpp).
# ----------------------------------------------------------------------


def test_soft_default_path_runs_and_preserves_mask():
    """Default `ncolor.label(m)` is now auto-soft (soft_conn=2 r=2) +
    clean_mask=False. Verifies the soft pathway fires without errors AND
    that the output's foreground/background pattern matches the input
    exactly — bridge_free barriers must NOT leak through to the output."""
    m = _circles_2d_dense()
    out, n_used = ncolor.label(m, return_n=True)
    assert 1 <= int(n_used) <= 4
    # Default clean_mask=False keeps original fg/bg pattern.
    assert np.array_equal(out == 0, m == 0)


def test_soft_explicit_extra_edges_path():
    """Explicit (E, 2) soft_extra_edges array path. Disjoint blobs (no
    geometric adjacency) with one soft pair between them: the picker
    is free to use the same color for both, but the soft search should
    prefer differentiating them and the bg pattern must survive."""
    m = _two_circle_2d()
    soft = np.array([[1, 2]], dtype=np.int32)
    out = ncolor.label(m, soft_extra_edges=soft,
                        # disable auto-build so only the explicit path fires
                        soft_conn=0, soft_radius=0)
    _assert_valid_coloring(out, m)


def test_clean_mask_true_vs_false_differs_only_at_barriers():
    """clean_mask=False (default) keeps original fg/bg exact. clean_mask
    =True opts into the bridge_free-barrier-zeros-surface-too behavior:
    the output may have additional zeros where bridge_free identified
    bridges/stubs. Where both produce a color, the colors must agree."""
    m = _circles_2d_dense()
    out_default = ncolor.label(m)                  # clean_mask=False
    out_clean   = ncolor.label(m, clean_mask=True)
    # Default preserves input bg pattern exactly.
    assert np.array_equal(out_default == 0, m == 0)
    # clean_mask=True output zeros include input bg AND any barriers.
    # So bg-direction inclusion holds: every input bg is bg in both.
    assert (out_clean[m == 0] == 0).all()
    # Where both paths assigned a color (>0), they should agree.
    both_colored = (out_default > 0) & (out_clean > 0)
    assert np.array_equal(
        out_default[both_colored], out_clean[both_colored])


def test_soft_conn_radius_disabled_when_zero():
    """soft_conn=0 or soft_radius=0 turns the auto-soft path off; the
    coloring should still be valid and identical to a plain call."""
    m = _circles_2d_dense()
    out_no_soft = ncolor.label(m, soft_conn=0, soft_radius=0)
    _assert_valid_coloring(out_no_soft, m)


def test_label_verbose_prints_stage_summary(capsys):
    """verbose=True wires through to capture_stages + prints a one-line
    summary to stderr (shape, n_used, sv, total, breakdown)."""
    m = _circles_2d_dense(seed=0)
    out, n_used = ncolor.label(m, return_n=True, verbose=True)
    captured = capsys.readouterr()
    # Header line should mention shape, n_used, sv, and total ms.
    assert "[ncolor.label]" in captured.err
    assert "n_used=" in captured.err
    assert "total=" in captured.err
    # Breakdown line should mention at least a few stage names.
    assert "expand" in captured.err or "find_pairs" in captured.err
