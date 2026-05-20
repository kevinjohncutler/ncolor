"""L1 vs L2 ncolor comparison on the example logo masks.

Mirrors examples/example.ipynb's styling: dark background, viridis
colormap, alpha = (label > 0). Renders four panels side by side:
  1. Original masks
  2. cpp p=2 (Felzenszwalb / Euclidean) coloring
  3. cpp p=1 (Saito-Toriwaki / Manhattan) coloring
  4. Pixel-level difference map (where L1 vs L2 disagree)
"""
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage import io as skio
from skimage import measure

import ncolor


def rescale_to(T, vmin, vmax):
    """Rescale T to [0, 1] using FIXED bounds (not per-panel min/max).
    Critical for consistent viridis shading across panels — otherwise the
    same label number gets different colors when one panel has bg pixels
    (value 0) and another doesn't."""
    if vmax == vmin:
        return np.zeros_like(T, dtype=float)
    return (np.clip(T, vmin, vmax).astype(float) - vmin) / (vmax - vmin)


def rescale(T):
    return rescale_to(T, T.min(), T.max())


def crop_bbx(mask, pad=10, iterations=3, im_pad=0, area_cutoff=0):
    bw = binary_dilation(mask > 0, iterations=iterations)
    clusters = measure.label(bw)
    regions = measure.regionprops(clusters)
    sz = mask.shape
    ylim = [im_pad, sz[0] - im_pad]
    xlim = [im_pad, sz[1] - im_pad]
    for props in regions:
        if props.area > area_cutoff:
            y1, x1, y2, x2 = props.bbox
            y1 = max(y1 - pad, ylim[0]); x1 = max(x1 - pad, xlim[0])
            y2 = min(y2 + pad, ylim[1]); x2 = min(x2 + pad, xlim[1])
            if (y2 > y1) and (x2 > x1):
                return (y1, y2, x1, x2)
    return (0, sz[0], 0, sz[1])


repo_root = Path(ncolor.__file__).resolve().parents[2]
masks = skio.imread(repo_root / "test_files" / "example.png")
y1, y2, x1, x2 = crop_bbx(masks)
m = masks[y1:y2, x1:x2]

# Format once so subsequent calls share the same compacted label
# numbering (avoids the bottom row using a different LUT than the top).
masks_fmt = ncolor.format_labels(masks.astype(np.int32))

# Get the LUT and the expanded labels for each metric. The LUT is
# indexed by formatted-expanded label values, so applying it to the same
# `expand_labels` output gives the full-coverage Voronoi coloring with
# the SAME color assignment as the bg-masked output.
# wrap=False (top half of figure): standard non-toroidal topology.
# wrap=True  (bottom row of figure): treat image as a torus — left/right
# and top/bottom edges become neighbors, so perimeter cells gain
# adjacencies and the coloring uses 4 colors more uniformly.
lut_l2 = ncolor.label(masks_fmt, expand=True, p=2, return_lut=True)
lut_l1 = ncolor.label(masks_fmt, expand=True, p=1, return_lut=True)
lut_l2w = ncolor.label(masks_fmt, expand=True, p=2, wrap=True, return_lut=True)
lut_l1w = ncolor.label(masks_fmt, expand=True, p=1, wrap=True, return_lut=True)
# wrap + balance: Welsh-Powell visit order (highest-degree first) on top
# of toroidal Voronoi. Targets visual uniformity of the 4-color count.
lut_l2wb = ncolor.label(masks_fmt, expand=True, p=2, wrap=True, balance=True, return_lut=True)
lut_l1wb = ncolor.label(masks_fmt, expand=True, p=1, wrap=True, balance=True, return_lut=True)
exp_l2 = ncolor.expand_labels(masks_fmt, p=2)
exp_l1 = ncolor.expand_labels(masks_fmt, p=1)
exp_l2w = ncolor.expand_labels(masks_fmt, p=2, wrap=True)
exp_l1w = ncolor.expand_labels(masks_fmt, p=1, wrap=True)

# Row 0 (masked): fg pixels get lut[label], bg pixels get 0.
nc_l2 = (lut_l2[exp_l2] * (masks_fmt > 0))[y1:y2, x1:x2]
nc_l1 = (lut_l1[exp_l1] * (masks_fmt > 0))[y1:y2, x1:x2]

# Row 1 (full Voronoi): every pixel gets lut[label] — same LUT as row 0.
ncraw_l2 = lut_l2[exp_l2][y1:y2, x1:x2]
ncraw_l1 = lut_l1[exp_l1][y1:y2, x1:x2]

# Row 2 (wrap=True): toroidal Voronoi expansion AND toroidal find_pairs.
# Apply LUT to the toroidal expansion (cells' territories now wrap across
# image edges, so perimeter cells have full constraint pressure).
ncwrap_l2 = lut_l2w[exp_l2w][y1:y2, x1:x2]
ncwrap_l1 = lut_l1w[exp_l1w][y1:y2, x1:x2]

# Row 3 (wrap=True, masked): the actual user-facing output of
# ncolor.label(masks, p=*, wrap=True). Same coloring as row 2 but
# bg-masked back to the original cell pattern.
ncwrap_masked_l2 = (lut_l2w[exp_l2w] * (masks_fmt > 0))[y1:y2, x1:x2]
ncwrap_masked_l1 = (lut_l1w[exp_l1w] * (masks_fmt > 0))[y1:y2, x1:x2]

# Row 4 (wrap + balance, masked): adding Welsh-Powell visit ordering.
ncwb_masked_l2 = (lut_l2wb[exp_l2w] * (masks_fmt > 0))[y1:y2, x1:x2]
ncwb_masked_l1 = (lut_l1wb[exp_l1w] * (masks_fmt > 0))[y1:y2, x1:x2]

# Stats.
n_pix = m.size
diff_pix = int(np.sum(nc_l1 != nc_l2))
print(f"Crop shape: {m.shape}")
print(f"L2 colors used: {int(nc_l2.max())},  L1 colors used: {int(nc_l1.max())}")
print(f"Pixels matching: {(1 - diff_pix / n_pix) * 100:.2f}%  ({n_pix - diff_pix:,} of {n_pix:,})")
print(f"Pixels differing: {diff_pix:,}")
print()
print(f"Cell-count per color (using each LUT directly):")
print(f"  L2 (no wrap):  {[int((lut_l2[1:]  == c).sum()) for c in range(1, 5)]}")
print(f"  L1 (no wrap):  {[int((lut_l1[1:]  == c).sum()) for c in range(1, 5)]}")
print(f"  L2 (wrap):     {[int((lut_l2w[1:] == c).sum()) for c in range(1, 5)]}")
print(f"  L1 (wrap):     {[int((lut_l1w[1:] == c).sum()) for c in range(1, 5)]}")

# Visual styling — match the notebook.
plt.style.use("dark_background")
mpl.rcParams["figure.dpi"] = 200
cmap = mpl.colormaps.get_cmap("viridis")


def colored_panel(arr, mask, *, vmin=None, vmax=None):
    """viridis-colored panel with alpha = (mask > 0).

    When ``vmin`` / ``vmax`` are given, use them as fixed colormap bounds
    so the same label number maps to the same viridis shade across
    panels — otherwise per-panel min/max scaling makes 'color 4' look
    different when one panel includes bg (0) and another doesn't.
    """
    if vmin is None:
        rgba = cmap(rescale(arr))
    else:
        rgba = cmap(rescale_to(arr, vmin, vmax))
    rgba[..., -1] = (mask > 0).astype(float)
    return rgba


# Five-row layout (top → bottom):
#   row 0 — wrap=False, masked (default ncolor.label output)
#   row 1 — wrap=False, full Voronoi expansion (same LUT as row 0)
#   row 2 — wrap=True,  masked
#   row 3 — wrap=True,  full Voronoi expansion (toroidal)
#   row 4 — wrap=True + balance=True, masked (recommended config —
#           Welsh-Powell BFS ordering for the most uniform 4-color
#           distribution)
fig, axes = plt.subplots(5, 4, figsize=(16, 20),
                         gridspec_kw={"wspace": 0.02, "hspace": 0.10})

# Color values produced by ncolor.label are in {0, 1, ..., n_colors}.
# Pin viridis bounds so the same color value renders identically in every
# panel, regardless of whether bg pixels (value 0) are present or not.
n_colors = max(int(nc_l2.max()), int(nc_l1.max()),
               int(ncraw_l2.max()), int(ncraw_l1.max()),
               int(ncwrap_l2.max()), int(ncwrap_l1.max()),
               int(ncwrap_masked_l2.max()), int(ncwrap_masked_l1.max()),
               int(ncwb_masked_l2.max()), int(ncwb_masked_l1.max()))
COLOR_VMIN, COLOR_VMAX = 0, n_colors

# Row 0 — masked output (only foreground cells visible).
axes[0, 0].imshow(colored_panel(m, m))
axes[0, 0].set_title("input masks", color="white", fontsize=11)
axes[0, 1].imshow(colored_panel(nc_l2, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[0, 1].set_title("ncolor.label(p=2)\nFelzenszwalb (Euclidean)\nbg-masked", color="white", fontsize=10)
axes[0, 2].imshow(colored_panel(nc_l1, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[0, 2].set_title("ncolor.label(p=1)\nSaito-Toriwaki (Manhattan)\nbg-masked", color="white", fontsize=10)

diff = (nc_l1 != nc_l2) & (m > 0)
diff_rgba = np.zeros((*diff.shape, 4))
diff_rgba[..., 0] = 1.0
diff_rgba[..., 3] = diff.astype(float)
axes[0, 3].imshow(colored_panel(nc_l2, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[0, 3].imshow(diff_rgba, alpha=0.85)
axes[0, 3].set_title(f"L1 ≠ L2 (red, masked)\n{(diff_pix / n_pix) * 100:.1f}% of fg pixels",
                     color="white", fontsize=10)

# Row 1 — raw Voronoi expansion (every pixel labeled / colored).
full = np.ones_like(m)  # alpha = 1 everywhere
axes[1, 0].imshow(colored_panel(exp_l2[y1:y2, x1:x2], full))
axes[1, 0].set_title("expand_labels(p=2)\nraw label map", color="white", fontsize=10)
axes[1, 1].imshow(colored_panel(ncraw_l2, full, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[1, 1].set_title("p=2 coloring\n(full Voronoi)", color="white", fontsize=10)
axes[1, 2].imshow(colored_panel(ncraw_l1, full, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[1, 2].set_title("p=1 coloring\n(full Voronoi)", color="white", fontsize=10)

raw_diff = ncraw_l1 != ncraw_l2
raw_diff_pct = raw_diff.mean() * 100
raw_diff_rgba = np.zeros((*raw_diff.shape, 4))
raw_diff_rgba[..., 0] = 1.0
raw_diff_rgba[..., 3] = raw_diff.astype(float)
axes[1, 3].imshow(colored_panel(ncraw_l2, full, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[1, 3].imshow(raw_diff_rgba, alpha=0.85)
axes[1, 3].set_title(f"L1 ≠ L2 (red, full)\n{raw_diff_pct:.1f}% of all pixels",
                     color="white", fontsize=10)

# Row 2 — wrap=True, masked (the *actual* user-facing output of
# ncolor.label(masks, p=..., wrap=True) — what callers see).
axes[2, 0].imshow(colored_panel(m, m))
axes[2, 0].set_title("input masks", color="white", fontsize=11)
axes[2, 1].imshow(colored_panel(ncwrap_masked_l2, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[2, 1].set_title("p=2 + wrap=True\nmasked output", color="white", fontsize=10)
axes[2, 2].imshow(colored_panel(ncwrap_masked_l1, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[2, 2].set_title("p=1 + wrap=True\nmasked output (recommended)", color="white", fontsize=10)

masked_wrap_diff = (ncwrap_masked_l1 != ncwrap_masked_l2) & (m > 0)
masked_wrap_diff_rgba = np.zeros((*masked_wrap_diff.shape, 4))
masked_wrap_diff_rgba[..., 0] = 1.0
masked_wrap_diff_rgba[..., 3] = masked_wrap_diff.astype(float)
axes[2, 3].imshow(colored_panel(ncwrap_masked_l2, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[2, 3].imshow(masked_wrap_diff_rgba, alpha=0.85)
axes[2, 3].set_title(f"L1 ≠ L2 (wrap, masked)\n{masked_wrap_diff.sum() / (m > 0).sum() * 100:.1f}% of fg",
                     color="white", fontsize=10)

# Row 3 — wrap=True, full Voronoi (toroidal expansion).
axes[3, 0].imshow(colored_panel(exp_l1w[y1:y2, x1:x2], full))
axes[3, 0].set_title("expand_labels(p=1, wrap=True)\ntoroidal Voronoi", color="white", fontsize=10)
axes[3, 1].imshow(colored_panel(ncwrap_l2, full, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[3, 1].set_title("p=2 + wrap=True\nfull Voronoi", color="white", fontsize=10)
axes[3, 2].imshow(colored_panel(ncwrap_l1, full, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[3, 2].set_title("p=1 + wrap=True\nfull Voronoi", color="white", fontsize=10)

wrap_diff = ncwrap_l1 != ncwrap_l2
wrap_diff_pct = wrap_diff.mean() * 100
wrap_diff_rgba = np.zeros((*wrap_diff.shape, 4))
wrap_diff_rgba[..., 0] = 1.0
wrap_diff_rgba[..., 3] = wrap_diff.astype(float)
axes[3, 3].imshow(colored_panel(ncwrap_l2, full, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[3, 3].imshow(wrap_diff_rgba, alpha=0.85)
axes[3, 3].set_title(f"L1 ≠ L2 (red, wrap full)\n{wrap_diff_pct:.1f}% of all pixels",
                     color="white", fontsize=10)

# Row 4 — wrap + balance (Welsh-Powell), masked output.
counts_l2wb = [int((lut_l2wb[1:] == c).sum()) for c in range(1, 5)]
counts_l1wb = [int((lut_l1wb[1:] == c).sum()) for c in range(1, 5)]
axes[4, 0].imshow(colored_panel(m, m))
axes[4, 0].set_title("input masks", color="white", fontsize=11)
axes[4, 1].imshow(colored_panel(ncwb_masked_l2, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[4, 1].set_title(f"p=2, wrap=True, balance=True\ncounts: {counts_l2wb}",
                     color="white", fontsize=10)
axes[4, 2].imshow(colored_panel(ncwb_masked_l1, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[4, 2].set_title(f"p=1, wrap=True, balance=True (recommended)\ncounts: {counts_l1wb}",
                     color="white", fontsize=10)
wb_diff = (ncwb_masked_l1 != ncwb_masked_l2) & (m > 0)
wb_diff_rgba = np.zeros((*wb_diff.shape, 4))
wb_diff_rgba[..., 0] = 1.0
wb_diff_rgba[..., 3] = wb_diff.astype(float)
axes[4, 3].imshow(colored_panel(ncwb_masked_l2, m, vmin=COLOR_VMIN, vmax=COLOR_VMAX))
axes[4, 3].imshow(wb_diff_rgba, alpha=0.85)
axes[4, 3].set_title(f"L1 ≠ L2 (wrap+balance, masked)\n{wb_diff.sum() / (m > 0).sum() * 100:.1f}% of fg",
                     color="white", fontsize=10)

for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])

out_path = repo_root / "examples" / "l1_vs_l2.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="black", edgecolor="none")
plt.close(fig)
print(f"\nSaved figure: {out_path}")
