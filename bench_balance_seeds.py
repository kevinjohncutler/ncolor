"""Compare L1+balance vs L2+nobalance color distributions across 10 seeds
on the example cellpose mask (test_files/example.png).

ncolor.label is deterministic per input, so to vary the "seed" we permute
the input mask's label IDs. The graph topology is identical, but the
Welsh-Powell visit order changes (it's degree-then-ID), so tie-broken
choices differ — a real seed knob.

For each (p, balance, seed) the script prints an "alignment score" — the
number of axis-aligned same-color groups of 3+ cells whose centroids land
in a narrow row/column band. Higher score → more visible stripes.

Output: bench_balance_seeds.png (2 × 10 grid of viridis panels matching
the examples/ visual style).
"""
from __future__ import annotations
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io as skio, measure
from scipy.ndimage import binary_dilation

import ncolor


# Same crop helper as examples/balance_comparison.py.
def crop_bbx(mask, pad=10, iterations=3):
    bw = binary_dilation(mask > 0, iterations=iterations)
    clusters = measure.label(bw)
    regions = measure.regionprops(clusters)
    sz = mask.shape
    for props in regions:
        if props.area > 0:
            y1, x1, y2, x2 = props.bbox
            return (max(y1 - pad, 0), max(x1 - pad, 0),
                    min(y2 + pad, sz[0]), min(x2 + pad, sz[1]))
    return (0, 0, sz[0], sz[1])


masks = skio.imread(ROOT / "test_files" / "example.png")
masks_fmt = ncolor.format_labels(masks.astype(np.int32))
y1, x1, y2, x2 = crop_bbx(masks_fmt)


def permute_label_ids(arr: np.ndarray, seed: int) -> np.ndarray:
    """Reassign label IDs using a permutation; preserves bg=0."""
    rng = np.random.default_rng(seed)
    n_labels = int(arr.max())
    perm = rng.permutation(n_labels) + 1  # 1..N permuted
    lut = np.zeros(n_labels + 1, dtype=arr.dtype)
    lut[1:] = perm
    return lut[arr]


def adj_same_color_pairs(label_image: np.ndarray, colored: np.ndarray) -> int:
    """Brute-force adjacency check (8-conn, pixel-pair-level): count adjacent
    pixel pairs that are in DIFFERENT cells in ``label_image`` AND have the
    SAME nonzero color in ``colored``. Should be 0 for a valid coloring."""
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


def alignment_score(label_image: np.ndarray, colored: np.ndarray,
                    band: int = 6) -> tuple[int, int]:
    """Count axis-aligned bands containing 3+ same-color cells.

    For each color c we project the centroids of all cells colored c
    onto the y-axis (and x-axis), bin them at ``band`` px, and count bins
    holding 3+ entries. The total per axis is the sum over colors.
    A "long stripe" (4-cell row of one color) contributes 2; a 5-cell
    row contributes 3; etc.
    """
    rprops = measure.regionprops(label_image)
    centroids = np.array([r.centroid for r in rprops])  # (N, 2) (y, x)
    cell_colors = np.array([
        colored[int(r.centroid[0]), int(r.centroid[1])] for r in rprops
    ])
    h_count = v_count = 0
    H, W = label_image.shape
    for c in np.unique(cell_colors):
        if c == 0:
            continue
        idx = cell_colors == c
        if idx.sum() < 3:
            continue
        ys = centroids[idx, 0]
        xs = centroids[idx, 1]
        # Bin centroids by y (horizontal stripes) and x (vertical stripes).
        for axis_pts, axis_extent, count_into in (
            (ys, H, "h"),
            (xs, W, "v"),
        ):
            n_bins = max(1, int(np.ceil(axis_extent / band)))
            counts = np.bincount(
                np.minimum((axis_pts // band).astype(int), n_bins - 1),
                minlength=n_bins,
            )
            stripes = np.maximum(counts - 2, 0).sum()  # 3-row=1, 4-row=2…
            if count_into == "h":
                h_count += stripes
            else:
                v_count += stripes
    return int(h_count), int(v_count)


CONFIGS = [
    ("L1+bal",   1, True),
    ("L1+nobal", 1, False),
    ("L2+bal",   2, True),
    ("L2+nobal", 2, False),
]
SEEDS = list(range(10))


def color_one(permuted, p, balance):
    out, n = ncolor.label(permuted, p=p, balance=balance,
                          expand=True, return_n=True)
    viol = adj_same_color_pairs(permuted, out)
    h, v = alignment_score(permuted, out)
    return out, n, viol, h, v


# Per-seed records for the figure.
records = {}  # records[(name, seed)] = (out_crop, n, viol, h, v)
for seed in SEEDS:
    permuted = permute_label_ids(masks_fmt, seed)
    for name, p, balance in CONFIGS:
        out, n, viol, h, v = color_one(permuted, p, balance)
        records[(name, seed)] = (out[y1:y2, x1:x2], n, viol, h, v)

print("Per-seed table:")
header = f"{'seed':<6}" + "".join(
    f" {name+' n stripes':>16} {'viol':>4}" for name, _, _ in CONFIGS
)
print(header)
for seed in SEEDS:
    row = f"  {seed:<4}"
    for name, _, _ in CONFIGS:
        _, n, viol, h, v = records[(name, seed)]
        row += f"  n={n} {h:>3}h{v:>3}v {viol:>4}"
    print(row)


# Wide-sweep study: hash every coloring, find duplicate colorings (within
# a config and across configs), measure failure rate.
import hashlib

def hash_coloring(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()[:12]

N_SEEDS_LARGE = 500
print(f"\nWide sweep over {N_SEEDS_LARGE} permutation seeds:")
fail_count = {name: 0 for name, _, _ in CONFIGS}
viol_count = {name: 0 for name, _, _ in CONFIGS}
n_distribution = {name: {} for name, _, _ in CONFIGS}
hash_seeds = {name: {} for name, _, _ in CONFIGS}     # hash → list of seeds
fail_seeds = {name: [] for name, _, _ in CONFIGS}     # seeds where n > 4

for seed in range(N_SEEDS_LARGE):
    permuted = permute_label_ids(masks_fmt, seed)
    for name, p, balance in CONFIGS:
        out, n = ncolor.label(permuted, p=p, balance=balance,
                              expand=True, return_n=True)
        viol = adj_same_color_pairs(permuted, out)
        if n > 4:
            fail_count[name] += 1
            fail_seeds[name].append(seed)
        if viol:
            viol_count[name] += 1
        n_distribution[name][n] = n_distribution[name].get(n, 0) + 1
        h = hash_coloring(out)
        hash_seeds[name].setdefault(h, []).append(seed)

print(f"\n{'config':<10} {'fail %':>7} {'unique colorings':>20} "
      f"{'biggest collision':>20} {'adj viol':>10}")
for name, _, _ in CONFIGS:
    n_unique = len(hash_seeds[name])
    biggest = max(len(s) for s in hash_seeds[name].values())
    pct = 100 * fail_count[name] / N_SEEDS_LARGE
    print(f"  {name:<8} {pct:>6.1f}% {n_unique:>20} "
          f"{biggest:>20} {viol_count[name]:>10}")
    dist = sorted(n_distribution[name].items())
    dist_str = ", ".join(f"n={n}: {c}" for n, c in dist)
    print(f"             n distribution: {{{dist_str}}}")
    if fail_count[name] <= 30:
        print(f"             bumped seeds: {fail_seeds[name]}")

# Cross-config collisions: do any two configs produce the SAME coloring on
# the same seed? (Would be surprising — different metric or order should
# break equivalence.)
print(f"\nCross-config matches (same seed → same coloring across configs):")
match_pairs = []
for seed in range(N_SEEDS_LARGE):
    hashes = {}
    for name, p, balance in CONFIGS:
        # find this seed's hash for this config
        for h, seeds in hash_seeds[name].items():
            if seed in seeds:
                hashes[name] = h
                break
    # any duplicate hash means two configs produced identical coloring
    rev = {}
    for name, h in hashes.items():
        rev.setdefault(h, []).append(name)
    for h, names in rev.items():
        if len(names) > 1:
            match_pairs.append((seed, names))
n_matches = len(match_pairs)
print(f"  {n_matches}/{N_SEEDS_LARGE} seeds had ≥2 configs produce the "
      f"same coloring.")
for seed, names in match_pairs[:5]:
    print(f"    seed {seed}: {names}")
if n_matches > 5:
    print(f"    ... and {n_matches - 5} more")


# -- Visualization: discrete 5-color categorical palette so color 5 (when
# the BFS heuristic bumps from 4 to 5) is visually distinct from color 4.
# Five hand-picked colors that stay legible on a black background and
# remain perceptually distinct under color-blindness simulation.
plt.style.use("dark_background")
mpl.rcParams["figure.dpi"] = 200
PALETTE = ["#000000",  # 0 — bg (alpha-masked anyway)
           "#1f77b4",  # 1 — blue
           "#ff7f0e",  # 2 — orange
           "#2ca02c",  # 3 — green
           "#d62728",  # 4 — red
           "#ffd92f"]  # 5 — yellow (only when the BFS bumps cur_n)
cmap = mpl.colors.ListedColormap(PALETTE)
norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(PALETTE) + 0.5), cmap.N)
mask_crop = masks_fmt[y1:y2, x1:x2]


def colored_panel(arr, mask):
    rgba = cmap(norm(arr))
    rgba[..., -1] = (mask > 0).astype(float)
    return rgba


fig, axes = plt.subplots(
    len(CONFIGS), len(SEEDS),
    figsize=(2.2 * len(SEEDS), 2.0 * len(CONFIGS) + 0.8),
    gridspec_kw={"wspace": 0.04, "hspace": 0.32},
)
for row_i, (name, p, balance) in enumerate(CONFIGS):
    for col, seed in enumerate(SEEDS):
        out_crop, n, viol, h, v = records[(name, seed)]
        bump = f"⚠ n={n}" if n > 4 else f"n={n}"
        title = (f"seed {seed}\n{name} · {bump}\n{h}h {v}v ✓{viol}"
                 if row_i == 0
                 else f"{name} · {bump}\n{h}h {v}v ✓{viol}")
        axes[row_i, col].imshow(colored_panel(out_crop, mask_crop))
        axes[row_i, col].set_title(title, color="white", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])
# Row labels.
row_y = np.linspace(0.86, 0.18, len(CONFIGS))
for y, (name, _, _) in zip(row_y, CONFIGS):
    fig.text(0.03, y, name, color="white", fontsize=11,
             fontweight="bold", rotation=90, ha="center", va="center")

out_path = ROOT / "bench_balance_seeds.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="black", edgecolor="none")
plt.close(fig)
print(f"\nfigure: {out_path}")
