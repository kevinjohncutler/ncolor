"""8-panel comparison: cpp/numba × L1/L2 × balance on/off (no wrap).

Visualises how each (backend, metric, balance) combination colors the
example logo. Same viridis colormap and dark-bg styling as
examples/example.ipynb. Uses subprocess to flip NCOLOR_BACKEND for the
numba-side panels.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage import io as skio
from skimage import measure

import ncolor

repo_root = Path(ncolor.__file__).resolve().parents[2]
masks = skio.imread(repo_root / "test_files" / "example.png")


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


y1, x1, y2, x2 = crop_bbx(masks)
m = masks[y1:y2, x1:x2]
masks_fmt = ncolor.format_labels(masks.astype(np.int32))

# CPP outputs
cpp_outputs = {}
for p in (1, 2):
    for balance in (False, True):
        out = ncolor.label(masks_fmt, p=p, balance=balance)[y1:y2, x1:x2]
        lut = ncolor.label(masks_fmt, p=p, balance=balance, return_lut=True)
        cpp_outputs[(p, balance)] = (out, lut)


# Numba outputs (subprocess so NCOLOR_BACKEND=numba reloads cleanly)
NUMBA_SCRIPT = textwrap.dedent(r'''
    import sys, os
    sys.path.insert(0, '<ncolor>/src')
    import numpy as np
    from pathlib import Path
    import skimage.io as skio
    import ncolor

    repo_root = Path(ncolor.__file__).resolve().parents[2]
    masks = skio.imread(repo_root / 'test_files' / 'example.png')
    masks_fmt = ncolor.format_labels(masks.astype(np.int32))

    p = int(sys.argv[1])
    balance = sys.argv[2] == '1'
    out_path = sys.argv[3]

    if p == 2:
        lab_exp = ncolor.expand_labels(masks_fmt, metric='l2')
    else:
        lab_exp = ncolor.expand_labels(masks_fmt, metric='l1')

    out = ncolor.label(lab_exp, expand=False, balance=balance)
    out = out * (masks_fmt > 0)
    lut = ncolor.label(lab_exp, expand=False, balance=balance, return_lut=True)
    np.savez(out_path, out=out, lut=lut)
''')


numba_outputs = {}
env = {**os.environ, "NCOLOR_BACKEND": "numba"}
for p in (1, 2):
    for balance in (False, True):
        tmp = f"/tmp/_nb_balance_{p}_{int(balance)}.npz"
        subprocess.run([sys.executable, "-c", NUMBA_SCRIPT, str(p),
                        "1" if balance else "0", tmp],
                       check=True, env=env, timeout=180)
        d = np.load(tmp)
        numba_outputs[(p, balance)] = (d["out"][y1:y2, x1:x2], d["lut"])


# Stats
def color_stats(lut):
    counts = [int((lut[1:] == c).sum()) for c in range(1, 5)]
    return counts, float(np.std(counts))


# Plot
plt.style.use("dark_background")
mpl.rcParams["figure.dpi"] = 200
cmap = mpl.colormaps.get_cmap("viridis")


def colored_panel(arr, mask, vmin, vmax):
    """Viridis-colored panel with alpha = (mask > 0)."""
    norm = (np.clip(arr, vmin, vmax).astype(float) - vmin) / max(vmax - vmin, 1)
    rgba = cmap(norm)
    rgba[..., -1] = (mask > 0).astype(float)
    return rgba


COLOR_VMIN, COLOR_VMAX = 0, 4

fig, axes = plt.subplots(2, 4, figsize=(16, 9),
                         gridspec_kw={"wspace": 0.02, "hspace": 0.18})

# Row 0: cpp.  Row 1: numba.
# Cols: L1 off, L1 on, L2 off, L2 on.
configs = [(1, False), (1, True), (2, False), (2, True)]
labels = ["L1, balance=False", "L1, balance=True",
          "L2, balance=False", "L2, balance=True"]

for col, ((p, balance), title) in enumerate(zip(configs, labels)):
    out_cpp, lut_cpp = cpp_outputs[(p, balance)]
    counts_c, sigma_c = color_stats(lut_cpp)
    axes[0, col].imshow(colored_panel(out_cpp, m, COLOR_VMIN, COLOR_VMAX))
    axes[0, col].set_title(f"cpp · {title}\ncounts {counts_c}  σ={sigma_c:.2f}",
                           color="white", fontsize=10)

    out_nb, lut_nb = numba_outputs[(p, balance)]
    counts_n, sigma_n = color_stats(lut_nb)
    axes[1, col].imshow(colored_panel(out_nb, m, COLOR_VMIN, COLOR_VMAX))
    axes[1, col].set_title(f"numba · {title}\ncounts {counts_n}  σ={sigma_n:.2f}",
                           color="white", fontsize=10)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

# Row labels on the left.
fig.text(0.02, 0.74, "cpp", color="white", fontsize=14, fontweight="bold",
         rotation=90, ha="center", va="center")
fig.text(0.02, 0.30, "numba", color="white", fontsize=14, fontweight="bold",
         rotation=90, ha="center", va="center")

out_path = repo_root / "examples" / "balance_comparison.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor="black", edgecolor="none")
plt.close(fig)

print("Cell-count summary (no wrap, just balance + metric):")
print(f"{'backend':<8} {'metric':<5} {'balance':<7} {'counts':<22} {'σ':<6}")
for backend, outs in [("cpp", cpp_outputs), ("numba", numba_outputs)]:
    for (p, balance), (_, lut) in outs.items():
        counts, sigma = color_stats(lut)
        print(f"  {backend:<6} L{p}    {str(balance):<7} {str(counts):<22} {sigma:.2f}")
print(f"\nSaved figure: {out_path}")
