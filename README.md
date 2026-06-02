[![PyPI version](https://img.shields.io/pypi/v/ncolor.svg?color=green)](https://pypi.org/project/ncolor/)
[![Downloads](https://static.pepy.tech/personalized-badge/ncolor?period=total&units=international_system&left_color=gray&right_color=green&left_text=Downloads)](https://pepy.tech/project/ncolor)
[![Tests](badges/tests.svg)](badges/tests.svg)
[![Coverage](badges/coverage.svg)](badges/coverage.svg)

# ncolor <img src="https://github.com/kevinjohncutler/ncolor/blob/main/logo.png?raw=true" width="400" title="bacteria" alt="bacteria" align="right" vspace = "0">

Fast remapping of instance labels `1,2,3,...,M` to a smaller set of repeating, disjoint labels `1,2,...,N`. The [four color theorem](https://en.wikipedia.org/wiki/Four_color_theorem) guarantees `N ≤ 4` for any 2D segmentation. The picker will fall back to `N = 5` if a 4-coloring cannot be found within the time budget. Also works for 3D labels (`< 8` typically) and higher dimensions.

## Install

```bash
pip install ncolor
```

Pulls a precompiled wheel (Linux x86_64 / aarch64, macOS arm64, Windows AMD64). Only runtime deps are `numpy` and `platformdirs`.

## Usage

```python
import ncolor
ncolor_masks = ncolor.label(masks)                    # 4-color
ncolor_masks, n = ncolor.label(masks, return_n=True)  # +color count
labels = ncolor.format_labels(masks)                  # compact to 1..N
labels = ncolor.format_labels(masks, clean=True)      # + split disjoint pieces, drop tiny components
```

Expand-labels is on by default (so that close-but-not-touching cells tend to be assigned distinct colors). Pass `expand=False` for 3D inputs where cells can over-expand. Thanks to Ryan Peters ([@ryanirl](https://github.com/ryanirl)) for the original suggestion.

## New in v2

v2 is a complete C++ rewrite. Every stage of the pipeline including label expansion has been optimized, resulting in 7–12× speedups end-to-end. The new default expand removes 1-pixel bridges and spurs before the picker sees them, and an auto-soft constraint refines the hard 4-coloring via local search to differentiate near-adjacent cells. Together these break the K₅-shaped convergence clusters that forced the v1 numba pipeline up to `N = 5`. See [CHANGELOG.md](CHANGELOG.md) for the full list of changes and the migration table from v1.

The rewrite also brings drop-in C++ replacements for the scikit-image and `scipy.ndimage` / `edt` calls the old pipeline relied on, with no extra install:

| ncolor | replaces | typical speedup |
|---|---|---|
| `ncolor.connected_components` | `skimage.measure.label` | 1.5–3× |
| `ncolor.regionprops` | `skimage.measure.regionprops` (vectorized subset: area / bbox / centroid) | 1.5–3× |
| `ncolor.expand_labels` | `skimage.segmentation.expand_labels` + `scipy.ndimage.distance_transform_edt` | ND-parallel L1 / L2 in-engine; no scipy or `edt` dependency |
| `ncolor.delete_spurs` | hand-rolled morphology / not in scikit-image | ND, parallel |


For C++ engine internals, file-by-file architecture, and threadpool design, see [ARCHITECTURE.md](ARCHITECTURE.md).
