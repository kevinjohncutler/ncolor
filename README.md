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

Pulls a precompiled wheel (Linux x86_64 / aarch64, macOS arm64 / x86_64, Windows AMD64 / ARM64) for CPython 3.11 to 3.15. Only runtime deps are `numpy` and `platformdirs`.

## Usage

```python
import ncolor
ncolor_masks = ncolor.label(masks)                    # 4-color
ncolor_masks, n = ncolor.label(masks, return_n=True)  # + color count
labels = ncolor.format_labels(masks)                  # compact to 1..N
labels = ncolor.format_labels(masks, clean=True)      # + split disjoint pieces, drop tiny components
ncolor.release_buffers()                              # free the scratch kept between calls (optional)
```

Expand-labels is on by default (so that close-but-not-touching cells tend to be assigned distinct colors). Pass `expand=False` for 3D inputs where cells can over-expand. Thanks to Ryan Peters ([@ryanirl](https://github.com/ryanirl)) for the original suggestion.

Threading needs nothing special. A lone call uses every core; calls that overlap are handed narrower engines whose threads add up to about one machine, so several images color at once:

```python
import concurrent.futures as cf, ncolor

with cf.ThreadPoolExecutor(4) as pool:
    colored = list(pool.map(ncolor.label, images))   # 1.3-2x on 4 threads
```

Nothing is measured or configured for this. Whether spreading calls out beats taking turns was swept over four machines and eight image sizes, and it depends on the image at least as much as on the machine, so there is no per-machine answer worth caching: on an 8-core i9 the same arrangement ran 1.3x faster at 512 by 512, 0.8x at 1024, and 1.1x at 4096. Averaged over sizes it pays on every machine tested, from 1.07x on that i9 to 1.9x on a 64-core Threadripper, with a worst single case of 0.76x. `NCOLOR_MAX_ENGINES=1` turns it off and calls take turns on one pool as they always have. Engines are built only when calls actually overlap, so a single-threaded program holds one, and once the parallel phase is over calls go back to full width. At most `NCOLOR_MAX_ENGINES` (4) are created; each keeps the working set of the largest image it has seen, roughly 20 bytes per pixel. `ncolor.Engine` does the same by hand for callers who would rather size the threads themselves.

Any integer, bool or float label array is accepted; the cast to the engine's int32 runs in parallel inside the call. Labels beyond the int32 range are compacted automatically by `label` and `format_labels`. The engine keeps the scratch memory of the largest image it has processed until `release_buffers()` is called.

## Repeated coloring

For an unchanged label image, prepare its expansion and contact graphs once:

```python
prepared = ncolor.prepare_labels(masks, expand_mode="clean")
colored = prepared.color(n=4)
colored_again = prepared.color(n=5)
print(prepared.nbytes)  # owned array bytes
```

Preparation owns a snapshot, so later edits to `masks` do not affect it.
Create a new snapshot when the geometry, expansion, connectivity, or weight
settings change. Each coloring call can change the color target, search
depth, and perceptual palette. `Engine.prepare_labels(...)` and
`prepared.color(engine=engine)` provide explicit worker control. A snapshot
retains one, two, or four bytes per pixel according to label count, plus its
graph arrays. Release it when done. `prepared.color(return_lut=True)` skips
image rendering when no output buffer is supplied.
See [performance exploration](PERFORMANCE_EXPLORATION.md) for measured
tradeoffs and experiments that were not promoted into the library.

## Vector geometry (GeoDataFrames, GeoJSON)

Data that starts out as polygons does not have to be rasterized to be colored. `ncolor.geo.label` reads the adjacency straight off the geometry, so nothing is lost on the way in:

```bash
pip install "ncolor[geo]"      # adds shapely; geopandas is optional
```

```python
import geopandas as gpd
from ncolor import geo

gdf = gpd.read_file("cells.geojson")
gdf["color"] = geo.label(gdf)                         # 4-color, one per feature
gdf.plot(column="color", cmap="viridis")
```

It accepts a `GeoDataFrame`, a `GeoSeries`, a GeoJSON file / string / `dict`, a list of Shapely geometries, or anything with a `__geo_interface__`, and returns a `uint8` array of colors in `1..n` in input order (0 for missing or empty geometries). Pass `return_frame=True` for a `GeoDataFrame` with the colors in a column instead.

Two knobs mirror the raster ones:

| kwarg | meaning | raster analogue |
|---|---|---|
| `min_shared_length` | contact must be a shared *border*, not a point. The default `0.0` drops corner-only touches (the "Four Corners" rule), which is what keeps the graph planar and 4-colorable. `None` counts them, and then `n = 5` is likely. | `conn=1` vs `conn=2` |
| `tolerance` | treat features within this distance as touching, for polygons whose neighbors are separated by a hairline gap after vectorization or reprojection. In CRS units. | `connect_radius` |

With a `tolerance`, a positive `min_shared_length` is the vector form of the raster `min_contact` filter: dilating polygons to close gaps also creates one-point leaks between features that were never really neighbors, and those are what push a coloring from 4 up to 5. On a 5000-cell Voronoi tessellation with hairline gaps, `tolerance=0.2` alone needs 5 colors; adding `min_shared_length=1.0` brings it back to 4.

Pairs the filters reject become *soft* constraints: they still get different colors where that is free, without constraining the coloring. `ncolor.geo.connect` returns the adjacency pairs alone (0-indexed by row), and `ncolor.color_graph` colors any edge list you build yourself:

```python
colors = ncolor.color_graph(edges, n_vertices=len(nodes))   # 0-indexed pairs
```

## New in v2

v2 is a complete C++ rewrite. Every stage of the pipeline including label expansion has been optimized, resulting in 7–12× speedups end-to-end. The new default expand removes 1-pixel bridges and spurs before the picker sees them, and an auto-soft constraint refines the hard 4-coloring via local search to differentiate near-adjacent cells. Together these break the K₅-shaped convergence clusters that forced the v1 numba pipeline up to `N = 5`. See [CHANGELOG.md](CHANGELOG.md) for the full list of changes and the migration table from v1.

The rewrite also brings drop-in C++ replacements for the image-analysis and distance-transform calls the old pipeline relied on, with no extra install:

| ncolor | replaces | typical speedup |
|---|---|---|
| `ncolor.connected_components` | `skimage.measure.label` | 1.5–3× |
| `ncolor.regionprops` | `skimage.measure.regionprops` (vectorized subset: area / bbox / centroid) | 1.5–3× |
| `ncolor.expand_labels` | `skimage.segmentation.expand_labels` + `scipy.ndimage.distance_transform_edt` | Parallel L1 / L2 expansion in any dimension; no extra dependency |
| `ncolor.delete_spurs` | hand-rolled morphology / not in scikit-image | ND, parallel |


Connected components is a separate utility and is also used by
`format_labels(clean=True)` to split disconnected pieces of a label.
Ordinary `label()` calls do not run it: their `expand_mode="clean"` uses
bridge and spur cleanup within label expansion. Large component jobs now
use parallel slabs and boundary merging; small arrays stay serial.
`Engine(n_threads=4).connected_components(mask)` selects an explicit worker
budget. Component IDs retain the serial scan order.

The latest measured optimizations preserve label propagation while avoiding
an unnecessary final distance transpose and an intermediate input copy.
Distance-returning and distance-weighted calls retain the distances they
need. See [PERFORMANCE_EXPLORATION.md](PERFORMANCE_EXPLORATION.md) for direct
before/after comparisons, memory costs, and experiments that were not kept.


For C++ engine internals, file-by-file architecture, and threadpool design, see [ARCHITECTURE.md](ARCHITECTURE.md).
