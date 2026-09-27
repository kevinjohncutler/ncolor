[![PyPI version](https://img.shields.io/pypi/v/ncolor.svg?color=green)](https://pypi.org/project/ncolor/)
[![Downloads](https://static.pepy.tech/personalized-badge/ncolor?period=total&units=international_system&left_color=gray&right_color=green&left_text=Downloads)](https://pepy.tech/project/ncolor)
[![Tests](badges/tests.svg)](badges/tests.svg)
[![Coverage](badges/coverage.svg)](badges/coverage.svg)

# ncolor <img src="https://github.com/kevinjohncutler/ncolor/blob/main/logo.png?raw=true" width="400" title="bacteria" alt="bacteria" align="right" vspace = "0">

Fast remapping of instance labels `1,2,3,...,M` to a smaller set of repeating, disjoint labels `1,2,...,N`. The [four color theorem](https://en.wikipedia.org/wiki/Four_color_theorem) guarantees `N <= 4` for planar adjacency graphs. Raster contacts can produce nonplanar graphs. The picker will fall back to `N = 5` if a 4-coloring cannot be found within the time budget. Also works for 3D labels (`< 8` typically) and higher dimensions.

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
    colored = list(pool.map(ncolor.label, images))   # 1.4-3.5x faster than taking turns
```

Nothing is calibrated or configured for this. With four threads labeling images from 512 by 512 to 4096 by 4096, overlapping calls ran 1.4x to 3.5x faster than taking turns on an Apple M5 Max and an AMD Ryzen 9 7950X ([benchmarks](BENCHMARKS.md)). `NCOLOR_MAX_ENGINES=1` turns it off and calls take turns on one pool as they always have. Engines are built only when calls actually overlap, so a single-threaded program holds one, and once the parallel phase is over calls go back to full width. At most `NCOLOR_MAX_ENGINES` (4) are created; each keeps the working set of the largest image it has seen, roughly 20 bytes per pixel. `ncolor.Engine` does the same by hand for callers who would rather size the threads themselves.

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
uses one, two, or four bytes per pixel according to label count, plus its
graph arrays. Large sparse images can instead store only foreground
positions and labels. Empty snapshots retain no per-pixel map. Compact
storage reduces snapshot memory and repeated rendering time, though
preparation can cost more. The snapshot's memory is freed when the object is
deleted (`del prepared`). `prepared.color(return_lut=True)` skips image
rendering when no output buffer is supplied.

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

v2 is a complete C++ rewrite. With matched settings, `label` runs 6.3x to 24.1x faster than 1.5.3 on 4 workers and 2.65x to 7.4x faster on one. These are ranges over five test images on an Apple M5 Max and an AMD Ryzen 9 7950X; see [BENCHMARKS.md](BENCHMARKS.md) for per-image times, methods, and how to reproduce them. The new default expand removes 1-pixel bridges and spurs before the picker sees them, and an auto-soft constraint refines the hard 4-coloring via local search to differentiate near-adjacent cells. Together these break the K₅-shaped convergence clusters that forced the v1 numba pipeline up to `N = 5`. See [CHANGELOG.md](CHANGELOG.md) for the full list of changes and the migration table from v1.

The rewrite also brings drop-in C++ replacements for the image-analysis and distance-transform calls the old pipeline relied on, with no extra install:

| ncolor | replaces | speed, 1 worker | speed, 4 workers |
|---|---|---|---|
| `ncolor.connected_components` | `skimage.measure.label` | 0.38x to 2.4x | 0.92x to 6.4x |
| `ncolor.regionprops` | `skimage.measure.regionprops_table` (area, bbox, centroid) | 1.95x to 27.8x | 4.7x to 27.8x |
| `ncolor.expand_labels` | `scipy.ndimage.distance_transform_edt` nearest-label fill | 2.6x to 7.1x | 6.6x to 24.2x |
| `ncolor.expand_labels` | `skimage.segmentation.expand_labels` | 3.4x to 8.3x | 10.0x to 27.9x |
| `ncolor.delete_spurs` | hand-rolled morphology, not in scikit-image | not compared | not compared |

Speeds are ranges over the benchmark inputs on both machines, measured against SciPy 1.18.1 and scikit-image 0.26.0, which run these operations on one thread. `expand_labels` also supports L1 expansion in any dimension. scikit-image's `measure.label` is faster on some dense masks: down to 0.38x on one thread on a 70% filled 3D volume at full connectivity, and 0.92x with four workers on the same volume.

All of these run on the engine's workers, and `Engine(n_threads=...)` sets the budget per call. Results never depend on the worker count.

Connected components is a separate utility and is also used by
`format_labels(clean=True)` to split disconnected pieces of a label.
Ordinary `label()` calls do not run it: their `expand_mode="clean"` uses
bridge and spur cleanup within label expansion. Large component jobs now
use parallel slabs and boundary merging; small arrays stay serial.
`Engine(n_threads=4).connected_components(mask)` selects an explicit worker
budget. Component IDs retain the serial scan order.

For C++ engine internals, file-by-file architecture, and threadpool design, see [ARCHITECTURE.md](ARCHITECTURE.md).
