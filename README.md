[![PyPI version](https://img.shields.io/pypi/v/ncolor.svg?color=green)](https://pypi.org/project/ncolor/)
[![Downloads](https://static.pepy.tech/personalized-badge/ncolor?period=total&units=international_system&left_color=gray&right_color=green&left_text=Downloads)](https://pepy.tech/project/ncolor)
[![Tests](badges/tests.svg)](badges/tests.svg)
[![Coverage](badges/coverage.svg)](badges/coverage.svg)

# ncolor <img src="https://github.com/kevinjohncutler/ncolor/blob/main/logo.png?raw=true" width="400" title="bacteria" alt="bacteria" align="right" vspace = "0">

Fast remapping of instance labels 1,2,3,...,M to a smaller set of repeating, disjoint labels, 1,2,...,N. The [four color theorem](https://en.wikipedia.org/wiki/Four_color_theorem) guarantees that at most four colors are required for any 2D segmentation/map, but the stochastic algorithms of `ncolor` will opt for 5 or 6 to give an acceptable result if it fails to find a 4-color mapping quickly. Also works for 3D labels (&lt;8 colors typically required) and perhaps higher dimensions as well.

Versions prior to 2.0 used a numba implementation. **v2.0 is a complete C++ rewrite** that's roughly **5–30× faster** on the core labeling pipeline (`label`, `expand_labels`) and **60–180× faster** on the cleanup helpers (`format_labels(clean=True)`, `delete_spurs`), with runtime dependencies shrunk to just `numpy` + `platformdirs`.

## Install

```bash
pip install ncolor
```

This pulls a precompiled wheel (Linux x86_64 / aarch64, macOS arm64,
Windows AMD64) and only `numpy` + `platformdirs`. The entire public
API — `label`, `expand_labels`, `format_labels` (including
`clean=True`), `connected_components`, `regionprops`,
`delete_spurs` — runs C++-only on this minimal install.

## Usage

If you have an integer array called `masks`, you may transform it into an N-color representation as follows:

```python
import ncolor 
ncolor_masks = ncolor.label(masks)
```

If you need the number of unique labels returned:
```python
ncolor_masks, num_labels = ncolor.label(masks, return_n=True)
```

To transform a label image into contiguous `1..N`:
```python
labels = ncolor.format_labels(masks)
```

Pass `clean=True` to additionally split disjoint components and drop
regions smaller than `min_area`:
```python
labels = ncolor.format_labels(masks, clean=True, min_area=9)
```

The integer array `ncolor_masks` can be visualized using any color map you prefer. The example in this README uses the viridis colormap. See `example.ipynb` for more details.

Thanks to Ryan Peters ([@ryanirl](https://github.com/ryanirl)) for suggesting the `expand_labels` function. This is applied by default to 2D images (optionally for 3D images with `expand=True`, but this can give bad results since objects in 3D have a lot more wiggle room to make contact when expanded). This preprocessing step eliminates cases where close (but not touching) or dispersed objects previously received the same label. The C++ engine ships its own ND parallel L1 (Saito-Toriwaki separable sweep) and L2 (Felzenszwalb parabolic envelope) implementations — no `scipy.ndimage`/`edt` dependency. If undesired for 2D images, use `expand=False`.

`ncolor` also ships drop-ins for the two scikit-image entry points the
above pipeline used to call:

```python
labels, n = ncolor.connected_components(mask, conn=2)   # like skimage.measure.label
props     = ncolor.regionprops(labels, n)               # area, bbox, centroid (vectorized)
```

These run 1.5–3× faster than `scikit-image` on typical instance-mask
inputs.

## Backend

The C++ engine in `ncolor._backend` is the only backend, built from
`cpp/binding.cpp` into a single pybind11 extension
`ncolor._backend._impl`. It owns a persistent thread pool and runs
`expand` → `find_pairs` → `color` → `apply_lut` end-to-end under one
`gil_scoped_release`.

The C++ engine auto-calibrates its thread count once per machine
(~50–300 ms hidden under the user's first `import ncolor`) and caches
the result. See "Calibration cache & NAS-mounted source" below for the
cache location and lifecycle.

## Calibration cache & NAS-mounted source

`ncolor` stores two things in the per-user cache directory resolved by
[platformdirs](https://pypi.org/project/platformdirs/) at
`platformdirs.user_cache_dir("ncolor")`:

1. `smt_threads.json` — the per-host SMT/HT-aware thread count chosen
   by `ncolor._backend._smt.calibrate()`. Keyed by `(hostname, CPU
   model)`. Read on every `Solver()` / `ConnectEngine()` /
   `ExpandEngine()` construction via `auto_threads()`. Skip calibration
   with `NCOLOR_NO_CALIBRATE=1` (CI / cross-compile builds).
2. `lib/<mtime_ns>_<size>/_impl.<so|pyd>` — only used when the package
   directory is on a network filesystem. The compiled extension is
   copied here and `dlopen`'d locally. See below.

Both paths resolve to the OS-native location:

| OS      | Path                                       |
|---------|--------------------------------------------|
| Linux   | `~/.cache/ncolor/`                         |
| macOS   | `~/Library/Caches/ncolor/`                 |
| Windows | `%LOCALAPPDATA%\ncolor\ncolor\Cache\`      |

### Why the local-disk `.so` cache exists

If you `pip install ncolor` from a wheel (the normal path), the compiled
extension lives in `site-packages/`, which is on local disk on every
reasonable system. The loader in `ncolor._backend.__init__` fast-paths
to a direct `importlib` load and never touches the platformdirs cache.

The cache only kicks in when the **package source itself lives on a
network filesystem** — i.e., a developer install (`pip install -e .` or
`setup.py build_ext --inplace`) where the source tree is NAS-mounted
(`smbfs` / `nfs` / `UNC`). Two OS-level bugs break direct loading in that
case:

* **macOS smbfs** — `dyld` calls `fcntl()` for code-signature validation
  during `dlopen`, and SMB hangs on those calls; `dlopen` blocks
  indefinitely in `JustInTimeLoader::withRegions`.
* **Windows UNC** — `LoadLibrary` raises *Access is denied* on certain
  server configurations (depends on the share's ACLs).

The loader detects this case (`smbfs` / `nfs` / `afpfs` on POSIX, `UNC` anchor
on Windows) and copies the `.so` / `.pyd` to the local cache before
`dlopen`'ing from there. Cache key is `(mtime_ns, size)` so a rebuild
gets a fresh local path — dyld retains stale path-keyed state from
prior failed loads at the same path, so reusing the same path can still
hang. On macOS the loader also strips `com.apple.quarantine` from the
copy.

Net effect: end-users with pip-installed wheels never see this code
path. Developers running an editable install from a NAS-mounted source
tree get transparent local-cache `.so` loading without doing anything.
The artifact ships *somewhere* readable (in the package directory, since
that's the only thing wheels can do); the load-time NAS workaround is
purely a runtime concern.

To clear caches manually:

```bash
# macOS
rm -rf ~/Library/Caches/ncolor

# Linux
rm -rf ~/.cache/ncolor

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\ncolor"
```

The next `import ncolor` will recalibrate (~250 ms) and recopy the
`.so` if needed.
