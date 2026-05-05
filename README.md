[![PyPI version](https://img.shields.io/pypi/v/ncolor.svg?color=green)](https://pypi.org/project/ncolor/)
[![Downloads](https://static.pepy.tech/personalized-badge/ncolor?period=total&units=international_system&left_color=grey&right_color=green&left_text=Downloads)](https://pepy.tech/project/ncolor)
[![Tests](badges/tests.svg)](badges/tests.svg)
[![Coverage](badges/coverage.svg)](badges/coverage.svg)

# ncolor <img src="https://github.com/kevinjohncutler/ncolor/blob/main/logo.png?raw=true" width="400" title="bacteria" alt="bacteria" align="right" vspace = "0">

Fast remapping of instance labels 1,2,3,...,M to a smaller set of repeating, disjoint labels, 1,2,...,N. The [four color theorem](https://en.wikipedia.org/wiki/Four_color_theorem) guarantees that at most four colors are required for any 2D segmentation/map, but the stochastic algorithms of `ncolor` will opt for 5 or 6 to give an acceptable result if it fails to find a 4-color mapping quickly. Also works for 3D labels (&lt;8 colors typically required) and perhaps higher dimensions as well.

### Usage
If you have an integer array called `masks`, you may transform it into an N-color representation as follows:

```python
import ncolor 
ncolor_masks = ncolor.label(masks)
```

If you need the number of unique labels returned:
```python
ncolor_masks, num_labels = ncolor.label(masks,return_n=True)

```
If you need to convert back to `0,...,N` object labels:
```python
labels = ncolor.format_labels(ncolor_masks,clean=True)

```

Note that `format_labels` with ```clean=True``` will also remove small labels (<9px) by default. This behavior can be changed with the `min_area` parameter. 

    
The integer array `ncolor_masks` can be visualized using any color map you prefer. The example in this README uses the viridis colormap. See `example.ipynb` for more details.

Thanks to Ryan Peters ([@ryanirl](https://github.com/ryanirl)) for suggesting the `expand_labels` function. This is applied by default to 2D images (optionally for 3D images with `expand=True`, but this can give bad results since objects in 3D have a lot more wiggle room to make contact when expanded). This preprocessing step eliminates cases where close (but not touching) or dispersed objects previously received the same label. I dug a layer back to use `ndimage.distance_transform_edt` for a speed boost. If undesired for 2D images, use `expand=False`. 

## Backends

Default backend is the C++ engine in `ncolor._backend` — built from
`cpp/binding.cpp` into a single pybind11 extension `ncolor._backend._impl`.
It implements `expand` → `find_pairs` → `color` → `apply_lut` end-to-end
with a persistent thread pool and bit-identical output to the numba
reference.

Set `NCOLOR_BACKEND=numba` to fall back to the original numba
implementation in `ncolor._numba_legacy/`. Useful as a sanity check when
diagnosing a regression; the legacy path will be removed once the C++
engine has been the default for a release cycle.

The C++ engine auto-calibrates its thread count once per machine
(~50–300 ms hidden under the user's first `import ncolor`) and caches
the result. See "Calibration cache & NAS-mounted source" below for the
cache location and lifecycle.

## Calibration cache & NAS-mounted source

ncolor stores two things in the per-user cache directory resolved by
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

If you `pip install` ncolor from a wheel (the normal path), the compiled
extension lives in `site-packages/`, which is on local disk on every
reasonable system. The loader in `ncolor._backend.__init__` fast-paths
to a direct `importlib` load and never touches the platformdirs cache.

The cache only kicks in when the **package source itself lives on a
network filesystem** — i.e., a developer install (`pip install -e .` or
`setup.py build_ext --inplace`) where the source tree is NAS-mounted
(`smbfs` / `nfs` / UNC). Two OS-level bugs break direct loading in that
case:

* **macOS smbfs** — `dyld` calls `fcntl()` for code-signature validation
  during `dlopen`, and SMB hangs on those calls; `dlopen` blocks
  indefinitely in `JustInTimeLoader::withRegions`.
* **Windows UNC** — `LoadLibrary` raises *Access is denied* on certain
  server configurations (depends on the share's ACLs).

The loader detects this case (smbfs / nfs / afpfs on POSIX, UNC anchor
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
