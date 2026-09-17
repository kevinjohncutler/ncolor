# Benchmarks

Measured on 2026-09-17 against ncolor 1.5.3 (the last version 1 release),
ncolor 2.2.0, and the current SciPy and scikit-image releases. Ratios are
the other time divided by the ncolor time, so a value above 1 means ncolor
is faster. Every ncolor, SciPy, and scikit-image output was checked against
a reference before its time was recorded.

## Summary

Ranges cover both machines and every input in the tables below.

| Operation | Compared with | 4 workers | 1 worker |
|---|---|---|---|
| `label` | ncolor 1.5.3, matched settings | 6.6x to 24.4x | 2.6x to 7.3x |
| `label` | ncolor 2.2.0, default settings | 0.96x to 1.18x | 1.01x to 1.25x |
| `expand_labels` | SciPy `distance_transform_edt` feature transform | 7.0x to 25.4x | 2.6x to 7.2x |
| `expand_labels` | scikit-image `expand_labels` | 10.7x to 28.8x | 3.4x to 8.4x |
| `regionprops` | scikit-image `regionprops_table` | 1.9x to 27.2x | same (single-threaded) |
| `connected_components` | scikit-image `measure.label` | 1.0x to 6.6x | 0.38x to 2.4x |
| four threads calling `label` | the same calls taking turns | 1.5x to 3.2x | not applicable |

SciPy and scikit-image run these operations on one thread, so the one-worker
column is the like-for-like comparison. On one thread, scikit-image labels
dense masks at full connectivity faster than ncolor: 0.38x on a 70% filled
2 x 513 x 517 volume, and 0.81x on the Ryzen for 70% filled 2D masks with diagonal
connectivity.

## Environment

| Item | Apple M5 Max | AMD Ryzen 9 7950X |
|---|---|---|
| Operating system | macOS 27.0 | Ubuntu 26.04 LTS |
| Python | 3.13.14 | 3.12.10 |
| Compiler | Apple clang 21.0, -O3 -march=native | gcc 15.2, -O3 -march=native |

Both machines used numpy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, numba
0.67.0, and fastremap 1.20.0, installed in a fresh virtual environment.
ncolor 2.2.0 and the current code were built from source with the same
flags. Published wheels target a baseline instruction set rather than
`-march=native`, so their times can differ from these.

## Method

Each version runs in its own process. A process warms each operation with
four calls, then times fifteen calls and records their median. The version
order rotates between rounds, and the tables report the median over rounds:
six rounds at 4 workers and three at 1 worker. ncolor 1.5.3 uses Numba with
the same thread count.

`label` runs in two configurations. Default settings are each version's
defaults. Matched settings use face connectivity, no input formatting,
standard expansion, and no soft constraints, which is the closest
configuration to version 1. The version 1 defaults fail the foreground check
on these inputs, so 1.5.3 is compared only with matched settings.

The label images are the logo from `test_files/example.png`, the
`test_files/synthetic_800.npz` fixture, and boxes placed at random. The masks
are random pixels at 10% and 70% fill.

`expand_labels` is compared with the nearest-label fill from SciPy's
Euclidean distance transform (`return_indices=True`) and with scikit-image's
`expand_labels(distance=inf)`. Pixels equidistant from two labels may be
assigned to either label; each such pixel is checked to be a true tie.

The concurrent-caller rows time 16 distinct 23%-filled images, spread over
four threads. They compare ncolor's default engine allocation with
`NCOLOR_MAX_ENGINES=1`, where calls take turns on one pool.

## Results

### label, 4 workers

| Image | CPU | Default ms | vs 2.2.0 | Matched ms | vs 2.2.0 | vs 1.5.3 |
|---|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.33 | 1.05x | 0.19 | 1.00x | 8.57x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.45 | 0.96x | 0.24 | 0.96x | 6.59x |
| synthetic, 900 x 900, 682 labels | M5 Max | 3.34 | 1.06x | 2.29 | 1.02x | 11.72x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 4.03 | 1.04x | 2.75 | 1.04x | 8.80x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.97 | 1.13x | 1.35 | 1.16x | 22.72x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 2.80 | 1.18x | 1.96 | 1.26x | 20.23x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 9.21 | 1.18x | 6.40 | 1.24x | 24.43x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 16.09 | 1.15x | 11.07 | 1.28x | 16.59x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 6.55 | 1.02x | 2.49 | 1.00x | 21.42x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 8.74 | 1.00x | 3.27 | 1.05x | 19.51x |

### label, 1 worker

| Image | CPU | Default ms | vs 2.2.0 | Matched ms | vs 2.2.0 | vs 1.5.3 |
|---|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.54 | 1.04x | 0.31 | 1.03x | 4.94x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.91 | 1.01x | 0.53 | 1.01x | 3.03x |
| synthetic, 900 x 900, 682 labels | M5 Max | 9.87 | 1.04x | 7.13 | 1.01x | 3.72x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 13.13 | 1.01x | 9.33 | 1.01x | 2.59x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 6.28 | 1.18x | 4.20 | 1.21x | 7.27x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 9.80 | 1.13x | 6.81 | 1.20x | 5.80x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 30.83 | 1.25x | 22.20 | 1.29x | 7.04x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 47.25 | 1.16x | 33.72 | 1.24x | 5.45x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 22.03 | 1.02x | 7.71 | 1.03x | 7.21x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 30.65 | 1.01x | 10.77 | 1.05x | 5.95x |

### expand_labels, 4 workers

| Image | CPU | ncolor ms | vs 2.2.0 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.08 | 1.13x | 12.72x | 14.79x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.16 | 0.82x | 7.01x | 10.68x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.58 | 1.06x | 12.51x | 14.58x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 2.06 | 1.07x | 9.43x | 12.44x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.05 | 1.17x | 25.40x | 28.84x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.76 | 1.21x | 21.25x | 24.92x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 5.79 | 1.19x | 23.58x | 26.16x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 11.92 | 1.27x | 13.93x | 17.04x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.55 | 1.08x | 21.61x | 24.92x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 2.30 | 1.12x | 19.78x | 25.71x |

### expand_labels, 1 worker

| Image | CPU | ncolor ms | vs 2.2.0 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.23 | 1.05x | 4.73x | 5.56x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.32 | 1.05x | 3.55x | 5.44x |
| synthetic, 900 x 900, 682 labels | M5 Max | 5.66 | 1.02x | 3.56x | 4.21x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 7.55 | 1.03x | 2.57x | 3.41x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 3.68 | 1.14x | 7.20x | 8.37x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 6.06 | 1.15x | 6.21x | 7.27x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 21.32 | 1.20x | 6.58x | 7.30x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 31.37 | 1.21x | 5.29x | 6.46x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 5.54 | 1.05x | 6.10x | 7.04x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 8.20 | 1.09x | 5.60x | 7.19x |

### regionprops

| Image | CPU | ncolor ms | vs regionprops | vs regionprops_table |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.12 | 19.24x | 20.10x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.15 | 26.06x | 27.16x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.52 | 7.82x | 8.12x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.80 | 10.50x | 10.94x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.57 | 2.03x | 2.19x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.44 | 2.79x | 2.89x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 3.49 | 3.26x | 3.48x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 5.73 | 2.77x | 2.88x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.50 | 1.87x | 1.90x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 1.28 | 2.32x | 2.40x |

### connected_components

| Mask | Fill | Conn | CPU | 4 workers ms | vs 2.2.0 | vs scikit-image | 1 worker ms | vs 2.2.0 | vs scikit-image |
|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 10% | 1 | M5 Max | 0.81 | 2.32x | 2.63x | 1.88 | 1.02x | 1.14x |
| 1024 x 1024 | 10% | 1 | Ryzen 9 7950X | 0.74 | 2.36x | 3.10x | 1.70 | 1.03x | 1.36x |
| 1024 x 1024 | 10% | 2 | M5 Max | 0.86 | 2.47x | 2.56x | 2.12 | 1.04x | 1.05x |
| 1024 x 1024 | 10% | 2 | Ryzen 9 7950X | 0.82 | 2.46x | 2.85x | 2.14 | 0.95x | 1.11x |
| 1024 x 1024 | 70% | 1 | M5 Max | 1.91 | 2.76x | 2.78x | 5.41 | 1.01x | 0.98x |
| 1024 x 1024 | 70% | 1 | Ryzen 9 7950X | 1.85 | 2.85x | 2.96x | 5.43 | 0.97x | 1.01x |
| 1024 x 1024 | 70% | 2 | M5 Max | 2.02 | 2.82x | 2.74x | 5.78 | 1.01x | 0.95x |
| 1024 x 1024 | 70% | 2 | Ryzen 9 7950X | 2.18 | 2.97x | 2.48x | 6.70 | 0.97x | 0.81x |
| 2048 x 2048 | 10% | 1 | M5 Max | 3.12 | 2.42x | 2.73x | 7.54 | 1.00x | 1.13x |
| 2048 x 2048 | 10% | 1 | Ryzen 9 7950X | 3.52 | 2.18x | 2.73x | 7.11 | 1.03x | 1.35x |
| 2048 x 2048 | 10% | 2 | M5 Max | 3.32 | 2.58x | 2.69x | 8.45 | 1.01x | 1.07x |
| 2048 x 2048 | 10% | 2 | Ryzen 9 7950X | 3.85 | 2.20x | 2.54x | 8.88 | 0.95x | 1.10x |
| 2048 x 2048 | 70% | 1 | M5 Max | 7.45 | 2.86x | 2.85x | 21.49 | 0.99x | 1.00x |
| 2048 x 2048 | 70% | 1 | Ryzen 9 7950X | 7.81 | 2.74x | 2.83x | 21.89 | 0.98x | 1.01x |
| 2048 x 2048 | 70% | 2 | M5 Max | 7.89 | 2.94x | 2.80x | 23.15 | 1.04x | 0.96x |
| 2048 x 2048 | 70% | 2 | Ryzen 9 7950X | 9.00 | 2.91x | 2.43x | 26.91 | 0.97x | 0.81x |
| 2 x 513 x 517 | 10% | 1 | M5 Max | 0.65 | 2.24x | 2.38x | 1.43 | 1.03x | 1.11x |
| 2 x 513 x 517 | 10% | 1 | Ryzen 9 7950X | 0.69 | 2.10x | 2.45x | 1.44 | 1.03x | 1.18x |
| 2 x 513 x 517 | 10% | 3 | M5 Max | 1.62 | 1.60x | 1.46x | 2.57 | 1.01x | 0.94x |
| 2 x 513 x 517 | 10% | 3 | Ryzen 9 7950X | 1.52 | 1.68x | 1.65x | 2.46 | 1.04x | 1.02x |
| 2 x 513 x 517 | 70% | 1 | M5 Max | 2.11 | 2.64x | 1.84x | 5.58 | 1.00x | 0.70x |
| 2 x 513 x 517 | 70% | 1 | Ryzen 9 7950X | 2.05 | 2.83x | 2.03x | 5.42 | 1.07x | 0.77x |
| 2 x 513 x 517 | 70% | 3 | M5 Max | 6.45 | 2.64x | 1.01x | 16.84 | 1.01x | 0.38x |
| 2 x 513 x 517 | 70% | 3 | Ryzen 9 7950X | 5.86 | 2.83x | 1.06x | 15.82 | 1.05x | 0.39x |
| 96 x 96 x 96 | 10% | 1 | M5 Max | 0.80 | 2.33x | 4.10x | 1.83 | 1.05x | 1.81x |
| 96 x 96 x 96 | 10% | 1 | Ryzen 9 7950X | 0.73 | 2.33x | 5.08x | 1.75 | 0.96x | 2.11x |
| 96 x 96 x 96 | 10% | 3 | M5 Max | 1.03 | 2.50x | 5.94x | 2.57 | 1.04x | 2.41x |
| 96 x 96 x 96 | 10% | 3 | Ryzen 9 7950X | 0.99 | 2.72x | 6.61x | 2.69 | 1.00x | 2.44x |
| 96 x 96 x 96 | 70% | 1 | M5 Max | 2.18 | 2.71x | 3.51x | 6.02 | 1.02x | 1.27x |
| 96 x 96 x 96 | 70% | 1 | Ryzen 9 7950X | 2.20 | 2.86x | 4.02x | 6.54 | 0.96x | 1.33x |
| 96 x 96 x 96 | 70% | 3 | M5 Max | 5.60 | 2.66x | 2.87x | 14.94 | 1.04x | 1.08x |
| 96 x 96 x 96 | 70% | 3 | Ryzen 9 7950X | 5.93 | 3.09x | 2.69x | 17.92 | 1.02x | 0.89x |

### Four concurrent callers

| Image | CPU | Overlapping ms | Taking turns ms | Speedup |
|---|---|---|---|---|
| 512 x 512 | M5 Max | 6.6 | 21.1 | 3.17x |
| 512 x 512 | Ryzen 9 7950X | 9.2 | 22.4 | 2.44x |
| 1024 x 1024 | M5 Max | 26.7 | 57.7 | 2.16x |
| 1024 x 1024 | Ryzen 9 7950X | 45.5 | 76.5 | 1.68x |
| 2048 x 2048 | M5 Max | 283.9 | 427.3 | 1.51x |
| 2048 x 2048 | Ryzen 9 7950X | 362.2 | 577.1 | 1.59x |
| 4096 x 4096 | M5 Max | 703.3 | 1498.7 | 2.13x |
| 4096 x 4096 | Ryzen 9 7950X | 1107.2 | 2112.2 | 1.91x |


Times under a millisecond vary by up to 1.8x between rounds, so small
differences on the logo image are within noise.

## Reproducing

Export each version into its own directory, create a virtual environment
with the dependency versions above plus `pybind11`, `setuptools`,
`setuptools_scm`, and `platformdirs`, and build the two C++ versions in
place with `python setup.py build_ext --inplace`. Run the interpreter from
that environment directly; `python -S` would load packages from the base
interpreter instead.

```bash
export NCOLOR_NO_CALIBRATE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python bench/release_comparison.py corpus --output work/corpus.npz
python bench/release_comparison.py run --version current --source src \
  --revision "$(git rev-parse HEAD)" --round 1 --threads 4 \
  --corpus work/corpus.npz --output work/t4/round_1_current.json
python bench/release_comparison.py run --version 2.2.0 --source work/v2.2.0/src \
  --revision "$(git rev-parse 'v2.2.0^{commit}')" --round 1 --threads 4 \
  --corpus work/corpus.npz --output work/t4/round_1_2.2.0.json
python bench/release_comparison.py run --version 1.5.3 --source work/v1.5.3/src \
  --revision "$(git rev-parse 'v1.5.3^{commit}')" --round 1 --threads 4 \
  --corpus work/corpus.npz --output work/t4/round_1_1.5.3.json
python bench/release_comparison.py run --version external --round 1 --threads 4 \
  --corpus work/corpus.npz --output work/t4/round_1_external.json
# Repeat for more rounds in a rotated version order, and with --threads 1
# (and the thread variables set to 1) into work/t1.
python bench/release_comparison.py report --directory work/t4 --output work/t4/summary.json
python bench/release_comparison.py report --directory work/t1 --output work/t1/summary.json
python bench/concurrent_callers.py compare --sizes 512 1024 2048 4096 --rounds 5 > work/concurrency.txt
python bench/release_tables.py "CPU name=work"
```
