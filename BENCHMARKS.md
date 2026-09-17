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
| `label` | ncolor 1.5.3, matched settings | 6.3x to 24.1x | 2.6x to 7.2x |
| `label` | ncolor 2.2.0, default settings | 1.00x to 1.19x | 1.00x to 1.21x |
| `expand_labels` | SciPy `distance_transform_edt` feature transform | 6.9x to 24.8x | 2.6x to 7.1x |
| `expand_labels` | scikit-image `expand_labels` | 10.6x to 28.3x | 3.4x to 8.2x |
| `regionprops` | scikit-image `regionprops_table` | 4.1x to 26.6x | 1.9x to 26.0x |
| `connected_components` | scikit-image `measure.label` | 1.0x to 6.6x | 0.37x to 2.5x |
| four threads calling `label` | the same calls taking turns | 1.5x to 3.1x | not applicable |

SciPy and scikit-image run these operations on one thread, so the one-worker
column is the like-for-like comparison. On one thread, scikit-image labels
dense masks at full connectivity faster than ncolor: 0.37x on a 70% filled
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
| logo, 241 x 205, 160 labels | M5 Max | 0.32 | 1.10x | 0.19 | 0.99x | 8.02x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.47 | 1.01x | 0.25 | 0.99x | 6.32x |
| synthetic, 900 x 900, 682 labels | M5 Max | 3.28 | 1.08x | 2.25 | 1.04x | 11.56x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 4.05 | 1.17x | 2.78 | 1.04x | 8.71x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.95 | 1.13x | 1.33 | 1.16x | 22.39x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 2.82 | 1.18x | 1.96 | 1.26x | 20.29x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 9.09 | 1.19x | 6.39 | 1.24x | 24.10x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 16.29 | 1.12x | 11.18 | 1.23x | 16.45x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 6.47 | 1.04x | 2.49 | 1.01x | 21.31x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 8.74 | 1.00x | 3.27 | 1.05x | 19.52x |

### label, 1 worker

| Image | CPU | Default ms | vs 2.2.0 | Matched ms | vs 2.2.0 | vs 1.5.3 |
|---|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.53 | 1.06x | 0.31 | 1.06x | 5.14x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.91 | 1.00x | 0.55 | 0.99x | 2.92x |
| synthetic, 900 x 900, 682 labels | M5 Max | 9.84 | 1.05x | 7.05 | 1.02x | 3.73x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 13.16 | 1.01x | 9.38 | 1.01x | 2.58x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 6.10 | 1.20x | 4.20 | 1.20x | 7.19x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 9.77 | 1.13x | 6.85 | 1.19x | 5.82x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 30.75 | 1.21x | 22.36 | 1.24x | 6.86x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 47.50 | 1.16x | 33.55 | 1.23x | 5.46x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 21.86 | 1.05x | 7.67 | 1.05x | 6.87x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 30.45 | 1.01x | 10.72 | 1.05x | 5.92x |

### expand_labels, 4 workers

| Image | CPU | ncolor ms | vs 2.2.0 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.09 | 1.09x | 12.03x | 14.44x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.16 | 1.09x | 6.92x | 10.60x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.58 | 1.04x | 12.26x | 14.40x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 2.09 | 1.06x | 9.31x | 12.23x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.04 | 1.17x | 24.80x | 28.25x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.77 | 1.19x | 21.04x | 24.91x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 5.79 | 1.20x | 23.55x | 25.96x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 11.82 | 1.29x | 14.05x | 17.16x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.55 | 1.08x | 21.21x | 24.42x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 2.29 | 1.12x | 19.80x | 25.79x |

### expand_labels, 1 worker

| Image | CPU | ncolor ms | vs 2.2.0 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.22 | 1.06x | 4.78x | 5.80x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.33 | 1.02x | 3.43x | 5.23x |
| synthetic, 900 x 900, 682 labels | M5 Max | 5.54 | 1.03x | 3.53x | 4.11x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 7.63 | 1.02x | 2.55x | 3.35x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 3.62 | 1.15x | 7.14x | 8.22x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 6.06 | 1.15x | 6.13x | 7.25x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 21.83 | 1.10x | 6.20x | 6.94x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 31.30 | 1.18x | 5.31x | 6.48x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 5.35 | 1.10x | 6.26x | 7.16x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 8.19 | 1.09x | 5.55x | 7.11x |

### regionprops, 4 workers

| Image | CPU | ncolor ms | vs 2.2.0 | vs regionprops | vs regionprops_table |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.12 | 0.98x | 18.82x | 19.83x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.15 | 0.98x | 25.43x | 26.60x |
| synthetic, 900 x 900, 682 labels | M5 Max | 0.84 | 1.74x | 13.90x | 14.32x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.02 | 1.77x | 18.62x | 19.37x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 0.51 | 3.22x | 6.25x | 6.55x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 0.62 | 2.34x | 6.54x | 6.80x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 1.92 | 1.93x | 5.41x | 5.45x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 1.40 | 4.07x | 11.31x | 11.72x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 0.69 | 2.14x | 3.99x | 4.08x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 0.57 | 2.49x | 5.13x | 5.34x |

### regionprops, 1 worker

| Image | CPU | ncolor ms | vs 2.2.0 | vs regionprops | vs regionprops_table |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.11 | 0.99x | 20.19x | 21.08x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.15 | 0.97x | 25.04x | 26.02x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.51 | 0.96x | 7.67x | 7.96x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.80 | 1.00x | 10.56x | 10.98x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.31 | 1.21x | 2.56x | 2.59x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.25 | 1.14x | 3.20x | 3.32x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 3.33 | 0.93x | 3.53x | 3.64x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 4.99 | 1.14x | 3.19x | 3.30x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.49 | 1.01x | 1.88x | 1.91x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 1.31 | 1.09x | 2.26x | 2.34x |

### connected_components

| Mask | Fill | Conn | CPU | 4 workers ms | vs 2.2.0 | vs scikit-image | 1 worker ms | vs 2.2.0 | vs scikit-image |
|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 10% | 1 | M5 Max | 0.81 | 2.26x | 2.56x | 1.85 | 1.04x | 1.12x |
| 1024 x 1024 | 10% | 1 | Ryzen 9 7950X | 0.74 | 2.36x | 3.13x | 1.69 | 1.03x | 1.35x |
| 1024 x 1024 | 10% | 2 | M5 Max | 0.86 | 2.39x | 2.49x | 2.05 | 1.04x | 1.06x |
| 1024 x 1024 | 10% | 2 | Ryzen 9 7950X | 0.83 | 2.46x | 2.86x | 2.14 | 0.95x | 1.10x |
| 1024 x 1024 | 70% | 1 | M5 Max | 1.91 | 2.68x | 2.67x | 5.35 | 1.00x | 0.97x |
| 1024 x 1024 | 70% | 1 | Ryzen 9 7950X | 1.86 | 2.84x | 2.94x | 5.41 | 0.97x | 1.00x |
| 1024 x 1024 | 70% | 2 | M5 Max | 2.02 | 2.76x | 2.64x | 5.82 | 0.97x | 0.92x |
| 1024 x 1024 | 70% | 2 | Ryzen 9 7950X | 2.17 | 2.99x | 2.50x | 6.67 | 0.97x | 0.81x |
| 2048 x 2048 | 10% | 1 | M5 Max | 3.12 | 2.34x | 2.63x | 7.60 | 0.99x | 1.10x |
| 2048 x 2048 | 10% | 1 | Ryzen 9 7950X | 3.49 | 2.09x | 2.76x | 7.14 | 1.08x | 1.34x |
| 2048 x 2048 | 10% | 2 | M5 Max | 3.31 | 2.49x | 2.62x | 8.24 | 1.01x | 1.06x |
| 2048 x 2048 | 10% | 2 | Ryzen 9 7950X | 3.81 | 2.25x | 2.57x | 8.88 | 0.95x | 1.10x |
| 2048 x 2048 | 70% | 1 | M5 Max | 7.50 | 2.75x | 2.78x | 20.89 | 0.99x | 1.00x |
| 2048 x 2048 | 70% | 1 | Ryzen 9 7950X | 7.80 | 2.74x | 2.83x | 22.01 | 0.97x | 1.00x |
| 2048 x 2048 | 70% | 2 | M5 Max | 7.84 | 2.87x | 2.78x | 23.44 | 0.97x | 0.93x |
| 2048 x 2048 | 70% | 2 | Ryzen 9 7950X | 8.95 | 2.93x | 2.47x | 26.86 | 0.97x | 0.81x |
| 2 x 513 x 517 | 10% | 1 | M5 Max | 0.65 | 2.19x | 2.34x | 1.45 | 1.00x | 1.04x |
| 2 x 513 x 517 | 10% | 1 | Ryzen 9 7950X | 0.75 | 2.00x | 2.27x | 1.44 | 1.03x | 1.17x |
| 2 x 513 x 517 | 10% | 3 | M5 Max | 1.62 | 1.56x | 1.44x | 2.58 | 0.99x | 0.90x |
| 2 x 513 x 517 | 10% | 3 | Ryzen 9 7950X | 1.52 | 1.69x | 1.64x | 2.48 | 1.03x | 1.01x |
| 2 x 513 x 517 | 70% | 1 | M5 Max | 2.11 | 2.61x | 1.81x | 5.61 | 0.98x | 0.67x |
| 2 x 513 x 517 | 70% | 1 | Ryzen 9 7950X | 2.09 | 2.77x | 2.00x | 5.43 | 1.06x | 0.77x |
| 2 x 513 x 517 | 70% | 3 | M5 Max | 6.39 | 2.63x | 0.97x | 17.12 | 0.99x | 0.37x |
| 2 x 513 x 517 | 70% | 3 | Ryzen 9 7950X | 5.78 | 2.87x | 1.07x | 15.73 | 1.05x | 0.40x |
| 96 x 96 x 96 | 10% | 1 | M5 Max | 0.80 | 2.25x | 4.02x | 1.77 | 1.04x | 1.82x |
| 96 x 96 x 96 | 10% | 1 | Ryzen 9 7950X | 0.72 | 2.35x | 5.14x | 1.73 | 0.97x | 2.09x |
| 96 x 96 x 96 | 10% | 3 | M5 Max | 1.03 | 2.40x | 5.85x | 2.49 | 1.01x | 2.39x |
| 96 x 96 x 96 | 10% | 3 | Ryzen 9 7950X | 0.99 | 2.72x | 6.61x | 2.64 | 1.01x | 2.46x |
| 96 x 96 x 96 | 70% | 1 | M5 Max | 2.18 | 2.64x | 3.47x | 5.80 | 0.99x | 1.31x |
| 96 x 96 x 96 | 70% | 1 | Ryzen 9 7950X | 2.19 | 2.87x | 4.01x | 6.54 | 0.96x | 1.34x |
| 96 x 96 x 96 | 70% | 3 | M5 Max | 5.59 | 2.56x | 2.80x | 14.74 | 0.97x | 1.07x |
| 96 x 96 x 96 | 70% | 3 | Ryzen 9 7950X | 5.88 | 3.12x | 2.70x | 17.86 | 1.03x | 0.89x |

### Four concurrent callers

| Image | CPU | Overlapping ms | Taking turns ms | Speedup |
|---|---|---|---|---|
| 512 x 512 | M5 Max | 6.5 | 20.5 | 3.13x |
| 512 x 512 | Ryzen 9 7950X | 9.6 | 20.6 | 2.15x |
| 1024 x 1024 | M5 Max | 26.6 | 57.1 | 2.14x |
| 1024 x 1024 | Ryzen 9 7950X | 46.3 | 74.4 | 1.61x |
| 2048 x 2048 | M5 Max | 281.5 | 428.9 | 1.52x |
| 2048 x 2048 | Ryzen 9 7950X | 369.6 | 581.4 | 1.57x |
| 4096 x 4096 | M5 Max | 625.9 | 1447.5 | 2.31x |
| 4096 x 4096 | Ryzen 9 7950X | 1105.2 | 2120.1 | 1.92x |


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
