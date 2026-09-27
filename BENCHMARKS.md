# Benchmarks

Measured on 2026-09-26 on commit 9e604c2 against ncolor 1.5.3 (the last
version 1 release) and the current SciPy and scikit-image releases. Ratios
are the other time divided by the ncolor time, so a value above 1 means
ncolor is faster. Every ncolor, SciPy, and scikit-image output was checked
against a reference before its time was recorded.

## Summary

Ranges cover both machines and every input in the tables below.

| Operation | Compared with | 4 workers | 1 worker |
|---|---|---|---|
| `label` | ncolor 1.5.3, matched settings | 6.3x to 24.1x | 2.65x to 7.4x |
| `expand_labels` | ncolor 1.5.3 `expand_labels` | 6.3x to 24.5x | 2.6x to 7.1x |
| `expand_labels` | SciPy `distance_transform_edt` feature transform | 6.6x to 24.2x | 2.6x to 7.1x |
| `expand_labels` | scikit-image `expand_labels` | 10.0x to 27.9x | 3.4x to 8.3x |
| `regionprops` | scikit-image `regionprops_table` | 4.7x to 27.8x | 1.95x to 27.8x |
| `connected_components` | scikit-image `measure.label` | 0.92x to 6.4x | 0.38x to 2.4x |
| four threads calling `label` | the same calls taking turns | 1.4x to 3.5x | not applicable |

SciPy and scikit-image run these operations on one thread, so the one-worker
column is the like-for-like comparison. `connected_components` is the one
operation where scikit-image is sometimes faster. On one thread it is ahead
on the 70% filled 2 x 513 x 517 volume (0.38x to 0.40x at full
connectivity, 0.70x to 0.78x with faces only), on the 70% filled 2D masks
with diagonal connectivity (0.80x to 0.96x), on the 10% filled
2 x 513 x 517 volume at full connectivity on the M5 (0.94x), and on the
70% filled 96 x 96 x 96 volume at full connectivity on the Ryzen (0.89x).
With four workers the only case below 1x is the 70% filled 2 x 513 x 517
volume at full connectivity (0.92x to 0.98x).

## Environment

| Item | Apple M5 Max | AMD Ryzen 9 7950X |
|---|---|---|
| Operating system | macOS 27.0 | Ubuntu 26.04 LTS |
| Python | 3.13.14 | 3.12.10 |
| Compiler | Apple clang 21.0, -O3 -march=native | gcc 15.2, -O3 -march=native, 32-byte branch alignment |

Both machines used numpy 2.5.3, SciPy 1.18.1, scikit-image 0.26.0, numba
0.67.0, and fastremap 1.20.0, installed in a fresh virtual environment. The
current code was built from source with the flags above; `setup.py` adds the
branch alignment itself on x86. Published wheels target a baseline
instruction set rather than `-march=native`, so their times can differ from
these.

## Method

Each version runs in its own process. A process warms each operation with
four calls, then times fifteen calls and records their median. The version
order rotates between rounds, and the tables report the median over rounds:
six rounds at 4 workers and three at 1 worker. ncolor 1.5.3 uses Numba with
the same thread count.

`label` runs in two configurations. Default settings are the current
defaults. Matched settings use face connectivity, no input formatting,
standard expansion, and no soft constraints, which is the closest
configuration to version 1. The version 1 defaults fail the foreground check
on these inputs, so 1.5.3 is compared only with matched settings.

The label images are the logo from `test_files/example.png`, the
`test_files/synthetic_800.npz` fixture, and boxes placed at random. The masks
are random pixels at 10% and 70% fill.

`expand_labels` is compared with version 1's own `expand_labels`, with the
nearest-label fill from SciPy's Euclidean distance transform
(`return_indices=True`), and with scikit-image's
`expand_labels(distance=inf)`. Pixels equidistant from two labels may be
assigned to either label; each such pixel is checked to be a true tie.
`regionprops` and `connected_components` did not exist in version 1.

The concurrent-caller rows time 16 distinct 23%-filled images, spread over
four threads. They compare ncolor's default engine allocation with
`NCOLOR_MAX_ENGINES=1`, where calls take turns on one pool.

## Results

### label, 4 workers

| Image | CPU | Default ms | Matched ms | vs 1.5.3 |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.30 | 0.17 | 9.27x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.44 | 0.25 | 6.32x |
| synthetic, 900 x 900, 682 labels | M5 Max | 3.16 | 2.27 | 11.74x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 4.55 | 2.70 | 8.89x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 2.01 | 1.42 | 21.79x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 2.76 | 1.90 | 20.86x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 9.13 | 6.80 | 23.26x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 16.16 | 10.99 | 16.63x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 5.97 | 2.21 | 24.14x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 7.33 | 2.85 | 22.21x |

### label, 1 worker

| Image | CPU | Default ms | Matched ms | vs 1.5.3 |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.51 | 0.31 | 5.09x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.84 | 0.50 | 3.18x |
| synthetic, 900 x 900, 682 labels | M5 Max | 9.64 | 7.38 | 3.70x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 12.31 | 9.07 | 2.65x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 6.40 | 4.54 | 6.93x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 9.64 | 6.67 | 5.89x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 31.66 | 23.54 | 7.02x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 46.73 | 33.06 | 5.53x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 20.34 | 7.56 | 7.41x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 25.76 | 10.27 | 6.16x |

### expand_labels, 4 workers

| Image | CPU | ncolor ms | vs 1.5.3 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.09 | 12.04x | 12.16x | 14.34x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.17 | 6.31x | 6.60x | 10.03x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.60 | 12.60x | 12.36x | 14.67x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 2.06 | 9.42x | 9.40x | 12.58x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.09 | 24.46x | 24.17x | 27.91x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.75 | 20.43x | 21.36x | 24.76x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 5.96 | 23.56x | 23.33x | 26.06x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 11.74 | 13.88x | 14.07x | 17.20x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.62 | 20.68x | 21.27x | 24.08x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 2.28 | 20.08x | 19.83x | 25.70x |

### expand_labels, 1 worker

| Image | CPU | ncolor ms | vs 1.5.3 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.22 | 4.74x | 4.93x | 5.78x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.31 | 3.46x | 3.60x | 5.50x |
| synthetic, 900 x 900, 682 labels | M5 Max | 5.82 | 3.47x | 3.53x | 4.09x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 7.45 | 2.60x | 2.58x | 3.42x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 3.83 | 7.14x | 7.12x | 8.26x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 6.10 | 5.87x | 6.10x | 7.16x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 21.48 | 6.75x | 6.94x | 7.38x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 31.16 | 5.24x | 5.32x | 6.52x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 5.67 | 6.11x | 6.13x | 7.17x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 8.15 | 5.54x | 5.53x | 7.15x |

### regionprops, 4 workers

| Image | CPU | ncolor ms | vs regionprops | vs regionprops_table |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.12 | 19.40x | 20.31x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.14 | 26.70x | 27.83x |
| synthetic, 900 x 900, 682 labels | M5 Max | 0.64 | 18.82x | 18.98x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.02 | 18.48x | 19.24x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 0.51 | 6.70x | 6.84x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 0.36 | 11.19x | 11.64x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 1.94 | 6.25x | 6.40x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 1.39 | 11.32x | 11.74x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 0.61 | 4.65x | 4.74x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 0.53 | 5.56x | 5.78x |

### regionprops, 1 worker

| Image | CPU | ncolor ms | vs regionprops | vs regionprops_table |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.12 | 19.78x | 20.70x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.14 | 26.64x | 27.77x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.54 | 7.99x | 8.24x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.70 | 11.01x | 11.44x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.77 | 2.00x | 2.04x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.24 | 3.21x | 3.33x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 5.25 | 2.08x | 2.27x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 4.95 | 3.19x | 3.31x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.52 | 1.90x | 1.95x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 1.41 | 2.08x | 2.16x |

### connected_components

| Mask | Fill | Conn | CPU | 4 workers ms | vs scikit-image | 1 worker ms | vs scikit-image |
|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 10% | 1 | M5 Max | 0.83 | 2.56x | 1.91 | 1.15x |
| 1024 x 1024 | 10% | 1 | Ryzen 9 7950X | 0.78 | 2.94x | 1.71 | 1.34x |
| 1024 x 1024 | 10% | 2 | M5 Max | 0.87 | 2.53x | 2.16 | 1.05x |
| 1024 x 1024 | 10% | 2 | Ryzen 9 7950X | 0.84 | 2.78x | 1.97 | 1.20x |
| 1024 x 1024 | 70% | 1 | M5 Max | 1.94 | 2.75x | 5.48 | 1.00x |
| 1024 x 1024 | 70% | 1 | Ryzen 9 7950X | 1.86 | 2.90x | 5.39 | 1.01x |
| 1024 x 1024 | 70% | 2 | M5 Max | 2.07 | 2.66x | 5.93 | 0.96x |
| 1024 x 1024 | 70% | 2 | Ryzen 9 7950X | 2.21 | 2.42x | 6.74 | 0.80x |
| 2048 x 2048 | 10% | 1 | M5 Max | 3.14 | 2.70x | 7.68 | 1.13x |
| 2048 x 2048 | 10% | 1 | Ryzen 9 7950X | 3.68 | 2.60x | 7.21 | 1.34x |
| 2048 x 2048 | 10% | 2 | M5 Max | 3.36 | 2.65x | 8.99 | 1.02x |
| 2048 x 2048 | 10% | 2 | Ryzen 9 7950X | 3.79 | 2.56x | 8.20 | 1.19x |
| 2048 x 2048 | 70% | 1 | M5 Max | 7.58 | 2.83x | 21.92 | 1.01x |
| 2048 x 2048 | 70% | 1 | Ryzen 9 7950X | 7.87 | 2.79x | 21.75 | 1.01x |
| 2048 x 2048 | 70% | 2 | M5 Max | 8.00 | 2.81x | 24.09 | 0.95x |
| 2048 x 2048 | 70% | 2 | Ryzen 9 7950X | 9.09 | 2.39x | 27.13 | 0.80x |
| 2 x 513 x 517 | 10% | 1 | M5 Max | 0.66 | 2.40x | 1.46 | 1.10x |
| 2 x 513 x 517 | 10% | 1 | Ryzen 9 7950X | 0.70 | 2.39x | 1.40 | 1.20x |
| 2 x 513 x 517 | 10% | 3 | M5 Max | 1.68 | 1.43x | 2.62 | 0.94x |
| 2 x 513 x 517 | 10% | 3 | Ryzen 9 7950X | 1.61 | 1.54x | 2.43 | 1.02x |
| 2 x 513 x 517 | 70% | 1 | M5 Max | 2.11 | 1.85x | 5.66 | 0.70x |
| 2 x 513 x 517 | 70% | 1 | Ryzen 9 7950X | 2.07 | 2.02x | 5.35 | 0.78x |
| 2 x 513 x 517 | 70% | 3 | M5 Max | 7.01 | 0.92x | 17.20 | 0.38x |
| 2 x 513 x 517 | 70% | 3 | Ryzen 9 7950X | 6.33 | 0.98x | 15.27 | 0.40x |
| 96 x 96 x 96 | 10% | 1 | M5 Max | 0.81 | 4.15x | 1.87 | 1.82x |
| 96 x 96 x 96 | 10% | 1 | Ryzen 9 7950X | 0.75 | 4.81x | 1.77 | 2.08x |
| 96 x 96 x 96 | 10% | 3 | M5 Max | 1.03 | 6.01x | 2.68 | 2.38x |
| 96 x 96 x 96 | 10% | 3 | Ryzen 9 7950X | 1.01 | 6.43x | 2.70 | 2.42x |
| 96 x 96 x 96 | 70% | 1 | M5 Max | 2.16 | 3.61x | 6.13 | 1.31x |
| 96 x 96 x 96 | 70% | 1 | Ryzen 9 7950X | 2.25 | 3.90x | 6.56 | 1.34x |
| 96 x 96 x 96 | 70% | 3 | M5 Max | 5.58 | 2.95x | 15.42 | 1.07x |
| 96 x 96 x 96 | 70% | 3 | Ryzen 9 7950X | 5.94 | 2.66x | 17.85 | 0.89x |

### Four concurrent callers

| Image | CPU | Overlapping ms | Taking turns ms | Speedup |
|---|---|---|---|---|
| 512 x 512 | M5 Max | 6.4 | 22.5 | 3.51x |
| 512 x 512 | Ryzen 9 7950X | 8.6 | 21.0 | 2.43x |
| 1024 x 1024 | M5 Max | 26.0 | 57.9 | 2.23x |
| 1024 x 1024 | Ryzen 9 7950X | 43.4 | 71.8 | 1.65x |
| 2048 x 2048 | M5 Max | 277.4 | 421.7 | 1.52x |
| 2048 x 2048 | Ryzen 9 7950X | 355.5 | 539.2 | 1.52x |
| 4096 x 4096 | M5 Max | 394.4 | 660.2 | 1.67x |
| 4096 x 4096 | Ryzen 9 7950X | 857.8 | 1228.8 | 1.43x |


Times under a millisecond vary by up to 2.1x between rounds, so small
differences on the logo image are within noise.

## Reproducing

Export 1.5.3 into its own directory, create a virtual environment with the
dependency versions above plus `pybind11`, `setuptools`, `setuptools_scm`,
and `platformdirs`, and build the current code in place with
`python setup.py build_ext --inplace`. Run the interpreter from that
environment directly; `python -S` would load packages from the base
interpreter instead.

```bash
export NCOLOR_NO_CALIBRATE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python bench/release_comparison.py corpus --output work/corpus.npz
python bench/release_comparison.py run --version current --source src \
  --revision "$(git rev-parse HEAD)" --round 1 --threads 4 \
  --corpus work/corpus.npz --output work/t4/round_1_current.json
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
