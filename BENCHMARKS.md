# Benchmarks

Measured on 2026-09-27 on commit 9e604c2 against ncolor 1.5.3 (the last
version 1 release) and the current SciPy and scikit-image releases. Ratios
are the other time divided by the ncolor time, so a value above 1 means
ncolor is faster. Every ncolor, SciPy, and scikit-image output was checked
against a reference before its time was recorded.

## Summary

Ranges cover both machines and every input in the tables below. The top of
the `label` range is the volume of packed 3D cells (63x on four workers,
18x on one); on the 2D images and sparse 3D boxes it is 6.3x to 25.3x on
four workers and 2.7x to 7.3x on one.

| Operation | Compared with | 4 workers | 1 worker |
|---|---|---|---|
| `label` | ncolor 1.5.3, matched settings | 6.3x to 63.5x | 2.7x to 17.9x |
| `expand_labels` | ncolor 1.5.3 `expand_labels` | 6.2x to 25.2x | 2.6x to 7.1x |
| `expand_labels` | SciPy `distance_transform_edt` feature transform | 6.5x to 25.8x | 2.6x to 7.6x |
| `expand_labels` | scikit-image `expand_labels` | 9.9x to 29.2x | 3.5x to 8.5x |
| `regionprops` | scikit-image `regionprops_table` | 5.4x to 34.5x | 1.9x to 27.8x |
| `connected_components` | scikit-image `measure.label` | 0.96x to 6.5x | 0.39x to 2.4x |
| four threads calling `label` | the same calls taking turns | 1.4x to 3.1x | not applicable |

SciPy and scikit-image run these operations on one thread, so the one-worker
column is the like-for-like comparison. `connected_components` is the one
operation where scikit-image is sometimes faster. On one thread it is ahead
on the 70% filled 2 x 513 x 517 volume (0.39x to 0.40x at full
connectivity, 0.70x to 0.78x with faces only), on the 70% filled 2D masks
with diagonal connectivity (0.80x to 0.95x), on the 10% filled
2 x 513 x 517 volume at full connectivity on the M5 (0.91x), and on the
70% filled 96 x 96 x 96 volume at full connectivity on the Ryzen (0.89x).
With four workers the only case below 1x is the 70% filled 2 x 513 x 517
volume at full connectivity (0.96x to 0.97x).

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
`test_files/synthetic_800.npz` fixture, boxes placed at random, and a
128 x 128 x 128 volume of cells packed like tissue (the Voronoi regions of
2700 random seeds, about 14 neighbors each, separated by one-voxel
boundaries). The masks are random pixels at 10% and 70% fill.

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
| logo, 241 x 205, 160 labels | M5 Max | 0.32 | 0.18 | 8.98x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.44 | 0.25 | 6.33x |
| synthetic, 900 x 900, 682 labels | M5 Max | 3.16 | 2.27 | 11.83x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 3.98 | 2.71 | 8.89x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.96 | 1.39 | 22.67x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 2.78 | 1.91 | 20.79x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 9.09 | 6.72 | 23.75x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 16.11 | 10.97 | 16.65x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 5.87 | 2.16 | 25.34x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 7.31 | 2.86 | 22.73x |
| packed cells, 128 x 128 x 128, 2694 labels | M5 Max | 65.87 | 28.62 | 63.20x |
| packed cells, 128 x 128 x 128, 2694 labels | Ryzen 9 7950X | 76.63 | 32.10 | 63.49x |

### label, 1 worker

| Image | CPU | Default ms | Matched ms | vs 1.5.3 |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.50 | 0.31 | 4.94x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.84 | 0.50 | 3.18x |
| synthetic, 900 x 900, 682 labels | M5 Max | 9.68 | 7.20 | 3.65x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 12.33 | 9.01 | 2.68x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 6.45 | 4.56 | 6.66x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 9.66 | 6.71 | 5.90x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 30.89 | 22.99 | 6.81x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 46.59 | 32.91 | 5.54x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 19.71 | 7.36 | 7.30x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 25.71 | 10.28 | 6.19x |
| packed cells, 128 x 128 x 128, 2694 labels | M5 Max | 210.32 | 101.86 | 17.88x |
| packed cells, 128 x 128 x 128, 2694 labels | Ryzen 9 7950X | 243.52 | 114.74 | 17.76x |

### expand_labels, 4 workers

| Image | CPU | ncolor ms | vs 1.5.3 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.09 | 12.14x | 12.50x | 14.56x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.17 | 6.20x | 6.49x | 9.88x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.60 | 12.44x | 12.86x | 15.10x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 2.06 | 9.43x | 9.39x | 12.39x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.07 | 25.16x | 25.82x | 29.24x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.75 | 20.46x | 21.17x | 24.95x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 5.95 | 23.47x | 24.52x | 27.01x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 11.59 | 14.05x | 14.25x | 17.43x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.59 | 21.37x | 22.44x | 25.89x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 2.28 | 20.27x | 19.89x | 25.70x |
| packed cells, 128 x 128 x 128, 2694 labels | M5 Max | 8.03 | 17.91x | 19.13x | 20.88x |
| packed cells, 128 x 128 x 128, 2694 labels | Ryzen 9 7950X | 10.86 | 16.71x | 17.64x | 21.39x |

### expand_labels, 1 worker

| Image | CPU | ncolor ms | vs 1.5.3 | vs SciPy EDT | vs scikit-image |
|---|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.23 | 4.63x | 4.79x | 5.54x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.32 | 3.46x | 3.57x | 5.43x |
| synthetic, 900 x 900, 682 labels | M5 Max | 5.76 | 3.40x | 3.51x | 4.16x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 7.42 | 2.61x | 2.60x | 3.48x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 3.72 | 7.10x | 7.58x | 8.45x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 6.03 | 5.97x | 6.20x | 7.14x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 21.03 | 6.62x | 6.89x | 7.63x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 32.15 | 5.08x | 5.15x | 6.30x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 5.45 | 6.33x | 6.35x | 7.31x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 8.16 | 5.65x | 5.54x | 7.24x |
| packed cells, 128 x 128 x 128, 2694 labels | M5 Max | 29.05 | 5.15x | 5.12x | 5.46x |
| packed cells, 128 x 128 x 128, 2694 labels | Ryzen 9 7950X | 38.80 | 4.77x | 5.00x | 6.05x |

### regionprops, 4 workers

| Image | CPU | ncolor ms | vs regionprops | vs regionprops_table |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.12 | 19.79x | 20.52x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.15 | 26.17x | 27.42x |
| synthetic, 900 x 900, 682 labels | M5 Max | 0.60 | 20.50x | 21.60x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.02 | 18.42x | 19.17x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 0.51 | 6.76x | 6.98x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 0.51 | 7.82x | 8.12x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 1.85 | 6.73x | 6.94x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 1.40 | 11.35x | 11.74x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 0.56 | 5.21x | 5.40x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 0.53 | 5.58x | 5.81x |
| packed cells, 128 x 128 x 128, 2694 labels | M5 Max | 1.94 | 32.72x | 34.53x |
| packed cells, 128 x 128 x 128, 2694 labels | Ryzen 9 7950X | 3.03 | 29.83x | 31.04x |

### regionprops, 1 worker

| Image | CPU | ncolor ms | vs regionprops | vs regionprops_table |
|---|---|---|---|---|
| logo, 241 x 205, 160 labels | M5 Max | 0.12 | 19.91x | 20.85x |
| logo, 241 x 205, 160 labels | Ryzen 9 7950X | 0.14 | 26.66x | 27.75x |
| synthetic, 900 x 900, 682 labels | M5 Max | 1.54 | 7.94x | 8.31x |
| synthetic, 900 x 900, 682 labels | Ryzen 9 7950X | 1.70 | 11.08x | 11.52x |
| sparse boxes, 1024 x 1024, 131 labels | M5 Max | 1.27 | 2.74x | 2.82x |
| sparse boxes, 1024 x 1024, 131 labels | Ryzen 9 7950X | 1.24 | 3.23x | 3.36x |
| sparse boxes, 2048 x 2048, 523 labels | M5 Max | 3.60 | 3.54x | 3.60x |
| sparse boxes, 2048 x 2048, 523 labels | Ryzen 9 7950X | 4.94 | 3.18x | 3.30x |
| boxes, 96 x 96 x 96, 50 labels | M5 Max | 1.54 | 1.89x | 1.93x |
| boxes, 96 x 96 x 96, 50 labels | Ryzen 9 7950X | 1.41 | 2.08x | 2.16x |
| packed cells, 128 x 128 x 128, 2694 labels | M5 Max | 6.89 | 9.09x | 9.39x |
| packed cells, 128 x 128 x 128, 2694 labels | Ryzen 9 7950X | 9.04 | 9.94x | 10.25x |

### connected_components

| Mask | Fill | Conn | CPU | 4 workers ms | vs scikit-image | 1 worker ms | vs scikit-image |
|---|---|---|---|---|---|---|---|
| 1024 x 1024 | 10% | 1 | M5 Max | 0.82 | 2.78x | 1.90 | 1.15x |
| 1024 x 1024 | 10% | 1 | Ryzen 9 7950X | 0.77 | 2.94x | 1.71 | 1.33x |
| 1024 x 1024 | 10% | 2 | M5 Max | 0.87 | 2.67x | 2.14 | 1.05x |
| 1024 x 1024 | 10% | 2 | Ryzen 9 7950X | 0.83 | 2.78x | 1.96 | 1.20x |
| 1024 x 1024 | 70% | 1 | M5 Max | 1.93 | 2.90x | 5.46 | 1.00x |
| 1024 x 1024 | 70% | 1 | Ryzen 9 7950X | 1.86 | 2.90x | 5.39 | 1.02x |
| 1024 x 1024 | 70% | 2 | M5 Max | 2.10 | 2.74x | 5.99 | 0.94x |
| 1024 x 1024 | 70% | 2 | Ryzen 9 7950X | 2.21 | 2.43x | 6.69 | 0.81x |
| 2048 x 2048 | 10% | 1 | M5 Max | 3.19 | 2.74x | 7.53 | 1.19x |
| 2048 x 2048 | 10% | 1 | Ryzen 9 7950X | 3.67 | 2.61x | 7.19 | 1.31x |
| 2048 x 2048 | 10% | 2 | M5 Max | 3.39 | 2.72x | 8.56 | 1.09x |
| 2048 x 2048 | 10% | 2 | Ryzen 9 7950X | 3.78 | 2.58x | 8.18 | 1.19x |
| 2048 x 2048 | 70% | 1 | M5 Max | 7.67 | 2.87x | 21.57 | 1.03x |
| 2048 x 2048 | 70% | 1 | Ryzen 9 7950X | 7.87 | 2.78x | 21.69 | 1.01x |
| 2048 x 2048 | 70% | 2 | M5 Max | 8.12 | 2.81x | 23.75 | 0.95x |
| 2048 x 2048 | 70% | 2 | Ryzen 9 7950X | 9.08 | 2.39x | 27.12 | 0.80x |
| 2 x 513 x 517 | 10% | 1 | M5 Max | 0.66 | 2.49x | 1.59 | 1.02x |
| 2 x 513 x 517 | 10% | 1 | Ryzen 9 7950X | 0.70 | 2.37x | 1.41 | 1.20x |
| 2 x 513 x 517 | 10% | 3 | M5 Max | 1.71 | 1.44x | 2.71 | 0.91x |
| 2 x 513 x 517 | 10% | 3 | Ryzen 9 7950X | 1.62 | 1.53x | 2.43 | 1.02x |
| 2 x 513 x 517 | 70% | 1 | M5 Max | 2.14 | 1.88x | 5.66 | 0.70x |
| 2 x 513 x 517 | 70% | 1 | Ryzen 9 7950X | 2.06 | 2.01x | 5.34 | 0.78x |
| 2 x 513 x 517 | 70% | 3 | M5 Max | 7.12 | 0.96x | 16.92 | 0.39x |
| 2 x 513 x 517 | 70% | 3 | Ryzen 9 7950X | 6.37 | 0.97x | 15.31 | 0.40x |
| 96 x 96 x 96 | 10% | 1 | M5 Max | 0.82 | 4.06x | 1.89 | 1.80x |
| 96 x 96 x 96 | 10% | 1 | Ryzen 9 7950X | 0.75 | 4.87x | 1.76 | 2.06x |
| 96 x 96 x 96 | 10% | 3 | M5 Max | 1.05 | 5.96x | 2.60 | 2.41x |
| 96 x 96 x 96 | 10% | 3 | Ryzen 9 7950X | 1.01 | 6.45x | 2.68 | 2.41x |
| 96 x 96 x 96 | 70% | 1 | M5 Max | 2.18 | 3.59x | 6.04 | 1.33x |
| 96 x 96 x 96 | 70% | 1 | Ryzen 9 7950X | 2.25 | 3.89x | 6.58 | 1.33x |
| 96 x 96 x 96 | 70% | 3 | M5 Max | 5.71 | 2.89x | 14.79 | 1.13x |
| 96 x 96 x 96 | 70% | 3 | Ryzen 9 7950X | 5.94 | 2.65x | 17.77 | 0.89x |

### Four concurrent callers

| Image | CPU | Overlapping ms | Taking turns ms | Speedup |
|---|---|---|---|---|
| 512 x 512 | M5 Max | 6.4 | 19.7 | 3.06x |
| 512 x 512 | Ryzen 9 7950X | 8.9 | 20.7 | 2.32x |
| 1024 x 1024 | M5 Max | 25.6 | 53.3 | 2.09x |
| 1024 x 1024 | Ryzen 9 7950X | 44.0 | 69.9 | 1.59x |
| 2048 x 2048 | M5 Max | 274.6 | 406.2 | 1.48x |
| 2048 x 2048 | Ryzen 9 7950X | 347.1 | 538.7 | 1.55x |
| 4096 x 4096 | M5 Max | 388.8 | 642.1 | 1.65x |
| 4096 x 4096 | Ryzen 9 7950X | 849.0 | 1215.6 | 1.43x |


Times under a millisecond vary by up to 2.8x between rounds, so small
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
