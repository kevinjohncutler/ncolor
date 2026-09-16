# Release performance comparison

Measured September 16, 2026. Current production code is `a6d09de`, compared
with the unmodified tagged sources for 2.2.0 (`6858d805`) and 1.5.3
(`07fb076c`). These are source-build comparisons, not downloaded wheel
benchmarks. All builds and working copies remained on shared storage.

## What the measurements establish

Default coloring options with four workers are 1.04x to 1.13x faster than
2.2.0 on the measured two-dimensional cases; the three-dimensional case
is essentially unchanged. With one worker, the two larger two-dimensional
cases improve 1.17x and 1.19x. Connected components benefit mainly from the
new parallel implementation: 1.47x to 3.16x versus 2.2.0 with four workers,
while one-worker results remain approximately unchanged.

Matched standard-expansion coloring is 6.93x to 22.61x faster than 1.5.3.
That is a comparison of equivalent settings and valid foreground coverage,
not a claim that version 1 and version 2 produce identical color assignments
or require the same number of colors.

## Method and limitations

- Apple M1 Ultra, macOS, Python 3.12.9, NumPy 2.4.6, SciPy 1.16.0,
  scikit-image 0.26.0, Numba 0.65.1, and fastremap 1.17.1.
- Both C++ versions use Apple Clang 21.0.0 with the same source-build
  flags: `-O3 -march=native -ffp-contract=fast -funroll-loops`.
- Three independent processes per version/configuration, four warmup calls,
  then 15 timed calls. Tables report the median of the three process medians.
  Version order alternates or rotates between rounds. Imports, compilation,
  validation, and input loading are excluded. These are warmed-up timings.
- Native engine budgets are explicitly four workers or one worker, rather
  than automatic calibration. External package calls retain their ordinary
  implementations. `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=4`, and
  `NUMBA_NUM_THREADS=4` were set for the four-worker suite.
- Every version receives the same saved input arrays. Two fixtures are
  bundled with the repository; the remaining images are reproducible random
  boxes and binary masks. This is not a substitute for a large microscopy
  corpus, another CPU architecture, or Windows/Linux wheel measurements.
- The small logo case is noisy: the 2.2.0 default-coloring process medians
  were 1.287, 0.582, and 0.588 ms. Larger default-coloring cases were much
  steadier. Neither the best single time nor a universal speedup is quoted.
- An additional validation pass checks every expansion pixel whose chosen
  label differs from SciPy: the nearest source carrying that label must be
  at exactly the same distance. These differences are valid distance ties.
  Connected-component arrays match scikit-image exactly. Region properties
  match for every measured area, bounding box, and centroid.

The ratio convention throughout is reference time divided by current time.
A ratio below 1 means current ncolor is slower.

## Default coloring versus 2.2.0

These use each version's default coloring options, target four colors and
search depth 30, with an explicitly selected engine worker budget.

| Input | 2.2.0 ms, 4 workers | Current ms, 4 workers | Speedup | Speedup, 1 worker |
|---|---|---|---|---|
| Bundled logo | 0.588 | 0.543 | 1.08x | 1.09x |
| Bundled synthetic 800 | 5.741 | 5.498 | 1.04x | 1.04x |
| 1024 x 1024 boxes | 3.686 | 3.268 | 1.13x | 1.17x |
| 2048 x 2048 boxes | 15.506 | 13.882 | 1.12x | 1.19x |
| 96 x 96 x 96 boxes | 10.907 | 10.884 | 1.00x | 1.02x |

In the separate matched standard-mode configuration, current performance
versus 2.2.0 ranges from 0.90x to 1.13x: the small logo is about 10% slower
and the three-dimensional case about 3% slower. The improvements are not
uniform across configurations; the complete matched timings remain in the
raw summary alongside the default results.

The tested default and matched coloring outputs have identical fingerprints
between 2.2.0 and current code across all three four-worker rounds. The
three-dimensional default case uses seven colors on both versions, despite
a target of four. Prepared snapshots did not exist in 2.2.0, so their recent
memory and recoloring gains should be cited against the development
checkpoint in `PERFORMANCE_EXPLORATION.md`, not mislabeled as a 2.2.0 ratio.

## Matched coloring versus version 1

The common settings are face connectivity, target four colors, search depth
30, standard Euclidean expansion, no soft contacts, and `format_input=False`
on already-normalized labels. Version 1's standard expansion has different
distance-tie choices, and the coloring algorithms differ. Actual color
counts are included instead of assuming the requested target was reached.

| Input | 1.5.3 ms | Current ms | Speedup | Colors, 1.5.3/current |
|---|---|---|---|---|
| Bundled logo | 2.537 | 0.366 | 6.93x | 4/4 |
| Bundled synthetic 800 | 41.344 | 3.778 | 10.94x | 5/4 |
| 1024 x 1024 boxes | 39.307 | 2.411 | 16.30x | 4/4 |
| 2048 x 2048 boxes | 222.311 | 9.832 | 22.61x | 5/4 |
| 96 x 96 x 96 boxes | 86.576 | 4.243 | 20.40x | 7/7 |

The 1.5.3 defaults have a historical correctness problem: expansion removes
all background, then automatic formatting treats the minimum positive label
as background. This erased 184, 479, 36, 196, and 2,788 foreground pixels
respectively on these five inputs. Those timings remain in the raw results
but are explicitly invalid and excluded from speedup calculations. The
matched configuration above avoids that bug on the common normalized inputs.
The older `bench_vs_pypi.py` did not check this and used minimum times;
its headline should not be carried forward.

## Label expansion against other packages

All rows expand labels over the entire image using Euclidean distance.
The SciPy reference computes feature indices with distances disabled and
then gathers source labels. The scikit-image reference uses
`expand_labels(distance=np.inf)`, which also computes distances internally.
This is label propagation, not a distance-only comparison. Valid distance
ties can select different source labels.

| Input | Current ms | Versus 2.2.0 | Versus SciPy feature transform | Versus scikit-image expansion |
|---|---|---|---|---|
| Bundled logo | 0.152 | 1.02x | 11.39x | 13.40x |
| Bundled synthetic 800 | 2.573 | 1.08x | 11.94x | 13.76x |
| 1024 x 1024 boxes | 1.794 | 1.22x | 18.56x | 21.80x |
| 2048 x 2048 boxes | 8.229 | 1.16x | 23.63x | 25.95x |
| 96 x 96 x 96 boxes | 2.503 | 1.12x | 22.47x | 25.83x |

## Connected components against scikit-image

Both packages receive boolean masks and return the label array and component
count. Face connectivity is compared with face connectivity; full diagonal
connectivity is compared with full diagonal connectivity. The 2.2.0 utility
was serial even when the engine budget elsewhere was four workers.

| Shape | Foreground | Connectivity | Current ms | Versus 2.2.0 | Versus scikit-image |
|---|---|---|---|---|
| 1024 x 1024 | 10% | face | 1.318 | 1.98x | 2.43x |
| 1024 x 1024 | 10% | full | 1.393 | 2.08x | 2.40x |
| 1024 x 1024 | 70% | face | 2.696 | 2.74x | 2.93x |
| 1024 x 1024 | 70% | full | 2.899 | 2.80x | 2.74x |
| 2048 x 2048 | 10% | face | 4.891 | 2.08x | 2.50x |
| 2048 x 2048 | 10% | full | 4.944 | 2.30x | 2.60x |
| 2048 x 2048 | 70% | face | 10.087 | 2.81x | 3.04x |
| 2048 x 2048 | 70% | full | 10.689 | 2.92x | 2.86x |
| 96 x 96 x 96 | 10% | face | 1.290 | 1.92x | 3.69x |
| 96 x 96 x 96 | 10% | full | 1.577 | 2.16x | 5.41x |
| 96 x 96 x 96 | 70% | face | 3.093 | 2.56x | 3.66x |
| 96 x 96 x 96 | 70% | full | 7.489 | 2.61x | 2.94x |
| 2 x 513 x 517 | 10% | face | 1.004 | 2.44x | 2.31x |
| 2 x 513 x 517 | 10% | full | 2.774 | 1.47x | 1.26x |
| 2 x 513 x 517 | 70% | face | 3.021 | 3.16x | 1.89x |
| 2 x 513 x 517 | 70% | full | 10.594 | 2.51x | 0.85x |

The blanket old claim of 1.5x to 3x faster is incomplete. Most measured
cases improve, including gains beyond that range, but dense, thin,
fully diagonal three-dimensional masks are 0.85x as fast as scikit-image
(about 18% more time). This is a concrete next optimization target.
One-worker ncolor versus 2.2.0 ranges from 0.98x to 1.02x here, confirming
that the improvement is primarily parallel execution rather than a faster
serial component kernel.

## Region properties against scikit-image

Both implementations must compute area, bounding box, and centroid for every
region. The object-based reference accesses all three properties and packs
arrays; timing lazy object construction alone would omit required work.
The separate `regionprops_table` comparison avoids that ambiguity entirely.

| Input | Current ms | Versus regionprops objects | Versus regionprops_table |
|---|---|---|---|
| Bundled logo | 0.188 | 22.25x | 23.11x |
| Bundled synthetic 800 | 2.348 | 9.01x | 9.24x |
| 1024 x 1024 boxes | 2.264 | 2.31x | 2.37x |
| 2048 x 2048 boxes | 8.824 | 2.38x | 2.44x |
| 96 x 96 x 96 boxes | 2.186 | 1.91x | 1.96x |

The large ratios on the logo and 800-region fixture reflect avoiding
per-region Python work. They should not be called a universal or typical
22x gain. The measured range against the object reference is 1.91x to
22.25x. Current region-property timings remain approximately unchanged
relative to 2.2.0 (0.97x to 1.00x).

## Draft README wording for release

The existing README has not been changed to advertise an unreleased build.
The following wording is ready to adapt once the release candidate and
cross-platform checks are complete:

> On bundled and synthetic inputs, matched standard-expansion coloring is
> 6.9x to 22.6x faster than ncolor 1.5.3 after compilation warmup. Relative
> to 2.2.0, default two-dimensional coloring is 1.04x to 1.13x faster with
> four workers. Measurements use an Apple M1 Ultra and fixed inputs;
> results vary with image geometry, connectivity, and worker count.

| Utility | Reference | Measured four-worker result |
|---|---|---|
| Connected components | scikit-image label | 0.85x to 5.41x; most tested cases faster, dense thin diagonal case slower |
| Area, bounding box, centroid | scikit-image regionprops | 1.91x to 22.25x, dependent on region count and size |
| Full Euclidean label expansion | SciPy feature transform plus label gather | 11.39x to 23.63x |
| Full Euclidean label expansion | scikit-image expand_labels | 13.40x to 25.95x |

The region-property routine is currently serial; the suite's four-worker
engine setting does not make that routine parallel. Avoid presenting the
entire table as equal-thread algorithmic speedups.

## Next work, in priority order

1. Extend this benchmark to representative real segmentation volumes and
   another architecture, then validate release wheels on supported platforms.
   Freeze dependency versions and rerun against the actual release candidate
   before publishing README figures. Do not multiply development-checkpoint
   ratios to estimate a release comparison.
2. Profile dense, thin, fully diagonal connected components. It is the
   concrete measured case where scikit-image still wins; optimize seam
   processing and worker assignment while preserving exact numbering.
3. Explore weighted contact extraction in retained feature-transform storage,
   starting with contact-count weights, which do not consume distances.
4. Revisit higher-dimensional cleanup locality and sparse prepared render
   maps only against the existing compact representation. Parallel component
   merging remains another hypothesis, not a demonstrated speedup.

There are remaining hypotheses, but broad undirected optimization is at
lower returns. These measurements support targeted work rather than a
claim that every avenue has been exhausted.

## Reproduction and raw evidence

`bench/release_comparison.py` provides `corpus`, `run`, and `report` commands.
Use isolated source exports of the tags, build the two native versions with
the same interpreter/compiler/flags, and warm version 1's Numba kernels.
Keep the exports and generated corpus in an ignored benchmark directory.
For example, from the repository root after preparing the source exports:

```bash
export BENCH_ROOT="$PWD/scratch/release_comparison"
export NCOLOR_NO_CALIBRATE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4
python bench/release_comparison.py corpus --output "$BENCH_ROOT/corpus.npz"
python bench/release_comparison.py run --source src --version current \
  --revision "$(git rev-parse HEAD)" --round 1 --threads 4 \
  --corpus "$BENCH_ROOT/corpus.npz" --output bench/release_results/round_1_current.json
python bench/release_comparison.py run --source "$BENCH_ROOT/v2.2.0/src" \
  --version 2.2.0 --revision "$(git rev-parse 'v2.2.0^{commit}')" --round 1 \
  --corpus "$BENCH_ROOT/corpus.npz" --output bench/release_results/round_1_2.2.0.json
python bench/release_comparison.py run --source "$BENCH_ROOT/v1.5.3/src" \
  --version 1.5.3 --revision "$(git rev-parse 'v1.5.3^{commit}')" --round 1 \
  --corpus "$BENCH_ROOT/corpus.npz" --output bench/release_results/round_1_1.5.3.json
python bench/release_comparison.py run --version external --round 1 \
  --corpus "$BENCH_ROOT/corpus.npz" --output bench/release_results/round_1_external.json
python bench/release_comparison.py report --directory bench/release_results \
  --output bench/release_results/summary.json
```

Repeat rounds 2 and 3 with reversed/rotated version order. For one-worker
native comparisons, use `--threads 1` and a separate output directory.
On environments with unrelated editable-install hooks, `python -S` is
supported: the harness appends dependency locations without running those
hooks. The run command verifies that the requested source was imported.

Raw samples, versions, input fingerprints, validity flags, and actual color
counts are under `bench/release_results/`. Four-worker aggregation is
`summary.json`; one-worker native aggregation is `one_worker/summary.json`.
`validation.json` is an additional correctness pass and is deliberately
excluded from timing aggregation. The reporting tests reject mixed input
corpora, environments, and revisions, and exclude invalid baseline outputs
from speedup claims.
