# Cross-platform optimization follow-up

This round follows checkpoint `92c7c69`. Two changes are integrated into
ordinary code paths: weighted contact extraction in retained feature-transform
storage and sparse prepared render maps. No public option is required.
The component shortcut and worker-threshold experiments were rejected after
cross-platform regression checks.

## Integrated changes

Minimum, maximum, count, and bounded integer-mean contact reductions can
consume the two-dimensional feature transform's final layout without first
restoring the label and distance images. Label propagation is preserved.
Mean retention requires a conservative bound proving that every integer
distance sum is exactly representable in double precision. Harmonic and
inverse-mean reductions retain their original traversal order. Existing
shape, cleanup, and distance-width restrictions still apply.

Nonempty prepared images with at least 262,144 pixels and at most 1/32
rendered foreground store foreground positions and compact labels instead
of a full render map. Rendering clears the complete output before scattering
foreground colors, including when an output buffer is reused. Dense
preparation stops its occupancy check early. The dense rendering loop has
no per-pixel sparse-representation branch.

## Final measurements

`bench/final_targets.py` measures whole calls, including output allocation,
clearing, graph coloring, and sparse writes. Three fresh-process rounds
alternate baseline and final-wheel order, with four warmups and 15 samples
per case. The tables use the median of each build's three process medians.
Ratios divide baseline time by final time; below 1 is slower.
The prepared benchmark requests 32 colors to keep timed search limits out
of the rendering comparison. These are not default-four-color speedups.

Both builds use matching portable compiler flags on each host. Linux uses
an AMD 3995WX, four workers, and four physical cores sharing a cache.
The Mac comparison uses an Apple M5 Max and four workers. Every output
fingerprint matches baseline across all 84 cases in all three rounds on
both hosts. Component arrays also match scikit-image exactly.

### Sparse prepared rendering

| Shape | Foreground fraction | Linux speedup | Mac speedup | Snapshot before, bytes | Snapshot after, bytes | Linux break-even calls | Mac break-even calls |
|---|---|---|---|---|---|---|---|
| 512 x 512 | 0.001 | 3.24x | 1.95x | 270840 | 11072 | 3 | 3 |
| 512 x 512 | 0.01 | 2.37x | 1.38x | 537880 | 40082 | 4 | 9 |
| 2048 x 2048 | 0.001 | 2.59x | 2.56x | 8534760 | 188552 | 3 | 4 |
| 2048 x 2048 | 0.01 | 1.73x | 1.84x | 8619216 | 650458 | 6 | 6 |
| 96 x 96 x 96 | 0.001 | 10.36x | 6.57x | 887112 | 9981 | 4 | 4 |
| 96 x 96 x 96 | 0.01 | 5.83x | 5.75x | 887224 | 80860 | 5 | 6 |

Sparse preparation takes approximately 5% to 21% longer in these cases.
This is a repeated-coloring tradeoff. Break-even counts divide measured
preparation overhead by the recoloring saving, rounded up. Dense rendering
controls range from 0.95x to 1.07x across the two hosts; dense preparation
also pays a small occupancy-check cost. Collecting positions during an
existing preparation pass is a useful next experiment.

`bench/paired_targets.py --render` additionally alternates individual calls
between warmed, isolated builds. It confirms the sparse recoloring gains.
Its inputs use a different random seed from the broad benchmark, so its
ratios need not match the table exactly.

### Weighted contacts

`bench/weighted_layout.py` alternates the shipped private layout switch
within one process. Whole-call improvements for the 1024 x 1024 cases are:

| Host | Standard expansion | Clean expansion |
|---|---|---|
| Linux, AMD 3995WX | 1.15x to 1.29x | 1.08x to 1.17x |
| Mac, Apple M5 Max | 1.08x to 1.14x | 1.06x to 1.12x |

The ranges cover minimum, maximum, mean, and count reductions. Every paired
output and color count matches. Harmonic reductions are unchanged controls.
These are measured cases, not promises for every shape or workload.

The Apple M1 Ultra also showed weighted and sparse-rendering gains in
paired tests. Its broad runs developed large absolute timing swings,
including unchanged controls. Background activity was present, but the
cause of the full variation was not established. Those runs are retained
as diagnostic evidence, not used for the headline table. Do not multiply
these development ratios by historical release ratios or compare absolute
times from different sessions.

## Rejected candidates and remaining work

- A full-connectivity component shortcut reused an occupied north neighbor,
  avoiding redundant unions on dense two-dimensional masks and thin slabs.
  Dense cases improved, but repeated portable Linux checks exposed sparse
  slowdowns, including after reversing process startup order. Occupancy
  gating fixed the initial Mac sparse regression but did not solve the
  Linux issue. Removing the component change restored most sparse controls;
  a compile-time specialization still lost on some cases. The component
  kernel is unchanged from the baseline. This round does not claim to have
  closed the dense thin-volume competitor gap.
- Raising the serial threshold from 8,192 to 131,072 pixels did not hold up
  across platforms. With Linux placement controlled, one-worker results
  stayed within about 1%, but four-worker calls around 32K to 64K pixels
  slowed by roughly 8% to 12%. The original threshold remains. Simply
  reducing claimed work does not eliminate the cost of waking a whole pool.
- Broader retained layouts still lose cleanup locality. A Linux phase
  benchmark takes 11.219 ms in the original layout versus 26.410 ms retained
  for clean, nonperiodic 2 x 513 x 517 data. A small four-dimensional clean
  case also regresses. These paths remain disabled in production. Full
  label images, hard contacts, and soft contacts match in all 36 native
  experiment cases on each platform. Rotation puts the length-two axis
  innermost, increasing row setups from 1,026 to 265,221 for this shape.

Useful next experiments are selective worker wakeups with a bounded worker
budget, collecting sparse render positions during an existing preparation
pass, and a strided cleanup scan along a longer axis. A component shortcut
would need to preserve the compiler's sparse-loop behavior, not just the
algorithm's operation count. Each experiment needs whole-call timing,
exact numbering or contact equivalence, and memory checks before promotion.

## Validation and release scope

Regression tests cover all 512 binary 3 x 3 neighborhoods, thin slab seams,
retained weighted layouts with singleton axes, irregular periodic contacts
and contact thresholds, and clearing reused sparse output buffers.
Windows testing also exposed operating system mocks that changed Python's
global host identity. Those tests now substitute a module-local object and
exercise the intended mount-detection branch on Windows as well as Unix.

Installed portable wheels were built with native CPU targeting disabled.
Linux uses the release x86-64-v2 target; Windows uses Microsoft Visual C++.
These are host wheel checks, not a claim that every supported Python version
or every manylinux container has been tested.

| Platform | Python | Passed | Skipped |
|---|---|---|---|
| Linux x86-64 | 3.12.11 | 1125 | 12 |
| Mac, Apple M5 Max | 3.13.14 | 1166 | 4 |
| Mac, Apple M1 Ultra | 3.12.9 | 1124 | 13 |
| Windows x86-64 | 3.11.9 | 1153 | 17 |

The native contract harness is checked with AddressSanitizer and
UndefinedBehaviorSanitizer. Linux's 100-call memory runs vary by 0, 4,096,
and 0 bytes in their final windows for recoloring, rebuilding snapshots,
and connected components, respectively. On the M5 Max, the first 100-call
component check crossed its 8,388,608-byte range limit with a late 8,421,376-byte
resident-memory step. A fresh 400-call follow-up plateaued in all three
operations, with zero range in each final 20-call window. Both measurements
are retained; a one-time resident-memory step is not hidden by changing the
assertion threshold.

An additional local 384 x 392 label fixture was checked through the release
harness. Its portable Linux candidate median default coloring time is
1.309 ms and feature expansion is 0.633 ms, versus 6.031 ms for SciPy's
feature transform plus label gather and 8.096 ms for scikit-image expansion.
Those calls do not exercise the rejected component shortcut. This small
fixture has unverified provenance and is not a substitute for a large
representative microscopy corpus. Its input fingerprint is recorded; the
input image is not redistributed.

The README now links qualified measurements instead of repeating blanket
speedup claims. Tagged-release numbers remain explicitly tied to their
original checkpoint in `RELEASE_BENCHMARKS.md`. No release, tag, or branch
has been pushed.

## Reproduction and cleanup

Final measurements are under `bench/final_results/linux_release/` and
`bench/final_results/mac_m5_release/`. `summary.json` aggregates the broad
three-round comparison. `manifest.json` records source hashes and validation.
`candidates/` preserves rejected experiments and noisier earlier runs with
a separate provenance note.

Use isolated baseline and final package directories, identical compiler
flags for each comparison, and explicit worker counts:

```bash
python -S bench/final_targets.py --source "$PACKAGE_DIR" --threads 4 --output "$RESULT_FILE"
python -S bench/paired_targets.py --before "$BASELINE_DIR" --after "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/paired_targets.py --sparse --before "$BASELINE_DIR" --after "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/paired_targets.py --render --before "$BASELINE_DIR" --after "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/weighted_layout.py --source "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/worker_passes.py --source "$PACKAGE_DIR" --threads 4 --output "$RESULT_FILE"
python bench/next_memory.py --iterations 400 --output "$RESULT_FILE"
```

On Linux, prefix timing commands with `taskset -c "$CPU_SET"` for a documented
group of physical cores. The memory script measures the installed package.
The worker experiment temporarily changed thresholds in `distinct_labels_`,
`parallel_max_label_`, and `apply_color_lut_`; that change is absent from the
final source.

Six byte-identical loose duplicates were removed. Two obsolete source
conflict copies and one empty duplicate result were moved into ignored
scratch storage after comparison. Distinct result files, figures, and the
additional image were preserved.
