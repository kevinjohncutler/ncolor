# Feature transform and component performance exploration

Measured September 16, 2026. The baseline is the saved build from the start
of this exploration, after the previous correctness audit. It is not the
published 2.2.0 build. Measurements use macOS, Apple M5 Max, Python 3.13,
and four workers unless specified. The completion round below adds a
second Apple Silicon machine. These synthetic measurements are not
portable speedup guarantees.

## What changed

Expansion still propagates the nearest source label. The Euclidean feature
transform now avoids transposing the final distance field back when the
caller only needs expanded labels. It also converts public expansion inputs
directly into working label storage, eliminating an intermediate image copy.
Distance-returning calls and distance-weighted coloring retain complete
distance fields. Cleanup, periodic boundaries, and tie rules are preserved.
This is a bandwidth optimization of label propagation, not a replacement
with a distance-only transform.

Connected components are auxiliary to ordinary `label()`. They are used by
`format_labels(clean=True)` to separate disconnected pieces belonging to
the same source label. This differs from `label(expand_mode="clean")`, whose
expansion removes bridges and spurs. Large component inputs now run in
contiguous slabs, merge connections across slab boundaries, and remap to the
same first-appearance numbering as the serial implementation. Small inputs
and one-dimensional inputs keep the serial path. An engine can supply its
worker budget through `Engine.connected_components(...)`.

The merge allocates union-find storage only if components actually cross a
slab boundary. It reuses that storage for the final remapping. Completely
disconnected slabs only need label offsets. Per-label cleanup retains the
exact source-label table, including signed source values.

Formatting uses private byte presence tables for compact label domains
(maximum normalized label below 4096). Parallel scratch is capped at one
mebibyte (MiB). Larger domains retain the shared atomic presence table.
The serial small-domain path also uses ordinary byte stores. This removes
atomic-load and branch overhead even when there is no worker contention.

## Measurements

Each reported time is the median of three process-run medians. Feature and
component runs use nine timed calls after two warmups; formatting runs use
21. Baseline/current order alternates across rounds. Every one of the 24
feature/component workloads and 12 formatting workloads has the same output
fingerprint across all six process runs, including shape, dtype, label
numbering, and source tables. The raw timings remain in
`bench/audit_results/final_*.json`; the consolidated values and fingerprints
are in `bench/audit_results/exploration_summary.json`.

| Workload | Before ms | After ms | Speedup |
|---|---|---|---|
| Standard expansion, 2048 x 2048 | 4.888 | 3.995 | 1.22x |
| Clean expansion, 2048 x 2048 | 5.952 | 4.477 | 1.33x |
| Complete standard coloring, 2048 x 2048 | 12.457 | 11.225 | 1.11x |
| Complete clean coloring, 2048 x 2048 | 13.170 | 12.259 | 1.07x |
| Binary components, 2048 x 2048, 10% foreground | 8.609 | 3.507 | 2.45x |
| Binary components, 2048 x 2048, 70% foreground | 22.944 | 7.978 | 2.88x |
| Binary components, 96 x 96 x 96, 10% foreground | 2.443 | 0.959 | 2.55x |
| Binary components, 96 x 96 x 96, 70% foreground | 11.345 | 4.090 | 2.77x |
| Component cleanup, 2048 x 2048 | 43.125 | 23.228 | 1.86x |
| Component cleanup, 96 x 96 x 96 | 23.270 | 9.340 | 2.49x |

Across 512 x 512, 2048 x 2048, and 96 x 96 x 96 inputs, standalone expansion
improved 1.10x to 1.33x. Complete coloring improved less because expansion
accounts for only part of the work. The small standard-coloring case changed
from 0.706 to 0.720 ms, a 2% slowdown. The unchanged serial per-label control
ranged from 1% to 11% slower, so small changes should not be overinterpreted.

For compact label domains, end-to-end formatting improved 1.23x to 2.04x
with one worker and 1.27x to 1.62x with four. The 65,536-label cases retain
the old algorithm; they were approximately unchanged with four workers and
4% to 9% slower with one. All timings, including regressions and controls,
are included in the consolidated results.

## Memory tradeoff

Parallel components use more peak resident memory than the serial scan.
Isolated 100-call tests used 2048 x 2048 masks and measured resident memory
after dropping each result and collecting Python garbage. Peak includes the
interpreter, inputs, outputs, and allocator caches, not just native scratch.

| Mask | Serial peak MiB | Parallel peak MiB | Parallel last-20 range MiB |
|---|---|---|---|
| sparse | 75.86 | 89.67 | 0.02 |
| dense | 73.27 | 80.02 | 0.28 |
| checkerboard | 99.16 | 121.56 | 0.00 |

An initial 20-call checkerboard test failed the memory-stability threshold
because resident memory jumped late in the run. Extending measurement to
100 calls showed a plateau in all three cases, with no more than 0.28 MiB
variation over the final 20 calls. The stepwise growth is consistent with
worker allocator caches warming, but that interpretation is not proof of
unbounded-leak absence. The measured parallel peak cost is about 7 to 22 MiB.
Earlier short-run results are retained alongside `memory_long_*.json`.

## Hypotheses tested and future work

Private packed bitsets lost to private byte flags in the presence-table
microbenchmark. With 32 labels, bitsets took 1.549 ms versus 0.422 ms for
bytes and 0.640 ms for shared atomics. Saving scratch bytes introduced a
dependent read-modify-write on frequently reused words. Large-domain private
bytes looked promising in isolation but did not justify replacing the
existing path in end-to-end measurements. An initial serial-path regression
was corrected by using the private byte table there too.

Reusing contacts is useful when recoloring unchanged geometry. The runnable
`bench/reuse_graph.py` experiment uses the existing graph-coloring interface
and a lookup table to recolor the original image. It produced identical
colors and took 8.032 ms versus 12.669 ms for the complete pipeline (1.58x).
This was a separate 25-sample experiment. The follow-up below implements
an explicit owned snapshot, with fixed topology settings and no automatic
cache invalidation.

Fusing final propagation with contact extraction remains a promising next
experiment. The obstacle is that the final Euclidean sweep uses transposed
storage, while contacts depend on neighbors in the completed label image.
Clean expansion can still remove labels after a sweep, so emitting contacts
too early can create incorrect edges. A tiled final pass with boundary
halos and contact extraction after cleanup should be compared against exact
expanded labels and edge sets before being adopted.

The follow-up below evaluates component splitting along a better axis for
very thin volumes. Optional scratch-retention limits remain a separate
candidate for engines that alternate large and small images; those need
measurements that include reallocation costs.

## Reproduction and validation

Use separate processes and the same interpreter for a saved baseline package
and this checkout. Set `PYTHONPATH` to select each package, and run these
commands without competing compute jobs:

```bash
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python bench/feature_experiments.py --output bench/audit_results/feature_current.json
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python bench/presence_format.py --output bench/audit_results/presence_current.json
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python bench/component_memory.py --case checkerboard --iterations 100 --output bench/audit_results/memory_current.json
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python bench/reuse_graph.py --output bench/audit_results/reuse_current.json
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python -m pytest -q tests
```

New contracts cover exact parallel component numbering and sources across
ranks and slab seams, empty and uniform masks, strided inputs, disconnected
slabs, distance-buffer reuse, weighted calls after unweighted calls, wide
distances, periodic boundaries, and presence-table dispatch boundaries.
Native tests additionally check label-only transposes and exact serial versus
parallel components.

Final validation: 972 tests passed and four skipped, including 43 added
Python contracts. The native contract executable passed AddressSanitizer
and UndefinedBehaviorSanitizer. All 24 C++ headers compiled independently;
the standalone example produced zero coloring conflicts. After reconciling
stale file-provider copies with the direct network mount and rebuilding,
all 24 feature/component fingerprints were checked again and matched.

Six extended broad audit rounds (seeds 505, 606, 707, 808, 909, and 1010)
passed 726 correctness cases each. Each round ran 200 additional calls per
resource workload, retained the complete memory history, and checked the
final 40 calls. Resident-memory range in that final window was at most
0.94 MiB, under the unchanged 8 MiB limit, with stable worker counts.
The shorter mixed-workload seed-707 run had also caught late allocator
warmup; its failed measurements are preserved in
`exploration_round_707_short_failure.json`. The harness now writes its
results before asserting, so a failed resource check remains inspectable.

Absolute timing stability remains qualified: sub-millisecond medians varied
by up to 39% in the extended rounds, and a subsequent process check found
an unrelated local application consuming roughly five CPU cores. It was
not stopped. Thus the earlier audit's strict all-workloads timing-stability
criterion cannot be claimed for this follow-up on this busy host. In the
three paired large-image comparisons, standard expansion improved
1.19x to 1.30x, clean expansion 1.23x to 1.34x, sparse components
2.40x to 2.58x, and dense components 2.81x to 2.88x. Those consistent
relative gains support keeping the changes, while the small timing
regressions need confirmation on an idle machine.


## Follow-up: prepared graphs, thin volumes, byte inputs, and fused contacts

This follow-up compares against local checkpoint `f6d25cb`, not release
2.2.0. Three separate-process rounds alternate baseline/current order.
The 64 formatting/coloring workloads and 60 component workloads have exact
matching output fingerprints in all six process runs. Each timing below
is the median of the three process medians. Formatting uses 15 samples,
coloring controls nine, components 11, and prepared calls 15, after warmup.
Raw measurements and the complete comparison, including controls and
regressions, are in `bench/audit_results/next_final_*.json`.

### Integrated changes

`ncolor.prepare_labels(...)` creates an owned snapshot of the normalized
render labels and the hard/soft contact graphs. `prepared.color(...)`
runs the same coloring picker and soft search, then renders the output.
It can vary the color count, search depth, and perceptual palette without
repeating formatting, expansion, contact extraction, or graph construction.
Source mutation, engine reuse, and buffer release do not affect it. Geometry
or topology-setting changes require a new snapshot. Separate engines can
color one snapshot concurrently. This is explicit reuse, with no hidden
cache keyed by a mutable array.

| Workload | Complete ms | Prepare ms | Recolor ms | Recolor speedup |
|---|---|---|---|---|
| 512 x 512, standard | 0.907 | 0.452 | 0.291 | 3.12x |
| 2048 x 2048, standard | 12.369 | 7.811 | 5.301 | 2.33x |
| 2048 x 2048, clean | 13.546 | 8.848 | 5.284 | 2.56x |
| 2048 x 2048, weighted clean | 17.815 | 13.142 | 5.549 | 3.21x |
| 96 x 96 x 96, standard | 3.010 | 2.892 | 0.250 | 12.05x |
| 96 x 96 x 96, weighted clean | 5.748 | 5.841 | 0.231 | 24.86x |

Preparation is an additional cost before the first recolor. The large
three-dimensional gains use only 27 seed labels, so eliminating raster
scans dominates a very small graph solve; they should not be generalized
to difficult graphs. The 2048 x 2048 snapshot retains 17,066,040 bytes
unweighted or 17,195,064 bytes weighted, in addition to engine scratch.
Array payload is exposed by `prepared.nbytes`; container overhead is not.

Connected-component splitting now uses an inner axis for sufficiently
large, thin, face-connected volumes when this makes more workers useful.
Partitions remain contiguous and preserve exact serial component numbering.
The optimization is deliberately restricted to the measured profitable
case: the second active axis, at least 32 rows, and at least two partitions
per leading plane. Diagonal connectivity and thinner inner extents retain
the earlier partition strategy.

| Workload, four workers | Before ms | After ms | Speedup |
|---|---|---|---|
| 2 x 1024 x 1024 components, 10% foreground | 4.948 | 2.605 | 1.90x |
| 2 x 1024 x 1024 components, 70% foreground | 13.134 | 8.890 | 1.48x |
| 2 x 1024 x 1024 per-label components | 24.333 | 16.256 | 1.50x |
| 2048 x 2048 boolean formatting | 1.004 | 0.542 | 1.85x |
| 2048 x 2048 unsigned-byte formatting, 32 values | 0.720 | 0.543 | 1.32x |
| 2048 x 2048 signed-byte formatting | 0.869 | 0.562 | 1.55x |

For large byte inputs, formatting builds private presence flags directly
from source bytes and combines casting with remapping. The input dtype
provides the domain bound, eliminating the separate int32 range scan.
This applies only to sorted formatting with multiple workers and at least
500,000 pixels. First-seen numbering, small arrays, serial calls, and wider
dtypes retain their existing path. Full-range unsigned bytes were roughly
unchanged; sparse byte values improved more. Scratch remains bounded at
one mebibyte. The end-to-end coloring pipeline retains its existing cast
and background-capture path.

### Rejected variants and the contact-fusion prototype

Unrestricted inner-axis splitting was correct but merging diagonal and
multiple-prefix seams made some thin cases three to ten times slower.
Those paths were removed. Extending source-domain presence tables to
16-bit input also regressed representative workloads. Adding per-element
min/max tracking did not rescue it, and serial byte formatting regressed.
Fusing background capture into the coloring pipeline cost roughly 5% to
11% in initial measurements. None of these broader variants remains in
the production code. Earlier raw results are retained as
`next_thin_before.json`, `next_thin_after.json`, `next_narrow_format_*.json`,
and `next_narrow_adjusted.json`.

`bench/fused_transpose_contacts.cpp` is a standalone experiment using the
shipped transpose and contact kernels. It transposes tiles with a two-pixel
halo, writes the final label image, and extracts hard and soft contacts
while the tile is still in cache. It checks exact images and sorted edge
sets against the separate passes on every timed call. The first prototype
incorrectly retained hard edges in the soft set; explicit subtraction
fixed that before performance results were accepted.

The final sweep includes tiny arrays, partial tiles, regular and jittered
seeds, standard and cleaned label fields, and periodic/nonperiodic edges:
56 cases, each with two warmups and 25 alternating paired samples. For
2048 x 2048, this phase improved 1.09x to 1.20x. These are phase timings,
not whole-pipeline speedups. Clean fields were already cleaned before the
experiment's transpose, so it does not establish that contacts may be
emitted before bridge/spur cleanup. The prototype is not integrated into
the expansion path. Safe promotion would require cleanup in the retained
layout, exact tie and contact equivalence, and higher-dimensional and
weighted implementations whose full-pipeline benefit exceeds their cost.

### Validation and remaining measurement limits

The final suite passed 1,047 tests with four skipped. New contracts cover
prepared snapshots, palette/constraint parity, independent ownership,
concurrent readers, output validation, empty/isolated graphs, signed and
wide labels, thin component seams, and byte-format dispatch boundaries.
Native contracts passed AddressSanitizer and UndefinedBehaviorSanitizer,
including the adaptive component path and byte formatter. The contact-fusion
prototype also passed both sanitizers on all 56 cases.

Repeated recoloring, rebuilding prepared snapshots, and adaptive components
were checked for retained memory in three processes, with 100 calls per
workload in each process. Variation over the final 20 samples was at most
1.13 MiB, below the unchanged 8 MiB limit. Total peak resident memory
ranged from 126.95 to 150.30 MiB. These peaks include Python, input/output
arrays, the snapshot, and retained engine buffers. The histories are
retained so allocator warmup is visible.

An unrelated application was consuming several CPU cores during paired
runs. Some unchanged controls varied substantially: the three-plane sparse
component control had per-round speedups of 0.99x, 1.96x, and 0.86x.
Consequently, these results support the consistent relative gains above,
but do not establish stable absolute performance on an idle machine.
Correctness fingerprints remained identical. No unrelated process was
stopped to improve the measurements.

### Candidates selected for the completion round

The largest remaining opportunity is to keep final feature-transform
storage in place through cleanup and contact extraction, avoiding an
entire transfer rather than making one transfer modestly cheaper. Test
cleanup ordering and tie behavior before changing the layout. A second
candidate is a compact source-only render map for sparse prepared inputs,
which could reduce the four-byte-per-pixel snapshot cost; measure its
output-fill and random-access overhead. For components, parallelizing the
large prefix-plane merge deserves a separate experiment, with union-find
contention and exact numbering included in the cost.

Reproduce the follow-up with `bench/thin_components.py`,
`bench/narrow_format.py`, `bench/prepared_labels.py`, and
`bench/next_memory.py`, each taking `--output PATH`. Compile the standalone
contact experiment with C++17, optimization, pthread support, and `-I cpp`.
Normal network builds continue to use the backend's local-cache loader
when required by the mount. Compiling and keeping benchmark working copies
on network storage remains supported.


## Completion round: integrated layout and compact prepared rendering

This round compares with local checkpoint `f644da0`. All production changes
below are integrated; the benchmark-only contact-fusion prototype remains
an experiment. Builds and working copies stayed on shared storage, including
the provider-backed view used for native execution.

### Retain the feature-transform layout through contact extraction

The standard and clean Euclidean expansion drivers can now leave their final
labels and distances in transposed storage. Cleanup and contact extraction
consume that storage directly, avoiding the last label transpose. Public
expansion still returns the original layout, with unchanged tie behavior.
The coloring renderer uses the original foreground labels, so this remains
a feature transform that propagates labels throughout expansion.

Production dispatch is deliberately bounded by measurements: two active
axes, at least 262,144 pixels, both active extents at least 128, narrow
distances, no weighting, no altered output mask, and no additional despur
iterations. Singleton axes are supported, including connectivity requested
using the original rank. Weighted and higher-dimensional calls keep their
previous path. The broader retained-layout experiment was correct in three
and four dimensions, but thin three-dimensional cleanup could be about
three times slower. That variant was not enabled.

Three paired runs on each of two Apple Silicon machines checked exact output
fingerprints on every timed call. For 2048 x 2048 inputs, whole-pipeline
speedups were 1.011x to 1.088x on M5 Max and 1.026x to 1.058x on M1 Ultra.
Rectangular two-dimensional cases measured 1.017x to 1.202x with one worker
and 0.970x to 1.212x with four. These are modest, workload-dependent gains.
Unchanged three-dimensional controls had one 0.900x outlier on the primary
machine; the second machine's controls were 0.973x to 1.013x. Background
activity still prevents a claim of stable absolute timings on an idle host.

`bench/deferred_layout.cpp` now calls the shipped expansion drivers rather
than duplicating their internals. All 36 combinations of shape, cleanup,
and periodic boundaries compare restored labels and hard/soft contacts.
`bench/completion_performance.py` measures the complete production path
with alternating paired calls, including rectangular and fallback controls.

### Smaller prepared snapshots and lookup-only recoloring

Prepared render identifiers now use one, two, or four bytes according to
the maximum label identifier. Hard and soft graph storage is unchanged.
Lookup-table-only recoloring skips image allocation and rendering; an
explicit output array still receives the rendered image and is validated.

| Snapshot workload | Before bytes | After bytes | Reduction |
|---|---|---|---|
| 2048 x 2048, unweighted | 17066040 | 8677432 | 49.2% |
| 2048 x 2048, weighted | 17195064 | 8806456 | 48.8% |
| 96 x 96 x 96, unweighted | 3541208 | 887000 | 75.0% |
| 96 x 96 x 96, weighted | 3542072 | 887864 | 74.9% |

Across the prepared benchmarks, lookup-only recoloring improved 1.14x to
6.11x. Ordinary recoloring ranged from 0.95x to 1.12x. Unweighted 2048 x
2048 preparation improved 1.13x to 1.14x, while weighted 512 x 512 clean
preparation measured 0.94x in the aggregate. The memory reduction and
lookup-only savings are the robust benefits; preparation is not uniformly
faster. Complete images and lookup tables matched the checkpoint exactly.

### Correctness and cleanup

Unformatted negative label identifiers now raise a clear exception before
they can index a lookup table. Normal formatting continues to support
negative source labels, and fractional floating values retain their existing
integer truncation behavior. Tests cover serial and parallel cast boundaries
and engine reuse after rejection.

Extension cache publication now verifies complete winner contents after a
rename collision, tolerates a bounded visibility delay, and serializes
competing writers within a process. Network filesystem tests exposed empty
cached reads immediately after concurrent renames and permissions that did
not follow local filesystem assumptions. Tests now exercise those behaviors
without assuming the temporary directory is local. The cache destination
and the existing network-loading policy are unchanged.

Unused duplicate experiment logic was removed when the retained-layout
benchmark switched to the shipped drivers. Broader variants that regressed
performance remain disabled rather than expanding production dispatch.

### Final validation and stopping evidence

The complete primary suite passed 1,087 tests with four skipped. The second
Mac passed 1,044 with 13 skipped; its native compiler harness was excluded
and optional geometry dependencies were unavailable. Geometry was covered
on the primary machine. Both native contract tests and the 36-case layout
experiment passed AddressSanitizer and UndefinedBehaviorSanitizer after
the final native changes. After the last loader adjustment, all 44 resource
tests passed in three consecutive runs on the second machine. The primary
resource and coverage suites then passed all 130 tests.

The final three broad primary rounds each checked 816 correctness cases
and 13 resource workloads over 300 blocks, or 1,200 calls per workload.
All three had zero resident-memory variation over the final ten blocks,
and worker counts stayed constant. Three secondary rounds checked 810
cases and 12 workloads over 100 blocks, explicitly omitting geometry;
final-window variation was 901,120 bytes, 278,528 bytes, and zero.
The existing eight-mebibyte limit was not relaxed.

One earlier 100-block run exceeded that limit with a 12,533,760-byte final
window range. Its resident memory had first fallen by 88 mebibytes and
then partially refilled; its final usage remained below its initial usage.
That pattern is consistent with allocator/page reclamation rather than
unbounded retained growth, but it remains recorded as a failed run.
The three longer runs above were added to investigate it, not substituted
for it in the raw history. Earlier rounds and all final histories are
retained in `bench/audit_results/completion_*`.

Repeated broad review now yields no further material correctness findings.
Valid-input output fingerprints remain identical, memory reaches a plateau
in the longer runs, and paired performance establishes the bounded gains
above. Absolute timing noise and small regressions are disclosed rather
than described as universal improvements. This is the stopping point for
this round; the experiments below are future scope.

Reproduction entry points are `bench/audit_round.py` (including explicit
`--skip-geometry` for hosts without that extra), `bench/prepared_labels.py`,
`bench/completion_performance.py`, and `bench/deferred_layout.cpp`.
Prepared before/after summaries are in `completion_prepared_summary.json`;
layout raw results use `completion_layout_*`, `completion_second_host_layout_*`,
`completion_rectangular_*`, and `deferred_layout_final.csv` under
`bench/audit_results/`.

### Future work beyond this completed round

- Measure retained-layout weighted contact extraction with matching distance
  storage and all reduction modes. Contact-count weighting is a useful first
  case because it does not need distance magnitudes.
- Investigate cleanup locality for higher-dimensional retained layouts before
  enabling them. Exact equivalence alone did not justify their measured cost.
- Compare sparse source-only prepared render maps against the new compact
  dense representation, including output clearing and irregular writes.
- Explore parallel prefix-plane component merging only with deterministic
  numbering, contention, and seam overhead included in whole-operation timing.

The earlier thin-component and private byte-presence improvements remain
integrated. Connected components support cleanup and per-label processing
as well as the public convenience operation; they are not an extra pass
required by every coloring call.


## Tagged-release comparison

[Release benchmarks](RELEASE_BENCHMARKS.md) now compare the completed code
with tagged 2.2.0 and 1.5.3 sources and equivalent scikit-image/SciPy
operations. That report contains repeated one-worker and four-worker
measurements, the historical version 1 foreground-formatting caveat,
a dense thin-component case that remains slower than scikit-image, and
draft README wording for release. Use those direct comparisons for release
claims rather than multiplying the development-checkpoint ratios above.


The Linux/x86-64 follow-up in that report now covers one, four, and 64
workers on a 64-core AMD 3995WX. It confirms hardware-dependent gains and
identifies new targets: validated casting, 3D coloring, native region
properties, and dense thin diagonal components. Stage profiles with
controlled core placement distinguish these from broad scheduling noise.
Use the revised priority list there for the next optimization round.

## Regression follow-up

[Regression recheck](REGRESSION_RECHECK.md) revisits those targets with fresh
before/after measurements. The integrated fixes restore vectorizable
negative-label validation, specialize ordinary coloring palettes without
reducing the supported range, and reduce thin-volume component seam work.
The report distinguishes regressions against ncolor 2.2.0 from remaining
competitor gaps and from scheduling-dependent timing variation.


## Cross-platform follow-up

[The final-target report](FINAL_OPTIMIZATIONS.md) follows up on weighted
retained layouts, sparse prepared rendering, dense thin components, worker
thresholds, and higher-dimensional cleanup locality. It distinguishes
integrated changes from rejected candidates and includes installed-wheel
validation. The earlier future-work list above describes the state before
that follow-up; use the new report for current decisions.
