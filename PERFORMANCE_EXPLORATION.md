# Feature transform and component performance exploration

Measured September 16, 2026. The baseline is the saved build from the start
of this exploration, after the previous correctness audit. It is not the
published 2.2.0 build. Measurements use macOS, Apple M5 Max, Python 3.13,
and four workers unless specified. These are synthetic workloads on one
machine, not portable speedup guarantees.

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
This is a separate 25-sample experiment, not a new automatic graph cache.
An explicit prepared-contact object could make this easier to use, provided
it records expansion, connectivity, periodicity, and weighting settings.
Mutating the image or those settings must invalidate the prepared graph.

Fusing final propagation with contact extraction remains a promising next
experiment. The obstacle is that the final Euclidean sweep uses transposed
storage, while contacts depend on neighbors in the completed label image.
Clean expansion can still remove labels after a sweep, so emitting contacts
too early can create incorrect edges. A tiled final pass with boundary
halos and contact extraction after cleanup should be compared against exact
expanded labels and edge sets before being adopted.

Other candidates are component splitting along a better axis for very thin
volumes, and optional scratch-retention limits for engines that alternate
large and small images. Both need measurements that include copying and
reallocation costs, not just isolated kernel time.

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
