# Iterative audit after 2.2.0

The stopping rule is three consecutive broad review rounds with no material
findings, passing correctness checks, stable retained memory, and stable
performance. A functional, memory, or concurrency defect resets the count.
Minor cleanup is allowed, with relevant checks rerun after editing.

## Reproduction and acceptance

Build the extension from this checkout, then run:

```sh
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python -m pytest -q tests
PYTHONPATH=src NCOLOR_NO_CALIBRATE=1 python bench/audit_round.py --seed 202 --output bench/audit_results/round_1.json
```

The audit script requires NumPy, Shapely, and psutil. Each seed changes the
correctness inputs but leaves the nine performance workloads and memory
workloads fixed. Each timing is the median of 15 samples after warmup.
The retained-memory check repeats all workloads 40 times, alternates buffer
release, collects garbage, and permits at most 8 mebibytes (MiB) of resident-memory
variation after two warm blocks. Worker counts must remain constant.
Allocators can retain freed memory, so returning resident memory to its
initial value is not a requirement.

Across qualifying rounds, each workload's largest median must be no more
than 20 percent above its smallest. A noisy result requires investigation
and another measurement, not silently removing an unfavorable sample.

Correctness checks cover independent brute-force distance references in
one through five dimensions; binary and per-label connected components;
region properties; signed, wide, floating, strided, and Boolean label
formatting; signed adjacency; weighted raster coloring with both expansion
modes, periodic boundaries, contact thresholds, and soft edges; arbitrary
graph coloring; geometry contact rules; and concurrent shared-engine calls.
The pytest suite separately covers cleanup, invalid inputs, buffer ownership,
pool exhaustion, and native allocation contracts.

## Material findings fixed before qualifying rounds

- Worker exceptions now reach the caller after all participants finish,
  preserving a reusable pool instead of aborting or hanging.
- Thread counts reject nonfinite and overflowing values; fractional counts
  handle an unavailable system CPU count.
- Extension and calibration caches publish complete files atomically.
  Invalid calibration records are ignored, truncated extension copies are
  replaced, and remote-mount matching respects directory boundaries.
- Label-presence tables use atomic flags. Sparse adjacency scratch scales
  with pixel count instead of the largest label ID, and buffer release frees
  the counting scratch.
- Adjacency key packing preserves both signed label IDs, including pairs
  where both labels are negative.
- Minimum-contact filtering applies at radius 1. Filtered hard contacts can
  remain soft preferences, and explicitly empty soft edges disable automatic
  preferences.
- Work dispatch produces ordered, bounded ranges for uneven chunk sizes,
  handles a zero chunk budget consistently, and avoids size arithmetic
  overflow near the native size limit.

The initial seed-101 measurement is retained as `bench/audit_results/baseline.json`.
Its 726 correctness cases passed, its memory range was 16 kibibytes (KiB), and its worker
count was constant. Source review then found the dispatch defect, so this
measurement does not count toward stopping.

## Review log

The stopping rule was met on 2026-09-15. Each round used a fresh seed and
passed 726 correctness cases. The source review covered all major areas
across the sequence, with these additional emphases:

| Round | Seed | Review emphasis | Findings | Memory range MiB | Worker count |
|---|---|---|---|---|---|
| 1 | 202 | Numeric limits, dimensional kernels, dispatch, scratch ownership | Only unused helper and test-fixture cleanup | 0.656 | 4, constant |
| 2 | 303 | Coloring fallbacks, edge reducers, hard and soft constraints, packaging | No production defect; corrected an audit fixture that assumed six labels were present | 1.281 | 4, constant |
| 3 | 404 | Buffer release, engine registry lifetimes, metadata access, standalone headers | Clarified soft-edge getter documentation | 1.438 | 4, constant |

All 2,178 correctness cases passed. Four optional/platform tests were skipped
in the full pytest run; 929 passed. The native allocation, dispatch, worker
exception, and numeric harness passed with address and undefined-behavior
sanitizers. Every `.hpp` header compiled independently as C++17. The
standalone example compiled and ran: 100 cells, 180 adjacencies, four colors,
zero conflicts. The final pytest rerun after cleanup passed in 9.85 seconds.
`git diff --check` also passed.

To reproduce the sanitizer harness:

```sh
c++ -std=c++17 -O1 -g -pthread -fsanitize=address,undefined -fno-omit-frame-pointer -I cpp tests/native_generalization.cpp -o /tmp/ncolor-audit-native
/tmp/ncolor-audit-native
```

The round-2 audit fixture correction only affects explicitly added edges:
choose the actual maximum formatted label instead of assuming label 6
exists. The seed-202 fixture contained all six labels, so its first-round
result remains valid.

| Workload | Round 1 ms | Round 2 ms | Round 3 ms | Largest spread percent |
|---|---|---|---|---|
| Components, 512 x 512 | 1.375 | 1.353 | 1.358 | 1.6 |
| Components, eight leading singleton axes | 1.309 | 1.330 | 1.424 | 8.8 |
| Adjacency, 2048 x 2048 | 0.793 | 0.843 | 0.801 | 6.2 |
| Formatting, 1024 x 1024 | 0.535 | 0.535 | 0.547 | 2.2 |
| Manhattan expansion, 512 x 512 | 0.246 | 0.244 | 0.247 | 0.9 |
| Euclidean expansion, 512 x 512 | 0.192 | 0.192 | 0.188 | 2.1 |
| Standard coloring, 512 x 512 | 0.767 | 0.726 | 0.697 | 10.0 |
| Clean coloring, 512 x 512 | 0.735 | 0.720 | 0.728 | 2.0 |
| Geometry, 1,225 polygons | 42.036 | 41.318 | 40.872 | 2.8 |

Spread is `(largest median / smallest median - 1) * 100`, calculated from
unrounded samples. The raw samples and resident-memory readings are in
`bench/audit_results/round_1.json`, `round_2.json`, and `round_3.json`.
No measured workload crossed the 20 percent acceptance threshold. These
are stability measurements of this checkout, not speedup claims against
release 2.2.0.

Cleanup completed during review: moved a Windows test-buffer generator out
of the installed calibration module into tests; removed the unused old
minimum-only hash insertion helper. The function named
`color_graph_csr_legacy` is still called by the current coloring picker and
is not dead code.

## Accumulating performance ideas

These are investigation candidates, not measured improvements or promises.

| Review | Candidate | Evidence needed before implementation |
|---|---|---|
| Resource baseline | Density-aware presence tables, comparing atomic flags with thread-local bitsets | Measure dense and sparse label domains across thread counts, including merge cost and scratch size |
| Resource baseline | Better distinct-label capacity estimates for mixed signed inputs | Measure sorting cost and hash-table retries on sparse and repeated IDs |
| Resource baseline | Optional retained-memory budget per engine | Measure repeated small workloads after one large volume, including the cost of reallocating |
| Dispatch review | Parallel checked conversion in distance helpers | Separate conversion and transform costs for float and wide integer volumes |
| Dispatch review | Sparse-region or spatial-index distance queries | Compare against full-volume per-class transforms while preserving periodic and exact distance behavior |
| Round 1 | Avoid distance-buffer passes when callers only need expanded labels | Profile bandwidth and allocations separately for both metrics and cleanup modes |
| Round 1 | Parallel component strips with boundary merging | Benchmark large volumes against the current serial scan, including fragmented masks and deterministic numbering |
| Round 2 | Bucket or heap scheduling for saturation-based graph coloring | Profile large sparse graphs and verify deterministic tie rules and search quality |
| Round 2 | Remove automatic soft edges already imposed by explicit hard constraints | Measure redundant search work and how removal changes triangle-based soft weights |
| Round 2 | Batch small graph or image calls | Measure dispatch overhead and throughput while capping total worker count |
| Round 3 | Reuse geometry spatial indexes and buffered features | Benchmark repeated immutable collections and define mutation invalidation explicitly |
| Round 3 | Return coloring metadata in one native result snapshot | Measure small-call overhead against the current separate locked getters |
| Round 3 | Trim sparse-source scratch adaptively after workload size changes | Compare retained memory and allocation churn with explicit buffer release |

## Scope limits

Measurements in this audit run on macOS with Apple Silicon and Python 3.13.
They do not establish Windows, Linux, other processor, or free-threaded
Python performance. Fault injection covers several platform branches;
it is not a substitute for native platform validation before a release.
Passing randomized cases and sanitizers is evidence, not proof of the absence
of defects. The existing raster coloring contract still requires at least
two dimensions; expansion and connected components support one dimension.


## Follow-up performance exploration

The September 16 follow-up implements and measures feature-transform buffer
traffic reduction, parallel component labeling, and private label-presence
tables. See [PERFORMANCE_EXPLORATION.md](PERFORMANCE_EXPLORATION.md) for exact
output comparisons, timing controls, memory costs, and additional hypotheses.
The earlier three-round stopping assessment above remains the historical
baseline for this follow-up.
