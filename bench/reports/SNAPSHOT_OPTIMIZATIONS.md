# Snapshot preparation and rendering

This round compares the retained implementation with checkpoint `a6d878a`.
It keeps three targeted changes and rejects a broader dispatch shortcut.
The preceding measurements remain in [FINAL_OPTIMIZATIONS.md](FINAL_OPTIMIZATIONS.md).

## Integrated changes

- Sparse preparation checks its occupancy limit between 4,096-pixel blocks,
  allowing the inner count to vectorize. A platform byte search skips
  background runs when collecting foreground positions. The exact 1/32
  density cutoff is preserved. Small nonempty snapshots skip the occupancy
  scan because they always use dense storage.
- Empty snapshots store no pixel map. Recoloring clears the requested output
  directly, including reused buffers and lookup-table calls with an explicit
  output. Default empty two-dimensional snapshots report 16 bytes of owned
  array data instead of one byte per image pixel plus shape storage.
- Dense prepared render maps with fewer than 8,192 entries render serially,
  matching the ordinary renderer's threshold. Sparse maps use a lower limit
  of 1,024 foreground entries because their scattered writes benefit from
  parallelism sooner. A large image with a small foreground
  map can benefit too. Larger maps retain parallel rendering.

Snapshot byte counts exclude container overhead and the engine's reusable
scratch buffers. Releasing a snapshot and releasing engine scratch remain
separate operations.

## Measurements

The final comparisons use portable wheels, four workers, and matching
compiler flags per host. Mac measurements use an Apple M5 Max. Linux uses
an AMD 3995WX pinned to four physical cores sharing a cache. The Windows
virtual machine was shut down before final Linux timing began.

`bench/snapshot_preparation.py` alternates calls between warmed, isolated
baseline and final processes. Each process retains only its active fixture.
A 2 ms pause outside each timed call lets the inactive pool park its workers
before the other process runs. Each case has four warmups and 15 measured
calls, repeated in three fresh process pairs. Preparation timing excludes
the subsequent rendering used to validate the snapshot. Rendering includes
output allocation and clearing and requests 32 colors, avoiding timed search
limits. These are not default-four-color speedup claims.

Follow-up rendering checks use 32 calls per timed sample to distinguish
steady throughput from waking an idle worker pool.

The tables use the median of the three per-round baseline/final time ratios.
This preserves the pairing when absolute timings drift between rounds.
The summary also retains the ratio of medians for comparison. Ratios below
1 indicate a slowdown.

| Case | Mac speedup | Linux speedup |
|---|---|---|
| Prepare 512 x 512, 0.1% foreground | 1.11x | 1.13x |
| Prepare 2048 x 2048, 0.1% foreground | 1.16x | 1.08x |
| Prepare 2048 x 2048, 1% foreground | 1.16x | 1.10x |
| Recolor 64 x 64, dense | 1.23x | 1.32x |
| Recolor 96 x 96 x 96, 0.1% foreground | 1.80x | 1.19x |
| Recolor empty 512 x 512 | 7.97x | 7.27x |
| Recolor empty 2048 x 2048 | 14.04x | 4.71x |

These are case-specific gains. Dense large-image preparation and recoloring
remain near parity overall. Three fresh batched rechecks of six rendering
controls gave aggregate ratios of 0.981x to 1.031x on Mac and 0.987x to 1.007x on Linux.
The corrected sparse-render case with about 8,000 foreground entries was
0.996x on Mac and 1.007x on Linux.

Exact-name outlier rechecks also cover the smaller sparse-render case.
At about 2,600 foreground entries, final batched ratios were 0.987x on Mac and 0.986x on Linux.
Empty preparation on Mac was 0.929x for 2048 x 2048 and 0.948x for 96 x 96 x 96 in the batched recheck.
The empty-map change is retained for its storage reduction and much faster
recoloring; it is not a preparation-speed improvement for every workload.

Single-call and separate-process comparisons still contain slower outliers.
Targeted batched checks of weighted expansion and connected components were
0.991x to 1.026x. This supports stable steady throughput,
not a guarantee that every idle-call latency improves. Raw outliers remain
in the results rather than being discarded.


The broader 84-case benchmark covers connected components, weighted
expansion, preparation, and recoloring. Output fingerprints are checked
against baseline across all three rounds. The focused benchmark also checks
every rendered output, including empty, singleton-axis, sparse, and dense
inputs. Nonempty snapshot sizes are unchanged.

## Rejected and narrowed experiments

An initial candidate ran any one-chunk dispatch directly on the caller.
This removes atomic claims and worker wakeups in a native microbenchmark,
but whole-call checks exposed regressions in prepared coloring. A dense Mac
case was slower even when requesting only the color lookup table. Removing
the generic shortcut restored that coloring control; the targeted small-map
renderer then recovered the useful rendering gains.

The final generic dispatcher is unchanged from `a6d878a`. The native
16-worker controls were also unstable, so their ratios are not application
speedup claims. Initial results, an exact candidate patch, and the isolated
coloring comparison are retained under `bench/round4_results/candidates/`.
The initial Linux round that overlapped virtual-machine tests is excluded
from performance conclusions.

The first small-render candidate used the same 8,192-entry limit for dense
and sparse maps. Batched checks reproduced a Linux slowdown at about 8,000
scattered foreground entries. Lowering the sparse limit to 4,096 fixed that
case, but exact batched checks exposed a smaller slowdown at about 2,600
entries. The final limit is 1,024, retaining parallel writes in both cases.

A separate experiment limited small feature-transform envelope passes to one
worker. Although sparse preparation suggested a benefit, dense and wrapped
expansion lost substantial performance on both hosts. The 8,192-pixel cutoff
was rejected. Expansion scheduling remains unchanged.

## Validation

Tests cover empty snapshots after engine reuse and buffer release, reused
outputs, lookup-only requests with output buffers, both sides of the density
cutoff, partial count blocks, clustered foreground, singleton axes, and
recovery after a single-chunk dispatch throws an exception. Small threaded
expansion is also checked against one-worker output across dimensions,
wrapped boundaries, and clean and standard modes.

| Installed wheel | Python | Passed | Skipped |
|---|---|---|---|
| Mac, Apple M5 Max | 3.13.14 | 1216 | 4 |
| Mac, Apple M1 Ultra | 3.12.9 | 1174 | 13 |
| Linux x86-64 | 3.12.11 | 1175 | 12 |
| Windows x86-64 | 3.11.9 | 1203 | 17 |

The native contract harness also passes AddressSanitizer and
UndefinedBehaviorSanitizer. Memory checks repeat recoloring, snapshot
construction, and components 400 times each.

| Host | Recolor final range | Rebuild final range | Components final range |
|---|---|---|---|
| Mac | 0 bytes | 0 bytes | 0 bytes |
| Linux | 0 bytes | 0 bytes | 0 bytes |

Ranges cover the final 20 resident-memory observations. Empty 2048 x 2048
snapshot array storage falls from 4,194,320 bytes to 16 bytes.


These are host-wheel checks, not the complete supported-Python or manylinux
release matrix. No branch, tag, or package has been published.

## What to investigate next

`bench/snapshot_stages.py` measures the shipped preparation path. On the Mac's
2048 x 2048 cases, expansion takes approximately 59% to 70% of preparation,
contact extraction 18% to 25%, and snapshot-map construction 4% to 8%.
That limits the remaining benefit of optimizing snapshot scans alone.

- Preserve label propagation while combining expansion cleanup and contact
  discovery where their traversal orders permit it. Compare whole calls,
  since a faster isolated pass can lose to extra layout conversions.
- Investigate finalizing empty snapshots earlier in normalization. Measure
  construction latency separately from the large recoloring and storage
  gains, and avoid adding another full input scan.
- Collect foreground positions during an existing pass if it saves more
  traffic than the extra bookkeeping costs. The current exact occupancy
  check avoids allocating large temporary position vectors for dense data.
- Investigate compact position indices for snapshots whose total size fits
  a 32-bit index, with a wide-index fallback and explicit boundary checks.
- Revisit selective worker wakeups as a separate thread-pool design change.
  Reducing chunk count alone does not change how many workers participate.
  Include idle-pool latency and subsequent parallel calls in those tests.

## Reproduction

Raw final results are in `bench/round4_results/mac/` and
`bench/round4_results/linux/`; the manifest identifies source and wheel
hashes. Use isolated package directories for the baseline and final builds:

```bash
python -S bench/snapshot_preparation.py --before "$BASELINE_DIR" --after "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/snapshot_preparation.py --batch 32 --case "render/(512, 512)/0.03" --before "$BASELINE_DIR" --after "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/snapshot_preparation.py --focus --before "$BASELINE_DIR" --after "$PACKAGE_DIR" --output "$RESULT_FILE"
python -S bench/final_targets.py --source "$PACKAGE_DIR" --threads 4 --output "$RESULT_FILE"
python -S bench/snapshot_stages.py --source "$PACKAGE_DIR" --output "$RESULT_FILE"
python bench/next_memory.py --iterations 400 --output "$RESULT_FILE"
```

On Linux, prefix timing commands with `taskset -c "$CPU_SET"`. The memory
script measures the installed package. Keep compiler flags, worker count,
and CPU placement consistent. Do not multiply these ratios by measurements
from earlier sessions.
