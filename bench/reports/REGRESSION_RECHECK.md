# Regression recheck

This follow-up remeasures the slow cases in `RELEASE_BENCHMARKS.md` against
both tagged ncolor 2.2.0 and the development snapshot at `0f9741e`.
The latter is the before-fix baseline. The same saved input corpus is used
for every version and host.

The previously reported 0.85x Mac result compared connected components with
scikit-image, not with an older ncolor release. It was already faster than
ncolor 2.2.0, but its remaining competitor gap was reproducible.

## Changes

- Negative-input validation now reduces integer sign bits while copying
  labels and capturing background. This preserves validation without the
  Boolean reduction dependency that slowed the unformatted-input path.
- The queue coloring kernel uses a 64-bit mask and 64 counters for palettes
  below 64 colors. The existing 256-bit path still supports all 255 colors.
  Both specializations use the same algorithm and produce identical results.
- Connected-component seam merging checks outer bounds once per row and
  inner bounds only at row endpoints. Repeated pairs of local component
  identifiers need only one union. Raster order and final numbering stay
  unchanged. Scratch storage grows only with the neighbor count.

Scanning a seam separately for each neighbor offset was also tested and
rejected: it lost per-pixel reuse and slowed the dense Mac case to about
14.6 ms. The integrated implementation retains the original pixel order.

## Measurement procedure

Whole-call measurements use `bench/release_comparison.py`: four warmups,
15 samples, and three fresh-process rounds with alternating version order.
Each host uses its own matching compiler and Python environment. Ordinary
comparisons use four workers. Input and output checks run outside timed
sections. Builds and benchmark runs do not overlap on a host.

`bench/release_stages.py` times the actual shipped pipeline stages. Linux
stage checks restrict execution to four physical cores sharing a cache.
`bench/regression_properties.py` separately measures properties with no
coloring worker pool, 101 samples per round, and a single fixed CPU on Linux.
These diagnostic measurements are separate from whole-call timing tables.

Raw samples, checks, and environment metadata are under
[`bench/regression_results/`](../regression_results/).

## Results

Median of three process medians, four workers. Times are milliseconds.

| CPU | Case | Before | After | Gain | After vs scikit-image |
|---|---|---|---|---|---|
| M1 Ultra | Thin dense components | 10.545 | 8.197 | 1.29x | 1.09x |
| M1 Ultra | Small unformatted coloring | 0.389 | 0.322 | 1.21x | Not compared |
| M1 Ultra | 3D standard coloring | 4.237 | 4.013 | 1.06x | Not compared |
| AMD 3995WX | Thin dense components | 15.182 | 13.071 | 1.16x | 0.79x |
| AMD 3995WX | Small unformatted coloring | 0.803 | 0.549 | 1.46x | Not compared |
| AMD 3995WX | 3D standard coloring | 8.394 | 7.449 | 1.13x | Not compared |

The Mac component gap is closed on this corpus. Linux improves, but its
13.07 ms dense thin-volume result remains slower than scikit-image's
10.32 ms. This is a remaining competitor gap, not a regression from ncolor
2.2.0: the tagged version takes 25.13 ms on the same Linux case.

### Confirmed causes

Apple Clang's vectorizer diagnostics report that the old validated int32
copy loop was not vectorized. The integer sign reduction is vectorized
with width 16 and interleave count four. The actual small-image cast stage
falls from 0.0466 to 0.0070 ms on the Mac and from 0.0705 to 0.0132 ms in
Linux's four-core diagnostic. Negative labels, overflow checks, and float
truncation semantics remain validated.

The GNU Compiler Collection 15.2 probe confirms the same change on x86-64:
zero vectorized loops before, and a vectorized serial copy afterward using
32-byte vectors, with narrower remainder loops.

The 3D coloring stage falls from 0.680 to 0.516 ms on the Mac and from
1.221 to 0.706 ms in the controlled Linux run. Generalizing every palette
to four machine words and 256 counters had imposed a measurable cost on
ordinary small palettes. Specializing the storage removes most of that
cost without restoring the old color-count limit.

### Placement-sensitive results

Unrestricted Linux runs still show approximately 8% small-image coloring
slowdowns and occasional 15% to 20% property slowdowns against 2.2.0.
Those do not survive identical core placement and isolated property calls.
They should not be silently removed from the raw results or mistaken for
a persistent kernel regression.

| Linux worker budget | Case | 2.2.0 ms | After ms | Ratio |
|---|---|---|---|---|
| 1 | Small default coloring | 1.398 | 1.382 | 1.01x |
| 1 | Small unformatted coloring | 0.817 | 0.804 | 1.02x |
| 1 | 3D default coloring | 49.785 | 49.733 | 1.00x |
| 1 | 3D standard coloring | 18.045 | 17.184 | 1.05x |
| 4 | Small default coloring | 0.717 | 0.710 | 1.01x |
| 4 | Small unformatted coloring | 0.399 | 0.382 | 1.05x |
| 4 | 3D default coloring | 17.183 | 16.906 | 1.02x |
| 4 | 3D standard coloring | 6.569 | 6.225 | 1.06x |
| 64 | Small default coloring | 11.551 | 7.239 | 1.60x |
| 64 | Small unformatted coloring | 11.033 | 11.093 | 0.99x |
| 64 | 3D default coloring | 21.561 | 18.856 | 1.14x |
| 64 | 3D standard coloring | 14.680 | 11.936 | 1.23x |

The 64-worker restriction spans 64 distinct physical cores. It makes both
versions much slower than unrestricted scheduling and remains variable
between rounds. The earlier 0.89x 3D result becomes 0.88x in the new
unrestricted comparison, but reverses under this restriction. A reliable
64-worker version ratio cannot be inferred from these runs. Profiles put
the placement penalty across expansion, contact extraction, and other
parallel passes, rather than in one new kernel. Worker placement and
barrier coordination need a separate controlled scaling study. Restricting
a job to all physical cores is not a general performance recommendation.

For properties, the isolated Linux timings for 1024 x 1024, 2048 x 2048,
and 96 x 96 x 96 inputs differ from 2.2.0 by less than 1%. The small case
differs by about 1%. Mac isolated property timings likewise remain within
1%. No property implementation change was justified by these measurements.

## Correctness and reproducibility

All four-worker before/after validation records match exactly on both
hosts, including coloring fingerprints, color counts, foreground coverage,
component numbering, expansion fingerprints, and property checks.
Additional tests compare the compact and wide palette kernels across
weighted and unweighted modes, exercise the 63/64-color boundary, check
sign reductions at serial/parallel thresholds, and compare thin-volume
components with an independent reference at four foreground densities.

Validation completed:

- Mac Python 3.13: 1,107 passed and four skipped; the optional standalone
  native harness was excluded from this run.
- Mac Python 3.12: 116 focused optimization tests passed.
- Linux Python 3.12: 1,067 passed and 12 skipped, including the native harness.
- Final native-harness and benchmark-metadata checks: 11 passed on Linux.
- AddressSanitizer and UndefinedBehaviorSanitizer: native harness passed.
  Its allocation tracker now also overrides nonthrowing allocation, which
  avoids a mismatched-allocator diagnostic from standard-library temporary
  buffers. Sanitizer checks were not disabled.

The source manifest records the exact post-fix production tree, including
SHA-256 (Secure Hash Algorithm 256-bit) checksums of its source files.
Current-run metadata links to that tree; the before-fix snapshot is
`0f9741e`. The saved corpus fingerprints match the original release suite.

The benchmark runner accepts repeated `--case` arguments for focused
rechecks and records CPU affinity. Its summary rejects mixed recorded
placement, as well as mismatched versions, inputs, and environments.
Property diagnostics can be rerun with:

```sh
python -S bench/regression_properties.py --source "$SOURCE" --version "$VERSION" \
  --corpus "$CORPUS" --output "$OUTPUT"
```

On Linux, prefix this command with `taskset -c "$CPU_ID"` and use the same
CPU for every source version. Keep compilation and other benchmark jobs
separate from measurement.
